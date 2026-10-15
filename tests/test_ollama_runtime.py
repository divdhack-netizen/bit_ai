"""OllamaRuntime lifecycle tests: cache, unload and residency."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from assistant.ollama_runtime import (
    OllamaRuntime,
    RuntimeKey,
    StreamingOllamaLLM,
    _as_stream_mapping,
)


pytestmark = pytest.mark.unit


def _factory() -> MagicMock:
    """LLM factory that returns a new mock on every call."""
    return MagicMock(side_effect=lambda **kwargs: MagicMock(**kwargs))


def test_get_llm_reuses_cache_same_key() -> None:
    """Identical profiles share the same LangChain wrapper."""
    factory = _factory()
    runtime = OllamaRuntime(llm_factory=factory, manage_residency=False)
    first = runtime.get_llm("default", 4096, 768)
    second = runtime.get_llm("default", 4096, 768)
    assert first is second
    assert factory.call_count == 1
    runtime.close(unload=False)


def test_llm_cache_lru_evict_does_not_call_unload() -> None:
    """Past the cap, Python wrappers are dropped with no unload POST."""
    from assistant.ollama_runtime import LLM_CACHE_LIMIT

    http = MagicMock()
    factory = _factory()
    runtime = OllamaRuntime(
        llm_factory=factory,
        http_client=http,
        manage_residency=False,
    )
    for index in range(LLM_CACHE_LIMIT + 1):
        runtime.get_llm("default", 1000 + index, 8)
    assert factory.call_count == LLM_CACHE_LIMIT + 1
    assert len(runtime._llm_cache) == LLM_CACHE_LIMIT
    http.post.assert_not_called()
    runtime.get_llm("default", 1000, 8)
    assert factory.call_count == LLM_CACHE_LIMIT + 2
    runtime.close(unload=False)


def test_switch_tag_updates_active_tag() -> None:
    """Switching model updates the application's active tag."""
    runtime = OllamaRuntime(llm_factory=_factory(), manage_residency=False)
    runtime.get_llm("fast", 4096, 512)
    assert runtime.active_tag == "gemma4:e4b"
    runtime.get_llm("default", 4096, 768)
    assert runtime.active_tag == "gemma4:12b"
    runtime.close(unload=False)


def test_unload_active_without_residency_does_not_open_http() -> None:
    """With manage_residency=False, unload does not contact Ollama."""
    http = MagicMock()
    runtime = OllamaRuntime(
        llm_factory=_factory(),
        http_client=http,
        manage_residency=False,
    )
    runtime.get_llm("default", 4096, 768)
    runtime.unload_active()
    http.post.assert_not_called()
    runtime.close(unload=False)


def test_runtime_key_distinguishes_context_and_reasoning() -> None:
    """Different context or reasoning do not collide in the cache."""
    assert RuntimeKey("gemma4:12b", 4096) != RuntimeKey("gemma4:12b", 6144)
    assert RuntimeKey("gemma4:12b", 4096, reasoning=False) != RuntimeKey(
        "gemma4:12b",
        4096,
        reasoning=True,
    )


def test_get_llm_reuses_cache_with_different_num_predict() -> None:
    """The output cap does not rebuild the HTTP client."""
    factory = _factory()
    runtime = OllamaRuntime(llm_factory=factory, manage_residency=False)
    first = runtime.get_llm("default", 4096, 768)
    second = runtime.get_llm("default", 4096, 256)
    assert first is second
    assert factory.call_count == 1
    runtime.close(unload=False)


def test_get_llm_applies_num_gpu_from_host_options() -> None:
    """Host options reach the LangChain wrapper only when set."""
    from assistant.host import InferenceHostOptions

    captured: dict[str, object] = {}

    def factory(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        return MagicMock(**kwargs)

    options = InferenceHostOptions(
        num_gpu=-1,
        num_thread=4,
        keep_alive="5m",
        kv_cache_type="q8_0",
        flash_attention=True,
        max_loaded_models=1,
        num_parallel=1,
    )
    runtime = OllamaRuntime(
        llm_factory=factory,
        manage_residency=False,
        host_options=options,
        keep_alive="5m",
    )
    runtime.get_llm("default", 4096, 768)
    assert captured["num_gpu"] == -1
    assert captured["num_thread"] == 4
    assert captured["reasoning"] is True
    assert captured["base_url"]
    runtime.close(unload=False)


def test_get_llm_reasoning_false_only_if_requested() -> None:
    """Separate wrappers for reasoning True/False; False is the explicit thermal cut."""
    captured: list[dict[str, object]] = []

    def factory(**kwargs: object) -> MagicMock:
        captured.append(dict(kwargs))
        return MagicMock(**kwargs)

    runtime = OllamaRuntime(llm_factory=factory, manage_residency=False)
    on = runtime.get_llm("default", 4096, 768)
    off = runtime.get_llm("default", 4096, 768, reasoning=False)
    assert off is not on
    assert captured[0]["reasoning"] is True
    assert captured[1]["reasoning"] is False
    runtime.close(unload=False)


def test_base_url_from_ollama_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without an explicit base_url, the already normalized OLLAMA_HOST is read."""
    monkeypatch.setenv("OLLAMA_HOST", "10.1.2.3:11434")
    runtime = OllamaRuntime(llm_factory=_factory(), manage_residency=False)
    assert runtime._base_url == "http://10.1.2.3:11434"
    runtime.close(unload=False)


def test_preload_with_memory_warning_does_not_touch_active_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    On a memory-risk profile, preload does not unload the resident model
    and does not start generate on the requested tag.
    """
    http = MagicMock()
    runtime = OllamaRuntime(
        llm_factory=_factory(),
        http_client=http,
        manage_residency=True,
    )
    runtime._active_tag = "gemma4:12b"
    monkeypatch.setattr(
        "assistant.ollama_runtime.memory_warning",
        lambda variant, **_kwargs: (
            f"{variant.tag} troppo grande per preload immediato"
        ),
    )
    # Insufficient headroom simulated by the monkeypatch.
    warning = runtime.preload("fast")
    assert warning is not None
    assert "troppo grande" in warning
    assert runtime.last_memory_warning == warning
    # The useful model stays resident: no POST to Ollama.
    assert runtime.active_tag == "gemma4:12b"
    http.post.assert_not_called()
    runtime.close(unload=False)


class _StreamPull:
    """Minimal context manager that mimics httpx.Client.stream for `/api/pull`."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self):
        yield from self._lines

    def __enter__(self) -> _StreamPull:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


def test_pull_model_invalidates_catalog_and_notifies_progress() -> None:
    """The NDJSON stream updates progress and clears the `/api/tags` cache."""
    http = MagicMock()
    http.stream.return_value = _StreamPull(
        [
            '{"status":"pulling manifest"}',
            '{"status":"downloading","total":100,"completed":50}',
            '{"status":"downloading","total":100,"completed":50}',
            '{"status":"success"}',
        ]
    )
    runtime = OllamaRuntime(
        llm_factory=_factory(),
        http_client=http,
        manage_residency=True,
    )
    runtime._available_models_cache = {"altro"}
    events: list[str] = []
    runtime.pull_model("gemma4:12b", events.append)
    assert runtime._available_models_cache is None
    assert any("50%" in event for event in events)
    assert sum("50%" in event for event in events) == 1
    http.stream.assert_called_once()
    runtime.close(unload=False)


def _llm_with_stream(payloads: list[dict[str, object]]) -> StreamingOllamaLLM:
    """Build a wrapper that does not talk to Ollama and emits the given payloads."""
    llm = StreamingOllamaLLM(
        model="x",
        validate_model_on_init=False,
        reasoning=True,
    )
    llm._create_generate_stream = lambda *_args, **_kwargs: iter(payloads)
    return llm


def test_generate_params_does_not_forward_num_predict_to_client() -> None:
    """num_predict lives in options, not as a kwarg of Client.generate."""
    llm = StreamingOllamaLLM(
        model="x",
        validate_model_on_init=False,
        num_predict=64,
    )
    params = llm._generate_params("ciao", num_predict=128)
    assert "num_predict" not in params
    assert params["options"]["num_predict"] == 128
    assert params["options"]["top_k"] == 64
    assert params["options"]["num_batch"] == 512


def test_stream_emits_live_reasoning_with_thinking_true() -> None:
    """Ollama `thinking` tokens reach the callback before the text."""
    llm = _llm_with_stream(
        [
            {"thinking": "perche", "response": ""},
            {"thinking": "", "response": "ok", "done": True},
        ]
    )
    events: list[tuple[str, bool]] = []
    manager = MagicMock()
    manager.on_llm_new_token.side_effect = (
        lambda token, **kwargs: events.append((token, bool(kwargs.get("thinking"))))
    )
    chunks = list(llm._stream("prompt", run_manager=manager))
    assert events == [("perche", True), ("ok", False)]
    assert chunks[-1].text == "ok"


def test_stream_reasoning_false_promotes_thinking_to_text() -> None:
    """Without native CoT, `thinking` tokens must reach the ReAct parser."""
    llm = StreamingOllamaLLM(
        model="x",
        validate_model_on_init=False,
        reasoning=False,
    )
    llm._create_generate_stream = lambda *_args, **_kwargs: iter(
        [
            {"thinking": "Action: scrivi_file\nAction Input: a.js\n---\nconsole.log(1)\n", "response": ""},
            {
                "thinking": "",
                "response": "",
                "done": True,
                "done_reason": "length",
            },
        ]
    )
    events: list[tuple[str, bool]] = []
    manager = MagicMock()
    manager.on_llm_new_token.side_effect = (
        lambda token, **kwargs: events.append((token, bool(kwargs.get("thinking"))))
    )
    chunks = list(llm._stream("prompt", run_manager=manager))
    assert any(not is_think for _token, is_think in events)
    assert "scrivi_file" in "".join(chunk.text for chunk in chunks)


def test_stream_exhausted_thinking_becomes_react_text() -> None:
    """If native CoT fills num_predict, the ReAct parser still receives the thinking."""
    llm = _llm_with_stream(
        [
            {"thinking": "Thought: salvo.\nAction: scrivi_file\nAction Input: a.js\n---\nconsole.log(1)\n", "response": ""},
            {
                "thinking": "",
                "response": "",
                "done": True,
                "done_reason": "length",
                "eval_count": 1024,
            },
        ]
    )
    events: list[tuple[str, bool]] = []
    manager = MagicMock()
    manager.on_llm_new_token.side_effect = (
        lambda token, **kwargs: events.append((token, bool(kwargs.get("thinking"))))
    )
    final = llm._stream_with_aggregation("prompt", run_manager=manager)
    assert all(is_think for _token, is_think in events)
    assert "Action: scrivi_file" in final.text
    assert "a.js" in final.text


def test_as_stream_mapping_generate_response_without_keys() -> None:
    """Ollama 0.33 frames are Pydantic objects, not dicts with `.keys()`."""

    class _GenerateResponse:
        def __init__(self) -> None:
            self._data = {
                "thinking": "",
                "response": "",
                "done": True,
                "done_reason": "length",
                "eval_count": 768,
            }

        def get(self, key: str, default: object = None) -> object:
            return self._data.get(key, default)

        def model_dump(self) -> dict[str, object]:
            return dict(self._data)

    mapped = _as_stream_mapping(_GenerateResponse())
    assert mapped["done"] is True
    assert mapped["done_reason"] == "length"
    assert "keys" not in dir(_GenerateResponse())


def test_stream_done_generate_response_does_not_raise() -> None:
    """The final `done` frame must not call `.keys()` on the Ollama payload."""

    class _GenerateResponse:
        def get(self, key: str, default: object = None) -> object:
            data = {"thinking": "x", "response": "", "done": True, "done_reason": "stop"}
            return data.get(key, default)

        def model_dump(self) -> dict[str, object]:
            return {
                "thinking": "x",
                "response": "",
                "done": True,
                "done_reason": "stop",
            }

    llm = _llm_with_stream([_GenerateResponse()])
    chunks = list(llm._stream("prompt", run_manager=MagicMock()))
    assert chunks[-1].generation_info is not None
    assert chunks[-1].generation_info["done"] is True


def test_react_aggregation_emits_thinking_before_content() -> None:
    """ReAct uses `_stream_with_aggregation`: CoT must still be live."""
    llm = _llm_with_stream(
        [
            {"thinking": "a", "response": ""},
            {"thinking": "b", "response": ""},
            {"response": "ciao", "done": True},
        ]
    )
    order: list[str] = []
    manager = MagicMock()
    manager.on_llm_new_token.side_effect = lambda token, **kwargs: order.append(
        "think" if kwargs.get("thinking") else "text"
    )
    final = llm._stream_with_aggregation("prompt", run_manager=manager)
    assert order == ["think", "think", "text"]
    assert final.text == "ciao"
    assert final.generation_info is not None
    assert final.generation_info["thinking"] == "ab"


def test_aggregation_many_thinking_chunks_without_merge() -> None:
    """Many CoT tokens must not merge generation_info with `GenerationChunk +=`."""
    payloads: list[dict[str, object]] = [
        {"thinking": str(index), "response": ""} for index in range(50)
    ]
    payloads.append({"response": "fine", "done": True})
    llm = _llm_with_stream(payloads)
    final = llm._stream_with_aggregation("prompt", run_manager=MagicMock())
    assert final.text == "fine"
    assert final.generation_info is not None
    assert final.generation_info["thinking"] == "".join(str(index) for index in range(50))


def test_server_env_diagnostics_reports_flash_attention_off() -> None:
    """A daemon env without FA produces a warning with a launchctl hint."""
    from assistant.ollama_runtime import server_env_diagnostics

    warning = server_env_diagnostics(
        {"OLLAMA_FLASH_ATTENTION": "false", "OLLAMA_NUM_PARALLEL": "1"}
    )
    assert warning is not None
    assert "FLASH_ATTENTION" in warning
    assert "launchctl setenv" in warning
    assert server_env_diagnostics(
        {"OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_NUM_PARALLEL": "1"}
    ) is None


def test_preload_sends_num_ctx_and_host_options() -> None:
    """The preload POST aligns num_ctx/num_gpu with the session wrapper."""
    from assistant.host import InferenceHostOptions

    http = MagicMock()
    http.post.return_value.raise_for_status.return_value = None
    http.get.return_value.json.return_value = {"models": []}
    http.get.return_value.raise_for_status.return_value = None
    options = InferenceHostOptions(
        num_gpu=-1,
        num_thread=4,
        keep_alive="30m",
        kv_cache_type="q8_0",
        flash_attention=True,
        max_loaded_models=1,
        num_parallel=1,
    )
    runtime = OllamaRuntime(
        llm_factory=_factory(),
        http_client=http,
        manage_residency=True,
        host_options=options,
        keep_alive="30m",
    )
    runtime.preload("default")
    # /api/show (model cap for the session window), then load and prefill.
    generate = [
        call.kwargs["json"]
        for call in http.post.call_args_list
        if call.args and call.args[0] == "/api/generate"
    ]
    assert len(generate) == 2
    first, second = generate
    assert first["options"]["num_gpu"] == -1
    assert first["options"]["num_thread"] == 4
    assert first["options"]["num_ctx"] >= 4096
    assert second["options"]["num_predict"] == 1
    assert second["prompt"]
    runtime.close(unload=False)


def test_preload_without_residency_does_not_read_daemon_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests with a fake factory do not pay for pgrep/ps on the daemon."""
    spy = MagicMock(return_value=None)
    monkeypatch.setattr("assistant.ollama_runtime.pid_daemon_ollama", spy)
    http = MagicMock()
    http.post.return_value.raise_for_status.return_value = None
    runtime = OllamaRuntime(
        llm_factory=_factory(),
        http_client=http,
        manage_residency=False,
    )
    runtime.preload("default")
    spy.assert_not_called()
    runtime.close(unload=False)


def test_preload_diagnostics_reuses_env_if_pid_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A PID change recomputes the env; the same PID skips ps/proc."""
    from assistant.host import InferenceHostOptions

    http = MagicMock()
    http.post.return_value.raise_for_status.return_value = None
    http.get.return_value.json.return_value = {"models": []}
    http.get.return_value.raise_for_status.return_value = None
    pid_spy = MagicMock(return_value=4242)
    env_spy = MagicMock(
        return_value={"OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_NUM_PARALLEL": "1"}
    )
    monkeypatch.setattr("assistant.ollama_runtime.pid_daemon_ollama", pid_spy)
    monkeypatch.setattr("assistant.ollama_runtime.environ_daemon_ollama", env_spy)
    options = InferenceHostOptions(
        num_gpu=-1,
        num_thread=4,
        keep_alive="30m",
        kv_cache_type="q8_0",
        flash_attention=True,
        max_loaded_models=1,
        num_parallel=1,
    )
    runtime = OllamaRuntime(
        llm_factory=_factory(),
        http_client=http,
        manage_residency=True,
        host_options=options,
        keep_alive="30m",
    )
    runtime.preload("default")
    runtime.preload("default")
    assert pid_spy.call_count == 2
    assert env_spy.call_count == 1
    pid_spy.return_value = 4243
    runtime.preload("default")
    assert env_spy.call_count == 2
    runtime.close(unload=False)



def test_describe_image_sends_base64_without_thinking() -> None:
    """Vision uses the session model: base64 image, think off."""
    import base64

    http = MagicMock()
    http.post.return_value.json.return_value = {"response": "Una balena."}
    runtime = OllamaRuntime(
        llm_factory=_factory(),
        http_client=http,
        manage_residency=True,
    )
    runtime._available_models_cache = frozenset({"gemma4:12b"})
    text = runtime.describe_image("default", b"PNG", "coda.png")
    assert text == "Una balena."
    body = http.post.call_args.kwargs["json"]
    assert body["model"] == "gemma4:12b"
    assert body["images"] == [base64.b64encode(b"PNG").decode("ascii")]
    assert body["think"] is False
    runtime.close(unload=False)


def test_describe_image_without_answer_is_unreadable_document() -> None:
    """Ollama missing: a readable error for the model, not a turn crash."""
    from assistant.readers import UnreadableDocument

    runtime = OllamaRuntime(llm_factory=_factory(), manage_residency=False)
    with pytest.raises(UnreadableDocument, match="coda.png"):
        runtime.describe_image("default", b"PNG", "coda.png")
    runtime.close(unload=False)


def test_session_context_single_for_all_routes() -> None:
    """Preload, card, direct, ReAct and summary ask for the same num_ctx: no reload."""
    from assistant.console import ConsoleUI
    from assistant.host import HostCapabilities, InferenceHostOptions
    from assistant.session import AgentSession

    memory = 16 * 1024**3
    host = HostCapabilities(
        os="darwin",
        arch="arm64",
        chip_family="m4",
        chip_brand="Apple M4",
        memory_bytes=memory,
        inference_memory_bytes=memory,
        performance_cores=4,
        has_discrete_gpu=False,
        cooling="fanless_conservative",
    )
    options = InferenceHostOptions(
        num_gpu=-1,
        num_thread=None,
        keep_alive="30m",
        kv_cache_type="q4_0",
        flash_attention=True,
        max_loaded_models=1,
        num_parallel=1,
    )
    runtime = OllamaRuntime(
        llm_factory=_factory(),
        manage_residency=False,
        host=host,
        host_options=options,
    )
    session_ctx = runtime.session_context("default")
    # q4_0 on 16 GB: the session window already rises to the expanded step.
    assert session_ctx == 8192
    session = AgentSession(
        MagicMock(spec=ConsoleUI),
        llm_factory=_factory(),
        ollama_runtime=runtime,
        host=host,
    )
    try:
        for route in ("direct", "react"):
            profile, _history = session._profile_with_history(
                "default", route, None, "domanda breve", 128
            )
            assert profile.context_tokens == session_ctx
    finally:
        session.close()
        runtime.close(unload=False)
