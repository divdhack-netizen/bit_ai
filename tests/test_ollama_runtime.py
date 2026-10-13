"""Test ciclo di vita OllamaRuntime: cache, unload e residency."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from assistente.ollama_runtime import (
    OllamaRuntime,
    RuntimeKey,
    StreamingOllamaLLM,
    _as_stream_mapping,
)


pytestmark = pytest.mark.unit


def _factory() -> MagicMock:
    """Factory LLM che restituisce un nuovo mock a ogni chiamata."""
    return MagicMock(side_effect=lambda **kwargs: MagicMock(**kwargs))


def test_get_llm_riusa_cache_stessa_chiave() -> None:
    """Profili identici condividono lo stesso wrapper LangChain."""
    factory = _factory()
    runtime = OllamaRuntime(llm_factory=factory, manage_residency=False)
    first = runtime.get_llm("default", 4096, 768)
    second = runtime.get_llm("default", 4096, 768)
    assert first is second
    assert factory.call_count == 1
    runtime.close(unload=False)


def test_llm_cache_lru_evict_non_chiama_unload() -> None:
    """Oltre il tetto si droppano wrapper Python, senza POST di unload."""
    from assistente.ollama_runtime import LLM_CACHE_LIMIT

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


def test_switch_tag_aggiorna_active_tag() -> None:
    """Il cambio modello aggiorna il tag attivo applicativo."""
    runtime = OllamaRuntime(llm_factory=_factory(), manage_residency=False)
    runtime.get_llm("fast", 4096, 512)
    assert runtime.active_tag == "gemma4:e4b"
    runtime.get_llm("default", 4096, 768)
    assert runtime.active_tag == "gemma4:12b"
    runtime.close(unload=False)


def test_unload_active_senza_residency_non_apre_http() -> None:
    """Con manage_residency=False unload non contatta Ollama."""
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


def test_runtime_key_distingue_contesto_e_reasoning() -> None:
    """Contesto o reasoning diversi non collidono nella cache."""
    assert RuntimeKey("gemma4:12b", 4096) != RuntimeKey("gemma4:12b", 6144)
    assert RuntimeKey("gemma4:12b", 4096, reasoning=False) != RuntimeKey(
        "gemma4:12b",
        4096,
        reasoning=True,
    )


def test_get_llm_riusa_cache_con_num_predict_diverso() -> None:
    """Il tetto di output non ricostruisce il client HTTP."""
    factory = _factory()
    runtime = OllamaRuntime(llm_factory=factory, manage_residency=False)
    first = runtime.get_llm("default", 4096, 768)
    second = runtime.get_llm("default", 4096, 256)
    assert first is second
    assert factory.call_count == 1
    runtime.close(unload=False)


def test_get_llm_applica_num_gpu_dalle_opzioni_host() -> None:
    """Le opzioni host finiscono nel wrapper LangChain solo se valorizzate."""
    from assistente.host import InferenceHostOptions

    catturato: dict[str, object] = {}

    def factory(**kwargs: object) -> MagicMock:
        catturato.update(kwargs)
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
    assert catturato["num_gpu"] == -1
    assert catturato["num_thread"] == 4
    assert catturato["reasoning"] is True
    assert catturato["base_url"]
    runtime.close(unload=False)


def test_get_llm_reasoning_false_solo_se_richiesto() -> None:
    """Wrapper distinti per reasoning True/False; False è il taglio termico esplicito."""
    catturati: list[dict[str, object]] = []

    def factory(**kwargs: object) -> MagicMock:
        catturati.append(dict(kwargs))
        return MagicMock(**kwargs)

    runtime = OllamaRuntime(llm_factory=factory, manage_residency=False)
    acceso = runtime.get_llm("default", 4096, 768)
    spento = runtime.get_llm("default", 4096, 768, reasoning=False)
    assert spento is not acceso
    assert catturati[0]["reasoning"] is True
    assert catturati[1]["reasoning"] is False
    runtime.close(unload=False)


def test_base_url_da_ollama_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Senza base_url esplicito si legge OLLAMA_HOST già normalizzato."""
    monkeypatch.setenv("OLLAMA_HOST", "10.1.2.3:11434")
    runtime = OllamaRuntime(llm_factory=_factory(), manage_residency=False)
    assert runtime._base_url == "http://10.1.2.3:11434"
    runtime.close(unload=False)


def test_preload_con_memory_warning_non_tocca_modello_attivo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Su profilo a rischio memoria, preload non scarica il modello residente
    e non avvia generate sul tag richiesto.
    """
    http = MagicMock()
    runtime = OllamaRuntime(
        llm_factory=_factory(),
        http_client=http,
        manage_residency=True,
    )
    runtime._active_tag = "gemma4:12b"
    monkeypatch.setattr(
        "assistente.ollama_runtime.memory_warning",
        lambda variant, **_kwargs: (
            f"{variant.tag} troppo grande per preload immediato"
        ),
    )
    # Headroom insufficiente simulato dal monkeypatch.
    warning = runtime.preload("fast")
    assert warning is not None
    assert "troppo grande" in warning
    assert runtime.last_memory_warning == warning
    # Il modello utile resta residente: nessun POST a Ollama.
    assert runtime.active_tag == "gemma4:12b"
    http.post.assert_not_called()
    runtime.close(unload=False)


class _StreamPull:
    """Context manager minimo che imita httpx.Client.stream per `/api/pull`."""

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


def test_pull_model_invalida_catalogo_e_notifica_progresso() -> None:
    """Lo stream NDJSON aggiorna il progresso e svuota la cache `/api/tags`."""
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
    eventi: list[str] = []
    runtime.pull_model("gemma4:12b", eventi.append)
    assert runtime._available_models_cache is None
    assert any("50%" in evento for evento in eventi)
    assert sum("50%" in evento for evento in eventi) == 1
    http.stream.assert_called_once()
    runtime.close(unload=False)


def _llm_con_stream(payloads: list[dict[str, object]]) -> StreamingOllamaLLM:
    """Costruisce un wrapper che non parla con Ollama e emette i payload dati."""
    llm = StreamingOllamaLLM(
        model="x",
        validate_model_on_init=False,
        reasoning=True,
    )
    llm._create_generate_stream = lambda *_args, **_kwargs: iter(payloads)
    return llm


def test_generate_params_non_inoltra_num_predict_a_client() -> None:
    """num_predict vive in options, non come kwarg di Client.generate."""
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


def test_stream_emette_reasoning_live_con_thinking_true() -> None:
    """I token Ollama `thinking` arrivano al callback prima del testo."""
    llm = _llm_con_stream(
        [
            {"thinking": "perche", "response": ""},
            {"thinking": "", "response": "ok", "done": True},
        ]
    )
    eventi: list[tuple[str, bool]] = []
    manager = MagicMock()
    manager.on_llm_new_token.side_effect = (
        lambda token, **kwargs: eventi.append((token, bool(kwargs.get("thinking"))))
    )
    chunks = list(llm._stream("prompt", run_manager=manager))
    assert eventi == [("perche", True), ("ok", False)]
    assert chunks[-1].text == "ok"


def test_stream_reasoning_false_promuove_thinking_a_testo() -> None:
    """Senza CoT nativo i token `thinking` devono arrivare al parser ReAct."""
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
    eventi: list[tuple[str, bool]] = []
    manager = MagicMock()
    manager.on_llm_new_token.side_effect = (
        lambda token, **kwargs: eventi.append((token, bool(kwargs.get("thinking"))))
    )
    chunks = list(llm._stream("prompt", run_manager=manager))
    assert any(not is_think for _token, is_think in eventi)
    assert "scrivi_file" in "".join(chunk.text for chunk in chunks)


def test_stream_thinking_esaurito_diventa_testo_react() -> None:
    """Se il CoT nativo riempie num_predict, il parser ReAct riceve comunque il thinking."""
    llm = _llm_con_stream(
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
    eventi: list[tuple[str, bool]] = []
    manager = MagicMock()
    manager.on_llm_new_token.side_effect = (
        lambda token, **kwargs: eventi.append((token, bool(kwargs.get("thinking"))))
    )
    final = llm._stream_with_aggregation("prompt", run_manager=manager)
    assert all(is_think for _token, is_think in eventi)
    assert "Action: scrivi_file" in final.text
    assert "a.js" in final.text


def test_as_stream_mapping_generate_response_senza_keys() -> None:
    """I frame Ollama 0.33 sono Pydantic, non dict con `.keys()`."""

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


def test_stream_done_generate_response_non_solleva() -> None:
    """Il frame finale `done` non deve chiamare `.keys()` sul payload Ollama."""

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

    llm = _llm_con_stream([_GenerateResponse()])
    chunks = list(llm._stream("prompt", run_manager=MagicMock()))
    assert chunks[-1].generation_info is not None
    assert chunks[-1].generation_info["done"] is True


def test_aggregazione_react_emette_thinking_prima_del_contenuto() -> None:
    """ReAct usa `_stream_with_aggregation`: il CoT deve comunque essere live."""
    llm = _llm_con_stream(
        [
            {"thinking": "a", "response": ""},
            {"thinking": "b", "response": ""},
            {"response": "ciao", "done": True},
        ]
    )
    ordine: list[str] = []
    manager = MagicMock()
    manager.on_llm_new_token.side_effect = lambda token, **kwargs: ordine.append(
        "think" if kwargs.get("thinking") else "text"
    )
    final = llm._stream_with_aggregation("prompt", run_manager=manager)
    assert ordine == ["think", "think", "text"]
    assert final.text == "ciao"
    assert final.generation_info is not None
    assert final.generation_info["thinking"] == "ab"


def test_aggregazione_molti_chunk_thinking_senza_merge() -> None:
    """Tanti token CoT non devono unire generation_info con `GenerationChunk +=`."""
    payloads: list[dict[str, object]] = [
        {"thinking": str(indice), "response": ""} for indice in range(50)
    ]
    payloads.append({"response": "fine", "done": True})
    llm = _llm_con_stream(payloads)
    final = llm._stream_with_aggregation("prompt", run_manager=MagicMock())
    assert final.text == "fine"
    assert final.generation_info is not None
    assert final.generation_info["thinking"] == "".join(str(indice) for indice in range(50))


def test_diagnostica_server_env_segnala_flash_attention_spenta() -> None:
    """Env del daemon senza FA produce un avviso con hint launchctl."""
    from assistente.ollama_runtime import diagnostica_server_env

    avviso = diagnostica_server_env(
        {"OLLAMA_FLASH_ATTENTION": "false", "OLLAMA_NUM_PARALLEL": "1"}
    )
    assert avviso is not None
    assert "FLASH_ATTENTION" in avviso
    assert "launchctl setenv" in avviso
    assert diagnostica_server_env(
        {"OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_NUM_PARALLEL": "1"}
    ) is None


def test_preload_invia_num_ctx_e_opzioni_host() -> None:
    """Il POST di preload allinea num_ctx/num_gpu al wrapper di sessione."""
    from assistente.host import InferenceHostOptions

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
    # /api/show (tetto del modello per la finestra di sessione), poi load e prefill.
    generate = [
        call.kwargs["json"]
        for call in http.post.call_args_list
        if call.args and call.args[0] == "/api/generate"
    ]
    assert len(generate) == 2
    primo, secondo = generate
    assert primo["options"]["num_gpu"] == -1
    assert primo["options"]["num_thread"] == 4
    assert primo["options"]["num_ctx"] >= 4096
    assert secondo["options"]["num_predict"] == 1
    assert secondo["prompt"]
    runtime.close(unload=False)


def test_preload_senza_residency_non_legge_pid_daemon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I test con factory fake non pagano pgrep/ps sul daemon."""
    spy = MagicMock(return_value=None)
    monkeypatch.setattr("assistente.ollama_runtime.pid_daemon_ollama", spy)
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


def test_preload_diagnostica_riusa_env_se_pid_invariato(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cambio PID ricalcola l'env; stesso PID salta ps/proc."""
    from assistente.host import InferenceHostOptions

    http = MagicMock()
    http.post.return_value.raise_for_status.return_value = None
    http.get.return_value.json.return_value = {"models": []}
    http.get.return_value.raise_for_status.return_value = None
    pid_spy = MagicMock(return_value=4242)
    env_spy = MagicMock(
        return_value={"OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_NUM_PARALLEL": "1"}
    )
    monkeypatch.setattr("assistente.ollama_runtime.pid_daemon_ollama", pid_spy)
    monkeypatch.setattr("assistente.ollama_runtime.environ_daemon_ollama", env_spy)
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



def test_descrivi_immagine_invia_base64_senza_thinking() -> None:
    """La vision usa il modello della sessione: immagine base64, think spento."""
    import base64

    http = MagicMock()
    http.post.return_value.json.return_value = {"response": "Una balena."}
    runtime = OllamaRuntime(
        llm_factory=_factory(),
        http_client=http,
        manage_residency=True,
    )
    runtime._available_models_cache = frozenset({"gemma4:12b"})
    testo = runtime.descrivi_immagine("default", b"PNG", "coda.png")
    assert testo == "Una balena."
    body = http.post.call_args.kwargs["json"]
    assert body["model"] == "gemma4:12b"
    assert body["images"] == [base64.b64encode(b"PNG").decode("ascii")]
    assert body["think"] is False
    runtime.close(unload=False)


def test_descrivi_immagine_senza_risposta_e_documento_non_leggibile() -> None:
    """Ollama assente: errore leggibile per il modello, non un crash del turno."""
    from assistente.lettori import DocumentoNonLeggibile

    runtime = OllamaRuntime(llm_factory=_factory(), manage_residency=False)
    with pytest.raises(DocumentoNonLeggibile, match="coda.png"):
        runtime.descrivi_immagine("default", b"PNG", "coda.png")
    runtime.close(unload=False)


def test_contesto_sessione_unico_per_tutte_le_route() -> None:
    """Preload, scheda, direct, ReAct e riassunto chiedono lo stesso num_ctx: niente reload."""
    from assistente.console import ConsoleUI
    from assistente.host import HostCapabilities, InferenceHostOptions
    from assistente.session import SessioneAgente

    memoria = 16 * 1024**3
    host = HostCapabilities(
        os="darwin",
        arch="arm64",
        chip_family="m4",
        chip_brand="Apple M4",
        memory_bytes=memoria,
        inference_memory_bytes=memoria,
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
    sessione_ctx = runtime.contesto_sessione("default")
    # q4_0 su 16 GB: la finestra di sessione sale già allo scalino espanso.
    assert sessione_ctx == 8192
    sessione = SessioneAgente(
        MagicMock(spec=ConsoleUI),
        llm_factory=_factory(),
        ollama_runtime=runtime,
        host=host,
    )
    try:
        for route in ("direct", "react"):
            profilo, _storico = sessione._profilo_con_storico(
                "default", route, None, "domanda breve", 128
            )
            assert profilo.context_tokens == sessione_ctx
    finally:
        sessione.close()
        runtime.close(unload=False)
