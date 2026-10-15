"""Central lifecycle of the Ollama clients and models."""

from __future__ import annotations

import base64
import json
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk
from langchain_ollama import OllamaLLM

from assistant.host import (
    HostCapabilities,
    InferenceHostOptions,
    can_expand_context,
    detect_host,
    environ_daemon_ollama,
    inference_host_options,
    neutral_host_options,
    ollama_base_url,
    pid_daemon_ollama,
    processor_looks_cpu_only,
)
from assistant.config import LLM_TEMPERATURE
from assistant.react_parser import extract_executable_react_block
from assistant.inference_profiles import (
    CONTEXT_LADDER,
    context_length_from_show,
    expanded_context,
    memory_warning,
    normalize_ollama_names,
    resolve_model_variant,
    select_installed_tag,
    session_context_tokens,
)
from assistant.prompt_budget import note_prompt_bytes
OLLAMA_CONTROL_TIMEOUT_SECONDS = 3.0
# Preload may need to load the weights into unified memory: 3s is not enough.
OLLAMA_PRELOAD_TIMEOUT_SECONDS = 180.0
# A model pull can take tens of minutes on a slow network.
OLLAMA_PULL_TIMEOUT_SECONDS = 3600.0
# Gemma 4: recommended top_k/top_p, low temperature for the tools' JSON;
# no presence/repeat penalty on code.
LLM_TOP_K = 64
LLM_TOP_P = 0.95
LLM_REPEAT_PENALTY = 1.0
LLM_PRESENCE_PENALTY = 0.0
LLM_NUM_BATCH = 512
LLM_CACHE_LIMIT = 4
# Vision: transcription and description of an image from the documents folder.
NUM_PREDICT_VISION = 700
PROMPT_VISION = (
    "Trascrivi fedelmente tutto il testo visibile nell'immagine. "
    "Poi descrivi in modo oggettivo cosa mostra: soggetti, tabelle, grafici, "
    "didascalie. Rispondi in italiano e non inventare dettagli non visibili."
)
_STREAM_FRAME_KEYS = (
    "thinking",
    "response",
    "done",
    "done_reason",
    "eval_count",
    "prompt_eval_count",
    "error",
    "eval_duration",
    "load_duration",
    "prompt_eval_duration",
    "total_duration",
)


def _as_stream_mapping(stream_resp: Any) -> dict[str, Any]:
    """Normalize an Ollama frame (dict or GenerateResponse) without a full dump."""
    if isinstance(stream_resp, dict):
        return stream_resp
    getter = getattr(stream_resp, "get", None)
    if callable(getter):
        return {
            key: getter(key)
            for key in _STREAM_FRAME_KEYS
            if getter(key) is not None
        }
    mapped: dict[str, Any] = {}
    for key in _STREAM_FRAME_KEYS:
        if hasattr(stream_resp, key):
            value = getattr(stream_resp, key)
            if value is not None:
                mapped[key] = value
    if mapped:
        return mapped
    dump = getattr(stream_resp, "model_dump", None)
    if callable(dump):
        dumped = dump()
        if isinstance(dumped, dict):
            return dumped
    return {}


class StreamingOllamaLLM(OllamaLLM):
    """OllamaLLM that forwards reasoning tokens as they arrive."""

    def attach_continuation(self, continuation: Any) -> None:
        """Attach the session's prefix cache to this wrapper."""
        object.__setattr__(self, "_continuazione_prefisso", continuation)

    def _continuation(self) -> Any | None:
        return getattr(self, "_continuazione_prefisso", None)

    def _generate_params(
        self,
        prompt: str,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Inject the per-call num_predict and the model's samplers into the Ollama options."""
        num_predict = kwargs.pop("num_predict", self.num_predict)
        # The JSON schema goes in the Ollama payload, not among the sampling options.
        schema_format = kwargs.pop("format", None)
        temperature = kwargs.pop("temperature", None)
        params = super()._generate_params(prompt, stop=stop, **kwargs)
        # LangChain may pass runtime keys back to generate(); Ollama rejects them.
        params.pop("num_predict", None)
        raw_options = params.get("options")
        if raw_options is None:
            options: dict[str, Any] = {}
        elif hasattr(raw_options, "model_dump"):
            options = {
                key: value
                for key, value in raw_options.model_dump().items()
                if value is not None
            }
        else:
            options = dict(raw_options)
        if num_predict is not None:
            options["num_predict"] = num_predict
        if temperature is not None:
            options["temperature"] = temperature
        options["repeat_penalty"] = LLM_REPEAT_PENALTY
        options["presence_penalty"] = LLM_PRESENCE_PENALTY
        options["top_k"] = LLM_TOP_K
        options["top_p"] = LLM_TOP_P
        options["num_batch"] = LLM_NUM_BATCH
        options.pop("format", None)
        params["options"] = options
        if schema_format is not None:
            params["format"] = schema_format
        return params

    def _stream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Emit thinking live, then the text, in a single pass over the Ollama stream."""
        reasoning = kwargs.get("reasoning", self.reasoning)
        already_retried = bool(kwargs.pop("_already_retried", False))
        continuation = self._continuation()
        if continuation is not None and "context" not in kwargs:
            prompt, extra = continuation.prepare(prompt)
            if extra:
                kwargs = {**kwargs, **extra}
        note_prompt_bytes(prompt)
        vis_chars = 0
        think_parts: list[str] = []
        retry_without_reasoning = False
        for stream_resp in self._create_generate_stream(prompt, stop, **kwargs):
            if isinstance(stream_resp, str):
                continue
            payload = _as_stream_mapping(stream_resp)
            raw_thinking = payload.get("thinking") or ""
            text = payload.get("response") or ""
            if raw_thinking:
                think_parts.append(str(raw_thinking))
            if reasoning:
                thinking = raw_thinking
            else:
                thinking = None
                executable = extract_executable_react_block(str(raw_thinking))
                if executable is not None and not text:
                    text = executable
            if text:
                vis_chars += len(text)
            recovered = False
            if payload.get("done") is True and vis_chars == 0 and think_parts:
                joined = "".join(think_parts)
                executable = extract_executable_react_block(joined)
                if executable is not None:
                    text = executable
                    vis_chars = len(text)
                    recovered = True
                elif reasoning and not already_retried:
                    retry_without_reasoning = True
            if payload.get("done") is True:
                generation_info: dict[str, Any] | None = dict(payload)
                if thinking:
                    generation_info["reasoning_content"] = thinking
            elif thinking:
                generation_info = {"reasoning_content": thinking}
            else:
                generation_info = None
            if run_manager is not None:
                if thinking:
                    run_manager.on_llm_new_token(
                        str(thinking),
                        thinking=True,
                        verbose=self.verbose,
                    )
                if text and not recovered:
                    run_manager.on_llm_new_token(text, verbose=self.verbose)
            yield GenerationChunk(text=text, generation_info=generation_info)
        if retry_without_reasoning:
            kwargs["reasoning"] = False
            kwargs["_already_retried"] = True
            yield from self._stream(
                prompt,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
            return

    def _stream_with_aggregation(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> GenerationChunk:
        """Aggregate texts and CoT into lists, without `GenerationChunk +=`."""
        del verbose
        texts: list[str] = []
        thoughts: list[str] = []
        done_info: dict[str, Any] | None = None
        received = False
        for chunk in self._stream(
            prompt,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        ):
            received = True
            if chunk.text:
                texts.append(chunk.text)
            info = chunk.generation_info
            if not info:
                continue
            reasoning = info.get("reasoning_content")
            if reasoning:
                thoughts.append(str(reasoning))
            if info.get("done") is True:
                done_info = dict(info)
        if not received:
            raise ValueError("No data received from Ollama stream.")
        generation_info = dict(done_info) if done_info else {}
        if thoughts:
            generation_info["thinking"] = "".join(thoughts)
        return GenerationChunk(
            text="".join(texts),
            generation_info=generation_info or None,
        )


@dataclass(frozen=True, slots=True)
class RuntimeKey:
    """Identify a LangChain wrapper without confusing it with the resident weights."""

    tag: str
    context_tokens: int
    num_gpu: int | None = None
    num_thread: int | None = None
    reasoning: bool = True


@dataclass(frozen=True, slots=True)
class RunningModel:
    """Minimal snapshot returned by `/api/ps`."""

    name: str
    size_bytes: int
    processor: str


class OllamaRuntime:
    """Owns the LLM wrappers, residency control and the Ollama HTTP connection."""

    def __init__(
        self,
        llm_factory: Callable[..., OllamaLLM] = StreamingOllamaLLM,
        *,
        base_url: str | None = None,
        keep_alive: str | None = None,
        http_client: httpx.Client | None = None,
        manage_residency: bool = True,
        host: HostCapabilities | None = None,
        host_options: InferenceHostOptions | None = None,
    ) -> None:
        """
        Initialize the runtime without loading any model.

        Args:
            llm_factory: Injectable constructor for tests.
            base_url: Ollama endpoint; None reads `OLLAMA_HOST`.
            keep_alive: Requested residency; None uses the host policy.
            http_client: Injectable control client; None creates one.
            manage_residency: If False, no automatic unloads are sent in tests.
            host: Machine snapshot; None detects only if residency is active.
            host_options: Explicit override of num_gpu/num_thread/keep-alive.
        """
        self._llm_factory = llm_factory
        self._host = host
        # Tests without residency stay on the historical profile (no num_gpu).
        if host_options is not None:
            self._host_options = host_options
        elif manage_residency:
            resolved_host = host if host is not None else detect_host()
            self._host = resolved_host
            self._host_options = inference_host_options(resolved_host)
        else:
            self._host_options = neutral_host_options()
        resolved_url = base_url if base_url is not None else ollama_base_url()
        self._base_url = resolved_url.rstrip("/")
        self._keep_alive = (
            keep_alive if keep_alive is not None else self._host_options.keep_alive
        )
        self._http_client = http_client or httpx.Client(
            base_url=self._base_url,
            timeout=OLLAMA_CONTROL_TIMEOUT_SECONDS,
        )
        self._owns_http_client = http_client is None
        self._manage_residency = manage_residency
        self._llm_cache: OrderedDict[RuntimeKey, OllamaLLM] = OrderedDict()
        self._active_tag: str | None = None
        self._last_control_error: str | None = None
        self._last_tag_fallback: str | None = None
        self._last_cpu_fallback: str | None = None
        self._last_memory_warning: str | None = None
        self._last_server_env_warning: str | None = None
        self._diag_pid: int | None = None
        self._diag_env_cached = False
        self._available_models_cache: frozenset[str] | None = None
        self._context_cap_for_tag: dict[str, int] = {}
        self._session_context_per_alias: dict[str, int] = {}
        self._preloaded_prefix: tuple[str, int, tuple[int, ...]] | None = None
        self._lock = threading.RLock()

    @property
    def kv_cache_type(self) -> str:
        """KV type requested from the daemon (`q4_0` enables the context steps)."""
        return self._host_options.kv_cache_type

    def session_context(self, alias: str) -> int:
        """
        Single per-alias session `num_ctx`, already expanded if memory allows it.

        Preload, card, direct, ReAct, summary and vision all use this
        window: Ollama reloads the weights on every `num_ctx` change.
        """
        cached = self._session_context_per_alias.get(alias)
        if cached is not None:
            return cached
        base = session_context_tokens(alias, host=self._host)
        host = self._host
        memory_bytes = (
            (host.inference_memory_bytes or host.memory_bytes) if host is not None else 0
        )
        chosen = expanded_context(
            base,
            alias=alias,
            kv_cache_type=self.kv_cache_type,
            memory_bytes=memory_bytes,
            weight_gb=resolve_model_variant(alias).size_gb,
            can_expand=host is not None and can_expand_context(host),
            tight=False,
            model_ceiling=self.model_context_cap(alias),
        )
        self._session_context_per_alias[alias] = chosen
        return chosen

    @property
    def active_tag(self) -> str | None:
        """Return the tag the application process considers active."""
        with self._lock:
            return self._active_tag

    @property
    def last_control_error(self) -> str | None:
        """Expose the last preload/unload error without interrupting inference."""
        with self._lock:
            return self._last_control_error

    @property
    def last_tag_fallback(self) -> str | None:
        """
        Describe an installed-tag fallback, if any.

        Returns:
            A readable message, or None if the preferred tag was used.
        """
        with self._lock:
            return self._last_tag_fallback

    @property
    def last_cpu_fallback(self) -> str | None:
        """Warning if `/api/ps` shows the model on CPU when a GPU was expected."""
        with self._lock:
            return self._last_cpu_fallback

    @property
    def last_memory_warning(self) -> str | None:
        """Warning if the preload was skipped because of swap risk."""
        with self._lock:
            return self._last_memory_warning

    @property
    def last_server_env_warning(self) -> str | None:
        """Warning if the Ollama daemon has no flash attention or num_parallel=1."""
        with self._lock:
            return self._last_server_env_warning

    @property
    def host_options(self) -> InferenceHostOptions:
        """Hardware options applied to the LangChain wrappers."""
        return self._host_options

    def invalidate_model_catalog(self) -> None:
        """Invalidate the `/api/tags` cache after a pull or a model change."""
        with self._lock:
            self._available_models_cache = None

    def ping(self) -> bool:
        """
        Check that the Ollama daemon answers on `/api/version`.

        Returns:
            True only if the HTTP response is 2xx and the JSON is an object.
        """
        payload = self._get_json("/api/version")
        return isinstance(payload, dict)

    def list_available_model_names(self) -> frozenset[str]:
        """
        Expose the already normalized `/api/tags` catalog.

        Returns:
            Empty set if Ollama is unreachable.
        """
        return self._list_available_model_names()

    def pull_model(
        self,
        tag: str,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        """
        Download an Ollama artifact via a streaming `POST /api/pull`.

        The progress callback is throttled: the same state and the same whole
        percentage are not repeated, to avoid flooding the console.

        Args:
            tag: Tag to pull (the profile's preferred one or a library name).
            on_progress: Optional callback with a readable status line.

        Raises:
            httpx.HTTPError: Failed transport or HTTP status.
            RuntimeError: Ollama payload with an `error` field, or an unreadable stream.
        """
        progress = on_progress or (lambda _message: None)
        last_status = ""
        last_percent = -1
        try:
            with self._http_client.stream(
                "POST",
                "/api/pull",
                json={"name": tag, "stream": True},
                timeout=OLLAMA_PULL_TIMEOUT_SECONDS,
            ) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines():
                    payload = _parse_pull_line(tag, raw_line)
                    if payload is None:
                        continue
                    error_message = payload.get("error")
                    if error_message:
                        raise RuntimeError(str(error_message))
                    status, percent = _pull_progress_state(payload)
                    # Avoid dozens of UI refreshes for the same byte range.
                    if status == last_status and percent == last_percent:
                        continue
                    last_status = status
                    last_percent = percent
                    progress(_format_pull_progress(tag, status, percent))
        except httpx.HTTPError as error:
            self._last_control_error = f"{type(error).__name__}: {error}"
            raise
        self.invalidate_model_catalog()
        self._last_control_error = None

    def get_llm(
        self,
        alias: str,
        context_tokens: int,
        output_tokens: int,
        *,
        reasoning: bool = True,
    ) -> OllamaLLM:
        """
        Return a cached wrapper and keep only one tag logically active.

        Args:
            alias: fast/default alias (fixed quantization in the matrix).
            context_tokens: Ollama window for this route.
            output_tokens: Generation limit.
            reasoning: Native Ollama CoT; False only for the SERIOUS/CRITICAL thermal cut.

        Returns:
            `OllamaLLM` configured with zero temperature and an explicit keep-alive.
        """
        variant = resolve_model_variant(alias)
        tag = self._resolve_installed_tag(variant.tag)
        key = RuntimeKey(
            tag,
            context_tokens,
            self._host_options.num_gpu,
            self._host_options.num_thread,
            reasoning,
        )
        with self._lock:
            self._switch_active_model(key.tag)
            # Identical profiles share the wrapper and the LangChain HTTP pool.
            cached = self._llm_cache.get(key)
            if cached is not None:
                self._llm_cache.move_to_end(key)
                self._apply_num_predict(cached, output_tokens)
                return cached
            factory_kwargs: dict[str, Any] = {
                "model": key.tag,
                "base_url": self._base_url,
                "temperature": LLM_TEMPERATURE,
                "num_ctx": key.context_tokens,
                "num_predict": output_tokens,
                "reasoning": key.reasoning,
                "keep_alive": self._keep_alive,
                "repeat_penalty": LLM_REPEAT_PENALTY,
                "top_k": LLM_TOP_K,
                "top_p": LLM_TOP_P,
            }
            # None leaves the default to Ollama: do not send the field.
            if key.num_gpu is not None:
                factory_kwargs["num_gpu"] = key.num_gpu
            if key.num_thread is not None:
                factory_kwargs["num_thread"] = key.num_thread
            created = self._llm_factory(**factory_kwargs)
            self._llm_cache[key] = created
            while len(self._llm_cache) > LLM_CACHE_LIMIT:
                self._llm_cache.popitem(last=False)
            return created

    @staticmethod
    def _apply_num_predict(llm: OllamaLLM, output_tokens: int) -> None:
        """Align the generation cap without rebuilding the HTTP client."""
        try:
            llm.num_predict = output_tokens
        except (TypeError, ValueError, AttributeError):
            # Mocked factories or frozen wrappers: the caller passes num_predict to the stream.
            object.__setattr__(llm, "num_predict", output_tokens)

    def preload(self, alias: str) -> str | None:
        """
        Preload the weights if the profile fits in unified memory.

        If `memory_warning` is set, the resident model is left alone.
        Otherwise it runs an eager unload/load.

        Args:
            alias: Model alias (quantization derived from the fixed matrix).

        Returns:
            Optional memory warning; server errors go to `last_control_error`.
        """
        variant = resolve_model_variant(alias)
        warning = memory_warning(variant, host=self._host)
        with self._lock:
            self._last_memory_warning = warning
        if warning is not None:
            self._last_tag_fallback = None
            self._last_control_error = None
            self._last_cpu_fallback = None
            return warning
        tag = self._resolve_installed_tag(variant.tag)
        context_tokens = self.session_context(alias)
        from assistant.direct_pipeline import DIRECT_SYSTEM_PREFIX

        with self._lock:
            self._switch_active_model(tag)
            self._post_generate(
                tag,
                keep_alive=self._keep_alive,
                context_tokens=context_tokens,
            )
            # Same text the direct prompt opens with when there is no project memory.
            prefix = f"{DIRECT_SYSTEM_PREFIX}\n\n"
            payload = self._post_generate(
                tag,
                keep_alive=self._keep_alive,
                context_tokens=context_tokens,
                prompt=prefix,
                num_predict=1,
            )
            self._store_prefix(prefix, context_tokens, payload)
        self._last_cpu_fallback = self._cpu_fallback_diagnostics(tag)
        self._update_server_env_diagnostics()
        return warning

    def describe_image(self, alias: str, image: bytes, name: str) -> str:
        """
        Read an image with the active model's vision (no second model).

        Args:
            alias: Session alias, already resident after the preload.
            image: File bytes (png, jpg, webp).
            name: Relative path, only for messages.

        Returns:
            Transcription and description of the image.

        Raises:
            UnreadableDocument: Ollama unreachable or empty answer.
        """
        from assistant.readers import UnreadableDocument

        variant = resolve_model_variant(alias)
        tag = self._resolve_installed_tag(variant.tag)
        # Same num_ctx as the preload: Ollama does not reload the weights for vision.
        context_tokens = self.session_context(alias)
        with self._lock:
            self._switch_active_model(tag)
            payload = self._post_generate(
                tag,
                keep_alive=self._keep_alive,
                context_tokens=context_tokens,
                prompt=PROMPT_VISION,
                num_predict=NUM_PREDICT_VISION,
                images=[base64.b64encode(image).decode("ascii")],
                think=False,
                timeout_seconds=OLLAMA_PRELOAD_TIMEOUT_SECONDS,
            )
        text = str(payload.get("response", "")).strip() if payload else ""
        if not text:
            detail = self._last_control_error or "risposta vuota"
            raise UnreadableDocument(f"Immagine non leggibile: {name} ({detail})")
        return text

    def unload_active(self) -> None:
        """Best-effort unload of the active model to free unified memory."""
        with self._lock:
            # No active tag means the runtime must not contact Ollama.
            if self._active_tag is None:
                return
            self._post_generate(self._active_tag, keep_alive=0)
            self._active_tag = None

    def running_models(self) -> tuple[RunningModel, ...]:
        """
        Query `/api/ps` for hardware diagnostics.

        Returns:
            Resident models; an empty tuple when Ollama is unreachable.
        """
        payload = self._get_json("/api/ps")
        if not isinstance(payload, dict):
            return ()
        models: list[RunningModel] = []
        for raw_model in payload.get("models", []):
            models.append(
                RunningModel(
                    name=str(raw_model.get("name", "")),
                    size_bytes=int(raw_model.get("size", 0) or 0),
                    processor=str(raw_model.get("processor", "")),
                )
            )
        return tuple(models)

    def context_preloaded_prefix(
        self,
        prompt: str,
        context_tokens: int,
    ) -> tuple[int, ...] | None:
        """Ids of the startup prefill, if prompt and window match."""
        stored = self._preloaded_prefix
        if stored is None:
            return None
        saved_prompt, saved_context, ids = stored
        if saved_prompt == prompt and saved_context == context_tokens and ids:
            return ids
        return None

    def _store_prefix(
        self,
        prompt: str,
        context_tokens: int,
        payload: dict[str, Any] | None,
    ) -> None:
        context = payload.get("context") if isinstance(payload, dict) else None
        if not isinstance(context, list) or not context:
            return
        self._preloaded_prefix = (
            prompt,
            context_tokens,
            tuple(int(item) for item in context),
        )

    def prefill_raw_prefix(
        self,
        alias: str,
        context_tokens: int,
        prompt: str,
    ) -> tuple[int, ...] | None:
        """
        Prefill a prefix different from the startup one.

        Control timeout: a failure must not block the turn.
        """
        if not self._manage_residency or not prompt.strip():
            return None
        variant = resolve_model_variant(alias)
        tag = self._resolve_installed_tag(variant.tag)
        payload = self._post_generate(
            tag,
            keep_alive=self._keep_alive,
            context_tokens=context_tokens,
            prompt=prompt,
            num_predict=0,
            raw=True,
            timeout_seconds=OLLAMA_CONTROL_TIMEOUT_SECONDS,
        )
        context = payload.get("context") if isinstance(payload, dict) else None
        if not isinstance(context, list) or not context:
            return None
        return tuple(int(item) for item in context)

    def model_context_cap(self, alias: str) -> int:
        """Context declared by `ollama show`, or the last step of the ladder."""
        ceiling = CONTEXT_LADDER[-1]
        if not self._manage_residency:
            return ceiling
        variant = resolve_model_variant(alias)
        tag = self._resolve_installed_tag(variant.tag)
        cached = self._context_cap_for_tag.get(tag)
        if cached is not None:
            return cached
        try:
            response = self._http_client.post("/api/show", json={"name": tag})
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError, TypeError):
            self._context_cap_for_tag[tag] = ceiling
            return ceiling
        if not isinstance(payload, dict):
            return ceiling
        declared = context_length_from_show(payload)
        if declared is None:
            self._context_cap_for_tag[tag] = ceiling
            return ceiling
        limited = min(ceiling, declared)
        self._context_cap_for_tag[tag] = limited
        return limited

    def close(self, *, unload: bool = True) -> None:
        """
        Release the model and the control connection.

        Args:
            unload: If True, free the weights immediately; False lets keep-alive expire.
        """
        # Interactive shutdown favours freeing the laptop's memory.
        if unload:
            self.unload_active()
        # An injected client remains the caller's responsibility.
        if self._owns_http_client:
            self._http_client.close()

    def _resolve_installed_tag(self, preferred_tag: str) -> str:
        """
        Map the profile tag onto an artifact actually present in Ollama.

        Args:
            preferred_tag: Tag declared by the hardware-aware profile.

        Returns:
            The chosen installed tag, or the preferred one if the catalog is
            unavailable or no fallback is present.
        """
        # Tests with a mocked factory must not depend on the local catalog.
        if not self._manage_residency:
            with self._lock:
                self._last_tag_fallback = None
            return preferred_tag
        available = self._list_available_model_names()
        # Without a catalog we do not invent fallbacks: the preload will report the real error.
        if not available:
            with self._lock:
                self._last_tag_fallback = None
            return preferred_tag
        selected = select_installed_tag(preferred_tag, available)
        with self._lock:
            # Tell the CLI when a library alias is used (e.g. gemma4).
            if selected != preferred_tag:
                self._last_tag_fallback = (
                    f"Modello {preferred_tag} assente: uso {selected}."
                )
            else:
                self._last_tag_fallback = None
        return selected

    def _list_available_model_names(self) -> frozenset[str]:
        """
        Read and store the model names from `/api/tags`, already normalized.

        Returns:
            Empty set if Ollama is unreachable.
        """
        with self._lock:
            if self._available_models_cache is not None:
                return self._available_models_cache
            payload = self._get_json("/api/tags")
            if not isinstance(payload, dict):
                return frozenset()
            names: set[str] = set()
            for raw_model in payload.get("models", []):
                for key in ("name", "model"):
                    value = raw_model.get(key)
                    if value:
                        names.add(str(value))
            cached = frozenset(normalize_ollama_names(names))
            self._available_models_cache = cached
            return cached

    def _get_json(self, path: str) -> Any | None:
        """Best-effort JSON GET; None if the daemon does not answer or the body is invalid."""
        try:
            response = self._http_client.get(path)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError, TypeError) as error:
            self._last_control_error = f"{type(error).__name__}: {error}"
            return None

    def _switch_active_model(self, next_tag: str) -> None:
        """
        Unload the previous tag before selecting a different one.

        Args:
            next_tag: Model that will serve the next generation.
        """
        # Reusing the same artifact avoids reloads between different output profiles.
        if self._active_tag == next_tag:
            return
        # On 16 GB two resident artifacts can cause swapping during the switch.
        if (
            self._manage_residency
            and self._active_tag is not None
        ):
            self._post_generate(self._active_tag, keep_alive=0)
        self._active_tag = next_tag

    def _post_generate(
        self,
        tag: str,
        *,
        keep_alive: str | int,
        context_tokens: int | None = None,
        prompt: str = "",
        num_predict: int | None = None,
        raw: bool = False,
        timeout_seconds: float | None = None,
        images: list[str] | None = None,
        think: bool | None = None,
    ) -> dict[str, Any] | None:
        """
        Send the preload/unload control without propagating availability errors.

        Args:
            tag: Ollama tag to control.
            keep_alive: Positive duration for preload, or zero for unload.
            context_tokens: `num_ctx` to align with the LangChain wrappers.
            prompt: Text to prefill (empty = load the weights only).
            num_predict: Generation cap; 1 on the prefix warmup.
            raw: If True, the prompt bypasses the Ollama template.
            timeout_seconds: Explicit cap; None uses a quick unload or a long preload.
            images: Base64 images for vision; None for plain control calls.
            think: Native CoT; None leaves the model default.

        Returns:
            JSON payload, or None if residency is off or the call fails.
        """
        # Tests with residency disabled must not open local sockets.
        if not self._manage_residency:
            return None
        body: dict[str, Any] = {
            "model": tag,
            "prompt": prompt,
            "stream": False,
            "keep_alive": keep_alive,
        }
        if raw:
            body["raw"] = True
        if images:
            body["images"] = images
        if think is not None:
            body["think"] = think
        options: dict[str, Any] = {}
        if keep_alive not in {0, "0"}:
            if context_tokens is not None:
                options["num_ctx"] = context_tokens
            if num_predict is not None:
                options["num_predict"] = num_predict
            if self._host_options.num_gpu is not None:
                options["num_gpu"] = self._host_options.num_gpu
            if self._host_options.num_thread is not None:
                options["num_thread"] = self._host_options.num_thread
        if options:
            body["options"] = options
        if timeout_seconds is not None:
            request_timeout = timeout_seconds
        elif keep_alive in {0, "0"}:
            request_timeout = OLLAMA_CONTROL_TIMEOUT_SECONDS
        else:
            request_timeout = OLLAMA_PRELOAD_TIMEOUT_SECONDS
        try:
            response = self._http_client.post(
                "/api/generate",
                json=body,
                timeout=request_timeout,
            )
            response.raise_for_status()
            self._last_control_error = None
            payload = response.json()
        except (httpx.HTTPError, ValueError, TypeError) as error:
            # The real generation will produce a more useful error if Ollama is offline.
            self._last_control_error = f"{type(error).__name__}: {error}"
            return None
        return payload if isinstance(payload, dict) else None

    def _expects_gpu(self) -> bool:
        """True if the host policy requested GPU offload (`num_gpu` ≠ 0)."""
        num_gpu = self._host_options.num_gpu
        return num_gpu is not None and num_gpu != 0

    def _update_server_env_diagnostics(self) -> None:
        """Re-read FA/num_parallel only if the daemon PID changed."""
        if not self._manage_residency:
            return
        pid = pid_daemon_ollama()
        if self._diag_env_cached and pid == self._diag_pid:
            return
        self._diag_pid = pid
        self._diag_env_cached = True
        if pid is None:
            self._last_server_env_warning = None
            return
        self._last_server_env_warning = server_env_diagnostics(
            environ_daemon_ollama(pid)
        )

    def _cpu_fallback_diagnostics(self, tag: str) -> str | None:
        """
        Warn if Ollama loaded the model on the CPU instead of the GPU.

        Args:
            tag: Artifact just preloaded.

        Returns:
            System message, or None if the GPU is fine or diagnostics are missing.
        """
        # Without an expected GPU (CPU-only Linux, Intel, tests) the warning would be noise.
        if not self._expects_gpu() or not self._manage_residency:
            return None
        models = self.running_models()
        if not models:
            return None
        selected = next(
            (item for item in models if tag in item.name or item.name in tag),
            models[0],
        )
        if not processor_looks_cpu_only(selected.processor):
            return None
        return (
            f"Modello su CPU ({selected.processor}): verificare Metal/CUDA. "
            f"Tag: {selected.name}."
        )


def _parse_pull_line(tag: str, raw_line: str | bytes) -> dict[str, Any] | None:
    """
    Parse an NDJSON line from `/api/pull`.

    Returns:
        Object payload, or None for empty lines.

    Raises:
        RuntimeError: Unreadable JSON or unexpected type.
    """
    if isinstance(raw_line, bytes):
        line = raw_line.decode("utf-8", errors="replace").strip()
    else:
        line = raw_line.strip()
    if not line:
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Risposta pull non JSON per {tag}: {line[:120]}"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"Payload pull inatteso per {tag}.")
    return payload


def _pull_progress_state(payload: dict[str, Any]) -> tuple[str, int]:
    """Return the textual state and the whole percentage (0-100, or -1 if unknown)."""
    status = str(payload.get("status") or "download")
    total_raw = payload.get("total")
    if not isinstance(total_raw, (int, float)) or total_raw <= 0:
        return status, -1
    try:
        completed = float(payload.get("completed") or 0)
    except (TypeError, ValueError):
        completed = 0.0
    percent = int(min(100.0, max(0.0, (completed / float(total_raw)) * 100.0)))
    return status, percent


def _format_pull_progress(tag: str, status: str, percent: int) -> str:
    """Reduce state/percentage to one line for the console."""
    if percent < 0:
        return f"{tag}: {status}"
    return f"{tag}: {status} {percent}%"


_LAUNCHCTL_HINT = (
    "launchctl setenv OLLAMA_FLASH_ATTENTION 1; "
    "launchctl setenv OLLAMA_KV_CACHE_TYPE q4_0; "
    "launchctl setenv OLLAMA_NUM_PARALLEL 1"
)


def server_env_diagnostics(
    environ: dict[str, str] | None = None,
) -> str | None:
    """
    Warn if the Ollama daemon has no flash attention or num_parallel=1.

    Args:
        environ: Injectable mapping; None reads the daemon process env.

    Returns:
        Message for the CLI, or None if fine or the process is not visible.
    """
    if environ is None:
        pid = pid_daemon_ollama()
        if pid is None:
            return None
        environ = environ_daemon_ollama(pid)
    if not environ:
        return None
    flash = str(environ.get("OLLAMA_FLASH_ATTENTION", "")).strip().casefold()
    parallel = str(environ.get("OLLAMA_NUM_PARALLEL", "1")).strip()
    problems: list[str] = []
    if flash not in {"1", "true"}:
        problems.append("OLLAMA_FLASH_ATTENTION non è 1")
    if parallel not in {"1", ""}:
        problems.append(f"OLLAMA_NUM_PARALLEL={parallel} (atteso 1)")
    if not problems:
        return None
    return (
        "Daemon Ollama: "
        + "; ".join(problems)
        + f". Per un `serve` già avviato: {_LAUNCHCTL_HINT}."
    )
