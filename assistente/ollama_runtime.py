"""Ciclo di vita centralizzato dei client e dei modelli Ollama."""

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

from assistente.host import (
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
from assistente.config import TEMPERATURA_LLM
from assistente.react_parser import estrai_blocco_react_eseguibile
from assistente.inference_profiles import (
    CONTEXT_LADDER,
    context_length_from_show,
    contesto_espanso,
    memory_warning,
    normalize_ollama_names,
    resolve_model_variant,
    select_installed_tag,
    session_context_tokens,
)
from assistente.prompt_budget import note_prompt_bytes
OLLAMA_CONTROL_TIMEOUT_SECONDS = 3.0
# Il preload può richiedere il load dei pesi su unified memory: 3s sono insufficienti.
OLLAMA_PRELOAD_TIMEOUT_SECONDS = 180.0
# Un pull del modello può durare decine di minuti su rete lenta.
OLLAMA_PULL_TIMEOUT_SECONDS = 3600.0
LLM_TEMPERATURE = TEMPERATURA_LLM
# Gemma 4: top_k/top_p consigliati, temperatura bassa per il JSON dei tool;
# niente presence/repeat penalty sul codice.
LLM_TOP_K = 64
LLM_TOP_P = 0.95
LLM_REPEAT_PENALTY = 1.0
LLM_PRESENCE_PENALTY = 0.0
LLM_NUM_BATCH = 512
LLM_CACHE_LIMIT = 4
# Vision: trascrizione e descrizione di un'immagine della cartella documenti.
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
    """Normalizza un frame Ollama (dict o GenerateResponse) senza dump completo."""
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
    """OllamaLLM che inoltra i token di ragionamento mentre arrivano."""

    def collega_continuazione(self, continuazione: Any) -> None:
        """Associa la cache di prefisso della sessione a questo wrapper."""
        object.__setattr__(self, "_continuazione_prefisso", continuazione)

    def _continuazione(self) -> Any | None:
        return getattr(self, "_continuazione_prefisso", None)

    def _generate_params(
        self,
        prompt: str,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Inietta num_predict per-chiamata e i sampler del modello nelle options Ollama."""
        num_predict = kwargs.pop("num_predict", self.num_predict)
        # Lo schema JSON sta nel payload Ollama, non fra le options di sampling.
        formato = kwargs.pop("format", None)
        temperatura = kwargs.pop("temperature", None)
        params = super()._generate_params(prompt, stop=stop, **kwargs)
        # LangChain può rilanciare chiavi runtime a generate(); Ollama le rifiuta.
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
        if temperatura is not None:
            options["temperature"] = temperatura
        options["repeat_penalty"] = LLM_REPEAT_PENALTY
        options["presence_penalty"] = LLM_PRESENCE_PENALTY
        options["top_k"] = LLM_TOP_K
        options["top_p"] = LLM_TOP_P
        options["num_batch"] = LLM_NUM_BATCH
        options.pop("format", None)
        params["options"] = options
        if formato is not None:
            params["format"] = formato
        return params

    def _stream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Emette thinking live, poi il testo, da un solo passaggio sullo stream Ollama."""
        reasoning = kwargs.get("reasoning", self.reasoning)
        already_retried = bool(kwargs.pop("_already_retried", False))
        continuazione = self._continuazione()
        if continuazione is not None and "context" not in kwargs:
            prompt, extra = continuazione.prepara(prompt)
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
                eseguibile = estrai_blocco_react_eseguibile(str(raw_thinking))
                if eseguibile is not None and not text:
                    text = eseguibile
            if text:
                vis_chars += len(text)
            recovered = False
            if payload.get("done") is True and vis_chars == 0 and think_parts:
                joined = "".join(think_parts)
                eseguibile = estrai_blocco_react_eseguibile(joined)
                if eseguibile is not None:
                    text = eseguibile
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
        """Aggrega testi e CoT in liste, senza `GenerationChunk +=`."""
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
    """Identifica un wrapper LangChain senza confonderlo con i pesi residenti."""

    tag: str
    context_tokens: int
    num_gpu: int | None = None
    num_thread: int | None = None
    reasoning: bool = True


@dataclass(frozen=True, slots=True)
class RunningModel:
    """Snapshot minimale restituito da `/api/ps`."""

    name: str
    size_bytes: int
    processor: str


class OllamaRuntime:
    """Possiede wrapper LLM, controllo residenza e connessione HTTP Ollama."""

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
        Inizializza il runtime senza caricare modelli.

        Args:
            llm_factory: Costruttore iniettabile per test.
            base_url: Endpoint Ollama; None legge `OLLAMA_HOST`.
            keep_alive: Residenza richiesta; None usa la policy del host.
            http_client: Client di controllo iniettabile; None ne crea uno.
            manage_residency: Se False non invia unload automatici nei test.
            host: Snapshot macchina; None rileva solo se la residency è attiva.
            host_options: Override esplicito di num_gpu/num_thread/keep-alive.
        """
        self._llm_factory = llm_factory
        self._host = host
        # I test senza residency restano sul profilo storico (niente num_gpu).
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
        self._tetto_contesto_per_tag: dict[str, int] = {}
        self._contesto_sessione_per_alias: dict[str, int] = {}
        self._prefisso_precaricato: tuple[str, int, tuple[int, ...]] | None = None
        self._lock = threading.RLock()

    @property
    def kv_cache_type(self) -> str:
        """Tipo di KV richiesto al daemon (`q4_0` abilita gli scalini di contesto)."""
        return self._host_options.kv_cache_type

    def contesto_sessione(self, alias: str) -> int:
        """
        Unico `num_ctx` di sessione per alias, già espanso se la memoria lo consente.

        Preload, scheda, direct, ReAct, riassunto e vision usano tutti questa
        finestra: Ollama ricarica i pesi a ogni cambio di `num_ctx`.
        """
        cached = self._contesto_sessione_per_alias.get(alias)
        if cached is not None:
            return cached
        base = session_context_tokens(alias, host=self._host)
        host = self._host
        memory_bytes = (
            (host.inference_memory_bytes or host.memory_bytes) if host is not None else 0
        )
        scelto = contesto_espanso(
            base,
            alias=alias,
            kv_cache_type=self.kv_cache_type,
            memory_bytes=memory_bytes,
            weight_gb=resolve_model_variant(alias).size_gb,
            can_expand=host is not None and can_expand_context(host),
            tight=False,
            model_ceiling=self.tetto_contesto_modello(alias),
        )
        self._contesto_sessione_per_alias[alias] = scelto
        return scelto

    @property
    def active_tag(self) -> str | None:
        """Restituisce il tag considerato attivo dal processo applicativo."""
        with self._lock:
            return self._active_tag

    @property
    def last_control_error(self) -> str | None:
        """Espone l'ultimo errore preload/unload senza interrompere l'inferenza."""
        with self._lock:
            return self._last_control_error

    @property
    def last_tag_fallback(self) -> str | None:
        """
        Descrive un eventuale fallback di tag installato.

        Returns:
            Messaggio leggibile oppure None se è stato usato il tag preferito.
        """
        with self._lock:
            return self._last_tag_fallback

    @property
    def last_cpu_fallback(self) -> str | None:
        """Avviso se `/api/ps` mostra il modello su CPU mentre ci si aspettava GPU."""
        with self._lock:
            return self._last_cpu_fallback

    @property
    def last_memory_warning(self) -> str | None:
        """Avviso se il preload è stato saltato per rischio swap."""
        with self._lock:
            return self._last_memory_warning

    @property
    def last_server_env_warning(self) -> str | None:
        """Avviso se il daemon Ollama non ha flash attention o num_parallel=1."""
        with self._lock:
            return self._last_server_env_warning

    @property
    def host_options(self) -> InferenceHostOptions:
        """Opzioni hardware applicate ai wrapper LangChain."""
        return self._host_options

    def invalidate_model_catalog(self) -> None:
        """Invalida la cache di `/api/tags` dopo pull o cambio modelli."""
        with self._lock:
            self._available_models_cache = None

    def ping(self) -> bool:
        """
        Verifica che il daemon Ollama risponda su `/api/version`.

        Returns:
            True solo se la risposta HTTP è 2xx e il JSON è un oggetto.
        """
        payload = self._get_json("/api/version")
        return isinstance(payload, dict)

    def list_available_model_names(self) -> frozenset[str]:
        """
        Espone il catalogo `/api/tags` già normalizzato.

        Returns:
            Insieme vuoto se Ollama non è raggiungibile.
        """
        return self._list_available_model_names()

    def pull_model(
        self,
        tag: str,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        """
        Scarica un artefatto Ollama via `POST /api/pull` in streaming.

        Il callback di progresso è throttled: stesso stato e stessa percentuale
        intera non vengono ripetuti, per non saturare la console.

        Args:
            tag: Tag da pullare (preferito del profilo o nome libreria).
            on_progress: Callback opzionale con una riga di stato leggibile.

        Raises:
            httpx.HTTPError: Trasporto o status HTTP non riuscito.
            RuntimeError: Payload Ollama con campo `error` o stream illeggibile.
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
                    status, percent = _stato_progresso_pull(payload)
                    # Evita decine di refresh UI per lo stesso byte-range.
                    if status == last_status and percent == last_percent:
                        continue
                    last_status = status
                    last_percent = percent
                    progress(_formatta_progresso_pull(tag, status, percent))
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
        Restituisce un wrapper cacheato e mantiene un solo tag logicamente attivo.

        Args:
            alias: Alias fast/default (quant fissa nella matrice).
            context_tokens: Finestra Ollama per questa route.
            output_tokens: Limite generazione.
            reasoning: CoT nativo Ollama; False solo per il taglio termico SERIO/CRITICO.

        Returns:
            `OllamaLLM` configurato con temperatura zero e keep-alive esplicito.
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
            # Profili identici condividono wrapper e pool HTTP LangChain.
            cached = self._llm_cache.get(key)
            if cached is not None:
                self._llm_cache.move_to_end(key)
                self._applica_num_predict(cached, output_tokens)
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
            # None lascia a Ollama il default: non inviare il campo.
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
    def _applica_num_predict(llm: OllamaLLM, output_tokens: int) -> None:
        """Allinea il tetto della generazione senza ricostruire il client HTTP."""
        try:
            llm.num_predict = output_tokens
        except (TypeError, ValueError, AttributeError):
            # Factory mockate o wrapper frozen: il chiamante passa num_predict allo stream.
            object.__setattr__(llm, "num_predict", output_tokens)

    def preload(self, alias: str) -> str | None:
        """
        Precarica i pesi se il profilo rientra nella memoria unificata.

        Se `memory_warning` è attivo non tocca il modello residente.
        Altrimenti esegue unload/load eager.

        Args:
            alias: Alias modello (quant derivata dalla matrice fissa).

        Returns:
            Avviso memoria opzionale; errori server in `last_control_error`.
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
        context_tokens = self.contesto_sessione(alias)
        from assistente.direct_pipeline import PREFISSO_SISTEMA_DIRECT

        with self._lock:
            self._switch_active_model(tag)
            self._post_generate(
                tag,
                keep_alive=self._keep_alive,
                context_tokens=context_tokens,
            )
            # Stesso testo con cui apre il prompt direct senza memoria progetto.
            prefisso = f"{PREFISSO_SISTEMA_DIRECT}\n\n"
            payload = self._post_generate(
                tag,
                keep_alive=self._keep_alive,
                context_tokens=context_tokens,
                prompt=prefisso,
                num_predict=1,
            )
            self._memorizza_prefisso(prefisso, context_tokens, payload)
        self._last_cpu_fallback = self._diagnostica_cpu_fallback(tag)
        self._aggiorna_diagnostica_server_env()
        return warning

    def descrivi_immagine(self, alias: str, immagine: bytes, nome: str) -> str:
        """
        Legge un'immagine con la vision del modello attivo (niente secondo modello).

        Args:
            alias: Alias della sessione, già residente dopo il preload.
            immagine: Byte del file (png, jpg, webp).
            nome: Path relativo, solo per i messaggi.

        Returns:
            Trascrizione e descrizione dell'immagine.

        Raises:
            DocumentoNonLeggibile: Ollama non raggiungibile o risposta vuota.
        """
        from assistente.lettori import DocumentoNonLeggibile

        variant = resolve_model_variant(alias)
        tag = self._resolve_installed_tag(variant.tag)
        # Stesso num_ctx del preload: Ollama non ricarica i pesi per la vision.
        context_tokens = self.contesto_sessione(alias)
        with self._lock:
            self._switch_active_model(tag)
            payload = self._post_generate(
                tag,
                keep_alive=self._keep_alive,
                context_tokens=context_tokens,
                prompt=PROMPT_VISION,
                num_predict=NUM_PREDICT_VISION,
                images=[base64.b64encode(immagine).decode("ascii")],
                think=False,
                timeout_seconds=OLLAMA_PRELOAD_TIMEOUT_SECONDS,
            )
        testo = str(payload.get("response", "")).strip() if payload else ""
        if not testo:
            dettaglio = self._last_control_error or "risposta vuota"
            raise DocumentoNonLeggibile(f"Immagine non leggibile: {nome} ({dettaglio})")
        return testo

    def unload_active(self) -> None:
        """Scarica best-effort il modello attivo per liberare memoria unificata."""
        with self._lock:
            # Nessun tag attivo significa che il runtime non deve contattare Ollama.
            if self._active_tag is None:
                return
            self._post_generate(self._active_tag, keep_alive=0)
            self._active_tag = None

    def running_models(self) -> tuple[RunningModel, ...]:
        """
        Interroga `/api/ps` per la diagnostica hardware.

        Returns:
            Modelli residenti; tupla vuota quando Ollama non è raggiungibile.
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

    def context_prefisso_precaricato(
        self,
        prompt: str,
        context_tokens: int,
    ) -> tuple[int, ...] | None:
        """Id del prefill di avvio, se prompt e finestra coincidono."""
        stored = self._prefisso_precaricato
        if stored is None:
            return None
        saved_prompt, saved_context, ids = stored
        if saved_prompt == prompt and saved_context == context_tokens and ids:
            return ids
        return None

    def _memorizza_prefisso(
        self,
        prompt: str,
        context_tokens: int,
        payload: dict[str, Any] | None,
    ) -> None:
        context = payload.get("context") if isinstance(payload, dict) else None
        if not isinstance(context, list) or not context:
            return
        self._prefisso_precaricato = (
            prompt,
            context_tokens,
            tuple(int(item) for item in context),
        )

    def prefill_prefisso_raw(
        self,
        alias: str,
        context_tokens: int,
        prompt: str,
    ) -> tuple[int, ...] | None:
        """
        Prefilla un prefisso diverso da quello di avvio.

        Timeout di controllo: un fallimento non deve bloccare il turno.
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

    def tetto_contesto_modello(self, alias: str) -> int:
        """Contesto dichiarato da `ollama show`, oppure l'ultimo scalino della ladder."""
        ceiling = CONTEXT_LADDER[-1]
        if not self._manage_residency:
            return ceiling
        variant = resolve_model_variant(alias)
        tag = self._resolve_installed_tag(variant.tag)
        cached = self._tetto_contesto_per_tag.get(tag)
        if cached is not None:
            return cached
        try:
            response = self._http_client.post("/api/show", json={"name": tag})
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError, TypeError):
            self._tetto_contesto_per_tag[tag] = ceiling
            return ceiling
        if not isinstance(payload, dict):
            return ceiling
        declared = context_length_from_show(payload)
        if declared is None:
            self._tetto_contesto_per_tag[tag] = ceiling
            return ceiling
        limited = min(ceiling, declared)
        self._tetto_contesto_per_tag[tag] = limited
        return limited

    def close(self, *, unload: bool = True) -> None:
        """
        Rilascia modello e connessione di controllo.

        Args:
            unload: Se True libera subito i pesi; False lascia scadere keep-alive.
        """
        # Lo shutdown interattivo privilegia il rilascio della memoria del portatile.
        if unload:
            self.unload_active()
        # Un client iniettato resta responsabilità del chiamante.
        if self._owns_http_client:
            self._http_client.close()

    def _resolve_installed_tag(self, preferred_tag: str) -> str:
        """
        Mappa il tag del profilo su un artefatto realmente presente in Ollama.

        Args:
            preferred_tag: Tag dichiarato dal profilo hardware-aware.

        Returns:
            Tag installato scelto, oppure il preferito se il catalogo non è
            disponibile o nessun fallback è presente.
        """
        # I test con factory mockata non devono dipendere dal catalogo locale.
        if not self._manage_residency:
            with self._lock:
                self._last_tag_fallback = None
            return preferred_tag
        available = self._list_available_model_names()
        # Senza catalogo non inventiamo fallback: il preload segnalerà l'errore reale.
        if not available:
            with self._lock:
                self._last_tag_fallback = None
            return preferred_tag
        selected = select_installed_tag(preferred_tag, available)
        with self._lock:
            # Comunica alla CLI quando si usa un alias libreria (es. gemma4).
            if selected != preferred_tag:
                self._last_tag_fallback = (
                    f"Modello {preferred_tag} assente: uso {selected}."
                )
            else:
                self._last_tag_fallback = None
        return selected

    def _list_available_model_names(self) -> frozenset[str]:
        """
        Legge e memorizza i nomi modello da `/api/tags`, già normalizzati.

        Returns:
            Insieme vuoto se Ollama non è raggiungibile.
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
        """GET JSON best-effort; None se il daemon non risponde o il body è invalido."""
        try:
            response = self._http_client.get(path)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError, TypeError) as error:
            self._last_control_error = f"{type(error).__name__}: {error}"
            return None

    def _switch_active_model(self, next_tag: str) -> None:
        """
        Scarica il tag precedente prima di selezionarne uno differente.

        Args:
            next_tag: Modello che servirà la prossima generazione.
        """
        # Riutilizzare lo stesso artefatto evita reload tra profili di output diversi.
        if self._active_tag == next_tag:
            return
        # Su 16 GB due artefatti residenti possono causare swap durante il cambio.
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
        Invia il controllo preload/unload senza propagare errori di disponibilità.

        Args:
            tag: Tag Ollama da controllare.
            keep_alive: Durata positiva per preload oppure zero per unload.
            context_tokens: `num_ctx` da allineare ai wrapper LangChain.
            prompt: Testo da prefillsare (vuoto = solo load pesi).
            num_predict: Tetto generazione; 1 sul warmup del prefisso.
            raw: Se True il prompt non passa dal template Ollama.
            timeout_seconds: Tetto esplicito; None usa unload rapido o preload lungo.
            images: Immagini base64 per la vision; None per i soli controlli.
            think: CoT nativo; None lascia il default del modello.

        Returns:
            Payload JSON, oppure None se la residency è spenta o la chiamata fallisce.
        """
        # I test con residency disabilitata non devono aprire socket locali.
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
            # La generazione reale produrrà un errore più utile se Ollama è offline.
            self._last_control_error = f"{type(error).__name__}: {error}"
            return None
        return payload if isinstance(payload, dict) else None

    def _expects_gpu(self) -> bool:
        """True se la policy host ha richiesto offload GPU (`num_gpu` ≠ 0)."""
        num_gpu = self._host_options.num_gpu
        return num_gpu is not None and num_gpu != 0

    def _aggiorna_diagnostica_server_env(self) -> None:
        """Rilegge FA/num_parallel solo se il PID del daemon è cambiato."""
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
        self._last_server_env_warning = diagnostica_server_env(
            environ_daemon_ollama(pid)
        )

    def _diagnostica_cpu_fallback(self, tag: str) -> str | None:
        """
        Avvisa se Ollama ha caricato il modello sulla CPU invece che sulla GPU.

        Args:
            tag: Artefatto appena precaricato.

        Returns:
            Messaggio di sistema, oppure None se GPU ok o diagnostica assente.
        """
        # Senza GPU attesa (CPU-only Linux, Intel, test) l'avviso sarebbe rumore.
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
    Interpreta una riga NDJSON di `/api/pull`.

    Returns:
        Payload oggetto, oppure None per linee vuote.

    Raises:
        RuntimeError: JSON illeggibile o tipo inatteso.
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


def _stato_progresso_pull(payload: dict[str, Any]) -> tuple[str, int]:
    """Restituisce stato testuale e percentuale intera (0-100, o -1 se ignota)."""
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


def _formatta_progresso_pull(tag: str, status: str, percent: int) -> str:
    """Riduce stato/percentuale a una riga per la console."""
    if percent < 0:
        return f"{tag}: {status}"
    return f"{tag}: {status} {percent}%"


_LAUNCHCTL_HINT = (
    "launchctl setenv OLLAMA_FLASH_ATTENTION 1; "
    "launchctl setenv OLLAMA_KV_CACHE_TYPE q4_0; "
    "launchctl setenv OLLAMA_NUM_PARALLEL 1"
)


def diagnostica_server_env(
    environ: dict[str, str] | None = None,
) -> str | None:
    """
    Avvisa se il daemon Ollama non ha flash attention o num_parallel=1.

    Args:
        environ: Mapping iniettabile; None legge l'env del processo daemon.

    Returns:
        Messaggio per la CLI, oppure None se ok o se il processo non è visibile.
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
    problemi: list[str] = []
    if flash not in {"1", "true"}:
        problemi.append("OLLAMA_FLASH_ATTENTION non è 1")
    if parallel not in {"1", ""}:
        problemi.append(f"OLLAMA_NUM_PARALLEL={parallel} (atteso 1)")
    if not problemi:
        return None
    return (
        "Daemon Ollama: "
        + "; ".join(problemi)
        + f". Per un `serve` già avviato: {_LAUNCHCTL_HINT}."
    )
