"""Tool riassumi_documento: riassunto workspace via Ollama, con fallback estrattivo."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import Tool
from langchain_ollama import OllamaLLM

from assistente.config import (
    ALIAS_MODELLO_DEFAULT,
    MAX_FRASI_RIASSUNTO,
    MIN_CARATTERI_FRASE_RIASSUNTO,
    SEPARATORE_FILE,
)
from assistente.file_service import WorkspaceFileService
from assistente.inference_profiles import profile_for_route
from assistente.ollama_runtime import OllamaRuntime
from assistente.tools import (
    espandi_newline_letterali,
    get_default_file_service,
    leggi_documento_da_testo,
    normalizza_path_da_llm,
    recupera_path_esistente,
    scrivi_file_da_testo,
)
from assistente.tool_contract import ToolContract, build_tool, validate_non_empty

DESCRIZIONE_TOOL_RIASSUMI_DOCUMENTO = (
    "Riassume un documento della cartella (txt, md, pdf, docx, immagine). "
    f"Input: path; oppure path_in, {SEPARATORE_FILE}, path_out .txt per salvarlo. "
    "Senza destinazione resta in chat."
)

_PROMPT_RIASSUNTO = (
    "Riassumi in italiano il documento seguente in al massimo "
    f"{MAX_FRASI_RIASSUNTO} frasi concise. "
    "Solo i punti essenziali; niente prefazioni né markdown.\n\n"
    "DOCUMENTO:\n{contenuto}"
)
_TRAILING_REACT_MARKERS = (
    "<channel|>",
    "\nAction:",
    "\nThought:",
    "\nFinal Answer:",
    "\nObservation:",
)
_LEAKED_SERVICE_LINE = "\nservice: WorkspaceFileService"

SummaryLLMProvider = Callable[[int, int], OllamaLLM]
SummaryCallbackProvider = Callable[[], BaseCallbackHandler | None]
_LEGACY_RUNTIME: OllamaRuntime | None = None


@dataclass(frozen=True, slots=True)
class _RichiestaRiassunto:
    """Path sorgente e destinazione opzionale per persistenza."""

    sorgente: str
    destinazione: str | None = None


class _SummaryToolAdapter:
    """Metodo bound a singolo input che nasconde il servizio al prompt ReAct."""

    def __init__(
        self,
        service: WorkspaceFileService,
        alias: str,
        llm_provider: SummaryLLMProvider | None,
        callback_provider: SummaryCallbackProvider | None,
    ) -> None:
        """
        Collega il riassunto allo stesso gate e tracking dei tool file.

        Args:
            service: Boundary file della sessione corrente.
            alias: Modello usato soltanto dal percorso legacy senza provider.
            llm_provider: Runtime della sessione, se disponibile.
            callback_provider: Callback del turno corrente per metriche/stream.
        """
        self._service = service
        self._alias = alias
        self._llm_provider = llm_provider
        self._callback_provider = callback_provider

    def summarize(self, action_input: str) -> str:
        """
        Esegue riassumi_documento sul servizio associato.

        Returns:
            Riassunto, conferma di scrittura oppure diagnostica.
        """
        callback = (
            self._callback_provider()
            if self._callback_provider is not None
            else None
        )
        return riassumi_documento_da_testo(
            action_input,
            service=self._service,
            alias=self._alias,
            llm_provider=self._llm_provider,
            callback=callback,
        )


def _runtime_riassunto() -> OllamaRuntime:
    """Runtime condiviso per il fallback fuori da SessioneAgente."""
    global _LEGACY_RUNTIME
    # Un solo client evita unload/reload a ogni riassunto standalone.
    if _LEGACY_RUNTIME is None:
        _LEGACY_RUNTIME = OllamaRuntime()
    return _LEGACY_RUNTIME


def _llm_riassunto(
    alias: str = ALIAS_MODELLO_DEFAULT,
) -> OllamaLLM:
    """
    Costruisce il fallback legacy riusando OllamaRuntime (stesse opzioni host).

    Args:
        alias: Modello attivo da rispettare anche fuori da SessioneAgente.

    Returns:
        Wrapper Ollama cacheato; il server riusa i pesi tramite keep-alive.
    """
    profile = profile_for_route(alias, "summary")
    return _runtime_riassunto().get_llm(
        alias,
        profile.context_tokens,
        profile.output_tokens,
    )


def _e_messaggio_errore(testo: str) -> bool:
    """True se la stringa è un messaggio d'errore tool."""
    return testo.startswith("Errore")


def _riassunto_estrattivo(contenuto: str) -> str:
    """
    Fallback senza LLM: prime frasi utili fino a MAX_FRASI_RIASSUNTO.

    Input: testo documento. Output: riassunto, oppure Errore: se non estraibile.
    """
    pezzi: list[str] = []
    # Unifica terminatori di frase prima dello split.
    normalizzato = contenuto.replace("!", ".").replace("?", ".")
    for grezzo in normalizzato.split("."):
        frase = " ".join(grezzo.split())
        # Scarta titoli/rumore troppo corti.
        if len(frase) < MIN_CARATTERI_FRASE_RIASSUNTO:
            continue
        pezzi.append(frase)
        if len(pezzi) >= MAX_FRASI_RIASSUNTO:
            break
    if not pezzi:
        return "Errore: impossibile estrarre frasi utili dal documento."
    return ". ".join(pezzi) + "."


def _invoca_llm_riassunto(
    contenuto: str,
    *,
    alias: str = ALIAS_MODELLO_DEFAULT,
    llm_provider: SummaryLLMProvider | None = None,
    callback: BaseCallbackHandler | None = None,
) -> str:
    """
    Genera riassunto con Ollama; se vuoto, usa fallback estrattivo.

    Input: testo già troncato dalla lettura workspace.
    Output: riassunto, oppure Errore riassunto LLM (...).
    """
    prompt = _PROMPT_RIASSUNTO.format(contenuto=contenuto)
    profile = profile_for_route(alias, "summary")
    llm = (
        llm_provider(profile.context_tokens, profile.output_tokens)
        if llm_provider is not None
        else _llm_riassunto(alias)
    )
    try:
        # Con callback, lo stream alimenta metriche senza confondersi con la Final Answer.
        if callback is not None:
            chunks = llm.stream(
                prompt,
                config=RunnableConfig(callbacks=[callback]),
            )
            risposta = "".join(str(chunk) for chunk in chunks)
        else:
            risposta = llm.invoke(prompt)
    except Exception as errore:
        return f"Errore riassunto LLM ({type(errore).__name__}): {errore}"
    testo = str(risposta).strip()
    return testo if testo else _riassunto_estrattivo(contenuto)


def _parse_richiesta(
    action_input: str,
    service: WorkspaceFileService,
) -> _RichiestaRiassunto | str:
    """
    Interpreta Action Input: path, oppure path_sorgente / --- / path_destinazione.

    Output: richiesta tipizzata, oppure messaggio Errore:.
    """
    testo = _clean_action_input(action_input)
    if not testo:
        return "Errore: path vuoto. Fornisci un path relativo nel workspace."

    if SEPARATORE_FILE not in testo:
        testo = recupera_path_esistente(testo, service, cartella=False)
        path_solo = normalizza_path_da_llm(testo, service)
        if path_solo is None:
            return f"Errore: path non valido: {testo}"
        return _RichiestaRiassunto(sorgente=path_solo)

    grezzo_sorgente, _, grezzo_dest = testo.partition(SEPARATORE_FILE)
    # I modelli citano spesso ogni path tra apici, come nel prompt utente.
    grezzo_sorgente = grezzo_sorgente.strip().strip("'\"`")
    grezzo_dest = grezzo_dest.strip().strip("'\"`")
    path_sorgente = normalizza_path_da_llm(grezzo_sorgente, service)
    path_destinazione = normalizza_path_da_llm(grezzo_dest, service)
    if path_sorgente is None:
        return f"Errore: path sorgente non valido: {grezzo_sorgente.strip()}"
    if path_destinazione is None:
        return (
            f"Errore: path destinazione non valido: {grezzo_dest.strip()}. "
            f"Formato: path_sorgente\\n{SEPARATORE_FILE}\\npath_destinazione"
        )
    return _RichiestaRiassunto(sorgente=path_sorgente, destinazione=path_destinazione)


def _clean_action_input(action_input: str) -> str:
    """
    Rimuove quote esterne e blocchi ReAct successivi catturati dal parser permissivo.

    Returns:
        Solo il payload del primo Action Input, preservando le tre righe del tool.
    """
    text = str(action_input or "").strip()
    # Gemma può emettere un secondo blocco nello stesso chunk; il primo input è quello valido.
    for marker in _TRAILING_REACT_MARKERS:
        # Tronca soltanto quando il marcatore compare dopo un payload non vuoto.
        if marker in text:
            text = text.split(marker, 1)[0].rstrip()
    # Una vecchia firma tool poteva comparire come riga service:; non è mai parte di un path.
    if _LEAKED_SERVICE_LINE in text:
        text = text.split(_LEAKED_SERVICE_LINE, 1)[0].rstrip()
    # Rimuove una coppia di quote che racchiude l'intero payload multilinea.
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1]
    # Espande dopo il troncamento: newline reali del secondo blocco non devono bloccare \\n.
    return espandi_newline_letterali(text.strip())


def riassumi_documento_da_testo(
    action_input: str,
    *,
    service: WorkspaceFileService | None = None,
    alias: str = ALIAS_MODELLO_DEFAULT,
    llm_provider: SummaryLLMProvider | None = None,
    callback: BaseCallbackHandler | None = None,
) -> str:
    """
    Legge un path workspace, riassume e opzionalmente scrive il risultato.

    Args:
        action_input: Path oppure protocollo sorgente/separatore/destinazione.
        service: Boundary workspace opzionale.
        alias: Modello attivo del fallback legacy.
        llm_provider: Runtime della sessione per evitare un secondo modello.
        callback: Observer LLM corrente per stream e telemetria.

    Returns:
        Riassunto; se salvato, esito conclusivo + riassunto.
    """
    selected_service = service or get_default_file_service()
    richiesta = _parse_richiesta(action_input, selected_service)
    if isinstance(richiesta, str):
        return richiesta
    # Stessa destinazione già scritta nel turno: niente seconda generazione.
    if (
        richiesta.destinazione is not None
        and richiesta.destinazione in selected_service.written_files
    ):
        return (
            f"Riassunto già salvato in {richiesta.destinazione} in questo turno. "
            "Compito completato: rispondi con Final Answer."
        )

    contenuto = leggi_documento_da_testo(
        richiesta.sorgente,
        service=selected_service,
    )
    if _e_messaggio_errore(contenuto):
        return contenuto

    riassunto = _invoca_llm_riassunto(
        contenuto,
        alias=alias,
        llm_provider=llm_provider,
        callback=callback,
    )
    if _e_messaggio_errore(riassunto):
        return riassunto
    if richiesta.destinazione is None:
        return riassunto

    # Riusa scrivi_file per sandbox, sanitizzazione e anti-loop.
    scrittura = scrivi_file_da_testo(
        f"{richiesta.destinazione}\n{SEPARATORE_FILE}\n{riassunto}",
        service=selected_service,
    )
    if _e_messaggio_errore(scrittura):
        return scrittura
    # Esito conclusivo: l'invito di scrivi_file a ripetere farebbe riscrivere il file.
    return (
        f"Riassunto salvato in {richiesta.destinazione}. Compito completato: "
        f"rispondi con Final Answer.\n\n{riassunto}"
    )


def crea_tool_riassunto(
    service: WorkspaceFileService | None = None,
    *,
    alias: str = ALIAS_MODELLO_DEFAULT,
    llm_provider: SummaryLLMProvider | None = None,
    callback_provider: SummaryCallbackProvider | None = None,
) -> list[Tool]:
    """
    Registra il tool ReAct di riassunto legato al servizio file della sessione.

    Returns:
        Lista con un Tool che condivide gate e tracking delle altre operazioni file.
    """
    selected_service = service or get_default_file_service()
    adapter = _SummaryToolAdapter(
        selected_service,
        alias,
        llm_provider,
        callback_provider,
    )
    return [
        build_tool(
            ToolContract(
                name="riassumi_documento",
                is_read_only=False,
                is_concurrency_safe=False,
                validate=validate_non_empty,
                description=DESCRIZIONE_TOOL_RIASSUMI_DOCUMENTO,
            ),
            adapter.summarize,
        ),
    ]
