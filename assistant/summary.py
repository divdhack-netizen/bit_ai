"""riassumi_documento tool: workspace summary via Ollama, with an extractive fallback."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import Tool
from langchain_ollama import OllamaLLM

from assistant.config import (
    DEFAULT_MODEL_ALIAS,
    MAX_SUMMARY_SENTENCES,
    MIN_SUMMARY_SENTENCE_CHARS,
    FILE_SEPARATOR,
)
from assistant.file_service import WorkspaceFileService
from assistant.inference_profiles import profile_for_route
from assistant.ollama_runtime import OllamaRuntime
from assistant.tools import (
    expand_literal_newlines,
    get_default_file_service,
    read_document_from_text,
    normalize_llm_path,
    recover_existing_path,
    write_file_from_text,
)
from assistant.tool_contract import ToolContract, build_tool, validate_non_empty

SUMMARIZE_DOCUMENT_TOOL_DESCRIPTION = (
    "Riassume un documento della cartella (txt, md, pdf, docx, immagine). "
    f"Input: path; oppure path_in, {FILE_SEPARATOR}, path_out .txt per salvarlo. "
    "Senza destinazione resta in chat."
)

_SUMMARY_PROMPT = (
    "Riassumi in italiano il documento seguente in al massimo "
    f"{MAX_SUMMARY_SENTENCES} frasi concise. "
    "Solo i punti essenziali; niente prefazioni né markdown.\n\n"
    "DOCUMENTO:\n{content}"
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
class _SummaryRequest:
    """Source path and optional destination for persistence."""

    source: str
    destination: str | None = None


class _SummaryToolAdapter:
    """Single-input bound method that hides the service from the ReAct prompt."""

    def __init__(
        self,
        service: WorkspaceFileService,
        alias: str,
        llm_provider: SummaryLLMProvider | None,
        callback_provider: SummaryCallbackProvider | None,
    ) -> None:
        """
        Bind the summary to the same gate and tracking as the file tools.

        Args:
            service: File boundary of the current session.
            alias: Model used only by the legacy path without a provider.
            llm_provider: The session runtime, if available.
            callback_provider: Callback of the current turn for metrics/stream.
        """
        self._service = service
        self._alias = alias
        self._llm_provider = llm_provider
        self._callback_provider = callback_provider

    def summarize(self, action_input: str) -> str:
        """
        Run riassumi_documento on the bound service.

        Returns:
            Summary, write confirmation or diagnostics.
        """
        callback = (
            self._callback_provider()
            if self._callback_provider is not None
            else None
        )
        return summarize_document_from_text(
            action_input,
            service=self._service,
            alias=self._alias,
            llm_provider=self._llm_provider,
            callback=callback,
        )


def _summary_runtime() -> OllamaRuntime:
    """Shared runtime for the fallback outside AgentSession."""
    global _LEGACY_RUNTIME
    # A single client avoids unload/reload on every standalone summary.
    if _LEGACY_RUNTIME is None:
        _LEGACY_RUNTIME = OllamaRuntime()
    return _LEGACY_RUNTIME


def _summary_llm(
    alias: str = DEFAULT_MODEL_ALIAS,
) -> OllamaLLM:
    """
    Build the legacy fallback reusing OllamaRuntime (same host options).

    Args:
        alias: Active model to respect even outside AgentSession.

    Returns:
        Cached Ollama wrapper; the server reuses the weights through keep-alive.
    """
    profile = profile_for_route(alias, "summary")
    return _summary_runtime().get_llm(
        alias,
        profile.context_tokens,
        profile.output_tokens,
    )


def _is_error_message(text: str) -> bool:
    """True if the string is a tool error message."""
    return text.startswith("Errore")


def _extractive_summary(content: str) -> str:
    """
    Fallback without the LLM: first useful sentences up to MAX_SUMMARY_SENTENCES.

    Input: document text. Output: summary, or Errore: if nothing can be extracted.
    """
    pieces: list[str] = []
    # Unify sentence terminators before splitting.
    normalized = content.replace("!", ".").replace("?", ".")
    for raw in normalized.split("."):
        sentence = " ".join(raw.split())
        # Drop titles/noise that are too short.
        if len(sentence) < MIN_SUMMARY_SENTENCE_CHARS:
            continue
        pieces.append(sentence)
        if len(pieces) >= MAX_SUMMARY_SENTENCES:
            break
    if not pieces:
        return "Errore: impossibile estrarre frasi utili dal documento."
    return ". ".join(pieces) + "."


def _invoke_summary_llm(
    content: str,
    *,
    alias: str = DEFAULT_MODEL_ALIAS,
    llm_provider: SummaryLLMProvider | None = None,
    callback: BaseCallbackHandler | None = None,
) -> str:
    """
    Generate a summary with Ollama; if empty, use the extractive fallback.

    Input: text already truncated by the workspace read.
    Output: summary, or Errore riassunto LLM (...).
    """
    prompt = _SUMMARY_PROMPT.format(content=content)
    profile = profile_for_route(alias, "summary")
    llm = (
        llm_provider(profile.context_tokens, profile.output_tokens)
        if llm_provider is not None
        else _summary_llm(alias)
    )
    try:
        # With a callback, the stream feeds the metrics without mixing with the Final Answer.
        if callback is not None:
            chunks = llm.stream(
                prompt,
                config=RunnableConfig(callbacks=[callback]),
            )
            answer = "".join(str(chunk) for chunk in chunks)
        else:
            answer = llm.invoke(prompt)
    except Exception as error:
        return f"Errore riassunto LLM ({type(error).__name__}): {error}"
    text = str(answer).strip()
    return text if text else _extractive_summary(content)


def _parse_request(
    action_input: str,
    service: WorkspaceFileService,
) -> _SummaryRequest | str:
    """
    Parse the Action Input: a path, or source_path / --- / destination_path.

    Output: typed request, or an Errore: message.
    """
    text = _clean_action_input(action_input)
    if not text:
        return "Errore: path vuoto. Fornisci un path relativo nel workspace."

    if FILE_SEPARATOR not in text:
        text = recover_existing_path(text, service, folder=False)
        path_only = normalize_llm_path(text, service)
        if path_only is None:
            return f"Errore: path non valido: {text}"
        return _SummaryRequest(source=path_only)

    raw_source, _, raw_dest = text.partition(FILE_SEPARATOR)
    # Models often quote every path, as in the user prompt.
    raw_source = raw_source.strip().strip("'\"`")
    raw_dest = raw_dest.strip().strip("'\"`")
    source_path = normalize_llm_path(raw_source, service)
    destination_path = normalize_llm_path(raw_dest, service)
    if source_path is None:
        return f"Errore: path sorgente non valido: {raw_source.strip()}"
    if destination_path is None:
        return (
            f"Errore: path destinazione non valido: {raw_dest.strip()}. "
            f"Formato: path_sorgente\\n{FILE_SEPARATOR}\\npath_destinazione"
        )
    return _SummaryRequest(source=source_path, destination=destination_path)


def _clean_action_input(action_input: str) -> str:
    """
    Remove outer quotes and later ReAct blocks captured by the permissive parser.

    Returns:
        Only the payload of the first Action Input, keeping the tool's three lines.
    """
    text = str(action_input or "").strip()
    # Gemma may emit a second block in the same chunk; the first input is the valid one.
    for marker in _TRAILING_REACT_MARKERS:
        # Truncate only when the marker appears after a non-empty payload.
        if marker in text:
            text = text.split(marker, 1)[0].rstrip()
    # An old tool signature could show up as a service: line; it is never part of a path.
    if _LEAKED_SERVICE_LINE in text:
        text = text.split(_LEAKED_SERVICE_LINE, 1)[0].rstrip()
    # Remove a pair of quotes wrapping the whole multiline payload.
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1]
    # Expand after truncating: real newlines of the second block must not block \\n.
    return expand_literal_newlines(text.strip())


def summarize_document_from_text(
    action_input: str,
    *,
    service: WorkspaceFileService | None = None,
    alias: str = DEFAULT_MODEL_ALIAS,
    llm_provider: SummaryLLMProvider | None = None,
    callback: BaseCallbackHandler | None = None,
) -> str:
    """
    Read a workspace path, summarize it and optionally write the result.

    Args:
        action_input: A path, or the source/separator/destination protocol.
        service: Optional workspace boundary.
        alias: Active model of the legacy fallback.
        llm_provider: Session runtime, to avoid a second model.
        callback: Current LLM observer for stream and telemetry.

    Returns:
        Summary; if saved, the final outcome + summary.
    """
    selected_service = service or get_default_file_service()
    request = _parse_request(action_input, selected_service)
    if isinstance(request, str):
        return request
    # Same destination already written in this turn: no second generation.
    if (
        request.destination is not None
        and request.destination in selected_service.written_files
    ):
        return (
            f"Riassunto già salvato in {request.destination} in questo turno. "
            "Compito completato: rispondi con Final Answer."
        )

    content = read_document_from_text(
        request.source,
        service=selected_service,
    )
    if _is_error_message(content):
        return content

    summary = _invoke_summary_llm(
        content,
        alias=alias,
        llm_provider=llm_provider,
        callback=callback,
    )
    if _is_error_message(summary):
        return summary
    if request.destination is None:
        return summary

    # Reuse scrivi_file for sandboxing, sanitization and anti-loop.
    write = write_file_from_text(
        f"{request.destination}\n{FILE_SEPARATOR}\n{summary}",
        service=selected_service,
    )
    if _is_error_message(write):
        return write
    # Final outcome: scrivi_file's invitation to retry would make the model rewrite the file.
    return (
        f"Riassunto salvato in {request.destination}. Compito completato: "
        f"rispondi con Final Answer.\n\n{summary}"
    )


def create_summary_tool(
    service: WorkspaceFileService | None = None,
    *,
    alias: str = DEFAULT_MODEL_ALIAS,
    llm_provider: SummaryLLMProvider | None = None,
    callback_provider: SummaryCallbackProvider | None = None,
) -> list[Tool]:
    """
    Register the summary ReAct tool bound to the session's file service.

    Returns:
        List with one Tool sharing gate and tracking with the other file operations.
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
                description=SUMMARIZE_DOCUMENT_TOOL_DESCRIPTION,
            ),
            adapter.summarize,
        ),
    ]
