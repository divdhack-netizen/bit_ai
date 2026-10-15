"""Thin LangChain adapters for WorkspaceFileService and the controlled fallback."""

from __future__ import annotations

import re
from contextvars import ContextVar
from pathlib import Path

from langchain_core.tools import Tool

from assistant.config import (
    LISTING_MAX_RESULT_CHARS,
    NEW_BLOCK_MARKER,
    OLD_BLOCK_MARKER,
    FILE_SEPARATOR,
)
from assistant.file_service import (
    DirectoryOperationResult,
    SearchResult,
    WorkspaceFileService,
)
from assistant.react_parser import _cut_protocol_tail
from assistant.tool_contract import (
    ToolContract,
    build_tools,
    validate_non_empty,
)

WRITE_FILE_TOOL_DESCRIPTION = (
    f"Crea o sovrascrive un file .txt. Input: path relativo "
    f"(note/oggi.txt, non /note/oggi.txt), riga {FILE_SEPARATOR}, body grezzo. "
    "Solo estensione .txt. Un file per Action. Niente virgole nel path."
)
EDIT_FILE_TOOL_DESCRIPTION = (
    f"Modifica un file .txt esistente. Input: path, riga {OLD_BLOCK_MARKER}, "
    f"blocco esatto, riga {NEW_BLOCK_MARKER}, blocco nuovo. "
    "Un match unico. NEW può essere vuoto."
)
READ_DOCUMENT_TOOL_DESCRIPTION = (
    "Legge un documento della cartella: txt, md, pdf, docx, pages o immagine "
    "(png, jpg, webp: trascrizione e descrizione). Input: path relativo esatto."
)
SEARCH_TEXT_TOOL_DESCRIPTION = (
    "Cerca parole nel contenuto di tutti i documenti (txt, md, pdf, docx, pages; "
    "non le immagini). Input: poche parole chiave. "
    "Restituisce file, riga o pagina ed estratto, i più pertinenti per primi."
)
LIST_FOLDER_TOOL_DESCRIPTION = (
    "Elenca una cartella del workspace (un livello). "
    "Input: . per la cartella attuale, altrimenti il path relativo esatto "
    "della sottocartella nominata (es. notes)."
)
CREATE_FOLDER_TOOL_DESCRIPTION = (
    "Crea una cartella relativa, con i parent se mancano. "
    "Input: path relativo (notes/2026). exist_ok se è già una cartella."
)
LIST_TREE_TOOL_DESCRIPTION = (
    "Elenca una cartella fino a due livelli, senza venv/.git. "
    "Input: . per la cartella attuale, altrimenti path relativo (es. output)."
)
FIND_FILE_TOOL_DESCRIPTION = (
    "Trova file o cartelle per nome. Input: glob (*.pdf) o frammento (brief). "
    "Per il contenuto usa cerca_testo."
)

_SUCCESS_PREFIX = "File scritto con successo:"
_MODIFY_SUCCESS_PREFIX = "File modificato con successo:"
_MKDIR_SUCCESS_PREFIX = "Cartella creata:"
_MARKER_OLD_LINE_PATTERN = re.compile(
    rf"^{re.escape(OLD_BLOCK_MARKER)}\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_MARKER_NEW_LINE_PATTERN = re.compile(
    rf"^{re.escape(NEW_BLOCK_MARKER)}\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_PATH_PREFIX_PATTERN = re.compile(r"^(?:path|file)\s*[:=]\s*", re.IGNORECASE)
_INSTRUCTIONAL_PATH_PREFIX_PATTERN = re.compile(
    r"^(?:path|percorso)\s+relativo\s+",
    re.IGNORECASE,
)
_ROW_PREFIX_PATTERN = re.compile(r"^riga\s*\d+\s*=\s*", re.IGNORECASE)
_ROW_SEPARATOR_ARTIFACT_PATTERN = re.compile(
    rf"^riga\s*{re.escape(FILE_SEPARATOR)}\s*$",
    re.IGNORECASE,
)
_ROW_SUFFIX_PATTERN = re.compile(r",\s*riga\s*\d+\s*=.*$", re.IGNORECASE)
_CONTENT_PREFIX_PATTERN = re.compile(
    r"^(?:content|contenuto)\s*:\s*",
    re.IGNORECASE,
)
_OPENING_FENCE_PATTERN = re.compile(r"^```[^\n]*\n")
_CLOSING_FENCE_PATTERN = re.compile(r"\n?```\s*$")
_AGENT_ARTIFACT_PATTERN = re.compile(
    r"(?:^|\n---\s*|\n)(?:Thought|Action|Action Input)\s*:",
    re.IGNORECASE | re.MULTILINE,
)
# Gemma often emits a single-line ``path.ext,content`` instead of path/---/body.
_SINGLE_LINE_PATH_CONTENT_PATTERN = re.compile(
    r"^(?P<path>(?:[\w./-]+/)*[\w.-]+\.[A-Za-z0-9]+)\s*,\s*(?P<content>.+)$"
)
# Paths with an extension cited in the Thought/body when the first line omits the extension.
_PATH_WITH_EXTENSION_PATTERN = re.compile(
    r"(?P<path>[\w./-]+\.[A-Za-z0-9]+)"
)
_YAML_BLOCK_MARKERS = frozenset({"|", ">", "|-", ">-", "|+", ">+"})
_PROTOCOL_ONLY_CONTENT_PATTERN = re.compile(
    r"^(?:Thought|Action|Action Input)\s*:",
    re.IGNORECASE,
)
_PATH_AND_FENCE_PATTERN = re.compile(
    r"(?ms)^(?P<path>[^\n:`]+?)\s*:?\s*```[^\n]*\n"
    r"(?P<content>.*?)```"
)
_PATH_AND_SEPARATOR_PATTERN = re.compile(
    rf"(?ms)^(?P<path>[^\n]+)\n{re.escape(FILE_SEPARATOR)}\n"
    rf"(?P<content>.*?)(?=^[^\n]+\n{re.escape(FILE_SEPARATOR)}\n|\Z)"
)
_SEPARATOR_LINE_PATTERN = re.compile(
    rf"\n\s*{re.escape(FILE_SEPARATOR)}\s*(?:\n|$)",
)
_KNOWN_LANGUAGE_MARKERS = frozenset(
    {
        "bash",
        "css",
        "html",
        "java",
        "javascript",
        "js",
        "json",
        "markdown",
        "md",
        "py",
        "python",
        "sh",
        "ts",
        "txt",
        "typescript",
    }
)
_MAX_EXPLICIT_PATH_CHARACTERS = 240

_CONTEXT_SERVICE: ContextVar[WorkspaceFileService | None] = ContextVar(
    "compat_file_service",
    default=None,
)

REPEATED_READ_WARNING = (
    "Stessa Action del passo precedente: il risultato è già questo. "
    "Non ripeterla: scrivi ora Final Answer.\n"
)


class _FileToolAdapter:
    """Single-argument bound methods that do not expose the service to the LLM prompt."""

    def __init__(self, service: WorkspaceFileService) -> None:
        """
        Bind every file tool to the same boundary and the same turn state.

        Args:
            service: Service owned by the current session.
        """
        self._service = service

    def _with_anti_loop(self, tool: str, action_input: str, outcome: str) -> str:
        """Prepend a warning if the model repeats the same read back to back."""
        if self._service.mark_read(tool, action_input):
            return f"{REPEATED_READ_WARNING}{outcome}"
        return outcome

    def write(self, action_input: str) -> str:
        """
        Run scrivi_file on the bound service.

        Returns:
            ReAct-compatible text Observation.
        """
        return write_file_from_text(action_input, service=self._service)

    def replace(self, action_input: str) -> str:
        """
        Run modifica_file on the bound service.

        Returns:
            ReAct-compatible text Observation.
        """
        return edit_file_from_text(action_input, service=self._service)

    def read(self, action_input: str) -> str:
        """
        Run leggi_documento on the bound service.

        Returns:
            Content or text diagnostics.
        """
        outcome = read_document_from_text(
            action_input,
            service=self._service,
        )
        return self._with_anti_loop("read", action_input, outcome)

    def list(self, action_input: str) -> str:
        """
        Run elenca_cartella on the bound service.

        Returns:
            Listing or text diagnostics.
        """
        outcome = list_folder_from_text(action_input, service=self._service)
        return self._with_anti_loop("list", action_input, outcome)

    def mkdir(self, action_input: str) -> str:
        """Run crea_cartella on the bound service."""
        return create_folder_from_text(action_input, service=self._service)

    def tree(self, action_input: str) -> str:
        """Run elenca_albero on the bound service."""
        outcome = list_tree_from_text(action_input, service=self._service)
        return self._with_anti_loop("tree", action_input, outcome)

    def find(self, action_input: str) -> str:
        """Run trova_file on the bound service."""
        outcome = find_file_from_text(action_input, service=self._service)
        return self._with_anti_loop("find", action_input, outcome)

    def search(self, action_input: str) -> str:
        """Run cerca_testo on the bound service."""
        outcome = search_text_from_text(action_input, service=self._service)
        return self._with_anti_loop("search", action_input, outcome)

def get_default_file_service() -> WorkspaceFileService:
    """
    Return a compatibility service isolated in the current ContextVar.

    Returns:
        Authorized service for legacy direct calls; sessions use explicit instances.
    """
    service = _CONTEXT_SERVICE.get()
    # Each context creates its own state, avoiding contamination between threads or tasks.
    if service is None:
        service = WorkspaceFileService()
        service.begin_turn(write_allowed=True)
        _CONTEXT_SERVICE.set(service)
    return service

def reset_turn_state(service: WorkspaceFileService | None = None) -> None:
    """
    Reset legacy tracking or the explicit service, keeping direct authorization.

    Args:
        service: Instance to reset; None replaces the one in the current ContextVar.
    """
    # Direct test calls stay authorized; AgentSession applies its own gate.
    if service is not None:
        service.begin_turn(write_allowed=True)
        return
    replacement = WorkspaceFileService()
    replacement.begin_turn(write_allowed=True)
    _CONTEXT_SERVICE.set(replacement)

def set_write_allowed(
    allowed: bool,
    service: WorkspaceFileService | None = None,
) -> None:
    """
    Start a new turn state with the explicitly requested gate.

    Args:
        allowed: The user policy's decision.
        service: Session service; None uses the context-local legacy adapter.
    """
    _selected_service(service).begin_turn(write_allowed=allowed)

def turn_write_allowed(
    service: WorkspaceFileService | None = None,
) -> bool:
    """
    Expose the current gate for tests and compatibility adapters.

    Returns:
        write_allowed state of the selected service.
    """
    return _selected_service(service).write_allowed

def normalize_llm_path(
    raw_path: str,
    service: WorkspaceFileService | None = None,
) -> str | None:
    """
    Clean up typical model prefixes and delegate validation to the service.

    Returns:
        Normalized relative file path, or None.
    """
    cleaned = _clean_raw_path(raw_path)
    return _selected_service(service).normalize_file_path(cleaned)

def normalize_llm_folder_path(
    raw_path: str,
    service: WorkspaceFileService | None = None,
) -> str | None:
    """
    Normalize a folder path coming from an Action Input.

    Returns:
        Relative path, empty string for the root, or None.
    """
    cleaned = _clean_raw_path(raw_path)
    return _selected_service(service).normalize_directory_path(cleaned)

def validate_relative_path(
    relative_path: str,
    service: WorkspaceFileService | None = None,
) -> Path | None:
    """
    Keep the historical API, returning the absolute Path only after the sandbox.

    Returns:
        Absolute destination, or None.
    """
    resolved = _selected_service(service).resolve_file(relative_path)
    return resolved.absolute if resolved is not None else None

def validate_relative_folder_path(
    relative_path: str,
    service: WorkspaceFileService | None = None,
) -> Path | None:
    """
    Resolve a relative folder in the selected workspace.

    Returns:
        Absolute directory, or None.
    """
    resolved = _selected_service(service).resolve_directory(relative_path)
    return resolved.absolute if resolved is not None else None

def path_valid_for_workspace(
    relative_path: str,
    service: WorkspaceFileService | None = None,
) -> bool:
    """
    Check syntax and sandbox with a single resolution.

    Returns:
        True when the service would accept the path.
    """
    return _selected_service(service).resolve_file(relative_path) is not None

def path_from_write_observation(
    observation: str,
    service: WorkspaceFileService | None = None,
) -> str | None:
    """
    Extract the path from a historical tool message, for external compatibility.

    Returns:
        Normalized path, only for success observations.
    """
    # Errors and messages from other tools must not produce false paths.
    if not observation.startswith(_SUCCESS_PREFIX):
        return None
    # Only the first line holds the path; any hints are below it.
    raw_path = (
        observation.removeprefix(_SUCCESS_PREFIX).strip().splitlines()[0].strip()
    )
    return normalize_llm_path(raw_path, service)

def sanitize_file_content(content: str, relative_path: str) -> str:
    """
    Remove only outer ReAct wrappers and normalize a trailing newline.

    Args:
        content: Body separated from the path.
        relative_path: Kept in the API for compatibility; no repair by extension.

    Returns:
        Preserved content without outer fences or trailing artifacts.
    """
    del relative_path
    text = content.strip("\n")
    text = _OPENING_FENCE_PATTERN.sub("", text)
    text = _CLOSING_FENCE_PATTERN.sub("", text)
    # "Contenuto: |" leaves the YAML block-scalar marker as a spurious first line.
    lines = text.split("\n", 1)
    if lines and lines[0].strip() in _YAML_BLOCK_MARKERS:
        text = lines[1] if len(lines) > 1 else ""
    # «riga ---» ends up in the body when the model copies the instructions.
    lines = text.split("\n", 1)
    if lines and _ROW_SEPARATOR_ARTIFACT_PATTERN.match(lines[0].strip()):
        text = lines[1] if len(lines) > 1 else ""
    artifact = _AGENT_ARTIFACT_PATTERN.search(text)
    # Cut only protocol added after the body, never valid strings inside the file.
    if artifact is not None:
        text = text[: artifact.start()].rstrip("\n")
    return text.rstrip("\n") + "\n"


def expand_literal_newlines(text: str) -> str:
    """
    Expand ``\\n`` only in the single-line Action Inputs produced by Gemma.

    Returns:
        Unchanged text if it already has real newlines, otherwise the multiline form.
    """
    # Real newlines mean the escape sequences in the body may be intentional.
    if "\\n" not in text or "\n" in text:
        return text
    return text.replace("\\n", "\n")

def write_file_from_text(
    action_input: str,
    *,
    service: WorkspaceFileService | None = None,
) -> str:
    """
    Parse the ``path / --- / content`` protocol and delegate the atomic write.

    Returns:
        Stable success Observation, or a message starting with Errore:.
    """
    selected_service = _selected_service(service)
    raw_text = str(action_input or "").strip()
    # Empty inputs do not reach the service because they hold neither path nor body.
    if not raw_text:
        return (
            "Errore: input vuoto. Formato: prima riga = path, "
            f"seconda riga = {FILE_SEPARATOR}, poi contenuto."
        )
    parsed = _split_path_and_content(raw_text, selected_service)
    # An ambiguous format must not try to infer the path from the content.
    if parsed is None:
        return _format_error_message(raw_text)
    relative_path, raw_content = parsed
    # Empty files need a different, explicit request; this avoids accidental writes.
    if not raw_content.strip():
        return "Errore: contenuto mancante."

    clean_content = sanitize_file_content(raw_content, relative_path)
    # Body reduced to ReAct protocol only: a typical chained Action Input without ---.
    if not clean_content.strip() or _PROTOCOL_ONLY_CONTENT_PATTERN.match(
        clean_content.strip()
    ):
        return (
            "Errore: contenuto mancante o solo protocollo ReAct.\n"
            f"Usa:\npath_rel\n{FILE_SEPARATOR}\ncontenuto_completo\n"
            "(un file per Action; niente Action concatenate)."
        )
    result = selected_service.write_text(relative_path, clean_content)
    # Typed results are turned into strings only at the LangChain boundary.
    if not result.succeeded:
        return f"Errore: {result.error}"
    # Second line: pushes Gemma not to claim other files without an Observation.
    return (
        f"{_SUCCESS_PREFIX} {result.path}\n"
        "Se serve un altro file, ripeti Action: scrivi_file (un path per Action)."
    )

def edit_file_from_text(
    action_input: str,
    *,
    service: WorkspaceFileService | None = None,
) -> str:
    """
    Parse ``path / <<<OLD / block / <<<NEW / block`` and delegate the atomic patch.

    Returns:
        Stable success Observation, or a message starting with Errore:.
    """
    selected_service = _selected_service(service)
    raw_text = str(action_input or "").strip()
    if not raw_text:
        return (
            "Errore: input vuoto. Formato: path, "
            f"riga {OLD_BLOCK_MARKER}, blocco, "
            f"riga {NEW_BLOCK_MARKER}, blocco nuovo."
        )
    parsed = _split_path_old_new(raw_text, selected_service)
    if parsed is None:
        return (
            "Errore: formato non valido. "
            f"Usa: path_rel\\n{OLD_BLOCK_MARKER}\\nblocco_esatto\\n"
            f"{NEW_BLOCK_MARKER}\\nblocco_nuovo"
        )
    relative_path, old_block, new_block = parsed
    if old_block == "":
        return "Errore: blocco OLD vuoto. Copia il testo esatto da sostituire."
    if OLD_BLOCK_MARKER in new_block or NEW_BLOCK_MARKER in new_block:
        return (
            "Errore: blocco NEW contiene ancora marker <<<OLD/<<<NEW. "
            "Ripeti modifica_file con NEW uguale solo al codice da scrivere."
        )
    result = selected_service.replace_text(relative_path, old_block, new_block)
    if not result.succeeded:
        return f"Errore: {result.error}"
    return (
        f"{_MODIFY_SUCCESS_PREFIX} {result.path}\n"
        "Se serve un altro file, ripeti Action: modifica_file (un path per Action)."
    )

def read_document_from_text(
    action_input: str,
    *,
    service: WorkspaceFileService | None = None,
) -> str:
    """
    Read a document through the sandboxed service.

    Returns:
        Truncated content, or an Errore: message.
    """
    selected = _selected_service(service)
    path = recover_existing_path(action_input, selected, folder=False)
    result = selected.read_document(path)
    return result.content if result.succeeded else f"Errore: {result.error}"

def list_folder_from_text(
    action_input: str,
    *,
    service: WorkspaceFileService | None = None,
) -> str:
    """
    List a folder through the service and format the relative entries.

    Returns:
        Stable text listing, or an Errore: message.
    """
    selected = _selected_service(service)
    result = selected.list_directory(
        recover_existing_path(action_input, selected, folder=True)
    )
    # Typed diagnostics must not be confused with an empty folder.
    if not result.succeeded:
        return f"Errore: {result.error}"
    return _format_list(
        result,
        title=f"Contenuto di '{result.path}'",
        empty_message=f"Cartella vuota: {result.path}",
    )

def create_folder_from_text(
    action_input: str,
    *,
    service: WorkspaceFileService | None = None,
) -> str:
    """Create a relative folder through the sandboxed service."""
    result = _selected_service(service).create_directory(
        _clean_raw_path(str(action_input or ""))
    )
    if not result.succeeded:
        return f"Errore: {result.error}"
    return f"{_MKDIR_SUCCESS_PREFIX} {result.path}"

def list_tree_from_text(
    action_input: str,
    *,
    service: WorkspaceFileService | None = None,
) -> str:
    """List a folder down to a fixed depth."""
    selected = _selected_service(service)
    result = selected.list_tree(
        recover_existing_path(action_input, selected, folder=True)
    )
    if not result.succeeded:
        return f"Errore: {result.error}"
    return _format_list(
        result,
        title=f"Albero di '{result.path}'",
        empty_message=f"Albero vuoto: {result.path}",
    )

def find_file_from_text(
    action_input: str,
    *,
    service: WorkspaceFileService | None = None,
) -> str:
    """Find files and folders by name."""
    result = _selected_service(service).find_files(str(action_input or "").strip())
    if not result.succeeded:
        return f"Errore: {result.error}"
    return _format_list(
        result,
        title=f"Trovati per '{result.path}'",
        empty_message=f"Nessun risultato: {result.path}",
    )

def search_text_from_text(
    action_input: str,
    *,
    service: WorkspaceFileService | None = None,
) -> str:
    """Search for words in the content of the folder's documents."""
    result = _selected_service(service).search_text(str(action_input or ""))
    if not result.succeeded:
        return f"Errore: {result.error}"
    return _format_search(result)

def _format_search(result: SearchResult) -> str:
    """Render a SearchResult as a stable Observation: file, position, excerpt."""
    if not result.matches:
        text = f"Nessun documento contiene: {result.query}"
    else:
        lines = [f"Trovato '{result.query}' in:"]
        for match in result.matches:
            lines.append(f"  {match.relative_path} ({match.location}): {match.snippet}")
        if result.truncated:
            lines.append("  […risultati troncati: restringi la ricerca]")
        text = "\n".join(lines)
    if result.skipped:
        skipped = ", ".join(result.skipped[:5])
        text += f"\nNon leggibili per la ricerca: {skipped}"
    # With no results, images are the only content still left to look at.
    if not result.matches and result.images:
        images = ", ".join(result.images[:10])
        text += (
            f"\nImmagini non coperte dalla ricerca: {images}. "
            "Leggi con leggi_documento solo quelle pertinenti al tema."
        )
    return text

def _format_list(
    result: DirectoryOperationResult,
    *,
    title: str,
    empty_message: str,
) -> str:
    """Render a DirectoryOperationResult as a stable Observation."""
    if not result.entries:
        return empty_message
    lines = [f"{title}:"]
    for entry in result.entries:
        if entry.is_directory:
            lines.append(f"  [cartella] {entry.relative_path}/")
        else:
            lines.append(f"  {entry.relative_path} ({entry.size_bytes} byte)")
    if result.truncated:
        lines.append("  […elenco troncato]")
    return "\n".join(lines)

def create_file_tools(
    service: WorkspaceFileService | None = None,
    *,
    write_allowed: bool | None = None,
    workspace_label: str | None = None,
) -> list[Tool]:
    """
    Register file adapters bound to the session instance.

    Args:
        service: The session's workspace boundary.
        write_allowed: False omits the mutation tools; True annotates overwrite.
        workspace_label: Short root shown in the dynamic descriptions.
    """
    selected_service = _selected_service(service)
    adapter = _FileToolAdapter(selected_service)
    root_hint = workspace_label or selected_service.workspace.name
    listing_contract = dict(max_result_chars=LISTING_MAX_RESULT_CHARS)
    read_specs = (
        (
            ToolContract(
                name="leggi_documento",
                is_read_only=True,
                is_concurrency_safe=True,
                validate=validate_non_empty,
                description=f"{READ_DOCUMENT_TOOL_DESCRIPTION} Workspace: {root_hint}.",
            ),
            adapter.read,
        ),
        (
            ToolContract(
                name="elenca_cartella",
                is_read_only=True,
                is_concurrency_safe=True,
                **listing_contract,
                description=f"{LIST_FOLDER_TOOL_DESCRIPTION} Workspace: {root_hint}.",
            ),
            adapter.list,
        ),
        (
            ToolContract(
                name="elenca_albero",
                is_read_only=True,
                is_concurrency_safe=True,
                **listing_contract,
                description=f"{LIST_TREE_TOOL_DESCRIPTION} Workspace: {root_hint}.",
            ),
            adapter.tree,
        ),
        (
            ToolContract(
                name="trova_file",
                is_read_only=True,
                is_concurrency_safe=True,
                validate=validate_non_empty,
                **listing_contract,
                description=f"{FIND_FILE_TOOL_DESCRIPTION} Workspace: {root_hint}.",
            ),
            adapter.find,
        ),
        (
            ToolContract(
                name="cerca_testo",
                is_read_only=True,
                is_concurrency_safe=True,
                validate=validate_non_empty,
                **listing_contract,
                description=f"{SEARCH_TEXT_TOOL_DESCRIPTION} Workspace: {root_hint}.",
            ),
            adapter.search,
        ),
    )
    if write_allowed is False:
        return build_tools(read_specs)
    if write_allowed is True:
        write_suffix = (
            f" Workspace: {root_hint}. Scrittura autorizzata in questo turno."
        )
    else:
        write_suffix = f" Workspace: {root_hint}."
    write_specs = (
        (
            ToolContract(
                name="scrivi_file",
                is_read_only=False,
                is_concurrency_safe=False,
                validate=validate_non_empty,
                description=f"{WRITE_FILE_TOOL_DESCRIPTION}{write_suffix}",
            ),
            adapter.write,
        ),
        (
            ToolContract(
                name="modifica_file",
                is_read_only=False,
                is_concurrency_safe=False,
                validate=validate_non_empty,
                description=f"{EDIT_FILE_TOOL_DESCRIPTION}{write_suffix}",
            ),
            adapter.replace,
        ),
        (
            ToolContract(
                name="crea_cartella",
                is_read_only=False,
                is_concurrency_safe=False,
                validate=validate_non_empty,
                description=f"{CREATE_FOLDER_TOOL_DESCRIPTION}{write_suffix}",
            ),
            adapter.mkdir,
        ),
    )
    return build_tools((*write_specs, *read_specs))

def extract_explicit_paths(
    text: str,
    service: WorkspaceFileService | None = None,
) -> list[tuple[str, str]]:
    """
    Extract only explicit path+fence or path+separator blocks from the Final Answer.

    Returns:
        Unique pairs with an already normalized path and non-empty content.
    """
    selected_service = _selected_service(service)
    seen_paths: set[str] = set()
    extracted: list[tuple[str, str]] = []

    def append_if_valid(raw_path: str, block: str) -> None:
        """
        Add a pair only if the line looks like a path, not a sentence.

        Postconditions:
            Each path appears at most once and keeps its first content.
        """
        cleaned_path = raw_path.strip().rstrip(":").strip()
        # Length, punctuation and list markers reduce false positives from natural text.
        if (
            not cleaned_path
            or len(cleaned_path) > _MAX_EXPLICIT_PATH_CHARACTERS
            or any(marker in cleaned_path for marker in (":", "=", "`"))
            or cleaned_path.startswith(("-", "*"))
        ):
            return
        normalized = selected_service.normalize_file_path(cleaned_path)
        # Invalid paths, duplicates or empty bodies are not materializable fallbacks.
        if normalized is None or normalized in seen_paths or not block.strip():
            return
        seen_paths.add(normalized)
        extracted.append((normalized, block))

    # Fences are the most common format when the model skips the tool.
    for match in _PATH_AND_FENCE_PATTERN.finditer(text):
        append_if_valid(match.group("path"), match.group("content"))
    # The native path/---/content protocol takes precedence only for paths not seen yet.
    for match in _PATH_AND_SEPARATOR_PATTERN.finditer(text):
        append_if_valid(match.group("path"), match.group("content"))
    return extracted

def apply_fallback(
    final_output: str,
    turn_written_paths: list[str],
    *,
    service: WorkspaceFileService | None = None,
) -> list[str]:
    """
    Materialize missing files from the Final Answer only with an explicit gate.

    Args:
        final_output: Final output possibly holding explicit file blocks.
        turn_written_paths: Paths already completed by the tools, not to duplicate.
        service: The turn's service; None uses the context-local adapter.

    Returns:
        Only the paths created by the fallback, in order.
    """
    selected_service = _selected_service(service)
    # The fallback cannot turn a Q&A into a spontaneous write.
    if not selected_service.write_allowed:
        return []
    already_written = {
        normalized
        for path in turn_written_paths
        if (normalized := selected_service.normalize_file_path(path)) is not None
    }
    fallback_paths: list[str] = []
    # Evaluate each file separately: one tool write does not block the other missing files.
    for relative_path, raw_content in extract_explicit_paths(
        final_output,
        selected_service,
    ):
        # A path already written by the tool is complete and must not be overwritten by the final text.
        if relative_path in already_written:
            continue
        clean_content = sanitize_file_content(raw_content, relative_path)
        result = selected_service.create_text(relative_path, clean_content)
        # Errors or pre-existing files stay intact and are not announced as fallback.
        if not result.succeeded or result.path is None:
            continue
        already_written.add(result.path)
        fallback_paths.append(result.path)
    return fallback_paths

def _selected_service(
    service: WorkspaceFileService | None,
) -> WorkspaceFileService:
    """Return the explicit instance or the context-local legacy service."""
    return service if service is not None else get_default_file_service()

def _clean_raw_path(raw_path: str) -> str:
    """Remove quotes and structural prefixes without interpreting the path."""
    text = str(raw_path or "").strip().strip("\"'")
    text = _PATH_PREFIX_PATTERN.sub("", text)
    # Gemma sometimes copies the phrase «path relativo» in front of the real path.
    text = _INSTRUCTIONAL_PATH_PREFIX_PATTERN.sub("", text)
    text = _ROW_PREFIX_PATTERN.sub("", text)
    return _ROW_SUFFIX_PATTERN.sub("", text).strip()

# Below this threshold the tail is a name with spaces («notes vecchie»), not pasted prose.
_MIN_PASTED_TAIL_WORDS = 3


def _exists(
    service: WorkspaceFileService,
    path: str,
    *,
    folder: bool,
) -> bool:
    """True if the sandboxed path exists and has the expected type."""
    if folder:
        resolved = service.resolve_directory(path)
        return resolved is not None and resolved.absolute.is_dir()
    resolved = service.resolve_file(path)
    return resolved is not None and resolved.absolute.is_file()


def recover_existing_path(
    action_input: str,
    service: WorkspaceFileService,
    *,
    folder: bool,
) -> str:
    """
    Gemma often does not break the line after the Action Input and keeps going on it
    («Articolo_01.pagesTrovato il file…», «.Tuttavia non vedo…»): each time
    an iteration is lost. If the whole path does not exist, take the longest
    prefix that really exists, as long as the rest is clearly prose.

    Returns:
        The recovered path, or the cleaned input if there is no safe recovery.
    """
    cleaned = _clean_raw_path(str(action_input or ""))
    if _exists(service, cleaned, folder=folder):
        return cleaned
    for end in range(len(cleaned) - 1, 0, -1):
        prefix = cleaned[:end].rstrip()
        tail = cleaned[end:]
        if len(tail.split()) < _MIN_PASTED_TAIL_WORDS:
            continue
        # Credible boundary: a space, a glued capital letter or the root's «.».
        if not (tail[0].isspace() or tail[0].isupper() or prefix == "."):
            continue
        if prefix and _exists(service, prefix, folder=folder):
            return prefix
    return cleaned


def _format_error_message(raw_text: str) -> str:
    """
    Build a format Observation that steers Gemma towards path/---/body.

    Args:
        raw_text: Raw Action Input rejected by the parser.

    Returns:
        Errore: message with a concrete example, especially for the comma shorthand.
    """
    first_line = raw_text.split("\n", 1)[0].strip()
    # Typical Gemma failure: ``tmp/a_duo,AAA`` instead of the three-line protocol.
    if "," in first_line:
        return (
            "Errore: non usare destinazione,contenuto sulla stessa riga.\n"
            "Formato obbligatorio (UN file per Action):\n"
            f"tmp/a.txt\n{FILE_SEPARATOR}\nAAA\n"
            "Poi ripeti Action per il secondo file."
        )
    if first_line.startswith("/"):
        relative = first_line.lstrip("/")
        return (
            f"Errore: non usare lo slash iniziale. Path relativo: {relative}\n"
            f"Poi una riga {FILE_SEPARATOR} e il contenuto completo del file."
        )
    if FILE_SEPARATOR not in raw_text:
        example = first_line or "sito/index.html"
        return (
            "Errore: manca il contenuto del file.\n"
            f"Usa:\n{example}\n{FILE_SEPARATOR}\n"
            "<contenuto completo>"
        )
    return (
        "Errore: formato non valido.\n"
        f"Usa path relativo:\nsito/index.html\n{FILE_SEPARATOR}\n"
        "<contenuto>"
    )

def _normalize_content_prefix(text: str) -> str:
    """Remove content/language markers the model added in front of the body."""
    lines = text.strip("\n").split("\n", 1)
    first_line = _CONTENT_PREFIX_PATTERN.sub("", lines[0].strip())
    remainder = lines[1] if len(lines) > 1 else ""
    # An isolated language marker comes from the serialized fence, not from the file.
    if first_line.casefold() in _KNOWN_LANGUAGE_MARKERS:
        first_line = ""
    # YAML block-scalar markers after "Contenuto:" are not part of the file.
    if first_line in _YAML_BLOCK_MARKERS:
        first_line = ""
    # «riga ---» artifact copied from the prompt instructions.
    if _ROW_SEPARATOR_ARTIFACT_PATTERN.match(first_line):
        first_line = ""
    # Reassemble without introducing an empty line when one of the two parts is missing.
    if first_line and remainder:
        return f"{first_line}\n{remainder}"
    return first_line or remainder

def _recover_extension_from_context(
    normalized_path: str,
    context: str,
    service: WorkspaceFileService,
) -> str:
    """
    If the path has no extension but the context cites ``path.ext``, use the full path.

    Args:
        normalized_path: Path already sandboxed from the first Action Input line.
        context: Rest of the Action Input (body + any Thought).
        service: Service used to re-validate the recovered candidate.

    Returns:
        The original path, or the version with the extension recovered from the context.
    """
    # A path that already has an extension must not be reinterpreted.
    if Path(normalized_path).suffix:
        return normalized_path
    # Look for citations like tmp/a_duo.txt in the same Action Input.
    for match in _PATH_WITH_EXTENSION_PATTERN.finditer(context):
        candidate_raw = match.group("path")
        candidate = service.normalize_file_path(_clean_raw_path(candidate_raw))
        # Accept only extensions of the same stem (tmp/a_duo → tmp/a_duo.txt).
        if candidate is None:
            continue
        stem = Path(candidate).with_suffix("").as_posix()
        if stem == normalized_path:
            return candidate
    return normalized_path

def _split_single_line_path_content(
    expanded: str,
    service: WorkspaceFileService,
) -> tuple[str, str] | None:
    """
    Parse the single-line ``path.ext,content`` often used by Gemma.

    Returns:
        A path/content pair, or None if it is not the shorthand with an extension.
    """
    match = _SINGLE_LINE_PATH_CONTENT_PATTERN.match(expanded.strip())
    # Without an explicit extension the shorthand is ambiguous and is rejected upstream.
    if match is None:
        return None
    normalized_path = service.normalize_file_path(_clean_raw_path(match.group("path")))
    content = match.group("content").strip()
    # An invalid path or empty body must not produce a silent write.
    if normalized_path is None or not content:
        return None
    return normalized_path, content

def _split_path_and_content(
    raw_text: str,
    service: WorkspaceFileService,
) -> tuple[str, str] | None:
    """Split the Action Input in the tolerated formats and normalize only the path."""
    expanded = expand_literal_newlines(raw_text)
    # The explicit separator allows multiline bodies and content with newlines.
    separator_match = _SEPARATOR_LINE_PATTERN.search(expanded)
    if separator_match is not None:
        raw_path = expanded[: separator_match.start()]
        content = expanded[separator_match.end() :]
        normalized_path = service.normalize_file_path(_clean_raw_path(raw_path))
        # An invalid path makes the whole payload ambiguous.
        if normalized_path is None:
            return None
        recovered = _recover_extension_from_context(
            normalized_path,
            content,
            service,
        )
        return recovered, content.strip("\n")

    # Single-line path.ext,body: a tolerated alternative to the three-line protocol.
    if "\n" not in expanded:
        single_line = _split_single_line_path_content(expanded, service)
        if single_line is not None:
            return single_line

    first_line, separator, remainder = expanded.partition("\n")
    normalized_path = service.normalize_file_path(_clean_raw_path(first_line))
    # The compact format requires both a separator newline and useful content.
    if normalized_path is None or not separator or not remainder.strip():
        return None
    recovered = _recover_extension_from_context(
        normalized_path,
        remainder,
        service,
    )
    return recovered, _normalize_content_prefix(remainder)

def _sanitize_patch_block(block: str) -> str:
    """
    Remove only outer markdown fences, without cutting inner code.

    Returns:
        Block preserved byte for byte apart from optional ``` wrappers.
    """
    text = block
    text = _OPENING_FENCE_PATTERN.sub("", text)
    text = _CLOSING_FENCE_PATTERN.sub("", text)
    return text

def _strip_block_delimiters(block: str) -> str:
    """Strip at most one boundary newline left by the protocol markers."""
    if block.startswith("\r\n"):
        block = block[2:]
    elif block.startswith("\n"):
        block = block[1:]
    if block.endswith("\r\n"):
        block = block[:-2]
    elif block.endswith("\n"):
        block = block[:-1]
    return block

def _split_path_old_new(
    raw_text: str,
    service: WorkspaceFileService,
) -> tuple[str, str, str] | None:
    """
    Split the modifica_file Action Input into path, OLD and NEW.

    Returns:
        Normalized tuple, or None if the markers or the path are missing.
    """
    expanded = expand_literal_newlines(raw_text)
    old_match = _MARKER_OLD_LINE_PATTERN.search(expanded)
    new_match = _MARKER_NEW_LINE_PATTERN.search(expanded)
    if old_match is None or new_match is None:
        return None
    if new_match.start() <= old_match.start():
        return None
    path_header = expanded[: old_match.start()].strip()
    if not path_header:
        return None
    path_line = path_header.splitlines()[0]
    normalized_path = service.normalize_file_path(_clean_raw_path(path_line))
    if normalized_path is None:
        return None
    old_block = _cut_protocol_tail(
        _sanitize_patch_block(
            _strip_block_delimiters(expanded[old_match.end() : new_match.start()])
        )
    )
    new_block = _cut_protocol_tail(
        _sanitize_patch_block(
            _strip_block_delimiters(expanded[new_match.end() :])
        )
    )
    return normalized_path, old_block, new_block
