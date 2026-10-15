"""The model proposes the card; the validator settles write and destination."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from assistant.comprehension import (
    FILE_DESTINATION_CUE,
    READ_CUE,
    SAVE_CUE,
    FILE_EXTENSIONS,
    PATH_FILE,
    unsupported_external_action,
    extract_named_folder,
)
from assistant.turn_schema import TurnSchema

MISSING_FILE_MESSAGE = "in quale file?"

_DOCUMENT_EXTENSION = re.compile(
    rf"\.(?:{'|'.join(FILE_EXTENSIONS)})$",
    re.IGNORECASE,
)
_SOFT_KINDS = frozenset({"chat", "explain"})

# Clarifications: what cannot be done and what can be offered instead.
_UNSUPPORTED_ACTION_MESSAGES: dict[str, str] = {
    "email": (
        "Non posso inviare email o messaggi. Posso però prepararti il testo da "
        "mandare: dimmi destinatario e contenuto e te lo scrivo in chat o in un file."
    ),
    "generazione": (
        "Non posso generare immagini, video o PDF. Posso però descriverti la scena "
        "a parole o prepararti il contenuto come testo, anche in un file .txt."
    ),
    "shell": (
        "Non posso eseguire comandi o script nel terminale. Posso però spiegarti il "
        "comando o prepararti uno script da eseguire tu."
    ),
    "prenotazione": (
        "Non posso prenotare o acquistare nulla. Posso però prepararti i dati e i "
        "passaggi da seguire."
    ),
    "audio": (
        "Non posso ascoltare o trascrivere file audio. Incollami il testo e ci "
        "lavoro volentieri."
    ),
}


@dataclass(frozen=True, slots=True)
class ValidationContext:
    """Session state the validator can read, without calling the LLM."""

    last: TurnSchema | None = None
    previous_document: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationOutcome:
    """Card ready for the executor. If clarify is filled, the turn stops."""

    schema: TurnSchema


def validate_turn(
    schema: TurnSchema,
    request: str,
    state: ValidationContext | None = None,
) -> ValidationOutcome:
    """
    Apply empty subject, write and destination.

    The request is used to recover a single missing document name,
    to complete the folder named by the user and to drop the write
    from a read with no save.
    """
    context = state or ValidationContext()
    current = _inherit_empty_subject(schema, context)
    current = _normalize_destination(current)
    current = _prefer_request_name(current, request)
    current = _downgrade_write_without_destination(current)
    current = _promote_document_from_subject(current, request)
    current = _downgrade_read_without_save(current, request)
    current = _downgrade_workspace_without_target(current)
    current = _file_clarification(current, context)
    current = _drop_useless_clarification(current)
    # Last step: an unsupported external action must not reach the tools.
    current = _unsupported_action_clarification(current, request)
    return ValidationOutcome(schema=current)


def _inherit_empty_subject(
    schema: TurnSchema,
    state: ValidationContext,
) -> TurnSchema:
    """If the model leaves the theme empty, reuse the last subject. The kind stays."""
    if schema.subject.strip():
        return schema
    last = state.last
    if last is None or not last.subject.strip():
        return schema
    return replace(schema, subject=last.subject)


def _normalize_destination(schema: TurnSchema) -> TurnSchema:
    """Move into paths a file name that ended up in folder by mistake."""
    if schema.paths:
        return schema

    def _is_file(text: str) -> bool:
        return bool(_DOCUMENT_EXTENSION.search((text or "").strip()))

    folder = schema.folder.strip()
    if folder and _is_file(folder):
        return replace(schema, paths=(folder,), folder="")
    return schema

def _prefer_request_name(
    schema: TurnSchema,
    request: str,
) -> TurnSchema:
    """If the model shortens or translates the folder, use the name from the request."""
    from_text = extract_named_folder(request)
    if not from_text:
        return schema
    folder = schema.folder.strip()
    if not folder or folder.casefold() == from_text.casefold():
        return schema
    text = (request or "").casefold()
    translated = folder.casefold() not in text
    if translated or _incomplete_folder(folder, from_text):
        return replace(schema, folder=from_text)
    return schema


def _incomplete_folder(folder: str, wait: str) -> bool:
    """True if folder is just one segment of the path the request names in full."""
    segments = [part.casefold() for part in wait.split("/") if part]
    if len(segments) < 2:
        return False
    return folder.casefold().strip("/") in segments


def _downgrade_write_without_destination(schema: TurnSchema) -> TurnSchema:
    """write true without a path, folder or delivery file does not open the workspace."""
    if not schema.write:
        return schema
    if schema.paths or schema.folder or schema.deliver == "file":
        if schema.kind in _SOFT_KINDS and (schema.paths or schema.folder):
            return replace(schema, kind="workspace")
        return schema
    kind = "explain" if schema.kind == "workspace" else schema.kind
    if kind == "chat":
        kind = "explain"
    return replace(schema, write=False, kind=kind)


def _promote_document_from_subject(
    schema: TurnSchema,
    request: str,
) -> TurnSchema:
    """A single document name, first from the subject and then from the request."""
    if schema.deliver != "file" or schema.paths:
        return schema
    for text in (schema.subject, request):
        found = tuple(dict.fromkeys(PATH_FILE.findall(text or "")))
        if len(found) == 1:
            name = found[0]
            # Same name in folder: it is the file, not a directory.
            folder = "" if schema.folder.strip() == name else schema.folder
            return replace(schema, paths=(name,), folder=folder)
    return schema


def _unsupported_action_clarification(
    schema: TurnSchema,
    request: str,
) -> TurnSchema:
    """Replace an impossible action with an honest, local clarification.

    The turn opens no tool: the session just answers with the message.
    """
    capability = unsupported_external_action(request)
    if capability is None:
        return schema
    return replace(
        schema,
        kind="chat",
        write=False,
        deliver="chat",
        paths=(),
        folder="",
        clarify=_UNSUPPORTED_ACTION_MESSAGES[capability],
    )


def _downgrade_workspace_without_target(schema: TurnSchema) -> TurnSchema:
    """A workspace without write, path, folder or delivery is not actionable.

    The model falls back to ``kind=workspace`` when it does not recognize the task:
    without a real destination the tools have nothing to do, so the turn becomes a
    local answer instead of opening ReAct for nothing.
    """
    if schema.kind != "workspace":
        return schema
    if schema.write or schema.paths or schema.folder or schema.deliver == "file":
        return schema
    return replace(schema, kind="explain")


def _downgrade_read_without_save(
    schema: TurnSchema,
    request: str,
) -> TurnSchema:
    """A read with no verb and no save destination does not write."""
    if schema.deliver != "file" and not schema.write:
        return schema
    text = request or ""
    if not READ_CUE.search(text):
        return schema
    if SAVE_CUE.search(text) or FILE_DESTINATION_CUE.search(text):
        return schema
    return replace(schema, write=False, deliver="chat")


def _file_clarification(
    schema: TurnSchema,
    state: ValidationContext,
) -> TurnSchema:
    """deliver file without a document and without a previous file does not write to the root."""
    if schema.deliver != "file":
        return schema
    if any(_DOCUMENT_EXTENSION.search(path) for path in schema.paths):
        return schema
    if state.previous_document:
        return schema
    return replace(schema, clarify=MISSING_FILE_MESSAGE)


_CODE_CLARIFY = frozenset({MISSING_FILE_MESSAGE})


def _drop_useless_clarification(schema: TurnSchema) -> TurnSchema:
    """The model's question stays only if the write has no destination."""
    if not schema.clarify or schema.clarify in _CODE_CLARIFY:
        return schema
    clear_workspace = schema.kind == "workspace" and bool(schema.paths or schema.folder)
    if clear_workspace:
        return replace(schema, clarify="")
    return schema
