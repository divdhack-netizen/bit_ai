"""JSON turn card: local repair, Ollama schema and stable-prefix prompt."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, fields
from typing import Any

NUM_PREDICT_SCHEMA = 160
MAX_PATHS = 3
MAX_SUBJECT_CHARS = 160
MAX_CLARIFY_CHARS = 200

KINDS = ("chat", "explain", "workspace")
DELIVERS = ("chat", "file")

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_OPENING_DELIMITER = "<richiesta>"
_CLOSING_DELIMITER = "</richiesta>"


@dataclass(frozen=True, slots=True)
class TurnSchema:
    """Flat turn card. Empty strings and empty tuples are the defaults."""

    kind: str = "chat"
    write: bool = False
    deliver: str = "chat"
    subject: str = ""
    paths: tuple[str, ...] = ()
    folder: str = ""
    clarify: str = ""


@dataclass(frozen=True, slots=True)
class RepairOutcome:
    """Successful parse, or the error to send back to the model once."""

    schema: TurnSchema | None
    error: str | None
    repaired: bool


@dataclass(frozen=True, slots=True)
class FollowupBundle:
    """Short context from the last schema, without the whole history."""

    kind: str = ""
    subject: str = ""
    path: str = ""
    outcome: str = ""


def schema_json_ollama() -> dict[str, Any]:
    """Single-object schema passed to Ollama's format field, outside options."""
    string = {"type": "string"}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "kind": {"type": "string", "enum": list(KINDS)},
            "write": {"type": "boolean"},
            "deliver": {"type": "string", "enum": list(DELIVERS)},
            "subject": string,
            "paths": {
                "type": "array",
                "items": string,
                "maxItems": MAX_PATHS,
            },
            "folder": string,
            "clarify": string,
        },
        "required": [item.name for item in fields(TurnSchema)],
    }


SCHEMA_PREFIX = """Compila solo un oggetto JSON con i campi dello schema.
L'agente lavora solo sui documenti della cartella corrente, senza Internet.
kind: chat, explain, workspace.
chat solo per saluti, ringraziamenti o commenti sulla conversazione.
explain per ogni domanda su un tema: la risposta va cercata nei documenti.
workspace per leggere, cercare, elencare, riassumere o scrivere file.
write è true solo per creare, salvare o modificare un file .txt.
Un path di destinazione per un risultato (riassunto, note, output) è write true e deliver file.
Leggere e rispondere in chat è write false e deliver chat.
clarify vuota se la richiesta è chiara.
Se la richiesta continua il turno precedente e non nomina un tema nuovo, riusa kind e subject.

Esempio. Richiesta: quanto pesa una balenottera azzurra?
{"kind":"explain","write":false,"deliver":"chat","subject":"peso balenottera azzurra","paths":[],"folder":"","clarify":""}

Esempio. Richiesta: leggi il file note.pdf e dimmi di cosa parla.
{"kind":"workspace","write":false,"deliver":"chat","subject":"note.pdf","paths":["note.pdf"],"folder":"","clarify":""}

Esempio. Richiesta: riassumi report.docx e salva il riassunto in sintesi.txt.
{"kind":"workspace","write":true,"deliver":"file","subject":"riassunto report.docx","paths":["report.docx","sintesi.txt"],"folder":"","clarify":""}

Esempio. Richiesta: grazie, perfetto.
{"kind":"chat","write":false,"deliver":"chat","subject":"","paths":[],"folder":"","clarify":""}
"""


def format_schema_prompt(
    request: str,
    iso: str,
    followup: FollowupBundle | None = None,
    error: str = "",
) -> str:
    """
    Stable prefix, then date, follow-up, any parse error and the request.

    The user text stays between delimiters and does not enter the instructions.
    """
    bundle = followup or FollowupBundle()
    error_block = f"Errore JSON: {error.strip()}\n" if error.strip() else ""
    return (
        f"{SCHEMA_PREFIX}\n"
        f"Data: {iso}\n"
        f"{_format_followup(bundle)}\n"
        f"{error_block}"
        f"{_OPENING_DELIMITER}\n"
        f"{str(request or '').strip()}\n"
        f"{_CLOSING_DELIMITER}\n"
    )


def load_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """First JSON object in the text, or a parse error."""
    excerpt = _extract_object(_without_fence(text))
    if excerpt is None:
        return None, "JSON oggetto assente."
    try:
        payload = json.loads(excerpt)
    except json.JSONDecodeError as error:
        return None, f"JSON non valido: {error.msg}."
    if not isinstance(payload, dict):
        return None, "JSON non è un oggetto."
    return payload, None


def repair_json_schema(text: str) -> RepairOutcome:
    """
    Strip fences, take the first object and fill in the defaults.

    Returns:
        Schema if kind is one of the allowed values, otherwise the parse error.
    """
    payload, error = load_object(text)
    if payload is None:
        return RepairOutcome(schema=None, error=error, repaired=False)
    repaired = set(payload) != {item.name for item in fields(TurnSchema)}
    kind = _enum_text(payload.get("kind"), KINDS)
    if kind is None:
        return RepairOutcome(schema=None, error="Manca kind valido.", repaired=True)
    deliver = _enum_text(payload.get("deliver"), DELIVERS) or "chat"
    schema = TurnSchema(
        kind=kind,
        write=_as_bool(payload.get("write")),
        deliver=deliver,
        subject=_short_text(payload.get("subject"), MAX_SUBJECT_CHARS),
        paths=_short_list(payload.get("paths"), MAX_PATHS, MAX_SUBJECT_CHARS),
        folder=_short_text(payload.get("folder"), MAX_SUBJECT_CHARS),
        clarify=_short_text(payload.get("clarify"), MAX_CLARIFY_CHARS),
    )
    return RepairOutcome(schema=schema, error=None, repaired=repaired)


def _format_followup(bundle: FollowupBundle) -> str:
    """Short lines: the 9B prefill must not re-read the history."""
    outcome = " ".join(bundle.outcome.split())[:240]
    return (
        "Follow-up:\n"
        f"- kind: {bundle.kind}\n"
        f"- subject: {bundle.subject}\n"
        f"- path: {bundle.path}\n"
        f"- esito: {outcome}"
    )


def _without_fence(text: str) -> str:
    """Prefer the content of the first fence, otherwise the raw text."""
    match = _FENCE.search(text or "")
    if match is None:
        return text or ""
    return match.group(1)


def _extract_object(text: str) -> str | None:
    """First object with balanced braces, ignoring braces inside strings."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _enum_text(value: object, allowed: tuple[str, ...]) -> str | None:
    """Case-sensitive enum value, or None if missing or not in the list."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if cleaned in allowed:
        return cleaned
    return None


def _as_bool(value: object) -> bool:
    """Accept bools and the true/false strings that a small model often emits."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "1", "sì", "si", "yes"}
    return False


def _short_text(value: object, limit: int) -> str:
    """Single-line string, cut at the limit."""
    if not isinstance(value, str):
        return ""
    collapsed = " ".join(value.replace("\n", " ").split())
    return collapsed[:limit]


def _short_list(value: object, maximum: int, entry_limit: int) -> tuple[str, ...]:
    """First non-empty entries, each already cut."""
    if not isinstance(value, list):
        return ()
    entries: list[str] = []
    for item in value:
        text = _short_text(item, entry_limit)
        if not text:
            continue
        entries.append(text)
        if len(entries) >= maximum:
            break
    return tuple(entries)
