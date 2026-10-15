"""Turn routing from the card, local greeting, history spill and scope lock."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from assistant.turn_schema import TurnSchema

TurnPath = Literal["direct", "react"]

ANTI_GOLD_PLATING = (
    "Scope lock: fai solo quanto chiesto. "
    "Non aggiungere feature, refactor o miglioramenti non richiesti. "
    "Non creare helper/utilità per operazioni una tantum. "
    "Tre righe simili sono meglio di un'astrazione prematura."
)

LOCAL_CHAT_REASON = "chat breve senza contenuto da ricercare"

_CHAT_PUNCTUATION = re.compile(r"[,;:]+")
_LOCAL_CHAT_INTENT = re.compile(
    r"^(?:"
    r"ciao(?:\s+ciao)*|hey|hi|hello|salve|"
    r"buongiorno|buonasera|buonanotte|"
    r"grazie(?:\s+mille)?|thanks|thank\s+you|"
    r"prego|di\s+niente|"
    r"(?:ok(?:ay)?|va\s+bene|perfetto|ottimo|capito)(?:\s+grazie(?:\s+mille)?)?|"
    r"come\s+stai|how\s+are\s+you"
    r")[\s!.?]*$",
    re.IGNORECASE,
)

_TOOLLESS_KINDS = frozenset({"chat", "explain"})


@dataclass(frozen=True, slots=True)
class RequestAnalysis:
    """Turn signals taken from the schema, not from the sentence wording."""

    write: bool
    workspace: bool
    explanation: bool


@dataclass(frozen=True, slots=True)
class TurnDecision:
    """Outcome used by AgentSession for routing and the write gate."""

    path: TurnPath
    write_allowed: bool
    analysis: RequestAnalysis
    path_reason: str = ""


def neutral_analysis() -> RequestAnalysis:
    """No signal: the text is not classified."""
    return RequestAnalysis(write=False, workspace=False, explanation=False)


def analysis_from_schema(schema: TurnSchema) -> RequestAnalysis:
    """Turn the card into signals for the prompt, without reading the sentence."""
    return RequestAnalysis(
        write=schema.write or schema.deliver == "file",
        workspace=schema.kind == "workspace" or bool(schema.paths or schema.folder),
        explanation=schema.kind == "explain",
    )


def decision_from_schema(schema: TurnSchema) -> TurnDecision:
    """
    Route from the card fields only.

    Closed corpus: every content question goes through the documents (ReAct).
    Only pure chat, with no destination, stays a direct generation.
    """
    analysis = analysis_from_schema(schema)
    has_destination = bool(
        schema.paths
        or schema.folder
        or schema.write
        or schema.deliver == "file"
    )
    if schema.kind == "chat" and not has_destination:
        path: TurnPath = "direct"
        reason = "chat senza contenuto da cercare"
    elif schema.kind in _TOOLLESS_KINDS and not has_destination:
        path = "react"
        reason = "domanda da verificare nei documenti"
    else:
        path = "react"
        reason = "compito con tool"
    write = _write_allowed(schema, path)
    return TurnDecision(
        path=path,
        write_allowed=write,
        analysis=analysis,
        path_reason=reason,
    )


def _file_read_without_write(schema: TurnSchema) -> bool:
    """A path without write, folder or delivery is the file to read."""
    return bool(
        schema.paths
        and not schema.write
        and schema.deliver != "file"
        and not schema.folder.strip()
    )


def _write_allowed(schema: TurnSchema, path: TurnPath) -> bool:
    """ReAct writes, except for file reads and questions about the documents."""
    if schema.write or schema.deliver == "file":
        return True
    if _file_read_without_write(schema):
        return False
    if schema.kind in _TOOLLESS_KINDS:
        return False
    return path == "react"


def open_react_decision() -> TurnDecision:
    """Missing schema: full tools, writes inside the sandbox."""
    return TurnDecision(
        path="react",
        write_allowed=True,
        analysis=neutral_analysis(),
        path_reason="schema assente: ReAct completo",
    )


def greeting_decision() -> TurnDecision:
    """Exact greeting: one local generation, no schema and no tools."""
    return TurnDecision(
        path="direct",
        write_allowed=False,
        analysis=neutral_analysis(),
        path_reason=LOCAL_CHAT_REASON,
    )


def _is_local_chat(request: str) -> bool:
    """
    True if the turn is only a greeting, a thank-you or a confirmation.

    «ciao, chi è il presidente?» stays False: it is not just a greeting.
    """
    text = (request or "").strip()
    if not text:
        return True
    if _LOCAL_CHAT_INTENT.fullmatch(text):
        return True
    normalized = re.sub(r"\s+", " ", _CHAT_PUNCTUATION.sub(" ", text)).strip()
    return bool(_LOCAL_CHAT_INTENT.fullmatch(normalized))


def format_intent_context(decision: TurnDecision) -> str:
    """Serialize the already chosen route for the ReAct prompt."""
    analysis = decision.analysis
    intent_flags = [
        name
        for flag, name in (
            (analysis.write, "scrittura"),
            (analysis.workspace, "workspace"),
            (analysis.explanation, "concettuale"),
        )
        if flag
    ] or ["chat"]
    return "\n".join(
        (
            "INTENT CONTEXT:",
            f"- Intent: {', '.join(intent_flags)}",
            f"- Percorso: {decision.path} ({decision.path_reason})",
            f"- Scrittura consentita: {'sì' if decision.write_allowed else 'no'}",
        )
    )


def capabilities_react() -> tuple[str, ...]:
    """ReAct tools: folder files, summary and date. No network."""
    return ("file", "summary", "date")


# Spilling the chat to a file: no new theme, only the history.
_DOCUMENT_EXTENSIONS = r"txt|md|json|ya?ml|html?"
_DOCUMENT_PATH = re.compile(
    rf"[\w./-]+\.(?:{_DOCUMENT_EXTENSIONS})\b",
    re.IGNORECASE,
)
_TAIL_STYLE = re.compile(
    r"\.?\s*sii?\s+prolisso(?:\s+e\s+approfondisci)?\.?\s*$",
    re.IGNORECASE,
)
_SCAFFOLD_CLAUSE = re.compile(
    r"(?:"
    r"(?:nella|dentro|in)\s+(?:la\s+)?cartella\s+[^\s.,;:]+|"
    r"(?:crea(?:re|ndo)?|realizza(?:re)?)\s+(?:una\s+|la\s+)?cartella"
    r"(?:\s+dal\s+nome)?\s+[^\s.,;:]+|"
    r"al\s+suo\s+interno|"
    r"in\s+quella\s+cartella|"
    r"all['’]interno\s+(?:di(?:lla)?|della\s+cartella|di\s+essa)\s*[^\s.,;:]*|"
    r"crea(?:re|ndo)?\s+(?:un\s+|il\s+)?file\s+[\w./-]+\.\w+|"
    r"(?:un|il)\s+documento\s+[\w./-]+\.\w+|"
    r"in\s+questo\s+documento"
    r")",
    re.IGNORECASE,
)
_TRAILING_FILE_INSTRUCTION = re.compile(
    r"\s+(?:e\s+)?(?:scriv\w*|salv\w*).{0,100}"
    rf"[\w./ -]+\.(?:{_DOCUMENT_EXTENSIONS})\s*$",
    re.IGNORECASE,
)
_WORD = re.compile(r"[^\W\d_]{3,}", re.UNICODE)
_META_STOPWORDS = frozenset(
    {
        "all",
        "approfondisci",
        "bene",
        "cartella",
        "che",
        "chi",
        "cio",
        "ciò",
        "come",
        "con",
        "consultare",
        "cosa",
        "crea",
        "creare",
        "dal",
        "del",
        "della",
        "delle",
        "degli",
        "directory",
        "documents",
        "dunque",
        "esclusivamente",
        "file",
        "fonti",
        "fornisci",
        "interno",
        "nella",
        "nome",
        "online",
        "per",
        "più",
        "prolisso",
        "qual",
        "quale",
        "quali",
        "quelle",
        "quindi",
        "riesci",
        "scrivere",
        "scrivi",
        "sono",
        "sul",
        "suo",
        "trovare",
        "tutto",
        "una",
        "uno",
    }
)
# Spill verbs and deixis: they are not a topic to search for.
_THEMELESS_WORDS = _META_STOPWORDS | frozenset(
    {
        "adesso",
        "avevi",
        "chat",
        "conversazione",
        "detta",
        "detto",
        "documento",
        "gia",
        "già",
        "hai",
        "memoria",
        "nel",
        "nelle",
        "nello",
        "negli",
        "ora",
        "precedente",
        "precedenti",
        "prima",
        "quanto",
        "qui",
        "quella",
        "quelli",
        "quello",
        "questa",
        "queste",
        "questi",
        "questo",
        "riporta",
        "riportala",
        "riportalo",
        "riportami",
        "riportare",
        "risultato",
        "risultati",
        "salva",
        "salvala",
        "salvalo",
        "salvami",
        "salvare",
        "scritta",
        "scritti",
        "scritto",
        "sopra",
        "storico",
        "turno",
        "turni",
    }
)
_SPILL_CUE = re.compile(
    r"(?:"
    r"\briporta(?:mi|re|lo|la)?\s+tutto\b|"
    r"\bsul\s+file\b|"
    r"\bsalva(?:re|lo|la|mi)?\b"
    r")",
    re.IGNORECASE,
)


def _clean_theme(theme: str) -> str:
    """Strip style tails («sii prolisso») from the theme."""
    return _TAIL_STYLE.sub("", (theme or "").strip()).strip(" .;,")


def named_document_path(text: str) -> str | None:
    """First document name mentioned in the sentence, or None."""
    paths = tuple(dict.fromkeys(_DOCUMENT_PATH.findall((text or "").strip())))
    return paths[0] if paths else None


def is_history_spill(text: str) -> bool:
    """
    True if the sentence asks to spill the chat, with no new theme.

    «riportami tutto», «sul file» and «salva» count only if, once the document
    and the save verbs are removed, no topic to search is left. Two paths
    stay ambiguous and do not take this shortcut.
    """
    raw = (text or "").strip()
    if not raw or _SPILL_CUE.search(raw) is None:
        return False
    paths = tuple(dict.fromkeys(_DOCUMENT_PATH.findall(raw)))
    if len(paths) > 1:
        return False
    return not _has_new_theme(raw)


def _has_new_theme(text: str) -> bool:
    """True if a topic remains once paths and save verbs are removed."""
    without_path = _DOCUMENT_PATH.sub(" ", text or "")
    cleaned = _strip_scaffold(without_path)
    return any(
        term not in _THEMELESS_WORDS
        for term in _WORD.findall(cleaned.casefold())
    )


def _strip_scaffold(query: str) -> str:
    """Remove folder/file clauses without touching the theme."""
    stripped = _SCAFFOLD_CLAUSE.sub(" ", query or "")
    stripped = _TRAILING_FILE_INSTRUCTION.sub("", stripped)
    return _clean_theme(" ".join(stripped.split()))
