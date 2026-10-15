"""Deterministic comprehension: text rules, local card and destinations.

The local card decides clear turns without the LLM (questions, reads, a single
new destination). When in doubt it returns None and the session asks the model for the card.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from assistant.turn_schema import MAX_PATHS, MAX_SUBJECT_CHARS, TurnSchema

# Single source of file extensions for comprehension, validator and session.
# Every new file type is added here only.
FILE_EXTENSIONS: tuple[str, ...] = (
    "py", "js", "ts", "tsx", "jsx",
    "htm", "html", "css",
    "json", "md", "txt",
    "java", "toml", "yml", "yaml",
    "pdf", "docx", "pages",
    "png", "jpg", "jpeg", "webp",
)
_FILE_EXTENSIONS = "|".join(FILE_EXTENSIONS)

_SPACES = re.compile(r"\s+")
# File paths cited in the text, in order of appearance.
PATH_FILE = re.compile(
    rf"[\w./-]+\.(?:{_FILE_EXTENSIONS})\b",
    re.IGNORECASE,
)
_FOLDER = re.compile(
    r"\b(?:cartella|directory|dir)\s+[`'\"]?([A-Za-z0-9_./-]+)[`'\"]?",
    re.IGNORECASE,
)
_FOLDER_FROM_NAME = re.compile(
    r"\b(?:cartella|directory|dir)\s+dal\s+nome\s+[`'\"]?([A-Za-z0-9_./-]+)[`'\"]?",
    re.IGNORECASE,
)
_CREATED_FOLDER = re.compile(
    r"\bcrea(?:re|ndo)?\s+(?:una\s+|la\s+)?(?:cartella|directory|dir)\s+"
    r"(?:dal\s+nome\s+)?[`'\"]?/?([A-Za-z0-9_./-]+)[`'\"]?",
    re.IGNORECASE,
)
_CONTAINER_FOLDER = re.compile(
    r"\b(?:nella|dentro(?:\s+la)?|in)\s+(?:la\s+)?"
    r"(?:cartella|directory|dir)\s+[`'\"]?/?([A-Za-z0-9_./-]+)[`'\"]?",
    re.IGNORECASE,
)
_FOLDER_STOPWORDS = frozenset(
    {
        "dal", "di", "nome", "una", "la", "un", "il",
        # «cartella attuale/corrente/presente» is the root, not a folder named «attuale».
        "attuale", "corrente", "presente", "questa", "stessa", "principale",
    }
)
# Document type without a file name (e.g. «documento .txt»).
_UNNAMED_DOCUMENT = re.compile(
    rf"(?:"
    rf"(?:un\s+)?documento\s*\.?\s*(?:{_FILE_EXTENSIONS})\b|"
    rf"(?:in\s+)?(?:un\s+)?file\s+(?:di\s+testo|txt)\b|"
    rf"\bfile\s+\.(?:{_FILE_EXTENSIONS})\b"
    rf")",
    re.IGNORECASE,
)
_WRITE_GERUND = re.compile(
    r"(?:"
    r"scrivend\w*|salvand\w*|creand\w*|generand\w*|"
    r"mettendol[oaie]|riassumendol[oaie]|"
    r"writing\s+(?:it\s+)?(?:to|in|into)|"
    r"saving\s+(?:it\s+)?(?:to|as|in)"
    r")",
    re.IGNORECASE,
)
NO_FILE_CONSTRAINT = re.compile(
    r"(?:"
    r"\bniente\s+file\b|"
    r"\bnessun\s+file\b|"
    r"\bsenza\s+(?:creare\s+|salvare\s+|scrivere\s+)?file\b|"
    r"\bnon\s+(?:creare|salvare|scrivere|toccare)\s+"
    r"(?:un\s+|il\s+|alcun\s+)?file\b|"
    r"\bsolo\s+(?:in\s+)?chat\b|"
    r"\bno\s+files?\b|"
    r"\bdon'?t\s+(?:write|save|create)\s+(?:a\s+)?files?\b"
    r")",
    re.IGNORECASE,
)
# Short deictic follow-up: «e il secondo?» is not a standalone question.
_SHORT_DEICTIC = re.compile(
    r"^(?:"
    r"(?:e\s+)?(?:il|la|lo|i|le)\s+"
    r"(?:prim[oaie]|second[oaie]|terz[oaie]|quart[oaie]|quint[oaie]|ultim[oaie]?)"
    r"|"
    r"(?:e\s+)?(?:quello|quella|questo|questa|quelli|quelle)"
    r"|"
    r"the\s+(?:first|second|third|fourth|fifth|last)(?:\s+one)?"
    r")\s*\??$",
    re.IGNORECASE,
)
READ_CUE = re.compile(
    r"\b(?:leggi|leggere|riassumi|riassumere|dimmi|spiega|riporta)\b",
    re.IGNORECASE,
)
# Clitics («salvalo», «scrivimi») stay on the same verb.
SAVE_CUE = re.compile(
    r"\b(?:scrivi\w*|salva\w*|crea\w*|accoda\w*)\b",
    re.IGNORECASE,
)
# «riassumi il documento in y.txt» has a read verb, but the file is the destination.
FILE_DESTINATION_CUE = re.compile(
    rf"\b(?:in|su|sul|sulla|dentro|nel|nello|nella)\s+"
    rf"(?:(?:un|il|lo|la)\s+)?(?:file\s+)?"
    rf"[\w./-]+\.(?:{_FILE_EXTENSIONS})\b",
    re.IGNORECASE,
)
# Mutations of existing files and implicit writes («appuntami», «in un txt»):
# the local card does not decide them, they go to the LLM.
_AMBIGUOUS_CUE = re.compile(
    r"\b(?:modific\w*|aggiorn\w*|aggiung\w*|corregg\w*|corress\w*|sostituisc\w*|"
    r"cambi\w*|rinomin\w*|elimin\w*|cancell\w*|rimuov\w*|spost\w*|"
    r"inserisc\w*|mett\w*|append\w*|"
    r"appunt\w*|annot\w*|archivi\w*|memorizz\w*|registr\w*|"
    r"segna(?:mi|lo|la|li|le|telo)?)\b"
    r"|(?<![.\w])txt\b",
    re.IGNORECASE,
)
# Below this threshold, with no path or folder, the text is too short to decide.
_MIN_LOCAL_WORDS = 3
# Real-world actions the agent cannot perform: better to say so than to pretend.
# The prefix avoids descriptive false positives («spiegami come mandare…»,
# «una guida per prenotare…»), which are ordinary informational questions.
_NON_INSTRUCTION = (
    r"(?<!\bcome\s)(?<!\bper\s)(?<!\bspiegami\s)(?<!\bspiega\s)(?<!\bsi\s)"
)
_UNSUPPORTED_ACTIONS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "email",
        re.compile(
            _NON_INSTRUCTION
            + r"\b(?:manda|mandami|mandagli|mandale|mandi|mandare|"
            r"invia|inviami|inviagli|inviare|inviarlo|inviarla|spedisci|spedire)\b"
            r"[\s\S]{0,40}\b(?:e-?mail|mail|messaggio|messaggi|sms|whatsapp|telegram)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "generazione",
        re.compile(
            _NON_INSTRUCTION
            + r"\b(?:genera|generami|generare|crea|creami|creare|disegna|"
            r"disegnami|disegnare|produci|renderizza)\b[\s\S]{0,40}"
            r"(?:\b(?:immagine|foto|fotografia|illustrazione|disegno|"
            r"render|video|logo)\b|\bpdf\b(?!\.))",
            re.IGNORECASE,
        ),
    ),
    (
        "shell",
        re.compile(
            _NON_INSTRUCTION
            + r"\b(?:esegui|eseguilo|eseguila|eseguire|lancia|lanciare|"
            r"avvia|avviare)\b[\s\S]{0,40}"
            r"\b(?:comando|command|shell|terminale|script)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "prenotazione",
        re.compile(
            _NON_INSTRUCTION
            + r"\b(?:prenota|prenotami|prenotare|prenotiamo)\b[\s\S]{0,40}"
            r"\b(?:volo|treno|hotel|albergo|tavolo|ristorante|camera|"
            r"biglietto|biglietti|noleggio|auto)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "audio",
        re.compile(
            _NON_INSTRUCTION
            + r"(?:\b(?:ascolta|ascoltare|trascrivi|trascrivere)\b[\s\S]{0,40}"
            r"\baudio\b"
            r"|\baudio\b[\s\S]{0,40}\b(?:trascrivi|trascrivere|ascolta)\b)",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class TurnComprehension:
    """Theme and paths of the turn, already fixed by the card."""

    task: str
    paths: tuple[str, ...]


def format_comprehension_block(comprehension: TurnComprehension) -> str:
    """
    Serialize COMPITO for the prompt, or an empty string if no injection is needed.

    Returns:
        A «COMPITO: …» line, or "".
    """
    if not comprehension.task:
        return ""
    return f"COMPITO: {comprehension.task}"


def normalize(request: str) -> str:
    """Collapse whitespace; does not rewrite the wording (no spellcheck)."""
    return _SPACES.sub(" ", str(request or "")).strip()


def extract_paths(text: str) -> tuple[str, ...]:
    """Explicit file paths, without duplicates, in order of appearance."""
    return tuple(dict.fromkeys(PATH_FILE.findall(text or "")))


def unsupported_external_action(request: str) -> str | None:
    """
    Requested external capability that the agent does not have, or None.

    Returns:
        One of email, generazione, shell, prenotazione, audio; or None.
    """
    text = request or ""
    for capability, pattern in _UNSUPPORTED_ACTIONS:
        if pattern.search(text):
            return capability
    return None


def local_card(
    request: str,
    exists: Callable[[str], bool],
) -> TurnSchema | None:
    """
    A certain card without the LLM, or None if the text is ambiguous.

    It decides only reads and questions, and writes with an explicit verb and a
    single new destination. Edits, deictic follow-ups and writes without a
    destination stay with the model.

    Args:
        request: Original message of the turn.
        exists: True if the path already exists in the documents folder (source).

    Returns:
        TurnSchema to pass to the validator, or None.
    """
    text = normalize(request)
    if not text or _SHORT_DEICTIC.fullmatch(text):
        return None
    subject = text[:MAX_SUBJECT_CHARS]
    if unsupported_external_action(text) is not None:
        # validate_turn writes the honest clarification.
        return TurnSchema(kind="explain", subject=subject)
    paths = extract_paths(text)
    folder = extract_named_folder(text) or ""
    if not paths and not folder and len(text.split()) < _MIN_LOCAL_WORDS:
        return None
    no_file = NO_FILE_CONSTRAINT.search(text) is not None
    if not no_file and _AMBIGUOUS_CUE.search(text):
        return None
    write_verb = bool(
        SAVE_CUE.search(text) or _WRITE_GERUND.search(text)
    )
    file_hint = bool(
        _UNNAMED_DOCUMENT.search(text) or FILE_DESTINATION_CUE.search(text)
    )
    if no_file or not (write_verb or file_hint):
        return TurnSchema(
            kind="workspace" if paths or folder else "explain",
            subject=subject,
            paths=paths[:MAX_PATHS],
            folder=folder,
        )
    new_paths = [path for path in paths if not exists(path)]
    if write_verb and len(new_paths) == 1 and len(paths) <= MAX_PATHS:
        return TurnSchema(
            kind="workspace",
            write=True,
            deliver="file",
            subject=subject,
            paths=paths,
            folder=folder,
        )
    return None


def _clean_folder_name(raw: str | None) -> str | None:
    """Normalize a folder name and drop grammatical tokens."""
    if not raw:
        return None
    candidate = raw.strip().rstrip("/.").lstrip("/")
    if not candidate or candidate == ".":
        return None
    if candidate.casefold() in _FOLDER_STOPWORDS:
        return None
    return candidate


def extract_named_folder(text: str) -> str | None:
    """Folder to create, an optional container, or the first useful mention."""
    from_match = _FOLDER_FROM_NAME.search(text)
    create_match = _CREATED_FOLDER.search(text)
    box_match = _CONTAINER_FOLDER.search(text)
    child = _clean_folder_name(
        (from_match or create_match).group(1) if (from_match or create_match) else None
    )
    parent = _clean_folder_name(
        box_match.group(1) if box_match else None
    )
    if parent and child and parent.casefold() != child.casefold():
        return f"{parent}/{child}"
    if child:
        return child
    if parent:
        return parent
    fall_match = _FOLDER.search(text)
    return _clean_folder_name(
        fall_match.group(1) if fall_match else None
    )


def join_path_in_folder(file_path: str, folder: str | None) -> str:
    """Nest the file in the named folder, without duplicating the prefix."""
    if not folder:
        return file_path
    if file_path == folder or file_path.startswith(f"{folder}/"):
        return file_path
    name = file_path.rsplit("/", 1)[-1]
    return f"{folder}/{name}"


def resolve_destination(
    excerpt: str | None,
    folder: str | None,
    previous: str | None,
) -> str | None:
    """
    Write destination: the turn path, nested, or the previous document.

    A follow-up that cites only the basename reuses the already written nested path.
    """
    if not excerpt:
        return previous
    nested = join_path_in_folder(excerpt, folder)
    if (
        previous
        and not folder
        and "/" not in excerpt
        and previous.rsplit("/", 1)[-1] == excerpt.rsplit("/", 1)[-1]
    ):
        return previous
    return nested
