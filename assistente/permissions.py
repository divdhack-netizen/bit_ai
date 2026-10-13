"""Regole allow/deny path per scritture e letture sopra la sandbox workspace."""

from __future__ import annotations

import json
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

from assistente.config import WORKSPACE

DEFAULT_PERMISSIONS_PATH = WORKSPACE / ".assistente" / "permissions.json"
LEGACY_PERMISSIONS_PATH = WORKSPACE / "assistente" / "permissions.json"


@dataclass(frozen=True, slots=True)
class WritePermissionRules:
    """
    Glob relativi al workspace.

    ``allow`` vuoto = tutti i path sandbox. ``deny`` vince sempre su ``allow``.

    Portata: ``allow`` e ``deny`` regolano le scritture e le mutazioni; la sola
    ``deny`` vale anche per letture ed elenchi (read_document, list_directory,
    list_tree, find_files), così i segreti restano illeggibili e invisibili.
    ``allow`` non amplia le letture: è un perimetro di scrittura, non una
    whitelist di lettura. Negare una cartella nega anche il suo contenuto.
    """

    allow: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()

    def is_write_allowed(self, relative_path: str) -> tuple[bool, str | None]:
        """Valuta un path POSIX già normalizzato. Deny ha priorità su allow."""
        path = _normalizza_path(relative_path)
        pattern_negato = _negato_con_antenati(path, self.deny)
        if pattern_negato is not None:
            return False, f"path negato dalle regole deny: {pattern_negato}"
        if not self.allow:
            return True, None
        for pattern in self.allow:
            if _glob_match(path, pattern) or _match_root(path, pattern):
                return True, None
        return False, "path non presente nelle regole allow"

    def is_read_allowed(self, relative_path: str) -> tuple[bool, str | None]:
        """
        Valuta la lettura di un path POSIX già normalizzato.

        Consulta solo la deny: i segreti restano illeggibili anche quando il
        file esiste. ``allow`` è perimetro di scrittura e non concede letture
        extra. La valutazione include gli antenati del path, così una cartella
        negata rende illeggibile tutto ciò che contiene.
        """
        path = _normalizza_path(relative_path)
        pattern_negato = _negato_con_antenati(path, self.deny)
        if pattern_negato is not None:
            return False, f"path negato dalle regole deny: {pattern_negato}"
        return True, None


def load_write_permission_rules(
    path: Path | None = None,
    *,
    cartella: Path | None = None,
) -> WritePermissionRules:
    """
    Carica JSON allow/deny. File assente o invalido → regole vuote (no-op).

    Path esplicito ha priorità; poi ``.assistente/permissions.json`` della
    cartella documenti, poi default e legacy del progetto. Chiavi ignote
    (es. ``_nota`` documentativa) sono scartate in silenzio.
    """
    if path is not None:
        candidates: tuple[Path | None, ...] = (path,)
    else:
        della_cartella = (
            cartella / ".assistente" / "permissions.json" if cartella is not None else None
        )
        candidates = (della_cartella, DEFAULT_PERMISSIONS_PATH, LEGACY_PERMISSIONS_PATH)
    for candidate in candidates:
        if candidate is None or not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        return WritePermissionRules(
            allow=_as_pattern_tuple(payload.get("allow")),
            deny=_as_pattern_tuple(payload.get("deny")),
        )
    return WritePermissionRules()


def _normalizza_path(relative_path: str) -> str:
    """
    Ripulisce il path senza toccare i nomi.

    Il vecchio ``lstrip("./")`` mangiava il punto iniziale dei dotfile
    (``.env`` diventava ``env``) e li esentava dalla deny: niente strip di
    caratteri, solo spazi e slash di margine.
    """
    return str(relative_path).strip().strip("/")


def _negato_da_deny(path: str, patterns: tuple[str, ...]) -> str | None:
    """Prima deny che matcha il path (o la sua radice per pattern ``**/``)."""
    for pattern in patterns:
        if _glob_match(path, pattern) or _match_root(path, pattern):
            return pattern
    return None


def _negato_con_antenati(path: str, patterns: tuple[str, ...]) -> str | None:
    """Deny sul path o su qualunque cartella antenata: negare la cartella nega il contenuto."""
    pattern = _negato_da_deny(path, patterns)
    if pattern is not None:
        return pattern
    parti = path.split("/")
    for indice in range(len(parti) - 1):
        pattern = _negato_da_deny("/".join(parti[: indice + 1]), patterns)
        if pattern is not None:
            return pattern
    return None


def _match_root(path: str, pattern: str) -> bool:
    """``**/x`` copre anche la radice del workspace, come nei glob convenzionali."""
    if not pattern.startswith("**/"):
        return False
    return _glob_match(path, pattern[3:])


def _glob_match(path: str, pattern: str) -> bool:
    """Match case-sensitive, poi casefold per path con casing misto."""
    return fnmatch(path, pattern) or fnmatch(path.casefold(), pattern.casefold())


def _as_pattern_tuple(value: object) -> tuple[str, ...]:
    """Accetta solo liste/tuple di stringhe non vuote."""
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
