"""Allow/deny path rules for writes and reads on top of the workspace sandbox."""

from __future__ import annotations

import json
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

from assistant.config import WORKSPACE

# Per-folder config directory; the Italian name is still read for existing setups.
CONFIG_DIR_NAME = ".assistant"
LEGACY_CONFIG_DIR_NAME = ".assistente"
DEFAULT_PERMISSIONS_PATH = WORKSPACE / CONFIG_DIR_NAME / "permissions.json"
LEGACY_PERMISSIONS_PATH = WORKSPACE / "assistant" / "permissions.json"


@dataclass(frozen=True, slots=True)
class WritePermissionRules:
    """
    Globs relative to the workspace.

    Empty ``allow`` = every sandbox path. ``deny`` always wins over ``allow``.

    Scope: ``allow`` and ``deny`` govern writes and mutations; ``deny`` alone
    also applies to reads and listings (read_document, list_directory,
    list_tree, find_files), so secrets stay unreadable and invisible.
    ``allow`` does not widen reads: it is a write perimeter, not a read
    whitelist. Denying a folder also denies its content.
    """

    allow: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()

    def is_write_allowed(self, relative_path: str) -> tuple[bool, str | None]:
        """Evaluate an already normalized POSIX path. Deny takes priority over allow."""
        path = _normalize_path(relative_path)
        denied_pattern = _denied_with_ancestors(path, self.deny)
        if denied_pattern is not None:
            return False, f"path negato dalle regole deny: {denied_pattern}"
        if not self.allow:
            return True, None
        for pattern in self.allow:
            if _glob_match(path, pattern) or _match_root(path, pattern):
                return True, None
        return False, "path non presente nelle regole allow"

    def is_read_allowed(self, relative_path: str) -> tuple[bool, str | None]:
        """
        Evaluate reading an already normalized POSIX path.

        Only the deny list is consulted: secrets stay unreadable even when the
        file exists. ``allow`` is a write perimeter and grants no extra reads.
        The check includes the path's ancestors, so a denied folder makes
        everything it contains unreadable.
        """
        path = _normalize_path(relative_path)
        denied_pattern = _denied_with_ancestors(path, self.deny)
        if denied_pattern is not None:
            return False, f"path negato dalle regole deny: {denied_pattern}"
        return True, None


def load_write_permission_rules(
    path: Path | None = None,
    *,
    folder: Path | None = None,
) -> WritePermissionRules:
    """
    Load the allow/deny JSON. Missing or invalid file → empty rules (no-op).

    An explicit path wins; then ``.assistant/permissions.json`` (or the legacy
    ``.assistente/``) in the documents folder, then the project default and
    legacy files. Unknown keys (e.g. the documentation-only ``_note``) are
    silently dropped.
    """
    if path is not None:
        candidates: tuple[Path | None, ...] = (path,)
    else:
        of_folder: tuple[Path | None, ...] = (
            (
                folder / CONFIG_DIR_NAME / "permissions.json",
                folder / LEGACY_CONFIG_DIR_NAME / "permissions.json",
            )
            if folder is not None
            else ()
        )
        candidates = (*of_folder, DEFAULT_PERMISSIONS_PATH, LEGACY_PERMISSIONS_PATH)
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


def _normalize_path(relative_path: str) -> str:
    """
    Clean up the path without touching the names.

    The old ``lstrip("./")`` ate the leading dot of dotfiles (``.env``
    became ``env``) and exempted them from deny: no character stripping,
    only surrounding spaces and slashes.
    """
    return str(relative_path).strip().strip("/")


def _denied_by_deny(path: str, patterns: tuple[str, ...]) -> str | None:
    """First deny pattern matching the path (or its root, for ``**/`` patterns)."""
    for pattern in patterns:
        if _glob_match(path, pattern) or _match_root(path, pattern):
            return pattern
    return None


def _denied_with_ancestors(path: str, patterns: tuple[str, ...]) -> str | None:
    """Deny on the path or any ancestor folder: denying the folder denies its content."""
    pattern = _denied_by_deny(path, patterns)
    if pattern is not None:
        return pattern
    parts = path.split("/")
    for index in range(len(parts) - 1):
        pattern = _denied_by_deny("/".join(parts[: index + 1]), patterns)
        if pattern is not None:
            return pattern
    return None


def _match_root(path: str, pattern: str) -> bool:
    """``**/x`` also covers the workspace root, as in conventional globs."""
    if not pattern.startswith("**/"):
        return False
    return _glob_match(path, pattern[3:])


def _glob_match(path: str, pattern: str) -> bool:
    """Case-sensitive match, then casefold for paths with mixed casing."""
    return fnmatch(path, pattern) or fnmatch(path.casefold(), pattern.casefold())


def _as_pattern_tuple(value: object) -> tuple[str, ...]:
    """Accept only lists/tuples of non-empty strings."""
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
