"""Persistent project memory (AGENTS.md / .assistant/MEMORY.md)."""

from __future__ import annotations

from pathlib import Path

from assistant.config import WORKSPACE
from assistant.prompt_budget import fit_text

_PROJECT_MEMORY_CANDIDATES = (
    "AGENTS.md",
    ".assistant/MEMORY.md",
    ".assistente/MEMORY.md",
    "CLAUDE.md",
)
_MAX_PROJECT_MEMORY_TOKENS = 512


def load_project_memory(workspace: Path | None = None) -> str:
    """First existing candidate file, or an empty string if missing/unreadable."""
    root = (workspace or WORKSPACE).resolve()
    for relative in _PROJECT_MEMORY_CANDIDATES:
        candidate = root / relative
        if not candidate.is_file():
            continue
        try:
            return candidate.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            continue
    return ""


def format_project_memory_block(
    workspace: Path | None = None,
    *,
    max_tokens: int = _MAX_PROJECT_MEMORY_TOKENS,
) -> str:
    """Labelled prompt block, or empty if there is no project memory."""
    raw = load_project_memory(workspace)
    if not raw:
        return ""
    return (
        "MEMORIA PROGETTO (convenzioni e vincoli persistenti; "
        "rispettali; non modificarli se non richiesto esplicitamente):\n"
        f"{fit_text(raw, max_tokens)}\n"
    )
