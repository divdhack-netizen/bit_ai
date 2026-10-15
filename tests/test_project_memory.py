"""Tests for project memory and structured history compaction."""

from __future__ import annotations

from pathlib import Path

import pytest

from assistant.memory import ConversationMemory
from assistant.project_memory import format_project_memory_block, load_project_memory
from assistant.prompt_budget import estimate_tokens

pytestmark = pytest.mark.unit


def test_load_project_memory_agents_md(tmp_path: Path) -> None:
    """AGENTS.md in the workspace is loaded as project memory."""
    (tmp_path / "AGENTS.md").write_text("# Rules\nUse pytest.\n", encoding="utf-8")
    text = load_project_memory(tmp_path)
    assert "Use pytest" in text
    block = format_project_memory_block(tmp_path)
    assert "MEMORIA PROGETTO" in block
    assert "Use pytest" in block


def test_load_project_memory_missing(tmp_path: Path) -> None:
    """Workspace without the file → empty string."""
    assert load_project_memory(tmp_path) == ""
    assert format_project_memory_block(tmp_path) == ""


def test_memory_structured_compaction_when_budget_tight() -> None:
    """
    With a low budget and many turns, the old ones end up in the history Compact.

    Recent user messages stay verbatim in the non-compacted turns.
    """
    memory = ConversationMemory(max_turns=6, max_chars=50_000)
    # Turns long enough that they cannot all fit in a tight budget.
    for index in range(5):
        memory.record(
            f"richiesta utente numero {index} con vincolo non-touch-legacy",
            f"Assistente ha letto file_{index}.py e risposto con dettagli " + ("x" * 200),
        )
    # A budget that forces some whole turns to be omitted.
    formatted = memory.format(max_tokens=180)
    assert estimate_tokens(formatted) <= 200
    # Structured Compact present, or at least the most recent turns.
    assert "Compact storico" in formatted or "Turno" in formatted
    # The last user message stays in the text (verbatim).
    assert "richiesta utente numero 4" in formatted
    assert "Compact storico" in formatted
