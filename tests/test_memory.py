"""Unit tests: conversation memory and the history block in the ReAct prompt."""

from __future__ import annotations

from assistant.memory import ConversationMemory, history_prompt_block
from assistant.prompt_budget import estimate_tokens
from assistant.turn_policy import decision_from_schema
from assistant.turn_schema import TurnSchema
from assistant.session import _build_task_description

_QUESTION = decision_from_schema(TurnSchema(kind="explain"))


def test_memory_records_and_formats() -> None:
    """After one turn, format shows Utente and Assistente."""
    memory = ConversationMemory()
    memory.record("domanda uno", "risposta uno")
    text = memory.format()
    assert "Utente: domanda uno" in text
    assert "Assistente: risposta uno" in text
    assert memory.has_turns() is True


def test_memory_ignores_empty_turns() -> None:
    """Incomplete records do not enter the history."""
    memory = ConversationMemory()
    memory.record("  ", "ok")
    memory.record("ok", "")
    assert memory.has_turns() is False


def test_memory_truncates_long_answer() -> None:
    """Answers over the limit keep the tail and report the cut."""
    memory = ConversationMemory(max_answer_chars=90)
    memory.record("q", "inizio\n" + ("x" * 400) + "\nfine esito")
    text = memory.format()
    assert "contenuto ridotto" in text
    assert "fine esito" in text


def test_memory_compact_includes_outcome_when_budget_suffices() -> None:
    """Omitted turns stay in the compact block, with an outcome line."""
    memory = ConversationMemory(max_turns=6, max_chars=50_000)
    memory.record("prima domanda", "elenco\n- alfa\n- beta")
    memory.record("seconda " + ("y" * 2000), "fatto\npath: note.md")
    text = memory.format(max_tokens=220)
    assert "Compact storico" in text
    assert "Esiti:" in text
    assert "elenco" in text
    assert estimate_tokens(text) <= 220


def test_memory_absolute_numbering_after_reduction() -> None:
    """Turns keep their session index, they are not renumbered from 1."""
    memory = ConversationMemory(max_turns=2)
    memory.record("u1", "a1")
    memory.record("u2", "a2")
    memory.record("u3", "a3")
    text = memory.format()
    assert "Turno 2" in text
    assert "Turno 3" in text
    assert "Turno 1" not in text


def test_memory_turn_limit() -> None:
    """Only the last N turns stay in the buffer."""
    memory = ConversationMemory(max_turns=2)
    memory.record("u1", "a1")
    memory.record("u2", "a2")
    memory.record("u3", "a3")
    text = memory.format()
    assert "u1" not in text
    assert "u2" in text
    assert "u3" in text


def test_memory_char_limit_reduces_without_repeated_format() -> None:
    """
    With a low budget, _reduce drops old turns while respecting max_chars.

    Also checks that the formatted length stays ≤ budget after many records.
    """
    memory = ConversationMemory(max_turns=6, max_chars=400, max_answer_chars=200)
    # Many long turns: it must drop the oldest without exceeding the budget.
    for index in range(12):
        memory.record(f"domanda-{index}", "x" * 180)
    assert len(memory.turns) < 12
    assert len(memory.format()) <= 400


def test_memory_clear() -> None:
    """clear removes the whole history."""
    memory = ConversationMemory()
    memory.record("q", "a")
    memory.clear()
    assert memory.has_turns() is False
    assert memory.format() == ""


def test_task_description_includes_history() -> None:
    """The ReAct prompt includes the history, but the facts stay in the documents."""
    task = _build_task_description(
        "approfondisci il punto 1",
        full_verification=False,
        decision=_QUESTION,
        history="Utente: elenca ricerche\nAssistente: 1. Miller 2. Unit-A",
    )
    assert "STORICO CONVERSAZIONE" in task
    assert "Miller" in task
    assert "approfondisci il punto 1" in task
    assert "verificati nei documenti" in task
    assert "cerca_web" not in task
    assert "esplicita i passaggi essenziali" in task
    assert task.index("Workflow:") < task.index("STORICO CONVERSAZIONE")
    assert task.index("STORICO CONVERSAZIONE") < task.index("RICHIESTA:")


def test_listing_task_description_cites_folder() -> None:
    """The listing ReAct task enforces the Action Input of the named folder."""
    task = _build_task_description(
        "Elenca i file nella cartella assistente senza modificarli.",
        full_verification=False,
        decision=_QUESTION,
    )
    assert "Action Input obbligatorio: assistente" in task
    assert "cerca_web" not in task
    assert "elenca i nomi osservati" in task


def test_history_prompt_block_keeps_do_not_deny() -> None:
    """Direct/ReAct keep the classic prefix with «non negarlo»."""
    block = history_prompt_block("Utente: ciao\nAssistente: ciao")
    assert "non negarlo" in block
    assert history_prompt_block("") == ""
