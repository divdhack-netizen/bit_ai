"""Routing from the schema: questions and tasks go through the documents via ReAct."""

from __future__ import annotations

from assistant.turn_policy import (
    _is_local_chat,
    analysis_from_schema,
    capabilities_react,
    decision_from_schema,
    open_react_decision,
    greeting_decision,
)
from assistant.turn_schema import TurnSchema
from assistant.session import (
    _AGENT_IDENTITY,
    _comprehension_from_schema,
    _build_task_description,
)

UNEXPECTED_SENTENCE = "prepara in questa cartella un piccolo gioco"


def test_explain_reads_documents_read_only() -> None:
    """Closed corpus: a question opens the read tools, not the model weights."""
    decision = decision_from_schema(
        TurnSchema(kind="explain", subject="balenottera"),
    )
    assert decision.path == "react"
    assert decision.write_allowed is False


def test_pure_chat_stays_direct() -> None:
    """A comment on the conversation has nothing to look up in the documents."""
    decision = decision_from_schema(TurnSchema(kind="chat"))
    assert decision.path == "direct"
    assert decision.write_allowed is False


def test_unexpected_workspace_is_react_with_write() -> None:
    """The workspace kind enables the tools, even if the sentence has no known verbs."""
    decision = decision_from_schema(
        TurnSchema(kind="workspace", subject="gioco", folder="gioco"),
    )
    assert decision.path == "react"
    assert decision.write_allowed is True
    assert UNEXPECTED_SENTENCE not in decision.path_reason


def test_explain_with_path_goes_react() -> None:
    """With a real destination the soft kind opens the tools."""
    decision = decision_from_schema(
        TurnSchema(kind="explain", paths=("note.md",)),
    )
    assert decision.path == "react"


def test_write_permission_is_not_write_intent() -> None:
    """A lookup-only ReAct opens the tools but does not ask for a write."""
    from assistant.turn_validator import validate_turn

    outcome = validate_turn(
        TurnSchema(kind="workspace", folder="progetto"),
        "elenca quello che c'è nella cartella progetto",
    )
    decision = decision_from_schema(outcome.schema)
    assert decision.path == "react"
    assert decision.write_allowed is True
    assert decision.analysis.write is False


def test_react_read_does_not_authorize_write() -> None:
    """The file to read stays on ReAct, with no write tool or instruction."""
    from assistant.turn_validator import validate_turn

    request = "leggi il file rassegna.txt e dimmi di cosa parla"
    outcome = validate_turn(
        TurnSchema(
            kind="workspace",
            write=True,
            deliver="file",
            subject="contenuto",
        ),
        request,
    )
    decision = decision_from_schema(outcome.schema)
    assert decision.path == "react"
    assert decision.write_allowed is False
    task = _build_task_description(
        request,
        full_verification=False,
        decision=decision,
        comprehension=_comprehension_from_schema(outcome.schema),
    )
    assert "Usa scrivi_file" not in task
    assert "Scrittura consentita: no" in task
    assert "niente scrivi_file" in task


def test_workspace_without_query_stays_react() -> None:
    """Without a query the task stays on ReAct, even with a destination folder."""
    decision = decision_from_schema(
        TurnSchema(
            kind="workspace",
            deliver="file",
            write=True,
            folder="progetto",
        ),
    )
    assert decision.path == "react"


def test_missing_schema_opens_full_react() -> None:
    """Unreadable JSON or thermal pressure: full tools, not a tool-less generation."""
    decision = open_react_decision()
    assert decision.path == "react"
    assert decision.write_allowed is True
    assert capabilities_react() == ("file", "summary", "date")


def test_exact_greeting_stays_direct() -> None:
    """Only a bare greeting skips the schema. A question after ciao does not."""
    assert greeting_decision().path == "direct"
    assert greeting_decision().write_allowed is False
    assert _is_local_chat("ciao") is True
    assert _is_local_chat("ciao, chi è il presidente?") is False


def test_react_enforces_corpus_without_repeating_it() -> None:
    """The corpus lives in the identity (stable prefix); the task does not repeat it."""
    decision = decision_from_schema(TurnSchema(kind="workspace", folder="gioco"))
    task = _build_task_description(
        UNEXPECTED_SENTENCE,
        full_verification=False,
        decision=decision,
        comprehension=_comprehension_from_schema(
            TurnSchema(kind="workspace", subject="gioco", folder="gioco"),
        ),
    )
    assert "Unica fonte: i documenti della cartella" in _AGENT_IDENTITY
    assert "Unica fonte: i documenti della cartella" not in task
    assert "cerca_testo" in _AGENT_IDENTITY
    assert "cerca_web" not in task


def test_analysis_from_schema_ignores_words() -> None:
    """The same card produces the same signals whatever the sentence."""
    schema = TurnSchema(kind="explain", subject="termodinamica")
    before = analysis_from_schema(schema)
    second = analysis_from_schema(schema)
    assert before == second
    assert before.explanation is True


def test_history_spill_without_new_theme() -> None:
    """«salva tutto in chat.txt» writes the history; a new theme does not."""
    from assistant.turn_policy import is_history_spill, named_document_path

    assert is_history_spill("salva tutto in chat.txt") is True
    assert named_document_path("salva tutto in chat.txt") == "chat.txt"
    assert is_history_spill("salva le abitudini delle balene in a.txt") is False


def test_non_txt_destination_becomes_txt() -> None:
    """A new .md destination becomes .txt; an existing source stays as it is."""
    from assistant.session import _adapt_txt_destinations, _write_destination

    schema = TurnSchema(
        kind="workspace",
        write=True,
        deliver="file",
        paths=("report.docx", "note.md"),
    )
    adapted, note = _adapt_txt_destinations(schema, lambda path: path == "report.docx")
    assert adapted is not None
    assert adapted.paths == ("report.docx", "note.txt")
    assert "note.md → note.txt" in note
    assert _write_destination(adapted.paths) == "note.txt"
    busy, _ = _adapt_txt_destinations(
        TurnSchema(kind="workspace", write=True, paths=("note.md",)),
        lambda path: path == "note.txt",
    )
    assert busy is not None and busy.paths == ("note_2.txt",)
    reading = TurnSchema(kind="workspace", paths=("note.md",))
    assert _adapt_txt_destinations(reading, lambda _path: False) == (reading, "")
