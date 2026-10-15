"""Stress: routing from the schema, ReAct parser, evaluate_turn, file sandbox."""

from __future__ import annotations

import pytest
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException

from assistant.config import FILE_SEPARATOR
from assistant.evaluate_turn import evaluate_turn
from assistant.file_service import WorkspaceFileService
from assistant.turn_policy import decision_from_schema, open_react_decision, greeting_decision
from assistant.turn_schema import TurnSchema
from assistant.react_parser import AssistantReActParser
from assistant.tools import reset_turn_state, write_file_from_text
from tests.conftest import DIR_UNIT_REL

pytestmark = pytest.mark.unit


def test_stress_routing_from_schema() -> None:
    """The wording does not pick the path: the schema does, and a missing schema opens ReAct."""
    assert greeting_decision().path == "direct"
    assert greeting_decision().write_allowed is False
    open_decision = open_react_decision()
    assert open_decision.path == "react"
    assert open_decision.write_allowed is True
    explanation = decision_from_schema(TurnSchema(kind="explain", subject="liste"))
    assert explanation.path == "react"
    assert explanation.write_allowed is False
    chat = decision_from_schema(TurnSchema(kind="chat"))
    assert chat.path == "direct"
    game = decision_from_schema(
        TurnSchema(kind="workspace", folder="gioco", write=True),
    )
    assert game.path == "react"
    assert game.write_allowed is True


_PARSER_CASES: tuple[tuple[str, str, str | None], ...] = (
    (
        "Thought: ok\nFinal Answer: 42",
        "finish",
        None,
    ),
    (
        "Thought: salvo\nAction: scrivi_file\nAction Input: a.txt\n---\nbody\n"
        "Final Answer: fatto",
        "action",
        "scrivi_file",
    ),
    (
        "Thought: elenco\nAction: elenca_cartella\nassistente",
        "action",
        "elenca_cartella",
    ),
    (
        "Thought: fine\nAction: Nessuna\nAction Input: n/a",
        "finish",
        None,
    ),
    (
        '{"tool":"elenca_cartella","input":"assistente"}',
        "finish_or_action",
        None,
    ),
    (
        "final answer: ciao in minuscolo",
        "finish",
        None,
    ),
    (
        "Thought: cerco\nAction: cerca_testo\nbalenottera azzurra",
        "action",
        "cerca_testo",
    ),
    (
        "```\nThought: ok\nFinal Answer: hi\n```",
        "finish",
        None,
    ),
    (
        "Thought: solo ragiono senza chiudere",
        "finish",
        None,
    ),
    (
        "<tool>elenca_cartella</tool><input>assistente</input>",
        "finish_or_action",
        None,
    ),
)


def test_stress_react_parser_does_not_explode() -> None:
    """Malformed LLM output must not crash the turn without recovery."""
    parser = AssistantReActParser()
    failures: list[str] = []
    for text, kind, tool in _PARSER_CASES:
        try:
            outcome = parser.parse(text)
        except OutputParserException as error:
            failures.append(f"raise {type(error).__name__} on {text!r:.80}")
            continue
        if kind == "finish" and not isinstance(outcome, AgentFinish):
            failures.append(f"want finish got {type(outcome).__name__} {text!r:.60}")
        elif kind == "action":
            if not isinstance(outcome, AgentAction) or (tool and outcome.tool != tool):
                failures.append(
                    f"want action {tool} got {type(outcome).__name__} "
                    f"{getattr(outcome, 'tool', None)} {text!r:.60}"
                )
        elif kind == "finish_or_action" and not isinstance(outcome, (AgentAction, AgentFinish)):
            failures.append(f"unparsed {type(outcome).__name__} {text!r:.60}")
    assert not failures, "parser failures:\n" + "\n".join(failures)


def test_stress_evaluate_turn_false_positives() -> None:
    """A didactic «Errore:» must not mark the turn incomplete when no write was requested."""
    tutorial = (
        "In HTTP, Errore: 404 significa risorsa assente. "
        "iteration limit non c'entra con questo esempio."
    )
    outcome = evaluate_turn(
        tutorial,
        write_requested=False,
        written_paths=[],
    )
    assert outcome.incomplete is False
    assert outcome.note == ()

    empty = evaluate_turn("", write_requested=True, written_paths=[])
    assert empty.incomplete is True
    assert empty.note


def test_enrich_answer_ignores_folders() -> None:
    """A created folder does not become «File scritti su disco»."""
    from assistant.session import _enrich_answer

    idle = "Agent stopped due to iteration limit or time limit."
    assert _enrich_answer(idle, []) == idle
    enriched = _enrich_answer(idle, ["note.md"])
    assert "File scritti su disco: note.md" in enriched


def test_stress_write_file_sandbox_and_contract(workspace_test_dir) -> None:
    """Path traversal and empty body fail; unicode and a valid contract pass."""
    reset_turn_state()
    service = WorkspaceFileService()
    service.begin_turn(True)

    traversal = write_file_from_text(
        f"../etc/passwd\n{FILE_SEPARATOR}\nroot",
        service=service,
    )
    assert traversal.startswith("Errore:"), traversal

    absolute = write_file_from_text(
        f"/etc/passwd\n{FILE_SEPARATOR}\nroot",
        service=service,
    )
    assert absolute.startswith("Errore:"), absolute

    empty_body = write_file_from_text(
        f"{DIR_UNIT_REL}/vuoto.txt\n{FILE_SEPARATOR}\n",
        service=service,
    )
    assert empty_body.startswith("Errore:"), empty_body

    comma_path = write_file_from_text(
        f"{DIR_UNIT_REL}/a,b.txt\n{FILE_SEPARATOR}\nok",
        service=service,
    )
    assert comma_path.startswith("Errore:"), comma_path

    unicode_path = f"{DIR_UNIT_REL}/città.txt"
    ok = write_file_from_text(
        f"{unicode_path}\n{FILE_SEPARATOR}\nciao",
        service=service,
    )
    assert ok.startswith("File scritto con successo:"), ok
