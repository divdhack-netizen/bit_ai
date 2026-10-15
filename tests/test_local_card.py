"""Local card: clear turns do not call the LLM, ambiguous ones do."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from assistant.comprehension import local_card
from assistant.config import ConfigRuntime, FileConfig
from assistant.console import ConsoleUI
from assistant.file_service import WorkspaceFileService
from assistant.host import PowerState
from assistant.turn_policy import _is_local_chat, decision_from_schema
from assistant.session import AgentSession
from assistant.turn_validator import ValidationContext, validate_turn
from benchmarks import hard_comprehension, comprehension_probe

pytestmark = pytest.mark.unit

# Minimum share of corpus turns decided without the LLM.
_MIN_COVERAGE = 0.7
_SOURCES = {".pdf", ".docx", ".pages", ".png", ".jpg"}


def _source_exists(path: str) -> bool:
    """Fake corpus: source documents exist, the .txt files are new destinations."""
    return Path(path).suffix.casefold() in _SOURCES


def _disagreements(case: object) -> list[str]:
    """Benchmark expectations broken by the local card, after the validator."""
    schema = local_card(case.text, _source_exists)
    assert schema is not None
    validated = validate_turn(
        schema,
        case.text,
        ValidationContext(
            last=getattr(case, "last", None),
            previous_document=getattr(case, "previous_document", None),
        ),
    ).schema
    route = decision_from_schema(validated).path
    write = validated.write or validated.deliver == "file"
    errors: list[str] = []
    if case.expected_kinds is not None and validated.kind not in case.expected_kinds:
        errors.append(f"kind={validated.kind}")
    wait = case.expected_route
    if wait is not None:
        ok = route in wait if isinstance(wait, frozenset) else route == wait
        if not ok:
            errors.append(f"route={route}")
    if case.expected_write is not None and write != case.expected_write:
        errors.append(f"write={write}")
    if case.expected_path is not None and not any(
        path.casefold().endswith(case.expected_path.casefold()) for path in validated.paths
    ):
        errors.append(f"paths={validated.paths}")
    if case.expected_folder is not None and (
        case.expected_folder.casefold() not in validated.folder.casefold()
    ):
        errors.append(f"folder={validated.folder!r}")
    expected_clarify = getattr(case, "expected_clarify", None)
    if expected_clarify is not None and bool(validated.clarify) != expected_clarify:
        errors.append(f"clarify={validated.clarify!r}")
    return errors


def test_local_card_agrees_with_benchmark_corpora() -> None:
    """When it decides on its own, the local card meets every benchmark expectation."""
    cases = [
        case
        for corpus in (hard_comprehension.CORPUS, comprehension_probe.CORPUS)
        for case in corpus
        if not getattr(case, "not_evaluated", False) and not _is_local_chat(case.text)
    ]
    decided = [case for case in cases if local_card(case.text, _source_exists)]
    errors = {case.text: _disagreements(case) for case in decided}
    assert {text: e for text, e in errors.items() if e} == {}
    assert len(decided) / len(cases) >= _MIN_COVERAGE


@pytest.mark.parametrize(
    ("text", "kind", "write", "paths"),
    [
        ("quanto pesa una balenottera azzurra?", "explain", False, ()),
        ("leggi avvistamenti.pdf e dimmi di cosa parla", "workspace", False, ("avvistamenti.pdf",)),
        (
            "riassumi report.docx e salva il riassunto in sintesi.txt",
            "workspace",
            True,
            ("report.docx", "sintesi.txt"),
        ),
        ("Scrivi il file note.md con il testo: prova", "workspace", True, ("note.md",)),
        ("dimmi cosa mangiano le orche, niente file, solo in chat", "explain", False, ()),
    ],
)
def test_local_card_decides_clear_cases(
    text: str,
    kind: str,
    write: bool,
    paths: tuple[str, ...],
) -> None:
    """Questions, reads and a single new destination ask nothing of the model."""
    schema = local_card(text, _source_exists)
    assert schema is not None
    assert schema.kind == kind
    assert schema.write is write
    assert schema.paths == paths


@pytest.mark.parametrize(
    "text",
    [
        "aggiorna note.txt con la data di oggi",
        "salvalo",
        "salvalo in un documento",
        "e il secondo?",
        "appuntami che domani piove",
        "archiviami questo ragionamento in un txt",
        "riassumi a.pdf in uno.txt e in due.txt e salvali",
        "e poi?",
    ],
)
def test_local_card_leaves_ambiguous_cases_to_model(text: str) -> None:
    """Edits, deixis and writes without a certain destination go to the LLM."""
    assert local_card(text, _source_exists) is None


def test_unsupported_action_becomes_clarification_without_llm() -> None:
    """An email opens no tools: the local card hands it to the validator."""
    text = "mandami una mail con il riassunto"
    schema = local_card(text, _source_exists)
    assert schema is not None
    validated = validate_turn(schema, text).schema
    assert validated.clarify
    assert validated.write is False


@pytest.fixture
def session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AgentSession:
    """Session on an isolated folder, with a fake LLM and mains power."""
    monkeypatch.setattr(
        "assistant.energy_policy.detect_power",
        lambda: PowerState(on_battery=False, low_power_mode=False),
    )
    service = WorkspaceFileService(FileConfig(workspace=tmp_path))
    agent = AgentSession(
        MagicMock(spec=ConsoleUI),
        file_service=service,
        llm_factory=MagicMock(),
    )
    yield agent
    agent.close()


def test_clear_turn_does_not_call_llm_card(session: AgentSession) -> None:
    """Local classification does not touch Ollama."""
    session._schema_from_model = MagicMock(side_effect=AssertionError("LLM chiamato"))
    origin, validation, decision = session._classify_turn(
        "quanto pesa una balenottera azzurra?",
        "default",
        None,
    )
    assert origin == "locale"
    assert validation is not None
    assert decision.path == "react"
    assert decision.write_allowed is False


def test_ambiguous_turn_asks_model_for_card(session: AgentSession) -> None:
    """Only ambiguous text pays for the comprehension call."""
    session._schema_from_model = MagicMock(return_value=(None, False, False))
    origin, _validation, decision = session._classify_turn(
        "salvalo",
        "default",
        None,
    )
    session._schema_from_model.assert_called_once()
    assert origin == "schema_fallback"
    assert decision.path == "react"


def test_spill_does_not_classify_turn(session: AgentSession) -> None:
    """«salva tutto in chat.txt» writes the history with no card and no search."""
    session.record_history_turn("ciao", "ciao a te")
    session._classify_turn = MagicMock(side_effect=AssertionError("classificato"))
    answer, written = session.run_request(
        "riportami tutto sul file chat.txt",
        ConfigRuntime(model="default", burn_protection=False),
    )
    assert written == ["chat.txt"]
    assert "chat.txt" in answer


def test_ollama_error_does_not_repeat_card(session: AgentSession) -> None:
    """A transport error does not pay for a second attempt: ReAct opens."""
    llm = MagicMock()
    llm.invoke.side_effect = ConnectionError("Ollama giù")
    session._runtime.get_llm = MagicMock(return_value=llm)
    validation, _repaired, retry = session._schema_from_model("salvalo", "default")
    assert validation is None
    assert retry is False
    assert llm.invoke.call_count == 1


def test_unreadable_json_retries_once(session: AgentSession) -> None:
    """The retry stays for broken JSON, with the error in the prompt."""
    llm = MagicMock()
    llm.invoke.side_effect = [
        "non è json",
        '{"kind":"workspace","write":true,"deliver":"file","subject":"x",'
        '"paths":["a.txt"],"folder":"","clarify":""}',
    ]
    session._runtime.get_llm = MagicMock(return_value=llm)
    validation, repaired, retry = session._schema_from_model("salvalo", "default")
    assert validation is not None
    assert validation.schema.paths == ("a.txt",)
    assert (repaired, retry) == (True, True)
    assert "Errore JSON" in llm.invoke.call_args_list[1].args[0]
