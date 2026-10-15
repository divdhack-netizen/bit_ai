"""Process log: JSONL, crash hook, incomplete turns and /errori."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from assistant.cli import _process_turn, _handle_command
from assistant.config import ConfigRuntime, REQUEST_PREVIEW_CHARS
from assistant.console import ConsoleUI
from assistant.evaluate_turn import evaluate_turn
from assistant.process_log import (
    ProcessLog,
    classify_turn_outcome,
    request_preview,
)
from assistant.session import CallbackUI, AgentSession

pytestmark = pytest.mark.unit


def test_request_preview_truncates_beyond_limit() -> None:
    """Beyond 200 characters only the preview and the real length are persisted."""
    text = "a" * (REQUEST_PREVIEW_CHARS + 17)
    preview, length = request_preview(text)
    assert length == len(text)
    assert len(preview) == REQUEST_PREVIEW_CHARS
    assert preview == text[:REQUEST_PREVIEW_CHARS]


def test_jsonl_round_trip(tmp_path: Path) -> None:
    """session_start and a generic event stay deserializable."""
    log = ProcessLog(directory=tmp_path)
    log.start_session(pid=1, python="3.14")
    log.emit("diagnostics", message="vuoto")
    events = log.read_events()
    assert events[0]["type"] == "session_start"
    assert events[0]["session_id"] == log.session_id
    assert events[1]["type"] == "diagnostics"
    assert events[1]["message"] == "vuoto"
    assert log.index_path.exists()


def test_start_turn_truncates_preview(tmp_path: Path) -> None:
    """turn_start does not write the whole prompt."""
    log = ProcessLog(directory=tmp_path)
    text = "richiesta " + ("x" * 400)
    log.start_turn("direct", "fast", text)
    startup = next(e for e in log.read_events() if e["type"] == "turn_start")
    assert startup["request_length"] == len(text.strip())
    assert len(startup["preview"]) == REQUEST_PREVIEW_CHARS


def test_evaluate_turn_incomplete_becomes_turn_end(tmp_path: Path) -> None:
    """The incomplete flag from evaluate_turn reaches the JSONL as the outcome."""
    log = ProcessLog(directory=tmp_path)
    ui = MagicMock(spec=ConsoleUI)
    session = AgentSession(ui, llm_factory=MagicMock(), log=log)
    try:
        log.start_turn("react", "fast", "scrivi foo.py")
        answer = "Agent stopped due to iteration limit"
        outcome = evaluate_turn(
            answer,
            write_requested=True,
            written_paths=[],
            max_iterations_hit=True,
        )
        assert outcome.incomplete is True
        session._finalize_turn(
            "scrivi foo.py",
            answer,
            write_requested=True,
            written_paths=[],
            max_iterations_hit=True,
        )
        end = next(e for e in log.read_events() if e["type"] == "turn_end")
        assert end["outcome"] == "incompleto"
        assert end["note"]
    finally:
        session.close()


def test_classify_diagnostics_and_empty() -> None:
    """Diagnostic strings and empty answers are not a success."""
    assert classify_turn_outcome(incomplete=False, answer="") == "vuoto"
    assert (
        classify_turn_outcome(
            incomplete=False,
            answer="Errore inferenza locale: Timeout",
        )
        == "errore"
    )
    assert classify_turn_outcome(incomplete=False, answer="ok") == "ok"


def test_process_turn_exception_writes_traceback(tmp_path: Path) -> None:
    """An exception in the CLI loop leaves type, message and traceback on disk."""
    log = ProcessLog(directory=tmp_path)
    ui = MagicMock()
    session = MagicMock()
    session.log = log

    def boom(_request: str, _config: ConfigRuntime) -> tuple[str, list[str]]:
        log.start_turn("direct", "fast", "ciao")
        raise ValueError("rotto")

    session.run_request.side_effect = boom
    ok = _process_turn("ciao", session, ConfigRuntime(), ui)
    assert ok is False
    error = next(e for e in log.read_events() if e["type"] == "turn_error")
    assert error["exception"] == "ValueError"
    assert "rotto" in error["message"]
    assert "ValueError" in error["traceback"]
    ui.print_error.assert_called_once()


def test_crash_hook_writes_event(tmp_path: Path) -> None:
    """The persisted sys.excepthook records crash and traceback, then delegates."""
    log = ProcessLog(directory=tmp_path)
    log.install_process_hooks()
    try:
        try:
            raise RuntimeError("boom")
        except RuntimeError as err:
            sys.excepthook(type(err), err, err.__traceback__)
        crash = next(e for e in log.read_events() if e["type"] == "crash")
        assert crash["exception"] == "RuntimeError"
        assert "boom" in crash["message"]
        assert "RuntimeError" in crash["traceback"]
    finally:
        log.close()


def test_stall_once_per_turn(tmp_path: Path) -> None:
    """The watchdog emits stall only once past the threshold."""
    clock = {"t": 0.0}
    log = ProcessLog(
        directory=tmp_path,
        stall_threshold_sec=10.0,
        heartbeat_clock=lambda: clock["t"],
    )
    log.start_turn("direct", "fast", "ciao")
    assert log.check_stall(10.0) is False
    clock["t"] = 11.0
    assert log.check_stall(10.0) is True
    assert log.check_stall(10.0) is False
    stalls = [e for e in log.read_events() if e["type"] == "stall"]
    assert len(stalls) == 1


def test_errors_command_reads_last_failures(tmp_path: Path) -> None:
    """/errori prints the non-ok events of the current session."""
    log = ProcessLog(directory=tmp_path)
    log.start_session(pid=1)
    log.emit("llm_error", exception="TimeoutError", message="timeout")
    session = MagicMock()
    session.log = log
    ui = MagicMock()
    consumed = _handle_command("/errori", ConfigRuntime(), ui, session)
    assert consumed is True
    text = ui.print_system_message.call_args.args[0]
    assert "llm_error" in text
    assert "timeout" in text


def test_errors_command_without_session() -> None:
    """Without a session the command does not raise and explains the missing log."""
    ui = MagicMock()
    consumed = _handle_command("/errori 3", ConfigRuntime(), ui)
    assert consumed is True
    text = ui.print_system_message.call_args.args[0]
    assert "registro" in text.casefold()


def test_callback_llm_error_does_not_discard_exception(tmp_path: Path) -> None:
    """on_llm_error propagates type and message to the JSONL."""
    log = ProcessLog(directory=tmp_path)
    log.start_turn("direct", "fast", "ciao")
    ui = MagicMock()
    ui.has_live_stream = False
    callback = CallbackUI(ui, log=log)
    callback.on_llm_start({}, ["prompt"])
    callback.on_llm_error(RuntimeError("ollama down"))
    error = next(e for e in log.read_events() if e["type"] == "llm_error")
    assert error["exception"] == "RuntimeError"
    assert "ollama down" in error["message"]


def test_keyboard_interrupt_does_not_emit_llm_error(tmp_path: Path) -> None:
    """Ctrl+C closes the turn as cancelled without passing it off as an LLM error."""
    log = ProcessLog(directory=tmp_path)
    ui = MagicMock(spec=ConsoleUI)
    ui.has_live_stream = False
    session = MagicMock()
    session.log = log

    def boom(_request: str, _config: ConfigRuntime) -> tuple[str, list[str]]:
        log.start_turn("react", "default", "questi file non esistono")
        callback = CallbackUI(ui, log=log)
        callback.on_llm_start({}, ["prompt"])
        callback.on_llm_error(KeyboardInterrupt())
        raise KeyboardInterrupt

    session.run_request.side_effect = boom
    ok = _process_turn(
        "questi file non esistono",
        session,
        ConfigRuntime(),
        ui,
    )
    assert ok is False
    types = [e["type"] for e in log.read_events()]
    assert "llm_error" not in types
    end = next(e for e in log.read_events() if e["type"] == "turn_end")
    assert end["outcome"] == "annullato"
    index = log.format_last_errors(10)
    assert "KeyboardInterrupt" not in index
    assert "annullato" not in index.casefold()


def test_process_turn_hides_fallback_for_tool_write() -> None:
    """ReAct tool paths must not appear in the Fallback panel."""
    ui = MagicMock()
    session = MagicMock()
    session.run_request.return_value = (
        "Navbar abbellita con successo.",
        ["sito/index.html"],
    )
    session.turn_fallback_path = []
    ok = _process_turn("abbellisci la navbar", session, ConfigRuntime(), ui)
    assert ok is True
    ui.finalize.assert_called_once()
    ui.print_fallback.assert_not_called()


def test_process_turn_shows_fallback_only_create_only() -> None:
    """The Fallback panel lists only the create-only paths."""
    ui = MagicMock()
    session = MagicMock()
    session.run_request.return_value = ("ok", ["note.txt"])
    session.turn_fallback_path = ["note.txt"]
    ok = _process_turn("salva", session, ConfigRuntime(), ui)
    assert ok is True
    ui.print_fallback.assert_called_once_with(["note.txt"])
