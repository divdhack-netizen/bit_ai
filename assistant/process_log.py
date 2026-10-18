"""Persistent log of crashes, incomplete turns and process failures."""

from __future__ import annotations

import atexit
import faulthandler
import json
import os
import sys
import threading
import time
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from assistant.config import (
    DIRECTORY_LOG,
    DEFAULT_SLASH_ERRORS,
    EVENT_MESSAGE_CHARS,
    REQUEST_PREVIEW_CHARS,
    STALL_POLL_SEC,
    STALL_THRESHOLD_SEC,
    TRACEBACK_MAX_CHARS,
    INDEX_LAST_ERRORS,
)

PROBLEM_TYPES = frozenset(
    {
        "startup",
        "crash",
        "diagnostics",
        "llm_error",
        "stall",
        "tool_error",
        "turn_error",
    }
)
NON_OK_TURN_OUTCOMES = frozenset({"errore", "incompleto", "vuoto"})
_DIAGNOSTICS_PREFIX = (
    "errore inferenza locale:",
)
_FAILED_SYNTHESIS_MARKER = "sintesi del modello è fallita"


def request_preview(
    text: str,
    limit: int = REQUEST_PREVIEW_CHARS,
) -> tuple[str, int]:
    """
    Truncate the request for the log without exposing the whole prompt.

    Returns:
        Preview (at most ``limit`` characters) and the original length.
    """
    clean = text.strip()
    length = len(clean)
    if length <= limit:
        return clean, length
    return clean[:limit], length


def truncate_message(
    text: str,
    limit: int = EVENT_MESSAGE_CHARS,
) -> str:
    """Shorten an event message to stay within the log budget."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n…"


def format_traceback(
    error: BaseException,
    limit: int = TRACEBACK_MAX_CHARS,
) -> str:
    """Serialize the exception stack, truncated if it exceeds the budget."""
    text = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    return truncate_message(text, limit)


def is_diagnostic_answer(answer: str) -> bool:
    """True if the user-facing text is a failure disguised as a diagnostic."""
    low = (answer or "").casefold()
    if not low.strip():
        return False
    if any(low.startswith(prefix) for prefix in _DIAGNOSTICS_PREFIX):
        return True
    return _FAILED_SYNTHESIS_MARKER in low


def classify_turn_outcome(*, incomplete: bool, answer: str) -> str:
    """
    Derive the persisted outcome from evaluate_turn and the answer text.

    Returns:
        ``incompleto``, ``vuoto``, ``errore`` or ``ok``.
    """
    if incomplete:
        return "incompleto"
    if not (answer or "").strip():
        return "vuoto"
    if is_diagnostic_answer(answer):
        return "errore"
    return "ok"


def non_ok_event(event: dict[str, Any]) -> bool:
    """True if the event must appear in ``/errori`` and in the markdown index."""
    event_type = event.get("type")
    if event_type == "turn_end":
        visible = event.get("outcome") in NON_OK_TURN_OUTCOMES
    else:
        visible = event_type in PROBLEM_TYPES
    if event_type == "llm_error" and event.get("exception") == "KeyboardInterrupt":
        visible = False
    return visible


def notify_diagnostics(
    callback: object,
    message: str,
    exception_type: str | None = None,
) -> None:
    """Forward a diagnostic to the callback if it exposes ``record_diagnostics``."""
    method = getattr(callback, "record_diagnostics", None)
    if callable(method):
        method(message, exception_type=exception_type)


def _iso_utc() -> str:
    """Stable UTC timestamp, without microseconds, for the JSONL lines."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class EventLog:
    """
    Observability API. The base implementation does not write to disk.

    Used in tests and when the CLI does not inject a persistent log.
    """

    def __init__(
        self,
        *,
        heartbeat_clock: Callable[[], float] | None = None,
    ) -> None:
        self._session_id = uuid4().hex
        self._turn_id: str | None = None
        self._turn_path: str | None = None
        self._turn_alias: str | None = None
        self._lock = threading.RLock()
        self._heartbeat_clock = heartbeat_clock or time.monotonic
        self._last_heartbeat = 0.0
        self._stall_emitted = False
        self._session_closed = False

    @property
    def session_id(self) -> str:
        """Identifier of the current CLI session."""
        return self._session_id

    @property
    def turn_id(self) -> str | None:
        """The active turn, or None outside an execution."""
        return self._turn_id

    def emit(self, event_type: str, **fields: Any) -> dict[str, Any]:
        """Build the event and hand it to the persistence layer, if any."""
        event = self._build_event(event_type, fields)
        self._persist(event)
        return event

    def _persist(self, event: dict[str, Any]) -> None:
        """Persistence hook: no I/O in the in-memory variant."""
        del event

    def _build_event(
        self,
        event_type: str,
        fields: dict[str, Any],
    ) -> dict[str, Any]:
        """Attach timestamp and session/turn correlation to the free-form fields."""
        event: dict[str, Any] = {
            "ts": _iso_utc(),
            "type": event_type,
            "session_id": self._session_id,
            "turn_id": self._turn_id,
        }
        event.update(fields)
        return event

    def start_session(self, **fields: Any) -> None:
        """Mark the start of the CLI process."""
        self._session_closed = False
        self.emit("session_start", **fields)

    def end_session(self, **fields: Any) -> None:
        """Mark the orderly shutdown; idempotent if already emitted."""
        if self._session_closed:
            return
        self._session_closed = True
        self._close_active_turn()
        self.emit("session_end", **fields)

    def start_turn(self, path: str, alias: str, request: str) -> str:
        """Open a turn and start the stall clock."""
        preview, length = request_preview(request)
        with self._lock:
            self._turn_id = uuid4().hex
            self._turn_path = path
            self._turn_alias = alias
            self._stall_emitted = False
            self._last_heartbeat = self._heartbeat_clock()
            turn_id = self._turn_id
        self.emit(
            "turn_start",
            path=path,
            alias=alias,
            preview=preview,
            request_length=length,
        )
        return turn_id

    def end_turn(
        self,
        outcome: str,
        note: tuple[str, ...] | list[str] = (),
        metrics: dict[str, float | int] | None = None,
    ) -> None:
        """Close the turn with its outcome and an optional metrics snapshot."""
        self.emit(
            "turn_end",
            outcome=outcome,
            note=list(note),
            metrics=metrics,
            path=self._turn_path,
            alias=self._turn_alias,
        )
        self._close_active_turn()

    def record_turn_error(self, error: BaseException) -> None:
        """Exception not handled by the pipeline, with traceback."""
        self.emit(
            "turn_error",
            exception=type(error).__name__,
            message=truncate_message(str(error)),
            traceback=format_traceback(error),
        )
        self._close_active_turn()

    def record_llm_error(self, error: BaseException) -> None:
        """Provider error during a generation."""
        self.heartbeat()
        self.emit(
            "llm_error",
            exception=type(error).__name__,
            message=truncate_message(str(error)),
            traceback=format_traceback(error),
        )

    def record_tool_error(self, name: str, error: BaseException) -> None:
        """ReAct or direct tool that raised an exception."""
        self.heartbeat()
        self.emit(
            "tool_error",
            tool=name,
            exception=type(error).__name__,
            message=truncate_message(str(error)),
            traceback=format_traceback(error),
        )

    def record_diagnostics(
        self,
        message: str,
        exception_type: str | None = None,
    ) -> None:
        """Failure turned into user-facing text (empty LLM, synthesis)."""
        self.emit(
            "diagnostics",
            message=truncate_message(message),
            exception=exception_type,
        )

    def record_startup(self, phase: str, error: BaseException) -> None:
        """Preload, download or wizard failure."""
        self.emit(
            "startup",
            phase=phase,
            exception=type(error).__name__,
            message=truncate_message(str(error)),
            traceback=format_traceback(error),
        )

    def record_crash(
        self,
        error: BaseException,
        *,
        thread: str | None = None,
    ) -> None:
        """Uncaught exception at process or thread level."""
        fields: dict[str, Any] = {
            "exception": type(error).__name__,
            "message": truncate_message(str(error)),
            "traceback": format_traceback(error),
        }
        if thread:
            fields["thread"] = thread
        self.emit("crash", **fields)

    def record_thermal(
        self,
        state: str,
        *,
        num_predict: int,
        max_iterations: int,
        alias: str,
    ) -> None:
        """Throttle that cuts output or iterations; explains truncated turns."""
        self.emit(
            "thermal",
            state=state,
            num_predict=num_predict,
            max_iterations=max_iterations,
            alias=alias,
        )

    def heartbeat(self) -> None:
        """Report activity (token or tool) to the stall watchdog."""
        with self._lock:
            self._last_heartbeat = self._heartbeat_clock()

    def check_stall(self, threshold_sec: float = STALL_THRESHOLD_SEC) -> bool:
        """
        Emit ``stall`` only once if the turn is silent past the threshold.

        Returns:
            True if a new stall event was emitted.
        """
        with self._lock:
            if self._turn_id is None or self._stall_emitted:
                return False
            silence = self._heartbeat_clock() - self._last_heartbeat
            if silence < threshold_sec:
                return False
            self._stall_emitted = True
        self.emit("stall", silence_sec=round(silence, 1), threshold_sec=threshold_sec)
        return True

    def last_errors(self, limit: int = DEFAULT_SLASH_ERRORS) -> list[dict[str, Any]]:
        """Non-ok events of the session; empty if there is no persistence."""
        del limit
        return []

    def format_last_errors(
        self,
        limit: int = DEFAULT_SLASH_ERRORS,
    ) -> str:
        """Compact text for ``/errori``."""
        errors = self.last_errors(limit)
        if not errors:
            return "Nessun errore registrato in questa sessione."
        lines = [f"Ultimi {len(errors)} eventi non-ok:"]
        for event in errors:
            lines.append(f"- {_error_line(event)}")
        return "\n".join(lines)

    def start_stall_check(self) -> None:
        """No-op: only the persistent log starts the thread."""
        return

    def install_process_hooks(self) -> None:
        """No-op: the crash hooks live on the persistent log."""
        return

    def close(self) -> None:
        """No-op for the in-memory variant."""
        return

    def _close_active_turn(self) -> None:
        """Reset the correlation and the watchdog of the current turn."""
        with self._lock:
            self._turn_id = None
            self._turn_path = None
            self._turn_alias = None
            self._stall_emitted = False


class ProcessLog(EventLog):
    """Append-only JSONL under ``output/logs/``, with crash dumps and a markdown index."""

    def __init__(
        self,
        directory: Path | None = None,
        *,
        stall_threshold_sec: float = STALL_THRESHOLD_SEC,
        stall_poll_sec: float = STALL_POLL_SEC,
        heartbeat_clock: Callable[[], float] | None = None,
    ) -> None:
        super().__init__(heartbeat_clock=heartbeat_clock)
        self._directory = directory if directory is not None else DIRECTORY_LOG
        self._directory.mkdir(parents=True, exist_ok=True)
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._path_jsonl = self._directory / f"{day}.jsonl"
        self._index_path = self._directory / "ultimi_errori.md"
        self._path_crash = self._directory / f"crash-{os.getpid()}.txt"
        self._stall_threshold_sec = stall_threshold_sec
        self._stall_poll_sec = stall_poll_sec
        self._crash_file: Any | None = None
        self._original_excepthook = sys.excepthook
        self._original_thread_hook = threading.excepthook
        self._hooks_installed = False
        self._atexit_registered = False
        self._closed = False
        self._stop_stall = threading.Event()
        self._thread_stall: threading.Thread | None = None

    @property
    def path_jsonl(self) -> Path:
        """JSONL file for the day the log was opened."""
        return self._path_jsonl

    @property
    def index_path(self) -> Path:
        """Markdown index of the last errors, rewritten on every non-ok event."""
        return self._index_path

    def _persist(self, event: dict[str, Any]) -> None:
        """Append a JSON line and, if non-ok, refresh the markdown index."""
        line = json.dumps(event, ensure_ascii=False, default=str)
        with self._lock:
            with self._path_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        if non_ok_event(event):
            self._update_index()

    def read_events(self) -> list[dict[str, Any]]:
        """Load every JSONL line of the session file, skipping corrupt ones."""
        if not self._path_jsonl.exists():
            return []
        events: list[dict[str, Any]] = []
        text = self._path_jsonl.read_text(encoding="utf-8")
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events

    def last_errors(self, limit: int = DEFAULT_SLASH_ERRORS) -> list[dict[str, Any]]:
        """Last non-ok events of the current session, in chronological order."""
        own = [
            event
            for event in self.read_events()
            if event.get("session_id") == self._session_id and non_ok_event(event)
        ]
        if limit <= 0:
            return []
        return own[-limit:]

    def start_stall_check(self) -> None:
        """Start the daemon thread that watches turns without a heartbeat."""
        if self._thread_stall is not None:
            return
        self._stop_stall.clear()
        thread = threading.Thread(
            target=self._loop_stall,
            name="assistant-log-stall",
            daemon=True,
        )
        self._thread_stall = thread
        thread.start()

    def stop_stall_check(self) -> None:
        """Stop the watchdog; waits at most one poll cycle."""
        self._stop_stall.set()
        thread = self._thread_stall
        if thread is not None and thread.is_alive():
            thread.join(timeout=self._stall_poll_sec + 1.0)
        self._thread_stall = None

    def install_process_hooks(self) -> None:
        """Enable faulthandler, excepthook and atexit only once."""
        if self._hooks_installed:
            return
        self._directory.mkdir(parents=True, exist_ok=True)
        crash_file = self._path_crash.open("a", encoding="utf-8")
        self._crash_file = crash_file
        faulthandler.enable(file=crash_file, all_threads=True)
        sys.excepthook = self._excepthook
        threading.excepthook = self._thread_excepthook
        if not self._atexit_registered:
            atexit.register(self._atexit_close)
            self._atexit_registered = True
        self._hooks_installed = True

    def close(self) -> None:
        """Stop watchdog and hooks; idempotent."""
        if self._closed:
            return
        self._closed = True
        self.stop_stall_check()
        self._restore_hooks()
        crash_file = self._crash_file
        self._crash_file = None
        if crash_file is not None:
            faulthandler.disable()
            crash_file.close()

    def _loop_stall(self) -> None:
        """Periodic poll: it does not kill the turn, it only records."""
        while not self._stop_stall.wait(self._stall_poll_sec):
            self.check_stall(self._stall_threshold_sec)

    def _update_index(self) -> None:
        """Rewrite ``ultimi_errori.md`` with the file's last non-ok events."""
        errors = [
            event
            for event in self.read_events()
            if non_ok_event(event)
        ][-INDEX_LAST_ERRORS:]
        lines = ["# Ultimi errori", ""]
        if not errors:
            lines.append("Nessun errore registrato.")
        else:
            for event in errors:
                lines.append(f"- {_error_line(event)}")
        self._index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _excepthook(
        self,
        event_type: type[BaseException],
        value: BaseException,
        trace: Any,
    ) -> None:
        """Record the crash and delegate to the previous hook (stderr/pytest)."""
        self.record_crash(value)
        original = self._original_excepthook
        if original is not None and original is not self._excepthook:
            original(event_type, value, trace)

    def _thread_excepthook(self, args: threading.ExceptHookArgs) -> None:
        """Same treatment for exceptions in non-main threads."""
        if args.exc_value is not None:
            name = args.thread.name if args.thread is not None else None
            self.record_crash(args.exc_value, thread=name)
        original = self._original_thread_hook
        if original is not None and original is not self._thread_excepthook:
            original(args)

    def _restore_hooks(self) -> None:
        """Restore the system excepthooks if this log owned them."""
        if not self._hooks_installed:
            return
        if sys.excepthook is self._excepthook:
            sys.excepthook = self._original_excepthook
        if threading.excepthook is self._thread_excepthook:
            threading.excepthook = self._original_thread_hook
        self._hooks_installed = False

    def _atexit_close(self) -> None:
        """If main did not close, emit session_end and release the files."""
        try:
            if self._closed:
                return
            self.end_session(reason="atexit")
            self.close()
        except Exception:
            return


def _error_line(event: dict[str, Any]) -> str:
    """One readable line for the TTY and the markdown index."""
    ts = str(event.get("ts", ""))
    event_type = str(event.get("type", "event"))
    outcome = event.get("outcome")
    path = event.get("path")
    alias = event.get("alias")
    message = event.get("message") or event.get("exception") or ""
    if event_type == "turn_end" and outcome:
        event_type = f"turn_end/{outcome}"
    context = ""
    if path or alias:
        context = f" [{path or '-'}/{alias or '-'}]"
    note = event.get("note") or []
    if note and not message:
        message = "; ".join(str(item) for item in note)
    suffix = f" {message}" if message else ""
    return f"{ts} **{event_type}**{context}{suffix}".strip()
