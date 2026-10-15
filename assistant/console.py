"""Console UI: timeline, ReAct stream, thermal bar and Rich panels."""

from __future__ import annotations

import os
import re
import select
import shutil
import signal
import sys
import threading
import time
from collections.abc import Callable
from enum import Enum, auto
from typing import TextIO

from rich.console import Console, Group
from rich.constrain import Constrain
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.rule import Rule
from rich.status import Status
from rich.text import Text
from rich.theme import Theme

from assistant.host import detect_power
from assistant.metrics import LLMTurnStats
from assistant.line_editor import LineEditor, KEYBOARD_GUIDE_LINES, text_with_selection
from assistant.prompt_tty import read_line_with_reflow, termios_available
from assistant.splash import run_splash
from assistant.thermal import (
    invalidate_thermal_cache,
    read_thermal_state,
    read_cpu_temperature_celsius,
    status_bar_line,
)
from assistant.tty_layout import (
    TTY_WIDTH_MARGIN,
    sticky_composer_height,
    clear_lines_above,
    cuu_clear_composer,
    session_header_markup,
    position_cursor_absolute,
    position_transcript_start,
    clear_absolute_lines,
    sticky_scroll_end_line,
    sticky_composer_start_line,
    visual_content_lines,
    panel_content_wrap_segments,
    scroll_region_sequence,
    sync_dimensions,
    print_composer,
    print_sticky_composer,
    print_sticky_header,
    print_user_message,
    print_panel,
    composer_line,
    truncate_visible,
    tty_size,
    read_ioctl_tty_size,
    reading_width,
)

MAX_TOOL_INPUT_LENGTH = 200
MAX_VERBOSE_OBSERVATION_LENGTH = 800
_THOUGHT_STYLE = "grey50 dim"
_THINKING_LABEL = "Pensiero "
_TOOL_SENTENCES = {
    "scrivi_file": "Scrivo",
    "modifica_file": "Modifico",
    "crea_cartella": "Creo la cartella",
    "elenca_cartella": "Elenco",
    "elenca_albero": "Elenco l'albero",
    "trova_file": "Cerco i file",
    "cerca_testo": "Cerco nei documenti",
    "leggi_documento": "Leggo",
    "riassumi_documento": "Riassumo",
    "data_oggi": "Controllo la data",
}
_LIVE_PANEL_REFRESH_PER_SECOND = 20
_LIVE_PANEL_REFRESH_INTERVAL_SEC = 1 / _LIVE_PANEL_REFRESH_PER_SECOND
# The thermal bar must not redraw on every chunk: keeps UI overhead under ~15%.
_STREAM_BAR_REFRESH_PER_SECOND = 4
_STREAM_BAR_REFRESH_INTERVAL_SEC = 1 / _STREAM_BAR_REFRESH_PER_SECOND
_THERMAL_POLLING_SEC = 2.0
_RESIZE_POLL_INTERVAL_SEC = 0.05
_START_MARKERS = ("Thought:", "Action:", "Action Input:", "Final Answer:")
_THOUGHT_END_MARKERS = ("\nAction:", "\nFinal Answer:", "\nObservation:")
_INPUT_END_MARKERS = ("\nObservation:",)
_ANSI = {
    "save": "\0337", "restore": "\0338", "reset_region": "\033[r",
    "clear_line": "\033[2K",
}

_THOUGHT_START_PATTERN = re.compile(r"Thought:\s*", re.IGNORECASE)
_ACTION_START_PATTERN = re.compile(r"(?:^|\n)Action:\s*", re.IGNORECASE)
_ACTION_INPUT_START_PATTERN = re.compile(r"(?:^|\n)Action Input:\s*", re.IGNORECASE)
_FINAL_ANSWER_START_PATTERN = re.compile(r"Final Answer:\s*", re.IGNORECASE)
_THOUGHT_END_PATTERN = re.compile(r"\n(?:Action|Final\s+Answer|Observation)\s*:", re.IGNORECASE)
_ACTION_INPUT_END_PATTERN = re.compile(r"\nObservation:\s*", re.IGNORECASE)
_EXTRACT_FINAL_ANSWER_PATTERN = re.compile(r"Final Answer:\s*(.*)", re.DOTALL | re.IGNORECASE)

_PREFIX_CACHE: dict[tuple[str, ...], tuple[str, ...]] = {}
_MAX_MARKER_LENGTH = max(
    len(m) for ms in (_START_MARKERS, _THOUGHT_END_MARKERS, _INPUT_END_MARKERS) for m in ms
)

_THEME = Theme({
    "user.label": "bold cyan", "agent.label": "bold green",
    "tool.name": "bold yellow", "tool.input": "dim", "tool.result": "green",
    "system": "dim", "error": "bold red", "warning": "bold yellow",
    "badge.on": "bold green", "badge.off": "dim",
    "panel.border": "cyan",
})
_STATE_KEY_FROM_LABEL = {
    "Verbose": "verbose",
    "Verifica completa": "full_verification",
    "Burn protection": "burn_protection",
    "Thinking": "thinking",
}


def _default_runtime_state() -> dict[str, bool]:
    """Session flags used by welcome and /help if the banner has not been printed yet."""
    return {
        "verbose": False,
        "full_verification": False,
        "burn_protection": True,
        "thinking": False,
    }


def _markup_badge(label: str, active: bool) -> str:
    """On/off badge with the console theme styles."""
    style = "badge.on" if active else "badge.off"
    value = "on" if active else "off"
    return f"{label}: [{style}]{value}[/]"


def _compact_status_line(state: dict[str, bool]) -> str:
    """One line of badges for the welcome panel."""
    return " · ".join([
        _markup_badge("Verbose", state["verbose"]),
        _markup_badge("Check", state["full_verification"]),
        _markup_badge("Burn", state["burn_protection"]),
        _markup_badge("Thinking", state.get("thinking", False)),
    ])


def _stdin_has_queued_lines(stream: TextIO) -> bool:
    if not stream.readable():
        return False
    try:
        fileno = stream.fileno()
    except (AttributeError, OSError, ValueError):
        pos = stream.tell()
        if stream.read(1):
            stream.seek(pos)
            return True
        return False
    try:
        ready, _, _ = select.select([fileno], [], [], 0)
    except (ValueError, OSError):
        return False
    return bool(ready)


def collect_multiline_input(first_line: str, stream: TextIO | None = None) -> str:
    source = stream or sys.stdin
    lines = [first_line]
    while _stdin_has_queued_lines(source):
        line = source.readline()
        if line == "":
            break
        lines.append(line.rstrip("\n\r"))
    return "\n".join(lines).strip()


def truncate_text(text: str, max_length: int) -> str:
    clean_text = text.strip()
    if len(clean_text) <= max_length:
        return clean_text
    return f"{clean_text[: max_length - 1]}…"


def summarize_tool_input(tool_name: str, action_input: str, file_separator: str) -> str:
    """Summarize the Action Input for the UI timeline (path or query)."""
    text = str(action_input).strip()
    if not text:
        return ""
    if tool_name in {"scrivi_file", "modifica_file"}:
        first_line = text.splitlines()[0].strip()
        if tool_name == "scrivi_file" and file_separator in text:
            first_line = text.partition(file_separator)[0].strip()
        return truncate_text(first_line, MAX_TOOL_INPUT_LENGTH)
    return truncate_text(text.replace("\n", " "), MAX_TOOL_INPUT_LENGTH)


def tool_activity_sentence(tool_name: str) -> str:
    """Human sentence for a tool; fallback: the name with spaces instead of underscores."""
    return _TOOL_SENTENCES.get(tool_name, tool_name.replace("_", " "))


def _prefixes(markers: tuple[str, ...]) -> tuple[str, ...]:
    if markers not in _PREFIX_CACHE:
        prefixes = {m.lower()[:n] for m in markers for n in range(1, len(m) + 1)}
        _PREFIX_CACHE[markers] = tuple(sorted(prefixes, key=len, reverse=True))
    return _PREFIX_CACHE[markers]


def _lookahead(text: str, markers: tuple[str, ...]) -> int:
    tail = text[-_MAX_MARKER_LENGTH:].lower()
    for prefix in _prefixes(markers):
        if tail.endswith(prefix):
            return len(prefix)
    return 0


def _extract_final_answer(text: str) -> str:
    match = _EXTRACT_FINAL_ANSWER_PATTERN.search(text)
    return match.group(1).strip() if match else text.strip()


def _console_reading_width(console: Console) -> int:
    """Reading cap relative to the already synced console width."""
    return reading_width(console.width)


def _render_assistant(body: object, width: int) -> Group:
    """Assistente title + constrained body: the same metaphor in stream and markdown."""
    title = Text.from_markup("[agent.label]Assistente[/]")
    return Group(title, Constrain(body, width=width))


class TurnTimeline:
    def __init__(self, console: Console) -> None:
        self._console = console
        self._status: Status | None = None
        self._verbose = False
        self._file_separator = "---"
        self._stream_action_name: str | None = None
        self._stream_input_emitted = False
        self._action_input_buffer = ""
        self._thinking_label_emitted = False
        self._thinking_line_open = False
        self._thinking_tail = ""
        self._thinking_tail_visible = False

    def reset_turn(self) -> None:
        self.end_tool_execution()
        self._reset_stream_cycle()

    def reset_llm_cycle(self) -> None:
        self.flush_pending_action_input()
        self.close_thinking()
        self._reset_stream_cycle()

    def set_verbose(self, active: bool) -> None:
        self._verbose = active

    def set_file_separator(self, separator: str) -> None:
        self._file_separator = separator

    def write_thinking(self, token: str) -> None:
        """Continuous CoT stream: tokens are appended like a flow being written."""
        if not token:
            return
        if not self._thinking_label_emitted:
            token = _THINKING_LABEL + token
            self._thinking_label_emitted = True
        self._emit_dim(token)

    def emit_thinking(self, text: str) -> None:
        if text:
            self.write_thinking(text)

    def close_thinking(self) -> None:
        if not self._thinking_line_open and not self._thinking_tail:
            return
        remainder = self._thinking_tail
        self._thinking_tail = ""
        self._clear_thinking_tail()
        if remainder:
            self._print_thinking_lines(remainder)
        elif self._thinking_line_open:
            self._console.print()
        self._thinking_line_open = False

    def _emit_dim(self, text: str) -> None:
        """Append the CoT and wrap on words, within the TTY width."""
        self._thinking_tail += text
        self._thinking_line_open = True
        width = max(1, self._console.width)
        while self._thinking_tail:
            split = self._thinking_tail.find("\n")
            if split >= 0:
                line = self._thinking_tail[:split]
                self._thinking_tail = self._thinking_tail[split + 1:]
                self._print_thinking_lines(line)
                continue
            segments = panel_content_wrap_segments(self._thinking_tail, width)
            if len(segments) <= 1:
                self._show_thinking_tail(segments[0] if segments else "")
                return
            for segment in segments[:-1]:
                self._print_thinking_line(segment)
            self._thinking_tail = segments[-1]
            self._show_thinking_tail(self._thinking_tail)
            return

    def _print_thinking_lines(self, line: str) -> None:
        width = max(1, self._console.width)
        if not line:
            self._print_thinking_line("")
            return
        for segment in panel_content_wrap_segments(line, width):
            self._print_thinking_line(segment)

    def _print_thinking_line(self, line: str) -> None:
        self._clear_thinking_tail()
        self._console.print(
            Text(line, style=_THOUGHT_STYLE),
            highlight=False,
            markup=False,
            overflow="crop",
            no_wrap=True,
        )

    def _show_thinking_tail(self, line: str) -> None:
        self._clear_thinking_tail()
        if not line:
            return
        self._console.print(
            Text(line, style=_THOUGHT_STYLE),
            end="",
            highlight=False,
            markup=False,
            overflow="crop",
            no_wrap=True,
        )
        self._thinking_tail_visible = True

    def _clear_thinking_tail(self) -> None:
        if not self._thinking_tail_visible:
            return
        self._console.file.write("\r\033[K")
        self._console.file.flush()
        self._thinking_tail_visible = False

    def emit_action_from_stream(self, tool_name: str) -> None:
        name = tool_name.strip()
        if not name:
            return
        self._stream_action_name = name
        self._stream_input_emitted = False
        self._action_input_buffer = ""

    def emit_action_input_from_stream(self, raw_input: str) -> None:
        if raw_input and self._stream_action_name:
            self._action_input_buffer += raw_input

    def _show_action_input(self, tool_name: str, action_input: str) -> None:
        summary = summarize_tool_input(tool_name, action_input, self._file_separator)
        if self._stream_action_name == tool_name and self._stream_input_emitted:
            return
        self._stream_action_name = tool_name
        self._print_activity(tool_name, summary)
        self._stream_input_emitted = True

    def record_tool_action(self, tool_name: str, action_input: str) -> None:
        if tool_name != "_Exception":
            self._show_action_input(tool_name, action_input)

    def start_tool_execution(self, tool_name: str) -> None:
        self.end_tool_execution()
        if self._stream_action_name == tool_name:
            self.flush_pending_action_input()
        elif not self._stream_input_emitted:
            self._print_activity(tool_name, "")
            self._stream_action_name = tool_name
            self._stream_input_emitted = True
        sentence = tool_activity_sentence(tool_name)
        self._status = self._console.status(f"{sentence}…", spinner="dots")
        self._status.start()

    def end_tool_execution(self) -> None:
        if self._status:
            self._status.stop()
            self._status = None

    def emit_tool_result(self, observation: str) -> None:
        if not self._verbose:
            return
        limit = MAX_VERBOSE_OBSERVATION_LENGTH
        self._console.print(Text(f"  {truncate_text(observation, limit)}", style="tool.result"))

    def emit_tool_error(self, message: str) -> None:
        self.end_tool_execution()
        self._console.print(Text(f"  {truncate_text(str(message), MAX_VERBOSE_OBSERVATION_LENGTH)}", style="error"))

    def flush_pending_action_input(self) -> None:
        if not self._stream_action_name or self._stream_input_emitted:
            return
        summary = summarize_tool_input(
            self._stream_action_name,
            self._action_input_buffer,
            self._file_separator,
        )
        self._print_activity(self._stream_action_name, summary)
        self._stream_input_emitted = True

    def _reset_stream_cycle(self) -> None:
        self._stream_action_name = None
        self._stream_input_emitted = False
        self._action_input_buffer = ""
        self._thinking_label_emitted = False
        self._thinking_line_open = False
        self._thinking_tail = ""
        self._thinking_tail_visible = False

    def _print_activity(self, tool_name: str, summary: str) -> None:
        sentence = tool_activity_sentence(tool_name)
        line = Text()
        line.append(sentence, style="tool.name")
        if summary:
            line.append(": ", style="tool.name")
            line.append(summary, style="tool.input")
        else:
            line.append("…", style="tool.name")
        if self._verbose:
            line.append(f"  ({tool_name})", style="dim")
        self._console.print(line)


class _StreamState(Enum):
    WAITING = auto()
    THOUGHT = auto()
    ACTION = auto()
    ACTION_INPUT = auto()
    FINAL_ANSWER = auto()


def _live_height(live: Live) -> int:
    """Height of the last Live frame; 0 if the render cannot be measured yet."""
    render = getattr(live, "_live_render", None)
    if render is None:
        return 0
    height = getattr(render, "last_render_height", 0)
    return height if isinstance(height, int) else 0


class RendererPanelStream:
    def __init__(
        self,
        console: Console,
        timeline: TurnTimeline,
        on_render: Callable[[], None] | None = None,
        reserved_lines: Callable[[], int] | None = None,
        suspend_scroll: Callable[[], None] | None = None,
        restore_scroll: Callable[[], None] | None = None,
    ) -> None:
        """
        ReAct stream → timeline + Live for the Final Answer (title + body, no box).

        Args:
            console: Rich console shared with the thermal bar.
            timeline: Emitter of the Thought/Action events.
            on_render: Callback after the actual Live refreshes only.
            reserved_lines: Sticky chrome (header + composer + status) to subtract.
            suspend_scroll: Removes the scroll region before a print taller than the viewport.
            restore_scroll: Puts header and status back after that print.
        """
        self._console = console
        self._timeline = timeline
        self._on_render = on_render
        self._reserved_lines = reserved_lines
        self._suspend_scroll = suspend_scroll
        self._restore_scroll = restore_scroll
        self._scroll_suspended = False
        self._live: Live | None = None
        self._live_open = False
        self._free_scroll = False
        self._scroll_tail = ""
        self._scroll_tail_visible = False
        self._state = _StreamState.WAITING
        self._lookahead_buf = ""
        self._final_buffer: list[str] = []
        self._stream_text = Text()
        self._next_live_refresh = 0.0
        self._action_name_buffer = ""
        self._refresh_count = 0
        self._live_lock = threading.Lock()
        self._refresh_dirty = False
        self._stop_refresh = threading.Event()
        self._refresh_thread: threading.Thread | None = None

    def _sync(self) -> None:
        extra = self._reserved_lines() if self._reserved_lines is not None else 0
        sync_dimensions(
            self._console,
            reserve_bar=True,
            reserved_lines=extra,
        )

    @property
    def refresh_count(self) -> int:
        """Live refreshes performed since the renderer started (benchmark diagnostics)."""
        return self._refresh_count

    def reset_turn(self) -> None:
        """Close any Live and reset the parser and content of the previous turn."""
        self._stop_live()
        self._reset_parser()
        self._reset_final_content()

    def start_direct_answer(self) -> None:
        """
        Open the Final Answer stream without waiting for ReAct markers.

        Used by the direct pipeline: the raw text is already the user answer,
        so every chunk must go into the rate-limited plain-text Panel.
        """
        # Close any Live/partials of a previous attempt in the same turn.
        if self._live_open:
            self._stop_live()
        self._state = _StreamState.FINAL_ANSWER
        self._lookahead_buf = ""
        self._action_name_buffer = ""
        self._reset_final_content()

    def reset_llm_call(self) -> None:
        """Close incomplete ReAct sections without losing an active Final Answer."""
        # A pending prefix must be processed before switching LLM cycle.
        if self._lookahead_buf:
            self._process(self._lookahead_buf)
            self._lookahead_buf = ""
        # The partial tool input must be visible before the timeline is reset.
        if self._state is _StreamState.ACTION_INPUT:
            self._timeline.flush_pending_action_input()
        # A tool name without a trailing newline is still a valid action.
        if self._state is _StreamState.ACTION:
            name = self._action_name_buffer.strip()
            # Empty fragments must not generate phantom tool events.
            if name:
                self._timeline.emit_action_from_stream(name)
            self._action_name_buffer = ""
        self._timeline.reset_llm_cycle()
        # A Final Answer can span LLM callbacks and must stay visible.
        if self._state is not _StreamState.FINAL_ANSWER:
            # A partial Live belongs to the failed attempt and must be closed.
            if self._live_open:
                self._stop_live()
            self._reset_final_content()
        # Intermediate states must not contaminate the retry's parser.
        if self._state not in (_StreamState.WAITING, _StreamState.FINAL_ANSWER):
            self._state = _StreamState.WAITING

    def write_token(self, token: str) -> None:
        """
        Forward a chunk to the ReAct parser, preserving any split markers.

        Args:
            token: Text fragment; an empty string has no effect.
        """
        # LLM callbacks may emit empty chunks, which need no parsing.
        if not token:
            return
        buffer = self._lookahead_buf + token
        self._lookahead_buf = ""
        self._process(buffer)

    def finalize(self, result: str) -> None:
        """
        Finalize the stream, converting the whole answer to Markdown only once.

        Args:
            result: Complete LLM answer used as a fallback without a final stream.
        """
        # A partial marker at the tail must be included before finalizing.
        if self._lookahead_buf:
            self._process(self._lookahead_buf)
            self._lookahead_buf = ""
        self._timeline.end_tool_execution()
        stream_text = "".join(self._final_buffer).strip()
        final_text = stream_text or _extract_final_answer(result)
        self._stop_live(final_text if final_text else None)
        self._reset_parser()
        self._reset_final_content()

    def reflow_on_resize(self) -> None:
        """Redraw the plain-text Panel after a resize consumed by the main thread."""
        # Rich Live is not safe to resize from the thermal bar thread.
        if threading.current_thread() is not threading.main_thread():
            return
        self._sync()
        # Without Live the sync is enough; any chrome is handled separately.
        if not self._live_open or self._live is None:
            return
        self._live.update(
            _render_assistant(self._stream_text, _console_reading_width(self._console)),
            refresh=True,
        )
        self._next_live_refresh = (
            time.monotonic() + _LIVE_PANEL_REFRESH_INTERVAL_SEC
        )
        self._notify_render()

    @property
    def has_open_live(self) -> bool:
        """True if the Final Answer stream is in progress with Live active."""
        return self._live_open

    def _reset_parser(self) -> None:
        """Reset the ReAct parser to its initial state without touching the content."""
        self._state = _StreamState.WAITING
        self._lookahead_buf = ""
        self._action_name_buffer = ""

    def _reset_final_content(self) -> None:
        """Reset the linear buffer and the plain-text renderable of the Final Answer."""
        self._final_buffer.clear()
        self._stream_text = Text()
        self._next_live_refresh = 0.0
        self._free_scroll = False
        self._scroll_tail = ""
        self._scroll_tail_visible = False
        self._scroll_suspended = False

    def _notify_render(self) -> None:
        """Notify a real refresh so the bar can restore itself."""
        # The callback is optional to allow isolated renderers in tests and clients.
        if self._on_render is not None:
            self._on_render()

    def _stop_live(self, update_final: str | None = None) -> None:
        """
        Close Live and, if available, replace the plain text with the final Markdown.

        Args:
            update_final: Full text to render; None keeps the plain text.
        """
        self._sync()
        width = _console_reading_width(self._console)
        # The tall stream is already in the scrollback: a later Markdown would duplicate the text.
        if self._free_scroll:
            self._stop_refresh_thread()
            self._close_scroll_tail()
            self._console.print()
            self._free_scroll = False
            self._restore_tty_scroll()
            self._notify_render()
            return
        # Without Live the fallback result must still be printed with the same metaphor.
        if not self._live_open or self._live is None:
            self._stop_refresh_thread()
            if update_final:
                renderable = _render_assistant(
                    Markdown(update_final, code_theme="monokai"),
                    width,
                )
                self._print_answer(renderable)
            return
        self._stop_refresh_thread()
        renderable = None
        if update_final is not None:
            renderable = _render_assistant(
                Markdown(update_final, code_theme="monokai"),
                width,
            )
        if renderable is not None and self._render_exceeds_viewport(renderable):
            # stop() with visible overflow reprints inside DECSTBM and cuts the start.
            self._release_live_without_dump(self._live)
            self._print_answer(renderable)
            self._notify_render()
            return
        if renderable is not None:
            self._live.update(renderable, refresh=True)
        self._live.stop()
        self._live = None
        self._live_open = False
        self._next_live_refresh = 0.0
        self._refresh_dirty = False
        self._notify_render()

    def _append_final(self, text: str) -> None:
        """
        Append a chunk in O(1) and update the plain text within the rate limit.

        Args:
            text: Final Answer fragment, kept without transformations.
        """
        # Empty fragments must not open a Live with no visible content.
        if not text:
            return
        with self._live_lock:
            self._final_buffer.append(text)
            self._stream_text.append(text)
            self._refresh_dirty = True
            free_scroll = self._free_scroll
        if free_scroll:
            self._emit_free_scroll_text(text)
            return
        # The first content must be rendered immediately, without a timer.
        if not self._live_open:
            self._start_live()
            return

    def _start_live(self) -> None:
        """Open Live with the first plain-text content already present and visible."""
        self._sync()
        width = _console_reading_width(self._console)
        self._console.print()
        self._live = Live(
            _render_assistant(self._stream_text, width),
            console=self._console,
            auto_refresh=False,
            refresh_per_second=_LIVE_PANEL_REFRESH_PER_SECOND,
            transient=False,
            redirect_stdout=False,
            redirect_stderr=False,
            # The ellipsis keeps long answers from spilling into the reserved bar.
            vertical_overflow="ellipsis",
        )
        self._live.start(refresh=True)
        self._live_open = True
        self._refresh_count += 1
        self._refresh_dirty = False
        if self._live_beyond_viewport(self._live):
            with self._live_lock:
                self._switch_to_free_scroll_locked(self._live)
            self._notify_render()
            return
        self._stop_refresh.clear()
        self._refresh_thread = threading.Thread(
            target=self._loop_refresh_live,
            name="assistant-live-refresh",
            daemon=True,
        )
        self._refresh_thread.start()
        self._next_live_refresh = (
            time.monotonic() + _LIVE_PANEL_REFRESH_INTERVAL_SEC
        )
        self._notify_render()

    def _loop_refresh_live(self) -> None:
        """Redraw Live outside the decode thread, at most 20 Hz."""
        while not self._stop_refresh.wait(_LIVE_PANEL_REFRESH_INTERVAL_SEC):
            if not self._run_live_refresh():
                return

    def _run_live_refresh(self) -> bool:
        """
        Apply a Live refresh if dirty; switch to free scroll if the Panel fills the TTY.

        Returns:
            False if Live was closed and the worker must exit.
        """
        notify = False
        with self._live_lock:
            live = self._live
            if live is None:
                return False
            if self._refresh_dirty:
                self._refresh_dirty = False
                live.refresh()
                self._refresh_count += 1
                notify = True
            if self._live_beyond_viewport(live):
                self._switch_to_free_scroll_locked(live)
                notify = True
                live = None
        if notify:
            self._notify_render()
        return live is not None

    def _live_beyond_viewport(self, live: Live) -> bool:
        """True when the Live frame takes the whole usable height (crop/ellipsis)."""
        height = _live_height(live)
        return height > 0 and height >= self._console.height

    def _switch_to_free_scroll_locked(self, live: Live) -> None:
        """
        Give up in-place redrawing and pour the text into the TTY scrollback.

        Rich Live with ellipsis overflow crops the answer to the viewport height
        and repositions the cursor on every refresh: the terminal has nothing
        to scroll until Live is closed. After this transition new tokens are
        only appended.
        """
        self._release_live_without_dump(live)
        self._free_scroll = True
        self._suspend_tty_scroll()
        self._refresh_dirty = False
        self._console.print(
            Text.from_markup("[agent.label]Assistente[/]"),
            highlight=False,
        )
        if self._stream_text.plain:
            self._emit_free_scroll_text(self._stream_text)

    def _release_live_without_dump(self, live: Live) -> None:
        """Close Live erasing only the area already drawn, without reprinting the Panel."""
        with live._lock:
            render = live._live_render
            if live.console.is_terminal and _live_height(live) > 0:
                live.console.control(render.restore_cursor())
            render._shape = None
        live.refresh = lambda: None
        live.transient = False
        live.stop()
        self._live = None
        self._live_open = False

    def _render_exceeds_viewport(self, renderable: object) -> bool:
        """True if the frame, without cropping, is taller than the console's usable area."""
        options = self._console.options.update(height=None)
        height = len(self._console.render_lines(renderable, options))
        return height > self._console.height

    def _print_answer(self, renderable: object) -> None:
        """Print outside DECSTBM so the history keeps the top lines too."""
        self._suspend_tty_scroll()
        self._console.print()
        self._console.print(renderable)
        self._restore_tty_scroll()

    def _suspend_tty_scroll(self) -> None:
        """Only once per answer: later prints use the whole scrollback."""
        if self._scroll_suspended or self._suspend_scroll is None:
            return
        self._suspend_scroll()
        self._scroll_suspended = True

    def _restore_tty_scroll(self) -> None:
        """Restore the scroll region after the tall print."""
        if not self._scroll_suspended:
            return
        self._scroll_suspended = False
        if self._restore_scroll is not None:
            self._restore_scroll()

    def _emit_free_scroll_text(self, text: str | Text) -> None:
        """Append text to the scrollback, wrapping on words within the TTY margin."""
        if isinstance(text, Text):
            fragment = text.plain
        else:
            fragment = text
        if not fragment:
            return
        self._scroll_tail += fragment
        self._flush_scroll_tail()

    def _flush_scroll_tail(self) -> None:
        """Print the lines already closed and keep only the last incomplete one on screen."""
        width = _console_reading_width(self._console)
        while self._scroll_tail:
            split = self._scroll_tail.find("\n")
            if split >= 0:
                line = self._scroll_tail[:split]
                self._scroll_tail = self._scroll_tail[split + 1:]
                self._finish_scroll_lines(line, width)
                continue
            segments = panel_content_wrap_segments(self._scroll_tail, width)
            if len(segments) <= 1:
                self._show_scroll_tail(segments[0] if segments else "")
                return
            for segment in segments[:-1]:
                self._finish_scroll_line(segment)
            self._scroll_tail = segments[-1]
            self._show_scroll_tail(self._scroll_tail)
            return

    def _finish_scroll_lines(self, line: str, width: int) -> None:
        """Break after an already finished line, splitting it if it exceeds the width."""
        if not line:
            self._finish_scroll_line("")
            return
        for segment in panel_content_wrap_segments(line, width):
            self._finish_scroll_line(segment)

    def _finish_scroll_line(self, line: str) -> None:
        """Replace the visible tail, if any, and close the line with a newline."""
        self._clear_scroll_tail()
        self._console.print(
            Text(line),
            highlight=False,
            markup=False,
            overflow="crop",
            no_wrap=True,
        )

    def _show_scroll_tail(self, line: str) -> None:
        """Rewrite the last incomplete line, without making the terminal wrap."""
        self._clear_scroll_tail()
        if not line:
            return
        self._console.print(
            Text(line),
            end="",
            highlight=False,
            markup=False,
            overflow="crop",
            no_wrap=True,
        )
        self._scroll_tail_visible = True

    def _clear_scroll_tail(self) -> None:
        """Go back to column 1 and erase the tail already drawn on the current line."""
        if not self._scroll_tail_visible:
            return
        self._console.file.write("\r\033[K")
        self._console.file.flush()
        self._scroll_tail_visible = False

    def _close_scroll_tail(self) -> None:
        """Close the last incomplete line so the prompt is not left mid-text."""
        if not self._scroll_tail and not self._scroll_tail_visible:
            return
        remainder = self._scroll_tail
        self._scroll_tail = ""
        self._clear_scroll_tail()
        if remainder:
            self._finish_scroll_lines(
                remainder,
                _console_reading_width(self._console),
            )

    def _stop_refresh_thread(self) -> None:
        """Stop the refresh worker and wait for it to exit."""
        self._stop_refresh.set()
        thread = self._refresh_thread
        self._refresh_thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def _update_live_if_due(self) -> None:
        """Mark the Panel as dirty: the worker performs the refresh."""
        if self._live is None:
            return
        self._refresh_dirty = True

    def _process(self, text: str) -> None:
        """
        Consume a fragment until every internal ReAct transition has been processed.

        Args:
            text: Text not yet classified by the parser.
        """
        remainder: str | None = text
        # A single chunk can contain several consecutive ReAct sections.
        while remainder:
            remainder = self._process_fragment(remainder)

    def _process_fragment(self, text: str) -> str | None:
        """
        Route the text according to the current ReAct state.

        Args:
            text: Fragment to consume in the current state.

        Returns:
            Remainder to re-examine after a transition, or None if consumed.
        """
        # After Final Answer every fragment belongs to the visible content.
        if self._state is _StreamState.FINAL_ANSWER:
            self._append_final(text)
            return None
        # Thought ends only when a compatible ReAct marker appears.
        if self._state is _StreamState.THOUGHT:
            return self._consume_until_marker(
                text, _THOUGHT_END_PATTERN, _THOUGHT_END_MARKERS,
                self._timeline.emit_thinking, self._timeline.close_thinking,
            )
        # The action name is delimited by the first newline, even across chunks.
        if self._state is _StreamState.ACTION:
            return self._consume_action(text)
        # Action Input ends at the Observation and is summarized by the timeline.
        if self._state is _StreamState.ACTION_INPUT:
            return self._consume_until_marker(
                text, _ACTION_INPUT_END_PATTERN, _INPUT_END_MARKERS,
                self._timeline.emit_action_input_from_stream,
                self._timeline.flush_pending_action_input,
            )
        return self._find_markers(text)

    def _consume_until_marker(
        self,
        text: str,
        end_pattern: re.Pattern[str],
        lookahead_markers: tuple[str, ...],
        on_chunk: Callable[[str], None],
        on_close: Callable[[], None] | None = None,
    ) -> str | None:
        """
        Emit text up to a marker, keeping its lookahead across chunks.

        Args:
            text: Current fragment.
            end_pattern: Regular expression that closes the section.
            lookahead_markers: Markers whose trailing prefixes must be preserved.
            on_chunk: Destination of the consumed content.
            on_close: Optional callback run on closing.

        Returns:
            Text from the marker onward to re-examine, or None.
        """
        end_match = end_pattern.search(text)
        # A complete marker closes the section and is re-examined from the WAITING state.
        if end_match:
            on_chunk(text[: end_match.start()])
            # Some sections need an explicit flush before the transition.
            if on_close:
                on_close()
            self._state = _StreamState.WAITING
            return text[end_match.start() :]
        n = _lookahead(text, lookahead_markers)
        # A prefix at the tail might complete in the next chunk.
        if n:
            on_chunk(text[:-n])
            self._lookahead_buf = text[-n:]
            return None
        on_chunk(text)
        return None

    def _consume_action(self, text: str) -> str | None:
        """
        Accumulate the tool name up to the newline that ends the Action declaration.

        Args:
            text: Fragment following the Action marker.

        Returns:
            Text from the newline onward to re-examine, or None.
        """
        idx = text.find("\n")
        # The newline completes the name and allows going back to marker search.
        if idx >= 0:
            name = (self._action_name_buffer + text[:idx]).strip()
            self._action_name_buffer = ""
            # An empty name does not represent a real tool invocation.
            if name:
                self._timeline.emit_action_from_stream(name)
            self._state = _StreamState.WAITING
            return text[idx:]
        self._action_name_buffer += text
        return None

    def _find_markers(self, text: str) -> str | None:
        """
        Look for the start of the next ReAct section, keeping the historical priority.

        Args:
            text: Fragment received while the parser is waiting.

        Returns:
            Content after the marker, or None if the chunk was consumed.
        """
        # The existing priority avoids regressions in marker recognition.
        for pattern, state, offset_fn in (
            (_FINAL_ANSWER_START_PATTERN, _StreamState.FINAL_ANSWER, lambda m: m.end()),
            (_THOUGHT_START_PATTERN, _StreamState.THOUGHT, lambda m: m.start()),
            (_ACTION_INPUT_START_PATTERN, _StreamState.ACTION_INPUT, lambda m: m.end()),
            (_ACTION_START_PATTERN, _StreamState.ACTION, lambda m: m.end()),
        ):
            match = pattern.search(text)
            # The first marker by historical priority determines the new state.
            if match:
                # A new Final Answer replaces only any partial content.
                if state is _StreamState.FINAL_ANSWER:
                    self._reset_final_content()
                # A new Action must not inherit fragments of the previous name.
                if state is _StreamState.ACTION:
                    self._action_name_buffer = ""
                self._state = state
                return text[offset_fn(match):]

        n = _lookahead(text, _START_MARKERS)
        # A split marker must stay pending until the next token.
        if n:
            # While waiting there is no previous final content to show.
            if self._state is _StreamState.WAITING:
                self._lookahead_buf = text[-n:]
            else:
                self._append_final(text[:-n])
                self._lookahead_buf = text[-n:]
            return None
        # States other than WAITING handle the text in the branches above.
        if self._state is not _StreamState.WAITING:
            return None
        # Text without markers does not go to the panel: avoids temporary answers being overwritten.
        return None


class ThermalBar:
    def __init__(self, console: Console, stream: TextIO | None = None) -> None:
        """
        Status bar on the last TTY line (thermal + LLM metrics).

        Args:
            console: Shared Rich console; width/height stay aligned with the TTY.
            stream: Output for ANSI sequences (default stdout).
        """
        self._console = console
        self._stream = stream or sys.stdout
        self._lock = threading.Lock()
        self._enable = False
        self._show_metrics = False
        self._reserved_lines: int | None = None
        self._header_lines = 0
        self._composer_lines = 0
        self._suspended_region = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._input_in_progress = False
        self._metrics: LLMTurnStats | None = None
        self._metrics_source: Callable[[], LLMTurnStats | None] | None = None
        self._last_line = ""
        self._resize_requested = threading.Event()
        self._on_resize: Callable[[], None] | None = None
        # Previous handler (callable / SIG_DFL / SIG_IGN) to restore in stop()
        self._previous_sigwinch: object | None = None
        self._sigwinch_installed = False

    def set_on_resize(self, callback: Callable[[], None] | None) -> None:
        """
        Register a callback run after the sync/scroll-region on a TTY resize.

        Args:
            callback: Function that queues the Live work on the main thread, or None.
        """
        self._on_resize = callback

    def start(self) -> None:
        if self._enable or not self._console.is_terminal:
            return
        self._enable = True
        self._stop.clear()
        self._resize_requested.clear()
        self._sync_console()
        self._reserve_last_line()
        self.redraw()
        self._install_sigwinch()
        self._thread = threading.Thread(target=self._polling, name="barra-termica", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._enable and self._thread is None:
            return
        self._stop.set()
        # Wake up the poll waiting for the resize flag
        self._resize_requested.set()
        if self._thread:
            self._thread.join(timeout=_THERMAL_POLLING_SEC + 1)
            self._thread = None
        self._restore_sigwinch()
        self._enable = False
        self._release()

    def set_show_metrics(self, active: bool) -> None:
        """In verbose the bar adds TTFT, prefill and the other latencies."""
        self._show_metrics = active

    def set_input_in_progress(self, active: bool) -> None:
        self._input_in_progress = active
        if not active:
            self.set_composer_lines(0)

    def set_composer_lines(self, lines: int) -> None:
        """Reserve N lines above the status for the sticky composer."""
        new = max(0, int(lines))
        if new == self._composer_lines:
            return
        self._composer_lines = new
        if self._enable and self._console.is_terminal:
            self._reserve_last_line()

    def chrome_lines(self) -> int:
        """Lines taken from the transcript: header + composer + status."""
        if not self._enable:
            return 1
        return self._header_lines + 1 + self._composer_lines

    @property
    def header_active(self) -> bool:
        """True if line 1 is outside the scroll region."""
        return self._enable and self._header_lines > 0

    @property
    def input_in_progress(self) -> bool:
        """True while the user prompt is being read."""
        return self._input_in_progress

    @property
    def enable(self) -> bool:
        """True if the bar has reserved the last TTY line."""
        return self._enable

    def after_clear(self) -> None:
        """Re-reserve the scroll region after a console.clear() (without redrawing yet)."""
        if not self._enable or not self._console.is_terminal:
            return
        self._reserve_last_line()
        self._last_line = ""

    def set_turn_metrics(self, metrics: LLMTurnStats | None) -> None:
        if metrics == self._metrics:
            return
        self._metrics = metrics
        self._last_line = ""
        self.redraw()

    def update_turn_metrics(self, metrics: LLMTurnStats | None) -> None:
        """Store the snapshot without redrawing: the next Live refresh shows it."""
        self._metrics = metrics

    def set_metrics_source(
        self,
        source: Callable[[], LLMTurnStats | None] | None,
    ) -> None:
        """
        Attach a live accumulator read on every bar redraw.

        Args:
            source: snapshot() of the turn's accumulator, or None at the end of the turn.
        """
        self._metrics_source = source

    def redraw(self, *, force: bool = False) -> None:
        """
        Update the status line at the bottom of the TTY.

        Args:
            force: If True, redraw even during input_in_progress (after clear/reflow).
        """
        if not self._enable or not self._console.is_terminal or self._suspended_region:
            return
        # During input avoid overwriting the cursor, unless an explicit reflow
        if self._input_in_progress and not force:
            return
        self._sync_console()
        columns, _ = tty_size(self._console)
        # Cell budget: leave a 1-column margin as before
        max_columns = max(0, columns - 1)
        metrics = self._metrics
        source = self._metrics_source
        # The thermal poll and Live must re-read the accumulator, not a frozen snapshot.
        if source is not None:
            metrics = source()
        line = status_bar_line(
            read_thermal_state(),
            read_cpu_temperature_celsius(),
            metrics,
            ansi=True,
            max_columns=max_columns,
            power=detect_power(),
            verbose=self._show_metrics,
        )
        if line != self._last_line:
            self._draw(line)
            self._last_line = line

    def _sync_console(self) -> None:
        sync_dimensions(
            self._console,
            reserve_bar=self._enable,
            reserved_lines=self.chrome_lines() if self._enable else 0,
        )

    def _write(self, sequence: str) -> None:
        with self._lock:
            self._stream.write(sequence)
            self._stream.flush()

    def _reserve_last_line(self) -> None:
        _, lines = tty_size(self._console)
        if lines < 2:
            self._header_lines = 0
            return
        header = 1 if lines >= 4 else 0
        low = 1 + self._composer_lines
        top = 1 + header
        bottom = lines - low
        if bottom < top:
            bottom = top
        a = _ANSI
        if header == 0 and self._composer_lines == 0:
            self._write(
                f"\n{a['save']}{scroll_region_sequence(top, bottom)}"
                f"{a['restore']}\033[1A"
            )
        else:
            self._write(f"{a['save']}{scroll_region_sequence(top, bottom)}{a['restore']}")
        self._header_lines = header
        self._reserved_lines = lines

    @property
    def suspended_region(self) -> bool:
        """True while the transcript uses the whole scrollback."""
        return self._suspended_region

    def suspend_scroll_region(self) -> None:
        """Remove DECSTBM: a print taller than the viewport goes into the scrollback."""
        if not self._enable or not self._console.is_terminal:
            return
        self._suspended_region = True
        a = _ANSI
        self._write(f"{a['save']}{a['reset_region']}{a['restore']}")

    def restore_scroll_region(self) -> None:
        """Reapply the scroll region and redraw the status line."""
        self._suspended_region = False
        if not self._enable or not self._console.is_terminal:
            return
        self._reserve_last_line()
        self._last_line = ""
        self.redraw(force=True)

    def make_room_below(self, lines: int) -> None:
        """
        Push the text up into the scrollback, leaving free lines at the bottom.

        The composer is then drawn on those lines, without covering the answer.
        """
        if lines < 1 or not self._enable or not self._console.is_terminal:
            return
        a = _ANSI
        self._suspended_region = True
        self._write(f"{a['save']}{a['reset_region']}{a['restore']}")
        self._write("\n" * lines)
        self.restore_scroll_region()

    def _release(self) -> None:
        if not self._console.is_terminal:
            return
        _, lines = tty_size(self._console)
        a = _ANSI
        self._write(f"{a['save']}{a['reset_region']}\033[{lines};1H{a['clear_line']}{a['restore']}")
        self._reserved_lines = None

    def _draw(self, text: str) -> None:
        columns, lines = tty_size(self._console)
        # On a height change: recompute the scroll region right away
        if self._reserved_lines is not None and self._reserved_lines != lines:
            self._reserve_last_line()
        # ANSI-aware truncation: avoids splitting SGR and wide-glyph overflow
        truncated = truncate_visible(text, max(0, columns - 1))
        a = _ANSI
        self._write(f"{a['save']}\033[{lines};1H{a['clear_line']}{truncated}{a['restore']}")

    def _install_sigwinch(self) -> None:
        """Register SIGWINCH to flag a reflow; no-op on platforms without the signal."""
        # Windows / environments without SIGWINCH: the thermal poll stays the only sync
        if not hasattr(signal, "SIGWINCH") or self._sigwinch_installed:
            return

        def _handler(_signum: int, _frame: object) -> None:
            """Signal the resize without doing I/O or Rich operations in the signal handler."""
            # The signal handler wakes up the poll; the poll will queue the Live reflow.
            self._resize_requested.set()

        try:
            self._previous_sigwinch = signal.signal(signal.SIGWINCH, _handler)
            self._sigwinch_installed = True
        except (ValueError, OSError):
            # signal.signal only from the main thread: ignore if called elsewhere
            self._sigwinch_installed = False

    def _restore_sigwinch(self) -> None:
        """Restore the SIGWINCH handler that preceded the session."""
        if not self._sigwinch_installed or not hasattr(signal, "SIGWINCH"):
            return
        try:
            previous = self._previous_sigwinch
            # None → SIG_DFL if no handler was saved
            if previous is None:
                signal.signal(signal.SIGWINCH, signal.SIG_DFL)
            else:
                signal.signal(signal.SIGWINCH, previous)
        except (ValueError, OSError):
            # Best-effort: do not block the bar's shutdown
            pass
        self._sigwinch_installed = False
        self._previous_sigwinch = None

    def _apply_resize(self) -> None:
        """
        Sync dimensions and notify the main thread to redraw the UI.

        The Panels are redrawn on the main thread (flag), not here:
        avoids races between the polling thread and input()/Live.
        """
        self._sync_console()
        # Update the scroll region right away (even during input: the main thread will clear)
        if self._enable and self._console.is_terminal and not self._suspended_region:
            self._reserve_last_line()
            self._last_line = ""
            # Bar: only if not in input; during input the main thread calls force=True
            if not self._input_in_progress:
                self.redraw()
        # The callback must only set thread-safe flags for the main thread.
        if self._on_resize is not None:
            self._on_resize()

    def _polling(self) -> None:
        """
        Daemon loop: reacts to resizes within ~50ms and updates the thermal state every 2s.
        """
        next_thermal = time.monotonic()
        while not self._stop.is_set():
            # Short wait: balances resize latency and CPU load
            self._stop.wait(timeout=_RESIZE_POLL_INTERVAL_SEC)
            # Exit requested: do not process further events
            if self._stop.is_set():
                break
            # The worker syncs the bar and only queues the Live reflow.
            if self._resize_requested.is_set():
                self._resize_requested.clear()
                self._apply_resize()
            now = time.monotonic()
            # Periodic thermal poll, independent of resizes
            if now >= next_thermal:
                next_thermal = now + _THERMAL_POLLING_SEC
                invalidate_thermal_cache()
                self.redraw()


class ConsoleUI:
    def __init__(self) -> None:
        """Initialize Console, timeline, streaming and thread-safe resize coordination."""
        ioctl = read_ioctl_tty_size()
        if ioctl is not None:
            columns, lines = ioctl
        else:
            try:
                dim = shutil.get_terminal_size(fallback=(120, 24))
            except OSError:
                dim = os.terminal_size((120, 24))
            columns, lines = dim.columns, dim.lines
        self.console = Console(
            theme=_THEME,
            width=max(1, columns - TTY_WIDTH_MARGIN),
            height=max(1, lines - 1),
        )
        self._timeline = TurnTimeline(self.console)
        self.thermal_bar = ThermalBar(self.console)
        self._stream = RendererPanelStream(
            self.console,
            self._timeline,
            on_render=self._after_stream_render,
            reserved_lines=self.thermal_bar.chrome_lines,
            suspend_scroll=self.thermal_bar.suspend_scroll_region,
            restore_scroll=self._restore_answer_scroll,
        )
        self._welcome_state: dict[str, bool] | None = None
        # Documents folder (closed corpus) shown in the header, welcome and /help.
        self._documents_folder = ""
        self._model_label = ""
        self._ui_needs_redraw = threading.Event()
        self._live_resize_requested = threading.Event()
        self._next_stream_bar_refresh = 0.0
        self._current_input_buffer = ""
        self._current_prompt_title = ""
        self._drawn_input_width = 0
        self._sticky_composer_lines = 0
        self._progress_in_progress = False
        self._deferred_header = False
        self.thermal_bar.set_on_resize(self._handle_resize)

    def _restore_answer_scroll(self) -> None:
        """After a tall answer: scroll region, status and header go back in place."""
        self.thermal_bar.restore_scroll_region()
        self._draw_sticky_header()

    def _after_stream_render(self) -> None:
        """
        Update the thermal bar at a limited rate during the stream.

        Prevents every Live refresh (up to 20/s) from waking thermal reads and
        ANSI redraws, which would otherwise create backpressure towards Ollama.
        """
        now = time.monotonic()
        # The turn's first render updates the perceived TTFT metrics right away.
        if now < self._next_stream_bar_refresh:
            return
        self._next_stream_bar_refresh = (
            now + _STREAM_BAR_REFRESH_INTERVAL_SEC
        )
        self.thermal_bar.redraw()

    def _sync(self) -> None:
        extra = self.thermal_bar.chrome_lines() if self.thermal_bar.enable else 1
        sync_dimensions(
            self.console,
            reserve_bar=True,
            reserved_lines=extra,
        )

    def _use_sticky_chrome(self) -> bool:
        return (
            self.thermal_bar.enable
            and self.console.is_terminal
            and sys.stdout.isatty()
        )

    def _draw_sticky_header(self) -> None:
        if self.thermal_bar.suspended_region:
            # A paint on line 1 while the screen scrolls ends up in the history.
            self._deferred_header = True
            return
        if not self._use_sticky_chrome() or not self.thermal_bar.header_active:
            return
        self._deferred_header = False
        state = self._welcome_state or _default_runtime_state()
        print_sticky_header(
            self.console,
            session_header_markup(
                state,
                self._model_label,
                self._documents_folder,
            ),
            is_terminal=True,
        )

    def _draw_sticky_composer(self, editor: LineEditor) -> None:
        text = editor.text()
        _columns, tty_lines = tty_size(self.console)
        header = self.thermal_bar.header_active
        new = sticky_composer_height(
            text,
            self.console.width,
            tty_lines,
            header=header,
        )
        if self._sticky_composer_lines > new:
            clear_absolute_lines(
                sticky_composer_start_line(tty_lines, self._sticky_composer_lines),
                self._sticky_composer_lines,
                is_terminal=True,
            )
        self._sticky_composer_lines = new
        self.thermal_bar.set_composer_lines(new)
        print_sticky_composer(
            self.console,
            self._current_prompt_title,
            text_with_selection(text, editor.anchor, editor.cursor),
            text,
            editor.cursor,
            tty_lines,
            new,
            is_terminal=True,
        )
        self._drawn_input_width = self.console.width

    def _close_sticky_composer(self) -> None:
        if not self._use_sticky_chrome():
            self._sticky_composer_lines = 0
            return
        _columns, tty_lines = tty_size(self.console)
        old = self._sticky_composer_lines
        if old > 0:
            clear_absolute_lines(
                sticky_composer_start_line(tty_lines, old),
                old,
                is_terminal=True,
            )
        self._sticky_composer_lines = 0
        self.thermal_bar.set_composer_lines(0)
        next_line = sticky_scroll_end_line(
            tty_lines,
            header=self.thermal_bar.header_active,
            composer_lines=old,
        ) + 1
        max_line = sticky_scroll_end_line(
            tty_lines,
            header=self.thermal_bar.header_active,
            composer_lines=0,
        )
        position_cursor_absolute(
            min(next_line, max_line),
            1,
            is_terminal=True,
        )

    def _handle_resize(self) -> None:
        """Classify the resize in the bar thread without running Rich Live operations."""
        # An active Live will be resized by the next callback on the main thread.
        if self._stream.has_open_live:
            self._live_resize_requested.set()
            return
        # Without Live, the next prompt adapts its width without a wipe.
        self._ui_needs_redraw.set()

    def _consume_requested_live_resize(self) -> None:
        """Consume on the main thread a Live resize queued by the thermal bar."""
        # Without a pending request there is no need to query or update Rich Live.
        if not self._live_resize_requested.is_set():
            return
        # An anomalous stream callback on a worker must leave the flag to main.
        if threading.current_thread() is not threading.main_thread():
            return
        self._live_resize_requested.clear()
        self._stream.reflow_on_resize()
        self._draw_sticky_header()

    def _welcome_body(self, state: dict[str, bool]) -> str:
        """
        Compact body of the startup Panel: one intro line and the badges.

        Args:
            state: Runtime flags (`verbose`, `full_verification`,
                `burn_protection`, `thinking`).

        Returns:
            Rich markup. The full commands live in `_help_body`.
        """
        return (
            "Assistente pronto. Rispondo solo dai documenti della cartella "
            f"[bold]{escape(self._documents_folder or '.')}[/] e scrivo solo file .txt.\n"
            "Scrivi una richiesta (exit/quit/esci). Comandi: /help\n"
            f"{_compact_status_line(state)}"
        )

    def _help_body(self, state: dict[str, bool]) -> str:
        """On-demand manual: status, slash commands and keyboard guide."""
        lines_state = "\n".join([
            f"  {_markup_badge('Verbose', state['verbose'])}",
            f"  {_markup_badge('Check', state['full_verification'])}",
            f"  {_markup_badge('Burn protection', state['burn_protection'])}",
            f"  {_markup_badge('Thinking', state.get('thinking', False))}",
        ])
        toggle_syntax = escape("[on|off|toggle]")
        model_syntax = escape("[fast|default]")
        commands = "\n".join([
            f"  Flag:     /verbose | /check | /burn_protection | /thinking  {toggle_syntax}",
            f"  Runtime:  /model {model_syntax}",
            "  Sessione: /clear — azzera storico conversazione",
            "            /errori — ultimi errori di processo",
            "            /help — questa guida",
            *KEYBOARD_GUIDE_LINES,
        ])
        folder = escape(self._documents_folder or ".")
        return (
            f"Cartella documenti: {folder}\n\n"
            f"Stato:\n{lines_state}\n\nComandi:\n{commands}"
        )

    def _reprint_chrome(
        self,
        input_text: str | None = None,
        *,
        anchor: int | None = None,
        cursor: int = 0,
    ) -> None:
        """Reflow header and composer after a resize, without clearing the history."""
        if not self.console.is_terminal or not sys.stdout.isatty():
            return
        self._progress_in_progress = False
        if self._use_sticky_chrome():
            self._sync()
            self._draw_sticky_header()
            if input_text is not None:
                editor = LineEditor()
                editor.insert(input_text)
                editor.cursor = max(0, min(cursor, len(editor.text())))
                editor.anchor = anchor
                self._draw_sticky_composer(editor)
            self.thermal_bar.redraw(force=True)
            return
        if input_text is None:
            self._sync()
            self.thermal_bar.redraw(force=True)
            return
        old_width = self._drawn_input_width or self.console.width
        clear_lines_above(
            cuu_clear_composer(input_text, old_width),
            is_terminal=True,
        )
        self._sync()
        print_composer(
            self.console,
            self._current_prompt_title,
            text_with_selection(input_text, anchor, cursor),
            sync=False,
        )
        self._drawn_input_width = self.console.width
        self.thermal_bar.redraw(force=True)

    def print_help(self) -> None:
        """Print the guide panel (status, commands, keyboard)."""
        self._finalize_progress()
        self._sync()
        state = self._welcome_state or _default_runtime_state()
        print_panel(
            self.console,
            self._help_body(state),
            "[agent.label]Aiuto[/]",
            padding=(0, 1),
            sync=False,
        )
        self.thermal_bar.redraw()

    def prepare_new_stream_turn(self) -> None:
        """Reset timeline, renderer, pending resizes and metrics for the new turn."""
        self._timeline.reset_turn()
        self._stream.reset_turn()
        self._live_resize_requested.clear()
        self._next_stream_bar_refresh = 0.0
        self.thermal_bar.set_metrics_source(None)
        self.thermal_bar.set_turn_metrics(None)

    def start_direct_answer(self) -> None:
        """
        Prepare the stream for an answer without the ReAct protocol.

        Returns:
            None. Side effect: the next stream goes into the final body.
        """
        self._stream.start_direct_answer()

    def set_verbose(self, active: bool) -> None:
        self._timeline.set_verbose(active)
        self.thermal_bar.set_show_metrics(active)
        if self._welcome_state is not None:
            self._welcome_state["verbose"] = active

    def set_file_separator(self, separator: str) -> None:
        self._timeline.set_file_separator(separator)

    def record_tool_action(self, name: str, action_input: str) -> None:
        self._finalize_progress()
        self._timeline.record_tool_action(name, action_input)

    def start_tool_execution(self, name: str) -> None:
        self._finalize_progress()
        self._timeline.start_tool_execution(name)

    def end_tool_execution(self) -> None:
        self._timeline.end_tool_execution()

    def emit_tool_result(self, observation: str) -> None:
        self._timeline.emit_tool_result(observation)

    def emit_tool_error(self, message: str) -> None:
        self._timeline.emit_tool_error(message)

    def write_token(self, token: str) -> None:
        """
        Process a chunk and apply any pending Live resize on the main thread.

        Args:
            token: LLM fragment to forward unchanged to the ReAct parser.
        """
        self._finalize_progress()
        self._stream.write_token(token)
        self._consume_requested_live_resize()

    def write_thinking(self, token: str) -> None:
        """Write a native CoT fragment to the timeline, without opening the panel."""
        # The progress line (e.g. «Comprendo la richiesta») is live and has no newline:
        # without closing it, the first token stays glued to that sentence.
        self._finalize_progress()
        self._timeline.write_thinking(token)

    def close_thinking(self) -> None:
        """Close the Pensiero line before the answer or the next LLM cycle."""
        self._timeline.close_thinking()

    def reset_llm_call(self) -> None:
        self._stream.reset_llm_call()

    def finalize(self, result: str) -> None:
        """
        Apply the last pending resize before closing and finalizing Live.

        Args:
            result: Complete LLM answer used by the renderer as a fallback.
        """
        self._consume_requested_live_resize()
        self._stream.finalize(result)

    def set_turn_metrics(self, metrics: LLMTurnStats | None) -> None:
        self.thermal_bar.set_turn_metrics(metrics)

    def update_turn_metrics(self, metrics: LLMTurnStats | None) -> None:
        self.thermal_bar.update_turn_metrics(metrics)

    def set_metrics_source(
        self,
        source: Callable[[], LLMTurnStats | None] | None,
    ) -> None:
        self.thermal_bar.set_metrics_source(source)

    @property
    def has_live_stream(self) -> bool:
        """True if the Final Answer Panel is open and already responsible for the bar refresh."""
        return self._stream.has_open_live

    @property
    def stream_refresh_count(self) -> int:
        """Number of Live refreshes of the answer panel (rendering diagnostics)."""
        return self._stream.refresh_count

    def clear_terminal(self) -> None:
        if not self.console.is_terminal or not sys.stdout.isatty():
            return
        self._progress_in_progress = False
        self.console.clear()
        if self.thermal_bar.enable:
            self.thermal_bar.after_clear()
            self._draw_sticky_header()

    def print_splash(
        self,
        pause: Callable[[float], None] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        """
        Play the startup HUD (animation + greeting) while the model prewarms.

        Args:
            pause: Wait between frames. Default `time.sleep`.
            monotonic: Injectable clock. Default `time.monotonic`.

        Returns:
            None. On a non-terminal console, or if the splash fails, it does not block the session.
        """
        if not self.console.is_terminal:
            return
        try:
            run_splash(self.console, pause=pause, monotonic=monotonic)
        except Exception:
            # The splash must not prevent the technical Panel or the wizard.
            return

    def print_welcome(
        self, verbose: bool,
        full_verification: bool = False, burn_protection: bool = True,
        thinking: bool = False,
        model_label: str = "",
        folder: str = "",
    ) -> None:
        self._welcome_state = {
            "verbose": verbose,
            "full_verification": full_verification,
            "burn_protection": burn_protection,
            "thinking": thinking,
        }
        if folder:
            self._documents_folder = folder
        if model_label:
            self._model_label = model_label
        self.clear_terminal()
        self._sync()
        self.set_verbose(verbose)
        self._draw_sticky_header()
        if self._use_sticky_chrome():
            position_transcript_start(
                header=self.thermal_bar.header_active,
                is_terminal=True,
            )
        self.console.print()
        print_panel(
            self.console,
            self._welcome_body(self._welcome_state),
            "[agent.label]Assistente[/]",
            padding=(0, 1),
            sync=False,
        )
        self.console.print()

    def set_model_label(self, label: str) -> None:
        """Update the model in the sticky header without a wipe."""
        self._model_label = label
        self._draw_sticky_header()

    def read_user_request(self, model_label: str) -> str:
        self._model_label = model_label
        self.thermal_bar.set_input_in_progress(True)
        self._sync()
        if self._use_sticky_chrome():
            _, tty_lines = tty_size(self.console)
            space = sticky_composer_height(
                "",
                self.console.width,
                tty_lines,
                header=self.thermal_bar.header_active,
            )
            self.thermal_bar.make_room_below(space)
            self._draw_sticky_header()
        if self._ui_needs_redraw.is_set():
            self._ui_needs_redraw.clear()
            self._draw_sticky_header()
            self.thermal_bar.redraw(force=True)
        title_markup = f"[user.label]Tu[/] [dim]· {model_label}[/]"
        self._current_prompt_title = title_markup
        sticky = self._use_sticky_chrome()
        if not sticky:
            self.console.print()
            print_composer(
                self.console,
                title_markup,
                Text(""),
                sync=False,
            )
        self._drawn_input_width = self.console.width
        cursor_in_box = self.console.is_terminal and termios_available()
        try:
            if cursor_in_box:
                first_line = read_line_with_reflow(
                    self.console,
                    title_markup,
                    resize_event=self._ui_needs_redraw,
                    on_resize=lambda text, anchor, cursor: self._reprint_chrome(
                        input_text=text,
                        anchor=anchor,
                        cursor=cursor,
                    ),
                    set_buffer=lambda t: setattr(self, "_current_input_buffer", t),
                    on_draw=(
                        self._draw_sticky_composer if sticky else None
                    ),
                )
            else:
                first_line = input("› ")
            text = collect_multiline_input(first_line, sys.stdin)
        except (EOFError, KeyboardInterrupt):
            if cursor_in_box:
                if sticky:
                    self._close_sticky_composer()
                else:
                    visual = visual_content_lines(
                        composer_line(self._current_input_buffer),
                        self.console.width,
                    )
                    sys.stdout.write(f"\033[{max(1, visual)}B\r")
                    sys.stdout.flush()
            self.thermal_bar.set_input_in_progress(False)
            self.thermal_bar.redraw(force=True)
            raise

        if cursor_in_box:
            if sticky:
                self._close_sticky_composer()
            else:
                clear_lines_above(
                    cuu_clear_composer(text, self.console.width),
                    is_terminal=self.console.is_terminal,
                )
        self._sync()
        print_user_message(
            self.console,
            title_markup,
            text,
            sync=False,
        )
        self._drawn_input_width = self.console.width
        self.thermal_bar.set_input_in_progress(False)
        self._draw_sticky_header()
        self.thermal_bar.redraw(force=True)
        return text

    def close_progress(self) -> None:
        """Close the live line (e.g. «Comprendo la richiesta») before the next output."""
        self._finalize_progress()

    def _finalize_progress(self) -> None:
        """Close the pull's live line: the last state stays visible below."""
        if not self._progress_in_progress:
            return
        self.console.file.write("\n")
        self.console.file.flush()
        self._progress_in_progress = False

    def _print(self, *args, **kwargs) -> None:
        self._finalize_progress()
        self._sync()
        self.console.print(*args, **kwargs)
        self.thermal_bar.redraw()

    def print_toggle(self, name: str, active: bool) -> None:
        if name == "Verbose":
            self.set_verbose(active)
        elif self._welcome_state is not None:
            key = _STATE_KEY_FROM_LABEL.get(name)
            if key is not None:
                self._welcome_state[key] = active
        style = "badge.on" if active else "badge.off"
        self._print(f"[{style}]{name}: {'attivo' if active else 'disattivo'}[/]")
        self._draw_sticky_header()

    def print_error(self, message: str) -> None:
        self._finalize_progress()
        self._sync()
        print_panel(
            self.console,
            Text(message, style="error"),
            "Errore",
            border_style="red",
            padding=(0, 2),
            sync=False,
        )
        self.thermal_bar.redraw()

    def print_fallback(self, written_paths: list[str]) -> None:
        self._finalize_progress()
        self._sync()
        lines = "\n".join(f"- `{p}`" for p in written_paths)
        print_panel(
            self.console,
            Markdown(
                "File scritti automaticamente perché l'agente non ha usato i tool:\n\n"
                f"{lines}"
            ),
            "[warning]Fallback[/]",
            border_style="yellow",
            padding=(1, 2),
            sync=False,
        )
        self.thermal_bar.redraw()

    def print_system_message(self, message: str) -> None:
        # Escaping is mandatory: Ollama/httpx errors may contain [Errno N] or [/]
        # which Rich reads as markup, crashing the command loop.
        self._print(f"[system]{escape(message)}[/]")

    def ask_confirmation(self, question: str) -> bool:
        """
        Ask for explicit yes/no consent.

        Empty Enter, EOF and any value other than s/si/sì/y/yes count as no.
        """
        answer = self._read_wizard_line(f"{question} [s/N]").casefold()
        return answer in {"s", "si", "sì", "y", "yes"}

    def ask_download_confirmation(self, question: str) -> bool:
        """
        Quick consent to the pull: empty Enter and Y count as yes.

        n/no cancel. Any other non-empty text counts as no.
        """
        answer = self._read_wizard_line(f"{question} [Y/n]").casefold()
        if answer in {"n", "no"}:
            return False
        return answer in {"", "y", "yes", "s", "si", "sì"}

    def ask_text(self, question: str) -> str:
        """Read a free-form line (model selection, indexes or aliases)."""
        return self._read_wizard_line(question)

    def update_progress(self, message: str) -> None:
        """
        Update the progress line in place (install or pull).

        On a TTY the percentage stays on the same line (EL + CR). Outside a TTY
        each state goes on a new line, for capturable logs and tests.
        """
        if not self.console.is_terminal:
            self.print_system_message(message)
            return
        self._sync()
        width = max(1, self.console.width - 1)
        visible = truncate_visible(message, width)
        destination = self.console.file
        destination.write(f"{_ANSI['clear_line']}\r")
        destination.flush()
        self.console.print(f"[system]{escape(visible)}[/]", end="")
        self._progress_in_progress = True
        self.thermal_bar.redraw()

    def _read_wizard_line(self, question: str) -> str:
        """Shared wizard prompt; EOF = empty string, KeyboardInterrupt propagates."""
        self.print_system_message(question)
        try:
            return input("› ").strip()
        except EOFError:
            return ""

    def print_thermal_warning(self, message: str) -> None:
        self._print(f"[warning]Termico: {escape(message)}[/]")

    def print_separator(self) -> None:
        self._print(Rule(style="dim"))
