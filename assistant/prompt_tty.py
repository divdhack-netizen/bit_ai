"""TTY line input: cbreak I/O, clipboard and composer reprint without a box."""

from __future__ import annotations

import os
import select
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable

from rich.console import Console

from assistant.line_editor import (
    KeyDecoder,
    LineEditor,
    KeyEvent,
    normalize_paste,
    text_with_selection,
)
from assistant.tty_layout import (
    clear_lines_above,
    cuu_clear_composer,
    position_composer_cursor,
    position_cursor_in_composer,
    print_composer,
)

try:
    import termios
    import tty
except ImportError:  # pragma: no cover - Windows
    termios = None  # type: ignore[assignment]
    tty = None  # type: ignore[assignment]

_RESIZE_POLL_INTERVAL_SEC = 0.05
_TIMEOUT_ESCAPE_SEC = 0.05
_BURST_STDIN_BYTES = 4096
_BRACKETED_PASTE_ON = "\033[?2004h"
_BRACKETED_PASTE_OFF = "\033[?2004l"
_DECCKM_OFF = "\033[?1l"
_MODIFY_OTHER_KEYS_ON = "\033[>4;2m"
_MODIFY_OTHER_KEYS_OFF = "\033[>4;0m"
_CLIPBOARD_TIMEOUT_SEC = 1.0
_TERMINAL_MODE = (
    _BRACKETED_PASTE_ON + _DECCKM_OFF + _MODIFY_OTHER_KEYS_ON
)
_TERMINAL_MODE_OFF = _BRACKETED_PASTE_OFF + _MODIFY_OTHER_KEYS_OFF

_kill_ring = ""


def termios_available() -> bool:
    """True if the stdin/stdout TTYs support cbreak input."""
    return (
        termios is not None
        and tty is not None
        and sys.stdin.isatty()
        and sys.stdout.isatty()
    )


def ctrl_c_action(editor: LineEditor) -> str:
    """Ctrl+C: copy only with an explicit selection, otherwise interrupt."""
    if editor.selection_range() is not None:
        return "copy"
    return "interrupt"


def _reprint_composer(
    console: Console,
    editor: LineEditor,
    *,
    title_markup: str,
    text_before: str,
    is_tty: bool,
    erase_width: int | None = None,
) -> None:
    """Single visual path: clear + composer + cursor at the index."""
    text = editor.text()
    effective_erase_width = (
        erase_width if erase_width is not None else console.width
    )
    clear_lines_above(
        cuu_clear_composer(text_before, effective_erase_width),
        is_terminal=is_tty,
    )
    print_composer(
        console,
        title_markup,
        text_with_selection(text, editor.anchor, editor.cursor),
        sync=False,
    )
    position_composer_cursor(
        text,
        editor.cursor,
        console.width,
        is_terminal=is_tty,
    )


def _clipboard_commands() -> tuple[list[str], list[str]]:
    if sys.platform == "darwin":
        return ["pbcopy"], ["pbpaste"]
    if shutil.which("wl-copy"):
        return ["wl-copy"], ["wl-paste"]
    return (
        ["xclip", "-selection", "clipboard"],
        ["xclip", "-selection", "clipboard", "-o"],
    )


def _run_clipboard(command: list[str], *, stdin: str | None = None) -> str:
    try:
        result = subprocess.run(
            command,
            input=stdin,
            capture_output=stdin is None,
            text=True,
            timeout=_CLIPBOARD_TIMEOUT_SEC,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""
    if stdin is not None:
        return ""
    if result.returncode == 0 and result.stdout:
        return result.stdout
    return ""


def write_clipboard(text: str) -> None:
    """Copy to the system clipboard and to the local kill ring."""
    global _kill_ring
    if not text:
        return
    _kill_ring = text
    writes, _reads = _clipboard_commands()
    _run_clipboard(writes, stdin=text)


def read_clipboard() -> str:
    """Text from the clipboard, otherwise the last local cut/copy."""
    _writes, reads = _clipboard_commands()
    pasted = _run_clipboard(reads)
    if pasted:
        return normalize_paste(pasted)
    return normalize_paste(_kill_ring)


def _read_burst(fd: int) -> bytes:
    try:
        return os.read(fd, _BURST_STDIN_BYTES)
    except OSError:
        return b""


def read_line_with_reflow(
    console: Console,
    title_markup: str,
    *,
    resize_event: threading.Event,
    on_resize: Callable[[str, int | None, int], None],
    set_buffer: Callable[[str], None],
    on_draw: Callable[[LineEditor], None] | None = None,
) -> str:
    """
    Read a line in cbreak mode; on resize call `on_resize` on the main thread.

    Raises:
        RuntimeError: termios not available.
        EOFError / KeyboardInterrupt: same as native input.
    """
    if termios is None or tty is None:
        raise RuntimeError("termios non disponibile")

    fd = sys.stdin.fileno()
    old_attr = termios.tcgetattr(fd)
    editor = LineEditor()
    decoder = KeyDecoder()
    resize_event.clear()
    set_buffer("")
    is_tty = console.is_terminal and sys.stdout.isatty()

    def sync() -> None:
        set_buffer(editor.text())

    def draw(text_before: str) -> None:
        sync()
        if on_draw is not None:
            on_draw(editor)
            return
        _reprint_composer(
            console,
            editor,
            title_markup=title_markup,
            text_before=text_before,
            is_tty=is_tty,
        )

    def apply_redraw() -> None:
        sync()
        on_resize(editor.text(), editor.anchor, editor.cursor)
        if on_draw is None:
            position_composer_cursor(
                editor.text(),
                editor.cursor,
                console.width,
                is_terminal=is_tty,
            )

    def handle(event: KeyEvent) -> str | None:
        before = editor.text()
        editor_signal = editor.apply(event)
        if editor_signal == "enter":
            return "enter"
        if editor_signal == "eof":
            raise EOFError
        if editor_signal == "copy":
            if ctrl_c_action(editor) == "interrupt":
                raise KeyboardInterrupt
            piece = editor.clipboard_text()
            if piece:
                write_clipboard(piece)
            return None
        if editor_signal == "cut":
            piece = editor.cut()
            if piece:
                write_clipboard(piece)
            draw(before)
            return None
        if editor_signal == "paste":
            editor.insert(read_clipboard())
            draw(before)
            return None
        if editor_signal == "interrupt":
            raise KeyboardInterrupt
        draw(before)
        return None

    try:
        tty.setcbreak(fd)
        if is_tty:
            sys.stdout.write(_TERMINAL_MODE)
            sys.stdout.flush()
        if on_draw is not None:
            on_draw(editor)
        else:
            position_cursor_in_composer(is_terminal=is_tty)
        while True:
            if resize_event.is_set():
                resize_event.clear()
                apply_redraw()
            ready, _, _ = select.select(
                [fd], [], [], _RESIZE_POLL_INTERVAL_SEC,
            )
            if not ready:
                continue
            data = _read_burst(fd)
            if not data:
                raise EOFError
            events = decoder.feed(data)
            while decoder.wait():
                wait_sec = 0.5 if decoder.in_paste else _TIMEOUT_ESCAPE_SEC
                extra, _, _ = select.select([fd], [], [], wait_sec)
                if extra:
                    others = _read_burst(fd)
                    if not others:
                        events.extend(decoder.close_wait())
                        break
                    events.extend(decoder.feed(others))
                else:
                    events.extend(decoder.close_wait())
                    break
            for event in events:
                if handle(event) == "enter":
                    return editor.text()
    finally:
        if is_tty:
            sys.stdout.write(_TERMINAL_MODE_OFF)
            sys.stdout.flush()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)
        set_buffer("")
