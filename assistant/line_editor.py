"""Line editor: pure state, key decoder, UI guide. No TTY I/O."""

from __future__ import annotations

from dataclasses import dataclass

from rich.text import Text

SELECTION_STYLE = "bold black on cyan"

KEYBOARD_GUIDE_LINES = (
    "  Cursore:  frecce ← →  ·  Option+frecce = parole  ·  Ctrl+A / Ctrl+E = inizio/fine",
    "  Selezione: Ctrl+W o Ctrl+G, poi frecce (o Option+frecce); Esc esce",
    "            Shift+frecce se il terminale le inoltra  ·  Option+Shift = parole",
    "  Appunti:  Ctrl+C copia solo la selezione  ·  Ctrl+X taglia  ·  Ctrl+V incolla",
    "            Ctrl+C senza selezione interrompe sempre il prompt",
)

_FINAL_ACTIONS = {
    "D": "left",
    "C": "right",
    "A": "home",
    "B": "end",
    "H": "home",
    "F": "end",
    "d": "left",
    "c": "right",
}
_SS3_ACTIONS = {"D": "left", "C": "right", "A": "home", "B": "end"}
_TILDE_ACTIONS = {
    "3": "delete",
    "1": "home",
    "7": "home",
    "4": "end",
    "8": "end",
}
_KITTY_ACTIONS = {
    "57350": "left",
    "57351": "right",
    "57352": "home",
    "57353": "end",
}
_CTRL = {
    "\x01": "home",
    "\x03": "copy",
    "\x04": "eof",
    "\x05": "end",
    "\x07": "toggle_selection",
    "\x08": "backspace",
    "\x0d": "enter",
    "\x0a": "enter",
    "\x16": "paste",
    "\x17": "toggle_selection",
    "\x18": "cut",
    "\x7f": "backspace",
}
_PASTE_TERMINATOR = b"\x1b[201~"
_PASTE_START = b"[200~"


@dataclass(frozen=True)
class KeyEvent:
    """Logical key: movement, editing, or `text` to insert."""

    action: str
    shift: bool = False
    alt: bool = False
    text: str = ""


def _is_word_glyph(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def left_word_index(buffer: list[str], cursor: int) -> int:
    """Start of the previous or current word (Option+←)."""
    i = max(0, min(cursor, len(buffer)))
    if i == 0:
        return 0
    i -= 1
    while i > 0 and not _is_word_glyph(buffer[i]):
        i -= 1
    while i > 0 and _is_word_glyph(buffer[i - 1]):
        i -= 1
    return i


def right_word_index(buffer: list[str], cursor: int) -> int:
    """End of the current word / start of the next one (Option+→)."""
    i = max(0, min(cursor, len(buffer)))
    n = len(buffer)
    while i < n and _is_word_glyph(buffer[i]):
        i += 1
    while i < n and not _is_word_glyph(buffer[i]):
        i += 1
    return i


def normalize_paste(text: str) -> str:
    """Single line: newline → space, control characters dropped."""
    pieces: list[str] = []
    for ch in text.replace("\r\n", "\n").replace("\r", "\n"):
        if ch == "\n":
            pieces.append(" ")
        elif ch.isprintable():
            pieces.append(ch)
    return "".join(pieces)


def _csi_modifiers(seq: str) -> tuple[bool, bool]:
    if not seq.startswith("[") or ";" not in seq:
        return False, False
    parts = seq[1:-1].split(";")
    if len(parts) < 2:
        return False, False
    raw = parts[1].split(":")[0]
    if not raw.isdigit():
        return False, False
    bits = max(0, int(raw) - 1)
    return bool(bits & 1), bool(bits & 2) or bool(bits & 4)


def escape_interpretation(seq: str) -> KeyEvent | None:
    """Translate CSI/SS3/Meta-b/f/CSI u into an event. The decoder handles `[200~`."""
    if not seq:
        return None
    if seq == "b":
        return KeyEvent("left", alt=True)
    if seq == "f":
        return KeyEvent("right", alt=True)
    if seq == "[200~":
        return KeyEvent("paste_start")
    shift, alt = _csi_modifiers(seq)
    final = seq[-1]
    action = ""
    if seq.startswith("O") and len(seq) == 2:
        action = _SS3_ACTIONS.get(seq[1], "")
    elif final == "u" and seq.startswith("["):
        action = _KITTY_ACTIONS.get(seq[1:-1].split(";")[0], "")
    elif final == "~":
        action = _TILDE_ACTIONS.get(seq[1:-1].split(";")[0], "")
    elif seq.startswith("["):
        action = _FINAL_ACTIONS.get(final, "")
        if final in ("d", "c"):
            shift = True
    if not action:
        return None
    return KeyEvent(action, shift=shift, alt=alt)


def text_with_selection(text: str, anchor: int | None, cursor: int) -> Text:
    """Rich Panel body with the anchor→cursor range highlighted."""
    if not text:
        return Text(" ")
    if anchor is None or anchor == cursor:
        return Text(text)
    start, end = sorted((anchor, cursor))
    start = max(0, min(start, len(text)))
    end = max(0, min(end, len(text)))
    body = Text()
    body.append(text[:start])
    body.append(text[start:end], style=SELECTION_STYLE)
    body.append(text[end:])
    return body


class LineEditor:
    """Buffer, cursor and selection: the single source of truth for the input line."""

    def __init__(self) -> None:
        self.glyphs: list[str] = []
        self.cursor = 0
        self.anchor: int | None = None
        self.selection_mode = False

    def text(self) -> str:
        return "".join(self.glyphs)

    def selection_range(self) -> tuple[int, int] | None:
        if self.anchor is None or self.anchor == self.cursor:
            return None
        start, end = sorted((self.anchor, self.cursor))
        return start, end

    def clipboard_text(self) -> str:
        interval = self.selection_range()
        if interval is None:
            return self.text()
        start, end = interval
        return "".join(self.glyphs[start:end])

    def _clear_selection(self) -> None:
        self.anchor = None
        self.selection_mode = False

    def delete_selection(self) -> bool:
        interval = self.selection_range()
        if interval is None:
            return False
        start, end = interval
        del self.glyphs[start:end]
        self.cursor = start
        self._clear_selection()
        return True

    def insert(self, piece: str) -> None:
        if not piece:
            return
        self.delete_selection()
        glyphs = list(piece)
        self.glyphs[self.cursor:self.cursor] = glyphs
        self.cursor += len(glyphs)
        self._clear_selection()

    def delete_left(self) -> None:
        if self.delete_selection():
            return
        if self.cursor <= 0:
            return
        self.cursor -= 1
        del self.glyphs[self.cursor]

    def delete_right(self) -> None:
        if self.delete_selection():
            return
        if self.cursor >= len(self.glyphs):
            return
        del self.glyphs[self.cursor]

    def cut(self) -> str:
        piece = self.clipboard_text()
        if not piece:
            return ""
        if self.selection_range() is not None:
            self.delete_selection()
            return piece
        self.glyphs.clear()
        self.cursor = 0
        self._clear_selection()
        return piece

    def toggle_selection(self) -> None:
        if self.selection_mode:
            self._clear_selection()
            return
        self.selection_mode = True
        self.anchor = self.cursor

    def close_selection(self) -> bool:
        if self.selection_mode or self.anchor is not None:
            self._clear_selection()
            return True
        return False

    def _destination(self, action: str, alt: bool) -> int:
        if action == "home":
            return 0
        if action == "end":
            return len(self.glyphs)
        if action == "left":
            if alt:
                return left_word_index(self.glyphs, self.cursor)
            return max(0, self.cursor - 1)
        if action == "right":
            if alt:
                return right_word_index(self.glyphs, self.cursor)
            return min(len(self.glyphs), self.cursor + 1)
        return self.cursor

    def move(self, action: str, *, shift: bool = False, alt: bool = False) -> None:
        extend = shift or self.selection_mode
        start = self.cursor
        if not extend and self.anchor is not None and self.anchor != self.cursor:
            if action == "left" and not alt:
                self.cursor = min(self.anchor, self.cursor)
            elif action == "right" and not alt:
                self.cursor = max(self.anchor, self.cursor)
            else:
                self.cursor = self._destination(action, alt)
            self.anchor = None
            return
        destination = self._destination(action, alt)
        if extend:
            if self.anchor is None:
                self.anchor = start
        else:
            self.anchor = None
        self.cursor = destination

    def apply(self, event: KeyEvent) -> str | None:
        """
        Apply an event. Return a loop signal, or the clipboard action.

        Returns:
            enter / interrupt / eof / copy / cut / paste, otherwise None.
        """
        action = event.action
        if action == "enter":
            return "enter"
        if action == "interrupt":
            return "interrupt"
        if action == "eof":
            if not self.glyphs:
                return "eof"
            self.delete_right()
            return None
        if action == "insert":
            self.insert(event.text)
            return None
        if action in ("left", "right", "home", "end"):
            self.move(action, shift=event.shift, alt=event.alt)
            return None
        if action == "backspace":
            self.delete_left()
            return None
        if action == "delete":
            self.delete_right()
            return None
        if action == "toggle_selection":
            self.toggle_selection()
            return None
        if action == "escape":
            self.close_selection()
            return None
        if action in ("copy", "cut", "paste"):
            return action
        return None


class KeyDecoder:
    """Accumulate byte bursts and produce complete `KeyEvent`s."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._paste: bytearray | None = None

    def wait(self) -> bool:
        """True if ESC/CSI or a paste is incomplete."""
        if self._paste is not None:
            return True
        return bool(self._buf) and self._buf[0] == 0x1B

    @property
    def in_paste(self) -> bool:
        return self._paste is not None

    def feed(self, data: bytes) -> list[KeyEvent]:
        if data:
            self._buf.extend(data)
        return self._extract_complete()

    def close_wait(self) -> list[KeyEvent]:
        """Timeout: close a truncated paste and a bare ESC."""
        events: list[KeyEvent] = []
        if self._paste is not None:
            text = normalize_paste(
                bytes(self._paste).decode("utf-8", errors="replace"),
            )
            self._paste = None
            if text:
                events.append(KeyEvent("insert", text=text))
        if self._buf[:1] == b"\x1b":
            if len(self._buf) == 1:
                del self._buf[0]
                events.append(KeyEvent("escape"))
            else:
                self._buf.clear()
        events.extend(self._extract_complete())
        return events

    def _extract_complete(self) -> list[KeyEvent]:
        events: list[KeyEvent] = []
        while True:
            event = self._extract_one()
            if event is None:
                break
            events.append(event)
        return events

    def _extract_one(self) -> KeyEvent | None:
        if self._paste is not None:
            return self._extract_paste()
        if not self._buf:
            return None
        if self._buf[0] == 0x1B:
            return self._extract_escape()
        ch = self._extract_utf8()
        if ch is None:
            return None
        if ch in _CTRL:
            return KeyEvent(_CTRL[ch])
        if ch.isprintable():
            return KeyEvent("insert", text=ch)
        return self._extract_one()

    def _extract_utf8(self) -> str | None:
        if not self._buf:
            return None
        lead = self._buf[0]
        if lead < 0x80:
            return chr(self._buf.pop(0))
        if lead < 0xE0:
            expected = 2
        elif lead < 0xF0:
            expected = 3
        else:
            expected = 4
        if len(self._buf) < expected:
            return None
        block = bytes(self._buf[:expected])
        del self._buf[:expected]
        return block.decode("utf-8", errors="replace")

    def _extract_escape(self) -> KeyEvent | None:
        if len(self._buf) < 2:
            return None
        second = self._buf[1]
        if second == ord("O"):
            if len(self._buf) < 3:
                return None
            seq = "O" + chr(self._buf[2])
            del self._buf[:3]
            return escape_interpretation(seq)
        if second != ord("["):
            seq = chr(second)
            del self._buf[:2]
            return escape_interpretation(seq)
        index = 2
        while index < len(self._buf):
            final = self._buf[index]
            if 0x40 <= final <= 0x7E:
                seq = self._buf[1:index + 1].decode("ascii", errors="replace")
                del self._buf[:index + 1]
                if bytes(seq, "ascii") == _PASTE_START or seq == "[200~":
                    self._paste = bytearray()
                    return self._extract_one()
                return escape_interpretation(seq)
            index += 1
        return None

    def _extract_paste(self) -> KeyEvent | None:
        assert self._paste is not None
        end = self._buf.find(_PASTE_TERMINATOR)
        if end >= 0:
            self._paste.extend(self._buf[:end])
            del self._buf[:end + len(_PASTE_TERMINATOR)]
            text = normalize_paste(
                bytes(self._paste).decode("utf-8", errors="replace"),
            )
            self._paste = None
            if text:
                return KeyEvent("insert", text=text)
            return self._extract_one()
        prefix = self._buf.find(b"\x1b")
        if prefix == -1:
            self._paste.extend(self._buf)
            self._buf.clear()
            return None
        self._paste.extend(self._buf[:prefix])
        del self._buf[:prefix]
        if _PASTE_TERMINATOR.startswith(bytes(self._buf)):
            return None
        self._paste.extend(self._buf)
        self._buf.clear()
        return None
