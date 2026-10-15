"""Line editor and key decoder: pure state, no TTY."""

from __future__ import annotations

from assistant.line_editor import (
    KeyDecoder,
    LineEditor,
    KeyEvent,
    right_word_index,
    left_word_index,
    escape_interpretation,
    text_with_selection as highlight_selection,
)

import pytest

pytestmark = pytest.mark.unit


def test_insert_in_middle_and_backspace() -> None:
    editor = LineEditor()
    editor.insert("ciao")
    editor.cursor = 2
    editor.insert("X")
    assert editor.text() == "ciXao"
    assert editor.cursor == 3
    editor.delete_left()
    assert editor.text() == "ciao"
    assert editor.cursor == 2


def test_delete_right_and_eof_on_text() -> None:
    editor = LineEditor()
    editor.insert("ab")
    editor.cursor = 0
    editor.delete_right()
    assert editor.text() == "b"
    assert editor.apply(KeyEvent("eof")) is None
    assert editor.text() == ""
    assert editor.apply(KeyEvent("eof")) == "eof"


def test_word_jump() -> None:
    buffer = list("ciao mondo qui")
    assert left_word_index(buffer, 14) == 11
    assert left_word_index(buffer, 11) == 5
    assert right_word_index(buffer, 0) == 5
    assert right_word_index(buffer, 5) == 11
    editor = LineEditor()
    editor.insert("ciao mondo qui")
    editor.move("left", alt=True)
    assert editor.cursor == 11
    editor.move("left", alt=True)
    assert editor.cursor == 5


def test_shift_selection_and_toggle() -> None:
    editor = LineEditor()
    editor.insert("abcdef")
    editor.move("left", shift=True)
    editor.move("left", shift=True)
    assert editor.selection_range() == (4, 6)
    assert editor.clipboard_text() == "ef"
    editor.move("right")
    assert editor.selection_range() is None
    assert editor.cursor == 6
    editor.toggle_selection()
    editor.move("left")
    editor.move("left")
    assert editor.clipboard_text() == "ef"
    editor.close_selection()
    assert editor.selection_range() is None


def test_cut_selection_vs_line() -> None:
    editor = LineEditor()
    editor.insert("ciao mondo")
    editor.cursor = 5
    editor.move("right", shift=True, alt=True)
    assert editor.clipboard_text() == "mondo"
    assert editor.cut() == "mondo"
    assert editor.text() == "ciao "
    editor.insert("tutto")
    assert editor.cut() == "ciao tutto"
    assert editor.text() == ""


def test_insert_replaces_selection() -> None:
    editor = LineEditor()
    editor.insert("abcde")
    editor.cursor = 1
    editor.move("right", shift=True)
    editor.move("right", shift=True)
    editor.insert("X")
    assert editor.text() == "aXde"


def test_plain_text_with_selection() -> None:
    highlighted = highlight_selection("ciao mondo", 5, 10)
    assert highlighted.plain == "ciao mondo"


def test_arrow_interpretation_csi_ss3_kitty() -> None:
    assert escape_interpretation("[D").action == "left"
    assert escape_interpretation("OD").action == "left"
    assert escape_interpretation("[1;5D").action == "left"
    assert escape_interpretation("[1;5D").alt is True
    assert escape_interpretation("[C").action == "right"
    assert escape_interpretation("OC").action == "right"
    assert escape_interpretation("[3~").action == "delete"
    rxvt = escape_interpretation("[d")
    assert rxvt is not None and rxvt.shift is True
    kitty = escape_interpretation("[57350;2u")
    assert kitty is not None and kitty.action == "left" and kitty.shift is True
    shift_left = escape_interpretation("[1;2D")
    assert shift_left is not None and shift_left.shift is True
    option_left = escape_interpretation("[1;3D")
    assert option_left is not None and option_left.alt is True
    option_shift = escape_interpretation("[1;4C")
    assert option_shift is not None and option_shift.alt and option_shift.shift
    meta_b = escape_interpretation("b")
    assert meta_b is not None and meta_b.alt is True and meta_b.action == "left"


def test_decoder_burst_glyphs_and_arrows() -> None:
    decoder = KeyDecoder()
    glyphs = decoder.feed(b"ciao")
    assert [e.text for e in glyphs] == list("ciao")
    left = decoder.feed(b"\x1b[D")
    assert left[0].action == "left"
    shift = decoder.feed(b"\x1b[1;2D")
    assert shift[0].shift is True
    option = decoder.feed(b"\x1b[1;3C")
    assert option[0].alt is True and option[0].action == "right"


def test_decoder_bracketed_paste() -> None:
    decoder = KeyDecoder()
    events = decoder.feed(b"\x1b[200~ciao\nmondo\x1b[201~")
    assert len(events) == 1
    assert events[0].action == "insert"
    assert events[0].text == "ciao mondo"


def test_decoder_bare_esc_after_timeout() -> None:
    decoder = KeyDecoder()
    assert decoder.feed(b"\x1b") == []
    assert decoder.wait() is True
    closed = decoder.close_wait()
    assert closed[0].action == "escape"


def test_decoder_ctrl_w_and_enter() -> None:
    decoder = KeyDecoder()
    assert decoder.feed(b"\x17")[0].action == "toggle_selection"
    assert decoder.feed(b"\n")[0].action == "enter"
    assert decoder.feed(b"\x03")[0].action == "copy"


def test_ctrl_c_interrupts_without_selection() -> None:
    """Ctrl+C without an explicit selection is an interrupt, even with text present."""
    from assistant.prompt_tty import ctrl_c_action

    editor = LineEditor()
    editor.insert("ciao")
    assert ctrl_c_action(editor) == "interrupt"
    editor.cursor = 4
    editor.move("left", shift=True)
    editor.move("left", shift=True)
    assert editor.selection_range() == (2, 4)
    assert ctrl_c_action(editor) == "copy"
