"""Editor di riga e decoder tasti: stato puro, senza TTY."""

from __future__ import annotations

from assistente.editor_riga import (
    DecoderTasti,
    EditorRiga,
    EventoTasto,
    indice_parola_destra,
    indice_parola_sinistra,
    interpretazione_escape,
    testo_con_selezione as evidenzia_selezione,
)

import pytest

pytestmark = pytest.mark.unit


def test_inserisci_in_mezzo_e_backspace() -> None:
    editor = EditorRiga()
    editor.inserisci("ciao")
    editor.cursore = 2
    editor.inserisci("X")
    assert editor.testo() == "ciXao"
    assert editor.cursore == 3
    editor.cancella_sinistra()
    assert editor.testo() == "ciao"
    assert editor.cursore == 2


def test_cancella_destra_e_eof_su_testo() -> None:
    editor = EditorRiga()
    editor.inserisci("ab")
    editor.cursore = 0
    editor.cancella_destra()
    assert editor.testo() == "b"
    assert editor.applica(EventoTasto("eof")) is None
    assert editor.testo() == ""
    assert editor.applica(EventoTasto("eof")) == "eof"


def test_salto_parola() -> None:
    buffer = list("ciao mondo qui")
    assert indice_parola_sinistra(buffer, 14) == 11
    assert indice_parola_sinistra(buffer, 11) == 5
    assert indice_parola_destra(buffer, 0) == 5
    assert indice_parola_destra(buffer, 5) == 11
    editor = EditorRiga()
    editor.inserisci("ciao mondo qui")
    editor.muovi("left", alt=True)
    assert editor.cursore == 11
    editor.muovi("left", alt=True)
    assert editor.cursore == 5


def test_selezione_shift_e_toggle() -> None:
    editor = EditorRiga()
    editor.inserisci("abcdef")
    editor.muovi("left", shift=True)
    editor.muovi("left", shift=True)
    assert editor.intervallo_selezione() == (4, 6)
    assert editor.testo_clipboard() == "ef"
    editor.muovi("right")
    assert editor.intervallo_selezione() is None
    assert editor.cursore == 6
    editor.toggle_selezione()
    editor.muovi("left")
    editor.muovi("left")
    assert editor.testo_clipboard() == "ef"
    editor.chiudi_selezione()
    assert editor.intervallo_selezione() is None


def test_taglia_selezione_vs_riga() -> None:
    editor = EditorRiga()
    editor.inserisci("ciao mondo")
    editor.cursore = 5
    editor.muovi("right", shift=True, alt=True)
    assert editor.testo_clipboard() == "mondo"
    assert editor.taglia() == "mondo"
    assert editor.testo() == "ciao "
    editor.inserisci("tutto")
    assert editor.taglia() == "ciao tutto"
    assert editor.testo() == ""


def test_inserisci_sostituisce_selezione() -> None:
    editor = EditorRiga()
    editor.inserisci("abcde")
    editor.cursore = 1
    editor.muovi("right", shift=True)
    editor.muovi("right", shift=True)
    editor.inserisci("X")
    assert editor.testo() == "aXde"


def test_testo_con_selezione_plain() -> None:
    evidenziato = evidenzia_selezione("ciao mondo", 5, 10)
    assert evidenziato.plain == "ciao mondo"


def test_interpretazione_frecce_csi_ss3_kitty() -> None:
    assert interpretazione_escape("[D").azione == "left"
    assert interpretazione_escape("OD").azione == "left"
    assert interpretazione_escape("[1;5D").azione == "left"
    assert interpretazione_escape("[1;5D").alt is True
    assert interpretazione_escape("[C").azione == "right"
    assert interpretazione_escape("OC").azione == "right"
    assert interpretazione_escape("[3~").azione == "delete"
    rxvt = interpretazione_escape("[d")
    assert rxvt is not None and rxvt.shift is True
    kitty = interpretazione_escape("[57350;2u")
    assert kitty is not None and kitty.azione == "left" and kitty.shift is True
    shift_left = interpretazione_escape("[1;2D")
    assert shift_left is not None and shift_left.shift is True
    option_left = interpretazione_escape("[1;3D")
    assert option_left is not None and option_left.alt is True
    option_shift = interpretazione_escape("[1;4C")
    assert option_shift is not None and option_shift.alt and option_shift.shift
    meta_b = interpretazione_escape("b")
    assert meta_b is not None and meta_b.alt is True and meta_b.azione == "left"


def test_decoder_burst_glifi_e_frecce() -> None:
    decoder = DecoderTasti()
    glifi = decoder.feed(b"ciao")
    assert [e.testo for e in glifi] == list("ciao")
    sinistra = decoder.feed(b"\x1b[D")
    assert sinistra[0].azione == "left"
    shift = decoder.feed(b"\x1b[1;2D")
    assert shift[0].shift is True
    option = decoder.feed(b"\x1b[1;3C")
    assert option[0].alt is True and option[0].azione == "right"


def test_decoder_bracketed_paste() -> None:
    decoder = DecoderTasti()
    eventi = decoder.feed(b"\x1b[200~ciao\nmondo\x1b[201~")
    assert len(eventi) == 1
    assert eventi[0].azione == "insert"
    assert eventi[0].testo == "ciao mondo"


def test_decoder_esc_nudo_dopo_timeout() -> None:
    decoder = DecoderTasti()
    assert decoder.feed(b"\x1b") == []
    assert decoder.attesa() is True
    chiusi = decoder.chiudi_attesa()
    assert chiusi[0].azione == "escape"


def test_decoder_ctrl_w_e_enter() -> None:
    decoder = DecoderTasti()
    assert decoder.feed(b"\x17")[0].azione == "toggle_selezione"
    assert decoder.feed(b"\n")[0].azione == "enter"
    assert decoder.feed(b"\x03")[0].azione == "copy"


def test_ctrl_c_interrompe_senza_selezione() -> None:
    """Ctrl+C senza selezione esplicita è interrupt, anche se c'è testo."""
    from assistente.prompt_tty import azione_ctrl_c

    editor = EditorRiga()
    editor.inserisci("ciao")
    assert azione_ctrl_c(editor) == "interrupt"
    editor.cursore = 4
    editor.muovi("left", shift=True)
    editor.muovi("left", shift=True)
    assert editor.intervallo_selezione() == (2, 4)
    assert azione_ctrl_c(editor) == "copy"
