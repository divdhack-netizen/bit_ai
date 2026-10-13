"""Lettura riga TTY: I/O cbreak, clipboard e reprint del composer senza box."""

from __future__ import annotations

import os
import select
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable

from rich.console import Console

from assistente.editor_riga import (
    DecoderTasti,
    EditorRiga,
    EventoTasto,
    normalizza_incolla,
    testo_con_selezione,
)
from assistente.tty_layout import (
    cancella_righe_sopra,
    cuu_clear_composer,
    posiziona_cursore_composer,
    posiziona_cursore_nel_composer,
    stampa_composer,
)

try:
    import termios
    import tty
except ImportError:  # pragma: no cover - Windows
    termios = None  # type: ignore[assignment]
    tty = None  # type: ignore[assignment]

_INTERVALLO_POLL_RESIZE_SEC = 0.05
_TIMEOUT_ESCAPE_SEC = 0.05
_BURST_STDIN_BYTES = 4096
_BRACKETED_PASTE_ON = "\033[?2004h"
_BRACKETED_PASTE_OFF = "\033[?2004l"
_DECCKM_OFF = "\033[?1l"
_MODIFY_OTHER_KEYS_ON = "\033[>4;2m"
_MODIFY_OTHER_KEYS_OFF = "\033[>4;0m"
_CLIPBOARD_TIMEOUT_SEC = 1.0
_MODO_TERMINALE = (
    _BRACKETED_PASTE_ON + _DECCKM_OFF + _MODIFY_OTHER_KEYS_ON
)
_MODO_TERMINALE_OFF = _BRACKETED_PASTE_OFF + _MODIFY_OTHER_KEYS_OFF

_kill_ring = ""


def termios_disponibile() -> bool:
    """True se stdin/stdout TTY supportano input cbreak."""
    return (
        termios is not None
        and tty is not None
        and sys.stdin.isatty()
        and sys.stdout.isatty()
    )


def azione_ctrl_c(editor: EditorRiga) -> str:
    """Ctrl+C: copia solo con selezione esplicita, altrimenti interrompe."""
    if editor.intervallo_selezione() is not None:
        return "copy"
    return "interrupt"


def _ristampa_composer(
    console: Console,
    editor: EditorRiga,
    *,
    titolo_markup: str,
    testo_prima: str,
    is_tty: bool,
    larghezza_cancellazione: int | None = None,
) -> None:
    """Unico percorso visivo: clear + composer + cursore all'indice."""
    testo = editor.testo()
    larghezza_erase = (
        larghezza_cancellazione if larghezza_cancellazione is not None else console.width
    )
    cancella_righe_sopra(
        cuu_clear_composer(testo_prima, larghezza_erase),
        is_terminal=is_tty,
    )
    stampa_composer(
        console,
        titolo_markup,
        testo_con_selezione(testo, editor.ancora, editor.cursore),
        sincronizza=False,
    )
    posiziona_cursore_composer(
        testo,
        editor.cursore,
        console.width,
        is_terminal=is_tty,
    )


def _comandi_clipboard() -> tuple[list[str], list[str]]:
    if sys.platform == "darwin":
        return ["pbcopy"], ["pbpaste"]
    if shutil.which("wl-copy"):
        return ["wl-copy"], ["wl-paste"]
    return (
        ["xclip", "-selection", "clipboard"],
        ["xclip", "-selection", "clipboard", "-o"],
    )


def _esegui_clipboard(comando: list[str], *, stdin: str | None = None) -> str:
    try:
        risultato = subprocess.run(
            comando,
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
    if risultato.returncode == 0 and risultato.stdout:
        return risultato.stdout
    return ""


def scrivi_clipboard(testo: str) -> None:
    """Copia negli appunti di sistema e nell'anello locale."""
    global _kill_ring
    if not testo:
        return
    _kill_ring = testo
    scrive, _legge = _comandi_clipboard()
    _esegui_clipboard(scrive, stdin=testo)


def leggi_clipboard() -> str:
    """Testo dagli appunti, altrimenti l'ultimo taglia/copia locale."""
    _scrive, legge = _comandi_clipboard()
    letto = _esegui_clipboard(legge)
    if letto:
        return normalizza_incolla(letto)
    return normalizza_incolla(_kill_ring)


def _leggi_burst(fd: int) -> bytes:
    try:
        return os.read(fd, _BURST_STDIN_BYTES)
    except OSError:
        return b""


def leggi_riga_con_reflow(
    console: Console,
    titolo_markup: str,
    *,
    resize_event: threading.Event,
    on_resize: Callable[[str, int | None, int], None],
    set_buffer: Callable[[str], None],
    on_disegna: Callable[[EditorRiga], None] | None = None,
) -> str:
    """
    Legge una riga in cbreak; su resize chiama `on_resize` sul main thread.

    Raises:
        RuntimeError: termios non disponibile.
        EOFError / KeyboardInterrupt: come input nativo.
    """
    if termios is None or tty is None:
        raise RuntimeError("termios non disponibile")

    fd = sys.stdin.fileno()
    old_attr = termios.tcgetattr(fd)
    editor = EditorRiga()
    decoder = DecoderTasti()
    resize_event.clear()
    set_buffer("")
    is_tty = console.is_terminal and sys.stdout.isatty()

    def sincronizza() -> None:
        set_buffer(editor.testo())

    def disegna(testo_prima: str) -> None:
        sincronizza()
        if on_disegna is not None:
            on_disegna(editor)
            return
        _ristampa_composer(
            console,
            editor,
            titolo_markup=titolo_markup,
            testo_prima=testo_prima,
            is_tty=is_tty,
        )

    def applica_ridisegno() -> None:
        sincronizza()
        on_resize(editor.testo(), editor.ancora, editor.cursore)
        if on_disegna is None:
            posiziona_cursore_composer(
                editor.testo(),
                editor.cursore,
                console.width,
                is_terminal=is_tty,
            )

    def gestisci(evento: EventoTasto) -> str | None:
        prima = editor.testo()
        segnale = editor.applica(evento)
        if segnale == "enter":
            return "enter"
        if segnale == "eof":
            raise EOFError
        if segnale == "copy":
            if azione_ctrl_c(editor) == "interrupt":
                raise KeyboardInterrupt
            pezzo = editor.testo_clipboard()
            if pezzo:
                scrivi_clipboard(pezzo)
            return None
        if segnale == "cut":
            pezzo = editor.taglia()
            if pezzo:
                scrivi_clipboard(pezzo)
            disegna(prima)
            return None
        if segnale == "paste":
            editor.inserisci(leggi_clipboard())
            disegna(prima)
            return None
        if segnale == "interrupt":
            raise KeyboardInterrupt
        disegna(prima)
        return None

    try:
        tty.setcbreak(fd)
        if is_tty:
            sys.stdout.write(_MODO_TERMINALE)
            sys.stdout.flush()
        if on_disegna is not None:
            on_disegna(editor)
        else:
            posiziona_cursore_nel_composer(is_terminal=is_tty)
        while True:
            if resize_event.is_set():
                resize_event.clear()
                applica_ridisegno()
            pronto, _, _ = select.select(
                [fd], [], [], _INTERVALLO_POLL_RESIZE_SEC,
            )
            if not pronto:
                continue
            data = _leggi_burst(fd)
            if not data:
                raise EOFError
            eventi = decoder.feed(data)
            while decoder.attesa():
                attesa_sec = 0.5 if decoder.in_paste else _TIMEOUT_ESCAPE_SEC
                extra, _, _ = select.select([fd], [], [], attesa_sec)
                if extra:
                    altri = _leggi_burst(fd)
                    if not altri:
                        eventi.extend(decoder.chiudi_attesa())
                        break
                    eventi.extend(decoder.feed(altri))
                else:
                    eventi.extend(decoder.chiudi_attesa())
                    break
            for evento in eventi:
                if gestisci(evento) == "enter":
                    return editor.testo()
    finally:
        if is_tty:
            sys.stdout.write(_MODO_TERMINALE_OFF)
            sys.stdout.flush()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)
        set_buffer("")
