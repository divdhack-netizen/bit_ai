"""Test HUD splash: frame, ASCII, orologio fake, errori TTY."""

from __future__ import annotations

from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from rich.console import Console

from assistente.splash import (
    DURATA_ANIMAZIONE_SEC,
    INTERVALLO_FRAME_SEC,
    MESSAGGIO_SPLASH,
    PAUSA_DOPO_ANIMAZIONE_SEC,
    PAUSA_SALUTO_SEC,
    _supporta_unicode,
    esegui_splash,
    frame_splash,
)

pytestmark = pytest.mark.unit


class _Orologio:
    """Avanza solo quando `sleep` è chiamato: i test non aspettano il wall-clock."""

    def __init__(self) -> None:
        self.t = 0.0
        self.attese: list[float] = []

    def monotonic(self) -> float:
        return self.t

    def sleep(self, delta: float) -> None:
        self.attese.append(delta)
        self.t += delta


def _console(width: int = 80, height: int = 24) -> tuple[StringIO, Console]:
    """Console Rich in memoria per catturare l'output della splash."""
    buffer = StringIO()
    console = Console(
        file=buffer,
        width=width,
        height=height,
        force_terminal=True,
        color_system=None,
    )
    return buffer, console


def test_frame_iniziale_ha_anelli_senza_saluto() -> None:
    """A t=0 gli anelli sono visibili e il saluto non è nel HUD."""
    testo = frame_splash(0.0, width=80, height=24).plain
    assert "·" in testo or "." in testo
    assert "◉" in testo or "O" in testo
    assert "SYSTEMS ONLINE" not in testo
    assert MESSAGGIO_SPLASH not in testo


def test_frame_hud_completo_senza_saluto() -> None:
    """A fine animazione c'è SYSTEMS ONLINE, non la scritta di benvenuto."""
    testo = frame_splash(DURATA_ANIMAZIONE_SEC, width=80, height=24).plain
    assert "SYSTEMS ONLINE" in testo
    assert "INITIATING HUD" in testo
    assert MESSAGGIO_SPLASH not in testo


def test_supporta_unicode_falso_su_latin1() -> None:
    """Un encoding ristretto forza la palette ASCII."""
    console = SimpleNamespace(encoding="latin-1")
    assert _supporta_unicode(console) is False


def test_frame_ascii_senza_glifi_unicode() -> None:
    """In ascii_mode il HUD usa * / O e non i simboli Unicode."""
    testo = frame_splash(DURATA_ANIMAZIONE_SEC, 80, 24, ascii_mode=True).plain
    assert "O" in testo
    assert "*" in testo or "." in testo
    assert "◆" not in testo
    assert "◉" not in testo
    assert "SYSTEMS ONLINE" in testo


def test_frame_terminale_piccolo_non_solleva() -> None:
    """Un TTY 10×5 non deve crashare né produrre un frame vuoto inutilizzabile."""
    testo = frame_splash(DURATA_ANIMAZIONE_SEC, width=10, height=5, ascii_mode=True).plain
    assert "SYSTEMS ONLINE" in testo


def test_esegui_splash_stampa_saluto_con_orologio_fake() -> None:
    """HUD a 12 fps, pausa sul frame finale, poi il saluto. Copre il preload."""
    buffer, console = _console()
    orologio = _Orologio()
    esegui_splash(console, pausa=orologio.sleep, monotonic=orologio.monotonic)
    cattura = buffer.getvalue()
    assert MESSAGGIO_SPLASH in cattura
    assert orologio.attese[-2:] == [PAUSA_DOPO_ANIMAZIONE_SEC, PAUSA_SALUTO_SEC]
    assert all(delta == INTERVALLO_FRAME_SEC for delta in orologio.attese[:-2])
    assert orologio.attese[:-2]


def test_esegui_splash_live_in_errore_stampa_comunque_il_saluto() -> None:
    """Un fallimento di Rich Live non impedisce il saluto."""
    buffer, console = _console()
    orologio = _Orologio()
    with patch("assistente.splash.Live", side_effect=RuntimeError("live down")):
        esegui_splash(console, pausa=orologio.sleep, monotonic=orologio.monotonic)
    assert MESSAGGIO_SPLASH in buffer.getvalue()
