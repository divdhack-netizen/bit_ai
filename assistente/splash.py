"""HUD di avvio JARVIS: animazione breve, pausa, saluto a font ridotto."""

from __future__ import annotations

import math
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.align import Align
from rich.live import Live
from rich.text import Text

if TYPE_CHECKING:
    from rich.console import Console

MESSAGGIO_SPLASH = "Benvenuto!"
FPS_SPLASH = 12
INTERVALLO_FRAME_SEC = 1.0 / FPS_SPLASH
_T_ANELLI_FINE = 0.8
_T_SCAN_INIZIO = 0.5
_T_SCAN_FINE = 1.6
DURATA_ANIMAZIONE_SEC = _T_SCAN_FINE
PAUSA_DOPO_ANIMAZIONE_SEC = 2.0
PAUSA_SALUTO_SEC = 2.0

LOG_BOOT: tuple[tuple[float, str], ...] = (
    (0.50, "INITIATING HUD"),
    (0.80, "CORE LINK .............. OK"),
    (1.10, "VOICE INTERFACE ........ OK"),
    (1.40, "SYSTEMS ONLINE"),
)

_STILE_FILO = "cyan dim"
_STILE_ANELLO = "cyan"
_STILE_ATTIVO = "bold bright_cyan"
_STILE_SCAN = "bold bright_cyan"
_STILE_LOG = "cyan"
_STILE_LOG_FINALE = "bold bright_cyan"
_STILE_SALUTO = "bold cyan"

_ASPETTO_CELLA = 2.0
_SPESSORE_ANELLO = 0.85
_NUMERO_ANELLI = 3
_LARGHEZZA_HUD_MAX = 48
_ALTEZZA_HUD_MAX = 13
_GLIFI_UNICODE = ("·", "◉", "◆", "━")
_GLIFI_ASCII = (".", "O", "*", "-")
_CAMPIONE_UNICODE = "".join(_GLIFI_UNICODE) + MESSAGGIO_SPLASH
_ERRORI_SPLASH = (KeyboardInterrupt, OSError, RuntimeError)


@dataclass(frozen=True, slots=True)
class _CacheHud:
    """Distanze precalcolate per una griglia HUD di dimensione fissa."""

    larghezza: int
    altezza: int
    centro_x: int
    centro_y: int
    raggio_massimo: float
    distanze: tuple[tuple[float, ...], ...]


def _clamp01(valore: float) -> float:
    """Limita un progresso nel range [0, 1]."""
    return 0.0 if valore < 0.0 else 1.0 if valore > 1.0 else valore


def _supporta_unicode(console: Console) -> bool:
    """True se encoding e TERM possono mostrare i glifi HUD Unicode."""
    if os.environ.get("TERM") == "dumb":
        return False
    encoding = getattr(console, "encoding", None) or sys.stdout.encoding or "utf-8"
    try:
        _CAMPIONE_UNICODE.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def _dimensioni_hud(width: int, height: int) -> tuple[int, int]:
    """Larghezza e altezza HUD, dispari se possibile, minimo 1×1."""
    larghezza = max(1, min(max(1, width) - 2, _LARGHEZZA_HUD_MAX))
    altezza = max(1, min(max(1, height) - 2, _ALTEZZA_HUD_MAX))
    if larghezza > 1 and larghezza % 2 == 0:
        larghezza -= 1
    if altezza > 1 and altezza % 2 == 0:
        altezza -= 1
    return larghezza, altezza


def _cache_hud(width: int, height: int) -> _CacheHud:
    """Precalcola hypot di ogni cella: riusato per tutti i frame della splash."""
    larghezza, altezza = _dimensioni_hud(width, height)
    centro_x = larghezza // 2
    centro_y = altezza // 2
    raggio_massimo = min(centro_x, centro_y * _ASPETTO_CELLA)
    if raggio_massimo < 1.0:
        raggio_massimo = 0.0
    distanze: list[tuple[float, ...]] = []
    for riga_idx in range(altezza):
        # Una riga di distanze dal centro, con correzione aspect ratio TTY.
        riga = tuple(
            math.hypot(colonna_idx - centro_x, (riga_idx - centro_y) * _ASPETTO_CELLA)
            for colonna_idx in range(larghezza)
        )
        distanze.append(riga)
    return _CacheHud(
        larghezza=larghezza,
        altezza=altezza,
        centro_x=centro_x,
        centro_y=centro_y,
        raggio_massimo=raggio_massimo,
        distanze=tuple(distanze),
    )


def _log_visibili(elapsed_sec: float) -> tuple[str, ...]:
    """Righe di boot già sbloccate all'istante `elapsed_sec`."""
    return tuple(riga for soglia, riga in LOG_BOOT if elapsed_sec >= soglia)


def _riga_scan(elapsed_sec: float, altezza: int) -> int | None:
    """Indice riga della scan line, o None fuori dalla finestra di scansione."""
    if elapsed_sec < _T_SCAN_INIZIO or elapsed_sec > _T_SCAN_FINE:
        return None
    avanzamento = _clamp01(
        (elapsed_sec - _T_SCAN_INIZIO) / (_T_SCAN_FINE - _T_SCAN_INIZIO)
    )
    return min(altezza - 1, int(avanzamento * max(1, altezza)))


def frame_splash(
    elapsed_sec: float,
    width: int,
    height: int,
    *,
    ascii_mode: bool = False,
    cache: _CacheHud | None = None,
) -> Text:
    """
    Compone un fotogramma HUD (anelli, scan, log; senza saluto).

    Args:
        elapsed_sec: Secondi dall'inizio; oltre `DURATA_ANIMAZIONE_SEC` resta l'ultimo HUD.
        width: Larghezza console in celle.
        height: Altezza console in celle.
        ascii_mode: Se True usa `.` `O` `*` `-` al posto dei glifi Unicode.
        cache: Distanze precalcolate; se None viene costruita per `width`×`height`.

    Returns:
        `Text` multilinea pronto per `Align`.
    """
    tempo = min(max(0.0, elapsed_sec), DURATA_ANIMAZIONE_SEC)
    griglia = cache if cache is not None else _cache_hud(max(1, width), max(1, height))
    anello, nucleo, tick, scan = _GLIFI_ASCII if ascii_mode else _GLIFI_UNICODE
    sblocco = _clamp01(tempo / _T_ANELLI_FINE) if _T_ANELLI_FINE else 1.0
    anelli_attivi = max(1, math.ceil(sblocco * _NUMERO_ANELLI))
    raggi = ()
    if griglia.raggio_massimo > 0.0:
        raggi = tuple(
            griglia.raggio_massimo * (indice + 1) / _NUMERO_ANELLI
            for indice in range(anelli_attivi)
        )
    stile_filo = _STILE_ANELLO if anelli_attivi >= _NUMERO_ANELLI else _STILE_FILO
    righe_glifi: list[list[str]] = [
        [" " for _colonna in range(griglia.larghezza)]
        for _riga in range(griglia.altezza)
    ]
    if raggi:
        for riga_idx, distanze_riga in enumerate(griglia.distanze):
            celle = righe_glifi[riga_idx]
            for colonna_idx, distanza in enumerate(distanze_riga):
                for raggio in raggi:
                    if abs(distanza - raggio) <= _SPESSORE_ANELLO:
                        celle[colonna_idx] = anello
                        break
        angolo = tempo * 2.6
        raggio_tick = raggi[-1]
        tick_x = int(round(griglia.centro_x + raggio_tick * math.cos(angolo)))
        tick_y = int(
            round(griglia.centro_y + (raggio_tick * math.sin(angolo)) / _ASPETTO_CELLA)
        )
        if 0 <= tick_y < griglia.altezza and 0 <= tick_x < griglia.larghezza:
            righe_glifi[tick_y][tick_x] = tick
    if 0 <= griglia.centro_y < griglia.altezza and 0 <= griglia.centro_x < griglia.larghezza:
        righe_glifi[griglia.centro_y][griglia.centro_x] = nucleo

    riga_scan = _riga_scan(tempo, griglia.altezza)
    fotogramma = Text()
    for indice_riga, celle in enumerate(righe_glifi):
        if riga_scan is not None and indice_riga == riga_scan:
            fotogramma.append(scan * griglia.larghezza, style=_STILE_SCAN)
        else:
            testo_riga = "".join(celle)
            riga = Text(testo_riga, style=stile_filo)
            if nucleo in testo_riga:
                inizio_nucleo = testo_riga.find(nucleo)
                riga.stylize(
                    _STILE_ATTIVO,
                    inizio_nucleo,
                    inizio_nucleo + len(nucleo),
                )
            if tick in testo_riga and tick != nucleo:
                inizio_tick = testo_riga.find(tick)
                riga.stylize(
                    _STILE_ATTIVO,
                    inizio_tick,
                    inizio_tick + len(tick),
                )
            fotogramma.append_text(riga)
        fotogramma.append("\n")
    fotogramma.append("\n")
    for riga_log in _log_visibili(tempo):
        stile_log = _STILE_LOG_FINALE if riga_log == "SYSTEMS ONLINE" else _STILE_LOG
        fotogramma.append(f"  {riga_log}\n", style=stile_log)
    return fotogramma


def _tenta(azione: Callable[[], None]) -> None:
    """Esegue I/O TTY ignorando interruzioni e errori Rich/OS."""
    try:
        azione()
    except _ERRORI_SPLASH:
        return


def esegui_splash(
    console: Console,
    pausa: Callable[[float], None] | None = None,
    monotonic: Callable[[], float] | None = None,
) -> None:
    """
    Anima il HUD, attende, pulisce e mostra il saluto.

    Copre il preriscaldamento del modello: preload e splash partono insieme.

    Args:
        console: Console Rich di destinazione.
        pausa: Attesa tra i frame e tra le scene. Default `time.sleep`.
        monotonic: Orologio iniettabile. Default `time.monotonic`.

    Returns:
        None. Errori TTY non si propagano: il chiamante può aprire il Panel.
    """
    attesa = time.sleep if pausa is None else pausa
    orologio = time.monotonic if monotonic is None else monotonic
    ascii_mode = not _supporta_unicode(console)
    larghezza = max(1, int(console.width or 1))
    altezza = max(1, int(console.height or 1))
    cache = _cache_hud(larghezza, altezza)

    def _allinea(elapsed_sec: float) -> Align:
        return Align.center(
            frame_splash(
                elapsed_sec,
                larghezza,
                altezza,
                ascii_mode=ascii_mode,
                cache=cache,
            ),
            vertical="middle",
            height=altezza,
        )

    def _anima() -> None:
        console.clear()
        with Live(
            _allinea(0.0),
            console=console,
            refresh_per_second=FPS_SPLASH,
            transient=True,
            redirect_stdout=False,
            redirect_stderr=False,
        ) as live:
            inizio = orologio()
            while orologio() - inizio < DURATA_ANIMAZIONE_SEC:
                live.update(_allinea(orologio() - inizio))
                attesa(INTERVALLO_FRAME_SEC)

    _tenta(_anima)
    _tenta(lambda: console.print(_allinea(DURATA_ANIMAZIONE_SEC)))
    _tenta(lambda: attesa(PAUSA_DOPO_ANIMAZIONE_SEC))

    def _saluto() -> None:
        console.clear()
        console.print(
            Align.center(
                Text(MESSAGGIO_SPLASH, style=_STILE_SALUTO),
                vertical="middle",
                height=altezza,
            ),
        )
        attesa(PAUSA_SALUTO_SEC)

    _tenta(_saluto)
