"""JARVIS startup HUD: short animation, pause, greeting in a small font."""

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

SPLASH_MESSAGE = "Benvenuto!"
FPS_SPLASH = 12
FRAME_INTERVAL_SEC = 1.0 / FPS_SPLASH
_T_RINGS_END = 0.8
_T_SCAN_START = 0.5
_T_SCAN_END = 1.6
ANIMATION_DURATION_SEC = _T_SCAN_END
POST_ANIMATION_PAUSE_SEC = 2.0
GREETING_PAUSE_SEC = 2.0

LOG_BOOT: tuple[tuple[float, str], ...] = (
    (0.50, "INITIATING HUD"),
    (0.80, "CORE LINK .............. OK"),
    (1.10, "VOICE INTERFACE ........ OK"),
    (1.40, "SYSTEMS ONLINE"),
)

_WIRE_STYLE = "cyan dim"
_RING_STYLE = "cyan"
_ACTIVE_STYLE = "bold bright_cyan"
_SCAN_STYLE = "bold bright_cyan"
_LOG_STYLE = "cyan"
_FINAL_LOG_STYLE = "bold bright_cyan"
_GREETING_STYLE = "bold cyan"

_CELL_ASPECT = 2.0
_RING_THICKNESS = 0.85
_RING_COUNT = 3
_MAX_HUD_WIDTH = 48
_MAX_HUD_HEIGHT = 13
_UNICODE_GLYPHS = ("·", "◉", "◆", "━")
_ASCII_GLYPHS = (".", "O", "*", "-")
_UNICODE_SAMPLE = "".join(_UNICODE_GLYPHS) + SPLASH_MESSAGE
_SPLASH_ERRORS = (KeyboardInterrupt, OSError, RuntimeError)


@dataclass(frozen=True, slots=True)
class _CacheHud:
    """Precomputed distances for a fixed-size HUD grid."""

    hud_width: int
    hud_height: int
    center_x: int
    center_y: int
    max_radius: float
    distances: tuple[tuple[float, ...], ...]


def _clamp01(value: float) -> float:
    """Clamp a progress value to the [0, 1] range."""
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _supports_unicode(console: Console) -> bool:
    """True if the encoding and TERM can show the Unicode HUD glyphs."""
    if os.environ.get("TERM") == "dumb":
        return False
    encoding = getattr(console, "encoding", None) or sys.stdout.encoding or "utf-8"
    try:
        _UNICODE_SAMPLE.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def _hud_dimensions(width: int, height: int) -> tuple[int, int]:
    """HUD width and height, odd when possible, at least 1×1."""
    hud_width = max(1, min(max(1, width) - 2, _MAX_HUD_WIDTH))
    hud_height = max(1, min(max(1, height) - 2, _MAX_HUD_HEIGHT))
    if hud_width > 1 and hud_width % 2 == 0:
        hud_width -= 1
    if hud_height > 1 and hud_height % 2 == 0:
        hud_height -= 1
    return hud_width, hud_height


def _cache_hud(width: int, height: int) -> _CacheHud:
    """Precompute the hypot of each cell: reused for every splash frame."""
    hud_width, hud_height = _hud_dimensions(width, height)
    center_x = hud_width // 2
    center_y = hud_height // 2
    max_radius = min(center_x, center_y * _CELL_ASPECT)
    if max_radius < 1.0:
        max_radius = 0.0
    distances: list[tuple[float, ...]] = []
    for line_idx in range(hud_height):
        # One row of distances from the center, with TTY aspect-ratio correction.
        line = tuple(
            math.hypot(column_idx - center_x, (line_idx - center_y) * _CELL_ASPECT)
            for column_idx in range(hud_width)
        )
        distances.append(line)
    return _CacheHud(
        hud_width=hud_width,
        hud_height=hud_height,
        center_x=center_x,
        center_y=center_y,
        max_radius=max_radius,
        distances=tuple(distances),
    )


def _visible_logs(elapsed_sec: float) -> tuple[str, ...]:
    """Boot lines already unlocked at instant `elapsed_sec`."""
    return tuple(line for threshold, line in LOG_BOOT if elapsed_sec >= threshold)


def _scan_line(elapsed_sec: float, hud_height: int) -> int | None:
    """Row index of the scan line, or None outside the scan window."""
    if elapsed_sec < _T_SCAN_START or elapsed_sec > _T_SCAN_END:
        return None
    progress = _clamp01(
        (elapsed_sec - _T_SCAN_START) / (_T_SCAN_END - _T_SCAN_START)
    )
    return min(hud_height - 1, int(progress * max(1, hud_height)))


def frame_splash(
    elapsed_sec: float,
    width: int,
    height: int,
    *,
    ascii_mode: bool = False,
    cache: _CacheHud | None = None,
) -> Text:
    """
    Compose one HUD frame (rings, scan, log; no greeting).

    Args:
        elapsed_sec: Seconds since start; past `ANIMATION_DURATION_SEC` the last HUD stays.
        width: Console width in cells.
        height: Console height in cells.
        ascii_mode: If True, use `.` `O` `*` `-` instead of the Unicode glyphs.
        cache: Precomputed distances; if None they are built for `width`×`height`.

    Returns:
        Multiline `Text` ready for `Align`.
    """
    fake_time = min(max(0.0, elapsed_sec), ANIMATION_DURATION_SEC)
    grid = cache if cache is not None else _cache_hud(max(1, width), max(1, height))
    ring, core, tick, scan = _ASCII_GLYPHS if ascii_mode else _UNICODE_GLYPHS
    unlock = _clamp01(fake_time / _T_RINGS_END) if _T_RINGS_END else 1.0
    active_rings = max(1, math.ceil(unlock * _RING_COUNT))
    radii = ()
    if grid.max_radius > 0.0:
        radii = tuple(
            grid.max_radius * (index + 1) / _RING_COUNT
            for index in range(active_rings)
        )
    wire_style = _RING_STYLE if active_rings >= _RING_COUNT else _WIRE_STYLE
    glyph_lines: list[list[str]] = [
        [" " for _column in range(grid.hud_width)]
        for _line in range(grid.hud_height)
    ]
    if radii:
        for line_idx, line_distances in enumerate(grid.distances):
            cells = glyph_lines[line_idx]
            for column_idx, distance in enumerate(line_distances):
                for radius in radii:
                    if abs(distance - radius) <= _RING_THICKNESS:
                        cells[column_idx] = ring
                        break
        angle = fake_time * 2.6
        tick_radius = radii[-1]
        tick_x = int(round(grid.center_x + tick_radius * math.cos(angle)))
        tick_y = int(
            round(grid.center_y + (tick_radius * math.sin(angle)) / _CELL_ASPECT)
        )
        if 0 <= tick_y < grid.hud_height and 0 <= tick_x < grid.hud_width:
            glyph_lines[tick_y][tick_x] = tick
    if 0 <= grid.center_y < grid.hud_height and 0 <= grid.center_x < grid.hud_width:
        glyph_lines[grid.center_y][grid.center_x] = core

    scan_line = _scan_line(fake_time, grid.hud_height)
    frame = Text()
    for line_index, cells in enumerate(glyph_lines):
        if scan_line is not None and line_index == scan_line:
            frame.append(scan * grid.hud_width, style=_SCAN_STYLE)
        else:
            line_text = "".join(cells)
            line = Text(line_text, style=wire_style)
            if core in line_text:
                core_start = line_text.find(core)
                line.stylize(
                    _ACTIVE_STYLE,
                    core_start,
                    core_start + len(core),
                )
            if tick in line_text and tick != core:
                tick_start = line_text.find(tick)
                line.stylize(
                    _ACTIVE_STYLE,
                    tick_start,
                    tick_start + len(tick),
                )
            frame.append_text(line)
        frame.append("\n")
    frame.append("\n")
    for log_line in _visible_logs(fake_time):
        log_style = _FINAL_LOG_STYLE if log_line == "SYSTEMS ONLINE" else _LOG_STYLE
        frame.append(f"  {log_line}\n", style=log_style)
    return frame


def _try(action: Callable[[], None]) -> None:
    """Run TTY I/O, ignoring interrupts and Rich/OS errors."""
    try:
        action()
    except _SPLASH_ERRORS:
        return


def run_splash(
    console: Console,
    pause: Callable[[float], None] | None = None,
    monotonic: Callable[[], float] | None = None,
) -> None:
    """
    Animate the HUD, wait, clear and show the greeting.

    Covers the model prewarm: preload and splash start together.

    Args:
        console: Target Rich console.
        pause: Wait between frames and between scenes. Default `time.sleep`.
        monotonic: Injectable clock. Default `time.monotonic`.

    Returns:
        None. TTY errors do not propagate: the caller can open the Panel.
    """
    wait = time.sleep if pause is None else pause
    clock = time.monotonic if monotonic is None else monotonic
    ascii_mode = not _supports_unicode(console)
    hud_width = max(1, int(console.width or 1))
    hud_height = max(1, int(console.height or 1))
    cache = _cache_hud(hud_width, hud_height)

    def _align(elapsed_sec: float) -> Align:
        return Align.center(
            frame_splash(
                elapsed_sec,
                hud_width,
                hud_height,
                ascii_mode=ascii_mode,
                cache=cache,
            ),
            vertical="middle",
            height=hud_height,
        )

    def _animate() -> None:
        console.clear()
        with Live(
            _align(0.0),
            console=console,
            refresh_per_second=FPS_SPLASH,
            transient=True,
            redirect_stdout=False,
            redirect_stderr=False,
        ) as live:
            start = clock()
            while clock() - start < ANIMATION_DURATION_SEC:
                live.update(_align(clock() - start))
                wait(FRAME_INTERVAL_SEC)

    _try(_animate)
    _try(lambda: console.print(_align(ANIMATION_DURATION_SEC)))
    _try(lambda: wait(POST_ANIMATION_PAUSE_SEC))

    def _greeting() -> None:
        console.clear()
        console.print(
            Align.center(
                Text(SPLASH_MESSAGE, style=_GREETING_STYLE),
                vertical="middle",
                height=hud_height,
            ),
        )
        wait(GREETING_PAUSE_SEC)

    _try(_greeting)
