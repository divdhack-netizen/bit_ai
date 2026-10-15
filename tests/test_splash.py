"""Splash HUD tests: frames, ASCII, fake clock, TTY errors."""

from __future__ import annotations

from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from rich.console import Console

from assistant.splash import (
    ANIMATION_DURATION_SEC,
    FRAME_INTERVAL_SEC,
    SPLASH_MESSAGE,
    POST_ANIMATION_PAUSE_SEC,
    GREETING_PAUSE_SEC,
    _supports_unicode,
    run_splash,
    frame_splash,
)

pytestmark = pytest.mark.unit


class _Clock:
    """Advances only when `sleep` is called: tests do not wait on the wall clock."""

    def __init__(self) -> None:
        self.t = 0.0
        self.expected: list[float] = []

    def monotonic(self) -> float:
        return self.t

    def sleep(self, delta: float) -> None:
        self.expected.append(delta)
        self.t += delta


def _console(width: int = 80, height: int = 24) -> tuple[StringIO, Console]:
    """In-memory Rich console to capture the splash output."""
    buffer = StringIO()
    console = Console(
        file=buffer,
        width=width,
        height=height,
        force_terminal=True,
        color_system=None,
    )
    return buffer, console


def test_initial_frame_has_rings_without_greeting() -> None:
    """At t=0 the rings are visible and the greeting is not in the HUD."""
    text = frame_splash(0.0, width=80, height=24).plain
    assert "·" in text or "." in text
    assert "◉" in text or "O" in text
    assert "SYSTEMS ONLINE" not in text
    assert SPLASH_MESSAGE not in text


def test_full_hud_frame_without_greeting() -> None:
    """At the end of the animation there is SYSTEMS ONLINE, not the welcome text."""
    text = frame_splash(ANIMATION_DURATION_SEC, width=80, height=24).plain
    assert "SYSTEMS ONLINE" in text
    assert "INITIATING HUD" in text
    assert SPLASH_MESSAGE not in text


def test_supports_unicode_false_on_latin1() -> None:
    """A restricted encoding forces the ASCII palette."""
    console = SimpleNamespace(encoding="latin-1")
    assert _supports_unicode(console) is False


def test_ascii_frame_without_unicode_glyphs() -> None:
    """In ascii_mode the HUD uses * / O and not the Unicode symbols."""
    text = frame_splash(ANIMATION_DURATION_SEC, 80, 24, ascii_mode=True).plain
    assert "O" in text
    assert "*" in text or "." in text
    assert "◆" not in text
    assert "◉" not in text
    assert "SYSTEMS ONLINE" in text


def test_small_terminal_frame_does_not_raise() -> None:
    """A 10×5 TTY must not crash or produce an unusable empty frame."""
    text = frame_splash(ANIMATION_DURATION_SEC, width=10, height=5, ascii_mode=True).plain
    assert "SYSTEMS ONLINE" in text


def test_run_splash_prints_greeting_with_fake_clock() -> None:
    """HUD at 12 fps, a pause on the final frame, then the greeting. Covers the preload."""
    buffer, console = _console()
    clock = _Clock()
    run_splash(console, pause=clock.sleep, monotonic=clock.monotonic)
    capture = buffer.getvalue()
    assert SPLASH_MESSAGE in capture
    assert clock.expected[-2:] == [POST_ANIMATION_PAUSE_SEC, GREETING_PAUSE_SEC]
    assert all(delta == FRAME_INTERVAL_SEC for delta in clock.expected[:-2])
    assert clock.expected[:-2]


def test_run_splash_live_error_still_prints_greeting() -> None:
    """A Rich Live failure does not prevent the greeting."""
    buffer, console = _console()
    clock = _Clock()
    with patch("assistant.splash.Live", side_effect=RuntimeError("live down")):
        run_splash(console, pause=clock.sleep, monotonic=clock.monotonic)
    assert SPLASH_MESSAGE in buffer.getvalue()
