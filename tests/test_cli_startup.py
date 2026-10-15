"""CLI startup: splash and prewarm in parallel."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from assistant.cli import PREWARM_IN_PROGRESS_MESSAGE, _run_session
from assistant.config import ConfigRuntime
from assistant.host import PowerState
from assistant.inference_profiles import GIBIBYTE
from assistant.startup import HOMEBREW_BIN_APPLE, DetectedEnvironment

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _on_ac_power(monkeypatch: pytest.MonkeyPatch) -> None:
    """Startup tests do not depend on the host machine's Low Power Mode."""
    monkeypatch.setattr(
        "assistant.energy_policy.detect_power",
        lambda: PowerState(on_battery=False, low_power_mode=False),
    )

_MEMORY_16_GIB = 16 * GIBIBYTE
_BREW = HOMEBREW_BIN_APPLE
_OLLAMA = _BREW.parent / "ollama"
_SYNC_TIMEOUT_SEC = 2.0


def _environment_ready() -> DetectedEnvironment:
    """Snapshot with Ollama and Gemma 4 already available, no wizard."""
    return DetectedEnvironment(
        brew_path=_BREW,
        ollama_cli=_OLLAMA,
        ollama_api=True,
        available_model_names=frozenset({"gemma4:12b"}),
        memory_bytes=_MEMORY_16_GIB,
        machine="arm64",
    )


def _environment_incomplete() -> DetectedEnvironment:
    """Snapshot without the Ollama CLI/API or the matrix models."""
    return DetectedEnvironment(
        brew_path=None,
        ollama_cli=None,
        ollama_api=False,
        available_model_names=frozenset(),
        memory_bytes=_MEMORY_16_GIB,
        machine="arm64",
    )


def _ui_loop() -> MagicMock:
    """Fake console that ends the interactive loop with EOF."""
    ui = MagicMock()
    ui.thermal_bar = MagicMock()
    ui.read_user_request.side_effect = EOFError
    return ui


def test_preload_starts_during_splash(monkeypatch: pytest.MonkeyPatch) -> None:
    """The preload POST starts while the splash is still running."""
    preload_started = threading.Event()
    splash_in_progress = threading.Event()

    def print_splash(*_args: object, **_kwargs: object) -> None:
        splash_in_progress.set()
        assert preload_started.wait(timeout=_SYNC_TIMEOUT_SEC)

    def preload(_alias: str) -> None:
        preload_started.set()
        assert splash_in_progress.wait(timeout=_SYNC_TIMEOUT_SEC)

    runtime = MagicMock()
    runtime.preload.side_effect = preload
    runtime_cls = MagicMock(return_value=runtime)
    session = MagicMock()
    session_cls = MagicMock(return_value=session)
    wizard = MagicMock()
    monkeypatch.setattr("assistant.cli.OllamaRuntime", runtime_cls)
    monkeypatch.setattr("assistant.cli.AgentSession", session_cls)
    monkeypatch.setattr("assistant.cli.detect_environment", _environment_ready)
    monkeypatch.setattr("assistant.cli.detect_host", lambda: MagicMock())
    monkeypatch.setattr("assistant.cli.run_wizard", wizard)
    monkeypatch.setattr("assistant.cli.interactive_session", lambda _prompt: True)
    ui = _ui_loop()
    ui.print_splash.side_effect = print_splash

    code = _run_session(ConfigRuntime(), None, ui)

    assert code == 0
    runtime.preload.assert_called_once_with("default")
    session_cls.assert_called_once()
    session.prewarm_model.assert_not_called()
    session.announce_prewarm.assert_called_once_with("default")
    wizard.assert_not_called()
    ui.print_welcome.assert_called_once()


def test_incomplete_environment_splash_then_wizard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without Ollama/Gemma the splash runs and the preload waits for the wizard."""
    order: list[str] = []
    runtime_cls = MagicMock()
    session = MagicMock()
    session_cls = MagicMock(return_value=session)

    def print_splash(*_args: object, **_kwargs: object) -> None:
        order.append("splash")

    def wizard(_ui: object) -> DetectedEnvironment:
        order.append("wizard")
        return _environment_ready()

    monkeypatch.setattr("assistant.cli.OllamaRuntime", runtime_cls)
    monkeypatch.setattr("assistant.cli.AgentSession", session_cls)
    monkeypatch.setattr("assistant.cli.detect_environment", _environment_incomplete)
    monkeypatch.setattr("assistant.cli.run_wizard", wizard)
    monkeypatch.setattr("assistant.cli.detect_host", lambda: MagicMock())
    monkeypatch.setattr("assistant.cli.interactive_session", lambda _prompt: True)
    ui = _ui_loop()
    ui.print_splash.side_effect = print_splash

    code = _run_session(ConfigRuntime(), None, ui)

    assert code == 0
    assert order == ["splash", "wizard"]
    runtime_cls.assert_not_called()
    session_cls.assert_called_once()
    session.prewarm_model.assert_called_once_with("default")
    session.announce_prewarm.assert_not_called()


def test_slow_preload_announces_wait_after_splash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the load outlasts the HUD, the CLI reports the wait before the join."""
    preload_blocked = threading.Event()
    release_preload = threading.Event()

    def print_splash(*_args: object, **_kwargs: object) -> None:
        assert preload_blocked.wait(timeout=_SYNC_TIMEOUT_SEC)

    def print_system_message(message: str) -> None:
        if message == PREWARM_IN_PROGRESS_MESSAGE:
            release_preload.set()

    def preload(_alias: str) -> None:
        preload_blocked.set()
        assert release_preload.wait(timeout=_SYNC_TIMEOUT_SEC)

    runtime = MagicMock()
    runtime.preload.side_effect = preload
    monkeypatch.setattr("assistant.cli.OllamaRuntime", MagicMock(return_value=runtime))
    monkeypatch.setattr("assistant.cli.AgentSession", MagicMock())
    monkeypatch.setattr("assistant.cli.detect_environment", _environment_ready)
    monkeypatch.setattr("assistant.cli.detect_host", lambda: MagicMock())
    monkeypatch.setattr("assistant.cli.interactive_session", lambda _prompt: True)
    ui = _ui_loop()
    ui.print_splash.side_effect = print_splash
    ui.print_system_message.side_effect = print_system_message

    code = _run_session(ConfigRuntime(), None, ui)

    assert code == 0
    ui.print_system_message.assert_any_call(PREWARM_IN_PROGRESS_MESSAGE)


def test_worker_closes_runtime_if_preload_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A preload that raises closes the runtime; the CLI retries on the main thread."""
    runtime = MagicMock()
    runtime.preload.side_effect = RuntimeError("boom")
    runtime_cls = MagicMock(return_value=runtime)
    session = MagicMock()
    session_cls = MagicMock(return_value=session)
    monkeypatch.setattr("assistant.cli.OllamaRuntime", runtime_cls)
    monkeypatch.setattr("assistant.cli.AgentSession", session_cls)
    monkeypatch.setattr("assistant.cli.detect_environment", _environment_ready)
    monkeypatch.setattr("assistant.cli.detect_host", lambda: MagicMock())
    monkeypatch.setattr("assistant.cli.interactive_session", lambda _prompt: True)
    ui = _ui_loop()

    code = _run_session(ConfigRuntime(), None, ui)

    assert code == 0
    runtime.close.assert_called()
    session.prewarm_model.assert_called_once_with("default")
    session.announce_prewarm.assert_not_called()


def test_preload_lpm_loads_and_announces_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In Low Power Mode the worker preloads fast and the announcement uses the same alias."""
    monkeypatch.setattr(
        "assistant.energy_policy.detect_power",
        lambda: PowerState(on_battery=True, low_power_mode=True),
    )
    runtime = MagicMock()
    runtime_cls = MagicMock(return_value=runtime)
    session = MagicMock()
    session_cls = MagicMock(return_value=session)
    monkeypatch.setattr("assistant.cli.OllamaRuntime", runtime_cls)
    monkeypatch.setattr("assistant.cli.AgentSession", session_cls)
    monkeypatch.setattr("assistant.cli.detect_environment", _environment_ready)
    monkeypatch.setattr("assistant.cli.detect_host", lambda: MagicMock())
    monkeypatch.setattr("assistant.cli.interactive_session", lambda _prompt: True)
    ui = _ui_loop()

    code = _run_session(ConfigRuntime(model="default"), None, ui)

    assert code == 0
    runtime.preload.assert_called_once_with("fast")
    session.announce_prewarm.assert_called_once_with("fast")
    session.prewarm_model.assert_not_called()


def test_cli_imports_agent_session() -> None:
    """Startup still exposes AgentSession from the session module."""
    from assistant.cli import AgentSession as CliSession
    from assistant.session import AgentSession

    assert CliSession is AgentSession
