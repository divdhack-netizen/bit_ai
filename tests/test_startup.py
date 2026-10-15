"""First-run wizard tests with no real network or Homebrew."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from assistant.bootstrap import OLLAMA_LINUX_INSTALL_ARGV
from assistant.cli import _run_session
from assistant.config import ConfigRuntime
from assistant.inference_profiles import (
    GIBIBYTE,
    resolve_model_variant,
    select_installed_tag,
    variant_is_installed,
)
from assistant.startup import (
    HOMEBREW_BIN_APPLE,
    HOMEBREW_INSTALL_ARGV,
    DetectedEnvironment,
    SetupAborted,
    StartupHost,
    suggested_alias,
    align_session_alias,
    run_wizard,
    model_tier,
    non_interactive_setup_message,
    parse_model_selection,
    detect_environment,
    interactive_session,
)

pytestmark = pytest.mark.unit

_MEMORY_8_GIB = 8 * GIBIBYTE
_MEMORY_16_GIB = 16 * GIBIBYTE
_MEMORY_64_GIB = 64 * GIBIBYTE
_BREW = HOMEBREW_BIN_APPLE
_OLLAMA = _BREW.parent / "ollama"


class _Clock:
    """Deterministic clock so tests do not wait for the 20s healthcheck."""

    def __init__(self) -> None:
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t

    def sleep(self, delta: float) -> None:
        self.t += delta


def _run_ok(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")


def _ui() -> MagicMock:
    ui = MagicMock()
    ui.ask_confirmation.return_value = False
    ui.ask_text.return_value = ""
    return ui


def _host(**overrides: object) -> StartupHost:
    clock = _Clock()
    values: dict[str, object] = {
        "which": lambda _name: None,
        "path_is_file": lambda _path: False,
        "run": _run_ok,
        "popen": MagicMock(),
        "ping_ollama": lambda: False,
        "list_models": lambda: set(),
        "memory_bytes": lambda: _MEMORY_16_GIB,
        "machine": lambda: "arm64",
        "system": lambda: "Darwin",
        "environ": {"PATH": "/usr/bin"},
        "sleep": clock.sleep,
        "monotonic": clock.monotonic,
    }
    values.update(overrides)
    return StartupHost(**values)  # type: ignore[arg-type]


def test_missing_brew_without_consent_does_not_launch_subprocess() -> None:
    """Homebrew declined: no installer and SetupAborted."""
    run = MagicMock()
    ui = _ui()
    with pytest.raises(SetupAborted, match="Homebrew"):
        run_wizard(ui, _host(run=run))
    run.assert_not_called()


def test_brew_present_ollama_missing_runs_brew_install() -> None:
    """With brew and without ollama the wizard builds `brew install ollama`."""
    commands: list[list[str]] = []

    def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(list(argv))
        return _run_ok(argv)

    ui = _ui()
    ui.ask_confirmation.return_value = True
    with pytest.raises(SetupAborted, match="Ollama"):
        run_wizard(
            ui,
            _host(
                which=lambda name: str(_BREW) if name == "brew" else None,
                path_is_file=lambda path: path == _BREW,
                run=run,
            ),
        )
    normalized = [[str(part) for part in command] for command in commands]
    assert [_BREW.as_posix(), "install", "ollama"] in normalized
    assert HOMEBREW_INSTALL_ARGV not in commands


def test_model_tier_by_ram_8_16_64() -> None:
    """The two Gemma 4 sizes change tier as RAM varies."""
    fast = resolve_model_variant("fast")
    default = resolve_model_variant("default")
    assert model_tier(fast, _MEMORY_8_GIB) == "sconsigliato"
    assert model_tier(default, _MEMORY_8_GIB) == "sconsigliato"
    assert suggested_alias(_MEMORY_8_GIB) is None
    assert model_tier(fast, _MEMORY_16_GIB) == "consigliato"
    assert model_tier(default, _MEMORY_16_GIB) == "consigliato"
    assert suggested_alias(_MEMORY_16_GIB) == "default"
    assert model_tier(fast, _MEMORY_64_GIB) == "consigliato"
    assert model_tier(default, _MEMORY_64_GIB) == "consigliato"
    assert suggested_alias(_MEMORY_64_GIB) == "default"


def test_parse_model_selection_indexes_and_aliases() -> None:
    """Accepts `1 2` and names; rejects unknown tokens."""
    aliases = ("fast", "default")
    assert parse_model_selection("1 2", aliases) == ("fast", "default")
    assert parse_model_selection("2", aliases) == ("default",)
    assert parse_model_selection("fast default", aliases) == ("fast", "default")
    assert parse_model_selection("", aliases) == ()
    with pytest.raises(ValueError):
        parse_model_selection("llama", aliases)


def test_select_installed_tag_after_dummy_pull() -> None:
    """After pulling the library tag, the preferred E4B resolves the fallback."""
    catalog = {"gemma4"}
    assert variant_is_installed("gemma4:e4b", catalog)
    assert select_installed_tag("gemma4:e4b", catalog) == "gemma4"
    assert not variant_is_installed("gemma4:12b", catalog)


def test_prompt_without_environment_does_not_start_wizard(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--prompt` with missing dependencies exits 1 and does not call the wizard."""
    incomplete = DetectedEnvironment(
        brew_path=None,
        ollama_cli=None,
        ollama_api=False,
        available_model_names=frozenset(),
        memory_bytes=_MEMORY_16_GIB,
        machine="arm64",
    )
    monkeypatch.setattr("assistant.cli.detect_environment", lambda: incomplete)
    wizard = MagicMock()
    monkeypatch.setattr("assistant.cli.run_wizard", wizard)
    session_cls = MagicMock()
    monkeypatch.setattr("assistant.cli.AgentSession", session_cls)
    ui = MagicMock()
    ui.thermal_bar = MagicMock()
    code = _run_session(ConfigRuntime(), "ciao mondo", ui)
    assert code == 1
    wizard.assert_not_called()
    session_cls.assert_not_called()
    ui.print_splash.assert_not_called()
    assert ui.print_error.call_args.args[0] == non_interactive_setup_message()


def test_align_alias_if_default_missing() -> None:
    """If default is not in the catalog, the first installed alias is used."""
    environment = DetectedEnvironment(
        brew_path=_BREW,
        ollama_cli=_OLLAMA,
        ollama_api=True,
        available_model_names=frozenset({"gemma4:e4b"}),
        memory_bytes=_MEMORY_16_GIB,
        machine="arm64",
    )
    assert environment.ready
    assert align_session_alias("default", environment) == "fast"


def test_interactive_session_false_with_prompt() -> None:
    """`--prompt` is never treated as a TTY for the wizard."""
    assert interactive_session("testo", stdin_tty=True) is False
    assert interactive_session(None, stdin_tty=True) is True
    assert interactive_session(None, stdin_tty=False) is False


def test_detect_environment_uses_canonical_brew_path() -> None:
    """Without brew on PATH, /opt/homebrew/bin/brew is still valid."""
    environment = detect_environment(
        _host(
            path_is_file=lambda path: path == _BREW,
            ping_ollama=lambda: True,
            list_models=lambda: {"gemma4:12b"},
        )
    )
    assert environment.brew_path == _BREW
    assert environment.has_model


def test_wizard_pulls_chosen_models() -> None:
    """Selecting `default` calls pull on the matrix tag."""
    pulled: list[str] = []
    catalog: set[str] = set()

    def path_is_file(path: Path) -> bool:
        return path in {_BREW, _OLLAMA}

    def pull_model(tag: str, _progress: object) -> None:
        pulled.append(tag)
        catalog.add(tag)

    ui = _ui()
    ui.ask_text.return_value = "default"
    environment = run_wizard(
        ui,
        _host(
            which=lambda name: str(_BREW) if name == "brew" else str(_OLLAMA),
            path_is_file=path_is_file,
            ping_ollama=lambda: True,
            list_models=lambda: set(catalog),
            pull_model=pull_model,
            environ={"PATH": "/opt/homebrew/bin"},
        ),
    )
    assert pulled == ["gemma4:12b"]
    assert environment.has_model
    ui.ask_confirmation.assert_not_called()


def test_invalid_selection_retries_then_pulls() -> None:
    """A wrong token does not abort: on the second attempt the pull starts."""
    pulled: list[str] = []
    catalog: set[str] = set()
    ui = _ui()
    ui.ask_text.side_effect = ["llama", "default"]
    run_wizard(
        ui,
        _host(
            which=lambda name: str(_BREW) if name == "brew" else str(_OLLAMA),
            path_is_file=lambda path: path in {_BREW, _OLLAMA},
            ping_ollama=lambda: True,
            list_models=lambda: set(catalog),
            pull_model=lambda tag, _p: (pulled.append(tag), catalog.add(tag)),
            environ={"PATH": "/opt/homebrew/bin"},
        ),
    )
    assert pulled == ["gemma4:12b"]
    ui.print_error.assert_called()


def test_linux_ready_without_homebrew() -> None:
    """On Linux Ollama + Gemma 4 are enough: brew is not a prerequisite."""
    ollama = Path("/usr/bin/ollama")
    environment = detect_environment(
        _host(
            system=lambda: "Linux",
            machine=lambda: "x86_64",
            which=lambda name: str(ollama) if name == "ollama" else None,
            path_is_file=lambda path: path == ollama,
            ping_ollama=lambda: True,
            list_models=lambda: {"gemma4:12b"},
        )
    )
    assert environment.brew_path is None
    assert environment.ready
    assert environment.ready


def test_linux_wizard_without_ollama_uses_official_installer() -> None:
    """Without the Ollama CLI on Linux, Homebrew is not touched."""
    commands: list[list[str]] = []

    def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(list(argv))
        return _run_ok(argv)

    ui = _ui()
    ui.ask_confirmation.return_value = True
    with pytest.raises(SetupAborted, match="Ollama"):
        run_wizard(
            ui,
            _host(
                system=lambda: "Linux",
                machine=lambda: "x86_64",
                run=run,
            ),
        )
    assert OLLAMA_LINUX_INSTALL_ARGV in commands
    assert HOMEBREW_INSTALL_ARGV not in commands


def test_linux_wizard_install_rejection_does_not_launch_subprocess() -> None:
    """Consent denied on Linux: no installer."""
    run = MagicMock()
    ui = _ui()
    with pytest.raises(SetupAborted, match="Ollama"):
        run_wizard(ui, _host(system=lambda: "Linux", machine=lambda: "x86_64", run=run))
    run.assert_not_called()
