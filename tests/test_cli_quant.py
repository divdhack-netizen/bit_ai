"""CLI model tests without a quantization flag/command."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from assistant.cli import (
    EXIT_COMMANDS,
    _model_label,
    _handle_command,
    _parse_args,
)
from assistant.config import ConfigRuntime
from assistant.host import PowerState
from assistant.inference_profiles import resolve_model_variant


@pytest.fixture(autouse=True)
def _on_ac_power(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI labels do not depend on the host machine's Low Power Mode."""
    monkeypatch.setattr(
        "assistant.energy_policy.detect_power",
        lambda: PowerState(on_battery=False, low_power_mode=False),
    )


def test_parse_args_without_quant(monkeypatch) -> None:
    """The CLI does not expose --quant; only --model remains."""
    monkeypatch.setattr(sys, "argv", ["main.py", "--model", "fast"])
    config, _prompt, _folder = _parse_args()
    assert config.model == "fast"
    assert not hasattr(config, "quantizzazione")
    assert config.thinking is False


def test_parse_args_default_model(monkeypatch) -> None:
    """Without --model the default is ConfigRuntime.model."""
    monkeypatch.setattr(sys, "argv", ["main.py"])
    config, _prompt, _folder = _parse_args()
    assert config.model == ConfigRuntime().model


def test_model_command_sets_fast() -> None:
    """/model fast updates ConfigRuntime and consumes the command."""
    config = ConfigRuntime(model="default")
    ui = MagicMock()
    consumed = _handle_command("/model fast", config, ui)
    assert consumed is True
    assert config.model == "fast"
    assert "fast" in ui.print_system_message.call_args.args[0].casefold()


def test_model_command_with_session_preload(monkeypatch) -> None:
    """/model calls prewarm_model with the alias only."""
    # Tight memory must not change the alias: quantization is fixed.
    monkeypatch.setattr(
        "assistant.inference_profiles.physical_memory_bytes",
        lambda: 16 * 1024**3,
    )
    config = ConfigRuntime(model="fast")
    ui = MagicMock()
    session = MagicMock()
    session.installed_model_names.return_value = frozenset({"gemma4:12b"})
    consumed = _handle_command("/model default", config, ui, session)
    assert consumed is True
    assert config.model == "default"
    session.prewarm_model.assert_called_once_with("default")
    session.download_model.assert_not_called()
    ui.ask_download_confirmation.assert_not_called()


def test_model_command_missing_download_y() -> None:
    """Missing model: warning, Y/n, pull, then switch."""
    config = ConfigRuntime(model="fast")
    ui = MagicMock()
    ui.ask_download_confirmation.return_value = True
    session = MagicMock()
    session.installed_model_names.return_value = frozenset({"gemma4:e4b"})
    consumed = _handle_command("/model default", config, ui, session)
    assert consumed is True
    assert config.model == "default"
    warning = ui.print_system_message.call_args_list[0].args[0]
    assert "non è presente" in warning
    assert "gemma4:12b" in warning
    ui.ask_download_confirmation.assert_called_once_with("Vuoi scaricarlo?")
    session.download_model.assert_called_once_with("gemma4:12b")
    session.prewarm_model.assert_called_once_with("default")


def test_model_command_missing_download_n() -> None:
    """n on the pull keeps the previous alias and does not preload."""
    config = ConfigRuntime(model="fast")
    ui = MagicMock()
    ui.ask_download_confirmation.return_value = False
    session = MagicMock()
    session.installed_model_names.return_value = frozenset({"gemma4:e4b"})
    consumed = _handle_command("/model default", config, ui, session)
    assert consumed is True
    assert config.model == "fast"
    session.download_model.assert_not_called()
    session.prewarm_model.assert_not_called()
    messages = [c.args[0] for c in ui.print_system_message.call_args_list]
    assert any("non è presente" in text for text in messages)
    assert any("Download annullato" in text for text in messages)


def test_model_command_list_marks_not_present() -> None:
    """/model without an alias lists installed vs not present."""
    config = ConfigRuntime(model="fast")
    ui = MagicMock()
    session = MagicMock()
    session.installed_model_names.return_value = frozenset({"gemma4:e4b"})
    _handle_command("/model", config, ui, session)
    text = ui.print_system_message.call_args.args[0]
    assert "fast (gemma4:e4b, installato)" in text
    assert "default (gemma4:12b, non presente)" in text
    assert "quality" not in text


def test_model_command_missing_pull_failed() -> None:
    """Failed pull: alias unchanged, no preload."""
    config = ConfigRuntime(model="fast")
    ui = MagicMock()
    ui.ask_download_confirmation.return_value = True
    session = MagicMock()
    session.installed_model_names.return_value = frozenset({"gemma4:e4b"})
    session.download_model.side_effect = RuntimeError("file not found")
    _handle_command("/model default", config, ui, session)
    assert config.model == "fast"
    session.prewarm_model.assert_not_called()
    ui.print_error.assert_called_once()


def test_ask_download_confirmation_y_n(monkeypatch) -> None:
    """Empty Enter and Y confirm; n declines."""
    from assistant.console import ConsoleUI

    ui = ConsoleUI()
    monkeypatch.setattr(ui, "print_system_message", lambda _m: None)
    monkeypatch.setattr("builtins.input", lambda _p: "")
    assert ui.ask_download_confirmation("Vuoi scaricarlo?") is True
    monkeypatch.setattr("builtins.input", lambda _p: "Y")
    assert ui.ask_download_confirmation("Vuoi scaricarlo?") is True
    monkeypatch.setattr("builtins.input", lambda _p: "n")
    assert ui.ask_download_confirmation("Vuoi scaricarlo?") is False


def test_label_derives_quant_from_alias() -> None:
    """fast/default→q4 in the fixed matrix."""
    assert resolve_model_variant("fast").quantization == "q4"
    assert resolve_model_variant("default").quantization == "q4"


def test_q_is_not_exit_command() -> None:
    """
    The letter q alone does not close the session.

    It stays out of EXIT_COMMANDS to avoid accidental exits.
    """
    assert "q" not in EXIT_COMMANDS


def test_parse_args_thinking_on(monkeypatch) -> None:
    """`--thinking` turns native CoT back on."""
    monkeypatch.setattr(sys, "argv", ["main.py", "--thinking"])
    config, _prompt, _folder = _parse_args()
    assert config.thinking is True


def test_parse_args_no_auto_fast(monkeypatch) -> None:
    """`--no-auto-fast` keeps the user alias even in Low Power Mode."""
    monkeypatch.setattr(sys, "argv", ["main.py", "--no-auto-fast"])
    config, _prompt, _folder = _parse_args()
    assert config.auto_fast_low_power is False


def test_lpm_model_label_shows_fast(monkeypatch) -> None:
    """The prompt shows the 4B when Low Power Mode forces fast, not the preferred 9B."""
    monkeypatch.setattr(
        "assistant.energy_policy.detect_power",
        lambda: PowerState(on_battery=True, low_power_mode=True),
    )
    config = ConfigRuntime(model="default", auto_fast_low_power=True)
    assert _model_label(config) == "fast · q4 · gemma4:e4b"
    assert config.model == "default"


def test_thinking_toggle_command() -> None:
    """`/thinking` and `/t` toggle native CoT."""
    config = ConfigRuntime()
    ui = MagicMock()
    assert config.thinking is False
    assert _handle_command("/thinking", config, ui) is True
    assert config.thinking is True
    assert _handle_command("/t off", config, ui) is True
    assert config.thinking is False


def test_parse_args_verbose_default_off(monkeypatch) -> None:
    """Without the flag, verbose is off: the bar does not show the LLM segments."""
    monkeypatch.setattr(sys, "argv", ["main.py"])
    config, _prompt, _folder = _parse_args()
    assert config.verbose is False


def test_help_command_prints_help() -> None:
    """`/help` and `/?` open the guide and consume the command."""
    config = ConfigRuntime()
    ui = MagicMock()
    assert _handle_command("/help", config, ui) is True
    ui.print_help.assert_called_once()
    assert _handle_command("/?", config, ui) is True
    assert ui.print_help.call_count == 2
