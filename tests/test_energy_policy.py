"""Tests for the Low Power Mode / battery energy policy."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from assistant.config import ConfigRuntime, resolve_ollama_tag
from assistant.console import ConsoleUI
from assistant.host import PowerState
from assistant.energy_policy import (
    alias_for_power,
    lightest_alias,
    prepare_energy_turn,
    reset_energy_for_tests,
)
from assistant.session import AgentSession

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_energy() -> None:
    reset_energy_for_tests()
    yield
    reset_energy_for_tests()


def test_lpm_alias_becomes_fast() -> None:
    """Low Power Mode forces fast when auto_fast is on."""
    config = ConfigRuntime(model="default", auto_fast_low_power=True)
    power = PowerState(on_battery=True, low_power_mode=True)
    assert alias_for_power("default", config, power=power) == "fast"


def test_lpm_alias_can_be_disabled() -> None:
    """`--no-auto-fast` keeps the requested alias."""
    config = ConfigRuntime(model="default", auto_fast_low_power=False)
    power = PowerState(on_battery=True, low_power_mode=True)
    assert alias_for_power("default", config, power=power) == "default"


def test_battery_without_lpm_does_not_degrade() -> None:
    """Battery only: a warning, no downgrade."""
    config = ConfigRuntime(model="default")
    ui = MagicMock()
    power = PowerState(on_battery=True, low_power_mode=False)
    context = prepare_energy_turn(config, ui, power=power)
    assert context.model_alias == "default"
    ui.print_system_message.assert_called_once()
    assert "batteria" in ui.print_system_message.call_args.args[0].casefold()


def test_lpm_announces_once() -> None:
    """The same LPM state does not repeat the warning."""
    config = ConfigRuntime(model="default")
    ui = MagicMock()
    power = PowerState(on_battery=True, low_power_mode=True)
    first = prepare_energy_turn(config, ui, power=power)
    second = prepare_energy_turn(config, ui, power=power)
    assert first.model_alias == "fast"
    assert second.model_alias == "fast"
    assert ui.print_system_message.call_count == 1


def test_lpm_turn_invokes_4b_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_request in LPM calls Ollama with gemma4:e4b, not with the preferred 12B."""
    from unittest.mock import MagicMock

    monkeypatch.setattr(
        "assistant.energy_policy.detect_power",
        lambda: PowerState(on_battery=True, low_power_mode=True),
    )
    ui = MagicMock(spec=ConsoleUI)
    llm = MagicMock()
    llm.stream.return_value = iter(["ok"])
    llm_factory = MagicMock(return_value=llm)
    session = AgentSession(ui, llm_factory=llm_factory)
    config = ConfigRuntime(
        model="default",
        burn_protection=False,
        auto_fast_low_power=True,
    )

    session.run_request("ciao", config)

    assert llm_factory.call_args.kwargs["model"] == resolve_ollama_tag("fast")
    assert llm_factory.call_args.kwargs["model"] != resolve_ollama_tag("default")
    assert config.model == "default"
    warnings = [
        call.args[0]
        for call in ui.print_system_message.call_args_list
        if call.args
    ]
    assert any("E4B" in text for text in warnings)


def test_lightest_alias_prefers_fast() -> None:
    """Thermal and energy pick the smaller one."""
    assert lightest_alias("fast", "default") == "fast"
    assert lightest_alias("default", "fast") == "fast"


def test_unknown_lightest_alias_degrades_to_default() -> None:
    """An alias outside the matrix is treated as default."""
    assert lightest_alias("sconosciuto", "default") == "default"


def test_lpm_power_key_precedes_battery() -> None:
    """LPM wins over battery; AC is the remaining state."""
    assert PowerState(on_battery=True, low_power_mode=True).key() == "lpm"
    assert PowerState(on_battery=True, low_power_mode=False).key() == "battery"
    assert PowerState().key() == "ac"
