"""Tests for the persistent thermal monitor, hysteresis and snapshot parsing."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from assistant.config import ConfigRuntime
from assistant.thermal import (
    ThermalSnapshot,
    ThermalState,
    ThermalPolicyState,
    _parse_snapshot_json,
    compute_throttle,
    prepare_turn,
    reset_thermal_policy_for_tests,
    snapshot_linux_sysfs,
    soft_thresholds_celsius,
    state_from_temperature,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_thermal() -> None:
    """Isolate each test from the global monitor/policy state."""
    reset_thermal_policy_for_tests()
    yield
    reset_thermal_policy_for_tests()


def test_parse_valid_json_snapshot() -> None:
    """A JSON line from the Swift monitor becomes a ThermalSnapshot."""
    snapshot = _parse_snapshot_json('{"thermalState":1,"tempC":72.5}')
    assert snapshot is not None
    assert snapshot.state is ThermalState.FAIR
    assert snapshot.temp_celsius == 72.5
    assert snapshot.read_from_monitor is True


def test_parse_snapshot_json_temp_null() -> None:
    """A null tempC is accepted when the SMC is not available."""
    snapshot = _parse_snapshot_json('{"thermalState":0,"tempC":null}')
    assert snapshot is not None
    assert snapshot.temp_celsius is None


def test_hysteresis_prevents_cooling_flap() -> None:
    """Returning to Nominal requires several consecutive samples."""
    policy = ThermalPolicyState()
    hot = ThermalSnapshot(ThermalState.FAIR, 80.0, True)
    cold = ThermalSnapshot(ThermalState.NOMINAL, 60.0, True)
    assert policy.update(hot) is ThermalState.FAIR
    # The first cooler sample must not downgrade the policy yet.
    assert policy.update(cold) is ThermalState.FAIR
    assert policy.update(cold) is ThermalState.FAIR
    assert policy.update(cold) is ThermalState.NOMINAL


def test_throttle_workload_first_reduces_output_before_pauses() -> None:
    """In Fair the policy cuts predict/iterations with no pre-turn pause."""
    throttle = compute_throttle(ThermalState.FAIR, "default")
    assert throttle.num_predict < 1536
    assert throttle.max_iterations <= 4
    assert throttle.pre_turn_pause_sec == 0.0


def test_soft_lookahead_requires_rising_trend() -> None:
    """Without a positive trend, a high temperature does not promote Fair from Nominal."""
    policy = ThermalPolicyState()
    # High but flat temperature: Apple Nominal stays Nominal.
    for _ in range(5):
        state = policy.update(ThermalSnapshot(ThermalState.NOMINAL, 80.0, True))
    assert state is ThermalState.NOMINAL


def test_cooled_thresholds_higher_than_fanless() -> None:
    """Pro/Linux throttle later than a fanless Air."""
    fair_fanless, serious_fanless = soft_thresholds_celsius("fanless_conservative")
    fair_cooled, serious_cooled = soft_thresholds_celsius("cooled")
    assert fair_cooled > fair_fanless
    assert serious_cooled > serious_fanless
    assert state_from_temperature(80.0, cooling="fanless_conservative") is ThermalState.FAIR
    assert state_from_temperature(80.0, cooling="cooled") is ThermalState.NOMINAL


def test_linux_sysfs_snapshot_uses_hottest_zone() -> None:
    """The Linux sensor takes the max in millicelsius and maps it to the cooled thresholds."""
    files = {
        "/sys/class/thermal/thermal_zone0/temp": "45000",
        "/sys/class/thermal/thermal_zone1/temp": "96000",
    }
    snapshot = snapshot_linux_sysfs(
        listdir=lambda _base: ["thermal_zone0", "thermal_zone1", "cooling_device0"],
        read_text=lambda path: files.get(path.as_posix()),
    )
    assert snapshot.temp_celsius == 96.0
    assert snapshot.state is ThermalState.SERIOUS
    empty = snapshot_linux_sysfs(listdir=lambda _base: [], read_text=lambda _path: None)
    assert empty.state is ThermalState.NOMINAL
    assert empty.temp_celsius is None


def test_prepare_serious_turn_disables_thinking_without_config_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SERIOUS turns CoT off as a load cut, independently of ConfigRuntime."""
    monkeypatch.setattr(
        "assistant.thermal.read_thermal_state",
        lambda: ThermalState.SERIOUS,
    )
    monkeypatch.setattr("assistant.thermal.time.sleep", lambda *_args, **_kwargs: None)
    ui = MagicMock()
    context = prepare_turn(ConfigRuntime(burn_protection=True), ui)
    assert context is not None
    assert context.thinking is False
    assert "thinking disattivato" in ui.print_thermal_warning.call_args.args[0]


def test_prepare_nominal_turn_keeps_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At nominal temperature CoT stays on."""
    monkeypatch.setattr(
        "assistant.thermal.read_thermal_state",
        lambda: ThermalState.NOMINAL,
    )
    ui = MagicMock()
    context = prepare_turn(ConfigRuntime(burn_protection=True), ui)
    assert context is not None
    assert context.thinking is True
    ui.print_thermal_warning.assert_not_called()
