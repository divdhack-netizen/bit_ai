"""Energy policy: Low Power Mode uses the fast alias without changing the preference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from assistant.config import ConfigRuntime
from assistant.host import PowerState, detect_power
from assistant.inference_profiles import MODEL_ALIAS_ORDER

_LPM_MESSAGE = (
    "Low Power Mode: uso fast (E4B); collega l'alimentatore o "
    "disattiva Risparmio energetico."
)
_BATTERY_MESSAGE = (
    "Alimentazione a batteria: i token/s possono calare. "
    "Collega l'alimentatore per prestazioni piene."
)


class EnergyUI(Protocol):
    """Minimal surface for energy warnings."""

    def print_system_message(self, message: str) -> None: ...


@dataclass(frozen=True, slots=True)
class EnergyContext:
    """Effective alias and power snapshot for the turn."""

    model_alias: str
    power: PowerState
    message: str | None


_last_announced_state: str | None = None


def reset_energy_for_tests() -> None:
    """Reset the one-shot announcement to its initial state."""
    global _last_announced_state
    _last_announced_state = None


def _alias_index(alias: str) -> int:
    """Position in MODEL_ALIAS_ORDER; an unknown alias counts as default."""
    try:
        return MODEL_ALIAS_ORDER.index(alias)
    except ValueError:
        return MODEL_ALIAS_ORDER.index("default")


def lightest_alias(first: str, second: str) -> str:
    """Pick the leftmost alias in the fast < default matrix."""
    return MODEL_ALIAS_ORDER[min(_alias_index(first), _alias_index(second))]


def alias_for_power(
    requested_alias: str,
    config: ConfigRuntime,
    *,
    power: PowerState | None = None,
) -> str:
    """
    Downgrade to fast only with Low Power Mode and the flag enabled.
    """
    state = power if power is not None else detect_power()
    if config.auto_fast_low_power and state.low_power_mode:
        return "fast"
    return requested_alias


def prepare_energy_turn(
    config: ConfigRuntime,
    ui: EnergyUI,
    *,
    power: PowerState | None = None,
) -> EnergyContext:
    """
    Derive the effective alias and announce a power change only once.
    """
    global _last_announced_state
    state = power if power is not None else detect_power()
    alias = alias_for_power(config.model, config, power=state)
    key = state.key()
    message: str | None = None
    if key == "lpm" and alias == "fast" and config.model != "fast":
        message = _LPM_MESSAGE
    elif key == "battery":
        message = _BATTERY_MESSAGE
    if message is not None and key != _last_announced_state:
        ui.print_system_message(message)
    _last_announced_state = key
    return EnergyContext(alias, state, message)
