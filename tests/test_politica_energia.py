"""Test policy energetica Low Power Mode / batteria."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from assistente.config import ConfigRuntime, risolvi_tag_ollama
from assistente.console import ConsoleUI
from assistente.host import PowerState
from assistente.politica_energia import (
    alias_per_alimentazione,
    alias_piu_leggero,
    prepara_turno_energia,
    reset_energia_per_test,
)
from assistente.session import SessioneAgente

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_energia() -> None:
    reset_energia_per_test()
    yield
    reset_energia_per_test()


def test_alias_lpm_diventa_fast() -> None:
    """Low Power Mode forza fast se auto_fast è attivo."""
    config = ConfigRuntime(modello="default", auto_fast_low_power=True)
    power = PowerState(on_battery=True, low_power_mode=True)
    assert alias_per_alimentazione("default", config, power=power) == "fast"


def test_alias_lpm_disattivabile() -> None:
    """`--no-auto-fast` conserva l'alias richiesto."""
    config = ConfigRuntime(modello="default", auto_fast_low_power=False)
    power = PowerState(on_battery=True, low_power_mode=True)
    assert alias_per_alimentazione("default", config, power=power) == "default"


def test_batteria_senza_lpm_non_degrada() -> None:
    """Solo batteria: avviso, nessun downgrade."""
    config = ConfigRuntime(modello="default")
    ui = MagicMock()
    power = PowerState(on_battery=True, low_power_mode=False)
    contesto = prepara_turno_energia(config, ui, power=power)
    assert contesto.alias_modello == "default"
    ui.stampa_messaggio_sistema.assert_called_once()
    assert "batteria" in ui.stampa_messaggio_sistema.call_args.args[0].casefold()


def test_lpm_annuncia_una_volta() -> None:
    """Lo stesso stato LPM non ripete l'avviso."""
    config = ConfigRuntime(modello="default")
    ui = MagicMock()
    power = PowerState(on_battery=True, low_power_mode=True)
    primo = prepara_turno_energia(config, ui, power=power)
    secondo = prepara_turno_energia(config, ui, power=power)
    assert primo.alias_modello == "fast"
    assert secondo.alias_modello == "fast"
    assert ui.stampa_messaggio_sistema.call_count == 1


def test_turno_lpm_invoca_tag_4b(monkeypatch: pytest.MonkeyPatch) -> None:
    """esegui_richiesta in LPM chiama Ollama con gemma4:e4b, non con il 12B della preferenza."""
    from unittest.mock import MagicMock

    monkeypatch.setattr(
        "assistente.politica_energia.detect_power",
        lambda: PowerState(on_battery=True, low_power_mode=True),
    )
    ui = MagicMock(spec=ConsoleUI)
    llm = MagicMock()
    llm.stream.return_value = iter(["ok"])
    llm_factory = MagicMock(return_value=llm)
    sessione = SessioneAgente(ui, llm_factory=llm_factory)
    config = ConfigRuntime(
        modello="default",
        burn_protection=False,
        auto_fast_low_power=True,
    )

    sessione.esegui_richiesta("ciao", config)

    assert llm_factory.call_args.kwargs["model"] == risolvi_tag_ollama("fast")
    assert llm_factory.call_args.kwargs["model"] != risolvi_tag_ollama("default")
    assert config.modello == "default"
    avvisi = [
        call.args[0]
        for call in ui.stampa_messaggio_sistema.call_args_list
        if call.args
    ]
    assert any("E4B" in testo for testo in avvisi)


def test_alias_piu_leggero_preferisce_fast() -> None:
    """Thermal e energia scelgono il più piccolo."""
    assert alias_piu_leggero("fast", "default") == "fast"
    assert alias_piu_leggero("default", "fast") == "fast"


def test_alias_piu_leggero_sconosciuto_degrada_a_default() -> None:
    """Un alias fuori matrice si tratta come default."""
    assert alias_piu_leggero("sconosciuto", "default") == "default"


def test_chiave_alimentazione_lpm_precede_batteria() -> None:
    """LPM vince sulla batteria; AC è lo stato residuo."""
    assert PowerState(on_battery=True, low_power_mode=True).chiave() == "lpm"
    assert PowerState(on_battery=True, low_power_mode=False).chiave() == "battery"
    assert PowerState().chiave() == "ac"
