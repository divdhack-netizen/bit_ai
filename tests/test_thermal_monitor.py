"""Test monitor termico persistente, isteresi e parsing snapshot."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from assistente.config import ConfigRuntime
from assistente.thermal import (
    SnapshotTermico,
    StatoTermico,
    ThermalPolicyState,
    _parse_snapshot_json,
    calcola_throttle,
    prepara_turno,
    reset_policy_termica_per_test,
    snapshot_linux_sysfs,
    soglie_soft_celsius,
    stato_da_temperatura,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_termico() -> None:
    """Isola ogni test da stato globale del monitor/policy."""
    reset_policy_termica_per_test()
    yield
    reset_policy_termica_per_test()


def test_parse_snapshot_json_valido() -> None:
    """Una riga JSON del monitor Swift diventa SnapshotTermico."""
    snapshot = _parse_snapshot_json('{"thermalState":1,"tempC":72.5}')
    assert snapshot is not None
    assert snapshot.stato is StatoTermico.DISCRETO
    assert snapshot.temp_celsius == 72.5
    assert snapshot.letto_da_monitor is True


def test_parse_snapshot_json_temp_null() -> None:
    """tempC null è accettato quando SMC non è disponibile."""
    snapshot = _parse_snapshot_json('{"thermalState":0,"tempC":null}')
    assert snapshot is not None
    assert snapshot.temp_celsius is None


def test_isteresi_impedisce_flap_raffreddamento() -> None:
    """Il ritorno a Nominal richiede più campioni consecutivi."""
    policy = ThermalPolicyState()
    caldo = SnapshotTermico(StatoTermico.DISCRETO, 80.0, True)
    freddo = SnapshotTermico(StatoTermico.NOMINALE, 60.0, True)
    assert policy.aggiorna(caldo) is StatoTermico.DISCRETO
    # Il primo campione più fresco non deve ancora declassare la policy.
    assert policy.aggiorna(freddo) is StatoTermico.DISCRETO
    assert policy.aggiorna(freddo) is StatoTermico.DISCRETO
    assert policy.aggiorna(freddo) is StatoTermico.NOMINALE


def test_throttle_workload_first_riduce_output_prima_delle_pause() -> None:
    """In Fair la policy taglia predict/iterazioni con pausa pre-turno nulla."""
    throttle = calcola_throttle(StatoTermico.DISCRETO, "default")
    assert throttle.num_predict < 1536
    assert throttle.max_iterazioni <= 4
    assert throttle.pausa_pre_turno_sec == 0.0


def test_anticipazione_soft_richiede_trend_in_salita() -> None:
    """Senza trend positivo la temperatura alta non promuove Fair da Nominal."""
    policy = ThermalPolicyState()
    # Temperatura alta ma piatta: Apple Nominal resta Nominal.
    for _ in range(5):
        stato = policy.aggiorna(SnapshotTermico(StatoTermico.NOMINALE, 80.0, True))
    assert stato is StatoTermico.NOMINALE


def test_soglie_cooled_piu_alte_delle_fanless() -> None:
    """Pro/Linux throttlano più tardi rispetto a un Air fanless."""
    fair_fanless, serio_fanless = soglie_soft_celsius("fanless_conservative")
    fair_cooled, serio_cooled = soglie_soft_celsius("cooled")
    assert fair_cooled > fair_fanless
    assert serio_cooled > serio_fanless
    assert stato_da_temperatura(80.0, cooling="fanless_conservative") is StatoTermico.DISCRETO
    assert stato_da_temperatura(80.0, cooling="cooled") is StatoTermico.NOMINALE


def test_snapshot_linux_sysfs_usa_la_zona_piu_calda() -> None:
    """Il sensore Linux prende il max in millicelsius e mappa sulle soglie cooled."""
    files = {
        "/sys/class/thermal/thermal_zone0/temp": "45000",
        "/sys/class/thermal/thermal_zone1/temp": "96000",
    }
    snapshot = snapshot_linux_sysfs(
        listdir=lambda _base: ["thermal_zone0", "thermal_zone1", "cooling_device0"],
        read_text=lambda path: files.get(path.as_posix()),
    )
    assert snapshot.temp_celsius == 96.0
    assert snapshot.stato is StatoTermico.SERIO
    vuoto = snapshot_linux_sysfs(listdir=lambda _base: [], read_text=lambda _path: None)
    assert vuoto.stato is StatoTermico.NOMINALE
    assert vuoto.temp_celsius is None


def test_prepara_turno_serio_disabilita_thinking_senza_flag_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SERIO spegne il CoT come taglio di carico, indipendente da ConfigRuntime."""
    monkeypatch.setattr(
        "assistente.thermal.leggi_stato_termico",
        lambda: StatoTermico.SERIO,
    )
    monkeypatch.setattr("assistente.thermal.time.sleep", lambda *_args, **_kwargs: None)
    ui = MagicMock()
    contesto = prepara_turno(ConfigRuntime(burn_protection=True), ui)
    assert contesto is not None
    assert contesto.thinking is False
    assert "thinking disattivato" in ui.stampa_avviso_termico.call_args.args[0]


def test_prepara_turno_nominale_mantiene_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A temperatura nominale il CoT resta acceso."""
    monkeypatch.setattr(
        "assistente.thermal.leggi_stato_termico",
        lambda: StatoTermico.NOMINALE,
    )
    ui = MagicMock()
    contesto = prepara_turno(ConfigRuntime(burn_protection=True), ui)
    assert contesto is not None
    assert contesto.thinking is True
    ui.stampa_avviso_termico.assert_not_called()
