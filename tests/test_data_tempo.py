"""Test unit: ContestoTemporale e tool data_oggi."""

from __future__ import annotations

from datetime import date

import pytest

from assistente.data_tempo import (
    ContestoTemporale,
    contesto_temporale,
    crea_tool_data,
    data_oggi_da_testo,
    formatta_data_odierna,
)


@pytest.fixture
def fissata() -> date:
    """Lunedì 3 agosto 2026 — data stabile per assert deterministici."""
    return date(2026, 8, 3)


def test_contesto_temporale_campi(fissata: date) -> None:
    """Snapshot: iso, anno ed etichetta IT coerenti con la data passata."""
    tempo = contesto_temporale(fissata)
    assert isinstance(tempo, ContestoTemporale)
    assert tempo.giorno == fissata
    assert tempo.iso == "2026-08-03"
    assert tempo.anno == 2026
    assert "lunedì" in tempo.etichetta
    assert "agosto" in tempo.etichetta
    assert "2026-08-03" in tempo.etichetta


def test_formatta_data_odierna_delega_a_contesto(fissata: date) -> None:
    """formatta_data_odierna coincide con ContestoTemporale.etichetta."""
    assert formatta_data_odierna(fissata) == contesto_temporale(fissata).etichetta


def test_data_oggi_da_testo_usa_date_today() -> None:
    """Tool live: ISO e anno di date.today(); input ignorato."""
    oggi = date.today()
    testo = data_oggi_da_testo(None)
    assert oggi.isoformat() in testo
    assert str(oggi.year) in testo


def test_crea_tool_data_registra_data_oggi() -> None:
    """Factory: un solo Tool nome data_oggi."""
    tools = crea_tool_data()
    assert len(tools) == 1
    assert tools[0].name == "data_oggi"
