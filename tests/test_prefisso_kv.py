"""Test del riuso KV: estensione del prompt e prefisso stabile."""

from __future__ import annotations

import pytest
from langchain_core.agents import AgentAction

from assistente.prefisso_kv import ContinuazionePrefisso
from assistente.session import _limita_passi_react

pytestmark = pytest.mark.unit


def test_prefisso_direct_non_rilegge_la_testa() -> None:
    """Con gli id del prefisso, il turno direct invia solo storia e richiesta."""
    cache = ContinuazionePrefisso()
    prefix = "Sei un assistente.\n\n"
    cache.imposta_chiave(("fast", 8192, False, prefix), prefix)
    cache.registra_prefisso((9, 8))
    inviato, extra = cache.prepara(prefix + "RICHIESTA:\nciao\n")
    assert inviato == "RICHIESTA:\nciao\n"
    assert extra == {"raw": True, "context": [9, 8]}


def test_limita_passi_react_tiene_la_write() -> None:
    """Il callable dello scratchpad non butta l'observation di una scrittura recente."""
    azione = AgentAction("scrivi_file", "nota.txt", "Thought: scrivo")
    lettura = AgentAction("leggi_documento", "a.py", "Thought: leggo")
    steps = [
        (lettura, "x" * 5000),
        (azione, "File scritto con successo: nota.txt"),
    ]
    limitati = _limita_passi_react(steps, max_tokens=80)
    osservazioni = [osservazione for _, osservazione in limitati]
    assert any("File scritto" in testo or "omessa" in testo for testo in osservazioni)
    assert len(limitati) == 2
