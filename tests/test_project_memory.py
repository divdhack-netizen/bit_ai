"""Test memoria progetto e compaction strutturata dello storico."""

from __future__ import annotations

from pathlib import Path

import pytest

from assistente.memoria import MemoriaConversazione
from assistente.project_memory import format_project_memory_block, load_project_memory
from assistente.prompt_budget import estimate_tokens

pytestmark = pytest.mark.unit


def test_load_project_memory_agents_md(tmp_path: Path) -> None:
    """AGENTS.md nella workspace viene caricato come memoria progetto."""
    (tmp_path / "AGENTS.md").write_text("# Rules\nUse pytest.\n", encoding="utf-8")
    text = load_project_memory(tmp_path)
    assert "Use pytest" in text
    block = format_project_memory_block(tmp_path)
    assert "MEMORIA PROGETTO" in block
    assert "Use pytest" in block


def test_load_project_memory_assente(tmp_path: Path) -> None:
    """Workspace senza file → stringa vuota."""
    assert load_project_memory(tmp_path) == ""
    assert format_project_memory_block(tmp_path) == ""


def test_memoria_compaction_strutturata_quando_budget_stretto() -> None:
    """
    Con budget basso e molti turni, i vecchi finiscono in Compact storico.

    I messaggi utente recenti restano verbatim nei turni non compattati.
    """
    memoria = MemoriaConversazione(max_turni=6, max_caratteri=50_000)
    # Turni abbastanza lunghi da non entrare tutti in un budget stretto.
    for index in range(5):
        memoria.registra(
            f"richiesta utente numero {index} con vincolo non-touch-legacy",
            f"Assistente ha letto file_{index}.py e risposto con dettagli " + ("x" * 200),
        )
    # Budget che forza l'omissione di alcuni turni interi.
    formatted = memoria.formatta(max_tokens=180)
    assert estimate_tokens(formatted) <= 200
    # Compact strutturato presente oppure almeno i turni più recenti.
    assert "Compact storico" in formatted or "Turno" in formatted
    # L'ultimo messaggio utente resta nel testo (verbatim).
    assert "richiesta utente numero 4" in formatted
    assert "Compact storico" in formatted
