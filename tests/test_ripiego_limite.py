"""Ripiego al limite iterazioni: la risposta arriva dall'ultimo passo utile."""

from __future__ import annotations

import pytest
from langchain_core.agents import AgentAction

from assistente.lettori import ISTRUZIONE_FORMATO_NON_SUPPORTATO
from assistente.session import _risposta_da_passi
from assistente.tools import AVVISO_LETTURA_RIPETUTA

pytestmark = pytest.mark.unit


def _passo(tool: str, tool_input: str, observation: str) -> tuple[AgentAction, str]:
    return AgentAction(tool, tool_input, ""), observation


def test_nessun_passo_nessun_ripiego() -> None:
    assert _risposta_da_passi([]) is None


def test_usa_l_ultimo_riassunto_riuscito() -> None:
    passi = [
        _passo("trova_file", "articolo_1", "Nessun risultato: articolo_1"),
        _passo("riassumi_documento", "Articolo_01.pages", "Il documento parla di IA."),
    ]
    assert _risposta_da_passi(passi) == "Il documento parla di IA."


def test_salta_errori_e_avviso_anti_loop() -> None:
    passi = [
        _passo("elenca_cartella", ".", "Contenuto di '.':\n  a.txt (1 byte)"),
        _passo("elenca_cartella", ".", f"{AVVISO_LETTURA_RIPETUTA}Contenuto di '.':\n  a.txt (1 byte)"),
        _passo("elenca_cartella", "notes", "Errore: Non è una cartella: notes"),
    ]
    assert _risposta_da_passi(passi) == "Contenuto di '.':\n  a.txt (1 byte)"


def test_formato_non_supportato_e_la_risposta() -> None:
    errore = (
        "Errore: Formato .zip non supportato o file binario: paperotti.zip. "
        f"Il file esiste ma non si può leggere; formati letti: txt.{ISTRUZIONE_FORMATO_NON_SUPPORTATO}"
    )
    passi = [_passo("leggi_documento", "paperotti.zip", errore)]
    risposta = _risposta_da_passi(passi)
    assert risposta is not None
    assert risposta.startswith("Formato .zip non supportato")
    assert "Non cercarlo" not in risposta


def test_documento_letto_diventa_estratto() -> None:
    passi = [_passo("leggi_documento", "a.txt", "x" * 3000)]
    risposta = _risposta_da_passi(passi)
    assert risposta is not None
    assert risposta.startswith("Ho letto a.txt") and risposta.endswith("…")


def test_cerca_testo_non_e_una_risposta() -> None:
    passi = [_passo("cerca_testo", "articolo_1", "Trovato 'articolo_1' in:\n  x.py (riga 1): article")]
    assert _risposta_da_passi(passi) is None
