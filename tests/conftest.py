"""Fixture condivise: workspace tmp/test_run, config e2e, gate Ollama."""

from __future__ import annotations

import shutil

import pytest

from assistente.config import (
    ALIAS_MODELLO_DEFAULT,
    WORKSPACE,
    ConfigRuntime,
)
from assistente.console import ConsoleUI
from assistente.ollama_runtime import OllamaRuntime
from assistente.prompt_budget import reset_token_estimate_for_tests
from assistente.session import SessioneAgente
from assistente.tools import reset_stato_turno

pytest_plugins = ["tests.report_plugin"]

DIR_TEST_REL = "tmp/test_run"
DIR_UNIT_REL = f"{DIR_TEST_REL}/unit"
DIR_E2E_REL = f"{DIR_TEST_REL}/e2e"


@pytest.fixture(autouse=True)
def stima_token_isolata():
    """Azzera la calibrazione byte/token: è globale e non deve passare tra test."""
    reset_token_estimate_for_tests()
    yield
    reset_token_estimate_for_tests()


@pytest.fixture
def workspace_test_dir():
    """Prepara tmp/test_run vuoto (unit/ + e2e/) e lo rimuove a fine test."""
    destinazione = WORKSPACE / DIR_TEST_REL
    if destinazione.exists():
        shutil.rmtree(destinazione)
    (destinazione / "unit").mkdir(parents=True)
    (destinazione / "e2e").mkdir(parents=True)
    reset_stato_turno()
    yield destinazione
    shutil.rmtree(destinazione, ignore_errors=True)
    reset_stato_turno()


@pytest.fixture
def config_e2e() -> ConfigRuntime:
    """Config e2e: alias default (quant fissa q4 nella matrice)."""
    return ConfigRuntime(
        verbose=False,
        verifica_completa=False,
        burn_protection=False,
        modello=ALIAS_MODELLO_DEFAULT,
    )


@pytest.fixture
def sessione_agente(config_e2e: ConfigRuntime) -> SessioneAgente:
    """Sessione agente con UI non verbose (adatta a pytest non-TTY)."""
    ui = ConsoleUI()
    ui.imposta_verbose(config_e2e.verbose)
    return SessioneAgente(ui)


@pytest.fixture(scope="session")
def ollama_disponibile() -> tuple[bool, str]:
    """
    Probe Ollama una sola volta per sessione pytest.

    Usa OllamaRuntime così il fallback ai tag installati coincide con l'agente.
    Returns:
        (ok, dettaglio): dettaglio è il tag attivo oppure il messaggio d'errore.
    """
    runtime = OllamaRuntime()
    try:
        # Budget minimi: bastano pochi token per validare disponibilità.
        llm = runtime.get_llm(ALIAS_MODELLO_DEFAULT, 512, 8)
        llm.invoke("ok")
        return True, runtime.active_tag or "sconosciuto"
    except Exception as errore:
        return False, f"{type(errore).__name__}: {errore}"
    finally:
        # Non unloadare: i test e2e successivi riusano lo stesso artefatto caldo.
        runtime.close(unload=False)


@pytest.fixture
def richiede_ollama(ollama_disponibile: tuple[bool, str]):
    """Skippa il test se Ollama/modello risolto non sono disponibili."""
    ok, dettaglio = ollama_disponibile
    if not ok:
        pytest.skip(
            f"Ollama/modello {ALIAS_MODELLO_DEFAULT} "
            f"non disponibile: {dettaglio}"
        )
