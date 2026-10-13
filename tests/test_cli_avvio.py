"""Avvio CLI: splash e preriscaldamento in parallelo."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from assistente.cli import MESSAGGIO_PRERISCALDAMENTO_IN_CORSO, _esegui_sessione
from assistente.config import ConfigRuntime
from assistente.host import PowerState
from assistente.inference_profiles import GIBIBYTE
from assistente.startup import HOMEBREW_BIN_APPLE, AmbienteRilevato

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _alimentazione_ac(monkeypatch: pytest.MonkeyPatch) -> None:
    """I test di avvio non dipendono dal Low Power Mode della macchina host."""
    monkeypatch.setattr(
        "assistente.politica_energia.detect_power",
        lambda: PowerState(on_battery=False, low_power_mode=False),
    )

_MEMORY_16_GIB = 16 * GIBIBYTE
_BREW = HOMEBREW_BIN_APPLE
_OLLAMA = _BREW.parent / "ollama"
_TIMEOUT_SINCRONIA_SEC = 2.0


def _ambiente_pronto() -> AmbienteRilevato:
    """Snapshot con Ollama e Gemma 4 già disponibili, senza wizard."""
    return AmbienteRilevato(
        brew_path=_BREW,
        ollama_cli=_OLLAMA,
        ollama_api=True,
        available_model_names=frozenset({"gemma4:12b"}),
        memory_bytes=_MEMORY_16_GIB,
        machine="arm64",
    )


def _ambiente_incompleto() -> AmbienteRilevato:
    """Snapshot senza CLI/API Ollama né modelli della matrice."""
    return AmbienteRilevato(
        brew_path=None,
        ollama_cli=None,
        ollama_api=False,
        available_model_names=frozenset(),
        memory_bytes=_MEMORY_16_GIB,
        machine="arm64",
    )


def _ui_loop() -> MagicMock:
    """Console finta che chiude il loop interattivo con EOF."""
    ui = MagicMock()
    ui.barra_termica = MagicMock()
    ui.leggi_richiesta_utente.side_effect = EOFError
    return ui


def test_preload_parte_durante_lo_splash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Il POST di preload parte mentre lo splash è ancora in corso."""
    preload_avviato = threading.Event()
    splash_in_corso = threading.Event()

    def stampa_splash(*_args: object, **_kwargs: object) -> None:
        splash_in_corso.set()
        assert preload_avviato.wait(timeout=_TIMEOUT_SINCRONIA_SEC)

    def preload(_alias: str) -> None:
        preload_avviato.set()
        assert splash_in_corso.wait(timeout=_TIMEOUT_SINCRONIA_SEC)

    runtime = MagicMock()
    runtime.preload.side_effect = preload
    runtime_cls = MagicMock(return_value=runtime)
    sessione = MagicMock()
    sessione_cls = MagicMock(return_value=sessione)
    wizard = MagicMock()
    monkeypatch.setattr("assistente.cli.OllamaRuntime", runtime_cls)
    monkeypatch.setattr("assistente.cli.SessioneAgente", sessione_cls)
    monkeypatch.setattr("assistente.cli.rileva_ambiente", _ambiente_pronto)
    monkeypatch.setattr("assistente.cli.detect_host", lambda: MagicMock())
    monkeypatch.setattr("assistente.cli.esegui_wizard", wizard)
    monkeypatch.setattr("assistente.cli.sessione_interattiva", lambda _prompt: True)
    ui = _ui_loop()
    ui.stampa_splash.side_effect = stampa_splash

    codice = _esegui_sessione(ConfigRuntime(), None, ui)

    assert codice == 0
    runtime.preload.assert_called_once_with("default")
    sessione_cls.assert_called_once()
    sessione.preriscalda_modello.assert_not_called()
    sessione.annuncia_preriscaldamento.assert_called_once_with("default")
    wizard.assert_not_called()
    ui.stampa_benvenuto.assert_called_once()


def test_ambiente_incompleto_splash_poi_wizard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Senza Ollama/Gemma lo splash gira, il preload aspetta il wizard."""
    ordine: list[str] = []
    runtime_cls = MagicMock()
    sessione = MagicMock()
    sessione_cls = MagicMock(return_value=sessione)

    def stampa_splash(*_args: object, **_kwargs: object) -> None:
        ordine.append("splash")

    def wizard(_ui: object) -> AmbienteRilevato:
        ordine.append("wizard")
        return _ambiente_pronto()

    monkeypatch.setattr("assistente.cli.OllamaRuntime", runtime_cls)
    monkeypatch.setattr("assistente.cli.SessioneAgente", sessione_cls)
    monkeypatch.setattr("assistente.cli.rileva_ambiente", _ambiente_incompleto)
    monkeypatch.setattr("assistente.cli.esegui_wizard", wizard)
    monkeypatch.setattr("assistente.cli.detect_host", lambda: MagicMock())
    monkeypatch.setattr("assistente.cli.sessione_interattiva", lambda _prompt: True)
    ui = _ui_loop()
    ui.stampa_splash.side_effect = stampa_splash

    codice = _esegui_sessione(ConfigRuntime(), None, ui)

    assert codice == 0
    assert ordine == ["splash", "wizard"]
    runtime_cls.assert_not_called()
    sessione_cls.assert_called_once()
    sessione.preriscalda_modello.assert_called_once_with("default")
    sessione.annuncia_preriscaldamento.assert_not_called()


def test_preload_lento_annuncia_attesa_dopo_splash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Se il load supera l'HUD, la CLI segnala l'attesa prima del join."""
    preload_bloccato = threading.Event()
    rilascia_preload = threading.Event()

    def stampa_splash(*_args: object, **_kwargs: object) -> None:
        assert preload_bloccato.wait(timeout=_TIMEOUT_SINCRONIA_SEC)

    def stampa_messaggio_sistema(messaggio: str) -> None:
        if messaggio == MESSAGGIO_PRERISCALDAMENTO_IN_CORSO:
            rilascia_preload.set()

    def preload(_alias: str) -> None:
        preload_bloccato.set()
        assert rilascia_preload.wait(timeout=_TIMEOUT_SINCRONIA_SEC)

    runtime = MagicMock()
    runtime.preload.side_effect = preload
    monkeypatch.setattr("assistente.cli.OllamaRuntime", MagicMock(return_value=runtime))
    monkeypatch.setattr("assistente.cli.SessioneAgente", MagicMock())
    monkeypatch.setattr("assistente.cli.rileva_ambiente", _ambiente_pronto)
    monkeypatch.setattr("assistente.cli.detect_host", lambda: MagicMock())
    monkeypatch.setattr("assistente.cli.sessione_interattiva", lambda _prompt: True)
    ui = _ui_loop()
    ui.stampa_splash.side_effect = stampa_splash
    ui.stampa_messaggio_sistema.side_effect = stampa_messaggio_sistema

    codice = _esegui_sessione(ConfigRuntime(), None, ui)

    assert codice == 0
    ui.stampa_messaggio_sistema.assert_any_call(MESSAGGIO_PRERISCALDAMENTO_IN_CORSO)


def test_worker_chiude_runtime_se_preload_alza(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Un preload che alza chiude il runtime; la CLI ritenta sul main."""
    runtime = MagicMock()
    runtime.preload.side_effect = RuntimeError("boom")
    runtime_cls = MagicMock(return_value=runtime)
    sessione = MagicMock()
    sessione_cls = MagicMock(return_value=sessione)
    monkeypatch.setattr("assistente.cli.OllamaRuntime", runtime_cls)
    monkeypatch.setattr("assistente.cli.SessioneAgente", sessione_cls)
    monkeypatch.setattr("assistente.cli.rileva_ambiente", _ambiente_pronto)
    monkeypatch.setattr("assistente.cli.detect_host", lambda: MagicMock())
    monkeypatch.setattr("assistente.cli.sessione_interattiva", lambda _prompt: True)
    ui = _ui_loop()

    codice = _esegui_sessione(ConfigRuntime(), None, ui)

    assert codice == 0
    runtime.close.assert_called()
    sessione.preriscalda_modello.assert_called_once_with("default")
    sessione.annuncia_preriscaldamento.assert_not_called()


def test_preload_lpm_carica_e_annuncia_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In Low Power Mode il worker precarica fast e l'annuncio usa lo stesso alias."""
    monkeypatch.setattr(
        "assistente.politica_energia.detect_power",
        lambda: PowerState(on_battery=True, low_power_mode=True),
    )
    runtime = MagicMock()
    runtime_cls = MagicMock(return_value=runtime)
    sessione = MagicMock()
    sessione_cls = MagicMock(return_value=sessione)
    monkeypatch.setattr("assistente.cli.OllamaRuntime", runtime_cls)
    monkeypatch.setattr("assistente.cli.SessioneAgente", sessione_cls)
    monkeypatch.setattr("assistente.cli.rileva_ambiente", _ambiente_pronto)
    monkeypatch.setattr("assistente.cli.detect_host", lambda: MagicMock())
    monkeypatch.setattr("assistente.cli.sessione_interattiva", lambda _prompt: True)
    ui = _ui_loop()

    codice = _esegui_sessione(ConfigRuntime(modello="default"), None, ui)

    assert codice == 0
    runtime.preload.assert_called_once_with("fast")
    sessione.annuncia_preriscaldamento.assert_called_once_with("fast")
    sessione.preriscalda_modello.assert_not_called()


def test_cli_importa_sessione_agente() -> None:
    """L'avvio continua a esporre SessioneAgente dal modulo session."""
    from assistente.cli import SessioneAgente as SessioneCli
    from assistente.session import SessioneAgente

    assert SessioneCli is SessioneAgente
