"""Test CLI modello senza flag/comando quantizzazione."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from assistente.cli import (
    COMANDI_USCITA,
    _etichetta_modello,
    _gestisci_comando,
    _parse_argomenti,
)
from assistente.config import ConfigRuntime
from assistente.host import PowerState
from assistente.inference_profiles import resolve_model_variant


@pytest.fixture(autouse=True)
def _alimentazione_ac(monkeypatch: pytest.MonkeyPatch) -> None:
    """Le etichette CLI non dipendono dal Low Power Mode della macchina host."""
    monkeypatch.setattr(
        "assistente.politica_energia.detect_power",
        lambda: PowerState(on_battery=False, low_power_mode=False),
    )


def test_parse_argomenti_senza_quant(monkeypatch) -> None:
    """La CLI non espone --quant; resta solo --model."""
    monkeypatch.setattr(sys, "argv", ["main.py", "--model", "fast"])
    config, _prompt, _cartella = _parse_argomenti()
    assert config.modello == "fast"
    assert not hasattr(config, "quantizzazione")
    assert config.thinking is False


def test_parse_argomenti_default_modello(monkeypatch) -> None:
    """Senza --model il default è ConfigRuntime.modello."""
    monkeypatch.setattr(sys, "argv", ["main.py"])
    config, _prompt, _cartella = _parse_argomenti()
    assert config.modello == ConfigRuntime().modello


def test_comando_model_imposta_fast() -> None:
    """/model fast aggiorna ConfigRuntime e consuma il comando."""
    config = ConfigRuntime(modello="default")
    ui = MagicMock()
    consumato = _gestisci_comando("/model fast", config, ui)
    assert consumato is True
    assert config.modello == "fast"
    assert "fast" in ui.stampa_messaggio_sistema.call_args.args[0].casefold()


def test_comando_model_con_sessione_preload(monkeypatch) -> None:
    """/model richiama preriscalda_modello solo con l'alias."""
    # Memoria stretta non deve cambiare l'alias: quant è fissa.
    monkeypatch.setattr(
        "assistente.inference_profiles.physical_memory_bytes",
        lambda: 16 * 1024**3,
    )
    config = ConfigRuntime(modello="fast")
    ui = MagicMock()
    sessione = MagicMock()
    sessione.nomi_modelli_installati.return_value = frozenset({"gemma4:12b"})
    consumato = _gestisci_comando("/model default", config, ui, sessione)
    assert consumato is True
    assert config.modello == "default"
    sessione.preriscalda_modello.assert_called_once_with("default")
    sessione.scarica_modello.assert_not_called()
    ui.chiedi_conferma_download.assert_not_called()


def test_comando_model_assente_download_y() -> None:
    """Modello mancante: avviso, Y/n, pull e poi switch."""
    config = ConfigRuntime(modello="fast")
    ui = MagicMock()
    ui.chiedi_conferma_download.return_value = True
    sessione = MagicMock()
    sessione.nomi_modelli_installati.return_value = frozenset({"gemma4:e4b"})
    consumato = _gestisci_comando("/model default", config, ui, sessione)
    assert consumato is True
    assert config.modello == "default"
    avviso = ui.stampa_messaggio_sistema.call_args_list[0].args[0]
    assert "non è presente" in avviso
    assert "gemma4:12b" in avviso
    ui.chiedi_conferma_download.assert_called_once_with("Vuoi scaricarlo?")
    sessione.scarica_modello.assert_called_once_with("gemma4:12b")
    sessione.preriscalda_modello.assert_called_once_with("default")


def test_comando_model_assente_download_n() -> None:
    """n sul pull lascia l'alias precedente e non precarica."""
    config = ConfigRuntime(modello="fast")
    ui = MagicMock()
    ui.chiedi_conferma_download.return_value = False
    sessione = MagicMock()
    sessione.nomi_modelli_installati.return_value = frozenset({"gemma4:e4b"})
    consumato = _gestisci_comando("/model default", config, ui, sessione)
    assert consumato is True
    assert config.modello == "fast"
    sessione.scarica_modello.assert_not_called()
    sessione.preriscalda_modello.assert_not_called()
    messaggi = [c.args[0] for c in ui.stampa_messaggio_sistema.call_args_list]
    assert any("non è presente" in testo for testo in messaggi)
    assert any("Download annullato" in testo for testo in messaggi)


def test_comando_model_elenco_segna_non_presente() -> None:
    """/model senza alias elenca installato vs non presente."""
    config = ConfigRuntime(modello="fast")
    ui = MagicMock()
    sessione = MagicMock()
    sessione.nomi_modelli_installati.return_value = frozenset({"gemma4:e4b"})
    _gestisci_comando("/model", config, ui, sessione)
    testo = ui.stampa_messaggio_sistema.call_args.args[0]
    assert "fast (gemma4:e4b, installato)" in testo
    assert "default (gemma4:12b, non presente)" in testo
    assert "quality" not in testo


def test_comando_model_assente_pull_fallito() -> None:
    """Pull fallito: alias invariato, niente preload."""
    config = ConfigRuntime(modello="fast")
    ui = MagicMock()
    ui.chiedi_conferma_download.return_value = True
    sessione = MagicMock()
    sessione.nomi_modelli_installati.return_value = frozenset({"gemma4:e4b"})
    sessione.scarica_modello.side_effect = RuntimeError("file not found")
    _gestisci_comando("/model default", config, ui, sessione)
    assert config.modello == "fast"
    sessione.preriscalda_modello.assert_not_called()
    ui.stampa_errore.assert_called_once()


def test_chiedi_conferma_download_y_n(monkeypatch) -> None:
    """Invio vuoto e Y confermano; n rifiuta."""
    from assistente.console import ConsoleUI

    ui = ConsoleUI()
    monkeypatch.setattr(ui, "stampa_messaggio_sistema", lambda _m: None)
    monkeypatch.setattr("builtins.input", lambda _p: "")
    assert ui.chiedi_conferma_download("Vuoi scaricarlo?") is True
    monkeypatch.setattr("builtins.input", lambda _p: "Y")
    assert ui.chiedi_conferma_download("Vuoi scaricarlo?") is True
    monkeypatch.setattr("builtins.input", lambda _p: "n")
    assert ui.chiedi_conferma_download("Vuoi scaricarlo?") is False


def test_etichetta_deriva_quant_da_alias() -> None:
    """fast/default→q4 nella matrice fissa."""
    assert resolve_model_variant("fast").quantization == "q4"
    assert resolve_model_variant("default").quantization == "q4"


def test_q_non_e_comando_uscita() -> None:
    """
    La lettera q non chiude la sessione da sola.

    Resta esclusa da COMANDI_USCITA per evitare uscite accidentali.
    """
    assert "q" not in COMANDI_USCITA


def test_parse_argomenti_thinking_on(monkeypatch) -> None:
    """`--thinking` riaccende il CoT nativo."""
    monkeypatch.setattr(sys, "argv", ["main.py", "--thinking"])
    config, _prompt, _cartella = _parse_argomenti()
    assert config.thinking is True


def test_parse_argomenti_no_auto_fast(monkeypatch) -> None:
    """`--no-auto-fast` lascia l'alias utente anche in Low Power Mode."""
    monkeypatch.setattr(sys, "argv", ["main.py", "--no-auto-fast"])
    config, _prompt, _cartella = _parse_argomenti()
    assert config.auto_fast_low_power is False


def test_etichetta_modello_lpm_mostra_fast(monkeypatch) -> None:
    """Il prompt mostra il 4B quando Low Power Mode forza fast, non la preferenza 9B."""
    monkeypatch.setattr(
        "assistente.politica_energia.detect_power",
        lambda: PowerState(on_battery=True, low_power_mode=True),
    )
    config = ConfigRuntime(modello="default", auto_fast_low_power=True)
    assert _etichetta_modello(config) == "fast · q4 · gemma4:e4b"
    assert config.modello == "default"


def test_comando_thinking_toggle() -> None:
    """`/thinking` e `/t` togglano il CoT nativo."""
    config = ConfigRuntime()
    ui = MagicMock()
    assert config.thinking is False
    assert _gestisci_comando("/thinking", config, ui) is True
    assert config.thinking is True
    assert _gestisci_comando("/t off", config, ui) is True
    assert config.thinking is False


def test_parse_argomenti_verbose_default_off(monkeypatch) -> None:
    """Senza flag, verbose è spento: la barra non mostra i segmenti LLM."""
    monkeypatch.setattr(sys, "argv", ["main.py"])
    config, _prompt, _cartella = _parse_argomenti()
    assert config.verbose is False


def test_comando_help_stampa_aiuto() -> None:
    """`/help` e `/?` aprono la guida e consumano il comando."""
    config = ConfigRuntime()
    ui = MagicMock()
    assert _gestisci_comando("/help", config, ui) is True
    ui.stampa_aiuto.assert_called_once()
    assert _gestisci_comando("/?", config, ui) is True
    assert ui.stampa_aiuto.call_count == 2
