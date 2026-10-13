"""Registro di processo: JSONL, crash hook, turni incompleti e /errori."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from assistente.cli import _elabora_turno, _gestisci_comando
from assistente.config import ConfigRuntime, PREVIEW_RICHIESTA_CARATTERI
from assistente.console import ConsoleUI
from assistente.evaluate_turn import evaluate_turn
from assistente.registro_processo import (
    RegistroProcesso,
    classifica_esito_turno,
    preview_richiesta,
)
from assistente.session import CallbackUI, SessioneAgente

pytestmark = pytest.mark.unit


def test_preview_richiesta_tronca_oltre_il_limite() -> None:
    """Oltre 200 caratteri si persiste solo l'anteprima e la lunghezza reale."""
    testo = "a" * (PREVIEW_RICHIESTA_CARATTERI + 17)
    anteprima, lunghezza = preview_richiesta(testo)
    assert lunghezza == len(testo)
    assert len(anteprima) == PREVIEW_RICHIESTA_CARATTERI
    assert anteprima == testo[:PREVIEW_RICHIESTA_CARATTERI]


def test_jsonl_round_trip(tmp_path: Path) -> None:
    """session_start e un evento generico restano deserializzabili."""
    registro = RegistroProcesso(directory=tmp_path)
    registro.inizia_sessione(pid=1, python="3.14")
    registro.emetti("diagnostica", messaggio="vuoto")
    eventi = registro.leggi_eventi()
    assert eventi[0]["tipo"] == "session_start"
    assert eventi[0]["session_id"] == registro.session_id
    assert eventi[1]["tipo"] == "diagnostica"
    assert eventi[1]["messaggio"] == "vuoto"
    assert registro.path_indice.exists()


def test_inizia_turno_tronca_la_preview(tmp_path: Path) -> None:
    """turn_start non scrive il prompt intero."""
    registro = RegistroProcesso(directory=tmp_path)
    testo = "richiesta " + ("x" * 400)
    registro.inizia_turno("direct", "fast", testo)
    avvio = next(e for e in registro.leggi_eventi() if e["tipo"] == "turn_start")
    assert avvio["lunghezza_richiesta"] == len(testo.strip())
    assert len(avvio["preview"]) == PREVIEW_RICHIESTA_CARATTERI


def test_evaluate_turn_incompleto_diventa_turn_end(tmp_path: Path) -> None:
    """Il flag incompleto di evaluate_turn arriva nel JSONL come esito."""
    registro = RegistroProcesso(directory=tmp_path)
    ui = MagicMock(spec=ConsoleUI)
    sessione = SessioneAgente(ui, llm_factory=MagicMock(), registro=registro)
    try:
        registro.inizia_turno("react", "fast", "scrivi foo.py")
        risposta = "Agent stopped due to iteration limit"
        esito = evaluate_turn(
            risposta,
            scrittura_richiesta=True,
            path_scritti=[],
            max_iterations_hit=True,
        )
        assert esito.incompleto is True
        sessione._finalizza_turno(
            "scrivi foo.py",
            risposta,
            scrittura_richiesta=True,
            path_scritti=[],
            max_iterations_hit=True,
        )
        fine = next(e for e in registro.leggi_eventi() if e["tipo"] == "turn_end")
        assert fine["esito"] == "incompleto"
        assert fine["note"]
    finally:
        sessione.close()


def test_classifica_diagnostica_e_vuoto() -> None:
    """Le stringhe diagnostiche e le risposte vuote non sono un successo."""
    assert classifica_esito_turno(incompleto=False, risposta="") == "vuoto"
    assert (
        classifica_esito_turno(
            incompleto=False,
            risposta="Errore inferenza locale: Timeout",
        )
        == "errore"
    )
    assert classifica_esito_turno(incompleto=False, risposta="ok") == "ok"


def test_elabora_turno_eccezione_scrive_traceback(tmp_path: Path) -> None:
    """Un'eccezione nel loop CLI lascia tipo, messaggio e traceback su disco."""
    registro = RegistroProcesso(directory=tmp_path)
    ui = MagicMock()
    sessione = MagicMock()
    sessione.registro = registro

    def boom(_richiesta: str, _config: ConfigRuntime) -> tuple[str, list[str]]:
        registro.inizia_turno("direct", "fast", "ciao")
        raise ValueError("rotto")

    sessione.esegui_richiesta.side_effect = boom
    ok = _elabora_turno("ciao", sessione, ConfigRuntime(), ui)
    assert ok is False
    errore = next(e for e in registro.leggi_eventi() if e["tipo"] == "turn_error")
    assert errore["eccezione"] == "ValueError"
    assert "rotto" in errore["messaggio"]
    assert "ValueError" in errore["traceback"]
    ui.stampa_errore.assert_called_once()


def test_hook_crash_scrive_evento(tmp_path: Path) -> None:
    """sys.excepthook persistito registra crash e traceback, poi delega."""
    registro = RegistroProcesso(directory=tmp_path)
    registro.instala_hook_processo()
    try:
        try:
            raise RuntimeError("boom")
        except RuntimeError as err:
            sys.excepthook(type(err), err, err.__traceback__)
        crash = next(e for e in registro.leggi_eventi() if e["tipo"] == "crash")
        assert crash["eccezione"] == "RuntimeError"
        assert "boom" in crash["messaggio"]
        assert "RuntimeError" in crash["traceback"]
    finally:
        registro.chiudi()


def test_stall_una_sola_volta_per_turno(tmp_path: Path) -> None:
    """Il watchdog emette stall una volta sola oltre la soglia."""
    orologio = {"t": 0.0}
    registro = RegistroProcesso(
        directory=tmp_path,
        stall_soglia_sec=10.0,
        orologio_heartbeat=lambda: orologio["t"],
    )
    registro.inizia_turno("direct", "fast", "ciao")
    assert registro.controlla_stall(10.0) is False
    orologio["t"] = 11.0
    assert registro.controlla_stall(10.0) is True
    assert registro.controlla_stall(10.0) is False
    stalls = [e for e in registro.leggi_eventi() if e["tipo"] == "stall"]
    assert len(stalls) == 1


def test_comando_errori_legge_gli_ultimi_fallimenti(tmp_path: Path) -> None:
    """/errori stampa gli eventi non-ok della sessione corrente."""
    registro = RegistroProcesso(directory=tmp_path)
    registro.inizia_sessione(pid=1)
    registro.emetti("llm_error", eccezione="TimeoutError", messaggio="timeout")
    sessione = MagicMock()
    sessione.registro = registro
    ui = MagicMock()
    consumato = _gestisci_comando("/errori", ConfigRuntime(), ui, sessione)
    assert consumato is True
    testo = ui.stampa_messaggio_sistema.call_args.args[0]
    assert "llm_error" in testo
    assert "timeout" in testo


def test_comando_errori_senza_sessione() -> None:
    """Senza sessione il comando non alza e spiega l'assenza di registro."""
    ui = MagicMock()
    consumato = _gestisci_comando("/errori 3", ConfigRuntime(), ui)
    assert consumato is True
    testo = ui.stampa_messaggio_sistema.call_args.args[0]
    assert "registro" in testo.casefold()


def test_callback_llm_error_non_scarta_l_eccezione(tmp_path: Path) -> None:
    """on_llm_error propaga tipo e messaggio al JSONL."""
    registro = RegistroProcesso(directory=tmp_path)
    registro.inizia_turno("direct", "fast", "ciao")
    ui = MagicMock()
    ui.ha_stream_live = False
    callback = CallbackUI(ui, registro=registro)
    callback.on_llm_start({}, ["prompt"])
    callback.on_llm_error(RuntimeError("ollama down"))
    errore = next(e for e in registro.leggi_eventi() if e["tipo"] == "llm_error")
    assert errore["eccezione"] == "RuntimeError"
    assert "ollama down" in errore["messaggio"]


def test_keyboard_interrupt_non_emette_llm_error(tmp_path: Path) -> None:
    """Ctrl+C chiude il turno come annullato senza spacciarlo per errore LLM."""
    registro = RegistroProcesso(directory=tmp_path)
    ui = MagicMock(spec=ConsoleUI)
    ui.ha_stream_live = False
    sessione = MagicMock()
    sessione.registro = registro

    def boom(_richiesta: str, _config: ConfigRuntime) -> tuple[str, list[str]]:
        registro.inizia_turno("react", "default", "questi file non esistono")
        callback = CallbackUI(ui, registro=registro)
        callback.on_llm_start({}, ["prompt"])
        callback.on_llm_error(KeyboardInterrupt())
        raise KeyboardInterrupt

    sessione.esegui_richiesta.side_effect = boom
    ok = _elabora_turno(
        "questi file non esistono",
        sessione,
        ConfigRuntime(),
        ui,
    )
    assert ok is False
    tipi = [e["tipo"] for e in registro.leggi_eventi()]
    assert "llm_error" not in tipi
    fine = next(e for e in registro.leggi_eventi() if e["tipo"] == "turn_end")
    assert fine["esito"] == "annullato"
    indice = registro.formatta_ultimi_errori(10)
    assert "KeyboardInterrupt" not in indice
    assert "annullato" not in indice.casefold()


def test_elabora_turno_non_mostra_fallback_per_scrittura_tool() -> None:
    """I path dei tool ReAct non devono comparire nel pannello Fallback."""
    ui = MagicMock()
    sessione = MagicMock()
    sessione.esegui_richiesta.return_value = (
        "Navbar abbellita con successo.",
        ["sito/index.html"],
    )
    sessione.path_fallback_turno = []
    ok = _elabora_turno("abbellisci la navbar", sessione, ConfigRuntime(), ui)
    assert ok is True
    ui.finalizza.assert_called_once()
    ui.stampa_fallback.assert_not_called()


def test_elabora_turno_mostra_fallback_solo_create_only() -> None:
    """Il pannello Fallback elenca soltanto i path del create-only."""
    ui = MagicMock()
    sessione = MagicMock()
    sessione.esegui_richiesta.return_value = ("ok", ["note.txt"])
    sessione.path_fallback_turno = ["note.txt"]
    ok = _elabora_turno("salva", sessione, ConfigRuntime(), ui)
    assert ok is True
    ui.stampa_fallback.assert_called_once_with(["note.txt"])
