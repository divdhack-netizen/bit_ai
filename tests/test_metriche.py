"""Test unitari delle metriche condivise da ReAct e percorso direct."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from assistente.metriche import (
    AccumulatoreMetricheLLM,
    formatta_segmento_statistiche,
    stima_token_da_testo,
)
from assistente.session import CallbackUI

pytestmark = pytest.mark.unit

_CONTEXT_TOKENS = 3072
_PROMPT_TOKENS = 120
_OUTPUT_TOKENS = 40
_EVALUATION_DURATION_NS = 2_000_000_000


class _FakeClock:
    """Clock monotono controllabile dal test senza sleep reali."""

    def __init__(self) -> None:
        """Inizia dall'istante zero."""
        self.value = 0.0

    def __call__(self) -> float:
        """
        Restituisce l'istante impostato dal test.

        Returns:
            Secondi monotoni simulati.
        """
        return self.value


def _ollama_response() -> SimpleNamespace:
    """
    Costruisce la struttura LangChain minima letta da _info_ollama.

    Returns:
        Oggetto con generation_info equivalente a una risposta Ollama.
    """
    info = {
        "prompt_eval_count": _PROMPT_TOKENS,
        "eval_count": _OUTPUT_TOKENS,
        "eval_duration": _EVALUATION_DURATION_NS,
    }
    generation = SimpleNamespace(generation_info=info)
    return SimpleNamespace(generations=[[generation]])


def test_accumulatore_misura_ttft_llm_e_tool() -> None:
    """Lo snapshot combina metadata Ollama e tempi wall-clock deterministici."""
    clock = _FakeClock()
    metrics = AccumulatoreMetricheLLM(_CONTEXT_TOKENS, clock=clock)

    metrics.inizia_chiamata()
    clock.value = 0.125
    metrics.registra_primo_token()
    clock.value = 0.5
    metrics.registra(_ollama_response())
    metrics.inizia_tool()
    clock.value = 0.7
    snapshot = metrics.termina_tool()

    assert snapshot is not None
    assert snapshot.chiamate_llm == 1
    assert snapshot.token_input == _PROMPT_TOKENS
    assert snapshot.token_output == _OUTPUT_TOKENS
    assert snapshot.token_al_secondo == pytest.approx(20.0)
    assert snapshot.ttft_ms == pytest.approx(125.0)
    assert snapshot.durata_llm_ms == pytest.approx(500.0)
    assert snapshot.durata_tool_ms == pytest.approx(200.0)
    export = snapshot.as_dict()
    assert export["ttft_ms"] == pytest.approx(125.0)
    assert export["contesto_massimo"] == _CONTEXT_TOKENS


def test_stima_token_da_testo_arrotonda_per_eccesso() -> None:
    """Un frammento corto conta come un token; quattro caratteri restano un token."""
    assert stima_token_da_testo("") == 0
    assert stima_token_da_testo("ab") == 1
    assert stima_token_da_testo("abcd") == 1
    assert stima_token_da_testo("abcde") == 2


def test_snapshot_live_aggiorna_token_e_durata_durante_lo_stream() -> None:
    """Prima di on_llm_end la barra deve già mostrare stima, TTFT e durata aperta."""
    clock = _FakeClock()
    metrics = AccumulatoreMetricheLLM(_CONTEXT_TOKENS, clock=clock)

    metrics.inizia_chiamata()
    metrics.stima_prompt("abcd" * 8)
    live_prefill = metrics.snapshot()
    assert live_prefill is not None
    assert live_prefill.chiamate_llm == 1
    assert live_prefill.token_input == 8
    assert live_prefill.contesto_usato == 8
    assert live_prefill.token_output == 0

    clock.value = 0.1
    metrics.registra_primo_token()
    metrics.registra_token_stream("ciao")
    clock.value = 0.3
    metrics.registra_token_stream(" mondo")
    live = metrics.snapshot()

    assert live is not None
    assert live.token_output == 3
    assert live.ttft_ms == pytest.approx(100.0)
    assert live.durata_llm_ms == pytest.approx(300.0)
    assert live.token_al_secondo == pytest.approx(15.0)


def test_registra_sostituisce_le_stime_live_con_i_conteggi_ollama() -> None:
    """eval_count ufficiale non si somma alla stima caratteri dello stream."""
    clock = _FakeClock()
    metrics = AccumulatoreMetricheLLM(_CONTEXT_TOKENS, clock=clock)

    metrics.inizia_chiamata()
    metrics.stima_prompt("abcd" * 8)
    clock.value = 0.1
    metrics.registra_primo_token()
    metrics.registra_token_stream("x" * 40)
    clock.value = 0.5
    snapshot = metrics.registra(_ollama_response())

    assert snapshot is not None
    assert snapshot.token_input == _PROMPT_TOKENS
    assert snapshot.token_output == _OUTPUT_TOKENS
    assert snapshot.token_al_secondo == pytest.approx(20.0)
    assert snapshot.durata_llm_ms == pytest.approx(500.0)


def test_errore_llm_chiude_il_timer_senza_token_fittizi() -> None:
    """Un provider fallito conserva chiamata e durata ma lascia i token a zero."""
    clock = _FakeClock()
    metrics = AccumulatoreMetricheLLM(_CONTEXT_TOKENS, clock=clock)

    metrics.inizia_chiamata()
    clock.value = 0.25
    snapshot = metrics.registra_errore_llm()

    assert snapshot is not None
    assert snapshot.chiamate_llm == 1
    assert snapshot.token_input == 0
    assert snapshot.token_output == 0
    assert snapshot.durata_llm_ms == pytest.approx(250.0)


def test_formattazione_include_metriche_di_latenza() -> None:
    """La barra espone TTFT, chiamate e durata tool oltre ai token."""
    clock = _FakeClock()
    metrics = AccumulatoreMetricheLLM(_CONTEXT_TOKENS, clock=clock)
    metrics.inizia_chiamata()
    clock.value = 0.1
    metrics.registra_primo_token()
    clock.value = 0.2
    snapshot = metrics.registra(_ollama_response())

    assert snapshot is not None
    formatted = formatta_segmento_statistiche(snapshot, verbose=True)
    assert "LLM: 1" in formatted
    assert "TTFT: 100 ms" in formatted
    assert "tool: 0 ms" in formatted
    compatta = formatta_segmento_statistiche(snapshot)
    assert "tok/s" in compatta
    assert "token:" in compatta
    assert "ctx:" in compatta
    assert "token in:" not in compatta
    assert "TTFT:" not in compatta


def test_callback_pubblica_metriche_all_avvio_della_chiamata() -> None:
    """Dopo i tool la barra non deve restare a LLM 0/0 durante il prefill."""
    ui = MagicMock()
    ui.ha_stream_live = False
    callback = CallbackUI(ui, contesto_massimo=_CONTEXT_TOKENS)

    callback.on_llm_start({}, ["abcd" * 8])

    ui.imposta_metriche_turno.assert_called_once()
    snapshot = ui.imposta_metriche_turno.call_args.args[0]
    assert snapshot is not None
    assert snapshot.chiamate_llm == 1
    assert snapshot.token_input == 8
    assert snapshot.contesto_usato == 8


def test_callback_aggiorna_metriche_live_senza_ridisegno_con_panel_aperto() -> None:
    """Con Live attivo i token aggiornano lo snapshot; il panel rinfresca la barra."""
    ui = MagicMock()
    ui.ha_stream_live = True
    callback = CallbackUI(ui, contesto_massimo=_CONTEXT_TOKENS)
    callback.on_llm_start({}, ["prompt"])
    ui.imposta_metriche_turno.reset_mock()

    callback.on_llm_new_token("ciao")

    ui.aggiorna_metriche_turno.assert_called_once()
    ui.imposta_metriche_turno.assert_not_called()
    snapshot = ui.aggiorna_metriche_turno.call_args.args[0]
    assert snapshot is not None
    assert snapshot.token_output >= 1
    ui.scrivi_token.assert_called_once_with("ciao")


def test_callback_instrada_thinking_sulla_timeline() -> None:
    """Il CoT nativo va in `scrivi_pensiero`; il contenuto chiude e apre la risposta."""
    ui = MagicMock()
    ui.ha_stream_live = False
    callback = CallbackUI(ui, contesto_massimo=_CONTEXT_TOKENS)
    callback.on_llm_start({}, ["prompt"])
    callback.on_llm_new_token("idea", thinking=True)
    ui.scrivi_pensiero.assert_called_once_with("idea")
    ui.scrivi_token.assert_not_called()
    callback.on_llm_new_token("ciao")
    ui.chiudi_pensiero.assert_called_once()
    ui.scrivi_token.assert_called_once_with("ciao")


def test_callback_ignora_token_vuoti_per_ttft() -> None:
    """Stringhe vuote non devono far scattare TTFT né il renderer."""
    ui = MagicMock()
    ui.ha_stream_live = False
    callback = CallbackUI(ui, contesto_massimo=_CONTEXT_TOKENS)
    callback.on_llm_start({}, ["prompt"])
    callback.on_llm_new_token("")
    ui.scrivi_token.assert_not_called()
    ui.scrivi_pensiero.assert_not_called()
    assert callback._metriche.snapshot().ttft_ms == 0.0
