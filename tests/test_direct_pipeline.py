"""Test pipeline diretta a singola generazione."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.callbacks import BaseCallbackHandler

from assistente.direct_pipeline import PREFISSO_SISTEMA_DIRECT, DirectAnswerPipeline

pytestmark = pytest.mark.unit


class _FakeLLM:
    """LLM di test che emette chunk deterministici."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    def stream(self, input: str, config=None, **kwargs):
        # Espone il prompt per asserzioni sul contenuto inviato al modello.
        self.last_prompt = input
        yield from self._chunks


def test_direct_pipeline_prompt_mette_intent_dopo_storico() -> None:
    """Il prefisso KV resta stabile: identità, memoria, storico, poi intent."""
    ui = MagicMock()
    llm = _FakeLLM(["ok"])
    pipeline = DirectAnswerPipeline(
        llm_provider=lambda *_args: llm,
        ui=ui,
    )
    pipeline.answer(
        request="Ciao",
        model_alias="default",
        history="Utente: prima\nAssistente: dopo",
        context_tokens=4096,
        output_tokens=64,
        callback=MagicMock(spec=BaseCallbackHandler),
        project_memory="MEMORIA PROGETTO:\nregole",
        intent_context="INTENT CONTEXT:\n- Percorso: direct",
    )
    prompt = llm.last_prompt
    idx_mem = prompt.index("MEMORIA PROGETTO")
    idx_hist = prompt.index("STORICO CONVERSAZIONE")
    idx_intent = prompt.index("INTENT CONTEXT")
    idx_req = prompt.index("RICHIESTA:")
    assert idx_mem < idx_hist < idx_intent < idx_req


def test_direct_pipeline_una_sola_generazione_streaming() -> None:
    """La pipeline apre lo stream diretto e concatena i chunk."""
    ui = MagicMock()
    llm = _FakeLLM(["Ciao ", "mondo"])
    pipeline = DirectAnswerPipeline(
        llm_provider=lambda *_args: llm,
        ui=ui,
    )
    callback = MagicMock(spec=BaseCallbackHandler)
    answer = pipeline.answer(
        request="Saluta",
        model_alias="default",
        history="",
        context_tokens=4096,
        output_tokens=768,
        callback=callback,
    )
    assert answer == "Ciao mondo"
    ui.inizia_risposta_diretta.assert_called_once()
    assert "Saluta" in llm.last_prompt


def test_direct_pipeline_errore_provider_diventa_diagnostica() -> None:
    """Un'eccezione del client non propaga lo stack all'utente."""

    class _BrokenLLM:
        def stream(self, input: str, config=None, **kwargs):
            raise RuntimeError("boom")

    ui = MagicMock()
    pipeline = DirectAnswerPipeline(
        llm_provider=lambda *_args: _BrokenLLM(),
        ui=ui,
    )
    answer = pipeline.answer(
        request="x",
        model_alias="fast",
        history="",
        context_tokens=2048,
        output_tokens=256,
        callback=MagicMock(spec=BaseCallbackHandler),
    )
    assert answer.startswith("Errore inferenza locale:")


def test_direct_pipeline_senza_reasoning_non_riprova() -> None:
    """Con thinking off una risposta vuota è diagnostica, non un secondo stream."""

    class _EmptyOnce:
        calls = 0

        def stream(self, input: str, config=None, **kwargs):
            del input, config, kwargs
            type(self).calls += 1
            yield ""

    ui = MagicMock()
    pipeline = DirectAnswerPipeline(
        llm_provider=lambda *_args: _EmptyOnce(),
        ui=ui,
    )
    answer = pipeline.answer(
        request="x",
        model_alias="default",
        history="",
        context_tokens=4096,
        output_tokens=64,
        callback=MagicMock(spec=BaseCallbackHandler),
        reasoning=False,
    )
    assert "risposta vuota" in answer
    assert _EmptyOnce.calls == 1


def test_direct_pipeline_non_risponde_con_conoscenze_proprie() -> None:
    """Closed-corpus: la chat diretta rimanda ai documenti, non ai pesi del modello."""
    assert "solo sui documenti della cartella" in PREFISSO_SISTEMA_DIRECT
    assert "non dare fatti" in PREFISSO_SISTEMA_DIRECT


def test_direct_pipeline_thinking_una_sola_chiamata() -> None:
    """Con CoT il budget sale in-place: niente seconda generazione."""

    class _CountingLLM:
        calls = 0
        last_kwargs: dict = {}

        def stream(self, input: str, config=None, **kwargs):
            del input, config
            type(self).calls += 1
            type(self).last_kwargs = dict(kwargs)
            yield "Esito: l'hardware locale resta il collo di bottiglia."

    ui = MagicMock()
    pipeline = DirectAnswerPipeline(
        llm_provider=lambda *_args: _CountingLLM(),
        ui=ui,
    )
    answer = pipeline.answer(
        request="opinione",
        model_alias="default",
        history="",
        context_tokens=4096,
        output_tokens=768,
        callback=MagicMock(spec=BaseCallbackHandler),
        reasoning=True,
    )
    assert "collo di bottiglia" in answer
    assert _CountingLLM.calls == 1
    assert _CountingLLM.last_kwargs.get("num_predict") == 1536
