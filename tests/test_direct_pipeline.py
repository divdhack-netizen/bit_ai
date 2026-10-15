"""Tests for the single-generation direct pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.callbacks import BaseCallbackHandler

from assistant.direct_pipeline import DIRECT_SYSTEM_PREFIX, DirectAnswerPipeline

pytestmark = pytest.mark.unit


class _FakeLLM:
    """Test LLM that emits deterministic chunks."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    def stream(self, input: str, config=None, **kwargs):
        # Expose the prompt for asserts on what was sent to the model.
        self.last_prompt = input
        yield from self._chunks


def test_direct_pipeline_prompt_puts_intent_after_history() -> None:
    """The KV prefix stays stable: identity, memory, history, then intent."""
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


def test_direct_pipeline_single_streaming_generation() -> None:
    """The pipeline opens the direct stream and joins the chunks."""
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
    ui.start_direct_answer.assert_called_once()
    assert "Saluta" in llm.last_prompt


def test_direct_pipeline_provider_error_becomes_diagnostics() -> None:
    """A client exception does not leak the stack trace to the user."""

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


def test_direct_pipeline_without_reasoning_does_not_retry() -> None:
    """With thinking off, an empty answer is a diagnostic, not a second stream."""

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


def test_direct_pipeline_does_not_answer_with_own_knowledge() -> None:
    """Closed corpus: the direct chat points to the documents, not to the model weights."""
    assert "solo sui documenti della cartella" in DIRECT_SYSTEM_PREFIX
    assert "non dare fatti" in DIRECT_SYSTEM_PREFIX


def test_direct_pipeline_thinking_single_call() -> None:
    """With CoT the budget grows in place: no second generation."""

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
