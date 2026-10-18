"""Unit tests for the metrics shared by ReAct and the direct path."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from assistant.metrics import (
    LLMMetricsAccumulator,
    format_stats_segment,
    estimate_tokens_from_text,
)
from assistant.session import CallbackUI

pytestmark = pytest.mark.unit

_CONTEXT_TOKENS = 3072
_PROMPT_TOKENS = 120
_OUTPUT_TOKENS = 40
_EVALUATION_DURATION_NS = 2_000_000_000


class _FakeClock:
    """Monotonic clock controlled by the test, with no real sleeps."""

    def __init__(self) -> None:
        """Start at time zero."""
        self.value = 0.0

    def __call__(self) -> float:
        """
        Return the instant set by the test.

        Returns:
            Simulated monotonic seconds.
        """
        return self.value


def _ollama_response() -> SimpleNamespace:
    """
    Build the minimal LangChain structure read by _info_ollama.

    Returns:
        Object with generation_info equivalent to an Ollama response.
    """
    info = {
        "prompt_eval_count": _PROMPT_TOKENS,
        "eval_count": _OUTPUT_TOKENS,
        "eval_duration": _EVALUATION_DURATION_NS,
    }
    generation = SimpleNamespace(generation_info=info)
    return SimpleNamespace(generations=[[generation]])


def test_accumulator_measures_llm_and_tool_ttft() -> None:
    """The snapshot combines Ollama metadata and deterministic wall-clock times."""
    clock = _FakeClock()
    metrics = LLMMetricsAccumulator(_CONTEXT_TOKENS, clock=clock)

    metrics.start_call()
    clock.value = 0.125
    metrics.record_first_token()
    clock.value = 0.5
    metrics.record(_ollama_response())
    metrics.start_tool()
    clock.value = 0.7
    snapshot = metrics.end_tool()

    assert snapshot is not None
    assert snapshot.llm_calls == 1
    assert snapshot.token_input == _PROMPT_TOKENS
    assert snapshot.token_output == _OUTPUT_TOKENS
    assert snapshot.tokens_per_second == pytest.approx(20.0)
    assert snapshot.ttft_ms == pytest.approx(125.0)
    assert snapshot.llm_duration_ms == pytest.approx(500.0)
    assert snapshot.tool_duration_ms == pytest.approx(200.0)
    export = snapshot.as_dict()
    assert export["ttft_ms"] == pytest.approx(125.0)
    assert export["contesto_massimo"] == _CONTEXT_TOKENS


def test_estimate_tokens_from_text_rounds_up() -> None:
    """A short fragment counts as one token; four characters stay one token."""
    assert estimate_tokens_from_text("") == 0
    assert estimate_tokens_from_text("ab") == 1
    assert estimate_tokens_from_text("abcd") == 1
    assert estimate_tokens_from_text("abcde") == 2


def test_live_snapshot_updates_tokens_and_duration_during_stream() -> None:
    """Before on_llm_end the bar must already show the estimate, TTFT and open duration."""
    clock = _FakeClock()
    metrics = LLMMetricsAccumulator(_CONTEXT_TOKENS, clock=clock)

    metrics.start_call()
    metrics.prompt_estimate("abcd" * 8)
    live_prefill = metrics.snapshot()
    assert live_prefill is not None
    assert live_prefill.llm_calls == 1
    assert live_prefill.token_input == 8
    assert live_prefill.used_context == 8
    assert live_prefill.token_output == 0

    clock.value = 0.1
    metrics.record_first_token()
    metrics.record_stream_token("ciao")
    clock.value = 0.3
    metrics.record_stream_token(" mondo")
    live = metrics.snapshot()

    assert live is not None
    assert live.token_output == 3
    assert live.ttft_ms == pytest.approx(100.0)
    assert live.llm_duration_ms == pytest.approx(300.0)
    assert live.tokens_per_second == pytest.approx(15.0)


def test_record_replaces_live_estimates_with_ollama_counts() -> None:
    """The official eval_count is not added to the stream's character estimate."""
    clock = _FakeClock()
    metrics = LLMMetricsAccumulator(_CONTEXT_TOKENS, clock=clock)

    metrics.start_call()
    metrics.prompt_estimate("abcd" * 8)
    clock.value = 0.1
    metrics.record_first_token()
    metrics.record_stream_token("x" * 40)
    clock.value = 0.5
    snapshot = metrics.record(_ollama_response())

    assert snapshot is not None
    assert snapshot.token_input == _PROMPT_TOKENS
    assert snapshot.token_output == _OUTPUT_TOKENS
    assert snapshot.tokens_per_second == pytest.approx(20.0)
    assert snapshot.llm_duration_ms == pytest.approx(500.0)


def test_llm_error_closes_timer_without_dummy_tokens() -> None:
    """A failed provider keeps the call and duration but leaves tokens at zero."""
    clock = _FakeClock()
    metrics = LLMMetricsAccumulator(_CONTEXT_TOKENS, clock=clock)

    metrics.start_call()
    clock.value = 0.25
    snapshot = metrics.record_llm_error()

    assert snapshot is not None
    assert snapshot.llm_calls == 1
    assert snapshot.token_input == 0
    assert snapshot.token_output == 0
    assert snapshot.llm_duration_ms == pytest.approx(250.0)


def test_formatting_includes_latency_metrics() -> None:
    """The bar shows TTFT, calls and tool duration on top of the tokens."""
    clock = _FakeClock()
    metrics = LLMMetricsAccumulator(_CONTEXT_TOKENS, clock=clock)
    metrics.start_call()
    clock.value = 0.1
    metrics.record_first_token()
    clock.value = 0.2
    snapshot = metrics.record(_ollama_response())

    assert snapshot is not None
    formatted = format_stats_segment(snapshot, verbose=True)
    assert "LLM: 1" in formatted
    assert "TTFT: 100 ms" in formatted
    assert "tool: 0 ms" in formatted
    compact = format_stats_segment(snapshot)
    assert "tok/s" in compact
    assert "token:" in compact
    assert "ctx:" in compact
    assert "token in:" not in compact
    assert "TTFT:" not in compact


def test_callback_publishes_metrics_at_call_start() -> None:
    """After the tools the bar must not stay at LLM 0/0 during the prefill."""
    ui = MagicMock()
    ui.has_live_stream = False
    callback = CallbackUI(ui, max_context=_CONTEXT_TOKENS)

    callback.on_llm_start({}, ["abcd" * 8])

    ui.set_turn_metrics.assert_called_once()
    snapshot = ui.set_turn_metrics.call_args.args[0]
    assert snapshot is not None
    assert snapshot.llm_calls == 1
    assert snapshot.token_input == 8
    assert snapshot.used_context == 8


def test_callback_updates_live_metrics_without_redraw_with_open_panel() -> None:
    """With Live active the tokens update the snapshot; the panel refreshes the bar."""
    ui = MagicMock()
    ui.has_live_stream = True
    callback = CallbackUI(ui, max_context=_CONTEXT_TOKENS)
    callback.on_llm_start({}, ["prompt"])
    ui.set_turn_metrics.reset_mock()

    callback.on_llm_new_token("ciao")

    ui.update_turn_metrics.assert_called_once()
    ui.set_turn_metrics.assert_not_called()
    snapshot = ui.update_turn_metrics.call_args.args[0]
    assert snapshot is not None
    assert snapshot.token_output >= 1
    ui.write_token.assert_called_once_with("ciao")


def test_callback_routes_thinking_to_timeline() -> None:
    """Native CoT goes to `write_thinking`; content closes it and opens the answer."""
    ui = MagicMock()
    ui.has_live_stream = False
    callback = CallbackUI(ui, max_context=_CONTEXT_TOKENS)
    callback.on_llm_start({}, ["prompt"])
    callback.on_llm_new_token("idea", thinking=True)
    ui.write_thinking.assert_called_once_with("idea")
    ui.write_token.assert_not_called()
    callback.on_llm_new_token("ciao")
    ui.close_thinking.assert_called_once()
    ui.write_token.assert_called_once_with("ciao")


def test_callback_ignores_empty_tokens_for_ttft() -> None:
    """Empty strings must not trigger TTFT or the renderer."""
    ui = MagicMock()
    ui.has_live_stream = False
    callback = CallbackUI(ui, max_context=_CONTEXT_TOKENS)
    callback.on_llm_start({}, ["prompt"])
    callback.on_llm_new_token("")
    ui.write_token.assert_not_called()
    ui.write_thinking.assert_not_called()
    assert callback._metrics.snapshot().ttft_ms == 0.0
