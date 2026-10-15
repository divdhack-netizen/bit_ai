"""Aggregate per-turn metrics: tokens, TTFT and LLM and tool durations."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rich.cells import cell_len

from assistant.prompt_budget import calibrate_token_estimate

_NANOSECONDS_PER_SECOND = 1_000_000_000
_MILLISECONDS_PER_SECOND = 1_000
# Ollama exposes eval_count only on the `done` chunk; while streaming it is estimated from characters.
_CHARS_PER_TOKEN = 4


def estimate_tokens_from_text(text: str) -> int:
    """
    Estimate the tokens of a fragment when the provider has not closed the stream yet.

    Returns:
        Integer ≥ 1 for non-empty text; 0 if the fragment is empty.
    """
    if not text:
        return 0
    return max(1, (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


@dataclass(frozen=True, slots=True)
class LLMTurnStats:
    """Immutable snapshot shown in the CLI status bar."""

    token_input: int
    token_output: int
    tokens_per_second: float
    used_context: int
    max_context: int
    llm_calls: int
    ttft_ms: float
    llm_duration_ms: float
    tool_duration_ms: float
    load_duration_ms: float = 0.0
    prefill_duration_ms: float = 0.0
    provider_duration_ms: float = 0.0
    thermal_duration_ms: float = 0.0
    rendering_duration_ms: float = 0.0
    end_to_end_duration_ms: float = 0.0
    last_ttft_ms: float = 0.0
    max_ttft_ms: float = 0.0
    peak_context: int = 0

    def as_dict(self) -> dict[str, float | int]:
        """
        Expose the snapshot in JSON-serializable form for benchmarks and logs.

        Returns:
            Flat mapping of every numeric metric of the turn.
        """
        return {
            "token_input": self.token_input,
            "token_output": self.token_output,
            "token_al_secondo": self.tokens_per_second,
            "contesto_usato": self.used_context,
            "contesto_massimo": self.max_context,
            "chiamate_llm": self.llm_calls,
            "ttft_ms": self.ttft_ms,
            "durata_llm_ms": self.llm_duration_ms,
            "durata_tool_ms": self.tool_duration_ms,
            "durata_load_ms": self.load_duration_ms,
            "durata_prefill_ms": self.prefill_duration_ms,
            "durata_provider_ms": self.provider_duration_ms,
            "durata_termica_ms": self.thermal_duration_ms,
            "durata_rendering_ms": self.rendering_duration_ms,
            "durata_end_to_end_ms": self.end_to_end_duration_ms,
            "ttft_ultima_ms": self.last_ttft_ms,
            "ttft_massima_ms": self.max_ttft_ms,
            "contesto_picco": self.peak_context,
        }


def _format_number(value: int) -> str:
    """
    Format integers with a space as the thousands separator.

    Returns:
        A readable number, independent of the process locale.
    """
    return f"{value:,}".replace(",", " ")


def _context_percent(stats: LLMTurnStats) -> int:
    """
    Compute the percentage of context used out of the available maximum.

    Args:
        stats: Turn snapshot with used_context and max_context.

    Returns:
        Integer 0–100; 0 if the maximum is not positive (avoids division by zero).
    """
    # Without a known cap the percentage is undefined: show 0
    if stats.max_context <= 0:
        return 0
    return round(100 * stats.used_context / stats.max_context)


def priority_stats_segments(
    stats: LLMTurnStats,
    *,
    verbose: bool = False,
) -> tuple[str, ...]:
    """
    Expose the metrics in priority order for narrow terminals.

    Default: tok/s, total tokens (in+out), context percentage.
    With verbose: also LLM/TTFT/tool/prefill/load.

    Args:
        stats: Immutable snapshot of the current turn.
        verbose: If True, append the latency segments.

    Returns:
        Tuple of independent fragments, joined with a double space in the bar.
    """
    percent = _context_percent(stats)
    totals = stats.token_input + stats.token_output
    base = (
        f"{stats.tokens_per_second:.1f} tok/s",
        f"token: {_format_number(totals)}",
        f"ctx: {percent}%",
    )
    if not verbose:
        return base
    return base + (
        f"LLM: {stats.llm_calls}/{stats.llm_duration_ms:.0f} ms",
        f"TTFT: {stats.ttft_ms:.0f} ms",
        f"tool: {stats.tool_duration_ms:.0f} ms",
        f"prefill: {stats.prefill_duration_ms:.0f} ms",
        f"load: {stats.load_duration_ms:.0f} ms",
    )


def format_stats_segment(
    stats: LLMTurnStats,
    max_width: int | None = None,
    *,
    verbose: bool = False,
) -> str:
    """
    Turn the turn metrics into a compact line for the terminal.

    If max_width is set, include only the priority segments that fit the
    width (visible cells, no ANSI).

    Args:
        stats: Turn snapshot.
        max_width: Optional limit in visible columns; None = everything.
        verbose: If True, also include the LLM/tool latencies.

    Returns:
        Segment with tok/s, total tokens and ctx % (plus latencies if verbose).
    """
    segments = priority_stats_segments(stats, verbose=verbose)
    # No cap: historical behaviour for tests and non-TTY logs
    if max_width is None:
        return "  ".join(segments)

    # Non-positive width: no metrics (only the thermal prefix remains)
    if max_width <= 0:
        return ""

    pieces: list[str] = []
    used = 0
    # Add segments while double space + piece fit the budget
    for segment in segments:
        separator = 2 if pieces else 0
        cost = separator + cell_len(segment)
        # Skip this and the following ones: the order already enforces priority
        if used + cost > max_width:
            break
        pieces.append(segment)
        used += cost
    return "  ".join(pieces)


def _info_ollama(response: Any) -> dict[str, Any] | None:
    """
    Extract generation_info from a LangChain response without depending on its concrete type.

    Returns:
        Ollama mapping, or None for mocked responses or other providers.
    """
    try:
        info = response.generations[0][0].generation_info
    except (AttributeError, IndexError, TypeError):
        return None
    return info if isinstance(info, dict) else None


class LLMMetricsAccumulator:
    """Per-turn accumulator fed by the LangChain callback and the direct tools."""

    def __init__(
        self,
        max_context: int,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        """
        Initialize empty counters with an injectable monotonic clock.

        Args:
            max_context: Token window of the current LLM profile.
            clock: Clock used for wall-clock durations and deterministic tests.
        """
        self._max_context = max_context
        self._clock = clock
        self._token_input = 0
        self._token_output = 0
        self._token_input_live = 0
        self._token_output_live = 0
        self._eval_duration_ns = 0
        self._load_duration_ns = 0
        self._prompt_eval_duration_ns = 0
        self._provider_total_duration_ns = 0
        self._used_context = 0
        self._peak_context = 0
        self._calls = 0
        self._ttft_total_seconds = 0.0
        self._ttft_samples = 0
        self._ttft_last_seconds = 0.0
        self._ttft_max_seconds = 0.0
        self._llm_total_seconds = 0.0
        self._tool_total_seconds = 0.0
        self._thermal_wait_seconds = 0.0
        self._render_total_seconds = 0.0
        self._turn_started_at = self._clock()
        self._llm_started_at: float | None = None
        self._tool_started_at: float | None = None
        self._first_token_at: float | None = None
        self._first_token_recorded = False

    def start_call(self) -> None:
        """Open an LLM measurement and increment the turn's call count."""
        self._llm_started_at = self._clock()
        self._first_token_recorded = False
        self._first_token_at = None
        self._token_input_live = 0
        self._token_output_live = 0
        self._calls += 1

    def prompt_estimate(self, text: str) -> None:
        """
        Estimate input and context tokens until Ollama publishes prompt_eval_count.

        Args:
            text: Full prompt passed to on_llm_start.
        """
        estimated = estimate_tokens_from_text(text)
        self._token_input_live = estimated
        self._used_context = estimated
        self._peak_context = max(self._peak_context, estimated)

    def record_stream_token(self, token: str) -> None:
        """
        Accumulate an estimate of output tokens to update the bar during the stream.

        Args:
            token: Fragment emitted by on_llm_new_token; empty ones do not count.
        """
        self._token_output_live += estimate_tokens_from_text(token)

    def record_first_token(self) -> None:
        """Record the TTFT only once for the current LLM call."""
        # Duplicate callbacks or tokens outside a call must not skew the average.
        if self._llm_started_at is None or self._first_token_recorded:
            return
        elapsed_seconds = self._clock() - self._llm_started_at
        self._ttft_total_seconds += elapsed_seconds
        self._ttft_samples += 1
        self._ttft_last_seconds = elapsed_seconds
        self._ttft_max_seconds = max(self._ttft_max_seconds, elapsed_seconds)
        self._first_token_recorded = True
        self._first_token_at = self._clock()

    def record(self, response: Any) -> LLMTurnStats | None:
        """
        Accumulate Ollama metadata and close the current wall-clock measurement.

        Returns:
            Updated snapshot, available even if the provider exposes no tokens.
        """
        info = _info_ollama(response)
        # Alternative providers may supply no metadata but still have a duration/call.
        if info is not None:
            prompt_eval = info.get("prompt_eval_count")
            eval_count = info.get("eval_count")
            # Each counter is added only when Ollama reports it.
            if prompt_eval is not None:
                evaluate_count = int(prompt_eval)
                calibrate_token_estimate(evaluate_count)
                self._token_input += evaluate_count
                self._token_input_live = 0
                self._used_context = int(prompt_eval)
                self._peak_context = max(
                    self._peak_context,
                    self._used_context,
                )
            else:
                # Without metadata, keep the character-based estimate already shown live.
                self._token_input += self._token_input_live
                self._token_input_live = 0
            # Produced tokens feed both the total and the aggregate speed.
            if eval_count is not None:
                self._token_output += int(eval_count)
                self._token_output_live = 0
            else:
                self._token_output += self._token_output_live
                self._token_output_live = 0
            eval_duration = info.get("eval_duration")
            # Ollama's native duration excludes transport and feeds tok/s precisely.
            if eval_duration is not None:
                self._eval_duration_ns += int(eval_duration)
            load_duration = info.get("load_duration")
            # Load time separates cold start from warm inference.
            if load_duration is not None:
                self._load_duration_ns += int(load_duration)
            prompt_eval_duration = info.get("prompt_eval_duration")
            # Prefill directly measures the cost of the input data.
            if prompt_eval_duration is not None:
                self._prompt_eval_duration_ns += int(prompt_eval_duration)
            total_duration = info.get("total_duration")
            # The provider duration allows comparison with the application wall clock.
            if total_duration is not None:
                self._provider_total_duration_ns += int(total_duration)
        else:
            self._commit_live_estimates()
        self._finish_llm_timing()
        return self.snapshot()

    def record_llm_error(self) -> LLMTurnStats | None:
        """
        Close the measurement after an error, keeping the estimates already observed.

        Returns:
            Snapshot keeping call, duration and stream tokens up to the failure.
        """
        self._commit_live_estimates()
        self._finish_llm_timing()
        return self.snapshot()

    def start_tool(self) -> None:
        """Start the current tool's timer; agent tools run sequentially."""
        self._tool_started_at = self._clock()

    def record_thermal_wait(self, seconds: float) -> None:
        """
        Accumulate deliberate pauses excluded from the provider TTFT.

        Args:
            seconds: Non-negative monotonic duration of the pause.
        """
        self._thermal_wait_seconds += max(0.0, seconds)

    def record_rendering(self, seconds: float) -> None:
        """
        Accumulate the synchronous cost of the UI callbacks.

        Args:
            seconds: Monotonic duration of rendering one chunk.
        """
        self._render_total_seconds += max(0.0, seconds)

    def end_tool(self) -> LLMTurnStats | None:
        """
        Add the current tool's duration and reset the timer.

        Returns:
            Current snapshot, to update the status bar immediately.
        """
        started_at = self._tool_started_at
        # An end event without a start can come from partial callbacks and is ignored.
        if started_at is not None:
            self._tool_total_seconds += self._clock() - started_at
            self._tool_started_at = None
        return self.snapshot()

    def snapshot(self) -> LLMTurnStats | None:
        """
        Produce a snapshot if at least one LLM or tool has been observed.

        Returns:
            Aggregate stats, or None before any measurable activity.
        """
        # Before the first event the UI must show no fake numbers.
        if not self._calls and self._tool_total_seconds <= 0:
            return None
        token_input = self._token_input + self._token_input_live
        token_output = self._token_output + self._token_output_live
        evaluation_seconds = self._eval_duration_ns / _NANOSECONDS_PER_SECOND
        # During decode, the wall clock since the first token anticipates Ollama's eval_duration.
        if self._first_token_at is not None and self._llm_started_at is not None:
            evaluation_seconds += self._clock() - self._first_token_at
        elif evaluation_seconds <= 0:
            decode_seconds = self._open_llm_duration() - self._ttft_last_seconds
            evaluation_seconds = max(0.0, decode_seconds)
        token_rate = (
            token_output / evaluation_seconds
            if evaluation_seconds > 0
            else 0.0
        )
        average_ttft_seconds = (
            self._ttft_total_seconds / self._ttft_samples
            if self._ttft_samples > 0
            else 0.0
        )
        return LLMTurnStats(
            token_input=token_input,
            token_output=token_output,
            tokens_per_second=token_rate,
            used_context=self._used_context,
            max_context=self._max_context,
            llm_calls=self._calls,
            ttft_ms=average_ttft_seconds * _MILLISECONDS_PER_SECOND,
            llm_duration_ms=self._open_llm_duration() * _MILLISECONDS_PER_SECOND,
            tool_duration_ms=self._tool_total_seconds * _MILLISECONDS_PER_SECOND,
            load_duration_ms=(
                self._load_duration_ns
                / _NANOSECONDS_PER_SECOND
                * _MILLISECONDS_PER_SECOND
            ),
            prefill_duration_ms=(
                self._prompt_eval_duration_ns
                / _NANOSECONDS_PER_SECOND
                * _MILLISECONDS_PER_SECOND
            ),
            provider_duration_ms=(
                self._provider_total_duration_ns
                / _NANOSECONDS_PER_SECOND
                * _MILLISECONDS_PER_SECOND
            ),
            thermal_duration_ms=(
                self._thermal_wait_seconds * _MILLISECONDS_PER_SECOND
            ),
            rendering_duration_ms=(
                self._render_total_seconds * _MILLISECONDS_PER_SECOND
            ),
            end_to_end_duration_ms=(
                (self._clock() - self._turn_started_at)
                * _MILLISECONDS_PER_SECOND
            ),
            last_ttft_ms=(
                self._ttft_last_seconds * _MILLISECONDS_PER_SECOND
            ),
            max_ttft_ms=(
                self._ttft_max_seconds * _MILLISECONDS_PER_SECOND
            ),
            peak_context=self._peak_context,
        )

    def _commit_live_estimates(self) -> None:
        """Move the streaming estimates into the turn's confirmed totals."""
        self._token_input += self._token_input_live
        self._token_input_live = 0
        self._token_output += self._token_output_live
        self._token_output_live = 0

    def _open_llm_duration(self) -> float:
        """Sum the closed calls and the LLM call still in progress, if any."""
        duration = self._llm_total_seconds
        started_at = self._llm_started_at
        if started_at is not None:
            duration += self._clock() - started_at
        return duration

    def _finish_llm_timing(self) -> None:
        """Close the current LLM timer idempotently."""
        started_at = self._llm_started_at
        # Duplicate end/error events must not add the same duration twice.
        if started_at is None:
            return
        self._llm_total_seconds += self._clock() - started_at
        self._llm_started_at = None
        self._first_token_at = None
