"""Conservative budgets to avoid overflowing the Ollama context."""

from __future__ import annotations

import math
from dataclasses import dataclass

UTF8_BYTES_PER_ESTIMATED_TOKEN = 3.0
MIN_BYTES_PER_ESTIMATED_TOKEN = 2.0
MAX_BYTES_PER_ESTIMATED_TOKEN = 4.5
DEFAULT_SCRATCHPAD_RESERVE_TOKENS = 768
DEFAULT_SYSTEM_RESERVE_TOKENS = 384
TRUNCATION_MARKER = "\n[…contenuto ridotto per rispettare il contesto…]\n"
_BYTES_PER_TOKEN = UTF8_BYTES_PER_ESTIMATED_TOKEN
_PENDING_PROMPT_BYTES = 0


@dataclass(frozen=True, slots=True)
class PromptBudget:
    """
    Split a window between input, output and operating reserves.

    Attributes:
        context_tokens: Total window allocated by Ollama.
        output_tokens: Maximum answer tokens.
        scratchpad_tokens: Reserve for ReAct Actions and Observations.
        system_tokens: Reserve for template, tool descriptions and markers.
    """

    context_tokens: int
    output_tokens: int
    scratchpad_tokens: int = DEFAULT_SCRATCHPAD_RESERVE_TOKENS
    system_tokens: int = DEFAULT_SYSTEM_RESERVE_TOKENS

    @property
    def input_tokens(self) -> int:
        """
        Compute the budget available to the variable prompt data.

        Returns:
            At least one token, even with an inconsistent configuration.
        """
        reserved_tokens = (
            self.output_tokens
            + self.scratchpad_tokens
            + self.system_tokens
        )
        return max(1, self.context_tokens - reserved_tokens)


def note_prompt_bytes(text: str) -> None:
    """Remember the bytes of the prompt just sent, pending prompt_eval_count."""
    global _PENDING_PROMPT_BYTES
    _PENDING_PROMPT_BYTES = len(text.encode("utf-8")) if text else 0


def calibrate_token_estimate(prompt_eval_count: int) -> None:
    """
    Align bytes per token with the real Ollama count, within a clamp.

    Without a count or without pending bytes the factor stays as before.
    """
    global _BYTES_PER_TOKEN, _PENDING_PROMPT_BYTES
    if prompt_eval_count <= 0 or _PENDING_PROMPT_BYTES <= 0:
        return
    ratio = _PENDING_PROMPT_BYTES / prompt_eval_count
    _BYTES_PER_TOKEN = min(
        MAX_BYTES_PER_ESTIMATED_TOKEN,
        max(MIN_BYTES_PER_ESTIMATED_TOKEN, ratio),
    )
    _PENDING_PROMPT_BYTES = 0


def reset_token_estimate_for_tests() -> None:
    """Reset the estimate to 3 bytes per token. Tests only."""
    global _BYTES_PER_TOKEN, _PENDING_PROMPT_BYTES
    _BYTES_PER_TOKEN = UTF8_BYTES_PER_ESTIMATED_TOKEN
    _PENDING_PROMPT_BYTES = 0


def estimate_tokens(text: str) -> int:
    """
    Estimate tokens without loading a tokenizer into memory.

    The estimate uses UTF-8 bytes divided by the session factor (default 3,
    then calibrated on prompt_eval_count). Ollama's counts remain the source of truth.

    Args:
        text: Text to evaluate, possibly empty.

    Returns:
        Estimated tokens, zero only for empty input.
    """
    if not text:
        return 0
    byte_count = len(text.encode("utf-8"))
    return max(1, math.ceil(byte_count / _BYTES_PER_TOKEN))


def _decode_prefix(payload: bytes, byte_limit: int) -> str:
    """
    Decode a UTF-8 prefix without leaving a split character.

    Args:
        payload: Original bytes.
        byte_limit: Maximum number of bytes to keep.

    Returns:
        Valid Unicode prefix.
    """
    return payload[:byte_limit].decode("utf-8", errors="ignore").rstrip()


def _decode_suffix(payload: bytes, byte_limit: int) -> str:
    """
    Decode a UTF-8 suffix without leaving a split character.

    Args:
        payload: Original bytes.
        byte_limit: Maximum number of bytes to keep.

    Returns:
        Valid Unicode suffix.
    """
    return payload[-byte_limit:].decode("utf-8", errors="ignore").lstrip()


def fit_text(
    text: str,
    max_tokens: int,
    *,
    keep_tail: bool = False,
    marker: str = TRUNCATION_MARKER,
) -> str:
    """
    Reduce a text to the estimated budget, keeping either the head or the tail.

    Args:
        text: Original content.
        max_tokens: Maximum budget; non-positive values produce an empty string.
        keep_tail: If True, keep the recent part, useful for the history.
        marker: Explicit indication of the cut.

    Returns:
        The original text if it fits, otherwise the truncated text with the marker.
    """
    # A zero budget must not leave markers without content.
    if max_tokens <= 0 or not text:
        return ""
    # The common path avoids copies when the content is already within the limit.
    if estimate_tokens(text) <= max_tokens:
        return text
    payload = text.encode("utf-8")
    marker_bytes = marker.encode("utf-8")
    maximum_bytes = math.floor(max_tokens * UTF8_BYTES_PER_ESTIMATED_TOKEN)
    content_bytes = max(0, maximum_bytes - len(marker_bytes))
    # A budget shorter than the marker still keeps a useful fragment.
    if content_bytes <= 0:
        return _decode_suffix(payload, maximum_bytes) if keep_tail else _decode_prefix(
            payload,
            maximum_bytes,
        )
    # History favours the last turns; documents and sources favour the start.
    if keep_tail:
        return f"{marker}{_decode_suffix(payload, content_bytes)}"
    return f"{_decode_prefix(payload, content_bytes)}{marker}"


def fit_observation(text: str, max_tokens: int) -> str:
    """
    Keep the head and tail of a long Observation.

    Args:
        text: Full tool output.
        max_tokens: Budget assigned to the scratchpad.

    Returns:
        The original Observation, or an explicit head+tail window.
    """
    # Short Observations must not be altered or reallocated.
    if estimate_tokens(text) <= max_tokens:
        return text
    # Half head and half tail keep both the structure and the final diagnostics.
    side_tokens = max(1, (max_tokens - estimate_tokens(TRUNCATION_MARKER)) // 2)
    head = fit_text(text, side_tokens, marker="")
    tail = fit_text(text, side_tokens, keep_tail=True, marker="")
    return f"{head.rstrip()}{TRUNCATION_MARKER}{tail.lstrip()}"


# Microcompact marker aligned with Claude Code's clear_tool_uses pattern.
MICROCOMPACT_MARKER_TEMPLATE = "[Observation precedente omessa: {tool} {path}]"
# How many recent observations stay intact.
DEFAULT_MICROCOMPACT_KEEP_RECENT = 2


def microcompact_observations(
    observations: list[tuple[str, str, str]],
    *,
    keep_recent: int = DEFAULT_MICROCOMPACT_KEEP_RECENT,
    force: bool = False,
    token_threshold: int = 1800,
) -> list[tuple[str, str, str]]:
    """
    Replace old read-only observations with placeholders past the threshold.

    Errors, writes and the last ``keep_recent`` stay intact.
    """
    if not observations:
        return []
    total_tokens = sum(estimate_tokens(content) for _, _, content in observations)
    if not force and total_tokens <= token_threshold:
        return list(observations)
    cutoff = max(0, len(observations) - keep_recent)
    compacted: list[tuple[str, str, str]] = []
    for index, (tool_name, path_hint, content) in enumerate(observations):
        lower = content.casefold()
        keep_full = (
            index >= cutoff
            or "errore" in lower
            or "error" in lower
            or tool_name
            in {
                "scrivi_file",
                "modifica_file",
                "riassumi_documento",
                "crea_cartella",
            }
            or "file scritto" in lower
            or "cartella creata" in lower
        )
        if keep_full:
            compacted.append((tool_name, path_hint, content))
            continue
        compacted.append(
            (
                tool_name,
                path_hint,
                MICROCOMPACT_MARKER_TEMPLATE.format(
                    tool=tool_name or "tool",
                    path=path_hint or "-",
                ),
            )
        )
    return compacted


def _scratchpad_text(steps: list[tuple[str, str]]) -> str:
    """Reproduce the ReAct log + Observation format used by LangChain."""
    return "".join(
        f"{log}\nObservation: {observation}\nThought:" for log, observation in steps
    )


def limit_scratchpad_steps(
    steps: list[tuple[str, str, str, str]],
    max_tokens: int,
) -> list[tuple[str, str]]:
    """
    Compress the ReAct steps within a cumulative token cap.

    Args:
        steps: (tool, path, log, observation) tuples in chronological order.
        max_tokens: Scratchpad budget. Non-positive empties the contents.

    Returns:
        (log, observation) pairs in the same order, already within the budget
        when the cap leaves room for at least one marker.
    """
    if not steps:
        return []
    if max_tokens <= 0:
        return [("", "") for _ in steps]
    observations = [(tool, path, observation) for tool, path, _, observation in steps]
    rendered = _scratchpad_text(
        [(log, observation) for _, _, log, observation in steps]
    )
    compacted = microcompact_observations(
        observations,
        force=estimate_tokens(rendered) > max_tokens,
    )
    mutable = [
        (log, content)
        for (_, _, log, _), (_, _, content) in zip(steps, compacted, strict=True)
    ]
    if estimate_tokens(_scratchpad_text(mutable)) <= max_tokens:
        return mutable
    last_log, last_observation = mutable[-1]
    room = max(1, max_tokens // 4)
    mutable = [
        (
            log,
            MICROCOMPACT_MARKER_TEMPLATE.format(tool=tool or "tool", path=path or "-"),
        )
        for tool, path, log, _observation in steps[:-1]
    ]
    mutable.append(
        (
            fit_text(last_log, room, keep_tail=True),
            fit_observation(last_observation, room),
        )
    )
    return mutable
