"""Tests for input budgeting and Observation/text trimming."""

from __future__ import annotations

import pytest

from assistant.prompt_budget import (
    PromptBudget,
    calibrate_token_estimate,
    estimate_tokens,
    fit_observation,
    fit_text,
    limit_scratchpad_steps,
    note_prompt_bytes,
    reset_token_estimate_for_tests,
)

pytestmark = pytest.mark.unit


def test_estimate_tokens_empty_is_zero() -> None:
    """Empty text → 0 estimated tokens."""
    assert estimate_tokens("") == 0


def test_estimate_tokens_conservative_on_unicode() -> None:
    """Multibyte characters weigh more than raw codepoints."""
    assert estimate_tokens("àèìòù") >= 2


def test_prompt_budget_reserves_output_and_scratchpad() -> None:
    """The input budget leaves mandatory room for output and scratchpad."""
    budget = PromptBudget(context_tokens=4096, output_tokens=768, scratchpad_tokens=768)
    assert budget.input_tokens == 4096 - 768 - 768 - budget.system_tokens


def test_fit_text_inserts_marker_when_truncating() -> None:
    """Long text is reduced with an explicit marker."""
    text = "alfa " * 400
    reduced = fit_text(text, max_tokens=20)
    assert len(reduced) < len(text)
    assert "contenuto ridotto" in reduced


def test_limit_scratchpad_respects_budget() -> None:
    """Old Observations become placeholders and the text stays under the cap."""
    steps = [
        ("leggi_documento", "a.py", "Thought: leggo", "HEAD " + ("x" * 4000) + " TAIL"),
        ("elenca_cartella", "src", "Thought: elenco", "uno\ndue\ntre\n" * 80),
        ("scrivi_file", "b.py", "Thought: scrivo", "File scritto con successo: b.py"),
        ("leggi_documento", "b.py", "Thought: rileggo", "contenuto finale"),
    ]
    limited = limit_scratchpad_steps(steps, max_tokens=120)
    assert len(limited) == len(steps)
    assert "omessa" in limited[0][1] or "contenuto ridotto" in limited[0][1]
    assert "File scritto" in limited[2][1] or "omessa" in limited[2][1]
    text = "".join(f"{log}\nObservation: {obs}\n" for log, obs in limited)
    assert estimate_tokens(text) <= 180


def test_token_calibration_stays_in_clamp() -> None:
    """The real ratio updates the estimate without leaving 2.0–4.5 bytes per token."""
    reset_token_estimate_for_tests()
    note_prompt_bytes("a" * 1000)
    calibrate_token_estimate(100)
    assert estimate_tokens("a" * 100) == 23
    note_prompt_bytes("a" * 1000)
    calibrate_token_estimate(10000)
    assert estimate_tokens("abcd") >= 1
    reset_token_estimate_for_tests()
    assert estimate_tokens("abcd") == 2


def test_fit_observation_keeps_head_and_tail() -> None:
    """Long Observations keep the start and end, which are useful for debugging."""
    text = "HEAD-" + ("x" * 5000) + "-TAIL"
    reduced = fit_observation(text, max_tokens=40)
    assert reduced.startswith("HEAD-")
    assert reduced.endswith("-TAIL")
    assert "contenuto ridotto" in reduced
