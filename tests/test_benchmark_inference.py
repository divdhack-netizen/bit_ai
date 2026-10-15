"""Tests for the inference benchmark: CLI dry run, quality rubric and comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from assistant.inference_profiles import GIBIBYTE
from benchmarks.benchmark_inference import (
    _CASES,
    _quality_pass,
    _should_skip_variant,
    aggregate_groups,
    main,
    evaluate_quality,
)
from benchmarks.compare_benchmarks import compare
from benchmarks.compare_benchmarks import main as compare_main

pytestmark = pytest.mark.unit


def test_quality_pass_requires_all_fragments() -> None:
    """The functional check is case-insensitive and conjunctive."""
    assert _quality_pass("Def Clamp(value)", ["def clamp"]) is True
    assert _quality_pass("hello", ["def clamp"]) is False


def test_should_skip_fast_on_insufficient_memory() -> None:
    """fast is skipped if weights plus headroom do not fit."""
    reason = _should_skip_variant(
        "fast",
        total_memory_bytes=8 * GIBIBYTE,
    )
    assert reason is not None
    assert "GB" in reason or "GiB" in reason


def test_main_list_cases(capsys: pytest.CaptureFixture[str]) -> None:
    """--list-cases lists the fixed cases without touching Ollama."""
    assert main(["--list-cases"]) == 0
    out = capsys.readouterr().out
    assert "chat:" in out
    assert "file_write:" in out
    assert "notes_write:" in out
    assert "evidenze_fornite:" in out
    assert "append_note:" not in out
    assert "web" not in out


def test_main_dry_run_json(capsys: pytest.CaptureFixture[str]) -> None:
    """--dry-run prints the planned matrix as JSON."""
    assert main(["--dry-run", "--alias", "default", "--case", "chat", "--repeat", "2"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["planned"]
    assert payload["planned"][0]["case"] == "chat"
    assert payload["planned"][0]["quantization"] == "q4"
    assert payload["planned"][0]["repeat"] == 2


def test_coding_rubric_runs_code(tmp_path: Path) -> None:
    """The functional check runs the generated code in a separate interpreter."""
    correct = (
        "```python\n"
        "def clamp(value, low, high):\n"
        '    """Limita value in [low, high]."""\n'
        "    return max(low, min(high, value))\n"
        "```"
    )
    wrong = "def clamp(value, low, high):\n    return value\n"
    assert evaluate_quality(_CASES["coding"], correct, workspace=tmp_path).passed
    outcome = evaluate_quality(_CASES["coding"], wrong, workspace=tmp_path)
    assert not outcome.passed
    assert any(f.startswith("codice:") for f in outcome.failures)


def test_count_and_length_rubric() -> None:
    """Out-of-tolerance numbers and overlong answers fail with an explicit reason."""
    case = _CASES["long_context"]
    assert evaluate_quality(case, "Circa 120.").passed
    assert not evaluate_quality(case, "Molte volte.").passed
    too_long = "x" * 500
    assert "lunghezza: 500 > 400" in evaluate_quality(case, too_long).failures


def test_llm_calls_rubric(tmp_path: Path) -> None:
    """Too many generations for a single-call case are a violation."""
    case = _CASES["evidenze_fornite"]
    answer = "Il pilota citato è Charles Leclerc."
    assert evaluate_quality(case, answer, llm_calls=1, workspace=tmp_path).passed
    outcome = evaluate_quality(case, answer, llm_calls=2, workspace=tmp_path)
    assert "llm_calls: 2 > 1" in outcome.failures


def test_hello_file_rubric(tmp_path: Path) -> None:
    """The file check reads the actual content on the workspace disk."""
    case = _CASES["file_write"]
    assert not evaluate_quality(case, "Fatto.", workspace=tmp_path).passed
    target = tmp_path / "tmp/benchmark_inference_hello.txt"
    target.parent.mkdir(parents=True)
    target.write_text("benchmark-ok\n", encoding="utf-8")
    assert evaluate_quality(case, "Fatto.", workspace=tmp_path).passed


def _result(alias: str, case: str, **override: object) -> dict[str, object]:
    base: dict[str, object] = {
        "alias": alias,
        "case": case,
        "mode": "agent_no_ui",
        "cold": False,
        "quality_pass": True,
        "total_ms": 5000.0,
        "ttft_ms": 400.0,
        "decode_tokens_per_s": 20.0,
        "prefill_ms": 300.0,
        "prompt_tokens": 300,
        "output_tokens_count": 150,
        "llm_calls": 1,
        "rendering_ms": 0.0,
        "quality_failures": [],
    }
    base.update(override)
    return base


def test_aggregate_groups_uses_medians_and_ignores_cold() -> None:
    """Cold measurements do not enter the per-group medians."""
    results = [
        _result("fast", "chat", total_ms=1000.0),
        _result("fast", "chat", total_ms=3000.0),
        _result("fast", "chat", total_ms=90000.0, cold=True),
    ]
    groups = aggregate_groups(results)
    assert len(groups) == 1
    assert groups[0]["n"] == 2
    assert groups[0]["median_total_ms"] == 2000.0


def test_compare_rejects_quality_regression_and_accepts_improvement() -> None:
    """The verdict is per alias: lower quality rejects, faster e2e accepts."""
    before = {"results": [_result("fast", "chat"), _result("default", "chat")]}
    after = {
        "results": [
            _result("fast", "chat", quality_pass=False),
            _result("default", "chat", total_ms=4000.0),
        ]
    }
    _rows, verdicts = compare(before, after)
    per_alias = {v.alias: v for v in verdicts}
    assert not per_alias["fast"].accepted
    assert any("qualità" in r for r in per_alias["fast"].reasons)
    assert per_alias["default"].accepted


def test_compare_rejects_more_prompt_tokens_unless_flag() -> None:
    """Rising prompt tokens reject, unless --allow-more-tokens is set."""
    before = {"results": [_result("fast", "chat", prompt_tokens=300)]}
    after = {"results": [_result("fast", "chat", prompt_tokens=400)]}
    _rows, verdicts = compare(before, after)
    assert not verdicts[0].accepted
    _rows, verdicts = compare(before, after, allow_more_tokens=True)
    assert verdicts[0].accepted


def test_compare_main_markdown(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI prints the Markdown table and returns 0 when it accepts."""
    before = tmp_path / "prima.json"
    after = tmp_path / "dopo.json"
    before.write_text(json.dumps({"results": [_result("fast", "chat")]}), encoding="utf-8")
    after.write_text(
        json.dumps({"results": [_result("fast", "chat", total_ms=4500.0)]}),
        encoding="utf-8",
    )
    assert compare_main([str(before), str(after), "--markdown"]) == 0
    out = capsys.readouterr().out
    assert "| alias |" in out
    assert "fast: ACCETTA" in out
