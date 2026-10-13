"""Test CLI dry-run, rubrica qualità e confronto del benchmark di inferenza."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from assistente.inference_profiles import GIBIBYTE
from benchmarks.benchmark_inference import (
    _CASES,
    _quality_pass,
    _should_skip_variant,
    aggrega_gruppi,
    main,
    valuta_qualita,
)
from benchmarks.compare_benchmarks import confronta
from benchmarks.compare_benchmarks import main as compare_main

pytestmark = pytest.mark.unit


def test_quality_pass_richiede_tutti_i_frammenti() -> None:
    """Il controllo funzionale è case-insensitive e congiuntivo."""
    assert _quality_pass("Def Clamp(value)", ["def clamp"]) is True
    assert _quality_pass("hello", ["def clamp"]) is False


def test_should_skip_fast_su_memoria_insufficiente() -> None:
    """fast viene saltato se pesi+headroom non entrano."""
    reason = _should_skip_variant(
        "fast",
        total_memory_bytes=8 * GIBIBYTE,
    )
    assert reason is not None
    assert "GB" in reason or "GiB" in reason


def test_main_list_cases(capsys: pytest.CaptureFixture[str]) -> None:
    """--list-cases elenca i casi fissi senza toccare Ollama."""
    assert main(["--list-cases"]) == 0
    out = capsys.readouterr().out
    assert "chat:" in out
    assert "file_write:" in out
    assert "notes_write:" in out
    assert "evidenze_fornite:" in out
    assert "append_note:" not in out
    assert "web" not in out


def test_main_dry_run_json(capsys: pytest.CaptureFixture[str]) -> None:
    """--dry-run mostra la matrice pianificata in JSON."""
    assert main(["--dry-run", "--alias", "default", "--case", "chat", "--repeat", "2"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["planned"]
    assert payload["planned"][0]["case"] == "chat"
    assert payload["planned"][0]["quantization"] == "q4"
    assert payload["planned"][0]["repeat"] == 2


def test_rubrica_coding_esegue_il_codice(tmp_path: Path) -> None:
    """Il check funzionale esegue il codice generato in un interprete separato."""
    corretto = (
        "```python\n"
        "def clamp(value, low, high):\n"
        '    """Limita value in [low, high]."""\n'
        "    return max(low, min(high, value))\n"
        "```"
    )
    sbagliato = "def clamp(value, low, high):\n    return value\n"
    assert valuta_qualita(_CASES["coding"], corretto, workspace=tmp_path).passed
    esito = valuta_qualita(_CASES["coding"], sbagliato, workspace=tmp_path)
    assert not esito.passed
    assert any(f.startswith("codice:") for f in esito.failures)


def test_rubrica_conteggio_e_lunghezza() -> None:
    """Numeri fuori tolleranza e risposte troppo lunghe falliscono con motivo esplicito."""
    caso = _CASES["long_context"]
    assert valuta_qualita(caso, "Circa 120.").passed
    assert not valuta_qualita(caso, "Molte volte.").passed
    troppo_lungo = "x" * 500
    assert "lunghezza: 500 > 400" in valuta_qualita(caso, troppo_lungo).failures


def test_rubrica_llm_calls(tmp_path: Path) -> None:
    """Troppe generazioni per un caso a singola chiamata sono una violazione."""
    caso = _CASES["evidenze_fornite"]
    risposta = "Il pilota citato è Charles Leclerc."
    assert valuta_qualita(caso, risposta, llm_calls=1, workspace=tmp_path).passed
    esito = valuta_qualita(caso, risposta, llm_calls=2, workspace=tmp_path)
    assert "llm_calls: 2 > 1" in esito.failures


def test_rubrica_file_hello(tmp_path: Path) -> None:
    """Il check file legge il contenuto reale sul disco del workspace."""
    caso = _CASES["file_write"]
    assert not valuta_qualita(caso, "Fatto.", workspace=tmp_path).passed
    target = tmp_path / "tmp/benchmark_inference_hello.txt"
    target.parent.mkdir(parents=True)
    target.write_text("benchmark-ok\n", encoding="utf-8")
    assert valuta_qualita(caso, "Fatto.", workspace=tmp_path).passed


def _risultato(alias: str, case: str, **override: object) -> dict[str, object]:
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


def test_aggrega_gruppi_usa_mediane_e_ignora_cold() -> None:
    """Le misure cold non entrano nelle mediane per gruppo."""
    results = [
        _risultato("fast", "chat", total_ms=1000.0),
        _risultato("fast", "chat", total_ms=3000.0),
        _risultato("fast", "chat", total_ms=90000.0, cold=True),
    ]
    gruppi = aggrega_gruppi(results)
    assert len(gruppi) == 1
    assert gruppi[0]["n"] == 2
    assert gruppi[0]["median_total_ms"] == 2000.0


def test_confronta_rifiuta_regressione_qualita_e_accetta_miglioramento() -> None:
    """Il verdetto è per alias: qualità in calo rifiuta, e2e più veloce accetta."""
    prima = {"results": [_risultato("fast", "chat"), _risultato("default", "chat")]}
    dopo = {
        "results": [
            _risultato("fast", "chat", quality_pass=False),
            _risultato("default", "chat", total_ms=4000.0),
        ]
    }
    _rows, verdetti = confronta(prima, dopo)
    per_alias = {v.alias: v for v in verdetti}
    assert not per_alias["fast"].accepted
    assert any("qualità" in r for r in per_alias["fast"].reasons)
    assert per_alias["default"].accepted


def test_confronta_rifiuta_piu_prompt_token_salvo_flag() -> None:
    """Prompt token in aumento rifiutano, a meno di --allow-more-tokens."""
    prima = {"results": [_risultato("fast", "chat", prompt_tokens=300)]}
    dopo = {"results": [_risultato("fast", "chat", prompt_tokens=400)]}
    _rows, verdetti = confronta(prima, dopo)
    assert not verdetti[0].accepted
    _rows, verdetti = confronta(prima, dopo, allow_more_tokens=True)
    assert verdetti[0].accepted


def test_compare_main_markdown(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """La CLI stampa la tabella Markdown e restituisce 0 quando accetta."""
    prima = tmp_path / "prima.json"
    dopo = tmp_path / "dopo.json"
    prima.write_text(json.dumps({"results": [_risultato("fast", "chat")]}), encoding="utf-8")
    dopo.write_text(
        json.dumps({"results": [_risultato("fast", "chat", total_ms=4500.0)]}),
        encoding="utf-8",
    )
    assert compare_main([str(prima), str(dopo), "--markdown"]) == 0
    out = capsys.readouterr().out
    assert "| alias |" in out
    assert "fast: ACCETTA" in out
