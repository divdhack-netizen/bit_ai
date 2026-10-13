"""Confronta due report di benchmark_inference e produce un verdetto ACCETTA/RIFIUTA."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.benchmark_inference import aggrega_gruppi, aggrega_per_alias

DEFAULT_TOLERANCE = 0.03
# Il TTFT sotto il secondo è dominato dal rumore di scheduling: slack assoluto.
TTFT_ABSOLUTE_SLACK_MS = 60.0
_GROUP_KEY = ("alias", "case", "mode")


@dataclass(frozen=True, slots=True)
class Verdetto:
    """Esito del confronto per alias, con i motivi del rifiuto."""

    alias: str
    accepted: bool
    reasons: tuple[str, ...]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _groups_by_key(payload: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    groups = aggrega_gruppi(payload.get("results", []))
    return {tuple(str(g[k]) for k in _GROUP_KEY): g for g in groups}  # type: ignore[misc]


def _delta_pct(before: float, after: float) -> str:
    if before <= 0:
        return "n/d"
    return f"{(after - before) / before * 100:+.1f}%"


def _fmt(value: float, unit: str = "") -> str:
    return f"{value:.1f}{unit}"


def confronta(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    allow_more_tokens: bool = False,
) -> tuple[list[dict[str, Any]], list[Verdetto]]:
    """
    Confronta i gruppi comuni e deriva il verdetto per alias.

    Args:
        before: Report JSON di riferimento.
        after: Report JSON della modifica in valutazione.
        tolerance: Rumore tollerato su decode tok/s ed e2e (frazione).
        allow_more_tokens: Se True non rifiuta per prompt token o llm_calls in aumento.

    Returns:
        Righe di confronto per gruppo e verdetto per alias.
    """
    groups_before = _groups_by_key(before)
    groups_after = _groups_by_key(after)
    common = sorted(set(groups_before) & set(groups_after), key=lambda k: (k[0], k[2], k[1]))
    rows: list[dict[str, Any]] = []
    for key in common:
        prima, dopo = groups_before[key], groups_after[key]
        rows.append({"key": key, "before": prima, "after": dopo})

    verdicts: list[Verdetto] = []
    per_alias_before = aggrega_per_alias([groups_before[k] for k in common])
    per_alias_after = aggrega_per_alias([groups_after[k] for k in common])
    for alias in sorted(per_alias_before):
        prima = per_alias_before[alias]
        dopo = per_alias_after[alias]
        reasons: list[str] = []
        if dopo["quality_pass_groups"] < prima["quality_pass_groups"]:
            reasons.append(
                f"qualità: gruppi pass {prima['quality_pass_groups']} → {dopo['quality_pass_groups']}"
            )
        for metric_key, label in (
            ("median_decode_tokens_per_s_raw", "decode raw"),
            ("median_decode_tokens_per_s_agent", "decode agente"),
        ):
            if prima[metric_key] > 0 and dopo[metric_key] < prima[metric_key] * (1 - tolerance):
                reasons.append(
                    f"{label}: {prima[metric_key]:.1f} → {dopo[metric_key]:.1f} tok/s"
                )
        if prima["sum_median_total_ms_agent"] > 0 and dopo["sum_median_total_ms_agent"] > prima[
            "sum_median_total_ms_agent"
        ] * (1 + tolerance):
            reasons.append(
                "e2e agente: "
                f"{prima['sum_median_total_ms_agent'] / 1000:.1f}s → "
                f"{dopo['sum_median_total_ms_agent'] / 1000:.1f}s"
            )
        ttft_limit = prima["median_ttft_ms_agent"] * (1 + tolerance) + TTFT_ABSOLUTE_SLACK_MS
        if prima["median_ttft_ms_agent"] > 0 and dopo["median_ttft_ms_agent"] > ttft_limit:
            reasons.append(
                f"TTFT agente: {prima['median_ttft_ms_agent']:.0f} → "
                f"{dopo['median_ttft_ms_agent']:.0f} ms"
            )
        if not allow_more_tokens:
            if dopo["sum_median_prompt_tokens_agent"] > prima["sum_median_prompt_tokens_agent"] + 1:
                reasons.append(
                    "prompt token agente: "
                    f"{prima['sum_median_prompt_tokens_agent']:.0f} → "
                    f"{dopo['sum_median_prompt_tokens_agent']:.0f}"
                )
            if dopo["sum_median_llm_calls_agent"] > prima["sum_median_llm_calls_agent"] + 0.51:
                reasons.append(
                    "llm_calls agente: "
                    f"{prima['sum_median_llm_calls_agent']:.1f} → "
                    f"{dopo['sum_median_llm_calls_agent']:.1f}"
                )
        verdicts.append(Verdetto(alias, not reasons, tuple(reasons)))
    return rows, verdicts


def _render_table(rows: list[dict[str, Any]], *, markdown: bool) -> str:
    header = (
        "alias",
        "mode",
        "case",
        "pass",
        "e2e s",
        "ttft ms",
        "dec tok/s",
        "prompt tok",
        "out tok",
        "llm",
    )
    lines: list[str] = []
    if markdown:
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join("---" for _ in header) + "|")
    else:
        lines.append("  ".join(f"{h:<12}" for h in header))
    for row in rows:
        alias, case, mode = row["key"]
        b, a = row["before"], row["after"]
        cells = (
            alias,
            mode,
            case,
            f"{b['quality_pass_rate']:.0%}→{a['quality_pass_rate']:.0%}",
            f"{b['median_total_ms'] / 1000:.1f}→{a['median_total_ms'] / 1000:.1f} "
            f"({_delta_pct(b['median_total_ms'], a['median_total_ms'])})",
            f"{b['median_ttft_ms']:.0f}→{a['median_ttft_ms']:.0f}",
            f"{b['median_decode_tokens_per_s']:.1f}→{a['median_decode_tokens_per_s']:.1f} "
            f"({_delta_pct(b['median_decode_tokens_per_s'], a['median_decode_tokens_per_s'])})",
            f"{b['median_prompt_tokens']:.0f}→{a['median_prompt_tokens']:.0f}",
            f"{b['median_output_tokens']:.0f}→{a['median_output_tokens']:.0f}",
            f"{b['median_llm_calls']:.0f}→{a['median_llm_calls']:.0f}",
        )
        if markdown:
            lines.append("| " + " | ".join(cells) + " |")
        else:
            lines.append("  ".join(f"{c:<12}" for c in cells))
    return "\n".join(lines)


def _render_verdicts(verdicts: list[Verdetto], *, markdown: bool) -> str:
    lines: list[str] = []
    for verdict in verdicts:
        status = "ACCETTA" if verdict.accepted else "RIFIUTA"
        prefix = "- " if markdown else ""
        lines.append(f"{prefix}{verdict.alias}: {status}")
        for reason in verdict.reasons:
            lines.append(f"{'  - ' if markdown else '    '}{reason}")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Confronto prima/dopo di due report benchmark.")
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--allow-more-tokens", action="store_true")
    parser.add_argument("--markdown", action="store_true", help="Tabella in Markdown")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Entry-point CLI.

    Returns:
        0 se tutti gli alias sono accettati, 1 altrimenti, 2 senza gruppi comuni.
    """
    args = _parse_args(argv)
    rows, verdicts = confronta(
        _load(args.before),
        _load(args.after),
        tolerance=args.tolerance,
        allow_more_tokens=args.allow_more_tokens,
    )
    if not rows:
        print("Nessun gruppo (alias, caso, modalità) in comune tra i due report.", file=sys.stderr)
        return 2
    print(_render_table(rows, markdown=args.markdown))
    print()
    print(_render_verdicts(verdicts, markdown=args.markdown))
    return 0 if all(v.accepted for v in verdicts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
