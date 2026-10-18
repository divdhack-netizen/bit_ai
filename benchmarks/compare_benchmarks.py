"""Compare two benchmark_inference reports and produce an ACCETTA/RIFIUTA verdict."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.benchmark_inference import aggregate_groups, aggregate_by_alias

DEFAULT_TOLERANCE = 0.03
# A sub-second TTFT is dominated by scheduling noise: absolute slack.
TTFT_ABSOLUTE_SLACK_MS = 60.0
_GROUP_KEY = ("alias", "case", "mode")


@dataclass(frozen=True, slots=True)
class Verdict:
    """Per-alias comparison outcome, with the rejection reasons."""

    alias: str
    accepted: bool
    reasons: tuple[str, ...]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _groups_by_key(payload: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    groups = aggregate_groups(payload.get("results", []))
    return {tuple(str(g[k]) for k in _GROUP_KEY): g for g in groups}  # type: ignore[misc]


def _delta_pct(before: float, after: float) -> str:
    if before <= 0:
        return "n/d"
    return f"{(after - before) / before * 100:+.1f}%"


def _fmt(value: float, unit: str = "") -> str:
    return f"{value:.1f}{unit}"


def compare(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    allow_more_tokens: bool = False,
) -> tuple[list[dict[str, Any]], list[Verdict]]:
    """
    Compare the common groups and derive the verdict per alias.

    Args:
        before: Reference JSON report.
        after: JSON report of the change under evaluation.
        tolerance: Tolerated noise on decode tok/s and e2e (fraction).
        allow_more_tokens: If True, do not reject for rising prompt tokens or llm_calls.

    Returns:
        Comparison rows per group and the verdict per alias.
    """
    groups_before = _groups_by_key(before)
    groups_after = _groups_by_key(after)
    common = sorted(set(groups_before) & set(groups_after), key=lambda k: (k[0], k[2], k[1]))
    rows: list[dict[str, Any]] = []
    for key in common:
        group_before_item, group_after_item = groups_before[key], groups_after[key]
        rows.append({"key": key, "before": group_before_item, "after": group_after_item})

    verdicts: list[Verdict] = []
    per_alias_before = aggregate_by_alias([groups_before[k] for k in common])
    per_alias_after = aggregate_by_alias([groups_after[k] for k in common])
    for alias in sorted(per_alias_before):
        group_before_item = per_alias_before[alias]
        group_after_item = per_alias_after[alias]
        reasons: list[str] = []
        if group_after_item["quality_pass_groups"] < group_before_item["quality_pass_groups"]:
            reasons.append(
                f"qualità: gruppi pass {group_before_item['quality_pass_groups']} → {group_after_item['quality_pass_groups']}"
            )
        for metric_key, label in (
            ("median_decode_tokens_per_s_raw", "decode raw"),
            ("median_decode_tokens_per_s_agent", "decode agente"),
        ):
            if group_before_item[metric_key] > 0 and group_after_item[metric_key] < group_before_item[metric_key] * (1 - tolerance):
                reasons.append(
                    f"{label}: {group_before_item[metric_key]:.1f} → {group_after_item[metric_key]:.1f} tok/s"
                )
        if group_before_item["sum_median_total_ms_agent"] > 0 and group_after_item["sum_median_total_ms_agent"] > group_before_item[
            "sum_median_total_ms_agent"
        ] * (1 + tolerance):
            reasons.append(
                "e2e agente: "
                f"{group_before_item['sum_median_total_ms_agent'] / 1000:.1f}s → "
                f"{group_after_item['sum_median_total_ms_agent'] / 1000:.1f}s"
            )
        ttft_limit = group_before_item["median_ttft_ms_agent"] * (1 + tolerance) + TTFT_ABSOLUTE_SLACK_MS
        if group_before_item["median_ttft_ms_agent"] > 0 and group_after_item["median_ttft_ms_agent"] > ttft_limit:
            reasons.append(
                f"TTFT agente: {group_before_item['median_ttft_ms_agent']:.0f} → "
                f"{group_after_item['median_ttft_ms_agent']:.0f} ms"
            )
        if not allow_more_tokens:
            if group_after_item["sum_median_prompt_tokens_agent"] > group_before_item["sum_median_prompt_tokens_agent"] + 1:
                reasons.append(
                    "prompt token agente: "
                    f"{group_before_item['sum_median_prompt_tokens_agent']:.0f} → "
                    f"{group_after_item['sum_median_prompt_tokens_agent']:.0f}"
                )
            if group_after_item["sum_median_llm_calls_agent"] > group_before_item["sum_median_llm_calls_agent"] + 0.51:
                reasons.append(
                    "llm_calls agente: "
                    f"{group_before_item['sum_median_llm_calls_agent']:.1f} → "
                    f"{group_after_item['sum_median_llm_calls_agent']:.1f}"
                )
        verdicts.append(Verdict(alias, not reasons, tuple(reasons)))
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


def _render_verdicts(verdicts: list[Verdict], *, markdown: bool) -> str:
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
    CLI entry point.

    Returns:
        0 if every alias is accepted, 1 otherwise, 2 with no common groups.
    """
    args = _parse_args(argv)
    rows, verdicts = compare(
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
