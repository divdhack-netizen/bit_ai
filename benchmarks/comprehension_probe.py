"""End-to-end comprehension probe for the turn classifier.

Exercises the real comprehension path (schema prompt → Ollama → JSON
repair → validator → routing) on a corpus of requests written by a user.
Measures how many dimensions (kind, write, path, route) the model gets right,
separating accuracy by category and the cost of JSON repair.

Usage:
    python -m benchmarks.comprehension_probe --alias default
    python -m benchmarks.comprehension_probe --alias default fast --output output/...
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from assistant.comprehension import local_card
from assistant.ollama_runtime import OllamaRuntime
from assistant.turn_policy import _is_local_chat, decision_from_schema
from assistant.turn_schema import (
    NUM_PREDICT_SCHEMA,
    format_schema_prompt,
    repair_json_schema,
    schema_json_ollama,
)
from assistant.turn_validator import ValidationContext, validate_turn


@dataclass(frozen=True, slots=True)
class Case:
    """A user request with the comprehension outcome its meaning implies."""

    category: str
    text: str
    expected_kinds: frozenset[str] | None = None
    expected_write: bool | None = None
    expected_route: str | None = None
    expected_path: str | None = None
    expected_folder: str | None = None


# Corpus: requests as a person would write them, with the intended outcome.
# expected_kinds=None means «not evaluated» (e.g. a greeting, which does not call the schema).
CORPUS: tuple[Case, ...] = (
    Case("saluto", "ciao", expected_route="direct", expected_write=False),
    Case("saluto", "grazie mille", expected_route="direct", expected_write=False),
    Case(
        "domanda",
        "quanto pesa una balenottera azzurra adulta?",
        expected_kinds=frozenset({"explain"}),
        expected_route="react",
        expected_write=False,
    ),
    Case(
        "domanda",
        "qual è la differenza tra misticeti e odontoceti?",
        expected_kinds=frozenset({"explain"}),
        expected_route="react",
        expected_write=False,
    ),
    Case(
        "domanda",
        "cosa dicono i documenti sulle rotte migratorie?",
        expected_kinds=frozenset({"explain", "workspace"}),
        expected_route="react",
        expected_write=False,
    ),
    Case(
        "lettura",
        "leggi il file avvistamenti.pdf e dimmi di cosa parla",
        expected_kinds=frozenset({"explain", "workspace", "chat"}),
        expected_route="react",
        expected_write=False,
        expected_path="avvistamenti.pdf",
    ),
    Case(
        "lettura",
        "cosa mostra la foto coda.jpg?",
        expected_kinds=frozenset({"explain", "workspace"}),
        expected_route="react",
        expected_write=False,
        expected_path="coda.jpg",
    ),
    Case(
        "scrittura",
        "riassumi report.docx e salva il riassunto in sintesi.txt",
        expected_kinds=frozenset({"workspace"}),
        expected_route="react",
        expected_write=True,
        expected_path="sintesi.txt",
    ),
    Case(
        "scrittura",
        "nella cartella note crea il file domande.txt con le domande aperte",
        expected_kinds=frozenset({"workspace"}),
        expected_route="react",
        expected_write=True,
        expected_path="domande.txt",
        expected_folder="note",
    ),
    Case(
        "vincolo_negativo",
        "dimmi cosa mangiano le orche, niente file, solo in chat",
        expected_kinds=frozenset({"chat", "explain"}),
        expected_route="react",
        expected_write=False,
    ),
)


@dataclass
class Outcome:
    """Comprehension of a single case on a single alias."""

    category: str
    text: str
    outcome: str
    stated_kind: str = ""
    kind_ok: bool | None = None
    stated_route: str = ""
    route_ok: bool | None = None
    stated_write: bool = False
    write_ok: bool | None = None
    path_ok: bool | None = None
    folder_ok: bool | None = None
    repaired: bool = False
    latency_s: float = 0.0
    exceeded: bool = False
    detail: str = ""

    def evaluated(self) -> bool:
        """True if at least one dimension was expected (scored case)."""
        return any(
            flag is not None
            for flag in (
                self.kind_ok,
                self.route_ok,
                self.write_ok,
                self.path_ok,
                self.folder_ok,
            )
        )


def _evaluate(
    case: Case,
    outcome: str,
    schema: object | None,
    route: str,
    repaired: bool,
    latency: float,
) -> Outcome:
    """Compare the observed outcome with the case expectations, dimension by dimension."""
    case_outcome = Outcome(
        category=case.category,
        text=case.text,
        outcome=outcome,
        stated_route=route,
        repaired=repaired,
        latency_s=round(latency, 2),
    )
    if schema is None:
        case_outcome.route_ok = (
            None if case.expected_route is None else route == case.expected_route
        )
        case_outcome.write_ok = None if case.expected_write is None else case.expected_write is False
        case_outcome.exceeded = bool(case_outcome.route_ok) and (
            case.expected_write is None or case.expected_write is False
        )
        return case_outcome

    kind = str(getattr(schema, "kind", ""))
    write = bool(getattr(schema, "write", False) or getattr(schema, "deliver", "") == "file")
    paths = tuple(getattr(schema, "paths", ()))
    folder = str(getattr(schema, "folder", ""))
    case_outcome.stated_kind = kind
    case_outcome.stated_write = write

    if case.expected_kinds is not None:
        case_outcome.kind_ok = kind in case.expected_kinds
    if case.expected_route is not None:
        case_outcome.route_ok = route == case.expected_route
    if case.expected_write is not None:
        case_outcome.write_ok = write == case.expected_write
    if case.expected_path is not None:
        expected = case.expected_path.casefold()
        case_outcome.path_ok = any(p.casefold().endswith(expected) for p in paths)
    if case.expected_folder is not None:
        case_outcome.folder_ok = folder.replace("\\", "/").casefold() == (
            case.expected_folder.casefold()
        ) or case.expected_folder.casefold() in folder.casefold()

    evaluated = [
        flag
        for flag in (
            case_outcome.kind_ok,
            case_outcome.route_ok,
            case_outcome.write_ok,
            case_outcome.path_ok,
            case_outcome.folder_ok,
        )
        if flag is not None
    ]
    failed = [i for i, flag in enumerate(evaluated) if not flag]
    case_outcome.exceeded = all(evaluated) and bool(evaluated)
    if failed:
        case_outcome.detail = "fallite: " + ", ".join(str(i) for i in failed)
    return case_outcome


def run_case(
    runtime: OllamaRuntime,
    alias: str,
    case: Case,
) -> Outcome:
    """Fill in a case's card with the real model and evaluate it."""
    # The exact greeting skips the schema in production too: direct local route.
    if _is_local_chat(case.text):
        return _evaluate(case, "schema_skipped_saluto", None, "direct", False, 0.0)
    # As in production: the local card decides clear cases without the LLM.
    start = time.perf_counter()
    local = local_card(case.text, lambda _path: False)
    if local is not None:
        schema = validate_turn(local, case.text, ValidationContext()).schema
        return _evaluate(
            case,
            "locale",
            schema,
            decision_from_schema(schema).path,
            False,
            time.perf_counter() - start,
        )

    llm = runtime.get_llm(
        alias,
        runtime.session_context(alias),
        NUM_PREDICT_SCHEMA,
        reasoning=False,
    )
    prompt = format_schema_prompt(case.text, "2026-09-28")
    start = time.perf_counter()
    answer = llm.invoke(
        prompt,
        format=schema_json_ollama(),
        num_predict=NUM_PREDICT_SCHEMA,
        reasoning=False,
        temperature=0.0,  # same as production (Phase 7): measures the actual sampling
    )
    latency = time.perf_counter() - start
    text = answer if isinstance(answer, str) else ""
    repair = repair_json_schema(text)
    if repair.schema is None:
        return _evaluate(case, "schema_fallback", None, "", True, latency)

    validation = validate_turn(repair.schema, case.text, ValidationContext())
    schema = validation.schema
    decision = decision_from_schema(schema)
    return _evaluate(
        case,
        "schema",
        schema,
        decision.path,
        repair.repaired,
        latency,
    )


def _recap(alias: str, outcomes: list[Outcome], repeat: int) -> dict[str, object]:
    """Aggregate accuracy by dimension, category and kind stability."""
    scored = [e for e in outcomes if e.evaluated()]
    by_dimension: dict[str, dict[str, int]] = {}
    for name, field in (
        ("kind", "kind_ok"),
        ("route", "route_ok"),
        ("scrittura", "write_ok"),
        ("path", "path_ok"),
        ("cartella", "cartella_ok"),
    ):
        evaluated = [getattr(e, field) for e in outcomes if getattr(e, field) is not None]
        if evaluated:
            by_dimension[name] = {
                "ok": sum(1 for v in evaluated if v),
                "tot": len(evaluated),
            }
    by_category: dict[str, dict[str, int]] = {}
    for outcome in scored:
        entry = by_category.setdefault(outcome.category, {"ok": 0, "tot": 0})
        entry["tot"] += 1
        if outcome.exceeded:
            entry["ok"] += 1
    # Stability per request: all repetitions equal, and a constant kind
    by_text: dict[str, list[Outcome]] = {}
    for outcome in scored:
        by_text.setdefault(outcome.text, []).append(outcome)
    stable = sum(
        1
        for run in by_text.values()
        if sum(1 for e in run if e.exceeded) == len(run)
    )
    constant_kind = sum(
        1
        for run in by_text.values()
        if len({e.stated_kind for e in run}) == 1
    )
    total_ok = sum(1 for e in scored if e.exceeded)
    return {
        "alias": alias,
        "ripetizioni": repeat,
        "casi": len(by_text),
        "scored": len(scored),
        "superati": total_ok,
        "accuratezza": round(total_ok / len(scored), 3) if scored else 0.0,
        "casi_stabili": stable,
        "kind_costante": constant_kind,
        "fallback_schema": sum(1 for e in outcomes if e.outcome == "schema_fallback"),
        "risolti_in_locale": sum(1 for e in outcomes if e.outcome == "locale"),
        "riparati": sum(1 for e in outcomes if e.repaired),
        "latenza_media_s": round(
            sum(e.latency_s for e in outcomes) / len(outcomes), 2
        )
        if outcomes
        else 0.0,
        "per_dimensione": by_dimension,
        "per_categoria": by_category,
    }


def main() -> None:
    """Run the corpus on the requested aliases and print/report the summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alias",
        nargs="+",
        default=["default"],
        help="Alias modello da provare (fast, default, quality).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Tronca il corpus.")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Ripetizioni per caso: misura la stabilità della comprensione.",
    )
    parser.add_argument("--output", type=Path, default=None, help="JSON di output.")
    args = parser.parse_args()

    corpus = CORPUS[: args.limit] if args.limit else CORPUS
    runtime = OllamaRuntime(manage_residency=False)
    report: dict[str, object] = {
        "corpus": len(corpus),
        "repeat": args.repeat,
        "alias": {},
    }

    for alias in args.alias:
        outcomes: list[Outcome] = []
        print(f"\n=== alias {alias} (repeat={args.repeat}) ===")
        for lap in range(args.repeat):
            for case in corpus:
                outcome = run_case(runtime, alias, case)
                outcomes.append(outcome)
                state = "OK " if outcome.exceeded else "KO "
                print(
                    f"  {state}[r{lap}][{case.category}] "
                    f"{outcome.stated_kind or outcome.outcome} "
                    f"route={outcome.stated_route} {outcome.latency_s:.1f}s :: {case.text[:50]}"
                )
        recap = _recap(alias, outcomes, args.repeat)
        report["alias"][alias] = {
            "riepilogo": recap,
            "esiti": [asdict(e) for e in outcomes],
        }
        print(
            f"  -> accuratezza {recap['accuratezza']} "
            f"({recap['superati']}/{recap['scored']}) "
            f"| casi stabili {recap['casi_stabili']}/{recap['casi']}"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nReport scritto in {args.output}")


if __name__ == "__main__":
    main()
