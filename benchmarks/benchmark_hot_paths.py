"""Deterministic micro-benchmark of the Python paths that run before network and Ollama."""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

from assistant.comprehension import local_card
from assistant.turn_policy import decision_from_schema
from assistant.turn_schema import TurnSchema
from assistant.session import _build_task_description
from assistant.tools import normalize_llm_path

_WARMUP_ITERATIONS = 100
_MEASURED_ITERATIONS = 2_000
_NANOSECONDS_PER_MILLISECOND = 1_000_000
_P95_QUANTILE_INDEX = 94
_QUANTILE_BUCKETS = 100

_REQUEST = "Quanto pesa una balenottera azzurra adulta secondo i documenti?"
_FILE_PATH = "docs/rapporti/balene.pdf"
_SCHEMA = TurnSchema(
    kind="explain",
    subject="peso balenottera azzurra",
)
_DECISION_TURN = decision_from_schema(_SCHEMA)


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Aggregate timings for a single local operation, in milliseconds."""

    name: str
    iterations: int
    median_ms: float
    p95_ms: float


def _measure(name: str, operation: Callable[[], Any]) -> BenchmarkResult:
    """
    Measure a pure operation after a warm-up that stabilizes imports and Python caches.

    Args:
        name: Stable label used in the JSON report.
        operation: Zero-argument callable; its return value is ignored.

    Returns:
        Median and p95 statistics over the configured iterations.
    """
    # Warm up bytecode and internal caches so startup cost is not charged to the hot path.
    for _warmup_iteration in range(_WARMUP_ITERATIONS):
        operation()

    durations_ns: list[int] = []
    # Measure each call separately to keep the distribution, not just the average.
    for _measured_iteration in range(_MEASURED_ITERATIONS):
        started_ns = time.perf_counter_ns()
        operation()
        durations_ns.append(time.perf_counter_ns() - started_ns)

    quantiles_ns = statistics.quantiles(
        durations_ns,
        n=_QUANTILE_BUCKETS,
        method="inclusive",
    )
    return BenchmarkResult(
        name=name,
        iterations=_MEASURED_ITERATIONS,
        median_ms=statistics.median(durations_ns) / _NANOSECONDS_PER_MILLISECOND,
        p95_ms=quantiles_ns[_P95_QUANTILE_INDEX] / _NANOSECONDS_PER_MILLISECOND,
    )


def _no_file(_path: str) -> bool:
    """Fake existence check: the micro-benchmark does not touch the disk."""
    return False


def _build_react_task() -> str:
    """
    Build the closed-corpus ReAct instructions with a stable input.

    Returns:
        Complete task ready for the LLM; no external call is made.
    """
    return _build_task_description(
        _REQUEST,
        False,
        _DECISION_TURN,
    )


def main() -> None:
    """
    Run the micro-benchmarks and print a JSON document comparable across revisions.

    Postconditions:
        Modifies no files and makes no network or Ollama calls.
    """
    operations: tuple[tuple[str, Callable[[], Any]], ...] = (
        (
            "turn_policy",
            partial(decision_from_schema, _SCHEMA),
        ),
        (
            "scheda_locale",
            partial(local_card, _REQUEST, _no_file),
        ),
        ("react_task", _build_react_task),
        ("file_path", partial(normalize_llm_path, _FILE_PATH)),
    )
    results: list[BenchmarkResult] = []
    # Run the same paths in a stable order so the reports are easy to compare.
    for operation_name, operation in operations:
        results.append(_measure(operation_name, operation))

    payload = {
        "schema_version": 1,
        "results": [asdict(result) for result in results],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


# Allow direct execution without side effects at import time in tests.
if __name__ == "__main__":
    main()
