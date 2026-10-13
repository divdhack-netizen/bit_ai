"""Micro-benchmark deterministico dei percorsi Python che precedono rete e Ollama."""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

from assistente.comprensione import scheda_locale
from assistente.politica_turno import decisione_da_schema
from assistente.schema_turno import SchemaTurno
from assistente.session import _costruisci_descrizione_task
from assistente.tools import normalizza_path_da_llm

_WARMUP_ITERATIONS = 100
_MEASURED_ITERATIONS = 2_000
_NANOSECONDS_PER_MILLISECOND = 1_000_000
_P95_QUANTILE_INDEX = 94
_QUANTILE_BUCKETS = 100

_REQUEST = "Quanto pesa una balenottera azzurra adulta secondo i documenti?"
_FILE_PATH = "docs/rapporti/balene.pdf"
_SCHEMA = SchemaTurno(
    kind="explain",
    subject="peso balenottera azzurra",
)
_TURNO_DECISIONE = decisione_da_schema(_SCHEMA)


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Tempi aggregati di una singola operazione locale, espressi in millisecondi."""

    name: str
    iterations: int
    median_ms: float
    p95_ms: float


def _measure(name: str, operation: Callable[[], Any]) -> BenchmarkResult:
    """
    Misura un'operazione pura dopo un warm-up che stabilizza import e cache Python.

    Args:
        name: Etichetta stabile usata nel report JSON.
        operation: Callable senza argomenti; il valore restituito viene ignorato.

    Returns:
        Statistiche mediane e p95 sulle iterazioni configurate.
    """
    # Riscalda bytecode e cache interne per non attribuire il costo di startup al percorso caldo.
    for _warmup_iteration in range(_WARMUP_ITERATIONS):
        operation()

    durations_ns: list[int] = []
    # Misura ogni chiamata separatamente per conservare la distribuzione, non solo la media.
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


def _nessun_file(_path: str) -> bool:
    """Esistenza finta: il micro-benchmark non tocca il disco."""
    return False


def _build_react_task() -> str:
    """
    Compila le istruzioni ReAct closed-corpus con input stabile.

    Returns:
        Task completo pronto per l'LLM; non viene effettuata alcuna chiamata esterna.
    """
    return _costruisci_descrizione_task(
        _REQUEST,
        False,
        _TURNO_DECISIONE,
    )


def main() -> None:
    """
    Esegue i micro-benchmark e stampa un documento JSON confrontabile tra revisioni.

    Postconditions:
        Non modifica file e non effettua chiamate di rete o verso Ollama.
    """
    operations: tuple[tuple[str, Callable[[], Any]], ...] = (
        (
            "turn_policy",
            partial(decisione_da_schema, _SCHEMA),
        ),
        (
            "scheda_locale",
            partial(scheda_locale, _REQUEST, _nessun_file),
        ),
        ("react_task", _build_react_task),
        ("file_path", partial(normalizza_path_da_llm, _FILE_PATH)),
    )
    results: list[BenchmarkResult] = []
    # Esegue gli stessi percorsi in ordine stabile per produrre report facilmente confrontabili.
    for operation_name, operation in operations:
        results.append(_measure(operation_name, operation))

    payload = {
        "schema_version": 1,
        "results": [asdict(result) for result in results],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


# Consente l'esecuzione diretta senza introdurre side effect durante l'import nei test.
if __name__ == "__main__":
    main()
