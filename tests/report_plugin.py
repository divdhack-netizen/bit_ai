"""Pytest plugin: writes output/report_test.md at the end of the session."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

_REPORT_PATH = Path(__file__).resolve().parent.parent / "output" / "report_test.md"

# Main capabilities (table order); other tests fall into "altro (...)".
_CAPABILITY_PER_TEST = {
    "test_write_file_roundtrip": "scrittura file",
    "test_read_document_content": "lettura file",
    "test_list_folder_contains_file": "elenco cartella",
    "test_forbidden_paths_return_error": "sandbox path",
    "test_summarize_document_valid_path": "riassumi_documento (unit)",
    "test_search_text_finds_line_and_page": "ricerca nel contenuto",
    "test_docx_extracted_with_standard_library": "lettura docx",
    "test_pdf_text_per_page": "lettura pdf",
    "test_image_goes_through_vision": "lettura immagini",
    "test_e2e_file_summary_to_txt": "riassunto file → txt",
    "test_e2e_write_txt_file": "scrittura file e2e txt",
    "test_e2e_answers_from_document": "risposta dal documento",
    "test_e2e_rejects_non_txt_write": "scrittura solo .txt",
}

_ENVIRONMENT_NOTES = (
    "- E2E richiedono Ollama locale; alias `default` con tag `gemma4:12b` "
    "(fallback `gemma4` per E4B se il tag esplicito manca; `:latest` è normalizzato).",
    "- Path tool confinati alla cartella documenti (sandbox: niente `..` / assoluti).",
    "- Nessuna ricerca web: l'agente risponde solo dai documenti e scrive solo .txt.",
)


class _Outcome(TypedDict):
    nodeid: str
    name: str
    outcome: str
    when: str
    duration: float
    longrepr: str


_OUTCOMES: list[_Outcome] = []


def pytest_sessionstart(session) -> None:
    """Reset the outcome buffer at session start."""
    _OUTCOMES.clear()


def pytest_runtest_logreport(report) -> None:
    """Collect call outcomes and non-passing setups (skip/fail before the call)."""
    if report.when == "teardown":
        return
    if report.when == "setup" and report.outcome == "passed":
        return
    _OUTCOMES.append(
        {
            "nodeid": report.nodeid,
            "nome": report.nodeid.rsplit("::", 1)[-1],
            "outcome": report.outcome,
            "when": report.when,
            "duration": float(getattr(report, "duration", 0.0) or 0.0),
            "longrepr": str(report.longrepr) if report.longrepr else "",
        }
    )


def _choose_outcome(outcomes: list[_Outcome]) -> _Outcome:
    """Prefer the call phase when present."""
    for outcome in outcomes:
        if outcome["when"] == "call":
            return outcome
    return outcomes[-1]


def _note(outcome: _Outcome) -> str:
    """Short Markdown-safe note for the capability table."""
    if outcome["outcome"] == "passed":
        return "ok"
    raw = (outcome.get("longrepr") or outcome["outcome"]).replace("|", "/")
    return raw[:120]


def _test_base_name(name: str) -> str:
    """Strip the pytest ``[param]`` suffix to match `_CAPABILITY_PER_TEST`."""
    return name.partition("[")[0]


def pytest_sessionfinish(session, exitstatus) -> None:
    """Write output/report_test.md aggregated by capability."""
    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    per_cap: dict[str, list[_Outcome]] = defaultdict(list)
    for outcome in _OUTCOMES:
        key = _test_base_name(outcome["nome"])
        capability = _CAPABILITY_PER_TEST.get(key, f"altro ({outcome['nome']})")
        per_cap[capability].append(outcome)

    order = list(dict.fromkeys(_CAPABILITY_PER_TEST.values()))
    capacity = order + [c for c in per_cap if c not in order]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# Report test agente",
        "",
        f"Generato: {now}",
        f"Exit status pytest: `{exitstatus}`",
        "",
        "## Esito per capability",
        "",
        "| Capability | Esito | Test | Durata (s) | Note |",
        "|---|---|---|---|---|",
    ]
    for capability in capacity:
        outcomes = per_cap.get(capability)
        if not outcomes:
            lines.append(f"| {capability} | non eseguito | — | — | nessun test raccolto |")
            continue
        chosen = _choose_outcome(outcomes)
        lines.append(
            f"| {capability} | **{chosen['outcome']}** | `{chosen['nome']}` | "
            f"{chosen['duration']:.1f} | {_note(chosen)} |"
        )

    lines.extend(["", "## Dettaglio test", ""])
    for outcome in _OUTCOMES:
        lines.append(
            f"- `{outcome['nodeid']}` — {outcome['when']}/{outcome['outcome']} "
            f"({outcome['duration']:.1f}s)"
        )
        if outcome["outcome"] == "failed" and outcome["longrepr"]:
            lines.append(f"  - errore: {outcome['longrepr'].replace(chr(10), ' ')[:400]}")

    lines.extend(["", "## Note ambientali / vincoli", "", *_ENVIRONMENT_NOTES, ""])
    _REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
