"""Plugin pytest: scrive output/report_test.md a fine sessione."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

_REPORT_PATH = Path(__file__).resolve().parent.parent / "output" / "report_test.md"

# Capability principali (ordine tabella); altri test finiscono in "altro (...)".
_CAPABILITY_PER_TEST = {
    "test_scrivi_file_roundtrip": "scrittura file",
    "test_leggi_documento_contenuto": "lettura file",
    "test_elenca_cartella_contiene_file": "elenco cartella",
    "test_path_vietati_restituiscono_errore": "sandbox path",
    "test_riassumi_documento_path_valido": "riassumi_documento (unit)",
    "test_cerca_testo_trova_riga_e_pagina": "ricerca nel contenuto",
    "test_docx_estratto_con_standard_library": "lettura docx",
    "test_pdf_testo_per_pagina": "lettura pdf",
    "test_immagine_passa_dalla_vision": "lettura immagini",
    "test_e2e_riassunto_file_su_txt": "riassunto file → txt",
    "test_e2e_scrivi_file_txt": "scrittura file e2e txt",
    "test_e2e_risponde_dal_documento": "risposta dal documento",
    "test_e2e_rifiuta_scrittura_non_txt": "scrittura solo .txt",
}

_NOTE_AMBIENTALI = (
    "- E2E richiedono Ollama locale; alias `default` con tag `gemma4:12b` "
    "(fallback `gemma4` per E4B se il tag esplicito manca; `:latest` è normalizzato).",
    "- Path tool confinati alla cartella documenti (sandbox: niente `..` / assoluti).",
    "- Nessuna ricerca web: l'agente risponde solo dai documenti e scrive solo .txt.",
)


class _Esito(TypedDict):
    nodeid: str
    nome: str
    outcome: str
    when: str
    duration: float
    longrepr: str


_ESITI: list[_Esito] = []


def pytest_sessionstart(session) -> None:
    """Azzera buffer esiti all'avvio sessione."""
    _ESITI.clear()


def pytest_runtest_logreport(report) -> None:
    """Accumula esiti call e setup non-pass (skip/fail pre-call)."""
    if report.when == "teardown":
        return
    if report.when == "setup" and report.outcome == "passed":
        return
    _ESITI.append(
        {
            "nodeid": report.nodeid,
            "nome": report.nodeid.rsplit("::", 1)[-1],
            "outcome": report.outcome,
            "when": report.when,
            "duration": float(getattr(report, "duration", 0.0) or 0.0),
            "longrepr": str(report.longrepr) if report.longrepr else "",
        }
    )


def _scegli_esito(esiti: list[_Esito]) -> _Esito:
    """Preferisce la fase call se presente."""
    for esito in esiti:
        if esito["when"] == "call":
            return esito
    return esiti[-1]


def _nota(esito: _Esito) -> str:
    """Nota breve Markdown-safe per la tabella capability."""
    if esito["outcome"] == "passed":
        return "ok"
    raw = (esito.get("longrepr") or esito["outcome"]).replace("|", "/")
    return raw[:120]


def _nome_base_test(nome: str) -> str:
    """Rimuove il suffisso pytest ``[param]`` per matchare `_CAPABILITY_PER_TEST`."""
    return nome.partition("[")[0]


def pytest_sessionfinish(session, exitstatus) -> None:
    """Scrive output/report_test.md aggregato per capability."""
    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    per_cap: dict[str, list[_Esito]] = defaultdict(list)
    for esito in _ESITI:
        chiave = _nome_base_test(esito["nome"])
        capability = _CAPABILITY_PER_TEST.get(chiave, f"altro ({esito['nome']})")
        per_cap[capability].append(esito)

    ordine = list(dict.fromkeys(_CAPABILITY_PER_TEST.values()))
    capacita = ordine + [c for c in per_cap if c not in ordine]
    ora = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    linee = [
        "# Report test agente",
        "",
        f"Generato: {ora}",
        f"Exit status pytest: `{exitstatus}`",
        "",
        "## Esito per capability",
        "",
        "| Capability | Esito | Test | Durata (s) | Note |",
        "|---|---|---|---|---|",
    ]
    for capability in capacita:
        esiti = per_cap.get(capability)
        if not esiti:
            linee.append(f"| {capability} | non eseguito | — | — | nessun test raccolto |")
            continue
        scelto = _scegli_esito(esiti)
        linee.append(
            f"| {capability} | **{scelto['outcome']}** | `{scelto['nome']}` | "
            f"{scelto['duration']:.1f} | {_nota(scelto)} |"
        )

    linee.extend(["", "## Dettaglio test", ""])
    for esito in _ESITI:
        linee.append(
            f"- `{esito['nodeid']}` — {esito['when']}/{esito['outcome']} "
            f"({esito['duration']:.1f}s)"
        )
        if esito["outcome"] == "failed" and esito["longrepr"]:
            linee.append(f"  - errore: {esito['longrepr'].replace(chr(10), ' ')[:400]}")

    linee.extend(["", "## Note ambientali / vincoli", "", *_NOTE_AMBIENTALI, ""])
    _REPORT_PATH.write_text("\n".join(linee), encoding="utf-8")
