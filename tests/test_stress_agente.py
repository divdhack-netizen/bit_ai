"""Stress: route dallo schema, parser ReAct, evaluate_turn, sandbox dei file."""

from __future__ import annotations

import pytest
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException

from assistente.config import SEPARATORE_FILE
from assistente.evaluate_turn import evaluate_turn
from assistente.file_service import WorkspaceFileService
from assistente.politica_turno import decisione_da_schema, decisione_react_aperta, decisione_saluto
from assistente.schema_turno import SchemaTurno
from assistente.react_parser import AssistenteReActParser
from assistente.tools import reset_stato_turno, scrivi_file_da_testo
from tests.conftest import DIR_UNIT_REL

pytestmark = pytest.mark.unit


def test_stress_routing_dallo_schema() -> None:
    """La frase non sceglie il percorso: lo schema sì, e l'assenza di schema apre il ReAct."""
    assert decisione_saluto().percorso == "direct"
    assert decisione_saluto().scrittura_consentita is False
    aperto = decisione_react_aperta()
    assert aperto.percorso == "react"
    assert aperto.scrittura_consentita is True
    spiegazione = decisione_da_schema(SchemaTurno(kind="explain", subject="liste"))
    assert spiegazione.percorso == "react"
    assert spiegazione.scrittura_consentita is False
    chat = decisione_da_schema(SchemaTurno(kind="chat"))
    assert chat.percorso == "direct"
    gioco = decisione_da_schema(
        SchemaTurno(kind="workspace", folder="gioco", write=True),
    )
    assert gioco.percorso == "react"
    assert gioco.scrittura_consentita is True


_PARSER_CASES: tuple[tuple[str, str, str | None], ...] = (
    (
        "Thought: ok\nFinal Answer: 42",
        "finish",
        None,
    ),
    (
        "Thought: salvo\nAction: scrivi_file\nAction Input: a.txt\n---\nbody\n"
        "Final Answer: fatto",
        "action",
        "scrivi_file",
    ),
    (
        "Thought: elenco\nAction: elenca_cartella\nassistente",
        "action",
        "elenca_cartella",
    ),
    (
        "Thought: fine\nAction: Nessuna\nAction Input: n/a",
        "finish",
        None,
    ),
    (
        '{"tool":"elenca_cartella","input":"assistente"}',
        "finish_or_action",
        None,
    ),
    (
        "final answer: ciao in minuscolo",
        "finish",
        None,
    ),
    (
        "Thought: cerco\nAction: cerca_testo\nbalenottera azzurra",
        "action",
        "cerca_testo",
    ),
    (
        "```\nThought: ok\nFinal Answer: hi\n```",
        "finish",
        None,
    ),
    (
        "Thought: solo ragiono senza chiudere",
        "finish",
        None,
    ),
    (
        "<tool>elenca_cartella</tool><input>assistente</input>",
        "finish_or_action",
        None,
    ),
)


def test_stress_parser_react_non_esplode() -> None:
    """Output LLM malformati non devono crashare il turno senza recovery."""
    parser = AssistenteReActParser()
    failures: list[str] = []
    for testo, kind, tool in _PARSER_CASES:
        try:
            esito = parser.parse(testo)
        except OutputParserException as errore:
            failures.append(f"raise {type(errore).__name__} on {testo!r:.80}")
            continue
        if kind == "finish" and not isinstance(esito, AgentFinish):
            failures.append(f"want finish got {type(esito).__name__} {testo!r:.60}")
        elif kind == "action":
            if not isinstance(esito, AgentAction) or (tool and esito.tool != tool):
                failures.append(
                    f"want action {tool} got {type(esito).__name__} "
                    f"{getattr(esito, 'tool', None)} {testo!r:.60}"
                )
        elif kind == "finish_or_action" and not isinstance(esito, (AgentAction, AgentFinish)):
            failures.append(f"unparsed {type(esito).__name__} {testo!r:.60}")
    assert not failures, "parser failures:\n" + "\n".join(failures)


def test_stress_evaluate_turn_falsi_positivi() -> None:
    """«Errore:» didattico non deve marcare incompleto se la scrittura non era richiesta."""
    tutorial = (
        "In HTTP, Errore: 404 significa risorsa assente. "
        "iteration limit non c'entra con questo esempio."
    )
    esito = evaluate_turn(
        tutorial,
        scrittura_richiesta=False,
        path_scritti=[],
    )
    assert esito.incompleto is False
    assert esito.note == ()

    vuoto = evaluate_turn("", scrittura_richiesta=True, path_scritti=[])
    assert vuoto.incompleto is True
    assert vuoto.note


def test_arricchisci_risposta_ignora_le_cartelle() -> None:
    """Una cartella creata non diventa «File scritti su disco»."""
    from assistente.session import _arricchisci_risposta

    fermo = "Agent stopped due to iteration limit or time limit."
    assert _arricchisci_risposta(fermo, []) == fermo
    arricchita = _arricchisci_risposta(fermo, ["note.md"])
    assert "File scritti su disco: note.md" in arricchita


def test_stress_scrivi_file_sandbox_e_contratto(workspace_test_dir) -> None:
    """Path traversal e body vuoto falliscono; unicode e contratto valido passano."""
    reset_stato_turno()
    service = WorkspaceFileService()
    service.begin_turn(True)

    traversal = scrivi_file_da_testo(
        f"../etc/passwd\n{SEPARATORE_FILE}\nroot",
        service=service,
    )
    assert traversal.startswith("Errore:"), traversal

    absolute = scrivi_file_da_testo(
        f"/etc/passwd\n{SEPARATORE_FILE}\nroot",
        service=service,
    )
    assert absolute.startswith("Errore:"), absolute

    empty_body = scrivi_file_da_testo(
        f"{DIR_UNIT_REL}/vuoto.txt\n{SEPARATORE_FILE}\n",
        service=service,
    )
    assert empty_body.startswith("Errore:"), empty_body

    comma_path = scrivi_file_da_testo(
        f"{DIR_UNIT_REL}/a,b.txt\n{SEPARATORE_FILE}\nok",
        service=service,
    )
    assert comma_path.startswith("Errore:"), comma_path

    unicode_path = f"{DIR_UNIT_REL}/città.txt"
    ok = scrivi_file_da_testo(
        f"{unicode_path}\n{SEPARATORE_FILE}\nciao",
        service=service,
    )
    assert ok.startswith("File scritto con successo:"), ok
