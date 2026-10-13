"""Path strutturali e prompt: il COMPITO non si compila più dal lessico."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.callbacks import BaseCallbackHandler

from assistente.comprensione import (
    ComprensioneTurno,
    componi_path_in_cartella,
    estrai_cartella_nominata,
    formatta_blocco_comprensione,
    risolvi_destinazione,
)
from assistente.direct_pipeline import PREFISSO_SISTEMA_DIRECT, DirectAnswerPipeline
from assistente.politica_turno import decisione_da_schema
from assistente.schema_turno import SchemaTurno
from assistente.session import _comprensione_dallo_schema, _costruisci_descrizione_task

pytestmark = pytest.mark.unit


def test_cartella_e_path_si_compongono_senza_verbi() -> None:
    """Il nome di cartella è struttura, non un intento di scrittura."""
    testo = (
        "nella cartella Documents crea una cartella dal nome AI. "
        "Al suo interno il file problems.txt"
    )
    assert estrai_cartella_nominata(testo) == "Documents/AI"
    assert componi_path_in_cartella("problems.txt", "Documents/AI") == (
        "Documents/AI/problems.txt"
    )
    assert componi_path_in_cartella("notes/a.txt", "notes") == "notes/a.txt"
    assert componi_path_in_cartella("a.txt", None) == "a.txt"
    assert risolvi_destinazione(
        "problems.txt",
        None,
        "Documents/AI/problems.txt",
    ) == "Documents/AI/problems.txt"


def test_schema_subject_diventa_tema_non_un_copione() -> None:
    """Il tema dello schema entra nel prompt; non c'è un'istruzione di scrivi_file."""
    schema = SchemaTurno(kind="workspace", subject="gioco", folder="gioco")
    comprensione = _comprensione_dallo_schema(schema)
    blocco = formatta_blocco_comprensione(comprensione)
    assert blocco == "COMPITO: gioco"
    assert "scrivi_file" not in blocco
    task = _costruisci_descrizione_task(
        "prepara in questa cartella un piccolo gioco",
        verifica_completa=False,
        decisione=decisione_da_schema(schema),
        comprensione=comprensione,
    )
    assert task.index("COMPITO:") < task.index("RICHIESTA:")


def test_schema_senza_subject_non_inietta_compito() -> None:
    """Senza subject il prompt non inventa un COMPITO."""
    comprensione = _comprensione_dallo_schema(SchemaTurno(kind="explain"))
    assert formatta_blocco_comprensione(comprensione) == ""
    task = _costruisci_descrizione_task(
        "ciao",
        verifica_completa=False,
        decisione=decisione_da_schema(SchemaTurno(kind="chat")),
        comprensione=comprensione,
    )
    assert "COMPITO:" not in task


def test_prefisso_kv_direct_resta_stabile() -> None:
    """PREFISSO_SISTEMA_DIRECT apre il prompt; un tema esplicito arriva dopo lo storico."""
    ui = MagicMock()
    llm = MagicMock()
    llm.last_prompt = ""

    def stream(input: str, config=None, **kwargs):
        del config, kwargs
        llm.last_prompt = input
        yield "ok"

    llm.stream = stream
    pipeline = DirectAnswerPipeline(
        llm_provider=lambda *_args: llm,
        ui=ui,
    )
    comprensione = ComprensioneTurno(compito="storico della conversazione", paths=())
    pipeline.answer(
        request="riassumi",
        model_alias="default",
        history="Utente: prima\nAssistente: dopo",
        context_tokens=4096,
        output_tokens=64,
        callback=MagicMock(spec=BaseCallbackHandler),
        intent_context=formatta_blocco_comprensione(comprensione),
    )
    prompt = llm.last_prompt
    assert prompt.startswith(PREFISSO_SISTEMA_DIRECT)
    assert prompt.index("STORICO CONVERSAZIONE") < prompt.index("COMPITO:")
    assert "COMPITO:" not in prompt[: len(PREFISSO_SISTEMA_DIRECT)]
