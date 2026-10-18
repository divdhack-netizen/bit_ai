"""Validator: write, destination, empty subject and unsupported actions."""

from __future__ import annotations

import pytest

from assistant.turn_schema import TurnSchema
from assistant.turn_validator import (
    MISSING_FILE_MESSAGE,
    ValidationContext,
    validate_turn,
)


def test_workspace_without_target_falls_back_to_explain() -> None:
    """A workspace without a target falls back to a question: search the documents, no write."""
    from assistant.turn_policy import decision_from_schema

    outcome = validate_turn(
        TurnSchema(kind="workspace", subject="fai quello che ti ho chiesto prima"),
        "fai quello che ti ho chiesto prima",
    )
    assert outcome.schema.kind == "explain"
    decision = decision_from_schema(outcome.schema)
    assert decision.path == "react"
    assert decision.write_allowed is False


def test_workspace_with_folder_stays_workspace() -> None:
    """A destination folder makes the workspace actionable: it stays ReAct."""
    outcome = validate_turn(
        TurnSchema(kind="workspace", folder="gioco"),
        "prepara in questa cartella un gioco",
    )
    assert outcome.schema.kind == "workspace"


def test_workspace_with_file_delivery_stays_workspace() -> None:
    """deliver file is enough to open the workspace, which then asks for the name."""
    outcome = validate_turn(
        TurnSchema(kind="workspace", deliver="file"),
        "fammi un documento",
    )
    assert outcome.schema.kind == "workspace"
    assert outcome.schema.clarify == MISSING_FILE_MESSAGE


@pytest.mark.parametrize(
    ("text", "capability"),
    (
        ("mandagli un'email con il riassunto", "email"),
        ("invia un messaggio su WhatsApp a Marco", "email"),
        ("genera un'immagine di un tramonto sul mare", "generazione"),
        ("disegnami un logo per il progetto", "generazione"),
        ("esegui questo comando: rm -rf /tmp/dati", "shell"),
        ("lancia lo script di backup", "shell"),
        ("prenotami un volo per Roma domani", "prenotazione"),
        ("ascolta questo file audio e trascrivilo", "audio"),
    ),
)
def test_unsupported_external_action(text: str, capability: str) -> None:
    """An action the agent cannot perform becomes a local clarification."""
    from assistant.turn_policy import decision_from_schema
    from assistant.turn_validator import unsupported_external_action

    assert unsupported_external_action(text) == capability
    outcome = validate_turn(
        TurnSchema(
            kind="workspace",
            write=True,
            deliver="file",
            paths=("x.md",),
        ),
        text,
    )
    assert outcome.schema.clarify
    assert outcome.schema.kind == "chat"
    assert outcome.schema.write is False
    assert outcome.schema.paths == ()
    assert decision_from_schema(outcome.schema).path == "direct"


@pytest.mark.parametrize(
    "text",
    (
        "scrivi una email di prova in email.md",
        "riassumi questa email e salvala in riassunto.md",
        "crea una cartella per le immagini",
        "ordina la lista e salvala in lista.md",
        "disegna la struttura del database in schema.md",
        "esegui il calcolo e dimmi il risultato",
        "spiegami come prenotare un volo online",
        "spiegami come trascrivere un audio",
        "spiegami come inviare una email",
        "prepara una guida su come creare un pdf in guida.md",
    ),
)
def test_no_false_positive_on_external_actions(text: str) -> None:
    """Informational questions and local tasks stay ordinary tasks."""
    from assistant.turn_validator import unsupported_external_action

    assert unsupported_external_action(text) is None
    outcome = validate_turn(
        TurnSchema(
            kind="workspace",
            write=True,
            deliver="file",
            paths=("x.md",),
        ),
        text,
    )
    assert outcome.schema.clarify == ""


def test_write_without_destination_stays_explain() -> None:
    """write without a path does not open the workspace."""
    outcome = validate_turn(
        TurnSchema(kind="explain", write=True, subject="chiusura"),
        "spiegami come funziona una chiusura",
    )
    assert outcome.schema.kind == "explain"
    assert outcome.schema.write is False


def test_path_does_not_rewrite_kind() -> None:
    """A .py in the schema does not turn a question into a workspace."""
    outcome = validate_turn(
        TurnSchema(
            kind="explain",
            subject="economia",
            paths=("foo.py",),
        ),
        "cosa dice foo.py sull'economia della settimana",
    )
    assert outcome.schema.kind == "explain"
    assert outcome.schema.paths == ("foo.py",)


def test_empty_subject_inherits_only_theme() -> None:
    """The follow-up without a subject reuses the theme. The kind stays the model's."""
    last = TurnSchema(
        kind="explain",
        subject="economia",
    )
    outcome = validate_turn(
        TurnSchema(kind="chat"),
        "e quella di Repubblica?",
        ValidationContext(last=last),
    )
    assert outcome.schema.kind == "chat"
    assert outcome.schema.subject == "economia"


def test_deliver_file_with_name_in_subject_does_not_ask() -> None:
    """The name put in subject is the destination, not a clarification."""
    outcome = validate_turn(
        TurnSchema(
            kind="workspace",
            write=True,
            deliver="file",
            subject="gt.txt",
        ),
        "potresti riportarmi il tutto in file. Crealo e chiamalo gt.txt",
    )
    assert outcome.schema.paths == ("gt.txt",)
    assert outcome.schema.clarify == ""


def test_deliver_file_with_name_only_in_request_does_not_ask() -> None:
    """If the subject is generic, the name in the sentence is the destination."""
    outcome = validate_turn(
        TurnSchema(
            kind="workspace",
            write=True,
            deliver="file",
            subject="creazione_file",
        ),
        "crea un file gt.txt e scrivi tutto lì",
    )
    assert outcome.schema.paths == ("gt.txt",)
    assert outcome.schema.clarify == ""


def test_deliver_file_without_path_asks_for_file() -> None:
    """No write in the root if the document is missing."""
    outcome = validate_turn(
        TurnSchema(
            kind="workspace",
            subject="economia",
            deliver="file",
            write=True,
        ),
        "salva il riepilogo",
    )
    assert outcome.schema.clarify == MISSING_FILE_MESSAGE


def test_relevance_on_explain_stays_explain() -> None:
    """«Più importanti» does not open a roundup if the schema is explain."""
    outcome = validate_turn(
        TurnSchema(kind="explain", subject="termodinamica"),
        "i principi più importanti della termodinamica",
    )
    assert outcome.schema.kind == "explain"


def test_code_in_folder_becomes_path_without_clarification() -> None:
    """A .py written by the user is the destination, not a clarification."""
    outcome = validate_turn(
        TurnSchema(
            kind="workspace",
            write=True,
            deliver="file",
            folder="sort.py",
            subject="programma",
        ),
        "scrivimi un programma Python che ordina una lista e salvalo in sort.py",
    )
    assert outcome.schema.paths == ("sort.py",)
    assert outcome.schema.folder == ""
    assert outcome.schema.clarify == ""


def test_code_only_in_request_becomes_path() -> None:
    """If the subject is generic, the .py in the sentence is the destination."""
    outcome = validate_turn(
        TurnSchema(
            kind="workspace",
            write=True,
            deliver="file",
            subject="ordinamento",
        ),
        "crea il file main.py con l'algoritmo",
    )
    assert outcome.schema.paths == ("main.py",)
    assert outcome.schema.clarify == ""


def test_file_in_folder_becomes_path() -> None:
    """A file name put in folder becomes paths, even if the sentence does not repeat it."""
    outcome = validate_turn(
        TurnSchema(
            kind="workspace",
            write=True,
            deliver="file",
            folder="apple.md",
            subject="novita Apple",
        ),
        "cerca online le novita su Apple e scrivile",
    )
    assert outcome.schema.paths == ("apple.md",)
    assert outcome.schema.folder == ""
    assert outcome.schema.clarify == ""


def test_folder_not_translated() -> None:
    """The folder name stays as written, even if the model translates it."""
    request = (
        "nella cartella Documents crea una cartella dal nome AI. "
        "Al suo interno il file problemi.txt"
    )
    outcome = validate_turn(
        TurnSchema(
            kind="workspace",
            write=True,
            deliver="file",
            folder="Documenti/AI",
            paths=("Documenti/AI/problemi.txt",),
        ),
        request,
    )
    assert outcome.schema.folder == "Documents/AI"


def test_legitimate_folder_stays() -> None:
    """A folder without a file extension is left alone."""
    outcome = validate_turn(
        TurnSchema(
            kind="workspace",
            write=True,
            deliver="file",
            folder="progetto",
            subject="todo",
        ),
        "crea una todo list nella cartella progetto",
    )
    assert outcome.schema.folder == "progetto"
    assert outcome.schema.paths == ()


def test_partial_folder_becomes_full_path() -> None:
    """«AI» alone is not enough: the request names Documents/AI."""
    request = (
        "nella cartella Documents crea una cartella dal nome AI. "
        "Al suo interno il file problemi.txt"
    )
    outcome = validate_turn(
        TurnSchema(
            kind="workspace",
            write=True,
            deliver="file",
            folder="AI",
            paths=("problemi.txt",),
        ),
        request,
    )
    assert outcome.schema.folder == "Documents/AI"
    assert outcome.schema.paths == ("problemi.txt",)


def test_now_explain_stays_explain() -> None:
    """Freshness does not steal an explanation asked for now."""
    outcome = validate_turn(
        TurnSchema(kind="explain", subject="chiusura"),
        "adesso spiegami come funziona una chiusura",
    )
    assert outcome.schema.kind == "explain"
    assert outcome.schema.write is False


def test_read_does_not_write() -> None:
    """Reading a file and saying what it is about does not authorize a write."""
    outcome = validate_turn(
        TurnSchema(
            kind="workspace",
            write=True,
            deliver="file",
            subject="contenuto",
        ),
        "leggi il file rassegna.txt e dimmi di cosa parla",
    )
    assert outcome.schema.write is False
    assert outcome.schema.deliver == "chat"
    assert outcome.schema.paths == ("rassegna.txt",)
    assert outcome.schema.clarify == ""


def test_summary_to_file_writes() -> None:
    """«Riassumi in un file» stays a write: the file is the destination."""
    outcome = validate_turn(
        TurnSchema(
            kind="workspace",
            write=True,
            deliver="file",
            subject="riassunto",
        ),
        "riassumi il documento in y.txt",
    )
    assert outcome.schema.write is True
    assert outcome.schema.deliver == "file"
    assert outcome.schema.paths == ("y.txt",)


def test_recognized_extensions() -> None:
    """Code and documents share the same list of extensions."""
    from assistant.comprehension import PATH_FILE

    for name in ("sort.py", "main.js", "app.tsx", "rassegna.txt", "note.md"):
        assert PATH_FILE.search(name), name
