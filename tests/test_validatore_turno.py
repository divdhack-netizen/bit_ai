"""Validatore: scrittura, destinazione, subject vuoto e azioni non supportate."""

from __future__ import annotations

import pytest

from assistente.schema_turno import SchemaTurno
from assistente.validatore_turno import (
    MESSAGGIO_FILE_MANCANTE,
    ContestoValidazione,
    valida_turno,
)


def test_workspace_senza_target_torna_explain() -> None:
    """workspace senza target torna domanda: si cerca nei documenti, senza scrivere."""
    from assistente.politica_turno import decisione_da_schema

    esito = valida_turno(
        SchemaTurno(kind="workspace", subject="fai quello che ti ho chiesto prima"),
        "fai quello che ti ho chiesto prima",
    )
    assert esito.schema.kind == "explain"
    decisione = decisione_da_schema(esito.schema)
    assert decisione.percorso == "react"
    assert decisione.scrittura_consentita is False


def test_workspace_con_cartella_resta_workspace() -> None:
    """Una cartella di destinazione rende il workspace azionabile: resta ReAct."""
    esito = valida_turno(
        SchemaTurno(kind="workspace", folder="gioco"),
        "prepara in questa cartella un gioco",
    )
    assert esito.schema.kind == "workspace"


def test_workspace_con_consegna_file_resta_workspace() -> None:
    """deliver file basta ad aprire il workspace, che poi chiede il nome."""
    esito = valida_turno(
        SchemaTurno(kind="workspace", deliver="file"),
        "fammi un documento",
    )
    assert esito.schema.kind == "workspace"
    assert esito.schema.clarify == MESSAGGIO_FILE_MANCANTE


@pytest.mark.parametrize(
    ("testo", "capability"),
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
def test_azione_esterna_non_supportata(testo: str, capability: str) -> None:
    """Un'azione che l'agente non può compiere diventa un chiarimento locale."""
    from assistente.politica_turno import decisione_da_schema
    from assistente.validatore_turno import azione_esterna_non_supportata

    assert azione_esterna_non_supportata(testo) == capability
    esito = valida_turno(
        SchemaTurno(
            kind="workspace",
            write=True,
            deliver="file",
            paths=("x.md",),
        ),
        testo,
    )
    assert esito.schema.clarify
    assert esito.schema.kind == "chat"
    assert esito.schema.write is False
    assert esito.schema.paths == ()
    assert decisione_da_schema(esito.schema).percorso == "direct"


@pytest.mark.parametrize(
    "testo",
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
def test_nessun_falso_positivo_su_azioni_esterne(testo: str) -> None:
    """Le domande informative e i compiti locali restano compiti normali."""
    from assistente.validatore_turno import azione_esterna_non_supportata

    assert azione_esterna_non_supportata(testo) is None
    esito = valida_turno(
        SchemaTurno(
            kind="workspace",
            write=True,
            deliver="file",
            paths=("x.md",),
        ),
        testo,
    )
    assert esito.schema.clarify == ""


def test_write_senza_destinazione_resta_explain() -> None:
    """write senza path non apre il workspace."""
    esito = valida_turno(
        SchemaTurno(kind="explain", write=True, subject="chiusura"),
        "spiegami come funziona una chiusura",
    )
    assert esito.schema.kind == "explain"
    assert esito.schema.write is False


def test_path_non_riscrive_il_kind() -> None:
    """Un .py nello schema non trasforma una domanda in workspace."""
    esito = valida_turno(
        SchemaTurno(
            kind="explain",
            subject="economia",
            paths=("foo.py",),
        ),
        "cosa dice foo.py sull'economia della settimana",
    )
    assert esito.schema.kind == "explain"
    assert esito.schema.paths == ("foo.py",)


def test_subject_vuoto_eredita_solo_il_tema() -> None:
    """Il follow-up senza subject riusa il tema. Il kind resta quello del modello."""
    ultimo = SchemaTurno(
        kind="explain",
        subject="economia",
    )
    esito = valida_turno(
        SchemaTurno(kind="chat"),
        "e quella di Repubblica?",
        ContestoValidazione(ultimo=ultimo),
    )
    assert esito.schema.kind == "chat"
    assert esito.schema.subject == "economia"


def test_deliver_file_con_nome_nello_subject_non_chiede() -> None:
    """Il nome messo in subject è la destinazione, non un chiarimento."""
    esito = valida_turno(
        SchemaTurno(
            kind="workspace",
            write=True,
            deliver="file",
            subject="gt.txt",
        ),
        "potresti riportarmi il tutto in file. Crealo e chiamalo gt.txt",
    )
    assert esito.schema.paths == ("gt.txt",)
    assert esito.schema.clarify == ""


def test_deliver_file_con_nome_solo_nella_richiesta_non_chiede() -> None:
    """Se lo subject è generico, il nome nella frase è la destinazione."""
    esito = valida_turno(
        SchemaTurno(
            kind="workspace",
            write=True,
            deliver="file",
            subject="creazione_file",
        ),
        "crea un file gt.txt e scrivi tutto lì",
    )
    assert esito.schema.paths == ("gt.txt",)
    assert esito.schema.clarify == ""


def test_deliver_file_senza_path_chiede_il_file() -> None:
    """Niente scrittura in root se manca il documento."""
    esito = valida_turno(
        SchemaTurno(
            kind="workspace",
            subject="economia",
            deliver="file",
            write=True,
        ),
        "salva il riepilogo",
    )
    assert esito.schema.clarify == MESSAGGIO_FILE_MANCANTE


def test_rilevanza_su_explain_resta_explain() -> None:
    """«Più importanti» non apre una rassegna se lo schema è explain."""
    esito = valida_turno(
        SchemaTurno(kind="explain", subject="termodinamica"),
        "i principi più importanti della termodinamica",
    )
    assert esito.schema.kind == "explain"


def test_codice_in_folder_diventa_path_senza_chiarimento() -> None:
    """Un .py scritto dall'utente è la destinazione, non un chiarimento."""
    esito = valida_turno(
        SchemaTurno(
            kind="workspace",
            write=True,
            deliver="file",
            folder="sort.py",
            subject="programma",
        ),
        "scrivimi un programma Python che ordina una lista e salvalo in sort.py",
    )
    assert esito.schema.paths == ("sort.py",)
    assert esito.schema.folder == ""
    assert esito.schema.clarify == ""


def test_codice_solo_nella_richiesta_diventa_path() -> None:
    """Se lo subject è generico, il .py nella frase è la destinazione."""
    esito = valida_turno(
        SchemaTurno(
            kind="workspace",
            write=True,
            deliver="file",
            subject="ordinamento",
        ),
        "crea il file main.py con l'algoritmo",
    )
    assert esito.schema.paths == ("main.py",)
    assert esito.schema.clarify == ""


def test_file_in_folder_diventa_path() -> None:
    """Un nome file messo in folder diventa paths, anche se la frase non lo ripete."""
    esito = valida_turno(
        SchemaTurno(
            kind="workspace",
            write=True,
            deliver="file",
            folder="apple.md",
            subject="novita Apple",
        ),
        "cerca online le novita su Apple e scrivile",
    )
    assert esito.schema.paths == ("apple.md",)
    assert esito.schema.folder == ""
    assert esito.schema.clarify == ""


def test_cartella_non_tradotta() -> None:
    """Il nome di cartella resta quello scritto, anche se il modello lo traduce."""
    richiesta = (
        "nella cartella Documents crea una cartella dal nome AI. "
        "Al suo interno il file problemi.txt"
    )
    esito = valida_turno(
        SchemaTurno(
            kind="workspace",
            write=True,
            deliver="file",
            folder="Documenti/AI",
            paths=("Documenti/AI/problemi.txt",),
        ),
        richiesta,
    )
    assert esito.schema.folder == "Documents/AI"


def test_folder_legittimo_resta() -> None:
    """Una cartella senza estensione file non viene toccata."""
    esito = valida_turno(
        SchemaTurno(
            kind="workspace",
            write=True,
            deliver="file",
            folder="progetto",
            subject="todo",
        ),
        "crea una todo list nella cartella progetto",
    )
    assert esito.schema.folder == "progetto"
    assert esito.schema.paths == ()


def test_cartella_parziale_diventa_path_completo() -> None:
    """«AI» da solo non basta: la richiesta nomina Documents/AI."""
    richiesta = (
        "nella cartella Documents crea una cartella dal nome AI. "
        "Al suo interno il file problemi.txt"
    )
    esito = valida_turno(
        SchemaTurno(
            kind="workspace",
            write=True,
            deliver="file",
            folder="AI",
            paths=("problemi.txt",),
        ),
        richiesta,
    )
    assert esito.schema.folder == "Documents/AI"
    assert esito.schema.paths == ("problemi.txt",)


def test_adesso_spiegami_resta_explain() -> None:
    """La freschezza non ruba una spiegazione chiesta adesso."""
    esito = valida_turno(
        SchemaTurno(kind="explain", subject="chiusura"),
        "adesso spiegami come funziona una chiusura",
    )
    assert esito.schema.kind == "explain"
    assert esito.schema.write is False


def test_lettura_non_scrive() -> None:
    """Leggere un file e dire di cosa parla non autorizza la scrittura."""
    esito = valida_turno(
        SchemaTurno(
            kind="workspace",
            write=True,
            deliver="file",
            subject="contenuto",
        ),
        "leggi il file rassegna.txt e dimmi di cosa parla",
    )
    assert esito.schema.write is False
    assert esito.schema.deliver == "chat"
    assert esito.schema.paths == ("rassegna.txt",)
    assert esito.schema.clarify == ""


def test_riassunto_in_file_scrive() -> None:
    """«Riassumi in un file» resta una scrittura: il file è la destinazione."""
    esito = valida_turno(
        SchemaTurno(
            kind="workspace",
            write=True,
            deliver="file",
            subject="riassunto",
        ),
        "riassumi il documento in y.txt",
    )
    assert esito.schema.write is True
    assert esito.schema.deliver == "file"
    assert esito.schema.paths == ("y.txt",)


def test_estensioni_riconosciute() -> None:
    """Codice e documenti condividono lo stesso elenco di estensioni."""
    from assistente.comprensione import PATH_FILE

    for nome in ("sort.py", "main.js", "app.tsx", "rassegna.txt", "note.md"):
        assert PATH_FILE.search(nome), nome
