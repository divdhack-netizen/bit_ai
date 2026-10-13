"""Il modello propone la scheda; il validatore fissa scrittura e destinazione."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from assistente.comprensione import (
    CUE_DESTINAZIONE_FILE,
    CUE_LETTURA,
    CUE_SALVATAGGIO,
    ESTENSIONI_FILE,
    PATH_FILE,
    azione_esterna_non_supportata,
    estrai_cartella_nominata,
)
from assistente.schema_turno import SchemaTurno

MESSAGGIO_FILE_MANCANTE = "in quale file?"

_ESTENSIONE_DOCUMENTO = re.compile(
    rf"\.(?:{'|'.join(ESTENSIONI_FILE)})$",
    re.IGNORECASE,
)
_KINDS_MORBIDI = frozenset({"chat", "explain"})

# Chiarimenti: cosa non si può fare e cosa si può offrire al posto suo.
_MESSAGGI_AZIONE_NON_SUPPORTATA: dict[str, str] = {
    "email": (
        "Non posso inviare email o messaggi. Posso però prepararti il testo da "
        "mandare: dimmi destinatario e contenuto e te lo scrivo in chat o in un file."
    ),
    "generazione": (
        "Non posso generare immagini, video o PDF. Posso però descriverti la scena "
        "a parole o prepararti il contenuto come testo, anche in un file .txt."
    ),
    "shell": (
        "Non posso eseguire comandi o script nel terminale. Posso però spiegarti il "
        "comando o prepararti uno script da eseguire tu."
    ),
    "prenotazione": (
        "Non posso prenotare o acquistare nulla. Posso però prepararti i dati e i "
        "passaggi da seguire."
    ),
    "audio": (
        "Non posso ascoltare o trascrivere file audio. Incollami il testo e ci "
        "lavoro volentieri."
    ),
}


@dataclass(frozen=True, slots=True)
class ContestoValidazione:
    """Stato sessione che il validatore può leggere, senza chiamare LLM."""

    ultimo: SchemaTurno | None = None
    documento_precedente: str | None = None


@dataclass(frozen=True, slots=True)
class EsitoValidazione:
    """Scheda pronta per l'executor. Se clarify è piena il turno si ferma."""

    schema: SchemaTurno


def valida_turno(
    schema: SchemaTurno,
    richiesta: str,
    stato: ContestoValidazione | None = None,
) -> EsitoValidazione:
    """
    Applica subject vuoto, scrittura e destinazione.

    La richiesta serve a recuperare un unico nome di documento mancante,
    a completare la cartella nominata dall'utente e a togliere la scrittura
    da una lettura senza salvataggio.
    """
    contesto = stato or ContestoValidazione()
    corrente = _eredita_subject_vuoto(schema, contesto)
    corrente = _normalizza_destinazione(corrente)
    corrente = _preferisci_nome_richiesta(corrente, richiesta)
    corrente = _declassa_scrittura_senza_destinazione(corrente)
    corrente = _promuovi_documento_dallo_subject(corrente, richiesta)
    corrente = _declassa_lettura_senza_salvataggio(corrente, richiesta)
    corrente = _declassa_workspace_senza_target(corrente)
    corrente = _chiarimento_file(corrente, contesto)
    corrente = _cancella_chiarimento_inutile(corrente)
    # Ultimo passo: un'azione esterna non supportata non deve finire sui tool.
    corrente = _chiarimento_azione_non_supportata(corrente, richiesta)
    return EsitoValidazione(schema=corrente)


def _eredita_subject_vuoto(
    schema: SchemaTurno,
    stato: ContestoValidazione,
) -> SchemaTurno:
    """Se il modello lascia il tema vuoto, riusa l'ultimo subject. Il kind resta."""
    if schema.subject.strip():
        return schema
    ultimo = stato.ultimo
    if ultimo is None or not ultimo.subject.strip():
        return schema
    return replace(schema, subject=ultimo.subject)


def _normalizza_destinazione(schema: SchemaTurno) -> SchemaTurno:
    """Sposta in paths un nome file finito per errore in folder."""
    if schema.paths:
        return schema

    def _e_file(testo: str) -> bool:
        return bool(_ESTENSIONE_DOCUMENTO.search((testo or "").strip()))

    cartella = schema.folder.strip()
    if cartella and _e_file(cartella):
        return replace(schema, paths=(cartella,), folder="")
    return schema

def _preferisci_nome_richiesta(
    schema: SchemaTurno,
    richiesta: str,
) -> SchemaTurno:
    """Se il modello accorcia o traduce la cartella, usa il nome della richiesta."""
    dal_testo = estrai_cartella_nominata(richiesta)
    if not dal_testo:
        return schema
    cartella = schema.folder.strip()
    if not cartella or cartella.casefold() == dal_testo.casefold():
        return schema
    testo = (richiesta or "").casefold()
    tradotta = cartella.casefold() not in testo
    if tradotta or _cartella_incompleta(cartella, dal_testo):
        return replace(schema, folder=dal_testo)
    return schema


def _cartella_incompleta(cartella: str, attesa: str) -> bool:
    """True se folder è solo un segmento del path che la richiesta nomina per intero."""
    segmenti = [parte.casefold() for parte in attesa.split("/") if parte]
    if len(segmenti) < 2:
        return False
    return cartella.casefold().strip("/") in segmenti


def _declassa_scrittura_senza_destinazione(schema: SchemaTurno) -> SchemaTurno:
    """write true senza path, cartella o file di consegna non apre il workspace."""
    if not schema.write:
        return schema
    if schema.paths or schema.folder or schema.deliver == "file":
        if schema.kind in _KINDS_MORBIDI and (schema.paths or schema.folder):
            return replace(schema, kind="workspace")
        return schema
    kind = "explain" if schema.kind == "workspace" else schema.kind
    if kind == "chat":
        kind = "explain"
    return replace(schema, write=False, kind=kind)


def _promuovi_documento_dallo_subject(
    schema: SchemaTurno,
    richiesta: str,
) -> SchemaTurno:
    """Un solo nome documento, prima nello subject e poi nella richiesta."""
    if schema.deliver != "file" or schema.paths:
        return schema
    for testo in (schema.subject, richiesta):
        trovati = tuple(dict.fromkeys(PATH_FILE.findall(testo or "")))
        if len(trovati) == 1:
            nome = trovati[0]
            # Stesso nome in folder: è il file, non una directory.
            cartella = "" if schema.folder.strip() == nome else schema.folder
            return replace(schema, paths=(nome,), folder=cartella)
    return schema


def _chiarimento_azione_non_supportata(
    schema: SchemaTurno,
    richiesta: str,
) -> SchemaTurno:
    """Sostituisce un'azione impossibile con un chiarimento onesto e locale.

    Il turno non apre alcun tool: la sessione risponde con il messaggio e basta.
    """
    capability = azione_esterna_non_supportata(richiesta)
    if capability is None:
        return schema
    return replace(
        schema,
        kind="chat",
        write=False,
        deliver="chat",
        paths=(),
        folder="",
        clarify=_MESSAGGI_AZIONE_NON_SUPPORTATA[capability],
    )


def _declassa_workspace_senza_target(schema: SchemaTurno) -> SchemaTurno:
    """Un workspace senza scrittura, path, cartella o consegna non è azionabile.

    Il modello ripiega su ``kind=workspace`` quando non riconosce il compito: senza
    una destinazione reale i tool non hanno nulla da fare, quindi il turno torna una
    risposta locale invece di aprire il ReAct a vuoto.
    """
    if schema.kind != "workspace":
        return schema
    if schema.write or schema.paths or schema.folder or schema.deliver == "file":
        return schema
    return replace(schema, kind="explain")


def _declassa_lettura_senza_salvataggio(
    schema: SchemaTurno,
    richiesta: str,
) -> SchemaTurno:
    """Una lettura senza verbo né destinazione di salvataggio non scrive."""
    if schema.deliver != "file" and not schema.write:
        return schema
    testo = richiesta or ""
    if not CUE_LETTURA.search(testo):
        return schema
    if CUE_SALVATAGGIO.search(testo) or CUE_DESTINAZIONE_FILE.search(testo):
        return schema
    return replace(schema, write=False, deliver="chat")


def _chiarimento_file(
    schema: SchemaTurno,
    stato: ContestoValidazione,
) -> SchemaTurno:
    """deliver file senza documento e senza file precedente non scrive in root."""
    if schema.deliver != "file":
        return schema
    if any(_ESTENSIONE_DOCUMENTO.search(path) for path in schema.paths):
        return schema
    if stato.documento_precedente:
        return schema
    return replace(schema, clarify=MESSAGGIO_FILE_MANCANTE)


_CLARIFY_DEL_CODICE = frozenset({MESSAGGIO_FILE_MANCANTE})


def _cancella_chiarimento_inutile(schema: SchemaTurno) -> SchemaTurno:
    """La domanda del modello resta solo se la scrittura non ha destinazione."""
    if not schema.clarify or schema.clarify in _CLARIFY_DEL_CODICE:
        return schema
    workspace_chiaro = schema.kind == "workspace" and bool(schema.paths or schema.folder)
    if workspace_chiaro:
        return replace(schema, clarify="")
    return schema
