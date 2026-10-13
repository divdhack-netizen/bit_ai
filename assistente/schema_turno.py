"""Scheda JSON del turno: repair locale, schema Ollama e prompt a prefisso stabile."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, fields
from typing import Any

NUM_PREDICT_SCHEMA = 160
MAX_PATHS = 3
MAX_SUBJECT_CARATTERI = 160
MAX_CLARIFY_CARATTERI = 200

KINDS = ("chat", "explain", "workspace")
DELIVERS = ("chat", "file")

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_DELIMITATORE_APERTURA = "<richiesta>"
_DELIMITATORE_CHIUSURA = "</richiesta>"


@dataclass(frozen=True, slots=True)
class SchemaTurno:
    """Scheda piatta del turno. Le stringhe vuote e le tuple vuote sono i default."""

    kind: str = "chat"
    write: bool = False
    deliver: str = "chat"
    subject: str = ""
    paths: tuple[str, ...] = ()
    folder: str = ""
    clarify: str = ""


@dataclass(frozen=True, slots=True)
class EsitoRepair:
    """Parse riuscito, oppure l'errore da rimandare al modello una sola volta."""

    schema: SchemaTurno | None
    errore: str | None
    riparato: bool


@dataclass(frozen=True, slots=True)
class PacchettoFollowup:
    """Contesto corto dell'ultimo schema, senza lo storico intero."""

    kind: str = ""
    subject: str = ""
    path: str = ""
    esito: str = ""


def schema_json_ollama() -> dict[str, Any]:
    """Schema oggetto unico passato al campo format di Ollama, fuori da options."""
    stringa = {"type": "string"}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "kind": {"type": "string", "enum": list(KINDS)},
            "write": {"type": "boolean"},
            "deliver": {"type": "string", "enum": list(DELIVERS)},
            "subject": stringa,
            "paths": {
                "type": "array",
                "items": stringa,
                "maxItems": MAX_PATHS,
            },
            "folder": stringa,
            "clarify": stringa,
        },
        "required": [item.name for item in fields(SchemaTurno)],
    }


PREFISSO_SCHEMA = """Compila solo un oggetto JSON con i campi dello schema.
L'agente lavora solo sui documenti della cartella corrente, senza Internet.
kind: chat, explain, workspace.
chat solo per saluti, ringraziamenti o commenti sulla conversazione.
explain per ogni domanda su un tema: la risposta va cercata nei documenti.
workspace per leggere, cercare, elencare, riassumere o scrivere file.
write è true solo per creare, salvare o modificare un file .txt.
Un path di destinazione per un risultato (riassunto, note, output) è write true e deliver file.
Leggere e rispondere in chat è write false e deliver chat.
clarify vuota se la richiesta è chiara.
Se la richiesta continua il turno precedente e non nomina un tema nuovo, riusa kind e subject.

Esempio. Richiesta: quanto pesa una balenottera azzurra?
{"kind":"explain","write":false,"deliver":"chat","subject":"peso balenottera azzurra","paths":[],"folder":"","clarify":""}

Esempio. Richiesta: leggi il file note.pdf e dimmi di cosa parla.
{"kind":"workspace","write":false,"deliver":"chat","subject":"note.pdf","paths":["note.pdf"],"folder":"","clarify":""}

Esempio. Richiesta: riassumi report.docx e salva il riassunto in sintesi.txt.
{"kind":"workspace","write":true,"deliver":"file","subject":"riassunto report.docx","paths":["report.docx","sintesi.txt"],"folder":"","clarify":""}

Esempio. Richiesta: grazie, perfetto.
{"kind":"chat","write":false,"deliver":"chat","subject":"","paths":[],"folder":"","clarify":""}
"""


def formatta_prompt_schema(
    richiesta: str,
    iso: str,
    followup: PacchettoFollowup | None = None,
    errore: str = "",
) -> str:
    """
    Prefisso stabile, poi data, follow-up, eventuale errore di parse e la richiesta.

    Il testo utente resta fra delimitatori e non entra nelle istruzioni.
    """
    pacchetto = followup or PacchettoFollowup()
    blocco_errore = f"Errore JSON: {errore.strip()}\n" if errore.strip() else ""
    return (
        f"{PREFISSO_SCHEMA}\n"
        f"Data: {iso}\n"
        f"{_formatta_followup(pacchetto)}\n"
        f"{blocco_errore}"
        f"{_DELIMITATORE_APERTURA}\n"
        f"{str(richiesta or '').strip()}\n"
        f"{_DELIMITATORE_CHIUSURA}\n"
    )


def carica_oggetto(testo: str) -> tuple[dict[str, Any] | None, str | None]:
    """Primo oggetto JSON del testo, oppure un errore di parse."""
    estratto = _estrai_oggetto(_senza_fence(testo))
    if estratto is None:
        return None, "JSON oggetto assente."
    try:
        payload = json.loads(estratto)
    except json.JSONDecodeError as error:
        return None, f"JSON non valido: {error.msg}."
    if not isinstance(payload, dict):
        return None, "JSON non è un oggetto."
    return payload, None


def ripara_json_schema(testo: str) -> EsitoRepair:
    """
    Toglie fence, prende il primo oggetto e riempie i default.

    Returns:
        Schema se kind è fra i valori ammessi, altrimenti l'errore di parse.
    """
    payload, errore = carica_oggetto(testo)
    if payload is None:
        return EsitoRepair(schema=None, errore=errore, riparato=False)
    riparato = set(payload) != {item.name for item in fields(SchemaTurno)}
    kind = _enum_testo(payload.get("kind"), KINDS)
    if kind is None:
        return EsitoRepair(schema=None, errore="Manca kind valido.", riparato=True)
    deliver = _enum_testo(payload.get("deliver"), DELIVERS) or "chat"
    schema = SchemaTurno(
        kind=kind,
        write=_come_bool(payload.get("write")),
        deliver=deliver,
        subject=_testo_corto(payload.get("subject"), MAX_SUBJECT_CARATTERI),
        paths=_lista_corta(payload.get("paths"), MAX_PATHS, MAX_SUBJECT_CARATTERI),
        folder=_testo_corto(payload.get("folder"), MAX_SUBJECT_CARATTERI),
        clarify=_testo_corto(payload.get("clarify"), MAX_CLARIFY_CARATTERI),
    )
    return EsitoRepair(schema=schema, errore=None, riparato=riparato)


def _formatta_followup(pacchetto: PacchettoFollowup) -> str:
    """Righe corte: il prefill del 9B non deve rivedere lo storico."""
    esito = " ".join(pacchetto.esito.split())[:240]
    return (
        "Follow-up:\n"
        f"- kind: {pacchetto.kind}\n"
        f"- subject: {pacchetto.subject}\n"
        f"- path: {pacchetto.path}\n"
        f"- esito: {esito}"
    )


def _senza_fence(testo: str) -> str:
    """Preferisce il contenuto del primo fence, altrimenti il testo grezzo."""
    match = _FENCE.search(testo or "")
    if match is None:
        return testo or ""
    return match.group(1)


def _estrai_oggetto(testo: str) -> str | None:
    """Primo oggetto con graffe bilanciate, ignorando le graffe dentro le stringhe."""
    start = testo.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(testo)):
        char = testo[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return testo[start : index + 1]
    return None


def _enum_testo(value: object, ammessi: tuple[str, ...]) -> str | None:
    """Valore enum case-sensitive, oppure None se assente o fuori lista."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if cleaned in ammessi:
        return cleaned
    return None


def _come_bool(value: object) -> bool:
    """Accetta bool e le stringhe true/false che un modello piccolo emette spesso."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "1", "sì", "si", "yes"}
    return False


def _testo_corto(value: object, limite: int) -> str:
    """Stringa su una riga, tagliata al limite."""
    if not isinstance(value, str):
        return ""
    collapsed = " ".join(value.replace("\n", " ").split())
    return collapsed[:limite]


def _lista_corta(value: object, massimo: int, limite_voce: int) -> tuple[str, ...]:
    """Prime voci non vuote, ciascuna già tagliata."""
    if not isinstance(value, list):
        return ()
    voci: list[str] = []
    for item in value:
        testo = _testo_corto(item, limite_voce)
        if not testo:
            continue
        voci.append(testo)
        if len(voci) >= massimo:
            break
    return tuple(voci)
