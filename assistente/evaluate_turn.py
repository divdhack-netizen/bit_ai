"""Valutazione deterministica post-turno, senza secondo LLM."""

from __future__ import annotations

import re
from dataclasses import dataclass

_PATH_CITATI_PATTERN = re.compile(r"[\w./-]+\.(?:java|py|js|css|html?|json|txt|md)")
_MARCATORI_LIMITE = (
    "agent stopped due to iteration limit",
    "agent stopped due to time limit",
)
_ERRORE_TOOL_RIGA = re.compile(
    r"(?m)^(?:errore validazione|errore:|path non consentito|"
    r"scrittura non richiesta|file non trovato)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class EsitoValutazioneTurno:
    """Esito della checklist: testo finale, completezza e note diagnostiche."""

    risposta: str
    incompleto: bool
    note: tuple[str, ...] = ()


def _append_notes(text: str, notes: list[str]) -> str:
    """Accoda note non già presenti nel testo, senza duplicati."""
    enriched = text.rstrip()
    lower = enriched.casefold()
    for note in notes:
        if note.casefold() not in lower:
            enriched = f"{enriched}\n\n{note}"
            lower = enriched.casefold()
    return enriched


def evaluate_turn(
    risposta: str,
    *,
    scrittura_richiesta: bool,
    path_scritti: list[str],
    max_iterations_hit: bool = False,
) -> EsitoValutazioneTurno:
    """
    Checklist post-esecuzione: scrittura mancante, tetto iterazioni, errori tool.

    Args:
        scrittura_richiesta: True solo se l'utente ha chiesto di creare o
            modificare qualcosa. Non è il permesso dei tool: la route ReAct
            autorizza la scrittura anche per una semplice lettura.

    Returns:
        Esito con risposta eventualmente arricchita e flag ``incompleto``.
    """
    notes: list[str] = []
    text = str(risposta or "")
    lower = text.casefold()
    hit_limit = max_iterations_hit or any(marker in lower for marker in _MARCATORI_LIMITE)
    if hit_limit and not path_scritti:
        notes.append(
            "Turno interrotto per limite iterazioni: il lavoro può essere incompleto."
        )

    missing_write = scrittura_richiesta and not path_scritti
    if missing_write:
        cited = list(dict.fromkeys(_PATH_CITATI_PATTERN.findall(text)))
        if cited:
            notes.append(
                "Scrittura richiesta ma nessun file persistito "
                f"(citati: {', '.join(cited)})."
            )
        else:
            notes.append(
                "Scrittura richiesta ma nessun file o cartella è stato creato "
                "o aggiornato in questo turno."
            )
    elif _ERRORE_TOOL_RIGA.search(text) and not path_scritti:
        notes.append("Sono presenti errori tool nella risposta; verifica l'esito.")

    if not notes:
        return EsitoValutazioneTurno(risposta=text, incompleto=False)
    return EsitoValutazioneTurno(
        risposta=_append_notes(text, notes),
        incompleto=missing_write or hit_limit,
        note=tuple(notes),
    )
