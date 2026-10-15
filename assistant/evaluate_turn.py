"""Deterministic post-turn evaluation, with no second LLM."""

from __future__ import annotations

import re
from dataclasses import dataclass

_CITED_PATHS_PATTERN = re.compile(r"[\w./-]+\.(?:java|py|js|css|html?|json|txt|md)")
_LIMIT_MARKERS = (
    "agent stopped due to iteration limit",
    "agent stopped due to time limit",
)
_TOOL_ERROR_LINE = re.compile(
    r"(?m)^(?:errore validazione|errore:|path non consentito|"
    r"scrittura non richiesta|file non trovato)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class TurnEvaluationOutcome:
    """Checklist outcome: final text, completeness and diagnostic notes."""

    answer: str
    incomplete: bool
    note: tuple[str, ...] = ()


def _append_notes(text: str, notes: list[str]) -> str:
    """Append notes not already present in the text, without duplicates."""
    enriched = text.rstrip()
    lower = enriched.casefold()
    for note in notes:
        if note.casefold() not in lower:
            enriched = f"{enriched}\n\n{note}"
            lower = enriched.casefold()
    return enriched


def evaluate_turn(
    answer: str,
    *,
    write_requested: bool,
    written_paths: list[str],
    max_iterations_hit: bool = False,
) -> TurnEvaluationOutcome:
    """
    Post-execution checklist: missing write, iteration cap, tool errors.

    Args:
        write_requested: True only if the user asked to create or modify
            something. This is not the tool permission: the ReAct route
            allows writing even for a plain read.

    Returns:
        Outcome with the possibly enriched answer and the ``incomplete`` flag.
    """
    notes: list[str] = []
    text = str(answer or "")
    lower = text.casefold()
    hit_limit = max_iterations_hit or any(marker in lower for marker in _LIMIT_MARKERS)
    if hit_limit and not written_paths:
        notes.append(
            "Turno interrotto per limite iterazioni: il lavoro può essere incompleto."
        )

    missing_write = write_requested and not written_paths
    if missing_write:
        cited = list(dict.fromkeys(_CITED_PATHS_PATTERN.findall(text)))
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
    elif _TOOL_ERROR_LINE.search(text) and not written_paths:
        notes.append("Sono presenti errori tool nella risposta; verifica l'esito.")

    if not notes:
        return TurnEvaluationOutcome(answer=text, incomplete=False)
    return TurnEvaluationOutcome(
        answer=_append_notes(text, notes),
        incomplete=missing_write or hit_limit,
        note=tuple(notes),
    )
