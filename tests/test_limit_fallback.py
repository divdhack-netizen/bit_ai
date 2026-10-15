"""Fallback at the iteration limit: the answer comes from the last useful step."""

from __future__ import annotations

import pytest
from langchain_core.agents import AgentAction

from assistant.readers import UNSUPPORTED_FORMAT_INSTRUCTION
from assistant.session import _answer_from_steps
from assistant.tools import REPEATED_READ_WARNING

pytestmark = pytest.mark.unit


def _step(tool: str, tool_input: str, observation: str) -> tuple[AgentAction, str]:
    return AgentAction(tool, tool_input, ""), observation


def test_no_step_no_fallback() -> None:
    assert _answer_from_steps([]) is None


def test_uses_last_successful_summary() -> None:
    steps = [
        _step("trova_file", "articolo_1", "Nessun risultato: articolo_1"),
        _step("riassumi_documento", "Articolo_01.pages", "Il documento parla di IA."),
    ]
    assert _answer_from_steps(steps) == "Il documento parla di IA."


def test_skips_errors_and_anti_loop_warning() -> None:
    steps = [
        _step("elenca_cartella", ".", "Contenuto di '.':\n  a.txt (1 byte)"),
        _step("elenca_cartella", ".", f"{REPEATED_READ_WARNING}Contenuto di '.':\n  a.txt (1 byte)"),
        _step("elenca_cartella", "notes", "Errore: Non è una cartella: notes"),
    ]
    assert _answer_from_steps(steps) == "Contenuto di '.':\n  a.txt (1 byte)"


def test_unsupported_format_is_the_answer() -> None:
    error = (
        "Errore: Formato .zip non supportato o file binario: paperotti.zip. "
        f"Il file esiste ma non si può leggere; formati letti: txt.{UNSUPPORTED_FORMAT_INSTRUCTION}"
    )
    steps = [_step("leggi_documento", "paperotti.zip", error)]
    answer = _answer_from_steps(steps)
    assert answer is not None
    assert answer.startswith("Formato .zip non supportato")
    assert "Non cercarlo" not in answer


def test_read_document_becomes_excerpt() -> None:
    steps = [_step("leggi_documento", "a.txt", "x" * 3000)]
    answer = _answer_from_steps(steps)
    assert answer is not None
    assert answer.startswith("Ho letto a.txt") and answer.endswith("…")


def test_search_text_is_not_an_answer() -> None:
    steps = [_step("cerca_testo", "articolo_1", "Trovato 'articolo_1' in:\n  x.py (riga 1): article")]
    assert _answer_from_steps(steps) is None
