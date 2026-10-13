"""Regressione: messaggi sistema con markup accidentale non devono crashare."""

from __future__ import annotations

import pytest
from rich.console import Console
from rich.markup import MarkupError
from rich.text import Text

from assistente.console import ConsoleUI

pytestmark = pytest.mark.unit


def test_markup_legacy_senza_escape_solleva_su_tag_chiusura() -> None:
    """Senza escape, un `[/]` nel testo fa crashare Rich (MarkupError)."""
    console = Console(force_terminal=True, width=80)
    with pytest.raises(MarkupError):
        console.print("[system]errore tag [/] unexpected[/]")


def test_stampa_messaggio_sistema_escapa_tag_chiusura() -> None:
    """
    `stampa_messaggio_sistema` deve escape-are il payload: errori Ollama/httpx
    con `[/]` non devono far uscire il loop di parsing Rich.
    """
    ui = ConsoleUI()
    # Non deve sollevare.
    ui.stampa_messaggio_sistema("errore tag [/] unexpected")
    plain = Text.from_markup(
        f"[system]{__import__('rich.markup', fromlist=['escape']).escape('errore tag [/] unexpected')}[/]"
    ).plain
    assert "[/]" in plain
