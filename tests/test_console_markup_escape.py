"""Regression: system messages with accidental markup must not crash."""

from __future__ import annotations

import pytest
from rich.console import Console
from rich.markup import MarkupError
from rich.text import Text

from assistant.console import ConsoleUI

pytestmark = pytest.mark.unit


def test_legacy_markup_without_escape_raises_on_closing_tag() -> None:
    """Without escaping, a `[/]` in the text crashes Rich (MarkupError)."""
    console = Console(force_terminal=True, width=80)
    with pytest.raises(MarkupError):
        console.print("[system]errore tag [/] unexpected[/]")


def test_print_system_message_escapes_closing_tag() -> None:
    """
    `print_system_message` must escape the payload: Ollama/httpx errors
    containing `[/]` must not break Rich's parsing loop.
    """
    ui = ConsoleUI()
    # Must not raise.
    ui.print_system_message("errore tag [/] unexpected")
    plain = Text.from_markup(
        f"[system]{__import__('rich.markup', fromlist=['escape']).escape('errore tag [/] unexpected')}[/]"
    ).plain
    assert "[/]" in plain
