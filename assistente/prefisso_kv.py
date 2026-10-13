"""Riuso della KV del prefisso stabile direct."""

from __future__ import annotations

from typing import Any


class ContinuazionePrefisso:
    """Se il prompt inizia col prefisso già prefillato, invia solo il suffisso."""

    def __init__(self) -> None:
        self._chiave: tuple[Any, ...] | None = None
        self._prefix = ""
        self._prefix_ids: tuple[int, ...] | None = None

    def imposta_chiave(self, chiave: tuple[Any, ...], prefix: str) -> None:
        """Cambia alias, finestra o prefisso: gli id precedenti non valgono più."""
        if chiave == self._chiave and prefix == self._prefix:
            return
        self._chiave = chiave
        self._prefix = prefix
        self._prefix_ids = None

    def ha_prefisso(self, chiave: tuple[Any, ...]) -> bool:
        """True se gli id del prefisso sono già noti per questa chiave."""
        return self._chiave == chiave and bool(self._prefix_ids)

    def registra_prefisso(self, context_ids: tuple[int, ...] | list[int]) -> None:
        """Memorizza gli id restituiti dal prefill del solo prefisso."""
        if not context_ids:
            return
        self._prefix_ids = tuple(int(item) for item in context_ids)

    def prepara(self, prompt: str) -> tuple[str, dict[str, Any]]:
        """Suffisso con `raw` e `context`, oppure il prompt intero."""
        if (
            self._prefix
            and self._prefix_ids
            and prompt.startswith(self._prefix)
            and len(prompt) > len(self._prefix)
        ):
            return prompt[len(self._prefix) :], {
                "raw": True,
                "context": list(self._prefix_ids),
            }
        return prompt, {}

    def invalida(self) -> None:
        """Azzera il prefisso, per esempio dopo /clear."""
        self._chiave = None
        self._prefix = ""
        self._prefix_ids = None
