"""KV reuse for the stable direct prefix."""

from __future__ import annotations

from typing import Any


class PrefixContinuation:
    """If the prompt starts with the already prefilled prefix, send only the suffix."""

    def __init__(self) -> None:
        self._key: tuple[Any, ...] | None = None
        self._prefix = ""
        self._prefix_ids: tuple[int, ...] | None = None

    def set_key(self, key: tuple[Any, ...], prefix: str) -> None:
        """Change alias, window or prefix: the previous ids are no longer valid."""
        if key == self._key and prefix == self._prefix:
            return
        self._key = key
        self._prefix = prefix
        self._prefix_ids = None

    def has_prefix(self, key: tuple[Any, ...]) -> bool:
        """True if the prefix ids are already known for this key."""
        return self._key == key and bool(self._prefix_ids)

    def record_prefix(self, context_ids: tuple[int, ...] | list[int]) -> None:
        """Store the ids returned by prefilling the prefix alone."""
        if not context_ids:
            return
        self._prefix_ids = tuple(int(item) for item in context_ids)

    def prepare(self, prompt: str) -> tuple[str, dict[str, Any]]:
        """Suffix with `raw` and `context`, or the whole prompt."""
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

    def invalidate(self) -> None:
        """Reset the prefix, e.g. after /clear."""
        self._key = None
        self._prefix = ""
        self._prefix_ids = None
