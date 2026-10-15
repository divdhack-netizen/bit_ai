"""KV reuse tests: prompt extension and stable prefix."""

from __future__ import annotations

import pytest
from langchain_core.agents import AgentAction

from assistant.kv_prefix import PrefixContinuation
from assistant.session import _limit_react_steps

pytestmark = pytest.mark.unit


def test_direct_prefix_does_not_reread_head() -> None:
    """With the prefix ids, the direct turn sends only history and request."""
    cache = PrefixContinuation()
    prefix = "Sei un assistente.\n\n"
    cache.set_key(("fast", 8192, False, prefix), prefix)
    cache.record_prefix((9, 8))
    sent, extra = cache.prepare(prefix + "RICHIESTA:\nciao\n")
    assert sent == "RICHIESTA:\nciao\n"
    assert extra == {"raw": True, "context": [9, 8]}


def test_limit_react_steps_keeps_write() -> None:
    """The scratchpad callable does not drop the observation of a recent write."""
    action = AgentAction("scrivi_file", "nota.txt", "Thought: scrivo")
    reading = AgentAction("leggi_documento", "a.py", "Thought: leggo")
    steps = [
        (reading, "x" * 5000),
        (action, "File scritto con successo: nota.txt"),
    ]
    limited = _limit_react_steps(steps, max_tokens=80)
    observations = [observation for _, observation in limited]
    assert any("File scritto" in text or "omessa" in text for text in observations)
    assert len(limited) == 2
