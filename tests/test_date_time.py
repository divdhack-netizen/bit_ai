"""Unit tests: TemporalContext and the data_oggi tool."""

from __future__ import annotations

from datetime import date

import pytest

from assistant.date_time import (
    TemporalContext,
    temporal_context,
    create_date_tool,
    today_date_from_text,
    format_today_date,
)


@pytest.fixture
def pinned() -> date:
    """Monday 3 August 2026: a fixed date for deterministic asserts."""
    return date(2026, 8, 3)


def test_temporal_context_fields(pinned: date) -> None:
    """Snapshot: ISO, year and Italian label consistent with the given date."""
    fake_time = temporal_context(pinned)
    assert isinstance(fake_time, TemporalContext)
    assert fake_time.day == pinned
    assert fake_time.iso == "2026-08-03"
    assert fake_time.year == 2026
    assert "lunedì" in fake_time.label
    assert "agosto" in fake_time.label
    assert "2026-08-03" in fake_time.label


def test_format_today_date_delegates_to_context(pinned: date) -> None:
    """format_today_date matches TemporalContext.label."""
    assert format_today_date(pinned) == temporal_context(pinned).label


def test_today_date_from_text_uses_date_today() -> None:
    """Live tool: ISO and year from date.today(); input is ignored."""
    today = date.today()
    text = today_date_from_text(None)
    assert today.isoformat() in text
    assert str(today.year) in text


def test_create_date_tool_registers_data_oggi() -> None:
    """Factory: a single Tool named data_oggi."""
    tools = create_date_tool()
    assert len(tools) == 1
    assert tools[0].name == "data_oggi"
