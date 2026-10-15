"""System date: the data_oggi ReAct tool and the turn's time reference."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from langchain_core.tools import Tool

from assistant.tool_contract import ToolContract, build_tool

TODAY_DATE_TOOL_DESCRIPTION = (
    "Data odierna del sistema (ISO). Input: vuoto. "
    "Usala per calcolare età o intervalli rispetto alle date dei documenti."
)

# date.weekday(): 0=Monday … 6=Sunday; month: 1–12.
_DAYS = (
    "lunedì", "martedì", "mercoledì", "giovedì",
    "venerdì", "sabato", "domenica",
)
ITALIAN_MONTHS = (
    "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
    "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
)


@dataclass(frozen=True, slots=True)
class TemporalContext:
    """
    Immutable snapshot of the calendar date used in a turn.

    Attributes:
        day: reference date (usually date.today()).
        iso: YYYY-MM-DD.
        year: calendar year.
        label: readable Italian form + ISO + year.
    """

    day: date
    iso: str
    year: int
    label: str


def temporal_context(today: date | None = None) -> TemporalContext:
    """
    Build a TemporalContext from today (default: date.today()).

    Input: optional date for tests; None → system calendar.
    Output: snapshot with iso, year and Italian label already computed.
    """
    day = today if today is not None else date.today()
    iso = day.isoformat()
    label = (
        f"Data odierna: {_DAYS[day.weekday()]} {day.day} "
        f"{ITALIAN_MONTHS[day.month - 1]} {day.year} ({iso}). Anno: {day.year}."
    )
    return TemporalContext(
        day=day,
        iso=iso,
        year=day.year,
        label=label,
    )


def format_today_date(today: date | None = None) -> str:
    """Italian + ISO label of the date (wrapper around temporal_context)."""
    return temporal_context(today).label


def today_date_from_text(_input: object = "") -> str:
    """
    ReAct tool: the system's live calendar date.

    Input: ignored (LangChain may pass str/None/other).
    Output: label from format_today_date() on date.today().
    """
    return format_today_date()


def create_date_tool() -> list[Tool]:
    """Register the data_oggi ReAct tool."""
    return [
        build_tool(
            ToolContract(
                name="data_oggi",
                is_read_only=True,
                is_concurrency_safe=True,
                description=TODAY_DATE_TOOL_DESCRIPTION,
            ),
            today_date_from_text,
        ),
    ]
