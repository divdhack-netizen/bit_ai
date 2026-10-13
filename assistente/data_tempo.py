"""Data di sistema: tool ReAct data_oggi e riferimento temporale del turno."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from langchain_core.tools import Tool

from assistente.tool_contract import ToolContract, build_tool

DESCRIZIONE_TOOL_DATA_OGGI = (
    "Data odierna del sistema (ISO). Input: vuoto. "
    "Usala per calcolare età o intervalli rispetto alle date dei documenti."
)

# date.weekday(): 0=lunedì … 6=domenica; month: 1–12.
_GIORNI = (
    "lunedì", "martedì", "mercoledì", "giovedì",
    "venerdì", "sabato", "domenica",
)
MESI_ITALIANI = (
    "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
    "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
)


@dataclass(frozen=True, slots=True)
class ContestoTemporale:
    """
    Snapshot immutabile della data di calendario usata in un turno.

    Attributes:
        giorno: data di riferimento (di solito date.today()).
        iso: YYYY-MM-DD.
        anno: anno civile.
        etichetta: forma leggibile IT + ISO + anno.
    """

    giorno: date
    iso: str
    anno: int
    etichetta: str


def contesto_temporale(oggi: date | None = None) -> ContestoTemporale:
    """
    Costruisce un ContestoTemporale da oggi (default: date.today()).

    Input: data opzionale per test; None → calendario di sistema.
    Output: snapshot con iso, anno ed etichetta IT già calcolati.
    """
    giorno = oggi if oggi is not None else date.today()
    iso = giorno.isoformat()
    etichetta = (
        f"Data odierna: {_GIORNI[giorno.weekday()]} {giorno.day} "
        f"{MESI_ITALIANI[giorno.month - 1]} {giorno.year} ({iso}). Anno: {giorno.year}."
    )
    return ContestoTemporale(
        giorno=giorno,
        iso=iso,
        anno=giorno.year,
        etichetta=etichetta,
    )


def formatta_data_odierna(oggi: date | None = None) -> str:
    """Etichetta IT+ISO della data (wrapper su contesto_temporale)."""
    return contesto_temporale(oggi).etichetta


def data_oggi_da_testo(_input: object = "") -> str:
    """
    Tool ReAct: data di calendario live del sistema.

    Input: ignorato (LangChain può passare str/None/altro).
    Output: etichetta da formatta_data_odierna() su date.today().
    """
    return formatta_data_odierna()


def crea_tool_data() -> list[Tool]:
    """Registra il tool ReAct data_oggi."""
    return [
        build_tool(
            ToolContract(
                name="data_oggi",
                is_read_only=True,
                is_concurrency_safe=True,
                description=DESCRIZIONE_TOOL_DATA_OGGI,
            ),
            data_oggi_da_testo,
        ),
    ]
