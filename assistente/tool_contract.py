"""Contratto tipizzato per tool ReAct: validazione, cap output e metadati RO."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import Tool

from assistente.prompt_budget import fit_observation

DEFAULT_MAX_RESULT_CHARS = 3600
_MAX_OBSERVATION_TOKENS = 800
_CONTRACT_ATTR = "_tool_contract"


@dataclass(frozen=True, slots=True)
class ToolContract:
    """Metadati orchestrabili di un tool (schema, RO, limiti)."""

    name: str
    is_read_only: bool
    is_concurrency_safe: bool
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS
    validate: Callable[[str], str | None] | None = None
    description: str = ""


def validate_non_empty(action_input: str) -> str | None:
    """Rifiuta Action Input vuoti; None se l'input è usabile."""
    if not str(action_input or "").strip():
        return "Input vuoto: fornisci un Action Input non vuoto."
    return None


def wrap_tool_func(
    contract: ToolContract,
    func: Callable[[str], str],
) -> Callable[[str], str]:
    """
    Valida l'input prima dei side-effect e limita l'Observation.

    Un errore di validazione torna come Observation: il loop ReAct può autocorreggersi.
    """

    def guarded(action_input: str) -> str:
        raw = str(action_input if action_input is not None else "")
        if contract.validate is not None:
            error = contract.validate(raw)
            if error is not None:
                return f"Errore validazione {contract.name}: {error}"
        result = str(func(raw))
        if len(result) > contract.max_result_chars:
            result = (
                result[: contract.max_result_chars]
                + "\n[…observation troncata per max_result_chars…]"
            )
        return fit_observation(result, _MAX_OBSERVATION_TOKENS)

    setattr(guarded, _CONTRACT_ATTR, contract)
    return guarded


def build_tool(
    contract: ToolContract,
    func: Callable[[str], str],
    *,
    description: str | None = None,
) -> Tool:
    """Costruisce un Tool LangChain con contratto applicato e consultabile."""
    return Tool(
        name=contract.name,
        description=description if description is not None else contract.description,
        func=wrap_tool_func(contract, func),
    )


def build_tools(
    specs: Iterable[tuple[ToolContract, Callable[[str], str]]],
) -> list[Tool]:
    """Costruisce in ordine una lista di tool da coppie (contratto, funzione)."""
    return [build_tool(contract, func) for contract, func in specs]


def get_tool_contract(tool: Any) -> ToolContract | None:
    """Legge il contratto attaccato a ``tool.func``; None se assente."""
    func = getattr(tool, "func", None)
    contract = getattr(func, _CONTRACT_ATTR, None)
    return contract if isinstance(contract, ToolContract) else None
