"""Typed contract for ReAct tools: validation, output cap and read-only metadata."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import Tool

from assistant.prompt_budget import fit_observation

DEFAULT_MAX_RESULT_CHARS = 3600
_MAX_OBSERVATION_TOKENS = 800
_CONTRACT_ATTR = "_tool_contract"


@dataclass(frozen=True, slots=True)
class ToolContract:
    """Orchestration metadata for a tool (schema, read-only, limits)."""

    name: str
    is_read_only: bool
    is_concurrency_safe: bool
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS
    validate: Callable[[str], str | None] | None = None
    description: str = ""


def validate_non_empty(action_input: str) -> str | None:
    """Reject empty Action Inputs; None if the input is usable."""
    if not str(action_input or "").strip():
        return "Input vuoto: fornisci un Action Input non vuoto."
    return None


def wrap_tool_func(
    contract: ToolContract,
    func: Callable[[str], str],
) -> Callable[[str], str]:
    """
    Validate the input before side effects and cap the Observation.

    A validation error comes back as an Observation, so the ReAct loop can self-correct.
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
    """Build a LangChain Tool with the contract applied and inspectable."""
    return Tool(
        name=contract.name,
        description=description if description is not None else contract.description,
        func=wrap_tool_func(contract, func),
    )


def build_tools(
    specs: Iterable[tuple[ToolContract, Callable[[str], str]]],
) -> list[Tool]:
    """Build an ordered list of tools from (contract, function) pairs."""
    return [build_tool(contract, func) for contract, func in specs]


def get_tool_contract(tool: Any) -> ToolContract | None:
    """Read the contract attached to ``tool.func``; None if missing."""
    func = getattr(tool, "func", None)
    contract = getattr(func, _CONTRACT_ATTR, None)
    return contract if isinstance(contract, ToolContract) else None
