"""Contracts on the signatures shown to the model in the ReAct prompt."""

from __future__ import annotations

import inspect

import pytest

from assistant.file_service import WorkspaceFileService
from assistant.summary import create_summary_tool
from assistant.tools import create_file_tools

pytestmark = pytest.mark.unit


def test_bound_tools_expose_single_action_input() -> None:
    """
    Prevents injected dependencies from appearing in the signature and being copied by Gemma.

    Postconditions:
        Each callable shows only action_input, with no service/config in the prompt.
    """
    file_service = WorkspaceFileService()
    tools = create_file_tools(file_service) + create_summary_tool(file_service)
    # Inspect every function the same way the LangChain renderer does.
    for tool in tools:
        parameter_names = tuple(inspect.signature(tool.func).parameters)
        assert parameter_names == ("action_input",), (
            f"{tool.name} espone parametri inattesi: {parameter_names}"
        )


def test_minimal_closed_corpus_tool_set() -> None:
    """Read-only folder access, content search and three write tools."""
    names = {tool.name for tool in create_file_tools(WorkspaceFileService(), write_allowed=True)}
    assert names == {
        "leggi_documento",
        "elenca_cartella",
        "elenca_albero",
        "trova_file",
        "cerca_testo",
        "scrivi_file",
        "modifica_file",
        "crea_cartella",
    }
    read_only = {
        tool.name for tool in create_file_tools(WorkspaceFileService(), write_allowed=False)
    }
    assert "scrivi_file" not in read_only
    assert "cerca_testo" in read_only
