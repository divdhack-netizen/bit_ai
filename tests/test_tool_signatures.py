"""Contratti sulle firme mostrate al modello nel prompt ReAct."""

from __future__ import annotations

import inspect

import pytest

from assistente.file_service import WorkspaceFileService
from assistente.riassunto import crea_tool_riassunto
from assistente.tools import crea_tool_file

pytestmark = pytest.mark.unit


def test_tool_bound_espongono_un_solo_action_input() -> None:
    """
    Impedisce che dipendenze iniettate appaiano nella firma e vengano copiate da Gemma.

    Postconditions:
        Ogni callable mostra soltanto action_input, senza service/config nel prompt.
    """
    file_service = WorkspaceFileService()
    tools = crea_tool_file(file_service) + crea_tool_riassunto(file_service)
    # Ispeziona tutte le funzioni nello stesso modo usato dal renderer LangChain.
    for tool in tools:
        parameter_names = tuple(inspect.signature(tool.func).parameters)
        assert parameter_names == ("action_input",), (
            f"{tool.name} espone parametri inattesi: {parameter_names}"
        )


def test_set_minimo_di_tool_closed_corpus() -> None:
    """Solo lettura della cartella, ricerca nel contenuto e tre tool di scrittura."""
    nomi = {tool.name for tool in crea_tool_file(WorkspaceFileService(), write_allowed=True)}
    assert nomi == {
        "leggi_documento",
        "elenca_cartella",
        "elenca_albero",
        "trova_file",
        "cerca_testo",
        "scrivi_file",
        "modifica_file",
        "crea_cartella",
    }
    sola_lettura = {
        tool.name for tool in crea_tool_file(WorkspaceFileService(), write_allowed=False)
    }
    assert "scrivi_file" not in sola_lettura
    assert "cerca_testo" in sola_lettura
