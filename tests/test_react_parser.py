"""Test unit: parser ReAct tollerante (Action + Final Answer miste)."""

from __future__ import annotations

import pytest
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException

from assistente.react_parser import AssistenteReActParser

pytestmark = pytest.mark.unit


def test_parser_preferisce_action_quando_c_e_anche_final_answer() -> None:
    """
    Gemma spesso emette scrivi_file e Final Answer nello stesso turno.

    Input: Thought + Action + Action Input + Final Answer.
    Output: AgentAction (tool eseguito), non eccezione né solo Finish.
    """
    testo = (
        "Thought: salvo la ricetta sul disco.\n"
        "Action: scrivi_file\n"
        "Action Input: tartufi_lotus.txt\n"
        "---\n"
        "Ingredienti:\n- Biscotti Lotus\n"
        "Final Answer: Ho creato il file tartufi_lotus.txt"
    )
    esito = AssistenteReActParser().parse(testo)
    assert isinstance(esito, AgentAction)
    assert esito.tool == "scrivi_file"
    assert "tartufi_lotus.txt" in str(esito.tool_input)
    assert "Biscotti Lotus" in str(esito.tool_input)
    assert "Final Answer" not in str(esito.tool_input)


def test_parser_final_answer_semplice() -> None:
    """Solo Final Answer → AgentFinish senza tool."""
    testo = "Thought: ho la risposta.\nFinal Answer: ciao"
    esito = AssistenteReActParser().parse(testo)
    assert isinstance(esito, AgentFinish)
    assert esito.return_values["output"] == "ciao"


def test_parser_action_nessuna_diventa_final_answer() -> None:
    """Action: Nessuna chiude il turno col Thought, senza retry sul LLM."""
    testo = (
        "Thought: non serve alcun tool, la risposta è 42.\n"
        "Action: Nessuna\n"
        "Action Input: risposta concettuale"
    )
    esito = AssistenteReActParser().parse(testo)
    assert isinstance(esito, AgentFinish)
    assert "42" in esito.return_values["output"]


def test_parser_action_senza_etichetta_input() -> None:
    """Un path sotto Action senza «Action Input:» resta un tool valido."""
    testo = "Thought: elenco la cartella.\nAction: elenca_cartella\nassistente"
    esito = AssistenteReActParser().parse(testo)
    assert isinstance(esito, AgentAction)
    assert esito.tool == "elenca_cartella"
    assert "assistente" in str(esito.tool_input)


def test_parser_nome_tool_con_backtick() -> None:
    """Il CoT Qwen spesso wrappa Action: `scrivi_file`."""
    testo = (
        "Thought: salvo.\n"
        "Action: `scrivi_file`.\n"
        "Action Input: tmp/a.js\n---\nconsole.log(1)\n"
    )
    esito = AssistenteReActParser().parse(testo)
    assert isinstance(esito, AgentAction)
    assert esito.tool == "scrivi_file"


def test_parser_ignora_piano_cot_senza_action_eseguibile() -> None:
    """Un outline «Observation: Result» non deve diventare scrivi_file."""
    testo = (
        "Thinking Process:\n"
        "1. Action: scrivi_file\n"
        "3.  Observation: Result.\n"
        "        4.  Thought: Verify.\n"
        "        5.  Final Answer.\n"
    )
    esito = AssistenteReActParser().parse(testo)
    assert isinstance(esito, AgentFinish)
    output = str(esito.return_values.get("output", ""))
    assert "Observation: Result" not in output


def test_parser_modifica_file_con_final_answer_mista() -> None:
    """Action modifica_file multilineare vince su Final Answer nello stesso turno."""
    testo = (
        "Thought: aggiorno la funzione.\n"
        "Action: modifica_file\n"
        "Action Input: src/app.py\n"
        "<<<OLD\n"
        "    return 1\n"
        "<<<NEW\n"
        "    return 2\n"
        "Final Answer: Ho aggiornato src/app.py"
    )
    esito = AssistenteReActParser().parse(testo)
    assert isinstance(esito, AgentAction)
    assert esito.tool == "modifica_file"
    tool_input = str(esito.tool_input)
    assert "src/app.py" in tool_input
    assert "<<<OLD" in tool_input
    assert "    return 1" in tool_input
    assert "    return 2" in tool_input
    assert "Final Answer" not in tool_input


def test_parser_thought_policy_lungo_non_diventa_risposta() -> None:
    """Un Thought che litiga col gate scrittura non deve finire come Final Answer."""
    from langchain_core.exceptions import OutputParserException

    testo = (
        "Thought: L'utente richiede di abbellire la navbar. "
        "Scrittura consentita: no. Non posso usare scrivi_file o modifica_file. "
        + ("Non posso completare la richiesta. " * 20)
    )
    with pytest.raises(OutputParserException):
        AssistenteReActParser().parse(testo)


def test_parser_modifica_file_troncato_chiede_retry() -> None:
    """Un modifica_file senza <<<NEW non deve chiudere il turno con output vuoto."""
    from langchain_core.exceptions import OutputParserException

    testo = (
        "Thought: Il file sito/index.html è stato letto. "
        "Per rendere i link più grandi aumento font-size e padding.\n"
        "Action: modifica_file\n"
        "Action Input: sito/index.html\n"
        "<<<OLD\n"
        "        nav a {\n"
        "            color: white;\n"
        "            text-decoration: none;\n"
        "            margin: 0 1rem;\n"
        "            font-weight: bold;\n"
        "            pa"
    )
    with pytest.raises(OutputParserException, match="incompleta"):
        AssistenteReActParser().parse(testo)


def test_parser_crea_cartella_con_backtick() -> None:
    """Qwen wrappa spesso il nome tool; crea_cartella resta un Action."""
    testo = "Thought: creo notes.\nAction: `crea_cartella`\nAction Input: notes"
    esito = AssistenteReActParser().parse(testo)
    assert isinstance(esito, AgentAction)
    assert esito.tool == "crea_cartella"
    assert "notes" in str(esito.tool_input)


def test_tool_sconosciuto_non_chiude_il_turno() -> None:
    """Un tool inesistente torna al modello come errore di parse, non come risposta vuota."""
    testo = "Thought: provo.\nAction: cancella_tutto\nAction Input: /"
    with pytest.raises(OutputParserException):
        AssistenteReActParser().parse(testo)


@pytest.mark.parametrize(
    ("testo", "atteso"),
    [
        ("Action: elenca_cartella\nAction Input: attualeFinal Answer: I file sono: [...]", "attuale"),
        ("Action: leggi_documento\nAction Input: a.pages Observation: testo", "a.pages"),
        ("Action: elenca_cartella\nAction Input: .Action: elenca_cartella", "."),
    ],
)
def test_parser_taglia_final_answer_incollata_all_input(testo: str, atteso: str) -> None:
    """Gemma incolla la conclusione sulla riga dell'input: il tool riceve solo il path."""
    esito = AssistenteReActParser().parse(testo)
    assert isinstance(esito, AgentAction)
    assert esito.tool_input == atteso
