"""Unit tests: tolerant ReAct parser (mixed Action + Final Answer)."""

from __future__ import annotations

import pytest
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException

from assistant.react_parser import AssistantReActParser

pytestmark = pytest.mark.unit


def test_parser_prefers_action_when_final_answer_also_present() -> None:
    """
    Gemma often emits scrivi_file and Final Answer in the same turn.

    Input: Thought + Action + Action Input + Final Answer.
    Output: AgentAction (tool executed), not an exception or just a Finish.
    """
    text = (
        "Thought: salvo la ricetta sul disco.\n"
        "Action: scrivi_file\n"
        "Action Input: tartufi_lotus.txt\n"
        "---\n"
        "Ingredienti:\n- Biscotti Lotus\n"
        "Final Answer: Ho creato il file tartufi_lotus.txt"
    )
    outcome = AssistantReActParser().parse(text)
    assert isinstance(outcome, AgentAction)
    assert outcome.tool == "scrivi_file"
    assert "tartufi_lotus.txt" in str(outcome.tool_input)
    assert "Biscotti Lotus" in str(outcome.tool_input)
    assert "Final Answer" not in str(outcome.tool_input)


def test_parser_simple_final_answer() -> None:
    """Final Answer only → AgentFinish with no tool."""
    text = "Thought: ho la risposta.\nFinal Answer: ciao"
    outcome = AssistantReActParser().parse(text)
    assert isinstance(outcome, AgentFinish)
    assert outcome.return_values["output"] == "ciao"


def test_parser_action_none_becomes_final_answer() -> None:
    """Action: Nessuna closes the turn with the Thought, without retrying the LLM."""
    text = (
        "Thought: non serve alcun tool, la risposta è 42.\n"
        "Action: Nessuna\n"
        "Action Input: risposta concettuale"
    )
    outcome = AssistantReActParser().parse(text)
    assert isinstance(outcome, AgentFinish)
    assert "42" in outcome.return_values["output"]


def test_parser_action_without_input_label() -> None:
    """A path under Action without «Action Input:» stays a valid tool."""
    text = "Thought: elenco la cartella.\nAction: elenca_cartella\nassistente"
    outcome = AssistantReActParser().parse(text)
    assert isinstance(outcome, AgentAction)
    assert outcome.tool == "elenca_cartella"
    assert "assistente" in str(outcome.tool_input)


def test_parser_tool_name_with_backtick() -> None:
    """Qwen's CoT often wraps Action: `scrivi_file`."""
    text = (
        "Thought: salvo.\n"
        "Action: `scrivi_file`.\n"
        "Action Input: tmp/a.js\n---\nconsole.log(1)\n"
    )
    outcome = AssistantReActParser().parse(text)
    assert isinstance(outcome, AgentAction)
    assert outcome.tool == "scrivi_file"


def test_parser_ignores_cot_plan_without_executable_action() -> None:
    """An «Observation: Result» outline must not become scrivi_file."""
    text = (
        "Thinking Process:\n"
        "1. Action: scrivi_file\n"
        "3.  Observation: Result.\n"
        "        4.  Thought: Verify.\n"
        "        5.  Final Answer.\n"
    )
    outcome = AssistantReActParser().parse(text)
    assert isinstance(outcome, AgentFinish)
    output = str(outcome.return_values.get("output", ""))
    assert "Observation: Result" not in output


def test_parser_edit_file_with_mixed_final_answer() -> None:
    """A multiline modifica_file Action beats Final Answer in the same turn."""
    text = (
        "Thought: aggiorno la funzione.\n"
        "Action: modifica_file\n"
        "Action Input: src/app.py\n"
        "<<<OLD\n"
        "    return 1\n"
        "<<<NEW\n"
        "    return 2\n"
        "Final Answer: Ho aggiornato src/app.py"
    )
    outcome = AssistantReActParser().parse(text)
    assert isinstance(outcome, AgentAction)
    assert outcome.tool == "modifica_file"
    tool_input = str(outcome.tool_input)
    assert "src/app.py" in tool_input
    assert "<<<OLD" in tool_input
    assert "    return 1" in tool_input
    assert "    return 2" in tool_input
    assert "Final Answer" not in tool_input


def test_parser_long_policy_thought_does_not_become_answer() -> None:
    """A Thought that argues with the write gate must not end up as Final Answer."""
    from langchain_core.exceptions import OutputParserException

    text = (
        "Thought: L'utente richiede di abbellire la navbar. "
        "Scrittura consentita: no. Non posso usare scrivi_file o modifica_file. "
        + ("Non posso completare la richiesta. " * 20)
    )
    with pytest.raises(OutputParserException):
        AssistantReActParser().parse(text)


def test_parser_truncated_edit_file_asks_retry() -> None:
    """A modifica_file without <<<NEW must not close the turn with empty output."""
    from langchain_core.exceptions import OutputParserException

    text = (
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
        AssistantReActParser().parse(text)


def test_parser_create_folder_with_backtick() -> None:
    """Qwen often wraps the tool name; crea_cartella stays an Action."""
    text = "Thought: creo notes.\nAction: `crea_cartella`\nAction Input: notes"
    outcome = AssistantReActParser().parse(text)
    assert isinstance(outcome, AgentAction)
    assert outcome.tool == "crea_cartella"
    assert "notes" in str(outcome.tool_input)


def test_unknown_tool_does_not_close_turn() -> None:
    """A non-existent tool goes back to the model as a parse error, not an empty answer."""
    text = "Thought: provo.\nAction: cancella_tutto\nAction Input: /"
    with pytest.raises(OutputParserException):
        AssistantReActParser().parse(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Action: elenca_cartella\nAction Input: attualeFinal Answer: I file sono: [...]", "attuale"),
        ("Action: leggi_documento\nAction Input: a.pages Observation: testo", "a.pages"),
        ("Action: elenca_cartella\nAction Input: .Action: elenca_cartella", "."),
    ],
)
def test_parser_cuts_final_answer_pasted_into_input(text: str, expected: str) -> None:
    """Gemma pastes the conclusion on the input line: the tool receives only the path."""
    outcome = AssistantReActParser().parse(text)
    assert isinstance(outcome, AgentAction)
    assert outcome.tool_input == expected
