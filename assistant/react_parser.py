"""Tolerant ReAct parser: malformed Final Answer → AgentFinish."""

from __future__ import annotations

import re

from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from typing_extensions import override

from assistant.config import NEW_BLOCK_MARKER, OLD_BLOCK_MARKER

# Gemma often emits Action/Action Input instead of "Final Answer:".
_ACTION_FINAL_ANSWER = re.compile(
    r"Action\s*\d*\s*:\s*Final\s*Answer\s*"
    r"Action\s*\d*\s*Input\s*\d*\s*:\s*(.*)",
    re.DOTALL | re.IGNORECASE,
)
_ACTION_NAME_FINAL = re.compile(r"^Final\s*Answer$", re.IGNORECASE)
_ACTION_NAME_NULL = re.compile(
    r"^(?:nessun[oa]?|nessuno strumento|none|no tool|n/?a)$",
    re.IGNORECASE,
)
_ACTION_MARKER = re.compile(r"Action\s*\d*\s*:", re.DOTALL)
_KNOWN_TOOLS = (
    "scrivi_file",
    "modifica_file",
    "leggi_documento",
    "elenca_cartella",
    "elenca_albero",
    "trova_file",
    "cerca_testo",
    "crea_cartella",
    "data_oggi",
    "riassumi_documento",
)
_BLOCK_TOOLS = frozenset({"scrivi_file", "modifica_file"})
_WRITE_BLOCK = re.compile(
    r"Action\s*\d*\s*:\s*`?(?P<tool>scrivi_file)`?(?:\s*\([^)]*\))?\.?\s*\n"
    r"(?:Action\s*\d*\s*Input\s*\d*\s*:\s*)?"
    r"(?P<path>[\w./-]+\.\w+)\s*\n"
    r"---\s*\n"
    r"(?P<body>.*)",
    re.IGNORECASE | re.DOTALL,
)
_EDIT_BLOCK = re.compile(
    r"Action\s*\d*\s*:\s*`?modifica_file`?(?:\s*\([^)]*\))?\.?\s*\n"
    r"(?:Action\s*\d*\s*Input\s*\d*\s*:\s*)?"
    r"(?P<path>[\w./-]+\.\w+)\s*\n"
    rf"{re.escape(OLD_BLOCK_MARKER)}[ \t]*\n"
    r"(?P<old>.*?)"
    rf"^{re.escape(NEW_BLOCK_MARKER)}[ \t]*(?:\n|\Z)"
    r"(?P<new>.*)",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)
_OTHER_TOOL_BLOCK = re.compile(
    r"Action\s*\d*\s*:\s*`?(?P<tool>"
    r"leggi_documento|elenca_cartella|elenca_albero|trova_file|"
    r"cerca_testo|crea_cartella|"
    r"data_oggi|riassumi_documento)`?"
    r"(?:\s*\([^)]*\))?\s*\n"
    r"(?:Action\s*\d*\s*Input\s*\d*\s*:\s*)?"
    # riassumi_documento may continue with «---» and the destination path.
    r"(?P<input>\S[^\n]*(?:\n[ \t]*---[ \t]*\n[ \t]*\S[^\n]*)?)",
    re.IGNORECASE,
)
_ACTION_BLOCK = re.compile(
    r"Action\s*\d*\s*:\s*(?P<tool>[^\n]+)(?:\nAction\s*\d*\s*Input\s*\d*\s*:)?\s*"
    r"(?P<rest>.*)",
    re.DOTALL | re.IGNORECASE,
)
_THOUGHT_BLOCK = re.compile(
    r"Thought\s*:\s*(.*?)(?:\nAction\s*:|\nFinal\s+Answer\s*:|\Z)",
    re.DOTALL | re.IGNORECASE,
)
# Gemma sometimes pastes the conclusion on the input line: «attualeFinal Answer: …».
_PASTED_TAIL = re.compile(
    r"(?:Final\s*Answer|Observation|Thought|Action(?:\s*Input)?)\s*:", re.IGNORECASE
)
_FENCE = re.compile(r"^```(?:[\w+-]*)\s*\n?(.*?)\n?```$", re.DOTALL)
_FINAL_ANSWER_MARKER = "Final Answer:"
_FINAL_ANSWER_FLEX = re.compile(r"final\s+answer\s*:", re.IGNORECASE)
_REASONING_MARKERS = (
    "scrittura consentita",
    "scrivi_file",
    "modifica_file",
    "non è autorizzata",
    "non posso usare",
)
_THOUGHT_MAX_AS_ANSWER = 400
INCOMPLETE_BLOCK_OBSERVATION = (
    "Action Input del tool di scrittura incompleto (troncato). "
    "Ripeti SOLO con Thought breve e Action completa:\n"
    "Action: modifica_file\n"
    "Action Input: path_relativo\n"
    f"{OLD_BLOCK_MARKER}\n"
    "<testo esatto dal file>\n"
    f"{NEW_BLOCK_MARKER}\n"
    "<sostituzione>\n"
    "Per scrivi_file: path .txt, riga ---, body. "
    "Niente Final Answer finché manca Observation di successo."
)


def _prefer_action_if_mixed(text: str) -> str:
    """
    If the text has both an Action and a Final Answer, keep only the Action block.

    Gemma often closes with Final Answer in the same turn as scrivi_file;
    the base parser rejects that mix and the tool never runs.
    """
    if _FINAL_ANSWER_MARKER not in text:
        return text
    match_action = _ACTION_MARKER.search(text)
    if match_action is None:
        return text
    final_position = text.find(_FINAL_ANSWER_MARKER)
    action_position = match_action.start()
    # Final Answer after Action → cut before Final Answer (run the tool).
    if final_position > action_position:
        return text[:final_position].rstrip()
    # Final Answer before Action → start from Action.
    action_only = text[action_position:]
    if _FINAL_ANSWER_MARKER in action_only:
        return action_only.split(_FINAL_ANSWER_MARKER, 1)[0].rstrip()
    return action_only


def _is_incomplete_reasoning(text: str) -> bool:
    """True if the text is a Thought/CoT with no usable ReAct conclusion."""
    if not text or _FINAL_ANSWER_FLEX.search(text):
        return False
    low = text.casefold()
    if any(marker in low for marker in _REASONING_MARKERS) and len(text) > 200:
        return True
    if "thought:" in low and len(text) > _THOUGHT_MAX_AS_ANSWER:
        return True
    return len(text) > 800 and "action:" not in low


def _extract_thought(text: str) -> str:
    """Return the Thought, or the text without Action markers."""
    match = _THOUGHT_BLOCK.search(text)
    if match is not None and match.group(1).strip():
        return match.group(1).strip()
    return text.strip()


def _normalize_tool_name(name: str) -> str:
    """Strip markdown and recognize a known tool even if the CoT adds noise."""
    clean = name.strip().strip("`*_.,:;\"'").strip()
    low = clean.casefold()
    for known in _KNOWN_TOOLS:
        if low == known or low.startswith(f"{known} ") or low.startswith(f"{known}`"):
            return known
    return clean


_PATCH_PROTOCOL_TAIL = re.compile(
    rf"\n(?:{re.escape(OLD_BLOCK_MARKER)}|{re.escape(NEW_BLOCK_MARKER)})"
    r"|^(?:Thought|Action|Action Input)\s*:",
    re.IGNORECASE | re.MULTILINE,
)


def _cut_protocol_tail(body: str) -> str:
    """Remove chained Actions, extra patch markers or a Final Answer after the payload."""
    cut = _ACTION_MARKER.search(body)
    if cut is not None:
        body = body[: cut.start()]
    final = re.search(r"\nFinal\s+Answer\s*:", body, re.IGNORECASE)
    if final is not None:
        body = body[: final.start()]
    extra_patch = _PATCH_PROTOCOL_TAIL.search(body)
    if extra_patch is not None:
        body = body[: extra_patch.start()]
    return body


def _rebuild_write(match: re.Match[str]) -> str | None:
    """Rebuild an executable scrivi_file Action Input from the regex match."""
    body = _cut_protocol_tail(match.group("body")).strip("\n")
    path = match.group("path").strip()
    tool = match.group("tool").strip().casefold()
    if not path or not body.strip():
        return None
    return f"Action: {tool}\nAction Input: {path}\n---\n{body}"


def _rebuild_edit(match: re.Match[str]) -> str | None:
    """Rebuild an executable modifica_file Action Input from the regex match."""
    old_block = _cut_protocol_tail(match.group("old")).strip("\n")
    new_block = _cut_protocol_tail(match.group("new")).strip("\n")
    path = match.group("path").strip()
    if not path or not old_block.strip():
        return None
    return (
        f"Action: modifica_file\nAction Input: {path}\n"
        f"{OLD_BLOCK_MARKER}\n{old_block}\n"
        f"{NEW_BLOCK_MARKER}\n{new_block}"
    )


def _action_block_complete(tool: str, tool_input: str) -> bool:
    """True if multiline tools have an executable payload; otherwise True."""
    if tool not in _BLOCK_TOOLS:
        return True
    return (
        extract_executable_react_block(f"Action: {tool}\nAction Input: {tool_input}")
        is not None
    )


def _raise_incomplete_block(tool: str, text: str) -> None:
    """Ask the LLM to retry instead of closing the turn with empty output."""
    raise OutputParserException(
        f"Action {tool} incompleta: completa i marcatori di blocco.",
        observation=INCOMPLETE_BLOCK_OBSERVATION,
        llm_output=text,
        send_to_llm=True,
    )


def _reject_non_executable_tool(tool: str, text: str) -> None:
    """Unknown tool or incomplete block: an Observation, not an empty answer."""
    if tool in _BLOCK_TOOLS:
        _raise_incomplete_block(tool, text)
    raise OutputParserException(
        f"Action {tool} non eseguibile: usa un tool noto con Action Input completo.",
        observation=(
            f"Tool non eseguibile ({tool}). "
            "Emetti Action con un tool noto e Action Input completo."
        ),
        llm_output=text,
        send_to_llm=True,
    )


def extract_executable_react_block(text: str) -> str | None:
    """
    Isolate the last complete Action, ignoring CoT plans like «Observation: Result».

    Returns:
        Minimal ReAct block, or None if there is no executable invocation.
    """
    if not text:
        return None
    candidates: list[tuple[int, str | None]] = [
        (match.start(), _rebuild_write(match))
        for match in _WRITE_BLOCK.finditer(text)
    ]
    candidates.extend(
        (match.start(), _rebuild_edit(match))
        for match in _EDIT_BLOCK.finditer(text)
    )
    if candidates:
        candidates = [(pos, block) for pos, block in candidates if block]
    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[-1][1]
    others = list(_OTHER_TOOL_BLOCK.finditer(text))
    if not others:
        return None
    match = others[-1]
    tool = match.group("tool").casefold()
    tool_input = match.group("input")
    tail = _PASTED_TAIL.search(tool_input)
    if tail is not None:
        tool_input = tool_input[: tail.start()]
    tool_input = tool_input.strip().strip("`")
    if not tool_input or tool_input.casefold().startswith("observation"):
        return None
    return f"Action: {tool}\nAction Input: {tool_input}"


# Gemma 4 sometimes uses its own call syntax: tool(action_input: '...').
_FUNCTION_CALL = re.compile(
    r"^\s*`?(?P<tool>[a-z_]+)\((?:\s*(?:action_input|input)\s*[:=]\s*)?"
    r"(?P<quote>['\"])(?P<input>.*)(?P=quote)\s*\)`?\s*$",
    re.DOTALL,
)


def _from_function_call(text: str) -> str | None:
    """
    Convert a function-style call into the equivalent ReAct block.

    Returns:
        «Action / Action Input» text, or None if it is not a known call.
    """
    match = _FUNCTION_CALL.match(text)
    if match is None or match.group("tool") not in _KNOWN_TOOLS:
        return None
    tool_input = (
        match.group("input")
        .replace("\\n", "\n")
        .replace("\\'", "'")
        .replace('\\"', '"')
    )
    return f"Action: {match.group('tool')}\nAction Input: {tool_input}"


def _normalize_tool_input(raw: str) -> str:
    """Strip markdown fences around the Action Input."""
    stripped = raw.strip().strip('"').strip("'")
    fenced = _FENCE.match(stripped)
    if fenced is not None:
        return fenced.group(1).strip()
    return stripped


def _action_without_input_label(text: str) -> tuple[str, str] | None:
    """
    Recover tool and input when the «Action Input:» line is missing.

    Returns:
        (tool_name, input), or None if there is no real Action.
    """
    match = _ACTION_BLOCK.search(text)
    if match is None:
        return None
    tool = _normalize_tool_name(match.group("tool"))
    if _ACTION_NAME_FINAL.match(tool) or _ACTION_NAME_NULL.match(tool):
        return None
    return tool, _normalize_tool_input(match.group("rest") or "")


class AssistantReActParser(ReActSingleInputOutputParser):
    """
    Extend the standard ReAct parser.

    If the model uses `Action: Final Answer` + `Action Input: ...`,
    return AgentFinish instead of trying a non-existent tool.
    If Action and Final Answer coexist, prefer the Action (tool).
    `Action: Nessuna` becomes a Final Answer with the Thought, without a retry.
    """

    @override
    def parse(self, text: str) -> AgentAction | AgentFinish:
        """
        Interpret the LLM output as an Action or a Final Answer.

        Input: raw model text.
        Output: AgentFinish for a conclusion (even malformed), otherwise AgentAction.
        """
        call = _from_function_call(text)
        if call is not None:
            text = call
        executable = extract_executable_react_block(text)
        if executable is not None:
            text = executable
            text_to_parse = executable
        else:
            # Typical Gemma case: Action: Final Answer / Action Input: answer.
            final_match = _ACTION_FINAL_ANSWER.search(text)
            if final_match is not None:
                answer = final_match.group(1).strip().strip('"').strip("'")
                return AgentFinish({"output": answer}, text)

            normalized_text = _FINAL_ANSWER_FLEX.sub("Final Answer:", text, count=1)
            text_to_parse = _prefer_action_if_mixed(normalized_text)

        try:
            outcome = super().parse(text_to_parse)
        except OutputParserException as err:
            recovered_action = _action_without_input_label(text_to_parse)
            if recovered_action is None:
                if _is_incomplete_reasoning(text):
                    raise OutputParserException(
                        "Thought incompleto: emetti Action oppure Final Answer breve."
                    ) from err
                recovered = _extract_thought(text) or str(text).strip()
                return AgentFinish({"output": recovered}, text)
            tool, tool_input = recovered_action
            tool = _normalize_tool_name(tool)
            if _ACTION_NAME_NULL.match(tool):
                return AgentFinish({"output": _extract_thought(text)}, text)
            if tool not in _KNOWN_TOOLS:
                _reject_non_executable_tool(tool, text)
            if not _action_block_complete(tool, tool_input):
                if tool in _BLOCK_TOOLS:
                    return AgentFinish({"output": ""}, text)
                _reject_non_executable_tool(tool, text)
            return AgentAction(tool, tool_input, text)
        if isinstance(outcome, AgentAction):
            outcome.tool = _normalize_tool_name(outcome.tool)
            outcome.tool_input = _normalize_tool_input(str(outcome.tool_input))
        if isinstance(outcome, AgentAction) and _ACTION_NAME_FINAL.match(outcome.tool.strip()):
            # Phantom tool: the content is already the user-facing answer.
            return AgentFinish({"output": str(outcome.tool_input).strip()}, text)
        if isinstance(outcome, AgentAction) and _ACTION_NAME_NULL.match(outcome.tool.strip()):
            answer = _extract_thought(text) or str(outcome.tool_input).strip()
            return AgentFinish({"output": answer}, text)
        if isinstance(outcome, AgentAction) and (
            outcome.tool not in _KNOWN_TOOLS
            or not _action_block_complete(outcome.tool, str(outcome.tool_input))
        ):
            if outcome.tool in _BLOCK_TOOLS:
                _raise_incomplete_block(str(outcome.tool), text)
            _reject_non_executable_tool(str(outcome.tool), text)
        return outcome
