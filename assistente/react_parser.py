"""Parser ReAct tollerante: Final Answer malformata → AgentFinish."""

from __future__ import annotations

import re

from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from typing_extensions import override

from assistente.config import MARCATORE_BLOCCO_NEW, MARCATORE_BLOCCO_OLD

# Gemma spesso emette Action/Action Input invece di "Final Answer:".
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
_TOOL_NOTI = (
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
_TOOL_CON_BLOCCO = frozenset({"scrivi_file", "modifica_file"})
_BLOCCO_SCRIVI = re.compile(
    r"Action\s*\d*\s*:\s*`?(?P<tool>scrivi_file)`?(?:\s*\([^)]*\))?\.?\s*\n"
    r"(?:Action\s*\d*\s*Input\s*\d*\s*:\s*)?"
    r"(?P<path>[\w./-]+\.\w+)\s*\n"
    r"---\s*\n"
    r"(?P<body>.*)",
    re.IGNORECASE | re.DOTALL,
)
_BLOCCO_MODIFICA = re.compile(
    r"Action\s*\d*\s*:\s*`?modifica_file`?(?:\s*\([^)]*\))?\.?\s*\n"
    r"(?:Action\s*\d*\s*Input\s*\d*\s*:\s*)?"
    r"(?P<path>[\w./-]+\.\w+)\s*\n"
    rf"{re.escape(MARCATORE_BLOCCO_OLD)}[ \t]*\n"
    r"(?P<old>.*?)"
    rf"^{re.escape(MARCATORE_BLOCCO_NEW)}[ \t]*(?:\n|\Z)"
    r"(?P<new>.*)",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)
_BLOCCO_ALTRO_TOOL = re.compile(
    r"Action\s*\d*\s*:\s*`?(?P<tool>"
    r"leggi_documento|elenca_cartella|elenca_albero|trova_file|"
    r"cerca_testo|crea_cartella|"
    r"data_oggi|riassumi_documento)`?"
    r"(?:\s*\([^)]*\))?\s*\n"
    r"(?:Action\s*\d*\s*Input\s*\d*\s*:\s*)?"
    # riassumi_documento può proseguire con «---» e il path di destinazione.
    r"(?P<input>\S[^\n]*(?:\n[ \t]*---[ \t]*\n[ \t]*\S[^\n]*)?)",
    re.IGNORECASE,
)
_ACTION_BLOCCO = re.compile(
    r"Action\s*\d*\s*:\s*(?P<tool>[^\n]+)(?:\nAction\s*\d*\s*Input\s*\d*\s*:)?\s*"
    r"(?P<rest>.*)",
    re.DOTALL | re.IGNORECASE,
)
_THOUGHT_BLOCCO = re.compile(
    r"Thought\s*:\s*(.*?)(?:\nAction\s*:|\nFinal\s+Answer\s*:|\Z)",
    re.DOTALL | re.IGNORECASE,
)
# Gemma a volte incolla la conclusione sulla riga dell'input: «attualeFinal Answer: …».
_CODA_INCOLLATA = re.compile(
    r"(?:Final\s*Answer|Observation|Thought|Action(?:\s*Input)?)\s*:", re.IGNORECASE
)
_FENCE = re.compile(r"^```(?:[\w+-]*)\s*\n?(.*?)\n?```$", re.DOTALL)
_FINAL_ANSWER_MARKER = "Final Answer:"
_FINAL_ANSWER_FLEX = re.compile(r"final\s+answer\s*:", re.IGNORECASE)
_MARCATORI_RAGIONAMENTO = (
    "scrittura consentita",
    "scrivi_file",
    "modifica_file",
    "non è autorizzata",
    "non posso usare",
)
_THOUGHT_MAX_COME_RISPOSTA = 400
OSSERVAZIONE_BLOCCO_INCOMPLETO = (
    "Action Input del tool di scrittura incompleto (troncato). "
    "Ripeti SOLO con Thought breve e Action completa:\n"
    "Action: modifica_file\n"
    "Action Input: path_relativo\n"
    f"{MARCATORE_BLOCCO_OLD}\n"
    "<testo esatto dal file>\n"
    f"{MARCATORE_BLOCCO_NEW}\n"
    "<sostituzione>\n"
    "Per scrivi_file: path .txt, riga ---, body. "
    "Niente Final Answer finché manca Observation di successo."
)


def _preferisci_action_se_mista(text: str) -> str:
    """
    Se il testo ha sia Action sia Final Answer, tiene solo il blocco Action.

    Gemma spesso chiude con Final Answer nello stesso turno dello scrivi_file;
    il parser base rifiuta quel mix e il tool non viene mai eseguito.
    """
    if _FINAL_ANSWER_MARKER not in text:
        return text
    match_action = _ACTION_MARKER.search(text)
    if match_action is None:
        return text
    posizione_final = text.find(_FINAL_ANSWER_MARKER)
    posizione_action = match_action.start()
    # Final Answer dopo Action → tronca prima di Final Answer (esegui il tool).
    if posizione_final > posizione_action:
        return text[:posizione_final].rstrip()
    # Final Answer prima di Action → parti da Action.
    solo_action = text[posizione_action:]
    if _FINAL_ANSWER_MARKER in solo_action:
        return solo_action.split(_FINAL_ANSWER_MARKER, 1)[0].rstrip()
    return solo_action


def _e_ragionamento_incompleto(text: str) -> bool:
    """True se il testo è un Thought/CoT senza conclusione ReAct utilizzabile."""
    if not text or _FINAL_ANSWER_FLEX.search(text):
        return False
    basso = text.casefold()
    if any(marcatore in basso for marcatore in _MARCATORI_RAGIONAMENTO) and len(text) > 200:
        return True
    if "thought:" in basso and len(text) > _THOUGHT_MAX_COME_RISPOSTA:
        return True
    return len(text) > 800 and "action:" not in basso


def _estrai_thought(text: str) -> str:
    """Restituisce il Thought, o il testo senza marcatori Action."""
    match = _THOUGHT_BLOCCO.search(text)
    if match is not None and match.group(1).strip():
        return match.group(1).strip()
    return text.strip()


def _normalizza_nome_tool(nome: str) -> str:
    """Toglie markdown e riconosce un tool noto anche se il CoT aggiunge rumore."""
    pulito = nome.strip().strip("`*_.,:;\"'").strip()
    basso = pulito.casefold()
    for noto in _TOOL_NOTI:
        if basso == noto or basso.startswith(f"{noto} ") or basso.startswith(f"{noto}`"):
            return noto
    return pulito


_CODA_PROTOCOLLO_PATCH = re.compile(
    rf"\n(?:{re.escape(MARCATORE_BLOCCO_OLD)}|{re.escape(MARCATORE_BLOCCO_NEW)})"
    r"|^(?:Thought|Action|Action Input)\s*:",
    re.IGNORECASE | re.MULTILINE,
)


def _taglia_coda_protocollo(body: str) -> str:
    """Rimuove Action concatenate, marker patch extra o Final Answer dopo il payload."""
    taglio = _ACTION_MARKER.search(body)
    if taglio is not None:
        body = body[: taglio.start()]
    finale = re.search(r"\nFinal\s+Answer\s*:", body, re.IGNORECASE)
    if finale is not None:
        body = body[: finale.start()]
    extra_patch = _CODA_PROTOCOLLO_PATCH.search(body)
    if extra_patch is not None:
        body = body[: extra_patch.start()]
    return body


def _ricostruisci_scrivi(match: re.Match[str]) -> str | None:
    """Ricostruisce un Action Input scrivi_file eseguibile dal match regex."""
    body = _taglia_coda_protocollo(match.group("body")).strip("\n")
    path = match.group("path").strip()
    tool = match.group("tool").strip().casefold()
    if not path or not body.strip():
        return None
    return f"Action: {tool}\nAction Input: {path}\n---\n{body}"


def _ricostruisci_modifica(match: re.Match[str]) -> str | None:
    """Ricostruisce un Action Input modifica_file eseguibile dal match regex."""
    old_block = _taglia_coda_protocollo(match.group("old")).strip("\n")
    new_block = _taglia_coda_protocollo(match.group("new")).strip("\n")
    path = match.group("path").strip()
    if not path or not old_block.strip():
        return None
    return (
        f"Action: modifica_file\nAction Input: {path}\n"
        f"{MARCATORE_BLOCCO_OLD}\n{old_block}\n"
        f"{MARCATORE_BLOCCO_NEW}\n{new_block}"
    )


def _azione_blocco_completa(tool: str, tool_input: str) -> bool:
    """True se i tool multilineari hanno un payload eseguibile, altrimenti True."""
    if tool not in _TOOL_CON_BLOCCO:
        return True
    return (
        estrai_blocco_react_eseguibile(f"Action: {tool}\nAction Input: {tool_input}")
        is not None
    )


def _raise_blocco_incompleto(tool: str, text: str) -> None:
    """Chiede un retry LLM invece di chiudere il turno con output vuoto."""
    raise OutputParserException(
        f"Action {tool} incompleta: completa i marcatori di blocco.",
        observation=OSSERVAZIONE_BLOCCO_INCOMPLETO,
        llm_output=text,
        send_to_llm=True,
    )


def _rifiuta_tool_non_eseguibile(tool: str, text: str) -> None:
    """Tool sconosciuto o blocco incompleto: Observation, non una risposta vuota."""
    if tool in _TOOL_CON_BLOCCO:
        _raise_blocco_incompleto(tool, text)
    raise OutputParserException(
        f"Action {tool} non eseguibile: usa un tool noto con Action Input completo.",
        observation=(
            f"Tool non eseguibile ({tool}). "
            "Emetti Action con un tool noto e Action Input completo."
        ),
        llm_output=text,
        send_to_llm=True,
    )


def estrai_blocco_react_eseguibile(text: str) -> str | None:
    """
    Isola l'ultima Action completa, ignorando i piani CoT tipo «Observation: Result».

    Returns:
        Blocco ReAct minimo, oppure None se non c'è un'invocazione eseguibile.
    """
    if not text:
        return None
    candidati: list[tuple[int, str | None]] = [
        (match.start(), _ricostruisci_scrivi(match))
        for match in _BLOCCO_SCRIVI.finditer(text)
    ]
    candidati.extend(
        (match.start(), _ricostruisci_modifica(match))
        for match in _BLOCCO_MODIFICA.finditer(text)
    )
    if candidati:
        candidati = [(pos, block) for pos, block in candidati if block]
    if candidati:
        candidati.sort(key=lambda item: item[0])
        return candidati[-1][1]
    altri = list(_BLOCCO_ALTRO_TOOL.finditer(text))
    if not altri:
        return None
    match = altri[-1]
    tool = match.group("tool").casefold()
    tool_input = match.group("input")
    coda = _CODA_INCOLLATA.search(tool_input)
    if coda is not None:
        tool_input = tool_input[: coda.start()]
    tool_input = tool_input.strip().strip("`")
    if not tool_input or tool_input.casefold().startswith("observation"):
        return None
    return f"Action: {tool}\nAction Input: {tool_input}"


# Gemma 4 a volte usa la propria sintassi di chiamata: tool(action_input: '...').
_CHIAMATA_FUNZIONE = re.compile(
    r"^\s*`?(?P<tool>[a-z_]+)\((?:\s*(?:action_input|input)\s*[:=]\s*)?"
    r"(?P<quote>['\"])(?P<input>.*)(?P=quote)\s*\)`?\s*$",
    re.DOTALL,
)


def _da_chiamata_funzione(text: str) -> str | None:
    """
    Converte una chiamata in stile funzione nel blocco ReAct equivalente.

    Returns:
        Testo «Action / Action Input», oppure None se non è una chiamata nota.
    """
    match = _CHIAMATA_FUNZIONE.match(text)
    if match is None or match.group("tool") not in _TOOL_NOTI:
        return None
    tool_input = (
        match.group("input")
        .replace("\\n", "\n")
        .replace("\\'", "'")
        .replace('\\"', '"')
    )
    return f"Action: {match.group('tool')}\nAction Input: {tool_input}"


def _normalizza_tool_input(raw: str) -> str:
    """Toglie fence markdown intorno all'Action Input."""
    stripped = raw.strip().strip('"').strip("'")
    fenced = _FENCE.match(stripped)
    if fenced is not None:
        return fenced.group(1).strip()
    return stripped


def _action_senza_etichetta_input(text: str) -> tuple[str, str] | None:
    """
    Recupera tool e input quando manca la riga «Action Input:».

    Returns:
        (nome_tool, input) oppure None se non c'è un Action reale.
    """
    match = _ACTION_BLOCCO.search(text)
    if match is None:
        return None
    tool = _normalizza_nome_tool(match.group("tool"))
    if _ACTION_NAME_FINAL.match(tool) or _ACTION_NAME_NULL.match(tool):
        return None
    return tool, _normalizza_tool_input(match.group("rest") or "")


class AssistenteReActParser(ReActSingleInputOutputParser):
    """
    Estende il parser ReAct standard.

    Se il modello usa `Action: Final Answer` + `Action Input: ...`,
    restituisce AgentFinish invece di tentare un tool inesistente.
    Se Action e Final Answer coesistono, preferisce Action (tool).
    `Action: Nessuna` diventa Final Answer col Thought, senza retry.
    """

    @override
    def parse(self, text: str) -> AgentAction | AgentFinish:
        """
        Interpreta l'output LLM come Action o Final Answer.

        Input: testo grezzo del modello.
        Output: AgentFinish se conclusione (anche malformata), altrimenti AgentAction.
        """
        chiamata = _da_chiamata_funzione(text)
        if chiamata is not None:
            text = chiamata
        eseguibile = estrai_blocco_react_eseguibile(text)
        if eseguibile is not None:
            text = eseguibile
            testo_da_parsare = eseguibile
        else:
            # Caso tipico Gemma: Action: Final Answer / Action Input: risposta.
            match_finale = _ACTION_FINAL_ANSWER.search(text)
            if match_finale is not None:
                risposta = match_finale.group(1).strip().strip('"').strip("'")
                return AgentFinish({"output": risposta}, text)

            testo_normalizzato = _FINAL_ANSWER_FLEX.sub("Final Answer:", text, count=1)
            testo_da_parsare = _preferisci_action_se_mista(testo_normalizzato)

        try:
            esito = super().parse(testo_da_parsare)
        except OutputParserException as err:
            recuperato = _action_senza_etichetta_input(testo_da_parsare)
            if recuperato is None:
                if _e_ragionamento_incompleto(text):
                    raise OutputParserException(
                        "Thought incompleto: emetti Action oppure Final Answer breve."
                    ) from err
                recovered = _estrai_thought(text) or str(text).strip()
                return AgentFinish({"output": recovered}, text)
            tool, tool_input = recuperato
            tool = _normalizza_nome_tool(tool)
            if _ACTION_NAME_NULL.match(tool):
                return AgentFinish({"output": _estrai_thought(text)}, text)
            if tool not in _TOOL_NOTI:
                _rifiuta_tool_non_eseguibile(tool, text)
            if not _azione_blocco_completa(tool, tool_input):
                if tool in _TOOL_CON_BLOCCO:
                    return AgentFinish({"output": ""}, text)
                _rifiuta_tool_non_eseguibile(tool, text)
            return AgentAction(tool, tool_input, text)
        if isinstance(esito, AgentAction):
            esito.tool = _normalizza_nome_tool(esito.tool)
            esito.tool_input = _normalizza_tool_input(str(esito.tool_input))
        if isinstance(esito, AgentAction) and _ACTION_NAME_FINAL.match(esito.tool.strip()):
            # Tool fantasma: il contenuto è già la risposta utente.
            return AgentFinish({"output": str(esito.tool_input).strip()}, text)
        if isinstance(esito, AgentAction) and _ACTION_NAME_NULL.match(esito.tool.strip()):
            risposta = _estrai_thought(text) or str(esito.tool_input).strip()
            return AgentFinish({"output": risposta}, text)
        if isinstance(esito, AgentAction) and (
            esito.tool not in _TOOL_NOTI
            or not _azione_blocco_completa(esito.tool, str(esito.tool_input))
        ):
            if esito.tool in _TOOL_CON_BLOCCO:
                _raise_blocco_incompleto(str(esito.tool), text)
            _rifiuta_tool_non_eseguibile(str(esito.tool), text)
        return esito
