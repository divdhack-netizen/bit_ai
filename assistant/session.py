"""Agent session: LLM/executor cache, prompt and turn orchestration."""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.agents import AgentAction
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_ollama import OllamaLLM

from assistant.config import (
    DEFAULT_MODEL_ALIAS,
    LLM_CONTEXT_TOKENS,
    WRITABLE_EXTENSIONS,
    ConfigRuntime,
    MAX_AGENT_ITERATIONS,
    MAX_GENERATED_TOKENS,
    FILE_SEPARATOR,
    resolve_ollama_tag,
)
from assistant.console import ConsoleUI
from assistant.date_time import temporal_context, create_date_tool
from assistant.direct_pipeline import DIRECT_SYSTEM_PREFIX, DirectAnswerPipeline
from assistant.evaluate_turn import evaluate_turn
from assistant.file_service import WorkspaceFileService
from assistant.host import HostCapabilities, can_expand_context
from assistant.inference_profiles import (
    InferenceRoute,
    expanded_context,
    profile_for_route,
)
from assistant.memory import (
    CHAT_ONLY_QA_RULE,
    ConversationMemory,
    _outcome_line,
    history_prompt_block,
)
from assistant.metrics import LLMMetricsAccumulator, LLMTurnStats
from assistant.ollama_runtime import OllamaRuntime, StreamingOllamaLLM
from assistant.energy_policy import (
    lightest_alias,
    prepare_energy_turn,
)
from assistant.comprehension import (
    PATH_FILE,
    TurnComprehension,
    extract_named_folder,
    format_comprehension_block,
    normalize,
    resolve_destination,
    local_card,
)
from assistant.turn_policy import (
    ANTI_GOLD_PLATING,
    TurnDecision,
    _is_local_chat,
    capabilities_react,
    decision_from_schema,
    open_react_decision,
    greeting_decision,
    is_history_spill,
    format_intent_context,
    named_document_path,
)
from assistant.turn_schema import (
    NUM_PREDICT_SCHEMA,
    SCHEMA_PREFIX,
    FollowupBundle,
    TurnSchema,
    format_schema_prompt,
    repair_json_schema,
    schema_json_ollama,
)
from assistant.turn_validator import (
    MISSING_FILE_MESSAGE,
    ValidationContext,
    ValidationOutcome,
    validate_turn,
)
from assistant.project_memory import format_project_memory_block
from assistant.kv_prefix import PrefixContinuation
from assistant.prompt_budget import (
    PromptBudget,
    estimate_tokens,
    limit_scratchpad_steps,
)
from assistant.react_parser import AssistantReActParser
from assistant.process_log import (
    EventLog,
    classify_turn_outcome,
)
from assistant.readers import UNSUPPORTED_FORMAT_INSTRUCTION
from assistant.summary import create_summary_tool
from assistant.tools import REPEATED_READ_WARNING
from assistant.thermal import (
    ThermalTurnContext,
    ThermalState,
    adaptive_pause_between_iterations,
    prepare_turn,
)
from assistant.tools import (
    apply_fallback,
    create_file_tools,
)

_LIMIT_MARKERS = ("iteration limit", "time limit")
_FORMAT_ERROR_MESSAGE = (
    "Formato non valido. Esempio corretto senza tool:\n"
    "Thought: ho la risposta\n"
    "Final Answer: testo per l'utente\n"
    "Esempio con tool:\n"
    "Thought: elenco la cartella richiesta\n"
    "Action: elenca_cartella\n"
    "Action Input: assistente\n"
    "Non usare Action: Nessuna/None. Action deve essere un tool reale."
)


def _path_from_tool_input(tool_input: object) -> str:
    """First line or explicit path, for the placeholder of an omitted observation."""
    if isinstance(tool_input, dict):
        for key in ("path", "file_path", "sorgente"):
            if key in tool_input:
                return str(tool_input[key])[:180]
        return ""
    text = str(tool_input or "").strip()
    if not text:
        return ""
    return text.splitlines()[0][:180]


def _limit_react_steps(
    steps: list[tuple[AgentAction, str]],
    max_tokens: int,
) -> list[tuple[AgentAction, str]]:
    """Apply microcompact and the token cap before the next LLM call."""
    packed = [
        (
            str(action.tool),
            _path_from_tool_input(action.tool_input),
            str(action.log),
            str(observation),
        )
        for action, observation in steps
    ]
    limited = limit_scratchpad_steps(packed, max_tokens)
    trimmed: list[tuple[AgentAction, str]] = []
    for (action, observation), (log, content) in zip(steps, limited, strict=True):
        if log == str(action.log) and content == str(observation):
            trimmed.append((action, observation))
            continue
        trimmed.append((AgentAction(action.tool, action.tool_input, log), content))
    return trimmed


# Tools whose successful result is already a useful answer for the user.
_DIRECT_ANSWER_TOOLS = frozenset(
    # Not cerca_testo: its stem matches are often noise, not an answer.
    {"riassumi_documento", "elenca_cartella", "elenca_albero", "trova_file"}
)
_DOCUMENT_EXCERPT_CHARS = 1500
LIMIT_FALLBACK_NOTE = (
    "(Limite di passi raggiunto: risposta ricavata dall'ultimo risultato utile.)"
)


def _answer_from_steps(steps: list[tuple[AgentAction, str]]) -> str | None:
    """
    At the iteration limit, derive the answer from the last useful step instead of
    showing «Agent stopped…»: often the data had already been read.

    Returns:
        Text for the user, or None if no step produced anything.
    """
    for action, observation in reversed(steps):
        tool = str(action.tool)
        text = str(observation).removeprefix(REPEATED_READ_WARNING).strip()
        if not text:
            continue
        error = text.startswith("Errore")
        # An unsupported format is the right answer, not a failure to skip.
        if tool == "leggi_documento" and error and "non supportato" in text:
            return text.removeprefix("Errore:").replace(
                UNSUPPORTED_FORMAT_INSTRUCTION, ""
            ).strip()
        if error:
            continue
        if tool in _DIRECT_ANSWER_TOOLS and not text.startswith("Nessun"):
            return text
        if tool == "leggi_documento":
            path = str(action.tool_input).strip()
            excerpt = text[:_DOCUMENT_EXCERPT_CHARS].rstrip()
            tail = "…" if len(text) > _DOCUMENT_EXCERPT_CHARS else ""
            return f"Ho letto {path}. Inizio del documento:\n\n{excerpt}{tail}"
    return None


def _parsing_error_observation(error: OutputParserException) -> str:
    """Retry Observation: truncated write payload vs generic ReAct format."""
    observation = str(getattr(error, "observation", "") or "")
    if observation:
        return observation
    if "thought incompleto" in str(error).casefold():
        return "Thought incompleto: emetti Action con un tool reale oppure Final Answer breve."
    return _FORMAT_ERROR_MESSAGE


_EMPTY_HISTORY_MESSAGE = "Nessun turno precedente da salvare."
_VERIFY_WORDS = re.compile(
    r"\b(?:verifica|verificalo|controlla|controllalo|ricontrolla|ricontrollalo)\b",
    re.IGNORECASE,
)
_HISTORY_MIN_TOKENS = 128
_LIGHT_HISTORY_MAX_TOKENS = 1024
_LIGHT_HISTORY_QUOTA = 5
_HEAVY_HISTORY_QUOTA = 3
# Aligned with the bar refresh during Live: avoids thermal I/O on every token.
_LIVE_METRICS_INTERVAL_SEC = 0.25
# Closed corpus: the only source is the documents in the launch folder.
CORPUS_RULE = (
    "Unica fonte: i documenti della cartella. Non usare conoscenze tue né Internet. "
    "Se l'utente nomina un file: leggi_documento su quel file (trova_file se il "
    "path non è esatto); se il tool dà errore, riporta l'errore. "
    "Per una domanda: cerca_testo con le parole chiave, poi leggi_documento sui file "
    "trovati. Se non trova nulla, al massimo un secondo cerca_testo con sinonimi "
    "e le sole immagini pertinenti; poi fermati. Cita il file (e la pagina) "
    "di ogni informazione. Se i documenti non contengono la risposta, dillo "
    "chiaramente: non è presente nei documenti."
)
_AGENT_IDENTITY = (
    "Parli in italiano informale, dando del tu. Lavori solo sui documenti della "
    "cartella corrente (txt, md, pdf, docx, pages, immagini) e scrivi solo file .txt. "
    f"{CORPUS_RULE} Non inventare fatti, path o operazioni su disco. "
    f"Usa i tool che servono e fermati appena il risultato è verificato. {ANTI_GOLD_PLATING}"
)
_TEMPLATE_REACT = """{identity}
Tool disponibili:
{tools}

Protocollo:
Thought: piano essenziale
Action: uno tra [{tool_names}]
Action Input: input del tool
Observation: risultato
Thought: completato
Final Answer: risposta

Se non serve un tool, dopo Thought scrivi subito Final Answer.
Non usare Action: Nessuna/None. Mai Action: Final Answer.

Question: {input}
Thought:{agent_scratchpad}"""

CapabilitySet = tuple[str, ...]
EXECUTOR_CACHE_LIMIT = 4


@dataclass(frozen=True, slots=True)
class ExecutorKey:
    """Identify an AgentExecutor without confusing it with the Ollama weights."""

    alias: str
    max_iter: int
    num_predict: int
    context_tokens: int
    capabilities: CapabilitySet
    write_allowed: bool
    reasoning: bool


class TurnCancelled(RuntimeError):
    """Signal a cancellation requested by the user during the stream."""

def _build_task_description(
    request: str,
    full_verification: bool,
    decision: TurnDecision,
    history: str = "",
    *,
    project_memory: str = "",
    comprehension: TurnComprehension | None = None,
    note: str = "",
) -> str:
    """
    Build compact ReAct instructions using the decision already taken by routing.

    The corpus rule lives in the identity, in the stable prefix: it is not repeated here.

    Returns:
        Request enriched with the workflow, file gate and history it needs.
    """
    write_allowed = decision.write_allowed
    instructions: list[str] = []
    if project_memory:
        instructions.append(project_memory.strip())
    instructions.append("Workflow: minimo di Action, poi Final Answer.")
    if history:
        instructions.append(
            "Lo STORICO serve al filo del discorso; i fatti vanno verificati nei documenti."
        )
    folder = extract_named_folder(request)
    if write_allowed:
        destination = (
            _write_destination(comprehension.paths)
            if comprehension is not None
            else ""
        )
        if destination:
            instructions.append(
                f"Destinazione già nota: {destination}. "
                "Usa scrivi_file su quel path, senza chiedere il nome."
            )
            if note:
                instructions.append(f"{note} Non creare cartelle per questo.")
        elif folder is not None:
            instructions.append(
                f"Cartella già nominata: {folder}. "
                "Usa crea_cartella se manca, poi scrivi_file su un .txt. "
                "Niente elenca_cartella obbligatorio. Action Input senza slash iniziale."
            )
        else:
            instructions.append(
                "Path incerto: trova_file o elenca_cartella, poi leggi_documento."
            )
        instructions.append(
            "crea_cartella solo se l'utente chiede una cartella. File nuovo: scrivi_file "
            f"(path relativo .txt SENZA slash, riga {FILE_SEPARATOR}, body). "
            "Si scrivono solo file .txt: mai sovrascrivere i documenti sorgente. "
            "Patch su un .txt già letto: modifica_file. Un path per Action. "
            "Niente virgole nel path. Non dichiarare successo senza Observation positiva."
        )
    else:
        if folder is not None:
            instructions.append(
                f"elenca_cartella Action Input obbligatorio: {folder}. "
                "Mai '.' o la root del workspace. Per più livelli: elenca_albero."
            )
        else:
            instructions.append(
                "Contenuto: cerca_testo. Nome incerto: trova_file. "
                "Cartella attuale: elenca_cartella con Action Input . "
                "Panoramica: elenca_albero."
            )
        instructions.append(CHAT_ONLY_QA_RULE)

    # The dedicated tool avoids copying the whole document into the scratchpad.
    instructions.append("Per un documento usa riassumi_documento una sola volta.")
    # Check mode re-reads the file to verify the content actually persisted.
    if full_verification and write_allowed:
        instructions.append(
            "Dopo ogni scrittura rileggi lo stesso path e cita soltanto file verificati. "
            "La verifica inline è obbligatoria prima di Final Answer."
        )
    instructions.append(
        "Final Answer: esito breve. "
        "Per elenca_cartella o elenca_albero: elenca i nomi osservati, "
        "senza riassumere e senza ricopiare metadati o dimensioni."
    )
    instructions.append(
        "Thought: esplicita i passaggi essenziali prima di ogni Action."
    )
    task_block = (
        format_comprehension_block(comprehension) if comprehension is not None else ""
    )
    return "\n".join(
        part
        for part in (
            *instructions,
            history_prompt_block(history).rstrip(),
            format_intent_context(decision),
            task_block,
            f"RICHIESTA:\n{request.strip()}",
            (
                "Action Input di scrivi_file: path relativo .txt, "
                f"riga {FILE_SEPARATOR}, body grezzo. "
                "Nel body niente fence markdown né Thought/Action."
                if write_allowed
                else ""
            ),
        )
        if part
    ).strip()


def _requires_full_verification(request: str, full_verification_active: bool) -> bool:
    return full_verification_active or bool(_VERIFY_WORDS.search(request))


# Open ReAct does not shorten the budget based on the sentence wording.
OPEN_REACT_ITERATIONS = 7


def _extract_single_requested_path(
    request: str,
    service: WorkspaceFileService,
) -> str | None:
    """
    Extract and validate the single document destination named in the request.

    Args:
        request: Original text that explicitly authorizes the write.
        service: Workspace boundary used to normalize each candidate.

    Returns:
        Relative path when there is a single valid destination, otherwise None.
    """
    normalized_paths: list[str] = []
    # Validate every candidate so as not to silently pick an ambiguous path.
    for raw_path in PATH_FILE.findall(request):
        normalized_path = service.normalize_file_path(raw_path)
        # Disallowed paths must not reach the direct write.
        if normalized_path is not None:
            normalized_paths.append(normalized_path)
    unique_paths = list(dict.fromkeys(normalized_paths))
    # The deterministic path handles a single document per request.
    if len(unique_paths) != 1:
        return None
    return unique_paths[0]


def _write_destination(paths: tuple[str, ...]) -> str:
    """First writable path (.txt): the others are source documents to read."""
    for path in paths:
        if Path(path).suffix.casefold() in WRITABLE_EXTENSIONS:
            return path
    return ""


def _adapt_txt_destinations(
    schema: TurnSchema | None,
    exists: Callable[[str], bool],
) -> tuple[TurnSchema | None, str]:
    """
    Turn new destinations with another extension into .txt.

    A path that already exists is a document to read and stays as it is; one that
    does not exist, in a write turn, is the requested destination.

    Returns:
        Adapted schema and a note for the user (empty if nothing changed).
    """
    if schema is None or not (schema.write or schema.deliver == "file"):
        return schema, ""
    adapted: list[str] = []
    changes: list[tuple[str, str]] = []
    for path in schema.paths:
        suffix = Path(path).suffix.casefold()
        if suffix and suffix not in WRITABLE_EXTENSIONS and not exists(path):
            base = Path(path).with_suffix(".txt")
            new = base.as_posix()
            index = 2
            # The .txt with the same name may be a source document: never overwrite it.
            while exists(new):
                new = base.with_name(f"{base.stem}_{index}.txt").as_posix()
                index += 1
            changes.append((path, new))
            adapted.append(new)
        else:
            adapted.append(path)
    if not changes:
        return schema, ""
    listing = ", ".join(f"{old} → {new}" for old, new in changes)
    note = f"Nota: scrivo solo file .txt, quindi uso {listing}."
    subject = schema.subject
    for old, new in changes:
        subject = subject.replace(old, new)
    return replace(schema, paths=tuple(adapted), subject=subject), note


def _comprehension_from_schema(
    schema: TurnSchema | None,
    request: str = "",
) -> TurnComprehension:
    """
    Theme and paths from the schema, without building a COMPITO from the wording.

    A subject that repeats the start of the request (local card) does not become
    COMPITO: it would duplicate the prompt's RICHIESTA line.
    """
    if schema is None:
        return TurnComprehension("", ())
    task = schema.subject.strip()
    if task and normalize(request).casefold().startswith(task.casefold()):
        task = ""
    return TurnComprehension(task=task, paths=schema.paths)


class CallbackUI(BaseCallbackHandler):
    """Single callback for streaming, timeline and metrics of both agent paths."""

    def __init__(
        self,
        ui: ConsoleUI,
        pause_between_iterations_sec: float = 0.0,
        *,
        initial_state: ThermalState = ThermalState.NOMINAL,
        burn_protection: bool = False,
        max_context: int = LLM_CONTEXT_TOKENS,
        cancel_event: threading.Event | None = None,
        log: EventLog | None = None,
    ) -> None:
        """
        Initialize the per-turn state and the accumulator for the effective LLM profile.

        Args:
            ui: Console receiving tokens, timeline and stats.
            pause_between_iterations_sec: Base thermal backoff between ReAct generations.
            initial_state: Thermal snapshot used to adapt the pauses.
            burn_protection: Enables dynamic recomputation of the backoff.
            max_context: Token window of the model used on the current path.
            cancel_event: Shared signal to interrupt the turn.
            log: Crash/error observability; None uses an in-memory log.
        """
        super().__init__()
        self._ui = ui
        self._pause_sec = pause_between_iterations_sec
        self._initial_state = initial_state
        self._burn_protection = burn_protection
        self._cancel_event = cancel_event
        self._log = log if log is not None else EventLog()
        self._metrics = LLMMetricsAccumulator(max_context)
        self._next_metrics_redraw = 0.0
        self._thinking_open = False
        self._current_tool: str | None = None
        self._ui.set_metrics_source(self._metrics.snapshot)

    def _raise_if_cancelled(self) -> None:
        """Interrupt the turn if the user requested cancellation."""
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise TurnCancelled("Turno annullato dall'utente.")

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        """Apply the thermal pause and start the call's TTFT timer."""
        del serialized, kwargs
        self._raise_if_cancelled()
        pause = self._pause_sec
        # With burn protection the pause reflects the updated thermal state.
        if self._burn_protection:
            pause = adaptive_pause_between_iterations(self._initial_state, self._pause_sec)
        # The LLM timer starts after throttling, to measure the provider, not the thermal wait.
        if pause > 0:
            wait_started = time.perf_counter()
            time.sleep(pause)
            self._metrics.record_thermal_wait(
                time.perf_counter() - wait_started
            )
        self._metrics.start_call()
        self._log.heartbeat()
        if prompts:
            self._metrics.prompt_estimate(prompts[0])
        self._publish_metrics_now()

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Route native CoT to the timeline and the useful text to the panel."""
        self._raise_if_cancelled()
        self._log.heartbeat()
        if kwargs.get("thinking"):
            if not token:
                return
            self._thinking_open = True
            self._emit_thinking_token(token)
            return
        if not token:
            return
        self._close_open_thinking()
        self._emit_content_token(token)

    def _prepare_token_emission(self, token: str) -> float:
        """Record TTFT/count and return the render start time."""
        self._metrics.record_first_token()
        self._metrics.record_stream_token(token)
        self._update_live_metrics()
        return time.perf_counter()

    def _emit_thinking_token(self, token: str) -> None:
        """Write a native CoT fragment and measure its rendering."""
        started = self._prepare_token_emission(token)
        self._ui.write_thinking(token)
        self._metrics.record_rendering(time.perf_counter() - started)

    def _emit_content_token(self, token: str) -> None:
        """Write an answer fragment and measure its rendering."""
        started = self._prepare_token_emission(token)
        self._ui.write_token(token)
        self._metrics.record_rendering(time.perf_counter() - started)

    def _close_open_thinking(self) -> None:
        """Close the Pensiero stream if still visible."""
        if not self._thinking_open:
            return
        self._ui.close_thinking()
        self._thinking_open = False

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Close a successful generation and publish the updated snapshot."""
        del kwargs
        self._log.heartbeat()
        self._close_open_thinking()
        self._ui.reset_llm_call()
        self._ui.set_turn_metrics(self._metrics.record(response))

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Close timer and stream when the LLM provider raises an error."""
        del kwargs
        if not isinstance(error, (KeyboardInterrupt, TurnCancelled)):
            self._log.record_llm_error(error)
        self._close_open_thinking()
        self._ui.reset_llm_call()
        self._ui.set_turn_metrics(self._metrics.record_llm_error())

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        """Record the ReAct action before invoking the tool."""
        del kwargs
        # Parser pseudo-actions do not represent a useful tool for the timeline.
        if action.tool != "_Exception":
            self._current_tool = action.tool
            self._ui.record_tool_action(action.tool, str(action.tool_input))

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        """Start the timer and timeline of a tool launched by AgentExecutor."""
        del input_str, kwargs
        name = str(serialized.get("name", "tool"))
        self._current_tool = name
        self._metrics.start_tool()
        self._log.heartbeat()
        self._ui.start_tool_execution(name)

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """End a ReAct tool and update timeline and metrics."""
        del kwargs
        observation = str(output)
        self._current_tool = None
        self._log.heartbeat()
        self._ui.end_tool_execution()
        self._ui.emit_tool_result(observation)
        self._ui.set_turn_metrics(self._metrics.end_tool())

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        """End timer and timeline when a ReAct tool raises an error."""
        del kwargs
        name = self._current_tool or "tool"
        self._log.record_tool_error(name, error)
        self._current_tool = None
        self._ui.end_tool_execution()
        self._ui.emit_tool_error(str(error))
        self._ui.set_turn_metrics(self._metrics.end_tool())

    def record_diagnostics(
        self,
        message: str,
        exception_type: str | None = None,
    ) -> None:
        """Propagate a pipeline diagnostic (user text unchanged) to the log."""
        self._log.record_diagnostics(message, exception_type=exception_type)

    def last_metrics(self) -> LLMTurnStats | None:
        """LLM snapshot of the turn, or None if there has been no call yet."""
        return self._metrics.snapshot()

    def _publish_metrics_now(self) -> None:
        """Force a redraw of the bar with the current snapshot."""
        self._next_metrics_redraw = (
            time.monotonic() + _LIVE_METRICS_INTERVAL_SEC
        )
        self._ui.set_turn_metrics(self._metrics.snapshot())

    def _update_live_metrics(self) -> None:
        """
        Update the bar numbers without flooding thermal I/O on every token.

        With Live active the redraw already follows the panel refresh; without Live
        (ReAct Thought) it redraws at 4 Hz.
        """
        snapshot = self._metrics.snapshot()
        if self._ui.has_live_stream:
            self._ui.update_turn_metrics(snapshot)
            return
        now = time.monotonic()
        if now < self._next_metrics_redraw:
            self._ui.update_turn_metrics(snapshot)
            return
        self._next_metrics_redraw = now + _LIVE_METRICS_INTERVAL_SEC
        self._ui.set_turn_metrics(snapshot)


def _enrich_answer(result: str, written_paths: list[str]) -> str:
    """
    Align the Final Answer with the files actually persisted in the turn.

    Args:
        written_paths: Files only, never folders: a created directory is not a
            «written file» and must not be listed in place of the missing content.

    Returns:
        The original answer, or an enriched/corrected one if the cited paths do not match.
    """
    cited_paths = list(dict.fromkeys(PATH_FILE.findall(result)))
    written_set = set(written_paths)
    missing_cited = [path for path in cited_paths if path not in written_set]
    # The FA named files the tool did not materialize: correct the claim.
    if missing_cited:
        written = ", ".join(written_paths) if written_paths else "(nessuno)"
        missing = ", ".join(missing_cited)
        return (
            f"{result.rstrip()}\n\n"
            f"Attenzione: path citati ma NON presenti su disco: {missing}. "
            f"File effettivamente scritti in questo turno: {written}."
        )
    if not written_paths:
        return result
    listing = ", ".join(written_paths)
    low = result.lower()
    if any(marker in low for marker in _LIMIT_MARKERS):
        return f"Operazione interrotta per limite iterazioni. File scritti su disco: {listing}"
    if set(cited_paths) != written_set:
        return f"{result.rstrip()}\n\nFile verificati su disco: {listing}"
    return result

class AgentSession:
    """Session orchestrator: policy, memory, folder documents and ReAct."""

    def __init__(
        self,
        ui: ConsoleUI,
        file_service: WorkspaceFileService | None = None,
        llm_factory: Callable[..., OllamaLLM] | None = None,
        ollama_runtime: OllamaRuntime | None = None,
        host: HostCapabilities | None = None,
        log: EventLog | None = None,
    ) -> None:
        """
        Build a session with reusable, isolated dependencies.

        Args:
            ui: The session's console interface.
            file_service: Isolated disk boundary; None uses the current folder.
            llm_factory: Replaceable Ollama constructor, with no global patches in tests.
            ollama_runtime: Optional central runtime, useful in integration tests.
            host: Machine snapshot; None does not widen the budgets (tests and default 16 GB).
            log: Persistent observability; None uses an in-memory log.
        """
        self._ui = ui
        self._host = host
        self._log = log if log is not None else EventLog()
        self._llm_factory = llm_factory or StreamingOllamaLLM
        self._runtime = ollama_runtime or OllamaRuntime(
            self._llm_factory,
            # A mocked factory must not open the Ollama control channel.
            manage_residency=llm_factory is None,
            host=host,
        )
        self._executor_per_key: OrderedDict[ExecutorKey, AgentExecutor] = (
            OrderedDict()
        )
        self._file_service = file_service or WorkspaceFileService()
        # Images go through the vision of the current turn's model.
        self._turn_alias = DEFAULT_MODEL_ALIAS
        # Deterministic turn notice (e.g. destination converted to .txt).
        self._turn_note = ""
        self._file_service.set_image_describer(self._describe_image)
        self._cancel_event = threading.Event()
        self._active_callback: CallbackUI | None = None
        self._last_callback: CallbackUI | None = None
        initial_provider = self._bind_llm_provider(True)
        self._direct_pipeline = DirectAnswerPipeline(
            initial_provider,
            self._ui,
        )
        # The history is local to the session and is not persisted to disk.
        self._memory = ConversationMemory()
        self._continuation = PrefixContinuation()
        # Project memory (AGENTS.md) loaded once at session start.
        self._project_memory = format_project_memory_block(self._file_service.workspace)
        self._turn_fallback_path: list[str] = []
        self._last_document: str | None = None
        self._last_schema: TurnSchema | None = None

    @property
    def log(self) -> EventLog:
        """The session's process log, injected by the CLI or in memory."""
        return self._log

    def clear_memory(self) -> None:
        """Reset the conversation history (the /clear command)."""
        self._memory.clear()
        self._continuation.invalidate()
        self._last_document = None
        self._last_schema = None
        # Also invalidate the file cache: /clear starts again from a fresh context.
        self._file_service.invalidate_read_cache()

    def has_history(self) -> bool:
        """True if the session has at least one stored turn."""
        return self._memory.has_turns()

    def record_history_turn(self, user: str, assistant: str) -> None:
        """Insert a past exchange into the history (benchmarks and follow-up tests)."""
        self._memory.record(user, assistant)

    @property
    def turn_fallback_path(self) -> list[str]:
        """Paths created by the create-only fallback, not by the ReAct tools."""
        return list(self._turn_fallback_path)

    @property
    def workspace(self) -> Path:
        """Confined documents folder used by the file tools."""
        return self._file_service.workspace

    def _exists_in_corpus(self, path: str) -> bool:
        """True if the path already exists in the documents folder."""
        resolved = self._file_service.resolve_file(path)
        return resolved is not None and resolved.absolute.exists()

    def _describe_image(self, image: bytes, name: str) -> str:
        """Vision on the turn's model: no second model is loaded."""
        self._check_cancellation()
        self._log.record_diagnostics(f"vision: {name}")
        return self._runtime.describe_image(self._turn_alias, image, name)

    def unload_active_weights(self) -> None:
        """Unload the resident model to measure a cold start."""
        self._runtime.unload_active()

    def close(self) -> None:
        """Release the Ollama connection and model owned by the session."""
        self._runtime.close()

    def cancel_turn(self) -> None:
        """Request cooperative cancellation of the current generation."""
        self._cancel_event.set()

    def _bind_llm_provider(
        self,
        reasoning: bool,
    ) -> Callable[[str, int, int], OllamaLLM]:
        """Settle the turn's `reasoning` without mutable state on the session."""

        def provider(alias: str, num_ctx: int, num_predict: int) -> OllamaLLM:
            llm = self._runtime.get_llm(
                alias,
                num_ctx,
                num_predict,
                reasoning=reasoning,
            )
            if isinstance(llm, StreamingOllamaLLM):
                llm.attach_continuation(self._continuation)
            return llm

        return provider

    def installed_model_names(self) -> frozenset[str]:
        """Normalized Ollama catalog visible from `/model`."""
        return self._runtime.list_available_model_names()

    def download_model(self, tag: str) -> None:
        """Pull the artifact with progress in the console, then invalidate the catalog."""
        self._runtime.pull_model(tag, self._ui.update_progress)

    def prewarm_model(self, model_alias: str) -> None:
        """
        Preload the weights and print the outcomes to the console right away.

        Args:
            model_alias: Key present in AVAILABLE_MODELS.

        Returns:
            None. Used by `/model` and by the wizard, when stdout is free.
        """
        self._runtime.preload(model_alias)
        self.announce_prewarm(model_alias)

    def announce_prewarm(self, model_alias: str) -> None:
        """
        Print the outcomes already collected by a preload, without repeating the HTTP request.

        Args:
            model_alias: Alias used in the preload, for the fallback tag in the message.

        Returns:
            None. Does not call `preload` again: a second POST would reload the weights.
        """
        fallback = self._runtime.last_tag_fallback
        # Profile tag → installed library alias mapping (e.g. gemma4).
        if fallback is not None:
            self._ui.print_system_message(fallback)
        warning = self._runtime.last_memory_warning
        # An explicit profile is allowed, but the swap must be made visible.
        if warning is not None:
            self._ui.print_system_message(f"Avviso memoria: {warning}")
        cpu_fallback = self._runtime.last_cpu_fallback
        # GPU expected but Ollama loaded on the CPU: it does not block the turn.
        if cpu_fallback is not None:
            self._ui.print_system_message(cpu_fallback)
        server_env = self._runtime.last_server_env_warning
        if server_env is not None:
            self._ui.print_system_message(server_env)
        control_error = self._runtime.last_control_error
        # A missing Ollama does not prevent the CLI from starting: warn before the turn.
        if control_error is not None:
            preferred = resolve_ollama_tag(model_alias)
            active = self._runtime.active_tag or preferred
            self._ui.print_system_message(
                "Preriscaldamento non riuscito: verificare Ollama "
                f"({control_error}). Se il modello manca: "
                f"ollama pull {active}"
            )

    def _write_to_file(self, request: str, answer: str) -> str:
        """
        Persist a text into the single requested .txt destination.

        Args:
            request: Original prompt from which to derive the authorized path.
            answer: Text to save (the conversation history).

        Returns:
            Text with the path confirmation, or diagnostics without a false write.
        """
        extracted = _extract_single_requested_path(request, self._file_service)
        folder = extract_named_folder(request)
        destination = resolve_destination(
            extracted,
            folder,
            self._last_document,
        )
        if destination is not None:
            destination = (
                self._file_service.normalize_file_path(destination) or destination
            )
        # Several destinations or missing paths require explicit disambiguation.
        if destination is None:
            return (
                f"{answer}\n\n"
                "Errore scrittura: indica un unico path file relativo e valido."
            )
        if folder:
            mkdir_result = self._file_service.create_directory(folder)
            if not mkdir_result.succeeded:
                return f"{answer}\n\nErrore scrittura: {mkdir_result.error}"
        content = answer.rstrip() + "\n"
        write_result = self._file_service.write_text(destination, content)
        # The typed result prevents declaring success after a boundary error.
        if not write_result.succeeded:
            return f"{answer}\n\nErrore scrittura: {write_result.error}"
        self._last_document = write_result.path
        return f"{answer}\n\nFile scritto e verificato: {write_result.path}"

    def _spill_history(self, request: str) -> tuple[str, list[str]]:
        """
        Write the turns already in memory to the named document, without searching.

        Without a path it asks for the file name. The text is the one stored in
        full: the prompt formatter cuts long answers.
        """
        self._log.record_diagnostics("riversamento storico senza ricerca")
        if named_document_path(request) is None:
            return self._finalize_turn(
                request,
                MISSING_FILE_MESSAGE,
                write_requested=False,
            ), []
        self._file_service.begin_turn(True)
        result = self._write_to_file(request, self._stored_turns_text())
        written = self._file_service.written_paths
        return self._finalize_turn(
            request,
            result,
            write_requested=True,
            written_paths=written,
        ), written

    def _stored_turns_text(self) -> str:
        """Join the exchanges already recorded, without the prompt's cut."""
        blocks = [
            (
                f"--- Turno {turn.index} ---\n"
                f"Utente: {turn.user}\n"
                f"Assistente: {turn.assistant}"
            )
            for turn in self._memory.turns
        ]
        if not blocks:
            return _EMPTY_HISTORY_MESSAGE
        return "\n\n".join(blocks)

    def run_request(self, request: str, config: ConfigRuntime) -> tuple[str, list[str]]:
        """
        Run a complete turn applying a single policy and isolated services.

        Returns:
            Text answer and the paths materialized in the turn (tools and/or fallback).
        """
        self._cancel_event.clear()
        self._turn_fallback_path = []
        verify = _requires_full_verification(request, config.full_verification)
        energy = prepare_energy_turn(config, self._ui)
        context = prepare_turn(config, self._ui) if config.burn_protection else None
        alias = energy.model_alias
        if context is not None:
            alias = lightest_alias(alias, context.model_alias)
            self._record_throttle_if_cutting(context)
        self._turn_alias = alias
        thinking = config.thinking and (context.thinking if context else True)
        provider = self._bind_llm_provider(thinking)
        self._direct_pipeline._llm_provider = provider
        if is_history_spill(request):
            # Spilling the chat needs neither a card nor a search.
            self._ui.prepare_new_stream_turn()
            self._log.start_turn("direct", alias, request)
            return self._spill_history(request)
        origin, validation, decision = self._classify_turn(
            request,
            alias,
            context,
        )
        if validation is not None and not validation.schema.clarify:
            self._last_schema = validation.schema
        current_schema = validation.schema if validation is not None else None
        current_schema, self._turn_note = _adapt_txt_destinations(
            current_schema,
            self._exists_in_corpus,
        )
        if self._turn_note:
            self._ui.print_system_message(self._turn_note)
        comprehension = _comprehension_from_schema(current_schema, request)
        self._file_service.begin_turn(decision.write_allowed)
        self._ui.prepare_new_stream_turn()
        self._log.start_turn(decision.path, alias, request)
        self._log.record_diagnostics(
            f"schema origine={origin} percorso={decision.path}"
        )
        if validation is not None and validation.schema.clarify:
            return self._finalize_turn(
                request,
                validation.schema.clarify,
                write_requested=False,
            ), []
        if decision.path == "direct":
            return self._run_direct_turn(
                request,
                alias,
                thinking,
                config,
                context,
                comprehension,
            ), []
        return self._run_react_turn(
            request,
            alias,
            thinking,
            verify,
            decision,
            config,
            context,
            comprehension,
        )

    def _classify_turn(
        self,
        request: str,
        alias: str,
        context: ThermalTurnContext | None,
    ) -> tuple[str, ValidationOutcome | None, TurnDecision]:
        """
        Greeting and local card without the LLM; only ambiguous text asks the model for the card.

        Serious thermal pressure and an unreadable card open full ReAct.
        """
        if _is_local_chat(request):
            return "schema_skipped_saluto", None, greeting_decision()
        local = local_card(request, self._exists_in_corpus)
        if local is not None:
            validation = self._validate_schema(local, request)
            return "locale", validation, decision_from_schema(validation.schema)
        if context is not None and context.state >= ThermalState.SERIOUS:
            self._log.record_diagnostics("schema_skipped_thermal")
            return "schema_skipped_thermal", None, open_react_decision()
        self._ui.update_progress("Comprendo la richiesta")
        self._check_cancellation()
        try:
            validation, repair, retry = self._schema_from_model(request, alias)
        finally:
            self._ui.close_progress()
        if validation is None:
            self._log.record_diagnostics("schema_fallback")
            return "schema_fallback", None, open_react_decision()
        schema = validation.schema
        self._log.record_diagnostics(
            "schema "
            f"kind={schema.kind} deliver={schema.deliver} "
            f"repair={int(repair)} retry={int(retry)}"
        )
        return "schema", validation, decision_from_schema(schema)

    def _schema_from_model(
        self,
        request: str,
        alias: str,
    ) -> tuple[ValidationOutcome | None, bool, bool]:
        """
        One call and a single retry, only for unreadable JSON.

        An Ollama error does not repeat the call: the turn goes straight to ReAct.
        The KV prefix is SCHEMA_PREFIX, on the same window as the turns.
        """
        context_tokens = self._runtime.session_context(alias)
        self._ensure_schema_prefix(alias, context_tokens)
        llm = self._runtime.get_llm(
            alias,
            context_tokens,
            NUM_PREDICT_SCHEMA,
            reasoning=False,
        )
        if hasattr(llm, "attach_continuation"):
            llm.attach_continuation(self._continuation)
        followup = self._followup_bundle()
        iso = temporal_context().iso
        schema_format = schema_json_ollama()
        error = ""
        for attempt in range(2):
            text = self._complete_llm(
                llm,
                format_schema_prompt(request, iso, followup, error=error),
                format=schema_format,
                num_predict=NUM_PREDICT_SCHEMA,
                reasoning=False,
                temperature=0.0,
            )
            if text is None:
                self._log.record_diagnostics("schema: errore Ollama, niente retry")
                return None, False, attempt > 0
            repair = repair_json_schema(text)
            if repair.schema is not None:
                repaired = repair.repaired or attempt > 0
                return self._validate_schema(repair.schema, request), repaired, attempt > 0
            error = repair.error or "JSON non valido."
        return None, True, True

    def _validate_schema(
        self,
        schema: TurnSchema,
        request: str,
    ) -> ValidationOutcome:
        """Apply deixis and the write gate to the schema (local or from the model)."""
        return validate_turn(
            schema,
            request,
            ValidationContext(
                last=self._last_schema,
                previous_document=self._last_document,
            ),
        )

    def _followup_bundle(self) -> FollowupBundle:
        """Last schema and one outcome line, without the whole history."""
        outcome = ""
        if self._memory.has_turns():
            outcome = _outcome_line(self._memory.turns[-1].assistant)
        last = self._last_schema
        if last is None:
            return FollowupBundle(outcome=outcome)
        return FollowupBundle(
            kind=last.kind,
            subject=last.subject,
            path=last.paths[0] if last.paths else "",
            outcome=outcome,
        )

    def _ensure_schema_prefix(self, alias: str, context_tokens: int) -> None:
        """Prefill only the schema prefix, so the turn does not pay for the examples again."""
        key = (alias, context_tokens, False, SCHEMA_PREFIX)
        if self._continuation.has_prefix(key):
            return
        self._continuation.set_key(key, SCHEMA_PREFIX)
        ids = self._runtime.context_preloaded_prefix(SCHEMA_PREFIX, context_tokens)
        if ids is None:
            ids = self._runtime.prefill_raw_prefix(alias, context_tokens, SCHEMA_PREFIX)
        if ids:
            self._continuation.record_prefix(ids)

    def _complete_llm(self, llm: Any, prompt: str, **kwargs: Any) -> str | None:
        """
        Generation without the answer panel.

        Returns:
            The text (empty if the mock does not return a string), or None
            if Ollama raised an error: in that case retrying is pointless.
        """
        self._check_cancellation()
        try:
            result = llm.invoke(prompt, **kwargs)
        except TurnCancelled:
            raise
        except Exception:
            return None
        if isinstance(result, str):
            return result
        return ""

    def _check_cancellation(self) -> None:
        """Cooperative Ctrl+C across schema, searches and writes."""
        if self._cancel_event.is_set():
            raise TurnCancelled("Turno annullato dall'utente.")

    def _create_turn_callback(
        self,
        config: ConfigRuntime,
        context: ThermalTurnContext | None,
        context_tokens: int,
    ) -> CallbackUI:
        """
        Build the observer shared by a route.

        Args:
            config: Runtime flags, including burn protection.
            context: Optional thermal snapshot produced before the turn.
            context_tokens: Window used for the context percentage.

        Returns:
            Callback with per-turn cancellation and metrics.
        """
        throttle = context.throttle if context else None
        callback = CallbackUI(
            self._ui,
            throttle.pause_between_iterations_sec if throttle else 0.0,
            initial_state=context.state if context else ThermalState.NOMINAL,
            burn_protection=config.burn_protection,
            max_context=context_tokens,
            cancel_event=self._cancel_event,
            log=self._log,
        )
        self._last_callback = callback
        return callback

    def last_metrics(self) -> LLMTurnStats | None:
        """LLM snapshot of the last turn, or None."""
        if self._last_callback is None:
            return None
        return self._last_callback.last_metrics()

    def _pool_memory_bytes(self) -> int:
        """Inference pool from the host snapshot; 0 if the session does not know it."""
        if self._host is None:
            return 0
        return self._host.inference_memory_bytes or self._host.memory_bytes

    def _profile_with_history(
        self,
        alias: str,
        route: InferenceRoute,
        output_tokens: int | None,
        request: str,
        scratchpad_tokens: int,
    ) -> tuple[Any, str]:
        """
        Profile on the session window, with history within budget.

        Every route uses `session_context`: changing `num_ctx` forces Ollama to
        reload the weights. It goes up one step only if the prompt fills the
        window.
        """
        profile = profile_for_route(
            alias,
            route,
            output_tokens=output_tokens,
            host=self._host,
        )
        session = self._runtime.session_context(alias)
        if session != profile.context_tokens:
            profile = replace(profile, context_tokens=session)
        history = self._history_for_profile(
            profile.context_tokens,
            profile.output_tokens,
            scratchpad_tokens=scratchpad_tokens,
            route=route,
        )
        estimate = (
            estimate_tokens(request)
            + estimate_tokens(self._project_memory)
            + estimate_tokens(history)
        )
        chosen = expanded_context(
            profile.context_tokens,
            alias=alias,
            kv_cache_type=self._runtime.kv_cache_type,
            memory_bytes=self._pool_memory_bytes(),
            weight_gb=profile.variant.size_gb,
            can_expand=self._host is not None and can_expand_context(self._host),
            tight=False,
            prompt_tokens=estimate,
            model_ceiling=self._runtime.model_context_cap(alias),
        )
        if chosen == profile.context_tokens:
            return profile, history
        profile = replace(profile, context_tokens=chosen)
        return profile, self._history_for_profile(
            profile.context_tokens,
            profile.output_tokens,
            scratchpad_tokens=scratchpad_tokens,
            route=route,
        )

    def _direct_prefix(self) -> str:
        """Stable head of the direct prompt, identical to the one the pipeline builds."""
        memory = self._project_memory.strip()
        if memory:
            return f"{DIRECT_SYSTEM_PREFIX}\n\n{memory}\n\n"
        return f"{DIRECT_SYSTEM_PREFIX}\n\n"

    def _ensure_direct_prefix(
        self,
        alias: str,
        context_tokens: int,
        reasoning: bool,
    ) -> None:
        """Use the preload ids, or a short prefill if the prefix differs."""
        prefix = self._direct_prefix()
        key = (alias, context_tokens, reasoning, prefix)
        if self._continuation.has_prefix(key):
            return
        self._continuation.set_key(key, prefix)
        ids = self._runtime.context_preloaded_prefix(prefix, context_tokens)
        if ids is None:
            ids = self._runtime.prefill_raw_prefix(alias, context_tokens, prefix)
        if ids:
            self._continuation.record_prefix(ids)

    def _history_for_profile(
        self,
        context_tokens: int,
        output_tokens: int,
        *,
        scratchpad_tokens: int,
        route: InferenceRoute = "direct",
    ) -> str:
        """
        Select the recent turns within a share of the input budget.

        Args:
            context_tokens: Total window of the route.
            output_tokens: Generation reserve.
            scratchpad_tokens: Reserve for tools or prompt structure.
            route: Route used to size the history.

        Returns:
            Formatted history, conservatively limited.
        """
        budget = PromptBudget(
            context_tokens=context_tokens,
            output_tokens=output_tokens,
            scratchpad_tokens=scratchpad_tokens,
        )
        if route == "direct":
            history_tokens = min(
                max(_HISTORY_MIN_TOKENS, budget.input_tokens // _LIGHT_HISTORY_QUOTA),
                _LIGHT_HISTORY_MAX_TOKENS,
            )
        else:
            history_tokens = max(
                _HISTORY_MIN_TOKENS,
                budget.input_tokens // _HEAVY_HISTORY_QUOTA,
            )
        return self._memory.format(max_tokens=history_tokens)

    def _finalize_turn(
        self,
        request: str,
        answer: str,
        *,
        write_requested: bool,
        written_paths: list[str] | None = None,
        max_iterations_hit: bool = False,
    ) -> str:
        """
        Apply evaluate_turn and record the history in a single place.

        ``write_requested`` reflects the user's intent, not the tool permission:
        the ReAct route opens writing even for a plain read, and in that case
        missing files are not a problem to report.
        """
        evaluated = evaluate_turn(
            answer,
            write_requested=write_requested,
            written_paths=written_paths or [],
            max_iterations_hit=max_iterations_hit,
        )
        metrics = self.last_metrics()
        self._log.end_turn(
            classify_turn_outcome(
                incomplete=evaluated.incomplete,
                answer=answer,
            ),
            note=evaluated.note,
            metrics=metrics.as_dict() if metrics else None,
        )
        self._memory.record(request, evaluated.answer)
        return evaluated.answer

    def _record_throttle_if_cutting(self, context: ThermalTurnContext) -> None:
        """Record the thermal cuts that explain truncated turns without being a crash."""
        throttle = context.throttle
        cut = (
            throttle.num_predict < MAX_GENERATED_TOKENS
            or throttle.max_iterations < MAX_AGENT_ITERATIONS
            or throttle.disable_thinking
        )
        if not cut:
            return
        self._log.record_thermal(
            throttle.state.name,
            num_predict=throttle.num_predict,
            max_iterations=throttle.max_iterations,
            alias=throttle.model_alias,
        )

    def _run_direct_turn(
        self,
        request: str,
        alias: str,
        thinking: bool,
        config: ConfigRuntime,
        context: ThermalTurnContext | None,
        comprehension: TurnComprehension,
    ) -> str:
        """
        Run chat/follow-up with a single generation.

        Returns:
            Complete answer, already recorded in the session memory.
        """
        throttle = context.throttle if context else None
        profile, history = self._profile_with_history(
            alias,
            "direct",
            throttle.num_predict if throttle else None,
            request,
            128,
        )
        callback = self._create_turn_callback(
            config,
            context,
            profile.context_tokens,
        )
        self._ensure_direct_prefix(alias, profile.context_tokens, thinking)
        # Direct: KV prefix without INTENT CONTEXT, at most the COMPITO line.
        result = self._direct_pipeline.answer(
            request,
            alias,
            history,
            profile.context_tokens,
            profile.output_tokens,
            callback,
            intent_context=format_comprehension_block(comprehension),
            project_memory=self._project_memory,
            reasoning=thinking,
        )
        return self._finalize_turn(
            request,
            result,
            write_requested=False,
        )

    def _run_react_turn(
        self,
        request: str,
        alias: str,
        thinking: bool,
        verify: bool,
        decision: TurnDecision,
        config: ConfigRuntime,
        context: ThermalTurnContext | None,
        comprehension: TurnComprehension,
    ) -> tuple[str, list[str]]:
        """
        Run the workspace path with minimal tools and limited budgets.

        Returns:
            Final answer and the paths possibly materialized by the fallback.
        """
        write_allowed = decision.write_allowed
        throttle = context.throttle if context else None
        max_iterations = OPEN_REACT_ITERATIONS
        if throttle is not None:
            if write_allowed:
                max_iterations = max(max_iterations, throttle.max_iterations)
            else:
                max_iterations = min(max_iterations, throttle.max_iterations)
        profile, history = self._profile_with_history(
            alias,
            "react",
            throttle.num_predict if throttle else None,
            request,
            768,
        )
        executor = self._get_executor(
            alias,
            max_iterations,
            profile.output_tokens,
            profile.context_tokens,
            write_allowed=write_allowed,
            reasoning=thinking,
        )
        callback = self._create_turn_callback(
            config,
            context,
            profile.context_tokens,
        )
        self._active_callback = callback
        try:
            executor_outcome = executor.invoke(
                    {
                        "input": _build_task_description(
                            request,
                            verify,
                            decision,
                            history,
                            project_memory=self._project_memory,
                            comprehension=comprehension,
                            note=self._turn_note,
                        ),
                    },
                    RunnableConfig(callbacks=[callback]),
                )
        finally:
            self._active_callback = None
        result = str(executor_outcome["output"])
        hit_limit = any(marker in result.casefold() for marker in _LIMIT_MARKERS)
        if hit_limit:
            fallback = _answer_from_steps(executor_outcome.get("intermediate_steps", []))
            if fallback is not None:
                self._log.record_diagnostics("limite iterazioni: ripiego sull'ultimo passo")
                result = f"{fallback}\n\n{LIMIT_FALLBACK_NOTE}"
                hit_limit = False
        written_paths = self._file_service.written_paths
        if write_allowed:
            # Files only: a created folder is not a «written file» to list.
            result = _enrich_answer(
                result,
                self._file_service.written_files,
            )
            self._turn_fallback_path = apply_fallback(
                result,
                written_paths,
                service=self._file_service,
            )
            written_paths = self._file_service.written_paths
        final = self._finalize_turn(
            request,
            result,
            write_requested=decision.analysis.write,
            written_paths=written_paths,
            max_iterations_hit=hit_limit,
        )
        return final, written_paths

    def _get_executor(
        self,
        alias: str,
        max_iter: int,
        num_predict: int,
        context_tokens: int,
        *,
        write_allowed: bool,
        reasoning: bool,
    ) -> AgentExecutor:
        """Return an executor cached on the whole operating profile."""
        key = ExecutorKey(
            alias,
            max_iter,
            num_predict,
            context_tokens,
            capabilities_react(),
            write_allowed,
            reasoning,
        )
        cached = self._executor_per_key.get(key)
        if cached is not None:
            self._executor_per_key.move_to_end(key)
            return cached
        created = self._create_executor(
            alias,
            max_iter,
            num_predict,
            context_tokens,
            key.capabilities,
            write_allowed=write_allowed,
            reasoning=reasoning,
        )
        self._executor_per_key[key] = created
        while len(self._executor_per_key) > EXECUTOR_CACHE_LIMIT:
            self._executor_per_key.popitem(last=False)
        return created

    def _create_executor(
        self,
        alias: str,
        max_iter: int,
        num_predict: int,
        context_tokens: int,
        capabilities: CapabilitySet,
        *,
        write_allowed: bool,
        reasoning: bool,
    ) -> AgentExecutor:
        tools = self._create_tools_for_capability(
            alias,
            capabilities,
            write_allowed=write_allowed,
            reasoning=reasoning,
        )
        prompt = PromptTemplate.from_template(_TEMPLATE_REACT).partial(
            identity=_AGENT_IDENTITY
        )
        # Custom parser: avoids loops on "Action: Final Answer" (phantom tool).
        agent = create_react_agent(
            llm=self._runtime.get_llm(
                alias,
                context_tokens,
                num_predict,
                reasoning=reasoning,
            ),
            tools=tools,
            prompt=prompt,
            output_parser=AssistantReActParser(),
        )
        budget = PromptBudget(
            context_tokens=context_tokens,
            output_tokens=num_predict,
        )
        return AgentExecutor(
            agent=agent,
            tools=tools,
            max_iterations=max_iter,
            return_intermediate_steps=True,
            handle_parsing_errors=_parsing_error_observation,
            trim_intermediate_steps=lambda steps: _limit_react_steps(
                steps,
                budget.scratchpad_tokens,
            ),
            verbose=False,
        )

    def _create_tools_for_capability(
        self,
        alias: str,
        capabilities: CapabilitySet,
        *,
        write_allowed: bool = False,
        reasoning: bool = True,
    ) -> list[Any]:
        """
        Build only the tools relevant to the request.

        Args:
            alias: Active model shared with the summary.
            capabilities: Ordered families produced by the policy.
            write_allowed: Gate used in the dynamic descriptions.

        Returns:
            Deterministic LangChain list used by prompt and executor.
        """
        tools: list[Any] = []
        # The file tools are also needed by the summary to read and save.
        if "file" in capabilities:
            tools.extend(
                create_file_tools(
                    self._file_service,
                    write_allowed=write_allowed,
                    workspace_label=self._file_service.workspace.name,
                )
            )
        # The provider bound to the alias prevents implicitly loading the 12B.
        if "summary" in capabilities:
            tools.extend(
                create_summary_tool(
                    self._file_service,
                    alias=alias,
                    # The summary profile's window is only a budget: num_ctx stays
                    # the session one, so the summary does not reload the weights.
                    llm_provider=lambda _context, output: self._runtime.get_llm(
                        alias,
                        self._runtime.session_context(alias),
                        output,
                        reasoning=reasoning,
                    ),
                    callback_provider=lambda: self._active_callback,
                )
            )
        # The date is needed for calculations on dates cited in the documents.
        if "date" in capabilities:
            tools.extend(create_date_tool())
        return tools
