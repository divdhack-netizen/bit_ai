"""Single-generation pipeline for chat only, with no documents or tools."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Protocol

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig

from assistant.console import ConsoleUI
from assistant.memory import history_prompt_block
from assistant.turn_policy import ANTI_GOLD_PLATING
from assistant.process_log import notify_diagnostics

LLMProvider = Callable[[str, int, int], "StreamingLLM"]

# Stable prefix for the KV cache: identity and constraints, no per-turn parts.
DIRECT_SYSTEM_PREFIX = (
    "Parli in italiano informale, dando del tu, come in una chat. "
    "Lavori solo sui documenti della cartella corrente, senza Internet. "
    "Qui rispondi solo a saluti e commenti sulla conversazione: non dare fatti "
    "presi dalle tue conoscenze. Per un contenuto invita a fare la domanda, "
    "così la cerchi nei documenti. "
    "Non inventare operazioni, file, fonti o risultati. "
    f"{ANTI_GOLD_PLATING} "
    "Apri con la risposta e tieni i dettagli proporzionati a quanto è complicata la cosa. "
    "Usa lo storico solo se serve al seguito e, se c'è, non contraddirlo. "
    "Nei riassunti o confronti sullo storico, copri tutti i temi dei turni "
    "rilevanti, non solo l'ultimo."
)

_DIRECT_PROMPT = (
    f"{DIRECT_SYSTEM_PREFIX}\n\n"
    "{project_memory}"
    "{history}"
    "{intent_context}"
    "RICHIESTA:\n{request}\n\n"
    "RISPOSTA:\n"
)
# Native CoT consumes num_predict: a single call with a larger budget.
THINKING_OUTPUT_FACTOR = 2


class StreamingLLM(Protocol):
    """Minimal client contract used by the direct pipeline."""

    def stream(
        self,
        input: str,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        Generate text chunks.

        Returns:
            Synchronous iterator over the fragments produced by Ollama.
        """


class DirectAnswerPipeline:
    """Answer with a single prefill/decode and no ReAct protocol."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        ui: ConsoleUI,
    ) -> None:
        """
        Wire up the central provider and the renderer.

        Args:
            llm_provider: Factory/cache for Ollama tags and budgets.
            ui: Destination of the final stream.
        """
        self._llm_provider = llm_provider
        self._ui = ui

    def answer(
        self,
        request: str,
        model_alias: str,
        history: str,
        context_tokens: int,
        output_tokens: int,
        callback: BaseCallbackHandler,
        *,
        intent_context: str = "",
        project_memory: str = "",
        reasoning: bool = False,
    ) -> str:
        """
        Run a direct streaming answer.

        Args:
            request: Original message of the turn.
            model_alias: Selected model alias (fixed quantization).
            history: Recent turns already reduced to the budget.
            context_tokens: Window assigned to the route.
            output_tokens: Answer cap.
            callback: Observer for tokens and metrics.
            intent_context: Variable per-turn block (date, notes); empty in chat.
            project_memory: Stable project memory (AGENTS.md).
            reasoning: Native CoT; the output budget is raised in place.

        Returns:
            The complete answer, or the provider diagnostics.
        """
        prefix_memory = (
            f"{project_memory.strip()}\n\n" if project_memory.strip() else ""
        )
        prefix_intent = (
            f"{intent_context.strip()}\n\n" if intent_context.strip() else ""
        )
        prompt = _DIRECT_PROMPT.format(
            project_memory=prefix_memory,
            history=history_prompt_block(history),
            intent_context=prefix_intent,
            request=request.strip(),
        )
        effective_output = output_tokens
        if reasoning:
            effective_output = max(output_tokens, output_tokens * THINKING_OUTPUT_FACTOR)
        llm = self._llm_provider(
            model_alias,
            context_tokens,
            effective_output,
        )
        self._ui.start_direct_answer()
        config = RunnableConfig(callbacks=[callback])
        extra: dict[str, Any] = {"num_predict": effective_output}
        if not reasoning:
            extra["reasoning"] = False
        answer = self._stream_text(llm, prompt, config, callback, **extra)
        if answer.startswith("Errore inferenza locale:"):
            return answer
        # An empty answer is an explicit diagnostic, not a silent success.
        if not answer:
            empty = (
                "Errore inferenza locale: Ollama ha restituito una risposta vuota."
            )
            notify_diagnostics(callback, empty)
            return empty
        return answer

    def _stream_text(
        self,
        llm: StreamingLLM,
        prompt: str,
        config: RunnableConfig,
        callback: BaseCallbackHandler,
        **stream_kwargs: Any,
    ) -> str:
        """Join the chunks; turn provider exceptions into diagnostics."""
        chunks: list[str] = []
        try:
            for chunk in llm.stream(prompt, config=config, **stream_kwargs):
                chunks.append(str(chunk))
        except Exception as error:
            message = (
                f"Errore inferenza locale: {type(error).__name__}: {error}"
            )
            notify_diagnostics(callback, message, type(error).__name__)
            return message
        return "".join(chunks).strip()
