"""Pipeline a singola generazione per la sola chat, senza documenti né tool."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Protocol

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig

from assistente.console import ConsoleUI
from assistente.memoria import blocco_storico_prompt
from assistente.politica_turno import ANTI_GOLD_PLATING
from assistente.registro_processo import notifica_diagnostica

LLMProvider = Callable[[str, int, int], "StreamingLLM"]

# Prefisso stabile per KV cache: identità e vincoli, senza parti di turno.
PREFISSO_SISTEMA_DIRECT = (
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
    f"{PREFISSO_SISTEMA_DIRECT}\n\n"
    "{project_memory}"
    "{history}"
    "{intent_context}"
    "RICHIESTA:\n{request}\n\n"
    "RISPOSTA:\n"
)
# CoT nativo consuma num_predict: una sola chiamata con budget maggiorato.
THINKING_OUTPUT_FACTOR = 2


class StreamingLLM(Protocol):
    """Contratto minimo del client usato dalla pipeline diretta."""

    def stream(
        self,
        input: str,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        Genera chunk testuali.

        Returns:
            Iteratore sincrono dei frammenti prodotti da Ollama.
        """


class DirectAnswerPipeline:
    """Risponde con un solo prefill/decode e senza protocollo ReAct."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        ui: ConsoleUI,
    ) -> None:
        """
        Collega provider centralizzato e renderer.

        Args:
            llm_provider: Factory/cache per tag e budget Ollama.
            ui: Destinazione dello stream finale.
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
        Esegue una risposta diretta in streaming.

        Args:
            request: Messaggio originale del turno.
            model_alias: Alias modello selezionato (quant fissa).
            history: Turni recenti già ridotti al budget.
            context_tokens: Finestra assegnata alla route.
            output_tokens: Tetto della risposta.
            callback: Observer per token e metriche.
            intent_context: Blocco variabile del turno (data, note); vuoto in chat.
            project_memory: Memoria progetto stabile (AGENTS.md).
            reasoning: CoT nativo; il budget output viene maggiorato in-place.

        Returns:
            Risposta completa oppure diagnostica del provider.
        """
        prefix_memory = (
            f"{project_memory.strip()}\n\n" if project_memory.strip() else ""
        )
        prefix_intent = (
            f"{intent_context.strip()}\n\n" if intent_context.strip() else ""
        )
        prompt = _DIRECT_PROMPT.format(
            project_memory=prefix_memory,
            history=blocco_storico_prompt(history),
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
        self._ui.inizia_risposta_diretta()
        config = RunnableConfig(callbacks=[callback])
        extra: dict[str, Any] = {"num_predict": effective_output}
        if not reasoning:
            extra["reasoning"] = False
        answer = self._stream_text(llm, prompt, config, callback, **extra)
        if answer.startswith("Errore inferenza locale:"):
            return answer
        # Una risposta vuota è una diagnostica esplicita, non un successo silenzioso.
        if not answer:
            vuota = (
                "Errore inferenza locale: Ollama ha restituito una risposta vuota."
            )
            notifica_diagnostica(callback, vuota)
            return vuota
        return answer

    def _stream_text(
        self,
        llm: StreamingLLM,
        prompt: str,
        config: RunnableConfig,
        callback: BaseCallbackHandler,
        **stream_kwargs: Any,
    ) -> str:
        """Concatena i chunk; converte eccezioni provider in diagnostica."""
        chunks: list[str] = []
        try:
            for chunk in llm.stream(prompt, config=config, **stream_kwargs):
                chunks.append(str(chunk))
        except Exception as error:
            messaggio = (
                f"Errore inferenza locale: {type(error).__name__}: {error}"
            )
            notifica_diagnostica(callback, messaggio, type(error).__name__)
            return messaggio
        return "".join(chunks).strip()
