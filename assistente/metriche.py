"""Metriche aggregate per turno: token, TTFT e durata di LLM e tool."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rich.cells import cell_len

from assistente.prompt_budget import calibrate_token_estimate

_NANOSECONDS_PER_SECOND = 1_000_000_000
_MILLISECONDS_PER_SECOND = 1_000
# Ollama espone eval_count solo sul chunk `done`; in streaming si stima dai caratteri.
_CARATTERI_PER_TOKEN = 4


def stima_token_da_testo(testo: str) -> int:
    """
    Stima i token di un frammento quando il provider non ha ancora chiuso lo stream.

    Returns:
        Intero ≥ 1 per testo non vuoto; 0 se il frammento è vuoto.
    """
    if not testo:
        return 0
    return max(1, (len(testo) + _CARATTERI_PER_TOKEN - 1) // _CARATTERI_PER_TOKEN)


@dataclass(frozen=True, slots=True)
class StatisticheTurnoLLM:
    """Snapshot immutabile visualizzato nella barra di stato della CLI."""

    token_input: int
    token_output: int
    token_al_secondo: float
    contesto_usato: int
    contesto_massimo: int
    chiamate_llm: int
    ttft_ms: float
    durata_llm_ms: float
    durata_tool_ms: float
    durata_load_ms: float = 0.0
    durata_prefill_ms: float = 0.0
    durata_provider_ms: float = 0.0
    durata_termica_ms: float = 0.0
    durata_rendering_ms: float = 0.0
    durata_end_to_end_ms: float = 0.0
    ttft_ultima_ms: float = 0.0
    ttft_massima_ms: float = 0.0
    contesto_picco: int = 0

    def as_dict(self) -> dict[str, float | int]:
        """
        Espone lo snapshot in forma JSON-serializzabile per benchmark e log.

        Returns:
            Mapping piatto di tutte le metriche numeriche del turno.
        """
        return {
            "token_input": self.token_input,
            "token_output": self.token_output,
            "token_al_secondo": self.token_al_secondo,
            "contesto_usato": self.contesto_usato,
            "contesto_massimo": self.contesto_massimo,
            "chiamate_llm": self.chiamate_llm,
            "ttft_ms": self.ttft_ms,
            "durata_llm_ms": self.durata_llm_ms,
            "durata_tool_ms": self.durata_tool_ms,
            "durata_load_ms": self.durata_load_ms,
            "durata_prefill_ms": self.durata_prefill_ms,
            "durata_provider_ms": self.durata_provider_ms,
            "durata_termica_ms": self.durata_termica_ms,
            "durata_rendering_ms": self.durata_rendering_ms,
            "durata_end_to_end_ms": self.durata_end_to_end_ms,
            "ttft_ultima_ms": self.ttft_ultima_ms,
            "ttft_massima_ms": self.ttft_massima_ms,
            "contesto_picco": self.contesto_picco,
        }


def _formatta_numero(valore: int) -> str:
    """
    Formatta interi con spazio come separatore delle migliaia.

    Returns:
        Numero leggibile e indipendente dalla locale del processo.
    """
    return f"{valore:,}".replace(",", " ")


def _percentuale_contesto(statistiche: StatisticheTurnoLLM) -> int:
    """
    Calcola la percentuale di contesto usata sul massimo disponibile.

    Args:
        statistiche: Snapshot del turno con contesto_usato e contesto_massimo.

    Returns:
        Intero 0–100; 0 se il massimo non è positivo (evita divisione per zero).
    """
    # Senza tetto noto la percentuale non è definibile: mostra 0
    if statistiche.contesto_massimo <= 0:
        return 0
    return round(100 * statistiche.contesto_usato / statistiche.contesto_massimo)


def segmenti_statistiche_prioritari(
    statistiche: StatisticheTurnoLLM,
    *,
    verbose: bool = False,
) -> tuple[str, ...]:
    """
    Espone le metriche in ordine di priorità per terminali stretti.

    Default: tok/s, token totali (in+out), percentuale di contesto.
    Con verbose: anche LLM/TTFT/tool/prefill/load.

    Args:
        statistiche: Snapshot immutabile del turno corrente.
        verbose: Se True, accoda i segmenti di latenza.

    Returns:
        Tupla di frammenti indipendenti, uniti con doppio spazio nella barra.
    """
    percentuale = _percentuale_contesto(statistiche)
    totali = statistiche.token_input + statistiche.token_output
    base = (
        f"{statistiche.token_al_secondo:.1f} tok/s",
        f"token: {_formatta_numero(totali)}",
        f"ctx: {percentuale}%",
    )
    if not verbose:
        return base
    return base + (
        f"LLM: {statistiche.chiamate_llm}/{statistiche.durata_llm_ms:.0f} ms",
        f"TTFT: {statistiche.ttft_ms:.0f} ms",
        f"tool: {statistiche.durata_tool_ms:.0f} ms",
        f"prefill: {statistiche.durata_prefill_ms:.0f} ms",
        f"load: {statistiche.durata_load_ms:.0f} ms",
    )


def formatta_segmento_statistiche(
    statistiche: StatisticheTurnoLLM,
    max_larghezza: int | None = None,
    *,
    verbose: bool = False,
) -> str:
    """
    Converte le metriche del turno in una riga compatta per il terminale.

    Se max_larghezza è impostato, include solo i segmenti prioritari che
    entrano nella larghezza (celle visibili, senza ANSI).

    Args:
        statistiche: Snapshot del turno.
        max_larghezza: Limite opzionale in colonne visibili; None = tutto.
        verbose: Se True, include anche le latenze LLM/tool.

    Returns:
        Segmento tok/s, token totali e ctx % (più latenze se verbose).
    """
    segmenti = segmenti_statistiche_prioritari(statistiche, verbose=verbose)
    # Nessun tetto: comportamento storico per test e log non-TTY
    if max_larghezza is None:
        return "  ".join(segmenti)

    # Larghezza non positiva: niente metriche (resta solo il prefisso termico)
    if max_larghezza <= 0:
        return ""

    pezzi: list[str] = []
    usato = 0
    # Aggiunge segmenti finché doppio spazio + pezzo entrano nel budget
    for segmento in segmenti:
        separatore = 2 if pezzi else 0
        costo = separatore + cell_len(segmento)
        # Salta questo e i successivi: priorità già rispettata dall'ordine
        if usato + costo > max_larghezza:
            break
        pezzi.append(segmento)
        usato += costo
    return "  ".join(pezzi)


def _info_ollama(response: Any) -> dict[str, Any] | None:
    """
    Estrae generation_info da una risposta LangChain senza dipendere dal suo tipo concreto.

    Returns:
        Mapping Ollama oppure None per risposte mockate o provider differenti.
    """
    try:
        info = response.generations[0][0].generation_info
    except (AttributeError, IndexError, TypeError):
        return None
    return info if isinstance(info, dict) else None


class AccumulatoreMetricheLLM:
    """Accumulatore per-turno alimentato dal callback LangChain e dai tool diretti."""

    def __init__(
        self,
        contesto_massimo: int,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        """
        Inizializza contatori vuoti con un clock monotono iniettabile.

        Args:
            contesto_massimo: Finestra token del profilo LLM corrente.
            clock: Orologio usato per durate wall-clock e test deterministici.
        """
        self._contesto_massimo = contesto_massimo
        self._clock = clock
        self._token_input = 0
        self._token_output = 0
        self._token_input_live = 0
        self._token_output_live = 0
        self._eval_duration_ns = 0
        self._load_duration_ns = 0
        self._prompt_eval_duration_ns = 0
        self._provider_total_duration_ns = 0
        self._contesto_usato = 0
        self._contesto_picco = 0
        self._chiamate = 0
        self._ttft_total_seconds = 0.0
        self._ttft_samples = 0
        self._ttft_last_seconds = 0.0
        self._ttft_max_seconds = 0.0
        self._llm_total_seconds = 0.0
        self._tool_total_seconds = 0.0
        self._thermal_wait_seconds = 0.0
        self._render_total_seconds = 0.0
        self._turn_started_at = self._clock()
        self._llm_started_at: float | None = None
        self._tool_started_at: float | None = None
        self._first_token_at: float | None = None
        self._first_token_recorded = False

    def inizia_chiamata(self) -> None:
        """Apre una misurazione LLM e incrementa il numero di chiamate del turno."""
        self._llm_started_at = self._clock()
        self._first_token_recorded = False
        self._first_token_at = None
        self._token_input_live = 0
        self._token_output_live = 0
        self._chiamate += 1

    def stima_prompt(self, testo: str) -> None:
        """
        Stima token di input e contesto finché Ollama non pubblica prompt_eval_count.

        Args:
            testo: Prompt completo passato a on_llm_start.
        """
        stimati = stima_token_da_testo(testo)
        self._token_input_live = stimati
        self._contesto_usato = stimati
        self._contesto_picco = max(self._contesto_picco, stimati)

    def registra_token_stream(self, token: str) -> None:
        """
        Accumula una stima dei token di output per aggiornare la barra durante lo stream.

        Args:
            token: Frammento emesso da on_llm_new_token; i vuoti non contano.
        """
        self._token_output_live += stima_token_da_testo(token)

    def registra_primo_token(self) -> None:
        """Registra il TTFT una sola volta per la chiamata LLM corrente."""
        # Callback duplicati o token fuori da una chiamata non devono alterare la media.
        if self._llm_started_at is None or self._first_token_recorded:
            return
        elapsed_seconds = self._clock() - self._llm_started_at
        self._ttft_total_seconds += elapsed_seconds
        self._ttft_samples += 1
        self._ttft_last_seconds = elapsed_seconds
        self._ttft_max_seconds = max(self._ttft_max_seconds, elapsed_seconds)
        self._first_token_recorded = True
        self._first_token_at = self._clock()

    def registra(self, response: Any) -> StatisticheTurnoLLM | None:
        """
        Accumula metadata Ollama e chiude la misurazione wall-clock corrente.

        Returns:
            Snapshot aggiornato, disponibile anche se il provider non espone token.
        """
        info = _info_ollama(response)
        # I provider alternativi possono non fornire metadata ma hanno comunque durata/chiamata.
        if info is not None:
            prompt_eval = info.get("prompt_eval_count")
            eval_count = info.get("eval_count")
            # Ogni contatore viene aggiunto solo quando Ollama lo dichiara.
            if prompt_eval is not None:
                evaluate_count = int(prompt_eval)
                calibrate_token_estimate(evaluate_count)
                self._token_input += evaluate_count
                self._token_input_live = 0
                self._contesto_usato = int(prompt_eval)
                self._contesto_picco = max(
                    self._contesto_picco,
                    self._contesto_usato,
                )
            else:
                # Senza metadata si conferma la stima da caratteri già mostrata in live.
                self._token_input += self._token_input_live
                self._token_input_live = 0
            # I token prodotti alimentano sia il totale sia la velocità aggregata.
            if eval_count is not None:
                self._token_output += int(eval_count)
                self._token_output_live = 0
            else:
                self._token_output += self._token_output_live
                self._token_output_live = 0
            eval_duration = info.get("eval_duration")
            # La durata nativa Ollama esclude trasporto e alimenta tok/s in modo preciso.
            if eval_duration is not None:
                self._eval_duration_ns += int(eval_duration)
            load_duration = info.get("load_duration")
            # Il tempo di load distingue cold start da inferenza calda.
            if load_duration is not None:
                self._load_duration_ns += int(load_duration)
            prompt_eval_duration = info.get("prompt_eval_duration")
            # Il prefill misura direttamente il costo dei dati di input.
            if prompt_eval_duration is not None:
                self._prompt_eval_duration_ns += int(prompt_eval_duration)
            total_duration = info.get("total_duration")
            # La durata provider consente il confronto con il wall-clock applicativo.
            if total_duration is not None:
                self._provider_total_duration_ns += int(total_duration)
        else:
            self._commit_stime_live()
        self._finish_llm_timing()
        return self.snapshot()

    def registra_errore_llm(self) -> StatisticheTurnoLLM | None:
        """
        Chiude la misurazione dopo un errore conservando le stime già osservate.

        Returns:
            Snapshot che conserva chiamata, durata e token stream fino al fallimento.
        """
        self._commit_stime_live()
        self._finish_llm_timing()
        return self.snapshot()

    def inizia_tool(self) -> None:
        """Apre il timer del tool corrente; i tool dell'agente sono sequenziali."""
        self._tool_started_at = self._clock()

    def registra_attesa_termica(self, seconds: float) -> None:
        """
        Accumula pause deliberate escluse dal TTFT provider.

        Args:
            seconds: Durata monotona non negativa della pausa.
        """
        self._thermal_wait_seconds += max(0.0, seconds)

    def registra_rendering(self, seconds: float) -> None:
        """
        Accumula il costo sincrono dei callback UI.

        Args:
            seconds: Durata monotona del rendering di un chunk.
        """
        self._render_total_seconds += max(0.0, seconds)

    def termina_tool(self) -> StatisticheTurnoLLM | None:
        """
        Aggiunge la durata del tool corrente e azzera il timer.

        Returns:
            Snapshot corrente per aggiornare immediatamente la barra di stato.
        """
        started_at = self._tool_started_at
        # Un evento end senza start può arrivare da callback parziali e viene ignorato.
        if started_at is not None:
            self._tool_total_seconds += self._clock() - started_at
            self._tool_started_at = None
        return self.snapshot()

    def snapshot(self) -> StatisticheTurnoLLM | None:
        """
        Produce uno snapshot se almeno un LLM o un tool è stato osservato.

        Returns:
            Statistiche aggregate oppure None prima di qualsiasi attività misurabile.
        """
        # Prima del primo evento la UI deve rimanere priva di numeri fittizi.
        if not self._chiamate and self._tool_total_seconds <= 0:
            return None
        token_input = self._token_input + self._token_input_live
        token_output = self._token_output + self._token_output_live
        evaluation_seconds = self._eval_duration_ns / _NANOSECONDS_PER_SECOND
        # Durante il decode il wall-clock dal primo token anticipa eval_duration Ollama.
        if self._first_token_at is not None and self._llm_started_at is not None:
            evaluation_seconds += self._clock() - self._first_token_at
        elif evaluation_seconds <= 0:
            decode_seconds = self._durata_llm_aperta() - self._ttft_last_seconds
            evaluation_seconds = max(0.0, decode_seconds)
        token_rate = (
            token_output / evaluation_seconds
            if evaluation_seconds > 0
            else 0.0
        )
        average_ttft_seconds = (
            self._ttft_total_seconds / self._ttft_samples
            if self._ttft_samples > 0
            else 0.0
        )
        return StatisticheTurnoLLM(
            token_input=token_input,
            token_output=token_output,
            token_al_secondo=token_rate,
            contesto_usato=self._contesto_usato,
            contesto_massimo=self._contesto_massimo,
            chiamate_llm=self._chiamate,
            ttft_ms=average_ttft_seconds * _MILLISECONDS_PER_SECOND,
            durata_llm_ms=self._durata_llm_aperta() * _MILLISECONDS_PER_SECOND,
            durata_tool_ms=self._tool_total_seconds * _MILLISECONDS_PER_SECOND,
            durata_load_ms=(
                self._load_duration_ns
                / _NANOSECONDS_PER_SECOND
                * _MILLISECONDS_PER_SECOND
            ),
            durata_prefill_ms=(
                self._prompt_eval_duration_ns
                / _NANOSECONDS_PER_SECOND
                * _MILLISECONDS_PER_SECOND
            ),
            durata_provider_ms=(
                self._provider_total_duration_ns
                / _NANOSECONDS_PER_SECOND
                * _MILLISECONDS_PER_SECOND
            ),
            durata_termica_ms=(
                self._thermal_wait_seconds * _MILLISECONDS_PER_SECOND
            ),
            durata_rendering_ms=(
                self._render_total_seconds * _MILLISECONDS_PER_SECOND
            ),
            durata_end_to_end_ms=(
                (self._clock() - self._turn_started_at)
                * _MILLISECONDS_PER_SECOND
            ),
            ttft_ultima_ms=(
                self._ttft_last_seconds * _MILLISECONDS_PER_SECOND
            ),
            ttft_massima_ms=(
                self._ttft_max_seconds * _MILLISECONDS_PER_SECOND
            ),
            contesto_picco=self._contesto_picco,
        )

    def _commit_stime_live(self) -> None:
        """Trasferisce le stime di streaming nei totali confermati del turno."""
        self._token_input += self._token_input_live
        self._token_input_live = 0
        self._token_output += self._token_output_live
        self._token_output_live = 0

    def _durata_llm_aperta(self) -> float:
        """Somma le chiamate chiuse e l'eventuale chiamata LLM ancora in corso."""
        durata = self._llm_total_seconds
        started_at = self._llm_started_at
        if started_at is not None:
            durata += self._clock() - started_at
        return durata

    def _finish_llm_timing(self) -> None:
        """Chiude in modo idempotente il timer LLM corrente."""
        started_at = self._llm_started_at
        # Eventi end/error duplicati non devono sommare due volte la stessa durata.
        if started_at is None:
            return
        self._llm_total_seconds += self._clock() - started_at
        self._llm_started_at = None
        self._first_token_at = None
