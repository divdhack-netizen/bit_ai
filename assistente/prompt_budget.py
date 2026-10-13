"""Budget conservativi per evitare overflow del contesto Ollama."""

from __future__ import annotations

import math
from dataclasses import dataclass

UTF8_BYTES_PER_ESTIMATED_TOKEN = 3.0
MIN_BYTES_PER_ESTIMATED_TOKEN = 2.0
MAX_BYTES_PER_ESTIMATED_TOKEN = 4.5
DEFAULT_SCRATCHPAD_RESERVE_TOKENS = 768
DEFAULT_SYSTEM_RESERVE_TOKENS = 384
TRUNCATION_MARKER = "\n[…contenuto ridotto per rispettare il contesto…]\n"
_BYTES_PER_TOKEN = UTF8_BYTES_PER_ESTIMATED_TOKEN
_PROMPT_BYTES_IN_ATTESA = 0


@dataclass(frozen=True, slots=True)
class PromptBudget:
    """
    Ripartisce una finestra tra input, output e riserve operative.

    Attributes:
        context_tokens: Finestra totale allocata da Ollama.
        output_tokens: Token massimi della risposta.
        scratchpad_tokens: Riserva per Action e Observation ReAct.
        system_tokens: Riserva per template, descrizioni tool e marker.
    """

    context_tokens: int
    output_tokens: int
    scratchpad_tokens: int = DEFAULT_SCRATCHPAD_RESERVE_TOKENS
    system_tokens: int = DEFAULT_SYSTEM_RESERVE_TOKENS

    @property
    def input_tokens(self) -> int:
        """
        Calcola il budget disponibile ai dati variabili del prompt.

        Returns:
            Almeno un token, anche con una configurazione incoerente.
        """
        reserved_tokens = (
            self.output_tokens
            + self.scratchpad_tokens
            + self.system_tokens
        )
        return max(1, self.context_tokens - reserved_tokens)


def note_prompt_bytes(text: str) -> None:
    """Ricorda i byte del prompt appena inviato, in attesa di prompt_eval_count."""
    global _PROMPT_BYTES_IN_ATTESA
    _PROMPT_BYTES_IN_ATTESA = len(text.encode("utf-8")) if text else 0


def calibrate_token_estimate(prompt_eval_count: int) -> None:
    """
    Allinea i byte per token al conteggio reale Ollama, entro un clamp.

    Senza conteggio o senza byte in attesa il fattore resta quello precedente.
    """
    global _BYTES_PER_TOKEN, _PROMPT_BYTES_IN_ATTESA
    if prompt_eval_count <= 0 or _PROMPT_BYTES_IN_ATTESA <= 0:
        return
    ratio = _PROMPT_BYTES_IN_ATTESA / prompt_eval_count
    _BYTES_PER_TOKEN = min(
        MAX_BYTES_PER_ESTIMATED_TOKEN,
        max(MIN_BYTES_PER_ESTIMATED_TOKEN, ratio),
    )
    _PROMPT_BYTES_IN_ATTESA = 0


def reset_token_estimate_for_tests() -> None:
    """Riporta la stima a 3 byte per token. Solo per i test."""
    global _BYTES_PER_TOKEN, _PROMPT_BYTES_IN_ATTESA
    _BYTES_PER_TOKEN = UTF8_BYTES_PER_ESTIMATED_TOKEN
    _PROMPT_BYTES_IN_ATTESA = 0


def estimate_tokens(text: str) -> int:
    """
    Stima i token senza caricare un tokenizer in memoria.

    La stima usa byte UTF-8 divisi per il fattore di sessione (default 3,
    poi calibrato su prompt_eval_count). I conteggi Ollama restano la fonte.

    Args:
        text: Testo da valutare, anche vuoto.

    Returns:
        Token stimati, zero soltanto per input vuoto.
    """
    if not text:
        return 0
    byte_count = len(text.encode("utf-8"))
    return max(1, math.ceil(byte_count / _BYTES_PER_TOKEN))


def _decode_prefix(payload: bytes, byte_limit: int) -> str:
    """
    Decodifica un prefisso UTF-8 senza lasciare un carattere spezzato.

    Args:
        payload: Byte originali.
        byte_limit: Numero massimo di byte da conservare.

    Returns:
        Prefisso Unicode valido.
    """
    return payload[:byte_limit].decode("utf-8", errors="ignore").rstrip()


def _decode_suffix(payload: bytes, byte_limit: int) -> str:
    """
    Decodifica un suffisso UTF-8 senza lasciare un carattere spezzato.

    Args:
        payload: Byte originali.
        byte_limit: Numero massimo di byte da conservare.

    Returns:
        Suffisso Unicode valido.
    """
    return payload[-byte_limit:].decode("utf-8", errors="ignore").lstrip()


def fit_text(
    text: str,
    max_tokens: int,
    *,
    keep_tail: bool = False,
    marker: str = TRUNCATION_MARKER,
) -> str:
    """
    Riduce un testo al budget stimato conservando testa oppure coda.

    Args:
        text: Contenuto originale.
        max_tokens: Budget massimo; valori non positivi producono stringa vuota.
        keep_tail: Se True preserva la parte recente, utile allo storico.
        marker: Indicazione esplicita del taglio.

    Returns:
        Testo originale se entra, altrimenti testo troncato con marker.
    """
    # Un budget nullo non deve lasciare marker privi di contenuto.
    if max_tokens <= 0 or not text:
        return ""
    # Il percorso comune evita copie quando il contenuto è già entro il limite.
    if estimate_tokens(text) <= max_tokens:
        return text
    payload = text.encode("utf-8")
    marker_bytes = marker.encode("utf-8")
    maximum_bytes = math.floor(max_tokens * UTF8_BYTES_PER_ESTIMATED_TOKEN)
    content_bytes = max(0, maximum_bytes - len(marker_bytes))
    # Un budget più corto del marker conserva comunque un frammento utile.
    if content_bytes <= 0:
        return _decode_suffix(payload, maximum_bytes) if keep_tail else _decode_prefix(
            payload,
            maximum_bytes,
        )
    # Lo storico privilegia gli ultimi turni; documenti e fonti privilegiano l'inizio.
    if keep_tail:
        return f"{marker}{_decode_suffix(payload, content_bytes)}"
    return f"{_decode_prefix(payload, content_bytes)}{marker}"


def fit_observation(text: str, max_tokens: int) -> str:
    """
    Conserva testa e coda di una Observation lunga.

    Args:
        text: Output completo del tool.
        max_tokens: Budget assegnato allo scratchpad.

    Returns:
        Observation originale oppure finestra esplicita testa+coda.
    """
    # Le Observation brevi non devono essere alterate o riallocate.
    if estimate_tokens(text) <= max_tokens:
        return text
    # Metà testa e metà coda mantengono sia struttura sia diagnostica finale.
    side_tokens = max(1, (max_tokens - estimate_tokens(TRUNCATION_MARKER)) // 2)
    head = fit_text(text, side_tokens, marker="")
    tail = fit_text(text, side_tokens, keep_tail=True, marker="")
    return f"{head.rstrip()}{TRUNCATION_MARKER}{tail.lstrip()}"


# Marker microcompact allineato al pattern Claude Code clear_tool_uses.
MICROCOMPACT_MARKER_TEMPLATE = "[Observation precedente omessa: {tool} {path}]"
# Quante observation recenti restano intatte.
DEFAULT_MICROCOMPACT_KEEP_RECENT = 2


def microcompact_observations(
    observations: list[tuple[str, str, str]],
    *,
    keep_recent: int = DEFAULT_MICROCOMPACT_KEEP_RECENT,
    force: bool = False,
    token_threshold: int = 1800,
) -> list[tuple[str, str, str]]:
    """
    Sostituisce observation read-only vecchie con placeholder oltre soglia.

    Errori, write e le ultime ``keep_recent`` restano intatte.
    """
    if not observations:
        return []
    total_tokens = sum(estimate_tokens(content) for _, _, content in observations)
    if not force and total_tokens <= token_threshold:
        return list(observations)
    cutoff = max(0, len(observations) - keep_recent)
    compacted: list[tuple[str, str, str]] = []
    for index, (tool_name, path_hint, content) in enumerate(observations):
        lower = content.casefold()
        keep_full = (
            index >= cutoff
            or "errore" in lower
            or "error" in lower
            or tool_name
            in {
                "scrivi_file",
                "modifica_file",
                "riassumi_documento",
                "crea_cartella",
            }
            or "file scritto" in lower
            or "cartella creata" in lower
        )
        if keep_full:
            compacted.append((tool_name, path_hint, content))
            continue
        compacted.append(
            (
                tool_name,
                path_hint,
                MICROCOMPACT_MARKER_TEMPLATE.format(
                    tool=tool_name or "tool",
                    path=path_hint or "-",
                ),
            )
        )
    return compacted


def _testo_scratchpad(passi: list[tuple[str, str]]) -> str:
    """Riproduce il formato ReAct di log + Observation usato da LangChain."""
    return "".join(
        f"{log}\nObservation: {observation}\nThought:" for log, observation in passi
    )


def limita_passi_scratchpad(
    passi: list[tuple[str, str, str, str]],
    max_tokens: int,
) -> list[tuple[str, str]]:
    """
    Comprime i passi ReAct entro un tetto token cumulativo.

    Args:
        passi: Tuple (tool, path, log, observation) in ordine cronologico.
        max_tokens: Budget dello scratchpad. Non positivo svuota i contenuti.

    Returns:
        Coppie (log, observation) nello stesso ordine, già entro il budget
        quando il tetto lascia spazio ad almeno un marcatore.
    """
    if not passi:
        return []
    if max_tokens <= 0:
        return [("", "") for _ in passi]
    observations = [(tool, path, observation) for tool, path, _, observation in passi]
    rendered = _testo_scratchpad(
        [(log, observation) for _, _, log, observation in passi]
    )
    compacted = microcompact_observations(
        observations,
        force=estimate_tokens(rendered) > max_tokens,
    )
    mutable = [
        (log, content)
        for (_, _, log, _), (_, _, content) in zip(passi, compacted, strict=True)
    ]
    if estimate_tokens(_testo_scratchpad(mutable)) <= max_tokens:
        return mutable
    last_log, last_observation = mutable[-1]
    room = max(1, max_tokens // 4)
    mutable = [
        (
            log,
            MICROCOMPACT_MARKER_TEMPLATE.format(tool=tool or "tool", path=path or "-"),
        )
        for tool, path, log, _observation in passi[:-1]
    ]
    mutable.append(
        (
            fit_text(last_log, room, keep_tail=True),
            fit_observation(last_observation, room),
        )
    )
    return mutable
