"""Memoria conversazionale: storico turni entro budget e blocco STORICO del prompt."""

from __future__ import annotations

import re
from dataclasses import dataclass

from assistente.prompt_budget import estimate_tokens, fit_text

# Quanti scambi utente↔assistente tenere nel prompt (più recenti).
MAX_TURNI_STORICO = 6
# Budget caratteri totale per lo storico (lascia spazio a tool/scratchpad).
MAX_CARATTERI_STORICO = 7000
# Troncamento per singola risposta assistente nello storico (elenchi lunghi).
MAX_CARATTERI_RISPOSTA_STORICO = 3000
# Riserva fissa del blocco compatto quando i turni vecchi non entrano verbatim.
COMPACT_RESERVE_TOKENS = 200
# Caratteri di testa e coda nella riga di esito del compatto.
_ESITO_CARATTERI = 80
# Separatore tra turni in formatta().
_SEP_TURNI = "\n\n"
_SEP_TURNI_LEN = len(_SEP_TURNI)
# Path citati nelle risposte per la sezione «File toccati».
_PATH_CITATI = re.compile(r"[\w./-]+\.(?:java|py|js|css|html?|json|txt|md|toml|ya?ml)")

# Prefisso per direct/ReAct: coerenza del filo senza evidenze fresche.
_PREFISSO_STORICO = "STORICO CONVERSAZIONE (usa per follow-up; non negarlo):\n"
REGOLA_QA_SOLO_CHAT = (
    "Domande sui documenti: rispondi solo in chat; niente scrivi_file né salvataggi spontanei."
)


@dataclass(frozen=True)
class TurnoConversazione:
    """Uno scambio completo: richiesta utente e risposta assistente."""

    utente: str
    assistente: str
    indice: int


def blocco_storico_prompt(storico: str) -> str:
    """
    Serializza lo storico per i prompt ReAct e direct.

    Args:
        storico: Testo già formattato da MemoriaConversazione.formatta (o vuoto).

    Returns:
        Blocco etichettato con newline finali, oppure stringa vuota.
    """
    testo = storico.strip()
    # Senza turni non inquinare il prompt con un header vuoto.
    if not testo:
        return ""
    return f"{_PREFISSO_STORICO}{testo}\n\n"


class MemoriaConversazione:
    """
    Buffer FIFO degli ultimi turni della sessione interattiva.

    Input: messaggi utente e risposte finali già mostrate.
    Output: testo formattato da iniettare nel prompt LLM.
    """

    def __init__(
        self,
        max_turni: int = MAX_TURNI_STORICO,
        max_caratteri: int = MAX_CARATTERI_STORICO,
        max_caratteri_risposta: int = MAX_CARATTERI_RISPOSTA_STORICO,
    ) -> None:
        self._max_turni = max_turni
        self._max_caratteri = max_caratteri
        self._max_token_risposta = max(1, max_caratteri_risposta // 3)
        self._turni: list[TurnoConversazione] = []
        self._prossimo_indice = 1

    @property
    def turni(self) -> list[TurnoConversazione]:
        """Copia difensiva dei turni memorizzati."""
        return list(self._turni)

    def ha_turni(self) -> bool:
        """True se esiste almeno uno scambio precedente."""
        return bool(self._turni)

    def svuota(self) -> None:
        """Azzera lo storico (es. comando /clear)."""
        self._turni.clear()
        self._prossimo_indice = 1

    def registra(self, utente: str, assistente: str) -> None:
        """
        Aggiunge un turno e taglia i più vecchi oltre i limiti.

        Precondizione: `utente` e `assistente` non vuoti dopo strip.
        """
        testo_utente = utente.strip()
        testo_assistente = assistente.strip()
        # Skip turni incompleti: non contaminano lo storico.
        if not testo_utente or not testo_assistente:
            return
        self._turni.append(
            TurnoConversazione(
                utente=testo_utente,
                assistente=testo_assistente,
                indice=self._prossimo_indice,
            )
        )
        self._prossimo_indice += 1
        self._riduci()

    def formatta(self, max_tokens: int | None = None) -> str:
        """
        Serializza lo storico per il prompt.

        Args:
            max_tokens: Budget stimato opzionale; None preserva il formato storico.

        Returns:
            Stringa vuota se non ci sono turni; altrimenti turni recenti in
            ordine cronologico entro il budget richiesto. Se i turni non
            entrano, compatta i più vecchi in un blocco strutturato.
        """
        if not self._turni:
            return ""
        # Il percorso legacy mantiene tutti i turni già ridotti dai limiti a caratteri.
        if max_tokens is None:
            blocchi: list[str] = []
            # Itera dal più vecchio al più recente (ordine di lettura naturale).
            for turno in self._turni:
                blocchi.append(self._formatta_turno(turno.indice, turno))
            return _SEP_TURNI.join(blocchi)

        # Un budget nullo evita di aggiungere un header storico privo di contenuto.
        if max_tokens <= 0:
            return ""
        sep_tokens = estimate_tokens(_SEP_TURNI)
        full_blocks = [
            self._formatta_turno(turno.indice, turno) for turno in self._turni
        ]
        full = _SEP_TURNI.join(full_blocks)
        if estimate_tokens(full) <= max_tokens:
            return full
        reserve = min(COMPACT_RESERVE_TOKENS, max(1, max_tokens // 2))
        verbatim_budget = max(1, max_tokens - reserve)
        selected_blocks: list[tuple[TurnoConversazione, str]] = []
        used_tokens = 0
        for turno in reversed(self._turni):
            extra = sep_tokens if selected_blocks else 0
            room = verbatim_budget - used_tokens - extra
            if room <= 1:
                break
            verbatim = self._formatta_turno(turno.indice, turno)
            block = (
                verbatim
                if estimate_tokens(verbatim) <= room
                else self._formatta_entro(turno, room)
            )
            cost = estimate_tokens(block)
            if cost > room:
                break
            selected_blocks.append((turno, block))
            used_tokens += extra + cost
            if block != verbatim:
                break
        selected_blocks.reverse()
        if not selected_blocks and self._turni:
            reserve = min(reserve, max(1, max_tokens // 3))
            room = max(1, max_tokens - reserve)
            last = self._turni[-1]
            selected_blocks = [(last, self._formatta_entro(last, room))]
        omitted = self._turni[: len(self._turni) - len(selected_blocks)]
        recent_blocks = [block for _, block in selected_blocks]
        if not omitted:
            return _SEP_TURNI.join(recent_blocks)
        compact = fit_text(self._compatta_turni(omitted), reserve)
        combined = _SEP_TURNI.join([compact, *recent_blocks])
        if estimate_tokens(combined) <= max_tokens:
            return combined
        return self._entro_budget(compact, selected_blocks[-1][0], max_tokens)

    def _compatta_turni(self, turni: list[TurnoConversazione]) -> str:
        """Riassume euristicamente i turni esclusi dal budget, in un solo passaggio."""
        files: list[str] = []
        seen_files: set[str] = set()
        errors: list[str] = []
        user_messages: list[str] = []
        outcomes: list[str] = []
        for turno in turni:
            user_messages.append(turno.utente.strip())
            outcome = _riga_esito(turno.assistente)
            if outcome:
                outcomes.append(outcome)
            for path in _PATH_CITATI.findall(turno.assistente):
                if path not in seen_files:
                    seen_files.add(path)
                    files.append(path)
            lower = turno.assistente.casefold()
            if "errore" in lower or "error" in lower or "attenzione:" in lower:
                snippet = turno.assistente.strip().replace("\n", " ")
                errors.append(snippet[:160])
        files_line = ", ".join(files[:12]) if files else "(nessuno)"
        errors_line = " | ".join(errors[-3:]) if errors else "(nessuno)"
        constraints = " | ".join(user_messages[-3:])
        outcomes_line = " | ".join(outcomes[-3:]) if outcomes else "(nessuno)"
        return (
            "--- Compact storico ---\n"
            f"1. Richiesta primaria: {user_messages[0]}\n"
            f"2. Vincoli utente (verbatim recenti): {constraints}\n"
            f"3. File toccati: {files_line}\n"
            f"4. Errori/fix: {errors_line}\n"
            f"5. Lavoro corrente (ultimo turno compatto): {user_messages[-1]}\n"
            f"6. Esiti: {outcomes_line}"
        )

    def _entro_budget(
        self,
        compact: str,
        turno: TurnoConversazione,
        max_tokens: int,
    ) -> str:
        """Un solo taglio: tiene il compatto e la richiesta del turno recente."""
        utente = f"--- Turno {turno.indice} ---\nUtente: {turno.utente}"
        head = f"{compact}{_SEP_TURNI}{utente}" if compact else utente
        if estimate_tokens(head) >= max_tokens:
            return fit_text(utente, max_tokens)
        room = max_tokens - estimate_tokens(f"{head}\nAssistente: ")
        if room <= 1:
            return fit_text(head, max_tokens)
        body = fit_text(turno.assistente, room, keep_tail=True)
        return f"{head}\nAssistente: {body}"

    def _formatta_entro(self, turno: TurnoConversazione, max_tokens: int) -> str:
        """Accorcia la risposta dalla coda e conserva la richiesta utente."""
        if max_tokens <= 0:
            return ""
        utente = f"--- Turno {turno.indice} ---\nUtente: {turno.utente}"
        if estimate_tokens(utente) >= max_tokens:
            return fit_text(utente, max_tokens)
        header = f"{utente}\nAssistente: "
        room = max_tokens - estimate_tokens(header)
        if room <= 1:
            return utente
        body = fit_text(turno.assistente, room, keep_tail=True)
        return f"{header}{body}"

    def _formatta_turno(self, indice: int, turno: TurnoConversazione) -> str:
        """Un singolo blocco Turno N con risposta eventualmente troncata."""
        risposta = self._tronca_risposta(turno.assistente)
        return (
            f"--- Turno {indice} ---\n"
            f"Utente: {turno.utente}\n"
            f"Assistente: {risposta}"
        )

    def _tronca_risposta(self, testo: str) -> str:
        """Limita una risposta dello storico in token, conservando la coda (l'esito)."""
        if estimate_tokens(testo) <= self._max_token_risposta:
            return testo
        return fit_text(testo, self._max_token_risposta, keep_tail=True)

    def _lunghezza_formattata(self) -> int:
        """Somma lineare delle lunghezze esatte dei blocchi + separatori (O(n))."""
        if not self._turni:
            return 0
        totale = sum(
            len(self._formatta_turno(turno.indice, turno))
            for turno in self._turni
        )
        # Separatori tra N turni: N-1 volte "\\n\\n".
        return totale + _SEP_TURNI_LEN * (len(self._turni) - 1)

    def _riduci(self) -> None:
        """Applica limiti su numero turni e budget caratteri senza formatta() ripetuto."""
        # Prima: solo gli ultimi N turni.
        if len(self._turni) > self._max_turni:
            self._turni = self._turni[-self._max_turni :]
        # Poi: togli i più vecchi finché la stima entra nel budget.
        while len(self._turni) > 1 and self._lunghezza_formattata() > self._max_caratteri:
            self._turni.pop(0)


def _riga_esito(testo: str) -> str:
    """Prima e ultima riga non vuota, per ritrovare un elenco anche nel compatto."""
    lines = [line.strip() for line in testo.splitlines() if line.strip()]
    if not lines:
        return ""
    head = lines[0][:_ESITO_CARATTERI]
    tail = lines[-1][:_ESITO_CARATTERI]
    if head == tail:
        return head
    return f"{head} … {tail}"
