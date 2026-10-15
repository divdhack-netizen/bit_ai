"""Conversation memory: turn history within budget and the prompt's STORICO block."""

from __future__ import annotations

import re
from dataclasses import dataclass

from assistant.prompt_budget import estimate_tokens, fit_text

# How many user↔assistant exchanges to keep in the prompt (most recent).
MAX_HISTORY_TURNS = 6
# Total character budget for the history (leaves room for tools/scratchpad).
MAX_HISTORY_CHARS = 7000
# Truncation per single assistant answer in the history (long lists).
MAX_HISTORY_ANSWER_CHARS = 3000
# Fixed reserve for the compact block when old turns do not fit verbatim.
COMPACT_RESERVE_TOKENS = 200
# Head and tail characters in the compact block's outcome line.
_OUTCOME_CHARS = 80
# Separator between turns in format().
_TURN_SEP = "\n\n"
_TURN_SEP_LEN = len(_TURN_SEP)
# Paths cited in the answers, for the «File toccati» section.
_CITED_PATHS = re.compile(r"[\w./-]+\.(?:java|py|js|css|html?|json|txt|md|toml|ya?ml)")

# Prefix for direct/ReAct: thread consistency without fresh evidence.
_HISTORY_PREFIX = "STORICO CONVERSAZIONE (usa per follow-up; non negarlo):\n"
CHAT_ONLY_QA_RULE = (
    "Domande sui documenti: rispondi solo in chat; niente scrivi_file né salvataggi spontanei."
)


@dataclass(frozen=True)
class ConversationTurn:
    """One complete exchange: user request and assistant answer."""

    user: str
    assistant: str
    index: int


def history_prompt_block(history: str) -> str:
    """
    Serialize the history for the ReAct and direct prompts.

    Args:
        history: Text already formatted by ConversationMemory.format (or empty).

    Returns:
        Labelled block with trailing newlines, or an empty string.
    """
    text = history.strip()
    # With no turns, do not pollute the prompt with an empty header.
    if not text:
        return ""
    return f"{_HISTORY_PREFIX}{text}\n\n"


class ConversationMemory:
    """
    FIFO buffer of the last turns of the interactive session.

    Input: user messages and final answers already shown.
    Output: formatted text to inject into the LLM prompt.
    """

    def __init__(
        self,
        max_turns: int = MAX_HISTORY_TURNS,
        max_chars: int = MAX_HISTORY_CHARS,
        max_answer_chars: int = MAX_HISTORY_ANSWER_CHARS,
    ) -> None:
        self._max_turns = max_turns
        self._max_chars = max_chars
        self._max_answer_tokens = max(1, max_answer_chars // 3)
        self._turns: list[ConversationTurn] = []
        self._next_index = 1

    @property
    def turns(self) -> list[ConversationTurn]:
        """Defensive copy of the stored turns."""
        return list(self._turns)

    def has_turns(self) -> bool:
        """True if at least one previous exchange exists."""
        return bool(self._turns)

    def clear(self) -> None:
        """Reset the history (e.g. the /clear command)."""
        self._turns.clear()
        self._next_index = 1

    def record(self, user: str, assistant: str) -> None:
        """
        Add a turn and drop the oldest ones beyond the limits.

        Precondition: `user` and `assistant` are non-empty after strip.
        """
        user_text = user.strip()
        assistant_text = assistant.strip()
        # Skip incomplete turns: they do not contaminate the history.
        if not user_text or not assistant_text:
            return
        self._turns.append(
            ConversationTurn(
                user=user_text,
                assistant=assistant_text,
                index=self._next_index,
            )
        )
        self._next_index += 1
        self._reduce()

    def format(self, max_tokens: int | None = None) -> str:
        """
        Serialize the history for the prompt.

        Args:
            max_tokens: Optional estimated budget; None keeps the historical format.

        Returns:
            Empty string if there are no turns; otherwise recent turns in
            chronological order within the requested budget. If the turns do
            not fit, the oldest are compacted into a structured block.
        """
        if not self._turns:
            return ""
        # The legacy path keeps every turn already reduced by the character limits.
        if max_tokens is None:
            blocks: list[str] = []
            # Iterate from oldest to newest (natural reading order).
            for turn in self._turns:
                blocks.append(self._format_turn(turn.index, turn))
            return _TURN_SEP.join(blocks)

        # A zero budget avoids adding a history header with no content.
        if max_tokens <= 0:
            return ""
        sep_tokens = estimate_tokens(_TURN_SEP)
        full_blocks = [
            self._format_turn(turn.index, turn) for turn in self._turns
        ]
        full = _TURN_SEP.join(full_blocks)
        if estimate_tokens(full) <= max_tokens:
            return full
        reserve = min(COMPACT_RESERVE_TOKENS, max(1, max_tokens // 2))
        verbatim_budget = max(1, max_tokens - reserve)
        selected_blocks: list[tuple[ConversationTurn, str]] = []
        used_tokens = 0
        for turn in reversed(self._turns):
            extra = sep_tokens if selected_blocks else 0
            room = verbatim_budget - used_tokens - extra
            if room <= 1:
                break
            verbatim = self._format_turn(turn.index, turn)
            block = (
                verbatim
                if estimate_tokens(verbatim) <= room
                else self._format_within(turn, room)
            )
            cost = estimate_tokens(block)
            if cost > room:
                break
            selected_blocks.append((turn, block))
            used_tokens += extra + cost
            if block != verbatim:
                break
        selected_blocks.reverse()
        if not selected_blocks and self._turns:
            reserve = min(reserve, max(1, max_tokens // 3))
            room = max(1, max_tokens - reserve)
            last = self._turns[-1]
            selected_blocks = [(last, self._format_within(last, room))]
        omitted = self._turns[: len(self._turns) - len(selected_blocks)]
        recent_blocks = [block for _, block in selected_blocks]
        if not omitted:
            return _TURN_SEP.join(recent_blocks)
        compact = fit_text(self._compact_turns(omitted), reserve)
        combined = _TURN_SEP.join([compact, *recent_blocks])
        if estimate_tokens(combined) <= max_tokens:
            return combined
        return self._within_budget(compact, selected_blocks[-1][0], max_tokens)

    def _compact_turns(self, turns: list[ConversationTurn]) -> str:
        """Heuristically summarize the turns left out of the budget, in a single pass."""
        files: list[str] = []
        seen_files: set[str] = set()
        errors: list[str] = []
        user_messages: list[str] = []
        outcomes: list[str] = []
        for turn in turns:
            user_messages.append(turn.user.strip())
            outcome = _outcome_line(turn.assistant)
            if outcome:
                outcomes.append(outcome)
            for path in _CITED_PATHS.findall(turn.assistant):
                if path not in seen_files:
                    seen_files.add(path)
                    files.append(path)
            lower = turn.assistant.casefold()
            if "errore" in lower or "error" in lower or "attenzione:" in lower:
                snippet = turn.assistant.strip().replace("\n", " ")
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

    def _within_budget(
        self,
        compact: str,
        turn: ConversationTurn,
        max_tokens: int,
    ) -> str:
        """A single cut: keep the compact block and the request of the recent turn."""
        user = f"--- Turno {turn.index} ---\nUtente: {turn.user}"
        head = f"{compact}{_TURN_SEP}{user}" if compact else user
        if estimate_tokens(head) >= max_tokens:
            return fit_text(user, max_tokens)
        room = max_tokens - estimate_tokens(f"{head}\nAssistente: ")
        if room <= 1:
            return fit_text(head, max_tokens)
        body = fit_text(turn.assistant, room, keep_tail=True)
        return f"{head}\nAssistente: {body}"

    def _format_within(self, turn: ConversationTurn, max_tokens: int) -> str:
        """Shorten the answer from the tail and keep the user request."""
        if max_tokens <= 0:
            return ""
        user = f"--- Turno {turn.index} ---\nUtente: {turn.user}"
        if estimate_tokens(user) >= max_tokens:
            return fit_text(user, max_tokens)
        header = f"{user}\nAssistente: "
        room = max_tokens - estimate_tokens(header)
        if room <= 1:
            return user
        body = fit_text(turn.assistant, room, keep_tail=True)
        return f"{header}{body}"

    def _format_turn(self, index: int, turn: ConversationTurn) -> str:
        """A single Turno N block with a possibly truncated answer."""
        answer = self._truncate_answer(turn.assistant)
        return (
            f"--- Turno {index} ---\n"
            f"Utente: {turn.user}\n"
            f"Assistente: {answer}"
        )

    def _truncate_answer(self, text: str) -> str:
        """Cap a history answer in tokens, keeping the tail (the outcome)."""
        if estimate_tokens(text) <= self._max_answer_tokens:
            return text
        return fit_text(text, self._max_answer_tokens, keep_tail=True)

    def _formatted_length(self) -> int:
        """Linear sum of the exact block lengths + separators (O(n))."""
        if not self._turns:
            return 0
        total = sum(
            len(self._format_turn(turn.index, turn))
            for turn in self._turns
        )
        # Separators between N turns: N-1 times "\\n\\n".
        return total + _TURN_SEP_LEN * (len(self._turns) - 1)

    def _reduce(self) -> None:
        """Apply the turn-count and character-budget limits without repeated format()."""
        # First: only the last N turns.
        if len(self._turns) > self._max_turns:
            self._turns = self._turns[-self._max_turns :]
        # Then: drop the oldest until the estimate fits the budget.
        while len(self._turns) > 1 and self._formatted_length() > self._max_chars:
            self._turns.pop(0)


def _outcome_line(text: str) -> str:
    """First and last non-empty line, to recover a list even in the compact block."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    head = lines[0][:_OUTCOME_CHARS]
    tail = lines[-1][:_OUTCOME_CHARS]
    if head == tail:
        return head
    return f"{head} … {tail}"
