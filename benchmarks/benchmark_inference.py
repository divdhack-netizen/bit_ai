"""Repeatable benchmark of raw Ollama and the closed-corpus agent (direct/ReAct)."""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from assistant.config import WORKSPACE
from assistant.host import detect_power
from assistant.inference_profiles import (
    InferenceRoute,
    physical_memory_bytes,
    profile_for_route,
    resolve_model_variant,
)
from assistant.metrics import estimate_tokens_from_text
from assistant.thermal import (
    invalidate_thermal_cache,
    read_thermal_state,
    read_cpu_temperature_celsius,
)

ModeName = Literal["ollama_raw", "agent_no_ui", "agent_rich"]

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
NANOSECONDS_PER_MILLISECOND = 1_000_000.0
NANOSECONDS_PER_SECOND = 1_000_000_000.0
GIBIBYTE = 1024**3
DEFAULT_COOLDOWN_SECONDS = 8.0
DEFAULT_REPEAT = 1
HTTP_TIMEOUT_SECONDS = 600.0
OLLAMA_WARMUP_TIMEOUT_SECONDS = 180.0
FUNCTIONAL_CHECK_TIMEOUT_SECONDS = 10.0
RAW_TEMPERATURE = 0
ANSWER_PREVIEW_CHARS = 240

_NOTE_BENCHMARK_PATH = "tmp/benchmark_inference_note.md"
_HELLO_BENCHMARK_PATH = "tmp/benchmark_inference_hello.txt"
_NOTES_DIR_BENCHMARK_PATH = "tmp/benchmark_notes"
_NOTES_FILE_BENCHMARK_PATH = "tmp/benchmark_notes/oggi.txt"
_MISSING_BENCHMARK_PATH = "tmp/benchmark_file_che_non_esiste.md"
_NOTE_BENCHMARK_CONTENT = (
    "# Progetto Aurora\n\n"
    "- Responsabile: Giulia Ferri\n"
    "- Scadenza consegna: 12 marzo 2027\n"
    "- Budget approvato: 40k euro\n"
    "- Obiettivo: migrare il servizio pagamenti su architettura event-driven.\n"
)
_LONG_CONTEXT_LINES = 120
_LONG_CONTEXT_EXPECTED = _LONG_CONTEXT_LINES + 1
_LONG_CONTEXT_TOLERANCE = 0.25


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """
    Functional case with a deterministic quality rubric.

    Attributes:
        name: CLI key of the case.
        prompt: Request sent to the model or the agent.
        route: Route used for the token profile (num_ctx/num_predict).
        contains_all: Fragments that must all appear (case-insensitive).
        contains_any: At least one of these fragments must appear.
        forbidden: Fragments that must not appear.
        min_chars: Minimum answer length.
        max_chars: Maximum length; None = no limit.
        max_llm_calls: Cap on LLM generations for the agent turn.
        functional_check: Name of the executable check registered in `_FUNCTIONAL_CHECKS`.
        history: Previous (user, assistant) exchanges to put into the history.
        setup_files: Files (relative path, content) created before the case.
        cleanup_files: Files removed before the case so the check is genuine.
        raw_capable: False if the case only makes sense with the agent.
    """

    name: str
    prompt: str
    route: InferenceRoute
    contains_all: tuple[str, ...] = ()
    contains_any: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    min_chars: int = 1
    max_chars: int | None = None
    max_llm_calls: int = 1
    functional_check: str | None = None
    history: tuple[tuple[str, str], ...] = ()
    setup_files: tuple[tuple[str, str], ...] = ()
    cleanup_files: tuple[str, ...] = ()
    raw_capable: bool = True


_CASES: dict[str, BenchmarkCase] = {
    "chat": BenchmarkCase(
        name="chat",
        prompt=(
            "Spiega in tre frasi cosa fa un assistente ReAct locale "
            "senza usare tool."
        ),
        route="direct",
        contains_all=("ReAct",),
        max_chars=1200,
    ),
    "coding": BenchmarkCase(
        name="coding",
        prompt=(
            "Scrivi una funzione Python `clamp(value, low, high)` con docstring "
            "e un esempio d'uso in commento. Solo codice."
        ),
        route="direct",
        contains_all=("def clamp",),
        functional_check="clamp",
    ),
    "coding_bugfix": BenchmarkCase(
        name="coding_bugfix",
        prompt=(
            "Questa funzione Python solleva ZeroDivisionError con una lista vuota:\n"
            "def media(valori):\n"
            "    return sum(valori) / len(valori)\n"
            "Correggila in modo che restituisca 0.0 per lista vuota. Solo codice."
        ),
        route="direct",
        contains_all=("def media",),
        functional_check="media_bugfix",
    ),
    "tool_selection": BenchmarkCase(
        name="tool_selection",
        prompt=(
            "Elenca i file nella cartella assistente senza modificarli. "
            "Usa i tool del workspace."
        ),
        route="react",
        contains_any=("session.py", "cli.py", "config.py"),
        max_llm_calls=4,
        raw_capable=False,
    ),
    "file_write": BenchmarkCase(
        name="file_write",
        prompt=(
            f"Crea il file {_HELLO_BENCHMARK_PATH} con una sola riga: "
            "benchmark-ok"
        ),
        route="react",
        max_llm_calls=3,
        functional_check="file_hello",
        cleanup_files=(_HELLO_BENCHMARK_PATH,),
        raw_capable=False,
    ),
    "notes_write": BenchmarkCase(
        name="notes_write",
        prompt=(
            f"Crea la cartella {_NOTES_DIR_BENCHMARK_PATH} e il file "
            f"{_NOTES_FILE_BENCHMARK_PATH} con una sola riga: nota-ok"
        ),
        route="react",
        max_llm_calls=4,
        functional_check="file_notes",
        cleanup_files=(_NOTES_FILE_BENCHMARK_PATH,),
        raw_capable=False,
    ),
    "summary": BenchmarkCase(
        name="summary",
        prompt=(
            f"Riassumi in due frasi il documento {_NOTE_BENCHMARK_PATH}."
        ),
        route="react",
        contains_all=("Aurora",),
        contains_any=("Giulia", "marzo", "40"),
        max_llm_calls=5,
        setup_files=((_NOTE_BENCHMARK_PATH, _NOTE_BENCHMARK_CONTENT),),
        raw_capable=False,
    ),
    "react_multistep": BenchmarkCase(
        name="react_multistep",
        prompt=(
            f"Leggi {_NOTE_BENCHMARK_PATH} e dimmi chi è il responsabile "
            "e la data di scadenza."
        ),
        route="react",
        contains_all=("Giulia",),
        contains_any=("12 marzo", "marzo 2027"),
        max_llm_calls=3,
        setup_files=((_NOTE_BENCHMARK_PATH, _NOTE_BENCHMARK_CONTENT),),
        raw_capable=False,
    ),
    "file_inesistente": BenchmarkCase(
        name="file_inesistente",
        prompt=(
            f"Leggi il file {_MISSING_BENCHMARK_PATH} e dimmi di cosa parla."
        ),
        route="react",
        contains_any=(
            "non esiste",
            "non è stato trovato",
            "non trovato",
            "inesistente",
            "non risulta",
            "non è presente",
            "non ho trovato",
            "impossibile leggere",
            "non riesco a leggere",
            "non è possibile leggere",
        ),
        forbidden=("Aurora",),
        max_llm_calls=4,
        cleanup_files=(_MISSING_BENCHMARK_PATH,),
        raw_capable=False,
    ),
    "follow_up_storico": BenchmarkCase(
        name="follow_up_storico",
        prompt="Approfondisci il secondo punto in due frasi.",
        route="direct",
        contains_any=("document", "comportamento", "specific"),
        history=(
            (
                "Quali sono tre vantaggi dei test unitari?",
                "1. Rilevano le regressioni presto.\n"
                "2. Documentano il comportamento atteso del codice.\n"
                "3. Rendono sicuro il refactoring.",
            ),
        ),
        raw_capable=False,
    ),
    "evidenze_fornite": BenchmarkCase(
        name="evidenze_fornite",
        prompt=(
            "Usando solo queste evidenze, indica il pilota citato.\n"
            "EVIDENZE:\n1. Pilota: Charles Leclerc\n"
            "Rispondi con il nome completo."
        ),
        route="direct",
        contains_all=("Leclerc",),
        max_chars=400,
    ),
    "long_context": BenchmarkCase(
        name="long_context",
        prompt=(
            "Contesto:\n"
            + (
                "riga di contesto riempitivo per stressare il prefill.\n"
                * _LONG_CONTEXT_LINES
            )
            + "Domanda finale: quante volte compare la parola 'contesto' "
            "nel paragrafo precedente? Rispondi solo con un numero."
        ),
        route="direct",
        functional_check="conteggio_contesto",
        max_chars=400,
    ),
}


@dataclass(frozen=True, slots=True)
class CheckContext:
    """Data available to the functional checks."""

    answer: str
    workspace: Path


@dataclass(frozen=True, slots=True)
class QualityOutcome:
    """Rubric verdict with the failure reasons."""

    passed: bool
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkCaseResult:
    """Outcome of a single measurement, comparable via JSON."""

    case: str
    mode: str
    alias: str
    quantization: str
    tag: str
    context_tokens: int
    output_tokens: int
    cold: bool
    ok: bool
    quality_pass: bool
    error: str | None
    load_ms: float
    prefill_ms: float
    decode_ms: float
    ttft_ms: float
    total_ms: float
    prompt_tokens: int
    output_tokens_count: int
    prompt_tokens_per_s: float
    decode_tokens_per_s: float
    thermal_state: int
    temp_celsius: float | None
    ui_updates: int = 0
    answer_preview: str = ""
    notes: list[str] = field(default_factory=list)
    think: bool = False
    thinking_tokens: int = 0
    on_battery: bool = False
    low_power_mode: bool = False
    route: str = ""
    llm_calls: int = 0
    tool_ms: float = 0.0
    rendering_ms: float = 0.0
    repeat_index: int = 0
    quality_failures: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class BenchmarkSuiteReport:
    """Aggregate report of the executed matrix."""

    machine: dict[str, Any]
    results: list[BenchmarkCaseResult]
    skipped: list[dict[str, str]]


# ---------------------------------------------------------------------------
# Quality rubric
# ---------------------------------------------------------------------------

_FENCED_PYTHON = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_NUMBERS = re.compile(r"\d+")
_CODE_START = re.compile(r"^(?:def |import |from |class )", re.MULTILINE)


def _extract_python_code(answer: str) -> str:
    """Return the Python code contained in the answer, with or without a fence."""
    blocks = _FENCED_PYTHON.findall(answer)
    if blocks:
        return "\n\n".join(block.strip() for block in blocks)
    match = _CODE_START.search(answer)
    if match is None:
        return answer
    return answer[match.start() :]


def _run_isolated_code(code: str, asserts: str) -> str | None:
    """Run generated code in a separate interpreter; None if the asserts pass."""
    script = f"{code}\n\n{asserts}\n"
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", script],
            capture_output=True,
            text=True,
            timeout=FUNCTIONAL_CHECK_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "codice: timeout in esecuzione"
    if completed.returncode == 0:
        return None
    stderr_tail = completed.stderr.strip().splitlines()[-1:] or ["errore sconosciuto"]
    return f"codice: {stderr_tail[0][:160]}"


def _check_clamp(ctx: CheckContext) -> str | None:
    asserts = (
        "assert clamp(5, 0, 3) == 3\n"
        "assert clamp(-1, 0, 3) == 0\n"
        "assert clamp(2, 0, 3) == 2\n"
    )
    return _run_isolated_code(_extract_python_code(ctx.answer), asserts)


def _check_average_bugfix(ctx: CheckContext) -> str | None:
    asserts = (
        "assert media([]) == 0.0\n"
        "assert media([2, 4]) == 3.0\n"
        "assert media([5]) == 5.0\n"
    )
    return _run_isolated_code(_extract_python_code(ctx.answer), asserts)


def _check_file_hello(ctx: CheckContext) -> str | None:
    target = ctx.workspace / _HELLO_BENCHMARK_PATH
    if not target.exists():
        return f"file: {_HELLO_BENCHMARK_PATH} non creato"
    content = target.read_text(encoding="utf-8", errors="replace").strip()
    if content != "benchmark-ok":
        return f"file: contenuto inatteso {content[:60]!r}"
    return None


def _check_file_notes(ctx: CheckContext) -> str | None:
    folder = ctx.workspace / _NOTES_DIR_BENCHMARK_PATH
    target = ctx.workspace / _NOTES_FILE_BENCHMARK_PATH
    if not folder.is_dir():
        return f"cartella: {_NOTES_DIR_BENCHMARK_PATH} non creata"
    if not target.is_file():
        return f"file: {_NOTES_FILE_BENCHMARK_PATH} non creato"
    content = target.read_text(encoding="utf-8", errors="replace")
    if "nota-ok" not in content:
        return f"file: manca nota-ok in {content[:60]!r}"
    return None


def _check_context_count(ctx: CheckContext) -> str | None:
    numbers = [int(raw) for raw in _NUMBERS.findall(ctx.answer)]
    if not numbers:
        return "conteggio: nessun numero nella risposta"
    low = _LONG_CONTEXT_EXPECTED * (1 - _LONG_CONTEXT_TOLERANCE)
    high = _LONG_CONTEXT_EXPECTED * (1 + _LONG_CONTEXT_TOLERANCE)
    if any(low <= value <= high for value in numbers):
        return None
    return f"conteggio: {numbers[:3]} fuori da [{low:.0f}, {high:.0f}]"


_FUNCTIONAL_CHECKS: dict[str, Callable[[CheckContext], str | None]] = {
    "clamp": _check_clamp,
    "media_bugfix": _check_average_bugfix,
    "file_hello": _check_file_hello,
    "file_notes": _check_file_notes,
    "conteggio_contesto": _check_context_count,
}


def _contains_all(answer: str, fragments: tuple[str, ...] | list[str]) -> bool:
    haystack = answer.casefold()
    return all(fragment.casefold() in haystack for fragment in fragments)


def _quality_pass(answer: str, contains: list[str]) -> bool:
    """Case-insensitive conjunctive check; without constraints a non-empty answer is enough."""
    if not contains:
        return bool(answer.strip())
    return _contains_all(answer, contains)


def evaluate_quality(
    case: BenchmarkCase,
    answer: str,
    *,
    llm_calls: int = 0,
    workspace: Path = WORKSPACE,
    check_llm_calls: bool = True,
) -> QualityOutcome:
    """
    Apply the case's whole rubric to the answer.

    Args:
        case: Case with a declarative rubric.
        answer: Final text produced.
        llm_calls: LLM generations counted by the callback (agent only).
        workspace: Root for the file checks.
        check_llm_calls: False for the raw path, where the count does not exist.

    Returns:
        Outcome with the list of violated constraints.
    """
    failures: list[str] = []
    text = answer.strip()
    if len(text) < case.min_chars:
        failures.append(f"lunghezza: {len(text)} < {case.min_chars}")
    if case.max_chars is not None and len(text) > case.max_chars:
        failures.append(f"lunghezza: {len(text)} > {case.max_chars}")
    haystack = text.casefold()
    for fragment in case.contains_all:
        if fragment.casefold() not in haystack:
            failures.append(f"manca: {fragment!r}")
    if case.contains_any and not any(
        fragment.casefold() in haystack for fragment in case.contains_any
    ):
        failures.append(f"nessuno tra: {list(case.contains_any)}")
    for fragment in case.forbidden:
        if fragment.casefold() in haystack:
            failures.append(f"vietato: {fragment!r}")
    if check_llm_calls and llm_calls > case.max_llm_calls:
        failures.append(f"llm_calls: {llm_calls} > {case.max_llm_calls}")
    if case.functional_check is not None:
        check = _FUNCTIONAL_CHECKS[case.functional_check]
        failure = check(CheckContext(text, workspace))
        if failure is not None:
            failures.append(failure)
    return QualityOutcome(passed=not failures, failures=tuple(failures))


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


def _http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float = HTTP_TIMEOUT_SECONDS,
) -> Any:
    """JSON request to Ollama using only the standard library."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else {}


def _ns_to_ms(value_ns: int | float | None) -> float:
    if value_ns is None:
        return 0.0
    return float(value_ns) / NANOSECONDS_PER_MILLISECOND


def _rate(count: int, duration_ns: int | float | None) -> float:
    if not duration_ns or duration_ns <= 0 or count <= 0:
        return 0.0
    return count / (float(duration_ns) / NANOSECONDS_PER_SECOND)


def _rate_ms(count: int, duration_ms: float) -> float:
    if duration_ms <= 0 or count <= 0:
        return 0.0
    return count / (duration_ms / 1000.0)


def _ollama_version(base_url: str) -> str | None:
    try:
        payload = _http_json("GET", f"{base_url}/api/version", timeout=3.0)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    return str(payload.get("version")) if isinstance(payload, dict) else None


def _unload_best_effort(base_url: str, tag: str) -> None:
    try:
        _http_json(
            "POST",
            f"{base_url}/api/generate",
            {"model": tag, "prompt": "", "stream": False, "keep_alive": 0},
            timeout=60.0,
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        # The best-effort unload must not block the next measurement.
        pass


def _warm_up(base_url: str, tag: str, context_tokens: int) -> float:
    """Load the weights with a minimal generation; return the load time in ms (0 on failure)."""
    try:
        payload = _http_json(
            "POST",
            f"{base_url}/api/generate",
            {
                "model": tag,
                "prompt": "ok",
                "stream": False,
                "think": False,
                "keep_alive": "30m",
                "options": {"num_ctx": context_tokens, "num_predict": 1},
            },
            timeout=OLLAMA_WARMUP_TIMEOUT_SECONDS,
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return 0.0
    return _ns_to_ms(payload.get("load_duration")) if isinstance(payload, dict) else 0.0


def measure_ollama_raw(
    *,
    tag: str,
    prompt: str,
    context_tokens: int,
    output_tokens: int,
    base_url: str = DEFAULT_OLLAMA_URL,
    keep_alive: str = "5m",
    cold: bool = False,
    contains: list[str] | None = None,
    think: bool = False,
    extra_options: dict[str, Any] | None = None,
    case: BenchmarkCase | None = None,
) -> BenchmarkCaseResult:
    """
    Measure a `/api/generate` generation with no UI and no agent.

    Args:
        tag: Explicit Ollama tag.
        prompt: Text sent to the model.
        context_tokens: `num_ctx`.
        output_tokens: `num_predict`.
        base_url: Ollama endpoint.
        keep_alive: Requested residency.
        cold: If True, try to unload before measuring.
        contains: Required fragments when no full case is passed.
        think: Native CoT.
        extra_options: Additional Ollama options (A/B of num_thread, num_batch, ...).
        case: Case with a full rubric; takes priority over `contains`.

    Returns:
        Result with native load/prefill/decode/TTFT metrics.
    """
    if cold:
        _unload_best_effort(base_url, tag)

    invalidate_thermal_cache()
    thermal_before = int(read_thermal_state())
    temp_before = read_cpu_temperature_celsius()
    options: dict[str, Any] = {
        "temperature": RAW_TEMPERATURE,
        "num_ctx": context_tokens,
        "num_predict": output_tokens,
    }
    if extra_options:
        options.update(extra_options)
    started = time.perf_counter()
    first_token_at: float | None = None
    answer_parts: list[str] = []
    thinking_parts: list[str] = []
    final_info: dict[str, Any] = {}
    error: str | None = None
    try:
        request = urllib.request.Request(
            f"{base_url}/api/generate",
            data=json.dumps(
                {
                    "model": tag,
                    "prompt": prompt,
                    "stream": True,
                    "keep_alive": keep_alive,
                    "think": think,
                    "options": options,
                }
            ).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                chunk = json.loads(line)
                token = str(chunk.get("response") or "")
                thought = str(chunk.get("thinking") or "")
                if (token or thought) and first_token_at is None:
                    first_token_at = time.perf_counter()
                if token:
                    answer_parts.append(token)
                if thought:
                    thinking_parts.append(thought)
                if chunk.get("done"):
                    final_info = chunk
    except Exception as exc:  # noqa: BLE001 — the report must capture every failure
        error = f"{type(exc).__name__}: {exc}"

    total_ms = (time.perf_counter() - started) * 1000.0
    answer = "".join(answer_parts)
    ttft_ms = (
        (first_token_at - started) * 1000.0 if first_token_at is not None else 0.0
    )
    prompt_tokens = int(final_info.get("prompt_eval_count") or 0)
    output_count = int(final_info.get("eval_count") or 0)
    thinking_text = "".join(thinking_parts)
    power = detect_power()
    ok = error is None and bool(answer.strip())
    if case is not None:
        quality = evaluate_quality(case, answer, check_llm_calls=False)
    else:
        quality = QualityOutcome(
            passed=_quality_pass(answer, list(contains or [])),
            failures=(),
        )
    return BenchmarkCaseResult(
        case=case.name if case else "",
        mode="ollama_raw",
        alias="",
        quantization="",
        tag=tag,
        context_tokens=context_tokens,
        output_tokens=output_tokens,
        cold=cold,
        ok=ok,
        quality_pass=ok and quality.passed,
        error=error,
        load_ms=_ns_to_ms(final_info.get("load_duration")),
        prefill_ms=_ns_to_ms(final_info.get("prompt_eval_duration")),
        decode_ms=_ns_to_ms(final_info.get("eval_duration")),
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        prompt_tokens=prompt_tokens,
        output_tokens_count=output_count,
        prompt_tokens_per_s=_rate(prompt_tokens, final_info.get("prompt_eval_duration")),
        decode_tokens_per_s=_rate(output_count, final_info.get("eval_duration")),
        thermal_state=thermal_before,
        temp_celsius=temp_before,
        answer_preview=answer[:ANSWER_PREVIEW_CHARS],
        notes=[f"answer_chars={len(answer)}", f"thinking_chars={len(thinking_text)}"],
        think=think,
        thinking_tokens=estimate_tokens_from_text(thinking_text),
        on_battery=power.on_battery,
        low_power_mode=power.low_power_mode,
        route=case.route if case else "",
        llm_calls=1,
        quality_failures=list(quality.failures),
    )


class BenchmarkSession:
    """Agent session reused across cases: avoids unloading/reloading the weights on every measurement."""

    def __init__(self, alias: str, mode: ModeName) -> None:
        from unittest.mock import MagicMock

        from assistant.console import ConsoleUI
        from assistant.session import AgentSession

        self.alias = alias
        self.mode = mode
        if mode == "agent_rich":
            self.ui: Any = ConsoleUI()
        else:
            self.ui = MagicMock(spec=ConsoleUI)
            self.ui.console = MagicMock()
            self.ui.thermal_bar = MagicMock()
            self.ui.has_live_stream = False
            self.ui.stream_refresh_count = 0
        self.session = AgentSession(self.ui)

    def prepare_case(self, case: BenchmarkCase) -> None:
        """Restore history, counters and workspace files for the case."""
        self.session.clear_memory()
        for user, assistant in case.history:
            self.session.record_history_turn(user, assistant)
        for relative_path in case.cleanup_files:
            target = self.session.workspace / relative_path
            if target.exists():
                target.unlink()
        for relative_path, content in case.setup_files:
            target = self.session.workspace / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

    def download_model(self) -> None:
        self.session.unload_active_weights()

    def close(self) -> None:
        self.session.close()


def measure_agent_case(
    *,
    case: BenchmarkCase,
    bench_session: BenchmarkSession,
    cold: bool = False,
    repeat_index: int = 0,
) -> BenchmarkCaseResult:
    """
    Run an agent case on the reused session.

    Args:
        case: Case with prompt and rubric.
        bench_session: Session already built for alias and mode.
        cold: If True, unload the weights before the turn.
        repeat_index: Ordinal of the repetition (0 = first measurement).

    Returns:
        Result with end-to-end metrics and the quality verdict.
    """
    from assistant.config import ConfigRuntime

    alias = bench_session.alias
    profile = profile_for_route(alias, case.route)
    config = ConfigRuntime(
        verbose=False,
        full_verification=False,
        burn_protection=False,
        model=alias,
        thinking=False,
        auto_fast_low_power=False,
    )
    bench_session.prepare_case(case)
    if cold:
        bench_session.download_model()

    invalidate_thermal_cache()
    thermal_before = int(read_thermal_state())
    temp_before = read_cpu_temperature_celsius()
    session = bench_session.session
    started = time.perf_counter()
    error: str | None = None
    answer = ""
    try:
        answer, _paths = session.run_request(case.prompt, config)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    total_ms = (time.perf_counter() - started) * 1000.0
    metrics = session.last_metrics()
    metrics_dict = metrics.as_dict() if metrics is not None else {}
    llm_calls = int(metrics_dict.get("chiamate_llm", 0))
    quality = evaluate_quality(
        case,
        str(answer),
        llm_calls=llm_calls,
        workspace=session.workspace,
    )
    ui_updates = int(getattr(bench_session.ui, "stream_refresh_count", 0) or 0)
    power = detect_power()
    prompt_tokens = int(metrics_dict.get("token_input", 0))
    prefill_ms = float(metrics_dict.get("durata_prefill_ms", 0.0))
    ok = error is None and bool(str(answer).strip())

    return BenchmarkCaseResult(
        case=case.name,
        mode=bench_session.mode,
        alias=alias,
        quantization=profile.variant.quantization,
        tag=profile.variant.tag,
        context_tokens=profile.context_tokens,
        output_tokens=profile.output_tokens,
        cold=cold,
        ok=ok,
        quality_pass=ok and quality.passed,
        error=error,
        load_ms=float(metrics_dict.get("durata_load_ms", 0.0)),
        prefill_ms=prefill_ms,
        decode_ms=max(
            0.0,
            float(metrics_dict.get("durata_llm_ms", 0.0))
            - float(metrics_dict.get("ttft_ms", 0.0)),
        ),
        ttft_ms=float(metrics_dict.get("ttft_ms", 0.0)),
        total_ms=total_ms,
        prompt_tokens=prompt_tokens,
        output_tokens_count=int(metrics_dict.get("token_output", 0)),
        prompt_tokens_per_s=_rate_ms(prompt_tokens, prefill_ms),
        decode_tokens_per_s=float(metrics_dict.get("token_al_secondo", 0.0)),
        thermal_state=thermal_before,
        temp_celsius=temp_before,
        ui_updates=ui_updates,
        answer_preview=str(answer)[:ANSWER_PREVIEW_CHARS],
        notes=[f"answer_chars={len(str(answer))}"],
        on_battery=power.on_battery,
        low_power_mode=power.low_power_mode,
        route=case.route,
        llm_calls=llm_calls,
        tool_ms=float(metrics_dict.get("durata_tool_ms", 0.0)),
        rendering_ms=float(metrics_dict.get("durata_rendering_ms", 0.0)),
        repeat_index=repeat_index,
        quality_failures=list(quality.failures),
    )


def _should_skip_variant(
    alias: str,
    *,
    total_memory_bytes: int,
) -> str | None:
    """Skip reason from the memory guard, or None if runnable."""
    variant = resolve_model_variant(alias)
    memory_gib = total_memory_bytes / GIBIBYTE if total_memory_bytes > 0 else 0.0
    if not variant.fits_memory(total_memory_bytes):
        return (
            f"{variant.tag} (~{variant.size_gb:.1f} GB pesi) non entra "
            f"con headroom su {memory_gib:.0f} GiB"
        )
    return None


def _machine_info(base_url: str, label: str) -> dict[str, Any]:
    memory_bytes = physical_memory_bytes()
    power = detect_power()
    return {
        "label": label,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "ollama_version": _ollama_version(base_url),
        "memory_bytes": memory_bytes,
        "memory_gib": round(memory_bytes / GIBIBYTE, 2) if memory_bytes else 0.0,
        "thermal_state": int(read_thermal_state()),
        "temp_celsius": read_cpu_temperature_celsius(),
        "on_battery": power.on_battery,
        "low_power_mode": power.low_power_mode,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _print_progress(result: BenchmarkCaseResult) -> None:
    verdict = "OK " if result.quality_pass else "KO "
    failures = f" [{'; '.join(result.quality_failures)}]" if result.quality_failures else ""
    print(
        f"{verdict}{result.alias:<8}{result.mode:<12}{result.case:<20}"
        f"rep={result.repeat_index} e2e={result.total_ms / 1000:.1f}s "
        f"ttft={result.ttft_ms:.0f}ms dec={result.decode_tokens_per_s:.1f}tok/s "
        f"in={result.prompt_tokens} out={result.output_tokens_count} "
        f"llm={result.llm_calls}{failures}",
        flush=True,
    )


def run_suite(
    *,
    aliases: list[str],
    contexts: list[int],
    cases: list[str],
    modes: list[ModeName],
    base_url: str,
    cooldown_seconds: float,
    think: bool = False,
    repeat: int = DEFAULT_REPEAT,
    cold: bool = False,
    raw_options: dict[str, Any] | None = None,
    label: str = "",
) -> BenchmarkSuiteReport:
    """
    Run the controlled matrix and return a serializable report.

    Args:
        aliases: Subset of fast/default/quality.
        contexts: Optional num_ctx overrides; empty uses the route budget.
        cases: Functional cases to measure.
        modes: Measurement paths.
        base_url: Ollama endpoint.
        cooldown_seconds: Pause between measurements to reduce thermal carry-over.
        think: Native CoT for the raw path.
        repeat: Warm repetitions per (alias, case, mode).
        cold: Add a cold measurement before the warm repetitions.
        raw_options: Extra Ollama options for the raw path.
        label: Free-form label for the iteration log.

    Returns:
        Report with machine metadata, results and justified skips.
    """
    machine = _machine_info(base_url, label)
    memory_bytes = int(machine["memory_bytes"])
    results: list[BenchmarkCaseResult] = []
    skipped: list[dict[str, str]] = []

    def pause() -> None:
        if cooldown_seconds > 0:
            time.sleep(cooldown_seconds)

    for alias in aliases:
        skip_reason = _should_skip_variant(alias, total_memory_bytes=memory_bytes)
        if skip_reason is not None:
            variant = resolve_model_variant(alias)
            skipped.append(
                {"alias": alias, "quantization": variant.quantization, "reason": skip_reason}
            )
            continue
        sessions: dict[ModeName, BenchmarkSession] = {}
        warm_profile = profile_for_route(alias, "react")
        load_ms = _warm_up(base_url, warm_profile.variant.tag, warm_profile.context_tokens)
        print(f"warm-up {alias} ({warm_profile.variant.tag}): load {load_ms:.0f} ms", flush=True)
        try:
            for mode in modes:
                for case_name in cases:
                    case = _CASES[case_name]
                    if mode == "ollama_raw" and not case.raw_capable:
                        continue
                    profile = profile_for_route(alias, case.route)
                    context_values = contexts or [profile.context_tokens]
                    for context_tokens in context_values:
                        cold_flags = (True, *([False] * repeat)) if cold else (False,) * repeat
                        for index, cold_flag in enumerate(cold_flags):
                            if mode == "ollama_raw":
                                measured = measure_ollama_raw(
                                    tag=profile.variant.tag,
                                    prompt=case.prompt,
                                    context_tokens=context_tokens,
                                    output_tokens=profile.output_tokens,
                                    base_url=base_url,
                                    cold=cold_flag,
                                    think=think,
                                    extra_options=raw_options,
                                    case=case,
                                )
                                result = replace(
                                    measured,
                                    alias=alias,
                                    quantization=profile.variant.quantization,
                                    repeat_index=index,
                                )
                            else:
                                bench_session = sessions.get(mode)
                                if bench_session is None:
                                    bench_session = BenchmarkSession(alias, mode)
                                    sessions[mode] = bench_session
                                result = measure_agent_case(
                                    case=case,
                                    bench_session=bench_session,
                                    cold=cold_flag,
                                    repeat_index=index,
                                )
                            results.append(result)
                            _print_progress(result)
                            pause()
        finally:
            for bench_session in sessions.values():
                bench_session.close()
    return BenchmarkSuiteReport(machine=machine, results=results, skipped=skipped)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def aggregate_groups(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Aggregate the warm measurements per (alias, case, mode) with medians.

    Args:
        results: Serialized results (dicts) from one or more runs.

    Returns:
        One dictionary per group, sorted by alias, mode and case.
    """
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for item in results:
        if item.get("cold"):
            continue
        key = (str(item["alias"]), str(item["case"]), str(item["mode"]))
        groups.setdefault(key, []).append(item)
    aggregated: list[dict[str, Any]] = []
    for (alias, case, mode), items in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][1])):
        decodes = [float(i["decode_tokens_per_s"]) for i in items if i["decode_tokens_per_s"]]
        aggregated.append(
            {
                "alias": alias,
                "case": case,
                "mode": mode,
                "n": len(items),
                "quality_pass_rate": sum(1 for i in items if i["quality_pass"]) / len(items),
                "median_total_ms": _median([float(i["total_ms"]) for i in items]),
                "median_ttft_ms": _median([float(i["ttft_ms"]) for i in items]),
                "median_decode_tokens_per_s": _median(decodes),
                "median_prefill_ms": _median([float(i["prefill_ms"]) for i in items]),
                "median_prompt_tokens": _median([float(i["prompt_tokens"]) for i in items]),
                "median_output_tokens": _median(
                    [float(i["output_tokens_count"]) for i in items]
                ),
                "median_llm_calls": _median([float(i.get("llm_calls", 0)) for i in items]),
                "median_rendering_ms": _median(
                    [float(i.get("rendering_ms", 0.0)) for i in items]
                ),
                "failures": sorted(
                    {f for i in items for f in i.get("quality_failures", [])}
                ),
            }
        )
    return aggregated


def aggregate_by_alias(groups: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Reduce the groups to one profile per alias (medians of medians, e2e sums)."""
    per_alias: dict[str, dict[str, Any]] = {}
    for alias in sorted({g["alias"] for g in groups}):
        own = [g for g in groups if g["alias"] == alias]
        agent = [g for g in own if g["mode"] != "ollama_raw"]
        raw = [g for g in own if g["mode"] == "ollama_raw"]
        per_alias[alias] = {
            "groups": len(own),
            "quality_pass_groups": sum(1 for g in own if g["quality_pass_rate"] >= 1.0),
            "quality_pass_rate": _median([g["quality_pass_rate"] for g in own]),
            "median_decode_tokens_per_s_raw": _median(
                [g["median_decode_tokens_per_s"] for g in raw]
            ),
            "median_decode_tokens_per_s_agent": _median(
                [g["median_decode_tokens_per_s"] for g in agent]
            ),
            "sum_median_total_ms_agent": sum(g["median_total_ms"] for g in agent),
            "median_ttft_ms_agent": _median([g["median_ttft_ms"] for g in agent]),
            "sum_median_prompt_tokens_agent": sum(
                g["median_prompt_tokens"] for g in agent
            ),
            "sum_median_llm_calls_agent": sum(g["median_llm_calls"] for g in agent),
        }
    return per_alias


def _summarize(results: list[BenchmarkCaseResult]) -> dict[str, Any]:
    """Aggregate global, per-group and per-alias medians."""
    if not results:
        return {}
    serialized = [asdict(item) for item in results]
    totals = [item.total_ms for item in results if item.ok]
    decodes = [item.decode_tokens_per_s for item in results if item.decode_tokens_per_s]
    groups = aggregate_groups(serialized)
    return {
        "measure_count": len(results),
        "ok_count": sum(1 for item in results if item.ok),
        "quality_pass_count": sum(1 for item in results if item.quality_pass),
        "median_total_ms": _median(totals),
        "median_decode_tokens_per_s": _median(decodes),
        "thermal_states": sorted({item.thermal_state for item in results}),
        "groups": groups,
        "per_alias": aggregate_by_alias(groups),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_option(raw: str) -> tuple[str, Any]:
    """Turn `key=value` into a typed pair (int, float, bool or str)."""
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"opzione non valida: {raw!r} (atteso chiave=valore)")
    key, value = raw.split("=", 1)
    lowered = value.strip().casefold()
    if lowered in {"true", "false"}:
        return key.strip(), lowered == "true"
    try:
        return key.strip(), int(value)
    except ValueError:
        pass
    try:
        return key.strip(), float(value)
    except ValueError:
        return key.strip(), value


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Ollama/agente con metriche load/prefill/decode/TTFT "
            "e rubrica di qualità."
        )
    )
    parser.add_argument(
        "--alias",
        nargs="+",
        default=["default"],
        choices=["fast", "default"],
    )
    parser.add_argument(
        "--context",
        nargs="*",
        type=int,
        default=[],
        help="Override num_ctx; default = budget della route",
    )
    parser.add_argument(
        "--case",
        nargs="+",
        default=["chat", "coding"],
        choices=sorted(_CASES),
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        default=["ollama_raw"],
        choices=["ollama_raw", "agent_no_ui", "agent_rich"],
    )
    parser.add_argument(
        "--think",
        choices=["on", "off"],
        default="off",
        help="CoT nativo Ollama per ollama_raw (default off)",
    )
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT, help="Ripetizioni warm")
    parser.add_argument(
        "--cold",
        action="store_true",
        help="Aggiunge una misura cold (unload + load) prima delle ripetizioni",
    )
    parser.add_argument(
        "--raw-option",
        action="append",
        type=_parse_option,
        default=[],
        metavar="CHIAVE=VALORE",
        help="Opzione Ollama extra per ollama_raw (es. num_thread=8, num_batch=1024)",
    )
    parser.add_argument("--label", default="", help="Etichetta del run per il registro")
    parser.add_argument("--base-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN_SECONDS)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/benchmark_inference.json"),
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Elenca i casi disponibili ed esce",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra la matrice pianificata senza chiamare Ollama",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point of the benchmark.

    Returns:
        0 if the report was produced; 2 on configuration errors.
    """
    args = _parse_args(argv)
    if args.list_cases:
        for name, case in _CASES.items():
            print(
                f"{name}: route={case.route} max_llm_calls={case.max_llm_calls} "
                f"raw={'sì' if case.raw_capable else 'no'}"
            )
        return 0

    if args.dry_run:
        memory_bytes = physical_memory_bytes()
        planned: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        for alias in args.alias:
            reason = _should_skip_variant(alias, total_memory_bytes=memory_bytes)
            variant = resolve_model_variant(alias)
            if reason:
                skipped.append(
                    {"alias": alias, "quantization": variant.quantization, "reason": reason}
                )
                continue
            for case_name in args.case:
                planned.append(
                    {
                        "alias": alias,
                        "quantization": variant.quantization,
                        "case": case_name,
                        "modes": args.mode,
                        "repeat": args.repeat,
                        "cold": args.cold,
                    }
                )
        print(json.dumps({"planned": planned, "skipped": skipped}, indent=2, ensure_ascii=False))
        return 0

    if args.repeat < 1:
        print("--repeat deve essere >= 1", file=sys.stderr)
        return 2

    report = run_suite(
        aliases=list(args.alias),
        contexts=list(args.context),
        cases=list(args.case),
        modes=list(args.mode),
        base_url=args.base_url,
        cooldown_seconds=float(args.cooldown),
        think=args.think == "on",
        repeat=int(args.repeat),
        cold=bool(args.cold),
        raw_options=dict(args.raw_option) if args.raw_option else None,
        label=str(args.label),
    )
    payload = {
        "machine": report.machine,
        "skipped": report.skipped,
        "results": [asdict(item) for item in report.results],
        "summary": _summarize(report.results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Scritto {args.output} ({len(report.results)} misure, {len(report.skipped)} skip)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
