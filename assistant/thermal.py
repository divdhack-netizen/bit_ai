"""macOS thermal monitoring and agent performance throttling."""

from __future__ import annotations

import atexit
import json
import platform
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from rich.cells import cell_len

from assistant.config import (
    DEFAULT_MODEL_ALIAS,
    MAX_AGENT_ITERATIONS,
    MAX_GENERATED_TOKENS,
)
from assistant.host import CoolingProfile, PowerState, detect_host
from assistant.inference_profiles import MODEL_ALIAS_ORDER
from assistant.metrics import LLMTurnStats, format_stats_segment

# Separator between the thermal prefix and the metrics block (fixed width in cells)
_METRICS_SEPARATOR = "  ·  "

if TYPE_CHECKING:
    from assistant.config import ConfigRuntime
    from assistant.console import ConsoleUI

_FORCE_FAST_STEPS = 99
_SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"
_SCRIPT_MONITOR = _SCRIPT_DIR / "thermal_monitor.swift"
_THERMAL_STATE_COMMAND = [
    "swift",
    "-e",
    "import Foundation; print(ProcessInfo.processInfo.thermalState.rawValue)",
]
_TIMEOUT_SUBPROCESS_SEC = 3
_POLLING_INTERVAL_SEC = 2.0
_MONITOR_INTERVAL_MS = 2000
_TTL_CACHE_SEC = 1.5
_TEMP_CPU_MAX_PLAUSIBLE = 150.0
_NOT_CACHED = object()
_ANSI_DIM = "\033[2m"
_ANSI_RESET = "\033[0m"
# Short moving average: smooths SMC spikes without delaying the reaction too much.
_TEMP_AVERAGE_WINDOW = 5
# Hysteresis: stay in the hotter state until N cooler samples are seen.
_COOLING_HYSTERESIS_SAMPLES = 3
# Soft lookahead: lower for fanless (Air/base); higher for Pro/Max/Ultra and Linux.
_TEMP_SOFT_FAIR_FANLESS_CELSIUS = 78.0
_TEMP_SOFT_SERIOUS_FANLESS_CELSIUS = 88.0
_TEMP_SOFT_FAIR_COOLED_CELSIUS = 88.0
_TEMP_SOFT_SERIOUS_COOLED_CELSIUS = 95.0
_TEMP_CRITICAL_LINUX_CELSIUS = 99.0
_RISING_TREND_DELTA_CELSIUS = 1.5
_SYSFS_THERMAL = Path("/sys/class/thermal")


def soft_thresholds_celsius(cooling: CoolingProfile) -> tuple[float, float]:
    """
    Fair/Serious thresholds for temperature-based lookahead.

    Args:
        cooling: `fanless_conservative` (Air/base) or `cooled` (Pro/Linux).

    Returns:
        The pair (fair_celsius, serious_celsius).
    """
    # Fanless SoCs must throttle earlier; actively cooled ones tolerate more.
    if cooling == "cooled":
        return _TEMP_SOFT_FAIR_COOLED_CELSIUS, _TEMP_SOFT_SERIOUS_COOLED_CELSIUS
    return _TEMP_SOFT_FAIR_FANLESS_CELSIUS, _TEMP_SOFT_SERIOUS_FANLESS_CELSIUS


def state_from_temperature(
    temp_celsius: float,
    *,
    cooling: CoolingProfile = "cooled",
) -> ThermalState:
    """Map an absolute temperature onto ThermalState (Linux, without ProcessInfo)."""
    fair_celsius, serious_celsius = soft_thresholds_celsius(cooling)
    # Critical is sysfs only: Apple uses ProcessInfo.thermalState for that level.
    if temp_celsius >= _TEMP_CRITICAL_LINUX_CELSIUS:
        return ThermalState.CRITICAL
    if temp_celsius >= serious_celsius:
        return ThermalState.SERIOUS
    if temp_celsius >= fair_celsius:
        return ThermalState.FAIR
    return ThermalState.NOMINAL

STATE_LABELS = {0: "Nominal", 1: "Fair", 2: "Serious", 3: "Critical"}


class ThermalState(IntEnum):
    NOMINAL = 0
    FAIR = 1
    SERIOUS = 2
    CRITICAL = 3


_COLORS: dict[ThermalState, str] = {
    ThermalState.NOMINAL: "\033[32m",
    ThermalState.FAIR: "\033[33m",
    ThermalState.SERIOUS: "\033[38;5;208m",
    ThermalState.CRITICAL: "\033[31m",
}

# Workload-first profiles: they reduce output/iterations before introducing sleeps.
_NUM_PREDICT_FAIR = 1024
_MAX_ITER_FAIR = 4
_FAIR_PRE_PAUSE_SEC = 0.0
_FAIR_ITER_PAUSE_SEC = 1.0
_FAIR_DEGRADE_STEPS = 0

_NUM_PREDICT_SERIOUS = 640
_MAX_ITER_SERIOUS = 3
_SERIOUS_PRE_PAUSE_SEC = 2.0
_SERIOUS_ITER_PAUSE_SEC = 3.0
_SERIOUS_DEGRADE_STEPS = 1

_NUM_PREDICT_CRITICAL = 384
_MAX_ITER_CRITICAL = max(2, MAX_AGENT_ITERATIONS // 3)
_CRITICAL_ITER_PAUSE_SEC = 5.0

# (num_predict, max_iter, pre_pause, iter_pause, degradation_steps, no_thinking, wait_cooldown)
_PROFILES: dict[ThermalState, tuple[int, int, float, float, int, bool, bool]] = {
    ThermalState.NOMINAL: (
        MAX_GENERATED_TOKENS,
        MAX_AGENT_ITERATIONS,
        0.0,
        0.0,
        0,
        False,
        False,
    ),
    ThermalState.FAIR: (
        _NUM_PREDICT_FAIR,
        _MAX_ITER_FAIR,
        _FAIR_PRE_PAUSE_SEC,
        _FAIR_ITER_PAUSE_SEC,
        _FAIR_DEGRADE_STEPS,
        False,
        False,
    ),
    ThermalState.SERIOUS: (
        _NUM_PREDICT_SERIOUS,
        _MAX_ITER_SERIOUS,
        _SERIOUS_PRE_PAUSE_SEC,
        _SERIOUS_ITER_PAUSE_SEC,
        _SERIOUS_DEGRADE_STEPS,
        True,
        False,
    ),
    ThermalState.CRITICAL: (
        _NUM_PREDICT_CRITICAL,
        _MAX_ITER_CRITICAL,
        0.0,
        _CRITICAL_ITER_PAUSE_SEC,
        _FORCE_FAST_STEPS,
        True,
        True,
    ),
}


@dataclass(frozen=True)
class ThrottleParams:
    state: ThermalState
    model_alias: str
    num_predict: int
    max_iterations: int
    pre_turn_pause_sec: float
    pause_between_iterations_sec: float
    disable_thinking: bool


class ThermalTurnContext(NamedTuple):
    state: ThermalState
    model_alias: str
    thinking: bool
    throttle: ThrottleParams
    warning_message: str | None


@dataclass(frozen=True, slots=True)
class ThermalSnapshot:
    """Atomic read of the Apple state and the optional SMC temperature."""

    state: ThermalState
    temp_celsius: float | None
    read_from_monitor: bool


class _CachedReader:
    def __init__(self, ttl_sec: float, reader: Callable[[], Any]) -> None:
        self._ttl_sec = ttl_sec
        self._reader = reader
        self._entry: Any = _NOT_CACHED
        self._deadline = 0.0

    def read(self) -> Any:
        now = time.monotonic()
        if self._entry is not _NOT_CACHED and now < self._deadline:
            return self._entry
        value = self._reader()
        self._entry = value
        self._deadline = now + self._ttl_sec
        return value

    def invalidate(self) -> None:
        self._entry = _NOT_CACHED
        self._deadline = 0.0


class ThermalMonitorProcess:
    """
    Owns a single persistent Swift process for state + temperature.

    Avoids starting two Swift compilations/interpreters on every bar poll.
    """

    def __init__(
        self,
        script_path: Path = _SCRIPT_MONITOR,
        *,
        interval_ms: int = _MONITOR_INTERVAL_MS,
        popen: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    ) -> None:
        """
        Prepare the monitor without starting it.

        Args:
            script_path: Swift source of the persistent monitor.
            interval_ms: Snapshot emission period on the Swift side.
            popen: Injectable factory for tests.
        """
        self._script_path = script_path
        self._interval_ms = interval_ms
        self._popen = popen
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.RLock()
        self._last_snapshot: ThermalSnapshot | None = None
        self._reader: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def active(self) -> bool:
        """True if the child process is still running."""
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def start(self) -> bool:
        """
        Start the monitor if it is not running.

        Returns:
            True if the monitor is available after the call.
        """
        with self._lock:
            # Reuse the existing process to avoid duplicating IOKit/SMC.
            if self.active:
                return True
            # Outside macOS the monitor has no useful sensors.
            if not _on_macos() or not self._script_path.is_file():
                return False
            try:
                self._process = self._popen(
                    ["swift", str(self._script_path), str(self._interval_ms)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                )
            except OSError:
                self._process = None
                return False
            self._stop.clear()
            self._reader = threading.Thread(
                target=self._read_stdout,
                name="thermal-monitor-reader",
                daemon=True,
            )
            self._reader.start()
            return True

    def stop(self) -> None:
        """Best-effort termination of the child process and the reader thread."""
        with self._lock:
            self._stop.set()
            process = self._process
            self._process = None
        # Outside the lock, so the reader does not block on joining the lock itself.
        if process is not None:
            # SIGTERM lets the Swift loop exit and close IOKit.
            process.terminate()
            try:
                process.wait(timeout=_TIMEOUT_SUBPROCESS_SEC)
            except subprocess.TimeoutExpired:
                process.kill()
        reader = self._reader
        self._reader = None
        # The reader is a daemon: a short timeout avoids hangs at CLI shutdown.
        if reader is not None and reader.is_alive():
            reader.join(timeout=1.0)

    def snapshot(self) -> ThermalSnapshot | None:
        """
        Return the last snapshot received from the monitor.

        Returns:
            The cached snapshot, or None if the monitor has not produced data yet.
        """
        # Lazy start: the first read tries to launch the process.
        if not self.active:
            self.start()
        with self._lock:
            return self._last_snapshot

    def _read_stdout(self) -> None:
        """Consume JSON lines from the monitor while the process stays alive."""
        process = self._process
        # Without a stdout pipe there is nothing to parse.
        if process is None or process.stdout is None:
            return
        # Each line is an independent snapshot; malformed JSON is ignored.
        for line in process.stdout:
            if self._stop.is_set():
                break
            snapshot = _parse_snapshot_json(line)
            # Empty or incomplete lines must not erase the last valid value.
            if snapshot is None:
                continue
            with self._lock:
                self._last_snapshot = snapshot


class ThermalPolicyState:
    """Apply the moving average and hysteresis to the effective thermal state."""

    def __init__(self, cooling: CoolingProfile = "fanless_conservative") -> None:
        self._temperature: deque[float] = deque(maxlen=_TEMP_AVERAGE_WINDOW)
        self._policy_state = ThermalState.NOMINAL
        self._cooling_samples = 0
        self._fair_celsius, self._serious_celsius = soft_thresholds_celsius(cooling)

    def update(self, snapshot: ThermalSnapshot) -> ThermalState:
        """
        Combine the Apple state (authoritative) with soft temperature lookahead.

        Args:
            snapshot: Current state reading + optional temperature.

        Returns:
            State used by the throttling policy after hysteresis.
        """
        # Apple Serious/Critical always win over the local soft thresholds.
        apple_state = snapshot.state
        candidate_state = apple_state
        if snapshot.temp_celsius is not None:
            self._temperature.append(snapshot.temp_celsius)
            average = sum(self._temperature) / len(self._temperature)
            trend = 0.0
            # At least one pair is needed to estimate whether the temperature is rising.
            if len(self._temperature) >= 2:
                trend = self._temperature[-1] - self._temperature[0]
            # Anticipate Fair/Serious only on a rising trend, never below Apple Nominal.
            if (
                apple_state <= ThermalState.FAIR
                and average >= self._serious_celsius
                and trend >= _RISING_TREND_DELTA_CELSIUS
            ):
                candidate_state = max(candidate_state, ThermalState.SERIOUS)
            elif (
                apple_state == ThermalState.NOMINAL
                and average >= self._fair_celsius
                and trend >= _RISING_TREND_DELTA_CELSIUS
            ):
                candidate_state = max(candidate_state, ThermalState.FAIR)

        # Heating up: adopt the higher level immediately to protect the SoC.
        if candidate_state > self._policy_state:
            self._policy_state = candidate_state
            self._cooling_samples = 0
            return self._policy_state

        # Cooling down: requires more samples to avoid Fair↔Nominal flapping.
        if candidate_state < self._policy_state:
            self._cooling_samples += 1
            # Until the hysteresis is reached we stay on the hotter level.
            if self._cooling_samples < _COOLING_HYSTERESIS_SAMPLES:
                return self._policy_state
            self._policy_state = candidate_state
            self._cooling_samples = 0
            return self._policy_state

        self._cooling_samples = 0
        return self._policy_state


def _on_macos() -> bool:
    return platform.system() == "Darwin"


def _stdout_subprocess(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SUBPROCESS_SEC,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    text = result.stdout.strip()
    return text or None


def _parse_snapshot_json(line: str) -> ThermalSnapshot | None:
    """
    Parse a JSON line from the Swift monitor.

    Args:
        line: Raw stdout line.

    Returns:
        Typed snapshot, or None if the payload is unusable.
    """
    text = line.strip()
    # Empty lines are normal while the Swift runtime starts.
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    # Missing fields or wrong types indicate an incompatible protocol.
    if not isinstance(payload, dict) or "thermalState" not in payload:
        return None
    try:
        state_value = int(payload["thermalState"])
    except (TypeError, ValueError):
        return None
    state = _normalize_state(state_value)
    temp_raw = payload.get("tempC")
    temperature: float | None
    # JSON null → no temperature; numeric values are validated.
    if temp_raw is None:
        temperature = None
    else:
        try:
            temperature = float(temp_raw)
        except (TypeError, ValueError):
            temperature = None
        # Drop SMC outliers before updating the moving average and the policy.
        if (
            temperature is not None
            and (temperature <= 0 or temperature >= _TEMP_CPU_MAX_PLAUSIBLE)
        ):
            temperature = None
    return ThermalSnapshot(
        state=state,
        temp_celsius=temperature,
        read_from_monitor=True,
    )


def _normalize_state(value: int) -> ThermalState:
    """Clamp Apple's rawValue into the enum supported by the policy."""
    if value >= ThermalState.CRITICAL:
        return ThermalState.CRITICAL
    if value <= ThermalState.NOMINAL:
        return ThermalState.NOMINAL
    return ThermalState(value)


def _read_system_thermal_state() -> ThermalState:
    if not _on_macos():
        return ThermalState.NOMINAL
    output_text = _stdout_subprocess(_THERMAL_STATE_COMMAND)
    if output_text is None or not output_text.isdigit():
        return ThermalState.NOMINAL
    return _normalize_state(int(output_text))


def _darwin_oneshot() -> ThermalSnapshot | None:
    """A single JSON snapshot from the same script as the persistent monitor."""
    output_text = _stdout_subprocess(["swift", str(_SCRIPT_MONITOR), "once"])
    if output_text is None:
        return None
    parsed = _parse_snapshot_json(output_text)
    if parsed is None:
        return None
    return ThermalSnapshot(parsed.state, parsed.temp_celsius, False)


def read_sysfs_temperatures(
    *,
    base: Path = _SYSFS_THERMAL,
    read_text: Callable[[Path], str | None] | None = None,
    listdir: Callable[[Path], list[str]] | None = None,
) -> list[float]:
    """
    Read the `/sys/class/thermal/thermal_zone*/temp` zones in Celsius.

    Args:
        base: sysfs directory.
        read_text: Injectable reader; None uses Path.read_text.
        listdir: Injectable listing; None uses Path.iterdir.

    Returns:
        Plausible temperatures; an empty list if sysfs is missing.
    """
    names: list[str]
    if listdir is not None:
        names = listdir(base)
    else:
        try:
            names = [entry.name for entry in base.iterdir()]
        except OSError:
            return []
    reader = read_text
    temperature: list[float] = []
    for name in names:
        # cooling_device* is not a CPU/SoC temperature.
        if not name.startswith("thermal_zone"):
            continue
        raw: str | None
        path = base / name / "temp"
        if reader is not None:
            raw = reader(path)
        else:
            try:
                raw = path.read_text(encoding="utf-8")
            except OSError:
                raw = None
        if raw is None:
            continue
        try:
            millicelsius = int(raw.strip())
        except ValueError:
            continue
        celsius = millicelsius / 1000.0
        # Drop outliers before they enter the policy.
        if celsius <= 0 or celsius >= _TEMP_CPU_MAX_PLAUSIBLE:
            continue
        temperature.append(celsius)
    return temperature


def snapshot_linux_sysfs(
    *,
    read_text: Callable[[Path], str | None] | None = None,
    listdir: Callable[[Path], list[str]] | None = None,
) -> ThermalSnapshot:
    """Build a snapshot from the max of the sysfs zones; Nominal if there are none."""
    temperature = read_sysfs_temperatures(read_text=read_text, listdir=listdir)
    if not temperature:
        return ThermalSnapshot(ThermalState.NOMINAL, None, False)
    hottest = max(temperature)
    return ThermalSnapshot(
        state_from_temperature(hottest, cooling="cooled"),
        hottest,
        False,
    )


_monitor_process = ThermalMonitorProcess()
_policy_state = ThermalPolicyState(detect_host().cooling)
atexit.register(_monitor_process.stop)


def _read_current_snapshot() -> ThermalSnapshot:
    """
    Prefer the persistent Darwin monitor; one-shot or Linux sysfs fallback.

    Returns:
        Snapshot with state and optional temperature.
    """
    if _on_macos():
        snapshot = _monitor_process.snapshot()
        if snapshot is not None:
            return snapshot
        oneshot = _darwin_oneshot()
        if oneshot is not None:
            return oneshot
        return ThermalSnapshot(
            state=_read_system_thermal_state(),
            temp_celsius=None,
            read_from_monitor=False,
        )
    if platform.system() == "Linux":
        return snapshot_linux_sysfs()
    return ThermalSnapshot(ThermalState.NOMINAL, None, False)


def _read_policy_state() -> ThermalState:
    """Update the hysteresis and return the effective throttling state."""
    snapshot = _read_current_snapshot()
    return _policy_state.update(snapshot)


_state_cache = _CachedReader(_TTL_CACHE_SEC, _read_policy_state)
_cache_temp_cpu = _CachedReader(
    _TTL_CACHE_SEC,
    lambda: _read_current_snapshot().temp_celsius,
)


def read_thermal_state() -> ThermalState:
    return _state_cache.read()


def read_cpu_temperature_celsius() -> float | None:
    return _cache_temp_cpu.read()


def invalidate_thermal_cache() -> None:
    _state_cache.invalidate()
    _cache_temp_cpu.invalidate()


def reset_thermal_policy_for_tests() -> None:
    """
    Reset monitor and hysteresis to their initial state in unit tests.

    Must not be used by the production path: it breaks thermal continuity.
    """
    global _policy_state
    _monitor_process.stop()
    _policy_state = ThermalPolicyState()
    invalidate_thermal_cache()


def status_bar_line(
    state: ThermalState,
    temp_celsius: float | None,
    metrics: LLMTurnStats | None = None,
    *,
    ansi: bool = False,
    max_columns: int | None = None,
    power: PowerState | None = None,
    verbose: bool = False,
) -> str:
    """
    Build the thermal status line (+ metrics) for the last TTY line.

    Args:
        state: macOS thermal level (Nominal…Critical).
        temp_celsius: CPU temperature in °C, or None if unavailable.
        metrics: LLM snapshot of the turn; None omits the metrics block.
        ansi: If True, apply dim/state colours for the terminal.
        max_columns: Budget of visible columns; with metrics it enables the
            progressive layout and does not exceed this limit (ANSI sequences excluded).
        verbose: If True, the metrics segment also includes the latencies.

    Returns:
        Line ready to print on the reserved bar (possibly shortened).
    """
    label = STATE_LABELS.get(int(state), "?")
    degrees = "n/d" if temp_celsius is None else f"{temp_celsius:.0f}°C"
    if power is not None and power.low_power_mode:
        pwr = "pwr: LPM  "
    elif power is not None and power.on_battery:
        pwr = "pwr: batt  "
    elif power is not None:
        pwr = "pwr: ac  "
    else:
        pwr = ""
    prefix = f"{pwr}tmp: {label} - {degrees}"
    text = prefix

    # Add metrics only if present and there is room beyond prefix + separator
    if metrics is not None:
        # With a column cap: the prefix gets absolute priority
        if max_columns is not None:
            remainder = max_columns - cell_len(prefix) - cell_len(_METRICS_SEPARATOR)
            # Non-positive remainder: only the thermal prefix, no overflow
            if remainder > 0:
                segment = format_stats_segment(
                    metrics,
                    max_width=remainder,
                    verbose=verbose,
                )
                # Empty segment if no piece fits: avoids an orphan " · "
                if segment:
                    text = f"{prefix}{_METRICS_SEPARATOR}{segment}"
        else:
            # No cap: full string (historical behaviour)
            text = (
                f"{prefix}{_METRICS_SEPARATOR}"
                f"{format_stats_segment(metrics, verbose=verbose)}"
            )

    # Without ANSI, return plain text (tests / logs)
    if not ansi:
        return text

    color = _COLORS.get(state, "")
    # Colour only the state label; the rest stays dim
    if color:
        text = text.replace(
            label,
            f"{color}{label}{_ANSI_RESET}{_ANSI_DIM}",
            1,
        )
    return f"{_ANSI_DIM}{text}{_ANSI_RESET}"


def _effective_alias(requested_alias: str, degradation_steps: int) -> str:
    if degradation_steps <= 0:
        return requested_alias
    if degradation_steps >= _FORCE_FAST_STEPS:
        return MODEL_ALIAS_ORDER[0]
    try:
        index = MODEL_ALIAS_ORDER.index(requested_alias)
    except ValueError:
        index = MODEL_ALIAS_ORDER.index(DEFAULT_MODEL_ALIAS)
    return MODEL_ALIAS_ORDER[max(0, index - degradation_steps)]


def adaptive_pause_between_iterations(
    initial_state: ThermalState,
    base_pause_sec: float,
) -> float:
    """Re-read the thermal state only from SERIOUS up; otherwise keep the base pause."""
    if initial_state < ThermalState.SERIOUS:
        # In nominal/fair the background monitor is enough: no synchronous probe.
        return base_pause_sec
    invalidate_thermal_cache()
    current_state = read_thermal_state()
    if current_state <= initial_state:
        return base_pause_sec
    profile_pause = _PROFILES[current_state][3]
    return max(base_pause_sec, profile_pause)


def compute_throttle(state: ThermalState, model_alias: str) -> ThrottleParams:
    prof = _PROFILES[state]
    return ThrottleParams(
        state=state,
        model_alias=_effective_alias(model_alias, prof[4]),
        num_predict=prof[0],
        max_iterations=prof[1],
        pre_turn_pause_sec=prof[2],
        pause_between_iterations_sec=prof[3],
        disable_thinking=prof[5],
    )


def wait_for_cooling(
    target_state: ThermalState = ThermalState.FAIR,
    timeout_seconds: float = 60.0,
) -> ThermalState:
    start_time = time.monotonic()
    while True:
        _state_cache.invalidate()
        current_state = _state_cache.read()
        if current_state <= target_state:
            return current_state
        if time.monotonic() - start_time >= timeout_seconds:
            return current_state
        time.sleep(_POLLING_INTERVAL_SEC)


def prepare_turn(config: ConfigRuntime, ui: ConsoleUI) -> ThermalTurnContext | None:
    if not config.burn_protection:
        return None

    state = read_thermal_state()
    prof = _PROFILES[state]

    if prof[6]:
        label = STATE_LABELS.get(int(state), "Unknown")
        ui.print_thermal_warning(f"Temperatura {label}: attesa raffreddamento...")
        state = wait_for_cooling()
        prof = _PROFILES[state]

    throttle = compute_throttle(state, config.model)
    thinking = not throttle.disable_thinking
    warning_message = None
    if state != ThermalState.NOMINAL:
        label = STATE_LABELS.get(int(state), "Unknown")
        parts = [f"Temperatura {label}"]
        # Workload-first: report the output/iteration cuts first, then the pauses.
        if throttle.num_predict < MAX_GENERATED_TOKENS:
            parts.append(f"num_predict {throttle.num_predict}")
        if throttle.max_iterations < MAX_AGENT_ITERATIONS:
            parts.append(f"max_iter {throttle.max_iterations}")
        if throttle.model_alias != config.model:
            parts.append(f"modello {throttle.model_alias}")
        if throttle.disable_thinking:
            parts.append("thinking disattivato")
        if throttle.pre_turn_pause_sec > 0:
            parts.append(f"pausa {int(throttle.pre_turn_pause_sec)}s")
        warning_message = (
            parts[0] if len(parts) == 1 else f"{parts[0]} — {', '.join(parts[1:])}"
        )
        ui.print_thermal_warning(warning_message)

    if throttle.pre_turn_pause_sec > 0:
        time.sleep(throttle.pre_turn_pause_sec)

    return ThermalTurnContext(
        state,
        throttle.model_alias,
        thinking,
        throttle,
        warning_message,
    )
