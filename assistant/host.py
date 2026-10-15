"""Stdlib-only host detection: OS, SoC, memory and Ollama options."""

from __future__ import annotations

import os
import platform
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

GIBIBYTE = 1024**3
# Threshold above which the heavy routes' context can be widened.
AMPLE_MEMORY_BYTES = 32 * GIBIBYTE
# Session keep-alive: close() unloads the weights on exit.
DEFAULT_KEEP_ALIVE = "30m"
_POWER_CACHE_TTL_SEC = 60.0
_PMSET_TIMEOUT_SEC = 2.0
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
_SYSCTL_TIMEOUT_SEC = 2.0
_NVIDIA_SMI_TIMEOUT_SEC = 2.0

HostOs = Literal["darwin", "linux", "windows", "unknown"]
ChipFamily = Literal[
    "m1",
    "m1_pro",
    "m1_max",
    "m1_ultra",
    "m2",
    "m2_pro",
    "m2_max",
    "m2_ultra",
    "m3",
    "m3_pro",
    "m3_max",
    "m4",
    "m4_pro",
    "m4_max",
    "intel",
    "linux_cpu",
    "linux_nvidia",
    "unknown",
]
CoolingProfile = Literal["fanless_conservative", "cooled"]

# More specific matches before the base chip, so "M4 Pro" does not become `m4`.
_CHIP_PATTERNS: tuple[tuple[ChipFamily, str], ...] = (
    ("m4_max", "m4 max"),
    ("m4_pro", "m4 pro"),
    ("m4", "m4"),
    ("m3_max", "m3 max"),
    ("m3_pro", "m3 pro"),
    ("m3", "m3"),
    ("m2_ultra", "m2 ultra"),
    ("m2_max", "m2 max"),
    ("m2_pro", "m2 pro"),
    ("m2", "m2"),
    ("m1_ultra", "m1 ultra"),
    ("m1_max", "m1 max"),
    ("m1_pro", "m1 pro"),
    ("m1", "m1"),
)
_APPLE_COOLED_FAMILIES: frozenset[ChipFamily] = frozenset(
    {
        "m1_pro",
        "m1_max",
        "m1_ultra",
        "m2_pro",
        "m2_max",
        "m2_ultra",
        "m3_pro",
        "m3_max",
        "m4_pro",
        "m4_max",
    }
)


@dataclass(frozen=True, slots=True)
class HostCapabilities:
    """
    Immutable snapshot of the resources relevant to inference.

    Attributes:
        os: Normalized system family.
        arch: Raw `platform.machine()` (`arm64`, `x86_64`, `aarch64`).
        chip_family: SoC or accelerator class, already classified.
        chip_brand: Original probe string (sysctl or empty).
        memory_bytes: Physical system RAM; 0 if unknown.
        inference_memory_bytes: Pool holding the weights (unified RAM or VRAM).
        performance_cores: Apple P-cores, otherwise 0 if unknown.
        has_discrete_gpu: True only for GPUs with their own VRAM (not unified Metal).
        cooling: Conservative vs actively cooled thermal profile.
    """

    os: HostOs
    arch: str
    chip_family: ChipFamily
    chip_brand: str
    memory_bytes: int
    inference_memory_bytes: int
    performance_cores: int
    has_discrete_gpu: bool
    cooling: CoolingProfile


@dataclass(frozen=True, slots=True)
class PowerState:
    """Current power source: battery and Low Power Mode."""

    on_battery: bool = False
    low_power_mode: bool = False

    def key(self) -> Literal["lpm", "battery", "ac"]:
        """Stable key for one-shot warnings and state comparisons."""
        if self.low_power_mode:
            return "lpm"
        if self.on_battery:
            return "battery"
        return "ac"


@dataclass(frozen=True, slots=True)
class InferenceHostOptions:
    """Ollama client/server options derived from the host, independent of LangChain."""

    num_gpu: int | None
    num_thread: int | None
    keep_alive: str
    kv_cache_type: str
    flash_attention: bool
    max_loaded_models: int
    num_parallel: int


@dataclass(frozen=True, slots=True)
class HostProbes:
    """
    Injectable collaborators: tests do not run the real sysctl or nvidia-smi.

    Each callable returns None/0 when the data is not available, so detection
    degrades to a conservative profile instead of raising exceptions.
    """

    system: Callable[[], str] = platform.system
    machine: Callable[[], str] = platform.machine
    sysctl: Callable[[str], str | None] = lambda name: _read_sysctl(name)
    read_text: Callable[[Path], str | None] = lambda path: _read_text(path)
    nvidia_smi_output: Callable[[], str | None] = lambda: _read_nvidia_smi()
    cpu_count: Callable[[], int | None] = os.cpu_count
    posix_memory_bytes: Callable[[], int] = lambda: _posix_sysconf_memory()
    run_command: Callable[[list[str]], str | None] = lambda argv: _run_command(argv)
    list_dir: Callable[[Path], list[str] | None] = lambda path: _list_dir(path)


def normalize_os_name(system_name: str) -> HostOs:
    """
    Normalize `platform.system()` into the three supported families.

    Args:
        system_name: Raw value (`Darwin`, `Linux`, `Windows`, other).

    Returns:
        Lowercase identifier, or `unknown` for out-of-scope OSes.
    """
    normalized = system_name.strip().casefold()
    # Darwin covers macOS and the Apple derivatives used by the sysctl probes.
    if normalized == "darwin":
        return "darwin"
    # Linux is the second first-class target of the portability plan.
    if normalized == "linux":
        return "linux"
    # Windows stays a graceful degradation: no wizard and no thermal handling.
    if normalized in {"windows", "win32"}:
        return "windows"
    return "unknown"


def parse_apple_chip_family(brand: str) -> ChipFamily | None:
    """
    Classify an Apple `machdep.cpu.brand_string`.

    Args:
        brand: E.g. `Apple M4 Pro`; empty or Intel strings return None.

    Returns:
        SoC family, or None if the brand is not an Apple chip.
    """
    lowered = brand.strip().casefold()
    # Without text we do not invent an M-series: the caller will use arch/OS.
    if not lowered:
        return None
    for family, needle in _CHIP_PATTERNS:
        # Patterns are ordered from most specific: "m4 pro" before "m4".
        if needle in lowered:
            return family
    return None


def parse_meminfo_bytes(contents: str) -> int:
    """
    Extract `MemTotal` from `/proc/meminfo`.

    Args:
        contents: Full text of the proc file.

    Returns:
        RAM bytes, or 0 if the line is missing or not numeric.
    """
    for raw_line in contents.splitlines():
        # Only MemTotal describes the physical RAM; MemAvailable is a runtime leftover.
        if not raw_line.startswith("MemTotal:"):
            continue
        parts = raw_line.split()
        # Expected format: MemTotal: <kB> kB
        if len(parts) < 2:
            return 0
        try:
            kilobytes = int(parts[1])
        except ValueError:
            return 0
        # proc's kB is 1024 bytes; non-positive values are garbage.
        if kilobytes <= 0:
            return 0
        return kilobytes * 1024
    return 0


def parse_nvidia_smi_mib(output: str) -> int:
    """
    Parse `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits`.

    Args:
        output: One line per GPU, value in MiB.

    Returns:
        Bytes of the GPU with the most VRAM (Ollama usually uses one device).
    """
    max_mib = 0
    for raw_line in output.splitlines():
        token = raw_line.strip().split()[0] if raw_line.strip() else ""
        # Empty lines or leftover headers are not measurements.
        if not token:
            continue
        try:
            mebibytes = int(float(token))
        except ValueError:
            continue
        # Keep the largest GPU: summing would overestimate a single process.
        if mebibytes > max_mib:
            max_mib = mebibytes
    return max_mib * (1024**2) if max_mib > 0 else 0


def ollama_base_url(environ: Mapping[str, str] | None = None) -> str:
    """
    Resolve the Ollama daemon URL from `OLLAMA_HOST`.

    Args:
        environ: Injectable mapping; None uses `os.environ`.

    Returns:
        URL with a scheme and no trailing slash. Default `http://127.0.0.1:11434`.
    """
    env = os.environ if environ is None else environ
    raw = str(env.get("OLLAMA_HOST") or DEFAULT_OLLAMA_BASE_URL).strip()
    # Empty or whitespace-only string: fall back to the historical local default.
    if not raw:
        raw = DEFAULT_OLLAMA_BASE_URL
    # Ollama accepts `127.0.0.1:11434` without a scheme; httpx requires one.
    if "://" not in raw:
        raw = f"http://{raw}"
    return raw.rstrip("/")


def physical_memory_bytes(probes: HostProbes | None = None) -> int:
    """
    Return the system's physical RAM.

    Args:
        probes: Injectable probes; None uses the real ones.

    Returns:
        RAM bytes, or 0 if no probe works.
    """
    return detect_host(probes).memory_bytes


def detect_host(probes: HostProbes | None = None) -> HostCapabilities:
    """
    Detect OS, architecture, SoC, RAM, VRAM and cooling profile.

    Args:
        probes: Injectable collaborators for tests.

    Returns:
        Conservative snapshot: unknown fields stay 0 / `unknown`.
    """
    active = probes or HostProbes()
    os_name = normalize_os_name(active.system())
    arch = (active.machine() or "").strip() or "unknown"
    brand = ""
    chip_family: ChipFamily = "unknown"
    memory_bytes = 0
    inference_memory_bytes = 0
    performance_cores = 0
    has_discrete_gpu = False
    cooling: CoolingProfile = "cooled"

    # Darwin: sysctl is the authoritative source for chip, unified RAM and P-cores.
    if os_name == "darwin":
        brand = active.sysctl("machdep.cpu.brand_string") or ""
        parsed = parse_apple_chip_family(brand)
        if parsed is not None:
            chip_family = parsed
        elif arch in {"x86_64", "i386"}:
            # Intel Mac: no first-class unified Metal.
            chip_family = "intel"
        memory_bytes = _parse_positive_int(active.sysctl("hw.memsize"))
        # sysctl missing (VM, sandbox): POSIX fallback identical to the historical code.
        if memory_bytes <= 0:
            memory_bytes = active.posix_memory_bytes()
        inference_memory_bytes = memory_bytes
        performance_cores = _parse_positive_int(
            active.sysctl("hw.perflevel0.physicalcpu")
        )
        cooling = (
            "cooled"
            if chip_family in _APPLE_COOLED_FAMILIES or chip_family == "intel"
            else "fanless_conservative"
        )
        return HostCapabilities(
            os=os_name,
            arch=arch,
            chip_family=chip_family,
            chip_brand=brand,
            memory_bytes=memory_bytes,
            inference_memory_bytes=inference_memory_bytes,
            performance_cores=performance_cores,
            has_discrete_gpu=False,
            cooling=cooling,
        )

    # Linux: RAM from proc, optional VRAM from nvidia-smi, cooled thermal profile.
    if os_name == "linux":
        meminfo = active.read_text(Path("/proc/meminfo"))
        memory_bytes = parse_meminfo_bytes(meminfo) if meminfo else 0
        if memory_bytes <= 0:
            memory_bytes = active.posix_memory_bytes()
        vram_bytes = parse_nvidia_smi_mib(active.nvidia_smi_output() or "")
        # Discrete GPU: the weights live in VRAM and do not compete with the OS for RAM.
        if vram_bytes > 0:
            has_discrete_gpu = True
            inference_memory_bytes = vram_bytes
            chip_family = "linux_nvidia"
        else:
            inference_memory_bytes = memory_bytes
            chip_family = "linux_cpu"
        cpu_count = active.cpu_count()
        performance_cores = cpu_count if isinstance(cpu_count, int) and cpu_count > 0 else 0
        return HostCapabilities(
            os=os_name,
            arch=arch,
            chip_family=chip_family,
            chip_brand=brand,
            memory_bytes=memory_bytes,
            inference_memory_bytes=inference_memory_bytes,
            performance_cores=performance_cores,
            has_discrete_gpu=has_discrete_gpu,
            cooling="cooled",
        )

    # Windows and unknown OSes: POSIX fallback only, no GPU declared.
    memory_bytes = active.posix_memory_bytes()
    return HostCapabilities(
        os=os_name,
        arch=arch,
        chip_family="unknown",
        chip_brand=brand,
        memory_bytes=memory_bytes,
        inference_memory_bytes=memory_bytes,
        performance_cores=0,
        has_discrete_gpu=False,
        cooling="cooled",
    )


def inference_host_options(host: HostCapabilities) -> InferenceHostOptions:
    """
    Derive the Ollama options from the host snapshot.

    Args:
        host: Already detected capabilities.

    Returns:
        Client options (`num_gpu`/`num_thread`/`keep_alive`) and server advice.
    """
    num_gpu: int | None
    # Apple Silicon: every layer on Metal. Intel Mac: do not force the GPU.
    if host.os == "darwin" and host.chip_family not in {"intel", "unknown"}:
        num_gpu = -1
    elif host.os == "darwin":
        num_gpu = None
    # Linux NVIDIA: full offload. CPU-only: zero GPU layers.
    elif host.has_discrete_gpu:
        num_gpu = -1
    elif host.os == "linux":
        num_gpu = 0
    else:
        num_gpu = None

    num_thread: int | None = None
    # Metal: A/B iter2, Ollama auto (None) beats pinning to the 4 P-cores on the 9B.
    # Without a GPU, the P-cores avoid the E-cores that slow down CPU decode.
    if host.os == "darwin" and host.performance_cores > 0 and num_gpu != -1:
        num_thread = host.performance_cores

    # Session keep-alive: 30m on every class; close() unloads on exit.
    return InferenceHostOptions(
        num_gpu=num_gpu,
        num_thread=num_thread,
        keep_alive=DEFAULT_KEEP_ALIVE,
        kv_cache_type="q4_0",
        flash_attention=True,
        max_loaded_models=1,
        num_parallel=1,
    )


def neutral_host_options() -> InferenceHostOptions:
    """Options identical to the historical runtime, used by tests without residency."""
    return InferenceHostOptions(
        num_gpu=None,
        num_thread=None,
        keep_alive=DEFAULT_KEEP_ALIVE,
        kv_cache_type="q4_0",
        flash_attention=True,
        max_loaded_models=1,
        num_parallel=1,
    )


def can_expand_context(host: HostCapabilities) -> bool:
    """
    True for a Pro/Max/Ultra SoC or an inference pool ≥ 32 GiB.

    Args:
        host: Host snapshot. Without known memory there is no widening.

    Returns:
        False on the base-M 16 GB profile, so the historical budgets stay identical.
    """
    if host.chip_family in _APPLE_COOLED_FAMILIES:
        return True
    pool = host.inference_memory_bytes or host.memory_bytes
    return pool >= AMPLE_MEMORY_BYTES


def ollama_server_environ(
    base: Mapping[str, str],
    options: InferenceHostOptions,
) -> dict[str, str]:
    """
    Copy the environment and set the Ollama variables only for a spawned `ollama serve`.

    Args:
        base: Parent process environment.
        options: Policy derived from the host.

    Returns:
        New dict to pass to `Popen`. Does not mutate `base`.
    """
    env = dict(base)
    # Flash attention reduces the KV on Metal/CUDA; it belongs to the server process, not the client.
    if options.flash_attention:
        env["OLLAMA_FLASH_ATTENTION"] = "1"
    env["OLLAMA_KV_CACHE_TYPE"] = options.kv_cache_type
    env["OLLAMA_KEEP_ALIVE"] = options.keep_alive
    env["OLLAMA_MAX_LOADED_MODELS"] = str(options.max_loaded_models)
    env["OLLAMA_NUM_PARALLEL"] = str(options.num_parallel)
    return env


def processor_looks_cpu_only(processor: str) -> bool:
    """
    True if `/api/ps` describes a model resident on CPU only.

    Args:
        processor: Ollama's `processor` field (e.g. `100% GPU`, `CPU`).

    Returns:
        False if the field is empty or mentions a GPU: not a useful diagnosis.
    """
    text = processor.strip().casefold()
    if not text:
        return False
    # "100% GPU" and "GPU/CPU" are not a CPU fallback worth reporting.
    if "gpu" in text:
        return False
    return "cpu" in text


_power_cache: tuple[float, PowerState] | None = None


def reset_power_cache_for_tests() -> None:
    """Clear the power cache; tests only."""
    global _power_cache
    _power_cache = None


def parse_pmset_on_battery(output: str) -> bool:
    """True if `pmset -g batt` reports battery power."""
    lowered = output.casefold()
    if "ac power" in lowered and "now drawing from 'ac power'" in lowered:
        return False
    if "now drawing from 'battery power'" in lowered:
        return True
    lines = lowered.splitlines()
    first = lines[0] if lines else ""
    if not first:
        return False
    return "battery power" in lowered and "ac power" not in first


def parse_pmset_low_power_mode(output: str) -> bool:
    """True if `pmset -g` has `lowpowermode 1`."""
    for raw_line in output.splitlines():
        line = raw_line.strip().casefold()
        if line.startswith("lowpowermode"):
            return line.split()[-1] == "1"
    return False


def parse_linux_on_battery(
    statuses: list[str],
) -> bool:
    """True if at least one battery is discharging and none is charging."""
    normalized = [item.strip().casefold() for item in statuses if item.strip()]
    if any(item == "charging" for item in normalized):
        return False
    return any(item == "discharging" for item in normalized)


def detect_power(probes: HostProbes | None = None) -> PowerState:
    """
    Detect battery and Low Power Mode with a short cache.

    Args:
        probes: Injectable collaborators; None uses the real ones.

    Returns:
        Conservative snapshot: False/False if the probes fail.
    """
    global _power_cache
    now = time.monotonic()
    if probes is None and _power_cache is not None:
        expires_at, cached = _power_cache
        if now < expires_at:
            return cached
    active = probes or HostProbes()
    os_name = normalize_os_name(active.system())
    on_battery = False
    low_power_mode = False
    if os_name == "darwin":
        batt = active.run_command(["pmset", "-g", "batt"]) or ""
        settings = active.run_command(["pmset", "-g"]) or ""
        on_battery = parse_pmset_on_battery(batt)
        low_power_mode = parse_pmset_low_power_mode(settings)
    elif os_name == "linux":
        on_battery, low_power_mode = _linux_power_state(active)
    state = PowerState(on_battery=on_battery, low_power_mode=low_power_mode)
    if probes is None:
        _power_cache = (now + _POWER_CACHE_TTL_SEC, state)
    return state


def _linux_power_state(probes: HostProbes) -> tuple[bool, bool]:
    """Battery via sysfs; Low Power from platform_profile if present."""
    supply_root = Path("/sys/class/power_supply")
    names = probes.list_dir(supply_root) or []
    statuses: list[str] = []
    for name in names:
        lowered = name.casefold()
        if not lowered.startswith("bat"):
            continue
        raw = probes.read_text(supply_root / name / "status")
        if raw:
            statuses.append(raw)
    on_battery = parse_linux_on_battery(statuses)
    profile = probes.read_text(Path("/sys/firmware/acpi/platform_profile")) or ""
    lowered_profile = profile.strip().casefold()
    low_power_mode = lowered_profile in {"low-power", "low_power", "quiet", "power-saver"}
    return on_battery, low_power_mode


def _run_command(argv: list[str]) -> str | None:
    """Run a short command; None if missing, timed out or exit ≠ 0."""
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_PMSET_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    text = result.stdout.strip()
    return text or None


def pid_daemon_ollama() -> int | None:
    """
    PID of the `ollama` daemon process, excluding this interpreter.

    `pgrep -x` avoids matching the agent's command line. Among several PIDs
    the one whose cmdline contains `serve` is preferred.
    """
    raw = _run_command(["pgrep", "-x", "ollama"])
    if raw is None:
        return None
    self_pid = os.getpid()
    candidates: list[tuple[int, str]] = []
    for line in raw.splitlines():
        token = line.strip()
        if not token.isdigit():
            continue
        pid = int(token)
        if pid == self_pid:
            continue
        command = _run_command(["ps", "-o", "command=", "-p", str(pid)]) or ""
        if not _cmdline_looks_like_ollama_daemon(command):
            continue
        candidates.append((pid, command))
    for pid, command in candidates:
        if "serve" in command.casefold():
            return pid
    if candidates:
        return candidates[0][0]
    return None


def environ_daemon_ollama(pid: int) -> dict[str, str] | None:
    """Best-effort: the daemon's env; None if the process is not visible."""
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        raw = b""
    if raw:
        return _parse_proc_environ(raw)
    stdout = _run_command(["ps", "eww", "-p", str(pid)])
    if stdout is None:
        return None
    env = _parse_ps_eww_env(stdout)
    return env or None


def _cmdline_looks_like_ollama_daemon(command: str) -> bool:
    """True if the cmdline is the ollama binary, not the agent or pytest."""
    lowered = command.casefold()
    if not lowered.strip():
        return False
    if "assistant" in lowered or "pytest" in lowered or "python" in lowered:
        return False
    if "serve" in lowered:
        return True
    tokens = command.split()
    return bool(tokens) and Path(tokens[0]).name.casefold() == "ollama"


def _parse_proc_environ(raw: bytes) -> dict[str, str]:
    """Decode `/proc/<pid>/environ` into key/value pairs."""
    env: dict[str, str] = {}
    for pair in raw.split(b"\x00"):
        if b"=" not in pair:
            continue
        key, _, value = pair.partition(b"=")
        env[key.decode("utf-8", errors="replace")] = value.decode(
            "utf-8",
            errors="replace",
        )
    return env


def _parse_ps_eww_env(stdout: str) -> dict[str, str]:
    """Extract OLLAMA_* pairs from `ps eww` (best-effort, split on whitespace)."""
    env: dict[str, str] = {}
    for token in stdout.replace("\n", " ").split():
        if not token.startswith("OLLAMA_") or "=" not in token:
            continue
        key, _, value = token.partition("=")
        env[key] = value
    return env


def _list_dir(path: Path) -> list[str] | None:
    """List the names in a directory; None if missing."""
    try:
        return [entry.name for entry in path.iterdir()]
    except OSError:
        return None


def _parse_positive_int(raw: str | None) -> int:
    """Turn a numeric string into a positive integer; 0 if missing or invalid."""
    if raw is None:
        return 0
    try:
        value = int(raw.strip())
    except (TypeError, ValueError):
        return 0
    return value if value > 0 else 0


def _read_sysctl(name: str) -> str | None:
    """Run `sysctl -n`; None if the binary or the key is missing."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", name],
            capture_output=True,
            text=True,
            timeout=_SYSCTL_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    # Missing keys exit != 0; that is not an error to propagate.
    if result.returncode != 0:
        return None
    text = result.stdout.strip()
    return text or None


def _read_text(path: Path) -> str | None:
    """Read a text file; None if the file is missing or cannot be decoded."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _read_nvidia_smi() -> str | None:
    """Query nvidia-smi; None if missing, timed out or failing."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    text = result.stdout.strip()
    return text or None


def _posix_sysconf_memory() -> int:
    """
    RAM via `SC_PAGE_SIZE` × `SC_PHYS_PAGES`.

    Returns:
        Bytes, or 0 on platforms without sysconf (historical behaviour).
    """
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        # Platforms without sysconf will continue with an explicit profile.
        return 0
    return page_size * page_count
