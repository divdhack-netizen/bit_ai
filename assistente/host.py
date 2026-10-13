"""Rilevamento host stdlib-only: OS, SoC, memoria e opzioni Ollama."""

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
# Soglia oltre la quale si può allargare il contesto delle route pesanti.
AMPLE_MEMORY_BYTES = 32 * GIBIBYTE
# Keep-alive di sessione: close() scarica i pesi all'uscita.
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

# Match più specifici prima del chip base, così "M4 Pro" non diventa `m4`.
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
    Snapshot immutabile delle risorse utili all'inferenza.

    Attributes:
        os: Famiglia di sistema normalizzata.
        arch: `platform.machine()` grezzo (`arm64`, `x86_64`, `aarch64`).
        chip_family: SoC o classe acceleratore già classificata.
        chip_brand: Stringa originale del probe (sysctl o vuota).
        memory_bytes: RAM fisica di sistema; 0 se sconosciuta.
        inference_memory_bytes: Pool in cui stanno i pesi (unified RAM o VRAM).
        performance_cores: P-core Apple, altrimenti 0 se ignoti.
        has_discrete_gpu: True solo per GPU con VRAM propria (non Metal unificato).
        cooling: Profilo termico conservativo vs raffreddato.
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
    """Alimentazione corrente: batteria e Low Power Mode."""

    on_battery: bool = False
    low_power_mode: bool = False

    def chiave(self) -> Literal["lpm", "battery", "ac"]:
        """Chiave stabile per avvisi one-shot e confronti di stato."""
        if self.low_power_mode:
            return "lpm"
        if self.on_battery:
            return "battery"
        return "ac"


@dataclass(frozen=True, slots=True)
class InferenceHostOptions:
    """Opzioni client/server Ollama derivate dal host, senza dipendere da LangChain."""

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
    Collaboratori iniettabili: i test non eseguono sysctl né nvidia-smi reali.

    Ogni callable restituisce None/0 quando il dato non è disponibile, così il
    rilevamento degrada a un profilo conservativo invece di alzare eccezioni.
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
    Normalizza `platform.system()` nelle tre famiglie supportite.

    Args:
        system_name: Valore grezzo (`Darwin`, `Linux`, `Windows`, altro).

    Returns:
        Identificatore minuscolo, oppure `unknown` per OS fuori scope.
    """
    normalized = system_name.strip().casefold()
    # Darwin copre macOS e i derivati Apple usati dai probe sysctl.
    if normalized == "darwin":
        return "darwin"
    # Linux è il secondo target di prima classe del piano di portabilità.
    if normalized == "linux":
        return "linux"
    # Windows resta degrado grazioso: niente wizard né termica.
    if normalized in {"windows", "win32"}:
        return "windows"
    return "unknown"


def parse_apple_chip_family(brand: str) -> ChipFamily | None:
    """
    Classifica un `machdep.cpu.brand_string` Apple.

    Args:
        brand: Es. `Apple M4 Pro`; stringhe vuote o Intel restituiscono None.

    Returns:
        Famiglia SoC, oppure None se il brand non è un chip Apple.
    """
    lowered = brand.strip().casefold()
    # Senza testo non inventiamo un M-series: il caller userà arch/OS.
    if not lowered:
        return None
    for family, needle in _CHIP_PATTERNS:
        # I pattern sono ordinati dal più specifico: "m4 pro" prima di "m4".
        if needle in lowered:
            return family
    return None


def parse_meminfo_bytes(contents: str) -> int:
    """
    Estrae `MemTotal` da `/proc/meminfo`.

    Args:
        contents: Testo completo del file proc.

    Returns:
        Byte di RAM, oppure 0 se la riga manca o non è numerica.
    """
    for raw_line in contents.splitlines():
        # Solo MemTotal descrive la RAM fisica; MemAvailable è un residuo runtime.
        if not raw_line.startswith("MemTotal:"):
            continue
        parts = raw_line.split()
        # Formato atteso: MemTotal: <kB> kB
        if len(parts) < 2:
            return 0
        try:
            kilobytes = int(parts[1])
        except ValueError:
            return 0
        # kB di proc è 1024 byte; valori non positivi sono spazzatura.
        if kilobytes <= 0:
            return 0
        return kilobytes * 1024
    return 0


def parse_nvidia_smi_mib(output: str) -> int:
    """
    Interpreta `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits`.

    Args:
        output: Una riga per GPU, valore in MiB.

    Returns:
        Byte della GPU con più VRAM (Ollama usa di solito un dispositivo).
    """
    maxima_mib = 0
    for raw_line in output.splitlines():
        token = raw_line.strip().split()[0] if raw_line.strip() else ""
        # Righe vuote o header residui non sono misure.
        if not token:
            continue
        try:
            mebibytes = int(float(token))
        except ValueError:
            continue
        # Tiene la GPU più capiente: sommare over-stimerebbe un unico processo.
        if mebibytes > maxima_mib:
            maxima_mib = mebibytes
    return maxima_mib * (1024**2) if maxima_mib > 0 else 0


def ollama_base_url(environ: Mapping[str, str] | None = None) -> str:
    """
    Risolve l'URL del daemon Ollama da `OLLAMA_HOST`.

    Args:
        environ: Mapping iniettabile; None usa `os.environ`.

    Returns:
        URL con schema e senza slash finale. Default `http://127.0.0.1:11434`.
    """
    env = os.environ if environ is None else environ
    raw = str(env.get("OLLAMA_HOST") or DEFAULT_OLLAMA_BASE_URL).strip()
    # Stringa vuota o solo spazi: torna al default locale storico.
    if not raw:
        raw = DEFAULT_OLLAMA_BASE_URL
    # Ollama accetta `127.0.0.1:11434` senza schema; httpx richiede lo schema.
    if "://" not in raw:
        raw = f"http://{raw}"
    return raw.rstrip("/")


def physical_memory_bytes(probes: HostProbes | None = None) -> int:
    """
    Restituisce la RAM fisica del sistema.

    Args:
        probes: Probe iniettabili; None usa quelli reali.

    Returns:
        Byte di RAM, oppure 0 se nessun probe funziona.
    """
    return detect_host(probes).memory_bytes


def detect_host(probes: HostProbes | None = None) -> HostCapabilities:
    """
    Rileva OS, architettura, SoC, RAM, VRAM e profilo di raffreddamento.

    Args:
        probes: Collaboratori iniettabili per i test.

    Returns:
        Snapshot conservativo: campi ignoti restano 0 / `unknown`.
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

    # Darwin: sysctl è la fonte autorevole per chip, RAM unificata e P-core.
    if os_name == "darwin":
        brand = active.sysctl("machdep.cpu.brand_string") or ""
        parsed = parse_apple_chip_family(brand)
        if parsed is not None:
            chip_family = parsed
        elif arch in {"x86_64", "i386"}:
            # Intel Mac: niente Metal unificato di prima classe.
            chip_family = "intel"
        memory_bytes = _parse_positive_int(active.sysctl("hw.memsize"))
        # sysctl assente (VM, sandbox): fallback POSIX identico al codice storico.
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

    # Linux: RAM da proc, VRAM opzionale da nvidia-smi, termica raffreddata.
    if os_name == "linux":
        meminfo = active.read_text(Path("/proc/meminfo"))
        memory_bytes = parse_meminfo_bytes(meminfo) if meminfo else 0
        if memory_bytes <= 0:
            memory_bytes = active.posix_memory_bytes()
        vram_bytes = parse_nvidia_smi_mib(active.nvidia_smi_output() or "")
        # GPU discreta: i pesi stanno in VRAM, non competono con l'OS sulla RAM.
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

    # Windows e OS sconosciuti: solo fallback POSIX, nessuna GPU dichiarata.
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
    Deriva le opzioni Ollama dal snapshot host.

    Args:
        host: Capacità già rilevate.

    Returns:
        Opzioni client (`num_gpu`/`num_thread`/`keep_alive`) e consigli server.
    """
    num_gpu: int | None
    # Apple Silicon: tutti i layer su Metal. Intel Mac: non forzare GPU.
    if host.os == "darwin" and host.chip_family not in {"intel", "unknown"}:
        num_gpu = -1
    elif host.os == "darwin":
        num_gpu = None
    # Linux NVIDIA: offload completo. CPU-only: zero layer GPU.
    elif host.has_discrete_gpu:
        num_gpu = -1
    elif host.os == "linux":
        num_gpu = 0
    else:
        num_gpu = None

    num_thread: int | None = None
    # Metal: A/B iter2, Ollama auto (None) batte il pin ai 4 P-core sul 9B.
    # Senza GPU, i P-core evitano gli E-core che peggiorano il decode CPU.
    if host.os == "darwin" and host.performance_cores > 0 and num_gpu != -1:
        num_thread = host.performance_cores

    # Keep-alive di sessione: 30m su tutte le classi; close() scarica all'uscita.
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
    """Opzioni identiche al runtime storico, usate dai test senza residency."""
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
    True se SoC Pro/Max/Ultra oppure pool di inferenza ≥ 32 GiB.

    Args:
        host: Snapshot host. Senza memoria nota non si amplia.

    Returns:
        False sul profilo 16 GB M-base, così i budget storici restano identici.
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
    Copia l'ambiente e imposta le variabili Ollama solo per un `ollama serve` spawnato.

    Args:
        base: Ambiente del processo padre.
        options: Policy derivata dal host.

    Returns:
        Nuovo dict da passare a `Popen`. Non muta `base`.
    """
    env = dict(base)
    # Flash attention riduce la KV su Metal/CUDA; va nel processo server, non nel client.
    if options.flash_attention:
        env["OLLAMA_FLASH_ATTENTION"] = "1"
    env["OLLAMA_KV_CACHE_TYPE"] = options.kv_cache_type
    env["OLLAMA_KEEP_ALIVE"] = options.keep_alive
    env["OLLAMA_MAX_LOADED_MODELS"] = str(options.max_loaded_models)
    env["OLLAMA_NUM_PARALLEL"] = str(options.num_parallel)
    return env


def processor_looks_cpu_only(processor: str) -> bool:
    """
    True se `/api/ps` descrive un modello residente solo su CPU.

    Args:
        processor: Campo `processor` di Ollama (es. `100% GPU`, `CPU`).

    Returns:
        False se il campo è vuoto o menziona una GPU: non è una diagnosi utile.
    """
    text = processor.strip().casefold()
    if not text:
        return False
    # "100% GPU" e "GPU/CPU" non sono un fallback CPU da segnalare.
    if "gpu" in text:
        return False
    return "cpu" in text


_power_cache: tuple[float, PowerState] | None = None


def reset_power_cache_for_tests() -> None:
    """Svuota la cache alimentazione; solo per i test."""
    global _power_cache
    _power_cache = None


def parse_pmset_on_battery(output: str) -> bool:
    """True se `pmset -g batt` indica alimentazione a batteria."""
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
    """True se `pmset -g` ha `lowpowermode 1`."""
    for raw_line in output.splitlines():
        line = raw_line.strip().casefold()
        if line.startswith("lowpowermode"):
            return line.split()[-1] == "1"
    return False


def parse_linux_on_battery(
    statuses: list[str],
) -> bool:
    """True se almeno una batteria è in scarica e nessuna è in carica."""
    normalized = [item.strip().casefold() for item in statuses if item.strip()]
    if any(item == "charging" for item in normalized):
        return False
    return any(item == "discharging" for item in normalized)


def detect_power(probes: HostProbes | None = None) -> PowerState:
    """
    Rileva batteria e Low Power Mode con cache breve.

    Args:
        probes: Collaboratori iniettabili; None usa quelli reali.

    Returns:
        Snapshot conservativo: False/False se i probe falliscono.
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
    """Batteria via sysfs; Low Power da platform_profile se presente."""
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
    """Esegue un comando breve; None se assente, timeout o exit ≠ 0."""
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
    PID del processo `ollama` daemon, escluso questo interprete.

    `pgrep -x` evita di matchare la command line dell'agente. Tra più PID
    si preferisce quello la cui cmdline contiene `serve`.
    """
    raw = _run_command(["pgrep", "-x", "ollama"])
    if raw is None:
        return None
    self_pid = os.getpid()
    candidati: list[tuple[int, str]] = []
    for line in raw.splitlines():
        token = line.strip()
        if not token.isdigit():
            continue
        pid = int(token)
        if pid == self_pid:
            continue
        command = _run_command(["ps", "-o", "command=", "-p", str(pid)]) or ""
        if not _cmdline_sembra_daemon_ollama(command):
            continue
        candidati.append((pid, command))
    for pid, command in candidati:
        if "serve" in command.casefold():
            return pid
    if candidati:
        return candidati[0][0]
    return None


def environ_daemon_ollama(pid: int) -> dict[str, str] | None:
    """Best-effort: env del daemon; None se il processo non è visibile."""
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


def _cmdline_sembra_daemon_ollama(command: str) -> bool:
    """True se la cmdline è il binario ollama, non l'agente o pytest."""
    lowered = command.casefold()
    if not lowered.strip():
        return False
    if "assistente" in lowered or "pytest" in lowered or "python" in lowered:
        return False
    if "serve" in lowered:
        return True
    tokens = command.split()
    return bool(tokens) and Path(tokens[0]).name.casefold() == "ollama"


def _parse_proc_environ(raw: bytes) -> dict[str, str]:
    """Decodifica `/proc/<pid>/environ` in coppie chiave/valore."""
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
    """Estrae coppie OLLAMA_* da `ps eww` (best-effort, split su whitespace)."""
    env: dict[str, str] = {}
    for token in stdout.replace("\n", " ").split():
        if not token.startswith("OLLAMA_") or "=" not in token:
            continue
        key, _, value = token.partition("=")
        env[key] = value
    return env


def _list_dir(path: Path) -> list[str] | None:
    """Elenca i nomi in una directory; None se assente."""
    try:
        return [entry.name for entry in path.iterdir()]
    except OSError:
        return None


def _parse_positive_int(raw: str | None) -> int:
    """Converte una stringa numerica in intero positivo; 0 se assente o invalida."""
    if raw is None:
        return 0
    try:
        value = int(raw.strip())
    except (TypeError, ValueError):
        return 0
    return value if value > 0 else 0


def _read_sysctl(name: str) -> str | None:
    """Esegue `sysctl -n`; None se il binario manca o la chiave è assente."""
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
    # Chiavi inesistenti escono != 0; non è un errore da propagare.
    if result.returncode != 0:
        return None
    text = result.stdout.strip()
    return text or None


def _read_text(path: Path) -> str | None:
    """Legge un file di testo; None su file assente o non decodificabile."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _read_nvidia_smi() -> str | None:
    """Interroga nvidia-smi; None se assente, in timeout o in errore."""
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
        Byte, oppure 0 su piattaforme senza sysconf (comportamento storico).
    """
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        # Piattaforme prive di sysconf continueranno con un profilo esplicito.
        return 0
    return page_size * page_count
