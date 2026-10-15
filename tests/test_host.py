"""Host detection tests with fake probes: no real sysctl or nvidia-smi."""

from __future__ import annotations

from pathlib import Path

import pytest

from assistant.host import (
    AMPLE_MEMORY_BYTES,
    DEFAULT_OLLAMA_BASE_URL,
    GIBIBYTE,
    HostProbes,
    can_expand_context,
    detect_host,
    inference_host_options,
    ollama_base_url,
    parse_apple_chip_family,
    parse_meminfo_bytes,
    parse_nvidia_smi_mib,
    processor_looks_cpu_only,
)

pytestmark = pytest.mark.unit

_MEMORY_16_GIB = 16 * GIBIBYTE
_MEMORY_8_GIB = 8 * GIBIBYTE


def _probes(**overrides: object) -> HostProbes:
    """Build deterministic probes starting from a Darwin 16 GB snapshot."""
    values: dict[str, object] = {
        "system": lambda: "Darwin",
        "machine": lambda: "arm64",
        "sysctl": lambda name: {
            "machdep.cpu.brand_string": "Apple M4",
            "hw.memsize": str(_MEMORY_16_GIB),
            "hw.perflevel0.physicalcpu": "4",
        }.get(name),
        "read_text": lambda _path: None,
        "nvidia_smi_output": lambda: None,
        "cpu_count": lambda: 8,
        "posix_memory_bytes": lambda: _MEMORY_16_GIB,
    }
    values.update(overrides)
    return HostProbes(**values)  # type: ignore[arg-type]


def test_parse_apple_chip_family_pro_before_base() -> None:
    """M4 Pro must not collapse onto the base M4 chip."""
    assert parse_apple_chip_family("Apple M4 Pro") == "m4_pro"
    assert parse_apple_chip_family("Apple M4") == "m4"
    assert parse_apple_chip_family("Apple M1 Ultra") == "m1_ultra"
    assert parse_apple_chip_family("Intel Core i7") is None


def test_detect_host_m4_16gb_does_not_widen_context() -> None:
    """The Air/base 16 GB profile stays the historical one: no 8192 ctx."""
    host = detect_host(_probes())
    assert host.os == "darwin"
    assert host.chip_family == "m4"
    assert host.memory_bytes == _MEMORY_16_GIB
    assert host.inference_memory_bytes == _MEMORY_16_GIB
    assert host.has_discrete_gpu is False
    assert host.cooling == "fanless_conservative"
    assert host.performance_cores == 4
    assert can_expand_context(host) is False
    options = inference_host_options(host)
    assert options.num_gpu == -1
    assert options.num_thread is None
    assert options.keep_alive == "30m"
    assert options.kv_cache_type == "q4_0"


def test_detect_host_m4_pro_widens_even_below_32gb() -> None:
    """Pro/Max/Ultra have active cooling and can widen the heavy routes."""
    host = detect_host(
        _probes(
            sysctl=lambda name: {
                "machdep.cpu.brand_string": "Apple M4 Pro",
                "hw.memsize": str(24 * GIBIBYTE),
                "hw.perflevel0.physicalcpu": "8",
            }.get(name),
        )
    )
    assert host.chip_family == "m4_pro"
    assert host.cooling == "cooled"
    assert can_expand_context(host) is True


def test_linux_nvidia_uses_vram_as_inference_memory() -> None:
    """On Linux the weights are measured in VRAM, not in system RAM."""
    meminfo = "MemTotal:       32768000 kB\nMemAvailable:   20000000 kB\n"
    host = detect_host(
        _probes(
            system=lambda: "Linux",
            machine=lambda: "x86_64",
            read_text=lambda path: meminfo if path == Path("/proc/meminfo") else None,
            nvidia_smi_output=lambda: "8192\n",
        )
    )
    assert host.os == "linux"
    assert host.chip_family == "linux_nvidia"
    assert host.has_discrete_gpu is True
    assert host.inference_memory_bytes == 8192 * (1024**2)
    assert host.memory_bytes == 32768000 * 1024
    options = inference_host_options(host)
    assert options.num_gpu == -1
    assert options.num_thread is None


def test_linux_cpu_only_num_gpu_zero() -> None:
    """Without nvidia-smi the GPU offload must be explicitly disabled."""
    host = detect_host(
        _probes(
            system=lambda: "Linux",
            machine=lambda: "aarch64",
            read_text=lambda path: "MemTotal: 16384000 kB\n"
            if path == Path("/proc/meminfo")
            else None,
            nvidia_smi_output=lambda: None,
        )
    )
    assert host.chip_family == "linux_cpu"
    assert host.has_discrete_gpu is False
    assert inference_host_options(host).num_gpu == 0


def test_intel_mac_does_not_force_gpu() -> None:
    """Darwin x86 does not set num_gpu: Ollama decides on its own."""
    host = detect_host(
        _probes(
            machine=lambda: "x86_64",
            sysctl=lambda name: {
                "machdep.cpu.brand_string": "Intel(R) Core(TM) i7",
                "hw.memsize": str(_MEMORY_16_GIB),
            }.get(name),
        )
    )
    assert host.chip_family == "intel"
    assert host.cooling == "cooled"
    assert inference_host_options(host).num_gpu is None


def test_keep_alive_wide_from_32_gib() -> None:
    """From 32 GiB up the context widens; keep-alive stays at 30m."""
    host = detect_host(
        _probes(
            sysctl=lambda name: {
                "machdep.cpu.brand_string": "Apple M4 Max",
                "hw.memsize": str(AMPLE_MEMORY_BYTES),
                "hw.perflevel0.physicalcpu": "12",
            }.get(name),
        )
    )
    options = inference_host_options(host)
    assert options.keep_alive == "30m"
    assert options.kv_cache_type == "q4_0"
    assert can_expand_context(host) is True


def test_parse_meminfo_and_nvidia_smi() -> None:
    """The parsers stay robust to extra lines and multiple GPUs."""
    assert parse_meminfo_bytes("MemTotal: 1024 kB\n") == 1024 * 1024
    assert parse_meminfo_bytes("MemAvailable: 10 kB\n") == 0
    assert parse_nvidia_smi_mib("8192\n4096\n") == 8192 * (1024**2)


def test_ollama_base_url_normalizes_host_without_scheme() -> None:
    """OLLAMA_HOST without a scheme becomes an httpx-compatible URL."""
    assert ollama_base_url({}) == DEFAULT_OLLAMA_BASE_URL
    assert ollama_base_url({"OLLAMA_HOST": "127.0.0.1:11434"}) == (
        "http://127.0.0.1:11434"
    )
    assert ollama_base_url({"OLLAMA_HOST": "http://10.0.0.2:11434/"}) == (
        "http://10.0.0.2:11434"
    )


def test_processor_cpu_only() -> None:
    """CPU fallback diagnostics: ignore mixed GPU/CPU strings."""
    assert processor_looks_cpu_only("CPU") is True
    assert processor_looks_cpu_only("100% GPU") is False
    assert processor_looks_cpu_only("") is False


def test_parse_pmset_battery_and_low_power() -> None:
    """Darwin parser: AC vs battery and the lowpowermode flag."""
    from assistant.host import parse_pmset_low_power_mode, parse_pmset_on_battery

    assert parse_pmset_on_battery("Now drawing from 'Battery Power'\n") is True
    assert parse_pmset_on_battery("Now drawing from 'AC Power'\n") is False
    assert parse_pmset_on_battery("") is False
    assert parse_pmset_on_battery("   \n") is False
    assert parse_pmset_low_power_mode(" lowpowermode         1\n") is True
    assert parse_pmset_low_power_mode(" lowpowermode         0\n") is False


def test_detect_power_uses_injectable_probes() -> None:
    """detect_power does not run the real pmset when the probes are mocked."""
    from assistant.host import detect_power, reset_power_cache_for_tests

    reset_power_cache_for_tests()
    commands: list[list[str]] = []

    def run_command(argv: list[str]) -> str | None:
        commands.append(argv)
        if argv[-1] == "batt":
            return "Now drawing from 'Battery Power'"
        return "lowpowermode 1"

    power = detect_power(
        _probes(run_command=run_command)
    )
    assert power.on_battery is True
    assert power.low_power_mode is True
    assert commands


def test_ollama_daemon_pid_prefers_serve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Among several `ollama` processes, the one with `serve` in its cmdline wins."""
    from assistant.host import pid_daemon_ollama

    def fake_run(argv: list[str]) -> str | None:
        if argv[:2] == ["pgrep", "-x"]:
            return "100\n200"
        if argv == ["ps", "-o", "command=", "-p", "100"]:
            return "/Applications/Ollama.app/Contents/Resources/ollama"
        if argv == ["ps", "-o", "command=", "-p", "200"]:
            return "ollama serve"
        return None

    monkeypatch.setattr("assistant.host._run_command", fake_run)
    assert pid_daemon_ollama() == 200


def test_ollama_daemon_pid_ignores_agent_command_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Python/agent cmdline is not mistaken for the daemon."""
    from assistant.host import pid_daemon_ollama

    def fake_run(argv: list[str]) -> str | None:
        if argv[:2] == ["pgrep", "-x"]:
            return "50"
        return "python -m pytest assistant ollama serve"

    monkeypatch.setattr("assistant.host._run_command", fake_run)
    assert pid_daemon_ollama() is None
