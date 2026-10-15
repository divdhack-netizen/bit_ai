"""First-run wizard: Ollama on macOS/Linux and the Gemma 4 models."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import httpx

from assistant.bootstrap import (
    BREW_SH_URL,
    HOMEBREW_BIN_APPLE,
    HOMEBREW_BIN_INTEL,
    HOMEBREW_INSTALL_ARGV,
    SetupAborted,
    start_ollama,
    install_homebrew,
    install_ollama_with_brew,
    install_ollama_linux,
)
from assistant.host import DEFAULT_OLLAMA_BASE_URL
from assistant.inference_profiles import (
    GIBIBYTE,
    TIGHT_CONTEXT_LEFTOVER_GB,
    ModelVariant,
    model_variants,
    normalize_ollama_names,
    physical_memory_bytes,
    resolve_model_variant,
    variant_is_installed,
)
from assistant.ollama_runtime import OllamaRuntime

MODEL_FAMILY = "Gemma 4"
MAX_MODEL_SELECTION_ATTEMPTS = 5
NON_INTERACTIVE_SETUP_MESSAGE = (
    f"Ambiente incompleto (Ollama o un modello {MODEL_FAMILY}). "
    "Avvia `main.py` in un terminale interattivo oppure segui STARTUP.MD."
)
_EXPORT_SHELLENV = re.compile(
    r'^export\s+([A-Za-z_][A-Za-z0-9_]*)="(.*)";?\s*$'
)
_VAR_BRACE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_VAR_SIMPLE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
ModelTier = Literal["consigliato", "stretto", "sconsigliato"]
ProgressCallback = Callable[[str], None]


class StartupUI(Protocol):
    """Minimal surface used by the wizard (ConsoleUI or a fake in tests)."""

    def print_system_message(self, message: str) -> None: ...

    def print_error(self, message: str) -> None: ...

    def ask_confirmation(self, question: str) -> bool: ...

    def ask_text(self, question: str) -> str: ...

    def update_progress(self, message: str) -> None: ...


@dataclass(frozen=True, slots=True)
class DetectedEnvironment:
    """Snapshot of the dependencies required by STARTUP.MD."""

    brew_path: Path | None
    ollama_cli: Path | None
    ollama_api: bool
    available_model_names: frozenset[str]
    memory_bytes: int
    machine: str
    installed_aliases: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "installed_aliases",
            installed_model_aliases(self.available_model_names),
        )

    @property
    def brew_ok(self) -> bool:
        """True if detection resolved a brew path."""
        return self.brew_path is not None

    @property
    def ollama_ok(self) -> bool:
        """True if both the Ollama CLI and API are available."""
        return self.ollama_cli is not None and self.ollama_api

    @property
    def has_model(self) -> bool:
        """True if at least one matrix alias is in the Ollama catalog."""
        return bool(self.installed_aliases)

    @property
    def ready(self) -> bool:
        """True if the Ollama CLI+API and at least one matrix model are present."""
        return self.ollama_ok and self.has_model


@dataclass
class StartupHost:
    """
    Injectable collaborators for filesystem, processes and Ollama.

    A single HTTP client is created on the first ping/list/pull and reused
    for the whole wizard, so the healthcheck does not open a TCP connection on every poll.
    """

    which: Callable[..., str | None] = shutil.which
    path_is_file: Callable[[Path], bool] = Path.is_file
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run
    popen: Callable[..., subprocess.Popen[str]] = subprocess.Popen
    sleep: Callable[[float], None] = time.sleep
    monotonic: Callable[[], float] = time.monotonic
    memory_bytes: Callable[[], int] = physical_memory_bytes
    machine: Callable[[], str] = platform.machine
    system: Callable[[], str] = platform.system
    environ: MutableMapping[str, str] = field(default_factory=lambda: os.environ)
    runtime_factory: Callable[[], OllamaRuntime] = OllamaRuntime
    ping_ollama: Callable[[], bool] | None = None
    list_models: Callable[[], set[str]] | None = None
    pull_model: Callable[[str, ProgressCallback], None] | None = None
    _runtime: OllamaRuntime | None = field(default=None, init=False, repr=False)

    def find_on_path(self, name: str) -> str | None:
        """
        Look up an executable on the host's PATH, not the global `os.environ`.

        Tests pass an isolated `environ`: `shutil.which` must use it.
        Test lambdas do not accept `path=` and fall into the TypeError branch.
        """
        path_value = self.environ.get("PATH")
        try:
            return self.which(name, path=path_value)
        except TypeError:
            return self.which(name)

    def ping(self) -> bool:
        """`/api/version` healthcheck reusing the host's HTTP client."""
        if self.ping_ollama is not None:
            return self.ping_ollama()
        return self._ollama().ping()

    def list_catalog(self) -> set[str]:
        """`/api/tags` catalog on the same client as the ping."""
        if self.list_models is not None:
            return self.list_models()
        return self._ollama().list_available_model_names()

    def pull(self, tag: str, on_progress: ProgressCallback) -> None:
        """Streamed pull; network errors become SetupAborted."""
        if self.pull_model is not None:
            self.pull_model(tag, on_progress)
            return
        try:
            self._ollama().pull_model(tag, on_progress)
        except (httpx.HTTPError, RuntimeError) as error:
            raise SetupAborted(f"Pull {tag} fallito: {error}.") from error

    def close(self) -> None:
        """Close the owned Ollama client, without unloading the session weights."""
        if self._runtime is None:
            return
        self._runtime.close(unload=False)
        self._runtime = None

    def _ollama(self) -> OllamaRuntime:
        if self._runtime is None:
            self._runtime = self.runtime_factory()
        return self._runtime


def non_interactive_setup_message() -> str:
    """Text for `--prompt` / non-TTY stdin when the environment is incomplete."""
    return NON_INTERACTIVE_SETUP_MESSAGE


def interactive_session(
    single_prompt: str | None,
    stdin_tty: bool | None = None,
) -> bool:
    """
    True when the wizard may ask for consent.

    `--prompt` forbids the wizard even if stdin is a TTY.
    """
    if single_prompt is not None:
        return False
    if stdin_tty is None:
        stdin_tty = sys.stdin.isatty()
    return stdin_tty


def installed_model_aliases(available_names: Iterable[str]) -> tuple[str, ...]:
    """Matrix aliases present in the catalog, in fast → default order."""
    names = normalize_ollama_names(set(available_names))
    return tuple(
        variant.alias
        for variant in model_variants()
        if variant_is_installed(variant.tag, names)
    )


def model_tier(variant: ModelVariant, memory_bytes: int) -> ModelTier:
    """
    Classify an artifact against the detected RAM.

    Unknown RAM (≤0): no penalizing tier.
    """
    if memory_bytes <= 0:
        return "consigliato"
    if not variant.fits_memory(memory_bytes):
        return "sconsigliato"
    leftover_gb = (memory_bytes / GIBIBYTE) - variant.size_gb
    if leftover_gb < TIGHT_CONTEXT_LEFTOVER_GB:
        return "stretto"
    return "consigliato"


def suggested_alias(memory_bytes: int) -> str | None:
    """
    Suggest default if it is recommended, otherwise the first recommended,
    otherwise the first tight one; None if all are discouraged.
    """
    tiers = {
        variant.alias: model_tier(variant, memory_bytes)
        for variant in model_variants()
    }
    if tiers["default"] == "consigliato":
        return "default"
    narrow_fallback: str | None = None
    for variant in model_variants():
        tier = tiers[variant.alias]
        if tier == "consigliato":
            return variant.alias
        if tier == "stretto" and narrow_fallback is None:
            narrow_fallback = variant.alias
    return narrow_fallback


def parse_model_selection(raw: str, aliases: tuple[str, ...]) -> tuple[str, ...]:
    """
    Parse 1-based indexes, aliases, or a mix, without duplicates.

    Raises:
        ValueError: Unknown token or index out of range.
    """
    if not raw.strip():
        return ()
    known = {name: name for name in aliases}
    selected: list[str] = []
    seen: set[str] = set()
    for token in raw.replace(",", " ").split():
        lowered = token.casefold()
        if lowered.isdigit():
            index = int(lowered)
            if index < 1 or index > len(aliases):
                raise ValueError(f"Indice non valido: {token!r}.")
            alias = aliases[index - 1]
        elif lowered in known:
            alias = known[lowered]
        else:
            raise ValueError(
                f"Selezione non valida: {token!r}. "
                f"Usa indici 1-{len(aliases)} o {', '.join(aliases)}."
            )
        if alias not in seen:
            seen.add(alias)
            selected.append(alias)
    return tuple(selected)


def apply_brew_shellenv(output: str, environ: MutableMapping[str, str]) -> None:
    """Apply the `export`s of `brew shellenv` to the process environment."""
    for line in output.splitlines():
        match = _EXPORT_SHELLENV.match(line.strip())
        if match is None:
            continue
        key, value = match.group(1), match.group(2)
        environ[key] = _expand_shell_value(value, environ)


def find_brew(host: StartupHost) -> Path | None:
    """Resolve Homebrew on the process PATH or in the canonical macOS paths."""
    # Linuxbrew is not a prerequisite: on Linux it is ignored even if present.
    if host.system() != "Darwin":
        return None
    which_result = host.find_on_path("brew")
    candidates = []
    if which_result:
        candidates.append(Path(which_result))
    candidates.extend((HOMEBREW_BIN_APPLE, HOMEBREW_BIN_INTEL))
    return _first_executable(host, candidates)


def find_ollama_cli(host: StartupHost, brew_path: Path | None) -> Path | None:
    """Resolve `ollama` on PATH or in the same `bin/` as Homebrew."""
    which_result = host.find_on_path("ollama")
    candidates: list[Path] = []
    if which_result:
        candidates.append(Path(which_result))
    if brew_path is not None:
        candidates.append(brew_path.parent / "ollama")
    return _first_executable(host, candidates)


def detect_environment(host: StartupHost | None = None) -> DetectedEnvironment:
    """
    Read brew, Ollama and the model catalog.

    If `host` is None a temporary one is created and closed on exit, so
    `main.py` with an environment already set up leaves no HTTP sockets open.
    """
    owns_host = host is None
    active = host or StartupHost()
    try:
        brew_path = find_brew(active)
        if brew_path is not None:
            _ensure_brew_in_path(active, brew_path)
        ollama_cli = find_ollama_cli(active, brew_path)
        ollama_api = active.ping()
        names = active.list_catalog() if ollama_api else set()
        return DetectedEnvironment(
            brew_path=brew_path,
            ollama_cli=ollama_cli,
            ollama_api=ollama_api,
            available_model_names=frozenset(names),
            memory_bytes=active.memory_bytes(),
            machine=active.machine(),
        )
    finally:
        if owns_host:
            active.close()


def align_session_alias(requested_alias: str, environment: DetectedEnvironment) -> str:
    """Use the CLI alias if installed, otherwise the first matrix alias present."""
    installed = environment.installed_aliases
    if requested_alias in installed:
        return requested_alias
    return installed[0] if installed else requested_alias


def run_wizard(ui: StartupUI, host: StartupHost | None = None) -> DetectedEnvironment:
    """
    Run only the missing STARTUP.MD steps.

    Raises:
        SetupAborted: Consent denied or a step failed.
    """
    active = host or StartupHost()
    try:
        return _run_steps(ui, active)
    finally:
        active.close()


def _run_steps(ui: StartupUI, host: StartupHost) -> DetectedEnvironment:
    environment = detect_environment(host)
    system = host.system()
    # The Intel warning applies only to macOS: Linux x86_64 is a first-class target.
    if system == "Darwin" and environment.machine not in {"arm64", "aarch64"}:
        ui.print_system_message(
            f"CPU {environment.machine}: il target è Apple Silicon; si prosegue comunque."
        )
    if not environment.ollama_ok:
        if system == "Darwin":
            _ensure_ollama_darwin(ui, host, environment)
        elif system == "Linux":
            _ensure_ollama_linux(ui, host, environment)
        else:
            raise SetupAborted(
                "Questa piattaforma non ha un wizard di installazione. "
                f"Installa Ollama, avvialo e scarica un {MODEL_FAMILY} (STARTUP.MD)."
            )
        environment = detect_environment(host)
        if not environment.ollama_ok:
            raise SetupAborted(
                "Ollama non è raggiungibile su "
                f"{DEFAULT_OLLAMA_BASE_URL}. Verifica il servizio o `ollama serve`."
            )
    if len(environment.installed_aliases) < len(model_variants()):
        _select_and_download_models(ui, host, environment)
        environment = detect_environment(host)
    if not environment.has_model:
        raise SetupAborted(
            f"Nessun modello {MODEL_FAMILY} della matrice è installato. "
            "Ripeti l'avvio e scegli almeno un alias, oppure "
            "`ollama pull gemma4:12b`."
        )
    ui.print_system_message(f"Ambiente pronto: Ollama e {MODEL_FAMILY}.")
    return environment


def _ensure_ollama_darwin(
    ui: StartupUI,
    host: StartupHost,
    environment: DetectedEnvironment,
) -> None:
    """Homebrew + brew install only if the CLI is missing; then start the service/serve."""
    if not environment.brew_ok:
        install_homebrew(ui, host)
        brew_path = find_brew(host)
        if brew_path is None:
            raise SetupAborted(
                "Homebrew risulta ancora assente dopo l'installer. "
                f"Installa da {BREW_SH_URL} e rilancia."
            )
        _ensure_brew_in_path(host, brew_path)
        ui.print_system_message(
            'Per le prossime shell: eval "$(/opt/homebrew/bin/brew shellenv)" '
            "in ~/.zprofile"
        )
        environment = detect_environment(host)
    if environment.ollama_cli is None:
        install_ollama_with_brew(ui, host, environment.brew_path)
        environment = detect_environment(host)
    start_ollama(
        ui,
        host,
        environment.ollama_cli or find_ollama_cli(host, environment.brew_path),
        brew_path=find_brew(host),
        apply_server_env=True,
    )


def _ensure_ollama_linux(
    ui: StartupUI,
    host: StartupHost,
    environment: DetectedEnvironment,
) -> None:
    """Official installer if the CLI is missing; systemd or ollama serve."""
    if environment.ollama_cli is None:
        install_ollama_linux(ui, host)
        environment = detect_environment(host)
    start_ollama(
        ui,
        host,
        environment.ollama_cli or find_ollama_cli(host, None),
        apply_server_env=True,
    )


def _first_executable(host: StartupHost, candidates: Iterable[Path]) -> Path | None:
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if host.path_is_file(candidate):
            return candidate
    return None


def _directory_in_path(directory: str, path_value: str) -> bool:
    return directory in path_value.split(os.pathsep)


def _expand_shell_value(value: str, environ: Mapping[str, str]) -> str:
    """Expand `${VAR}` and `$VAR` in the order brew emits the exports."""

    def from_braces(match: re.Match[str]) -> str:
        return environ.get(match.group(1), match.group(0))

    def from_simple(match: re.Match[str]) -> str:
        return environ.get(match.group(1), match.group(0))

    expanded = _VAR_BRACE.sub(from_braces, value)
    return _VAR_SIMPLE.sub(from_simple, expanded)


def _ensure_brew_in_path(host: StartupHost, brew_path: Path) -> None:
    """
    Prepend Homebrew's bin and apply `brew shellenv` only if it is missing from PATH.

    So `main.py` with brew already configured does not launch a subprocess on every start.
    """
    brew_bin_dir = str(brew_path.parent)
    current = host.environ.get("PATH", "")
    if _directory_in_path(brew_bin_dir, current):
        return
    host.environ["PATH"] = (
        brew_bin_dir if not current else f"{brew_bin_dir}{os.pathsep}{current}"
    )
    try:
        result = host.run(
            [str(brew_path), "shellenv"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return
    if result.returncode != 0:
        return
    apply_brew_shellenv(result.stdout, host.environ)


def _select_and_download_models(
    ui: StartupUI,
    host: StartupHost,
    environment: DetectedEnvironment,
) -> None:
    aliases = tuple(variant.alias for variant in model_variants())
    ui.print_system_message(
        _model_catalog_text(environment, suggested_alias(environment.memory_bytes))
    )
    chosen = _read_model_selection(ui, aliases)
    if not chosen:
        if environment.has_model:
            ui.print_system_message(
                f"Nessun nuovo pull. Si userà un {MODEL_FAMILY} già presente in Ollama."
            )
            return
        raise SetupAborted(
            f"Nessun modello selezionato e nessun {MODEL_FAMILY} della matrice è installato."
        )
    present = normalize_ollama_names(set(environment.available_model_names))
    for alias in chosen:
        variant = resolve_model_variant(alias)
        if variant_is_installed(variant.tag, present):
            ui.print_system_message(f"{alias} è già installato, skip pull.")
            continue
        ui.update_progress(f"Pull {variant.tag}...")
        host.pull(variant.tag, ui.update_progress)
        present.add(variant.tag)


def _read_model_selection(
    ui: StartupUI,
    aliases: tuple[str, ...],
) -> tuple[str, ...]:
    question = (
        "Quali modelli scaricare? Indici o alias (es. 1 2 oppure fast default). "
        "Invio vuoto = nessuno."
    )
    last_error = "Selezione non valida."
    for _attempt in range(MAX_MODEL_SELECTION_ATTEMPTS):
        raw = ui.ask_text(question)
        try:
            return parse_model_selection(raw, aliases)
        except ValueError as error:
            last_error = str(error)
            ui.print_error(last_error)
    raise SetupAborted(last_error)


def _model_catalog_text(environment: DetectedEnvironment, suggested: str | None) -> str:
    if environment.memory_bytes > 0:
        ram = f"RAM rilevata: {environment.memory_bytes / GIBIBYTE:.0f} GB."
    else:
        ram = "RAM non rilevata: nessuna fascia punitiva."
    lines = [
        f"Modelli {MODEL_FAMILY} (solo questa famiglia). "
        "Suggerito evidenziato, non preselezionato.",
        ram,
    ]
    catalog = normalize_ollama_names(set(environment.available_model_names))
    for index, variant in enumerate(model_variants(), start=1):
        tier = model_tier(variant, environment.memory_bytes)
        present = variant_is_installed(variant.tag, catalog)
        marker = " ← suggerito" if variant.alias == suggested else ""
        state = "già presente" if present else "da scaricare"
        lines.append(
            f"{index}) {variant.alias} · {variant.tag} · ~{variant.size_gb:.1f} GB "
            f"· {tier} · {state}{marker}"
        )
    return "\n".join(lines)
