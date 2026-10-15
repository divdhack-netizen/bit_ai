"""Ollama installation and startup for Darwin (Homebrew) and Linux (official installer)."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Mapping, MutableMapping
from pathlib import Path
from typing import Any, Protocol

from assistant.host import (
    DEFAULT_OLLAMA_BASE_URL,
    InferenceHostOptions,
    detect_host,
    inference_host_options,
    ollama_server_environ,
)

HOMEBREW_BIN_APPLE = Path("/opt/homebrew/bin/brew")
HOMEBREW_BIN_INTEL = Path("/usr/local/bin/brew")
HOMEBREW_INSTALL_SCRIPT_URL = (
    "https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh"
)
HOMEBREW_INSTALL_ARGV = [
    "/bin/bash",
    "-c",
    f"$(curl -fsSL {HOMEBREW_INSTALL_SCRIPT_URL})",
]
BREW_SH_URL = "https://brew.sh"
OLLAMA_LINUX_INSTALL_URL = "https://ollama.com/install.sh"
OLLAMA_LINUX_INSTALL_ARGV = [
    "/bin/sh",
    "-c",
    f"$(curl -fsSL {OLLAMA_LINUX_INSTALL_URL})",
]
OLLAMA_HEALTH_WAIT_SECONDS = 20.0
OLLAMA_HEALTH_POLL_SECONDS = 0.5


class SetupAborted(RuntimeError):
    """Stop the wizard after a refusal or a failed step."""

    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class BootstrapUI(Protocol):
    """Consents and messages used by the installers, without depending on ConsoleUI."""

    def print_system_message(self, message: str) -> None: ...

    def ask_confirmation(self, question: str) -> bool: ...

    def update_progress(self, message: str) -> None: ...


class BootstrapHost(Protocol):
    """Minimal process/PATH/ping surface required by the installers."""

    environ: MutableMapping[str, str]
    run: Callable[..., subprocess.CompletedProcess[str]]
    popen: Callable[..., subprocess.Popen[str]]

    def find_on_path(self, name: str) -> str | None: ...

    def path_is_file(self, path: Path) -> bool: ...

    def ping(self) -> bool: ...

    def sleep(self, seconds: float) -> None: ...

    def monotonic(self) -> float: ...

    def system(self) -> str: ...


def request_consent(ui: BootstrapUI, question: str, rejection: str) -> None:
    """Raise SetupAborted if the user declines: no silent downloads."""
    if not ui.ask_confirmation(question):
        raise SetupAborted(rejection)


def run_command(
    host: BootstrapHost,
    argv: list[str],
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    """Run argv on the host; OSError becomes SetupAborted naming the binary involved."""
    try:
        return host.run(argv, **kwargs)
    except OSError as error:
        raise SetupAborted(f"Impossibile eseguire {argv[0]}: {error}.") from error


def popen_detached(
    host: BootstrapHost,
    argv: list[str],
    *,
    env: Mapping[str, str] | None = None,
) -> None:
    """
    Start a child detached from the TTY (POSIX start_new_session / Windows job).

    Args:
        host: Process collaborator.
        argv: Command to launch.
        env: Optional environment; None inherits the host's.
    """
    kwargs: dict[str, object] = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    # Windows has no start_new_session; the job flag keeps Ctrl+C off the CLI.
    if os.name == "nt":
        kwargs["creationflags"] = getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
    else:
        kwargs["start_new_session"] = True
    if env is not None:
        kwargs["env"] = dict(env)
    try:
        host.popen(argv, **kwargs)
    except OSError as error:
        raise SetupAborted(f"Impossibile avviare {argv[0]}: {error}.") from error


def wait_for_api(host: BootstrapHost) -> bool:
    """Wait for `/api/version` up to OLLAMA_HEALTH_WAIT_SECONDS."""
    deadline = host.monotonic() + OLLAMA_HEALTH_WAIT_SECONDS
    while host.monotonic() < deadline:
        if host.ping():
            return True
        host.sleep(OLLAMA_HEALTH_POLL_SECONDS)
    return host.ping()


def install_homebrew(ui: BootstrapUI, host: BootstrapHost) -> None:
    """Official Homebrew installer in the foreground, with explicit consent."""
    ui.print_system_message(
        "Homebrew non è installato. L'installer ufficiale può chiedere "
        f"la password amministratore. Documentazione: {BREW_SH_URL}"
    )
    request_consent(
        ui,
        "Vuoi installare Homebrew ora?",
        "Installazione Homebrew rifiutata. Senza brew non si può "
        "installare Ollama secondo STARTUP.MD.",
    )
    ui.print_system_message("Avvio installer Homebrew nel terminale...")
    result = run_command(host, HOMEBREW_INSTALL_ARGV, check=False)
    if result.returncode != 0:
        raise SetupAborted(
            f"Installer Homebrew uscito con codice {result.returncode}. "
            f"Nessun retry automatico; vedi {BREW_SH_URL}."
        )


def install_ollama_with_brew(
    ui: BootstrapUI,
    host: BootstrapHost,
    brew_path: Path | None,
) -> None:
    """`brew install ollama` only after consent; no .dmg installer."""
    if brew_path is None:
        raise SetupAborted("Homebrew assente: impossibile installare Ollama.")
    ui.print_system_message(
        "Ollama non è nel PATH. Verrà installato solo con `brew install ollama`."
    )
    request_consent(
        ui,
        "Vuoi installare Ollama con Homebrew ora?",
        "Installazione Ollama rifiutata.",
    )
    ui.update_progress("Eseguo brew install ollama...")
    result = run_command(
        host,
        [str(brew_path), "install", "ollama"],
        check=False,
    )
    if result.returncode != 0:
        raise SetupAborted(
            f"`brew install ollama` uscito con codice {result.returncode}."
        )


def install_ollama_linux(ui: BootstrapUI, host: BootstrapHost) -> None:
    """Official Linux installer in the foreground; Python does not wrap sudo."""
    ui.print_system_message(
        "Ollama non è nel PATH. Verrà usato l'installer ufficiale "
        f"({OLLAMA_LINUX_INSTALL_URL}), che può chiedere la password."
    )
    request_consent(
        ui,
        "Vuoi installare Ollama ora?",
        "Installazione Ollama rifiutata.",
    )
    ui.print_system_message("Avvio installer Ollama nel terminale...")
    result = run_command(host, OLLAMA_LINUX_INSTALL_ARGV, check=False)
    if result.returncode != 0:
        raise SetupAborted(
            f"Installer Ollama uscito con codice {result.returncode}. "
            f"Nessun retry automatico; vedi {OLLAMA_LINUX_INSTALL_URL}."
        )


def start_ollama(
    ui: BootstrapUI,
    host: BootstrapHost,
    ollama_cli: Path | None,
    *,
    brew_path: Path | None = None,
    apply_server_env: bool = False,
) -> None:
    """
    Get the API listening: brew services / systemd, then `ollama serve`.

    Args:
        ui: Progress and messages.
        host: Processes and ping.
        ollama_cli: Binary to use for `serve`; None uses `ollama` from PATH.
        brew_path: If present, try `brew services start` (Darwin only).
        apply_server_env: True only when this process spawns `ollama serve`.
    """
    system = host.system()
    # Darwin: the Homebrew service is the preferred path, with no custom env.
    if system == "Darwin" and brew_path is not None:
        ui.update_progress("Avvio brew services start ollama...")
        try:
            host.run(
                [str(brew_path), "services", "start", "ollama"],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            pass
        if wait_for_api(host):
            _warn_preexisting_daemon_env(ui)
            return
    # Linux: user unit, then system unit; either may not exist.
    if system == "Linux":
        if _start_systemctl(host, user=True) and wait_for_api(host):
            _warn_preexisting_daemon_env(ui)
            return
        if _start_systemctl(host, user=False) and wait_for_api(host):
            _warn_preexisting_daemon_env(ui)
            return
    binary = str(ollama_cli) if ollama_cli is not None else "ollama"
    ui.update_progress("API assente: avvio ollama serve in background...")
    env: Mapping[str, str] | None = None
    # Only the spawned child receives flash-attn / KV / parallel.
    if apply_server_env:
        options = _server_options()
        env = ollama_server_environ(host.environ, options)
    popen_detached(host, [binary, "serve"], env=env)
    if not wait_for_api(host):
        raise SetupAborted(
            f"Timeout: Ollama non risponde su {DEFAULT_OLLAMA_BASE_URL}."
        )


def _start_systemctl(host: BootstrapHost, *, user: bool) -> bool:
    """Try `systemctl start ollama`; False if the binary is missing."""
    argv = ["systemctl"]
    if user:
        argv.append("--user")
    argv.extend(["start", "ollama"])
    try:
        host.run(argv, check=False, capture_output=True, text=True)
    except OSError:
        return False
    return True


def _server_options() -> InferenceHostOptions:
    """Server policy from the real host snapshot (production wizard)."""
    return inference_host_options(detect_host())


def _warn_preexisting_daemon_env(ui: BootstrapUI) -> None:
    """A daemon already listening does not receive OLLAMA_*: the README is the guide."""
    ui.print_system_message(
        "Ollama era già avviabile come servizio: flash attention e KV cache "
        "si configurano sul daemon (vedi README), non dal client."
    )
