"""Installazione e avvio Ollama per Darwin (Homebrew) e Linux (installer ufficiale)."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Mapping, MutableMapping
from pathlib import Path
from typing import Any, Protocol

from assistente.host import (
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
    """Interrompe il wizard dopo un rifiuto o un passo fallito."""

    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class BootstrapUI(Protocol):
    """Consensi e messaggi usati dagli installer, senza dipendere da ConsoleUI."""

    def stampa_messaggio_sistema(self, messaggio: str) -> None: ...

    def chiedi_conferma(self, domanda: str) -> bool: ...

    def aggiorna_progresso(self, messaggio: str) -> None: ...


class BootstrapHost(Protocol):
    """Superficie minima di processi/PATH/ping richiesta dagli installer."""

    environ: MutableMapping[str, str]
    run: Callable[..., subprocess.CompletedProcess[str]]
    popen: Callable[..., subprocess.Popen[str]]

    def find_on_path(self, name: str) -> str | None: ...

    def path_is_file(self, path: Path) -> bool: ...

    def ping(self) -> bool: ...

    def sleep(self, seconds: float) -> None: ...

    def monotonic(self) -> float: ...

    def system(self) -> str: ...


def richiedi_consenso(ui: BootstrapUI, domanda: str, rifiuto: str) -> None:
    """Alza SetupAborted se l'utente rifiuta: nessun download silenzioso."""
    if not ui.chiedi_conferma(domanda):
        raise SetupAborted(rifiuto)


def esegui_comando(
    host: BootstrapHost,
    argv: list[str],
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    """Esegue argv sul host; OSError diventa SetupAborted con il binario in causa."""
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
    Avvia un figlio non legato al TTY (POSIX start_new_session / Windows job).

    Args:
        host: Collaboratore processi.
        argv: Comando da lanciare.
        env: Ambiente opzionale; None eredita quello del host.
    """
    kwargs: dict[str, object] = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    # Windows non ha start_new_session; il flag job evita Ctrl+C sulla CLI.
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


def attendi_api(host: BootstrapHost) -> bool:
    """Attende `/api/version` fino a OLLAMA_HEALTH_WAIT_SECONDS."""
    scadenza = host.monotonic() + OLLAMA_HEALTH_WAIT_SECONDS
    while host.monotonic() < scadenza:
        if host.ping():
            return True
        host.sleep(OLLAMA_HEALTH_POLL_SECONDS)
    return host.ping()


def installa_homebrew(ui: BootstrapUI, host: BootstrapHost) -> None:
    """Installer ufficiale Homebrew in foreground, con consenso esplicito."""
    ui.stampa_messaggio_sistema(
        "Homebrew non è installato. L'installer ufficiale può chiedere "
        f"la password amministratore. Documentazione: {BREW_SH_URL}"
    )
    richiedi_consenso(
        ui,
        "Vuoi installare Homebrew ora?",
        "Installazione Homebrew rifiutata. Senza brew non si può "
        "installare Ollama secondo STARTUP.MD.",
    )
    ui.stampa_messaggio_sistema("Avvio installer Homebrew nel terminale...")
    risultato = esegui_comando(host, HOMEBREW_INSTALL_ARGV, check=False)
    if risultato.returncode != 0:
        raise SetupAborted(
            f"Installer Homebrew uscito con codice {risultato.returncode}. "
            f"Nessun retry automatico; vedi {BREW_SH_URL}."
        )


def installa_ollama_con_brew(
    ui: BootstrapUI,
    host: BootstrapHost,
    brew_path: Path | None,
) -> None:
    """`brew install ollama` solo dopo consenso; nessun installer .dmg."""
    if brew_path is None:
        raise SetupAborted("Homebrew assente: impossibile installare Ollama.")
    ui.stampa_messaggio_sistema(
        "Ollama non è nel PATH. Verrà installato solo con `brew install ollama`."
    )
    richiedi_consenso(
        ui,
        "Vuoi installare Ollama con Homebrew ora?",
        "Installazione Ollama rifiutata.",
    )
    ui.aggiorna_progresso("Eseguo brew install ollama...")
    risultato = esegui_comando(
        host,
        [str(brew_path), "install", "ollama"],
        check=False,
    )
    if risultato.returncode != 0:
        raise SetupAborted(
            f"`brew install ollama` uscito con codice {risultato.returncode}."
        )


def installa_ollama_linux(ui: BootstrapUI, host: BootstrapHost) -> None:
    """Installer ufficiale Linux in foreground; Python non wrappa sudo."""
    ui.stampa_messaggio_sistema(
        "Ollama non è nel PATH. Verrà usato l'installer ufficiale "
        f"({OLLAMA_LINUX_INSTALL_URL}), che può chiedere la password."
    )
    richiedi_consenso(
        ui,
        "Vuoi installare Ollama ora?",
        "Installazione Ollama rifiutata.",
    )
    ui.stampa_messaggio_sistema("Avvio installer Ollama nel terminale...")
    risultato = esegui_comando(host, OLLAMA_LINUX_INSTALL_ARGV, check=False)
    if risultato.returncode != 0:
        raise SetupAborted(
            f"Installer Ollama uscito con codice {risultato.returncode}. "
            f"Nessun retry automatico; vedi {OLLAMA_LINUX_INSTALL_URL}."
        )


def avvia_ollama(
    ui: BootstrapUI,
    host: BootstrapHost,
    ollama_cli: Path | None,
    *,
    brew_path: Path | None = None,
    apply_server_env: bool = False,
) -> None:
    """
    Porta l'API in ascolto: brew services / systemd, poi `ollama serve`.

    Args:
        ui: Progresso e messaggi.
        host: Processi e ping.
        ollama_cli: Binario da usare per `serve`; None usa `ollama` nel PATH.
        brew_path: Se presente tenta `brew services start` (solo Darwin).
        apply_server_env: True solo quando questo processo spawna `ollama serve`.
    """
    sistema = host.system()
    # Darwin: il servizio Homebrew è il percorso preferito, senza env custom.
    if sistema == "Darwin" and brew_path is not None:
        ui.aggiorna_progresso("Avvio brew services start ollama...")
        try:
            host.run(
                [str(brew_path), "services", "start", "ollama"],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            pass
        if attendi_api(host):
            _avvisa_env_daemon_preesistente(ui)
            return
    # Linux: user unit, poi system unit; entrambi possono non esistere.
    if sistema == "Linux":
        if _avvia_systemctl(host, user=True) and attendi_api(host):
            _avvisa_env_daemon_preesistente(ui)
            return
        if _avvia_systemctl(host, user=False) and attendi_api(host):
            _avvisa_env_daemon_preesistente(ui)
            return
    binario = str(ollama_cli) if ollama_cli is not None else "ollama"
    ui.aggiorna_progresso("API assente: avvio ollama serve in background...")
    env: Mapping[str, str] | None = None
    # Solo il figlio spawnato riceve flash-attn / KV / parallel.
    if apply_server_env:
        options = _opzioni_server()
        env = ollama_server_environ(host.environ, options)
    popen_detached(host, [binario, "serve"], env=env)
    if not attendi_api(host):
        raise SetupAborted(
            f"Timeout: Ollama non risponde su {DEFAULT_OLLAMA_BASE_URL}."
        )


def _avvia_systemctl(host: BootstrapHost, *, user: bool) -> bool:
    """Tenta `systemctl start ollama`; False se il binario manca."""
    argv = ["systemctl"]
    if user:
        argv.append("--user")
    argv.extend(["start", "ollama"])
    try:
        host.run(argv, check=False, capture_output=True, text=True)
    except OSError:
        return False
    return True


def _opzioni_server() -> InferenceHostOptions:
    """Policy server dallo snapshot host reale (wizard di produzione)."""
    return inference_host_options(detect_host())


def _avvisa_env_daemon_preesistente(ui: BootstrapUI) -> None:
    """Il daemon già in ascolto non riceve OLLAMA_*: il README resta la guida."""
    ui.stampa_messaggio_sistema(
        "Ollama era già avviabile come servizio: flash attention e KV cache "
        "si configurano sul daemon (vedi README), non dal client."
    )
