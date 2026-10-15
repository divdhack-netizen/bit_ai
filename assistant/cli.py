"""Command line and interactive loop."""

from __future__ import annotations

import argparse
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from assistant.config import (
    ConfigRuntime,
    DEFAULT_SLASH_ERRORS,
    FileConfig,
    AVAILABLE_MODELS,
    FILE_SEPARATOR,
    resolve_ollama_tag,
)
from assistant.console import ConsoleUI
from assistant.file_service import WorkspaceFileService
from assistant.host import HostCapabilities, detect_host
from assistant.inference_profiles import (
    normalize_ollama_names,
    resolve_model_variant,
    variant_is_installed,
)
from assistant.ollama_runtime import OllamaRuntime
from assistant.energy_policy import alias_for_power
from assistant.process_log import EventLog, ProcessLog
from assistant.session import AgentSession, TurnCancelled
from assistant.startup import (
    DetectedEnvironment,
    SetupAborted,
    align_session_alias,
    run_wizard,
    non_interactive_setup_message,
    detect_environment,
    interactive_session,
)

EXIT_COMMANDS = frozenset({"exit", "quit", "esci"})
CLEAR_COMMANDS = frozenset({"/clear", "/reset", "/nuova"})
PREWARM_IN_PROGRESS_MESSAGE = "Preriscaldamento modello in corso..."
PRELOAD_THREAD_NAME = "assistant-preload"


@dataclass(slots=True)
class StartupOutcome:
    """Outcome of the preload worker: Ollama I/O with no UI objects."""

    environment: DetectedEnvironment | None
    runtime: OllamaRuntime | None
    host: HostCapabilities | None
    alias: str | None

_TOGGLE_COMMAND = re.compile(
    r"^/(?P<cmd>verbose|v|check|c|burn_protection|bp|thinking|t)"
    r"(?:\s+(?P<azione>on|off|toggle))?$",
    re.IGNORECASE,
)
_MODEL_COMMAND = re.compile(r"^/model(?:\s+(\w+))?$", re.IGNORECASE)
_ERRORS_COMMAND = re.compile(r"^/errori(?:\s+(\d+))?$", re.IGNORECASE)

_ALIAS_TOGGLE = {
    "v": "verbose",
    "c": "check",
    "bp": "burn_protection",
    "t": "thinking",
}
_TOGGLE_LABELS = {
    "verbose": "Verbose",
    "full_verification": "Verifica completa",
    "burn_protection": "Burn protection",
    "thinking": "Thinking",
}
_ACTIONS = {"on": True, "off": False}


def _parse_args() -> tuple[ConfigRuntime, str | None, Path]:
    """Turn the CLI arguments into configuration, prompt and documents folder."""
    defaults = ConfigRuntime()
    parser = argparse.ArgumentParser(
        description=(
            "Agente locale closed-corpus: risponde solo dai documenti della "
            "cartella (txt, md, pdf, docx, immagini) e scrive solo file .txt."
        )
    )
    parser.add_argument(
        "--folder",
        "--cartella",
        dest="folder",
        type=Path,
        default=None,
        help="Cartella dei documenti (default: la cartella corrente).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=defaults.verbose,
    )
    parser.add_argument(
        "--check",
        action=argparse.BooleanOptionalAction,
        default=defaults.full_verification,
    )
    parser.add_argument(
        "--burn-protection",
        action=argparse.BooleanOptionalAction,
        default=defaults.burn_protection,
    )
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=defaults.thinking,
    )
    parser.add_argument(
        "--auto-fast",
        action=argparse.BooleanOptionalAction,
        default=defaults.auto_fast_low_power,
    )
    parser.add_argument(
        "--model",
        choices=sorted(AVAILABLE_MODELS),
        default=defaults.model,
    )
    parser.add_argument("--prompt", default=None)
    args = parser.parse_args()
    config = ConfigRuntime(
        verbose=args.verbose,
        full_verification=args.check,
        burn_protection=args.burn_protection,
        model=args.model,
        thinking=args.thinking,
        auto_fast_low_power=args.auto_fast,
    )
    folder = (args.folder or Path.cwd()).expanduser().resolve()
    if not folder.is_dir():
        parser.error(f"cartella documenti inesistente: {folder}")
    return config, args.prompt, folder


def _toggle(value: bool, action: str | None) -> bool:
    """Apply on/off/toggle to a flag; unknown actions leave the value as is."""
    if action is None or action.lower() == "toggle":
        return not value
    return _ACTIONS.get(action.lower(), value)


def _attr_toggle(cmd: str) -> str:
    """Map short CLI aliases to the ConfigRuntime attribute name."""
    name = _ALIAS_TOGGLE.get(cmd, cmd)
    return "full_verification" if name == "check" else name


def _model_label(config: ConfigRuntime) -> str:
    """Format the effective alias (LPM → fast), quantization and Ollama tag for the prompt."""
    alias = alias_for_power(config.model, config)
    variant = resolve_model_variant(alias)
    return f"{variant.alias} · {variant.quantization} · {variant.tag}"


def _session_catalog(session: AgentSession | None) -> set[str]:
    """Normalized Ollama names visible to the session; empty without a runtime."""
    if session is None:
        return set()
    return normalize_ollama_names(set(session.installed_model_names()))


def _missing_model_text(alias: str) -> str:
    """First line of the /model prompt when the artifact is not in Ollama."""
    variant = resolve_model_variant(alias)
    return (
        f"Il modello {variant.alias} ({variant.tag}, ~{variant.size_gb:.1f} GB) "
        "non è presente."
    )


def _ensure_model_installed(
    alias: str,
    ui: ConsoleUI,
    session: AgentSession | None,
) -> bool:
    """
    True if the alias is usable: already in the catalog, or pull confirmed.

    Without a session (CLI tests) it does not query Ollama or ask to download.
    """
    if session is None:
        return True
    variant = resolve_model_variant(alias)
    if variant_is_installed(variant.tag, _session_catalog(session)):
        return True
    ui.print_system_message(_missing_model_text(alias))
    if not ui.ask_download_confirmation("Vuoi scaricarlo?"):
        ui.print_system_message(
            "Download annullato. Resta il modello precedente."
        )
        return False
    try:
        session.download_model(variant.tag)
    except Exception as error:
        session.log.record_startup("download_modello", error)
        ui.print_error(
            f"Download non riuscito: {type(error).__name__}: {error}"
        )
        return False
    return True


def _slash_model_list(session: AgentSession | None) -> str:
    """/model options with tag and install state when the catalog is known."""
    catalog = _session_catalog(session)
    parts: list[str] = []
    for name in sorted(AVAILABLE_MODELS):
        tag = resolve_ollama_tag(name)
        if session is None:
            parts.append(f"{name} ({tag})")
            continue
        state = "installato" if variant_is_installed(tag, catalog) else "non presente"
        parts.append(f"{name} ({tag}, {state})")
    return ", ".join(parts)


def _apply_model_and_preload(
    config: ConfigRuntime,
    ui: ConsoleUI,
    session: AgentSession | None,
    confirm: str,
    alias_preload: str | None = None,
) -> None:
    """
    Preload the runtime for the effective alias and confirm to the user.

    Args:
        config: Session configuration.
        ui: Console for system messages.
        session: Active session; None skips the preload (tests without a runtime).
        confirm: Final message (model set).
        alias_preload: Alias to load; None applies the energy policy to config.
    """
    if session is not None:
        alias = alias_preload or alias_for_power(config.model, config)
        session.prewarm_model(alias)
    ui.set_model_label(_model_label(config))
    ui.print_system_message(confirm)


def _handle_command(
    request: str,
    config: ConfigRuntime,
    ui: ConsoleUI,
    session: AgentSession | None = None,
) -> bool:
    """Handle a slash command and report whether the request was consumed."""
    if request.lower() in {"/help", "/?"}:
        ui.print_help()
        return True

    if request.lower() in CLEAR_COMMANDS:
        if session is not None:
            session.clear_memory()
        ui.print_system_message("Storico conversazione azzerato.")
        return True

    errors_match = _ERRORS_COMMAND.match(request)
    if errors_match:
        if session is None:
            ui.print_system_message(
                "Nessun registro di sessione disponibile."
            )
            return True
        raw = errors_match.group(1)
        limit = int(raw) if raw else DEFAULT_SLASH_ERRORS
        ui.print_system_message(session.log.format_last_errors(limit))
        return True

    match_toggle = _TOGGLE_COMMAND.match(request)
    if match_toggle:
        attr = _attr_toggle(match_toggle.group("cmd").lower())
        new = _toggle(getattr(config, attr), match_toggle.group("azione"))
        setattr(config, attr, new)
        ui.print_toggle(_TOGGLE_LABELS[attr], new)
        return True

    match_model = _MODEL_COMMAND.match(request)
    if match_model:
        alias = match_model.group(1)
        if alias is None or alias.lower() not in AVAILABLE_MODELS:
            ui.print_system_message(
                f"Modello attuale: {_model_label(config)}. "
                f"Disponibili: {_slash_model_list(session)}"
            )
            return True
        chosen = alias.lower()
        if not _ensure_model_installed(chosen, ui, session):
            return True
        config.model = chosen
        alias_preload = alias_for_power(chosen, config)
        _apply_model_and_preload(
            config,
            ui,
            session,
            confirm=f"Modello impostato: {_model_label(config)}",
            alias_preload=alias_preload,
        )
        return True

    return False


def _process_turn(
    request: str,
    session: AgentSession,
    config: ConfigRuntime,
    ui: ConsoleUI,
) -> bool:
    """Run a single turn, finalize the UI and turn exceptions into an exit status."""
    try:
        result, _ = session.run_request(request, config)
    except (KeyboardInterrupt, TurnCancelled):
        # Ctrl+C cancels the turn but keeps the session alive.
        session.cancel_turn()
        session.log.end_turn("annullato")
        ui.reset_llm_call()
        ui.end_tool_execution()
        ui.print_system_message("Turno annullato.")
        ui.print_separator()
        return False
    except Exception as error:
        session.log.record_turn_error(error)
        ui.print_error(f"Errore agente: {type(error).__name__}: {error}")
        ui.print_separator()
        return False
    ui.finalize(result)
    # Only the create-only fallback paths: ReAct tools are not an automatic save.
    path_fallback = session.turn_fallback_path
    if path_fallback:
        ui.print_fallback(path_fallback)
    ui.print_separator()
    return True


def _empty_outcome() -> StartupOutcome:
    """Neutral outcome: main retries the probe and does not inherit a runtime."""
    return StartupOutcome(None, None, None, None)


def _prefill_if_ready(
    config: ConfigRuntime,
    runtime_holder: list[OllamaRuntime],
) -> StartupOutcome:
    """
    Detect the environment and, if complete, preload the weights without touching the console.

    Args:
        config: Session alias to align with the Ollama catalog.
        runtime_holder: List shared with main to close the client on Ctrl+C.

    Returns:
        Snapshot and warm runtime, or a None runtime if the wizard is needed.
    """
    environment = detect_environment()
    # Without Ollama/model the POST would fail: the wizard stays on the UI thread.
    if not environment.ready:
        return StartupOutcome(environment, None, None, None)
    alias = align_session_alias(config.model, environment)
    alias = alias_for_power(alias, config)
    host = detect_host()
    runtime = OllamaRuntime(host=host)
    runtime_holder.append(runtime)
    try:
        runtime.preload(alias)
    except Exception:
        # main must not inherit a half-open HTTP client.
        runtime.close()
        runtime_holder.remove(runtime)
        raise
    return StartupOutcome(environment, runtime, host, alias)


def _run_with_splash(
    config: ConfigRuntime,
    single_prompt: str | None,
    ui: ConsoleUI,
    log: EventLog,
) -> StartupOutcome:
    """
    Start the preload in a thread and, if interactive, the splash over the same interval.

    Args:
        config: CLI configuration (model alias).
        single_prompt: If set, no HUD.
        ui: Console for the splash and the wait message.
        log: Destination of the ``startup`` event if the preload raises.

    Returns:
        The worker outcome, or an empty outcome if the thread raised.
    """
    runtime_holder: list[OllamaRuntime] = []
    outcomes: list[StartupOutcome] = []
    errors: list[BaseException] = []

    def work() -> None:
        try:
            outcomes.append(_prefill_if_ready(config, runtime_holder))
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(
        target=work,
        name=PRELOAD_THREAD_NAME,
        daemon=False,
    )
    thread.start()
    if interactive_session(single_prompt):
        ui.print_splash()
    if thread.is_alive():
        ui.print_system_message(PREWARM_IN_PROGRESS_MESSAGE)
    try:
        thread.join()
    except KeyboardInterrupt:
        for runtime in list(runtime_holder):
            runtime.close()
        raise
    if errors:
        error = errors[0]
        log.record_startup("preload", error)
        ui.print_system_message(
            "Preriscaldamento non riuscito: "
            f"{type(error).__name__}: {error}"
        )
        return _empty_outcome()
    if outcomes:
        return outcomes[0]
    return _empty_outcome()


def _prepare_environment(
    config: ConfigRuntime,
    single_prompt: str | None,
    ui: ConsoleUI,
    environment: DetectedEnvironment | None = None,
    log: EventLog | None = None,
) -> DetectedEnvironment | None:
    """
    Check Homebrew/Ollama/Gemma 4 and, if needed, start the STARTUP.MD wizard.

    Args:
        config: CLI runtime; `model` may be realigned with the catalog.
        single_prompt: If set, the wizard is forbidden (exit 1 if anything is missing).
        ui: Console for errors and consents.
        environment: Snapshot already detected in the startup thread; None repeats the probe.
        log: Destination of the failed-startup events.

    Returns:
        Ready snapshot, or None when startup must end with exit code 1.
    """
    observation = log if log is not None else EventLog()
    if environment is None:
        environment = detect_environment()
    if not environment.ready:
        if not interactive_session(single_prompt):
            ui.print_error(non_interactive_setup_message())
            observation.record_startup(
                "setup",
                RuntimeError(non_interactive_setup_message()),
            )
            return None
        try:
            environment = run_wizard(ui)
        except KeyboardInterrupt:
            observation.record_startup(
                "setup",
                KeyboardInterrupt("Setup interrotto."),
            )
            ui.print_error("Setup interrotto.")
            return None
        except SetupAborted as error:
            observation.record_startup("setup", error)
            ui.print_error(str(error))
            return None
    effective_alias = align_session_alias(config.model, environment)
    if effective_alias != config.model:
        ui.print_system_message(
            f"Modello {config.model} assente in Ollama: uso {effective_alias}. "
            "Cambia con /model."
        )
        config.model = effective_alias
    return environment


def _run_session(
    config: ConfigRuntime,
    single_prompt: str | None,
    ui: ConsoleUI,
    log: EventLog | None = None,
    folder: Path | None = None,
) -> int:
    """Run the single prompt or the interactive loop, always releasing resources."""
    documents_folder = (folder or Path.cwd()).resolve()
    file_service = WorkspaceFileService(FileConfig(workspace=documents_folder))
    ui.set_file_separator(FILE_SEPARATOR)
    ui.set_verbose(config.verbose)
    owned = log is None
    observation = log if log is not None else EventLog()
    observation.start_session(
        pid=os.getpid(),
        python=sys.version.split()[0],
        model=config.model,
    )
    session: AgentSession | None = None
    try:
        outcome = _run_with_splash(config, single_prompt, ui, observation)
        environment = _prepare_environment(
            config,
            single_prompt,
            ui,
            outcome.environment,
            observation,
        )
        if environment is None:
            if outcome.runtime is not None:
                outcome.runtime.close()
            return 1
        if outcome.runtime is not None:
            session = AgentSession(
                ui,
                file_service=file_service,
                ollama_runtime=outcome.runtime,
                host=outcome.host,
                log=observation,
            )
            loaded_alias = outcome.alias or alias_for_power(
                config.model, config
            )
            session.announce_prewarm(loaded_alias)
        else:
            session = AgentSession(
                ui,
                file_service=file_service,
                host=detect_host(),
                log=observation,
            )
            session.prewarm_model(
                alias_for_power(config.model, config)
            )
        ui.thermal_bar.start()
        ui.print_welcome(
            config.verbose,
            config.full_verification,
            config.burn_protection,
            config.thinking,
            model_label=_model_label(config),
            folder=str(documents_folder),
        )
        if single_prompt is not None:
            if not single_prompt.strip():
                ui.print_system_message("Prompt vuoto: nessuna elaborazione.")
                return 0
            return 0 if _process_turn(single_prompt, session, config, ui) else 1

        while True:
            try:
                request = ui.read_user_request(_model_label(config))
            except (EOFError, KeyboardInterrupt):
                ui.print_system_message("Sessione terminata.")
                break
            if not request:
                continue
            if request.lower() in EXIT_COMMANDS:
                ui.print_system_message("Sessione terminata.")
                break
            if _handle_command(request, config, ui, session):
                continue
            _process_turn(request, session, config, ui)
        return 0
    finally:
        observation.end_session()
        if session is not None:
            session.close()
        ui.thermal_bar.stop()
        if owned:
            observation.close()


def main() -> None:
    """CLI entry point that propagates the exit code to the process."""
    config, single_prompt, folder = _parse_args()
    log = ProcessLog()
    log.install_process_hooks()
    log.start_stall_check()
    ui = ConsoleUI()
    try:
        sys.exit(
            _run_session(
                config,
                single_prompt,
                ui,
                log=log,
                folder=folder,
            )
        )
    finally:
        log.close()
