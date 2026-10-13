"""Wizard di primo avvio: Ollama su macOS/Linux e modelli Gemma 4."""

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

from assistente.bootstrap import (
    BREW_SH_URL,
    HOMEBREW_BIN_APPLE,
    HOMEBREW_BIN_INTEL,
    HOMEBREW_INSTALL_ARGV,
    SetupAborted,
    avvia_ollama,
    installa_homebrew,
    installa_ollama_con_brew,
    installa_ollama_linux,
)
from assistente.host import DEFAULT_OLLAMA_BASE_URL
from assistente.inference_profiles import (
    GIBIBYTE,
    TIGHT_CONTEXT_LEFTOVER_GB,
    ModelVariant,
    model_variants,
    normalize_ollama_names,
    physical_memory_bytes,
    resolve_model_variant,
    variant_is_installed,
)
from assistente.ollama_runtime import OllamaRuntime

FAMIGLIA_MODELLI = "Gemma 4"
MAX_TENTATIVI_SELEZIONE_MODELLI = 5
MESSAGGIO_SETUP_NON_INTERATTIVO = (
    f"Ambiente incompleto (Ollama o un modello {FAMIGLIA_MODELLI}). "
    "Avvia `main.py` in un terminale interattivo oppure segui STARTUP.MD."
)
_EXPORT_SHELLENV = re.compile(
    r'^export\s+([A-Za-z_][A-Za-z0-9_]*)="(.*)";?\s*$'
)
_VAR_BRACE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_VAR_SIMPLE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
FasciaModello = Literal["consigliato", "stretto", "sconsigliato"]
ProgressCallback = Callable[[str], None]


class StartupUI(Protocol):
    """Superficie minima usata dal wizard (ConsoleUI o fake nei test)."""

    def stampa_messaggio_sistema(self, messaggio: str) -> None: ...

    def stampa_errore(self, messaggio: str) -> None: ...

    def chiedi_conferma(self, domanda: str) -> bool: ...

    def chiedi_testo(self, domanda: str) -> str: ...

    def aggiorna_progresso(self, messaggio: str) -> None: ...


@dataclass(frozen=True, slots=True)
class AmbienteRilevato:
    """Snapshot delle dipendenze richieste da STARTUP.MD."""

    brew_path: Path | None
    ollama_cli: Path | None
    ollama_api: bool
    available_model_names: frozenset[str]
    memory_bytes: int
    machine: str
    alias_installati: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "alias_installati",
            alias_modello_installati(self.available_model_names),
        )

    @property
    def brew_ok(self) -> bool:
        """True se il rilevamento ha risolto un path brew."""
        return self.brew_path is not None

    @property
    def ollama_ok(self) -> bool:
        """True se CLI e API Ollama sono entrambi disponibili."""
        return self.ollama_cli is not None and self.ollama_api

    @property
    def has_modello(self) -> bool:
        """True se almeno un alias della matrice è nel catalogo Ollama."""
        return bool(self.alias_installati)

    @property
    def pronto(self) -> bool:
        """True se CLI+API Ollama e almeno un modello della matrice ci sono."""
        return self.ollama_ok and self.has_modello


@dataclass
class StartupHost:
    """
    Collaboratori iniettabili per filesystem, processi e Ollama.

    Un solo client HTTP viene creato al primo ping/list/pull e riusato
    per tutto il wizard, così il healthcheck non apre un TCP a ogni poll.
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
        Cerca un eseguibile nel PATH del host, non in `os.environ` globale.

        I test passano un `environ` isolato: `shutil.which` deve usarlo.
        I lambda dei test non accettano `path=` e cadono nel ramo TypeError.
        """
        path_value = self.environ.get("PATH")
        try:
            return self.which(name, path=path_value)
        except TypeError:
            return self.which(name)

    def ping(self) -> bool:
        """Healthcheck `/api/version` riusando il client HTTP del host."""
        if self.ping_ollama is not None:
            return self.ping_ollama()
        return self._ollama().ping()

    def list_catalog(self) -> set[str]:
        """Catalogo `/api/tags` sullo stesso client del ping."""
        if self.list_models is not None:
            return self.list_models()
        return self._ollama().list_available_model_names()

    def pull(self, tag: str, on_progress: ProgressCallback) -> None:
        """Pull streamato; errori di rete diventano SetupAborted."""
        if self.pull_model is not None:
            self.pull_model(tag, on_progress)
            return
        try:
            self._ollama().pull_model(tag, on_progress)
        except (httpx.HTTPError, RuntimeError) as error:
            raise SetupAborted(f"Pull {tag} fallito: {error}.") from error

    def close(self) -> None:
        """Chiude il client Ollama posseduto, senza unload dei pesi di sessione."""
        if self._runtime is None:
            return
        self._runtime.close(unload=False)
        self._runtime = None

    def _ollama(self) -> OllamaRuntime:
        if self._runtime is None:
            self._runtime = self.runtime_factory()
        return self._runtime


def messaggio_setup_non_interattivo() -> str:
    """Testo per `--prompt` / stdin non TTY quando l'ambiente è incompleto."""
    return MESSAGGIO_SETUP_NON_INTERATTIVO


def sessione_interattiva(
    prompt_singolo: str | None,
    stdin_tty: bool | None = None,
) -> bool:
    """
    True quando il wizard può chiedere consensi.

    `--prompt` vieta il wizard anche se stdin è un TTY.
    """
    if prompt_singolo is not None:
        return False
    if stdin_tty is None:
        stdin_tty = sys.stdin.isatty()
    return stdin_tty


def alias_modello_installati(available_names: Iterable[str]) -> tuple[str, ...]:
    """Alias della matrice presenti nel catalogo, in ordine fast → default."""
    names = normalize_ollama_names(set(available_names))
    return tuple(
        variant.alias
        for variant in model_variants()
        if variant_is_installed(variant.tag, names)
    )


def fascia_modello(variant: ModelVariant, memory_bytes: int) -> FasciaModello:
    """
    Classifica un artefatto rispetto alla RAM rilevata.

    RAM sconosciuta (≤0): nessuna fascia punitiva.
    """
    if memory_bytes <= 0:
        return "consigliato"
    if not variant.fits_memory(memory_bytes):
        return "sconsigliato"
    leftover_gb = (memory_bytes / GIBIBYTE) - variant.size_gb
    if leftover_gb < TIGHT_CONTEXT_LEFTOVER_GB:
        return "stretto"
    return "consigliato"


def alias_suggerito(memory_bytes: int) -> str | None:
    """
    Suggerisce default se è consigliato, altrimenti il primo consigliato,
    altrimenti il primo stretto; None se tutti sconsigliati.
    """
    fasce = {
        variant.alias: fascia_modello(variant, memory_bytes)
        for variant in model_variants()
    }
    if fasce["default"] == "consigliato":
        return "default"
    stretto_fallback: str | None = None
    for variant in model_variants():
        fascia = fasce[variant.alias]
        if fascia == "consigliato":
            return variant.alias
        if fascia == "stretto" and stretto_fallback is None:
            stretto_fallback = variant.alias
    return stretto_fallback


def parse_model_selection(raw: str, aliases: tuple[str, ...]) -> tuple[str, ...]:
    """
    Interpreta indici 1-based, alias, o un misto, senza duplicati.

    Raises:
        ValueError: Token sconosciuto o indice fuori range.
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


def applica_brew_shellenv(output: str, environ: MutableMapping[str, str]) -> None:
    """Applica le `export` di `brew shellenv` all'ambiente del processo."""
    for line in output.splitlines():
        match = _EXPORT_SHELLENV.match(line.strip())
        if match is None:
            continue
        key, value = match.group(1), match.group(2)
        environ[key] = _espandi_shell_value(value, environ)


def trova_brew(host: StartupHost) -> Path | None:
    """Risolve Homebrew nel PATH del processo o nei path canonici macOS."""
    # Linuxbrew non è un prerequisito: su Linux si ignora anche se presente.
    if host.system() != "Darwin":
        return None
    which_result = host.find_on_path("brew")
    candidati = []
    if which_result:
        candidati.append(Path(which_result))
    candidati.extend((HOMEBREW_BIN_APPLE, HOMEBREW_BIN_INTEL))
    return _primo_eseguibile(host, candidati)


def trova_ollama_cli(host: StartupHost, brew_path: Path | None) -> Path | None:
    """Risolve `ollama` nel PATH o nella stessa `bin/` di Homebrew."""
    which_result = host.find_on_path("ollama")
    candidati: list[Path] = []
    if which_result:
        candidati.append(Path(which_result))
    if brew_path is not None:
        candidati.append(brew_path.parent / "ollama")
    return _primo_eseguibile(host, candidati)


def rileva_ambiente(host: StartupHost | None = None) -> AmbienteRilevato:
    """
    Legge brew, Ollama e catalogo modelli.

    Se `host` è None ne crea uno temporaneo e lo chiude in uscita, così
    `main.py` con ambiente già pronto non lascia socket HTTP aperti.
    """
    owns_host = host is None
    attivo = host or StartupHost()
    try:
        brew_path = trova_brew(attivo)
        if brew_path is not None:
            _assicura_brew_nel_path(attivo, brew_path)
        ollama_cli = trova_ollama_cli(attivo, brew_path)
        ollama_api = attivo.ping()
        names = attivo.list_catalog() if ollama_api else set()
        return AmbienteRilevato(
            brew_path=brew_path,
            ollama_cli=ollama_cli,
            ollama_api=ollama_api,
            available_model_names=frozenset(names),
            memory_bytes=attivo.memory_bytes(),
            machine=attivo.machine(),
        )
    finally:
        if owns_host:
            attivo.close()


def allinea_alias_sessione(alias_richiesto: str, ambiente: AmbienteRilevato) -> str:
    """Usa l'alias CLI se installato, altrimenti il primo della matrice presente."""
    installati = ambiente.alias_installati
    if alias_richiesto in installati:
        return alias_richiesto
    return installati[0] if installati else alias_richiesto


def esegui_wizard(ui: StartupUI, host: StartupHost | None = None) -> AmbienteRilevato:
    """
    Esegue solo i passi mancanti di STARTUP.MD.

    Raises:
        SetupAborted: Consenso negato o passo fallito.
    """
    attivo = host or StartupHost()
    try:
        return _esegui_passi(ui, attivo)
    finally:
        attivo.close()


def _esegui_passi(ui: StartupUI, host: StartupHost) -> AmbienteRilevato:
    ambiente = rileva_ambiente(host)
    sistema = host.system()
    # L'avviso Intel vale solo su macOS: Linux x86_64 è un target di prima classe.
    if sistema == "Darwin" and ambiente.machine not in {"arm64", "aarch64"}:
        ui.stampa_messaggio_sistema(
            f"CPU {ambiente.machine}: il target è Apple Silicon; si prosegue comunque."
        )
    if not ambiente.ollama_ok:
        if sistema == "Darwin":
            _assicura_ollama_darwin(ui, host, ambiente)
        elif sistema == "Linux":
            _assicura_ollama_linux(ui, host, ambiente)
        else:
            raise SetupAborted(
                "Questa piattaforma non ha un wizard di installazione. "
                f"Installa Ollama, avvialo e scarica un {FAMIGLIA_MODELLI} (STARTUP.MD)."
            )
        ambiente = rileva_ambiente(host)
        if not ambiente.ollama_ok:
            raise SetupAborted(
                "Ollama non è raggiungibile su "
                f"{DEFAULT_OLLAMA_BASE_URL}. Verifica il servizio o `ollama serve`."
            )
    if len(ambiente.alias_installati) < len(model_variants()):
        _seleziona_e_scarica_modelli(ui, host, ambiente)
        ambiente = rileva_ambiente(host)
    if not ambiente.has_modello:
        raise SetupAborted(
            f"Nessun modello {FAMIGLIA_MODELLI} della matrice è installato. "
            "Ripeti l'avvio e scegli almeno un alias, oppure "
            "`ollama pull gemma4:12b`."
        )
    ui.stampa_messaggio_sistema(f"Ambiente pronto: Ollama e {FAMIGLIA_MODELLI}.")
    return ambiente


def _assicura_ollama_darwin(
    ui: StartupUI,
    host: StartupHost,
    ambiente: AmbienteRilevato,
) -> None:
    """Homebrew + brew install solo se manca la CLI; poi avvio servizio/serve."""
    if not ambiente.brew_ok:
        installa_homebrew(ui, host)
        brew_path = trova_brew(host)
        if brew_path is None:
            raise SetupAborted(
                "Homebrew risulta ancora assente dopo l'installer. "
                f"Installa da {BREW_SH_URL} e rilancia."
            )
        _assicura_brew_nel_path(host, brew_path)
        ui.stampa_messaggio_sistema(
            'Per le prossime shell: eval "$(/opt/homebrew/bin/brew shellenv)" '
            "in ~/.zprofile"
        )
        ambiente = rileva_ambiente(host)
    if ambiente.ollama_cli is None:
        installa_ollama_con_brew(ui, host, ambiente.brew_path)
        ambiente = rileva_ambiente(host)
    avvia_ollama(
        ui,
        host,
        ambiente.ollama_cli or trova_ollama_cli(host, ambiente.brew_path),
        brew_path=trova_brew(host),
        apply_server_env=True,
    )


def _assicura_ollama_linux(
    ui: StartupUI,
    host: StartupHost,
    ambiente: AmbienteRilevato,
) -> None:
    """Installer ufficiale se manca la CLI; systemd oppure ollama serve."""
    if ambiente.ollama_cli is None:
        installa_ollama_linux(ui, host)
        ambiente = rileva_ambiente(host)
    avvia_ollama(
        ui,
        host,
        ambiente.ollama_cli or trova_ollama_cli(host, None),
        apply_server_env=True,
    )


def _primo_eseguibile(host: StartupHost, candidati: Iterable[Path]) -> Path | None:
    visti: set[Path] = set()
    for candidato in candidati:
        if candidato in visti:
            continue
        visti.add(candidato)
        if host.path_is_file(candidato):
            return candidato
    return None


def _directory_in_path(directory: str, path_value: str) -> bool:
    return directory in path_value.split(os.pathsep)


def _espandi_shell_value(value: str, environ: Mapping[str, str]) -> str:
    """Espande `${VAR}` e `$VAR` nell'ordine in cui brew emette le export."""

    def from_braces(match: re.Match[str]) -> str:
        return environ.get(match.group(1), match.group(0))

    def from_simple(match: re.Match[str]) -> str:
        return environ.get(match.group(1), match.group(0))

    expanded = _VAR_BRACE.sub(from_braces, value)
    return _VAR_SIMPLE.sub(from_simple, expanded)


def _assicura_brew_nel_path(host: StartupHost, brew_path: Path) -> None:
    """
    Antepone la bin di Homebrew e applica `brew shellenv` solo se manca dal PATH.

    Così `main.py` con brew già configurato non lancia un subprocess a ogni avvio.
    """
    brew_bin_dir = str(brew_path.parent)
    current = host.environ.get("PATH", "")
    if _directory_in_path(brew_bin_dir, current):
        return
    host.environ["PATH"] = (
        brew_bin_dir if not current else f"{brew_bin_dir}{os.pathsep}{current}"
    )
    try:
        risultato = host.run(
            [str(brew_path), "shellenv"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return
    if risultato.returncode != 0:
        return
    applica_brew_shellenv(risultato.stdout, host.environ)


def _seleziona_e_scarica_modelli(
    ui: StartupUI,
    host: StartupHost,
    ambiente: AmbienteRilevato,
) -> None:
    aliases = tuple(variant.alias for variant in model_variants())
    ui.stampa_messaggio_sistema(
        _testo_catalogo_modelli(ambiente, alias_suggerito(ambiente.memory_bytes))
    )
    scelti = _leggi_selezione_modelli(ui, aliases)
    if not scelti:
        if ambiente.has_modello:
            ui.stampa_messaggio_sistema(
                f"Nessun nuovo pull. Si userà un {FAMIGLIA_MODELLI} già presente in Ollama."
            )
            return
        raise SetupAborted(
            f"Nessun modello selezionato e nessun {FAMIGLIA_MODELLI} della matrice è installato."
        )
    presenti = normalize_ollama_names(set(ambiente.available_model_names))
    for alias in scelti:
        variant = resolve_model_variant(alias)
        if variant_is_installed(variant.tag, presenti):
            ui.stampa_messaggio_sistema(f"{alias} è già installato, skip pull.")
            continue
        ui.aggiorna_progresso(f"Pull {variant.tag}...")
        host.pull(variant.tag, ui.aggiorna_progresso)
        presenti.add(variant.tag)


def _leggi_selezione_modelli(
    ui: StartupUI,
    aliases: tuple[str, ...],
) -> tuple[str, ...]:
    domanda = (
        "Quali modelli scaricare? Indici o alias (es. 1 2 oppure fast default). "
        "Invio vuoto = nessuno."
    )
    ultimo_errore = "Selezione non valida."
    for _tentativo in range(MAX_TENTATIVI_SELEZIONE_MODELLI):
        grezzo = ui.chiedi_testo(domanda)
        try:
            return parse_model_selection(grezzo, aliases)
        except ValueError as error:
            ultimo_errore = str(error)
            ui.stampa_errore(ultimo_errore)
    raise SetupAborted(ultimo_errore)


def _testo_catalogo_modelli(ambiente: AmbienteRilevato, suggerito: str | None) -> str:
    if ambiente.memory_bytes > 0:
        ram = f"RAM rilevata: {ambiente.memory_bytes / GIBIBYTE:.0f} GB."
    else:
        ram = "RAM non rilevata: nessuna fascia punitiva."
    righe = [
        f"Modelli {FAMIGLIA_MODELLI} (solo questa famiglia). "
        "Suggerito evidenziato, non preselezionato.",
        ram,
    ]
    catalogo = normalize_ollama_names(set(ambiente.available_model_names))
    for indice, variant in enumerate(model_variants(), start=1):
        fascia = fascia_modello(variant, ambiente.memory_bytes)
        presente = variant_is_installed(variant.tag, catalogo)
        marker = " ← suggerito" if variant.alias == suggerito else ""
        stato = "già presente" if presente else "da scaricare"
        righe.append(
            f"{indice}) {variant.alias} · {variant.tag} · ~{variant.size_gb:.1f} GB "
            f"· {fascia} · {stato}{marker}"
        )
    return "\n".join(righe)
