"""Test wizard di primo avvio senza rete né Homebrew reali."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from assistente.bootstrap import OLLAMA_LINUX_INSTALL_ARGV
from assistente.cli import _esegui_sessione
from assistente.config import ConfigRuntime
from assistente.inference_profiles import (
    GIBIBYTE,
    resolve_model_variant,
    select_installed_tag,
    variant_is_installed,
)
from assistente.startup import (
    HOMEBREW_BIN_APPLE,
    HOMEBREW_INSTALL_ARGV,
    AmbienteRilevato,
    SetupAborted,
    StartupHost,
    alias_suggerito,
    allinea_alias_sessione,
    esegui_wizard,
    fascia_modello,
    messaggio_setup_non_interattivo,
    parse_model_selection,
    rileva_ambiente,
    sessione_interattiva,
)

pytestmark = pytest.mark.unit

_MEMORY_8_GIB = 8 * GIBIBYTE
_MEMORY_16_GIB = 16 * GIBIBYTE
_MEMORY_64_GIB = 64 * GIBIBYTE
_BREW = HOMEBREW_BIN_APPLE
_OLLAMA = _BREW.parent / "ollama"


class _Orologio:
    """Orologio deterministico per non aspettare i 20s di healthcheck."""

    def __init__(self) -> None:
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t

    def sleep(self, delta: float) -> None:
        self.t += delta


def _run_ok(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")


def _ui() -> MagicMock:
    ui = MagicMock()
    ui.chiedi_conferma.return_value = False
    ui.chiedi_testo.return_value = ""
    return ui


def _host(**overrides: object) -> StartupHost:
    orologio = _Orologio()
    valori: dict[str, object] = {
        "which": lambda _name: None,
        "path_is_file": lambda _path: False,
        "run": _run_ok,
        "popen": MagicMock(),
        "ping_ollama": lambda: False,
        "list_models": lambda: set(),
        "memory_bytes": lambda: _MEMORY_16_GIB,
        "machine": lambda: "arm64",
        "system": lambda: "Darwin",
        "environ": {"PATH": "/usr/bin"},
        "sleep": orologio.sleep,
        "monotonic": orologio.monotonic,
    }
    valori.update(overrides)
    return StartupHost(**valori)  # type: ignore[arg-type]


def test_brew_assente_senza_consenso_non_lancia_subprocess() -> None:
    """Rifiuto Homebrew: nessun installer e SetupAborted."""
    run = MagicMock()
    ui = _ui()
    with pytest.raises(SetupAborted, match="Homebrew"):
        esegui_wizard(ui, _host(run=run))
    run.assert_not_called()


def test_brew_presente_ollama_assente_esegue_brew_install() -> None:
    """Con brew e senza ollama il wizard costruisce `brew install ollama`."""
    comandi: list[list[str]] = []

    def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        comandi.append(list(argv))
        return _run_ok(argv)

    ui = _ui()
    ui.chiedi_conferma.return_value = True
    with pytest.raises(SetupAborted, match="Ollama"):
        esegui_wizard(
            ui,
            _host(
                which=lambda name: str(_BREW) if name == "brew" else None,
                path_is_file=lambda path: path == _BREW,
                run=run,
            ),
        )
    normalizzati = [[str(part) for part in comando] for comando in comandi]
    assert [_BREW.as_posix(), "install", "ollama"] in normalizzati
    assert HOMEBREW_INSTALL_ARGV not in comandi


def test_fascia_modello_per_ram_8_16_64() -> None:
    """Le due taglie Gemma 4 cambiano fascia al variare della RAM."""
    fast = resolve_model_variant("fast")
    default = resolve_model_variant("default")
    assert fascia_modello(fast, _MEMORY_8_GIB) == "sconsigliato"
    assert fascia_modello(default, _MEMORY_8_GIB) == "sconsigliato"
    assert alias_suggerito(_MEMORY_8_GIB) is None
    assert fascia_modello(fast, _MEMORY_16_GIB) == "consigliato"
    assert fascia_modello(default, _MEMORY_16_GIB) == "consigliato"
    assert alias_suggerito(_MEMORY_16_GIB) == "default"
    assert fascia_modello(fast, _MEMORY_64_GIB) == "consigliato"
    assert fascia_modello(default, _MEMORY_64_GIB) == "consigliato"
    assert alias_suggerito(_MEMORY_64_GIB) == "default"


def test_parse_model_selection_indici_e_alias() -> None:
    """Accetta `1 2` e nomi; rifiuta token sconosciuti."""
    aliases = ("fast", "default")
    assert parse_model_selection("1 2", aliases) == ("fast", "default")
    assert parse_model_selection("2", aliases) == ("default",)
    assert parse_model_selection("fast default", aliases) == ("fast", "default")
    assert parse_model_selection("", aliases) == ()
    with pytest.raises(ValueError):
        parse_model_selection("llama", aliases)


def test_select_installed_tag_dopo_pull_fittizio() -> None:
    """Dopo un pull del tag libreria, il preferito E4B risolve il fallback."""
    catalogo = {"gemma4"}
    assert variant_is_installed("gemma4:e4b", catalogo)
    assert select_installed_tag("gemma4:e4b", catalogo) == "gemma4"
    assert not variant_is_installed("gemma4:12b", catalogo)


def test_prompt_senza_ambiente_non_avvia_wizard(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--prompt` con dipendenze mancanti esce 1 e non chiama il wizard."""
    incompleto = AmbienteRilevato(
        brew_path=None,
        ollama_cli=None,
        ollama_api=False,
        available_model_names=frozenset(),
        memory_bytes=_MEMORY_16_GIB,
        machine="arm64",
    )
    monkeypatch.setattr("assistente.cli.rileva_ambiente", lambda: incompleto)
    wizard = MagicMock()
    monkeypatch.setattr("assistente.cli.esegui_wizard", wizard)
    sessione_cls = MagicMock()
    monkeypatch.setattr("assistente.cli.SessioneAgente", sessione_cls)
    ui = MagicMock()
    ui.barra_termica = MagicMock()
    codice = _esegui_sessione(ConfigRuntime(), "ciao mondo", ui)
    assert codice == 1
    wizard.assert_not_called()
    sessione_cls.assert_not_called()
    ui.stampa_splash.assert_not_called()
    assert ui.stampa_errore.call_args.args[0] == messaggio_setup_non_interattivo()


def test_allinea_alias_se_manca_default() -> None:
    """Se default non è in catalogo si usa il primo alias installato."""
    ambiente = AmbienteRilevato(
        brew_path=_BREW,
        ollama_cli=_OLLAMA,
        ollama_api=True,
        available_model_names=frozenset({"gemma4:e4b"}),
        memory_bytes=_MEMORY_16_GIB,
        machine="arm64",
    )
    assert ambiente.pronto
    assert allinea_alias_sessione("default", ambiente) == "fast"


def test_sessione_interattiva_false_con_prompt() -> None:
    """`--prompt` non è mai considerato TTY per il wizard."""
    assert sessione_interattiva("testo", stdin_tty=True) is False
    assert sessione_interattiva(None, stdin_tty=True) is True
    assert sessione_interattiva(None, stdin_tty=False) is False


def test_rileva_ambiente_usa_path_canonico_brew() -> None:
    """Senza brew nel PATH resta valido /opt/homebrew/bin/brew."""
    ambiente = rileva_ambiente(
        _host(
            path_is_file=lambda path: path == _BREW,
            ping_ollama=lambda: True,
            list_models=lambda: {"gemma4:12b"},
        )
    )
    assert ambiente.brew_path == _BREW
    assert ambiente.has_modello


def test_wizard_pull_dei_modelli_scelti() -> None:
    """La selezione `default` chiama pull sul tag della matrice."""
    pullati: list[str] = []
    catalogo: set[str] = set()

    def path_is_file(path: Path) -> bool:
        return path in {_BREW, _OLLAMA}

    def pull_model(tag: str, _progress: object) -> None:
        pullati.append(tag)
        catalogo.add(tag)

    ui = _ui()
    ui.chiedi_testo.return_value = "default"
    ambiente = esegui_wizard(
        ui,
        _host(
            which=lambda name: str(_BREW) if name == "brew" else str(_OLLAMA),
            path_is_file=path_is_file,
            ping_ollama=lambda: True,
            list_models=lambda: set(catalogo),
            pull_model=pull_model,
            environ={"PATH": "/opt/homebrew/bin"},
        ),
    )
    assert pullati == ["gemma4:12b"]
    assert ambiente.has_modello
    ui.chiedi_conferma.assert_not_called()


def test_selezione_invalida_riprova_poi_pull() -> None:
    """Un token errato non abortisce: al secondo tentativo il pull parte."""
    pullati: list[str] = []
    catalogo: set[str] = set()
    ui = _ui()
    ui.chiedi_testo.side_effect = ["llama", "default"]
    esegui_wizard(
        ui,
        _host(
            which=lambda name: str(_BREW) if name == "brew" else str(_OLLAMA),
            path_is_file=lambda path: path in {_BREW, _OLLAMA},
            ping_ollama=lambda: True,
            list_models=lambda: set(catalogo),
            pull_model=lambda tag, _p: (pullati.append(tag), catalogo.add(tag)),
            environ={"PATH": "/opt/homebrew/bin"},
        ),
    )
    assert pullati == ["gemma4:12b"]
    ui.stampa_errore.assert_called()


def test_linux_pronto_senza_homebrew() -> None:
    """Su Linux Ollama+Gemma 4 bastano: brew non è un prerequisito."""
    ollama = Path("/usr/bin/ollama")
    ambiente = rileva_ambiente(
        _host(
            system=lambda: "Linux",
            machine=lambda: "x86_64",
            which=lambda name: str(ollama) if name == "ollama" else None,
            path_is_file=lambda path: path == ollama,
            ping_ollama=lambda: True,
            list_models=lambda: {"gemma4:12b"},
        )
    )
    assert ambiente.brew_path is None
    assert ambiente.pronto
    assert ambiente.pronto


def test_linux_wizard_senza_ollama_usa_installer_ufficiale() -> None:
    """Senza CLI Ollama su Linux non si tocca Homebrew."""
    comandi: list[list[str]] = []

    def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        comandi.append(list(argv))
        return _run_ok(argv)

    ui = _ui()
    ui.chiedi_conferma.return_value = True
    with pytest.raises(SetupAborted, match="Ollama"):
        esegui_wizard(
            ui,
            _host(
                system=lambda: "Linux",
                machine=lambda: "x86_64",
                run=run,
            ),
        )
    assert OLLAMA_LINUX_INSTALL_ARGV in comandi
    assert HOMEBREW_INSTALL_ARGV not in comandi


def test_linux_wizard_rifiuto_install_non_lancia_subprocess() -> None:
    """Consenso negato su Linux: nessun installer."""
    run = MagicMock()
    ui = _ui()
    with pytest.raises(SetupAborted, match="Ollama"):
        esegui_wizard(ui, _host(system=lambda: "Linux", machine=lambda: "x86_64", run=run))
    run.assert_not_called()
