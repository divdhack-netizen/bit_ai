"""Riga di comando e loop interattivo."""

from __future__ import annotations

import argparse
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from assistente.config import (
    ConfigRuntime,
    ERRORI_SLASH_DEFAULT,
    FileConfig,
    MODELLI_DISPONIBILI,
    SEPARATORE_FILE,
    risolvi_tag_ollama,
)
from assistente.console import ConsoleUI
from assistente.file_service import WorkspaceFileService
from assistente.host import HostCapabilities, detect_host
from assistente.inference_profiles import (
    normalize_ollama_names,
    resolve_model_variant,
    variant_is_installed,
)
from assistente.ollama_runtime import OllamaRuntime
from assistente.politica_energia import alias_per_alimentazione
from assistente.registro_processo import RegistroEventi, RegistroProcesso
from assistente.session import SessioneAgente, TurnoAnnullato
from assistente.startup import (
    AmbienteRilevato,
    SetupAborted,
    allinea_alias_sessione,
    esegui_wizard,
    messaggio_setup_non_interattivo,
    rileva_ambiente,
    sessione_interattiva,
)

COMANDI_USCITA = frozenset({"exit", "quit", "esci"})
COMANDI_CLEAR = frozenset({"/clear", "/reset", "/nuova"})
MESSAGGIO_PRERISCALDAMENTO_IN_CORSO = "Preriscaldamento modello in corso..."
NOME_THREAD_PRELOAD = "assistente-preload"


@dataclass(slots=True)
class EsitoAvvio:
    """Esito del worker di preload: I/O Ollama senza oggetti UI."""

    ambiente: AmbienteRilevato | None
    runtime: OllamaRuntime | None
    host: HostCapabilities | None
    alias: str | None

_COMANDO_TOGGLE = re.compile(
    r"^/(?P<cmd>verbose|v|check|c|burn_protection|bp|thinking|t)"
    r"(?:\s+(?P<azione>on|off|toggle))?$",
    re.IGNORECASE,
)
_COMANDO_MODEL = re.compile(r"^/model(?:\s+(\w+))?$", re.IGNORECASE)
_COMANDO_ERRORI = re.compile(r"^/errori(?:\s+(\d+))?$", re.IGNORECASE)

_ALIAS_TOGGLE = {
    "v": "verbose",
    "c": "check",
    "bp": "burn_protection",
    "t": "thinking",
}
_ETICHETTE_TOGGLE = {
    "verbose": "Verbose",
    "verifica_completa": "Verifica completa",
    "burn_protection": "Burn protection",
    "thinking": "Thinking",
}
_AZIONI = {"on": True, "off": False}


def _parse_argomenti() -> tuple[ConfigRuntime, str | None, Path]:
    """Converte gli argomenti CLI in configurazione, prompt e cartella documenti."""
    defaults = ConfigRuntime()
    parser = argparse.ArgumentParser(
        description=(
            "Agente locale closed-corpus: risponde solo dai documenti della "
            "cartella (txt, md, pdf, docx, immagini) e scrive solo file .txt."
        )
    )
    parser.add_argument(
        "--cartella",
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
        default=defaults.verifica_completa,
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
        choices=sorted(MODELLI_DISPONIBILI),
        default=defaults.modello,
    )
    parser.add_argument("--prompt", default=None)
    args = parser.parse_args()
    config = ConfigRuntime(
        verbose=args.verbose,
        verifica_completa=args.check,
        burn_protection=args.burn_protection,
        modello=args.model,
        thinking=args.thinking,
        auto_fast_low_power=args.auto_fast,
    )
    cartella = (args.cartella or Path.cwd()).expanduser().resolve()
    if not cartella.is_dir():
        parser.error(f"cartella documenti inesistente: {cartella}")
    return config, args.prompt, cartella


def _toggle(valore: bool, azione: str | None) -> bool:
    """Applica on/off/toggle a un flag; azioni sconosciute lasciano il valore."""
    if azione is None or azione.lower() == "toggle":
        return not valore
    return _AZIONI.get(azione.lower(), valore)


def _attr_toggle(cmd: str) -> str:
    """Mappa alias CLI brevi al nome dell'attributo ConfigRuntime."""
    nome = _ALIAS_TOGGLE.get(cmd, cmd)
    return "verifica_completa" if nome == "check" else nome


def _etichetta_modello(config: ConfigRuntime) -> str:
    """Formatta alias effettivo (LPM → fast), quant e tag Ollama per il prompt."""
    alias = alias_per_alimentazione(config.modello, config)
    variant = resolve_model_variant(alias)
    return f"{variant.alias} · {variant.quantization} · {variant.tag}"


def _catalogo_sessione(sessione: SessioneAgente | None) -> set[str]:
    """Nomi Ollama normalizzati visibili dalla sessione; vuoto senza runtime."""
    if sessione is None:
        return set()
    return normalize_ollama_names(set(sessione.nomi_modelli_installati()))


def _testo_modello_assente(alias: str) -> str:
    """Prima riga del prompt /model quando l'artefatto non è in Ollama."""
    variant = resolve_model_variant(alias)
    return (
        f"Il modello {variant.alias} ({variant.tag}, ~{variant.size_gb:.1f} GB) "
        "non è presente."
    )


def _assicura_modello_installato(
    alias: str,
    ui: ConsoleUI,
    sessione: SessioneAgente | None,
) -> bool:
    """
    True se l'alias è usabile: già in catalogo, oppure pull confermato.

    Senza sessione (test CLI) non interroga Ollama e non chiede il download.
    """
    if sessione is None:
        return True
    variant = resolve_model_variant(alias)
    if variant_is_installed(variant.tag, _catalogo_sessione(sessione)):
        return True
    ui.stampa_messaggio_sistema(_testo_modello_assente(alias))
    if not ui.chiedi_conferma_download("Vuoi scaricarlo?"):
        ui.stampa_messaggio_sistema(
            "Download annullato. Resta il modello precedente."
        )
        return False
    try:
        sessione.scarica_modello(variant.tag)
    except Exception as errore:
        sessione.registro.registra_avvio("download_modello", errore)
        ui.stampa_errore(
            f"Download non riuscito: {type(errore).__name__}: {errore}"
        )
        return False
    return True


def _elenco_modelli_slash(sessione: SessioneAgente | None) -> str:
    """Opzioni /model con tag e stato installazione se il catalogo è noto."""
    catalogo = _catalogo_sessione(sessione)
    parti: list[str] = []
    for nome in sorted(MODELLI_DISPONIBILI):
        tag = risolvi_tag_ollama(nome)
        if sessione is None:
            parti.append(f"{nome} ({tag})")
            continue
        stato = "installato" if variant_is_installed(tag, catalogo) else "non presente"
        parti.append(f"{nome} ({tag}, {stato})")
    return ", ".join(parti)


def _applica_modello_e_preload(
    config: ConfigRuntime,
    ui: ConsoleUI,
    sessione: SessioneAgente | None,
    conferma: str,
    alias_preload: str | None = None,
) -> None:
    """
    Precarica il runtime per l'alias effettivo e conferma all'utente.

    Args:
        config: Configurazione della sessione.
        ui: Console per messaggi sistema.
        sessione: Sessione attiva; None salta il preload (test senza runtime).
        conferma: Messaggio finale (modello impostato).
        alias_preload: Alias da caricare; None usa la policy energetica su config.
    """
    if sessione is not None:
        alias = alias_preload or alias_per_alimentazione(config.modello, config)
        sessione.preriscalda_modello(alias)
    ui.imposta_etichetta_modello(_etichetta_modello(config))
    ui.stampa_messaggio_sistema(conferma)


def _gestisci_comando(
    richiesta: str,
    config: ConfigRuntime,
    ui: ConsoleUI,
    sessione: SessioneAgente | None = None,
) -> bool:
    """Gestisce un comando slash e indica se la richiesta è stata consumata."""
    if richiesta.lower() in {"/help", "/?"}:
        ui.stampa_aiuto()
        return True

    if richiesta.lower() in COMANDI_CLEAR:
        if sessione is not None:
            sessione.svuota_memoria()
        ui.stampa_messaggio_sistema("Storico conversazione azzerato.")
        return True

    match_errori = _COMANDO_ERRORI.match(richiesta)
    if match_errori:
        if sessione is None:
            ui.stampa_messaggio_sistema(
                "Nessun registro di sessione disponibile."
            )
            return True
        grezzo = match_errori.group(1)
        limite = int(grezzo) if grezzo else ERRORI_SLASH_DEFAULT
        ui.stampa_messaggio_sistema(sessione.registro.formatta_ultimi_errori(limite))
        return True

    match_toggle = _COMANDO_TOGGLE.match(richiesta)
    if match_toggle:
        attr = _attr_toggle(match_toggle.group("cmd").lower())
        nuovo = _toggle(getattr(config, attr), match_toggle.group("azione"))
        setattr(config, attr, nuovo)
        ui.stampa_toggle(_ETICHETTE_TOGGLE[attr], nuovo)
        return True

    match_model = _COMANDO_MODEL.match(richiesta)
    if match_model:
        alias = match_model.group(1)
        if alias is None or alias.lower() not in MODELLI_DISPONIBILI:
            ui.stampa_messaggio_sistema(
                f"Modello attuale: {_etichetta_modello(config)}. "
                f"Disponibili: {_elenco_modelli_slash(sessione)}"
            )
            return True
        scelto = alias.lower()
        if not _assicura_modello_installato(scelto, ui, sessione):
            return True
        config.modello = scelto
        alias_preload = alias_per_alimentazione(scelto, config)
        _applica_modello_e_preload(
            config,
            ui,
            sessione,
            conferma=f"Modello impostato: {_etichetta_modello(config)}",
            alias_preload=alias_preload,
        )
        return True

    return False


def _elabora_turno(
    richiesta: str,
    sessione: SessioneAgente,
    config: ConfigRuntime,
    ui: ConsoleUI,
) -> bool:
    """Esegue un singolo turno, finalizza la UI e converte eccezioni in exit status."""
    try:
        risultato, _ = sessione.esegui_richiesta(richiesta, config)
    except (KeyboardInterrupt, TurnoAnnullato):
        # Ctrl+C annulla il turno ma tiene viva la sessione.
        sessione.annulla_turno()
        sessione.registro.termina_turno("annullato")
        ui.reset_chiamata_llm()
        ui.termina_esecuzione_tool()
        ui.stampa_messaggio_sistema("Turno annullato.")
        ui.stampa_separatore()
        return False
    except Exception as errore:
        sessione.registro.registra_turn_error(errore)
        ui.stampa_errore(f"Errore agente: {type(errore).__name__}: {errore}")
        ui.stampa_separatore()
        return False
    ui.finalizza(risultato)
    # Solo i path del fallback create-only: i tool ReAct non sono un salvataggio automatico.
    path_fallback = sessione.path_fallback_turno
    if path_fallback:
        ui.stampa_fallback(path_fallback)
    ui.stampa_separatore()
    return True


def _esito_vuoto() -> EsitoAvvio:
    """Esito neutro: il main ritenta il probe e non eredita un runtime."""
    return EsitoAvvio(None, None, None, None)


def _precara_se_pronto(
    config: ConfigRuntime,
    runtime_holder: list[OllamaRuntime],
) -> EsitoAvvio:
    """
    Rileva l'ambiente e, se è completo, precarica i pesi senza toccare la console.

    Args:
        config: Alias di sessione da allineare al catalogo Ollama.
        runtime_holder: Lista condivisa col main per chiudere il client su Ctrl+C.

    Returns:
        Snapshot e runtime caldo, oppure runtime None se serve il wizard.
    """
    ambiente = rileva_ambiente()
    # Senza Ollama/modello il POST fallirebbe: il wizard resta sul thread UI.
    if not ambiente.pronto:
        return EsitoAvvio(ambiente, None, None, None)
    alias = allinea_alias_sessione(config.modello, ambiente)
    alias = alias_per_alimentazione(alias, config)
    host = detect_host()
    runtime = OllamaRuntime(host=host)
    runtime_holder.append(runtime)
    try:
        runtime.preload(alias)
    except Exception:
        # Il main non deve ereditare un client HTTP a metà.
        runtime.close()
        runtime_holder.remove(runtime)
        raise
    return EsitoAvvio(ambiente, runtime, host, alias)


def _esegui_con_splash(
    config: ConfigRuntime,
    prompt_singolo: str | None,
    ui: ConsoleUI,
    registro: RegistroEventi,
) -> EsitoAvvio:
    """
    Avvia il preload in un thread e, se interattivo, lo splash sullo stesso intervallo.

    Args:
        config: Configurazione CLI (alias modello).
        prompt_singolo: Se impostato, niente HUD.
        ui: Console per splash e messaggio di attesa.
        registro: Destinazione dell'evento ``avvio`` se il preload alza.

    Returns:
        Esito del worker, oppure esito vuoto se il thread ha alzato.
    """
    runtime_holder: list[OllamaRuntime] = []
    esiti: list[EsitoAvvio] = []
    errori: list[BaseException] = []

    def lavoro() -> None:
        try:
            esiti.append(_precara_se_pronto(config, runtime_holder))
        except Exception as errore:
            errori.append(errore)

    thread = threading.Thread(
        target=lavoro,
        name=NOME_THREAD_PRELOAD,
        daemon=False,
    )
    thread.start()
    if sessione_interattiva(prompt_singolo):
        ui.stampa_splash()
    if thread.is_alive():
        ui.stampa_messaggio_sistema(MESSAGGIO_PRERISCALDAMENTO_IN_CORSO)
    try:
        thread.join()
    except KeyboardInterrupt:
        for runtime in list(runtime_holder):
            runtime.close()
        raise
    if errori:
        errore = errori[0]
        registro.registra_avvio("preload", errore)
        ui.stampa_messaggio_sistema(
            "Preriscaldamento non riuscito: "
            f"{type(errore).__name__}: {errore}"
        )
        return _esito_vuoto()
    if esiti:
        return esiti[0]
    return _esito_vuoto()


def _prepara_ambiente(
    config: ConfigRuntime,
    prompt_singolo: str | None,
    ui: ConsoleUI,
    ambiente: AmbienteRilevato | None = None,
    registro: RegistroEventi | None = None,
) -> AmbienteRilevato | None:
    """
    Verifica Homebrew/Ollama/Gemma 4 e, se serve, avvia il wizard di STARTUP.MD.

    Args:
        config: Runtime CLI; `modello` può essere riallineato al catalogo.
        prompt_singolo: Se impostato, il wizard è vietato (uscita 1 se manca tutto).
        ui: Console per errori e consensi.
        ambiente: Snapshot già rilevato nel thread di avvio; None ripete il probe.
        registro: Destinazione degli eventi di avvio fallito.

    Returns:
        Snapshot pronto, oppure None quando l'avvio deve terminare con codice 1.
    """
    osservazione = registro if registro is not None else RegistroEventi()
    if ambiente is None:
        ambiente = rileva_ambiente()
    if not ambiente.pronto:
        if not sessione_interattiva(prompt_singolo):
            ui.stampa_errore(messaggio_setup_non_interattivo())
            osservazione.registra_avvio(
                "setup",
                RuntimeError(messaggio_setup_non_interattivo()),
            )
            return None
        try:
            ambiente = esegui_wizard(ui)
        except KeyboardInterrupt:
            osservazione.registra_avvio(
                "setup",
                KeyboardInterrupt("Setup interrotto."),
            )
            ui.stampa_errore("Setup interrotto.")
            return None
        except SetupAborted as errore:
            osservazione.registra_avvio("setup", errore)
            ui.stampa_errore(str(errore))
            return None
    alias_effettivo = allinea_alias_sessione(config.modello, ambiente)
    if alias_effettivo != config.modello:
        ui.stampa_messaggio_sistema(
            f"Modello {config.modello} assente in Ollama: uso {alias_effettivo}. "
            "Cambia con /model."
        )
        config.modello = alias_effettivo
    return ambiente


def _esegui_sessione(
    config: ConfigRuntime,
    prompt_singolo: str | None,
    ui: ConsoleUI,
    registro: RegistroEventi | None = None,
    cartella: Path | None = None,
) -> int:
    """Avvia prompt singolo o loop interattivo e rilascia sempre le risorse."""
    cartella_documenti = (cartella or Path.cwd()).resolve()
    file_service = WorkspaceFileService(FileConfig(workspace=cartella_documenti))
    ui.imposta_separatore_file(SEPARATORE_FILE)
    ui.imposta_verbose(config.verbose)
    posseduto = registro is None
    osservazione = registro if registro is not None else RegistroEventi()
    osservazione.inizia_sessione(
        pid=os.getpid(),
        python=sys.version.split()[0],
        modello=config.modello,
    )
    sessione: SessioneAgente | None = None
    try:
        esito = _esegui_con_splash(config, prompt_singolo, ui, osservazione)
        ambiente = _prepara_ambiente(
            config,
            prompt_singolo,
            ui,
            esito.ambiente,
            osservazione,
        )
        if ambiente is None:
            if esito.runtime is not None:
                esito.runtime.close()
            return 1
        if esito.runtime is not None:
            sessione = SessioneAgente(
                ui,
                file_service=file_service,
                ollama_runtime=esito.runtime,
                host=esito.host,
                registro=osservazione,
            )
            alias_caricato = esito.alias or alias_per_alimentazione(
                config.modello, config
            )
            sessione.annuncia_preriscaldamento(alias_caricato)
        else:
            sessione = SessioneAgente(
                ui,
                file_service=file_service,
                host=detect_host(),
                registro=osservazione,
            )
            sessione.preriscalda_modello(
                alias_per_alimentazione(config.modello, config)
            )
        ui.barra_termica.avvia()
        ui.stampa_benvenuto(
            config.verbose,
            config.verifica_completa,
            config.burn_protection,
            config.thinking,
            etichetta_modello=_etichetta_modello(config),
            cartella=str(cartella_documenti),
        )
        if prompt_singolo is not None:
            if not prompt_singolo.strip():
                ui.stampa_messaggio_sistema("Prompt vuoto: nessuna elaborazione.")
                return 0
            return 0 if _elabora_turno(prompt_singolo, sessione, config, ui) else 1

        while True:
            try:
                richiesta = ui.leggi_richiesta_utente(_etichetta_modello(config))
            except (EOFError, KeyboardInterrupt):
                ui.stampa_messaggio_sistema("Sessione terminata.")
                break
            if not richiesta:
                continue
            if richiesta.lower() in COMANDI_USCITA:
                ui.stampa_messaggio_sistema("Sessione terminata.")
                break
            if _gestisci_comando(richiesta, config, ui, sessione):
                continue
            _elabora_turno(richiesta, sessione, config, ui)
        return 0
    finally:
        osservazione.termina_sessione()
        if sessione is not None:
            sessione.close()
        ui.barra_termica.ferma()
        if posseduto:
            osservazione.chiudi()


def main() -> None:
    """Entry point CLI che propaga il codice di uscita al processo."""
    config, prompt_singolo, cartella = _parse_argomenti()
    registro = RegistroProcesso()
    registro.instala_hook_processo()
    registro.avvia_controllo_stall()
    ui = ConsoleUI()
    try:
        sys.exit(
            _esegui_sessione(
                config,
                prompt_singolo,
                ui,
                registro=registro,
                cartella=cartella,
            )
        )
    finally:
        registro.chiudi()
