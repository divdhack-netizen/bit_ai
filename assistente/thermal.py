"""Monitoraggio termico macOS e throttling prestazioni agente."""

from __future__ import annotations

import atexit
import json
import platform
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from rich.cells import cell_len

from assistente.config import (
    ALIAS_MODELLO_DEFAULT,
    MAX_ITERAZIONI_AGENTE,
    MAX_TOKEN_GENERATI,
)
from assistente.host import CoolingProfile, PowerState, detect_host
from assistente.inference_profiles import MODEL_ALIAS_ORDER
from assistente.metriche import StatisticheTurnoLLM, formatta_segmento_statistiche

# Separatore tra prefisso termico e blocco metriche (larghezza fissa in celle)
_SEPARATORE_METRICHE = "  ·  "

if TYPE_CHECKING:
    from assistente.config import ConfigRuntime
    from assistente.console import ConsoleUI

_PASSI_FORZA_FAST = 99
_SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"
_SCRIPT_MONITOR = _SCRIPT_DIR / "thermal_monitor.swift"
_COMANDO_STATO_TERMICO = [
    "swift",
    "-e",
    "import Foundation; print(ProcessInfo.processInfo.thermalState.rawValue)",
]
_TIMEOUT_SUBPROCESS_SEC = 3
_INTERVALLO_POLLING_SEC = 2.0
_INTERVALLO_MONITOR_MS = 2000
_TTL_CACHE_SEC = 1.5
_TEMP_CPU_MAX_PLAUSIBILE = 150.0
_NON_IN_CACHE = object()
_ANSI_DIM = "\033[2m"
_ANSI_RESET = "\033[0m"
# Media mobile breve: smussa spike SMC senza ritardare troppo la reazione.
_FINESTRA_MEDIA_TEMP = 5
# Isteresi: resta nello stato più caldo finché non vediamo N campioni più freschi.
_CAMPIONI_ISTERESI_RAFFREDDAMENTO = 3
# Anticipazione soft: fanless (Air/base) più bassa; Pro/Max/Ultra e Linux più alta.
_TEMP_SOFT_FAIR_FANLESS_CELSIUS = 78.0
_TEMP_SOFT_SERIO_FANLESS_CELSIUS = 88.0
_TEMP_SOFT_FAIR_COOLED_CELSIUS = 88.0
_TEMP_SOFT_SERIO_COOLED_CELSIUS = 95.0
_TEMP_CRITICO_LINUX_CELSIUS = 99.0
_DELTA_TREND_SALITA_CELSIUS = 1.5
_SYSFS_THERMAL = Path("/sys/class/thermal")


def soglie_soft_celsius(cooling: CoolingProfile) -> tuple[float, float]:
    """
    Soglie Fair/Serious per anticipazione da temperatura.

    Args:
        cooling: `fanless_conservative` (Air/base) o `cooled` (Pro/Linux).

    Returns:
        Coppia (fair_celsius, serio_celsius).
    """
    # I SoC senza ventola devono throttlare prima; i raffreddati tollerano di più.
    if cooling == "cooled":
        return _TEMP_SOFT_FAIR_COOLED_CELSIUS, _TEMP_SOFT_SERIO_COOLED_CELSIUS
    return _TEMP_SOFT_FAIR_FANLESS_CELSIUS, _TEMP_SOFT_SERIO_FANLESS_CELSIUS


def stato_da_temperatura(
    temp_celsius: float,
    *,
    cooling: CoolingProfile = "cooled",
) -> StatoTermico:
    """Mappa una temperatura assoluta su StatoTermico (Linux, senza ProcessInfo)."""
    fair_celsius, serio_celsius = soglie_soft_celsius(cooling)
    # Critical è solo sysfs: Apple usa ProcessInfo.thermalState per quel livello.
    if temp_celsius >= _TEMP_CRITICO_LINUX_CELSIUS:
        return StatoTermico.CRITICO
    if temp_celsius >= serio_celsius:
        return StatoTermico.SERIO
    if temp_celsius >= fair_celsius:
        return StatoTermico.DISCRETO
    return StatoTermico.NOMINALE

ETICHETTE_STATO = {0: "Nominal", 1: "Fair", 2: "Serious", 3: "Critical"}


class StatoTermico(IntEnum):
    NOMINALE = 0
    DISCRETO = 1
    SERIO = 2
    CRITICO = 3


_COLORI: dict[StatoTermico, str] = {
    StatoTermico.NOMINALE: "\033[32m",
    StatoTermico.DISCRETO: "\033[33m",
    StatoTermico.SERIO: "\033[38;5;208m",
    StatoTermico.CRITICO: "\033[31m",
}

# Profili workload-first: riducono output/iterazioni prima di introdurre sleep.
_NUM_PREDICT_DISCRETO = 1024
_MAX_ITER_DISCRETO = 4
_PAUSA_DISCRETO_PRE_SEC = 0.0
_PAUSA_DISCRETO_ITER_SEC = 1.0
_PASSI_DEGRADA_DISCRETO = 0

_NUM_PREDICT_SERIO = 640
_MAX_ITER_SERIO = 3
_PAUSA_SERIO_PRE_SEC = 2.0
_PAUSA_SERIO_ITER_SEC = 3.0
_PASSI_DEGRADA_SERIO = 1

_NUM_PREDICT_CRITICO = 384
_MAX_ITER_CRITICO = max(2, MAX_ITERAZIONI_AGENTE // 3)
_PAUSA_CRITICO_ITER_SEC = 5.0

# (num_predict, max_iter, pausa_pre, pausa_iter, passi_degradazione, no_thinking, attendi_cooldown)
_PROFILI: dict[StatoTermico, tuple[int, int, float, float, int, bool, bool]] = {
    StatoTermico.NOMINALE: (
        MAX_TOKEN_GENERATI,
        MAX_ITERAZIONI_AGENTE,
        0.0,
        0.0,
        0,
        False,
        False,
    ),
    StatoTermico.DISCRETO: (
        _NUM_PREDICT_DISCRETO,
        _MAX_ITER_DISCRETO,
        _PAUSA_DISCRETO_PRE_SEC,
        _PAUSA_DISCRETO_ITER_SEC,
        _PASSI_DEGRADA_DISCRETO,
        False,
        False,
    ),
    StatoTermico.SERIO: (
        _NUM_PREDICT_SERIO,
        _MAX_ITER_SERIO,
        _PAUSA_SERIO_PRE_SEC,
        _PAUSA_SERIO_ITER_SEC,
        _PASSI_DEGRADA_SERIO,
        True,
        False,
    ),
    StatoTermico.CRITICO: (
        _NUM_PREDICT_CRITICO,
        _MAX_ITER_CRITICO,
        0.0,
        _PAUSA_CRITICO_ITER_SEC,
        _PASSI_FORZA_FAST,
        True,
        True,
    ),
}


@dataclass(frozen=True)
class ParametriThrottle:
    stato: StatoTermico
    alias_modello: str
    num_predict: int
    max_iterazioni: int
    pausa_pre_turno_sec: float
    pausa_tra_iterazioni_sec: float
    disabilita_thinking: bool


class ContestoTurnoTermico(NamedTuple):
    stato: StatoTermico
    alias_modello: str
    thinking: bool
    throttle: ParametriThrottle
    messaggio_avviso: str | None


@dataclass(frozen=True, slots=True)
class SnapshotTermico:
    """Lettura atomica di stato Apple e temperatura SMC opzionale."""

    stato: StatoTermico
    temp_celsius: float | None
    letto_da_monitor: bool


class _LettoreConCache:
    def __init__(self, ttl_sec: float, lettore: Callable[[], Any]) -> None:
        self._ttl_sec = ttl_sec
        self._lettore = lettore
        self._voce: Any = _NON_IN_CACHE
        self._scadenza = 0.0

    def leggi(self) -> Any:
        ora = time.monotonic()
        if self._voce is not _NON_IN_CACHE and ora < self._scadenza:
            return self._voce
        valore = self._lettore()
        self._voce = valore
        self._scadenza = ora + self._ttl_sec
        return valore

    def invalida(self) -> None:
        self._voce = _NON_IN_CACHE
        self._scadenza = 0.0


class ThermalMonitorProcess:
    """
    Possiede un unico processo Swift persistente per stato + temperatura.

    Evita di avviare due compilazioni/interpreti Swift a ogni poll della barra.
    """

    def __init__(
        self,
        script_path: Path = _SCRIPT_MONITOR,
        *,
        interval_ms: int = _INTERVALLO_MONITOR_MS,
        popen: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    ) -> None:
        """
        Prepara il monitor senza avviarlo.

        Args:
            script_path: Sorgente Swift del monitor persistente.
            interval_ms: Periodo di emissione snapshot lato Swift.
            popen: Factory iniettabile per i test.
        """
        self._script_path = script_path
        self._interval_ms = interval_ms
        self._popen = popen
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.RLock()
        self._ultimo_snapshot: SnapshotTermico | None = None
        self._reader: threading.Thread | None = None
        self._ferma = threading.Event()

    @property
    def attivo(self) -> bool:
        """True se il processo figlio è ancora in esecuzione."""
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def avvia(self) -> bool:
        """
        Avvia il monitor se assente.

        Returns:
            True se il monitor è disponibile dopo la chiamata.
        """
        with self._lock:
            # Riutilizza il processo esistente per non duplicare IOKit/SMC.
            if self.attivo:
                return True
            # Fuori da macOS il monitor non ha sensori utili.
            if not _su_macos() or not self._script_path.is_file():
                return False
            try:
                self._process = self._popen(
                    ["swift", str(self._script_path), str(self._interval_ms)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                )
            except OSError:
                self._process = None
                return False
            self._ferma.clear()
            self._reader = threading.Thread(
                target=self._leggi_stdout,
                name="thermal-monitor-reader",
                daemon=True,
            )
            self._reader.start()
            return True

    def ferma(self) -> None:
        """Termina best-effort il processo figlio e il thread reader."""
        with self._lock:
            self._ferma.set()
            processo = self._process
            self._process = None
        # Fuori dal lock per non bloccare il reader sul join del lock stesso.
        if processo is not None:
            # SIGTERM permette al loop Swift di uscire e chiudere IOKit.
            processo.terminate()
            try:
                processo.wait(timeout=_TIMEOUT_SUBPROCESS_SEC)
            except subprocess.TimeoutExpired:
                processo.kill()
        reader = self._reader
        self._reader = None
        # Il reader è daemon: un timeout breve evita hang allo shutdown CLI.
        if reader is not None and reader.is_alive():
            reader.join(timeout=1.0)

    def snapshot(self) -> SnapshotTermico | None:
        """
        Restituisce l'ultimo snapshot ricevuto dal monitor.

        Returns:
            Snapshot cacheato oppure None se il monitor non ha ancora prodotto dati.
        """
        # Avvio lazy: la prima lettura tenta di lanciare il processo.
        if not self.attivo:
            self.avvia()
        with self._lock:
            return self._ultimo_snapshot

    def _leggi_stdout(self) -> None:
        """Consuma linee JSON dal monitor finché il processo resta vivo."""
        processo = self._process
        # Senza stdout pipe non c'è nulla da parseare.
        if processo is None or processo.stdout is None:
            return
        # Ogni riga è uno snapshot indipendente; JSON malformato viene ignorato.
        for riga in processo.stdout:
            if self._ferma.is_set():
                break
            snapshot = _parse_snapshot_json(riga)
            # Linee vuote o incomplete non devono cancellare l'ultimo valore valido.
            if snapshot is None:
                continue
            with self._lock:
                self._ultimo_snapshot = snapshot


class ThermalPolicyState:
    """Applica media mobile e isteresi sullo stato termico effettivo."""

    def __init__(self, cooling: CoolingProfile = "fanless_conservative") -> None:
        self._temperature: deque[float] = deque(maxlen=_FINESTRA_MEDIA_TEMP)
        self._stato_policy = StatoTermico.NOMINALE
        self._campioni_raffreddamento = 0
        self._fair_celsius, self._serio_celsius = soglie_soft_celsius(cooling)

    def aggiorna(self, snapshot: SnapshotTermico) -> StatoTermico:
        """
        Combina stato Apple (autorevole) con anticipazione soft da temperatura.

        Args:
            snapshot: Lettura corrente stato + temp opzionale.

        Returns:
            Stato usato dalla policy di throttling dopo isteresi.
        """
        # Apple Serious/Critical prevalgono sempre sulle soglie soft locali.
        stato_apple = snapshot.stato
        stato_candidato = stato_apple
        if snapshot.temp_celsius is not None:
            self._temperature.append(snapshot.temp_celsius)
            media = sum(self._temperature) / len(self._temperature)
            trend = 0.0
            # Serve almeno una coppia per stimare se la temperatura sta salendo.
            if len(self._temperature) >= 2:
                trend = self._temperature[-1] - self._temperature[0]
            # Anticipa Fair/Serious solo con trend in salita, mai sotto Nominal Apple.
            if (
                stato_apple <= StatoTermico.DISCRETO
                and media >= self._serio_celsius
                and trend >= _DELTA_TREND_SALITA_CELSIUS
            ):
                stato_candidato = max(stato_candidato, StatoTermico.SERIO)
            elif (
                stato_apple == StatoTermico.NOMINALE
                and media >= self._fair_celsius
                and trend >= _DELTA_TREND_SALITA_CELSIUS
            ):
                stato_candidato = max(stato_candidato, StatoTermico.DISCRETO)

        # Scaldare: adotta subito il livello più alto per proteggere il SoC.
        if stato_candidato > self._stato_policy:
            self._stato_policy = stato_candidato
            self._campioni_raffreddamento = 0
            return self._stato_policy

        # Raffreddare: richiede più campioni per evitare flap Fair↔Nominal.
        if stato_candidato < self._stato_policy:
            self._campioni_raffreddamento += 1
            # Finché non raggiungiamo l'isteresi restiamo sul livello più caldo.
            if self._campioni_raffreddamento < _CAMPIONI_ISTERESI_RAFFREDDAMENTO:
                return self._stato_policy
            self._stato_policy = stato_candidato
            self._campioni_raffreddamento = 0
            return self._stato_policy

        self._campioni_raffreddamento = 0
        return self._stato_policy


def _su_macos() -> bool:
    return platform.system() == "Darwin"


def _stdout_subprocess(comando: list[str]) -> str | None:
    try:
        risultato = subprocess.run(
            comando,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SUBPROCESS_SEC,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if risultato.returncode != 0:
        return None
    testo = risultato.stdout.strip()
    return testo or None


def _parse_snapshot_json(riga: str) -> SnapshotTermico | None:
    """
    Interpreta una riga JSON del monitor Swift.

    Args:
        riga: Linea grezza stdout.

    Returns:
        Snapshot tipizzato oppure None se il payload non è utilizzabile.
    """
    testo = riga.strip()
    # Linee vuote sono normali durante l'avvio del runtime Swift.
    if not testo:
        return None
    try:
        payload = json.loads(testo)
    except json.JSONDecodeError:
        return None
    # Campi assenti o di tipo errato indicano un protocollo incompatibile.
    if not isinstance(payload, dict) or "thermalState" not in payload:
        return None
    try:
        valore_stato = int(payload["thermalState"])
    except (TypeError, ValueError):
        return None
    stato = _normalizza_stato(valore_stato)
    temp_raw = payload.get("tempC")
    temperatura: float | None
    # null JSON → temperatura assente; valori numerici vengono validati.
    if temp_raw is None:
        temperatura = None
    else:
        try:
            temperatura = float(temp_raw)
        except (TypeError, ValueError):
            temperatura = None
        # Scarta outlier SMC prima di aggiornare media mobile e policy.
        if (
            temperatura is not None
            and (temperatura <= 0 or temperatura >= _TEMP_CPU_MAX_PLAUSIBILE)
        ):
            temperatura = None
    return SnapshotTermico(
        stato=stato,
        temp_celsius=temperatura,
        letto_da_monitor=True,
    )


def _normalizza_stato(valore: int) -> StatoTermico:
    """Clamp del rawValue Apple nell'enum supportato dalla policy."""
    if valore >= StatoTermico.CRITICO:
        return StatoTermico.CRITICO
    if valore <= StatoTermico.NOMINALE:
        return StatoTermico.NOMINALE
    return StatoTermico(valore)


def _leggi_stato_termico_da_sistema() -> StatoTermico:
    if not _su_macos():
        return StatoTermico.NOMINALE
    testo_uscita = _stdout_subprocess(_COMANDO_STATO_TERMICO)
    if testo_uscita is None or not testo_uscita.isdigit():
        return StatoTermico.NOMINALE
    return _normalizza_stato(int(testo_uscita))


def _darwin_oneshot() -> SnapshotTermico | None:
    """Un solo snapshot JSON dallo stesso script del monitor persistente."""
    testo_uscita = _stdout_subprocess(["swift", str(_SCRIPT_MONITOR), "once"])
    if testo_uscita is None:
        return None
    parsed = _parse_snapshot_json(testo_uscita)
    if parsed is None:
        return None
    return SnapshotTermico(parsed.stato, parsed.temp_celsius, False)


def leggi_temperature_sysfs(
    *,
    base: Path = _SYSFS_THERMAL,
    read_text: Callable[[Path], str | None] | None = None,
    listdir: Callable[[Path], list[str]] | None = None,
) -> list[float]:
    """
    Legge le zone `/sys/class/thermal/thermal_zone*/temp` in Celsius.

    Args:
        base: Directory sysfs.
        read_text: Lettore iniettabile; None usa Path.read_text.
        listdir: Elenco iniettabile; None usa Path.iterdir.

    Returns:
        Temperature plausibili; lista vuota se sysfs è assente.
    """
    names: list[str]
    if listdir is not None:
        names = listdir(base)
    else:
        try:
            names = [entry.name for entry in base.iterdir()]
        except OSError:
            return []
    reader = read_text
    temperature: list[float] = []
    for name in names:
        # cooling_device* non è una temperatura CPU/SoC.
        if not name.startswith("thermal_zone"):
            continue
        raw: str | None
        path = base / name / "temp"
        if reader is not None:
            raw = reader(path)
        else:
            try:
                raw = path.read_text(encoding="utf-8")
            except OSError:
                raw = None
        if raw is None:
            continue
        try:
            millicelsius = int(raw.strip())
        except ValueError:
            continue
        celsius = millicelsius / 1000.0
        # Scarta outlier prima di farli entrare nella policy.
        if celsius <= 0 or celsius >= _TEMP_CPU_MAX_PLAUSIBILE:
            continue
        temperature.append(celsius)
    return temperature


def snapshot_linux_sysfs(
    *,
    read_text: Callable[[Path], str | None] | None = None,
    listdir: Callable[[Path], list[str]] | None = None,
) -> SnapshotTermico:
    """Costruisce uno snapshot dal max delle zone sysfs; Nominal se assenti."""
    temperature = leggi_temperature_sysfs(read_text=read_text, listdir=listdir)
    if not temperature:
        return SnapshotTermico(StatoTermico.NOMINALE, None, False)
    hottest = max(temperature)
    return SnapshotTermico(
        stato_da_temperatura(hottest, cooling="cooled"),
        hottest,
        False,
    )


_monitor_process = ThermalMonitorProcess()
_policy_state = ThermalPolicyState(detect_host().cooling)
atexit.register(_monitor_process.ferma)


def _leggi_snapshot_corrente() -> SnapshotTermico:
    """
    Preferisce il monitor persistente Darwin; fallback one-shot o sysfs Linux.

    Returns:
        Snapshot con stato e temperatura opzionale.
    """
    if _su_macos():
        snapshot = _monitor_process.snapshot()
        if snapshot is not None:
            return snapshot
        oneshot = _darwin_oneshot()
        if oneshot is not None:
            return oneshot
        return SnapshotTermico(
            stato=_leggi_stato_termico_da_sistema(),
            temp_celsius=None,
            letto_da_monitor=False,
        )
    if platform.system() == "Linux":
        return snapshot_linux_sysfs()
    return SnapshotTermico(StatoTermico.NOMINALE, None, False)


def _leggi_stato_policy() -> StatoTermico:
    """Aggiorna isteresi e restituisce lo stato effettivo di throttling."""
    snapshot = _leggi_snapshot_corrente()
    return _policy_state.aggiorna(snapshot)


_cache_stato = _LettoreConCache(_TTL_CACHE_SEC, _leggi_stato_policy)
_cache_temp_cpu = _LettoreConCache(
    _TTL_CACHE_SEC,
    lambda: _leggi_snapshot_corrente().temp_celsius,
)


def leggi_stato_termico() -> StatoTermico:
    return _cache_stato.leggi()


def leggi_temperatura_cpu_celsius() -> float | None:
    return _cache_temp_cpu.leggi()


def invalida_cache_termica() -> None:
    _cache_stato.invalida()
    _cache_temp_cpu.invalida()


def reset_policy_termica_per_test() -> None:
    """
    Riporta monitor e isteresi allo stato iniziale nei test unitari.

    Non va usata dal percorso di produzione: invalida la continuità termica.
    """
    global _policy_state
    _monitor_process.ferma()
    _policy_state = ThermalPolicyState()
    invalida_cache_termica()


def riga_barra_stato(
    stato: StatoTermico,
    temp_celsius: float | None,
    metriche: StatisticheTurnoLLM | None = None,
    *,
    ansi: bool = False,
    max_colonne: int | None = None,
    power: PowerState | None = None,
    verbose: bool = False,
) -> str:
    """
    Costruisce la riga di stato termico (+ metriche) per l'ultima riga TTY.

    Args:
        stato: Livello termico macOS (Nominal…Critical).
        temp_celsius: Temperatura CPU in °C, o None se non disponibile.
        metriche: Snapshot LLM del turno; None omette il blocco metriche.
        ansi: Se True, applica colori dim/stato per il terminale.
        max_colonne: Budget colonne visibili; con metriche attiva il layout
            progressivo e non supera questo limite (escluse sequenze ANSI).
        verbose: Se True, il segmento metriche include anche le latenze.

    Returns:
        Riga pronta da stampare sulla barra riservata (eventualmente ridotta).
    """
    etichetta = ETICHETTE_STATO.get(int(stato), "?")
    gradi = "n/d" if temp_celsius is None else f"{temp_celsius:.0f}°C"
    if power is not None and power.low_power_mode:
        pwr = "pwr: LPM  "
    elif power is not None and power.on_battery:
        pwr = "pwr: batt  "
    elif power is not None:
        pwr = "pwr: ac  "
    else:
        pwr = ""
    prefisso = f"{pwr}tmp: {etichetta} - {gradi}"
    testo = prefisso

    # Aggiunge metriche solo se presenti e c'è spazio oltre prefisso + separatore
    if metriche is not None:
        # Con tetto colonne: lascia al prefisso la priorità assoluta
        if max_colonne is not None:
            residuo = max_colonne - cell_len(prefisso) - cell_len(_SEPARATORE_METRICHE)
            # Residuo non positivo: solo prefisso termico, niente overflow
            if residuo > 0:
                segmento = formatta_segmento_statistiche(
                    metriche,
                    max_larghezza=residuo,
                    verbose=verbose,
                )
                # Segmento vuoto se nessun pezzo entra: evita " · " orfano
                if segmento:
                    testo = f"{prefisso}{_SEPARATORE_METRICHE}{segmento}"
        else:
            # Nessun tetto: stringa completa (comportamento storico)
            testo = (
                f"{prefisso}{_SEPARATORE_METRICHE}"
                f"{formatta_segmento_statistiche(metriche, verbose=verbose)}"
            )

    # Senza ANSI restituisce testo piano (test / log)
    if not ansi:
        return testo

    colore = _COLORI.get(stato, "")
    # Colora solo l'etichetta di stato; il resto resta dim
    if colore:
        testo = testo.replace(
            etichetta,
            f"{colore}{etichetta}{_ANSI_RESET}{_ANSI_DIM}",
            1,
        )
    return f"{_ANSI_DIM}{testo}{_ANSI_RESET}"


def _alias_effettivo(alias_richiesto: str, passi_degradazione: int) -> str:
    if passi_degradazione <= 0:
        return alias_richiesto
    if passi_degradazione >= _PASSI_FORZA_FAST:
        return MODEL_ALIAS_ORDER[0]
    try:
        indice = MODEL_ALIAS_ORDER.index(alias_richiesto)
    except ValueError:
        indice = MODEL_ALIAS_ORDER.index(ALIAS_MODELLO_DEFAULT)
    return MODEL_ALIAS_ORDER[max(0, indice - passi_degradazione)]


def pausa_adattiva_tra_iterazioni(
    stato_iniziale: StatoTermico,
    pausa_base_sec: float,
) -> float:
    """Rilegge lo stato termico solo da SERIO in su; altrimenti tiene la pausa base."""
    if stato_iniziale < StatoTermico.SERIO:
        # In nominale/discreto il monitor in background basta: niente probe sincrono.
        return pausa_base_sec
    invalida_cache_termica()
    stato_attuale = leggi_stato_termico()
    if stato_attuale <= stato_iniziale:
        return pausa_base_sec
    pausa_profilo = _PROFILI[stato_attuale][3]
    return max(pausa_base_sec, pausa_profilo)


def calcola_throttle(stato: StatoTermico, alias_modello: str) -> ParametriThrottle:
    prof = _PROFILI[stato]
    return ParametriThrottle(
        stato=stato,
        alias_modello=_alias_effettivo(alias_modello, prof[4]),
        num_predict=prof[0],
        max_iterazioni=prof[1],
        pausa_pre_turno_sec=prof[2],
        pausa_tra_iterazioni_sec=prof[3],
        disabilita_thinking=prof[5],
    )


def attendi_raffreddamento(
    stato_target: StatoTermico = StatoTermico.DISCRETO,
    timeout_secondi: float = 60.0,
) -> StatoTermico:
    tempo_inizio = time.monotonic()
    while True:
        _cache_stato.invalida()
        stato_attuale = _cache_stato.leggi()
        if stato_attuale <= stato_target:
            return stato_attuale
        if time.monotonic() - tempo_inizio >= timeout_secondi:
            return stato_attuale
        time.sleep(_INTERVALLO_POLLING_SEC)


def prepara_turno(config: ConfigRuntime, ui: ConsoleUI) -> ContestoTurnoTermico | None:
    if not config.burn_protection:
        return None

    stato = leggi_stato_termico()
    prof = _PROFILI[stato]

    if prof[6]:
        etichetta = ETICHETTE_STATO.get(int(stato), "Unknown")
        ui.stampa_avviso_termico(f"Temperatura {etichetta}: attesa raffreddamento...")
        stato = attendi_raffreddamento()
        prof = _PROFILI[stato]

    throttle = calcola_throttle(stato, config.modello)
    thinking = not throttle.disabilita_thinking
    messaggio_avviso = None
    if stato != StatoTermico.NOMINALE:
        etichetta = ETICHETTE_STATO.get(int(stato), "Unknown")
        parti = [f"Temperatura {etichetta}"]
        # Workload-first: comunica prima i tagli a output/iterazioni, poi le pause.
        if throttle.num_predict < MAX_TOKEN_GENERATI:
            parti.append(f"num_predict {throttle.num_predict}")
        if throttle.max_iterazioni < MAX_ITERAZIONI_AGENTE:
            parti.append(f"max_iter {throttle.max_iterazioni}")
        if throttle.alias_modello != config.modello:
            parti.append(f"modello {throttle.alias_modello}")
        if throttle.disabilita_thinking:
            parti.append("thinking disattivato")
        if throttle.pausa_pre_turno_sec > 0:
            parti.append(f"pausa {int(throttle.pausa_pre_turno_sec)}s")
        messaggio_avviso = (
            parti[0] if len(parti) == 1 else f"{parti[0]} — {', '.join(parti[1:])}"
        )
        ui.stampa_avviso_termico(messaggio_avviso)

    if throttle.pausa_pre_turno_sec > 0:
        time.sleep(throttle.pausa_pre_turno_sec)

    return ContestoTurnoTermico(
        stato,
        throttle.alias_modello,
        thinking,
        throttle,
        messaggio_avviso,
    )
