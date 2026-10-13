"""Registro persistente di crash, turni incompleti e fallimenti di processo."""

from __future__ import annotations

import atexit
import faulthandler
import json
import os
import sys
import threading
import time
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from assistente.config import (
    DIRECTORY_LOG,
    ERRORI_SLASH_DEFAULT,
    MESSAGGIO_EVENTO_CARATTERI,
    PREVIEW_RICHIESTA_CARATTERI,
    STALL_POLL_SEC,
    STALL_SOGLIA_SEC,
    TRACEBACK_MAX_CARATTERI,
    ULTIMI_ERRORI_INDICE,
)

TIPI_PROBLEMA = frozenset(
    {
        "avvio",
        "crash",
        "diagnostica",
        "llm_error",
        "stall",
        "tool_error",
        "turn_error",
    }
)
ESITI_TURNO_NON_OK = frozenset({"errore", "incompleto", "vuoto"})
_PREFISS_DIAGNOSTICA = (
    "errore inferenza locale:",
)
_MARCATORE_SINTESI_FALLITA = "sintesi del modello è fallita"


def preview_richiesta(
    testo: str,
    limite: int = PREVIEW_RICHIESTA_CARATTERI,
) -> tuple[str, int]:
    """
    Tronca la richiesta per il log senza esporre il prompt intero.

    Returns:
        Anteprima (al più ``limite`` caratteri) e lunghezza originale.
    """
    pulito = testo.strip()
    lunghezza = len(pulito)
    if lunghezza <= limite:
        return pulito, lunghezza
    return pulito[:limite], lunghezza


def tronca_messaggio(
    testo: str,
    limite: int = MESSAGGIO_EVENTO_CARATTERI,
) -> str:
    """Accorcia un messaggio di evento restando entro il budget del registro."""
    if len(testo) <= limite:
        return testo
    return f"{testo[:limite]}\n…"


def formatta_traceback(
    errore: BaseException,
    limite: int = TRACEBACK_MAX_CARATTERI,
) -> str:
    """Serializza lo stack dell'eccezione, troncato se eccede il budget."""
    testo = "".join(
        traceback.format_exception(type(errore), errore, errore.__traceback__)
    )
    return tronca_messaggio(testo, limite)


def e_risposta_diagnostica(risposta: str) -> bool:
    """True se il testo utente è un fallimento mascherato da diagnostica."""
    basso = (risposta or "").casefold()
    if not basso.strip():
        return False
    if any(basso.startswith(prefisso) for prefisso in _PREFISS_DIAGNOSTICA):
        return True
    return _MARCATORE_SINTESI_FALLITA in basso


def classifica_esito_turno(*, incompleto: bool, risposta: str) -> str:
    """
    Deriva l'esito persistito da evaluate_turn e dal testo della risposta.

    Returns:
        ``incompleto``, ``vuoto``, ``errore`` o ``ok``.
    """
    if incompleto:
        return "incompleto"
    if not (risposta or "").strip():
        return "vuoto"
    if e_risposta_diagnostica(risposta):
        return "errore"
    return "ok"


def evento_non_ok(evento: dict[str, Any]) -> bool:
    """True se l'evento deve comparire in ``/errori`` e nell'indice markdown."""
    tipo = evento.get("tipo")
    if tipo == "turn_end":
        visibile = evento.get("esito") in ESITI_TURNO_NON_OK
    else:
        visibile = tipo in TIPI_PROBLEMA
    if tipo == "llm_error" and evento.get("eccezione") == "KeyboardInterrupt":
        visibile = False
    return visibile


def notifica_diagnostica(
    callback: object,
    messaggio: str,
    tipo_eccezione: str | None = None,
) -> None:
    """Inoltra una diagnostica al callback se espone ``registra_diagnostica``."""
    metodo = getattr(callback, "registra_diagnostica", None)
    if callable(metodo):
        metodo(messaggio, tipo_eccezione=tipo_eccezione)


def _iso_utc() -> str:
    """Timestamp UTC stabile, senza microsecondi, per le righe JSONL."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RegistroEventi:
    """
    API di osservabilità. L'implementazione di base non scrive su disco.

    Usata nei test e quando la CLI non inietta un registro persistente.
    """

    def __init__(
        self,
        *,
        orologio_heartbeat: Callable[[], float] | None = None,
    ) -> None:
        self._session_id = uuid4().hex
        self._turn_id: str | None = None
        self._percorso_turno: str | None = None
        self._alias_turno: str | None = None
        self._lock = threading.RLock()
        self._orologio_heartbeat = orologio_heartbeat or time.monotonic
        self._ultimo_heartbeat = 0.0
        self._stall_emesso = False
        self._sessione_chiusa = False

    @property
    def session_id(self) -> str:
        """Identificativo della sessione CLI corrente."""
        return self._session_id

    @property
    def turn_id(self) -> str | None:
        """Turno attivo, oppure None fuori da un'esecuzione."""
        return self._turn_id

    def emetti(self, tipo: str, **campi: Any) -> dict[str, Any]:
        """Costruisce l'evento e lo consegna all'eventuale persistenza."""
        evento = self._costruisci_evento(tipo, campi)
        self._persisti(evento)
        return evento

    def _persisti(self, evento: dict[str, Any]) -> None:
        """Hook di persistenza: nessuna I/O nella variante in memoria."""
        del evento

    def _costruisci_evento(
        self,
        tipo: str,
        campi: dict[str, Any],
    ) -> dict[str, Any]:
        """Associa timestamp e correlazione di sessione/turno ai campi liberi."""
        evento: dict[str, Any] = {
            "ts": _iso_utc(),
            "tipo": tipo,
            "session_id": self._session_id,
            "turn_id": self._turn_id,
        }
        evento.update(campi)
        return evento

    def inizia_sessione(self, **campi: Any) -> None:
        """Marca l'avvio del processo CLI."""
        self._sessione_chiusa = False
        self.emetti("session_start", **campi)

    def termina_sessione(self, **campi: Any) -> None:
        """Marca la chiusura ordinata; idempotente se già emesso."""
        if self._sessione_chiusa:
            return
        self._sessione_chiusa = True
        self._chiudi_turno_attivo()
        self.emetti("session_end", **campi)

    def inizia_turno(self, percorso: str, alias: str, richiesta: str) -> str:
        """Apre un turno e avvia il clock dello stall."""
        anteprima, lunghezza = preview_richiesta(richiesta)
        with self._lock:
            self._turn_id = uuid4().hex
            self._percorso_turno = percorso
            self._alias_turno = alias
            self._stall_emesso = False
            self._ultimo_heartbeat = self._orologio_heartbeat()
            turn_id = self._turn_id
        self.emetti(
            "turn_start",
            percorso=percorso,
            alias=alias,
            preview=anteprima,
            lunghezza_richiesta=lunghezza,
        )
        return turn_id

    def termina_turno(
        self,
        esito: str,
        note: tuple[str, ...] | list[str] = (),
        metriche: dict[str, float | int] | None = None,
    ) -> None:
        """Chiude il turno con esito e snapshot metriche opzionale."""
        self.emetti(
            "turn_end",
            esito=esito,
            note=list(note),
            metriche=metriche,
            percorso=self._percorso_turno,
            alias=self._alias_turno,
        )
        self._chiudi_turno_attivo()

    def registra_turn_error(self, errore: BaseException) -> None:
        """Eccezione non gestita dalla pipeline, con traceback."""
        self.emetti(
            "turn_error",
            eccezione=type(errore).__name__,
            messaggio=tronca_messaggio(str(errore)),
            traceback=formatta_traceback(errore),
        )
        self._chiudi_turno_attivo()

    def registra_errore_llm(self, errore: BaseException) -> None:
        """Errore del provider durante una generazione."""
        self.heartbeat()
        self.emetti(
            "llm_error",
            eccezione=type(errore).__name__,
            messaggio=tronca_messaggio(str(errore)),
            traceback=formatta_traceback(errore),
        )

    def registra_errore_tool(self, nome: str, errore: BaseException) -> None:
        """Tool ReAct o diretto che ha sollevato un'eccezione."""
        self.heartbeat()
        self.emetti(
            "tool_error",
            tool=nome,
            eccezione=type(errore).__name__,
            messaggio=tronca_messaggio(str(errore)),
            traceback=formatta_traceback(errore),
        )

    def registra_diagnostica(
        self,
        messaggio: str,
        tipo_eccezione: str | None = None,
    ) -> None:
        """Fallimento convertito in testo utente (LLM vuoto, sintesi)."""
        self.emetti(
            "diagnostica",
            messaggio=tronca_messaggio(messaggio),
            eccezione=tipo_eccezione,
        )

    def registra_avvio(self, fase: str, errore: BaseException) -> None:
        """Fallimento di preload, download o wizard."""
        self.emetti(
            "avvio",
            fase=fase,
            eccezione=type(errore).__name__,
            messaggio=tronca_messaggio(str(errore)),
            traceback=formatta_traceback(errore),
        )

    def registra_crash(
        self,
        errore: BaseException,
        *,
        thread: str | None = None,
    ) -> None:
        """Eccezione non catturata a livello di processo o thread."""
        campi: dict[str, Any] = {
            "eccezione": type(errore).__name__,
            "messaggio": tronca_messaggio(str(errore)),
            "traceback": formatta_traceback(errore),
        }
        if thread:
            campi["thread"] = thread
        self.emetti("crash", **campi)

    def registra_termica(
        self,
        stato: str,
        *,
        num_predict: int,
        max_iterazioni: int,
        alias: str,
    ) -> None:
        """Throttle che taglia output o iterazioni; spiega turni tronchi."""
        self.emetti(
            "termica",
            stato=stato,
            num_predict=num_predict,
            max_iterazioni=max_iterazioni,
            alias=alias,
        )

    def heartbeat(self) -> None:
        """Segnala attività (token o tool) per il watchdog di stall."""
        with self._lock:
            self._ultimo_heartbeat = self._orologio_heartbeat()

    def controlla_stall(self, soglia_sec: float = STALL_SOGLIA_SEC) -> bool:
        """
        Emette ``stall`` una sola volta se il turno è silenzioso oltre la soglia.

        Returns:
            True se è stato emesso un nuovo evento stall.
        """
        with self._lock:
            if self._turn_id is None or self._stall_emesso:
                return False
            silenzio = self._orologio_heartbeat() - self._ultimo_heartbeat
            if silenzio < soglia_sec:
                return False
            self._stall_emesso = True
        self.emetti("stall", silenzio_sec=round(silenzio, 1), soglia_sec=soglia_sec)
        return True

    def ultimi_errori(self, limite: int = ERRORI_SLASH_DEFAULT) -> list[dict[str, Any]]:
        """Eventi non-ok della sessione; vuoto se non c'è persistenza."""
        del limite
        return []

    def formatta_ultimi_errori(
        self,
        limite: int = ERRORI_SLASH_DEFAULT,
    ) -> str:
        """Testo compatto per ``/errori``."""
        errori = self.ultimi_errori(limite)
        if not errori:
            return "Nessun errore registrato in questa sessione."
        righe = [f"Ultimi {len(errori)} eventi non-ok:"]
        for evento in errori:
            righe.append(f"- {_riga_errore(evento)}")
        return "\n".join(righe)

    def avvia_controllo_stall(self) -> None:
        """No-op: solo il registro persistente avvia il thread."""
        return

    def instala_hook_processo(self) -> None:
        """No-op: gli hook di crash vivono sul registro persistente."""
        return

    def chiudi(self) -> None:
        """No-op per la variante in memoria."""
        return

    def _chiudi_turno_attivo(self) -> None:
        """Azzera correlazione e watchdog del turno corrente."""
        with self._lock:
            self._turn_id = None
            self._percorso_turno = None
            self._alias_turno = None
            self._stall_emesso = False


class RegistroProcesso(RegistroEventi):
    """JSONL append-only sotto ``output/logs/``, con dump crash e indice markdown."""

    def __init__(
        self,
        directory: Path | None = None,
        *,
        stall_soglia_sec: float = STALL_SOGLIA_SEC,
        stall_poll_sec: float = STALL_POLL_SEC,
        orologio_heartbeat: Callable[[], float] | None = None,
    ) -> None:
        super().__init__(orologio_heartbeat=orologio_heartbeat)
        self._directory = directory if directory is not None else DIRECTORY_LOG
        self._directory.mkdir(parents=True, exist_ok=True)
        giorno = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._path_jsonl = self._directory / f"{giorno}.jsonl"
        self._path_indice = self._directory / "ultimi_errori.md"
        self._path_crash = self._directory / f"crash-{os.getpid()}.txt"
        self._stall_soglia_sec = stall_soglia_sec
        self._stall_poll_sec = stall_poll_sec
        self._crash_file: Any | None = None
        self._excepthook_originale = sys.excepthook
        self._thread_hook_originale = threading.excepthook
        self._hook_installati = False
        self._atexit_registrato = False
        self._chiuso = False
        self._ferma_stall = threading.Event()
        self._thread_stall: threading.Thread | None = None

    @property
    def path_jsonl(self) -> Path:
        """File JSONL della giornata in cui è stato aperto il registro."""
        return self._path_jsonl

    @property
    def path_indice(self) -> Path:
        """Indice markdown degli ultimi errori, riscritto a ogni evento non-ok."""
        return self._path_indice

    def _persisti(self, evento: dict[str, Any]) -> None:
        """Appende una riga JSON e, se non-ok, riallinea l'indice markdown."""
        riga = json.dumps(evento, ensure_ascii=False, default=str)
        with self._lock:
            with self._path_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(riga + "\n")
        if evento_non_ok(evento):
            self._aggiorna_indice()

    def leggi_eventi(self) -> list[dict[str, Any]]:
        """Carica tutte le righe JSONL del file di sessione, saltando le corrotte."""
        if not self._path_jsonl.exists():
            return []
        eventi: list[dict[str, Any]] = []
        testo = self._path_jsonl.read_text(encoding="utf-8")
        for riga in testo.splitlines():
            if not riga.strip():
                continue
            try:
                eventi.append(json.loads(riga))
            except json.JSONDecodeError:
                continue
        return eventi

    def ultimi_errori(self, limite: int = ERRORI_SLASH_DEFAULT) -> list[dict[str, Any]]:
        """Ultimi eventi non-ok della sessione corrente, in ordine cronologico."""
        propri = [
            evento
            for evento in self.leggi_eventi()
            if evento.get("session_id") == self._session_id and evento_non_ok(evento)
        ]
        if limite <= 0:
            return []
        return propri[-limite:]

    def avvia_controllo_stall(self) -> None:
        """Avvia il thread daemon che osserva i turni senza heartbeat."""
        if self._thread_stall is not None:
            return
        self._ferma_stall.clear()
        thread = threading.Thread(
            target=self._loop_stall,
            name="assistente-registro-stall",
            daemon=True,
        )
        self._thread_stall = thread
        thread.start()

    def ferma_controllo_stall(self) -> None:
        """Ferma il watchdog; attende al più un ciclo di poll."""
        self._ferma_stall.set()
        thread = self._thread_stall
        if thread is not None and thread.is_alive():
            thread.join(timeout=self._stall_poll_sec + 1.0)
        self._thread_stall = None

    def instala_hook_processo(self) -> None:
        """Abilita faulthandler, excepthook e atexit una sola volta."""
        if self._hook_installati:
            return
        self._directory.mkdir(parents=True, exist_ok=True)
        crash_file = self._path_crash.open("a", encoding="utf-8")
        self._crash_file = crash_file
        faulthandler.enable(file=crash_file, all_threads=True)
        sys.excepthook = self._excepthook
        threading.excepthook = self._thread_excepthook
        if not self._atexit_registrato:
            atexit.register(self._atexit_chiusura)
            self._atexit_registrato = True
        self._hook_installati = True

    def chiudi(self) -> None:
        """Ferma watchdog e hook; idempotente."""
        if self._chiuso:
            return
        self._chiuso = True
        self.ferma_controllo_stall()
        self._ripristina_hook()
        crash_file = self._crash_file
        self._crash_file = None
        if crash_file is not None:
            faulthandler.disable()
            crash_file.close()

    def _loop_stall(self) -> None:
        """Poll periodico: non uccide il turno, registra soltanto."""
        while not self._ferma_stall.wait(self._stall_poll_sec):
            self.controlla_stall(self._stall_soglia_sec)

    def _aggiorna_indice(self) -> None:
        """Riscrive ``ultimi_errori.md`` con gli ultimi eventi non-ok del file."""
        errori = [
            evento
            for evento in self.leggi_eventi()
            if evento_non_ok(evento)
        ][-ULTIMI_ERRORI_INDICE:]
        linee = ["# Ultimi errori", ""]
        if not errori:
            linee.append("Nessun errore registrato.")
        else:
            for evento in errori:
                linee.append(f"- {_riga_errore(evento)}")
        self._path_indice.write_text("\n".join(linee) + "\n", encoding="utf-8")

    def _excepthook(
        self,
        tipo: type[BaseException],
        valore: BaseException,
        traccia: Any,
    ) -> None:
        """Registra il crash e delega all'hook precedente (stderr/pytest)."""
        self.registra_crash(valore)
        originale = self._excepthook_originale
        if originale is not None and originale is not self._excepthook:
            originale(tipo, valore, traccia)

    def _thread_excepthook(self, args: threading.ExceptHookArgs) -> None:
        """Stesso trattamento per eccezioni nei thread non principali."""
        if args.exc_value is not None:
            nome = args.thread.name if args.thread is not None else None
            self.registra_crash(args.exc_value, thread=nome)
        originale = self._thread_hook_originale
        if originale is not None and originale is not self._thread_excepthook:
            originale(args)

    def _ripristina_hook(self) -> None:
        """Rimette excepthook di sistema se questo registro li possedeva."""
        if not self._hook_installati:
            return
        if sys.excepthook is self._excepthook:
            sys.excepthook = self._excepthook_originale
        if threading.excepthook is self._thread_excepthook:
            threading.excepthook = self._thread_hook_originale
        self._hook_installati = False

    def _atexit_chiusura(self) -> None:
        """Se il main non ha chiuso, emette session_end e rilascia i file."""
        try:
            if self._chiuso:
                return
            self.termina_sessione(motivo="atexit")
            self.chiudi()
        except Exception:
            return


def _riga_errore(evento: dict[str, Any]) -> str:
    """Una riga leggibile per TTY e indice markdown."""
    ts = str(evento.get("ts", ""))
    tipo = str(evento.get("tipo", "evento"))
    esito = evento.get("esito")
    percorso = evento.get("percorso")
    alias = evento.get("alias")
    messaggio = evento.get("messaggio") or evento.get("eccezione") or ""
    if tipo == "turn_end" and esito:
        tipo = f"turn_end/{esito}"
    contesto = ""
    if percorso or alias:
        contesto = f" [{percorso or '-'}/{alias or '-'}]"
    note = evento.get("note") or []
    if note and not messaggio:
        messaggio = "; ".join(str(item) for item in note)
    suffisso = f" {messaggio}" if messaggio else ""
    return f"{ts} **{tipo}**{contesto}{suffisso}".strip()
