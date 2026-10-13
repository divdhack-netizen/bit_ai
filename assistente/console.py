"""Console UI: timeline, stream ReAct, barra termica e panel Rich."""

from __future__ import annotations

import os
import re
import select
import shutil
import signal
import sys
import threading
import time
from collections.abc import Callable
from enum import Enum, auto
from typing import TextIO

from rich.console import Console, Group
from rich.constrain import Constrain
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.rule import Rule
from rich.status import Status
from rich.text import Text
from rich.theme import Theme

from assistente.host import detect_power
from assistente.metriche import StatisticheTurnoLLM
from assistente.editor_riga import EditorRiga, RIGHE_GUIDA_TASTIERA, testo_con_selezione
from assistente.prompt_tty import leggi_riga_con_reflow, termios_disponibile
from assistente.splash import esegui_splash
from assistente.thermal import (
    invalida_cache_termica,
    leggi_stato_termico,
    leggi_temperatura_cpu_celsius,
    riga_barra_stato,
)
from assistente.tty_layout import (
    MARGINE_LARGHEZZA_TTY,
    altezza_composer_sticky,
    cancella_righe_sopra,
    cuu_clear_composer,
    markup_header_sessione,
    posiziona_cursore_assoluto,
    posiziona_inizio_transcript,
    pulisci_righe_assolute,
    riga_fine_scroll_sticky,
    riga_inizio_composer_sticky,
    righe_visive_contenuto,
    segmenti_wrap_contenuto_panel,
    sequenza_regione_scroll,
    sincronizza_dimensioni,
    stampa_composer,
    stampa_composer_sticky,
    stampa_header_sticky,
    stampa_messaggio_utente,
    stampa_panel,
    riga_composer,
    tronca_visibile,
    tty_size,
    leggi_ioctl_tty_size,
    larghezza_lettura,
)

LUNGHEZZA_MASSIMA_INPUT_TOOL = 200
LUNGHEZZA_MASSIMA_OSSERVAZIONE_VERBOSE = 800
_STILE_THOUGHT = "grey50 dim"
_ETICHETTA_PENSIERO = "Pensiero "
_FRASI_TOOL = {
    "scrivi_file": "Scrivo",
    "modifica_file": "Modifico",
    "crea_cartella": "Creo la cartella",
    "elenca_cartella": "Elenco",
    "elenca_albero": "Elenco l'albero",
    "trova_file": "Cerco i file",
    "cerca_testo": "Cerco nei documenti",
    "leggi_documento": "Leggo",
    "riassumi_documento": "Riassumo",
    "data_oggi": "Controllo la data",
}
_REFRESH_PANEL_LIVE_PER_SECONDO = 20
_INTERVALLO_REFRESH_PANEL_LIVE_SEC = 1 / _REFRESH_PANEL_LIVE_PER_SECONDO
# La barra termica non deve ridisegnarsi a ogni chunk: limita overhead UI sotto ~15%.
_REFRESH_BARRA_DURANTE_STREAM_PER_SECONDO = 4
_INTERVALLO_REFRESH_BARRA_STREAM_SEC = 1 / _REFRESH_BARRA_DURANTE_STREAM_PER_SECONDO
_POLLING_TERMICA_SEC = 2.0
_INTERVALLO_POLL_RESIZE_SEC = 0.05
_MARCATORI_INIZIO = ("Thought:", "Action:", "Action Input:", "Final Answer:")
_MARCATORI_FINE_THOUGHT = ("\nAction:", "\nFinal Answer:", "\nObservation:")
_MARCATORI_FINE_INPUT = ("\nObservation:",)
_ANSI = {
    "save": "\0337", "restore": "\0338", "reset_region": "\033[r",
    "clear_line": "\033[2K",
}

_PATTERN_INIZIO_THOUGHT = re.compile(r"Thought:\s*", re.IGNORECASE)
_PATTERN_INIZIO_ACTION = re.compile(r"(?:^|\n)Action:\s*", re.IGNORECASE)
_PATTERN_INIZIO_ACTION_INPUT = re.compile(r"(?:^|\n)Action Input:\s*", re.IGNORECASE)
_PATTERN_INIZIO_FINAL_ANSWER = re.compile(r"Final Answer:\s*", re.IGNORECASE)
_PATTERN_FINE_THOUGHT = re.compile(r"\n(?:Action|Final\s+Answer|Observation)\s*:", re.IGNORECASE)
_PATTERN_FINE_AZIONE_INPUT = re.compile(r"\nObservation:\s*", re.IGNORECASE)
_PATTERN_ESTRAI_FINAL_ANSWER = re.compile(r"Final Answer:\s*(.*)", re.DOTALL | re.IGNORECASE)

_PREFISSI_CACHE: dict[tuple[str, ...], tuple[str, ...]] = {}
_LUNGHEZZA_MAX_MARCATORE = max(
    len(m) for ms in (_MARCATORI_INIZIO, _MARCATORI_FINE_THOUGHT, _MARCATORI_FINE_INPUT) for m in ms
)

_TEMA = Theme({
    "user.label": "bold cyan", "agent.label": "bold green",
    "tool.name": "bold yellow", "tool.input": "dim", "tool.result": "green",
    "system": "dim", "error": "bold red", "warning": "bold yellow",
    "badge.on": "bold green", "badge.off": "dim",
    "panel.border": "cyan",
})
_CHIAVE_STATO_DA_ETICHETTA = {
    "Verbose": "verbose",
    "Verifica completa": "verifica_completa",
    "Burn protection": "burn_protection",
    "Thinking": "thinking",
}


def _stato_runtime_default() -> dict[str, bool]:
    """Flag di sessione usati da welcome e /help se il banner non è ancora stato stampato."""
    return {
        "verbose": False,
        "verifica_completa": False,
        "burn_protection": True,
        "thinking": False,
    }


def _markup_badge(etichetta: str, attivo: bool) -> str:
    """Badge on/off con gli stili del tema console."""
    stile = "badge.on" if attivo else "badge.off"
    valore = "on" if attivo else "off"
    return f"{etichetta}: [{stile}]{valore}[/]"


def _riga_stato_compatta(stato: dict[str, bool]) -> str:
    """Una riga di badge per il panel di benvenuto."""
    return " · ".join([
        _markup_badge("Verbose", stato["verbose"]),
        _markup_badge("Check", stato["verifica_completa"]),
        _markup_badge("Burn", stato["burn_protection"]),
        _markup_badge("Thinking", stato.get("thinking", False)),
    ])


def _stdin_ha_righe_in_coda(stream: TextIO) -> bool:
    if not stream.readable():
        return False
    try:
        fileno = stream.fileno()
    except (AttributeError, OSError, ValueError):
        pos = stream.tell()
        if stream.read(1):
            stream.seek(pos)
            return True
        return False
    try:
        pronti, _, _ = select.select([fileno], [], [], 0)
    except (ValueError, OSError):
        return False
    return bool(pronti)


def raccogli_input_multilinea(prima_riga: str, stream: TextIO | None = None) -> str:
    sorgente = stream or sys.stdin
    righe = [prima_riga]
    while _stdin_ha_righe_in_coda(sorgente):
        riga = sorgente.readline()
        if riga == "":
            break
        righe.append(riga.rstrip("\n\r"))
    return "\n".join(righe).strip()


def tronca_testo(testo: str, lunghezza_massima: int) -> str:
    testo_pulito = testo.strip()
    if len(testo_pulito) <= lunghezza_massima:
        return testo_pulito
    return f"{testo_pulito[: lunghezza_massima - 1]}…"


def riassumi_input_tool(nome_tool: str, action_input: str, separatore_file: str) -> str:
    """Riassume l'Action Input per la timeline UI (path o query)."""
    testo = str(action_input).strip()
    if not testo:
        return ""
    if nome_tool in {"scrivi_file", "modifica_file"}:
        prima_riga = testo.splitlines()[0].strip()
        if nome_tool == "scrivi_file" and separatore_file in testo:
            prima_riga = testo.partition(separatore_file)[0].strip()
        return tronca_testo(prima_riga, LUNGHEZZA_MASSIMA_INPUT_TOOL)
    return tronca_testo(testo.replace("\n", " "), LUNGHEZZA_MASSIMA_INPUT_TOOL)


def frase_attivita_tool(nome_tool: str) -> str:
    """Frase umana per un tool; fallback: nome con spazi al posto degli underscore."""
    return _FRASI_TOOL.get(nome_tool, nome_tool.replace("_", " "))


def _prefissi(marcatori: tuple[str, ...]) -> tuple[str, ...]:
    if marcatori not in _PREFISSI_CACHE:
        prefissi = {m.lower()[:n] for m in marcatori for n in range(1, len(m) + 1)}
        _PREFISSI_CACHE[marcatori] = tuple(sorted(prefissi, key=len, reverse=True))
    return _PREFISSI_CACHE[marcatori]


def _anticipo(testo: str, marcatori: tuple[str, ...]) -> int:
    coda = testo[-_LUNGHEZZA_MAX_MARCATORE:].lower()
    for prefisso in _prefissi(marcatori):
        if coda.endswith(prefisso):
            return len(prefisso)
    return 0


def _estrai_final_answer(testo: str) -> str:
    match = _PATTERN_ESTRAI_FINAL_ANSWER.search(testo)
    return match.group(1).strip() if match else testo.strip()


def _larghezza_lettura_console(console: Console) -> int:
    """Cap di lettura rispetto alla larghezza console già sincronizzata."""
    return larghezza_lettura(console.width)


def _render_assistente(corpo: object, larghezza: int) -> Group:
    """Titolo Assistente + corpo vincolato: stessa metafora in stream e markdown."""
    titolo = Text.from_markup("[agent.label]Assistente[/]")
    return Group(titolo, Constrain(corpo, width=larghezza))


class TimelineTurno:
    def __init__(self, console: Console) -> None:
        self._console = console
        self._status: Status | None = None
        self._verbose = False
        self._separatore_file = "---"
        self._nome_azione_stream: str | None = None
        self._input_stream_emesso = False
        self._buffer_input_azione = ""
        self._etichetta_pensiero_emessa = False
        self._riga_pensiero_aperta = False
        self._coda_pensiero = ""
        self._coda_pensiero_visibile = False

    def reset_turno(self) -> None:
        self.termina_esecuzione_tool()
        self._reset_ciclo_stream()

    def reset_ciclo_llm(self) -> None:
        self.flush_input_azione_pendente()
        self.chiudi_pensiero()
        self._reset_ciclo_stream()

    def imposta_verbose(self, attivo: bool) -> None:
        self._verbose = attivo

    def imposta_separatore_file(self, separatore: str) -> None:
        self._separatore_file = separatore

    def scrivi_pensiero(self, token: str) -> None:
        """Stream continuo del CoT: i token si accodano come un flusso in scrittura."""
        if not token:
            return
        if not self._etichetta_pensiero_emessa:
            token = _ETICHETTA_PENSIERO + token
            self._etichetta_pensiero_emessa = True
        self._emetti_dim(token)

    def emetti_pensiero(self, testo: str) -> None:
        if testo:
            self.scrivi_pensiero(testo)

    def chiudi_pensiero(self) -> None:
        if not self._riga_pensiero_aperta and not self._coda_pensiero:
            return
        residuo = self._coda_pensiero
        self._coda_pensiero = ""
        self._cancella_coda_pensiero()
        if residuo:
            self._stampa_righe_pensiero(residuo)
        elif self._riga_pensiero_aperta:
            self._console.print()
        self._riga_pensiero_aperta = False

    def _emetti_dim(self, testo: str) -> None:
        """Accoda il CoT e va a capo sulle parole, dentro la larghezza del TTY."""
        self._coda_pensiero += testo
        self._riga_pensiero_aperta = True
        larghezza = max(1, self._console.width)
        while self._coda_pensiero:
            spezzato = self._coda_pensiero.find("\n")
            if spezzato >= 0:
                riga = self._coda_pensiero[:spezzato]
                self._coda_pensiero = self._coda_pensiero[spezzato + 1:]
                self._stampa_righe_pensiero(riga)
                continue
            segmenti = segmenti_wrap_contenuto_panel(self._coda_pensiero, larghezza)
            if len(segmenti) <= 1:
                self._mostra_coda_pensiero(segmenti[0] if segmenti else "")
                return
            for segmento in segmenti[:-1]:
                self._stampa_riga_pensiero(segmento)
            self._coda_pensiero = segmenti[-1]
            self._mostra_coda_pensiero(self._coda_pensiero)
            return

    def _stampa_righe_pensiero(self, riga: str) -> None:
        larghezza = max(1, self._console.width)
        if not riga:
            self._stampa_riga_pensiero("")
            return
        for segmento in segmenti_wrap_contenuto_panel(riga, larghezza):
            self._stampa_riga_pensiero(segmento)

    def _stampa_riga_pensiero(self, riga: str) -> None:
        self._cancella_coda_pensiero()
        self._console.print(
            Text(riga, style=_STILE_THOUGHT),
            highlight=False,
            markup=False,
            overflow="crop",
            no_wrap=True,
        )

    def _mostra_coda_pensiero(self, riga: str) -> None:
        self._cancella_coda_pensiero()
        if not riga:
            return
        self._console.print(
            Text(riga, style=_STILE_THOUGHT),
            end="",
            highlight=False,
            markup=False,
            overflow="crop",
            no_wrap=True,
        )
        self._coda_pensiero_visibile = True

    def _cancella_coda_pensiero(self) -> None:
        if not self._coda_pensiero_visibile:
            return
        self._console.file.write("\r\033[K")
        self._console.file.flush()
        self._coda_pensiero_visibile = False

    def emetti_azione_da_stream(self, nome_tool: str) -> None:
        nome = nome_tool.strip()
        if not nome:
            return
        self._nome_azione_stream = nome
        self._input_stream_emesso = False
        self._buffer_input_azione = ""

    def emetti_input_azione_da_stream(self, input_grezzo: str) -> None:
        if input_grezzo and self._nome_azione_stream:
            self._buffer_input_azione += input_grezzo

    def _mostra_azione_input(self, nome_tool: str, action_input: str) -> None:
        riassunto = riassumi_input_tool(nome_tool, action_input, self._separatore_file)
        if self._nome_azione_stream == nome_tool and self._input_stream_emesso:
            return
        self._nome_azione_stream = nome_tool
        self._stampa_attivita(nome_tool, riassunto)
        self._input_stream_emesso = True

    def registra_azione_tool(self, nome_tool: str, action_input: str) -> None:
        if nome_tool != "_Exception":
            self._mostra_azione_input(nome_tool, action_input)

    def inizia_esecuzione_tool(self, nome_tool: str) -> None:
        self.termina_esecuzione_tool()
        if self._nome_azione_stream == nome_tool:
            self.flush_input_azione_pendente()
        elif not self._input_stream_emesso:
            self._stampa_attivita(nome_tool, "")
            self._nome_azione_stream = nome_tool
            self._input_stream_emesso = True
        frase = frase_attivita_tool(nome_tool)
        self._status = self._console.status(f"{frase}…", spinner="dots")
        self._status.start()

    def termina_esecuzione_tool(self) -> None:
        if self._status:
            self._status.stop()
            self._status = None

    def emetti_risultato_tool(self, osservazione: str) -> None:
        if not self._verbose:
            return
        limite = LUNGHEZZA_MASSIMA_OSSERVAZIONE_VERBOSE
        self._console.print(Text(f"  {tronca_testo(osservazione, limite)}", style="tool.result"))

    def emetti_errore_tool(self, messaggio: str) -> None:
        self.termina_esecuzione_tool()
        self._console.print(Text(f"  {tronca_testo(str(messaggio), LUNGHEZZA_MASSIMA_OSSERVAZIONE_VERBOSE)}", style="error"))

    def flush_input_azione_pendente(self) -> None:
        if not self._nome_azione_stream or self._input_stream_emesso:
            return
        riassunto = riassumi_input_tool(
            self._nome_azione_stream,
            self._buffer_input_azione,
            self._separatore_file,
        )
        self._stampa_attivita(self._nome_azione_stream, riassunto)
        self._input_stream_emesso = True

    def _reset_ciclo_stream(self) -> None:
        self._nome_azione_stream = None
        self._input_stream_emesso = False
        self._buffer_input_azione = ""
        self._etichetta_pensiero_emessa = False
        self._riga_pensiero_aperta = False
        self._coda_pensiero = ""
        self._coda_pensiero_visibile = False

    def _stampa_attivita(self, nome_tool: str, riassunto: str) -> None:
        frase = frase_attivita_tool(nome_tool)
        riga = Text()
        riga.append(frase, style="tool.name")
        if riassunto:
            riga.append(": ", style="tool.name")
            riga.append(riassunto, style="tool.input")
        else:
            riga.append("…", style="tool.name")
        if self._verbose:
            riga.append(f"  ({nome_tool})", style="dim")
        self._console.print(riga)


class _StatoStream(Enum):
    ATTESA = auto()
    THOUGHT = auto()
    AZIONE = auto()
    AZIONE_INPUT = auto()
    FINAL_ANSWER = auto()


def _altezza_live(live: Live) -> int:
    """Altezza dell'ultimo frame Live; 0 se il render non è ancora misurabile."""
    render = getattr(live, "_live_render", None)
    if render is None:
        return 0
    altezza = getattr(render, "last_render_height", 0)
    return altezza if isinstance(altezza, int) else 0


class RendererPanelStream:
    def __init__(
        self,
        console: Console,
        timeline: TimelineTurno,
        on_render: Callable[[], None] | None = None,
        righe_riservate: Callable[[], int] | None = None,
        sospendi_scroll: Callable[[], None] | None = None,
        ripristina_scroll: Callable[[], None] | None = None,
    ) -> None:
        """
        Stream ReAct → timeline + Live per la Final Answer (titolo + corpo, senza box).

        Args:
            console: Console Rich condivisa con la barra termica.
            timeline: Emitter degli eventi Thought/Action.
            on_render: Callback dopo i soli refresh Live effettivi.
            righe_riservate: Chrome sticky (header + composer + stato) da sottrarre.
            sospendi_scroll: Toglie la scroll-region prima di una stampa più alta del viewport.
            ripristina_scroll: Rimette header e stato dopo quella stampa.
        """
        self._console = console
        self._timeline = timeline
        self._on_render = on_render
        self._righe_riservate = righe_riservate
        self._sospendi_scroll = sospendi_scroll
        self._ripristina_scroll = ripristina_scroll
        self._scroll_sospeso = False
        self._live: Live | None = None
        self._live_aperto = False
        self._scroll_libero = False
        self._coda_scroll = ""
        self._coda_scroll_visibile = False
        self._stato = _StatoStream.ATTESA
        self._anticipo_buf = ""
        self._buffer_finale: list[str] = []
        self._testo_stream = Text()
        self._prossimo_refresh_live = 0.0
        self._buffer_nome_azione = ""
        self._conteggio_refresh = 0
        self._live_lock = threading.Lock()
        self._refresh_dirty = False
        self._stop_refresh = threading.Event()
        self._refresh_thread: threading.Thread | None = None

    def _sincronizza(self) -> None:
        extra = self._righe_riservate() if self._righe_riservate is not None else 0
        sincronizza_dimensioni(
            self._console,
            riserva_barra=True,
            righe_riservate=extra,
        )

    @property
    def conteggio_refresh(self) -> int:
        """Refresh Live eseguiti dall'avvio del renderer (diagnostica benchmark)."""
        return self._conteggio_refresh

    def reset_turno(self) -> None:
        """Chiude eventuale Live e azzera parser e contenuto del turno precedente."""
        self._ferma_live()
        self._reset_parser()
        self._reset_contenuto_finale()

    def inizia_risposta_diretta(self) -> None:
        """
        Apre lo stream Final Answer senza attendere marcatori ReAct.

        Usato dalla pipeline direct: il testo grezzo è già la risposta utente,
        quindi ogni chunk deve entrare nel Panel plain-text rate-limited.
        """
        # Chiude eventuali Live/parziali di un tentativo precedente dello stesso turno.
        if self._live_aperto:
            self._ferma_live()
        self._stato = _StatoStream.FINAL_ANSWER
        self._anticipo_buf = ""
        self._buffer_nome_azione = ""
        self._reset_contenuto_finale()

    def reset_chiamata_llm(self) -> None:
        """Chiude le sezioni ReAct incomplete senza perdere una Final Answer attiva."""
        # Un prefisso sospeso va elaborato prima di cambiare ciclo LLM.
        if self._anticipo_buf:
            self._elabora(self._anticipo_buf)
            self._anticipo_buf = ""
        # L'input tool parziale deve essere visibile prima del reset della timeline.
        if self._stato is _StatoStream.AZIONE_INPUT:
            self._timeline.flush_input_azione_pendente()
        # Un nome tool senza newline finale è comunque un'azione valida.
        if self._stato is _StatoStream.AZIONE:
            nome = self._buffer_nome_azione.strip()
            # I frammenti vuoti non devono generare eventi tool fantasma.
            if nome:
                self._timeline.emetti_azione_da_stream(nome)
            self._buffer_nome_azione = ""
        self._timeline.reset_ciclo_llm()
        # Una Final Answer può attraversare callback LLM e deve restare visibile.
        if self._stato is not _StatoStream.FINAL_ANSWER:
            # Un Live parziale appartiene al tentativo fallito e va chiuso.
            if self._live_aperto:
                self._ferma_live()
            self._reset_contenuto_finale()
        # Gli stati intermedi non devono contaminare il parser del retry.
        if self._stato not in (_StatoStream.ATTESA, _StatoStream.FINAL_ANSWER):
            self._stato = _StatoStream.ATTESA

    def scrivi_token(self, token: str) -> None:
        """
        Inoltra un chunk al parser ReAct preservando eventuali marcatori spezzati.

        Args:
            token: Frammento testuale; una stringa vuota non produce effetti.
        """
        # I callback LLM possono emettere chunk vuoti, che non richiedono parsing.
        if not token:
            return
        buffer = self._anticipo_buf + token
        self._anticipo_buf = ""
        self._elabora(buffer)

    def finalizza(self, risultato: str) -> None:
        """
        Finalizza lo stream convertendo l'intera risposta in Markdown una sola volta.

        Args:
            risultato: Risposta LLM completa usata come fallback senza stream finale.
        """
        # Un marcatore parziale in coda deve essere incluso prima della finalizzazione.
        if self._anticipo_buf:
            self._elabora(self._anticipo_buf)
            self._anticipo_buf = ""
        self._timeline.termina_esecuzione_tool()
        testo_stream = "".join(self._buffer_finale).strip()
        testo_finale = testo_stream or _estrai_final_answer(risultato)
        self._ferma_live(testo_finale if testo_finale else None)
        self._reset_parser()
        self._reset_contenuto_finale()

    def reflow_su_resize(self) -> None:
        """Ridisegna il Panel plain-text dopo un resize consumato dal main thread."""
        # Rich Live non è sicuro da ridimensionare dal thread della barra termica.
        if threading.current_thread() is not threading.main_thread():
            return
        self._sincronizza()
        # Senza Live basta il sync; l'eventuale chrome verrà gestito separatamente.
        if not self._live_aperto or self._live is None:
            return
        self._live.update(
            _render_assistente(self._testo_stream, _larghezza_lettura_console(self._console)),
            refresh=True,
        )
        self._prossimo_refresh_live = (
            time.monotonic() + _INTERVALLO_REFRESH_PANEL_LIVE_SEC
        )
        self._notifica_render()

    @property
    def ha_live_aperto(self) -> bool:
        """True se lo stream Final Answer è in corso con Live attivo."""
        return self._live_aperto

    def _reset_parser(self) -> None:
        """Riporta il parser ReAct allo stato iniziale senza toccare il contenuto."""
        self._stato = _StatoStream.ATTESA
        self._anticipo_buf = ""
        self._buffer_nome_azione = ""

    def _reset_contenuto_finale(self) -> None:
        """Azzera buffer lineare e renderable plain-text della Final Answer."""
        self._buffer_finale.clear()
        self._testo_stream = Text()
        self._prossimo_refresh_live = 0.0
        self._scroll_libero = False
        self._coda_scroll = ""
        self._coda_scroll_visibile = False
        self._scroll_sospeso = False

    def _notifica_render(self) -> None:
        """Notifica un refresh reale affinché la barra possa ripristinarsi."""
        # La callback è opzionale per consentire renderer isolati nei test e nei client.
        if self._on_render is not None:
            self._on_render()

    def _ferma_live(self, aggiorna_finale: str | None = None) -> None:
        """
        Chiude Live e, se disponibile, sostituisce il plain-text con Markdown finale.

        Args:
            aggiorna_finale: Testo completo da renderizzare; None mantiene il plain-text.
        """
        self._sincronizza()
        larghezza = _larghezza_lettura_console(self._console)
        # Lo stream alto è già nello scrollback: un Markdown successivo duplicherebbe il testo.
        if self._scroll_libero:
            self._ferma_thread_refresh()
            self._chiudi_coda_scroll()
            self._console.print()
            self._scroll_libero = False
            self._ripristina_scroll_tty()
            self._notifica_render()
            return
        # Senza Live il risultato fallback va comunque stampato con la stessa metafora.
        if not self._live_aperto or self._live is None:
            self._ferma_thread_refresh()
            if aggiorna_finale:
                renderabile = _render_assistente(
                    Markdown(aggiorna_finale, code_theme="monokai"),
                    larghezza,
                )
                self._stampa_risposta(renderabile)
            return
        self._ferma_thread_refresh()
        renderabile = None
        if aggiorna_finale is not None:
            renderabile = _render_assistente(
                Markdown(aggiorna_finale, code_theme="monokai"),
                larghezza,
            )
        if renderabile is not None and self._render_supera_viewport(renderabile):
            # stop() con overflow visible ristampa dentro DECSTBM e taglia l'inizio.
            self._rilascia_live_senza_dump(self._live)
            self._stampa_risposta(renderabile)
            self._notifica_render()
            return
        if renderabile is not None:
            self._live.update(renderabile, refresh=True)
        self._live.stop()
        self._live = None
        self._live_aperto = False
        self._prossimo_refresh_live = 0.0
        self._refresh_dirty = False
        self._notifica_render()

    def _append_finale(self, testo: str) -> None:
        """
        Accoda un chunk in O(1) e aggiorna il plain-text entro il limite di frequenza.

        Args:
            testo: Frammento della Final Answer, conservato senza trasformazioni.
        """
        # Frammenti vuoti non devono aprire un Live privo di contenuto visibile.
        if not testo:
            return
        with self._live_lock:
            self._buffer_finale.append(testo)
            self._testo_stream.append(testo)
            self._refresh_dirty = True
            scroll_libero = self._scroll_libero
        if scroll_libero:
            self._emetti_testo_scroll_libero(testo)
            return
        # Il primo contenuto deve essere renderizzato immediatamente, senza timer.
        if not self._live_aperto:
            self._avvia_live()
            return

    def _avvia_live(self) -> None:
        """Apre Live con il primo contenuto plain-text già presente e visibile."""
        self._sincronizza()
        larghezza = _larghezza_lettura_console(self._console)
        self._console.print()
        self._live = Live(
            _render_assistente(self._testo_stream, larghezza),
            console=self._console,
            auto_refresh=False,
            refresh_per_second=_REFRESH_PANEL_LIVE_PER_SECONDO,
            transient=False,
            redirect_stdout=False,
            redirect_stderr=False,
            # L'ellipsis impedisce alle risposte lunghe di invadere la barra riservata.
            vertical_overflow="ellipsis",
        )
        self._live.start(refresh=True)
        self._live_aperto = True
        self._conteggio_refresh += 1
        self._refresh_dirty = False
        if self._live_oltre_viewport(self._live):
            with self._live_lock:
                self._passa_a_scroll_libero_locked(self._live)
            self._notifica_render()
            return
        self._stop_refresh.clear()
        self._refresh_thread = threading.Thread(
            target=self._loop_refresh_live,
            name="assistente-live-refresh",
            daemon=True,
        )
        self._refresh_thread.start()
        self._prossimo_refresh_live = (
            time.monotonic() + _INTERVALLO_REFRESH_PANEL_LIVE_SEC
        )
        self._notifica_render()

    def _loop_refresh_live(self) -> None:
        """Ridisegna Live fuori dal thread di decode, al massimo 20 Hz."""
        while not self._stop_refresh.wait(_INTERVALLO_REFRESH_PANEL_LIVE_SEC):
            if not self._esegui_refresh_live():
                return

    def _esegui_refresh_live(self) -> bool:
        """
        Applica un refresh Live se sporco; passa a scroll libero se il Panel satura il TTY.

        Returns:
            False se Live è stato chiuso e il worker deve uscire.
        """
        notifica = False
        with self._live_lock:
            live = self._live
            if live is None:
                return False
            if self._refresh_dirty:
                self._refresh_dirty = False
                live.refresh()
                self._conteggio_refresh += 1
                notifica = True
            if self._live_oltre_viewport(live):
                self._passa_a_scroll_libero_locked(live)
                notifica = True
                live = None
        if notifica:
            self._notifica_render()
        return live is not None

    def _live_oltre_viewport(self, live: Live) -> bool:
        """True quando il frame Live occupa tutta l'altezza utile (crop/ellipsis)."""
        altezza = _altezza_live(live)
        return altezza > 0 and altezza >= self._console.height

    def _passa_a_scroll_libero_locked(self, live: Live) -> None:
        """
        Abbandona il ridisegno in-place e versa il testo nello scrollback del TTY.

        Rich Live con overflow ellipsis ritaglia la risposta all'altezza del
        viewport e riposiziona il cursore a ogni refresh: il terminale non ha
        nulla da scorrere finché Live non viene chiuso. Dopo questa transizione
        i nuovi token vengono solo accodati.
        """
        self._rilascia_live_senza_dump(live)
        self._scroll_libero = True
        self._sospendi_scroll_tty()
        self._refresh_dirty = False
        self._console.print(
            Text.from_markup("[agent.label]Assistente[/]"),
            highlight=False,
        )
        if self._testo_stream.plain:
            self._emetti_testo_scroll_libero(self._testo_stream)

    def _rilascia_live_senza_dump(self, live: Live) -> None:
        """Chiude Live cancellando solo l'area già disegnata, senza ristampare il Panel."""
        with live._lock:
            render = live._live_render
            if live.console.is_terminal and _altezza_live(live) > 0:
                live.console.control(render.restore_cursor())
            render._shape = None
        live.refresh = lambda: None
        live.transient = False
        live.stop()
        self._live = None
        self._live_aperto = False

    def _render_supera_viewport(self, renderabile: object) -> bool:
        """True se il frame, senza crop, è più alto dell'area utile della console."""
        opzioni = self._console.options.update(height=None)
        altezza = len(self._console.render_lines(renderabile, opzioni))
        return altezza > self._console.height

    def _stampa_risposta(self, renderabile: object) -> None:
        """Stampa fuori da DECSTBM così lo storico tiene anche le righe in alto."""
        self._sospendi_scroll_tty()
        self._console.print()
        self._console.print(renderabile)
        self._ripristina_scroll_tty()

    def _sospendi_scroll_tty(self) -> None:
        """Una sola volta per risposta: le stampe successive usano lo scrollback intero."""
        if self._scroll_sospeso or self._sospendi_scroll is None:
            return
        self._sospendi_scroll()
        self._scroll_sospeso = True

    def _ripristina_scroll_tty(self) -> None:
        """Rimette la scroll-region dopo la stampa alta."""
        if not self._scroll_sospeso:
            return
        self._scroll_sospeso = False
        if self._ripristina_scroll is not None:
            self._ripristina_scroll()

    def _emetti_testo_scroll_libero(self, testo: str | Text) -> None:
        """Accoda testo nello scrollback, a capo sulle parole dentro il margine TTY."""
        if isinstance(testo, Text):
            frammento = testo.plain
        else:
            frammento = testo
        if not frammento:
            return
        self._coda_scroll += frammento
        self._svuota_coda_scroll()

    def _svuota_coda_scroll(self) -> None:
        """Stampa le righe già chiuse e tiene a video solo l'ultima incompleta."""
        larghezza = _larghezza_lettura_console(self._console)
        while self._coda_scroll:
            spezzato = self._coda_scroll.find("\n")
            if spezzato >= 0:
                riga = self._coda_scroll[:spezzato]
                self._coda_scroll = self._coda_scroll[spezzato + 1:]
                self._concludi_righe_scroll(riga, larghezza)
                continue
            segmenti = segmenti_wrap_contenuto_panel(self._coda_scroll, larghezza)
            if len(segmenti) <= 1:
                self._mostra_coda_scroll(segmenti[0] if segmenti else "")
                return
            for segmento in segmenti[:-1]:
                self._concludi_riga_scroll(segmento)
            self._coda_scroll = segmenti[-1]
            self._mostra_coda_scroll(self._coda_scroll)
            return

    def _concludi_righe_scroll(self, riga: str, larghezza: int) -> None:
        """Va a capo su una riga già terminata, spezzandola se supera la larghezza."""
        if not riga:
            self._concludi_riga_scroll("")
            return
        for segmento in segmenti_wrap_contenuto_panel(riga, larghezza):
            self._concludi_riga_scroll(segmento)

    def _concludi_riga_scroll(self, riga: str) -> None:
        """Sostituisce l'eventuale coda visibile e chiude la riga con un newline."""
        self._cancella_coda_scroll()
        self._console.print(
            Text(riga),
            highlight=False,
            markup=False,
            overflow="crop",
            no_wrap=True,
        )

    def _mostra_coda_scroll(self, riga: str) -> None:
        """Riscrive l'ultima riga incompleta, senza mandare a capo il terminale."""
        self._cancella_coda_scroll()
        if not riga:
            return
        self._console.print(
            Text(riga),
            end="",
            highlight=False,
            markup=False,
            overflow="crop",
            no_wrap=True,
        )
        self._coda_scroll_visibile = True

    def _cancella_coda_scroll(self) -> None:
        """Torna a colonna 1 e cancella la coda già disegnata sulla riga corrente."""
        if not self._coda_scroll_visibile:
            return
        self._console.file.write("\r\033[K")
        self._console.file.flush()
        self._coda_scroll_visibile = False

    def _chiudi_coda_scroll(self) -> None:
        """Chiude l'ultima riga incompleta così il prompt non resta a metà testo."""
        if not self._coda_scroll and not self._coda_scroll_visibile:
            return
        residuo = self._coda_scroll
        self._coda_scroll = ""
        self._cancella_coda_scroll()
        if residuo:
            self._concludi_righe_scroll(
                residuo,
                _larghezza_lettura_console(self._console),
            )

    def _ferma_thread_refresh(self) -> None:
        """Ferma il worker di refresh e attende l'uscita."""
        self._stop_refresh.set()
        thread = self._refresh_thread
        self._refresh_thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def _aggiorna_live_se_scaduto(self) -> None:
        """Segna il Panel come sporco: il worker esegue il refresh."""
        if self._live is None:
            return
        self._refresh_dirty = True

    def _elabora(self, testo: str) -> None:
        """
        Consuma un frammento finché ogni transizione ReAct interna è stata elaborata.

        Args:
            testo: Testo non ancora classificato dal parser.
        """
        residuo: str | None = testo
        # Un singolo chunk può contenere più sezioni ReAct consecutive.
        while residuo:
            residuo = self._elabora_frammento(residuo)

    def _elabora_frammento(self, testo: str) -> str | None:
        """
        Instrada il testo secondo lo stato ReAct corrente.

        Args:
            testo: Frammento da consumare nello stato corrente.

        Returns:
            Residuo da riesaminare dopo una transizione, oppure None se consumato.
        """
        # Dopo Final Answer ogni frammento appartiene al contenuto visibile.
        if self._stato is _StatoStream.FINAL_ANSWER:
            self._append_finale(testo)
            return None
        # Thought termina solo quando compare un marcatore ReAct compatibile.
        if self._stato is _StatoStream.THOUGHT:
            return self._consuma_fino_marcatore(
                testo, _PATTERN_FINE_THOUGHT, _MARCATORI_FINE_THOUGHT,
                self._timeline.emetti_pensiero, self._timeline.chiudi_pensiero,
            )
        # Il nome azione è delimitato dal primo newline, anche tra chunk diversi.
        if self._stato is _StatoStream.AZIONE:
            return self._consuma_azione(testo)
        # Action Input termina all'Observation e viene riassunto dalla timeline.
        if self._stato is _StatoStream.AZIONE_INPUT:
            return self._consuma_fino_marcatore(
                testo, _PATTERN_FINE_AZIONE_INPUT, _MARCATORI_FINE_INPUT,
                self._timeline.emetti_input_azione_da_stream,
                self._timeline.flush_input_azione_pendente,
            )
        return self._cerca_marcatori(testo)

    def _consuma_fino_marcatore(
        self,
        testo: str,
        pattern_fine: re.Pattern[str],
        marcatori_anticipo: tuple[str, ...],
        on_chunk: Callable[[str], None],
        on_chiusura: Callable[[], None] | None = None,
    ) -> str | None:
        """
        Emette testo fino a un marcatore, conservandone gli anticipi tra chunk.

        Args:
            testo: Frammento corrente.
            pattern_fine: Espressione regolare che chiude la sezione.
            marcatori_anticipo: Marcatori di cui preservare i prefissi finali.
            on_chunk: Destinazione del contenuto consumato.
            on_chiusura: Callback opzionale eseguita alla chiusura.

        Returns:
            Testo dal marcatore in poi da riesaminare, oppure None.
        """
        match_fine = pattern_fine.search(testo)
        # Un marcatore completo chiude la sezione e va riesaminato dallo stato ATTESA.
        if match_fine:
            on_chunk(testo[: match_fine.start()])
            # Alcune sezioni richiedono un flush esplicito prima della transizione.
            if on_chiusura:
                on_chiusura()
            self._stato = _StatoStream.ATTESA
            return testo[match_fine.start() :]
        n = _anticipo(testo, marcatori_anticipo)
        # Un prefisso in coda potrebbe completarsi nel chunk successivo.
        if n:
            on_chunk(testo[:-n])
            self._anticipo_buf = testo[-n:]
            return None
        on_chunk(testo)
        return None

    def _consuma_azione(self, testo: str) -> str | None:
        """
        Accumula il nome tool fino al newline che conclude la dichiarazione Action.

        Args:
            testo: Frammento successivo al marcatore Action.

        Returns:
            Testo dal newline in poi da riesaminare, oppure None.
        """
        idx = testo.find("\n")
        # Il newline rende completo il nome e consente di tornare alla ricerca marcatori.
        if idx >= 0:
            nome = (self._buffer_nome_azione + testo[:idx]).strip()
            self._buffer_nome_azione = ""
            # Un nome vuoto non rappresenta una vera invocazione tool.
            if nome:
                self._timeline.emetti_azione_da_stream(nome)
            self._stato = _StatoStream.ATTESA
            return testo[idx:]
        self._buffer_nome_azione += testo
        return None

    def _cerca_marcatori(self, testo: str) -> str | None:
        """
        Cerca l'inizio della prossima sezione ReAct mantenendo la priorità storica.

        Args:
            testo: Frammento ricevuto mentre il parser è in attesa.

        Returns:
            Contenuto successivo al marcatore, oppure None se il chunk è consumato.
        """
        # La priorità esistente evita regressioni nel riconoscimento dei marcatori.
        for pattern, stato, offset_fn in (
            (_PATTERN_INIZIO_FINAL_ANSWER, _StatoStream.FINAL_ANSWER, lambda m: m.end()),
            (_PATTERN_INIZIO_THOUGHT, _StatoStream.THOUGHT, lambda m: m.start()),
            (_PATTERN_INIZIO_ACTION_INPUT, _StatoStream.AZIONE_INPUT, lambda m: m.end()),
            (_PATTERN_INIZIO_ACTION, _StatoStream.AZIONE, lambda m: m.end()),
        ):
            match = pattern.search(testo)
            # Il primo marcatore secondo la priorità storica determina il nuovo stato.
            if match:
                # Una nuova Final Answer sostituisce solo eventuale contenuto parziale.
                if stato is _StatoStream.FINAL_ANSWER:
                    self._reset_contenuto_finale()
                # Una nuova Action non deve ereditare frammenti del nome precedente.
                if stato is _StatoStream.AZIONE:
                    self._buffer_nome_azione = ""
                self._stato = stato
                return testo[offset_fn(match):]

        n = _anticipo(testo, _MARCATORI_INIZIO)
        # Un marcatore spezzato deve restare sospeso fino al prossimo token.
        if n:
            # In attesa non esiste contenuto finale precedente da mostrare.
            if self._stato is _StatoStream.ATTESA:
                self._anticipo_buf = testo[-n:]
            else:
                self._append_finale(testo[:-n])
                self._anticipo_buf = testo[-n:]
            return None
        # Stati diversi da ATTESA gestiscono il testo nei rami superiori.
        if self._stato is not _StatoStream.ATTESA:
            return None
        # Testo senza marcatori non va nel panel: evita risposte temporanee sovrascritte.
        return None


class BarraTermica:
    def __init__(self, console: Console, stream: TextIO | None = None) -> None:
        """
        Barra di stato sull'ultima riga TTY (termica + metriche LLM).

        Args:
            console: Console Rich condivisa; width/height restano allineati al TTY.
            stream: Output per sequenze ANSI (default stdout).
        """
        self._console = console
        self._stream = stream or sys.stdout
        self._lock = threading.Lock()
        self._attiva = False
        self._mostra_metriche = False
        self._righe_riservate: int | None = None
        self._header_righe = 0
        self._righe_composer = 0
        self._regione_sospesa = False
        self._ferma = threading.Event()
        self._thread: threading.Thread | None = None
        self._input_in_corso = False
        self._metriche: StatisticheTurnoLLM | None = None
        self._sorgente_metriche: Callable[[], StatisticheTurnoLLM | None] | None = None
        self._ultima_riga = ""
        self._resize_richiesto = threading.Event()
        self._on_resize: Callable[[], None] | None = None
        # Handler precedente (callable / SIG_DFL / SIG_IGN) da ripristinare in ferma()
        self._sigwinch_precedente: object | None = None
        self._sigwinch_installato = False

    def imposta_on_resize(self, callback: Callable[[], None] | None) -> None:
        """
        Registra un callback eseguito dopo sync/scroll-region su resize TTY.

        Args:
            callback: Funzione che accoda il lavoro Live sul main thread, o None.
        """
        self._on_resize = callback

    def avvia(self) -> None:
        if self._attiva or not self._console.is_terminal:
            return
        self._attiva = True
        self._ferma.clear()
        self._resize_richiesto.clear()
        self._sincronizza_console()
        self._riserva_ultima_riga()
        self.ridisegna()
        self._installa_sigwinch()
        self._thread = threading.Thread(target=self._polling, name="barra-termica", daemon=True)
        self._thread.start()

    def ferma(self) -> None:
        if not self._attiva and self._thread is None:
            return
        self._ferma.set()
        # Sveglia il poll in attesa del flag resize
        self._resize_richiesto.set()
        if self._thread:
            self._thread.join(timeout=_POLLING_TERMICA_SEC + 1)
            self._thread = None
        self._ripristina_sigwinch()
        self._attiva = False
        self._rilascia()

    def imposta_mostra_metriche(self, attivo: bool) -> None:
        """In verbose la barra aggiunge TTFT, prefill e le altre latenze."""
        self._mostra_metriche = attivo

    def imposta_input_in_corso(self, attivo: bool) -> None:
        self._input_in_corso = attivo
        if not attivo:
            self.imposta_righe_composer(0)

    def imposta_righe_composer(self, righe: int) -> None:
        """Riserva N righe sopra lo stato per il composer sticky."""
        nuovo = max(0, int(righe))
        if nuovo == self._righe_composer:
            return
        self._righe_composer = nuovo
        if self._attiva and self._console.is_terminal:
            self._riserva_ultima_riga()

    def righe_chrome(self) -> int:
        """Righe sottratte al transcript: header + composer + stato."""
        if not self._attiva:
            return 1
        return self._header_righe + 1 + self._righe_composer

    @property
    def header_attivo(self) -> bool:
        """True se la riga 1 è fuori dalla scroll-region."""
        return self._attiva and self._header_righe > 0

    @property
    def input_in_corso(self) -> bool:
        """True durante la lettura del prompt utente."""
        return self._input_in_corso

    @property
    def attiva(self) -> bool:
        """True se la barra ha riservato l'ultima riga TTY."""
        return self._attiva

    def dopo_clear(self) -> None:
        """Re-riserva la scroll-region dopo un console.clear() (senza ridisegnare ancora)."""
        if not self._attiva or not self._console.is_terminal:
            return
        self._riserva_ultima_riga()
        self._ultima_riga = ""

    def imposta_metriche_turno(self, metriche: StatisticheTurnoLLM | None) -> None:
        if metriche == self._metriche:
            return
        self._metriche = metriche
        self._ultima_riga = ""
        self.ridisegna()

    def aggiorna_metriche_turno(self, metriche: StatisticheTurnoLLM | None) -> None:
        """Memorizza lo snapshot senza ridisegnare: il prossimo refresh Live lo mostra."""
        self._metriche = metriche

    def imposta_sorgente_metriche(
        self,
        sorgente: Callable[[], StatisticheTurnoLLM | None] | None,
    ) -> None:
        """
        Collega un accumulatore vivo letto a ogni ridisegno della barra.

        Args:
            sorgente: snapshot() dell'accumulatore del turno, o None a fine turno.
        """
        self._sorgente_metriche = sorgente

    def ridisegna(self, *, forza: bool = False) -> None:
        """
        Aggiorna la riga di stato in fondo al TTY.

        Args:
            forza: Se True, ridisegna anche durante input_in_corso (dopo clear/reflow).
        """
        if not self._attiva or not self._console.is_terminal or self._regione_sospesa:
            return
        # Durante input evita di sovrascrivere il cursore, salvo reflow esplicito
        if self._input_in_corso and not forza:
            return
        self._sincronizza_console()
        colonne, _ = tty_size(self._console)
        # Budget celle: lascia 1 colonna di margine come prima
        max_colonne = max(0, colonne - 1)
        metriche = self._metriche
        sorgente = self._sorgente_metriche
        # Il poll termico e il Live devono rileggere l'accumulatore, non uno snapshot fermo.
        if sorgente is not None:
            metriche = sorgente()
        riga = riga_barra_stato(
            leggi_stato_termico(),
            leggi_temperatura_cpu_celsius(),
            metriche,
            ansi=True,
            max_colonne=max_colonne,
            power=detect_power(),
            verbose=self._mostra_metriche,
        )
        if riga != self._ultima_riga:
            self._disegna(riga)
            self._ultima_riga = riga

    def _sincronizza_console(self) -> None:
        sincronizza_dimensioni(
            self._console,
            riserva_barra=self._attiva,
            righe_riservate=self.righe_chrome() if self._attiva else 0,
        )

    def _scrivi(self, sequenza: str) -> None:
        with self._lock:
            self._stream.write(sequenza)
            self._stream.flush()

    def _riserva_ultima_riga(self) -> None:
        _, righe = tty_size(self._console)
        if righe < 2:
            self._header_righe = 0
            return
        header = 1 if righe >= 4 else 0
        basso = 1 + self._righe_composer
        top = 1 + header
        bottom = righe - basso
        if bottom < top:
            bottom = top
        a = _ANSI
        if header == 0 and self._righe_composer == 0:
            self._scrivi(
                f"\n{a['save']}{sequenza_regione_scroll(top, bottom)}"
                f"{a['restore']}\033[1A"
            )
        else:
            self._scrivi(f"{a['save']}{sequenza_regione_scroll(top, bottom)}{a['restore']}")
        self._header_righe = header
        self._righe_riservate = righe

    @property
    def regione_sospesa(self) -> bool:
        """True mentre il transcript usa lo scrollback intero."""
        return self._regione_sospesa

    def sospendi_regione_scroll(self) -> None:
        """Toglie DECSTBM: una stampa più alta del viewport entra nello scrollback."""
        if not self._attiva or not self._console.is_terminal:
            return
        self._regione_sospesa = True
        a = _ANSI
        self._scrivi(f"{a['save']}{a['reset_region']}{a['restore']}")

    def ripristina_regione_scroll(self) -> None:
        """Riapplica la scroll-region e ridisegna la riga di stato."""
        self._regione_sospesa = False
        if not self._attiva or not self._console.is_terminal:
            return
        self._riserva_ultima_riga()
        self._ultima_riga = ""
        self.ridisegna(forza=True)

    def fai_spazio_in_basso(self, righe: int) -> None:
        """
        Spinge il testo in alto nello scrollback, lasciando righe libere in basso.

        Il composer viene poi disegnato su quelle righe, senza coprire la risposta.
        """
        if righe < 1 or not self._attiva or not self._console.is_terminal:
            return
        a = _ANSI
        self._regione_sospesa = True
        self._scrivi(f"{a['save']}{a['reset_region']}{a['restore']}")
        self._scrivi("\n" * righe)
        self.ripristina_regione_scroll()

    def _rilascia(self) -> None:
        if not self._console.is_terminal:
            return
        _, righe = tty_size(self._console)
        a = _ANSI
        self._scrivi(f"{a['save']}{a['reset_region']}\033[{righe};1H{a['clear_line']}{a['restore']}")
        self._righe_riservate = None

    def _disegna(self, testo: str) -> None:
        colonne, righe = tty_size(self._console)
        # Al cambio altezza: ricalcola la scroll-region subito
        if self._righe_riservate is not None and self._righe_riservate != righe:
            self._riserva_ultima_riga()
        # Troncamento ANSI-aware: evita di spezzare SGR e overflow wide glyph
        troncato = tronca_visibile(testo, max(0, colonne - 1))
        a = _ANSI
        self._scrivi(f"{a['save']}\033[{righe};1H{a['clear_line']}{troncato}{a['restore']}")

    def _installa_sigwinch(self) -> None:
        """Registra SIGWINCH per marcare un reflow; no-op su piattaforme senza segnale."""
        # Windows / ambienti senza SIGWINCH: il poll termico resta l'unico sync
        if not hasattr(signal, "SIGWINCH") or self._sigwinch_installato:
            return

        def _handler(_signum: int, _frame: object) -> None:
            """Segnala il resize senza eseguire I/O o operazioni Rich nel signal handler."""
            # Il signal handler sveglia il poll; il poll accoderà il reflow del Live.
            self._resize_richiesto.set()

        try:
            self._sigwinch_precedente = signal.signal(signal.SIGWINCH, _handler)
            self._sigwinch_installato = True
        except (ValueError, OSError):
            # signal.signal solo dal main thread: ignora se chiamato altrove
            self._sigwinch_installato = False

    def _ripristina_sigwinch(self) -> None:
        """Ripristina l'handler SIGWINCH precedente alla sessione."""
        if not self._sigwinch_installato or not hasattr(signal, "SIGWINCH"):
            return
        try:
            precedente = self._sigwinch_precedente
            # None → SIG_DFL se non c'era un handler salvato
            if precedente is None:
                signal.signal(signal.SIGWINCH, signal.SIG_DFL)
            else:
                signal.signal(signal.SIGWINCH, precedente)
        except (ValueError, OSError):
            # Best-effort: non bloccare lo shutdown della barra
            pass
        self._sigwinch_installato = False
        self._sigwinch_precedente = None

    def _applica_resize(self) -> None:
        """
        Sync dimensioni e notifica il main thread per il ridisegno UI.

        Il ridisegno dei Panel avviene sul main thread (flag), non qui:
        evita race tra thread polling e input()/Live.
        """
        self._sincronizza_console()
        # Aggiorna scroll-region subito (anche in input: il main thread farà clear)
        if self._attiva and self._console.is_terminal and not self._regione_sospesa:
            self._riserva_ultima_riga()
            self._ultima_riga = ""
            # Barra: solo se non in input; in input il main thread chiama forza=True
            if not self._input_in_corso:
                self.ridisegna()
        # La callback deve limitarsi a impostare flag thread-safe per il main thread.
        if self._on_resize is not None:
            self._on_resize()

    def _polling(self) -> None:
        """
        Loop daemon: reagisce ai resize entro ~50ms e aggiorna la termica ogni 2s.
        """
        prossimo_termico = time.monotonic()
        while not self._ferma.is_set():
            # Attesa breve: bilancia latenza resize e carico CPU
            self._ferma.wait(timeout=_INTERVALLO_POLL_RESIZE_SEC)
            # Uscita richiesta: non processare altri eventi
            if self._ferma.is_set():
                break
            # Il worker sincronizza la barra e accoda soltanto il reflow del Live.
            if self._resize_richiesto.is_set():
                self._resize_richiesto.clear()
                self._applica_resize()
            ora = time.monotonic()
            # Poll termico periodico indipendente dal resize
            if ora >= prossimo_termico:
                prossimo_termico = ora + _POLLING_TERMICA_SEC
                invalida_cache_termica()
                self.ridisegna()


class ConsoleUI:
    def __init__(self) -> None:
        """Inizializza Console, timeline, streaming e coordinamento resize thread-safe."""
        ioctl = leggi_ioctl_tty_size()
        if ioctl is not None:
            colonne, righe = ioctl
        else:
            try:
                dim = shutil.get_terminal_size(fallback=(120, 24))
            except OSError:
                dim = os.terminal_size((120, 24))
            colonne, righe = dim.columns, dim.lines
        self.console = Console(
            theme=_TEMA,
            width=max(1, colonne - MARGINE_LARGHEZZA_TTY),
            height=max(1, righe - 1),
        )
        self._timeline = TimelineTurno(self.console)
        self.barra_termica = BarraTermica(self.console)
        self._stream = RendererPanelStream(
            self.console,
            self._timeline,
            on_render=self._dopo_render_stream,
            righe_riservate=self.barra_termica.righe_chrome,
            sospendi_scroll=self.barra_termica.sospendi_regione_scroll,
            ripristina_scroll=self._ripristina_scroll_risposta,
        )
        self._stato_benvenuto: dict[str, bool] | None = None
        # Cartella documenti (closed-corpus) mostrata in header, benvenuto e /help.
        self._cartella_documenti = ""
        self._etichetta_modello = ""
        self._necessita_ridisegno_ui = threading.Event()
        self._resize_live_richiesto = threading.Event()
        self._prossimo_refresh_barra_stream = 0.0
        self._buffer_input_corrente = ""
        self._titolo_prompt_corrente = ""
        self._larghezza_input_disegnata = 0
        self._righe_composer_sticky = 0
        self._progresso_in_corso = False
        self._header_differito = False
        self.barra_termica.imposta_on_resize(self._gestisci_resize)

    def _ripristina_scroll_risposta(self) -> None:
        """Dopo la risposta alta: scroll-region, stato e header tornano al loro posto."""
        self.barra_termica.ripristina_regione_scroll()
        self._disegna_header_sticky()

    def _dopo_render_stream(self) -> None:
        """
        Aggiorna la barra termica con cadenza limitata durante lo stream.

        Evita che ogni refresh Live (fino a 20/s) risvegli letture termiche e
        ridisegni ANSI, che altrimenti creano backpressure verso Ollama.
        """
        ora = time.monotonic()
        # Il primo render del turno aggiorna subito le metriche TTFT percepite.
        if ora < self._prossimo_refresh_barra_stream:
            return
        self._prossimo_refresh_barra_stream = (
            ora + _INTERVALLO_REFRESH_BARRA_STREAM_SEC
        )
        self.barra_termica.ridisegna()

    def _sincronizza(self) -> None:
        extra = self.barra_termica.righe_chrome() if self.barra_termica.attiva else 1
        sincronizza_dimensioni(
            self.console,
            riserva_barra=True,
            righe_riservate=extra,
        )

    def _usa_chrome_sticky(self) -> bool:
        return (
            self.barra_termica.attiva
            and self.console.is_terminal
            and sys.stdout.isatty()
        )

    def _disegna_header_sticky(self) -> None:
        if self.barra_termica.regione_sospesa:
            # Un paint sulla riga 1, mentre lo schermo scorre, finisce nello storico.
            self._header_differito = True
            return
        if not self._usa_chrome_sticky() or not self.barra_termica.header_attivo:
            return
        self._header_differito = False
        stato = self._stato_benvenuto or _stato_runtime_default()
        stampa_header_sticky(
            self.console,
            markup_header_sessione(
                stato,
                self._etichetta_modello,
                self._cartella_documenti,
            ),
            is_terminal=True,
        )

    def _disegna_composer_sticky(self, editor: EditorRiga) -> None:
        testo = editor.testo()
        _colonne, righe_tty = tty_size(self.console)
        header = self.barra_termica.header_attivo
        nuova = altezza_composer_sticky(
            testo,
            self.console.width,
            righe_tty,
            header=header,
        )
        if self._righe_composer_sticky > nuova:
            pulisci_righe_assolute(
                riga_inizio_composer_sticky(righe_tty, self._righe_composer_sticky),
                self._righe_composer_sticky,
                is_terminal=True,
            )
        self._righe_composer_sticky = nuova
        self.barra_termica.imposta_righe_composer(nuova)
        stampa_composer_sticky(
            self.console,
            self._titolo_prompt_corrente,
            testo_con_selezione(testo, editor.ancora, editor.cursore),
            testo,
            editor.cursore,
            righe_tty,
            nuova,
            is_terminal=True,
        )
        self._larghezza_input_disegnata = self.console.width

    def _chiudi_composer_sticky(self) -> None:
        if not self._usa_chrome_sticky():
            self._righe_composer_sticky = 0
            return
        _colonne, righe_tty = tty_size(self.console)
        vecchie = self._righe_composer_sticky
        if vecchie > 0:
            pulisci_righe_assolute(
                riga_inizio_composer_sticky(righe_tty, vecchie),
                vecchie,
                is_terminal=True,
            )
        self._righe_composer_sticky = 0
        self.barra_termica.imposta_righe_composer(0)
        riga_prossima = riga_fine_scroll_sticky(
            righe_tty,
            header=self.barra_termica.header_attivo,
            righe_composer=vecchie,
        ) + 1
        riga_max = riga_fine_scroll_sticky(
            righe_tty,
            header=self.barra_termica.header_attivo,
            righe_composer=0,
        )
        posiziona_cursore_assoluto(
            min(riga_prossima, riga_max),
            1,
            is_terminal=True,
        )

    def _gestisci_resize(self) -> None:
        """Classifica il resize nel thread barra senza eseguire operazioni Rich Live."""
        # Un Live attivo verrà ridimensionato dal prossimo callback sul main thread.
        if self._stream.ha_live_aperto:
            self._resize_live_richiesto.set()
            return
        # Senza Live, il prompt successivo si adatta in larghezza senza wipe.
        self._necessita_ridisegno_ui.set()

    def _consuma_resize_live_richiesto(self) -> None:
        """Consuma sul main thread un resize Live accodato dalla barra termica."""
        # Senza richiesta pendente non serve interrogare o aggiornare Rich Live.
        if not self._resize_live_richiesto.is_set():
            return
        # Un callback di stream anomalo su un worker deve lasciare il flag al main.
        if threading.current_thread() is not threading.main_thread():
            return
        self._resize_live_richiesto.clear()
        self._stream.reflow_su_resize()
        self._disegna_header_sticky()

    def _corpo_benvenuto(self, stato: dict[str, bool]) -> str:
        """
        Corpo compatto del Panel di avvio: una riga di intro e i badge.

        Args:
            stato: Flag runtime (`verbose`, `verifica_completa`,
                `burn_protection`, `thinking`).

        Returns:
            Markup Rich. I comandi completi stanno in `_corpo_aiuto`.
        """
        return (
            "Assistente pronto. Rispondo solo dai documenti della cartella "
            f"[bold]{escape(self._cartella_documenti or '.')}[/] e scrivo solo file .txt.\n"
            "Scrivi una richiesta (exit/quit/esci). Comandi: /help\n"
            f"{_riga_stato_compatta(stato)}"
        )

    def _corpo_aiuto(self, stato: dict[str, bool]) -> str:
        """Manuale on-demand: stato, slash command e guida tastiera."""
        stato_righe = "\n".join([
            f"  {_markup_badge('Verbose', stato['verbose'])}",
            f"  {_markup_badge('Check', stato['verifica_completa'])}",
            f"  {_markup_badge('Burn protection', stato['burn_protection'])}",
            f"  {_markup_badge('Thinking', stato.get('thinking', False))}",
        ])
        sintassi_toggle = escape("[on|off|toggle]")
        sintassi_model = escape("[fast|default]")
        comandi = "\n".join([
            f"  Flag:     /verbose | /check | /burn_protection | /thinking  {sintassi_toggle}",
            f"  Runtime:  /model {sintassi_model}",
            "  Sessione: /clear — azzera storico conversazione",
            "            /errori — ultimi errori di processo",
            "            /help — questa guida",
            *RIGHE_GUIDA_TASTIERA,
        ])
        cartella = escape(self._cartella_documenti or ".")
        return (
            f"Cartella documenti: {cartella}\n\n"
            f"Stato:\n{stato_righe}\n\nComandi:\n{comandi}"
        )

    def _ristampa_chrome(
        self,
        testo_input: str | None = None,
        *,
        ancora: int | None = None,
        cursore: int = 0,
    ) -> None:
        """Reflow di header e composer dopo un resize, senza cancellare lo storico."""
        if not self.console.is_terminal or not sys.stdout.isatty():
            return
        self._progresso_in_corso = False
        if self._usa_chrome_sticky():
            self._sincronizza()
            self._disegna_header_sticky()
            if testo_input is not None:
                editor = EditorRiga()
                editor.inserisci(testo_input)
                editor.cursore = max(0, min(cursore, len(editor.testo())))
                editor.ancora = ancora
                self._disegna_composer_sticky(editor)
            self.barra_termica.ridisegna(forza=True)
            return
        if testo_input is None:
            self._sincronizza()
            self.barra_termica.ridisegna(forza=True)
            return
        larghezza_vecchia = self._larghezza_input_disegnata or self.console.width
        cancella_righe_sopra(
            cuu_clear_composer(testo_input, larghezza_vecchia),
            is_terminal=True,
        )
        self._sincronizza()
        stampa_composer(
            self.console,
            self._titolo_prompt_corrente,
            testo_con_selezione(testo_input, ancora, cursore),
            sincronizza=False,
        )
        self._larghezza_input_disegnata = self.console.width
        self.barra_termica.ridisegna(forza=True)

    def stampa_aiuto(self) -> None:
        """Stampa il panel di guida (stato, comandi, tastiera)."""
        self._finalizza_progresso()
        self._sincronizza()
        stato = self._stato_benvenuto or _stato_runtime_default()
        stampa_panel(
            self.console,
            self._corpo_aiuto(stato),
            "[agent.label]Aiuto[/]",
            padding=(0, 1),
            sincronizza=False,
        )
        self.barra_termica.ridisegna()

    def prepara_nuovo_turno_stream(self) -> None:
        """Azzera timeline, renderer, resize pendenti e metriche del nuovo turno."""
        self._timeline.reset_turno()
        self._stream.reset_turno()
        self._resize_live_richiesto.clear()
        self._prossimo_refresh_barra_stream = 0.0
        self.barra_termica.imposta_sorgente_metriche(None)
        self.barra_termica.imposta_metriche_turno(None)

    def inizia_risposta_diretta(self) -> None:
        """
        Prepara lo stream per una risposta senza protocollo ReAct.

        Returns:
            None. Effetto collaterale: lo stream successivo va nel corpo finale.
        """
        self._stream.inizia_risposta_diretta()

    def imposta_verbose(self, attivo: bool) -> None:
        self._timeline.imposta_verbose(attivo)
        self.barra_termica.imposta_mostra_metriche(attivo)
        if self._stato_benvenuto is not None:
            self._stato_benvenuto["verbose"] = attivo

    def imposta_separatore_file(self, separatore: str) -> None:
        self._timeline.imposta_separatore_file(separatore)

    def registra_azione_tool(self, nome: str, action_input: str) -> None:
        self._finalizza_progresso()
        self._timeline.registra_azione_tool(nome, action_input)

    def inizia_esecuzione_tool(self, nome: str) -> None:
        self._finalizza_progresso()
        self._timeline.inizia_esecuzione_tool(nome)

    def termina_esecuzione_tool(self) -> None:
        self._timeline.termina_esecuzione_tool()

    def emetti_risultato_tool(self, osservazione: str) -> None:
        self._timeline.emetti_risultato_tool(osservazione)

    def emetti_errore_tool(self, messaggio: str) -> None:
        self._timeline.emetti_errore_tool(messaggio)

    def scrivi_token(self, token: str) -> None:
        """
        Elabora un chunk e applica sul main thread l'eventuale resize Live pendente.

        Args:
            token: Frammento LLM da inoltrare senza modifiche al parser ReAct.
        """
        self._finalizza_progresso()
        self._stream.scrivi_token(token)
        self._consuma_resize_live_richiesto()

    def scrivi_pensiero(self, token: str) -> None:
        """Scrive un frammento di CoT nativo in timeline, senza aprire il panel."""
        # La riga di progresso (es. «Comprendo la richiesta») è live e senza newline:
        # senza chiusura il primo token resta attaccato a quella frase.
        self._finalizza_progresso()
        self._timeline.scrivi_pensiero(token)

    def chiudi_pensiero(self) -> None:
        """Chiude la riga Pensiero prima della risposta o del ciclo LLM successivo."""
        self._timeline.chiudi_pensiero()

    def reset_chiamata_llm(self) -> None:
        self._stream.reset_chiamata_llm()

    def finalizza(self, risultato: str) -> None:
        """
        Applica l'ultimo resize pendente prima di chiudere e finalizzare il Live.

        Args:
            risultato: Risposta LLM completa usata dal renderer come fallback.
        """
        self._consuma_resize_live_richiesto()
        self._stream.finalizza(risultato)

    def imposta_metriche_turno(self, metriche: StatisticheTurnoLLM | None) -> None:
        self.barra_termica.imposta_metriche_turno(metriche)

    def aggiorna_metriche_turno(self, metriche: StatisticheTurnoLLM | None) -> None:
        self.barra_termica.aggiorna_metriche_turno(metriche)

    def imposta_sorgente_metriche(
        self,
        sorgente: Callable[[], StatisticheTurnoLLM | None] | None,
    ) -> None:
        self.barra_termica.imposta_sorgente_metriche(sorgente)

    @property
    def ha_stream_live(self) -> bool:
        """True se il Panel Final Answer è aperto e già responsabile del refresh barra."""
        return self._stream.ha_live_aperto

    @property
    def conteggio_refresh_stream(self) -> int:
        """Numero di refresh Live del panel risposta (diagnostica rendering)."""
        return self._stream.conteggio_refresh

    def pulisci_terminale(self) -> None:
        if not self.console.is_terminal or not sys.stdout.isatty():
            return
        self._progresso_in_corso = False
        self.console.clear()
        if self.barra_termica.attiva:
            self.barra_termica.dopo_clear()
            self._disegna_header_sticky()

    def stampa_splash(
        self,
        pausa: Callable[[float], None] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        """
        Riproduce l'HUD di avvio (animazione + saluto) mentre il modello si preriscalda.

        Args:
            pausa: Attesa tra i frame. Default `time.sleep`.
            monotonic: Orologio iniettabile. Default `time.monotonic`.

        Returns:
            None. Su console non-terminale, o se la splash fallisce, non blocca la sessione.
        """
        if not self.console.is_terminal:
            return
        try:
            esegui_splash(self.console, pausa=pausa, monotonic=monotonic)
        except Exception:
            # La splash non deve impedire il Panel tecnico né il wizard.
            return

    def stampa_benvenuto(
        self, verbose: bool,
        verifica_completa: bool = False, burn_protection: bool = True,
        thinking: bool = False,
        etichetta_modello: str = "",
        cartella: str = "",
    ) -> None:
        self._stato_benvenuto = {
            "verbose": verbose,
            "verifica_completa": verifica_completa,
            "burn_protection": burn_protection,
            "thinking": thinking,
        }
        if cartella:
            self._cartella_documenti = cartella
        if etichetta_modello:
            self._etichetta_modello = etichetta_modello
        self.pulisci_terminale()
        self._sincronizza()
        self.imposta_verbose(verbose)
        self._disegna_header_sticky()
        if self._usa_chrome_sticky():
            posiziona_inizio_transcript(
                header=self.barra_termica.header_attivo,
                is_terminal=True,
            )
        self.console.print()
        stampa_panel(
            self.console,
            self._corpo_benvenuto(self._stato_benvenuto),
            "[agent.label]Assistente[/]",
            padding=(0, 1),
            sincronizza=False,
        )
        self.console.print()

    def imposta_etichetta_modello(self, etichetta: str) -> None:
        """Aggiorna il modello nell'header sticky senza wipe."""
        self._etichetta_modello = etichetta
        self._disegna_header_sticky()

    def leggi_richiesta_utente(self, etichetta_modello: str) -> str:
        self._etichetta_modello = etichetta_modello
        self.barra_termica.imposta_input_in_corso(True)
        self._sincronizza()
        if self._usa_chrome_sticky():
            _, righe_tty = tty_size(self.console)
            spazio = altezza_composer_sticky(
                "",
                self.console.width,
                righe_tty,
                header=self.barra_termica.header_attivo,
            )
            self.barra_termica.fai_spazio_in_basso(spazio)
            self._disegna_header_sticky()
        if self._necessita_ridisegno_ui.is_set():
            self._necessita_ridisegno_ui.clear()
            self._disegna_header_sticky()
            self.barra_termica.ridisegna(forza=True)
        titolo_markup = f"[user.label]Tu[/] [dim]· {etichetta_modello}[/]"
        self._titolo_prompt_corrente = titolo_markup
        sticky = self._usa_chrome_sticky()
        if not sticky:
            self.console.print()
            stampa_composer(
                self.console,
                titolo_markup,
                Text(""),
                sincronizza=False,
            )
        self._larghezza_input_disegnata = self.console.width
        cursore_nel_box = self.console.is_terminal and termios_disponibile()
        try:
            if cursore_nel_box:
                prima_riga = leggi_riga_con_reflow(
                    self.console,
                    titolo_markup,
                    resize_event=self._necessita_ridisegno_ui,
                    on_resize=lambda testo, ancora, cursore: self._ristampa_chrome(
                        testo_input=testo,
                        ancora=ancora,
                        cursore=cursore,
                    ),
                    set_buffer=lambda t: setattr(self, "_buffer_input_corrente", t),
                    on_disegna=(
                        self._disegna_composer_sticky if sticky else None
                    ),
                )
            else:
                prima_riga = input("› ")
            testo = raccogli_input_multilinea(prima_riga, sys.stdin)
        except (EOFError, KeyboardInterrupt):
            if cursore_nel_box:
                if sticky:
                    self._chiudi_composer_sticky()
                else:
                    visive = righe_visive_contenuto(
                        riga_composer(self._buffer_input_corrente),
                        self.console.width,
                    )
                    sys.stdout.write(f"\033[{max(1, visive)}B\r")
                    sys.stdout.flush()
            self.barra_termica.imposta_input_in_corso(False)
            self.barra_termica.ridisegna(forza=True)
            raise

        if cursore_nel_box:
            if sticky:
                self._chiudi_composer_sticky()
            else:
                cancella_righe_sopra(
                    cuu_clear_composer(testo, self.console.width),
                    is_terminal=self.console.is_terminal,
                )
        self._sincronizza()
        stampa_messaggio_utente(
            self.console,
            titolo_markup,
            testo,
            sincronizza=False,
        )
        self._larghezza_input_disegnata = self.console.width
        self.barra_termica.imposta_input_in_corso(False)
        self._disegna_header_sticky()
        self.barra_termica.ridisegna(forza=True)
        return testo

    def chiudi_progresso(self) -> None:
        """Chiude la riga live (es. «Comprendo la richiesta») prima dell'output successivo."""
        self._finalizza_progresso()

    def _finalizza_progresso(self) -> None:
        """Chiude la riga live del pull: l'ultimo stato resta visibile sotto."""
        if not self._progresso_in_corso:
            return
        self.console.file.write("\n")
        self.console.file.flush()
        self._progresso_in_corso = False

    def _print(self, *args, **kwargs) -> None:
        self._finalizza_progresso()
        self._sincronizza()
        self.console.print(*args, **kwargs)
        self.barra_termica.ridisegna()

    def stampa_toggle(self, nome: str, attivo: bool) -> None:
        if nome == "Verbose":
            self.imposta_verbose(attivo)
        elif self._stato_benvenuto is not None:
            chiave = _CHIAVE_STATO_DA_ETICHETTA.get(nome)
            if chiave is not None:
                self._stato_benvenuto[chiave] = attivo
        stile = "badge.on" if attivo else "badge.off"
        self._print(f"[{stile}]{nome}: {'attivo' if attivo else 'disattivo'}[/]")
        self._disegna_header_sticky()

    def stampa_errore(self, messaggio: str) -> None:
        self._finalizza_progresso()
        self._sincronizza()
        stampa_panel(
            self.console,
            Text(messaggio, style="error"),
            "Errore",
            border_style="red",
            padding=(0, 2),
            sincronizza=False,
        )
        self.barra_termica.ridisegna()

    def stampa_fallback(self, path_scritti: list[str]) -> None:
        self._finalizza_progresso()
        self._sincronizza()
        righe = "\n".join(f"- `{p}`" for p in path_scritti)
        stampa_panel(
            self.console,
            Markdown(
                "File scritti automaticamente perché l'agente non ha usato i tool:\n\n"
                f"{righe}"
            ),
            "[warning]Fallback[/]",
            border_style="yellow",
            padding=(1, 2),
            sincronizza=False,
        )
        self.barra_termica.ridisegna()

    def stampa_messaggio_sistema(self, messaggio: str) -> None:
        # Escape obbligatorio: errori Ollama/httpx possono contenere [Errno N] o [/]
        # che Rich interpreta come markup e fanno crashare il loop comandi.
        self._print(f"[system]{escape(messaggio)}[/]")

    def chiedi_conferma(self, domanda: str) -> bool:
        """
        Chiede un consenso esplicito sì/no.

        Invio vuoto, EOF e qualsiasi valore diverso da s/si/sì/y/yes valgono no.
        """
        risposta = self._leggi_riga_wizard(f"{domanda} [s/N]").casefold()
        return risposta in {"s", "si", "sì", "y", "yes"}

    def chiedi_conferma_download(self, domanda: str) -> bool:
        """
        Consenso rapido al pull: invio vuoto e Y valgono sì.

        n/no annullano. Qualsiasi altro testo non vuoto vale no.
        """
        risposta = self._leggi_riga_wizard(f"{domanda} [Y/n]").casefold()
        if risposta in {"n", "no"}:
            return False
        return risposta in {"", "y", "yes", "s", "si", "sì"}

    def chiedi_testo(self, domanda: str) -> str:
        """Legge una riga libera (selezione modelli, indici o alias)."""
        return self._leggi_riga_wizard(domanda)

    def aggiorna_progresso(self, messaggio: str) -> None:
        """
        Aggiorna in place la riga di avanzamento (install o pull).

        Su TTY la percentuale resta sulla stessa riga (EL + CR). Fuori da TTY
        ogni stato va a capo, per log e test catturabili.
        """
        if not self.console.is_terminal:
            self.stampa_messaggio_sistema(messaggio)
            return
        self._sincronizza()
        larghezza = max(1, self.console.width - 1)
        visibile = tronca_visibile(messaggio, larghezza)
        destinazione = self.console.file
        destinazione.write(f"{_ANSI['clear_line']}\r")
        destinazione.flush()
        self.console.print(f"[system]{escape(visibile)}[/]", end="")
        self._progresso_in_corso = True
        self.barra_termica.ridisegna()

    def _leggi_riga_wizard(self, domanda: str) -> str:
        """Prompt condiviso del wizard; EOF = stringa vuota, KeyboardInterrupt sale."""
        self.stampa_messaggio_sistema(domanda)
        try:
            return input("› ").strip()
        except EOFError:
            return ""

    def stampa_avviso_termico(self, messaggio: str) -> None:
        self._print(f"[warning]Termico: {escape(messaggio)}[/]")

    def stampa_separatore(self) -> None:
        self._print(Rule(style="dim"))
