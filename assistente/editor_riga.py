"""Editor di riga: stato puro, decoder tasti, guida UI. Nessun I/O TTY."""

from __future__ import annotations

from dataclasses import dataclass

from rich.text import Text

STILE_SELEZIONE = "bold black on cyan"

RIGHE_GUIDA_TASTIERA = (
    "  Cursore:  frecce ← →  ·  Option+frecce = parole  ·  Ctrl+A / Ctrl+E = inizio/fine",
    "  Selezione: Ctrl+W o Ctrl+G, poi frecce (o Option+frecce); Esc esce",
    "            Shift+frecce se il terminale le inoltra  ·  Option+Shift = parole",
    "  Appunti:  Ctrl+C copia solo la selezione  ·  Ctrl+X taglia  ·  Ctrl+V incolla",
    "            Ctrl+C senza selezione interrompe sempre il prompt",
)

_AZIONI_FINALE = {
    "D": "left",
    "C": "right",
    "A": "home",
    "B": "end",
    "H": "home",
    "F": "end",
    "d": "left",
    "c": "right",
}
_AZIONI_SS3 = {"D": "left", "C": "right", "A": "home", "B": "end"}
_AZIONI_TILDE = {
    "3": "delete",
    "1": "home",
    "7": "home",
    "4": "end",
    "8": "end",
}
_AZIONI_KITTY = {
    "57350": "left",
    "57351": "right",
    "57352": "home",
    "57353": "end",
}
_CTRL = {
    "\x01": "home",
    "\x03": "copy",
    "\x04": "eof",
    "\x05": "end",
    "\x07": "toggle_selezione",
    "\x08": "backspace",
    "\x0d": "enter",
    "\x0a": "enter",
    "\x16": "paste",
    "\x17": "toggle_selezione",
    "\x18": "cut",
    "\x7f": "backspace",
}
_TERMINATORE_PASTE = b"\x1b[201~"
_INIZIO_PASTE = b"[200~"


@dataclass(frozen=True)
class EventoTasto:
    """Tasto logico: movimento, editing, oppure `testo` da inserire."""

    azione: str
    shift: bool = False
    alt: bool = False
    testo: str = ""


def _e_glifo_parola(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def indice_parola_sinistra(buffer: list[str], cursore: int) -> int:
    """Inizio della parola precedente o corrente (Option+←)."""
    i = max(0, min(cursore, len(buffer)))
    if i == 0:
        return 0
    i -= 1
    while i > 0 and not _e_glifo_parola(buffer[i]):
        i -= 1
    while i > 0 and _e_glifo_parola(buffer[i - 1]):
        i -= 1
    return i


def indice_parola_destra(buffer: list[str], cursore: int) -> int:
    """Fine della parola corrente / inizio della successiva (Option+→)."""
    i = max(0, min(cursore, len(buffer)))
    n = len(buffer)
    while i < n and _e_glifo_parola(buffer[i]):
        i += 1
    while i < n and not _e_glifo_parola(buffer[i]):
        i += 1
    return i


def normalizza_incolla(testo: str) -> str:
    """Una sola riga: newline → spazio, scarta i controlli."""
    pezzi: list[str] = []
    for ch in testo.replace("\r\n", "\n").replace("\r", "\n"):
        if ch == "\n":
            pezzi.append(" ")
        elif ch.isprintable():
            pezzi.append(ch)
    return "".join(pezzi)


def _modificatori_csi(seq: str) -> tuple[bool, bool]:
    if not seq.startswith("[") or ";" not in seq:
        return False, False
    parti = seq[1:-1].split(";")
    if len(parti) < 2:
        return False, False
    raw = parti[1].split(":")[0]
    if not raw.isdigit():
        return False, False
    bits = max(0, int(raw) - 1)
    return bool(bits & 1), bool(bits & 2) or bool(bits & 4)


def interpretazione_escape(seq: str) -> EventoTasto | None:
    """Traduce CSI/SS3/Meta-b/f/CSI u in evento. `[200~` lo gestisce il decoder."""
    if not seq:
        return None
    if seq == "b":
        return EventoTasto("left", alt=True)
    if seq == "f":
        return EventoTasto("right", alt=True)
    if seq == "[200~":
        return EventoTasto("paste_start")
    shift, alt = _modificatori_csi(seq)
    finale = seq[-1]
    azione = ""
    if seq.startswith("O") and len(seq) == 2:
        azione = _AZIONI_SS3.get(seq[1], "")
    elif finale == "u" and seq.startswith("["):
        azione = _AZIONI_KITTY.get(seq[1:-1].split(";")[0], "")
    elif finale == "~":
        azione = _AZIONI_TILDE.get(seq[1:-1].split(";")[0], "")
    elif seq.startswith("["):
        azione = _AZIONI_FINALE.get(finale, "")
        if finale in ("d", "c"):
            shift = True
    if not azione:
        return None
    return EventoTasto(azione, shift=shift, alt=alt)


def testo_con_selezione(testo: str, ancora: int | None, cursore: int) -> Text:
    """Corpo Panel Rich con intervallo ancora→cursore evidenziato."""
    if not testo:
        return Text(" ")
    if ancora is None or ancora == cursore:
        return Text(testo)
    inizio, fine = sorted((ancora, cursore))
    inizio = max(0, min(inizio, len(testo)))
    fine = max(0, min(fine, len(testo)))
    corpo = Text()
    corpo.append(testo[:inizio])
    corpo.append(testo[inizio:fine], style=STILE_SELEZIONE)
    corpo.append(testo[fine:])
    return corpo


class EditorRiga:
    """Buffer, cursore e selezione: unica fonte di verità della riga di input."""

    def __init__(self) -> None:
        self.glifi: list[str] = []
        self.cursore = 0
        self.ancora: int | None = None
        self.modalita_selezione = False

    def testo(self) -> str:
        return "".join(self.glifi)

    def intervallo_selezione(self) -> tuple[int, int] | None:
        if self.ancora is None or self.ancora == self.cursore:
            return None
        inizio, fine = sorted((self.ancora, self.cursore))
        return inizio, fine

    def testo_clipboard(self) -> str:
        intervallo = self.intervallo_selezione()
        if intervallo is None:
            return self.testo()
        inizio, fine = intervallo
        return "".join(self.glifi[inizio:fine])

    def _azzera_selezione(self) -> None:
        self.ancora = None
        self.modalita_selezione = False

    def elimina_selezione(self) -> bool:
        intervallo = self.intervallo_selezione()
        if intervallo is None:
            return False
        inizio, fine = intervallo
        del self.glifi[inizio:fine]
        self.cursore = inizio
        self._azzera_selezione()
        return True

    def inserisci(self, pezzo: str) -> None:
        if not pezzo:
            return
        self.elimina_selezione()
        glifi = list(pezzo)
        self.glifi[self.cursore:self.cursore] = glifi
        self.cursore += len(glifi)
        self._azzera_selezione()

    def cancella_sinistra(self) -> None:
        if self.elimina_selezione():
            return
        if self.cursore <= 0:
            return
        self.cursore -= 1
        del self.glifi[self.cursore]

    def cancella_destra(self) -> None:
        if self.elimina_selezione():
            return
        if self.cursore >= len(self.glifi):
            return
        del self.glifi[self.cursore]

    def taglia(self) -> str:
        pezzo = self.testo_clipboard()
        if not pezzo:
            return ""
        if self.intervallo_selezione() is not None:
            self.elimina_selezione()
            return pezzo
        self.glifi.clear()
        self.cursore = 0
        self._azzera_selezione()
        return pezzo

    def toggle_selezione(self) -> None:
        if self.modalita_selezione:
            self._azzera_selezione()
            return
        self.modalita_selezione = True
        self.ancora = self.cursore

    def chiudi_selezione(self) -> bool:
        if self.modalita_selezione or self.ancora is not None:
            self._azzera_selezione()
            return True
        return False

    def _destinazione(self, azione: str, alt: bool) -> int:
        if azione == "home":
            return 0
        if azione == "end":
            return len(self.glifi)
        if azione == "left":
            if alt:
                return indice_parola_sinistra(self.glifi, self.cursore)
            return max(0, self.cursore - 1)
        if azione == "right":
            if alt:
                return indice_parola_destra(self.glifi, self.cursore)
            return min(len(self.glifi), self.cursore + 1)
        return self.cursore

    def muovi(self, azione: str, *, shift: bool = False, alt: bool = False) -> None:
        estendi = shift or self.modalita_selezione
        partenza = self.cursore
        if not estendi and self.ancora is not None and self.ancora != self.cursore:
            if azione == "left" and not alt:
                self.cursore = min(self.ancora, self.cursore)
            elif azione == "right" and not alt:
                self.cursore = max(self.ancora, self.cursore)
            else:
                self.cursore = self._destinazione(azione, alt)
            self.ancora = None
            return
        destinazione = self._destinazione(azione, alt)
        if estendi:
            if self.ancora is None:
                self.ancora = partenza
        else:
            self.ancora = None
        self.cursore = destinazione

    def applica(self, evento: EventoTasto) -> str | None:
        """
        Applica un evento. Ritorna un segnale di loop, o l'azione clipboard.

        Returns:
            enter / interrupt / eof / copy / cut / paste, altrimenti None.
        """
        azione = evento.azione
        if azione == "enter":
            return "enter"
        if azione == "interrupt":
            return "interrupt"
        if azione == "eof":
            if not self.glifi:
                return "eof"
            self.cancella_destra()
            return None
        if azione == "insert":
            self.inserisci(evento.testo)
            return None
        if azione in ("left", "right", "home", "end"):
            self.muovi(azione, shift=evento.shift, alt=evento.alt)
            return None
        if azione == "backspace":
            self.cancella_sinistra()
            return None
        if azione == "delete":
            self.cancella_destra()
            return None
        if azione == "toggle_selezione":
            self.toggle_selezione()
            return None
        if azione == "escape":
            self.chiudi_selezione()
            return None
        if azione in ("copy", "cut", "paste"):
            return azione
        return None


class DecoderTasti:
    """Accumula burst di byte e produce `EventoTasto` completi."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._paste: bytearray | None = None

    def attesa(self) -> bool:
        """True se ESC/CSI o paste sono incompleti."""
        if self._paste is not None:
            return True
        return bool(self._buf) and self._buf[0] == 0x1B

    @property
    def in_paste(self) -> bool:
        return self._paste is not None

    def feed(self, data: bytes) -> list[EventoTasto]:
        if data:
            self._buf.extend(data)
        return self._estrai_completi()

    def chiudi_attesa(self) -> list[EventoTasto]:
        """Timeout: chiude paste troncato e ESC nudo."""
        eventi: list[EventoTasto] = []
        if self._paste is not None:
            testo = normalizza_incolla(
                bytes(self._paste).decode("utf-8", errors="replace"),
            )
            self._paste = None
            if testo:
                eventi.append(EventoTasto("insert", testo=testo))
        if self._buf[:1] == b"\x1b":
            if len(self._buf) == 1:
                del self._buf[0]
                eventi.append(EventoTasto("escape"))
            else:
                self._buf.clear()
        eventi.extend(self._estrai_completi())
        return eventi

    def _estrai_completi(self) -> list[EventoTasto]:
        eventi: list[EventoTasto] = []
        while True:
            evento = self._estrai_uno()
            if evento is None:
                break
            eventi.append(evento)
        return eventi

    def _estrai_uno(self) -> EventoTasto | None:
        if self._paste is not None:
            return self._estrai_paste()
        if not self._buf:
            return None
        if self._buf[0] == 0x1B:
            return self._estrai_escape()
        ch = self._estrai_utf8()
        if ch is None:
            return None
        if ch in _CTRL:
            return EventoTasto(_CTRL[ch])
        if ch.isprintable():
            return EventoTasto("insert", testo=ch)
        return self._estrai_uno()

    def _estrai_utf8(self) -> str | None:
        if not self._buf:
            return None
        lead = self._buf[0]
        if lead < 0x80:
            return chr(self._buf.pop(0))
        if lead < 0xE0:
            attesi = 2
        elif lead < 0xF0:
            attesi = 3
        else:
            attesi = 4
        if len(self._buf) < attesi:
            return None
        blocco = bytes(self._buf[:attesi])
        del self._buf[:attesi]
        return blocco.decode("utf-8", errors="replace")

    def _estrai_escape(self) -> EventoTasto | None:
        if len(self._buf) < 2:
            return None
        secondo = self._buf[1]
        if secondo == ord("O"):
            if len(self._buf) < 3:
                return None
            seq = "O" + chr(self._buf[2])
            del self._buf[:3]
            return interpretazione_escape(seq)
        if secondo != ord("["):
            seq = chr(secondo)
            del self._buf[:2]
            return interpretazione_escape(seq)
        indice = 2
        while indice < len(self._buf):
            finale = self._buf[indice]
            if 0x40 <= finale <= 0x7E:
                seq = self._buf[1:indice + 1].decode("ascii", errors="replace")
                del self._buf[:indice + 1]
                if bytes(seq, "ascii") == _INIZIO_PASTE or seq == "[200~":
                    self._paste = bytearray()
                    return self._estrai_uno()
                return interpretazione_escape(seq)
            indice += 1
        return None

    def _estrai_paste(self) -> EventoTasto | None:
        assert self._paste is not None
        fine = self._buf.find(_TERMINATORE_PASTE)
        if fine >= 0:
            self._paste.extend(self._buf[:fine])
            del self._buf[:fine + len(_TERMINATORE_PASTE)]
            testo = normalizza_incolla(
                bytes(self._paste).decode("utf-8", errors="replace"),
            )
            self._paste = None
            if testo:
                return EventoTasto("insert", testo=testo)
            return self._estrai_uno()
        prefisso = self._buf.find(b"\x1b")
        if prefisso == -1:
            self._paste.extend(self._buf)
            self._buf.clear()
            return None
        self._paste.extend(self._buf[:prefisso])
        del self._buf[:prefisso]
        if _TERMINATORE_PASTE.startswith(bytes(self._buf)):
            return None
        self._paste.extend(self._buf)
        self._buf.clear()
        return None
