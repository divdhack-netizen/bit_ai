"""Layout TTY: sync dimensioni, troncamento visibile e Panel anti-wrap."""

from __future__ import annotations

import os
import re
import shutil
import sys

from rich._wrap import divide_line
from rich.cells import cell_len
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

# Margine anti-wrap: Terminal.app va a capo sulla colonna finale
MARGINE_LARGHEZZA_TTY = 1
# Angoli + spazi titolo Panel
MARGINE_TITOLO_PANEL = 4
# Padding Panel input utente: (verticale, orizzontale)
PADDING_PANEL_INPUT = (0, 1)
# Prefisso del composer senza box
PREFISSO_COMPOSER = "› "
# Cap di lettura markdown/stream: non allungare le righe su terminali ultrawide
LARGHEZZA_LETTURA_MARKDOWN = 88
# Composer sticky: titolo + corpo, mai più del viewport utile
MAX_RIGHE_COMPOSER_STICKY = 6
# Header sticky: una riga fuori dalla scroll-region
RIGHE_HEADER_STICKY = 1
# Stile bordo condiviso (theme key in ConsoleUI)
STILE_BORDO_PANEL = "panel.border"
# Sequenze SGR usate dalla barra termica
_PATTERN_ANSI_SGR = re.compile(r"\033\[[0-9;]*m")


def leggi_ioctl_tty_size() -> tuple[int, int] | None:
    """
    Legge colonne/righe tramite ioctl sul file descriptor del TTY.

    Returns:
        (colonne, righe) se disponibili, altrimenti None (pipe/redirect).
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            fileno = stream.fileno()
        except (AttributeError, OSError, ValueError):
            continue
        try:
            size = os.get_terminal_size(fileno)
        except OSError:
            continue
        if size.columns > 0 and size.lines > 0:
            return size.columns, size.lines
    return None


def _tty_size_senza_env() -> tuple[int, int] | None:
    """Fallback shutil ignorando COLUMNS/LINES (spesso stale dopo uno shrink)."""
    salvati: dict[str, str] = {}
    for chiave in ("COLUMNS", "LINES"):
        if chiave in os.environ:
            salvati[chiave] = os.environ.pop(chiave)
    try:
        try:
            size = shutil.get_terminal_size()
        except OSError:
            return None
        if size.columns > 0 and size.lines > 0:
            return size.columns, size.lines
        return None
    finally:
        os.environ.update(salvati)


def tty_size(console: Console) -> tuple[int, int]:
    """
    Legge colonne/righe TTY privilegiando ioctl (segue enlarge e shrink).

    Non usa min(ioctl, COLUMNS): un COLUMNS piccolo dopo lo shrink
    bloccherebbe l'enlarge.
    """
    ioctl = leggi_ioctl_tty_size()
    if ioctl is not None:
        return ioctl
    senza_env = _tty_size_senza_env()
    if senza_env is not None:
        return senza_env
    return max(1, console.width), max(1, console.height)


def sincronizza_dimensioni(
    console: Console,
    *,
    riserva_barra: bool = False,
    righe_riservate: int = 0,
) -> tuple[int, int]:
    """
    Allinea size della Console al TTY (width = colonne - 1 anti-wrap).

    Args:
        riserva_barra: Se True e `righe_riservate` è 0, sottrae 1 riga (stato).
        righe_riservate: Header + composer + stato già calcolati dal chrome.

    Returns:
        Dimensioni TTY (colonne, righe) pre-margine.
    """
    colonne, righe = tty_size(console)
    larghezza = max(1, colonne - MARGINE_LARGHEZZA_TTY)
    extra = righe_riservate
    if extra <= 0 and riserva_barra:
        extra = 1
    if extra > 0 and righe > extra:
        altezza = max(1, righe - extra)
    else:
        altezza = max(1, righe)
    console.size = (larghezza, altezza)
    os.environ["COLUMNS"] = str(colonne)
    os.environ["LINES"] = str(righe)
    return colonne, righe


def titolo_panel_adattivo(markup: str, max_celle: int) -> str:
    """Tronca un title Panel Rich alla larghezza utile (ellipsis se serve)."""
    if max_celle <= 0:
        return ""
    testo = Text.from_markup(markup)
    if cell_len(testo.plain) <= max_celle:
        return markup
    troncato = testo.copy()
    troncato.truncate(max_celle, overflow="ellipsis")
    return troncato.markup


def rimuovi_ansi_sgr(testo: str) -> str:
    """Rimuove sequenze SGR da una stringa ANSI."""
    return _PATTERN_ANSI_SGR.sub("", testo)


def larghezza_visibile(testo: str) -> int:
    """Larghezza in celle ignorando gli SGR ANSI."""
    return cell_len(rimuovi_ansi_sgr(testo))


def tronca_visibile(testo: str, max_colonne: int) -> str:
    """Tronca alla larghezza visibile senza spezzare sequenze ANSI."""
    if max_colonne <= 0:
        return ""
    if larghezza_visibile(testo) <= max_colonne:
        return testo
    risultato: list[str] = []
    celle_usate = 0
    indice = 0
    lunghezza = len(testo)
    while indice < lunghezza:
        if testo[indice] == "\033":
            match = _PATTERN_ANSI_SGR.match(testo, indice)
            if match:
                risultato.append(match.group(0))
                indice = match.end()
                continue
        carattere = testo[indice]
        costo = cell_len(carattere)
        if celle_usate + costo > max_colonne:
            break
        risultato.append(carattere)
        celle_usate += costo
        indice += 1
    return "".join(risultato) + "\033[0m"


def colonna_cursore_panel(padding_orizzontale: int) -> int:
    """Colonna 1-based del primo carattere di contenuto in un Panel Rich."""
    return 2 + padding_orizzontale


def larghezza_interna_panel(
    larghezza_console: int,
    padding_orizzontale: int = PADDING_PANEL_INPUT[1],
) -> int:
    """Celle utili nel Panel (senza bordi né padding); minimo 1."""
    # 2 celle: bordi verticali sinistro/destro.
    return max(1, larghezza_console - 2 - (2 * padding_orizzontale))


def segmenti_wrap_contenuto_panel(
    testo: str,
    larghezza_utile: int,
) -> list[str]:
    """
    Spezza `testo` come il corpo di un Panel Rich (word wrap + fold).

    La geometria a celle intere (`len % larghezza`) non basta: Rich va a capo
    sulla parola, quindi l'ultima riga è più corta del resto della divisione.
    """
    if not testo:
        return [""]
    utile = max(1, larghezza_utile)
    tagli = divide_line(testo, utile, fold=True)
    if not tagli:
        return [testo]
    segmenti: list[str] = []
    inizio = 0
    for taglio in tagli:
        segmenti.append(testo[inizio:taglio])
        inizio = taglio
    segmenti.append(testo[inizio:])
    if segmenti and segmenti[-1] == "" and len(segmenti) > 1:
        segmenti.pop()
    return segmenti or [""]


def geometria_cursore_contenuto(
    testo: str,
    larghezza_utile: int,
) -> tuple[int, int]:
    """
    Offset colonna (0-based nel contenuto) e righe visive a fine `testo`.

    Returns:
        `(colonna_extra, righe_visive)`. Su riga piena `colonna_extra == larghezza_utile`.
    """
    if not testo:
        return 0, 1
    utile = max(1, larghezza_utile)
    segmenti = segmenti_wrap_contenuto_panel(testo, utile)
    ultima = segmenti[-1]
    colonna_extra = cell_len(ultima)
    if colonna_extra == 0:
        return utile, max(1, len(segmenti) - 1)
    return colonna_extra, len(segmenti)


def righe_visive_contenuto(testo: str, larghezza_utile: int) -> int:
    """Righe visive di `testo` nella larghezza utile del Panel (minimo 1)."""
    return geometria_cursore_contenuto(testo, larghezza_utile)[1]


def righe_salita_panel(righe_contenuto: int, padding_verticale: int) -> int:
    """CUU dal cursore sotto il Panel fino alla prima riga contenuto."""
    return 1 + padding_verticale + max(1, righe_contenuto) + padding_verticale


def righe_da_contenuto_a_bordo_superiore(
    padding_verticale: int,
    *,
    riga_contenuto: int = 1,
) -> int:
    """
    CUU dal cursore DENTRO il contenuto fino al bordo superiore del Panel.

    In editing cbreak usare questa geometria (non `righe_salita_panel`).
    """
    return 1 + padding_verticale + max(0, riga_contenuto - 1)


def cuu_clear_panel_input(
    testo: str,
    larghezza_console: int,
    padding: tuple[int, int] = PADDING_PANEL_INPUT,
) -> int:
    """
    CUU dal fine-contenuto (wrap incluso) al bordo superiore del Panel input.

    Usato su Enter / reprint: una sola fonte di verità per la salita.
    """
    padding_v, padding_h = padding
    utile = larghezza_interna_panel(larghezza_console, padding_h)
    return righe_da_contenuto_a_bordo_superiore(
        padding_v,
        riga_contenuto=righe_visive_contenuto(testo, utile),
    )


def stampa_panel(
    console: Console,
    corpo: object,
    title_markup: str,
    *,
    border_style: str = STILE_BORDO_PANEL,
    padding: tuple[int, int] = (0, 1),
    sincronizza: bool = True,
) -> None:
    """Stampa un Panel con title troncato e width esplicita."""
    if sincronizza:
        sincronizza_dimensioni(console, riserva_barra=True)
    larghezza = console.width
    titolo = titolo_panel_adattivo(
        title_markup,
        max(1, larghezza - MARGINE_TITOLO_PANEL),
    )
    console.print(Panel(
        corpo,
        title=titolo,
        title_align="left",
        border_style=border_style,
        padding=padding,
        width=larghezza,
        expand=True,
    ))


def posiziona_cursore_nel_panel_input(
    *,
    righe_contenuto: int = 1,
    padding: tuple[int, int] = PADDING_PANEL_INPUT,
    colonna_extra: int = 0,
    is_terminal: bool = True,
) -> None:
    """Sposta il cursore sulla riga di contenuto di un Panel appena stampato."""
    if not is_terminal or not sys.stdout.isatty():
        return
    padding_v, padding_h = padding
    righe_su = righe_salita_panel(righe_contenuto, padding_v)
    colonna = colonna_cursore_panel(padding_h) + max(0, colonna_extra)
    sys.stdout.write(f"\033[{righe_su}A\033[{colonna}G")
    sys.stdout.flush()


def posiziona_cursore_a_indice(
    testo: str,
    indice: int,
    larghezza_utile: int,
    *,
    padding: tuple[int, int] = PADDING_PANEL_INPUT,
    is_terminal: bool = True,
) -> None:
    """
    Dopo `stampa_panel`, mette il cursore sull'indice Unicode in `testo`.

    Dal basso del box: `CUU` cresce se l'indice non è sull'ultima riga wrap.
    """
    if not is_terminal or not sys.stdout.isatty():
        return
    indice = max(0, min(indice, len(testo)))
    col_pref, righe_pref = geometria_cursore_contenuto(testo[:indice], larghezza_utile)
    _col_full, righe_full = geometria_cursore_contenuto(testo, larghezza_utile)
    padding_v, padding_h = padding
    righe_su = 2 + padding_v + (righe_full - righe_pref)
    colonna = colonna_cursore_panel(padding_h) + col_pref
    sys.stdout.write(f"\033[{righe_su}A\033[{colonna}G")
    sys.stdout.flush()


def larghezza_lettura(larghezza_console: int) -> int:
    """Larghezza utile per markdown e stream: mai oltre il cap di lettura."""
    return max(1, min(int(larghezza_console), LARGHEZZA_LETTURA_MARKDOWN))


def riga_composer(testo: str) -> str:
    """Riga di editing: prefisso più contenuto utente."""
    return f"{PREFISSO_COMPOSER}{testo}"


def cuu_clear_composer(testo: str, larghezza_console: int) -> int:
    """
    CUU dall'ultima riga del composer al titolo, titolo incluso.

    Stessa approssimazione di `cuu_clear_panel_input`: si usa l'altezza del
    contenuto, non la riga esatta del cursore.
    """
    utile = max(1, larghezza_console)
    return righe_visive_contenuto(riga_composer(testo), utile)


def stampa_composer(
    console: Console,
    titolo_markup: str,
    corpo: Text,
    *,
    sincronizza: bool = False,
) -> None:
    """Stampa titolo + riga `› ` senza box, per l'editing della richiesta."""
    if sincronizza:
        sincronizza_dimensioni(console, riserva_barra=True)
    titolo = titolo_panel_adattivo(titolo_markup, max(1, console.width))
    console.print(Text.from_markup(titolo), highlight=False)
    riga = Text()
    riga.append(PREFISSO_COMPOSER, style="user.label")
    if corpo.plain != " ":
        riga.append_text(corpo)
    console.print(riga, highlight=False, soft_wrap=True)


def stampa_messaggio_utente(
    console: Console,
    titolo_markup: str,
    testo: str,
    *,
    sincronizza: bool = True,
) -> None:
    """Turno utente confermato: titolo e corpo, senza prefisso di editing."""
    if sincronizza:
        sincronizza_dimensioni(console, riserva_barra=True)
    titolo = titolo_panel_adattivo(titolo_markup, max(1, console.width))
    console.print(Text.from_markup(titolo), highlight=False)
    console.print(Text(testo) if testo else Text(" ", style="dim"), highlight=False)


def posiziona_cursore_nel_composer(*, is_terminal: bool = True) -> None:
    """Dopo `stampa_composer` vuoto, mette il cursore dopo `› `."""
    if not is_terminal or not sys.stdout.isatty():
        return
    colonna = 1 + cell_len(PREFISSO_COMPOSER)
    sys.stdout.write(f"\033[1A\033[{colonna}G")
    sys.stdout.flush()


def posiziona_cursore_composer(
    testo: str,
    indice: int,
    larghezza_console: int,
    *,
    is_terminal: bool = True,
) -> None:
    """Dopo `stampa_composer`, mette il cursore sull'indice Unicode in `testo`."""
    if not is_terminal or not sys.stdout.isatty():
        return
    utile = max(1, larghezza_console)
    indice = max(0, min(indice, len(testo)))
    composto = riga_composer(testo)
    indice_vis = len(PREFISSO_COMPOSER) + indice
    col_pref, righe_pref = geometria_cursore_contenuto(composto[:indice_vis], utile)
    _col_full, righe_full = geometria_cursore_contenuto(composto, utile)
    righe_su = 1 + (righe_full - righe_pref)
    colonna = 1 + col_pref
    sys.stdout.write(f"\033[{righe_su}A\033[{colonna}G")
    sys.stdout.flush()


def cancella_righe_sopra(numero_righe: int, *, is_terminal: bool = True) -> None:
    """
    Torna a colonna 1, sale di N righe e cancella fino a fine schermo (`ED 0`).

    Il CR è obbligatorio: `CUU` preserva la colonna; senza, un reprint a metà
    riga duplica il titolo del Panel.
    """
    if numero_righe < 1 or not is_terminal or not sys.stdout.isatty():
        return
    sys.stdout.write(f"\r\033[{numero_righe}A\033[J")
    sys.stdout.flush()


def markup_header_sessione(
    stato: dict[str, bool],
    etichetta_modello: str,
    cartella: str = "",
) -> str:
    """Una riga sticky: modello, cartella documenti, thinking e hint `/help`."""
    thinking = "thinking" if stato.get("thinking", False) else "thinking off"
    dim: list[str] = []
    modello = etichetta_modello.strip()
    if modello:
        dim.append(modello)
    if cartella:
        dim.append(escape(f"cartella {cartella}"))
    dim.extend([thinking, "/help"])
    return f"[agent.label]Assistente[/] [dim]· {' · '.join(dim)}[/]"


def altezza_composer(testo: str, larghezza_console: int) -> int:
    """Titolo più righe visive del corpo `› ` + testo."""
    utile = max(1, larghezza_console)
    return 1 + righe_visive_contenuto(riga_composer(testo), utile)


def altezza_composer_sticky(
    testo: str,
    larghezza_console: int,
    righe_tty: int,
    *,
    header: bool,
) -> int:
    """Altezza riservata al composer, limitata al viewport sotto header e stato."""
    pieno = altezza_composer(testo, larghezza_console)
    chrome = (RIGHE_HEADER_STICKY if header else 0) + 1
    massimo = min(MAX_RIGHE_COMPOSER_STICKY, max(2, righe_tty - chrome - 1))
    return max(2, min(pieno, massimo))


def riga_inizio_scroll_sticky(*, header: bool) -> int:
    """Prima riga 1-based della scroll-region sotto l'header."""
    return 1 + (RIGHE_HEADER_STICKY if header else 0)


def riga_inizio_composer_sticky(righe_tty: int, righe_composer: int) -> int:
    """Riga 1-based della prima riga composer (lo stato occupa l'ultima)."""
    return max(1, int(righe_tty) - max(1, righe_composer))


def riga_fine_scroll_sticky(
    righe_tty: int,
    *,
    header: bool,
    righe_composer: int,
) -> int:
    """Ultima riga 1-based della scroll-region (sopra composer e stato)."""
    basso = 1 + max(0, righe_composer)
    top = 1 + (RIGHE_HEADER_STICKY if header else 0)
    return max(top, int(righe_tty) - basso)


def sequenza_regione_scroll(riga_prima: int, riga_ultima: int) -> str:
    """DECSTBM: freeze header e fascia bassa fuori dallo scroll."""
    prima = max(1, riga_prima)
    ultima = max(prima, riga_ultima)
    return f"\033[{prima};{ultima}r"


def _render_text_ansi(console: Console, text: Text, max_colonne: int) -> str:
    """Render Rich senza scrivere sul TTY; ignora i control sequence."""
    utile = max(1, max_colonne)
    opzioni = console.options.update(width=utile, no_wrap=True)
    pezzi: list[str] = []
    for segmento in console.render(text, opzioni):
        if segmento.control:
            continue
        stile = segmento.style
        if stile is None:
            pezzi.append(segmento.text)
            continue
        pezzi.append(stile.render(segmento.text, color_system=console._color_system))
    return tronca_visibile("".join(pezzi).replace("\n", ""), utile)


def render_markup_linea(console: Console, markup: str, max_colonne: int) -> str:
    """Markup Rich → ANSI di una riga, troncata alle celle visibili."""
    return _render_text_ansi(console, Text.from_markup(markup), max_colonne)


def pulisci_righe_assolute(
    riga_inizio: int,
    numero: int,
    *,
    is_terminal: bool = True,
) -> None:
    """Cancella N righe a coordinate assolute, senza toccare lo scrollback."""
    if numero < 1 or not is_terminal or not sys.stdout.isatty():
        return
    parti = [f"\033[{riga_inizio + indice};1H\033[2K" for indice in range(numero)]
    sys.stdout.write("".join(parti))
    sys.stdout.flush()


def disegna_riga_assoluta(
    riga: int,
    testo_ansi: str,
    *,
    is_terminal: bool = True,
    restaura_cursore: bool = True,
) -> None:
    """Scrive una riga a CUP assoluto; di default non ruba il cursore."""
    if not is_terminal or not sys.stdout.isatty() or riga < 1:
        return
    # ?7l spegne l'autowrap: un header più largo non finisce nella riga sotto.
    corpo = f"\033[{riga};1H\033[?7l\033[2K{testo_ansi}\033[?7h"
    if restaura_cursore:
        sys.stdout.write(f"\0337{corpo}\0338")
    else:
        sys.stdout.write(corpo)
    sys.stdout.flush()


def posiziona_cursore_assoluto(
    riga: int,
    colonna: int,
    *,
    is_terminal: bool = True,
) -> None:
    """CUP 1-based; no-op fuori da TTY."""
    if not is_terminal or not sys.stdout.isatty():
        return
    sys.stdout.write(f"\033[{max(1, riga)};{max(1, colonna)}H")
    sys.stdout.flush()


def posiziona_inizio_transcript(
    *,
    header: bool,
    is_terminal: bool = True,
) -> None:
    """Cursore sulla prima riga della scroll-region, colonna 1."""
    posiziona_cursore_assoluto(
        riga_inizio_scroll_sticky(header=header),
        1,
        is_terminal=is_terminal,
    )


def posiziona_fine_transcript(
    righe_tty: int,
    *,
    header: bool,
    righe_composer: int = 0,
    is_terminal: bool = True,
) -> None:
    """Cursore sull'ultima riga della scroll-region, colonna 1."""
    riga = riga_fine_scroll_sticky(
        righe_tty,
        header=header,
        righe_composer=righe_composer,
    )
    posiziona_cursore_assoluto(riga, 1, is_terminal=is_terminal)


def stampa_header_sticky(
    console: Console,
    markup: str,
    *,
    is_terminal: bool = True,
) -> None:
    """Disegna l'header sulla riga 1 senza wipe dello storico."""
    if not is_terminal or not sys.stdout.isatty():
        return
    colonne, _righe = tty_size(console)
    testo = render_markup_linea(console, markup, max(1, colonne - 1))
    disegna_riga_assoluta(1, testo, is_terminal=True, restaura_cursore=True)


def stampa_composer_sticky(
    console: Console,
    titolo_markup: str,
    corpo: Text,
    testo: str,
    cursore: int,
    righe_tty: int,
    righe_composer: int,
    *,
    is_terminal: bool = True,
) -> None:
    """Disegna titolo + `› ` sopra la barra, a coordinate fisse."""
    if not is_terminal or not sys.stdout.isatty():
        stampa_composer(console, titolo_markup, corpo, sincronizza=False)
        return
    utile = max(1, console.width)
    inizio = riga_inizio_composer_sticky(righe_tty, righe_composer)
    titolo = render_markup_linea(console, titolo_markup, utile)
    riga_corpo = Text()
    riga_corpo.append(PREFISSO_COMPOSER, style="user.label")
    if corpo.plain != " ":
        riga_corpo.append_text(corpo)
    avvolte = riga_corpo.wrap(console, utile) or [Text("")]
    corpo_righe = max(1, righe_composer - 1)
    visibili = avvolte[-corpo_righe:]
    while len(visibili) < corpo_righe:
        visibili.append(Text(""))
    disegna_riga_assoluta(inizio, titolo, is_terminal=True, restaura_cursore=False)
    for offset, pezzo in enumerate(visibili, start=1):
        ansi = _render_text_ansi(console, pezzo, utile)
        disegna_riga_assoluta(
            inizio + offset,
            ansi,
            is_terminal=True,
            restaura_cursore=False,
        )
    composto = riga_composer(testo)
    indice = max(0, min(cursore, len(testo)))
    indice_vis = len(PREFISSO_COMPOSER) + indice
    col_pref, riga_rel = geometria_cursore_contenuto(composto[:indice_vis], utile)
    riga_corpo_inizio = inizio + 1
    riga_cursore = riga_corpo_inizio + riga_rel - 1
    riga_max = inizio + righe_composer - 1
    riga_cursore = min(max(riga_corpo_inizio, riga_cursore), riga_max)
    posiziona_cursore_assoluto(riga_cursore, 1 + col_pref, is_terminal=True)
