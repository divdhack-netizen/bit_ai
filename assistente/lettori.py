"""Estrazione testo dai documenti della cartella: testo, PDF, docx, Pages e immagini."""

from __future__ import annotations

import zipfile
from collections.abc import Callable
from pathlib import Path
from xml.etree import ElementTree

ESTENSIONI_PDF = frozenset({".pdf"})
ESTENSIONI_DOCX = frozenset({".docx"})
ESTENSIONI_PAGES = frozenset({".pages"})
# Keynote e Numbers: stesso contenitore IWA, ma testo frammentato in slide/celle.
ESTENSIONI_IWORK_NON_SUPPORTATE = frozenset({".key", ".numbers"})
ESTENSIONI_IMMAGINE = frozenset({".png", ".jpg", ".jpeg", ".webp"})
# Immagini incorporate in un PDF scansionato: oltre questo tetto il costo vision esplode.
MAX_IMMAGINI_PDF = 4

# Un blocco di testo IWA sotto questa soglia è metadato (UUID, URL, stili), non corpo.
_MIN_CARATTERI_TESTO_IWA = 200

# (byte immagine, path relativo) → testo; None se la vision non è disponibile.
DescrittoreImmagine = Callable[[bytes, str], str]

_NS_WORD = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


# Coda rivolta al modello: il ripiego a fine passi la toglie prima di mostrarla.
ISTRUZIONE_FORMATO_NON_SUPPORTATO = (
    " Non cercarlo con altri tool: rispondi subito all'utente che il formato "
    "non è supportato."
)


class DocumentoNonLeggibile(Exception):
    """Formato non supportato o file corrotto: il messaggio va al modello."""


def e_immagine(path: Path) -> bool:
    """True per i formati letti con la vision del modello."""
    return path.suffix.casefold() in ESTENSIONI_IMMAGINE


def estrai_testo(
    path: Path,
    relative: str,
    *,
    descrivi_immagine: DescrittoreImmagine | None = None,
) -> str:
    """
    Testo leggibile del documento, scelto dall'estensione.

    Args:
        path: File assoluto già validato dalla sandbox.
        relative: Path relativo, usato nei messaggi e per la vision.
        descrivi_immagine: Vision del modello; None rifiuta le immagini.

    Returns:
        Testo del documento. I PDF portano un marcatore ``[pagina N]``.

    Raises:
        DocumentoNonLeggibile: formato binario non supportato o file corrotto.
    """
    suffix = path.suffix.casefold()
    if suffix in ESTENSIONI_PDF:
        return _testo_pdf(path, relative, descrivi_immagine)
    if suffix in ESTENSIONI_DOCX:
        return _testo_docx(path, relative)
    if suffix in ESTENSIONI_PAGES:
        return _testo_pages(path, relative)
    if suffix in ESTENSIONI_IWORK_NON_SUPPORTATE:
        raise DocumentoNonLeggibile(
            f"Formato {suffix} (Keynote/Numbers) non supportato: {relative}. "
            "Chiedi all'utente di esportarlo in PDF."
        )
    if suffix in ESTENSIONI_IMMAGINE:
        if descrivi_immagine is None:
            raise DocumentoNonLeggibile(
                f"Lettura immagini non disponibile in questa sessione: {relative}"
            )
        return descrivi_immagine(path.read_bytes(), relative)
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise DocumentoNonLeggibile(
            f"Formato {suffix or 'senza estensione'} non supportato o file binario: "
            f"{relative}. Il file esiste ma non si può leggere; formati letti: "
            f"txt, md, pdf, docx, pages, immagini.{ISTRUZIONE_FORMATO_NON_SUPPORTATO}"
        ) from error


def _testo_pdf(
    path: Path,
    relative: str,
    descrivi_immagine: DescrittoreImmagine | None,
) -> str:
    """Testo per pagina; un PDF scansionato passa dalle immagini incorporate."""
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(path)
        pagine = [
            (indice, (pagina.extract_text() or "").strip())
            for indice, pagina in enumerate(reader.pages, start=1)
        ]
    except (PdfReadError, OSError, ValueError) as error:
        raise DocumentoNonLeggibile(f"PDF non leggibile: {relative} ({error})") from error
    blocchi = [f"[pagina {indice}]\n{testo}" for indice, testo in pagine if testo]
    if blocchi:
        return "\n\n".join(blocchi)
    # Nessun testo: probabile scansione. La vision legge le prime immagini.
    if descrivi_immagine is None:
        raise DocumentoNonLeggibile(
            f"PDF senza testo estraibile (forse scansionato): {relative}"
        )
    letti: list[str] = []
    for indice, pagina in enumerate(reader.pages, start=1):
        for immagine in pagina.images:
            if len(letti) >= MAX_IMMAGINI_PDF:
                break
            descrizione = descrivi_immagine(immagine.data, f"{relative} pagina {indice}")
            letti.append(f"[pagina {indice}]\n{descrizione.strip()}")
    if not letti:
        raise DocumentoNonLeggibile(f"PDF senza testo né immagini: {relative}")
    return "\n\n".join(letti)


def _testo_docx(path: Path, relative: str) -> str:
    """Paragrafi di word/document.xml, con la sola standard library."""
    try:
        with zipfile.ZipFile(path) as archivio:
            xml = archivio.read("word/document.xml")
        radice = ElementTree.fromstring(xml)
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError, OSError) as error:
        raise DocumentoNonLeggibile(f"docx non leggibile: {relative} ({error})") from error
    paragrafi: list[str] = []
    for paragrafo in radice.iter(f"{_NS_WORD}p"):
        parti: list[str] = []
        for nodo in paragrafo.iter():
            if nodo.tag == f"{_NS_WORD}t" and nodo.text:
                parti.append(nodo.text)
            elif nodo.tag == f"{_NS_WORD}tab":
                parti.append("\t")
            elif nodo.tag in {f"{_NS_WORD}br", f"{_NS_WORD}cr"}:
                parti.append("\n")
        testo = "".join(parti).strip()
        if testo:
            paragrafi.append(testo)
    return "\n".join(paragrafi)


def _testo_pages(path: Path, relative: str) -> str:
    """Corpo di un documento Pages da Index/Document.iwa, con la sola standard library."""
    try:
        with zipfile.ZipFile(path) as archivio:
            iwa = archivio.read("Index/Document.iwa")
        grezzo = _decomprimi_iwa(iwa)
    except (zipfile.BadZipFile, KeyError, OSError, IndexError, ValueError) as error:
        raise DocumentoNonLeggibile(
            f"Pages non leggibile: {relative} ({error}). "
            "Chiedi all'utente di esportarlo in PDF o docx."
        ) from error
    blocchi = _stringhe_protobuf(grezzo, _MIN_CARATTERI_TESTO_IWA)
    if not blocchi:
        raise DocumentoNonLeggibile(f"Pages senza testo estraibile: {relative}")
    # U+2028 è l'a capo «morbido» di Pages; U+FFFC segna oggetti incorporati.
    testo = "\n\n".join(blocchi).replace("\u2028", "\n").replace("\ufffc", "")
    return testo.strip()


def _decomprimi_iwa(dati: bytes) -> bytes:
    """IWA = blocchi Snappy senza CRC: header 0x00 + lunghezza a 24 bit little-endian."""
    uscita = bytearray()
    indice = 0
    while indice < len(dati):
        if dati[indice] != 0:
            raise ValueError("header IWA inatteso")
        lunghezza = int.from_bytes(dati[indice + 1 : indice + 4], "little")
        uscita += _snappy(dati[indice + 4 : indice + 4 + lunghezza])
        indice += 4 + lunghezza
    return bytes(uscita)


def _leggi_varint(dati: bytes, indice: int) -> tuple[int, int]:
    """Varint protobuf/Snappy: (valore, indice dopo il varint)."""
    valore = 0
    shift = 0
    while True:
        byte = dati[indice]
        indice += 1
        valore |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            return valore, indice
        if shift > 35:
            raise ValueError("varint troppo lungo")


def _snappy(blocco: bytes) -> bytes:
    """Decompressore Snappy raw (formato di blocco, senza framing)."""
    _, indice = _leggi_varint(blocco, 0)
    uscita = bytearray()
    while indice < len(blocco):
        tag = blocco[indice]
        indice += 1
        tipo = tag & 3
        if tipo == 0:
            lunghezza = tag >> 2
            if lunghezza >= 60:
                extra = lunghezza - 59
                lunghezza = int.from_bytes(blocco[indice : indice + extra], "little")
                indice += extra
            lunghezza += 1
            uscita += blocco[indice : indice + lunghezza]
            indice += lunghezza
            continue
        if tipo == 1:
            lunghezza = ((tag >> 2) & 7) + 4
            offset = ((tag >> 5) << 8) | blocco[indice]
            indice += 1
        elif tipo == 2:
            lunghezza = (tag >> 2) + 1
            offset = int.from_bytes(blocco[indice : indice + 2], "little")
            indice += 2
        else:
            lunghezza = (tag >> 2) + 1
            offset = int.from_bytes(blocco[indice : indice + 4], "little")
            indice += 4
        if offset == 0 or offset > len(uscita):
            raise ValueError("offset Snappy non valido")
        # Copia byte a byte: le sorgenti possono sovrapporsi alla destinazione.
        for _ in range(lunghezza):
            uscita.append(uscita[-offset])
    return bytes(uscita)


def _stringhe_protobuf(dati: bytes, minimo: int) -> list[str]:
    """
    Campi stringa protobuf (wire type 2) lunghi almeno ``minimo`` caratteri.

    Senza gli schema Apple si riconosce il campo dal tag e dalla lunghezza
    esatta del payload, che deve essere UTF-8 valido e in prevalenza testo.
    """
    trovati: list[str] = []
    indice = 0
    while indice < len(dati) - 2:
        if dati[indice] & 7 != 2:
            indice += 1
            continue
        try:
            lunghezza, inizio = _leggi_varint(dati, indice + 1)
        except (IndexError, ValueError):
            indice += 1
            continue
        fine = inizio + lunghezza
        if lunghezza < minimo or fine > len(dati):
            indice += 1
            continue
        try:
            testo = dati[inizio:fine].decode("utf-8")
        except UnicodeDecodeError:
            indice += 1
            continue
        lettere = sum(carattere.isalpha() for carattere in testo)
        if lettere < len(testo) * 0.5:
            indice += 1
            continue
        trovati.append(testo)
        indice = fine
    return trovati
