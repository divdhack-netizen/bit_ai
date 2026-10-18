"""Text extraction from the folder's documents: text, PDF, docx, Pages and images."""

from __future__ import annotations

import zipfile
from collections.abc import Callable
from pathlib import Path
from xml.etree import ElementTree

PDF_EXTENSIONS = frozenset({".pdf"})
DOCX_EXTENSIONS = frozenset({".docx"})
PAGES_EXTENSIONS = frozenset({".pages"})
# Keynote and Numbers: same IWA container, but text fragmented into slides/cells.
UNSUPPORTED_IWORK_EXTENSIONS = frozenset({".key", ".numbers"})
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
# Images embedded in a scanned PDF: beyond this cap the vision cost explodes.
MAX_PDF_IMAGES = 4

# An IWA text block under this threshold is metadata (UUIDs, URLs, styles), not body.
_MIN_IWA_TEXT_CHARS = 200

# (image bytes, relative path) → text; None if vision is not available.
ImageDescriber = Callable[[bytes, str], str]

_NS_WORD = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


# Tail addressed to the model: the end-of-steps fallback strips it before showing it.
UNSUPPORTED_FORMAT_INSTRUCTION = (
    " Non cercarlo con altri tool: rispondi subito all'utente che il formato "
    "non è supportato."
)


class UnreadableDocument(Exception):
    """Unsupported format or corrupt file: the message goes to the model."""


def is_image(path: Path) -> bool:
    """True for the formats read through the model's vision."""
    return path.suffix.casefold() in IMAGE_EXTENSIONS


def extract_text(
    path: Path,
    relative: str,
    *,
    describe_image: ImageDescriber | None = None,
) -> str:
    """
    Readable text of the document, chosen by extension.

    Args:
        path: Absolute file already validated by the sandbox.
        relative: Relative path, used in messages and for vision.
        describe_image: The model's vision; None rejects images.

    Returns:
        The document text. PDFs carry a ``[pagina N]`` marker.

    Raises:
        UnreadableDocument: unsupported binary format or corrupt file.
    """
    suffix = path.suffix.casefold()
    if suffix in PDF_EXTENSIONS:
        return _pdf_text(path, relative, describe_image)
    if suffix in DOCX_EXTENSIONS:
        return _docx_text(path, relative)
    if suffix in PAGES_EXTENSIONS:
        return _pages_text(path, relative)
    if suffix in UNSUPPORTED_IWORK_EXTENSIONS:
        raise UnreadableDocument(
            f"Formato {suffix} (Keynote/Numbers) non supportato: {relative}. "
            "Chiedi all'utente di esportarlo in PDF."
        )
    if suffix in IMAGE_EXTENSIONS:
        if describe_image is None:
            raise UnreadableDocument(
                f"Lettura immagini non disponibile in questa sessione: {relative}"
            )
        return describe_image(path.read_bytes(), relative)
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise UnreadableDocument(
            f"Formato {suffix or 'senza estensione'} non supportato o file binario: "
            f"{relative}. Il file esiste ma non si può leggere; formati letti: "
            f"txt, md, pdf, docx, pages, immagini.{UNSUPPORTED_FORMAT_INSTRUCTION}"
        ) from error


def _pdf_text(
    path: Path,
    relative: str,
    describe_image: ImageDescriber | None,
) -> str:
    """Text per page; a scanned PDF goes through its embedded images."""
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(path)
        pages = [
            (index, (page.extract_text() or "").strip())
            for index, page in enumerate(reader.pages, start=1)
        ]
    except (PdfReadError, OSError, ValueError) as error:
        raise UnreadableDocument(f"PDF non leggibile: {relative} ({error})") from error
    blocks = [f"[pagina {index}]\n{text}" for index, text in pages if text]
    if blocks:
        return "\n\n".join(blocks)
    # No text: probably a scan. Vision reads the first images.
    if describe_image is None:
        raise UnreadableDocument(
            f"PDF senza testo estraibile (forse scansionato): {relative}"
        )
    read_parts: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        for image in page.images:
            if len(read_parts) >= MAX_PDF_IMAGES:
                break
            description = describe_image(image.data, f"{relative} pagina {index}")
            read_parts.append(f"[pagina {index}]\n{description.strip()}")
    if not read_parts:
        raise UnreadableDocument(f"PDF senza testo né immagini: {relative}")
    return "\n\n".join(read_parts)


def _docx_text(path: Path, relative: str) -> str:
    """Paragraphs of word/document.xml, using only the standard library."""
    try:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml")
        root = ElementTree.fromstring(xml)
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError, OSError) as error:
        raise UnreadableDocument(f"docx non leggibile: {relative} ({error})") from error
    paragraphs: list[str] = []
    for paragraph in root.iter(f"{_NS_WORD}p"):
        parts: list[str] = []
        for node in paragraph.iter():
            if node.tag == f"{_NS_WORD}t" and node.text:
                parts.append(node.text)
            elif node.tag == f"{_NS_WORD}tab":
                parts.append("\t")
            elif node.tag in {f"{_NS_WORD}br", f"{_NS_WORD}cr"}:
                parts.append("\n")
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def _pages_text(path: Path, relative: str) -> str:
    """Body of a Pages document from Index/Document.iwa, using only the standard library."""
    try:
        with zipfile.ZipFile(path) as archive:
            iwa = archive.read("Index/Document.iwa")
        raw = _decompress_iwa(iwa)
    except (zipfile.BadZipFile, KeyError, OSError, IndexError, ValueError) as error:
        raise UnreadableDocument(
            f"Pages non leggibile: {relative} ({error}). "
            "Chiedi all'utente di esportarlo in PDF o docx."
        ) from error
    blocks = _protobuf_strings(raw, _MIN_IWA_TEXT_CHARS)
    if not blocks:
        raise UnreadableDocument(f"Pages senza testo estraibile: {relative}")
    # U+2028 is Pages' «soft» line break; U+FFFC marks embedded objects.
    text = "\n\n".join(blocks).replace("\u2028", "\n").replace("\ufffc", "")
    return text.strip()


def _decompress_iwa(data: bytes) -> bytes:
    """IWA = Snappy blocks without CRC: 0x00 header + 24-bit little-endian length."""
    output = bytearray()
    index = 0
    while index < len(data):
        if data[index] != 0:
            raise ValueError("header IWA inatteso")
        length = int.from_bytes(data[index + 1 : index + 4], "little")
        output += _snappy(data[index + 4 : index + 4 + length])
        index += 4 + length
    return bytes(output)


def _read_varint(data: bytes, index: int) -> tuple[int, int]:
    """Protobuf/Snappy varint: (value, index after the varint)."""
    value = 0
    shift = 0
    while True:
        byte = data[index]
        index += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            return value, index
        if shift > 35:
            raise ValueError("varint troppo lungo")


def _snappy(block: bytes) -> bytes:
    """Raw Snappy decompressor (block format, no framing)."""
    _, index = _read_varint(block, 0)
    output = bytearray()
    while index < len(block):
        tag = block[index]
        index += 1
        event_type = tag & 3
        if event_type == 0:
            length = tag >> 2
            if length >= 60:
                extra = length - 59
                length = int.from_bytes(block[index : index + extra], "little")
                index += extra
            length += 1
            output += block[index : index + length]
            index += length
            continue
        if event_type == 1:
            length = ((tag >> 2) & 7) + 4
            offset = ((tag >> 5) << 8) | block[index]
            index += 1
        elif event_type == 2:
            length = (tag >> 2) + 1
            offset = int.from_bytes(block[index : index + 2], "little")
            index += 2
        else:
            length = (tag >> 2) + 1
            offset = int.from_bytes(block[index : index + 4], "little")
            index += 4
        if offset == 0 or offset > len(output):
            raise ValueError("offset Snappy non valido")
        # Byte-by-byte copy: the source may overlap the destination.
        for _ in range(length):
            output.append(output[-offset])
    return bytes(output)


def _protobuf_strings(data: bytes, minimum: int) -> list[str]:
    """
    Protobuf string fields (wire type 2) at least ``minimum`` characters long.

    Without Apple's schemas the field is recognized by its tag and the exact
    payload length, which must be valid UTF-8 and mostly text.
    """
    found: list[str] = []
    index = 0
    while index < len(data) - 2:
        if data[index] & 7 != 2:
            index += 1
            continue
        try:
            length, start = _read_varint(data, index + 1)
        except (IndexError, ValueError):
            index += 1
            continue
        end = start + length
        if length < minimum or end > len(data):
            index += 1
            continue
        try:
            text = data[start:end].decode("utf-8")
        except UnicodeDecodeError:
            index += 1
            continue
        letters = sum(char.isalpha() for char in text)
        if letters < len(text) * 0.5:
            index += 1
            continue
        found.append(text)
        index = end
    return found
