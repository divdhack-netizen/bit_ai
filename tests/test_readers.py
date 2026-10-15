"""Closed-corpus readers: txt, pdf, docx and images, plus cerca_testo."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from assistant.config import FileConfig
from assistant.file_service import WorkspaceFileService
from assistant.readers import UnreadableDocument, extract_text
from assistant.tools import search_text_from_text, read_document_from_text

pytestmark = pytest.mark.unit

_DOCX_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
    "<w:body>"
    "<w:p><w:r><w:t>La balenottera azzurra</w:t></w:r>"
    "<w:r><w:t xml:space=\"preserve\"> mangia krill.</w:t></w:r></w:p>"
    "<w:p><w:r><w:t>Seconda</w:t><w:tab/><w:t>riga</w:t></w:r></w:p>"
    "</w:body></w:document>"
)


def _write_docx(path: Path) -> None:
    """Minimal docx: only word/document.xml, as the reader reads it."""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", _DOCX_XML)


def _write_pdf(path: Path, lines: list[str]) -> None:
    """PDF with one page per line, with Helvetica text that pypdf can extract."""
    objects: list[bytes] = []
    n_pages = len(lines)
    font_id = 3 + 2 * n_pages
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n_pages))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    for index, line in enumerate(lines):
        content = f"BT /F1 12 Tf 72 720 Td ({line}) Tj ET".encode()
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {4 + 2 * index} 0 R "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>"
            ).encode()
        )
        objects.append(
            b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n"
            + content + b"\nendstream"
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    body = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(body))
        body += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(body)
    body += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        body += f"{offset:010d} 00000 n \n".encode()
    body += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode()
    path.write_bytes(bytes(body))


def _service(folder: Path, **kwargs: object) -> WorkspaceFileService:
    return WorkspaceFileService(FileConfig(workspace=folder), **kwargs)  # type: ignore[arg-type]


def test_docx_extracted_with_standard_library(tmp_path: Path) -> None:
    """Docx paragraphs and tabs come out as plain text."""
    document = tmp_path / "report.docx"
    _write_docx(document)
    text = extract_text(document, "report.docx")
    assert text == "La balenottera azzurra mangia krill.\nSeconda\triga"


def test_pdf_text_per_page(tmp_path: Path) -> None:
    """Each page carries the [pagina N] marker for citations."""
    document = tmp_path / "studio.pdf"
    _write_pdf(document, ["Le orche cacciano in gruppo", "Le megattere cantano"])
    text = extract_text(document, "studio.pdf")
    assert "[pagina 1]\nLe orche cacciano in gruppo" in text
    assert "[pagina 2]\nLe megattere cantano" in text


def test_binary_file_not_supported(tmp_path: Path) -> None:
    """An unknown binary is a readable error, not a crash."""
    document = tmp_path / "dati.bin"
    document.write_bytes(b"\xff\xfe\x00\x01")
    with pytest.raises(UnreadableDocument, match="non supportato.*dati.bin"):
        extract_text(document, "dati.bin")


def _varint(value: int) -> bytes:
    output = bytearray()
    while value >= 0x80:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)
    return bytes(output)


def _snappy_literal(data: bytes) -> bytes:
    """Valid Snappy with a single literal block (length in 2 extra bytes)."""
    return _varint(len(data)) + bytes([61 << 2]) + (len(data) - 1).to_bytes(2, "little") + data


def _write_pages(path: Path, text: str) -> None:
    """Minimal Pages: Index/Document.iwa with one protobuf string field."""
    payload = b"\x0a\x05UUID1" + b"\x1a" + _varint(len(text.encode())) + text.encode()
    block = _snappy_literal(payload)
    iwa = b"\x00" + len(block).to_bytes(3, "little") + block
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("Index/Document.iwa", iwa)
        archive.writestr("preview.jpg", b"\xff\xd8")


_PAGES_TEXT = (
    "Agenti Estintivi\u2028sottotitolo\n\nLa balenottera azzurra mangia krill "
    "ogni giorno e attraversa gli oceani durante la migrazione stagionale, "
    "percorrendo migliaia di chilometri tra le zone di alimentazione e quelle "
    "di riproduzione."
)


def test_pages_extracted_from_document_iwa(tmp_path: Path) -> None:
    """A .pages file reads without Pages or dependencies: text body, normalized newlines."""
    document = tmp_path / "Articolo_01.pages"
    _write_pages(document, _PAGES_TEXT)
    text = extract_text(document, "Articolo_01.pages")
    assert text.startswith("Agenti Estintivi\nsottotitolo")
    assert "migliaia di chilometri" in text
    assert "UUID1" not in text


def test_pages_searchable_and_readable_by_tools(tmp_path: Path) -> None:
    """Regression: «riassumi Articolo_01.pages» answered «non presente»."""
    _write_pages(tmp_path / "Articolo_01.pages", _PAGES_TEXT)
    service = WorkspaceFileService(FileConfig(workspace=tmp_path))
    assert "krill" in read_document_from_text("Articolo_01.pages", service=service)
    assert "Articolo_01.pages" in search_text_from_text("krill", service=service)


def test_corrupt_pages_is_a_readable_error(tmp_path: Path) -> None:
    document = tmp_path / "rotto.pages"
    document.write_bytes(b"non uno zip")
    with pytest.raises(UnreadableDocument, match="PDF o docx"):
        extract_text(document, "rotto.pages")


def test_image_goes_through_vision(tmp_path: Path) -> None:
    """Images go to the describer (the model's vision), with name and bytes."""
    (tmp_path / "coda.png").write_bytes(b"PNGDATA")
    calls: list[tuple[bytes, str]] = []

    def describe(image: bytes, name: str) -> str:
        calls.append((image, name))
        return "Coda di balena che emerge dall'acqua."

    service = _service(tmp_path, describe_image=describe)
    assert read_document_from_text("coda.png", service=service) == (
        "Coda di balena che emerge dall'acqua."
    )
    # The second read uses the cache: no second vision call.
    read_document_from_text("coda.png", service=service)
    assert calls == [(b"PNGDATA", "coda.png")]


def test_image_without_vision_is_an_error(tmp_path: Path) -> None:
    """Without a describer, reading an image returns an explicit Errore."""
    (tmp_path / "coda.jpg").write_bytes(b"JPG")
    result = read_document_from_text("coda.jpg", service=_service(tmp_path))
    assert result.startswith("Errore:")
    assert "immagini" in result


def test_search_text_finds_line_and_page(tmp_path: Path) -> None:
    """Lines with every term, position per line (txt/docx) or page (pdf)."""
    (tmp_path / "note.txt").write_text(
        "introduzione\nIl KRILL è il cibo della balenottera\n",
        encoding="utf-8",
    )
    _write_docx(tmp_path / "report.docx")
    _write_pdf(tmp_path / "studio.pdf", ["Nessun cetaceo qui", "Balenottera e krill a sud"])
    (tmp_path / "coda.png").write_bytes(b"PNG")
    result = search_text_from_text("krill balenottera", service=_service(tmp_path))
    assert "note.txt (riga 2)" in result
    assert "report.docx (riga 1)" in result
    assert "studio.pdf (pagina 2)" in result
    assert "coda.png" not in result


def test_search_text_tolerates_forms_and_sorts_by_relevance(tmp_path: Path) -> None:
    """«migrazione» finds «migrano»; the line with more words comes first."""
    (tmp_path / "a.txt").write_text(
        "Le megattere cantano a lungo\n"
        "Le megattere migrano per 8.000 chilometri\n",
        encoding="utf-8",
    )
    result = search_text_from_text(
        "migrazione megattere chilometri",
        service=_service(tmp_path),
    )
    lines = result.splitlines()
    assert "a.txt (riga 2)" in lines[1]
    assert "a.txt (riga 1)" in lines[2]


def test_search_text_no_results_and_empty_query(tmp_path: Path) -> None:
    """No match is an explicit outcome; an empty query is an error."""
    (tmp_path / "note.txt").write_text("solo orche\n", encoding="utf-8")
    service = _service(tmp_path)
    assert search_text_from_text("narvalo", service=service) == (
        "Nessun documento contiene: narvalo"
    )
    (tmp_path / "foto.png").write_bytes(b"PNG")
    assert "Immagini non coperte dalla ricerca: foto.png" in search_text_from_text(
        "narvalo",
        service=service,
    )
    assert search_text_from_text("  ", service=service).startswith("Errore:")


def test_write_txt_only(tmp_path: Path) -> None:
    """scrivi/modifica accept only .txt: sources are never touched."""
    service = _service(tmp_path)
    service.begin_turn(write_allowed=True)
    assert service.write_text("sintesi.txt", "ok\n").succeeded
    rejection = service.write_text("note.md", "no\n")
    assert not rejection.succeeded
    assert ".txt" in str(rejection.error)
    assert not (tmp_path / "note.md").exists()
    _write_docx(tmp_path / "report.docx")
    patch = service.replace_text("report.docx", "krill", "plancton")
    assert not patch.succeeded
