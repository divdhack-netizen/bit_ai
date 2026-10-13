"""Lettori closed-corpus: txt, pdf, docx e immagini, più cerca_testo."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from assistente.config import FileConfig
from assistente.file_service import WorkspaceFileService
from assistente.lettori import DocumentoNonLeggibile, estrai_testo
from assistente.tools import cerca_testo_da_testo, leggi_documento_da_testo

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


def _scrivi_docx(path: Path) -> None:
    """docx minimale: solo word/document.xml, come lo legge il lettore."""
    with zipfile.ZipFile(path, "w") as archivio:
        archivio.writestr("word/document.xml", _DOCX_XML)


def _scrivi_pdf(path: Path, righe: list[str]) -> None:
    """PDF a una pagina per riga, con testo Helvetica estraibile da pypdf."""
    oggetti: list[bytes] = []
    n_pagine = len(righe)
    font_id = 3 + 2 * n_pagine
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n_pagine))
    oggetti.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    oggetti.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pagine} >>".encode())
    for indice, riga in enumerate(righe):
        contenuto = f"BT /F1 12 Tf 72 720 Td ({riga}) Tj ET".encode()
        oggetti.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {4 + 2 * indice} 0 R "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>"
            ).encode()
        )
        oggetti.append(
            b"<< /Length " + str(len(contenuto)).encode() + b" >>\nstream\n"
            + contenuto + b"\nendstream"
        )
    oggetti.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    corpo = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for numero, oggetto in enumerate(oggetti, start=1):
        offsets.append(len(corpo))
        corpo += f"{numero} 0 obj\n".encode() + oggetto + b"\nendobj\n"
    xref = len(corpo)
    corpo += f"xref\n0 {len(oggetti) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        corpo += f"{offset:010d} 00000 n \n".encode()
    corpo += (
        f"trailer\n<< /Size {len(oggetti) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode()
    path.write_bytes(bytes(corpo))


def _service(cartella: Path, **kwargs: object) -> WorkspaceFileService:
    return WorkspaceFileService(FileConfig(workspace=cartella), **kwargs)  # type: ignore[arg-type]


def test_docx_estratto_con_standard_library(tmp_path: Path) -> None:
    """Paragrafi e tab del docx arrivano come testo semplice."""
    documento = tmp_path / "report.docx"
    _scrivi_docx(documento)
    testo = estrai_testo(documento, "report.docx")
    assert testo == "La balenottera azzurra mangia krill.\nSeconda\triga"


def test_pdf_testo_per_pagina(tmp_path: Path) -> None:
    """Ogni pagina porta il marcatore [pagina N] per le citazioni."""
    documento = tmp_path / "studio.pdf"
    _scrivi_pdf(documento, ["Le orche cacciano in gruppo", "Le megattere cantano"])
    testo = estrai_testo(documento, "studio.pdf")
    assert "[pagina 1]\nLe orche cacciano in gruppo" in testo
    assert "[pagina 2]\nLe megattere cantano" in testo


def test_file_binario_non_supportato(tmp_path: Path) -> None:
    """Un binario sconosciuto è un errore leggibile, non un crash."""
    documento = tmp_path / "dati.bin"
    documento.write_bytes(b"\xff\xfe\x00\x01")
    with pytest.raises(DocumentoNonLeggibile, match="non supportato.*dati.bin"):
        estrai_testo(documento, "dati.bin")


def _varint(valore: int) -> bytes:
    uscita = bytearray()
    while valore >= 0x80:
        uscita.append((valore & 0x7F) | 0x80)
        valore >>= 7
    uscita.append(valore)
    return bytes(uscita)


def _snappy_letterale(dati: bytes) -> bytes:
    """Snappy valido con un solo blocco letterale (lunghezza in 2 byte extra)."""
    return _varint(len(dati)) + bytes([61 << 2]) + (len(dati) - 1).to_bytes(2, "little") + dati


def _scrivi_pages(path: Path, testo: str) -> None:
    """Pages minimale: Index/Document.iwa con un campo stringa protobuf."""
    payload = b"\x0a\x05UUID1" + b"\x1a" + _varint(len(testo.encode())) + testo.encode()
    blocco = _snappy_letterale(payload)
    iwa = b"\x00" + len(blocco).to_bytes(3, "little") + blocco
    with zipfile.ZipFile(path, "w") as archivio:
        archivio.writestr("Index/Document.iwa", iwa)
        archivio.writestr("preview.jpg", b"\xff\xd8")


_TESTO_PAGES = (
    "Agenti Estintivi\u2028sottotitolo\n\nLa balenottera azzurra mangia krill "
    "ogni giorno e attraversa gli oceani durante la migrazione stagionale, "
    "percorrendo migliaia di chilometri tra le zone di alimentazione e quelle "
    "di riproduzione."
)


def test_pages_estratto_da_document_iwa(tmp_path: Path) -> None:
    """Un .pages si legge senza Pages né dipendenze: corpo testo, a capo normalizzati."""
    documento = tmp_path / "Articolo_01.pages"
    _scrivi_pages(documento, _TESTO_PAGES)
    testo = estrai_testo(documento, "Articolo_01.pages")
    assert testo.startswith("Agenti Estintivi\nsottotitolo")
    assert "migliaia di chilometri" in testo
    assert "UUID1" not in testo


def test_pages_cercabile_e_leggibile_dai_tool(tmp_path: Path) -> None:
    """Regressione: «riassumi Articolo_01.pages» rispondeva «non presente»."""
    _scrivi_pages(tmp_path / "Articolo_01.pages", _TESTO_PAGES)
    service = WorkspaceFileService(FileConfig(workspace=tmp_path))
    assert "krill" in leggi_documento_da_testo("Articolo_01.pages", service=service)
    assert "Articolo_01.pages" in cerca_testo_da_testo("krill", service=service)


def test_pages_corrotto_e_un_errore_leggibile(tmp_path: Path) -> None:
    documento = tmp_path / "rotto.pages"
    documento.write_bytes(b"non uno zip")
    with pytest.raises(DocumentoNonLeggibile, match="PDF o docx"):
        estrai_testo(documento, "rotto.pages")


def test_immagine_passa_dalla_vision(tmp_path: Path) -> None:
    """Le immagini vanno al descrittore (vision del modello), con nome e byte."""
    (tmp_path / "coda.png").write_bytes(b"PNGDATA")
    chiamate: list[tuple[bytes, str]] = []

    def descrivi(immagine: bytes, nome: str) -> str:
        chiamate.append((immagine, nome))
        return "Coda di balena che emerge dall'acqua."

    service = _service(tmp_path, descrivi_immagine=descrivi)
    assert leggi_documento_da_testo("coda.png", service=service) == (
        "Coda di balena che emerge dall'acqua."
    )
    # La seconda lettura usa la cache: niente seconda chiamata vision.
    leggi_documento_da_testo("coda.png", service=service)
    assert chiamate == [(b"PNGDATA", "coda.png")]


def test_immagine_senza_vision_e_un_errore(tmp_path: Path) -> None:
    """Senza descrittore la lettura immagine restituisce un Errore esplicito."""
    (tmp_path / "coda.jpg").write_bytes(b"JPG")
    risultato = leggi_documento_da_testo("coda.jpg", service=_service(tmp_path))
    assert risultato.startswith("Errore:")
    assert "immagini" in risultato


def test_cerca_testo_trova_riga_e_pagina(tmp_path: Path) -> None:
    """Righe con tutti i termini, posizione per riga (txt/docx) o pagina (pdf)."""
    (tmp_path / "note.txt").write_text(
        "introduzione\nIl KRILL è il cibo della balenottera\n",
        encoding="utf-8",
    )
    _scrivi_docx(tmp_path / "report.docx")
    _scrivi_pdf(tmp_path / "studio.pdf", ["Nessun cetaceo qui", "Balenottera e krill a sud"])
    (tmp_path / "coda.png").write_bytes(b"PNG")
    risultato = cerca_testo_da_testo("krill balenottera", service=_service(tmp_path))
    assert "note.txt (riga 2)" in risultato
    assert "report.docx (riga 1)" in risultato
    assert "studio.pdf (pagina 2)" in risultato
    assert "coda.png" not in risultato


def test_cerca_testo_tollera_le_forme_e_ordina_per_pertinenza(tmp_path: Path) -> None:
    """«migrazione» trova «migrano»; la riga con più parole viene prima."""
    (tmp_path / "a.txt").write_text(
        "Le megattere cantano a lungo\n"
        "Le megattere migrano per 8.000 chilometri\n",
        encoding="utf-8",
    )
    risultato = cerca_testo_da_testo(
        "migrazione megattere chilometri",
        service=_service(tmp_path),
    )
    righe = risultato.splitlines()
    assert "a.txt (riga 2)" in righe[1]
    assert "a.txt (riga 1)" in righe[2]


def test_cerca_testo_senza_risultati_e_query_vuota(tmp_path: Path) -> None:
    """Nessun match è un esito esplicito; una query vuota è un errore."""
    (tmp_path / "note.txt").write_text("solo orche\n", encoding="utf-8")
    service = _service(tmp_path)
    assert cerca_testo_da_testo("narvalo", service=service) == (
        "Nessun documento contiene: narvalo"
    )
    (tmp_path / "foto.png").write_bytes(b"PNG")
    assert "Immagini non coperte dalla ricerca: foto.png" in cerca_testo_da_testo(
        "narvalo",
        service=service,
    )
    assert cerca_testo_da_testo("  ", service=service).startswith("Errore:")


def test_scrittura_solo_txt(tmp_path: Path) -> None:
    """scrivi/modifica accettano solo .txt: i sorgenti non si toccano."""
    service = _service(tmp_path)
    service.begin_turn(write_allowed=True)
    assert service.write_text("sintesi.txt", "ok\n").succeeded
    rifiuto = service.write_text("note.md", "no\n")
    assert not rifiuto.succeeded
    assert ".txt" in str(rifiuto.error)
    assert not (tmp_path / "note.md").exists()
    _scrivi_docx(tmp_path / "report.docx")
    patch = service.replace_text("report.docx", "krill", "plancton")
    assert not patch.succeeded
