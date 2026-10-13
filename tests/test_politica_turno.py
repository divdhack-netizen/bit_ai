"""Route dallo schema: domande e compiti passano dai documenti via ReAct."""

from __future__ import annotations

from assistente.politica_turno import (
    _e_chat_locale,
    analisi_da_schema,
    capabilities_react,
    decisione_da_schema,
    decisione_react_aperta,
    decisione_saluto,
)
from assistente.schema_turno import SchemaTurno
from assistente.session import (
    _IDENTITA_AGENTE,
    _comprensione_dallo_schema,
    _costruisci_descrizione_task,
)

FRASE_IMPREVISTA = "prepara in questa cartella un piccolo gioco"


def test_explain_va_sui_documenti_in_sola_lettura() -> None:
    """Closed-corpus: una domanda apre i tool di lettura, non i pesi del modello."""
    decisione = decisione_da_schema(
        SchemaTurno(kind="explain", subject="balenottera"),
    )
    assert decisione.percorso == "react"
    assert decisione.scrittura_consentita is False


def test_chat_pura_resta_direct() -> None:
    """Un commento sulla conversazione non ha nulla da cercare nei documenti."""
    decisione = decisione_da_schema(SchemaTurno(kind="chat"))
    assert decisione.percorso == "direct"
    assert decisione.scrittura_consentita is False


def test_workspace_imprevisto_e_react_con_scrittura() -> None:
    """Il kind workspace autorizza i tool, anche se la frase non ha verbi noti."""
    decisione = decisione_da_schema(
        SchemaTurno(kind="workspace", subject="gioco", folder="gioco"),
    )
    assert decisione.percorso == "react"
    assert decisione.scrittura_consentita is True
    assert FRASE_IMPREVISTA not in decisione.motivo_percorso


def test_explain_con_path_va_react() -> None:
    """Con una destinazione reale il kind morbido apre i tool."""
    decisione = decisione_da_schema(
        SchemaTurno(kind="explain", paths=("note.md",)),
    )
    assert decisione.percorso == "react"


def test_permesso_scrittura_non_e_intento_di_scrittura() -> None:
    """Un ReAct di sola consultazione apre i tool ma non chiede una scrittura."""
    from assistente.validatore_turno import valida_turno

    esito = valida_turno(
        SchemaTurno(kind="workspace", folder="progetto"),
        "elenca quello che c'è nella cartella progetto",
    )
    decisione = decisione_da_schema(esito.schema)
    assert decisione.percorso == "react"
    assert decisione.scrittura_consentita is True
    assert decisione.analisi.scrittura is False


def test_lettura_react_non_autorizza_la_scrittura() -> None:
    """Il file da leggere resta sul ReAct, senza tool né istruzione di scrittura."""
    from assistente.validatore_turno import valida_turno

    richiesta = "leggi il file rassegna.txt e dimmi di cosa parla"
    esito = valida_turno(
        SchemaTurno(
            kind="workspace",
            write=True,
            deliver="file",
            subject="contenuto",
        ),
        richiesta,
    )
    decisione = decisione_da_schema(esito.schema)
    assert decisione.percorso == "react"
    assert decisione.scrittura_consentita is False
    task = _costruisci_descrizione_task(
        richiesta,
        verifica_completa=False,
        decisione=decisione,
        comprensione=_comprensione_dallo_schema(esito.schema),
    )
    assert "Usa scrivi_file" not in task
    assert "Scrittura consentita: no" in task
    assert "niente scrivi_file" in task


def test_workspace_senza_query_resta_react() -> None:
    """Senza query il compito resta sul ReAct, anche con una cartella di destinazione."""
    decisione = decisione_da_schema(
        SchemaTurno(
            kind="workspace",
            deliver="file",
            write=True,
            folder="progetto",
        ),
    )
    assert decisione.percorso == "react"


def test_schema_assente_apre_react_completo() -> None:
    """JSON illeggibile o termica: tool completi, non una generazione senza strumenti."""
    decisione = decisione_react_aperta()
    assert decisione.percorso == "react"
    assert decisione.scrittura_consentita is True
    assert capabilities_react() == ("file", "summary", "date")


def test_saluto_esatto_resta_direct() -> None:
    """Solo il saluto intero salta lo schema. Una domanda dopo ciao no."""
    assert decisione_saluto().percorso == "direct"
    assert decisione_saluto().scrittura_consentita is False
    assert _e_chat_locale("ciao") is True
    assert _e_chat_locale("ciao, chi è il presidente?") is False


def test_react_impone_il_corpus_senza_ripeterlo() -> None:
    """Il corpus sta nell'identità (prefisso stabile); il task non lo ripete."""
    decisione = decisione_da_schema(SchemaTurno(kind="workspace", folder="gioco"))
    task = _costruisci_descrizione_task(
        FRASE_IMPREVISTA,
        verifica_completa=False,
        decisione=decisione,
        comprensione=_comprensione_dallo_schema(
            SchemaTurno(kind="workspace", subject="gioco", folder="gioco"),
        ),
    )
    assert "Unica fonte: i documenti della cartella" in _IDENTITA_AGENTE
    assert "Unica fonte: i documenti della cartella" not in task
    assert "cerca_testo" in _IDENTITA_AGENTE
    assert "cerca_web" not in task


def test_analisi_da_schema_ignora_le_parole() -> None:
    """La stessa scheda produce gli stessi segnali qualunque sia la frase."""
    schema = SchemaTurno(kind="explain", subject="termodinamica")
    prima = analisi_da_schema(schema)
    seconda = analisi_da_schema(schema)
    assert prima == seconda
    assert prima.spiegazione is True


def test_riversamento_storico_senza_tema_nuovo() -> None:
    """«salva tutto in chat.txt» scrive lo storico; un tema nuovo no."""
    from assistente.politica_turno import e_riversamento_storico, path_documento_nominato

    assert e_riversamento_storico("salva tutto in chat.txt") is True
    assert path_documento_nominato("salva tutto in chat.txt") == "chat.txt"
    assert e_riversamento_storico("salva le abitudini delle balene in a.txt") is False


def test_destinazione_non_txt_diventa_txt() -> None:
    """Una destinazione nuova .md diventa .txt; un sorgente esistente resta com'è."""
    from assistente.session import _adatta_destinazioni_txt, _destinazione_scrittura

    schema = SchemaTurno(
        kind="workspace",
        write=True,
        deliver="file",
        paths=("report.docx", "note.md"),
    )
    adattato, nota = _adatta_destinazioni_txt(schema, lambda path: path == "report.docx")
    assert adattato is not None
    assert adattato.paths == ("report.docx", "note.txt")
    assert "note.md → note.txt" in nota
    assert _destinazione_scrittura(adattato.paths) == "note.txt"
    occupato, _ = _adatta_destinazioni_txt(
        SchemaTurno(kind="workspace", write=True, paths=("note.md",)),
        lambda path: path == "note.txt",
    )
    assert occupato is not None and occupato.paths == ("note_2.txt",)
    lettura = SchemaTurno(kind="workspace", paths=("note.md",))
    assert _adatta_destinazioni_txt(lettura, lambda _path: False) == (lettura, "")
