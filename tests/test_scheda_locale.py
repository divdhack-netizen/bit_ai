"""Scheda locale: i turni chiari non chiamano l'LLM, gli ambigui sì."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from assistente.comprensione import scheda_locale
from assistente.config import ConfigRuntime, FileConfig
from assistente.console import ConsoleUI
from assistente.file_service import WorkspaceFileService
from assistente.host import PowerState
from assistente.politica_turno import _e_chat_locale, decisione_da_schema
from assistente.session import SessioneAgente
from assistente.validatore_turno import ContestoValidazione, valida_turno
from benchmarks import comprensione_dura, comprensione_probe

pytestmark = pytest.mark.unit

# Soglia minima di turni del corpus decisi senza LLM.
_COPERTURA_MINIMA = 0.7
_SORGENTI = {".pdf", ".docx", ".pages", ".png", ".jpg"}


def _esiste_sorgente(path: str) -> bool:
    """Corpus finto: documenti sorgente presenti, i .txt sono destinazioni nuove."""
    return Path(path).suffix.casefold() in _SORGENTI


def _disaccordi(caso: object) -> list[str]:
    """Attese del benchmark violate dalla scheda locale, dopo il validatore."""
    schema = scheda_locale(caso.testo, _esiste_sorgente)
    assert schema is not None
    validato = valida_turno(
        schema,
        caso.testo,
        ContestoValidazione(
            ultimo=getattr(caso, "ultimo", None),
            documento_precedente=getattr(caso, "documento_precedente", None),
        ),
    ).schema
    route = decisione_da_schema(validato).percorso
    write = validato.write or validato.deliver == "file"
    errori: list[str] = []
    if caso.kind_attesi is not None and validato.kind not in caso.kind_attesi:
        errori.append(f"kind={validato.kind}")
    attesa = caso.route_attesa
    if attesa is not None:
        ok = route in attesa if isinstance(attesa, frozenset) else route == attesa
        if not ok:
            errori.append(f"route={route}")
    if caso.write_atteso is not None and write != caso.write_atteso:
        errori.append(f"write={write}")
    if caso.path_atteso is not None and not any(
        path.casefold().endswith(caso.path_atteso.casefold()) for path in validato.paths
    ):
        errori.append(f"paths={validato.paths}")
    if caso.cartella_attesa is not None and (
        caso.cartella_attesa.casefold() not in validato.folder.casefold()
    ):
        errori.append(f"folder={validato.folder!r}")
    clarify_attesa = getattr(caso, "clarify_attesa", None)
    if clarify_attesa is not None and bool(validato.clarify) != clarify_attesa:
        errori.append(f"clarify={validato.clarify!r}")
    return errori


def test_scheda_locale_concorda_con_i_corpus_dei_benchmark() -> None:
    """Quando decide da sola, la scheda locale rispetta tutte le attese dei benchmark."""
    casi = [
        caso
        for corpus in (comprensione_dura.CORPUS, comprensione_probe.CORPUS)
        for caso in corpus
        if not getattr(caso, "non_valutato", False) and not _e_chat_locale(caso.testo)
    ]
    decisi = [caso for caso in casi if scheda_locale(caso.testo, _esiste_sorgente)]
    errori = {caso.testo: _disaccordi(caso) for caso in decisi}
    assert {testo: e for testo, e in errori.items() if e} == {}
    assert len(decisi) / len(casi) >= _COPERTURA_MINIMA


@pytest.mark.parametrize(
    ("testo", "kind", "write", "paths"),
    [
        ("quanto pesa una balenottera azzurra?", "explain", False, ()),
        ("leggi avvistamenti.pdf e dimmi di cosa parla", "workspace", False, ("avvistamenti.pdf",)),
        (
            "riassumi report.docx e salva il riassunto in sintesi.txt",
            "workspace",
            True,
            ("report.docx", "sintesi.txt"),
        ),
        ("Scrivi il file note.md con il testo: prova", "workspace", True, ("note.md",)),
        ("dimmi cosa mangiano le orche, niente file, solo in chat", "explain", False, ()),
    ],
)
def test_scheda_locale_decide_i_casi_chiari(
    testo: str,
    kind: str,
    write: bool,
    paths: tuple[str, ...],
) -> None:
    """Domande, letture e una sola destinazione nuova non chiedono nulla al modello."""
    schema = scheda_locale(testo, _esiste_sorgente)
    assert schema is not None
    assert schema.kind == kind
    assert schema.write is write
    assert schema.paths == paths


@pytest.mark.parametrize(
    "testo",
    [
        "aggiorna note.txt con la data di oggi",
        "salvalo",
        "salvalo in un documento",
        "e il secondo?",
        "appuntami che domani piove",
        "archiviami questo ragionamento in un txt",
        "riassumi a.pdf in uno.txt e in due.txt e salvali",
        "e poi?",
    ],
)
def test_scheda_locale_lascia_al_modello_i_casi_ambigui(testo: str) -> None:
    """Modifiche, deissi e scritture senza una destinazione certa vanno all'LLM."""
    assert scheda_locale(testo, _esiste_sorgente) is None


def test_azione_non_supportata_diventa_chiarimento_senza_llm() -> None:
    """Un'email non apre tool: la scheda locale la consegna al validatore."""
    testo = "mandami una mail con il riassunto"
    schema = scheda_locale(testo, _esiste_sorgente)
    assert schema is not None
    validato = valida_turno(schema, testo).schema
    assert validato.clarify
    assert validato.write is False


@pytest.fixture
def sessione(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SessioneAgente:
    """Sessione su una cartella isolata, con LLM finto e alimentazione da rete."""
    monkeypatch.setattr(
        "assistente.politica_energia.detect_power",
        lambda: PowerState(on_battery=False, low_power_mode=False),
    )
    service = WorkspaceFileService(FileConfig(workspace=tmp_path))
    agente = SessioneAgente(
        MagicMock(spec=ConsoleUI),
        file_service=service,
        llm_factory=MagicMock(),
    )
    yield agente
    agente.close()


def test_turno_chiaro_non_chiama_la_scheda_llm(sessione: SessioneAgente) -> None:
    """La classificazione locale non tocca Ollama."""
    sessione._schema_da_modello = MagicMock(side_effect=AssertionError("LLM chiamato"))
    origine, validazione, decisione = sessione._classifica_turno(
        "quanto pesa una balenottera azzurra?",
        "default",
        None,
    )
    assert origine == "locale"
    assert validazione is not None
    assert decisione.percorso == "react"
    assert decisione.scrittura_consentita is False


def test_turno_ambiguo_chiede_la_scheda_al_modello(sessione: SessioneAgente) -> None:
    """Solo il testo ambiguo paga la chiamata di comprensione."""
    sessione._schema_da_modello = MagicMock(return_value=(None, False, False))
    origine, _validazione, decisione = sessione._classifica_turno(
        "salvalo",
        "default",
        None,
    )
    sessione._schema_da_modello.assert_called_once()
    assert origine == "schema_fallback"
    assert decisione.percorso == "react"


def test_riversamento_non_classifica_il_turno(sessione: SessioneAgente) -> None:
    """«salva tutto in chat.txt» scrive lo storico senza scheda né ricerca."""
    sessione.registra_turno_storico("ciao", "ciao a te")
    sessione._classifica_turno = MagicMock(side_effect=AssertionError("classificato"))
    risposta, scritti = sessione.esegui_richiesta(
        "riportami tutto sul file chat.txt",
        ConfigRuntime(modello="default", burn_protection=False),
    )
    assert scritti == ["chat.txt"]
    assert "chat.txt" in risposta


def test_errore_ollama_non_ripete_la_scheda(sessione: SessioneAgente) -> None:
    """Un errore di trasporto non paga un secondo tentativo: si apre il ReAct."""
    llm = MagicMock()
    llm.invoke.side_effect = ConnectionError("Ollama giù")
    sessione._runtime.get_llm = MagicMock(return_value=llm)
    validazione, _riparato, retry = sessione._schema_da_modello("salvalo", "default")
    assert validazione is None
    assert retry is False
    assert llm.invoke.call_count == 1


def test_json_illeggibile_riprova_una_volta(sessione: SessioneAgente) -> None:
    """Il retry resta per il JSON rotto, con l'errore nel prompt."""
    llm = MagicMock()
    llm.invoke.side_effect = [
        "non è json",
        '{"kind":"workspace","write":true,"deliver":"file","subject":"x",'
        '"paths":["a.txt"],"folder":"","clarify":""}',
    ]
    sessione._runtime.get_llm = MagicMock(return_value=llm)
    validazione, riparato, retry = sessione._schema_da_modello("salvalo", "default")
    assert validazione is not None
    assert validazione.schema.paths == ("a.txt",)
    assert (riparato, retry) == (True, True)
    assert "Errore JSON" in llm.invoke.call_args_list[1].args[0]
