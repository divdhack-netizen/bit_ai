"""Test e2e con Ollama (modello default): riassunto, scrivi_file e risposte dai documenti."""

from __future__ import annotations

from pathlib import Path

import pytest

from assistente.config import SEPARATORE_FILE, WORKSPACE, ConfigRuntime, FileConfig
from assistente.console import ConsoleUI
from assistente.file_service import WorkspaceFileService
from assistente.session import SessioneAgente
from tests.conftest import DIR_E2E_REL

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(300)]

_TESTO_SORGENTE = """
Il linguaggio Python è ampiamente usato per automazione, data science e backend web.
È interpretato, tipizzato dinamicamente e dispone di una vasta libreria standard.
Il progetto coding_gemma4 implementa un assistente di programmazione locale basato su
LangChain ReAct e modelli Gemma 4 eseguiti tramite Ollama sul computer dell'utente.
I tool dell'agente permettono di scrivere e leggere file nel workspace, elencare
cartelle e cercare parole nel contenuto dei documenti della cartella.
Il tool riassumi_documento produce un riassunto dedicato via Ollama.
Questo paragrafo aggiuntivo serve ad aumentare la lunghezza del documento sorgente così
che un riassunto corretto sia più corto dell'originale e non una copia integrale.
In ambienti di sviluppo Apple Silicon si consiglia di abilitare flash attention e
quantizzazione KV cache per ridurre latenza e consumo memoria durante le sessioni lunghe.
Le iterazioni ReAct sono limitate: l'agente deve scegliere Action e Action Input corretti
senza ripetere inutilmente la stessa scrittura file.
""".strip()

_MIN_CARATTERI_RIASSUNTO = 40
# Marker fisso per assert exact sul body .txt dopo sanitizzazione (newline finale).
_PROBE_TXT_BODY = "probe-ok-2026"
_PROBE_TXT_ON_DISK = f"{_PROBE_TXT_BODY}\n"
# Fatto inventato: il modello può conoscerlo solo leggendo il documento.
_FATTO_CORPUS = (
    "Nel registro dell'osservatorio, la balena chiamata Zeroglut ha percorso "
    "4.217 chilometri durante la migrazione del 2025.\n"
)


def _workspace_text(path_rel: str) -> str | None:
    """
    Legge UTF-8 da WORKSPACE/path_rel, oppure None se il file manca.

    Args:
        path_rel: Path relativo al workspace.

    Returns:
        Contenuto del file oppure None.
    """
    destinazione = WORKSPACE / path_rel
    # Evita FileNotFoundError nei messaggi di assert contestuali.
    if not destinazione.is_file():
        return None
    return destinazione.read_text(encoding="utf-8")


def _raise_with_context(
    errore: BaseException,
    *,
    risposta: str,
    path_scritti: list[str],
    path_rel: str,
) -> None:
    """
    Rilancia AssertionError arricchita con risposta, path_scritti e contenuto disco.

    Args:
        errore: Eccezione originale (AssertionError o SyntaxError).
        risposta: Final Answer dell'agente.
        path_scritti: Path materializzati restituiti da esegui_richiesta.
        path_rel: Path relativo sotto esame.
    """
    contenuto = _workspace_text(path_rel)
    raise AssertionError(
        f"{errore}; risposta={risposta!r} path_scritti={path_scritti!r} "
        f"contenuto={contenuto!r}"
    ) from errore


def _prompt_scrivi_file(path_rel: str, body: str) -> str:
    """
    Costruisce un prompt ReAct vincolato al contratto path / --- / body.

    Args:
        path_rel: Destinazione relativa nel workspace.
        body: Contenuto esatto da scrivere (senza fence markdown).

    Returns:
        Prompt utente per esegui_richiesta.
    """
    return (
        f"Crea il file {path_rel}. "
        "Usa una sola volta scrivi_file con Action Input a tre righe: "
        f"path, poi '{SEPARATORE_FILE}', poi solo questo body esatto:\n"
        f"{body}\n"
        "Niente markdown, niente testo extra nel body."
    )


def _assert_workspace_text(path_rel: str, expected: str) -> None:
    """
    Verifica che path_rel esista con contenuto esatto (inclusa newline finale).

    Args:
        path_rel: Path relativo atteso su disco.
        expected: Byte-stringa UTF-8 attesa dopo sanitizzazione tool.
    """
    destinazione = WORKSPACE / path_rel
    assert destinazione.is_file(), f"File assente: {path_rel}"
    assert destinazione.read_text(encoding="utf-8") == expected


def _assert_riassunto_file(path_rel: str, sorgente: str | None = None) -> str:
    """
    Verifica che path_rel esista e contenga un riassunto non banale.

    Output: contenuto strip del file.
    """
    destinazione = WORKSPACE / path_rel
    assert destinazione.is_file(), f"File assente: {path_rel}"
    contenuto = destinazione.read_text(encoding="utf-8").strip()
    assert len(contenuto) >= _MIN_CARATTERI_RIASSUNTO, contenuto
    if sorgente is not None:
        assert contenuto != sorgente
        assert len(contenuto) < len(sorgente)
    return contenuto


def test_e2e_riassunto_file_su_txt(
    workspace_test_dir: Path,
    sessione_agente: SessioneAgente,
    config_e2e: ConfigRuntime,
    richiede_ollama,
) -> None:
    """Agente: riassumi_documento sorgente --- destinazione → file su disco."""
    path_sorgente = f"{DIR_E2E_REL}/sorgente.txt"
    path_riassunto = f"{DIR_E2E_REL}/riassunto.txt"
    (WORKSPACE / path_sorgente).write_text(_TESTO_SORGENTE, encoding="utf-8")

    prompt = (
        f"Usa una sola volta riassumi_documento con Action Input a tre righe: "
        f"'{path_sorgente}', poi '---', poi '{path_riassunto}'. "
        "Non usare scrivi_file."
    )
    risposta, path_scritti = sessione_agente.esegui_richiesta(prompt, config_e2e)
    try:
        _assert_riassunto_file(path_riassunto, sorgente=_TESTO_SORGENTE)
    except AssertionError as errore:
        _raise_with_context(
            errore,
            risposta=risposta,
            path_scritti=path_scritti,
            path_rel=path_riassunto,
        )


def test_e2e_scrivi_file_txt(
    workspace_test_dir: Path,
    sessione_agente: SessioneAgente,
    config_e2e: ConfigRuntime,
    richiede_ollama,
) -> None:
    """Agente ReAct: scrivi_file crea un .txt con body esatto su disco."""
    path_rel = f"{DIR_E2E_REL}/probe.txt"
    risposta, path_scritti = sessione_agente.esegui_richiesta(
        _prompt_scrivi_file(path_rel, _PROBE_TXT_BODY),
        config_e2e,
    )
    try:
        _assert_workspace_text(path_rel, _PROBE_TXT_ON_DISK)
        assert path_rel in path_scritti, path_scritti
    except AssertionError as errore:
        _raise_with_context(
            errore,
            risposta=risposta,
            path_scritti=path_scritti,
            path_rel=path_rel,
        )


def test_e2e_crea_cartella_e_diario_txt(
    workspace_test_dir: Path,
    sessione_agente: SessioneAgente,
    config_e2e: ConfigRuntime,
    richiede_ollama,
) -> None:
    """Agente: crea_cartella + scrivi_file su un diario di testo."""
    cartella = f"{DIR_E2E_REL}/e2e_notes"
    path_rel = f"{cartella}/diario.txt"
    marker = "probe-diario-2026"
    prompt = (
        f"Crea la cartella {cartella} e al suo interno il file {path_rel}. "
        "Usa una sola volta scrivi_file con Action Input a tre righe: "
        f"path, poi '{SEPARATORE_FILE}', poi solo questo body esatto:\n"
        f"{marker}\n"
        "Niente markdown, niente testo extra nel body."
    )
    risposta, path_scritti = sessione_agente.esegui_richiesta(prompt, config_e2e)
    try:
        assert (WORKSPACE / cartella).is_dir()
        _assert_workspace_text(path_rel, f"{marker}\n")
        assert path_rel in path_scritti or cartella in path_scritti, path_scritti
    except AssertionError as errore:
        _raise_with_context(
            errore,
            risposta=risposta,
            path_scritti=path_scritti,
            path_rel=path_rel,
        )


@pytest.fixture
def sessione_corpus(
    workspace_test_dir: Path,
    config_e2e: ConfigRuntime,
) -> tuple[SessioneAgente, Path]:
    """Sessione confinata a una cartella con un solo documento noto."""
    corpus = workspace_test_dir / "e2e" / "balena"
    corpus.mkdir(parents=True)
    (corpus / "registro.txt").write_text(_FATTO_CORPUS, encoding="utf-8")
    ui = ConsoleUI()
    ui.imposta_verbose(config_e2e.verbose)
    service = WorkspaceFileService(FileConfig(workspace=corpus))
    return SessioneAgente(ui, file_service=service), corpus


def test_e2e_risponde_dal_documento(
    sessione_corpus: tuple[SessioneAgente, Path],
    config_e2e: ConfigRuntime,
    richiede_ollama,
) -> None:
    """Un fatto presente solo nel documento arriva nella risposta, con il file."""
    sessione, _corpus = sessione_corpus
    risposta, _scritti = sessione.esegui_richiesta(
        "Quanti chilometri ha percorso la balena Zeroglut?",
        config_e2e,
    )
    assert "4.217" in risposta or "4217" in risposta, risposta
    assert "registro" in risposta.casefold(), risposta


def test_e2e_rifiuta_scrittura_non_txt(
    sessione_corpus: tuple[SessioneAgente, Path],
    config_e2e: ConfigRuntime,
    richiede_ollama,
) -> None:
    """Una destinazione .md non viene mai creata: si scrive solo .txt."""
    sessione, corpus = sessione_corpus
    sessione.esegui_richiesta(
        "Scrivi il file note.md con il testo: prova.",
        config_e2e,
    )
    assert not (corpus / "note.md").exists()
