"""E2E tests with Ollama (default model): summary, scrivi_file and answers from the documents."""

from __future__ import annotations

from pathlib import Path

import pytest

from assistant.config import FILE_SEPARATOR, WORKSPACE, ConfigRuntime, FileConfig
from assistant.console import ConsoleUI
from assistant.file_service import WorkspaceFileService
from assistant.session import AgentSession
from tests.conftest import DIR_E2E_REL

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(300)]

_SOURCE_TEXT = """
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

_MIN_SUMMARY_CHARS = 40
# Fixed marker for an exact assert on the .txt body after sanitization (trailing newline).
_PROBE_TXT_BODY = "probe-ok-2026"
_PROBE_TXT_ON_DISK = f"{_PROBE_TXT_BODY}\n"
# Invented fact: the model can only know it by reading the document.
_CORPUS_FACT = (
    "Nel registro dell'osservatorio, la balena chiamata Zeroglut ha percorso "
    "4.217 chilometri durante la migrazione del 2025.\n"
)


def _workspace_text(path_rel: str) -> str | None:
    """
    Read UTF-8 from WORKSPACE/path_rel, or None if the file is missing.

    Args:
        path_rel: Path relative to the workspace.

    Returns:
        The file content, or None.
    """
    destination = WORKSPACE / path_rel
    # Avoid FileNotFoundError in the contextual assert messages.
    if not destination.is_file():
        return None
    return destination.read_text(encoding="utf-8")


def _raise_with_context(
    error: BaseException,
    *,
    answer: str,
    written_paths: list[str],
    path_rel: str,
) -> None:
    """
    Re-raise an AssertionError enriched with the answer, written_paths and disk content.

    Args:
        error: Original exception (AssertionError or SyntaxError).
        answer: The agent's Final Answer.
        written_paths: Materialized paths returned by run_request.
        path_rel: Relative path under test.
    """
    content = _workspace_text(path_rel)
    raise AssertionError(
        f"{error}; risposta={answer!r} path_scritti={written_paths!r} "
        f"contenuto={content!r}"
    ) from error


def _write_file_prompt(path_rel: str, body: str) -> str:
    """
    Build a ReAct prompt bound to the path / --- / body contract.

    Args:
        path_rel: Relative destination in the workspace.
        body: Exact content to write (no markdown fence).

    Returns:
        User prompt for run_request.
    """
    return (
        f"Crea il file {path_rel}. "
        "Usa una sola volta scrivi_file con Action Input a tre righe: "
        f"path, poi '{FILE_SEPARATOR}', poi solo questo body esatto:\n"
        f"{body}\n"
        "Niente markdown, niente testo extra nel body."
    )


def _assert_workspace_text(path_rel: str, expected: str) -> None:
    """
    Check that path_rel exists with the exact content (trailing newline included).

    Args:
        path_rel: Relative path expected on disk.
        expected: Expected UTF-8 string after tool sanitization.
    """
    destination = WORKSPACE / path_rel
    assert destination.is_file(), f"File assente: {path_rel}"
    assert destination.read_text(encoding="utf-8") == expected


def _assert_summary_file(path_rel: str, source: str | None = None) -> str:
    """
    Check that path_rel exists and holds a non-trivial summary.

    Output: the stripped file content.
    """
    destination = WORKSPACE / path_rel
    assert destination.is_file(), f"File assente: {path_rel}"
    content = destination.read_text(encoding="utf-8").strip()
    assert len(content) >= _MIN_SUMMARY_CHARS, content
    if source is not None:
        assert content != source
        assert len(content) < len(source)
    return content


def test_e2e_file_summary_to_txt(
    workspace_test_dir: Path,
    agent_session: AgentSession,
    config_e2e: ConfigRuntime,
    requires_ollama,
) -> None:
    """Agent: riassumi_documento source --- destination → file on disk."""
    source_path = f"{DIR_E2E_REL}/sorgente.txt"
    summary_path = f"{DIR_E2E_REL}/riassunto.txt"
    (WORKSPACE / source_path).write_text(_SOURCE_TEXT, encoding="utf-8")

    prompt = (
        f"Usa una sola volta riassumi_documento con Action Input a tre righe: "
        f"'{source_path}', poi '---', poi '{summary_path}'. "
        "Non usare scrivi_file."
    )
    answer, written_paths = agent_session.run_request(prompt, config_e2e)
    try:
        _assert_summary_file(summary_path, source=_SOURCE_TEXT)
    except AssertionError as error:
        _raise_with_context(
            error,
            answer=answer,
            written_paths=written_paths,
            path_rel=summary_path,
        )


def test_e2e_write_txt_file(
    workspace_test_dir: Path,
    agent_session: AgentSession,
    config_e2e: ConfigRuntime,
    requires_ollama,
) -> None:
    """ReAct agent: scrivi_file creates a .txt with the exact body on disk."""
    path_rel = f"{DIR_E2E_REL}/probe.txt"
    answer, written_paths = agent_session.run_request(
        _write_file_prompt(path_rel, _PROBE_TXT_BODY),
        config_e2e,
    )
    try:
        _assert_workspace_text(path_rel, _PROBE_TXT_ON_DISK)
        assert path_rel in written_paths, written_paths
    except AssertionError as error:
        _raise_with_context(
            error,
            answer=answer,
            written_paths=written_paths,
            path_rel=path_rel,
        )


def test_e2e_create_folder_and_journal_txt(
    workspace_test_dir: Path,
    agent_session: AgentSession,
    config_e2e: ConfigRuntime,
    requires_ollama,
) -> None:
    """Agent: crea_cartella + scrivi_file on a text journal."""
    folder = f"{DIR_E2E_REL}/e2e_notes"
    path_rel = f"{folder}/diario.txt"
    marker = "probe-diario-2026"
    prompt = (
        f"Crea la cartella {folder} e al suo interno il file {path_rel}. "
        "Usa una sola volta scrivi_file con Action Input a tre righe: "
        f"path, poi '{FILE_SEPARATOR}', poi solo questo body esatto:\n"
        f"{marker}\n"
        "Niente markdown, niente testo extra nel body."
    )
    answer, written_paths = agent_session.run_request(prompt, config_e2e)
    try:
        assert (WORKSPACE / folder).is_dir()
        _assert_workspace_text(path_rel, f"{marker}\n")
        assert path_rel in written_paths or folder in written_paths, written_paths
    except AssertionError as error:
        _raise_with_context(
            error,
            answer=answer,
            written_paths=written_paths,
            path_rel=path_rel,
        )


@pytest.fixture
def corpus_session(
    workspace_test_dir: Path,
    config_e2e: ConfigRuntime,
) -> tuple[AgentSession, Path]:
    """Session confined to a folder with a single known document."""
    corpus = workspace_test_dir / "e2e" / "balena"
    corpus.mkdir(parents=True)
    (corpus / "registro.txt").write_text(_CORPUS_FACT, encoding="utf-8")
    ui = ConsoleUI()
    ui.set_verbose(config_e2e.verbose)
    service = WorkspaceFileService(FileConfig(workspace=corpus))
    return AgentSession(ui, file_service=service), corpus


def test_e2e_answers_from_document(
    corpus_session: tuple[AgentSession, Path],
    config_e2e: ConfigRuntime,
    requires_ollama,
) -> None:
    """A fact present only in the document reaches the answer, with the file."""
    session, _corpus = corpus_session
    answer, _written = session.run_request(
        "Quanti chilometri ha percorso la balena Zeroglut?",
        config_e2e,
    )
    assert "4.217" in answer or "4217" in answer, answer
    assert "registro" in answer.casefold(), answer


def test_e2e_rejects_non_txt_write(
    corpus_session: tuple[AgentSession, Path],
    config_e2e: ConfigRuntime,
    requires_ollama,
) -> None:
    """A .md destination is never created: only .txt is written."""
    session, corpus = corpus_session
    session.run_request(
        "Scrivi il file note.md con il testo: prova.",
        config_e2e,
    )
    assert not (corpus / "note.md").exists()
