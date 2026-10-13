"""Test unitari sul tool riassumi_documento (LLM mockato)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from assistente.config import SEPARATORE_FILE, WORKSPACE, risolvi_tag_ollama
from assistente.ollama_runtime import OllamaRuntime
from assistente.riassunto import _llm_riassunto, riassumi_documento_da_testo
from assistente.tools import (
    get_default_file_service,
    reset_stato_turno,
    scrivi_file_da_testo,
)
from tests.conftest import DIR_UNIT_REL

pytestmark = pytest.mark.unit

_CONTENUTO = (
    "Python è usato per automazione e data science. "
    "L'agente locale usa LangChain e Ollama con Gemma 4."
)
_RIASSUNTO_MOCK = "Python e agente LangChain/Ollama con Gemma 4."


def _scrivi_unit(nome: str, contenuto: str = _CONTENUTO) -> str:
    """Scrive un file sotto tmp/test_run/unit e restituisce il path relativo."""
    path_rel = f"{DIR_UNIT_REL}/{nome}"
    scrivi_file_da_testo(f"{path_rel}\n{SEPARATORE_FILE}\n{contenuto}")
    return path_rel


def _llm_mock(valore: str = _RIASSUNTO_MOCK) -> MagicMock:
    """OllamaLLM fake con invoke che restituisce valore."""
    llm = MagicMock()
    llm.invoke.return_value = valore
    return llm


def test_riassumi_documento_path_valido(workspace_test_dir) -> None:
    """Path valido → riassunto dall'LLM mock; prompt include il contenuto."""
    reset_stato_turno()
    path_rel = _scrivi_unit("da_riassumere.txt")
    llm = _llm_mock()
    with patch("assistente.riassunto._llm_riassunto", return_value=llm):
        esito = riassumi_documento_da_testo(path_rel)
    assert esito == _RIASSUNTO_MOCK
    assert "Python" in llm.invoke.call_args.args[0]


def test_riassumi_documento_path_vietato(workspace_test_dir) -> None:
    """Traversal → Errore: senza invocare l'LLM."""
    reset_stato_turno()
    with patch("assistente.riassunto._llm_riassunto") as llm_fn:
        esito = riassumi_documento_da_testo("../README.md")
    assert esito.startswith("Errore:"), esito
    llm_fn.assert_not_called()


def test_riassumi_documento_file_assente(workspace_test_dir) -> None:
    """File assente → Errore: senza invocare l'LLM."""
    reset_stato_turno()
    path_rel = f"{DIR_UNIT_REL}/inesistente.txt"
    assert not (WORKSPACE / path_rel).exists()
    with patch("assistente.riassunto._llm_riassunto") as llm_fn:
        esito = riassumi_documento_da_testo(path_rel)
    assert esito.startswith("Errore:"), esito
    llm_fn.assert_not_called()


def test_riassumi_documento_fallback_estrattivo_se_llm_vuoto(workspace_test_dir) -> None:
    """Risposta LLM vuota → fallback estrattivo con contenuto utile."""
    reset_stato_turno()
    path_rel = _scrivi_unit("fallback.txt")
    with patch("assistente.riassunto._llm_riassunto", return_value=_llm_mock("")):
        esito = riassumi_documento_da_testo(path_rel)
    assert not esito.startswith("Errore:"), esito
    assert "Python" in esito


def test_riassumi_documento_salva_su_destinazione(workspace_test_dir) -> None:
    """Forma sorgente / --- / destinazione persiste il riassunto su disco."""
    reset_stato_turno()
    path_in = _scrivi_unit("sorgente_save.txt")
    path_out = f"{DIR_UNIT_REL}/out_riassunto.txt"
    with patch("assistente.riassunto._llm_riassunto", return_value=_llm_mock()):
        esito = riassumi_documento_da_testo(f"{path_in}\n{SEPARATORE_FILE}\n{path_out}")
    assert "Riassunto salvato in" in esito
    assert "ripeti Action" not in esito
    assert _RIASSUNTO_MOCK in esito
    assert (WORKSPACE / path_out).read_text(encoding="utf-8").startswith(_RIASSUNTO_MOCK)
    # Il tracking del servizio include anche scritture indirette via riassumi_documento.
    assert path_out in get_default_file_service().written_paths
    # Una seconda chiamata identica nello stesso turno non rigenera il riassunto.
    llm = _llm_mock()
    with patch("assistente.riassunto._llm_riassunto", return_value=llm):
        ripetuto = riassumi_documento_da_testo(f"{path_in}\n{SEPARATORE_FILE}\n{path_out}")
    assert "già salvato" in ripetuto
    llm.invoke.assert_not_called()


def test_riassunto_ignora_un_secondo_blocco_react_catturato(
    workspace_test_dir,
) -> None:
    """Il primo Action Input resta valido anche se Gemma accoda marker di un'altra azione."""
    reset_stato_turno()
    path_in = _scrivi_unit("sorgente_react.txt")
    path_out = f"{DIR_UNIT_REL}/out_react.txt"
    action_input = (
        f"'{path_in}\\n---\\n{path_out}'"
        "<channel|>Action: riassumi_documento\n"
        "Action Input: input duplicato"
    )

    with patch("assistente.riassunto._llm_riassunto", return_value=_llm_mock()):
        result = riassumi_documento_da_testo(action_input)

    assert result.startswith("Riassunto salvato in")
    assert (WORKSPACE / path_out).is_file()


def test_llm_riassunto_default_usa_tag_ollama_corretto(monkeypatch: pytest.MonkeyPatch) -> None:
    """_llm_riassunto() costruisce OllamaLLM con il tag q4 di default."""
    import assistente.riassunto as modulo_riassunto

    catturato: dict[str, object] = {}

    class _FakeOllamaLLM:
        def __init__(self, **kwargs: object) -> None:
            catturato.update(kwargs)

    runtime = OllamaRuntime(llm_factory=_FakeOllamaLLM, manage_residency=False)
    monkeypatch.setattr(modulo_riassunto, "_runtime_riassunto", lambda: runtime)
    _llm_riassunto("default")
    assert catturato["model"] == risolvi_tag_ollama("default")
    # Il fallback legacy usa il profilo summary, non un secondo modello hardcoded E4B-only.
    assert catturato["num_ctx"] == 6144
    assert catturato["num_predict"] == 512
