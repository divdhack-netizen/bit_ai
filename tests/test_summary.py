"""Unit tests for the riassumi_documento tool (mocked LLM)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from assistant.config import FILE_SEPARATOR, WORKSPACE, resolve_ollama_tag
from assistant.ollama_runtime import OllamaRuntime
from assistant.summary import _summary_llm, summarize_document_from_text
from assistant.tools import (
    get_default_file_service,
    reset_turn_state,
    write_file_from_text,
)
from tests.conftest import DIR_UNIT_REL

pytestmark = pytest.mark.unit

_CONTENT = (
    "Python è usato per automazione e data science. "
    "L'agente locale usa LangChain e Ollama con Gemma 4."
)
_MOCK_SUMMARY = "Python e agente LangChain/Ollama con Gemma 4."


def _write_unit(name: str, content: str = _CONTENT) -> str:
    """Write a file under tmp/test_run/unit and return its relative path."""
    path_rel = f"{DIR_UNIT_REL}/{name}"
    write_file_from_text(f"{path_rel}\n{FILE_SEPARATOR}\n{content}")
    return path_rel


def _llm_mock(value: str = _MOCK_SUMMARY) -> MagicMock:
    """Fake OllamaLLM whose invoke returns a value."""
    llm = MagicMock()
    llm.invoke.return_value = value
    return llm


def test_summarize_document_valid_path(workspace_test_dir) -> None:
    """Valid path → summary from the mock LLM; the prompt includes the content."""
    reset_turn_state()
    path_rel = _write_unit("da_riassumere.txt")
    llm = _llm_mock()
    with patch("assistant.summary._summary_llm", return_value=llm):
        outcome = summarize_document_from_text(path_rel)
    assert outcome == _MOCK_SUMMARY
    assert "Python" in llm.invoke.call_args.args[0]


def test_summarize_document_forbidden_path(workspace_test_dir) -> None:
    """Traversal → Errore: without invoking the LLM."""
    reset_turn_state()
    with patch("assistant.summary._summary_llm") as llm_fn:
        outcome = summarize_document_from_text("../README.md")
    assert outcome.startswith("Errore:"), outcome
    llm_fn.assert_not_called()


def test_summarize_document_missing_file(workspace_test_dir) -> None:
    """Missing file → Errore: without invoking the LLM."""
    reset_turn_state()
    path_rel = f"{DIR_UNIT_REL}/inesistente.txt"
    assert not (WORKSPACE / path_rel).exists()
    with patch("assistant.summary._summary_llm") as llm_fn:
        outcome = summarize_document_from_text(path_rel)
    assert outcome.startswith("Errore:"), outcome
    llm_fn.assert_not_called()


def test_summarize_document_extractive_fallback_if_llm_empty(workspace_test_dir) -> None:
    """Empty LLM answer → extractive fallback with useful content."""
    reset_turn_state()
    path_rel = _write_unit("fallback.txt")
    with patch("assistant.summary._summary_llm", return_value=_llm_mock("")):
        outcome = summarize_document_from_text(path_rel)
    assert not outcome.startswith("Errore:"), outcome
    assert "Python" in outcome


def test_summarize_document_saves_to_destination(workspace_test_dir) -> None:
    """The source / --- / destination form persists the summary to disk."""
    reset_turn_state()
    path_in = _write_unit("sorgente_save.txt")
    path_out = f"{DIR_UNIT_REL}/out_riassunto.txt"
    with patch("assistant.summary._summary_llm", return_value=_llm_mock()):
        outcome = summarize_document_from_text(f"{path_in}\n{FILE_SEPARATOR}\n{path_out}")
    assert "Riassunto salvato in" in outcome
    assert "ripeti Action" not in outcome
    assert _MOCK_SUMMARY in outcome
    assert (WORKSPACE / path_out).read_text(encoding="utf-8").startswith(_MOCK_SUMMARY)
    # The service tracking also includes indirect writes through riassumi_documento.
    assert path_out in get_default_file_service().written_paths
    # A second identical call in the same turn does not regenerate the summary.
    llm = _llm_mock()
    with patch("assistant.summary._summary_llm", return_value=llm):
        repeated = summarize_document_from_text(f"{path_in}\n{FILE_SEPARATOR}\n{path_out}")
    assert "già salvato" in repeated
    llm.invoke.assert_not_called()


def test_summary_ignores_second_captured_react_block(
    workspace_test_dir,
) -> None:
    """The first Action Input stays valid even if Gemma appends markers of another action."""
    reset_turn_state()
    path_in = _write_unit("sorgente_react.txt")
    path_out = f"{DIR_UNIT_REL}/out_react.txt"
    action_input = (
        f"'{path_in}\\n---\\n{path_out}'"
        "<channel|>Action: riassumi_documento\n"
        "Action Input: input duplicato"
    )

    with patch("assistant.summary._summary_llm", return_value=_llm_mock()):
        result = summarize_document_from_text(action_input)

    assert result.startswith("Riassunto salvato in")
    assert (WORKSPACE / path_out).is_file()


def test_default_summary_llm_uses_correct_ollama_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    """_summary_llm() builds OllamaLLM with the default q4 tag."""
    import assistant.summary as summary_module

    captured: dict[str, object] = {}

    class _FakeOllamaLLM:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    runtime = OllamaRuntime(llm_factory=_FakeOllamaLLM, manage_residency=False)
    monkeypatch.setattr(summary_module, "_summary_runtime", lambda: runtime)
    _summary_llm("default")
    assert captured["model"] == resolve_ollama_tag("default")
    # The legacy fallback uses the summary profile, not a second hardcoded E4B-only model.
    assert captured["num_ctx"] == 6144
    assert captured["num_predict"] == 512
