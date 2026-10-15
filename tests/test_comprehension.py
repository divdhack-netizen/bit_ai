"""Structural paths and prompt: the COMPITO is no longer built from the wording."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.callbacks import BaseCallbackHandler

from assistant.comprehension import (
    TurnComprehension,
    join_path_in_folder,
    extract_named_folder,
    format_comprehension_block,
    resolve_destination,
)
from assistant.direct_pipeline import DIRECT_SYSTEM_PREFIX, DirectAnswerPipeline
from assistant.turn_policy import decision_from_schema
from assistant.turn_schema import TurnSchema
from assistant.session import _comprehension_from_schema, _build_task_description

pytestmark = pytest.mark.unit


def test_folder_and_path_compose_without_verbs() -> None:
    """The folder name is structure, not a write intent."""
    text = (
        "nella cartella Documents crea una cartella dal nome AI. "
        "Al suo interno il file problems.txt"
    )
    assert extract_named_folder(text) == "Documents/AI"
    assert join_path_in_folder("problems.txt", "Documents/AI") == (
        "Documents/AI/problems.txt"
    )
    assert join_path_in_folder("notes/a.txt", "notes") == "notes/a.txt"
    assert join_path_in_folder("a.txt", None) == "a.txt"
    assert resolve_destination(
        "problems.txt",
        None,
        "Documents/AI/problems.txt",
    ) == "Documents/AI/problems.txt"


def test_schema_subject_becomes_theme_not_script() -> None:
    """The schema theme goes into the prompt; there is no scrivi_file instruction."""
    schema = TurnSchema(kind="workspace", subject="gioco", folder="gioco")
    comprehension = _comprehension_from_schema(schema)
    block = format_comprehension_block(comprehension)
    assert block == "COMPITO: gioco"
    assert "scrivi_file" not in block
    task = _build_task_description(
        "prepara in questa cartella un piccolo gioco",
        full_verification=False,
        decision=decision_from_schema(schema),
        comprehension=comprehension,
    )
    assert task.index("COMPITO:") < task.index("RICHIESTA:")


def test_schema_without_subject_does_not_inject_task() -> None:
    """Without a subject the prompt does not invent a COMPITO."""
    comprehension = _comprehension_from_schema(TurnSchema(kind="explain"))
    assert format_comprehension_block(comprehension) == ""
    task = _build_task_description(
        "ciao",
        full_verification=False,
        decision=decision_from_schema(TurnSchema(kind="chat")),
        comprehension=comprehension,
    )
    assert "COMPITO:" not in task


def test_direct_kv_prefix_stays_stable() -> None:
    """DIRECT_SYSTEM_PREFIX opens the prompt; an explicit theme comes after the history."""
    ui = MagicMock()
    llm = MagicMock()
    llm.last_prompt = ""

    def stream(input: str, config=None, **kwargs):
        del config, kwargs
        llm.last_prompt = input
        yield "ok"

    llm.stream = stream
    pipeline = DirectAnswerPipeline(
        llm_provider=lambda *_args: llm,
        ui=ui,
    )
    comprehension = TurnComprehension(task="storico della conversazione", paths=())
    pipeline.answer(
        request="riassumi",
        model_alias="default",
        history="Utente: prima\nAssistente: dopo",
        context_tokens=4096,
        output_tokens=64,
        callback=MagicMock(spec=BaseCallbackHandler),
        intent_context=format_comprehension_block(comprehension),
    )
    prompt = llm.last_prompt
    assert prompt.startswith(DIRECT_SYSTEM_PREFIX)
    assert prompt.index("STORICO CONVERSAZIONE") < prompt.index("COMPITO:")
    assert "COMPITO:" not in prompt[: len(DIRECT_SYSTEM_PREFIX)]
