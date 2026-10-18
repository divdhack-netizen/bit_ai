"""Unit tests for tool_contract, evaluate_turn, permissions and microcompact."""

from __future__ import annotations

from pathlib import Path

import pytest

from assistant.evaluate_turn import evaluate_turn
from assistant.file_service import WorkspaceFileService
from assistant.permissions import WritePermissionRules, load_write_permission_rules
from assistant.prompt_budget import microcompact_observations
from assistant.tool_contract import (
    ToolContract,
    get_tool_contract,
    validate_non_empty,
    wrap_tool_func,
)
from assistant.tools import create_file_tools
from assistant.config import FileConfig

pytestmark = pytest.mark.unit


def test_wrap_tool_func_validation_error() -> None:
    """Empty input on validate_non_empty → error Observation, no side effect."""
    calls: list[str] = []

    def impl(action_input: str) -> str:
        calls.append(action_input)
        return "ok"

    contract = ToolContract(
        name="demo",
        is_read_only=True,
        is_concurrency_safe=True,
        validate=validate_non_empty,
    )
    guarded = wrap_tool_func(contract, impl)
    result = guarded("   ")
    assert result.startswith("Errore validazione demo:")
    assert calls == []


def test_create_file_tools_annotates_contracts(tmp_path: Path) -> None:
    """The file tools expose tool_contract with the correct read-only flags."""
    service = WorkspaceFileService(
        FileConfig(workspace=tmp_path),
        permission_rules=WritePermissionRules(),
    )
    tools = create_file_tools(service, write_allowed=True, workspace_label="ws")
    by_name = {tool.name: tool for tool in tools}
    write_contract = get_tool_contract(by_name["scrivi_file"])
    modify_contract = get_tool_contract(by_name["modifica_file"])
    read_contract = get_tool_contract(by_name["leggi_documento"])
    assert write_contract is not None
    assert modify_contract is not None
    assert read_contract is not None
    assert write_contract.is_read_only is False
    assert modify_contract.is_read_only is False
    assert read_contract.is_read_only is True
    assert "Scrittura autorizzata" in by_name["scrivi_file"].description
    assert "Solo estensione .txt" in by_name["scrivi_file"].description
    assert "crea_cartella" in by_name
    assert "accoda_file" not in by_name
    assert "sposta_path" not in by_name
    assert get_tool_contract(by_name["crea_cartella"]).is_read_only is False
    assert get_tool_contract(by_name["elenca_albero"]).is_read_only is True


def test_create_file_tools_omits_write_if_gate_closed(tmp_path: Path) -> None:
    """Gate False: no scrivi_file/modifica_file in the ReAct prompt."""
    service = WorkspaceFileService(
        FileConfig(workspace=tmp_path),
        permission_rules=WritePermissionRules(),
    )
    tools = create_file_tools(service, write_allowed=False, workspace_label="ws")
    names = {tool.name for tool in tools}
    assert names == {
        "leggi_documento",
        "elenca_cartella",
        "elenca_albero",
        "trova_file",
        "cerca_testo",
    }


def test_evaluate_turn_missing_write() -> None:
    """Write requested without a path → incomplete with an explicit note."""
    outcome = evaluate_turn(
        "Ho creato hello.py",
        write_requested=True,
        written_paths=[],
    )
    assert outcome.incomplete is True
    assert "nessun file" in outcome.answer.casefold() or "persistito" in outcome.answer.casefold()


def test_evaluate_turn_ok() -> None:
    """Chat turn without a write → complete."""
    outcome = evaluate_turn(
        "Una closure cattura variabili",
        write_requested=False,
        written_paths=[],
    )
    assert outcome.incomplete is False
    assert outcome.answer == "Una closure cattura variabili"


def test_evaluate_turn_without_intent_does_not_report_write() -> None:
    """The ReAct route enables the tools but does not imply a wanted write."""
    outcome = evaluate_turn(
        "Agent stopped due to iteration limit",
        write_requested=False,
        written_paths=[],
        max_iterations_hit=True,
    )
    assert outcome.incomplete is True
    assert all("Scrittura richiesta" not in note for note in outcome.note)
    assert any("limite iterazioni" in note for note in outcome.note)


def test_evaluate_turn_mkdir_is_not_missing_write() -> None:
    """A persisted folder closes the write gate."""
    outcome = evaluate_turn(
        "Ho creato notes",
        write_requested=True,
        written_paths=["notes"],
    )
    assert outcome.incomplete is False


def test_permission_deny_blocks_write(tmp_path: Path) -> None:
    """A deny glob on secrets.* rejects the write even with the gate open."""
    service = WorkspaceFileService(
        FileConfig(workspace=tmp_path),
        permission_rules=WritePermissionRules(deny=("secrets.*",)),
    )
    service.begin_turn(True)
    result = service.write_text("secrets.txt", "TOKEN=1\n")
    assert result.succeeded is False
    assert result.error is not None
    assert "permesso" in result.error.casefold() or "deny" in result.error.casefold()


def test_load_write_permission_rules_missing(tmp_path: Path) -> None:
    """Missing file → empty rules (allow-all on top of the sandbox)."""
    rules = load_write_permission_rules(tmp_path / "missing.json")
    assert rules.allow == ()
    assert rules.deny == ()
    assert rules.is_write_allowed("tmp/a.txt") == (True, None)


def test_file_read_cache_hit(tmp_path: Path) -> None:
    """A second read of the same file reuses the session cache."""
    service = WorkspaceFileService(
        FileConfig(workspace=tmp_path),
        permission_rules=WritePermissionRules(),
    )
    service.begin_turn(True)
    service.write_text("note.txt", "hello cache")
    first = service.read_document("note.txt")
    second = service.read_document("note.txt")
    assert first.succeeded and second.succeeded
    assert first.content == second.content == "hello cache"
    # After a write, the path's cache is invalidated.
    service.begin_turn(True)
    service.write_text("note.txt", "updated\n")
    third = service.read_document("note.txt")
    assert third.content == "updated\n"


def test_permission_deny_covers_root_dotfile_and_root_glob() -> None:
    """lstrip('./') used to exempt dotfiles and **/X did not cover the root."""
    rules = WritePermissionRules(
        deny=(".env", ".env.*", "**/.env", "**/credentials.json", "**/secrets.*"),
    )
    for path in (".env", ".env.local", "credentials.json", "secrets.yaml", "sub/.env"):
        allowed, reason = rules.is_write_allowed(path)
        assert allowed is False, path
        assert reason is not None and "negato" in reason
    # Normal paths stay free with an empty allow list.
    assert rules.is_write_allowed("notes/a.md") == (True, None)


def test_permission_is_read_allowed_uses_only_deny() -> None:
    """Deny also blocks reads; allow restricts writes, it does not widen reads."""
    rules = WritePermissionRules(allow=("pub/*",), deny=("secrets.*",))
    assert rules.is_read_allowed("secrets.env")[0] is False
    # Outside the allow list but readable: allow is not a read whitelist.
    assert rules.is_read_allowed("altro/file.md") == (True, None)
    assert rules.is_read_allowed("pub/x.md") == (True, None)


def test_permission_deny_folder_covers_children() -> None:
    """Denying the folder denies its content too, without a «**» glob."""
    rules = WritePermissionRules(deny=("privato",))
    assert rules.is_read_allowed("privato/chiavi.txt")[0] is False
    assert rules.is_write_allowed("privato/chiavi.txt")[0] is False
    assert rules.is_read_allowed("pubblico/nota.md") == (True, None)


def test_permission_deny_blocks_read(tmp_path: Path) -> None:
    """The denied secret is unreadable even if it exists on disk (R5)."""
    (tmp_path / "secrets.env").write_text("TOKEN=1\n", encoding="utf-8")
    (tmp_path / "note.txt").write_text("ciao\n", encoding="utf-8")
    service = WorkspaceFileService(
        FileConfig(workspace=tmp_path),
        permission_rules=WritePermissionRules(deny=("secrets.*",)),
    )
    denied = service.read_document("secrets.env")
    assert denied.succeeded is False
    assert denied.content == ""
    assert "Lettura non consentita" in str(denied.error)
    # Clean paths stay readable.
    assert service.read_document("note.txt").succeeded is True


def test_microcompact_omits_old_observations() -> None:
    """Old read-only Observations become placeholders; errors and writes stay."""
    observations = [
        ("leggi_documento", "a.py", "contenuto lungo " * 50),
        ("elenca_cartella", ".", "file1\nfile2"),
        ("scrivi_file", "b.py", "File scritto con successo: b.py"),
        ("modifica_file", "b.py", "File modificato con successo: b.py"),
        ("leggi_documento", "b.py", "nuovo contenuto"),
    ]
    compacted = microcompact_observations(observations, keep_recent=1, force=True)
    assert "omessa" in compacted[0][2]
    assert "File scritto" in compacted[2][2]
    assert "File modificato" in compacted[3][2]
    # The last one stays intact (keep_recent=1).
    assert compacted[-1][2] == "nuovo contenuto"
