"""Test unitari per tool_contract, evaluate_turn, permissions e microcompact."""

from __future__ import annotations

from pathlib import Path

import pytest

from assistente.evaluate_turn import evaluate_turn
from assistente.file_service import WorkspaceFileService
from assistente.permissions import WritePermissionRules, load_write_permission_rules
from assistente.prompt_budget import microcompact_observations
from assistente.tool_contract import (
    ToolContract,
    get_tool_contract,
    validate_non_empty,
    wrap_tool_func,
)
from assistente.tools import crea_tool_file
from assistente.config import FileConfig

pytestmark = pytest.mark.unit


def test_wrap_tool_func_validation_error() -> None:
    """Input vuoto su validate_non_empty → Observation di errore, no side effect."""
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


def test_crea_tool_file_annota_contratti(tmp_path: Path) -> None:
    """I tool file espongono tool_contract con flag RO corretti."""
    service = WorkspaceFileService(
        FileConfig(workspace=tmp_path),
        permission_rules=WritePermissionRules(),
    )
    tools = crea_tool_file(service, write_allowed=True, workspace_label="ws")
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


def test_crea_tool_file_omette_write_se_gate_chiuso(tmp_path: Path) -> None:
    """Gate False: niente scrivi_file/modifica_file nel prompt ReAct."""
    service = WorkspaceFileService(
        FileConfig(workspace=tmp_path),
        permission_rules=WritePermissionRules(),
    )
    tools = crea_tool_file(service, write_allowed=False, workspace_label="ws")
    names = {tool.name for tool in tools}
    assert names == {
        "leggi_documento",
        "elenca_cartella",
        "elenca_albero",
        "trova_file",
        "cerca_testo",
    }


def test_evaluate_turn_scrittura_mancante() -> None:
    """Scrittura richiesta senza path → incompleto con nota esplicita."""
    esito = evaluate_turn(
        "Ho creato hello.py",
        scrittura_richiesta=True,
        path_scritti=[],
    )
    assert esito.incompleto is True
    assert "nessun file" in esito.risposta.casefold() or "persistito" in esito.risposta.casefold()


def test_evaluate_turn_ok() -> None:
    """Turno chat senza scrittura → completo."""
    esito = evaluate_turn(
        "Una closure cattura variabili",
        scrittura_richiesta=False,
        path_scritti=[],
    )
    assert esito.incompleto is False
    assert esito.risposta == "Una closure cattura variabili"


def test_evaluate_turn_senza_intento_non_segnala_la_scrittura() -> None:
    """La route ReAct autorizza i tool, ma non implica una scrittura voluta."""
    esito = evaluate_turn(
        "Agent stopped due to iteration limit",
        scrittura_richiesta=False,
        path_scritti=[],
        max_iterations_hit=True,
    )
    assert esito.incompleto is True
    assert all("Scrittura richiesta" not in nota for nota in esito.note)
    assert any("limite iterazioni" in nota for nota in esito.note)


def test_evaluate_turn_mkdir_non_e_scrittura_mancante() -> None:
    """Una cartella persistita chiude il gate scrittura."""
    esito = evaluate_turn(
        "Ho creato notes",
        scrittura_richiesta=True,
        path_scritti=["notes"],
    )
    assert esito.incompleto is False


def test_permission_deny_blocks_write(tmp_path: Path) -> None:
    """Deny glob su secrets.* rifiuta la scrittura anche con gate aperto."""
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
    """File assente → regole vuote (allow-all sopra sandbox)."""
    rules = load_write_permission_rules(tmp_path / "missing.json")
    assert rules.allow == ()
    assert rules.deny == ()
    assert rules.is_write_allowed("tmp/a.txt") == (True, None)


def test_file_read_cache_hit(tmp_path: Path) -> None:
    """Seconda lettura dello stesso file riusa la cache di sessione."""
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
    # Dopo write, la cache del path viene invalidata.
    service.begin_turn(True)
    service.write_text("note.txt", "updated\n")
    third = service.read_document("note.txt")
    assert third.content == "updated\n"


def test_permission_deny_copre_dotfile_root_e_glob_radice() -> None:
    """Prima lstrip('./') esentava i dotfile e **/X non copriva la radice."""
    rules = WritePermissionRules(
        deny=(".env", ".env.*", "**/.env", "**/credentials.json", "**/secrets.*"),
    )
    for path in (".env", ".env.local", "credentials.json", "secrets.yaml", "sub/.env"):
        allowed, reason = rules.is_write_allowed(path)
        assert allowed is False, path
        assert reason is not None and "negato" in reason
    # I path normali restano liberi con allow vuoto.
    assert rules.is_write_allowed("notes/a.md") == (True, None)


def test_permission_is_read_allowed_usa_solo_deny() -> None:
    """La deny blocca anche la lettura; allow restringe le scritture, non amplia le letture."""
    rules = WritePermissionRules(allow=("pub/*",), deny=("secrets.*",))
    assert rules.is_read_allowed("secrets.env")[0] is False
    # Fuori dall'allow ma leggibile: allow non è una whitelist di lettura.
    assert rules.is_read_allowed("altro/file.md") == (True, None)
    assert rules.is_read_allowed("pub/x.md") == (True, None)


def test_permission_deny_cartella_copre_i_figli() -> None:
    """Negare la cartella nega anche il contenuto, senza glob «**»."""
    rules = WritePermissionRules(deny=("privato",))
    assert rules.is_read_allowed("privato/chiavi.txt")[0] is False
    assert rules.is_write_allowed("privato/chiavi.txt")[0] is False
    assert rules.is_read_allowed("pubblico/nota.md") == (True, None)


def test_permission_deny_blocks_read(tmp_path: Path) -> None:
    """Il segreto negato è illeggibile anche se esiste sul disco (R5)."""
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
    # I path puliti restano leggibili.
    assert service.read_document("note.txt").succeeded is True


def test_microcompact_omette_observation_vecchie() -> None:
    """Observation RO vecchie diventano placeholder; errori e write restano."""
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
    # L'ultima resta intatta (keep_recent=1).
    assert compacted[-1][2] == "nuovo contenuto"
