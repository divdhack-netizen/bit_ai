"""Test unitari del servizio file atomico, sandboxed e isolato per sessione."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from assistente.config import FileConfig
from assistente.file_service import WorkspaceFileService

pytestmark = pytest.mark.unit

_ORIGINAL_CONTENT = "contenuto originale\n"
_UPDATED_CONTENT = "contenuto aggiornato\n"


def _service(workspace: Path, *, max_read_characters: int = 12_000) -> WorkspaceFileService:
    """
    Costruisce un servizio confinato nella directory temporanea del test.

    Returns:
        Servizio deny-by-default con configurazione indipendente dal repository.
    """
    return WorkspaceFileService(
        FileConfig(
            workspace=workspace,
            max_read_characters=max_read_characters,
        )
    )


def test_servizio_blocca_scrittura_senza_intent(tmp_path: Path) -> None:
    """Lo stato iniziale deny-by-default impedisce persistenza accidentale."""
    service = _service(tmp_path)

    result = service.write_text("blocked.txt", "non scrivere")

    assert result.succeeded is False
    assert "non autorizza" in str(result.error)
    assert not (tmp_path / "blocked.txt").exists()


def test_scrittura_atomica_supporta_spazi_e_unicode(
    tmp_path: Path,
) -> None:
    """Spazi e caratteri non ASCII sono validi nei path .txt."""
    service = _service(tmp_path)
    service.begin_turn(write_allowed=True)
    relative_path = "documenti utili/ricetta_è_finale.txt"

    result = service.write_text(relative_path, _UPDATED_CONTENT)

    assert result.succeeded
    assert result.path == relative_path
    assert (tmp_path / relative_path).read_text(encoding="utf-8") == _UPDATED_CONTENT
    assert service.written_paths == [relative_path]


def test_errore_replace_lascia_intatto_il_file_precedente(tmp_path: Path) -> None:
    """Un fallimento prima del commit atomico non corrompe né tronca il file esistente."""
    destination = tmp_path / "report.txt"
    destination.write_text(_ORIGINAL_CONTENT, encoding="utf-8")
    service = _service(tmp_path)
    service.begin_turn(write_allowed=True)

    with patch(
        "assistente.file_service.os.replace",
        side_effect=OSError("disco non disponibile"),
    ):
        result = service.write_text("report.txt", _UPDATED_CONTENT)

    assert result.succeeded is False
    assert destination.read_text(encoding="utf-8") == _ORIGINAL_CONTENT
    assert list(tmp_path.glob(".agent-write-*.tmp")) == []
    assert service.written_paths == []


def test_path_prosa_observation_non_e_un_file(tmp_path: Path) -> None:
    """I titoli del piano CoT non devono essere accettati come path."""
    service = _service(tmp_path)
    service.begin_turn(write_allowed=True)
    result = service.write_text("3.  Observation: Result.", "junk")
    assert result.succeeded is False
    assert not list(tmp_path.iterdir())


def test_symlink_verso_esterno_viene_rifiutato(tmp_path: Path) -> None:
    """Un componente symlink non può trasformare un path relativo in una fuga dal sandbox."""
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (workspace / "link").symlink_to(outside, target_is_directory=True)
    service = _service(workspace)
    service.begin_turn(write_allowed=True)

    result = service.write_text("link/escape.txt", "segreto")

    assert result.succeeded is False
    assert not (outside / "escape.txt").exists()


def test_create_only_non_sovrascrive_file_esistente(tmp_path: Path) -> None:
    """La modalità usata dal fallback conserva sempre un file già materializzato."""
    destination = tmp_path / "existing.txt"
    destination.write_text(_ORIGINAL_CONTENT, encoding="utf-8")
    service = _service(tmp_path)
    service.begin_turn(write_allowed=True)

    result = service.create_text("existing.txt", _UPDATED_CONTENT)

    assert result.succeeded is False
    assert "già esistente" in str(result.error)
    assert destination.read_text(encoding="utf-8") == _ORIGINAL_CONTENT


def test_scrittura_identica_ripetuta_viene_bloccata(tmp_path: Path) -> None:
    """L'anti-loop impedisce una seconda write identica ma consente contenuti corretti."""
    service = _service(tmp_path)
    service.begin_turn(write_allowed=True)

    first = service.write_text("loop.txt", _ORIGINAL_CONTENT)
    repeated = service.write_text("loop.txt", _ORIGINAL_CONTENT)
    corrected = service.write_text("loop.txt", _UPDATED_CONTENT)

    assert first.succeeded
    assert repeated.succeeded is False
    assert "già completata" in str(repeated.error)
    assert corrected.succeeded
    assert (tmp_path / "loop.txt").read_text(encoding="utf-8") == _UPDATED_CONTENT
    assert service.written_paths == ["loop.txt"]


def test_stato_turno_e_isolato_tra_servizi(tmp_path: Path) -> None:
    """Autorizzazione e tracking di una sessione non contaminano un'altra."""
    first_workspace = tmp_path / "first"
    second_workspace = tmp_path / "second"
    first_workspace.mkdir()
    second_workspace.mkdir()
    first = _service(first_workspace)
    second = _service(second_workspace)
    first.begin_turn(write_allowed=True)
    second.begin_turn(write_allowed=False)

    first_result = first.write_text("ok.txt", _ORIGINAL_CONTENT)
    second_result = second.write_text("blocked.txt", _ORIGINAL_CONTENT)

    assert first_result.succeeded
    assert second_result.succeeded is False
    assert first.written_paths == ["ok.txt"]
    assert second.written_paths == []


def test_lettura_tronca_e_listing_ordina_cartelle_prima_dei_file(
    tmp_path: Path,
) -> None:
    """Read/list rispettano budget e restituiscono metadati deterministici."""
    character_limit = 10
    service = _service(tmp_path, max_read_characters=character_limit)
    service.begin_turn(write_allowed=True)
    (tmp_path / "folder").mkdir()
    (tmp_path / "long.txt").write_text("x" * 20, encoding="utf-8")

    read_result = service.read_document("long.txt")
    list_result = service.list_directory(".")

    assert read_result.succeeded
    assert "troncato" in read_result.content
    assert list_result.succeeded
    assert [entry.relative_path for entry in list_result.entries] == [
        "folder",
        "long.txt",
    ]


def test_replace_text_match_unico(tmp_path: Path) -> None:
    """Una occorrenza esatta viene sostituita e tracciata come write del turno."""
    destination = tmp_path / "app.txt"
    destination.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    service = _service(tmp_path)
    service.begin_turn(write_allowed=True)

    result = service.replace_text("app.txt", "beta", "delta")

    assert result.succeeded
    assert result.path == "app.txt"
    assert destination.read_text(encoding="utf-8") == "alpha\ndelta\ngamma\n"
    assert service.written_paths == ["app.txt"]


def test_replace_text_zero_match(tmp_path: Path) -> None:
    """OLD assente non tocca il file e chiede di rileggere."""
    destination = tmp_path / "app.txt"
    destination.write_text("alpha\nbeta\n", encoding="utf-8")
    service = _service(tmp_path)
    service.begin_turn(write_allowed=True)

    result = service.replace_text("app.txt", "omega", "delta")

    assert result.succeeded is False
    assert "non trovato" in str(result.error)
    assert destination.read_text(encoding="utf-8") == "alpha\nbeta\n"
    assert service.written_paths == []


def test_replace_text_match_doppio(tmp_path: Path) -> None:
    """Più occorrenze identiche sono ambigue e non modificano il disco."""
    destination = tmp_path / "app.txt"
    destination.write_text("beta\nalpha\nbeta\n", encoding="utf-8")
    service = _service(tmp_path)
    service.begin_turn(write_allowed=True)

    result = service.replace_text("app.txt", "beta", "delta")

    assert result.succeeded is False
    assert "ambiguo" in str(result.error)
    assert destination.read_text(encoding="utf-8") == "beta\nalpha\nbeta\n"


def test_replace_text_file_assente(tmp_path: Path) -> None:
    """modifica_file non crea file mancanti."""
    service = _service(tmp_path)
    service.begin_turn(write_allowed=True)

    result = service.replace_text("manca.txt", "old", "new")

    assert result.succeeded is False
    assert "non trovato" in str(result.error).casefold()
    assert not (tmp_path / "manca.txt").exists()


def test_replace_text_bloccato_senza_intent(tmp_path: Path) -> None:
    """Deny-by-default vale anche per le patch."""
    (tmp_path / "app.py").write_text("alpha\n", encoding="utf-8")
    service = _service(tmp_path)

    result = service.replace_text("app.py", "alpha", "beta")

    assert result.succeeded is False
    assert "non autorizza" in str(result.error)
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "alpha\n"


def test_replace_text_poi_fingerprint_identico(tmp_path: Path) -> None:
    """Dopo una patch, una write dello stesso contenuto nello stesso turno è anti-loop."""
    destination = tmp_path / "app.txt"
    destination.write_text("alpha\n", encoding="utf-8")
    service = _service(tmp_path)
    service.begin_turn(write_allowed=True)

    patched = service.replace_text("app.txt", "alpha", "beta")
    repeated = service.write_text("app.txt", "beta\n")

    assert patched.succeeded
    assert repeated.succeeded is False
    assert "non lo sovrascrivo" in str(repeated.error)
    assert destination.read_text(encoding="utf-8") == "beta\n"


def test_documento_preesistente_non_si_sovrascrive(tmp_path: Path) -> None:
    """Un .txt già nella cartella è una sorgente: niente overwrite, suggerisce un nome nuovo."""
    destination = tmp_path / "note.txt"
    destination.write_text("appunti originali\n", encoding="utf-8")
    service = _service(tmp_path)
    service.begin_turn(write_allowed=True)

    rifiuto = service.write_text("note.txt", "prova\n")
    assert rifiuto.succeeded is False
    assert "note_2.txt" in str(rifiuto.error)
    assert destination.read_text(encoding="utf-8") == "appunti originali\n"

    # Un file creato dall'agente nella sessione resta riscrivibile, anche al turno dopo.
    assert service.write_text("sintesi.txt", "prima\n").succeeded
    service.begin_turn(write_allowed=True)
    assert service.write_text("sintesi.txt", "seconda\n").succeeded
    assert (tmp_path / "sintesi.txt").read_text(encoding="utf-8") == "seconda\n"


def test_replace_text_ignora_troncamento_lettura(tmp_path: Path) -> None:
    """La patch legge il file intero, anche oltre max_read_characters."""
    character_limit = 10
    unique_tail = "UNIQUE_TAIL"
    destination = tmp_path / "long.txt"
    destination.write_text(("x" * 40) + unique_tail, encoding="utf-8")
    service = _service(tmp_path, max_read_characters=character_limit)
    service.begin_turn(write_allowed=True)

    result = service.replace_text("long.txt", unique_tail, "DONE")

    assert result.succeeded
    assert destination.read_text(encoding="utf-8").endswith("DONE")
    read_result = service.read_document("long.txt")
    assert "troncato" in read_result.content


def test_normalize_slash_iniziale_workspace(tmp_path: Path) -> None:
    """Slash iniziale su una cartella progetto resta nel sandbox; /tmp e /etc no."""
    service = _service(tmp_path)
    assert service.normalize_file_path("/sito/index.html") == "sito/index.html"
    assert service.normalize_file_path("/tmp/fuori.txt") is None
    assert service.normalize_file_path("/etc/passwd") is None


def test_create_directory_exist_ok_e_file_in_the_way(tmp_path: Path) -> None:
    """mkdir è idempotente su una dir; rifiuta se il path è un file."""
    service = _service(tmp_path)
    service.begin_turn(write_allowed=True)
    first = service.create_directory("notes/2026")
    second = service.create_directory("notes/2026")
    (tmp_path / "file.txt").write_text("x", encoding="utf-8")
    blocked = service.create_directory("file.txt")

    assert first.succeeded
    assert (tmp_path / "notes/2026").is_dir()
    assert second.succeeded is False
    assert "già completata" in str(second.error)
    assert blocked.succeeded is False
    assert "non è una cartella" in str(blocked.error).casefold()
    assert service.written_paths == ["notes/2026"]


def test_written_files_esclude_le_cartelle(tmp_path: Path) -> None:
    """Una cartella è progresso del turno, non un file scritto da citare."""
    service = _service(tmp_path)
    service.begin_turn(write_allowed=True)
    service.create_directory("notes")
    service.write_text("notes/oggi.txt", "testo")

    assert service.written_paths == ["notes", "notes/oggi.txt"]
    assert service.written_files == ["notes/oggi.txt"]


def test_list_tree_salta_venv(tmp_path: Path) -> None:
    """venv non compare nell'albero; i file utili sì."""
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "oggi.md").write_text("ok", encoding="utf-8")
    (tmp_path / "venv").mkdir()
    (tmp_path / "venv" / "x.py").write_text("no", encoding="utf-8")
    service = _service(tmp_path)

    tree = service.list_tree(".")
    names = [entry.relative_path for entry in tree.entries]
    assert "notes" in names
    assert "notes/oggi.md" in names
    assert all("venv" not in name.split("/") for name in names)


def test_find_files_glob_e_sottostringa(tmp_path: Path) -> None:
    """Glob e frammento matchano il nome, non il contenuto."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "brief.md").write_text("segreto", encoding="utf-8")
    (tmp_path / "docs" / "altro.txt").write_text("brief nel body", encoding="utf-8")
    service = _service(tmp_path)

    globbed = service.find_files("*.md")
    substr = service.find_files("brief")
    glob_names = [entry.relative_path for entry in globbed.entries]
    sub_names = [entry.relative_path for entry in substr.entries]
    assert "docs/brief.md" in glob_names
    assert "docs/altro.txt" not in glob_names
    assert "docs/brief.md" in sub_names
    assert "docs/altro.txt" not in sub_names


def test_create_directory_bloccata_senza_intent(tmp_path: Path) -> None:
    """Deny-by-default vale anche per mkdir."""
    service = _service(tmp_path)
    result = service.create_directory("notes")
    assert result.succeeded is False
    assert "non autorizza" in str(result.error)
    assert not (tmp_path / "notes").exists()


def test_deny_nasconde_segreti_da_elenchi_e_letture(tmp_path: Path) -> None:
    """R5: la deny copre anche letture ed elenchi, con cartelle e radice inclusi."""
    from assistente.permissions import WritePermissionRules

    (tmp_path / "privato").mkdir()
    (tmp_path / "privato" / "chiavi.txt").write_text("s3cr3t", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.txt").write_text("ok", encoding="utf-8")
    (tmp_path / "note.md").write_text("ok", encoding="utf-8")
    service = WorkspaceFileService(
        FileConfig(workspace=tmp_path),
        permission_rules=WritePermissionRules(deny=("privato/**",)),
    )
    service.begin_turn(True)

    # Lettura negata con diagnostica dedicata e contenuto vuoto.
    denied = service.read_document("privato/chiavi.txt")
    assert denied.succeeded is False
    assert denied.content == ""
    assert "Lettura non consentita" in str(denied.error)

    # Elenchi: la cartella il cui contenuto è tutto negato sparisce.
    nomi_root = [entry.relative_path for entry in service.list_directory(".").entries]
    assert "privato" not in nomi_root
    assert "docs" in nomi_root and "note.md" in nomi_root
    nomi_tree = [entry.relative_path for entry in service.list_tree(".").entries]
    assert not any(name.startswith("privato") for name in nomi_tree)
    nomi_find = [entry.relative_path for entry in service.find_files("*.txt").entries]
    assert "privato/chiavi.txt" not in nomi_find
    assert "docs/a.txt" in nomi_find

    # Scrittura comunque negata dalle stesse regole.
    assert service.write_text("privato/x.txt", "a").succeeded is False


def test_deny_cartella_blocca_elenco_e_figli(tmp_path: Path) -> None:
    """Negare la cartella per nome la rende invisibile e non elencabile."""
    from assistente.permissions import WritePermissionRules

    (tmp_path / "segreti").mkdir()
    (tmp_path / "segreti" / "k.txt").write_text("x", encoding="utf-8")
    (tmp_path / "aperto.txt").write_text("ok", encoding="utf-8")
    service = WorkspaceFileService(
        FileConfig(workspace=tmp_path),
        permission_rules=WritePermissionRules(deny=("segreti",)),
    )
    service.begin_turn(True)

    nomi_root = [entry.relative_path for entry in service.list_directory(".").entries]
    assert "segreti" not in nomi_root
    errore = service.list_directory("segreti")
    assert errore.succeeded is False
    assert "Lettura non consentita" in str(errore.error)
    # Il figlio è illeggibile per antenato negato, senza alcun glob.
    assert service.read_document("segreti/k.txt").succeeded is False
    assert service.read_document("aperto.txt").succeeded is True


def test_env_radice_negato_dalle_regole_repo(tmp_path: Path) -> None:
    """Le regole di default negano .env in radice e credentials.json in root (bug lstrip)."""
    from assistente.permissions import load_write_permission_rules

    rules = load_write_permission_rules()
    if not rules.deny:
        pytest.skip("Nessuna regola di permesso installata nell'ambiente di test.")
    for path in (".env", ".env.local", "credentials.json", "secrets.yaml"):
        assert rules.is_write_allowed(path)[0] is False, path
        assert rules.is_read_allowed(path)[0] is False, path


@pytest.mark.parametrize("alias", [".", "./", "attuale", "cartella corrente", "Root"])
def test_alias_cartella_attuale_e_root(tmp_path: Path, alias: str) -> None:
    """Regressione: «elencami i file nella cartella attuale» non trovava la root."""
    assert _service(tmp_path).normalize_directory_path(alias) == ""


def test_nome_della_cartella_stessa_e_root(tmp_path: Path) -> None:
    """Il modello scrive «Documents» quando lanciato da ~/Documents."""
    workspace = tmp_path / "Documents"
    workspace.mkdir()
    assert _service(workspace).normalize_directory_path("Documents") == ""


def test_sottocartella_reale_vince_sull_alias(tmp_path: Path) -> None:
    (tmp_path / "attuale").mkdir()
    assert _service(tmp_path).normalize_directory_path("attuale") == "attuale"


@pytest.mark.parametrize("query", ["articolo_1", "Articolo 1", "ARTICOLO-01"])
def test_trova_file_tollera_separatori_e_zeri(tmp_path: Path, query: str) -> None:
    """Regressione: «un file di nome articolo_1» non trovava Articolo_01.pages."""
    (tmp_path / "Articolo_01.pages").write_bytes(b"x")
    result = _service(tmp_path).find_files(query)
    assert [entry.relative_path for entry in result.entries] == ["Articolo_01.pages"]


def test_trova_file_approssimato_solo_come_ripiego(tmp_path: Path) -> None:
    """Se la sottostringa trova qualcosa, il confronto approssimato non allarga i risultati."""
    (tmp_path / "note_1.txt").write_bytes(b"x")
    (tmp_path / "note1.txt").write_bytes(b"x")
    nomi = [entry.relative_path for entry in _service(tmp_path).find_files("note_1").entries]
    assert nomi == ["note_1.txt"]
    assert _service(tmp_path).find_files("articolo_2").entries == ()


@pytest.mark.parametrize("query", ["articolo_1.txt", "articolo_1*"])
def test_trova_file_ignora_estensione_e_glob_indovinati(tmp_path: Path, query: str) -> None:
    (tmp_path / "Articolo_01.pages").write_bytes(b"x")
    nomi = [entry.relative_path for entry in _service(tmp_path).find_files(query).entries]
    assert nomi == ["Articolo_01.pages"]
