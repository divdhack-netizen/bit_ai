"""Unit tests for the atomic, sandboxed, per-session file service."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from assistant.config import FileConfig
from assistant.file_service import WorkspaceFileService

pytestmark = pytest.mark.unit

_ORIGINAL_CONTENT = "contenuto originale\n"
_UPDATED_CONTENT = "contenuto aggiornato\n"


def _service(workspace: Path, *, max_read_characters: int = 12_000) -> WorkspaceFileService:
    """
    Build a service confined to the test's temporary directory.

    Returns:
        Deny-by-default service with configuration independent of the repository.
    """
    return WorkspaceFileService(
        FileConfig(
            workspace=workspace,
            max_read_characters=max_read_characters,
        )
    )


def test_service_blocks_write_without_intent(tmp_path: Path) -> None:
    """The deny-by-default initial state prevents accidental persistence."""
    service = _service(tmp_path)

    result = service.write_text("blocked.txt", "non scrivere")

    assert result.succeeded is False
    assert "non autorizza" in str(result.error)
    assert not (tmp_path / "blocked.txt").exists()


def test_atomic_write_supports_spaces_and_unicode(
    tmp_path: Path,
) -> None:
    """Spaces and non-ASCII characters are valid in .txt paths."""
    service = _service(tmp_path)
    service.begin_turn(write_allowed=True)
    relative_path = "documenti utili/ricetta_è_finale.txt"

    result = service.write_text(relative_path, _UPDATED_CONTENT)

    assert result.succeeded
    assert result.path == relative_path
    assert (tmp_path / relative_path).read_text(encoding="utf-8") == _UPDATED_CONTENT
    assert service.written_paths == [relative_path]


def test_replace_error_leaves_previous_file_intact(tmp_path: Path) -> None:
    """A failure before the atomic commit neither corrupts nor truncates the existing file."""
    destination = tmp_path / "report.txt"
    destination.write_text(_ORIGINAL_CONTENT, encoding="utf-8")
    service = _service(tmp_path)
    service.begin_turn(write_allowed=True)

    with patch(
        "assistant.file_service.os.replace",
        side_effect=OSError("disco non disponibile"),
    ):
        result = service.write_text("report.txt", _UPDATED_CONTENT)

    assert result.succeeded is False
    assert destination.read_text(encoding="utf-8") == _ORIGINAL_CONTENT
    assert list(tmp_path.glob(".agent-write-*.tmp")) == []
    assert service.written_paths == []


def test_prose_observation_path_is_not_a_file(tmp_path: Path) -> None:
    """CoT plan headings must not be accepted as paths."""
    service = _service(tmp_path)
    service.begin_turn(write_allowed=True)
    result = service.write_text("3.  Observation: Result.", "junk")
    assert result.succeeded is False
    assert not list(tmp_path.iterdir())


def test_symlink_to_outside_is_rejected(tmp_path: Path) -> None:
    """A symlink component cannot turn a relative path into a sandbox escape."""
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


def test_create_only_does_not_overwrite_existing_file(tmp_path: Path) -> None:
    """The mode used by the fallback always keeps an already materialized file."""
    destination = tmp_path / "existing.txt"
    destination.write_text(_ORIGINAL_CONTENT, encoding="utf-8")
    service = _service(tmp_path)
    service.begin_turn(write_allowed=True)

    result = service.create_text("existing.txt", _UPDATED_CONTENT)

    assert result.succeeded is False
    assert "già esistente" in str(result.error)
    assert destination.read_text(encoding="utf-8") == _ORIGINAL_CONTENT


def test_repeated_identical_write_is_blocked(tmp_path: Path) -> None:
    """Anti-loop blocks a second identical write but allows corrected content."""
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


def test_turn_state_is_isolated_between_services(tmp_path: Path) -> None:
    """One session's authorization and tracking do not contaminate another."""
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


def test_read_truncates_and_listing_sorts_folders_first(
    tmp_path: Path,
) -> None:
    """Read/list respect the budget and return deterministic metadata."""
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


def test_replace_text_single_match(tmp_path: Path) -> None:
    """A single exact occurrence is replaced and tracked as a turn write."""
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
    """A missing OLD does not touch the file and asks for a re-read."""
    destination = tmp_path / "app.txt"
    destination.write_text("alpha\nbeta\n", encoding="utf-8")
    service = _service(tmp_path)
    service.begin_turn(write_allowed=True)

    result = service.replace_text("app.txt", "omega", "delta")

    assert result.succeeded is False
    assert "non trovato" in str(result.error)
    assert destination.read_text(encoding="utf-8") == "alpha\nbeta\n"
    assert service.written_paths == []


def test_replace_text_double_match(tmp_path: Path) -> None:
    """Multiple identical occurrences are ambiguous and do not modify the disk."""
    destination = tmp_path / "app.txt"
    destination.write_text("beta\nalpha\nbeta\n", encoding="utf-8")
    service = _service(tmp_path)
    service.begin_turn(write_allowed=True)

    result = service.replace_text("app.txt", "beta", "delta")

    assert result.succeeded is False
    assert "ambiguo" in str(result.error)
    assert destination.read_text(encoding="utf-8") == "beta\nalpha\nbeta\n"


def test_replace_text_missing_file(tmp_path: Path) -> None:
    """modifica_file does not create missing files."""
    service = _service(tmp_path)
    service.begin_turn(write_allowed=True)

    result = service.replace_text("manca.txt", "old", "new")

    assert result.succeeded is False
    assert "non trovato" in str(result.error).casefold()
    assert not (tmp_path / "manca.txt").exists()


def test_replace_text_blocked_without_intent(tmp_path: Path) -> None:
    """Deny-by-default applies to patches too."""
    (tmp_path / "app.py").write_text("alpha\n", encoding="utf-8")
    service = _service(tmp_path)

    result = service.replace_text("app.py", "alpha", "beta")

    assert result.succeeded is False
    assert "non autorizza" in str(result.error)
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "alpha\n"


def test_replace_text_then_identical_fingerprint(tmp_path: Path) -> None:
    """After a patch, a write of the same content in the same turn is anti-loop."""
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


def test_preexisting_document_is_not_overwritten(tmp_path: Path) -> None:
    """A .txt already in the folder is a source: no overwrite, a new name is suggested."""
    destination = tmp_path / "note.txt"
    destination.write_text("appunti originali\n", encoding="utf-8")
    service = _service(tmp_path)
    service.begin_turn(write_allowed=True)

    rejection = service.write_text("note.txt", "prova\n")
    assert rejection.succeeded is False
    assert "note_2.txt" in str(rejection.error)
    assert destination.read_text(encoding="utf-8") == "appunti originali\n"

    # A file created by the agent in the session stays rewritable, even on the next turn.
    assert service.write_text("sintesi.txt", "prima\n").succeeded
    service.begin_turn(write_allowed=True)
    assert service.write_text("sintesi.txt", "seconda\n").succeeded
    assert (tmp_path / "sintesi.txt").read_text(encoding="utf-8") == "seconda\n"


def test_replace_text_ignores_read_truncation(tmp_path: Path) -> None:
    """The patch reads the whole file, even beyond max_read_characters."""
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


def test_normalize_leading_slash_workspace(tmp_path: Path) -> None:
    """A leading slash on a project folder stays in the sandbox; /tmp and /etc do not."""
    service = _service(tmp_path)
    assert service.normalize_file_path("/sito/index.html") == "sito/index.html"
    assert service.normalize_file_path("/tmp/fuori.txt") is None
    assert service.normalize_file_path("/etc/passwd") is None


def test_create_directory_exist_ok_and_file_in_the_way(tmp_path: Path) -> None:
    """mkdir is idempotent on a dir; it refuses if the path is a file."""
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


def test_written_files_excludes_folders(tmp_path: Path) -> None:
    """A folder is turn progress, not a written file to cite."""
    service = _service(tmp_path)
    service.begin_turn(write_allowed=True)
    service.create_directory("notes")
    service.write_text("notes/oggi.txt", "testo")

    assert service.written_paths == ["notes", "notes/oggi.txt"]
    assert service.written_files == ["notes/oggi.txt"]


def test_list_tree_skips_venv(tmp_path: Path) -> None:
    """venv does not appear in the tree; useful files do."""
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


def test_find_files_glob_and_substring(tmp_path: Path) -> None:
    """Glob and fragment match the name, not the content."""
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


def test_create_directory_blocked_without_intent(tmp_path: Path) -> None:
    """Deny-by-default applies to mkdir too."""
    service = _service(tmp_path)
    result = service.create_directory("notes")
    assert result.succeeded is False
    assert "non autorizza" in str(result.error)
    assert not (tmp_path / "notes").exists()


def test_deny_hides_secrets_from_listings_and_reads(tmp_path: Path) -> None:
    """R5: deny also covers reads and listings, folders and root included."""
    from assistant.permissions import WritePermissionRules

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

    # Denied read with dedicated diagnostics and empty content.
    denied = service.read_document("privato/chiavi.txt")
    assert denied.succeeded is False
    assert denied.content == ""
    assert "Lettura non consentita" in str(denied.error)

    # Listings: a folder whose content is all denied disappears.
    root_names = [entry.relative_path for entry in service.list_directory(".").entries]
    assert "privato" not in root_names
    assert "docs" in root_names and "note.md" in root_names
    tree_names = [entry.relative_path for entry in service.list_tree(".").entries]
    assert not any(name.startswith("privato") for name in tree_names)
    find_names = [entry.relative_path for entry in service.find_files("*.txt").entries]
    assert "privato/chiavi.txt" not in find_names
    assert "docs/a.txt" in find_names

    # Writes are still denied by the same rules.
    assert service.write_text("privato/x.txt", "a").succeeded is False


def test_deny_folder_blocks_listing_and_children(tmp_path: Path) -> None:
    """Denying the folder by name makes it invisible and unlistable."""
    from assistant.permissions import WritePermissionRules

    (tmp_path / "segreti").mkdir()
    (tmp_path / "segreti" / "k.txt").write_text("x", encoding="utf-8")
    (tmp_path / "aperto.txt").write_text("ok", encoding="utf-8")
    service = WorkspaceFileService(
        FileConfig(workspace=tmp_path),
        permission_rules=WritePermissionRules(deny=("segreti",)),
    )
    service.begin_turn(True)

    root_names = [entry.relative_path for entry in service.list_directory(".").entries]
    assert "segreti" not in root_names
    error = service.list_directory("segreti")
    assert error.succeeded is False
    assert "Lettura non consentita" in str(error.error)
    # The child is unreadable because of a denied ancestor, with no glob at all.
    assert service.read_document("segreti/k.txt").succeeded is False
    assert service.read_document("aperto.txt").succeeded is True


def test_root_env_denied_by_repo_rules(tmp_path: Path) -> None:
    """The default rules deny .env at the root and credentials.json at the root (lstrip bug)."""
    from assistant.permissions import load_write_permission_rules

    rules = load_write_permission_rules()
    if not rules.deny:
        pytest.skip("Nessuna regola di permesso installata nell'ambiente di test.")
    for path in (".env", ".env.local", "credentials.json", "secrets.yaml"):
        assert rules.is_write_allowed(path)[0] is False, path
        assert rules.is_read_allowed(path)[0] is False, path


@pytest.mark.parametrize("alias", [".", "./", "attuale", "cartella corrente", "Root"])
def test_current_folder_alias_is_root(tmp_path: Path, alias: str) -> None:
    """Regression: «elencami i file nella cartella attuale» did not find the root."""
    assert _service(tmp_path).normalize_directory_path(alias) == ""


def test_own_folder_name_is_root(tmp_path: Path) -> None:
    """The model writes «Documents» when launched from ~/Documents."""
    workspace = tmp_path / "Documents"
    workspace.mkdir()
    assert _service(workspace).normalize_directory_path("Documents") == ""


def test_real_subfolder_wins_over_alias(tmp_path: Path) -> None:
    (tmp_path / "attuale").mkdir()
    assert _service(tmp_path).normalize_directory_path("attuale") == "attuale"


@pytest.mark.parametrize("query", ["articolo_1", "Articolo 1", "ARTICOLO-01"])
def test_find_file_tolerates_separators_and_zeros(tmp_path: Path, query: str) -> None:
    """Regression: «un file di nome articolo_1» did not find Articolo_01.pages."""
    (tmp_path / "Articolo_01.pages").write_bytes(b"x")
    result = _service(tmp_path).find_files(query)
    assert [entry.relative_path for entry in result.entries] == ["Articolo_01.pages"]


def test_find_file_approximate_only_as_fallback(tmp_path: Path) -> None:
    """If the substring finds something, the approximate match does not widen the results."""
    (tmp_path / "note_1.txt").write_bytes(b"x")
    (tmp_path / "note1.txt").write_bytes(b"x")
    names = [entry.relative_path for entry in _service(tmp_path).find_files("note_1").entries]
    assert names == ["note_1.txt"]
    assert _service(tmp_path).find_files("articolo_2").entries == ()


@pytest.mark.parametrize("query", ["articolo_1.txt", "articolo_1*"])
def test_find_file_ignores_guessed_extension_and_glob(tmp_path: Path, query: str) -> None:
    (tmp_path / "Articolo_01.pages").write_bytes(b"x")
    names = [entry.relative_path for entry in _service(tmp_path).find_files(query).entries]
    assert names == ["Articolo_01.pages"]
