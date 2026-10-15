"""Unit tests for the file tools: write, read, listing, path sandbox."""

from __future__ import annotations

from pathlib import Path

import pytest

from assistant.config import (
    FileConfig,
    NEW_BLOCK_MARKER,
    OLD_BLOCK_MARKER,
    FILE_SEPARATOR,
    WORKSPACE,
)
from assistant.tools import (
    apply_fallback,
    create_folder_from_text,
    list_tree_from_text,
    list_folder_from_text,
    set_write_allowed,
    read_document_from_text,
    edit_file_from_text,
    reset_turn_state,
    sanitize_file_content,
    write_file_from_text,
    find_file_from_text,
)
from assistant.file_service import WorkspaceFileService
from tests.conftest import DIR_UNIT_REL

pytestmark = pytest.mark.unit

# sanitize_file_content normalizes with strip + a mandatory trailing newline.
_HELLO_CONTENT = "ciao agente\nseconda riga UTF-8: àèì"
_HELLO_CONTENT_ON_DISK = _HELLO_CONTENT + "\n"


def _write_input(path_rel: str, content: str) -> str:
    """
    Build the Action Input of the scrivi_file tool.

    Input: relative path and text content.
    Output: the string path\\n---\\ncontent.
    """
    return f"{path_rel}\n{FILE_SEPARATOR}\n{content}"


def _edit_input(path_rel: str, old: str, new: str) -> str:
    """Build the Action Input of the modifica_file tool."""
    return (
        f"{path_rel}\n"
        f"{OLD_BLOCK_MARKER}\n{old}\n"
        f"{NEW_BLOCK_MARKER}\n{new}"
    )


def test_write_file_roundtrip(workspace_test_dir) -> None:
    """
    Check that scrivi_file creates the file and returns a success message.

    Precondition: workspace_test_dir is prepared.
    Postcondition: file on disk with the expected content.
    """
    reset_turn_state()
    path_rel = f"{DIR_UNIT_REL}/hello.txt"
    # Write through the same contract used by the ReAct agent.
    observation = write_file_from_text(_write_input(path_rel, _HELLO_CONTENT))
    assert observation.startswith("File scritto con successo:"), observation
    assert path_rel in observation
    destination = WORKSPACE / path_rel
    # Confirm persistence on disk independently of the tool message.
    assert destination.is_file()
    assert destination.read_text(encoding="utf-8") == _HELLO_CONTENT_ON_DISK


def test_read_document_content(workspace_test_dir) -> None:
    """
    Check a round-trip read of a freshly written UTF-8 file.

    Input: file created in tmp/test_run/unit.
    Expected output: identical content (no truncation below 12k).
    """
    reset_turn_state()
    path_rel = f"{DIR_UNIT_REL}/leggi_me.txt"
    write_file_from_text(_write_input(path_rel, _HELLO_CONTENT))
    pasted = read_document_from_text(path_rel)
    # Fails if the tool returns Errore: instead of the body.
    assert not pasted.startswith("Errore:"), pasted
    assert pasted == _HELLO_CONTENT_ON_DISK


def test_list_folder_contains_file(workspace_test_dir) -> None:
    """
    Check that elenca_cartella lists a file created in the unit folder.

    Input: the tmp/test_run/unit folder with a known file.
    Output: a listing that mentions the file name.
    """
    reset_turn_state()
    path_rel = f"{DIR_UNIT_REL}/elenco.txt"
    write_file_from_text(_write_input(path_rel, "voce elenco"))
    listing = list_folder_from_text(DIR_UNIT_REL)
    assert not listing.startswith("Errore:"), listing
    # The listing includes the relative path or just the name; both forms are accepted.
    assert "elenco.txt" in listing


def test_write_file_accepts_literal_newlines(workspace_test_dir) -> None:
    """
    Check that an Action Input with escaped \\n (a single line) is accepted.
    """
    reset_turn_state()
    path_rel = f"{DIR_UNIT_REL}/escaped.txt"
    # Simulate typical single-line LLM input with escapes.
    observation = write_file_from_text(
        f"{path_rel}\\n{FILE_SEPARATOR}\\ncontenuto ok"
    )
    assert observation.startswith("File scritto con successo:"), observation
    assert (WORKSPACE / path_rel).read_text(encoding="utf-8") == "contenuto ok\n"


def test_forbidden_paths_return_error(workspace_test_dir) -> None:
    """
    Check that absolute paths and `..` traversal are rejected.

    Input: paths not allowed by the workspace sandbox.
    Output: strings starting with Errore:.
    """
    reset_turn_state()
    # Absolute path: forbidden by the sanitizer.
    abs_err = write_file_from_text(
        _write_input("/tmp/fuori.txt", "non deve scrivere")
    )
    assert abs_err.startswith("Errore:"), abs_err

    # Traversal outside the workspace: forbidden.
    err_parent = write_file_from_text(
        _write_input("../escape.txt", "non deve scrivere")
    )
    assert err_parent.startswith("Errore:"), err_parent

    # A read with a forbidden path must fail the same way.
    read_err = read_document_from_text("../README.md")
    assert read_err.startswith("Errore:"), read_err


def test_write_file_accepts_leading_slash_in_workspace(workspace_test_dir) -> None:
    """«/sito/slash_rel.txt» is the relative path sito/slash_rel.txt, not an OS path."""
    reset_turn_state()
    path_rel = "sito/slash_rel.txt"
    observation = write_file_from_text(
        f"/{path_rel}\n{FILE_SEPARATOR}\n<!DOCTYPE html><html></html>"
    )
    assert observation.startswith("File scritto con successo:"), observation
    assert path_rel in observation
    destination = WORKSPACE / path_rel
    assert destination.is_file()
    assert "<!DOCTYPE html>" in destination.read_text(encoding="utf-8")
    destination.unlink()
    parent = destination.parent
    if parent.exists() and not any(parent.iterdir()):
        parent.rmdir()


def test_write_file_blocked_without_intent(workspace_test_dir) -> None:
    """
    Runtime gate: without intent the tool does not write to disk.

    Postcondition: error Observation and no file.
    """
    reset_turn_state()
    set_write_allowed(False)
    path_rel = f"{DIR_UNIT_REL}/proibito.txt"
    observation = write_file_from_text(_write_input(path_rel, "non deve esistere"))
    assert observation.startswith("Errore: scrittura non richiesta"), observation
    assert not (WORKSPACE / path_rel).exists()


def test_fallback_blocked_without_intent(workspace_test_dir) -> None:
    """apply_fallback does not materialize paths from the Final Answer without intent."""
    reset_turn_state()
    set_write_allowed(False)
    path_rel = f"{DIR_UNIT_REL}/fallback_proibito.txt"
    result = f"{path_rel}\n---\ncontenuto fantasma\n"
    written = apply_fallback(result, [])
    assert written == []
    assert not (WORKSPACE / path_rel).exists()


def test_fallback_materializes_only_missing_files(workspace_test_dir) -> None:
    """A tool write that already succeeded does not prevent recovering a second file."""
    reset_turn_state()
    first_path = f"{DIR_UNIT_REL}/gia_scritto.txt"
    second_path = f"{DIR_UNIT_REL}/mancante.txt"
    first_result = write_file_from_text(
        _write_input(first_path, "contenuto tool")
    )
    assert first_result.startswith("File scritto con successo:")
    final_answer = (
        f"{first_path}\n---\nnon sovrascrivere\n"
        f"{second_path}\n---\ncontenuto fallback\n"
    )

    fallback_paths = apply_fallback(final_answer, [first_path])

    assert fallback_paths == [second_path]
    assert (WORKSPACE / first_path).read_text(encoding="utf-8") == "contenuto tool\n"
    assert (WORKSPACE / second_path).read_text(encoding="utf-8") == "contenuto fallback\n"


def test_fallback_does_not_overwrite_untracked_preexisting_file(
    workspace_test_dir,
) -> None:
    """Create-only recovery keeps existing data even if the callback does not know about it."""
    reset_turn_state()
    path_rel = f"{DIR_UNIT_REL}/preesistente.txt"
    destination = WORKSPACE / path_rel
    destination.write_text("dato utente\n", encoding="utf-8")
    final_answer = f"{path_rel}\n---\ncontenuto agente\n"

    fallback_paths = apply_fallback(final_answer, [])

    assert fallback_paths == []
    assert destination.read_text(encoding="utf-8") == "dato utente\n"


def test_repeated_identical_write_is_blocked(workspace_test_dir) -> None:
    """The service prevents a ReAct loop from rewriting the same body."""
    reset_turn_state()
    path_rel = f"{DIR_UNIT_REL}/loop.txt"
    action_input = _write_input(path_rel, "una sola volta")

    first = write_file_from_text(action_input)
    repeated = write_file_from_text(action_input)

    assert first.startswith("File scritto con successo:")
    assert repeated.startswith("Errore:")
    assert "già completata" in repeated


def test_sanitization_does_not_invent_html_closings() -> None:
    """The adapter strips the protocol but does not semantically change incomplete HTML."""
    raw_html = "<html><body><p>contenuto"

    sanitized = sanitize_file_content(raw_html, "pagina.html")

    assert sanitized == raw_html + "\n"
    assert "</html>" not in sanitized


def test_write_file_rejects_path_with_comma(workspace_test_dir) -> None:
    """
    Gemma emits ``tmp/a_duo,AAA``: it must not create a file with a comma in its name.
    """
    reset_turn_state()
    destination = WORKSPACE / "tmp/a_duo,AAA"
    # Remove any leftovers of earlier reproductions in the same workspace.
    destination.unlink(missing_ok=True)
    observation = write_file_from_text("tmp/a_duo,AAA\nAction: scrivi_file")
    assert observation.startswith("Errore:"), observation
    assert not destination.exists()


def test_write_file_accepts_singleline_shorthand_with_extension(
    workspace_test_dir,
) -> None:
    """The ``path.ext,content`` shorthand is interpreted correctly."""
    reset_turn_state()
    path_rel = f"{DIR_UNIT_REL}/comma.txt"
    observation = write_file_from_text(f"{path_rel},corpo monolinea")
    assert observation.startswith("File scritto con successo:"), observation
    assert (WORKSPACE / path_rel).read_text(encoding="utf-8") == "corpo monolinea\n"


def test_write_file_removes_yaml_block_scalar_marker(workspace_test_dir) -> None:
    """``Contenuto: |`` must not leave ``|`` as the first line of the file."""
    reset_turn_state()
    path_rel = f"{DIR_UNIT_REL}/pipe.txt"
    action_input = f"{path_rel}\nContenuto: |\ndef hello():\n    print('ciao')\n"
    observation = write_file_from_text(action_input)
    assert observation.startswith("File scritto con successo:"), observation
    content = (WORKSPACE / path_rel).read_text(encoding="utf-8")
    assert not content.startswith("|")
    assert "def hello():" in content


def test_write_file_rejects_react_protocol_only_body(workspace_test_dir) -> None:
    """Chained Action Inputs must not write ReAct artifacts to disk."""
    reset_turn_state()
    path_rel = f"{DIR_UNIT_REL}/proto.txt"
    action_input = (
        f"{path_rel}\n"
        "Action: scrivi_file\n"
        "Action Input: altro.txt\n"
        "Thought: fatto"
    )
    observation = write_file_from_text(action_input)
    assert observation.startswith("Errore:"), observation
    assert not (WORKSPACE / path_rel).exists()


def test_write_file_cleans_relative_path_prefix(workspace_test_dir) -> None:
    """«path relativo tmp/x.txt» is reduced to the actual path."""
    reset_turn_state()
    path_rel = f"{DIR_UNIT_REL}/prefisso.txt"
    observation = write_file_from_text(
        f"path relativo {path_rel}\n{FILE_SEPARATOR}\nok"
    )
    assert observation.startswith("File scritto con successo:"), observation
    assert path_rel in observation
    assert (WORKSPACE / path_rel).read_text(encoding="utf-8") == "ok\n"


def test_write_file_removes_separator_line_artifact(workspace_test_dir) -> None:
    """A first line «riga ---» must not end up in the file."""
    reset_turn_state()
    path_rel = f"{DIR_UNIT_REL}/riga_sep.txt"
    observation = write_file_from_text(
        f"{path_rel}\nriga ---\ncorpo vero\n"
    )
    assert observation.startswith("File scritto con successo:"), observation
    assert (WORKSPACE / path_rel).read_text(encoding="utf-8") == "corpo vero\n"


def test_write_file_accepts_incomplete_html(workspace_test_dir) -> None:
    """An .html file is a text file: it is persisted even without </html>."""
    reset_turn_state()
    path_rel = f"{DIR_UNIT_REL}/troncato.txt"
    body = "<!DOCTYPE html>\n<html lang=\"it\">\n<body><nav>x</nav>\n"
    observation = write_file_from_text(_write_input(path_rel, body))
    assert observation.startswith("File scritto con successo:"), observation
    assert (WORKSPACE / path_rel).read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_write_file_short_html_overwrite(workspace_test_dir) -> None:
    """A short HTML stub may overwrite a longer document."""
    reset_turn_state()
    path_rel = f"{DIR_UNIT_REL}/sito.txt"
    original = (
        "<!DOCTYPE html>\n<html lang=\"it\"><head><title>Zeroglut</title></head>"
        "<body>" + ("<p>contenuto</p>" * 40) + "</body></html>\n"
    )
    assert write_file_from_text(_write_input(path_rel, original)).startswith(
        "File scritto con successo:"
    )
    stub = (
        "<!DOCTYPE html>\n<html><head><title>Sito Web</title></head>"
        "<body><nav class=\"navbar\"></nav></body></html>\n"
    )
    observation = write_file_from_text(_write_input(path_rel, stub))
    assert observation.startswith("File scritto con successo:"), observation
    assert (WORKSPACE / path_rel).read_text(encoding="utf-8") == stub


def test_write_file_accepts_similar_size_html_overwrite(workspace_test_dir) -> None:
    """A full HTML rewrite of similar size is a valid overwrite."""
    reset_turn_state()
    path_rel = f"{DIR_UNIT_REL}/landing.txt"
    original = (
        "<!DOCTYPE html>\n<html lang=\"it\"><head><title>Zeroglut</title></head>"
        "<body>" + ("<p>contenuto</p>" * 20) + "</body></html>\n"
    )
    assert write_file_from_text(_write_input(path_rel, original)).startswith(
        "File scritto con successo:"
    )
    updated = original.replace("contenuto", "sezione")
    observation = write_file_from_text(_write_input(path_rel, updated))
    assert observation.startswith("File scritto con successo:"), observation
    assert (WORKSPACE / path_rel).read_text(encoding="utf-8") == updated


def test_write_file_recovers_extension_from_thought(workspace_test_dir) -> None:
    """
    If the first line omits the extension but the Thought cites path.ext, path.ext is written.
    """
    reset_turn_state()
    path_rel = f"{DIR_UNIT_REL}/a_duo.txt"
    action_input = (
        f"{DIR_UNIT_REL}/a_duo\n"
        "Contenuto: AAA\n"
        f"Thought: creo il file {path_rel} con AAA"
    )
    observation = write_file_from_text(action_input)
    assert observation.startswith("File scritto con successo:"), observation
    assert path_rel in observation
    assert (WORKSPACE / path_rel).read_text(encoding="utf-8") == "AAA\n"
    assert not (WORKSPACE / f"{DIR_UNIT_REL}/a_duo").exists()


def test_edit_file_roundtrip(workspace_test_dir) -> None:
    """The path/OLD/NEW protocol replaces a single block on disk."""
    reset_turn_state()
    path_rel = f"{DIR_UNIT_REL}/patch.txt"
    write_file_from_text(_write_input(path_rel, "def foo():\n    return 1"))
    observation = edit_file_from_text(
        _edit_input(path_rel, "    return 1", "    return 2")
    )
    assert observation.startswith("File modificato con successo:"), observation
    assert path_rel in observation
    assert (WORKSPACE / path_rel).read_text(encoding="utf-8") == (
        "def foo():\n    return 2\n"
    )


def test_edit_file_cuts_old_tail_in_new(workspace_test_dir) -> None:
    """A NEW followed by a spurious <<<OLD must not end up in the file."""
    reset_turn_state()
    path_rel = f"{DIR_UNIT_REL}/nav.txt"
    original = "nav a {\n    padding: 0.5rem 0.9rem;\n}\n"
    assert write_file_from_text(_write_input(path_rel, original)).startswith(
        "File scritto con successo:"
    )
    observation = edit_file_from_text(
        _edit_input(
            path_rel,
            "    padding: 0.5rem 0.9rem;",
            "    padding: 0.75rem 1.1rem;\n<<<OLD\n    garbage corrotta;",
        )
    )
    assert observation.startswith("File modificato con successo:"), observation
    text = (WORKSPACE / path_rel).read_text(encoding="utf-8")
    assert "<<<OLD" not in text
    assert "garbage" not in text
    assert "padding: 0.75rem 1.1rem;" in text


def test_edit_file_rejects_malformed_format(workspace_test_dir) -> None:
    """Without <<<OLD/<<<NEW markers the tool does not touch the disk."""
    reset_turn_state()
    path_rel = f"{DIR_UNIT_REL}/patch.txt"
    write_file_from_text(_write_input(path_rel, "ciao"))
    observation = edit_file_from_text(f"{path_rel}\n---\nciao")
    assert observation.startswith("Errore:"), observation
    assert (WORKSPACE / path_rel).read_text(encoding="utf-8") == "ciao\n"


def test_edit_file_empty_new_deletes_block(workspace_test_dir) -> None:
    """An empty NEW deletes the OLD block if the match is unique."""
    reset_turn_state()
    path_rel = f"{DIR_UNIT_REL}/trim.txt"
    write_file_from_text(_write_input(path_rel, "keep\nremove\nkeep"))
    observation = edit_file_from_text(_edit_input(path_rel, "remove\n", ""))
    assert observation.startswith("File modificato con successo:"), observation
    assert (WORKSPACE / path_rel).read_text(encoding="utf-8") == "keep\nkeep\n"


def test_edit_file_rejects_traversal(workspace_test_dir) -> None:
    """Paths outside the sandbox stay errors, as with scrivi_file."""
    reset_turn_state()
    observation = edit_file_from_text(
        _edit_input("../escape.txt", "old", "new")
    )
    assert observation.startswith("Errore:"), observation


def test_create_folder_blocked_without_intent(workspace_test_dir) -> None:
    """Closed gate: no mkdir."""
    reset_turn_state()
    set_write_allowed(False)
    observation = create_folder_from_text(f"{DIR_UNIT_REL}/nope")
    assert observation.startswith("Errore: scrittura non richiesta"), observation
    assert not (WORKSPACE / DIR_UNIT_REL / "nope").exists()


def test_list_tree_and_find_file(workspace_test_dir) -> None:
    """Tree at depth 2 and a glob on the name."""
    reset_turn_state()
    nested = f"{DIR_UNIT_REL}/docs/sub/note.txt"
    write_file_from_text(_write_input(nested, "hello"))
    tree = list_tree_from_text(DIR_UNIT_REL)
    assert "docs/" in tree
    found = find_file_from_text("*.txt")
    assert "note.txt" in found


@pytest.mark.parametrize(
    ("action_input", "expected"),
    [
        ("Articolo_01.txtTrovato il file Articolo_01.txt. Procedo alla lettura.", "krill"),
        ("Articolo_01.txt Procedo subito alla lettura del file", "krill"),
    ],
)
def test_read_document_recovers_path_with_pasted_prose(
    tmp_path: Path, action_input: str, expected: str
) -> None:
    """Regression: Gemma continues on the Action Input line and burns iterations."""
    (tmp_path / "Articolo_01.txt").write_text("la balena mangia krill", encoding="utf-8")
    service = WorkspaceFileService(FileConfig(workspace=tmp_path))
    assert expected in read_document_from_text(action_input, service=service)


@pytest.mark.parametrize(
    "action_input",
    [
        ".Tuttavia, non vedo il file tra i documenti della cartella.",
        ".elenca_cartella Action Input obbligatorio: attuale. Mai la root.",
        "attualeTuttavia, non hai fornito l'output del tool",
    ],
)
def test_list_folder_recovers_root_with_pasted_prose(
    tmp_path: Path, action_input: str
) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    service = WorkspaceFileService(FileConfig(workspace=tmp_path))
    assert "a.txt" in list_folder_from_text(action_input, service=service)


def test_name_with_spaces_is_not_truncated(tmp_path: Path) -> None:
    """A non-existent «notes vecchie» must not silently become «notes»."""
    (tmp_path / "notes").mkdir()
    service = WorkspaceFileService(FileConfig(workspace=tmp_path))
    assert list_folder_from_text("notes vecchie", service=service).startswith("Errore")


def test_repeated_consecutive_read_warns_model(tmp_path: Path) -> None:
    """Gemma repeated elenca_cartella '.' until the iteration limit."""
    from assistant.tools import create_file_tools

    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    service = WorkspaceFileService(FileConfig(workspace=tmp_path))
    service.begin_turn(write_allowed=False)
    list_tool = next(t for t in create_file_tools(service, write_allowed=False) if t.name == "elenca_cartella")
    first = list_tool.func(".")
    second = list_tool.func(".")
    assert not first.startswith("Stessa Action")
    assert second.startswith("Stessa Action") and "a.txt" in second
    service.begin_turn(write_allowed=False)
    assert not list_tool.func(".").startswith("Stessa Action")
