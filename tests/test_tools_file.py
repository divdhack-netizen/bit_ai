"""Test unitari sui tool file: scrittura, lettura, elenco, sandbox path."""

from __future__ import annotations

from pathlib import Path

import pytest

from assistente.config import (
    FileConfig,
    MARCATORE_BLOCCO_NEW,
    MARCATORE_BLOCCO_OLD,
    SEPARATORE_FILE,
    WORKSPACE,
)
from assistente.tools import (
    applica_fallback,
    crea_cartella_da_testo,
    elenca_albero_da_testo,
    elenca_cartella_da_testo,
    imposta_scrittura_consentita,
    leggi_documento_da_testo,
    modifica_file_da_testo,
    reset_stato_turno,
    sanitizza_contenuto_file,
    scrivi_file_da_testo,
    trova_file_da_testo,
)
from assistente.file_service import WorkspaceFileService
from tests.conftest import DIR_UNIT_REL

pytestmark = pytest.mark.unit

# sanitizza_contenuto_file normalizza con strip + newline finale obbligatorio.
_CONTENUTO_HELLO = "ciao agente\nseconda riga UTF-8: àèì"
_CONTENUTO_HELLO_SU_DISCO = _CONTENUTO_HELLO + "\n"


def _input_scrivi(path_rel: str, contenuto: str) -> str:
    """
    Costruisce l'Action Input del tool scrivi_file.

    Input: path relativo e contenuto testuale.
    Output: stringa path\\n---\\ncontenuto.
    """
    return f"{path_rel}\n{SEPARATORE_FILE}\n{contenuto}"


def _input_modifica(path_rel: str, old: str, new: str) -> str:
    """Costruisce l'Action Input del tool modifica_file."""
    return (
        f"{path_rel}\n"
        f"{MARCATORE_BLOCCO_OLD}\n{old}\n"
        f"{MARCATORE_BLOCCO_NEW}\n{new}"
    )


def test_scrivi_file_roundtrip(workspace_test_dir) -> None:
    """
    Verifica che scrivi_file crei il file e restituisca messaggio di successo.

    Precondition: workspace_test_dir preparato.
    Postcondition: file presente su disco con contenuto atteso.
    """
    reset_stato_turno()
    path_rel = f"{DIR_UNIT_REL}/hello.txt"
    # Scrive tramite lo stesso contratto usato dall'agente ReAct.
    osservazione = scrivi_file_da_testo(_input_scrivi(path_rel, _CONTENUTO_HELLO))
    assert osservazione.startswith("File scritto con successo:"), osservazione
    assert path_rel in osservazione
    destinazione = WORKSPACE / path_rel
    # Conferma persistenza su disco indipendente dal messaggio tool.
    assert destinazione.is_file()
    assert destinazione.read_text(encoding="utf-8") == _CONTENUTO_HELLO_SU_DISCO


def test_leggi_documento_contenuto(workspace_test_dir) -> None:
    """
    Verifica lettura round-trip di un file UTF-8 appena scritto.

    Input: file creato in tmp/test_run/unit.
    Output atteso: contenuto identico (nessun troncamento sotto 12k).
    """
    reset_stato_turno()
    path_rel = f"{DIR_UNIT_REL}/leggi_me.txt"
    scrivi_file_da_testo(_input_scrivi(path_rel, _CONTENUTO_HELLO))
    letto = leggi_documento_da_testo(path_rel)
    # Fallisce se il tool restituisce Errore: invece del corpo.
    assert not letto.startswith("Errore:"), letto
    assert letto == _CONTENUTO_HELLO_SU_DISCO


def test_elenca_cartella_contiene_file(workspace_test_dir) -> None:
    """
    Verifica che elenca_cartella elenchi un file creato nella cartella unit.

    Input: cartella tmp/test_run/unit con un file noto.
    Output: listing che cita il nome file.
    """
    reset_stato_turno()
    path_rel = f"{DIR_UNIT_REL}/elenco.txt"
    scrivi_file_da_testo(_input_scrivi(path_rel, "voce elenco"))
    listing = elenca_cartella_da_testo(DIR_UNIT_REL)
    assert not listing.startswith("Errore:"), listing
    # Il listing include path relativo o solo nome; accettiamo entrambe le forme.
    assert "elenco.txt" in listing


def test_scrivi_file_accetta_newline_letterali(workspace_test_dir) -> None:
    """
    Verifica che Action Input con \\n escaped (una sola riga) venga accettato.
    """
    reset_stato_turno()
    path_rel = f"{DIR_UNIT_REL}/escaped.txt"
    # Simula tipico input LLM su una riga con escape.
    osservazione = scrivi_file_da_testo(
        f"{path_rel}\\n{SEPARATORE_FILE}\\ncontenuto ok"
    )
    assert osservazione.startswith("File scritto con successo:"), osservazione
    assert (WORKSPACE / path_rel).read_text(encoding="utf-8") == "contenuto ok\n"


def test_path_vietati_restituiscono_errore(workspace_test_dir) -> None:
    """
    Verifica che path assoluti e traversal `..` siano rifiutati.

    Input: path non consentiti dal sandbox workspace.
    Output: stringhe che iniziano con Errore:.
    """
    reset_stato_turno()
    # Path assoluto: vietato dal sanitizer.
    err_assoluto = scrivi_file_da_testo(
        _input_scrivi("/tmp/fuori.txt", "non deve scrivere")
    )
    assert err_assoluto.startswith("Errore:"), err_assoluto

    # Traversal fuori workspace: vietato.
    err_parent = scrivi_file_da_testo(
        _input_scrivi("../escape.txt", "non deve scrivere")
    )
    assert err_parent.startswith("Errore:"), err_parent

    # Lettura con path vietato deve fallire allo stesso modo.
    err_lettura = leggi_documento_da_testo("../README.md")
    assert err_lettura.startswith("Errore:"), err_lettura


def test_scrivi_file_accetta_slash_iniziale_nel_workspace(workspace_test_dir) -> None:
    """«/sito/slash_rel.txt» è path relativo sito/slash_rel.txt, non un path OS."""
    reset_stato_turno()
    path_rel = "sito/slash_rel.txt"
    osservazione = scrivi_file_da_testo(
        f"/{path_rel}\n{SEPARATORE_FILE}\n<!DOCTYPE html><html></html>"
    )
    assert osservazione.startswith("File scritto con successo:"), osservazione
    assert path_rel in osservazione
    destinazione = WORKSPACE / path_rel
    assert destinazione.is_file()
    assert "<!DOCTYPE html>" in destinazione.read_text(encoding="utf-8")
    destinazione.unlink()
    parent = destinazione.parent
    if parent.exists() and not any(parent.iterdir()):
        parent.rmdir()


def test_scrivi_file_bloccato_senza_intent(workspace_test_dir) -> None:
    """
    Gate runtime: senza intent il tool non scrive su disco.

    Postcondition: Observation di errore e file assente.
    """
    reset_stato_turno()
    imposta_scrittura_consentita(False)
    path_rel = f"{DIR_UNIT_REL}/proibito.txt"
    osservazione = scrivi_file_da_testo(_input_scrivi(path_rel, "non deve esistere"))
    assert osservazione.startswith("Errore: scrittura non richiesta"), osservazione
    assert not (WORKSPACE / path_rel).exists()


def test_fallback_bloccato_senza_intent(workspace_test_dir) -> None:
    """applica_fallback non materializza path da Final Answer senza intent."""
    reset_stato_turno()
    imposta_scrittura_consentita(False)
    path_rel = f"{DIR_UNIT_REL}/fallback_proibito.txt"
    risultato = f"{path_rel}\n---\ncontenuto fantasma\n"
    scritti = applica_fallback(risultato, [])
    assert scritti == []
    assert not (WORKSPACE / path_rel).exists()


def test_fallback_materializza_solo_i_file_mancanti(workspace_test_dir) -> None:
    """Una write tool già riuscita non impedisce il recupero di un secondo file."""
    reset_stato_turno()
    first_path = f"{DIR_UNIT_REL}/gia_scritto.txt"
    second_path = f"{DIR_UNIT_REL}/mancante.txt"
    first_result = scrivi_file_da_testo(
        _input_scrivi(first_path, "contenuto tool")
    )
    assert first_result.startswith("File scritto con successo:")
    final_answer = (
        f"{first_path}\n---\nnon sovrascrivere\n"
        f"{second_path}\n---\ncontenuto fallback\n"
    )

    fallback_paths = applica_fallback(final_answer, [first_path])

    assert fallback_paths == [second_path]
    assert (WORKSPACE / first_path).read_text(encoding="utf-8") == "contenuto tool\n"
    assert (WORKSPACE / second_path).read_text(encoding="utf-8") == "contenuto fallback\n"


def test_fallback_non_sovrascrive_un_file_preesistente_non_tracciato(
    workspace_test_dir,
) -> None:
    """Il recupero create-only conserva dati presenti anche se il callback non li conosce."""
    reset_stato_turno()
    path_rel = f"{DIR_UNIT_REL}/preesistente.txt"
    destination = WORKSPACE / path_rel
    destination.write_text("dato utente\n", encoding="utf-8")
    final_answer = f"{path_rel}\n---\ncontenuto agente\n"

    fallback_paths = applica_fallback(final_answer, [])

    assert fallback_paths == []
    assert destination.read_text(encoding="utf-8") == "dato utente\n"


def test_scrittura_identica_ripetuta_viene_bloccata(workspace_test_dir) -> None:
    """Il servizio impedisce che un loop ReAct riscriva lo stesso body."""
    reset_stato_turno()
    path_rel = f"{DIR_UNIT_REL}/loop.txt"
    action_input = _input_scrivi(path_rel, "una sola volta")

    first = scrivi_file_da_testo(action_input)
    repeated = scrivi_file_da_testo(action_input)

    assert first.startswith("File scritto con successo:")
    assert repeated.startswith("Errore:")
    assert "già completata" in repeated


def test_sanitizzazione_non_inventa_chiusure_html() -> None:
    """L'adapter rimuove il protocollo ma non modifica semanticamente HTML incompleto."""
    raw_html = "<html><body><p>contenuto"

    sanitized = sanitizza_contenuto_file(raw_html, "pagina.html")

    assert sanitized == raw_html + "\n"
    assert "</html>" not in sanitized


def test_scrivi_file_rifiuta_path_con_virgola(workspace_test_dir) -> None:
    """
    Gemma emette ``tmp/a_duo,AAA``: non deve creare un file con la virgola nel nome.
    """
    reset_stato_turno()
    destinazione = WORKSPACE / "tmp/a_duo,AAA"
    # Rimuove eventuali residui di riproduzioni precedenti nello stesso workspace.
    destinazione.unlink(missing_ok=True)
    osservazione = scrivi_file_da_testo("tmp/a_duo,AAA\nAction: scrivi_file")
    assert osservazione.startswith("Errore:"), osservazione
    assert not destinazione.exists()


def test_scrivi_file_accetta_shorthand_monolinea_con_estensione(
    workspace_test_dir,
) -> None:
    """Shorthand ``path.ext,contenuto`` viene interpretato correttamente."""
    reset_stato_turno()
    path_rel = f"{DIR_UNIT_REL}/comma.txt"
    osservazione = scrivi_file_da_testo(f"{path_rel},corpo monolinea")
    assert osservazione.startswith("File scritto con successo:"), osservazione
    assert (WORKSPACE / path_rel).read_text(encoding="utf-8") == "corpo monolinea\n"


def test_scrivi_file_rimuove_marker_yaml_block_scalar(workspace_test_dir) -> None:
    """``Contenuto: |`` non deve lasciare ``|`` come prima riga del file."""
    reset_stato_turno()
    path_rel = f"{DIR_UNIT_REL}/pipe.txt"
    action_input = f"{path_rel}\nContenuto: |\ndef hello():\n    print('ciao')\n"
    osservazione = scrivi_file_da_testo(action_input)
    assert osservazione.startswith("File scritto con successo:"), osservazione
    contenuto = (WORKSPACE / path_rel).read_text(encoding="utf-8")
    assert not contenuto.startswith("|")
    assert "def hello():" in contenuto


def test_scrivi_file_rifiuta_body_solo_protocollo_react(workspace_test_dir) -> None:
    """Action Input concatenati non devono scrivere artefatti ReAct su disco."""
    reset_stato_turno()
    path_rel = f"{DIR_UNIT_REL}/proto.txt"
    action_input = (
        f"{path_rel}\n"
        "Action: scrivi_file\n"
        "Action Input: altro.txt\n"
        "Thought: fatto"
    )
    osservazione = scrivi_file_da_testo(action_input)
    assert osservazione.startswith("Errore:"), osservazione
    assert not (WORKSPACE / path_rel).exists()


def test_scrivi_file_pulisce_prefisso_path_relativo(workspace_test_dir) -> None:
    """«path relativo tmp/x.txt» viene ridotto al path effettivo."""
    reset_stato_turno()
    path_rel = f"{DIR_UNIT_REL}/prefisso.txt"
    osservazione = scrivi_file_da_testo(
        f"path relativo {path_rel}\n{SEPARATORE_FILE}\nok"
    )
    assert osservazione.startswith("File scritto con successo:"), osservazione
    assert path_rel in osservazione
    assert (WORKSPACE / path_rel).read_text(encoding="utf-8") == "ok\n"


def test_scrivi_file_rimuove_artefatto_riga_separatore(workspace_test_dir) -> None:
    """Una prima riga «riga ---» non deve finire nel file."""
    reset_stato_turno()
    path_rel = f"{DIR_UNIT_REL}/riga_sep.txt"
    osservazione = scrivi_file_da_testo(
        f"{path_rel}\nriga ---\ncorpo vero\n"
    )
    assert osservazione.startswith("File scritto con successo:"), osservazione
    assert (WORKSPACE / path_rel).read_text(encoding="utf-8") == "corpo vero\n"


def test_scrivi_file_accetta_html_incompleto(workspace_test_dir) -> None:
    """Un .html è un file di testo: anche senza </html> viene persistito."""
    reset_stato_turno()
    path_rel = f"{DIR_UNIT_REL}/troncato.txt"
    body = "<!DOCTYPE html>\n<html lang=\"it\">\n<body><nav>x</nav>\n"
    osservazione = scrivi_file_da_testo(_input_scrivi(path_rel, body))
    assert osservazione.startswith("File scritto con successo:"), osservazione
    assert (WORKSPACE / path_rel).read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_scrivi_file_overwrite_html_corto(workspace_test_dir) -> None:
    """Uno stub HTML corto può sovrascrivere un documento più lungo."""
    reset_stato_turno()
    path_rel = f"{DIR_UNIT_REL}/sito.txt"
    originale = (
        "<!DOCTYPE html>\n<html lang=\"it\"><head><title>Zeroglut</title></head>"
        "<body>" + ("<p>contenuto</p>" * 40) + "</body></html>\n"
    )
    assert scrivi_file_da_testo(_input_scrivi(path_rel, originale)).startswith(
        "File scritto con successo:"
    )
    stub = (
        "<!DOCTYPE html>\n<html><head><title>Sito Web</title></head>"
        "<body><nav class=\"navbar\"></nav></body></html>\n"
    )
    osservazione = scrivi_file_da_testo(_input_scrivi(path_rel, stub))
    assert osservazione.startswith("File scritto con successo:"), osservazione
    assert (WORKSPACE / path_rel).read_text(encoding="utf-8") == stub


def test_scrivi_file_accetta_overwrite_html_taglia_simile(workspace_test_dir) -> None:
    """Una riscrittura HTML completa di taglia simile è overwrite valido."""
    reset_stato_turno()
    path_rel = f"{DIR_UNIT_REL}/landing.txt"
    originale = (
        "<!DOCTYPE html>\n<html lang=\"it\"><head><title>Zeroglut</title></head>"
        "<body>" + ("<p>contenuto</p>" * 20) + "</body></html>\n"
    )
    assert scrivi_file_da_testo(_input_scrivi(path_rel, originale)).startswith(
        "File scritto con successo:"
    )
    aggiornato = originale.replace("contenuto", "sezione")
    osservazione = scrivi_file_da_testo(_input_scrivi(path_rel, aggiornato))
    assert osservazione.startswith("File scritto con successo:"), osservazione
    assert (WORKSPACE / path_rel).read_text(encoding="utf-8") == aggiornato


def test_scrivi_file_recupera_estensione_dal_thought(workspace_test_dir) -> None:
    """
    Se la prima riga omette l'estensione ma il Thought cita path.ext, scrive path.ext.
    """
    reset_stato_turno()
    path_rel = f"{DIR_UNIT_REL}/a_duo.txt"
    action_input = (
        f"{DIR_UNIT_REL}/a_duo\n"
        "Contenuto: AAA\n"
        f"Thought: creo il file {path_rel} con AAA"
    )
    osservazione = scrivi_file_da_testo(action_input)
    assert osservazione.startswith("File scritto con successo:"), osservazione
    assert path_rel in osservazione
    assert (WORKSPACE / path_rel).read_text(encoding="utf-8") == "AAA\n"
    assert not (WORKSPACE / f"{DIR_UNIT_REL}/a_duo").exists()


def test_modifica_file_roundtrip(workspace_test_dir) -> None:
    """Protocollo path/OLD/NEW sostituisce un blocco unico sul disco."""
    reset_stato_turno()
    path_rel = f"{DIR_UNIT_REL}/patch.txt"
    scrivi_file_da_testo(_input_scrivi(path_rel, "def foo():\n    return 1"))
    osservazione = modifica_file_da_testo(
        _input_modifica(path_rel, "    return 1", "    return 2")
    )
    assert osservazione.startswith("File modificato con successo:"), osservazione
    assert path_rel in osservazione
    assert (WORKSPACE / path_rel).read_text(encoding="utf-8") == (
        "def foo():\n    return 2\n"
    )


def test_modifica_file_taglia_coda_old_nel_new(workspace_test_dir) -> None:
    """Un NEW seguito da <<<OLD spurio non deve finire nel file."""
    reset_stato_turno()
    path_rel = f"{DIR_UNIT_REL}/nav.txt"
    originale = "nav a {\n    padding: 0.5rem 0.9rem;\n}\n"
    assert scrivi_file_da_testo(_input_scrivi(path_rel, originale)).startswith(
        "File scritto con successo:"
    )
    osservazione = modifica_file_da_testo(
        _input_modifica(
            path_rel,
            "    padding: 0.5rem 0.9rem;",
            "    padding: 0.75rem 1.1rem;\n<<<OLD\n    garbage corrotta;",
        )
    )
    assert osservazione.startswith("File modificato con successo:"), osservazione
    testo = (WORKSPACE / path_rel).read_text(encoding="utf-8")
    assert "<<<OLD" not in testo
    assert "garbage" not in testo
    assert "padding: 0.75rem 1.1rem;" in testo


def test_modifica_file_rifiuta_formato_malformato(workspace_test_dir) -> None:
    """Senza marker <<<OLD/<<<NEW il tool non tocca il disco."""
    reset_stato_turno()
    path_rel = f"{DIR_UNIT_REL}/patch.txt"
    scrivi_file_da_testo(_input_scrivi(path_rel, "ciao"))
    osservazione = modifica_file_da_testo(f"{path_rel}\n---\nciao")
    assert osservazione.startswith("Errore:"), osservazione
    assert (WORKSPACE / path_rel).read_text(encoding="utf-8") == "ciao\n"


def test_modifica_file_new_vuoto_cancella_blocco(workspace_test_dir) -> None:
    """NEW vuoto elimina il blocco OLD se il match è unico."""
    reset_stato_turno()
    path_rel = f"{DIR_UNIT_REL}/trim.txt"
    scrivi_file_da_testo(_input_scrivi(path_rel, "keep\nremove\nkeep"))
    osservazione = modifica_file_da_testo(_input_modifica(path_rel, "remove\n", ""))
    assert osservazione.startswith("File modificato con successo:"), osservazione
    assert (WORKSPACE / path_rel).read_text(encoding="utf-8") == "keep\nkeep\n"


def test_modifica_file_rifiuta_traversal(workspace_test_dir) -> None:
    """Path fuori sandbox restano errori, come scrivi_file."""
    reset_stato_turno()
    osservazione = modifica_file_da_testo(
        _input_modifica("../escape.txt", "old", "new")
    )
    assert osservazione.startswith("Errore:"), osservazione


def test_crea_cartella_bloccata_senza_intent(workspace_test_dir) -> None:
    """Gate chiuso: niente mkdir."""
    reset_stato_turno()
    imposta_scrittura_consentita(False)
    osservazione = crea_cartella_da_testo(f"{DIR_UNIT_REL}/nope")
    assert osservazione.startswith("Errore: scrittura non richiesta"), osservazione
    assert not (WORKSPACE / DIR_UNIT_REL / "nope").exists()


def test_elenca_albero_e_trova_file(workspace_test_dir) -> None:
    """Albero a profondità 2 e glob sul nome."""
    reset_stato_turno()
    nested = f"{DIR_UNIT_REL}/docs/sub/note.txt"
    scrivi_file_da_testo(_input_scrivi(nested, "hello"))
    albero = elenca_albero_da_testo(DIR_UNIT_REL)
    assert "docs/" in albero
    trovati = trova_file_da_testo("*.txt")
    assert "note.txt" in trovati


@pytest.mark.parametrize(
    ("action_input", "atteso"),
    [
        ("Articolo_01.txtTrovato il file Articolo_01.txt. Procedo alla lettura.", "krill"),
        ("Articolo_01.txt Procedo subito alla lettura del file", "krill"),
    ],
)
def test_leggi_documento_recupera_path_con_prosa_incollata(
    tmp_path: Path, action_input: str, atteso: str
) -> None:
    """Regressione: Gemma prosegue sulla riga dell'Action Input e brucia iterazioni."""
    (tmp_path / "Articolo_01.txt").write_text("la balena mangia krill", encoding="utf-8")
    service = WorkspaceFileService(FileConfig(workspace=tmp_path))
    assert atteso in leggi_documento_da_testo(action_input, service=service)


@pytest.mark.parametrize(
    "action_input",
    [
        ".Tuttavia, non vedo il file tra i documenti della cartella.",
        ".elenca_cartella Action Input obbligatorio: attuale. Mai la root.",
        "attualeTuttavia, non hai fornito l'output del tool",
    ],
)
def test_elenca_cartella_recupera_root_con_prosa_incollata(
    tmp_path: Path, action_input: str
) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    service = WorkspaceFileService(FileConfig(workspace=tmp_path))
    assert "a.txt" in elenca_cartella_da_testo(action_input, service=service)


def test_nome_con_spazi_non_viene_troncato(tmp_path: Path) -> None:
    """«notes vecchie» inesistente non deve diventare silenziosamente «notes»."""
    (tmp_path / "notes").mkdir()
    service = WorkspaceFileService(FileConfig(workspace=tmp_path))
    assert elenca_cartella_da_testo("notes vecchie", service=service).startswith("Errore")


def test_lettura_ripetuta_di_fila_avvisa_il_modello(tmp_path: Path) -> None:
    """Gemma ripeteva elenca_cartella '.' fino al limite iterazioni."""
    from assistente.tools import crea_tool_file

    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    service = WorkspaceFileService(FileConfig(workspace=tmp_path))
    service.begin_turn(write_allowed=False)
    elenca = next(t for t in crea_tool_file(service, write_allowed=False) if t.name == "elenca_cartella")
    primo = elenca.func(".")
    secondo = elenca.func(".")
    assert not primo.startswith("Stessa Action")
    assert secondo.startswith("Stessa Action") and "a.txt" in secondo
    service.begin_turn(write_allowed=False)
    assert not elenca.func(".").startswith("Stessa Action")
