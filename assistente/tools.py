"""Adapter LangChain sottili per WorkspaceFileService e fallback controllato."""

from __future__ import annotations

import re
from contextvars import ContextVar
from pathlib import Path

from langchain_core.tools import Tool

from assistente.config import (
    LISTING_MAX_RESULT_CHARS,
    MARCATORE_BLOCCO_NEW,
    MARCATORE_BLOCCO_OLD,
    SEPARATORE_FILE,
)
from assistente.file_service import (
    DirectoryOperationResult,
    SearchResult,
    WorkspaceFileService,
)
from assistente.react_parser import _taglia_coda_protocollo
from assistente.tool_contract import (
    ToolContract,
    build_tools,
    validate_non_empty,
)

DESCRIZIONE_TOOL_SCRIVI_FILE = (
    f"Crea o sovrascrive un file .txt. Input: path relativo "
    f"(note/oggi.txt, non /note/oggi.txt), riga {SEPARATORE_FILE}, body grezzo. "
    "Solo estensione .txt. Un file per Action. Niente virgole nel path."
)
DESCRIZIONE_TOOL_MODIFICA_FILE = (
    f"Modifica un file .txt esistente. Input: path, riga {MARCATORE_BLOCCO_OLD}, "
    f"blocco esatto, riga {MARCATORE_BLOCCO_NEW}, blocco nuovo. "
    "Un match unico. NEW può essere vuoto."
)
DESCRIZIONE_TOOL_LEGGI_DOCUMENTO = (
    "Legge un documento della cartella: txt, md, pdf, docx, pages o immagine "
    "(png, jpg, webp: trascrizione e descrizione). Input: path relativo esatto."
)
DESCRIZIONE_TOOL_CERCA_TESTO = (
    "Cerca parole nel contenuto di tutti i documenti (txt, md, pdf, docx, pages; "
    "non le immagini). Input: poche parole chiave. "
    "Restituisce file, riga o pagina ed estratto, i più pertinenti per primi."
)
DESCRIZIONE_TOOL_ELENCA_CARTELLA = (
    "Elenca una cartella del workspace (un livello). "
    "Input: . per la cartella attuale, altrimenti il path relativo esatto "
    "della sottocartella nominata (es. notes)."
)
DESCRIZIONE_TOOL_CREA_CARTELLA = (
    "Crea una cartella relativa, con i parent se mancano. "
    "Input: path relativo (notes/2026). exist_ok se è già una cartella."
)
DESCRIZIONE_TOOL_ELENCA_ALBERO = (
    "Elenca una cartella fino a due livelli, senza venv/.git. "
    "Input: . per la cartella attuale, altrimenti path relativo (es. output)."
)
DESCRIZIONE_TOOL_TROVA_FILE = (
    "Trova file o cartelle per nome. Input: glob (*.pdf) o frammento (brief). "
    "Per il contenuto usa cerca_testo."
)

_SUCCESS_PREFIX = "File scritto con successo:"
_MODIFY_SUCCESS_PREFIX = "File modificato con successo:"
_MKDIR_SUCCESS_PREFIX = "Cartella creata:"
_MARKER_OLD_LINE_PATTERN = re.compile(
    rf"^{re.escape(MARCATORE_BLOCCO_OLD)}\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_MARKER_NEW_LINE_PATTERN = re.compile(
    rf"^{re.escape(MARCATORE_BLOCCO_NEW)}\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_PATH_PREFIX_PATTERN = re.compile(r"^(?:path|file)\s*[:=]\s*", re.IGNORECASE)
_INSTRUCTIONAL_PATH_PREFIX_PATTERN = re.compile(
    r"^(?:path|percorso)\s+relativo\s+",
    re.IGNORECASE,
)
_ROW_PREFIX_PATTERN = re.compile(r"^riga\s*\d+\s*=\s*", re.IGNORECASE)
_ROW_SEPARATOR_ARTIFACT_PATTERN = re.compile(
    rf"^riga\s*{re.escape(SEPARATORE_FILE)}\s*$",
    re.IGNORECASE,
)
_ROW_SUFFIX_PATTERN = re.compile(r",\s*riga\s*\d+\s*=.*$", re.IGNORECASE)
_CONTENT_PREFIX_PATTERN = re.compile(
    r"^(?:content|contenuto)\s*:\s*",
    re.IGNORECASE,
)
_OPENING_FENCE_PATTERN = re.compile(r"^```[^\n]*\n")
_CLOSING_FENCE_PATTERN = re.compile(r"\n?```\s*$")
_AGENT_ARTIFACT_PATTERN = re.compile(
    r"(?:^|\n---\s*|\n)(?:Thought|Action|Action Input)\s*:",
    re.IGNORECASE | re.MULTILINE,
)
# Gemma spesso emette monolinea ``path.ext,contenuto`` al posto di path/---/body.
_SINGLE_LINE_PATH_CONTENT_PATTERN = re.compile(
    r"^(?P<path>(?:[\w./-]+/)*[\w.-]+\.[A-Za-z0-9]+)\s*,\s*(?P<content>.+)$"
)
# Path con estensione citati nel Thought/body quando la prima riga omette l'estensione.
_PATH_WITH_EXTENSION_PATTERN = re.compile(
    r"(?P<path>[\w./-]+\.[A-Za-z0-9]+)"
)
_YAML_BLOCK_MARKERS = frozenset({"|", ">", "|-", ">-", "|+", ">+"})
_PROTOCOL_ONLY_CONTENT_PATTERN = re.compile(
    r"^(?:Thought|Action|Action Input)\s*:",
    re.IGNORECASE,
)
_PATH_AND_FENCE_PATTERN = re.compile(
    r"(?ms)^(?P<path>[^\n:`]+?)\s*:?\s*```[^\n]*\n"
    r"(?P<content>.*?)```"
)
_PATH_AND_SEPARATOR_PATTERN = re.compile(
    rf"(?ms)^(?P<path>[^\n]+)\n{re.escape(SEPARATORE_FILE)}\n"
    rf"(?P<content>.*?)(?=^[^\n]+\n{re.escape(SEPARATORE_FILE)}\n|\Z)"
)
_SEPARATOR_LINE_PATTERN = re.compile(
    rf"\n\s*{re.escape(SEPARATORE_FILE)}\s*(?:\n|$)",
)
_KNOWN_LANGUAGE_MARKERS = frozenset(
    {
        "bash",
        "css",
        "html",
        "java",
        "javascript",
        "js",
        "json",
        "markdown",
        "md",
        "py",
        "python",
        "sh",
        "ts",
        "txt",
        "typescript",
    }
)
_MAX_EXPLICIT_PATH_CHARACTERS = 240

_CONTEXT_SERVICE: ContextVar[WorkspaceFileService | None] = ContextVar(
    "compat_file_service",
    default=None,
)

AVVISO_LETTURA_RIPETUTA = (
    "Stessa Action del passo precedente: il risultato è già questo. "
    "Non ripeterla: scrivi ora Final Answer.\n"
)


class _FileToolAdapter:
    """Bound methods a singolo argomento, senza esporre il servizio al prompt LLM."""

    def __init__(self, service: WorkspaceFileService) -> None:
        """
        Collega tutti i tool file alla stessa boundary e allo stesso stato turno.

        Args:
            service: Servizio posseduto dalla sessione corrente.
        """
        self._service = service

    def _con_anti_loop(self, tool: str, action_input: str, esito: str) -> str:
        """Antepone un avviso se il modello ripete la stessa lettura di fila."""
        if self._service.segna_lettura(tool, action_input):
            return f"{AVVISO_LETTURA_RIPETUTA}{esito}"
        return esito

    def write(self, action_input: str) -> str:
        """
        Esegue scrivi_file sul servizio associato.

        Returns:
            Observation testuale compatibile con ReAct.
        """
        return scrivi_file_da_testo(action_input, service=self._service)

    def replace(self, action_input: str) -> str:
        """
        Esegue modifica_file sul servizio associato.

        Returns:
            Observation testuale compatibile con ReAct.
        """
        return modifica_file_da_testo(action_input, service=self._service)

    def read(self, action_input: str) -> str:
        """
        Esegue leggi_documento sul servizio associato.

        Returns:
            Contenuto o diagnostica testuale.
        """
        esito = leggi_documento_da_testo(
            action_input,
            service=self._service,
        )
        return self._con_anti_loop("read", action_input, esito)

    def list(self, action_input: str) -> str:
        """
        Esegue elenca_cartella sul servizio associato.

        Returns:
            Listing o diagnostica testuale.
        """
        esito = elenca_cartella_da_testo(action_input, service=self._service)
        return self._con_anti_loop("list", action_input, esito)

    def mkdir(self, action_input: str) -> str:
        """Esegue crea_cartella sul servizio associato."""
        return crea_cartella_da_testo(action_input, service=self._service)

    def tree(self, action_input: str) -> str:
        """Esegue elenca_albero sul servizio associato."""
        esito = elenca_albero_da_testo(action_input, service=self._service)
        return self._con_anti_loop("tree", action_input, esito)

    def find(self, action_input: str) -> str:
        """Esegue trova_file sul servizio associato."""
        esito = trova_file_da_testo(action_input, service=self._service)
        return self._con_anti_loop("find", action_input, esito)

    def search(self, action_input: str) -> str:
        """Esegue cerca_testo sul servizio associato."""
        esito = cerca_testo_da_testo(action_input, service=self._service)
        return self._con_anti_loop("search", action_input, esito)

def get_default_file_service() -> WorkspaceFileService:
    """
    Restituisce un servizio di compatibilità isolato nel ContextVar corrente.

    Returns:
        Servizio autorizzato per chiamate dirette legacy; le sessioni usano istanze esplicite.
    """
    service = _CONTEXT_SERVICE.get()
    # Ogni contesto crea il proprio stato, evitando contaminazioni tra thread o task.
    if service is None:
        service = WorkspaceFileService()
        service.begin_turn(write_allowed=True)
        _CONTEXT_SERVICE.set(service)
    return service

def reset_stato_turno(service: WorkspaceFileService | None = None) -> None:
    """
    Azzera tracking legacy o il servizio esplicito mantenendo autorizzazione diretta.

    Args:
        service: Istanza da azzerare; None sostituisce quella del ContextVar corrente.
    """
    # Le chiamate dirette dei test restano autorizzate; SessioneAgente applica il proprio gate.
    if service is not None:
        service.begin_turn(write_allowed=True)
        return
    replacement = WorkspaceFileService()
    replacement.begin_turn(write_allowed=True)
    _CONTEXT_SERVICE.set(replacement)

def imposta_scrittura_consentita(
    consentito: bool,
    service: WorkspaceFileService | None = None,
) -> None:
    """
    Avvia un nuovo stato turno con il gate esplicito richiesto.

    Args:
        consentito: Decisione della policy utente.
        service: Servizio di sessione; None usa l'adapter legacy context-local.
    """
    _selected_service(service).begin_turn(write_allowed=consentito)

def scrittura_consentita_turno(
    service: WorkspaceFileService | None = None,
) -> bool:
    """
    Espone il gate corrente per test e adapter di compatibilità.

    Returns:
        Stato write_allowed del servizio selezionato.
    """
    return _selected_service(service).write_allowed

def normalizza_path_da_llm(
    path_grezzo: str,
    service: WorkspaceFileService | None = None,
) -> str | None:
    """
    Ripulisce prefissi tipici del modello e delega la validazione al servizio.

    Returns:
        Path file relativo normalizzato oppure None.
    """
    cleaned = _clean_raw_path(path_grezzo)
    return _selected_service(service).normalize_file_path(cleaned)

def normalizza_path_cartella_da_llm(
    path_grezzo: str,
    service: WorkspaceFileService | None = None,
) -> str | None:
    """
    Normalizza un path cartella proveniente da Action Input.

    Returns:
        Path relativo, stringa vuota per root oppure None.
    """
    cleaned = _clean_raw_path(path_grezzo)
    return _selected_service(service).normalize_directory_path(cleaned)

def valida_path_relativo(
    path_relativo: str,
    service: WorkspaceFileService | None = None,
) -> Path | None:
    """
    Mantiene l'API storica restituendo il Path assoluto solo dopo il sandbox.

    Returns:
        Destinazione assoluta oppure None.
    """
    resolved = _selected_service(service).resolve_file(path_relativo)
    return resolved.absolute if resolved is not None else None

def valida_path_cartella_relativo(
    path_relativo: str,
    service: WorkspaceFileService | None = None,
) -> Path | None:
    """
    Risolve una cartella relativa nel workspace selezionato.

    Returns:
        Directory assoluta oppure None.
    """
    resolved = _selected_service(service).resolve_directory(path_relativo)
    return resolved.absolute if resolved is not None else None

def path_valido_per_workspace(
    path_relativo: str,
    service: WorkspaceFileService | None = None,
) -> bool:
    """
    Verifica sintassi e sandbox con un'unica risoluzione.

    Returns:
        True quando il servizio accetterebbe il path.
    """
    return _selected_service(service).resolve_file(path_relativo) is not None

def path_da_osservazione_scrittura(
    osservazione: str,
    service: WorkspaceFileService | None = None,
) -> str | None:
    """
    Estrae il path da un messaggio tool storico, per compatibilità esterna.

    Returns:
        Path normalizzato soltanto per osservazioni di successo.
    """
    # Errori e messaggi di altri tool non devono produrre falsi path.
    if not osservazione.startswith(_SUCCESS_PREFIX):
        return None
    # Solo la prima riga contiene il path; eventuali hint stanno sotto.
    raw_path = (
        osservazione.removeprefix(_SUCCESS_PREFIX).strip().splitlines()[0].strip()
    )
    return normalizza_path_da_llm(raw_path, service)

def sanitizza_contenuto_file(contenuto: str, path_relativo: str) -> str:
    """
    Rimuove soltanto wrapper ReAct esterni e uniforma una newline finale.

    Args:
        contenuto: Body separato dal path.
        path_relativo: Conservato nell'API per compatibilità; nessuna riparazione per estensione.

    Returns:
        Contenuto preservato senza fence esterne o artefatti successivi.
    """
    del path_relativo
    text = contenuto.strip("\n")
    text = _OPENING_FENCE_PATTERN.sub("", text)
    text = _CLOSING_FENCE_PATTERN.sub("", text)
    # "Contenuto: |" lascia il marker YAML block-scalar come prima riga spuria.
    lines = text.split("\n", 1)
    if lines and lines[0].strip() in _YAML_BLOCK_MARKERS:
        text = lines[1] if len(lines) > 1 else ""
    # «riga ---» finisce nel body quando il modello copia le istruzioni.
    lines = text.split("\n", 1)
    if lines and _ROW_SEPARATOR_ARTIFACT_PATTERN.match(lines[0].strip()):
        text = lines[1] if len(lines) > 1 else ""
    artifact = _AGENT_ARTIFACT_PATTERN.search(text)
    # Taglia soltanto protocollo aggiunto dopo il body, mai stringhe valide interne al file.
    if artifact is not None:
        text = text[: artifact.start()].rstrip("\n")
    return text.rstrip("\n") + "\n"


def espandi_newline_letterali(testo: str) -> str:
    """
    Espande ``\\n`` soltanto negli Action Input monolinea prodotti da Gemma.

    Returns:
        Testo invariato se contiene già newline reali, altrimenti forma multilinea.
    """
    # Newline reali indicano che le sequenze escape nel body possono essere intenzionali.
    if "\\n" not in testo or "\n" in testo:
        return testo
    return testo.replace("\\n", "\n")

def scrivi_file_da_testo(
    action_input: str,
    *,
    service: WorkspaceFileService | None = None,
) -> str:
    """
    Parsea il protocollo ``path / --- / contenuto`` e delega la write atomica.

    Returns:
        Observation stabile di successo oppure messaggio che inizia con Errore:.
    """
    selected_service = _selected_service(service)
    raw_text = str(action_input or "").strip()
    # Input vuoti non arrivano al servizio perché non contengono path né body.
    if not raw_text:
        return (
            "Errore: input vuoto. Formato: prima riga = path, "
            f"seconda riga = {SEPARATORE_FILE}, poi contenuto."
        )
    parsed = _split_path_and_content(raw_text, selected_service)
    # Un formato ambiguo non deve tentare di dedurre path dal contenuto.
    if parsed is None:
        return _messaggio_errore_formato(raw_text)
    relative_path, raw_content = parsed
    # File vuoti richiedono una richiesta esplicita diversa; evitano write accidentali.
    if not raw_content.strip():
        return "Errore: contenuto mancante."

    clean_content = sanitizza_contenuto_file(raw_content, relative_path)
    # Body ridotto a solo protocollo ReAct: tipico Action Input concatenato senza ---.
    if not clean_content.strip() or _PROTOCOL_ONLY_CONTENT_PATTERN.match(
        clean_content.strip()
    ):
        return (
            "Errore: contenuto mancante o solo protocollo ReAct.\n"
            f"Usa:\npath_rel\n{SEPARATORE_FILE}\ncontenuto_completo\n"
            "(un file per Action; niente Action concatenate)."
        )
    result = selected_service.write_text(relative_path, clean_content)
    # I risultati tipizzati vengono convertiti in stringa soltanto al confine LangChain.
    if not result.succeeded:
        return f"Errore: {result.error}"
    # Seconda riga: spinge Gemma a non dichiarare altri file senza Observation.
    return (
        f"{_SUCCESS_PREFIX} {result.path}\n"
        "Se serve un altro file, ripeti Action: scrivi_file (un path per Action)."
    )

def modifica_file_da_testo(
    action_input: str,
    *,
    service: WorkspaceFileService | None = None,
) -> str:
    """
    Parsea ``path / <<<OLD / blocco / <<<NEW / blocco`` e delega la patch atomica.

    Returns:
        Observation stabile di successo oppure messaggio che inizia con Errore:.
    """
    selected_service = _selected_service(service)
    raw_text = str(action_input or "").strip()
    if not raw_text:
        return (
            "Errore: input vuoto. Formato: path, "
            f"riga {MARCATORE_BLOCCO_OLD}, blocco, "
            f"riga {MARCATORE_BLOCCO_NEW}, blocco nuovo."
        )
    parsed = _split_path_old_new(raw_text, selected_service)
    if parsed is None:
        return (
            "Errore: formato non valido. "
            f"Usa: path_rel\\n{MARCATORE_BLOCCO_OLD}\\nblocco_esatto\\n"
            f"{MARCATORE_BLOCCO_NEW}\\nblocco_nuovo"
        )
    relative_path, old_block, new_block = parsed
    if old_block == "":
        return "Errore: blocco OLD vuoto. Copia il testo esatto da sostituire."
    if MARCATORE_BLOCCO_OLD in new_block or MARCATORE_BLOCCO_NEW in new_block:
        return (
            "Errore: blocco NEW contiene ancora marker <<<OLD/<<<NEW. "
            "Ripeti modifica_file con NEW uguale solo al codice da scrivere."
        )
    result = selected_service.replace_text(relative_path, old_block, new_block)
    if not result.succeeded:
        return f"Errore: {result.error}"
    return (
        f"{_MODIFY_SUCCESS_PREFIX} {result.path}\n"
        "Se serve un altro file, ripeti Action: modifica_file (un path per Action)."
    )

def leggi_documento_da_testo(
    action_input: str,
    *,
    service: WorkspaceFileService | None = None,
) -> str:
    """
    Legge un documento tramite il servizio sandboxed.

    Returns:
        Contenuto troncato oppure messaggio Errore:.
    """
    selected = _selected_service(service)
    path = recupera_path_esistente(action_input, selected, cartella=False)
    result = selected.read_document(path)
    return result.content if result.succeeded else f"Errore: {result.error}"

def elenca_cartella_da_testo(
    action_input: str,
    *,
    service: WorkspaceFileService | None = None,
) -> str:
    """
    Elenca una cartella tramite il servizio e formatta entry relative.

    Returns:
        Listing testuale stabile oppure messaggio Errore:.
    """
    selected = _selected_service(service)
    result = selected.list_directory(
        recupera_path_esistente(action_input, selected, cartella=True)
    )
    # La diagnostica tipizzata non deve essere confusa con una cartella vuota.
    if not result.succeeded:
        return f"Errore: {result.error}"
    return _formatta_elenco(
        result,
        titolo=f"Contenuto di '{result.path}'",
        empty_message=f"Cartella vuota: {result.path}",
    )

def crea_cartella_da_testo(
    action_input: str,
    *,
    service: WorkspaceFileService | None = None,
) -> str:
    """Crea una cartella relativa tramite il servizio sandboxed."""
    result = _selected_service(service).create_directory(
        _clean_raw_path(str(action_input or ""))
    )
    if not result.succeeded:
        return f"Errore: {result.error}"
    return f"{_MKDIR_SUCCESS_PREFIX} {result.path}"

def elenca_albero_da_testo(
    action_input: str,
    *,
    service: WorkspaceFileService | None = None,
) -> str:
    """Elenca una cartella a profondità fissa."""
    selected = _selected_service(service)
    result = selected.list_tree(
        recupera_path_esistente(action_input, selected, cartella=True)
    )
    if not result.succeeded:
        return f"Errore: {result.error}"
    return _formatta_elenco(
        result,
        titolo=f"Albero di '{result.path}'",
        empty_message=f"Albero vuoto: {result.path}",
    )

def trova_file_da_testo(
    action_input: str,
    *,
    service: WorkspaceFileService | None = None,
) -> str:
    """Cerca file e cartelle per nome."""
    result = _selected_service(service).find_files(str(action_input or "").strip())
    if not result.succeeded:
        return f"Errore: {result.error}"
    return _formatta_elenco(
        result,
        titolo=f"Trovati per '{result.path}'",
        empty_message=f"Nessun risultato: {result.path}",
    )

def cerca_testo_da_testo(
    action_input: str,
    *,
    service: WorkspaceFileService | None = None,
) -> str:
    """Cerca parole nel contenuto dei documenti della cartella."""
    result = _selected_service(service).search_text(str(action_input or ""))
    if not result.succeeded:
        return f"Errore: {result.error}"
    return _formatta_ricerca(result)

def _formatta_ricerca(result: SearchResult) -> str:
    """Rende un SearchResult in Observation stabile: file, posizione, estratto."""
    if not result.matches:
        testo = f"Nessun documento contiene: {result.query}"
    else:
        lines = [f"Trovato '{result.query}' in:"]
        for match in result.matches:
            lines.append(f"  {match.relative_path} ({match.location}): {match.snippet}")
        if result.truncated:
            lines.append("  […risultati troncati: restringi la ricerca]")
        testo = "\n".join(lines)
    if result.skipped:
        saltati = ", ".join(result.skipped[:5])
        testo += f"\nNon leggibili per la ricerca: {saltati}"
    # Senza risultati, le immagini sono l'unico contenuto ancora da guardare.
    if not result.matches and result.images:
        immagini = ", ".join(result.images[:10])
        testo += (
            f"\nImmagini non coperte dalla ricerca: {immagini}. "
            "Leggi con leggi_documento solo quelle pertinenti al tema."
        )
    return testo

def _formatta_elenco(
    result: DirectoryOperationResult,
    *,
    titolo: str,
    empty_message: str,
) -> str:
    """Rende un DirectoryOperationResult in Observation stabile."""
    if not result.entries:
        return empty_message
    lines = [f"{titolo}:"]
    for entry in result.entries:
        if entry.is_directory:
            lines.append(f"  [cartella] {entry.relative_path}/")
        else:
            lines.append(f"  {entry.relative_path} ({entry.size_bytes} byte)")
    if result.truncated:
        lines.append("  […elenco troncato]")
    return "\n".join(lines)

def crea_tool_file(
    service: WorkspaceFileService | None = None,
    *,
    write_allowed: bool | None = None,
    workspace_label: str | None = None,
) -> list[Tool]:
    """
    Registra adapter file legati all'istanza di sessione.

    Args:
        service: Boundary workspace della sessione.
        write_allowed: False omette i tool di mutazione; True annota overwrite.
        workspace_label: Root breve mostrata nelle descrizioni dinamiche.
    """
    selected_service = _selected_service(service)
    adapter = _FileToolAdapter(selected_service)
    root_hint = workspace_label or selected_service.workspace.name
    listing_contract = dict(max_result_chars=LISTING_MAX_RESULT_CHARS)
    read_specs = (
        (
            ToolContract(
                name="leggi_documento",
                is_read_only=True,
                is_concurrency_safe=True,
                validate=validate_non_empty,
                description=f"{DESCRIZIONE_TOOL_LEGGI_DOCUMENTO} Workspace: {root_hint}.",
            ),
            adapter.read,
        ),
        (
            ToolContract(
                name="elenca_cartella",
                is_read_only=True,
                is_concurrency_safe=True,
                **listing_contract,
                description=f"{DESCRIZIONE_TOOL_ELENCA_CARTELLA} Workspace: {root_hint}.",
            ),
            adapter.list,
        ),
        (
            ToolContract(
                name="elenca_albero",
                is_read_only=True,
                is_concurrency_safe=True,
                **listing_contract,
                description=f"{DESCRIZIONE_TOOL_ELENCA_ALBERO} Workspace: {root_hint}.",
            ),
            adapter.tree,
        ),
        (
            ToolContract(
                name="trova_file",
                is_read_only=True,
                is_concurrency_safe=True,
                validate=validate_non_empty,
                **listing_contract,
                description=f"{DESCRIZIONE_TOOL_TROVA_FILE} Workspace: {root_hint}.",
            ),
            adapter.find,
        ),
        (
            ToolContract(
                name="cerca_testo",
                is_read_only=True,
                is_concurrency_safe=True,
                validate=validate_non_empty,
                **listing_contract,
                description=f"{DESCRIZIONE_TOOL_CERCA_TESTO} Workspace: {root_hint}.",
            ),
            adapter.search,
        ),
    )
    if write_allowed is False:
        return build_tools(read_specs)
    if write_allowed is True:
        write_suffix = (
            f" Workspace: {root_hint}. Scrittura autorizzata in questo turno."
        )
    else:
        write_suffix = f" Workspace: {root_hint}."
    write_specs = (
        (
            ToolContract(
                name="scrivi_file",
                is_read_only=False,
                is_concurrency_safe=False,
                validate=validate_non_empty,
                description=f"{DESCRIZIONE_TOOL_SCRIVI_FILE}{write_suffix}",
            ),
            adapter.write,
        ),
        (
            ToolContract(
                name="modifica_file",
                is_read_only=False,
                is_concurrency_safe=False,
                validate=validate_non_empty,
                description=f"{DESCRIZIONE_TOOL_MODIFICA_FILE}{write_suffix}",
            ),
            adapter.replace,
        ),
        (
            ToolContract(
                name="crea_cartella",
                is_read_only=False,
                is_concurrency_safe=False,
                validate=validate_non_empty,
                description=f"{DESCRIZIONE_TOOL_CREA_CARTELLA}{write_suffix}",
            ),
            adapter.mkdir,
        ),
    )
    return build_tools((*write_specs, *read_specs))

def estrai_path_espliciti(
    testo: str,
    service: WorkspaceFileService | None = None,
) -> list[tuple[str, str]]:
    """
    Estrae solo blocchi espliciti path+fence o path+separatore dalla Final Answer.

    Returns:
        Coppie uniche con path già normalizzato e contenuto non vuoto.
    """
    selected_service = _selected_service(service)
    seen_paths: set[str] = set()
    extracted: list[tuple[str, str]] = []

    def append_if_valid(raw_path: str, block: str) -> None:
        """
        Aggiunge una coppia soltanto se la riga sembra un path, non una frase.

        Postconditions:
            Ogni path compare al massimo una volta e conserva il primo contenuto.
        """
        cleaned_path = raw_path.strip().rstrip(":").strip()
        # Lunghezza, punteggiatura e marker elenco riducono falsi positivi da testo naturale.
        if (
            not cleaned_path
            or len(cleaned_path) > _MAX_EXPLICIT_PATH_CHARACTERS
            or any(marker in cleaned_path for marker in (":", "=", "`"))
            or cleaned_path.startswith(("-", "*"))
        ):
            return
        normalized = selected_service.normalize_file_path(cleaned_path)
        # Path invalidi, duplicati o body vuoti non sono fallback materializzabili.
        if normalized is None or normalized in seen_paths or not block.strip():
            return
        seen_paths.add(normalized)
        extracted.append((normalized, block))

    # I fence sono il formato più comune quando il modello salta il tool.
    for match in _PATH_AND_FENCE_PATTERN.finditer(testo):
        append_if_valid(match.group("path"), match.group("content"))
    # Il protocollo nativo path/---/contenuto ha precedenza solo per path non già visti.
    for match in _PATH_AND_SEPARATOR_PATTERN.finditer(testo):
        append_if_valid(match.group("path"), match.group("content"))
    return extracted

def applica_fallback(
    risultato: str,
    path_scritti_turno: list[str],
    *,
    service: WorkspaceFileService | None = None,
) -> list[str]:
    """
    Materializza file mancanti dalla Final Answer soltanto con gate esplicito.

    Args:
        risultato: Output finale potenzialmente contenente blocchi file espliciti.
        path_scritti_turno: Path già completati dai tool, da non duplicare.
        service: Servizio del turno; None usa l'adapter context-local.

    Returns:
        Soli path creati dal fallback, in ordine.
    """
    selected_service = _selected_service(service)
    # Il fallback non può trasformare una Q&A in una scrittura spontanea.
    if not selected_service.write_allowed:
        return []
    already_written = {
        normalized
        for path in path_scritti_turno
        if (normalized := selected_service.normalize_file_path(path)) is not None
    }
    fallback_paths: list[str] = []
    # Valuta ogni file separatamente: una write tool non blocca gli altri file mancanti.
    for relative_path, raw_content in estrai_path_espliciti(
        risultato,
        selected_service,
    ):
        # Un path già scritto dal tool è completo e non deve essere sovrascritto dal testo finale.
        if relative_path in already_written:
            continue
        clean_content = sanitizza_contenuto_file(raw_content, relative_path)
        result = selected_service.create_text(relative_path, clean_content)
        # Errori o file preesistenti restano intatti e non vengono annunciati come fallback.
        if not result.succeeded or result.path is None:
            continue
        already_written.add(result.path)
        fallback_paths.append(result.path)
    return fallback_paths

def _selected_service(
    service: WorkspaceFileService | None,
) -> WorkspaceFileService:
    """Restituisce l'istanza esplicita o il servizio legacy context-local."""
    return service if service is not None else get_default_file_service()

def _clean_raw_path(raw_path: str) -> str:
    """Rimuove quote e prefissi strutturali senza interpretare il path."""
    text = str(raw_path or "").strip().strip("\"'")
    text = _PATH_PREFIX_PATTERN.sub("", text)
    # Gemma a volte copia la frase «path relativo» davanti al path vero.
    text = _INSTRUCTIONAL_PATH_PREFIX_PATTERN.sub("", text)
    text = _ROW_PREFIX_PATTERN.sub("", text)
    return _ROW_SUFFIX_PATTERN.sub("", text).strip()

# Sotto questa soglia la coda è un nome con spazi («notes vecchie»), non prosa incollata.
_MIN_PAROLE_CODA_INCOLLATA = 3


def _esiste(
    service: WorkspaceFileService,
    path: str,
    *,
    cartella: bool,
) -> bool:
    """True se il path sandboxed esiste ed è del tipo atteso."""
    if cartella:
        resolved = service.resolve_directory(path)
        return resolved is not None and resolved.absolute.is_dir()
    resolved = service.resolve_file(path)
    return resolved is not None and resolved.absolute.is_file()


def recupera_path_esistente(
    action_input: str,
    service: WorkspaceFileService,
    *,
    cartella: bool,
) -> str:
    """
    Gemma spesso non va a capo dopo l'Action Input e prosegue sulla stessa riga
    («Articolo_01.pagesTrovato il file…», «.Tuttavia non vedo…»): ogni volta
    un'iterazione persa. Se il path intero non esiste, prende il prefisso più
    lungo che esiste davvero, purché il resto sia chiaramente prosa.

    Returns:
        Path recuperato, oppure l'input ripulito se non c'è un recupero sicuro.
    """
    cleaned = _clean_raw_path(str(action_input or ""))
    if _esiste(service, cleaned, cartella=cartella):
        return cleaned
    for fine in range(len(cleaned) - 1, 0, -1):
        prefisso = cleaned[:fine].rstrip()
        coda = cleaned[fine:]
        if len(coda.split()) < _MIN_PAROLE_CODA_INCOLLATA:
            continue
        # Confine credibile: spazio, maiuscola incollata o il «.» della root.
        if not (coda[0].isspace() or coda[0].isupper() or prefisso == "."):
            continue
        if prefisso and _esiste(service, prefisso, cartella=cartella):
            return prefisso
    return cleaned


def _messaggio_errore_formato(raw_text: str) -> str:
    """
    Costruisce un Observation di formato che guida Gemma verso path/---/body.

    Args:
        raw_text: Action Input grezzo rifiutato dal parser.

    Returns:
        Messaggio Errore: con esempio concreto, specialmente per lo shorthand a virgola.
    """
    first_line = raw_text.split("\n", 1)[0].strip()
    # Tipico fallimento Gemma: ``tmp/a_duo,AAA`` invece del protocollo a tre righe.
    if "," in first_line:
        return (
            "Errore: non usare destinazione,contenuto sulla stessa riga.\n"
            "Formato obbligatorio (UN file per Action):\n"
            f"tmp/a.txt\n{SEPARATORE_FILE}\nAAA\n"
            "Poi ripeti Action per il secondo file."
        )
    if first_line.startswith("/"):
        relativo = first_line.lstrip("/")
        return (
            f"Errore: non usare lo slash iniziale. Path relativo: {relativo}\n"
            f"Poi una riga {SEPARATORE_FILE} e il contenuto completo del file."
        )
    if SEPARATORE_FILE not in raw_text:
        esempio = first_line or "sito/index.html"
        return (
            "Errore: manca il contenuto del file.\n"
            f"Usa:\n{esempio}\n{SEPARATORE_FILE}\n"
            "<contenuto completo>"
        )
    return (
        "Errore: formato non valido.\n"
        f"Usa path relativo:\nsito/index.html\n{SEPARATORE_FILE}\n"
        "<contenuto>"
    )

def _normalize_content_prefix(text: str) -> str:
    """Rimuove marker content/linguaggio aggiunti davanti al body dal modello."""
    lines = text.strip("\n").split("\n", 1)
    first_line = _CONTENT_PREFIX_PATTERN.sub("", lines[0].strip())
    remainder = lines[1] if len(lines) > 1 else ""
    # Un language marker isolato proviene dal fence serializzato e non dal file.
    if first_line.casefold() in _KNOWN_LANGUAGE_MARKERS:
        first_line = ""
    # Marker YAML block-scalar dopo "Contenuto:" non fanno parte del file.
    if first_line in _YAML_BLOCK_MARKERS:
        first_line = ""
    # Artefatto «riga ---» copiato dalle istruzioni del prompt.
    if _ROW_SEPARATOR_ARTIFACT_PATTERN.match(first_line):
        first_line = ""
    # Ricompone senza introdurre una riga vuota quando una delle due parti manca.
    if first_line and remainder:
        return f"{first_line}\n{remainder}"
    return first_line or remainder

def _recover_extension_from_context(
    normalized_path: str,
    context: str,
    service: WorkspaceFileService,
) -> str:
    """
    Se il path è senza estensione ma il contesto cita ``path.ext``, usa il path completo.

    Args:
        normalized_path: Path già sandboxed dalla prima riga Action Input.
        context: Resto dell'Action Input (body + eventuali Thought).
        service: Servizio per ri-validare il candidato recuperato.

    Returns:
        Path originale oppure versione con estensione recuperata dal contesto.
    """
    # Un path che ha già un'estensione non va reinterpretato.
    if Path(normalized_path).suffix:
        return normalized_path
    # Cerca citazioni tipo tmp/a_duo.txt nello stesso Action Input.
    for match in _PATH_WITH_EXTENSION_PATTERN.finditer(context):
        candidate_raw = match.group("path")
        candidate = service.normalize_file_path(_clean_raw_path(candidate_raw))
        # Accetta solo estensioni dello stesso stem (tmp/a_duo → tmp/a_duo.txt).
        if candidate is None:
            continue
        stem = Path(candidate).with_suffix("").as_posix()
        if stem == normalized_path:
            return candidate
    return normalized_path

def _split_single_line_path_content(
    expanded: str,
    service: WorkspaceFileService,
) -> tuple[str, str] | None:
    """
    Interpreta monolinea ``path.ext,contenuto`` usata spesso da Gemma.

    Returns:
        Coppia path/contenuto oppure None se non è lo shorthand con estensione.
    """
    match = _SINGLE_LINE_PATH_CONTENT_PATTERN.match(expanded.strip())
    # Senza estensione esplicita lo shorthand è ambiguo e va rifiutato a monte.
    if match is None:
        return None
    normalized_path = service.normalize_file_path(_clean_raw_path(match.group("path")))
    content = match.group("content").strip()
    # Path invalido o body vuoto non devono produrre una write silenziosa.
    if normalized_path is None or not content:
        return None
    return normalized_path, content

def _split_path_and_content(
    raw_text: str,
    service: WorkspaceFileService,
) -> tuple[str, str] | None:
    """Separa Action Input nei formati tollerati e normalizza solo il path."""
    expanded = espandi_newline_letterali(raw_text)
    # Il separatore esplicito consente body multilinea e contenuti con newline.
    separator_match = _SEPARATOR_LINE_PATTERN.search(expanded)
    if separator_match is not None:
        raw_path = expanded[: separator_match.start()]
        content = expanded[separator_match.end() :]
        normalized_path = service.normalize_file_path(_clean_raw_path(raw_path))
        # Un path invalido rende ambiguo tutto il payload.
        if normalized_path is None:
            return None
        recovered = _recover_extension_from_context(
            normalized_path,
            content,
            service,
        )
        return recovered, content.strip("\n")

    # Monolinea path.ext,body: alternativa tollerata al protocollo a tre righe.
    if "\n" not in expanded:
        single_line = _split_single_line_path_content(expanded, service)
        if single_line is not None:
            return single_line

    first_line, separator, remainder = expanded.partition("\n")
    normalized_path = service.normalize_file_path(_clean_raw_path(first_line))
    # Il formato compatto richiede sia newline separatore sia contenuto utile.
    if normalized_path is None or not separator or not remainder.strip():
        return None
    recovered = _recover_extension_from_context(
        normalized_path,
        remainder,
        service,
    )
    return recovered, _normalize_content_prefix(remainder)

def _sanitizza_blocco_patch(blocco: str) -> str:
    """
    Rimuove soltanto fence markdown esterne, senza tagliare codice interno.

    Returns:
        Blocco preservato byte-per-byte a parte wrapper ``` opzionali.
    """
    text = blocco
    text = _OPENING_FENCE_PATTERN.sub("", text)
    text = _CLOSING_FENCE_PATTERN.sub("", text)
    return text

def _strip_block_delimiters(blocco: str) -> str:
    """Toglie al più una newline di confine lasciata dai marker di protocollo."""
    if blocco.startswith("\r\n"):
        blocco = blocco[2:]
    elif blocco.startswith("\n"):
        blocco = blocco[1:]
    if blocco.endswith("\r\n"):
        blocco = blocco[:-2]
    elif blocco.endswith("\n"):
        blocco = blocco[:-1]
    return blocco

def _split_path_old_new(
    raw_text: str,
    service: WorkspaceFileService,
) -> tuple[str, str, str] | None:
    """
    Separa Action Input di modifica_file in path, OLD e NEW.

    Returns:
        Tupla normalizzata oppure None se i marker o il path mancano.
    """
    expanded = espandi_newline_letterali(raw_text)
    old_match = _MARKER_OLD_LINE_PATTERN.search(expanded)
    new_match = _MARKER_NEW_LINE_PATTERN.search(expanded)
    if old_match is None or new_match is None:
        return None
    if new_match.start() <= old_match.start():
        return None
    path_header = expanded[: old_match.start()].strip()
    if not path_header:
        return None
    path_line = path_header.splitlines()[0]
    normalized_path = service.normalize_file_path(_clean_raw_path(path_line))
    if normalized_path is None:
        return None
    old_block = _taglia_coda_protocollo(
        _sanitizza_blocco_patch(
            _strip_block_delimiters(expanded[old_match.end() : new_match.start()])
        )
    )
    new_block = _taglia_coda_protocollo(
        _sanitizza_blocco_patch(
            _strip_block_delimiters(expanded[new_match.end() :])
        )
    )
    return normalized_path, old_block, new_block
