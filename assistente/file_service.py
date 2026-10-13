"""Servizio file per-sessione: sandbox, scritture atomiche e tracking affidabile."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from fnmatch import fnmatch
from pathlib import Path

from assistente.config import DEFAULT_FILE_CONFIG, DIRECTORY_LOG, FileConfig
from assistente.lettori import (
    DescrittoreImmagine,
    DocumentoNonLeggibile,
    e_immagine,
    estrai_testo,
)
from assistente.permissions import WritePermissionRules, load_write_permission_rules

_WRITE_DENIED = (
    "scrittura non richiesta. Il turno corrente non autorizza "
    "creazione o modifica di file."
)

_WINDOWS_DRIVE_PATTERN = re.compile(r"^[a-zA-Z]:[\\/]")
_CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x1f\x7f]")
# Slash iniziale su queste radici sarebbe un escape dal workspace (/tmp, /etc, …).
_ABSOLUTE_ESCAPE_ROOTS = frozenset(
    {
        "Applications",
        "Library",
        "System",
        "Users",
        "Volumes",
        "bin",
        "dev",
        "etc",
        "home",
        "opt",
        "private",
        "proc",
        "root",
        "sbin",
        "tmp",
        "usr",
        "var",
    }
)
# Virgole/punto-e-virgola nei segmenti sono tipici di Action Input "path,contenuto" malformati.
_AMBIGUOUS_PATH_CHARACTER_PATTERN = re.compile(r"[,;:]")
_PATH_PROSA = re.compile(
    r"(?:observation|thought|final\s+answer|action\s+input)|^\d+\.\s",
    re.IGNORECASE,
)
_MARCATORE_PAGINA = re.compile(r"^\[pagina (\d+)\]$")
_TERMINE_RICERCA = re.compile(r"\w{2,}", re.UNICODE)
# Radice di ricerca: «migrazione» e «migrano» condividono «migra».
_LUNGHEZZA_RADICE = 5
# Parole che compaiono ovunque e non identificano un tema.
_PAROLE_VUOTE = frozenset(
    {
        "il", "lo", "la", "le", "gli", "un", "una", "uno", "di", "da", "in",
        "su", "per", "con", "tra", "fra", "che", "chi", "cosa", "come",
        "del", "della", "delle", "dei", "degli", "nel", "nella", "nei",
        "sul", "sulla", "al", "alla", "ai", "agli", "ed", "and", "the", "of",
        "quanto", "quanti", "quante", "quale", "quali", "sono", "dove",
    }
)
_SNIPPET_CARATTERI = 200
_TEMP_FILE_PREFIX = ".agent-write-"
_TEMP_FILE_SUFFIX = ".tmp"
# Cache LRU per-sessione: evita re-read identiche nello stesso turno.
_FILE_CACHE_CAPACITY = 32


@dataclass(frozen=True, slots=True)
class _CachedFileContent:
    """Entry cache con fingerprint mtime+size per invalidazione leggera."""

    content: str
    mtime_ns: int
    size_bytes: int

@dataclass(frozen=True, slots=True)


class FileOperationResult:
    """Esito tipizzato di lettura o scrittura su un singolo path."""

    path: str | None = None
    content: str = ""
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """True quando l'operazione non contiene una diagnostica d'errore."""
        return self.error is None

@dataclass(frozen=True, slots=True)


class DirectoryEntry:
    """Voce immutabile di directory esposta senza Path assoluti."""

    relative_path: str
    is_directory: bool
    size_bytes: int

@dataclass(frozen=True, slots=True)


class DirectoryOperationResult:
    """Esito tipizzato dell'elenco directory."""

    path: str
    entries: tuple[DirectoryEntry, ...] = ()
    error: str | None = None
    truncated: bool = False

    @property
    def succeeded(self) -> bool:
        """True quando la directory è valida ed è stata letta."""
        return self.error is None

@dataclass(frozen=True, slots=True)


class SearchMatch:
    """Riga di documento che contiene tutti i termini cercati."""

    relative_path: str
    location: str
    snippet: str

@dataclass(frozen=True, slots=True)


class SearchResult:
    """Esito di cerca_testo: righe trovate, file saltati e troncamento."""

    query: str
    matches: tuple[SearchMatch, ...] = ()
    skipped: tuple[str, ...] = ()
    # Immagini della cartella: la ricerca non le copre, vanno lette a parte.
    images: tuple[str, ...] = ()
    error: str | None = None
    truncated: bool = False

    @property
    def succeeded(self) -> bool:
        """True quando la query era valida."""
        return self.error is None

@dataclass(frozen=True, slots=True)


class ResolvedWorkspacePath:
    """Path relativo normalizzato e destinazione assoluta già sandboxed."""

    relative: str
    absolute: Path

@dataclass(slots=True)


class FileTurnState:
    """Stato mutabile confinato a una sessione e azzerato a ogni turno."""

    write_allowed: bool = False
    # written_paths è il segnale di progresso del turno (include le cartelle);
    # written_files sono i soli file, quelli che si possono citare come «scritti».
    written_paths: list[str] = field(default_factory=list)
    written_files: list[str] = field(default_factory=list)
    successful_write_fingerprints: set[str] = field(default_factory=set)
    # Ultima chiamata in sola lettura (tool, input): Gemma tende a ripeterla in loop.
    ultima_lettura: tuple[str, str] | None = None

class _WriteMode(Enum):
    """Overwrite del tool, create-only del fallback, patch puntuale di modifica_file."""

    OVERWRITE = auto()
    CREATE_ONLY = auto()
    PATCH = auto()

def _nome_libero(destination: Path) -> Path:
    """Primo «nome_2.txt», «nome_3.txt»… che non esiste ancora."""
    indice = 2
    while True:
        candidato = destination.with_name(f"{destination.stem}_{indice}{destination.suffix}")
        if not candidato.exists():
            return candidato
        indice += 1


def _radice(termine: str) -> str:
    """Prime lettere di una parola lunga: tollera plurali e coniugazioni."""
    if len(termine) <= _LUNGHEZZA_RADICE:
        return termine
    return termine[:_LUNGHEZZA_RADICE]


_NON_ALFANUMERICO = re.compile(r"[\W_]+")
_ZERI_INIZIALI = re.compile(r"(?<!\d)0+(?=\d)")


def _nome_compatto(nome: str) -> str:
    """«Articolo_01.pages» → «articolo1pages»: niente separatori né zeri iniziali."""
    compatto = _NON_ALFANUMERICO.sub("", nome.casefold())
    return _ZERI_INIZIALI.sub("", compatto)


_ALIAS_ROOT = frozenset(
    {
        "attuale", "corrente", "cartella attuale", "cartella corrente",
        "root", "workspace", "cartella di lavoro",
    }
)


class WorkspaceFileService:
    """Boundary unica per tutte le operazioni file richieste dall'agente."""

    def __init__(
        self,
        config: FileConfig = DEFAULT_FILE_CONFIG,
        permission_rules: WritePermissionRules | None = None,
        descrivi_immagine: DescrittoreImmagine | None = None,
    ) -> None:
        """
        Inizializza il servizio in modalità deny-by-default.

        Args:
            config: Cartella documenti e budget lettura immutabili.
            permission_rules: Allow/deny opzionali; None carica da file config.
            descrivi_immagine: Vision del modello per le immagini; None le rifiuta.
        """
        self._config = config
        self._workspace = config.workspace.resolve()
        self._turn = FileTurnState()
        # Regole statiche sopra la sandbox (pattern Claude Code livello 1).
        self._permission_rules = (
            permission_rules
            if permission_rules is not None
            else load_write_permission_rules(cartella=self._workspace)
        )
        self._descrivi_immagine = descrivi_immagine
        self._read_cache: OrderedDict[str, _CachedFileContent] = OrderedDict()
        # Testo integrale per cerca_testo: evita di riestrarre PDF e docx a ogni query.
        self._search_cache: dict[str, _CachedFileContent] = {}
        # File creati dall'agente in questa sessione: solo questi si sovrascrivono.
        self._creati_in_sessione: set[str] = set()

    @property
    def workspace(self) -> Path:
        """Root assoluta e risolta usata dal sandbox."""
        return self._workspace

    def imposta_descrittore_immagine(
        self,
        descrivi_immagine: DescrittoreImmagine | None,
    ) -> None:
        """Collega la vision del modello dopo la creazione del runtime."""
        self._descrivi_immagine = descrivi_immagine
        self.invalidate_read_cache()

    @property
    def write_allowed(self) -> bool:
        """True soltanto dopo begin_turn con intent di scrittura esplicito."""
        return self._turn.write_allowed

    @property
    def written_paths(self) -> list[str]:
        """Copia difensiva dei path persistiti con successo nel turno."""
        return list(self._turn.written_paths)

    @property
    def written_files(self) -> list[str]:
        """Copia difensiva dei soli file persistiti nel turno, senza cartelle."""
        return list(self._turn.written_files)

    def segna_lettura(self, tool: str, action_input: str) -> bool:
        """
        Registra una chiamata in sola lettura del turno.

        Returns:
            True se è identica alla precedente (il modello sta girando a vuoto).
        """
        chiave = (tool, " ".join(str(action_input or "").split()))
        ripetuta = self._turn.ultima_lettura == chiave
        self._turn.ultima_lettura = chiave
        return ripetuta

    def begin_turn(self, write_allowed: bool) -> None:
        """
        Azzera tracking e anti-loop applicando il gate deciso dalla policy.

        Args:
            write_allowed: Autorizzazione derivata esclusivamente dalla richiesta utente.
        """
        self._turn = FileTurnState(write_allowed=write_allowed)

    def invalidate_read_cache(self, relative_path: str | None = None) -> None:
        """
        Invalida la cache di lettura per un path o per l'intera sessione.

        Args:
            relative_path: Path POSIX da invalidare; None svuota tutta la cache.
        """
        # Invalidazione totale: utile dopo /clear o cambio workspace.
        if relative_path is None:
            self._read_cache.clear()
            self._search_cache.clear()
            return
        self._read_cache.pop(relative_path, None)
        self._search_cache.pop(relative_path, None)

    def normalize_file_path(self, raw_path: str) -> str | None:
        """
        Normalizza un path file relativo senza limitarne estensione, spazi o Unicode.

        Args:
            raw_path: Path già separato dal protocollo tool.

        Returns:
            Forma POSIX relativa oppure None per input ambiguo o pericoloso.
        """
        text = str(raw_path or "").strip().strip("\"'")
        # Input vuoti, caratteri di controllo e drive Windows non sono path portabili.
        if (
            not text
            or _CONTROL_CHARACTER_PATTERN.search(text)
            or _WINDOWS_DRIVE_PATTERN.match(text)
        ):
            return None
        # "tmp/a_duo,AAA" non è un path: è il protocollo path,contenuto compresso da Gemma.
        if _AMBIGUOUS_PATH_CHARACTER_PATTERN.search(text):
            return None
        if _PATH_PROSA.search(text):
            return None
        candidate = Path(text)
        # «/sito/index.html» è il path workspace sito/index.html, non il FS reale.
        if candidate.is_absolute():
            parts = [part for part in candidate.parts if part not in {"/", ""}]
            if not parts or parts[0] in _ABSOLUTE_ESCAPE_ROOTS:
                return None
            candidate = Path(*parts)
        parts = candidate.parts
        # Traversal e path corrente non identificano un file sicuro.
        if not parts or any(part in {"", ".", ".."} for part in parts):
            return None
        return Path(*parts).as_posix()

    def normalize_directory_path(self, raw_path: str) -> str | None:
        """
        Normalizza una cartella relativa; stringa vuota rappresenta la root.

        Returns:
            Path POSIX relativo, ``""`` per root oppure None se non valido.
        """
        text = str(raw_path or "").strip().strip("\"'")
        # Punto, slash e input vuoto sono alias espliciti della root workspace.
        if text in {"", ".", "/", "./"}:
            return ""
        # Gemma nomina la root a parole («attuale») o col nome della cartella stessa,
        # salvo che esista davvero una sottocartella con quel nome.
        if (
            text.casefold() in _ALIAS_ROOT or text == self._workspace.name
        ) and not (self._workspace / text).is_dir():
            return ""
        return self.normalize_file_path(text.strip("/"))

    def resolve_file(self, raw_path: str) -> ResolvedWorkspacePath | None:
        """
        Risolve un file verificando traversal, componenti symlink e confine workspace.

        Returns:
            Path relativo/assoluto oppure None quando il sandbox lo rifiuta.
        """
        normalized = self.normalize_file_path(raw_path)
        # La risoluzione non tenta di correggere path sintatticamente invalidi.
        if normalized is None:
            return None
        return self._resolve_normalized(normalized)

    def resolve_directory(self, raw_path: str) -> ResolvedWorkspacePath | None:
        """
        Risolve una directory relativa, inclusa la root rappresentata da input vuoto.

        Returns:
            Path sandboxed oppure None per percorso non consentito.
        """
        normalized = self.normalize_directory_path(raw_path)
        # Distingue path invalido da root, che usa intenzionalmente stringa vuota.
        if normalized is None:
            return None
        return self._resolve_normalized(normalized)

    def _deny_write(self) -> FileOperationResult:
        """Esito stabile quando il gate del turno è chiuso."""
        return FileOperationResult(error=_WRITE_DENIED)

    def is_read_allowed(self, relative_path: str) -> tuple[bool, str | None]:
        """Delega alla deny: la deny vale anche per le letture."""
        return self._permission_rules.is_read_allowed(relative_path)

    def _read_permission_error(
        self,
        relative_path: str,
        deny_reason: str | None,
    ) -> FileOperationResult:
        """Errore allow/deny in lettura senza toccare la cache né il disco."""
        return FileOperationResult(
            path=relative_path,
            error=f"Lettura non consentita dalle regole di permesso: {deny_reason}",
        )

    def _filter_voci_consentite(
        self,
        entries: list[DirectoryEntry],
    ) -> tuple[DirectoryEntry, ...]:
        """Esclude dall'elenco le voci coperte dalla deny: i segreti sono invisibili."""
        return tuple(entry for entry in entries if self._voce_visibile(entry))

    def _voce_visibile(self, entry: DirectoryEntry) -> bool:
        """Visibilità in elenco secondo la deny, con probe sul contenuto cartelle."""
        if not self.is_read_allowed(entry.relative_path)[0]:
            return False
        if not entry.is_directory:
            return True
        # Un glob che nega tutto il contenuto (es. «cart/**») nasconde anche la
        # cartella: il figlio ipotetico «x» serve da probe sul sottoalbero.
        return self.is_read_allowed(f"{entry.relative_path}/x")[0]

    def _record_mutation(
        self,
        relative_path: str,
        fingerprint: str,
        *,
        directory: bool = False,
    ) -> None:
        """Registra path e fingerprint anti-loop dopo una mutazione riuscita."""
        self._turn.successful_write_fingerprints.add(fingerprint)
        if relative_path not in self._turn.written_paths:
            self._turn.written_paths.append(relative_path)
        # Una cartella è progresso ma non è un file: non entra nei messaggi finali.
        if not directory and relative_path not in self._turn.written_files:
            self._turn.written_files.append(relative_path)

    def _permission_error(
        self,
        relative_path: str,
        deny_reason: str | None,
    ) -> FileOperationResult:
        """Errore allow/deny senza side-effect su disco."""
        return FileOperationResult(
            path=relative_path,
            error=f"Path non consentito dalle regole di permesso: {deny_reason}",
        )

    def write_text(self, raw_path: str, content: str) -> FileOperationResult:
        """
        Crea o sovrascrive atomicamente un file dopo gate e validazione.

        Args:
            raw_path: Path relativo nel workspace.
            content: Contenuto esatto già ripulito dal protocollo ReAct.

        Returns:
            Successo con path normalizzato oppure errore senza modifica parziale.
        """
        return self._write(raw_path, content, _WriteMode.OVERWRITE)

    def create_text(self, raw_path: str, content: str) -> FileOperationResult:
        """
        Crea atomicamente un file soltanto se non esiste, usato dal fallback.

        Returns:
            Successo o errore esplicito; non sovrascrive mai dati esistenti.
        """
        return self._write(raw_path, content, _WriteMode.CREATE_ONLY)

    def replace_text(self, raw_path: str, old: str, new: str) -> FileOperationResult:
        """
        Sostituisce un blocco esatto presente una sola volta, senza creare file.

        Args:
            raw_path: Path relativo nel workspace.
            old: Testo da sostituire; deve comparire esattamente una volta.
            new: Testo sostitutivo; vuoto elimina il blocco.

        Returns:
            Successo con path normalizzato oppure errore senza modifica parziale.
        """
        if not self._turn.write_allowed:
            return self._deny_write()
        if old == "":
            return FileOperationResult(
                error="Blocco OLD vuoto: fornisci il testo esatto da sostituire."
            )
        resolved = self.resolve_file(raw_path)
        if resolved is None:
            return FileOperationResult(error=f"Path non consentito: {raw_path}")
        suffix_error = self._suffix_error(resolved.relative)
        if suffix_error is not None:
            return suffix_error
        allowed, deny_reason = self._permission_rules.is_write_allowed(resolved.relative)
        if not allowed:
            return self._permission_error(resolved.relative, deny_reason)
        full_read = self._read_full_text(resolved)
        if not full_read.succeeded:
            return full_read
        occurrences = full_read.content.count(old)
        if occurrences == 0:
            return FileOperationResult(
                path=resolved.relative,
                error=(
                    f"Blocco OLD non trovato in '{resolved.relative}'. "
                    "Rileggi il file e copia il testo esatto."
                ),
            )
        if occurrences > 1:
            return FileOperationResult(
                path=resolved.relative,
                error=(
                    f"Blocco OLD ambiguo: {occurrences} occorrenze in "
                    f"'{resolved.relative}'. Allarga il contesto OLD."
                ),
            )
        updated = full_read.content.replace(old, new, 1)
        return self._write(resolved.relative, updated, _WriteMode.PATCH)

    def read_document(self, raw_path: str) -> FileOperationResult:
        """
        Estrae il testo (UTF-8, PDF, docx, immagini) e tronca al budget del contesto.

        Returns:
            Contenuto testuale, incluso un marcatore quando avviene il troncamento.
        """
        resolved = self.resolve_file(raw_path)
        # Un path rifiutato non deve essere distinto tra traversal e sintassi nei dettagli.
        if resolved is None:
            return FileOperationResult(error=f"Path non valido o non consentito: {raw_path}")
        # La deny protegge anche le letture: un segreto resta illeggibile anche se esiste.
        allowed, deny_reason = self.is_read_allowed(resolved.relative)
        if not allowed:
            return self._read_permission_error(resolved.relative, deny_reason)
        destination = resolved.absolute
        # Directory e file mancanti hanno diagnostiche diverse utili al modello.
        if not destination.exists():
            return FileOperationResult(
                path=resolved.relative,
                error=f"File non trovato: {resolved.relative}",
            )
        # Soltanto file regolari possono essere letti come documento.
        if not destination.is_file():
            return FileOperationResult(
                path=resolved.relative,
                error=f"Non è un file: {resolved.relative}",
            )
        try:
            file_stat = destination.stat()
        except OSError as error:
            return FileOperationResult(
                path=resolved.relative,
                error=f"Errore di lettura su '{resolved.relative}': {error}",
            )
        # Hit cache solo se mtime e size coincidono (evita contenuto stale).
        cached = self._read_cache.get(resolved.relative)
        if (
            cached is not None
            and cached.mtime_ns == file_stat.st_mtime_ns
            and cached.size_bytes == file_stat.st_size
        ):
            self._read_cache.move_to_end(resolved.relative)
            return FileOperationResult(path=resolved.relative, content=cached.content)
        try:
            content = estrai_testo(
                destination,
                resolved.relative,
                descrivi_immagine=self._descrivi_immagine,
            )
        except DocumentoNonLeggibile as error:
            return FileOperationResult(path=resolved.relative, error=str(error))
        except OSError as error:
            return FileOperationResult(
                path=resolved.relative,
                error=f"Errore di lettura su '{resolved.relative}': {error}",
            )

        truncated = self._truncate_document(content, resolved.relative)
        self._store_read_cache(
            resolved.relative,
            truncated,
            file_stat.st_mtime_ns,
            file_stat.st_size,
        )
        return FileOperationResult(path=resolved.relative, content=truncated)

    def _read_full_text(self, resolved: ResolvedWorkspacePath) -> FileOperationResult:
        """
        Legge UTF-8 intero da disco, senza cache né troncamento.

        Returns:
            Contenuto completo oppure errore di esistenza/encoding/IO.
        """
        destination = resolved.absolute
        if not destination.exists():
            return FileOperationResult(
                path=resolved.relative,
                error=f"File non trovato: {resolved.relative}",
            )
        if not destination.is_file():
            return FileOperationResult(
                path=resolved.relative,
                error=f"Non è un file: {resolved.relative}",
            )
        # I rami di scrittura che rileggono il file rispettano la stessa deny.
        allowed, deny_reason = self.is_read_allowed(resolved.relative)
        if not allowed:
            return self._read_permission_error(resolved.relative, deny_reason)
        try:
            content = destination.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return FileOperationResult(
                path=resolved.relative,
                error=f"File non testuale o encoding non UTF-8: {resolved.relative}",
            )
        except OSError as error:
            return FileOperationResult(
                path=resolved.relative,
                error=f"Errore di lettura su '{resolved.relative}': {error}",
            )
        return FileOperationResult(path=resolved.relative, content=content)

    def _store_read_cache(
        self,
        relative_path: str,
        content: str,
        mtime_ns: int,
        size_bytes: int,
    ) -> None:
        """Inserisce o aggiorna un'entry LRU rispettando la capacità massima."""
        self._read_cache[relative_path] = _CachedFileContent(
            content=content,
            mtime_ns=mtime_ns,
            size_bytes=size_bytes,
        )
        self._read_cache.move_to_end(relative_path)
        while len(self._read_cache) > _FILE_CACHE_CAPACITY:
            self._read_cache.popitem(last=False)

    def list_directory(self, raw_path: str) -> DirectoryOperationResult:
        """
        Elenca una directory sandboxed con cartelle prima dei file.

        Returns:
            Entry relative ordinate o errore tipizzato.
        """
        resolved = self.resolve_directory(raw_path)
        display_path = str(raw_path or "").strip() or "."
        # Un path non consentito non deve essere attraversato per determinarne il tipo.
        if resolved is None:
            return DirectoryOperationResult(
                path=display_path,
                error=f"Path cartella non valido o non consentito: {display_path}",
            )
        # Una cartella negata non viene nemmeno elencata.
        allowed, deny_reason = self.is_read_allowed(resolved.relative)
        if not allowed:
            return DirectoryOperationResult(
                path=display_path,
                error=f"Lettura non consentita dalle regole di permesso: {deny_reason}",
            )
        destination = resolved.absolute
        # File e cartelle mancanti sono entrambi incompatibili con l'operazione elenco.
        if not destination.is_dir():
            return DirectoryOperationResult(
                path=resolved.relative or ".",
                error=f"Non è una cartella: {resolved.relative or '.'}",
            )
        try:
            children = sorted(
                destination.iterdir(),
                key=self._directory_sort_key,
            )
        except OSError as error:
            return DirectoryOperationResult(
                path=resolved.relative or ".",
                error=f"Errore lettura cartella '{resolved.relative or '.'}': {error}",
            )

        entries: list[DirectoryEntry] = []
        for child in children:
            if child.is_symlink() or self._is_ignored_directory(child):
                continue
            entry = self._entry_from_path(child)
            if entry is not None:
                entries.append(entry)
        return DirectoryOperationResult(
            path=resolved.relative or ".",
            entries=self._filter_voci_consentite(entries),
        )

    def create_directory(self, raw_path: str) -> FileOperationResult:
        """
        Crea una cartella relativa, con i parent mancanti.

        Returns:
            Path normalizzato; exist_ok se è già una directory.
        """
        if not self._turn.write_allowed:
            return self._deny_write()
        normalized = self.normalize_directory_path(raw_path)
        if normalized is None:
            return FileOperationResult(error=f"Path non consentito: {raw_path}")
        if normalized == "":
            return FileOperationResult(
                error="Non è possibile creare la root del workspace."
            )
        resolved = self._resolve_normalized(normalized)
        if resolved is None:
            return FileOperationResult(error=f"Path non consentito: {raw_path}")
        allowed, deny_reason = self._permission_rules.is_write_allowed(resolved.relative)
        if not allowed:
            return self._permission_error(resolved.relative, deny_reason)
        fingerprint = self._write_fingerprint(f"mkdir:{resolved.relative}", "")
        if fingerprint in self._turn.successful_write_fingerprints:
            return FileOperationResult(
                path=resolved.relative,
                error=(
                    f"Azione di scrittura già completata per: {resolved.relative}. "
                    "Passa al prossimo file o a Final Answer."
                )
            )
        destination = resolved.absolute
        if destination.exists() and not destination.is_dir():
            return FileOperationResult(
                path=resolved.relative,
                error=f"La destinazione non è una cartella: {resolved.relative}",
            )
        try:
            destination.mkdir(parents=True, exist_ok=True)
            resolved_after = self._resolve_normalized(resolved.relative)
            if (
                resolved_after is None
                or resolved_after.absolute != destination
                or not destination.is_dir()
            ):
                return FileOperationResult(
                    path=resolved.relative,
                    error="Il path è cambiato durante la validazione.",
                )
        except OSError as error:
            return FileOperationResult(
                path=resolved.relative,
                error=f"Errore creazione cartella '{resolved.relative}': {error}",
            )
        self._record_mutation(resolved.relative, fingerprint, directory=True)
        return FileOperationResult(path=resolved.relative)

    def list_tree(self, raw_path: str) -> DirectoryOperationResult:
        """
        Elenca una cartella fino a profondità fissa, saltando il rumore.

        Returns:
            Entry relative o errore tipizzato; ``truncated`` se si è raggiunto il tetto.
        """
        resolved = self.resolve_directory(raw_path)
        display_path = str(raw_path or "").strip() or "."
        if resolved is None:
            return DirectoryOperationResult(
                path=display_path,
                error=f"Path cartella non valido o non consentito: {display_path}",
            )
        destination = resolved.absolute
        if not destination.is_dir():
            return DirectoryOperationResult(
                path=resolved.relative or ".",
                error=f"Non è una cartella: {resolved.relative or '.'}",
            )
        entries, truncated = self._collect_tree_entries(
            destination,
            max_depth=self._config.tree_depth,
            max_entries=self._config.tree_max_entries,
        )
        return DirectoryOperationResult(
            path=resolved.relative or ".",
            entries=self._filter_voci_consentite(entries),
            truncated=truncated,
        )

    def find_files(self, query: str) -> DirectoryOperationResult:
        """
        Trova file e cartelle il cui nome globba o contiene la query.

        Returns:
            Match relativi dalla root workspace, tetto ``find_max_matches``.
        """
        needle = str(query or "").strip().strip("\"'")
        display = needle or "."
        if not needle:
            return DirectoryOperationResult(
                path=display,
                error="Query vuota: indica un glob (*.md) o un frammento di nome.",
            )
        entries, truncated = self._cerca_nomi(
            lambda name: self._name_matches(name, needle)
        )
        # «articolo_1» deve trovare «Articolo_01.pages»: separatori, zeri iniziali,
        # glob ed estensione indovinata dal modello («articolo_1.txt») non contano.
        # Solo come ripiego, per non allargare le ricerche esatte.
        compatta = _nome_compatto(Path(needle.strip("*?[]")).stem)
        if not entries and compatta:
            entries, truncated = self._cerca_nomi(
                lambda name: compatta in _nome_compatto(name)
            )
        return DirectoryOperationResult(
            path=display,
            entries=self._filter_voci_consentite(entries),
            truncated=truncated,
        )

    def _cerca_nomi(
        self,
        corrisponde: Callable[[str], bool],
    ) -> tuple[list[DirectoryEntry], bool]:
        """Voci del workspace il cui nome soddisfa il predicato, con tetto e ordine stabili."""
        entries: list[DirectoryEntry] = []
        truncated = False
        max_matches = self._config.find_max_matches
        for child, is_directory, size_bytes in self._walk_workspace():
            if not corrisponde(child.name):
                continue
            relative_path = child.relative_to(self._workspace).as_posix()
            entries.append(
                DirectoryEntry(
                    relative_path=relative_path,
                    is_directory=is_directory,
                    size_bytes=size_bytes,
                )
            )
            if len(entries) >= max_matches:
                truncated = True
                break
        entries.sort(
            key=lambda entry: (not entry.is_directory, entry.relative_path.casefold())
        )
        return entries, truncated

    def search_text(self, query: str) -> SearchResult:
        """
        Cerca righe pertinenti nel testo estratto dei documenti.

        Ogni parola della query vale per la sua radice (prime lettere), così
        «migrazione» trova anche «migrano». Le righe sono ordinate per numero
        di radici trovate: prima quelle che le contengono tutte.
        Le immagini sono escluse: leggerle richiede la vision, troppo costosa
        per una scansione dell'intera cartella.

        Returns:
            Righe con file e posizione (riga o pagina), tetto ``search_max_matches``.
        """
        needle = " ".join(str(query or "").split()).strip("\"'")
        radici = list(
            dict.fromkeys(
                _radice(termine.casefold())
                for termine in _TERMINE_RICERCA.findall(needle)
                if termine.casefold() not in _PAROLE_VUOTE
            )
        )
        if not radici:
            return SearchResult(
                query=needle,
                error="Query vuota: indica una parola o una frase da cercare.",
            )
        candidati: list[tuple[int, int, SearchMatch]] = []
        skipped: list[str] = []
        images: list[str] = []
        for child, is_directory, size_bytes in self._walk_workspace():
            if is_directory:
                continue
            if child.is_relative_to(DIRECTORY_LOG):
                continue
            relative = child.relative_to(self._workspace).as_posix()
            if not self.is_read_allowed(relative)[0]:
                continue
            if e_immagine(child):
                images.append(relative)
                continue
            if size_bytes > self._config.search_max_file_bytes:
                skipped.append(relative)
                continue
            testo = self._testo_per_ricerca(child, relative, size_bytes)
            if testo is None:
                skipped.append(relative)
                continue
            for location, riga in self._righe_con_posizione(testo):
                riga_cf = riga.casefold()
                trovate = [radice for radice in radici if radice in riga_cf]
                if not trovate:
                    continue
                match = SearchMatch(relative, location, self._snippet(riga, trovate[0]))
                candidati.append((-len(trovate), len(candidati), match))
        candidati.sort(key=lambda voce: voce[:2])
        limite = self._config.search_max_matches
        return SearchResult(
            query=needle,
            matches=tuple(match for _, _, match in candidati[:limite]),
            skipped=tuple(skipped),
            images=tuple(sorted(images)),
            truncated=len(candidati) > limite,
        )

    def _testo_per_ricerca(
        self,
        path: Path,
        relative: str,
        size_bytes: int,
    ) -> str | None:
        """Testo integrale cacheato su mtime+size; None se il formato non si legge."""
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            return None
        cached = self._search_cache.get(relative)
        if cached is not None and (cached.mtime_ns, cached.size_bytes) == (
            mtime_ns,
            size_bytes,
        ):
            return cached.content
        try:
            testo = estrai_testo(path, relative)
        except (DocumentoNonLeggibile, OSError):
            return None
        self._search_cache[relative] = _CachedFileContent(testo, mtime_ns, size_bytes)
        return testo

    @staticmethod
    def _righe_con_posizione(testo: str) -> Iterator[tuple[str, str]]:
        """Righe non vuote con «pagina N» nei PDF, altrimenti «riga N»."""
        pagina: str | None = None
        for numero, riga in enumerate(testo.splitlines(), start=1):
            pulita = riga.strip()
            if not pulita:
                continue
            marcatore = _MARCATORE_PAGINA.match(pulita)
            if marcatore is not None:
                pagina = marcatore.group(1)
                continue
            yield (f"pagina {pagina}" if pagina else f"riga {numero}"), pulita

    @staticmethod
    def _snippet(riga: str, termine: str) -> str:
        """Finestra di testo centrata sul primo termine trovato."""
        if len(riga) <= _SNIPPET_CARATTERI:
            return riga
        inizio = max(0, riga.casefold().find(termine) - _SNIPPET_CARATTERI // 3)
        estratto = riga[inizio : inizio + _SNIPPET_CARATTERI]
        prefisso = "…" if inizio > 0 else ""
        suffisso = "…" if inizio + _SNIPPET_CARATTERI < len(riga) else ""
        return f"{prefisso}{estratto}{suffisso}"

    def _suffix_error(self, relative_path: str) -> FileOperationResult | None:
        """Le scritture toccano solo file .txt: i documenti sorgente restano intatti."""
        suffix = Path(relative_path).suffix.casefold()
        if suffix in self._config.writable_suffixes:
            return None
        ammessi = ", ".join(sorted(self._config.writable_suffixes))
        suggerito = Path(relative_path).with_suffix(".txt").as_posix()
        return FileOperationResult(
            path=relative_path,
            error=(
                f"Scrittura consentita solo su file {ammessi}: {relative_path}. "
                f"Usa lo stesso path con .txt: {suggerito}"
            ),
        )

    def _write(
        self,
        raw_path: str,
        content: str,
        mode: _WriteMode,
    ) -> FileOperationResult:
        """Implementa create/overwrite atomici condividendo gate, sandbox e tracking."""
        # Il servizio deny-by-default impedisce bypass accidentali degli adapter LangChain.
        if not self._turn.write_allowed:
            return self._deny_write()
        resolved = self.resolve_file(raw_path)
        # Path non validi vengono rifiutati prima di creare directory temporanee.
        if resolved is None:
            return FileOperationResult(error=f"Path non consentito: {raw_path}")
        suffix_error = self._suffix_error(resolved.relative)
        if suffix_error is not None:
            return suffix_error
        # Livello allow/deny sopra la sandbox (permissions.json).
        allowed, deny_reason = self._permission_rules.is_write_allowed(resolved.relative)
        if not allowed:
            return self._permission_error(resolved.relative, deny_reason)
        destination = resolved.absolute
        # Il fallback create-only non può sovrascrivere un file già esistente.
        if mode is _WriteMode.CREATE_ONLY and destination.exists():
            return FileOperationResult(
                path=resolved.relative,
                error=f"File già esistente: {resolved.relative}",
            )
        # Una directory non può essere sostituita da contenuto file.
        if destination.exists() and not destination.is_file():
            return FileOperationResult(
                path=resolved.relative,
                error=f"La destinazione non è un file: {resolved.relative}",
            )
        # I documenti già presenti nella cartella non si sovrascrivono per intero.
        if (
            mode is _WriteMode.OVERWRITE
            and destination.exists()
            and resolved.relative not in self._creati_in_sessione
        ):
            nuovo = _nome_libero(destination).relative_to(self._workspace).as_posix()
            return FileOperationResult(
                path=resolved.relative,
                error=(
                    f"{resolved.relative} è un documento della cartella: non lo "
                    f"sovrascrivo. Scrivi in un file nuovo (es. {nuovo}) oppure "
                    "usa modifica_file per una modifica puntuale."
                ),
            )
        nuovo_file = not destination.exists()

        fingerprint = self._write_fingerprint(resolved.relative, content)
        # Blocca una scrittura identica già riuscita nello stesso turno, tipico loop ReAct.
        if fingerprint in self._turn.successful_write_fingerprints:
            return FileOperationResult(
                path=resolved.relative,
                error=(
                    f"Azione di scrittura già completata per: {resolved.relative}. "
                    "Passa al prossimo file o a Final Answer."
                ),
            )

        write_error = self._atomic_write(destination, content)
        # Errori atomici non vengono registrati, così un retry corretto resta possibile.
        if write_error is not None:
            return FileOperationResult(
                path=resolved.relative,
                error=f"Errore di scrittura su '{resolved.relative}': {write_error}",
            )

        self._record_mutation(resolved.relative, fingerprint)
        if nuovo_file:
            self._creati_in_sessione.add(resolved.relative)
        # La cache di lettura non deve servire contenuto stale dopo una write.
        self.invalidate_read_cache(resolved.relative)
        return FileOperationResult(path=resolved.relative)

    def _is_ignored_directory(self, path: Path) -> bool:
        """True per cartelle di rumore da non attraversare in listing/trova."""
        return path.is_dir() and path.name in self._config.walk_ignore_directory_names

    def _iter_visible_children(self, directory: Path) -> list[Path]:
        """Figli non symlink, senza cartelle ignorate, ordinati directory-prima."""
        try:
            children = list(directory.iterdir())
        except OSError:
            return []
        visible = [
            child
            for child in children
            if not child.is_symlink() and not self._is_ignored_directory(child)
        ]
        visible.sort(key=self._directory_sort_key)
        return visible

    def _entry_from_path(self, path: Path) -> DirectoryEntry | None:
        """Metadati relativi di un path già visibile, o None fuori sandbox."""
        try:
            relative_path = path.relative_to(self._workspace).as_posix()
        except ValueError:
            return None
        is_directory = path.is_dir()
        size_bytes = 0
        if not is_directory:
            try:
                size_bytes = path.stat().st_size
            except OSError:
                size_bytes = 0
        return DirectoryEntry(
            relative_path=relative_path,
            is_directory=is_directory,
            size_bytes=size_bytes,
        )

    def _collect_tree_entries(
        self,
        start: Path,
        *,
        max_depth: int,
        max_entries: int,
    ) -> tuple[list[DirectoryEntry], bool]:
        """BFS fino a max_depth; depth 1 = figli diretti della cartella richiesta."""
        entries: list[DirectoryEntry] = []
        truncated = False
        frontier: list[tuple[Path, int]] = [(start, 0)]
        while frontier:
            current, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            for child in self._iter_visible_children(current):
                entry = self._entry_from_path(child)
                if entry is None:
                    continue
                if len(entries) >= max_entries:
                    truncated = True
                    return entries, truncated
                entries.append(entry)
                if child.is_dir():
                    frontier.append((child, depth + 1))
        return entries, truncated

    def _walk_workspace(self) -> Iterator[tuple[Path, bool, int]]:
        """Attraversa il workspace saltando symlink e cartelle ignorate."""
        stack = [self._workspace]
        while stack:
            current = stack.pop()
            for child in self._iter_visible_children(current):
                is_directory = child.is_dir()
                size_bytes = 0
                if not is_directory:
                    try:
                        size_bytes = child.stat().st_size
                    except OSError:
                        size_bytes = 0
                yield child, is_directory, size_bytes
                if is_directory:
                    stack.append(child)

    @staticmethod
    def _name_matches(name: str, query: str) -> bool:
        """Glob se la query ha metacaratteri, altrimenti sottostringa casefold."""
        if any(marker in query for marker in "*?[]"):
            return fnmatch(name, query) or fnmatch(name.casefold(), query.casefold())
        return query.casefold() in name.casefold()

    def _resolve_normalized(self, normalized: str) -> ResolvedWorkspacePath | None:
        """Applica il confine workspace e rifiuta qualsiasi componente symlink esistente."""
        candidate = self._workspace if not normalized else self._workspace / normalized
        # Symlink intermedi o finali rendono ambiguo il confine in presenza di race.
        if self._contains_symlink(candidate):
            return None
        try:
            resolved = candidate.resolve(strict=False)
        except OSError:
            return None
        # resolve neutralizza componenti esistenti; il controllo impedisce escape dal root.
        if not resolved.is_relative_to(self._workspace):
            return None
        return ResolvedWorkspacePath(relative=normalized, absolute=resolved)

    def _contains_symlink(self, candidate: Path) -> bool:
        """Controlla ogni componente già esistente dalla root alla destinazione."""
        # La root stessa è trusted dalla configurazione e viene esclusa dal traversal.
        try:
            relative_parts = candidate.relative_to(self._workspace).parts
        except ValueError:
            return True
        current = self._workspace
        # Rifiuta il primo symlink osservato senza seguirlo né ispezionarne la destinazione.
        for part in relative_parts:
            current = current / part
            # is_symlink funziona anche per link spezzati, che exists non rileverebbe.
            if current.is_symlink():
                return True
        return False

    @staticmethod
    def _directory_sort_key(child: Path) -> tuple[bool, str]:
        """Ordina directory prima dei file e poi per nome case-insensitive."""
        return not child.is_dir(), child.name.casefold()

    def _atomic_write(self, destination: Path, content: str) -> str | None:
        """Scrive e fsync su file temporaneo, quindi sostituisce atomicamente la destinazione."""
        temporary_path: Path | None = None
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Una seconda validazione riduce la finestra tra controllo e creazione directory.
            resolved_after_mkdir = self._resolve_normalized(
                destination.relative_to(self._workspace).as_posix()
            )
            # Un symlink introdotto durante mkdir interrompe prima della scrittura.
            if (
                resolved_after_mkdir is None
                or resolved_after_mkdir.absolute != destination
            ):
                return "Il path è cambiato durante la validazione."

            file_descriptor, temporary_name = tempfile.mkstemp(
                dir=destination.parent,
                prefix=_TEMP_FILE_PREFIX,
                suffix=_TEMP_FILE_SUFFIX,
            )
            temporary_path = Path(temporary_name)
            # fdopen garantisce chiusura anche quando write, flush o fsync falliscono.
            with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())

            # Conserva i permessi del file esistente invece di sostituirli con quelli del temp.
            if destination.exists():
                current_mode = stat.S_IMODE(destination.stat().st_mode)
                os.chmod(temporary_path, current_mode)
            os.replace(temporary_path, destination)
            temporary_path = None
        except OSError as error:
            return str(error)
        finally:
            # Un errore prima di os.replace non deve lasciare artefatti nascosti nel workspace.
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    # La pulizia best-effort non deve nascondere l'errore di scrittura originale.
                    pass
        return None

    def _truncate_document(self, content: str, relative_path: str) -> str:
        """Tronca un documento al budget mantenendo quantità originale e origine."""
        character_limit = self._config.max_read_characters
        # Documenti entro budget vengono restituiti byte-per-byte dopo la decodifica.
        if len(content) <= character_limit:
            return content
        return (
            f"{content[:character_limit]}\n\n"
            f"[... troncato: mostrati {character_limit} "
            f"di {len(content)} caratteri in '{relative_path}' ...]"
        )

    @staticmethod
    def _write_fingerprint(relative_path: str, content: str) -> str:
        """Calcola una chiave compatta per bloccare riscritture identiche nel turno."""
        digest = hashlib.sha256()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content.encode("utf-8"))
        return digest.hexdigest()
