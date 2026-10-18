"""Per-session file service: sandbox, atomic writes and reliable tracking."""

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

from assistant.config import DEFAULT_FILE_CONFIG, DIRECTORY_LOG, FileConfig
from assistant.readers import (
    ImageDescriber,
    UnreadableDocument,
    is_image,
    extract_text,
)
from assistant.permissions import WritePermissionRules, load_write_permission_rules

_WRITE_DENIED = (
    "scrittura non richiesta. Il turno corrente non autorizza "
    "creazione o modifica di file."
)

_WINDOWS_DRIVE_PATTERN = re.compile(r"^[a-zA-Z]:[\\/]")
_CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x1f\x7f]")
# A leading slash on these roots would escape the workspace (/tmp, /etc, …).
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
# Commas/semicolons in segments are typical of malformed "path,content" Action Inputs.
_AMBIGUOUS_PATH_CHARACTER_PATTERN = re.compile(r"[,;:]")
_PROSE_PATH = re.compile(
    r"(?:observation|thought|final\s+answer|action\s+input)|^\d+\.\s",
    re.IGNORECASE,
)
_PAGE_MARKER = re.compile(r"^\[pagina (\d+)\]$")
_SEARCH_TERM = re.compile(r"\w{2,}", re.UNICODE)
# Search stem: «migrazione» and «migrano» share «migra».
_ROOT_LENGTH = 5
# Words that appear everywhere and do not identify a topic.
_STOPWORDS = frozenset(
    {
        "il", "lo", "la", "le", "gli", "un", "una", "uno", "di", "da", "in",
        "su", "per", "con", "tra", "fra", "che", "chi", "cosa", "come",
        "del", "della", "delle", "dei", "degli", "nel", "nella", "nei",
        "sul", "sulla", "al", "alla", "ai", "agli", "ed", "and", "the", "of",
        "quanto", "quanti", "quante", "quale", "quali", "sono", "dove",
    }
)
_SNIPPET_CHARS = 200
_TEMP_FILE_PREFIX = ".agent-write-"
_TEMP_FILE_SUFFIX = ".tmp"
# Per-session LRU cache: avoids identical re-reads in the same turn.
_FILE_CACHE_CAPACITY = 32


@dataclass(frozen=True, slots=True)
class _CachedFileContent:
    """Cache entry with an mtime+size fingerprint for cheap invalidation."""

    content: str
    mtime_ns: int
    size_bytes: int

@dataclass(frozen=True, slots=True)


class FileOperationResult:
    """Typed outcome of a read or write on a single path."""

    path: str | None = None
    content: str = ""
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """True when the operation contains no error diagnostic."""
        return self.error is None

@dataclass(frozen=True, slots=True)


class DirectoryEntry:
    """Immutable directory entry exposed without absolute Paths."""

    relative_path: str
    is_directory: bool
    size_bytes: int

@dataclass(frozen=True, slots=True)


class DirectoryOperationResult:
    """Typed outcome of a directory listing."""

    path: str
    entries: tuple[DirectoryEntry, ...] = ()
    error: str | None = None
    truncated: bool = False

    @property
    def succeeded(self) -> bool:
        """True when the directory is valid and was read."""
        return self.error is None

@dataclass(frozen=True, slots=True)


class SearchMatch:
    """Document line that contains every searched term."""

    relative_path: str
    location: str
    snippet: str

@dataclass(frozen=True, slots=True)


class SearchResult:
    """cerca_testo outcome: lines found, files skipped and truncation."""

    query: str
    matches: tuple[SearchMatch, ...] = ()
    skipped: tuple[str, ...] = ()
    # Images in the folder: search does not cover them, they must be read separately.
    images: tuple[str, ...] = ()
    error: str | None = None
    truncated: bool = False

    @property
    def succeeded(self) -> bool:
        """True when the query was valid."""
        return self.error is None

@dataclass(frozen=True, slots=True)


class ResolvedWorkspacePath:
    """Normalized relative path and absolute destination, already sandboxed."""

    relative: str
    absolute: Path

@dataclass(slots=True)


class FileTurnState:
    """Mutable state confined to one session and reset on every turn."""

    write_allowed: bool = False
    # written_paths is the turn's progress signal (it includes folders);
    # written_files holds only files, the ones that can be cited as «written».
    written_paths: list[str] = field(default_factory=list)
    written_files: list[str] = field(default_factory=list)
    successful_write_fingerprints: set[str] = field(default_factory=set)
    # Last read-only call (tool, input): Gemma tends to repeat it in a loop.
    last_read: tuple[str, str] | None = None

class _WriteMode(Enum):
    """Tool overwrite, fallback create-only, targeted modifica_file patch."""

    OVERWRITE = auto()
    CREATE_ONLY = auto()
    PATCH = auto()

def _free_name(destination: Path) -> Path:
    """First «name_2.txt», «name_3.txt»… that does not exist yet."""
    index = 2
    while True:
        candidate = destination.with_name(f"{destination.stem}_{index}{destination.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def _root(term: str) -> str:
    """First letters of a long word: tolerates plurals and conjugations."""
    if len(term) <= _ROOT_LENGTH:
        return term
    return term[:_ROOT_LENGTH]


_NON_ALPHANUMERIC = re.compile(r"[\W_]+")
_LEADING_ZEROS = re.compile(r"(?<!\d)0+(?=\d)")


def _compact_name(name: str) -> str:
    """«Articolo_01.pages» → «articolo1pages»: no separators or leading zeros."""
    compact = _NON_ALPHANUMERIC.sub("", name.casefold())
    return _LEADING_ZEROS.sub("", compact)


_ALIAS_ROOT = frozenset(
    {
        "attuale", "corrente", "cartella attuale", "cartella corrente",
        "root", "workspace", "cartella di lavoro",
    }
)


class WorkspaceFileService:
    """Single boundary for every file operation requested by the agent."""

    def __init__(
        self,
        config: FileConfig = DEFAULT_FILE_CONFIG,
        permission_rules: WritePermissionRules | None = None,
        describe_image: ImageDescriber | None = None,
    ) -> None:
        """
        Initialize the service in deny-by-default mode.

        Args:
            config: Immutable documents folder and read budget.
            permission_rules: Optional allow/deny; None loads them from the config file.
            describe_image: The model's vision for images; None rejects them.
        """
        self._config = config
        self._workspace = config.workspace.resolve()
        self._turn = FileTurnState()
        # Static rules on top of the sandbox (Claude Code level-1 pattern).
        self._permission_rules = (
            permission_rules
            if permission_rules is not None
            else load_write_permission_rules(folder=self._workspace)
        )
        self._describe_image = describe_image
        self._read_cache: OrderedDict[str, _CachedFileContent] = OrderedDict()
        # Full text for cerca_testo: avoids re-extracting PDFs and docx on every query.
        self._search_cache: dict[str, _CachedFileContent] = {}
        # Files created by the agent in this session: only these may be overwritten.
        self._created_in_session: set[str] = set()

    @property
    def workspace(self) -> Path:
        """Absolute, resolved root used by the sandbox."""
        return self._workspace

    def set_image_describer(
        self,
        describe_image: ImageDescriber | None,
    ) -> None:
        """Attach the model's vision after the runtime is created."""
        self._describe_image = describe_image
        self.invalidate_read_cache()

    @property
    def write_allowed(self) -> bool:
        """True only after begin_turn with an explicit write intent."""
        return self._turn.write_allowed

    @property
    def written_paths(self) -> list[str]:
        """Defensive copy of the paths successfully persisted in the turn."""
        return list(self._turn.written_paths)

    @property
    def written_files(self) -> list[str]:
        """Defensive copy of only the files persisted in the turn, without folders."""
        return list(self._turn.written_files)

    def mark_read(self, tool: str, action_input: str) -> bool:
        """
        Record a read-only call of the turn.

        Returns:
            True if it is identical to the previous one (the model is spinning).
        """
        key = (tool, " ".join(str(action_input or "").split()))
        repeated = self._turn.last_read == key
        self._turn.last_read = key
        return repeated

    def begin_turn(self, write_allowed: bool) -> None:
        """
        Reset tracking and anti-loop, applying the gate decided by the policy.

        Args:
            write_allowed: Authorization derived exclusively from the user request.
        """
        self._turn = FileTurnState(write_allowed=write_allowed)

    def invalidate_read_cache(self, relative_path: str | None = None) -> None:
        """
        Invalidate the read cache for one path or for the whole session.

        Args:
            relative_path: POSIX path to invalidate; None clears the whole cache.
        """
        # Full invalidation: useful after /clear or a workspace change.
        if relative_path is None:
            self._read_cache.clear()
            self._search_cache.clear()
            return
        self._read_cache.pop(relative_path, None)
        self._search_cache.pop(relative_path, None)

    def normalize_file_path(self, raw_path: str) -> str | None:
        """
        Normalize a relative file path without restricting extension, spaces or Unicode.

        Args:
            raw_path: Path already separated from the tool protocol.

        Returns:
            Relative POSIX form, or None for ambiguous or dangerous input.
        """
        text = str(raw_path or "").strip().strip("\"'")
        # Empty input, control characters and Windows drives are not portable paths.
        if (
            not text
            or _CONTROL_CHARACTER_PATTERN.search(text)
            or _WINDOWS_DRIVE_PATTERN.match(text)
        ):
            return None
        # "tmp/a_duo,AAA" is not a path: it is the path,content protocol squashed by Gemma.
        if _AMBIGUOUS_PATH_CHARACTER_PATTERN.search(text):
            return None
        if _PROSE_PATH.search(text):
            return None
        candidate = Path(text)
        # «/sito/index.html» is the workspace path sito/index.html, not the real FS.
        if candidate.is_absolute():
            parts = [part for part in candidate.parts if part not in {"/", ""}]
            if not parts or parts[0] in _ABSOLUTE_ESCAPE_ROOTS:
                return None
            candidate = Path(*parts)
        parts = candidate.parts
        # Traversal and the current path do not identify a safe file.
        if not parts or any(part in {"", ".", ".."} for part in parts):
            return None
        return Path(*parts).as_posix()

    def normalize_directory_path(self, raw_path: str) -> str | None:
        """
        Normalize a relative folder; an empty string represents the root.

        Returns:
            Relative POSIX path, ``""`` for the root, or None if invalid.
        """
        text = str(raw_path or "").strip().strip("\"'")
        # Dot, slash and empty input are explicit aliases of the workspace root.
        if text in {"", ".", "/", "./"}:
            return ""
        # Gemma names the root in words («attuale») or with the folder's own name,
        # unless a subfolder with that name really exists.
        if (
            text.casefold() in _ALIAS_ROOT or text == self._workspace.name
        ) and not (self._workspace / text).is_dir():
            return ""
        return self.normalize_file_path(text.strip("/"))

    def resolve_file(self, raw_path: str) -> ResolvedWorkspacePath | None:
        """
        Resolve a file, checking traversal, symlink components and the workspace boundary.

        Returns:
            Relative/absolute path, or None when the sandbox rejects it.
        """
        normalized = self.normalize_file_path(raw_path)
        # Resolution does not try to fix syntactically invalid paths.
        if normalized is None:
            return None
        return self._resolve_normalized(normalized)

    def resolve_directory(self, raw_path: str) -> ResolvedWorkspacePath | None:
        """
        Resolve a relative directory, including the root represented by empty input.

        Returns:
            Sandboxed path, or None for a disallowed path.
        """
        normalized = self.normalize_directory_path(raw_path)
        # Distinguish an invalid path from the root, which intentionally uses an empty string.
        if normalized is None:
            return None
        return self._resolve_normalized(normalized)

    def _deny_write(self) -> FileOperationResult:
        """Stable outcome when the turn gate is closed."""
        return FileOperationResult(error=_WRITE_DENIED)

    def is_read_allowed(self, relative_path: str) -> tuple[bool, str | None]:
        """Delegate to deny: deny applies to reads too."""
        return self._permission_rules.is_read_allowed(relative_path)

    def _read_permission_error(
        self,
        relative_path: str,
        deny_reason: str | None,
    ) -> FileOperationResult:
        """Allow/deny read error without touching the cache or the disk."""
        return FileOperationResult(
            path=relative_path,
            error=f"Lettura non consentita dalle regole di permesso: {deny_reason}",
        )

    def _filter_allowed_entries(
        self,
        entries: list[DirectoryEntry],
    ) -> tuple[DirectoryEntry, ...]:
        """Exclude entries covered by deny from the listing: secrets are invisible."""
        return tuple(entry for entry in entries if self._entry_visible(entry))

    def _entry_visible(self, entry: DirectoryEntry) -> bool:
        """Listing visibility according to deny, with a probe on folder contents."""
        if not self.is_read_allowed(entry.relative_path)[0]:
            return False
        if not entry.is_directory:
            return True
        # A glob that denies all the content (e.g. «cart/**») also hides the
        # folder: the hypothetical child «x» acts as a probe on the subtree.
        return self.is_read_allowed(f"{entry.relative_path}/x")[0]

    def _record_mutation(
        self,
        relative_path: str,
        fingerprint: str,
        *,
        directory: bool = False,
    ) -> None:
        """Record the path and the anti-loop fingerprint after a successful mutation."""
        self._turn.successful_write_fingerprints.add(fingerprint)
        if relative_path not in self._turn.written_paths:
            self._turn.written_paths.append(relative_path)
        # A folder is progress but not a file: it does not enter the final messages.
        if not directory and relative_path not in self._turn.written_files:
            self._turn.written_files.append(relative_path)

    def _permission_error(
        self,
        relative_path: str,
        deny_reason: str | None,
    ) -> FileOperationResult:
        """Allow/deny error with no side effect on disk."""
        return FileOperationResult(
            path=relative_path,
            error=f"Path non consentito dalle regole di permesso: {deny_reason}",
        )

    def write_text(self, raw_path: str, content: str) -> FileOperationResult:
        """
        Atomically create or overwrite a file after the gate and validation.

        Args:
            raw_path: Relative path in the workspace.
            content: Exact content already cleaned of the ReAct protocol.

        Returns:
            Success with the normalized path, or an error with no partial change.
        """
        return self._write(raw_path, content, _WriteMode.OVERWRITE)

    def create_text(self, raw_path: str, content: str) -> FileOperationResult:
        """
        Atomically create a file only if it does not exist, used by the fallback.

        Returns:
            Success or an explicit error; never overwrites existing data.
        """
        return self._write(raw_path, content, _WriteMode.CREATE_ONLY)

    def replace_text(self, raw_path: str, old: str, new: str) -> FileOperationResult:
        """
        Replace an exact block present exactly once, without creating files.

        Args:
            raw_path: Relative path in the workspace.
            old: Text to replace; must appear exactly once.
            new: Replacement text; empty deletes the block.

        Returns:
            Success with the normalized path, or an error with no partial change.
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
        Extract the text (UTF-8, PDF, docx, images) and truncate it to the context budget.

        Returns:
            Text content, including a marker when truncation happens.
        """
        resolved = self.resolve_file(raw_path)
        # A rejected path must not distinguish traversal from syntax in the details.
        if resolved is None:
            return FileOperationResult(error=f"Path non valido o non consentito: {raw_path}")
        # Deny protects reads too: a secret stays unreadable even if it exists.
        allowed, deny_reason = self.is_read_allowed(resolved.relative)
        if not allowed:
            return self._read_permission_error(resolved.relative, deny_reason)
        destination = resolved.absolute
        # Missing directories and files have different diagnostics, useful to the model.
        if not destination.exists():
            return FileOperationResult(
                path=resolved.relative,
                error=f"File non trovato: {resolved.relative}",
            )
        # Only regular files can be read as documents.
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
        # Cache hit only if mtime and size match (avoids stale content).
        cached = self._read_cache.get(resolved.relative)
        if (
            cached is not None
            and cached.mtime_ns == file_stat.st_mtime_ns
            and cached.size_bytes == file_stat.st_size
        ):
            self._read_cache.move_to_end(resolved.relative)
            return FileOperationResult(path=resolved.relative, content=cached.content)
        try:
            content = extract_text(
                destination,
                resolved.relative,
                describe_image=self._describe_image,
            )
        except UnreadableDocument as error:
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
        Read the whole UTF-8 file from disk, with no cache and no truncation.

        Returns:
            Full content, or an existence/encoding/IO error.
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
        # Write branches that re-read the file respect the same deny.
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
        """Insert or update an LRU entry, respecting the maximum capacity."""
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
        List a sandboxed directory with folders before files.

        Returns:
            Sorted relative entries, or a typed error.
        """
        resolved = self.resolve_directory(raw_path)
        display_path = str(raw_path or "").strip() or "."
        # A disallowed path must not be traversed to find out its type.
        if resolved is None:
            return DirectoryOperationResult(
                path=display_path,
                error=f"Path cartella non valido o non consentito: {display_path}",
            )
        # A denied folder is not even listed.
        allowed, deny_reason = self.is_read_allowed(resolved.relative)
        if not allowed:
            return DirectoryOperationResult(
                path=display_path,
                error=f"Lettura non consentita dalle regole di permesso: {deny_reason}",
            )
        destination = resolved.absolute
        # Missing files and folders are both incompatible with the listing operation.
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
            entries=self._filter_allowed_entries(entries),
        )

    def create_directory(self, raw_path: str) -> FileOperationResult:
        """
        Create a relative folder, including missing parents.

        Returns:
            Normalized path; exist_ok if it is already a directory.
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
        List a folder down to a fixed depth, skipping the noise.

        Returns:
            Relative entries or a typed error; ``truncated`` if the cap was reached.
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
            entries=self._filter_allowed_entries(entries),
            truncated=truncated,
        )

    def find_files(self, query: str) -> DirectoryOperationResult:
        """
        Find files and folders whose name globs or contains the query.

        Returns:
            Matches relative to the workspace root, capped at ``find_max_matches``.
        """
        needle = str(query or "").strip().strip("\"'")
        display = needle or "."
        if not needle:
            return DirectoryOperationResult(
                path=display,
                error="Query vuota: indica un glob (*.md) o un frammento di nome.",
            )
        entries, truncated = self._search_names(
            lambda name: self._name_matches(name, needle)
        )
        # «articolo_1» must find «Articolo_01.pages»: separators, leading zeros,
        # globs and the extension guessed by the model («articolo_1.txt») do not count.
        # Only as a fallback, so exact searches are not widened.
        compact = _compact_name(Path(needle.strip("*?[]")).stem)
        if not entries and compact:
            entries, truncated = self._search_names(
                lambda name: compact in _compact_name(name)
            )
        return DirectoryOperationResult(
            path=display,
            entries=self._filter_allowed_entries(entries),
            truncated=truncated,
        )

    def _search_names(
        self,
        matches: Callable[[str], bool],
    ) -> tuple[list[DirectoryEntry], bool]:
        """Workspace entries whose name satisfies the predicate, with a stable cap and order."""
        entries: list[DirectoryEntry] = []
        truncated = False
        max_matches = self._config.find_max_matches
        for child, is_directory, size_bytes in self._walk_workspace():
            if not matches(child.name):
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
        Search relevant lines in the extracted text of the documents.

        Each query word counts by its stem (first letters), so «migrazione»
        also finds «migrano». Lines are sorted by the number of stems found:
        those containing all of them come first.
        Images are excluded: reading them needs vision, too expensive for a
        scan of the whole folder.

        Returns:
            Lines with file and position (line or page), capped at ``search_max_matches``.
        """
        needle = " ".join(str(query or "").split()).strip("\"'")
        roots = list(
            dict.fromkeys(
                _root(term.casefold())
                for term in _SEARCH_TERM.findall(needle)
                if term.casefold() not in _STOPWORDS
            )
        )
        if not roots:
            return SearchResult(
                query=needle,
                error="Query vuota: indica una parola o una frase da cercare.",
            )
        candidates: list[tuple[int, int, SearchMatch]] = []
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
            if is_image(child):
                images.append(relative)
                continue
            if size_bytes > self._config.search_max_file_bytes:
                skipped.append(relative)
                continue
            text = self._search_text(child, relative, size_bytes)
            if text is None:
                skipped.append(relative)
                continue
            for location, line in self._lines_with_position(text):
                cf_line = line.casefold()
                found = [root for root in roots if root in cf_line]
                if not found:
                    continue
                match = SearchMatch(relative, location, self._snippet(line, found[0]))
                candidates.append((-len(found), len(candidates), match))
        candidates.sort(key=lambda entry: entry[:2])
        limit = self._config.search_max_matches
        return SearchResult(
            query=needle,
            matches=tuple(match for _, _, match in candidates[:limit]),
            skipped=tuple(skipped),
            images=tuple(sorted(images)),
            truncated=len(candidates) > limit,
        )

    def _search_text(
        self,
        path: Path,
        relative: str,
        size_bytes: int,
    ) -> str | None:
        """Full text cached on mtime+size; None if the format cannot be read."""
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
            text = extract_text(path, relative)
        except (UnreadableDocument, OSError):
            return None
        self._search_cache[relative] = _CachedFileContent(text, mtime_ns, size_bytes)
        return text

    @staticmethod
    def _lines_with_position(text: str) -> Iterator[tuple[str, str]]:
        """Non-empty lines with «pagina N» for PDFs, otherwise «riga N»."""
        page: str | None = None
        for number, line in enumerate(text.splitlines(), start=1):
            clean = line.strip()
            if not clean:
                continue
            marker = _PAGE_MARKER.match(clean)
            if marker is not None:
                page = marker.group(1)
                continue
            yield (f"pagina {page}" if page else f"riga {number}"), clean

    @staticmethod
    def _snippet(line: str, term: str) -> str:
        """Text window centered on the first term found."""
        if len(line) <= _SNIPPET_CHARS:
            return line
        start = max(0, line.casefold().find(term) - _SNIPPET_CHARS // 3)
        excerpt = line[start : start + _SNIPPET_CHARS]
        prefix = "…" if start > 0 else ""
        suffix = "…" if start + _SNIPPET_CHARS < len(line) else ""
        return f"{prefix}{excerpt}{suffix}"

    def _suffix_error(self, relative_path: str) -> FileOperationResult | None:
        """Writes touch only .txt files: source documents stay intact."""
        suffix = Path(relative_path).suffix.casefold()
        if suffix in self._config.writable_suffixes:
            return None
        allowed = ", ".join(sorted(self._config.writable_suffixes))
        suggested = Path(relative_path).with_suffix(".txt").as_posix()
        return FileOperationResult(
            path=relative_path,
            error=(
                f"Scrittura consentita solo su file {allowed}: {relative_path}. "
                f"Usa lo stesso path con .txt: {suggested}"
            ),
        )

    def _write(
        self,
        raw_path: str,
        content: str,
        mode: _WriteMode,
    ) -> FileOperationResult:
        """Atomic create/overwrite sharing gate, sandbox and tracking."""
        # The deny-by-default service prevents accidental bypasses through the LangChain adapters.
        if not self._turn.write_allowed:
            return self._deny_write()
        resolved = self.resolve_file(raw_path)
        # Invalid paths are rejected before creating temporary directories.
        if resolved is None:
            return FileOperationResult(error=f"Path non consentito: {raw_path}")
        suffix_error = self._suffix_error(resolved.relative)
        if suffix_error is not None:
            return suffix_error
        # Allow/deny layer on top of the sandbox (permissions.json).
        allowed, deny_reason = self._permission_rules.is_write_allowed(resolved.relative)
        if not allowed:
            return self._permission_error(resolved.relative, deny_reason)
        destination = resolved.absolute
        # The create-only fallback cannot overwrite an existing file.
        if mode is _WriteMode.CREATE_ONLY and destination.exists():
            return FileOperationResult(
                path=resolved.relative,
                error=f"File già esistente: {resolved.relative}",
            )
        # A directory cannot be replaced by file content.
        if destination.exists() and not destination.is_file():
            return FileOperationResult(
                path=resolved.relative,
                error=f"La destinazione non è un file: {resolved.relative}",
            )
        # Documents already in the folder are not overwritten wholesale.
        if (
            mode is _WriteMode.OVERWRITE
            and destination.exists()
            and resolved.relative not in self._created_in_session
        ):
            new = _free_name(destination).relative_to(self._workspace).as_posix()
            return FileOperationResult(
                path=resolved.relative,
                error=(
                    f"{resolved.relative} è un documento della cartella: non lo "
                    f"sovrascrivo. Scrivi in un file nuovo (es. {new}) oppure "
                    "usa modifica_file per una modifica puntuale."
                ),
            )
        new_file = not destination.exists()

        fingerprint = self._write_fingerprint(resolved.relative, content)
        # Block an identical write that already succeeded in the same turn, a typical ReAct loop.
        if fingerprint in self._turn.successful_write_fingerprints:
            return FileOperationResult(
                path=resolved.relative,
                error=(
                    f"Azione di scrittura già completata per: {resolved.relative}. "
                    "Passa al prossimo file o a Final Answer."
                ),
            )

        write_error = self._atomic_write(destination, content)
        # Atomic errors are not recorded, so a corrected retry stays possible.
        if write_error is not None:
            return FileOperationResult(
                path=resolved.relative,
                error=f"Errore di scrittura su '{resolved.relative}': {write_error}",
            )

        self._record_mutation(resolved.relative, fingerprint)
        if new_file:
            self._created_in_session.add(resolved.relative)
        # The read cache must not serve stale content after a write.
        self.invalidate_read_cache(resolved.relative)
        return FileOperationResult(path=resolved.relative)

    def _is_ignored_directory(self, path: Path) -> bool:
        """True for noise folders not to traverse in listing/find."""
        return path.is_dir() and path.name in self._config.walk_ignore_directory_names

    def _iter_visible_children(self, directory: Path) -> list[Path]:
        """Non-symlink children, without ignored folders, sorted directories first."""
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
        """Relative metadata of an already visible path, or None outside the sandbox."""
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
        """BFS down to max_depth; depth 1 = direct children of the requested folder."""
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
        """Traverse the workspace, skipping symlinks and ignored folders."""
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
        """Glob if the query has metacharacters, otherwise a casefold substring."""
        if any(marker in query for marker in "*?[]"):
            return fnmatch(name, query) or fnmatch(name.casefold(), query.casefold())
        return query.casefold() in name.casefold()

    def _resolve_normalized(self, normalized: str) -> ResolvedWorkspacePath | None:
        """Enforce the workspace boundary and reject any existing symlink component."""
        candidate = self._workspace if not normalized else self._workspace / normalized
        # Intermediate or final symlinks make the boundary ambiguous in the presence of races.
        if self._contains_symlink(candidate):
            return None
        try:
            resolved = candidate.resolve(strict=False)
        except OSError:
            return None
        # resolve neutralizes existing components; the check prevents escaping the root.
        if not resolved.is_relative_to(self._workspace):
            return None
        return ResolvedWorkspacePath(relative=normalized, absolute=resolved)

    def _contains_symlink(self, candidate: Path) -> bool:
        """Check every already existing component from the root to the destination."""
        # The root itself is trusted by the configuration and is excluded from the traversal.
        try:
            relative_parts = candidate.relative_to(self._workspace).parts
        except ValueError:
            return True
        current = self._workspace
        # Reject the first symlink seen, without following it or inspecting its target.
        for part in relative_parts:
            current = current / part
            # is_symlink also works for broken links, which exists would not detect.
            if current.is_symlink():
                return True
        return False

    @staticmethod
    def _directory_sort_key(child: Path) -> tuple[bool, str]:
        """Sort directories before files, then by case-insensitive name."""
        return not child.is_dir(), child.name.casefold()

    def _atomic_write(self, destination: Path, content: str) -> str | None:
        """Write and fsync a temporary file, then atomically replace the destination."""
        temporary_path: Path | None = None
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            # A second validation narrows the window between the check and the directory creation.
            resolved_after_mkdir = self._resolve_normalized(
                destination.relative_to(self._workspace).as_posix()
            )
            # A symlink introduced during mkdir aborts before the write.
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
            # fdopen guarantees closing even when write, flush or fsync fail.
            with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())

            # Keep the existing file's permissions instead of replacing them with the temp file's.
            if destination.exists():
                current_mode = stat.S_IMODE(destination.stat().st_mode)
                os.chmod(temporary_path, current_mode)
            os.replace(temporary_path, destination)
            temporary_path = None
        except OSError as error:
            return str(error)
        finally:
            # An error before os.replace must not leave hidden artifacts in the workspace.
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    # Best-effort cleanup must not hide the original write error.
                    pass
        return None

    def _truncate_document(self, content: str, relative_path: str) -> str:
        """Truncate a document to the budget, keeping the original size and origin."""
        character_limit = self._config.max_read_characters
        # Documents within budget are returned byte for byte after decoding.
        if len(content) <= character_limit:
            return content
        return (
            f"{content[:character_limit]}\n\n"
            f"[... troncato: mostrati {character_limit} "
            f"di {len(content)} caratteri in '{relative_path}' ...]"
        )

    @staticmethod
    def _write_fingerprint(relative_path: str, content: str) -> str:
        """Compute a compact key to block identical rewrites within the turn."""
        digest = hashlib.sha256()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content.encode("utf-8"))
        return digest.hexdigest()
