"""Central configuration: models, file service and session state."""

from dataclasses import dataclass, field
from pathlib import Path

from assistant.inference_profiles import (
    model_variants,
    resolve_model_variant,
    route_token_budget,
)

# Project root: default logs and permissions, not the documents folder.
WORKSPACE = Path(__file__).resolve().parent.parent
DIRECTORY_LOG = WORKSPACE / "output" / "logs"
REQUEST_PREVIEW_CHARS = 200
TRACEBACK_MAX_CHARS = 4000
EVENT_MESSAGE_CHARS = 500
STALL_THRESHOLD_SEC = 30.0
STALL_POLL_SEC = 5.0
INDEX_LAST_ERRORS = 50
DEFAULT_SLASH_ERRORS = 10

# Single matrix: the tags live in inference_profiles.model_variants().
AVAILABLE_MODELS = {
    variant.alias: variant.tag for variant in model_variants()
}
DEFAULT_MODEL_ALIAS = "default"


def resolve_ollama_tag(alias: str) -> str:
    """
    Resolve the model alias to the fixed Ollama tag of the local matrix.

    Args:
        alias: Key in AVAILABLE_MODELS (fast, default).

    Returns:
        Ollama tag to pass to OllamaLLM (E4B q4 / 12B q4).

    Raises:
        ValueError: If the alias is not supported.
    """
    return resolve_model_variant(alias).tag


LLM_TEMPERATURE = 0.3
LLM_CONTEXT_TOKENS, MAX_GENERATED_TOKENS = route_token_budget("react")
MAX_AGENT_ITERATIONS = 5
FILE_SEPARATOR = "---"
OLD_BLOCK_MARKER = "<<<OLD"
NEW_BLOCK_MARKER = "<<<NEW"
MAX_DOCUMENT_READ_CHARS = 12000
WALK_IGNORE_DIRECTORY_NAMES = frozenset(
    {"venv", ".venv", ".git", "__pycache__", "node_modules", ".cursor"}
)
TREE_DEFAULT_DEPTH = 2
TREE_MAX_ENTRIES = 80
FIND_MAX_MATCHES = 40
# cerca_testo: lines returned and cap per extracted file (large PDFs included).
SEARCH_MAX_MATCHES = 30
SEARCH_MAX_FILE_BYTES = 25_000_000
# Only writable extension: source documents stay untouchable.
WRITABLE_EXTENSIONS = frozenset({".txt"})
LISTING_MAX_RESULT_CHARS = 2200
# riassumi_documento tool: dedicated generation via Ollama.
MAX_SUMMARY_TOKENS = route_token_budget("summary")[1]
MAX_SUMMARY_SENTENCES = 8
MIN_SUMMARY_SENTENCE_CHARS = 20


@dataclass(frozen=True, slots=True)
class FileConfig:
    """Immutable parameters of the file service confined to the documents folder."""

    # Closed corpus: the folder the agent is launched from.
    workspace: Path = field(default_factory=Path.cwd)
    separator: str = FILE_SEPARATOR
    max_read_characters: int = MAX_DOCUMENT_READ_CHARS
    tree_depth: int = TREE_DEFAULT_DEPTH
    tree_max_entries: int = TREE_MAX_ENTRIES
    find_max_matches: int = FIND_MAX_MATCHES
    search_max_matches: int = SEARCH_MAX_MATCHES
    search_max_file_bytes: int = SEARCH_MAX_FILE_BYTES
    writable_suffixes: frozenset[str] = WRITABLE_EXTENSIONS
    walk_ignore_directory_names: frozenset[str] = WALK_IGNORE_DIRECTORY_NAMES


DEFAULT_FILE_CONFIG = FileConfig()


@dataclass
class ConfigRuntime:
    """Runtime session state (defaults = CLI defaults)."""

    verbose: bool = False
    full_verification: bool = False
    burn_protection: bool = True
    model: str = DEFAULT_MODEL_ALIAS
    # The model's native CoT: off by default; /thinking or --thinking turn it back on.
    thinking: bool = False
    # Low Power Mode: effective alias is fast, without changing the user preference.
    auto_fast_low_power: bool = True
