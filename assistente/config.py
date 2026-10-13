"""Configurazione centrale: modelli, servizio file e stato della sessione."""

from dataclasses import dataclass, field
from pathlib import Path

from assistente.inference_profiles import (
    model_variants,
    resolve_model_variant,
    route_token_budget,
)

# Radice del progetto: log e permessi di default, non la cartella documenti.
WORKSPACE = Path(__file__).resolve().parent.parent
DIRECTORY_LOG = WORKSPACE / "output" / "logs"
PREVIEW_RICHIESTA_CARATTERI = 200
TRACEBACK_MAX_CARATTERI = 4000
MESSAGGIO_EVENTO_CARATTERI = 500
STALL_SOGLIA_SEC = 30.0
STALL_POLL_SEC = 5.0
ULTIMI_ERRORI_INDICE = 50
ERRORI_SLASH_DEFAULT = 10

# Unica matrice: i tag vivono in inference_profiles.model_variants().
MODELLI_DISPONIBILI = {
    variant.alias: variant.tag for variant in model_variants()
}
ALIAS_MODELLO_DEFAULT = "default"


def risolvi_tag_ollama(alias: str) -> str:
    """
    Risolve l'alias modello nel tag Ollama fisso della matrice locale.

    Args:
        alias: Chiave in MODELLI_DISPONIBILI (fast, default).

    Returns:
        Tag Ollama da passare a OllamaLLM (E4B q4 / 12B q4).

    Raises:
        ValueError: Se l'alias non è supportato.
    """
    return resolve_model_variant(alias).tag


TEMPERATURA_LLM = 0.3
CONTESTO_LLM_TOKEN, MAX_TOKEN_GENERATI = route_token_budget("react")
MAX_ITERAZIONI_AGENTE = 5
SEPARATORE_FILE = "---"
MARCATORE_BLOCCO_OLD = "<<<OLD"
MARCATORE_BLOCCO_NEW = "<<<NEW"
MAX_CARATTERI_LETTURA_DOCUMENTO = 12000
WALK_IGNORE_DIRECTORY_NAMES = frozenset(
    {"venv", ".venv", ".git", "__pycache__", "node_modules", ".cursor"}
)
TREE_DEFAULT_DEPTH = 2
TREE_MAX_ENTRIES = 80
FIND_MAX_MATCHES = 40
# cerca_testo: righe restituite e tetto per file estratto (PDF grandi inclusi).
SEARCH_MAX_MATCHES = 30
SEARCH_MAX_FILE_BYTES = 25_000_000
# Unica estensione scrivibile: i documenti sorgente restano intoccabili.
ESTENSIONI_SCRIVIBILI = frozenset({".txt"})
LISTING_MAX_RESULT_CHARS = 2200
# Tool riassumi_documento: generazione dedicata via Ollama.
MAX_TOKEN_RIASSUNTO = route_token_budget("summary")[1]
MAX_FRASI_RIASSUNTO = 8
MIN_CARATTERI_FRASE_RIASSUNTO = 20


@dataclass(frozen=True, slots=True)
class FileConfig:
    """Parametri immutabili del servizio file confinato nella cartella documenti."""

    # Closed-corpus: la cartella da cui l'agente viene lanciato.
    workspace: Path = field(default_factory=Path.cwd)
    separator: str = SEPARATORE_FILE
    max_read_characters: int = MAX_CARATTERI_LETTURA_DOCUMENTO
    tree_depth: int = TREE_DEFAULT_DEPTH
    tree_max_entries: int = TREE_MAX_ENTRIES
    find_max_matches: int = FIND_MAX_MATCHES
    search_max_matches: int = SEARCH_MAX_MATCHES
    search_max_file_bytes: int = SEARCH_MAX_FILE_BYTES
    writable_suffixes: frozenset[str] = ESTENSIONI_SCRIVIBILI
    walk_ignore_directory_names: frozenset[str] = WALK_IGNORE_DIRECTORY_NAMES


DEFAULT_FILE_CONFIG = FileConfig()


@dataclass
class ConfigRuntime:
    """Stato runtime della sessione (default = default CLI)."""

    verbose: bool = False
    verifica_completa: bool = False
    burn_protection: bool = True
    modello: str = ALIAS_MODELLO_DEFAULT
    # CoT nativo del modello: off di default; /thinking o --thinking lo riaccendono.
    thinking: bool = False
    # Low Power Mode: alias effettivo fast, senza mutare la preferenza utente.
    auto_fast_low_power: bool = True
