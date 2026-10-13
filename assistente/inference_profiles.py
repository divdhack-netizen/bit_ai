"""Profili di inferenza hardware-aware per Ollama su memoria unificata o VRAM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from assistente.host import (
    DEFAULT_KEEP_ALIVE,
    GIBIBYTE,
    HostCapabilities,
    can_expand_context,
    inference_host_options,
    physical_memory_bytes as host_physical_memory_bytes,
)

ModelAlias = Literal["fast", "default"]
MODEL_ALIAS_ORDER: tuple[ModelAlias, ...] = ("fast", "default")
ResolvedQuantization = Literal["q4", "q8"]
InferenceRoute = Literal[
    "direct",
    "react",
    "summary",
]

MINIMUM_SYSTEM_HEADROOM_GB = 6.0
# Residuo (pesi già sottratti) sotto il quale fascia e KV delle route pesanti si restringono.
TIGHT_CONTEXT_LEFTOVER_GB = MINIMUM_SYSTEM_HEADROOM_GB + 2.0


@dataclass(frozen=True, slots=True)
class ModelVariant:
    """
    Descrive un artefatto Ollama con footprint noto.

    Attributes:
        alias: Livello funzionale richiesto dall'utente.
        quantization: Quantizzazione fissa dei pesi per l'alias.
        tag: Tag Ollama instruction-tuned esplicito.
        size_gb: Dimensione pubblicata dell'artefatto, esclusa la KV cache.
    """

    alias: ModelAlias
    quantization: ResolvedQuantization
    tag: str
    size_gb: float

    def fits_memory(
        self,
        total_memory_bytes: int,
        *,
        headroom_gb: float | None = None,
        discrete_gpu: bool = False,
    ) -> bool:
        """
        Verifica che i pesi entrino nel pool di inferenza.

        Args:
            total_memory_bytes: RAM unificata oppure VRAM; ≤0 = dato assente.
            headroom_gb: Margine OS sulla stessa pool; ignorato con GPU discreta.
            discrete_gpu: True se i pesi stanno in VRAM dedicata.

        Returns:
            True se il profilo è ragionevole o se la memoria non è rilevabile.
        """
        # Un valore sconosciuto non deve bloccare un override esplicito dell'utente.
        if total_memory_bytes <= 0:
            return True
        pool_gb = total_memory_bytes / GIBIBYTE
        # Sulla VRAM l'OS non compete: basta che i pesi ci stiano.
        if discrete_gpu:
            return pool_gb >= self.size_gb
        used_headroom = (
            MINIMUM_SYSTEM_HEADROOM_GB if headroom_gb is None else headroom_gb
        )
        return pool_gb >= self.size_gb + used_headroom


@dataclass(frozen=True, slots=True)
class InferenceProfile:
    """
    Raggruppa modello e budget di una singola route.

    Attributes:
        variant: Artefatto Ollama selezionato.
        route: Percorso applicativo che userà il modello.
        context_tokens: Finestra allocata alla richiesta.
        output_tokens: Tetto massimo della generazione.
        keep_alive: Durata di residenza richiesta al server.
    """

    variant: ModelVariant
    route: InferenceRoute
    context_tokens: int
    output_tokens: int
    keep_alive: str = DEFAULT_KEEP_ALIVE


# Matrice fissa Gemma 4: E4B q4, 12B q4 (nessuna scelta runtime).
_MODEL_VARIANTS: dict[ModelAlias, ModelVariant] = {
    "fast": ModelVariant("fast", "q4", "gemma4:e4b", 6.6),
    "default": ModelVariant("default", "q4", "gemma4:12b", 8.0),
}

# Tag libreria e alias Ollama equivalenti (gemma4:latest punta a E4B).
TAG_INSTALL_CANDIDATES: dict[str, tuple[str, ...]] = {
    "gemma4:e4b": ("gemma4:e4b", "gemma4:e4b-it-q4_K_M", "gemma4"),
    "gemma4:12b": ("gemma4:12b", "gemma4:12b-it-q4_K_M"),
}


def normalize_ollama_names(available_names: set[str]) -> set[str]:
    """
    Normalizza i nomi restituiti da `/api/tags` per il matching dei fallback.

    Args:
        available_names: Nomi grezzi (con o senza suffisso `:latest`).

    Returns:
        Insieme con i nomi originali e, se presente, la forma senza `:latest`.
    """
    normalized: set[str] = set()
    for raw_name in available_names:
        name = raw_name.strip()
        # Nomi vuoti non sono candidati di installazione.
        if not name:
            continue
        normalized.add(name)
        # gemma4:latest → gemma4
        if name.endswith(":latest"):
            normalized.add(name[: -len(":latest")])
    return normalized


def variant_is_installed(preferred_tag: str, available_names: set[str]) -> bool:
    """
    True se il tag preferito o un fallback di libreria è nel catalogo.

    Args:
        preferred_tag: Tag dichiarato dal profilo (es. `gemma4:12b`).
        available_names: Catalogo già normalizzato (`normalize_ollama_names`).

    Returns:
        False quando nessun candidato della famiglia è presente.
    """
    candidates = TAG_INSTALL_CANDIDATES.get(preferred_tag, (preferred_tag,))
    return any(candidate in available_names for candidate in candidates)


def select_installed_tag(preferred_tag: str, available_names: set[str]) -> str:
    """
    Sceglie un tag installato a partire dal preferito del profilo.

    Args:
        preferred_tag: Tag dichiarato dal profilo (es. `gemma4:12b`).
        available_names: Catalogo già normalizzato (`normalize_ollama_names`).

    Returns:
        Il preferito se presente, altrimenti il primo fallback installato;
        se nessuno è disponibile restituisce comunque il preferito.
    """
    candidates = TAG_INSTALL_CANDIDATES.get(preferred_tag, (preferred_tag,))
    for candidate in candidates:
        if candidate in available_names:
            return candidate
    return preferred_tag


_ROUTE_BUDGETS: dict[InferenceRoute, tuple[int, int]] = {
    "direct": (4096, 640),
    "react": (6144, 1024),
    "summary": (4096, 512),
}

# Route pesanti quando il residuo dopo i pesi è sotto TIGHT_CONTEXT_LEFTOVER_GB.
_TIGHT_KV_OVERRIDES: dict[InferenceRoute, tuple[int, int]] = {
    "react": (4096, 1024),
}

# Macchine Pro/Max/Ultra o ≥32 GiB: più KV sulle route lunghe, output invariato.
_AMPLE_OVERRIDES: dict[InferenceRoute, tuple[int, int]] = {
    "react": (8192, 1536),
}

# Scalini sopra la matrice storica. Si sale di uno per la sessione, e di un
# altro solo se il turno riempie la finestra.
CONTEXT_LADDER: tuple[int, ...] = (4096, 6144, 8192, 12288, 16384)
EXPAND_FILL_RATIO = 0.70
KV_CACHE_FOR_EXPANSION = "q4_0"
# KV fp16 stimata per un modello classe 9-12B: 2 * layer * kv_head * head_dim * 2 byte.
BYTES_PER_TOKEN_FP16_KV = 56 * 1024
_KV_SCALE = {"q4_0": 0.25, "q8_0": 0.5, "f16": 1.0}


def _budget_for_route(
    route: InferenceRoute,
    overrides: dict[InferenceRoute, tuple[int, int]] | None,
) -> tuple[int, int]:
    """Budget della route, con override puntuale senza copiare la matrice."""
    if overrides is not None:
        selected = overrides.get(route)
        if selected is not None:
            return selected
    return _ROUTE_BUDGETS[route]


def kv_cache_bytes(context_tokens: int, kv_cache_type: str) -> int:
    """Byte stimati della KV per una finestra, in base al tipo di cache."""
    scale = _KV_SCALE.get(kv_cache_type, _KV_SCALE["q8_0"])
    return int(max(0, context_tokens) * BYTES_PER_TOKEN_FP16_KV * scale)


def context_fits_pool(
    pool_bytes: int,
    weight_gb: float,
    context_tokens: int,
    kv_cache_type: str,
    *,
    headroom_gb: float = MINIMUM_SYSTEM_HEADROOM_GB,
) -> bool:
    """True se pesi, KV stimata e headroom di sistema stanno nel pool."""
    if pool_bytes <= 0 or context_tokens <= 0:
        return False
    weights = int(weight_gb * GIBIBYTE)
    headroom = int(headroom_gb * GIBIBYTE)
    return pool_bytes >= weights + kv_cache_bytes(context_tokens, kv_cache_type) + headroom


def next_context_step(current: int, *, ceiling: int) -> int:
    """Primo scalino della ladder strettamente sopra `current` e non oltre `ceiling`."""
    for step in CONTEXT_LADDER:
        if current < step <= ceiling:
            return step
    return current


def _passo_se_entra(
    current: int,
    ceiling: int,
    *,
    memory_bytes: int,
    weight_gb: float,
    kv_cache_type: str,
) -> int:
    """Uno scalino sopra `current`, solo se la KV stimata sta nel pool."""
    step = next_context_step(current, ceiling=ceiling)
    if step == current or memory_bytes <= 0:
        return current
    if not context_fits_pool(memory_bytes, weight_gb, step, kv_cache_type):
        return current
    return step


def contesto_espanso(
    base_tokens: int,
    *,
    alias: str,
    kv_cache_type: str,
    memory_bytes: int,
    weight_gb: float,
    can_expand: bool,
    tight: bool,
    prompt_tokens: int = 0,
    model_ceiling: int = CONTEXT_LADDER[-1],
) -> int:
    """
    Scalino di sessione, più uno solo se il prompt supera il 70% della finestra.

    Su 16 GB lo scalino di sessione richiede KV q4_0.
    Sulle macchine ampie la sessione arriva a 12288 e un turno pieno a 16384.
    """
    if tight or memory_bytes <= 0:
        return base_tokens
    session_tokens = base_tokens
    if kv_cache_type == KV_CACHE_FOR_EXPANSION:
        session_tokens = _passo_se_entra(
            base_tokens,
            12288 if can_expand else 8192,
            memory_bytes=memory_bytes,
            weight_gb=weight_gb,
            kv_cache_type=kv_cache_type,
        )
    if prompt_tokens <= int(session_tokens * EXPAND_FILL_RATIO):
        return session_tokens
    if kv_cache_type != KV_CACHE_FOR_EXPANSION and not can_expand:
        return session_tokens
    ceiling = min(model_ceiling, 16384 if can_expand else 8192)
    return _passo_se_entra(
        session_tokens,
        ceiling,
        memory_bytes=memory_bytes,
        weight_gb=weight_gb,
        kv_cache_type=kv_cache_type,
    )


def context_length_from_show(payload: dict[str, object]) -> int | None:
    """Legge il contesto nativo da `/api/show`, senza inventare uno scaling rope."""
    info = payload.get("model_info")
    if isinstance(info, dict):
        for key, value in info.items():
            if str(key).endswith("context_length") and isinstance(value, int) and value > 0:
                return value
    parameters = payload.get("parameters")
    if isinstance(parameters, str):
        for line in parameters.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == "num_ctx":
                try:
                    parsed = int(parts[1])
                except ValueError:
                    continue
                if parsed > 0:
                    return parsed
    return None


def route_token_budget(route: InferenceRoute) -> tuple[int, int]:
    """
    Espone il budget base di una route, senza override tight/ample.

    Args:
        route: Percorso applicativo.

    Returns:
        Coppia (num_ctx, num_predict) della matrice fissa.
    """
    return _ROUTE_BUDGETS[route]


def _context_overrides(
    variant: ModelVariant,
    memory_bytes: int,
    host: HostCapabilities | None,
) -> dict[InferenceRoute, tuple[int, int]] | None:
    """Override KV tight/ample condivisi da profilo e contesto di sessione."""
    overrides: dict[InferenceRoute, tuple[int, int]] | None = None
    if memory_bytes > 0:
        leftover_gb = (memory_bytes / GIBIBYTE) - variant.size_gb
        if leftover_gb < TIGHT_CONTEXT_LEFTOVER_GB:
            overrides = _TIGHT_KV_OVERRIDES
    if overrides is None and host is not None and _should_expand(host, memory_bytes):
        overrides = _AMPLE_OVERRIDES
    return overrides


def _session_context_from_overrides(
    overrides: dict[InferenceRoute, tuple[int, int]] | None,
) -> int:
    """Massimo num_ctx tra le route, con gli override già risolti."""
    return max(_budget_for_route(route, overrides)[0] for route in _ROUTE_BUDGETS)


def session_context_tokens(
    alias: str,
    *,
    total_memory_bytes: int | None = None,
    host: HostCapabilities | None = None,
) -> int:
    """
    Unico `num_ctx` di sessione: il massimo delle route sulla classe host.

    Evita reload Ollama quando si passa da direct a react.
    """
    memory_bytes = _memory_bytes(total_memory_bytes, host)
    variant = resolve_model_variant(alias)
    overrides = _context_overrides(variant, memory_bytes, host)
    return _session_context_from_overrides(overrides)


def _memory_bytes(
    total_memory_bytes: int | None,
    host: HostCapabilities | None,
) -> int:
    """
    Risolve il pool da usare per i guardrail.

    Args:
        total_memory_bytes: Valore iniettabile nei test; None legge host o sistema.
        host: Snapshot opzionale; ha priorità sulla RAM POSIX grezza.

    Returns:
        Byte di memoria di inferenza, oppure 0 se sconosciuta.
    """
    if total_memory_bytes is not None:
        return total_memory_bytes
    # Con host esplicito si usa VRAM o unified, non un secondo probe POSIX.
    if host is not None:
        return host.inference_memory_bytes or host.memory_bytes
    return physical_memory_bytes()


def physical_memory_bytes() -> int:
    """
    Restituisce la RAM fisica disponibile al sistema operativo.

    Returns:
        Numero di byte fisici, oppure zero quando nessun probe è disponibile.
    """
    return host_physical_memory_bytes()


def resolve_model_variant(alias: str) -> ModelVariant:
    """
    Risolve l'alias nel solo artefatto Ollama installato per quel livello.

    Args:
        alias: `fast` o `default`.

    Returns:
        Variante immutabile con quant e tag già fissati.

    Raises:
        ValueError: Se l'alias non è supportato.
    """
    normalized_alias = alias.casefold()
    if normalized_alias not in _MODEL_VARIANTS:
        raise ValueError(f"Alias modello non valido: {alias!r}.")
    return _MODEL_VARIANTS[normalized_alias]  # type: ignore[index]


def profile_for_route(
    alias: str,
    route: InferenceRoute,
    *,
    total_memory_bytes: int | None = None,
    output_tokens: int | None = None,
    host: HostCapabilities | None = None,
) -> InferenceProfile:
    """
    Costruisce il profilo completo per una route applicativa.

    Args:
        alias: Livello modello richiesto.
        route: Percorso direct/ReAct/summary.
        total_memory_bytes: RAM/VRAM iniettabile per i budget KV stretti.
        output_tokens: Override termico opzionale, limitato al budget della route.
        host: Capacità macchina; None non amplia i budget (profilo 16 GB).

    Returns:
        Profilo con contesto, output e keep-alive coerenti.
    """
    memory_bytes = _memory_bytes(total_memory_bytes, host)
    variant = resolve_model_variant(alias)
    overrides = _context_overrides(variant, memory_bytes, host)
    _, default_output_budget = _budget_for_route(route, overrides)
    selected_output_budget = default_output_budget
    context_budget = _session_context_from_overrides(overrides)
    # Il throttling può soltanto ridurre l'output, mai espanderlo oltre la route.
    if output_tokens is not None:
        selected_output_budget = min(default_output_budget, max(1, output_tokens))
    keep_alive = DEFAULT_KEEP_ALIVE
    # Senza snapshot host resta il 30m di sessione; con host si rilegge lo stesso valore.
    if host is not None:
        keep_alive = inference_host_options(host).keep_alive
    return InferenceProfile(
        variant=variant,
        route=route,
        context_tokens=context_budget,
        output_tokens=selected_output_budget,
        keep_alive=keep_alive,
    )


def memory_warning(
    variant: ModelVariant,
    *,
    total_memory_bytes: int | None = None,
    host: HostCapabilities | None = None,
) -> str | None:
    """
    Descrive un profilo che lascia headroom insufficiente.

    Args:
        variant: Modello già risolto.
        total_memory_bytes: Pool iniettabile; None usa host o sistema.
        host: Distingue VRAM da memoria unificata nel messaggio.

    Returns:
        Avviso leggibile oppure None quando il profilo è ragionevole.
    """
    memory_bytes = _memory_bytes(total_memory_bytes, host)
    discrete = host.has_discrete_gpu if host is not None else False
    if variant.fits_memory(memory_bytes, discrete_gpu=discrete):
        return None
    pool_label = "VRAM" if discrete else "unificati"
    return (
        f"{variant.tag} occupa circa {variant.size_gb:.1f} GB di pesi su "
        f"{memory_bytes / GIBIBYTE:.0f} GB {pool_label}; "
        "valutare un alias più leggero per evitare swap."
    )


def model_variants() -> tuple[ModelVariant, ...]:
    """
    Espone la matrice produzione in ordine stabile.

    Returns:
        Varianti ordinate fast → default.
    """
    return (
        _MODEL_VARIANTS["fast"],
        _MODEL_VARIANTS["default"],
    )


def _should_expand(host: HostCapabilities, memory_bytes: int) -> bool:
    """True se SoC/RAM lo consentono e default entra nel pool."""
    if not can_expand_context(host):
        return False
    discrete = host.has_discrete_gpu
    pool = memory_bytes or host.inference_memory_bytes or host.memory_bytes
    default = _MODEL_VARIANTS["default"]
    # Non allargare la KV se i pesi di default non stanno nel pool.
    return default.fits_memory(pool, discrete_gpu=discrete)
