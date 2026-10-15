"""Hardware-aware inference profiles for Ollama on unified memory or VRAM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from assistant.host import (
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
# Leftover (weights already subtracted) below which the heavy routes' tier and KV shrink.
TIGHT_CONTEXT_LEFTOVER_GB = MINIMUM_SYSTEM_HEADROOM_GB + 2.0


@dataclass(frozen=True, slots=True)
class ModelVariant:
    """
    Describe an Ollama artifact with a known footprint.

    Attributes:
        alias: Functional level requested by the user.
        quantization: Fixed weight quantization for the alias.
        tag: Explicit instruction-tuned Ollama tag.
        size_gb: Published size of the artifact, excluding the KV cache.
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
        Check that the weights fit the inference pool.

        Args:
            total_memory_bytes: Unified RAM or VRAM; ≤0 = unknown.
            headroom_gb: OS margin on the same pool; ignored with a discrete GPU.
            discrete_gpu: True if the weights live in dedicated VRAM.

        Returns:
            True if the profile is reasonable or the memory cannot be detected.
        """
        # An unknown value must not block an explicit user override.
        if total_memory_bytes <= 0:
            return True
        pool_gb = total_memory_bytes / GIBIBYTE
        # On VRAM the OS does not compete: it is enough that the weights fit.
        if discrete_gpu:
            return pool_gb >= self.size_gb
        used_headroom = (
            MINIMUM_SYSTEM_HEADROOM_GB if headroom_gb is None else headroom_gb
        )
        return pool_gb >= self.size_gb + used_headroom


@dataclass(frozen=True, slots=True)
class InferenceProfile:
    """
    Group the model and budget of a single route.

    Attributes:
        variant: Selected Ollama artifact.
        route: Application path that will use the model.
        context_tokens: Window allocated to the request.
        output_tokens: Maximum generation cap.
        keep_alive: Residency duration requested from the server.
    """

    variant: ModelVariant
    route: InferenceRoute
    context_tokens: int
    output_tokens: int
    keep_alive: str = DEFAULT_KEEP_ALIVE


# Fixed Gemma 4 matrix: E4B q4, 12B q4 (no runtime choice).
_MODEL_VARIANTS: dict[ModelAlias, ModelVariant] = {
    "fast": ModelVariant("fast", "q4", "gemma4:e4b", 6.6),
    "default": ModelVariant("default", "q4", "gemma4:12b", 8.0),
}

# Equivalent library tags and Ollama aliases (gemma4:latest points to E4B).
TAG_INSTALL_CANDIDATES: dict[str, tuple[str, ...]] = {
    "gemma4:e4b": ("gemma4:e4b", "gemma4:e4b-it-q4_K_M", "gemma4"),
    "gemma4:12b": ("gemma4:12b", "gemma4:12b-it-q4_K_M"),
}


def normalize_ollama_names(available_names: set[str]) -> set[str]:
    """
    Normalize the names returned by `/api/tags` for fallback matching.

    Args:
        available_names: Raw names (with or without the `:latest` suffix).

    Returns:
        Set with the original names and, when present, the form without `:latest`.
    """
    normalized: set[str] = set()
    for raw_name in available_names:
        name = raw_name.strip()
        # Empty names are not install candidates.
        if not name:
            continue
        normalized.add(name)
        # gemma4:latest → gemma4
        if name.endswith(":latest"):
            normalized.add(name[: -len(":latest")])
    return normalized


def variant_is_installed(preferred_tag: str, available_names: set[str]) -> bool:
    """
    True if the preferred tag or a library fallback is in the catalog.

    Args:
        preferred_tag: Tag declared by the profile (e.g. `gemma4:12b`).
        available_names: Already normalized catalog (`normalize_ollama_names`).

    Returns:
        False when no candidate of the family is present.
    """
    candidates = TAG_INSTALL_CANDIDATES.get(preferred_tag, (preferred_tag,))
    return any(candidate in available_names for candidate in candidates)


def select_installed_tag(preferred_tag: str, available_names: set[str]) -> str:
    """
    Pick an installed tag starting from the profile's preferred one.

    Args:
        preferred_tag: Tag declared by the profile (e.g. `gemma4:12b`).
        available_names: Already normalized catalog (`normalize_ollama_names`).

    Returns:
        The preferred tag if present, otherwise the first installed fallback;
        if none is available it still returns the preferred tag.
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

# Heavy routes when the leftover after the weights is below TIGHT_CONTEXT_LEFTOVER_GB.
_TIGHT_KV_OVERRIDES: dict[InferenceRoute, tuple[int, int]] = {
    "react": (4096, 1024),
}

# Pro/Max/Ultra or ≥32 GiB machines: more KV on the long routes, same output.
_AMPLE_OVERRIDES: dict[InferenceRoute, tuple[int, int]] = {
    "react": (8192, 1536),
}

# Steps above the historical matrix. The session goes up one step, and one
# more only if the turn fills the window.
CONTEXT_LADDER: tuple[int, ...] = (4096, 6144, 8192, 12288, 16384)
EXPAND_FILL_RATIO = 0.70
KV_CACHE_FOR_EXPANSION = "q4_0"
# Estimated fp16 KV for a 9-12B class model: 2 * layers * kv_heads * head_dim * 2 bytes.
BYTES_PER_TOKEN_FP16_KV = 56 * 1024
_KV_SCALE = {"q4_0": 0.25, "q8_0": 0.5, "f16": 1.0}


def _budget_for_route(
    route: InferenceRoute,
    overrides: dict[InferenceRoute, tuple[int, int]] | None,
) -> tuple[int, int]:
    """Route budget, with a targeted override and no copy of the matrix."""
    if overrides is not None:
        selected = overrides.get(route)
        if selected is not None:
            return selected
    return _ROUTE_BUDGETS[route]


def kv_cache_bytes(context_tokens: int, kv_cache_type: str) -> int:
    """Estimated KV bytes for a window, based on the cache type."""
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
    """True if the weights, estimated KV and system headroom fit the pool."""
    if pool_bytes <= 0 or context_tokens <= 0:
        return False
    weights = int(weight_gb * GIBIBYTE)
    headroom = int(headroom_gb * GIBIBYTE)
    return pool_bytes >= weights + kv_cache_bytes(context_tokens, kv_cache_type) + headroom


def next_context_step(current: int, *, ceiling: int) -> int:
    """First ladder step strictly above `current` and not beyond `ceiling`."""
    for step in CONTEXT_LADDER:
        if current < step <= ceiling:
            return step
    return current


def _step_if_fits(
    current: int,
    ceiling: int,
    *,
    memory_bytes: int,
    weight_gb: float,
    kv_cache_type: str,
) -> int:
    """One step above `current`, only if the estimated KV fits the pool."""
    step = next_context_step(current, ceiling=ceiling)
    if step == current or memory_bytes <= 0:
        return current
    if not context_fits_pool(memory_bytes, weight_gb, step, kv_cache_type):
        return current
    return step


def expanded_context(
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
    Session step, plus one more only if the prompt exceeds 70% of the window.

    On 16 GB the session step requires a q4_0 KV.
    On ample machines the session reaches 12288 and a full turn 16384.
    """
    if tight or memory_bytes <= 0:
        return base_tokens
    session_tokens = base_tokens
    if kv_cache_type == KV_CACHE_FOR_EXPANSION:
        session_tokens = _step_if_fits(
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
    return _step_if_fits(
        session_tokens,
        ceiling,
        memory_bytes=memory_bytes,
        weight_gb=weight_gb,
        kv_cache_type=kv_cache_type,
    )


def context_length_from_show(payload: dict[str, object]) -> int | None:
    """Read the native context from `/api/show`, without inventing rope scaling."""
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
    Expose a route's base budget, without the tight/ample overrides.

    Args:
        route: Application path.

    Returns:
        The (num_ctx, num_predict) pair of the fixed matrix.
    """
    return _ROUTE_BUDGETS[route]


def _context_overrides(
    variant: ModelVariant,
    memory_bytes: int,
    host: HostCapabilities | None,
) -> dict[InferenceRoute, tuple[int, int]] | None:
    """Tight/ample KV overrides shared by the profile and the session context."""
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
    """Maximum num_ctx across routes, with the overrides already resolved."""
    return max(_budget_for_route(route, overrides)[0] for route in _ROUTE_BUDGETS)


def session_context_tokens(
    alias: str,
    *,
    total_memory_bytes: int | None = None,
    host: HostCapabilities | None = None,
) -> int:
    """
    Single session `num_ctx`: the maximum across routes for the host class.

    Avoids Ollama reloads when switching from direct to react.
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
    Resolve the pool to use for the guardrails.

    Args:
        total_memory_bytes: Injectable value for tests; None reads the host or the system.
        host: Optional snapshot; takes priority over the raw POSIX RAM.

    Returns:
        Inference memory in bytes, or 0 if unknown.
    """
    if total_memory_bytes is not None:
        return total_memory_bytes
    # With an explicit host, use VRAM or unified memory, not a second POSIX probe.
    if host is not None:
        return host.inference_memory_bytes or host.memory_bytes
    return physical_memory_bytes()


def physical_memory_bytes() -> int:
    """
    Return the physical RAM available to the operating system.

    Returns:
        Number of physical bytes, or zero when no probe is available.
    """
    return host_physical_memory_bytes()


def resolve_model_variant(alias: str) -> ModelVariant:
    """
    Resolve the alias to the single Ollama artifact installed for that level.

    Args:
        alias: `fast` or `default`.

    Returns:
        Immutable variant with quantization and tag already fixed.

    Raises:
        ValueError: If the alias is not supported.
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
    Build the full profile for an application route.

    Args:
        alias: Requested model level.
        route: direct/ReAct/summary path.
        total_memory_bytes: Injectable RAM/VRAM for the tight KV budgets.
        output_tokens: Optional thermal override, capped at the route budget.
        host: Machine capabilities; None does not widen the budgets (16 GB profile).

    Returns:
        Profile with consistent context, output and keep-alive.
    """
    memory_bytes = _memory_bytes(total_memory_bytes, host)
    variant = resolve_model_variant(alias)
    overrides = _context_overrides(variant, memory_bytes, host)
    _, default_output_budget = _budget_for_route(route, overrides)
    selected_output_budget = default_output_budget
    context_budget = _session_context_from_overrides(overrides)
    # Throttling can only reduce the output, never expand it beyond the route.
    if output_tokens is not None:
        selected_output_budget = min(default_output_budget, max(1, output_tokens))
    keep_alive = DEFAULT_KEEP_ALIVE
    # Without a host snapshot the session's 30m stays; with a host the same value is re-read.
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
    Describe a profile that leaves insufficient headroom.

    Args:
        variant: Already resolved model.
        total_memory_bytes: Injectable pool; None uses the host or the system.
        host: Distinguishes VRAM from unified memory in the message.

    Returns:
        A readable warning, or None when the profile is reasonable.
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
    Expose the production matrix in a stable order.

    Returns:
        Variants ordered fast → default.
    """
    return (
        _MODEL_VARIANTS["fast"],
        _MODEL_VARIANTS["default"],
    )


def _should_expand(host: HostCapabilities, memory_bytes: int) -> bool:
    """True if SoC/RAM allow it and default fits the pool."""
    if not can_expand_context(host):
        return False
    discrete = host.has_discrete_gpu
    pool = memory_bytes or host.inference_memory_bytes or host.memory_bytes
    default = _MODEL_VARIANTS["default"]
    # Do not widen the KV if the default weights do not fit the pool.
    return default.fits_memory(pool, discrete_gpu=discrete)
