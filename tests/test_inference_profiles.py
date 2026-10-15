"""Tests for inference profiles with fixed quantization per alias."""

from __future__ import annotations

import pytest

from assistant.host import HostCapabilities
from assistant.inference_profiles import (
    GIBIBYTE,
    TAG_INSTALL_CANDIDATES,
    memory_warning,
    model_variants,
    normalize_ollama_names,
    profile_for_route,
    resolve_model_variant,
    expanded_context,
    context_fits_pool,
    context_length_from_show,
    session_context_tokens,
    select_installed_tag,
    variant_is_installed,
)

_MEMORY_12_GIB = 12 * GIBIBYTE
_MEMORY_16_GIB = 16 * GIBIBYTE
_MEMORY_32_GIB = 32 * GIBIBYTE


def test_resolve_model_variant_fast_always_q4() -> None:
    """fast always resolves to Gemma 4 E4B q4."""
    variant = resolve_model_variant("fast")
    assert variant.tag == "gemma4:e4b"
    assert variant.quantization == "q4"


def test_resolve_model_variant_default_always_q4() -> None:
    """default always resolves to Gemma 4 12B q4."""
    variant = resolve_model_variant("default")
    assert variant.tag == "gemma4:12b"
    assert variant.quantization == "q4"




def test_resolve_model_variant_invalid_alias() -> None:
    """Unknown aliases fail."""
    with pytest.raises(ValueError, match="Alias modello non valido"):
        resolve_model_variant("enorme")


def test_quantization_per_alias() -> None:
    """Quantization is pinned on the variant, not by a separate helper."""
    assert resolve_model_variant("FAST").quantization == "q4"
    assert resolve_model_variant("Default").quantization == "q4"




def test_memory_warning_for_fast_on_tight_memory() -> None:
    """E4B q4 below the minimum headroom produces a warning."""
    variant = resolve_model_variant("fast")
    warning = memory_warning(variant, total_memory_bytes=_MEMORY_16_GIB)
    assert warning is None
    warning_12 = memory_warning(variant, total_memory_bytes=_MEMORY_12_GIB)
    assert warning_12 is not None
    assert "gemma4:e4b" in warning_12


def test_memory_warning_absent_on_ample_memory() -> None:
    """With enough RAM there is no warning for default."""
    variant = resolve_model_variant("default")
    assert memory_warning(variant, total_memory_bytes=_MEMORY_32_GIB) is None


def test_profile_fast_on_16gb_uses_full_budgets() -> None:
    """E4B on 16 GB has plenty left over: react stays on the standard budgets."""
    profile = profile_for_route(
        "fast",
        "react",
        total_memory_bytes=_MEMORY_16_GIB,
    )
    assert profile.context_tokens == 6144
    assert profile.output_tokens == 1024


def test_profile_default_on_12gb_reduces_react() -> None:
    """12B on 12 GB leaves little over: react cuts the KV."""
    profile = profile_for_route(
        "default",
        "react",
        total_memory_bytes=_MEMORY_12_GIB,
    )
    assert profile.context_tokens == 4096
    assert profile.output_tokens == 1024


def test_model_variants_stable_order() -> None:
    """The matrix exposes exactly two fixed variants."""
    variants = model_variants()
    assert [item.alias for item in variants] == ["fast", "default"]
    assert [item.quantization for item in variants] == ["q4", "q4"]


def test_host_m4_16gb_matches_historical_budgets() -> None:
    """Apple M4 16 GB with an explicit host does not change ctx/predict from today."""
    host = HostCapabilities(
        os="darwin",
        arch="arm64",
        chip_family="m4",
        chip_brand="Apple M4",
        memory_bytes=_MEMORY_16_GIB,
        inference_memory_bytes=_MEMORY_16_GIB,
        performance_cores=4,
        has_discrete_gpu=False,
        cooling="fanless_conservative",
    )
    react = profile_for_route("default", "react", host=host)
    assert react.context_tokens == 6144
    assert react.output_tokens == 1024
    fast_react = profile_for_route(
        "fast",
        "react",
        total_memory_bytes=_MEMORY_16_GIB,
        host=host,
    )
    assert fast_react.context_tokens == 6144
    assert fast_react.output_tokens == 1024
    direct = profile_for_route("default", "direct", host=host)
    assert direct.context_tokens == 6144
    assert session_context_tokens("default", host=host) == 6144


def test_host_m4_pro_widens_react() -> None:
    """M4 Pro widens only the heavy routes when default fits."""
    host = HostCapabilities(
        os="darwin",
        arch="arm64",
        chip_family="m4_pro",
        chip_brand="Apple M4 Pro",
        memory_bytes=24 * GIBIBYTE,
        inference_memory_bytes=24 * GIBIBYTE,
        performance_cores=8,
        has_discrete_gpu=False,
        cooling="cooled",
    )
    react = profile_for_route("default", "react", host=host)
    assert react.context_tokens == 8192
    assert react.output_tokens == 1536
    direct = profile_for_route("default", "direct", host=host)
    assert direct.context_tokens == 8192


def test_16gb_context_rises_to_8192_only_with_kv_q4() -> None:
    """fast/default on 16 GB take one step up if the cache is q4_0 and the KV fits."""
    base = 6144
    assert (
        expanded_context(
            base,
            alias="default",
            kv_cache_type="q8_0",
            memory_bytes=_MEMORY_16_GIB,
            weight_gb=6.6,
            can_expand=False,
            tight=False,
        )
        == base
    )
    assert (
        expanded_context(
            base,
            alias="default",
            kv_cache_type="q4_0",
            memory_bytes=_MEMORY_16_GIB,
            weight_gb=6.6,
            can_expand=False,
            tight=False,
        )
        == 8192
    )
    assert (
        expanded_context(
            4096,
            alias="default",
            kv_cache_type="q4_0",
            memory_bytes=_MEMORY_16_GIB,
            weight_gb=8.0,
            can_expand=False,
            tight=True,
        )
        == 4096
    )


def test_full_turn_on_ample_host_takes_another_step() -> None:
    """Above 8192, the turn that fills the window can reach 16384."""
    kwargs = dict(
        alias="default",
        kv_cache_type="q4_0",
        memory_bytes=_MEMORY_32_GIB,
        weight_gb=6.6,
        can_expand=True,
        tight=False,
    )
    assert expanded_context(8192, **kwargs) == 12288
    full = expanded_context(
        8192,
        prompt_tokens=int(12288 * 0.8),
        model_ceiling=32768,
        **kwargs,
    )
    assert full == 16384
    assert context_fits_pool(_MEMORY_16_GIB, 6.6, 8192, "q4_0")
    assert context_length_from_show(
        {"model_info": {"gemma4.context_length": 32768}}
    ) == 32768


def test_tag_install_candidates_gemma4() -> None:
    """Library fallback for E4B/12B; `gemma4` (latest) applies only to E4B."""
    assert TAG_INSTALL_CANDIDATES["gemma4:e4b"] == (
        "gemma4:e4b",
        "gemma4:e4b-it-q4_K_M",
        "gemma4",
    )
    assert TAG_INSTALL_CANDIDATES["gemma4:12b"] == (
        "gemma4:12b",
        "gemma4:12b-it-q4_K_M",
    )
    catalog = {"gemma4:12b-it-q4_K_M"}
    assert variant_is_installed("gemma4:12b", catalog)
    assert select_installed_tag("gemma4:12b", catalog) == "gemma4:12b-it-q4_K_M"
    assert not variant_is_installed("gemma4:e4b", catalog)
    latest = normalize_ollama_names({"gemma4:latest"})
    assert select_installed_tag("gemma4:e4b", latest) == "gemma4"
    assert not variant_is_installed("gemma4:12b", latest)
