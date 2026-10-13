"""Test profili inferenza con quantizzazione fissa per alias."""

from __future__ import annotations

import pytest

from assistente.host import HostCapabilities
from assistente.inference_profiles import (
    GIBIBYTE,
    TAG_INSTALL_CANDIDATES,
    memory_warning,
    model_variants,
    normalize_ollama_names,
    profile_for_route,
    resolve_model_variant,
    contesto_espanso,
    context_fits_pool,
    context_length_from_show,
    session_context_tokens,
    select_installed_tag,
    variant_is_installed,
)

_MEMORY_12_GIB = 12 * GIBIBYTE
_MEMORY_16_GIB = 16 * GIBIBYTE
_MEMORY_32_GIB = 32 * GIBIBYTE


def test_resolve_model_variant_fast_sempre_q4() -> None:
    """fast risolve sempre a Gemma 4 E4B q4."""
    variant = resolve_model_variant("fast")
    assert variant.tag == "gemma4:e4b"
    assert variant.quantization == "q4"


def test_resolve_model_variant_default_sempre_q4() -> None:
    """default risolve sempre a Gemma 4 12B q4."""
    variant = resolve_model_variant("default")
    assert variant.tag == "gemma4:12b"
    assert variant.quantization == "q4"




def test_resolve_model_variant_alias_invalido() -> None:
    """Alias sconosciuti falliscono."""
    with pytest.raises(ValueError, match="Alias modello non valido"):
        resolve_model_variant("enorme")


def test_quantizzazione_per_alias() -> None:
    """La quantizzazione è fissata sulla variante, non da un helper separato."""
    assert resolve_model_variant("FAST").quantization == "q4"
    assert resolve_model_variant("Default").quantization == "q4"




def test_memory_warning_per_fast_su_memoria_stretta() -> None:
    """E4B q4 sotto headroom minimo produce un avviso."""
    variant = resolve_model_variant("fast")
    avviso = memory_warning(variant, total_memory_bytes=_MEMORY_16_GIB)
    assert avviso is None
    avviso_12 = memory_warning(variant, total_memory_bytes=_MEMORY_12_GIB)
    assert avviso_12 is not None
    assert "gemma4:e4b" in avviso_12


def test_memory_warning_assente_su_memoria_ampia() -> None:
    """Con abbastanza RAM non c'è avviso per default."""
    variant = resolve_model_variant("default")
    assert memory_warning(variant, total_memory_bytes=_MEMORY_32_GIB) is None


def test_profile_fast_su_16gb_usa_budget_pieni() -> None:
    """E4B su 16 GB ha residuo alto: react resta sui budget standard."""
    profile = profile_for_route(
        "fast",
        "react",
        total_memory_bytes=_MEMORY_16_GIB,
    )
    assert profile.context_tokens == 6144
    assert profile.output_tokens == 1024


def test_profile_default_su_12gb_riduce_react() -> None:
    """12B su 12 GB lascia poco residuo: react taglia la KV."""
    profile = profile_for_route(
        "default",
        "react",
        total_memory_bytes=_MEMORY_12_GIB,
    )
    assert profile.context_tokens == 4096
    assert profile.output_tokens == 1024


def test_model_variants_ordine_stabile() -> None:
    """La matrice espone esattamente due varianti fisse."""
    variants = model_variants()
    assert [item.alias for item in variants] == ["fast", "default"]
    assert [item.quantization for item in variants] == ["q4", "q4"]


def test_host_m4_16gb_coincide_con_budget_storici() -> None:
    """Apple M4 16 GB con host esplicito non cambia ctx/predict rispetto a oggi."""
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


def test_host_m4_pro_amplia_react() -> None:
    """M4 Pro allarga solo le route pesanti quando default entra."""
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


def test_contesto_16gb_sale_a_8192_solo_con_kv_q4() -> None:
    """fast/default su 16 GB fanno uno scalino se la cache è q4_0 e la KV entra."""
    base = 6144
    assert (
        contesto_espanso(
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
        contesto_espanso(
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
        contesto_espanso(
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


def test_turno_pieno_su_host_ampio_fa_un_altro_scalino() -> None:
    """Oltre 8192, il turno che riempie la finestra può arrivare a 16384."""
    kwargs = dict(
        alias="default",
        kv_cache_type="q4_0",
        memory_bytes=_MEMORY_32_GIB,
        weight_gb=6.6,
        can_expand=True,
        tight=False,
    )
    assert contesto_espanso(8192, **kwargs) == 12288
    pieno = contesto_espanso(
        8192,
        prompt_tokens=int(12288 * 0.8),
        model_ceiling=32768,
        **kwargs,
    )
    assert pieno == 16384
    assert context_fits_pool(_MEMORY_16_GIB, 6.6, 8192, "q4_0")
    assert context_length_from_show(
        {"model_info": {"gemma4.context_length": 32768}}
    ) == 32768


def test_tag_install_candidates_gemma4() -> None:
    """Fallback libreria per E4B/12B; `gemma4` (latest) vale solo per E4B."""
    assert TAG_INSTALL_CANDIDATES["gemma4:e4b"] == (
        "gemma4:e4b",
        "gemma4:e4b-it-q4_K_M",
        "gemma4",
    )
    assert TAG_INSTALL_CANDIDATES["gemma4:12b"] == (
        "gemma4:12b",
        "gemma4:12b-it-q4_K_M",
    )
    catalogo = {"gemma4:12b-it-q4_K_M"}
    assert variant_is_installed("gemma4:12b", catalogo)
    assert select_installed_tag("gemma4:12b", catalogo) == "gemma4:12b-it-q4_K_M"
    assert not variant_is_installed("gemma4:e4b", catalogo)
    latest = normalize_ollama_names({"gemma4:latest"})
    assert select_installed_tag("gemma4:e4b", latest) == "gemma4"
    assert not variant_is_installed("gemma4:12b", latest)
