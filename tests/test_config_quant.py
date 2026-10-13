"""Test risoluzione tag Ollama per alias (quant fissa)."""

from __future__ import annotations

import pytest

from assistente.config import MODELLI_DISPONIBILI, risolvi_tag_ollama
from assistente.inference_profiles import resolve_model_variant


@pytest.mark.parametrize(
    ("alias", "tag_atteso"),
    [
        ("fast", "gemma4:e4b"),
        ("default", "gemma4:12b"),
    ],
)
def test_risolvi_tag_ollama_mapping(alias: str, tag_atteso: str) -> None:
    """Ogni alias punta al solo tag della matrice locale."""
    assert risolvi_tag_ollama(alias) == tag_atteso
    assert MODELLI_DISPONIBILI[alias] == tag_atteso


def test_risolvi_tag_ollama_normalizza_maiuscole() -> None:
    """Gli alias sono case-insensitive."""
    assert risolvi_tag_ollama("DEFAULT") == "gemma4:12b"


def test_risolvi_tag_ollama_alias_invalido() -> None:
    """Alias sconosciuti sollevano ValueError."""
    with pytest.raises(ValueError, match="Alias modello non valido"):
        risolvi_tag_ollama("enorme")


def test_quantizzazione_per_alias_matrice_fissa() -> None:
    """E4B→q4, 12B→q4."""
    assert resolve_model_variant("fast").quantization == "q4"
    assert resolve_model_variant("default").quantization == "q4"


def test_resolve_model_variant_coerente_con_tag() -> None:
    """Variante e risolvi_tag_ollama espongono lo stesso artefatto."""
    for alias in MODELLI_DISPONIBILI:
        assert resolve_model_variant(alias).tag == risolvi_tag_ollama(alias)
