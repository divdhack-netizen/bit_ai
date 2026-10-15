"""Tests for resolving Ollama tags by alias (fixed quantization)."""

from __future__ import annotations

import pytest

from assistant.config import AVAILABLE_MODELS, resolve_ollama_tag
from assistant.inference_profiles import resolve_model_variant


@pytest.mark.parametrize(
    ("alias", "expected_tag"),
    [
        ("fast", "gemma4:e4b"),
        ("default", "gemma4:12b"),
    ],
)
def test_resolve_ollama_tag_mapping(alias: str, expected_tag: str) -> None:
    """Each alias points to the single tag of the local matrix."""
    assert resolve_ollama_tag(alias) == expected_tag
    assert AVAILABLE_MODELS[alias] == expected_tag


def test_resolve_ollama_tag_normalizes_uppercase() -> None:
    """Aliases are case-insensitive."""
    assert resolve_ollama_tag("DEFAULT") == "gemma4:12b"


def test_resolve_ollama_tag_invalid_alias() -> None:
    """Unknown aliases raise ValueError."""
    with pytest.raises(ValueError, match="Alias modello non valido"):
        resolve_ollama_tag("enorme")


def test_quantization_per_alias_fixed_matrix() -> None:
    """E4B→q4, 12B→q4."""
    assert resolve_model_variant("fast").quantization == "q4"
    assert resolve_model_variant("default").quantization == "q4"


def test_resolve_model_variant_consistent_with_tag() -> None:
    """The variant and resolve_ollama_tag expose the same artifact."""
    for alias in AVAILABLE_MODELS:
        assert resolve_model_variant(alias).tag == resolve_ollama_tag(alias)
