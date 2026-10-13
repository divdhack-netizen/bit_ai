"""Test budgeting input e trimming Observation/testo."""

from __future__ import annotations

import pytest

from assistente.prompt_budget import (
    PromptBudget,
    calibrate_token_estimate,
    estimate_tokens,
    fit_observation,
    fit_text,
    limita_passi_scratchpad,
    note_prompt_bytes,
    reset_token_estimate_for_tests,
)

pytestmark = pytest.mark.unit


def test_estimate_tokens_vuoto_e_zero() -> None:
    """Testo vuoto → 0 token stimati."""
    assert estimate_tokens("") == 0


def test_estimate_tokens_conservativo_su_unicode() -> None:
    """Caratteri multibyte pesano più dei codepoint grezzi."""
    assert estimate_tokens("àèìòù") >= 2


def test_prompt_budget_riserva_output_e_scratchpad() -> None:
    """Il budget input lascia spazio obbligatorio a output e scratchpad."""
    budget = PromptBudget(context_tokens=4096, output_tokens=768, scratchpad_tokens=768)
    assert budget.input_tokens == 4096 - 768 - 768 - budget.system_tokens


def test_fit_text_inserisce_marker_quando_tronca() -> None:
    """Un testo lungo viene ridotto con marker esplicito."""
    testo = "alfa " * 400
    ridotto = fit_text(testo, max_tokens=20)
    assert len(ridotto) < len(testo)
    assert "contenuto ridotto" in ridotto


def test_limita_scratchpad_rispetta_il_budget() -> None:
    """Observation vecchie diventano placeholder e il testo resta sotto il tetto."""
    passi = [
        ("leggi_documento", "a.py", "Thought: leggo", "HEAD " + ("x" * 4000) + " TAIL"),
        ("elenca_cartella", "src", "Thought: elenco", "uno\ndue\ntre\n" * 80),
        ("scrivi_file", "b.py", "Thought: scrivo", "File scritto con successo: b.py"),
        ("leggi_documento", "b.py", "Thought: rileggo", "contenuto finale"),
    ]
    limitati = limita_passi_scratchpad(passi, max_tokens=120)
    assert len(limitati) == len(passi)
    assert "omessa" in limitati[0][1] or "contenuto ridotto" in limitati[0][1]
    assert "File scritto" in limitati[2][1] or "omessa" in limitati[2][1]
    testo = "".join(f"{log}\nObservation: {obs}\n" for log, obs in limitati)
    assert estimate_tokens(testo) <= 180


def test_calibrazione_token_resta_nel_clamp() -> None:
    """Il rapporto reale aggiorna la stima, senza uscire da 2.0–4.5 byte per token."""
    reset_token_estimate_for_tests()
    note_prompt_bytes("a" * 1000)
    calibrate_token_estimate(100)
    assert estimate_tokens("a" * 100) == 23
    note_prompt_bytes("a" * 1000)
    calibrate_token_estimate(10000)
    assert estimate_tokens("abcd") >= 1
    reset_token_estimate_for_tests()
    assert estimate_tokens("abcd") == 2


def test_fit_observation_conserva_testa_e_coda() -> None:
    """Observation lunghe mantengono inizio e fine utili al debug."""
    testo = "HEAD-" + ("x" * 5000) + "-TAIL"
    ridotto = fit_observation(testo, max_tokens=40)
    assert ridotto.startswith("HEAD-")
    assert ridotto.endswith("-TAIL")
    assert "contenuto ridotto" in ridotto
