"""Shared fixtures: tmp/test_run workspace, e2e config, Ollama gate."""

from __future__ import annotations

import shutil

import pytest

from assistant.config import (
    DEFAULT_MODEL_ALIAS,
    WORKSPACE,
    ConfigRuntime,
)
from assistant.console import ConsoleUI
from assistant.ollama_runtime import OllamaRuntime
from assistant.prompt_budget import reset_token_estimate_for_tests
from assistant.session import AgentSession
from assistant.tools import reset_turn_state

pytest_plugins = ["tests.report_plugin"]

DIR_TEST_REL = "tmp/test_run"
DIR_UNIT_REL = f"{DIR_TEST_REL}/unit"
DIR_E2E_REL = f"{DIR_TEST_REL}/e2e"


@pytest.fixture(autouse=True)
def isolated_token_estimate():
    """Reset the byte/token calibration: it is global and must not leak between tests."""
    reset_token_estimate_for_tests()
    yield
    reset_token_estimate_for_tests()


@pytest.fixture
def workspace_test_dir():
    """Prepare an empty tmp/test_run (unit/ + e2e/) and remove it after the test."""
    destination = WORKSPACE / DIR_TEST_REL
    if destination.exists():
        shutil.rmtree(destination)
    (destination / "unit").mkdir(parents=True)
    (destination / "e2e").mkdir(parents=True)
    reset_turn_state()
    yield destination
    shutil.rmtree(destination, ignore_errors=True)
    reset_turn_state()


@pytest.fixture
def config_e2e() -> ConfigRuntime:
    """E2E config: default alias (fixed q4 quantization in the matrix)."""
    return ConfigRuntime(
        verbose=False,
        full_verification=False,
        burn_protection=False,
        model=DEFAULT_MODEL_ALIAS,
    )


@pytest.fixture
def agent_session(config_e2e: ConfigRuntime) -> AgentSession:
    """Agent session with a non-verbose UI (suitable for non-TTY pytest)."""
    ui = ConsoleUI()
    ui.set_verbose(config_e2e.verbose)
    return AgentSession(ui)


@pytest.fixture(scope="session")
def ollama_available() -> tuple[bool, str]:
    """
    Probe Ollama only once per pytest session.

    Uses OllamaRuntime so the fallback to installed tags matches the agent.
    Returns:
        (ok, detail): detail is the active tag or the error message.
    """
    runtime = OllamaRuntime()
    try:
        # Minimal budgets: a few tokens are enough to check availability.
        llm = runtime.get_llm(DEFAULT_MODEL_ALIAS, 512, 8)
        llm.invoke("ok")
        return True, runtime.active_tag or "sconosciuto"
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"
    finally:
        # Do not unload: the following e2e tests reuse the same warm artifact.
        runtime.close(unload=False)


@pytest.fixture
def requires_ollama(ollama_available: tuple[bool, str]):
    """Skip the test if Ollama or the resolved model is not available."""
    ok, detail = ollama_available
    if not ok:
        pytest.skip(
            f"Ollama/modello {DEFAULT_MODEL_ALIAS} "
            f"non disponibile: {detail}"
        )
