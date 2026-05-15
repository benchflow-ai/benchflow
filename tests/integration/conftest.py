"""Shared fixtures and skip logic for integration tests.

Integration tests hit real infrastructure (Daytona, LLM APIs) and are
gated behind the ``integration`` marker so a bare ``pytest`` never runs
them.  Invoke with::

    pytest -m integration tests/integration/

Environment variables required:

    GEMINI_API_KEY   — Google Gemini API key (or GOOGLE_API_KEY)
    DAYTONA_API_KEY  — Daytona backend credential

Optional:

    ANTHROPIC_API_KEY   — needed for claude-agent-acp
    OPENAI_API_KEY      — needed for codex-acp
"""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SKILLSBENCH_TASKS = [
    # Low complexity
    "jax-computing-basics",
    "python-scala-translation",
    "jpg-ocr-stat",
    # Medium complexity
    "grid-dispatch-operator",
    "threejs-to-obj",
    "data-to-d3",
    # High complexity
    "lake-warming-attribution",
    "weighted-gdp-calc",
    "shock-analysis-supply",
]

ALL_AGENTS = [
    "claude-agent-acp",
    "pi-acp",
    "openclaw",
    "codex-acp",
    "gemini",
    "opencode",
    "harvey-lab-harness",
    "openhands",
]

DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_ENVIRONMENT = "daytona"
DEFAULT_CONCURRENCY = 30

# Agent → required env var(s). Agent is skipped when none are set.
# For claude-agent-acp, OAuth tokens also count as valid credentials.
AGENT_REQUIRED_KEYS: dict[str, list[str]] = {
    "claude-agent-acp": [
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
    ],
    "pi-acp": [],
    "openclaw": [],
    "codex-acp": ["OPENAI_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "opencode": [],
    "harvey-lab-harness": [],
    "openhands": [],
}

# Agent → model override when DEFAULT_MODEL is incompatible with the agent's
# API protocol.  codex-acp speaks OpenAI Responses API and cannot use a
# Gemini model string.
AGENT_MODEL_OVERRIDES: dict[str, str] = {
    "claude-agent-acp": "claude-haiku-4-5-20251001",
    "codex-acp": "gpt-5.4-nano",
}

# Subscription auth files that substitute for API keys.
SUBSCRIPTION_AUTH_FILES: dict[str, str] = {
    "claude-agent-acp": "~/.claude/.credentials.json",
    "codex-acp": "~/.codex/auth.json",
}

HELLO_TASK = Path(__file__).parent.parent / "examples" / "hello-world-task"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_gemini_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def _has_daytona() -> bool:
    return bool(os.environ.get("DAYTONA_API_KEY"))


def has_creds_for_agent(agent: str) -> bool:
    """Return True if the host has credentials for *agent*."""
    required = AGENT_REQUIRED_KEYS.get(agent, [])
    if not required:
        # Agents with no explicit key requirement need at least Gemini key
        # since we drive them with gemini-3.1-flash-lite-preview.
        return _has_gemini_key()
    if any(os.environ.get(k) for k in required):
        return True
    sub_file = SUBSCRIPTION_AUTH_FILES.get(agent)
    return bool(sub_file and Path(sub_file).expanduser().is_file())


def model_for_agent(agent: str) -> str:
    """Return the model string appropriate for *agent*."""
    return AGENT_MODEL_OVERRIDES.get(agent, DEFAULT_MODEL)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def gemini_api_key() -> str:
    """Return the Gemini API key or skip."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        pytest.skip("GEMINI_API_KEY or GOOGLE_API_KEY not set")
    return key


@pytest.fixture(scope="session")
def integration_prereqs() -> None:
    """Session-wide gate: skip all integration tests when infra is missing."""
    if not _has_gemini_key():
        pytest.skip("No GEMINI_API_KEY or GOOGLE_API_KEY")
    if not _has_daytona():
        pytest.skip("No DAYTONA_API_KEY — Daytona backend unavailable")


@pytest.fixture
def jobs_dir(tmp_path: Path) -> Iterator[Path]:
    """Provide a temporary jobs directory for a single test."""
    d = tmp_path / f"jobs-{uuid.uuid4().hex[:8]}"
    d.mkdir()
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="session")
def session_jobs_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped jobs directory (shared across tests in a session)."""
    return tmp_path_factory.mktemp("integration-jobs")


@pytest.fixture(scope="session")
def skillsbench_tasks_dir() -> Path:
    """Resolve SkillsBench tasks directory (clones if needed)."""
    from benchflow.task_download import resolve_source

    return resolve_source("benchflow-ai/skillsbench", path="tasks", ref="main")
