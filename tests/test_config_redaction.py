"""Regression tests for ``config.json`` secret redaction (issue #410).

``_write_config`` persists ``agent_env`` into the rollout's ``config.json``.
Before #410, only KEY/TOKEN/SECRET/PASSWORD/CREDENTIALS substrings were
filtered, which let common auth-bearing names like ``COOKIE`` and
``AUTHORIZATION`` leak into the artifact (and from there into any dashboard
that mirrors it).

These tests pin the denylist via the underlying ``_is_secret_env_key``
predicate so a future copy-paste of just the substring tuple cannot silently
narrow the filter.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from benchflow.rollout import _is_secret_env_key, _write_config


@pytest.mark.parametrize(
    "name",
    [
        # Original denylist — must remain covered.
        "OPENAI_API_KEY",
        "GITHUB_TOKEN",
        "STRIPE_SECRET",
        "DB_PASSWORD",
        "AWS_CREDENTIALS",
        # Names added in #410.
        "COOKIE",
        "SESSION_COOKIE",
        "AUTHORIZATION",
        "MY_AUTH_HEADER",
        "BEARER_TOKEN",
        "SESSION_ID",
        # Case-insensitivity — env keys may be lowercase in user dicts even if
        # the OS canonicalizes them. Redaction must catch them anyway.
        "cookie",
        "Authorization",
        "my_auth",
    ],
)
def test_secret_env_keys_are_redacted(name: str) -> None:
    assert _is_secret_env_key(name), (
        f"{name!r} should be flagged as secret-bearing for config.json"
    )


@pytest.mark.parametrize(
    "name",
    [
        "PATH",
        "HOME",
        "USER",
        "LANG",
        "NORMAL_VAR",
        "HTTP_PROXY",
        "PYTHONPATH",
    ],
)
def test_non_secret_env_keys_are_preserved(name: str) -> None:
    assert not _is_secret_env_key(name), (
        f"{name!r} should not be flagged as secret-bearing"
    )


def test_write_config_drops_secret_env_vars(tmp_path: Path) -> None:
    """End-to-end: ``config.json`` must not contain the issue #410 names."""
    agent_env = {
        # Should be redacted.
        "COOKIE": "session=secret-cookie",
        "AUTHORIZATION": "Bearer secret-auth",
        "MY_AUTH_HEADER": "Bearer secret",
        "GITHUB_TOKEN": "ghp_secret",
        "OPENAI_API_KEY": "sk-secret",
        # Should be preserved.
        "NORMAL_VAR": "keep-me",
        "PATH": "/usr/bin:/bin",
    }

    _write_config(
        tmp_path,
        task_path=tmp_path / "task",
        agent="claude",
        model="claude-haiku-4-5",
        environment="docker",
        skills_dir=None,
        sandbox_user=None,
        context_root=None,
        timeout=300,
        started_at=datetime(2026, 1, 1),
        agent_env=agent_env,
    )

    config = json.loads((tmp_path / "config.json").read_text())
    recorded = config["agent_env"]

    # The dropped keys must not appear at all (even as a key with a redacted
    # placeholder), and their values must not appear anywhere in the file.
    raw = (tmp_path / "config.json").read_text()
    for redacted_name, redacted_value in (
        ("COOKIE", "secret-cookie"),
        ("AUTHORIZATION", "secret-auth"),
        ("MY_AUTH_HEADER", "Bearer secret"),
        ("GITHUB_TOKEN", "ghp_secret"),
        ("OPENAI_API_KEY", "sk-secret"),
    ):
        assert redacted_name not in recorded
        assert redacted_value not in raw

    # Non-secret entries are preserved unchanged.
    assert recorded["NORMAL_VAR"] == "keep-me"
    assert recorded["PATH"] == "/usr/bin:/bin"


def test_write_config_drops_usage_proxy_secret_base_urls(tmp_path: Path) -> None:
    """Guards PR #568: external usage proxy path prefixes are bearer secrets."""
    secret_base = "https://usage.example.test/__benchflow/secret-prefix"
    agent_env = {
        "BENCHFLOW_PROVIDER_BASE_URL": secret_base,
        "OPENAI_BASE_URL": secret_base,
        "NORMAL_VAR": "keep-me",
    }

    _write_config(
        tmp_path,
        task_path=tmp_path / "task",
        agent="codex-acp",
        model="gpt-4.1-mini",
        environment="daytona",
        skills_dir=None,
        sandbox_user=None,
        context_root=None,
        timeout=300,
        started_at=datetime(2026, 1, 1),
        agent_env=agent_env,
    )

    raw = (tmp_path / "config.json").read_text()
    recorded = json.loads(raw)["agent_env"]

    assert "BENCHFLOW_PROVIDER_BASE_URL" not in recorded
    assert "OPENAI_BASE_URL" not in recorded
    assert "__benchflow/secret-prefix" not in raw
    assert recorded["NORMAL_VAR"] == "keep-me"
