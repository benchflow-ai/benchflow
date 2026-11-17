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
from benchflow.skill_policy import SKILL_MODE_NO_SKILL, resolve_task_skill_policy


def _no_skill_policy(task_path: Path):
    return resolve_task_skill_policy(
        task_path=task_path,
        skill_mode=SKILL_MODE_NO_SKILL,
        runtime_skills_dir=None,
        declared_sandbox_skills_dir=None,
    )


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
        "DEFAULT_AUTH_REQUEST": (
            '{"methodId":"api-key","_meta":{"api-key":{"apiKey":"sk-secret"}}}'
        ),
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
        skill_policy=_no_skill_policy(tmp_path / "task"),
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
        ("DEFAULT_AUTH_REQUEST", "apiKey"),
    ):
        assert redacted_name not in recorded
        assert redacted_value not in raw

    # Non-secret entries are preserved unchanged.
    assert recorded["NORMAL_VAR"] == "keep-me"
    assert recorded["PATH"] == "/usr/bin:/bin"


@pytest.mark.parametrize(
    ("name", "value", "recorded"),
    [
        ("TESTS_PASS", "hunter2FAKE", False),
        ("TESTS_PASS", "12", True),
        ("GATES_PASS", "3", True),
        ("BLOCKER_PASS", "3", True),
        ("FIRST_PASS", "hunter2FAKE", False),
        ("DB_PASS", "hunter2FAKE", False),
        ("SSH_PASSPHRASE", "hunter2FAKE", False),
        ("HAS_PASSPHRASE", "false", True),
        ("SSH_KEY_PASSPHRASE", "false", False),
    ],
)
def test_write_config_records_a_pass_outcome_only_with_a_test_result(
    tmp_path: Path, name: str, value: str, recorded: bool
) -> None:
    """Review of PR #173 (RPUB, 2026-11-13), the lead's option 2:
    ``--agent-env TESTS_PASS=<opaque>`` was recorded in config.json while
    the trajectory redactor masked it. A test's or an outcome's pass name is
    recorded only while it holds a test result (entry_kind)."""
    _write_config(
        tmp_path,
        task_path=tmp_path / "task",
        agent="claude",
        model="claude-haiku-4-5",
        environment="docker",
        skill_policy=_no_skill_policy(tmp_path / "task"),
        sandbox_user=None,
        context_root=None,
        timeout=300,
        started_at=datetime(2026, 1, 1),
        agent_env={name: value},
    )
    raw = (tmp_path / "config.json").read_text()
    assert (name in json.loads(raw)["agent_env"]) is recorded
    if not recorded and value not in ("true", "false"):
        # (a switch word is in config.json anyway, as a JSON boolean)
        assert value not in raw


@pytest.mark.parametrize(
    ("name", "value", "recorded"),
    [
        ("HTTP_COOKIE", "theme=dark; sid=FAKEsid0123", False),
        ("HTTP_COOKIE", "12", True),
        ("HTTP_COOKIE", "8f14e45fceea167a5a36dedd4bea2543", False),
        ("SESSION_COOKIE", "8f14e45fceea167a5a36dedd4bea2543", False),
        ("NOTION_COOKIE", "FAKEsess0aZ9bY7Lm2Qx", False),
        ("accept_cookie", "true", True),
        # A name that ends in `authorization` holds a credential, prose
        # included (the lead, 2026-11-16 10:53Z; F2b, 2026-11-16 00:25Z: an
        # over-mask in the record only; the agent still gets the variable).
        ("HTTP_AUTHORIZATION", "required", False),
        ("HTTP_AUTHORIZATION", "Bearer FAKEtoken0123", False),
        ("REDIRECT_HTTP_AUTHORIZATION", "please sign in first", False),
        ("Authorization", "required", False),
        ("COOKIE", "strict", False),
    ],
)
def test_write_config_records_a_header_name_by_its_value(
    tmp_path: Path, name: str, value: str, recorded: bool
) -> None:
    """The lead's follow-up PR (2026-11-14): the header value rule is
    entry_kind's. ``--agent-env HTTP_COOKIE=<pairs>`` stays out of
    config.json, and so do ``HTTP_AUTHORIZATION=required`` (every name that
    ends in `authorization` holds a credential) and the header's own name
    (Authorization, COOKIE), whatever they hold."""
    _write_config(
        tmp_path,
        task_path=tmp_path / "task",
        agent="claude",
        model="claude-haiku-4-5",
        environment="docker",
        skill_policy=_no_skill_policy(tmp_path / "task"),
        sandbox_user=None,
        context_root=None,
        timeout=300,
        started_at=datetime(2026, 1, 1),
        agent_env={name: value},
    )
    raw = (tmp_path / "config.json").read_text()
    assert (name in json.loads(raw)["agent_env"]) is recorded
    if not recorded:
        assert value not in raw


@pytest.mark.parametrize(
    ("name", "value", "recorded"),
    [
        ("dbPass", "FAKEpass0aZ9", False),
        ("adminPass", "FAKEpass0aZ9", False),
        ("smtpPwd", "FAKEpass0aZ9", False),
        ("dbPass_2", "FAKEpass0aZ9", False),
        ("dbPassValue", "FAKEpass0aZ9", False),
        ("testsPass", "FAKEpass0aZ9", False),
        ("maxTokens", "4096", True),
        ("testsPass", "12", True),
        ("ByPass", "on", True),
    ],
)
def test_write_config_reads_a_camel_case_name_as_written(
    tmp_path: Path, name: str, value: str, recorded: bool
) -> None:
    """The lead's ruling (2026-11-14): the env filter read a name in upper
    case only, where dbPass is DBPASS, no password's name, so config.json
    recorded dbPass, adminPass, smtpPwd, dbPass_2 and dbPassValue. It reads
    the name's words too (dbPass as DB_PASS); a camel-case setting is still
    recorded, and testsPass holding a password is not."""
    assert _is_secret_env_key(name) is (name != "ByPass")
    _write_config(
        tmp_path,
        task_path=tmp_path / "task",
        agent="claude",
        model="claude-haiku-4-5",
        environment="docker",
        skill_policy=_no_skill_policy(tmp_path / "task"),
        sandbox_user=None,
        context_root=None,
        timeout=300,
        started_at=datetime(2026, 1, 1),
        agent_env={name: value},
    )
    raw = (tmp_path / "config.json").read_text()
    assert (name in json.loads(raw)["agent_env"]) is recorded
    if not recorded:
        assert value not in raw


@pytest.mark.parametrize(
    ("name", "value", "recorded"),
    [
        ("input_tokens_2", "1234", True),
        ("tokens_5m", "5", True),
        ("tokens_1h", "1234", True),
        ("TOKENS_2", "FAKEtokA,FAKEtokB", False),
        ("ACCESS_TOKEN_2", "FAKEtok0aZ9bY7Lm2Qx", False),
    ],
)
def test_write_config_records_a_numbered_token_count(
    tmp_path: Path, name: str, value: str, recorded: bool
) -> None:
    """The lead's rulings (2026-11-14): a token count stays a count with a
    number after its name, and a list of tokens or a token does not."""
    _write_config(
        tmp_path,
        task_path=tmp_path / "task",
        agent="claude",
        model="claude-haiku-4-5",
        environment="docker",
        skill_policy=_no_skill_policy(tmp_path / "task"),
        sandbox_user=None,
        context_root=None,
        timeout=300,
        started_at=datetime(2026, 1, 1),
        agent_env={name: value},
    )
    raw = (tmp_path / "config.json").read_text()
    assert (name in json.loads(raw)["agent_env"]) is recorded
    if not recorded:
        assert value not in raw


@pytest.mark.parametrize(
    ("value", "recorded"),
    [
        ("/run/x.json", True),
        ('{"type": "service_account", "project_id": "fake-project"}', False),
    ],
)
def test_write_config_records_a_credentials_file_path_only(
    tmp_path: Path, value: str, recorded: bool
) -> None:
    """The lead's ruling (2026-11-14): --agent-env
    GOOGLE_APPLICATION_CREDENTIALS=/run/x.json is recorded as given, and
    the service account's JSON inline is withheld (it was recorded when it
    held no private key)."""
    name = "GOOGLE_APPLICATION_CREDENTIALS"
    _write_config(
        tmp_path,
        task_path=tmp_path / "task",
        agent="claude",
        model="claude-haiku-4-5",
        environment="docker",
        skill_policy=_no_skill_policy(tmp_path / "task"),
        sandbox_user=None,
        context_root=None,
        timeout=300,
        started_at=datetime(2026, 1, 1),
        agent_env={name: value},
    )
    raw = (tmp_path / "config.json").read_text()
    if recorded:
        assert json.loads(raw)["agent_env"][name] == value
    else:
        assert name not in json.loads(raw)["agent_env"]
        assert "fake-project" not in raw


def test_write_config_drops_litellm_secret_base_urls(tmp_path: Path) -> None:
    """LiteLLM URLs with BenchFlow secret path segments must be redacted."""
    secret_base = "https://litellm.example.test/__benchflow/secret-prefix"
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
        skill_policy=_no_skill_policy(tmp_path / "task"),
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
