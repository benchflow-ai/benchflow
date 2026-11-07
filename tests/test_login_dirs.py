"""Logins kept in $CLAUDE_CONFIG_DIR and $CODEX_HOME are found (PLAN P8).

Claude Code reads its login from ``$CLAUDE_CONFIG_DIR/.credentials.json``
and Codex from ``$CODEX_HOME/auth.json`` when the variables are set. BenchFlow
looked only in ``~/.claude`` and ``~/.codex``, so such a login read as no
login: the run asked for an API key, ``bench doctor`` listed nothing, and the
crash log did not redact the file's tokens (Rust round backlog dx-100, Rust
edd7d060).
"""

from __future__ import annotations

import json

import pytest

from benchflow.agents.env import check_subscription_auth, login_label

CLAUDE_FILE = json.dumps(
    {
        "claudeAiOauth": {
            "accessToken": "fake-claude-access-0123456789abcdef",
            "refreshToken": "fake-claude-refresh-0123456789abcdef",
            "expiresAt": 4_102_444_800_000,
        }
    }
)
CODEX_FILE = json.dumps(
    {
        "auth_mode": "chatgpt",
        "tokens": {
            "access_token": "codex-access-fake-0123456789abcdef",
            "refresh_token": "codex-refresh-fake-0123456789abcdef",
        },
    }
)


@pytest.fixture
def moved(tmp_path, monkeypatch):
    """An empty home, and both logins in folders the variables name."""
    home = tmp_path / "home"
    home.mkdir()
    claude_dir = tmp_path / "elsewhere" / "claude-config"
    claude_dir.mkdir(parents=True)
    (claude_dir / ".credentials.json").write_text(CLAUDE_FILE)
    codex_dir = tmp_path / "elsewhere" / "codex-home"
    codex_dir.mkdir(parents=True)
    (codex_dir / "auth.json").write_text(CODEX_FILE)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_dir))
    monkeypatch.setenv("CODEX_HOME", str(codex_dir))
    return home, claude_dir, codex_dir


def test_a_run_finds_the_logins_the_variables_point_at(moved):
    assert check_subscription_auth("claude-agent-acp", "ANTHROPIC_API_KEY")
    assert check_subscription_auth("codex-acp", "OPENAI_API_KEY")


def test_without_the_variables_the_home_folders_count(moved, monkeypatch):
    home, _, _ = moved
    monkeypatch.delenv("CLAUDE_CONFIG_DIR")
    monkeypatch.delenv("CODEX_HOME")
    assert not check_subscription_auth("claude-agent-acp", "ANTHROPIC_API_KEY")
    assert not check_subscription_auth("codex-acp", "OPENAI_API_KEY")
    (home / ".claude").mkdir()
    (home / ".claude" / ".credentials.json").write_text(CLAUDE_FILE)
    assert check_subscription_auth("claude-agent-acp", "ANTHROPIC_API_KEY")
    # An empty variable is unset, as for the agents.
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "")
    assert check_subscription_auth("claude-agent-acp", "ANTHROPIC_API_KEY")


def test_a_variable_set_to_an_empty_folder_hides_the_home_login(moved, monkeypatch):
    # Claude Code itself reads only the folder the variable names.
    home, _, _ = moved
    (home / ".claude").mkdir()
    (home / ".claude" / ".credentials.json").write_text(CLAUDE_FILE)
    empty = home.parent / "empty"
    empty.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(empty))
    assert not check_subscription_auth("claude-agent-acp", "ANTHROPIC_API_KEY")


@pytest.mark.asyncio
async def test_the_moved_login_is_uploaded_where_the_agent_reads_it(moved, monkeypatch):
    from benchflow.agents import credentials

    uploads: list[tuple[str, str]] = []

    async def fake_upload(env, path, content, *, owner=None):
        uploads.append((path, content))

    monkeypatch.setattr(credentials, "upload_credential", fake_upload)
    await credentials.upload_subscription_auth(None, "claude-agent-acp", "/root")
    await credentials.upload_subscription_auth(None, "codex-acp", "/home/agent")
    assert uploads == [
        ("/root/.claude/.credentials.json", CLAUDE_FILE),
        ("/home/agent/.codex/auth.json", CODEX_FILE),
    ]


def test_the_login_label_names_the_variable(moved):
    env = {"_BENCHFLOW_SUBSCRIPTION_AUTH": "1"}
    assert (
        login_label("claude-agent-acp", None, env)
        == "host login ($CLAUDE_CONFIG_DIR/.credentials.json)"
    )
    assert login_label("codex-acp", None, env) == "host login ($CODEX_HOME/auth.json)"


def test_doctor_lists_the_moved_logins(moved):
    from benchflow.doctor import DoctorProbes, claude_auth, codex_auth

    home, claude_dir, codex_dir = moved
    probes = DoctorProbes(
        environ={"CLAUDE_CONFIG_DIR": str(claude_dir), "CODEX_HOME": str(codex_dir)},
        origins={},
        home=home,
    )
    claude = claude_auth(probes)
    assert claude.effective is not None
    assert claude.effective.name == "$CLAUDE_CONFIG_DIR/.credentials.json"
    assert claude.effective.kind == "login-file" and claude.effective.usable
    codex = codex_auth(probes)
    assert codex.effective is not None
    assert codex.effective.name == "$CODEX_HOME/auth.json"
    assert codex.effective.kind == "auth-json"
    # Without the variables doctor reads the home folders, as before.
    bare = DoctorProbes(environ={}, origins={}, home=home)
    assert claude_auth(bare).effective is None
    assert codex_auth(bare).effective is None


def test_the_crash_log_redacts_the_moved_login_files(moved):
    from benchflow.cli._errors import _login_file_secrets

    found = set(_login_file_secrets().values())
    assert CLAUDE_FILE in found
    assert CODEX_FILE in found


def test_a_login_files_mode_and_plan_are_not_masked(moved):
    """A Codex login with "auth_mode": "apikey" masked the word everywhere,
    doctor's own Google AI Studio link included; its tokens stay masked."""
    from benchflow._diagnostics import redact
    from benchflow.cli._errors import _secrets
    from benchflow.doctor import secret_values

    found = secret_values(
        {
            "CODEX_AUTH_JSON": json.dumps(
                {
                    "auth_mode": "apikey",
                    "OPENAI_API_KEY": "sk-fake-codex-0123456789abcdef",
                    "last_refresh": "2026-11-04T10:00:00Z",
                }
            )
        }
    )
    assert "sk-fake-codex-0123456789abcdef" in found
    assert "apikey" not in found and "2026-11-04T10:00:00Z" not in found
    secrets = _secrets([])
    link = "Google AI Studio: https://aistudio.google.com/apikey (login chatgpt)"
    assert redact(link, secrets) == link
    leaked = "access codex-access-fake-0123456789abcdef"
    assert "codex-access-fake" not in redact(leaked, secrets)


def test_login_paths_resolve_against_a_given_home_and_environment(tmp_path):
    from benchflow.agents.login_dirs import login_display, login_path

    home = tmp_path / "h"
    env = {"CLAUDE_CONFIG_DIR": "~/cc", "CODEX_HOME": str(tmp_path / "cx")}
    assert login_path("~/.claude/.credentials.json", environ=env, home=home) == (
        home / "cc" / ".credentials.json"
    )
    assert login_path("~/.codex/auth.json", environ=env, home=home) == (
        tmp_path / "cx" / "auth.json"
    )
    # Other agents' files and unset variables stay in the home folder.
    assert login_path("~/.gemini/oauth_creds.json", environ=env, home=home) == (
        home / ".gemini" / "oauth_creds.json"
    )
    assert login_path("~/.claude/.credentials.json", environ={}, home=home) == (
        home / ".claude" / ".credentials.json"
    )
    assert login_display("~/.codex/auth.json", environ=env) == "$CODEX_HOME/auth.json"
    assert login_display("~/.codex/auth.json", environ={}) == "~/.codex/auth.json"
    assert (
        login_display("~/.gemini/oauth_creds.json", environ=env)
        == "~/.gemini/oauth_creds.json"
    )
