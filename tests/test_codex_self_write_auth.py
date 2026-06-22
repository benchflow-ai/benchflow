"""codex-acp self-writes ~/.codex/auth.json in its own launcher.

Decoupling: the auth-file write moves out of core's credentials.py
(`write_credential_files`, keyed off the `_SHIM_ONLY` `credential_files` field a
data-only manifest can't carry) and into codex's `launch_cmd`, so the manifest
agent is self-contained — exactly as mimo/opencode self-write their config. The
write is conditional on OPENAI_API_KEY so subscription/host-auth mode is never
clobbered, and byte-identical to the former `credential_files` template.
"""

from __future__ import annotations

import json
import os
import subprocess

from benchflow.agents.registry import AGENTS


def _write_prefix(launch_cmd: str) -> str:
    """The launcher is ``<auth-write...>; exec <agent>``; return the part before exec."""
    return launch_cmd.split("; exec ", 1)[0]


def test_credential_files_relocated_off_core():
    cfg = AGENTS["codex-acp"]
    assert cfg.credential_files == []  # no longer core's write_credential_files job
    assert ".codex/auth.json" in cfg.launch_cmd
    assert "exec " in cfg.launch_cmd  # process is replaced by codex


def test_launcher_writes_auth_json_byte_identical_when_key_set(tmp_path):
    prefix = _write_prefix(AGENTS["codex-acp"].launch_cmd)
    env = {
        **os.environ,
        "OPENAI_API_KEY": "sk-test-123",
        "BENCHFLOW_AGENT_HOME": str(tmp_path),
    }
    subprocess.run(["sh", "-c", prefix], env=env, check=True)
    auth = tmp_path / ".codex" / "auth.json"
    assert auth.exists()
    # byte-identical to the old credential_files template '{"OPENAI_API_KEY": "{value}"}'
    assert auth.read_text() == '{"OPENAI_API_KEY": "sk-test-123"}'
    assert json.loads(auth.read_text()) == {"OPENAI_API_KEY": "sk-test-123"}


def test_launcher_does_not_clobber_subscription_auth_when_key_unset(tmp_path):
    prefix = _write_prefix(AGENTS["codex-acp"].launch_cmd)
    codex = tmp_path / ".codex"
    codex.mkdir(parents=True)
    (codex / "auth.json").write_text(
        "HOST-SUBSCRIPTION-CREDS"
    )  # placed by subscription_auth
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    env["BENCHFLOW_AGENT_HOME"] = str(tmp_path)
    subprocess.run(["sh", "-c", prefix], env=env, check=True)
    assert (codex / "auth.json").read_text() == "HOST-SUBSCRIPTION-CREDS"  # untouched
