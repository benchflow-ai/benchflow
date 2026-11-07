"""Credential writes stay private without changing tool umask (01727c09e)."""

import json
import os
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from benchflow.agents.credentials import (
    prepare_launcher_credential,
    remove_staged_credentials,
)
from benchflow.agents.registry import _MIMO_MANIFEST_LAUNCHER, AGENTS
from benchflow.native_harness.client import encode_prompt, launch_script


@pytest.mark.parametrize("agent", ["codex-acp", "openhands"])
@pytest.mark.parametrize("mask", ["0022", "0002", "0027"])
def test_credential_write_umask_does_not_reach_agent(tmp_path, agent, mask):
    stub = tmp_path / "agent"
    stub.write_text("#!/bin/sh\numask\n")
    stub.chmod(0o755)
    command = AGENTS[agent].launch_cmd
    if agent == "codex-acp":
        command = command.replace("/opt/runtime/bin/codex-acp", str(stub))
        files = [".codex/auth.json"]
    else:
        command = command.replace(
            "openhands acp --always-approve", f"{stub} acp --always-approve"
        )
        files = [".openhands/agent_settings.json"]
    run = subprocess.run(
        ["sh", "-c", f"umask {mask}; {command}"],
        env={
            "PATH": os.environ["PATH"],
            "HOME": str(tmp_path),
            "OPENAI_API_KEY": "fake",
            "BENCHFLOW_CODEX_HOME_CONFIG": "# test\n",
            "LLM_MODEL": "fake",
            "LLM_API_KEY": "fake",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert run.stdout.strip() == mask
    for name in files:
        assert (tmp_path / name).stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("mask", ["0022", "0002", "0027"])
def test_native_harness_prompt_umask_does_not_reach_agent(tmp_path, mask):
    """Private prompt, image tool umask."""
    stub = tmp_path / "agent.py"
    stub.write_text(
        "import os, sys\n"
        "print(f'{os.umask(0):04o}', flush=True)\n"
        "assert os.fstat(0).st_mode & 0o777 == 0o600\n"
        "assert sys.stdin.read() == 'private prompt\\n'\n"
    )
    command = launch_script(sys.executable, (str(stub),), "umask-test")
    command = command.replace("/tmp/runtime-native-umask-test", str(tmp_path / "run"))
    run = subprocess.run(
        ["bash", "-c", f"umask {mask}; {command}"],
        input=encode_prompt("private prompt\n") + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    assert run.stdout.splitlines() == [mask, '{"benchflow_native_exit": 0}']
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("mask", ["0022", "0002", "0027"])
@pytest.mark.parametrize("proxy", [True, False], ids=["proxy", "direct"])
def test_mimo_config_umask_does_not_reach_agent(tmp_path, mask, proxy):
    """MiMo config stays 0600."""
    stub = tmp_path / "node"
    stub.write_text("#!/bin/sh\numask\n")
    stub.chmod(0o755)
    command = _MIMO_MANIFEST_LAUNCHER.replace(
        "/opt/runtime/node/bin/node", shlex.quote(str(stub))
    )
    run = subprocess.run(
        ["sh", "-c", f"umask {mask}; {command}"],
        cwd=tmp_path,
        env={
            "PATH": os.environ["PATH"],
            "HOME": str(tmp_path),
            "OPENAI_BASE_URL": "http://proxy.invalid/v1",
            "OPENAI_API_KEY": "fake",
            "BENCHFLOW_LITELLM_MODEL_ALIAS": "fake-model" if proxy else "",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert run.stdout.strip() == mask
    for name in (".config/mimocode/mimocode.json", "mimocode.json"):
        path = tmp_path / name
        if proxy:
            assert path.stat().st_mode & 0o777 == 0o600
        else:
            assert not path.exists()


@pytest.mark.parametrize("agent", ["codex-acp", "openhands", "mimo"])
@pytest.mark.parametrize(
    "kind", ["symlink", "dangling", "hardlink", "regular", "absent"]
)
async def test_config_writers_preserve_prepared_credentials(tmp_path, agent, kind):
    """Guards 092eb18ad/3c9388059: C32/C33 still apply to all scoped writers."""

    class LocalSandbox:
        async def exec(self, command, **kwargs):
            done = subprocess.run(["sh", "-c", command], capture_output=True, text=True)
            return SimpleNamespace(
                return_code=done.returncode, stdout=done.stdout, stderr=done.stderr
            )

    sandbox = LocalSandbox()
    ledger = []
    stub = tmp_path / "agent"
    stub.write_text("#!/bin/sh\numask\n")
    stub.chmod(0o755)
    if agent == "mimo":
        command = _MIMO_MANIFEST_LAUNCHER.replace(
            "/opt/runtime/node/bin/node", shlex.quote(str(stub))
        )
        names = [".config/mimocode/mimocode.json", "mimocode.json"]
    elif agent == "codex-acp":
        command = AGENTS[agent].launch_cmd.replace(
            "/opt/runtime/bin/codex-acp", shlex.quote(str(stub))
        )
        names = [".codex/auth.json"]
    else:
        command = AGENTS[agent].launch_cmd.replace(
            "openhands acp --always-approve", shlex.quote(str(stub))
        )
        names = [".openhands/agent_settings.json"]
    saved_paths = []
    for i, name in enumerate(names):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        saved = tmp_path / f"outside-ledger-{i}"
        saved_paths.append(saved)
        if kind != "dangling":
            saved.write_text("{}")
        if kind in {"symlink", "dangling"}:
            path.symlink_to(saved)
        elif kind == "hardlink":
            os.link(saved, path)
        elif kind == "regular":
            path.write_text("{}")
        await prepare_launcher_credential(
            sandbox, str(path), owner=None, record=ledger.append
        )
    run = subprocess.run(
        ["sh", "-c", f"umask 0022; {command}"],
        cwd=tmp_path,
        env={
            "PATH": os.environ["PATH"],
            "HOME": str(tmp_path),
            "OPENAI_API_KEY": "dummy-login",
            "OPENAI_BASE_URL": "http://proxy.invalid/v1",
            "BENCHFLOW_LITELLM_MODEL_ALIAS": "dummy-model",
            "BENCHFLOW_CODEX_HOME_CONFIG": "# private config\n",
            "LLM_MODEL": "dummy-model",
            "LLM_API_KEY": "dummy-login",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert run.stdout.strip() == "0022"
    assert ledger == [str(tmp_path / name) for name in names]
    for name in names:
        path = tmp_path / name
        assert not path.is_symlink()
        assert path.stat().st_mode & 0o777 == 0o600
        if name.endswith(".toml"):
            assert path.read_text() == "# private config\n"
        else:
            assert json.loads(path.read_text())
            assert "dummy-login" in path.read_text()
    for saved in saved_paths:
        assert not saved.exists() if kind == "dangling" else saved.read_text() == "{}"
    await remove_staged_credentials(sandbox, None, ledger)
    assert all(not (tmp_path / name).exists() for name in names)
