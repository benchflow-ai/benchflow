"""CONF-R2-1: inherited values cannot change a connection's writer set.

Run the shipped writer prefixes against dummy files; substitute only the
final third-party executable. These are contained filesystem tests, not live
provider or benchmark trials. Each writer runs as the sandbox runs it since
#80: its deployed bytes and the launch environment under the agent's
neutral names, and the image's values under those names too.
"""

import base64
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchflow.agents import pi_acp_launcher
from benchflow.agents.codex_config import apply_codex_launch_config
from benchflow.agents.credentials import remove_staged_credentials
from benchflow.agents.env import CREDENTIAL_POLICY_MARKER
from benchflow.agents.launch_env import agent_launch_env
from benchflow.agents.registry import (
    _MIMO_MANIFEST_LAUNCHER,
    AGENTS,
    CODEX_ACP_BUILTIN_LAUNCH,
    _opencode_family_proxy_wrapper_install,
)
from benchflow.agents.visible_runtime import (
    visible_env,
    visible_launch,
    visible_script,
)
from tests.test_cli_other_confirm3 import LocalSandbox
from tests.test_sweep_credential_lifecycle import (
    Sandbox,
    _rollout,
    host,  # noqa: F401
    no_kept_services,  # noqa: F401
)

BASE = "http://dummy.invalid"
ALIAS = "dummy-model"
KEY = "dummy-login"
PROVIDER = json.dumps(
    {
        "model_provider": "dummy",
        "model_providers": {"dummy": {"base_url": BASE, "api_key": KEY}},
    }
)
TASK = '{"task_answer":true}'

# Every input which can enable a built-in destination, individually inherited.
INHERITED = [
    ("mimo", {"OPENAI_BASE_URL": BASE}, {"BENCHFLOW_LITELLM_MODEL_ALIAS": ALIAS}),
    ("mimo", {"BENCHFLOW_LITELLM_MODEL_ALIAS": ALIAS}, {"OPENAI_BASE_URL": BASE}),
    ("codex-acp", {"OPENAI_API_KEY": KEY}, {}),
    ("codex-acp", {"BENCHFLOW_CODEX_HOME_CONFIG": "dummy = 'login'"}, {}),
    (
        "pi-acp",
        {"BENCHFLOW_PROVIDER_PROTOCOL": "openai-completions"},
        {"BENCHFLOW_PROVIDER_BASE_URL": BASE},
    ),
    (
        "pi-acp",
        {"BENCHFLOW_PROVIDER_BASE_URL": BASE},
        {"BENCHFLOW_PROVIDER_PROTOCOL": "openai-completions"},
    ),
    ("opencode", {"BENCHFLOW_LITELLM_MODEL_ALIAS": ALIAS}, {}),
    ("opencode", {"BENCHFLOW_OPENCODE_GOOGLE_BASE_URL": BASE}, {}),
]
CASES = (
    [
        pytest.param(agent, inherited, explicit, False, id=f"inherited-on-{i}")
        for i, (agent, inherited, explicit) in enumerate(INHERITED)
    ]
    + [
        pytest.param(
            agent,
            inherited,
            {**explicit, **dict.fromkeys(inherited, "")},
            False,
            id=f"explicit-off-{i}",
        )
        for i, (agent, inherited, explicit) in enumerate(INHERITED)
    ]
    + [
        pytest.param(
            agent,
            dict.fromkeys(inherited, ""),
            {**explicit, **inherited},
            True,
            id=f"inherited-off-{i}",
        )
        for i, (agent, inherited, explicit) in enumerate(INHERITED)
        if "BENCHFLOW_CODEX_HOME_CONFIG" not in inherited
    ]
    + [
        pytest.param(
            "codex-acp",
            {"BENCHFLOW_CODEX_HOME_CONFIG": ""},
            {"CODEX_CONFIG": PROVIDER},
            False,
            id="codex-provider-on",
        ),
        pytest.param(
            "codex-acp",
            {"OPENAI_API_KEY": "dummy-image-login"},
            {"OPENAI_API_KEY": KEY, CREDENTIAL_POLICY_MARKER: '{"route":"none"}'},
            False,
            id="codex-unadmitted-key",
        ),
        pytest.param(
            "openhands", {"LLM_API_KEY": KEY}, {}, True, id="openhands-unconditional"
        ),
        pytest.param(
            "opencode",
            {"BENCHFLOW_LITELLM_MODEL_ALIAS": ALIAS},
            {"BENCHFLOW_LITELLM_MODEL_ALIAS": " \t\n"},
            False,
            id="opencode-blank",
        ),
    ]
)


def writer_command(agent, tmp_path):
    if agent == "mimo":
        return _MIMO_MANIFEST_LAUNCHER.rsplit("exec /opt/runtime/node", 1)[0]
    if agent == "codex-acp":
        return CODEX_ACP_BUILTIN_LAUNCH.rsplit("exec ", 1)[0] + "true"
    if agent == "openhands":
        return (
            AGENTS[agent].launch_cmd.split(
                'if [ "${BENCHFLOW_OPENHANDS_DISABLE_SUBAGENTS', 1
            )[0]
            + "true"
        )
    if agent == "pi-acp":
        # The launcher as _install_python_script deploys it.
        launcher = tmp_path / "pi_acp_launcher.py"
        launcher.write_text(visible_script(Path(pi_acp_launcher.__file__).read_text()))
        script = f"import runpy; runpy.run_path({str(launcher)!r})['setup_provider']()"
        return shlex.join([sys.executable, "-I", "-c", script])
    node = shutil.which("node")
    if node is None:
        pytest.skip("OpenCode writer needs Node")
    install = _opencode_family_proxy_wrapper_install(
        "opencode", ".config/opencode/opencode.json"
    )
    wrapper = base64.b64decode(shlex.split(install)[2]).decode()
    return wrapper.split('if [ "$(head -c2', 1)[0].replace(
        "/opt/runtime/node/bin/node", shlex.quote(node)
    )


@pytest.mark.parametrize("backend", ["docker", "remote-docker", "daytona", "modal"])
@pytest.mark.parametrize("wrapped", [False, True], ids=["builtin", "acpx"])
@pytest.mark.parametrize("agent,inherited,explicit,writes", CASES)
async def test_ledger_matches_actual_writer_with_inherited_environment(
    tmp_path, monkeypatch, backend, wrapped, agent, inherited, explicit, writes
):
    trial = _rollout(tmp_path, Sandbox(monkeypatch), "oracle", None, {})
    trial._config.environment = backend
    trial._env = LocalSandbox()
    trial._agent_cwd = str(tmp_path / "workspace")
    Path(trial._agent_cwd).mkdir()
    home_dir = tmp_path / "login-home"
    home_dir.mkdir()
    name = f"acpx:{agent}" if wrapped else agent
    if wrapped and agent == "codex-acp" and "CODEX_CONFIG" in explicit:
        # acpx replaces the direct launch command, and the runtime deliberately
        # does not call the direct Codex TOML producer for that spelling.
        writes = False
    config = replace(AGENTS[agent], name=name, credential_files=[])
    monkeypatch.setitem(AGENTS, name, config)
    # Exercise launcher ownership only; upload ownership has separate tests.
    trial._planes.write_credential_files = AsyncMock()
    trial._planes.upload_subscription_auth = AsyncMock()
    trial._planes.apply_web_tool_policy = AsyncMock()

    def pin(_agent, values):
        values.update(
            HOME=str(home_dir),
            BENCHFLOW_AGENT_HOME=str(home_dir),
            CODEX_HOME=str(home_dir / ".codex"),
        )
        return str(home_dir)

    trial._pin_credential_homes = pin
    paths = [
        Path(p.format(home=home_dir, workspace=trial._agent_cwd))
        for p in config.launcher_credential_files
    ]
    if agent == "codex-acp":
        paths.append(home_dir / ".codex/config.toml")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(TASK)
    values = dict(explicit)
    await trial._stage_connection_credentials(
        name,
        values,
        config,
        None,
        key=(name, None, ()),
        disallow_web_tools=False,
        disallow_hosted_search=False,
    )
    values, _ = apply_codex_launch_config(
        name, values, model=None, reasoning_effort=None, sandboxed=True
    )
    launch = agent_launch_env(name, None, values)
    done = subprocess.run(
        ["sh", "-c", visible_launch(writer_command(agent, tmp_path))],
        cwd=trial._agent_cwd,
        env={"PATH": os.defpath, **visible_env(inherited), **visible_env(launch)},
        capture_output=True,
        text=True,
    )
    # Pi deliberately refuses an explicitly empty endpoint before any write.
    if (
        agent == "pi-acp"
        and not writes
        and explicit.get("BENCHFLOW_PROVIDER_PROTOCOL") == "openai-completions"
    ):
        assert done.returncode != 0
    else:
        assert done.returncode == 0, done.stderr
    changed = {str(p) for p in paths if p.read_text() != TASK}
    assert bool(changed) is writes
    assert set(trial._credential_ledger) == changed
    await remove_staged_credentials(trial._env, None, trial._credential_ledger)
    for path in paths:
        if str(path) in changed:
            assert not path.exists()
        else:
            assert path.read_text() == TASK


@pytest.mark.parametrize("link", ["symlink", "hardlink"])
@pytest.mark.parametrize("destination", ["home", "workspace"])
async def test_mimo_inherited_endpoint_never_writes_through_old_alias(
    tmp_path, monkeypatch, link, destination
):
    trial = _rollout(tmp_path, Sandbox(monkeypatch), "oracle", None, {})
    trial._env = LocalSandbox()
    trial._agent_cwd = str(tmp_path)
    trial._planes.write_credential_files = AsyncMock()
    trial._planes.upload_subscription_auth = AsyncMock()
    home_dir = tmp_path / "login-home"
    home_dir.mkdir()
    trial._pin_credential_homes = lambda *args: str(home_dir)
    path = (
        home_dir / ".config/mimocode/mimocode.json"
        if destination == "home"
        else tmp_path / "mimocode.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    victim = tmp_path / "saved-config"
    victim.write_text(TASK)
    if link == "symlink":
        path.symlink_to(victim)
    else:
        path.hardlink_to(victim)
    values = {
        "HOME": str(home_dir),
        "BENCHFLOW_LITELLM_MODEL_ALIAS": ALIAS,
        "OPENAI_API_KEY": KEY,
    }
    await trial._stage_connection_credentials(
        "mimo",
        values,
        AGENTS["mimo"],
        None,
        key=("mimo", None, ()),
        disallow_web_tools=False,
        disallow_hosted_search=False,
    )
    subprocess.run(
        ["sh", "-c", visible_launch(writer_command("mimo", tmp_path))],
        cwd=tmp_path,
        env={
            "PATH": os.defpath,
            "OPENAI_BASE_URL": BASE,
            **visible_env(agent_launch_env("mimo", None, values)),
        },
        check=True,
    )
    assert victim.read_text() == TASK
    assert trial._credential_ledger == []
    await remove_staged_credentials(trial._env, None, trial._credential_ledger)
    assert path.read_text() == TASK


async def test_acpx_codex_wrapper_does_not_claim_direct_launcher_toml(
    tmp_path, monkeypatch
):
    from benchflow.agents.registry import resolve_agent

    config = resolve_agent("acpx/codex")
    monkeypatch.setitem(AGENTS, config.name, config)
    trial = _rollout(tmp_path, Sandbox(monkeypatch), "oracle", None, {})
    trial._env = LocalSandbox()
    trial._agent_cwd = str(tmp_path)
    trial._planes.write_credential_files = AsyncMock()
    trial._planes.upload_subscription_auth = AsyncMock()
    trial._pin_credential_homes = lambda *args: str(tmp_path)
    task_output = tmp_path / ".codex/config.toml"
    task_output.parent.mkdir()
    task_output.write_text(TASK)
    binary = tmp_path / "bin/acpx"
    binary.parent.mkdir()
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    values = {"CODEX_CONFIG": PROVIDER, "HOME": str(tmp_path)}
    await trial._stage_connection_credentials(
        config.name,
        values,
        config,
        None,
        key=(config.name, None, ()),
        disallow_web_tools=False,
        disallow_hosted_search=False,
    )
    # Run the actual wrapper; only the third-party client is a no-op.
    subprocess.run(
        ["sh", "-c", config.launch_cmd],
        cwd=tmp_path,
        env={
            "PATH": f"{binary.parent}:{os.defpath}",
            **agent_launch_env(config.name, None, values),
        },
        check=True,
    )
    assert str(task_output) not in trial._credential_ledger
    await remove_staged_credentials(trial._env, None, trial._credential_ledger)
    assert task_output.read_text() == TASK
