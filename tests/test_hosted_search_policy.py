"""Hosted-search switches and rollout wiring for network_mode='denylist'.

Guards the denylist egress mode, benchflow-ai/FrontierPhysics#365: a task
keeps internet access, listed URLs and hosts are unreachable through the
loopback egress proxy, and each harness's hosted (provider-side) search tool
is switched off because the proxy cannot see those requests.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.agents.install import apply_web_tool_policy
from benchflow.agents.manifest import _SHIM_ONLY
from benchflow.agents.registry import (
    AGENT_INSTALLERS,
    AGENT_LAUNCH,
    AGENTS,
    AgentConfig,
    register_agent,
)
from benchflow.rollout import Role, Rollout, RolloutConfig, Scene
from benchflow.rollout_planes import DefaultRolloutPlanes
from benchflow.sandbox.egress_denylist import EGRESS_DENYLIST_ENV, EgressDenylist
from benchflow.task import RolloutPaths

_USAGE_UNAVAILABLE = {
    "n_input_tokens": None,
    "n_output_tokens": None,
    "n_cache_read_tokens": None,
    "n_cache_creation_tokens": None,
    "total_tokens": None,
    "cost_usd": None,
    "usage_source": "unavailable",
    "price_source": None,
}
_BLOCKED_URL = "https://arxiv.org/abs/2401.00001"
_BLOCKED_HOST = "github.com"
_DENYLIST = EgressDenylist((_BLOCKED_URL,), (_BLOCKED_HOST,))


# Registry: per-harness hosted-search switches


def _run_hosted_search_cmd(agent_name: str, home: Path) -> None:
    """Execute an agent's disallow_hosted_search_setup_cmd against a temp home."""
    cmd = AGENTS[agent_name].disallow_hosted_search_setup_cmd
    assert cmd, f"{agent_name} has no disallow_hosted_search_setup_cmd"
    result = subprocess.run(
        ["bash", "-c", cmd],
        env={"BENCHFLOW_AGENT_HOME": str(home), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, f"{agent_name} setup_cmd failed: {result.stderr}"


def _run_no_web_cmd(agent_name: str, home: Path) -> None:
    result = subprocess.run(
        ["bash", "-c", AGENTS[agent_name].disallow_web_tools_setup_cmd],
        env={"BENCHFLOW_AGENT_HOME": str(home), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_claude_hosted_search_cmd_denies_websearch_but_keeps_webfetch(tmp_path):
    """WebFetch is a local fetch that goes through the proxy, so it stays on."""
    _run_hosted_search_cmd("claude-agent-acp", tmp_path)
    settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())

    deny = settings["permissions"]["deny"]
    assert "WebSearch" in deny
    assert "WebFetch" not in deny


def test_claude_hosted_search_cmd_is_idempotent(tmp_path):
    _run_hosted_search_cmd("claude-agent-acp", tmp_path)
    _run_hosted_search_cmd("claude-agent-acp", tmp_path)
    settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())

    assert settings["permissions"]["deny"].count("WebSearch") == 1


def test_gemini_hosted_search_cmd_excludes_search_and_fetch(tmp_path):
    """Gemini's web_fetch uses the hosted urlContext path first, so both go."""
    _run_hosted_search_cmd("gemini", tmp_path)
    settings = json.loads((tmp_path / ".gemini" / "settings.json").read_text())

    excluded = settings["tools"]["exclude"]
    assert "google_web_search" in excluded
    assert "web_fetch" in excluded


@pytest.mark.parametrize(
    ("agent_name", "config_path"),
    [
        ("opencode", ".config/opencode/opencode.json"),
        ("mimo", ".config/mimocode/mimocode.json"),
    ],
)
def test_opencode_family_hosted_search_cmd_leaves_webfetch_on(
    agent_name, config_path, tmp_path
):
    _run_hosted_search_cmd(agent_name, tmp_path)
    tools = json.loads((tmp_path / config_path).read_text())["tools"]

    assert tools["websearch"] is False
    assert "webfetch" not in tools


def test_opencode_hosted_search_cmd_merges_with_no_web_settings(tmp_path):
    """Both policies write the same config file without clobbering each other."""
    _run_no_web_cmd("opencode", tmp_path)
    _run_hosted_search_cmd("opencode", tmp_path)
    tools = json.loads(
        (tmp_path / ".config" / "opencode" / "opencode.json").read_text()
    )["tools"]

    assert tools == {"webfetch": False, "websearch": False}


def test_agents_without_hosted_search_switch_keep_defaults():
    """Guards PR #1093 after OpenClaw moves out of the built-in registry."""
    for name in ("pi-acp", "harvey-lab-harness", "deepagents", "openhands"):
        assert AGENTS[name].disallow_hosted_search_setup_cmd == "", name
        assert AGENTS[name].disallow_hosted_search_launch_suffix == "", name


def test_hosted_search_fields_are_shim_only():
    """A data-only manifest cannot carry the switches; core owns them."""
    assert {
        "disallow_hosted_search_setup_cmd",
        "disallow_hosted_search_launch_suffix",
    } <= _SHIM_ONLY


# Planes: launch suffix selection


def test_codex_launch_suffix_disables_web_search_for_hosted_search():
    planes = DefaultRolloutPlanes()
    base = AGENT_LAUNCH["codex-acp"]

    assert (
        planes.agent_launch(
            "codex-acp", disallow_web_tools=False, disallow_hosted_search=True
        )
        == f"{base} -c tools.web_search=false"
    )
    assert planes.agent_launch("codex-acp", disallow_web_tools=False) == base


def test_launch_without_suffix_is_unchanged_for_hosted_search():
    planes = DefaultRolloutPlanes()
    raw_command = "not-a-real-agent --acp"

    assert (
        planes.agent_launch(
            "claude-agent-acp", disallow_web_tools=False, disallow_hosted_search=True
        )
        == AGENT_LAUNCH["claude-agent-acp"]
    )
    assert (
        planes.agent_launch(
            raw_command, disallow_web_tools=False, disallow_hosted_search=True
        )
        == raw_command
    )


def test_no_web_launch_suffix_wins_over_hosted_search_suffix():
    register_agent(
        "hosted-search-probe",
        "true",
        "probe --acp",
        disallow_web_tools_launch_suffix=" --no-web",
        disallow_hosted_search_launch_suffix=" --no-search",
    )
    try:
        planes = DefaultRolloutPlanes()
        assert (
            planes.agent_launch(
                "hosted-search-probe",
                disallow_web_tools=True,
                disallow_hosted_search=True,
            )
            == "probe --acp --no-web"
        )
        assert (
            planes.agent_launch(
                "hosted-search-probe",
                disallow_web_tools=False,
                disallow_hosted_search=True,
            )
            == "probe --acp --no-search"
        )
        assert (
            planes.agent_launch("hosted-search-probe", disallow_web_tools=False)
            == "probe --acp"
        )
    finally:
        AGENTS.pop("hosted-search-probe", None)
        AGENT_INSTALLERS.pop("hosted-search-probe", None)
        AGENT_LAUNCH.pop("hosted-search-probe", None)


# Install: apply_web_tool_policy chooses the hosted-search command


def _shell_env() -> SimpleNamespace:
    calls: list[str] = []

    async def exec_cmd(cmd, *, timeout_sec=None, **kwargs):
        calls.append(cmd)
        result = subprocess.run(
            cmd, shell=True, text=True, capture_output=True, timeout=timeout_sec
        )
        return SimpleNamespace(
            return_code=result.returncode, stdout=result.stdout, stderr=result.stderr
        )

    return SimpleNamespace(exec=exec_cmd, calls=calls)


def _probe_agent_cfg(**overrides: Any) -> AgentConfig:
    fields = {
        "name": "probe",
        "install_cmd": "true",
        "launch_cmd": "true",
        "disallow_web_tools_setup_cmd": (
            'mkdir -p "$BENCHFLOW_AGENT_HOME" && '
            'printf no-web > "$BENCHFLOW_AGENT_HOME/policy"'
        ),
        "disallow_hosted_search_setup_cmd": (
            'mkdir -p "$BENCHFLOW_AGENT_HOME" && '
            'printf hosted-search > "$BENCHFLOW_AGENT_HOME/policy"'
        ),
    }
    fields.update(overrides)
    return AgentConfig(**fields)


@pytest.mark.asyncio
async def test_apply_web_tool_policy_runs_hosted_search_cmd(tmp_path):
    env = _shell_env()
    home = tmp_path / "home"

    await apply_web_tool_policy(
        env,
        "probe",
        _probe_agent_cfg(),
        str(home),
        disallow=False,
        disallow_hosted_search=True,
    )

    assert (home / "policy").read_text() == "hosted-search"
    assert len(env.calls) == 1
    assert env.calls[0].startswith("export BENCHFLOW_AGENT_HOME=")


@pytest.mark.asyncio
async def test_apply_web_tool_policy_prefers_no_web_cmd_when_both_requested(tmp_path):
    env = _shell_env()
    home = tmp_path / "home"

    await apply_web_tool_policy(
        env,
        "probe",
        _probe_agent_cfg(),
        str(home),
        disallow=True,
        disallow_hosted_search=True,
    )

    assert (home / "policy").read_text() == "no-web"
    assert len(env.calls) == 1


@pytest.mark.asyncio
async def test_apply_web_tool_policy_is_noop_without_either_flag():
    env = MagicMock()
    env.exec = AsyncMock()

    await apply_web_tool_policy(
        env, "probe", _probe_agent_cfg(), "/home/agent", disallow=False
    )

    env.exec.assert_not_called()


@pytest.mark.asyncio
async def test_apply_web_tool_policy_skips_agents_without_hosted_search_cmd():
    env = MagicMock()
    env.exec = AsyncMock()

    await apply_web_tool_policy(
        env,
        "probe",
        _probe_agent_cfg(disallow_hosted_search_setup_cmd=""),
        "/home/agent",
        disallow=False,
        disallow_hosted_search=True,
    )

    env.exec.assert_not_called()


@pytest.mark.asyncio
async def test_apply_web_tool_policy_reports_hosted_search_failures():
    env = _shell_env()

    with pytest.raises(RuntimeError, match="Failed to apply hosted-search policy"):
        await apply_web_tool_policy(
            env,
            "probe",
            _probe_agent_cfg(disallow_hosted_search_setup_cmd="false"),
            "/home/agent",
            disallow=False,
            disallow_hosted_search=True,
        )


@pytest.mark.asyncio
async def test_apply_web_tool_policy_repairs_ownership_for_hosted_search():
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))

    await apply_web_tool_policy(
        env,
        "claude-agent-acp",
        AGENTS["claude-agent-acp"],
        "/home/agent",
        disallow=False,
        disallow_hosted_search=True,
    )

    cmd = env.exec.await_args.args[0]
    assert "chown -R agent:agent /home/agent/.claude" in cmd
    assert "WebSearch" in cmd
    assert "WebFetch" not in cmd


# Rollout wiring


def _denylist_task(network_mode: str = "denylist") -> SimpleNamespace:
    sandbox = SimpleNamespace(
        network_mode=network_mode,
        blocked_urls=[_BLOCKED_URL],
        blocked_hosts=[_BLOCKED_HOST],
        allow_internet=True,
        skills_dir=None,
        docker_image=None,
        workdir=None,
    )
    return SimpleNamespace(
        name="denylist-task",
        config=SimpleNamespace(
            sandbox=sandbox,
            agent=SimpleNamespace(prompt_prefix=None, timeout_sec=60),
        ),
    )


def _fake_sandbox() -> MagicMock:
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))
    env.stop = AsyncMock()
    env.download_file = AsyncMock()
    return env


def _fake_acp_connection() -> tuple[Any, Any, Any, str]:
    """A connect_acp result whose client and session survive disconnect()."""
    client = MagicMock()
    client.close = AsyncMock()
    client.session = None
    session = MagicMock()
    session.latest_usage_totals.return_value = None
    return client, session, MagicMock(), "agent"


def _fake_planes(env: Any) -> MagicMock:
    planes = MagicMock()
    planes.extract_usage.return_value = dict(_USAGE_UNAVAILABLE)
    planes.resolve_locked_paths.return_value = []
    planes.resolve_agent_env.side_effect = lambda _agent, _model, agent_env: dict(
        agent_env or {}
    )
    planes.agent_launch.side_effect = (
        lambda agent, *, disallow_web_tools, disallow_hosted_search=False: agent
    )
    planes.create_environment.return_value = env
    planes.ensure_litellm_runtime = AsyncMock(
        side_effect=lambda **kwargs: (kwargs["agent_env"], None)
    )
    planes.start_egress_denylist = AsyncMock()
    planes.stop_egress_denylist = AsyncMock()
    planes.stop_provider_runtime = AsyncMock()
    planes.install_agent = AsyncMock(return_value=MagicMock())
    planes.write_credential_files = AsyncMock()
    planes.upload_subscription_auth = AsyncMock()
    planes.apply_web_tool_policy = AsyncMock()
    planes.connect_acp = AsyncMock(side_effect=lambda **k: _fake_acp_connection())
    planes.connect_session_factory = AsyncMock(
        side_effect=lambda **k: (None, _fake_acp_connection()[1], None, "agent")
    )
    return planes


def _rollout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    task: Any,
    planes: Any,
    agent: str = "claude-agent-acp",
    sandbox_user: str | None = "agent",
) -> Rollout:
    task_dir = tmp_path / "task"
    task_dir.mkdir(exist_ok=True)
    (task_dir / "instruction.md").write_text("Solve it.\n")
    rollout_dir = tmp_path / "rollout"

    def fake_init_rollout(task_path, job_name, rollout_name, jobs_dir):
        for subdir in ("agent", "verifier", "artifacts", "trajectory"):
            (rollout_dir / subdir).mkdir(parents=True, exist_ok=True)
        return (
            task,
            rollout_dir,
            RolloutPaths(rollout_dir=rollout_dir),
            datetime.now(),
            "job",
            "rollout",
        )

    monkeypatch.setattr("benchflow.rollout._init_rollout", fake_init_rollout)
    cfg = RolloutConfig(
        task_path=task_dir,
        agent=agent,
        model="test-model",
        sandbox_user=sandbox_user,
        planes=planes,
    )
    return Rollout(cfg)


@pytest.mark.asyncio
async def test_setup_defers_proxy_env_until_connect(tmp_path, monkeypatch):
    """Proxy and CA vars must not reach the LiteLLM launcher: the CA bundle does not exist yet."""
    env = _fake_sandbox()
    planes = _fake_planes(env)
    rollout = _rollout(tmp_path, monkeypatch, task=_denylist_task(), planes=planes)

    await rollout.setup()

    assert rollout._egress_denylist == _DENYLIST
    assert rollout._disallow_hosted_search is True
    assert rollout._disallow_web_tools is False
    assert EGRESS_DENYLIST_ENV not in rollout._agent_env
    assert "HTTPS_PROXY" not in rollout._agent_env
    assert "BENCHFLOW_DISALLOW_WEB_TOOLS" not in rollout._agent_env
    planes.agent_launch.assert_called_once_with(
        "claude-agent-acp", disallow_web_tools=False, disallow_hosted_search=True
    )


@pytest.mark.asyncio
async def test_setup_leaves_env_alone_outside_denylist_agent_runs(
    tmp_path, monkeypatch
):
    env = _fake_sandbox()
    planes = _fake_planes(env)
    rollout = _rollout(
        tmp_path,
        monkeypatch,
        task=_denylist_task("public"),
        planes=planes,
        agent="claude-agent-acp",
    )

    await rollout.setup()

    assert rollout._egress_denylist is None
    assert rollout._disallow_hosted_search is False
    assert EGRESS_DENYLIST_ENV not in rollout._agent_env
    assert "HTTPS_PROXY" not in rollout._agent_env
    planes.agent_launch.assert_called_once_with(
        "claude-agent-acp",
        disallow_web_tools=False,
        disallow_hosted_search=False,
    )


@pytest.mark.asyncio
async def test_setup_does_not_resolve_agent_launch_for_oracle(tmp_path, monkeypatch):
    """Guards PR #1093's non-agent rollout boundary under denylist tasks."""
    env = _fake_sandbox()
    planes = _fake_planes(env)
    rollout = _rollout(
        tmp_path,
        monkeypatch,
        task=_denylist_task("denylist"),
        planes=planes,
        agent="oracle",
    )

    await rollout.setup()

    assert rollout._egress_denylist is None
    assert rollout._disallow_hosted_search is False
    assert EGRESS_DENYLIST_ENV not in rollout._agent_env
    assert "HTTPS_PROXY" not in rollout._agent_env
    planes.agent_launch.assert_not_called()


@pytest.mark.asyncio
async def test_setup_fails_closed_without_sandbox_user(tmp_path, monkeypatch):
    env = _fake_sandbox()
    planes = _fake_planes(env)
    rollout = _rollout(
        tmp_path, monkeypatch, task=_denylist_task(), planes=planes, sandbox_user=None
    )

    with pytest.raises(
        ValueError, match="network_mode='denylist' requires a sandbox_user"
    ):
        await rollout.setup()

    planes.create_environment.assert_not_called()


@pytest.mark.asyncio
async def test_install_agent_requests_hosted_search_policy(tmp_path, monkeypatch):
    env = _fake_sandbox()
    planes = _fake_planes(env)
    planes.setup_sandbox_user = AsyncMock(return_value="/workspace")
    planes.snapshot_build_config = AsyncMock()
    planes.seed_verifier_workspace = AsyncMock()
    planes.deploy_skills = AsyncMock()
    planes.lockdown_paths = AsyncMock()
    rollout = _rollout(tmp_path, monkeypatch, task=_denylist_task(), planes=planes)
    await rollout.setup()
    env.exec.return_value = MagicMock(return_code=0, stdout="/workspace\n", stderr="")

    await rollout.install_agent()

    kwargs = planes.apply_web_tool_policy.await_args.kwargs
    assert kwargs == {"disallow": False, "disallow_hosted_search": True}


@pytest.mark.asyncio
async def test_connect_starts_proxy_before_acp_and_cleanup_stops_it(
    tmp_path, monkeypatch
):
    env = _fake_sandbox()
    planes = _fake_planes(env)
    order: list[str] = []
    planes.start_egress_denylist.side_effect = lambda *a, **k: order.append("start")
    planes.connect_acp.side_effect = lambda **k: (
        order.append("connect_acp"),
        _fake_acp_connection(),
    )[1]
    planes.stop_egress_denylist.side_effect = lambda *a, **k: order.append("stop")
    env.stop.side_effect = lambda **k: order.append("env-stop")
    rollout = _rollout(tmp_path, monkeypatch, task=_denylist_task(), planes=planes)
    await rollout.setup()

    await rollout.connect()

    litellm_kwargs = planes.ensure_litellm_runtime.await_args.kwargs
    assert litellm_kwargs["force_sandbox_local"] is True
    assert "SSL_CERT_FILE" not in litellm_kwargs["agent_env"]
    assert "HTTPS_PROXY" not in litellm_kwargs["agent_env"]
    planes.start_egress_denylist.assert_awaited_once_with(env, "agent", _DENYLIST)
    acp_env = planes.connect_acp.await_args.kwargs["agent_env"]
    assert acp_env[EGRESS_DENYLIST_ENV] == "1"
    assert acp_env["HTTPS_PROXY"] == "http://127.0.0.1:18628"
    assert acp_env["NO_PROXY"] == "127.0.0.1,localhost,::1"

    await rollout.cleanup()

    planes.stop_egress_denylist.assert_awaited_once_with(env, tmp_path / "rollout")
    assert order == ["start", "connect_acp", "stop", "env-stop"]


@pytest.mark.asyncio
async def test_connect_restarts_proxy_on_every_reconnect(tmp_path, monkeypatch):
    """A restored sandbox has no proxy running, so each connect starts it again."""
    env = _fake_sandbox()
    planes = _fake_planes(env)
    rollout = _rollout(tmp_path, monkeypatch, task=_denylist_task(), planes=planes)
    await rollout.setup()

    await rollout.connect()
    await rollout.disconnect()
    await rollout.connect()

    assert planes.start_egress_denylist.await_count == 2
    assert planes.connect_acp.await_count == 2


@pytest.mark.asyncio
async def test_connect_rejects_session_factory_agents(tmp_path, monkeypatch):
    """The uid firewall only runs on the ACP path, so a session-factory agent fails closed."""
    register_agent(
        "denylist-sf-probe",
        "true",
        "true",
        protocol="session-factory",
        session_factory="fake_mod:build_agent",
    )
    try:
        env = _fake_sandbox()
        planes = _fake_planes(env)
        rollout = _rollout(
            tmp_path,
            monkeypatch,
            task=_denylist_task(),
            planes=planes,
            agent="denylist-sf-probe",
        )
        await rollout.setup()

        with pytest.raises(
            RuntimeError, match="network_mode='denylist' requires an ACP agent"
        ):
            await rollout.connect()

        planes.start_egress_denylist.assert_not_awaited()
        planes.connect_session_factory.assert_not_awaited()
    finally:
        AGENTS.pop("denylist-sf-probe", None)
        AGENT_INSTALLERS.pop("denylist-sf-probe", None)
        AGENT_LAUNCH.pop("denylist-sf-probe", None)


@pytest.mark.asyncio
async def test_connect_as_applies_denylist_to_role_env(tmp_path):
    """A role connected without setup() derives the denylist from the task."""
    role = Role(name="coder", agent="gemini", model="gemini/test")
    cfg = RolloutConfig(
        task_path=tmp_path / "task",
        scenes=[
            Scene(
                roles=[
                    Role(name="primary", agent="claude-agent-acp", model="test-model"),
                    role,
                ]
            )
        ],
    )
    env = _fake_sandbox()
    planes = _fake_planes(env)
    planes.install_agent.return_value = AGENTS["gemini"]
    trial = Rollout.__new__(Rollout)
    trial._config = cfg
    trial._env = env
    trial._rollout_dir = tmp_path
    trial._timing = {}
    trial._agent_cwd = "/app"
    trial._phase = "idle"
    trial._task = _denylist_task()
    trial._planes = planes

    await trial.connect_as(role)

    planes.agent_launch.assert_called_once_with(
        "gemini", disallow_web_tools=False, disallow_hosted_search=True
    )
    assert (
        planes.ensure_litellm_runtime.await_args.kwargs["force_sandbox_local"] is True
    )
    assert planes.apply_web_tool_policy.await_args.kwargs == {
        "disallow": False,
        "disallow_hosted_search": True,
    }
    planes.start_egress_denylist.assert_awaited_once_with(env, "agent", _DENYLIST)
    acp_env = planes.connect_acp.await_args.kwargs["agent_env"]
    assert acp_env[EGRESS_DENYLIST_ENV] == "1"
    assert acp_env["HTTPS_PROXY"] == "http://127.0.0.1:18628"
    assert "BENCHFLOW_DISALLOW_WEB_TOOLS" not in acp_env


@pytest.mark.asyncio
async def test_connect_as_applies_denylist_when_primary_is_oracle(tmp_path):
    """Guards the oracle-primary gap found in review: a later real role still gets the denylist."""
    role = Role(name="coder", agent="gemini", model="gemini/test")
    cfg = RolloutConfig(
        task_path=tmp_path / "task",
        agent="oracle",
        scenes=[Scene(roles=[Role(name="primary", agent="oracle", model=None), role])],
    )
    env = _fake_sandbox()
    planes = _fake_planes(env)
    planes.install_agent.return_value = AGENTS["gemini"]
    trial = Rollout.__new__(Rollout)
    trial._config = cfg
    trial._env = env
    trial._rollout_dir = tmp_path
    trial._timing = {}
    trial._agent_cwd = "/app"
    trial._phase = "idle"
    trial._task = _denylist_task()
    trial._planes = planes
    trial._egress_denylist = None

    await trial.connect_as(role)

    planes.start_egress_denylist.assert_awaited_once_with(env, "agent", _DENYLIST)
    assert planes.connect_acp.await_args.kwargs["agent_env"][EGRESS_DENYLIST_ENV] == "1"


@pytest.mark.asyncio
async def test_no_web_policy_wins_over_denylist(tmp_path, monkeypatch):
    """--self-gen-no-internet must not hand the agent internet through the egress proxy."""
    env = _fake_sandbox()
    planes = _fake_planes(env)
    rollout = _rollout(tmp_path, monkeypatch, task=_denylist_task(), planes=planes)
    rollout._config = dataclasses.replace(rollout._config, self_gen_no_internet=True)

    await rollout.setup()
    await rollout.connect()

    assert rollout._disallow_web_tools is True
    assert rollout._egress_denylist is None
    planes.start_egress_denylist.assert_not_awaited()
    acp_env = planes.connect_acp.await_args.kwargs["agent_env"]
    assert acp_env["BENCHFLOW_DISALLOW_WEB_TOOLS"] == "1"
    assert "HTTPS_PROXY" not in acp_env


@pytest.mark.asyncio
async def test_task_runtime_rejects_denylist_tasks(monkeypatch):
    """The bash primitive never arms the proxy or firewall, so it fails closed."""
    from benchflow.rollout.task_runtime import TaskRuntime

    fake = MagicMock()
    fake.setup = AsyncMock()
    fake.cleanup = AsyncMock()
    fake._egress_denylist = _DENYLIST
    monkeypatch.setattr(
        "benchflow.rollout.Rollout.create", AsyncMock(return_value=fake)
    )
    runtime = TaskRuntime.__new__(TaskRuntime)
    runtime._started = False
    runtime.config = MagicMock()

    with pytest.raises(RuntimeError, match="requires an ACP agent rollout"):
        await runtime.start()
    fake.cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_connect_as_skips_denylist_for_oracle_role(tmp_path):
    role = Role(name="checker", agent="oracle", model=None)
    cfg = RolloutConfig(task_path=tmp_path / "task", scenes=[Scene(roles=[role])])
    env = _fake_sandbox()
    planes = _fake_planes(env)
    trial = Rollout.__new__(Rollout)
    trial._config = cfg
    trial._env = env
    trial._rollout_dir = tmp_path
    trial._timing = {}
    trial._agent_cwd = "/app"
    trial._phase = "idle"
    trial._task = _denylist_task()
    trial._planes = planes

    await trial.connect_as(role)

    planes.agent_launch.assert_called_once_with(
        "oracle", disallow_web_tools=False, disallow_hosted_search=False
    )
    planes.start_egress_denylist.assert_not_awaited()
    assert EGRESS_DENYLIST_ENV not in planes.connect_acp.await_args.kwargs["agent_env"]
