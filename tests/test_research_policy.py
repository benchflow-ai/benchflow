"""Regression coverage for private filtered-research runs."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.agents.registry import AGENTS
from benchflow.eval_plan import EvalCreateRequest, EvalPlanError, build_eval_plan
from benchflow.eval_sharding import EvalShard, _config_payload, _worker_payload_artifact
from benchflow.eval_worker import _evaluation_config
from benchflow.evaluation import EvaluationConfig
from benchflow.research_policy import (
    RESEARCH_GATEWAY_PORT,
    RESEARCH_MCP_NAME,
    RESEARCH_POLICY_PATH,
    ResolvedResearchPolicy,
    attach_research_mcp,
    install_research_gateway,
    load_research_policy,
)
from benchflow.rollout import (
    Rollout,
    RolloutConfig,
    _fastmcp_task_mcp_config,
    _task_mcp_specs_for_agent,
)
from benchflow.sandbox import _research_gateway_runtime as runtime
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.lockdown import enforce_agent_egress_firewall
from benchflow.task.config import SandboxConfig


def _policy_yaml(
    secret_url: str = "https://papers.example/abs/365", *, task_id: str = "task-a"
) -> str:
    return f"""\
version: 1
tasks:
  {task_id}:
    blocked_urls:
      - {secret_url}
    blocked_url_prefixes:
      - https://archive.example/private
    blocked_hosts:
      - forbidden.example
    blocked_terms:
      - Secret Paper Title
    blocked_content_sha256:
      - {hashlib.sha256(b"secret body").hexdigest()}
"""


def _resolved_policy(tmp_path) -> ResolvedResearchPolicy:
    path = tmp_path / "private-policy.yaml"
    path.write_text(_policy_yaml())
    return load_research_policy(path, task_id="task-a")


def _runtime_policy() -> runtime.Policy:
    body = b"secret body"
    return runtime.Policy(
        {
            "blocked_urls": ["https://papers.example/abs/365"],
            "blocked_url_prefixes": ["https://archive.example/private"],
            "blocked_hosts": ["forbidden.example"],
            "blocked_terms": ["Secret Paper Title"],
            "blocked_content_sha256": [hashlib.sha256(body).hexdigest()],
            "search_endpoint": "https://search.example/html/",
            "policy_sha256": "a" * 64,
        }
    )


def test_policy_resolves_one_task_without_exposing_values(tmp_path) -> None:
    """Guards FrontierPhysics #365: artifacts contain provenance, not deny rules."""
    policy = _resolved_policy(tmp_path)

    metadata = policy.artifact_metadata(enforced=False)
    serialized = json.dumps(metadata)

    assert policy.task_id == "task-a"
    assert metadata["blocked_url_count"] == 1
    assert metadata["blocked_term_count"] == 1
    assert metadata["enforced"] is False
    assert "papers.example" not in serialized
    assert "Secret Paper Title" not in serialized


def test_policy_enabled_batch_fails_closed_on_missing_task(tmp_path) -> None:
    """Guards FrontierPhysics #365 against silently unrestricted batch tasks."""
    path = tmp_path / "private-policy.yaml"
    path.write_text(_policy_yaml())

    with pytest.raises(ValueError, match="no entry for task 'task-b'"):
        load_research_policy(path, task_id="task-b")


@pytest.mark.parametrize(
    "url",
    [
        "http://papers.example/abs/365?download=1#fragment",
        "https://papers.example:443/abs/365",
        "https://papers.example/abs/365/",
        "https://papers.example/x/../abs/365",
        "https://archive.example/private/appendix.pdf",
        "https://sub.forbidden.example/anything",
    ],
)
def test_url_policy_closes_common_normalization_bypasses(url: str) -> None:
    """Guards FrontierPhysics #365 across scheme/query/dot-path/subdomain variants."""
    with pytest.raises(runtime.PolicyBlocked):
        _runtime_policy().check_url(url)


def test_content_policy_blocks_terms_and_full_body_fingerprints() -> None:
    """Guards FrontierPhysics #365 against allowed URLs leaking blocked content."""
    policy = _runtime_policy()

    with pytest.raises(runtime.PolicyBlocked):
        policy.check_body(b"This cites the SECRET PAPER TITLE.", "text/plain")
    with pytest.raises(runtime.PolicyBlocked):
        policy.check_body(b"secret body", "application/pdf")


def test_redirect_target_is_checked_before_second_connection(monkeypatch) -> None:
    """Guards FrontierPhysics #365 against redirects into a denied resource."""

    class Redirect:
        status = 302

        @staticmethod
        def getheader(name: str):
            return "https://papers.example/abs/365" if name == "Location" else None

        @staticmethod
        def read(_limit: int) -> bytes:
            return b""

    connections = []

    class Connection:
        def __init__(self, *_args) -> None:
            connections.append(self)

        def request(self, *_args, **_kwargs) -> None:
            return None

        def getresponse(self):
            return Redirect()

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        runtime, "_resolve_public_ip", lambda _host, _port: "93.184.216.34"
    )
    monkeypatch.setattr(runtime, "_PinnedHTTPSConnection", Connection)

    with pytest.raises(runtime.PolicyBlocked):
        runtime._request(
            _runtime_policy(),
            "https://allowed.example/start",
            max_bytes=1024,
        )

    assert len(connections) == 1


def test_search_filters_denied_urls_and_titles(monkeypatch) -> None:
    """Guards FrontierPhysics #365 by suppressing blocked-paper discovery results."""
    page = b"""
    <a class="result__a" href="https://allowed.example/a">Allowed result</a>
    <a class="result__a" href="https://papers.example/abs/365">Hidden by URL</a>
    <a class="result__a" href="https://mirror.example/x">Secret Paper Title</a>
    """
    monkeypatch.setattr(
        runtime,
        "_request",
        lambda *_args, **_kwargs: runtime.Response(
            "https://search.example/html/", 200, "text/html", page
        ),
    )

    assert runtime._search(_runtime_policy(), "allowed topic", 10) == [
        {"title": "Allowed result", "url": "https://allowed.example/a"}
    ]


def test_loopback_gateway_rejects_blocked_fetch_over_real_http() -> None:
    """Guards FrontierPhysics #365 at the MCP-relay/gateway HTTP boundary."""
    runtime.GatewayHandler.policy = _runtime_policy()
    server = runtime.ThreadingHTTPServer(("127.0.0.1", 0), runtime.GatewayHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}"
        with pytest.raises(runtime.GatewayError, match="blocked by research policy"):
            runtime._mcp_call(
                endpoint,
                "web_fetch",
                {"url": "https://papers.example/abs/365?mirror=1"},
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_research_mcp_spec_contains_no_private_policy_values() -> None:
    """Guards FrontierPhysics #365 across ACP and native-config harness injection."""
    task = SimpleNamespace(config=SimpleNamespace(sandbox=SandboxConfig()))

    attach_research_mcp(task)

    [server] = task.config.sandbox.mcp_servers
    assert server.name == RESEARCH_MCP_NAME
    assert server.transport == "stdio"
    assert "policy" not in " ".join(server.args).lower()
    assert "papers.example" not in json.dumps(server.model_dump())


@pytest.mark.parametrize(
    "agent",
    [
        "openhands",
        "pi-acp",
        "openclaw",
        "opencode",
        "mimo",
        "codex-acp",
        "claude-agent-acp",
    ],
)
def test_research_mcp_reaches_supported_harnesses(agent: str) -> None:
    """Guards FrontierPhysics #365 across the release integration agent roster."""
    task = SimpleNamespace(config=SimpleNamespace(sandbox=SandboxConfig()))
    attach_research_mcp(task)

    if AGENTS[agent].task_mcp_transport == "native-config":
        native = _fastmcp_task_mcp_config(task)
        assert RESEARCH_MCP_NAME in native["mcpServers"]
        assert _task_mcp_specs_for_agent(agent, task, AGENTS[agent]) == []
    else:
        specs = _task_mcp_specs_for_agent(agent, task, AGENTS[agent])
        assert [spec.name for spec in specs] == [RESEARCH_MCP_NAME]


@pytest.mark.asyncio
async def test_gateway_policy_upload_is_root_only_and_command_is_secret_free(
    tmp_path,
) -> None:
    """Guards FrontierPhysics #365: the sandbox policy never enters agent argv/env."""
    uploaded: list[tuple[str, str, str | None]] = []

    async def upload(source, target, *, mode=None):
        uploaded.append((str(target), Path(source).read_text(), mode))

    exec_calls: list[str] = []

    async def execute(command, **_kwargs):
        exec_calls.append(command)
        # Startup succeeds; the first health probe succeeds too.
        return ExecResult(stdout="", stderr="", return_code=0)

    env = SimpleNamespace(upload_file=upload, exec=execute)
    policy = _resolved_policy(tmp_path)

    await install_research_gateway(env, policy)

    policy_upload = next(item for item in uploaded if item[0] == RESEARCH_POLICY_PATH)
    assert policy_upload[2] == "600"
    assert "papers.example" in policy_upload[1]
    assert len(exec_calls) == 2
    assert all("papers.example" not in command for command in exec_calls)
    assert all("Secret Paper Title" not in command for command in exec_calls)


def test_docker_research_override_adds_net_admin_last(tmp_path) -> None:
    """Guards FrontierPhysics #365 by making UID-firewall setup fail closed."""
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox.rollout_paths = SimpleNamespace(rollout_dir=tmp_path)
    sandbox._research_policy_compose_path = None

    sandbox.configure_research_policy()

    path = sandbox._research_policy_compose_path
    assert path is not None
    assert json.loads(path.read_text()) == {
        "services": {"main": {"cap_add": ["NET_ADMIN"]}}
    }


def test_worker_private_payload_round_trip_and_public_redaction(tmp_path) -> None:
    """Guards FrontierPhysics #365 when evaluation sharding crosses a process."""
    private_path = tmp_path / "contains-paper-names.yaml"
    config = EvaluationConfig(research_policy_path=str(private_path))
    shard = EvalShard(index=0, task_names=("task-a",), concurrency=1)
    payload = _config_payload(config, shard=shard)
    artifact = _worker_payload_artifact({"config": payload})

    assert _evaluation_config(payload).research_policy_path == str(private_path)
    assert artifact["config"]["research_policy_path"] == "<private>"
    assert private_path.name not in json.dumps(artifact)


def test_eval_plan_threads_private_policy_and_rejects_unsupported_sandbox(
    tmp_path,
) -> None:
    """Guards FrontierPhysics #365 at the CLI planning boundary."""
    policy_path = tmp_path / "private.yaml"
    policy_path.write_text(_policy_yaml())
    tasks = tmp_path / "tasks"
    tasks.mkdir()

    plan = build_eval_plan(
        EvalCreateRequest(tasks_dir=tasks, research_policy=policy_path)
    )
    assert plan.make_eval_config().research_policy_path == str(policy_path.resolve())

    with pytest.raises(EvalPlanError, match="requires --sandbox docker"):
        build_eval_plan(
            EvalCreateRequest(
                tasks_dir=tasks,
                research_policy=policy_path,
                environment="daytona",
            )
        )


@pytest.mark.asyncio
async def test_rollout_setup_records_only_policy_summary_and_configures_docker(
    tmp_path,
) -> None:
    """Guards FrontierPhysics #365 through the real rollout setup composition."""
    task = Path(__file__).parent / "examples" / "hello-world-task"
    policy_path = tmp_path / "private.yaml"
    policy_path.write_text(_policy_yaml(task_id=task.name))
    rollout = Rollout(
        RolloutConfig(
            task_path=task,
            jobs_dir=tmp_path / "jobs",
            research_policy_path=policy_path,
        )
    )

    await rollout.setup()

    assert rollout._disallow_web_tools is True
    assert rollout._env._research_policy_compose_path is not None
    assert [server.name for server in rollout._task.config.sandbox.mcp_servers] == [
        RESEARCH_MCP_NAME
    ]
    artifact = json.loads((rollout._rollout_dir / "config.json").read_text())
    artifact_text = json.dumps(artifact)
    assert artifact["research_policy"]["enforced"] is False
    assert artifact["research_policy"]["blocked_url_count"] == 1
    assert "papers.example" not in artifact_text
    assert "Secret Paper Title" not in artifact_text


def _docker_available() -> bool:
    try:
        return (
            subprocess.run(
                ["docker", "info"], capture_output=True, timeout=15, check=False
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.integration
@pytest.mark.skipif(not _docker_available(), reason="docker daemon unavailable")
@pytest.mark.asyncio
async def test_research_policy_docker_egress_canary(tmp_path) -> None:
    """Guards FrontierPhysics #365 with a real Docker UID-firewall canary."""
    source = Path(__file__).parent / "examples" / "hello-world-task"
    task = tmp_path / "research-policy-canary"
    shutil.copytree(source, task)
    dockerfile = task / "environment" / "Dockerfile"
    dockerfile.write_text(
        dockerfile.read_text().replace(
            "apt-get install -y -qq curl",
            "apt-get install -y -qq curl python3 ca-certificates iptables",
        )
    )
    policy_path = tmp_path / "private.yaml"
    policy_path.write_text(
        _policy_yaml("https://example.com/blocked-answer", task_id=task.name)
    )
    rollout = Rollout(
        RolloutConfig(
            task_path=task,
            jobs_dir=tmp_path / "jobs",
            research_policy_path=policy_path,
        )
    )

    await rollout.setup()
    await rollout.start()
    try:
        cwd = await rollout._planes.setup_sandbox_user(
            rollout._env,
            "agent",
            workspace="/app",
            timeout_sec=120,
        )
        assert cwd == "/app"
        assert rollout._research_policy is not None
        await install_research_gateway(rollout._env, rollout._research_policy)
        await enforce_agent_egress_firewall(
            rollout._env,
            "agent",
            {
                "BENCHFLOW_DISALLOW_WEB_TOOLS": "1",
                "BENCHFLOW_PROVIDER_BASE_URL": "http://127.0.0.1:9999/v1",
            },
        )

        unreadable = await rollout._env.exec(
            f"test ! -r {RESEARCH_POLICY_PATH}", user="agent", timeout_sec=10
        )
        assert unreadable.return_code == 0

        health = await rollout._env.exec(
            f"curl -fsS http://127.0.0.1:{RESEARCH_GATEWAY_PORT}/health",
            user="agent",
            timeout_sec=10,
        )
        assert health.return_code == 0

        blocked = await rollout._env.exec(
            "curl -sS -o /tmp/blocked.out -w '%{http_code}' "
            f"-X POST http://127.0.0.1:{RESEARCH_GATEWAY_PORT}/fetch "
            "-H 'Content-Type: application/json' "
            '-d \'{"url":"https://example.com/blocked-answer?download=1"}\'',
            user="agent",
            timeout_sec=10,
        )
        assert blocked.return_code == 0
        assert blocked.stdout == "403"

        allowed = await rollout._env.exec(
            f"curl -fsS --max-time 20 -X POST "
            f"http://127.0.0.1:{RESEARCH_GATEWAY_PORT}/fetch "
            "-H 'Content-Type: application/json' "
            '-d \'{"url":"https://example.com/"}\'',
            user="agent",
            timeout_sec=30,
        )
        assert allowed.return_code == 0
        assert '"content"' in (allowed.stdout or "")

        bypass = await rollout._env.exec(
            "curl -fsS --max-time 5 https://example.com/",
            user="agent",
            timeout_sec=10,
        )
        assert bypass.return_code != 0
    finally:
        await rollout._env.stop(delete=True)
