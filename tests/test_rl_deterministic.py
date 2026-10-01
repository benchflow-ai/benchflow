"""Deterministic tier for the RL rollout API: real sandboxes, a scripted policy.

``bf.rollout_group`` runs ``claude-agent-acp`` in real Docker sandboxes
against the deterministic fake model served as a vLLM-shaped policy
(``tests/integration/deterministic/task/environment/fake_llm``: token ids and
logprobs from a fixed chat template, streamed, as Claude Code asks). The
policy is reached through ``bf.Policy``'s relay on loopback, as a trainer's
local server would be. No model key, no model cost.

Scenarios:

- a group streams typed results with exact token segments, helper calls kept
  out, per-call policy versions, and every call attested against the relay;
  the server's key and the relay credentials never reach the rollout files;
- group failures: an ACP handshake timeout set per group masks each attempt
  as a startup failure, retries it, and reports the phase and the timeouts;
- a root agent never finds the oracle in its sandbox;
- cancelling a group mid-flight, and killing its process (SIGTERM, SIGKILL),
  leaves no container behind (the SIGKILL case through the lease reaper).

Marked ``deterministic``; Docker only (on Daytona the gateway runs in the
sandbox and needs a public relay URL, which this tier does not stand up).
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import benchflow as bf
from benchflow.sandbox import leases
from benchflow.training.rollouts import StartupTimeouts
from tests.integration.deterministic import harness as h

pytestmark = pytest.mark.deterministic

SANDBOX, SKIP_REASON = h.select_sandbox()
POLICY_KEY = "det-policy-server-key-not-a-secret"
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def docker() -> None:
    if SANDBOX != "docker":
        pytest.skip(
            SKIP_REASON if SANDBOX is None else "the RL scenarios need a Docker sandbox"
        )


@pytest.fixture(scope="module")
def rl_root(tmp_path_factory, docker) -> Path:
    kept = os.environ.get(h.KEEP_JOBS_ENV)
    if kept:
        root = Path(kept) / "rl"
        root.mkdir(parents=True, exist_ok=True)
        return root
    return tmp_path_factory.mktemp("rl-deterministic")


@pytest.fixture(scope="module")
def fake_policy(rl_root) -> str:
    with h.host_fake_llm(rl_root / "fake-llm.jsonl") as url:
        yield url


@pytest.fixture(scope="module")
def tasks(rl_root) -> Path:
    root = rl_root / "tasks"
    root.mkdir(exist_ok=True)
    for variant in (
        h.TaskVariant("hello-pass", "hello-pass"),
        h.TaskVariant("oracle-probe", "oracle-probe"),
        h.TaskVariant("slow-pass", "pass-then-hang"),
    ):
        if not (root / variant.name).exists():
            h.materialize_task(variant, root)
    return root


@pytest.fixture(autouse=True)
def private_leases(rl_root, monkeypatch):
    monkeypatch.setenv(leases.LEASE_DIR_ENV, str(rl_root / "leases"))


def _policy(url: str, version: object = 1) -> bf.Policy:
    return bf.Policy(
        "vllm/fake-policy", base_url=f"{url}/v1", api_key=POLICY_KEY, version=version
    )


def _config(tasks: Path, name: str, rl_root: Path, **extra) -> bf.RolloutConfig:
    return bf.RolloutConfig(
        task_path=tasks / name,
        agent=h.AGENT,
        environment="docker",
        jobs_dir=rl_root / "jobs",
        **extra,
    )


def _files_containing(root: Path, needle: str) -> list[Path]:
    hits = []
    for path in root.rglob("*"):
        if path.is_file():
            try:
                if needle in path.read_text(errors="ignore"):
                    hits.append(path)
            except OSError:
                continue
    return hits


def _containers(label: str) -> list[str]:
    out = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"label={label}"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return out.stdout.split()


async def test_group_streams_typed_results_with_exact_tokens(
    fake_policy, tasks, rl_root
):
    policy = _policy(fake_policy, version=7)
    config = _config(tasks, "hello-pass", rl_root)
    async with policy:
        group = bf.rollout_group(config, n=2, policy=policy, group_id="det-stream")
        streamed = [rollout async for rollout in group]
        result = await group.wait()

    assert sorted(r.index for r in streamed) == [0, 1]
    for rollout in result.rollouts:
        assert rollout.reward == 1.0 and rollout.passed, rollout.to_dict()
        assert rollout.attribution == "score" and rollout.attribution_reason == "scored"
        assert rollout.policy_version == 7 and rollout.policy_versions == (7,)
        tokens = rollout.tokens
        assert tokens["attestation"]["status"] == "attested", tokens["attestation"]
        assert tokens["status"] == "exact", tokens["dropped"]
        assert rollout.segments, "no trainable token segment"
        for segment in rollout.segments:
            assert segment.kind == "agent" and segment.verify()
            assert len(segment.completion_ids) == len(segment.action_mask)
            assert sum(segment.action_mask) > 0 and 0 in segment.action_mask
            assert set(segment.policy_versions) == {7}
            # masked (environment) positions carry logprob 0.0
            assert all(
                lp == 0.0
                for lp, m in zip(segment.logprobs, segment.action_mask, strict=True)
                if m == 0
            )
        assert all(s.kind == "helper" for s in rollout.excluded_segments)
        relay_log = rollout.rollout_dir / "trajectory" / "policy_relay.jsonl"
        relayed = [json.loads(line) for line in relay_log.read_text().splitlines()]
        assert relayed and all(c["stream"] for c in relayed if c["status"] == "ok")
    # Advantages: both passed, so a zero-variance group (GRPO gives 0.0).
    assert result.zero_variance and [r.advantage for r in result.rollouts] == [0.0, 0.0]
    # Neither the server's key nor a relay credential is in any rollout file.
    jobs = rl_root / "jobs"
    assert _files_containing(jobs, POLICY_KEY) == []
    assert _files_containing(jobs, "bfr-") == []


async def test_group_failures_are_masked_retried_and_reported(
    fake_policy, tasks, rl_root
):
    policy = _policy(fake_policy)
    config = _config(tasks, "hello-pass", rl_root)
    async with policy:
        group = bf.rollout_group(
            config,
            n=1,
            policy=policy,
            attempts=2,
            group_id="det-startup",
            startup_timeouts=StartupTimeouts(acp_handshake_sec=0.05),
        )
        result = await group.wait()
    [rollout] = result.rollouts
    assert rollout.outcome == "masked" and rollout.reward is None, rollout.to_dict()
    assert rollout.attribution_reason == "agent_setup"
    assert rollout.failure == "infrastructure"
    # The handshake timeout is a typed transport failure (acp/runtime.py
    # _wait_for_acp_handshake): category pipe_closed, with the phase in its
    # structured diagnosis.
    assert rollout.error_category == "pipe_closed"
    raw = json.loads((rollout.rollout_dir / "result.json").read_text())
    assert raw["transport_error_info"]["transport_diagnosis"] == (
        "acp_initialize_timeout"
    )
    assert rollout.startup["failed_phase"] == "acp_initialize"
    assert rollout.startup["timeouts"]["acp_handshake_sec"] == 0.05
    assert rollout.attempt == 2 and len(rollout.attempts) == 1
    assert rollout.attempts[0].attribution_reason == "agent_setup"
    assert result.dropped == "no_scored_rollouts"


async def test_a_root_agent_never_finds_the_oracle(fake_policy, tasks, rl_root):
    assert (tasks / "oracle-probe" / "solution" / "solve.sh").is_file()
    policy = _policy(fake_policy)
    config = _config(tasks, "oracle-probe", rl_root, sandbox_user=None)
    async with policy:
        result = await bf.rollout_group(
            config, n=1, policy=policy, group_id="det-root"
        ).wait()
    [rollout] = result.rollouts
    assert rollout.reward == 1.0, (
        "the probe wrote LEAK: /oracle or /solution was in the root agent's sandbox"
    )


async def test_cancel_mid_group_leaves_no_container(fake_policy, tasks, rl_root):
    policy = _policy(fake_policy)
    config = _config(tasks, "slow-pass", rl_root)
    token = leases.lease_token()
    async with policy:
        group = bf.rollout_group(config, n=2, policy=policy, group_id="det-cancel")
        await group.start()
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            live = [r for r in leases.live() if r["provider"] == "docker"]
            if len(live) == 2 and all(
                _containers(f"com.docker.compose.project={r['id']}") for r in live
            ):
                break
            await asyncio.sleep(2)
        else:
            pytest.fail("the group's sandboxes never came up")
        projects = [r["id"] for r in leases.live() if r["provider"] == "docker"]
        await asyncio.sleep(10)  # let the agents start their tool calls
        await group.cancel()
        result = await group.wait()
    assert result.cancelled == 2
    assert leases.live() == []
    for project in projects:
        assert _containers(f"com.docker.compose.project={project}") == [], project
    assert leases.lease_state(token) == "this"


CHILD = """
import asyncio, sys
import benchflow as bf

async def main():
    policy = bf.Policy("vllm/fake-policy", base_url=sys.argv[1] + "/v1", version=0)
    config = bf.RolloutConfig(task_path=sys.argv[2], agent="claude-agent-acp",
                              environment="docker", jobs_dir=sys.argv[3])
    async with policy:
        group = bf.rollout_group(config, n=2, policy=policy, group_id=sys.argv[4])
        async for rollout in group:
            print("finished", rollout.index, flush=True)

asyncio.run(main())
"""


def _start_child(
    fake_policy: str, tasks: Path, rl_root: Path, name: str
) -> tuple[subprocess.Popen, Path]:
    lease_dir = rl_root / f"leases-{name}"
    env = {**os.environ, leases.LEASE_DIR_ENV: str(lease_dir)}
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            CHILD,
            fake_policy,
            str(tasks / "slow-pass"),
            str(rl_root / "jobs"),
            name,
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=open(rl_root / f"{name}.stderr", "w"),  # noqa: SIM115
        cwd=REPO,
    )
    return child, lease_dir


def _wait_for_projects(lease_dir: Path, child: subprocess.Popen, n: int) -> list[str]:
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        assert child.poll() is None, "the group process ended early"
        projects: list[str] = []
        for path in lease_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            projects += [
                r["id"] for r in data.get("resources", []) if r["provider"] == "docker"
            ]
        if len(projects) == n and all(
            _containers(f"com.docker.compose.project={p}") for p in projects
        ):
            return projects
        time.sleep(2)
    pytest.fail("the group's sandboxes never came up")


def test_sigterm_mid_group_deletes_its_sandboxes(fake_policy, tasks, rl_root):
    child, lease_dir = _start_child(fake_policy, tasks, rl_root, "det-sigterm")
    try:
        projects = _wait_for_projects(lease_dir, child, 2)
        time.sleep(10)
        child.send_signal(signal.SIGTERM)
        assert child.wait(timeout=240) == -signal.SIGTERM
    finally:
        if child.poll() is None:
            child.kill()
    for project in projects:
        assert _containers(f"com.docker.compose.project={project}") == [], project
    assert list(lease_dir.glob("*.json")) == []


def test_sigkill_mid_group_is_reaped_by_the_next_run(
    fake_policy, tasks, rl_root, monkeypatch
):
    child, lease_dir = _start_child(fake_policy, tasks, rl_root, "det-sigkill")
    try:
        projects = _wait_for_projects(lease_dir, child, 2)
        time.sleep(10)
        child.kill()
        child.wait(timeout=60)
    finally:
        if child.poll() is None:
            child.kill()
    # Nothing ran teardown: the sandboxes are still there.
    assert all(_containers(f"com.docker.compose.project={p}") for p in projects)
    monkeypatch.setenv(leases.LEASE_DIR_ENV, str(lease_dir))
    report = leases.reap_dead_leases()
    assert sorted(report["deleted"]) == sorted(f"docker:{p}" for p in projects)
    for project in projects:
        assert _containers(f"com.docker.compose.project={project}") == [], project
    assert list(lease_dir.glob("*.json")) == []
