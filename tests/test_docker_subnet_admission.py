"""Guards U14-R3-A-1, the incomplete run-wide pinning in 37cb4f7e."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchflow import batch, eval_sharding, evaluation, runtime
from benchflow.models import RolloutResult
from benchflow.rollout import RolloutConfig
from benchflow.sandbox import _docker_subnets as subnets
from tests._docker_daemon_fakes import server_id


@pytest.fixture(autouse=True)
def isolated_pins(monkeypatch):
    subnets.reset_allocators()
    monkeypatch.setattr(subnets, "_read_daemon_id", server_id)
    monkeypatch.setattr(subnets, "_PINNED", {})
    monkeypatch.setenv("BENCHFLOW_REMOTE_DOCKER_HOST", "ssh://dummy@remote.invalid")


def _tasks(tmp_path, subnet="172.20.0.0/16"):
    paths = [tmp_path / "plain", tmp_path / "pinned"]
    for path in paths:
        (path / "environment").mkdir(parents=True)
    (paths[1] / "environment" / "docker-compose.yaml").write_text(
        "services:\n  main:\n    image: fixture\n    networks: [fixed]\n"
        "networks:\n  fixed:\n    ipam:\n      config:\n"
        f'        - subnet: "{subnet}"\n'
    )
    return paths


def _allocator(backend="docker"):
    return subnets.SubnetAllocator(
        [ipaddress.IPv4Network("172.20.0.0/15")],
        start=48,
        daemon=subnets.backend_daemon_key(backend),
    )


@pytest.mark.parametrize("mode", ["as_completed", "arun_batch", "run_batch"])
@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
@pytest.mark.parametrize(
    "subnet",
    [
        "172.20.0.0/16",
        pytest.param(
            "${U14_NET:-172.20.0.0/16}",
            marks=pytest.mark.skipif(
                shutil.which("docker") is None,
                reason="Compose CLI required for interpolation",
            ),
        ),
    ],
)
async def test_batch_pins_later_tasks_before_first_trial(
    tmp_path, monkeypatch, mode, backend, subnet
):
    """Guards U14-R3-A-1: the first config cannot borrow a later config's /16."""
    paths = _tasks(tmp_path, subnet)
    configs = [
        RolloutConfig.from_legacy(task_path=p, agent="nop", environment=backend)
        for p in paths
    ]
    monkeypatch.setattr(batch, "_check_configs", lambda items: None)
    monkeypatch.setattr(runtime, "check_host", lambda items: None)
    picked = []
    allocator = _allocator(backend)

    async def trial(config):
        picked.append(allocator.allocate([subnets.NetworkNeed("default")])["default"])
        return RolloutResult(
            task_name=config.task_path.name, agent="nop", rewards={"reward": 0}
        )

    monkeypatch.setattr(batch, "_run_one", trial)
    if mode == "as_completed":
        results = [r async for r in batch.as_completed(iter(configs), concurrency=1)]
    elif mode == "arun_batch":
        results = await batch.arun_batch(iter(configs), concurrency=1)
    else:
        results = batch.run_batch(iter(configs), concurrency=1)
    assert len(results) == 2
    assert all(
        not net.overlaps(ipaddress.IPv4Network("172.20.0.0/16")) for net in picked
    ), picked


_WORKER = """
import asyncio, ipaddress, json, sys
from types import SimpleNamespace
from benchflow import eval_worker
from benchflow.sandbox import _docker_subnets as subnets
from tests._docker_daemon_fakes import server_id
subnets._read_daemon_id = server_id
assert subnets.pinned_spans() == []
class FakeEvaluation:
    def __init__(self, **kwargs):
        pass
    async def run(self):
        allocator = subnets.SubnetAllocator([ipaddress.IPv4Network("172.20.0.0/15")], start=48)
        net = allocator.allocate([subnets.NetworkNeed("default")])["default"]
        assert not net.overlaps(ipaddress.IPv4Network("172.20.0.0/16")), str(net)
        return SimpleNamespace(job_name="worker", total=1, passed=0, failed=1, errored=0, verifier_errored=0, elapsed_sec=0, score=0, score_excl_errors=0)
eval_worker.Evaluation = FakeEvaluation
asyncio.run(eval_worker.run_worker(__import__("pathlib").Path(sys.argv[1])))
"""


@pytest.mark.parametrize(
    "subnet",
    [
        "172.20.0.0/16",
        pytest.param(
            "${U14_NET:-172.20.0.0/16}",
            marks=pytest.mark.skipif(
                shutil.which("docker") is None,
                reason="Compose CLI required for interpolation",
            ),
        ),
    ],
)
async def test_every_shard_worker_gets_all_selected_task_pins(
    tmp_path, monkeypatch, subnet
):
    """Guards U14-R3-A-1 across real subprocess globals and JSON payloads."""
    paths = _tasks(tmp_path, subnet)
    monkeypatch.setattr(evaluation.Evaluation, "_get_task_dirs", lambda self: paths)
    from benchflow.review import automatic

    monkeypatch.setattr(automatic, "prepare_review", lambda *a: None)
    seen = []

    async def worker(payload_path, log_path, **kwargs):
        payload = json.loads(payload_path.read_text())
        seen.append(payload["config"]["include_tasks"])
        # Each fresh interpreter has its own empty _PINNED; parent state cannot
        # make this pass accidentally. No daemon or sandbox is used here.
        process = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-c", _WORKER, str(payload_path)],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0, process.stderr
        return json.loads(Path(payload["result_path"]).read_text())

    monkeypatch.setattr(eval_sharding, "_run_worker_with_retries", worker)
    await eval_sharding.run_sharded_evaluation(
        tasks_dir=tmp_path,
        jobs_dir=tmp_path / "jobs",
        config=evaluation.EvaluationConfig(
            agent="nop", environment="docker", concurrency=2
        ),
        worker_concurrency=1,
        worker_retries=0,
        worker_start_stagger_sec=0,
    )
    assert sorted(seen) == [["pinned"], ["plain"]]


def test_pinned_overlap_advice_covers_other_tasks_and_owned_networks():
    """Guards U14-R3-A-1: one-at-a-time is not a universal cure."""
    message = subnets.pinned_overlap_message(
        "fixture_fixed", [ipaddress.IPv4Network("172.20.0.0/16")]
    )
    assert "non-overlapping" in message
    assert "you own" in message
    assert "another task" in message
    assert "so retrying will not help" not in message
