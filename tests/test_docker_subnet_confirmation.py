"""Guards U14-C1/C2/C3 against the admission regressions at e01b1e57."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import shutil
from types import SimpleNamespace

import pytest

from benchflow import batch, evaluation, runtime
from benchflow.models import RolloutResult
from benchflow.rollout import RolloutConfig
from benchflow.sandbox import _docker_subnets as subnets
from benchflow.sandbox._base import ExecResult
from tests._docker_daemon_fakes import server_id


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    # These resume tests stub trial execution and test only subnet selection.
    monkeypatch.setattr(evaluation, "_refuse_unnamed_task_env", lambda *a, **k: None)
    monkeypatch.setattr(subnets, "_read_daemon_id", server_id)
    subnets.reset_allocators()
    monkeypatch.setenv("BENCHFLOW_REMOTE_DOCKER_HOST", "ssh://dummy@remote.invalid")
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    monkeypatch.delenv("U14_NET", raising=False)
    monkeypatch.delenv("FALLBACK", raising=False)
    yield
    subnets.reset_allocators()


def tasks(tmp_path, subnet="172.20.0.0/16"):
    paths = [tmp_path / "plain", tmp_path / "pinned"]
    for path in paths:
        (path / "environment").mkdir(parents=True)
        (path / "instruction.md").write_text("Dummy network fixture.")
        (path / "task.toml").write_text("[sandbox]\nallow_internet = true\n")
    (paths[1] / "environment" / "docker-compose.yaml").write_text(
        "services:\n  main:\n    image: fixture\n    networks: [fixed]\n"
        "networks:\n  fixed:\n    ipam:\n      config:\n"
        f'        - subnet: "{subnet}"\n'
    )
    return paths


@pytest.mark.parametrize(
    "expression,source,expected",
    [
        ("172.24.0.0/16", "literal", "172.24.0.0/16"),
        ("${U14_NET:-172.20.0.0/16}", "default", "172.20.0.0/16"),
        ("${U14_NET:-${FALLBACK:-172.20.0.0/16}}", "default", "172.20.0.0/16"),
        ("$U14_NET", "shell", "172.21.0.0/16"),
        ("${U14_NET}", "dotenv", "172.22.0.0/16"),
        ("${U14_NET}", "task", "172.23.0.0/16"),
    ],
)
@pytest.mark.skipif(shutil.which("docker") is None, reason="Compose CLI required")
def test_interpolated_pins_use_startup_precedence(
    tmp_path, monkeypatch, expression, source, expected
):
    """Guards U14-C1/e01b1e57: reserve the CIDR Compose actually creates."""
    _, pinned = tasks(tmp_path, expression)
    env = pinned / "environment"
    if source in ("shell", "task"):
        monkeypatch.setenv("U14_NET", "172.21.0.0/16")
        monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_ENV", "U14_NET")
    if source in ("shell", "dotenv", "task"):
        (env / ".env").write_text("U14_NET=172.22.0.0/16\n")
    if source == "task":
        (pinned / "task.toml").write_text(
            '[sandbox]\nallow_internet = true\n[sandbox.env]\nU14_NET = "172.23.0.0/16"\n'
        )
    got = subnets.pin_task_subnets([env / "docker-compose.yaml"])
    assert got == [ipaddress.IPv4Network(expected)]


@pytest.mark.parametrize("mode", ["as_completed", "arun_batch", "run_batch"])
@pytest.mark.parametrize("first_backend", ["docker", "remote-docker"])
async def test_mixed_batch_reserves_only_its_daemon(
    tmp_path, monkeypatch, mode, first_backend
):
    """Guards U14-C2/e01b1e57: a remote pin cannot exhaust local space, or vice versa."""
    paths = tasks(tmp_path)
    other = "remote-docker" if first_backend == "docker" else "docker"
    configs = [
        RolloutConfig.from_legacy(task_path=p, agent="nop", environment=b)
        for p, b in zip(paths, [first_backend, other], strict=True)
    ]
    monkeypatch.setenv(subnets.ENV_VAR, "172.20.0.0/16")
    monkeypatch.setattr(batch, "_check_configs", lambda items: None)
    monkeypatch.setattr(runtime, "check_host", lambda items: None)
    picked = []

    async def docker(args):
        data = {"name": "fixture", "pools": []} if args[0] == "info" else []
        return ExecResult(stdout=json.dumps(data), stderr="", return_code=0)

    async def trial(config):
        if config.task_path == paths[0]:
            env = dict(os.environ)
            if config.environment == "remote-docker":
                env["DOCKER_HOST"] = env["BENCHFLOW_REMOTE_DOCKER_HOST"]
                env.pop("DOCKER_CONTEXT", None)
            allocator = await subnets.allocator_for(subnets.daemon_key(env), docker)
            assert allocator is not None
            try:
                picked.append(allocator.allocate([subnets.NetworkNeed("default")]))
            except subnets.SubnetRangeExhausted as exc:
                picked.append(exc)
        return RolloutResult(
            task_name=config.task_path.name, agent="nop", rewards={"reward": 0}
        )

    monkeypatch.setattr(batch, "_run_one", trial)
    if mode == "as_completed":
        results = [r async for r in batch.as_completed(configs, concurrency=1)]
    elif mode == "arun_batch":
        results = await batch.arun_batch(configs, concurrency=1)
    else:
        results = batch.run_batch(configs, concurrency=1)
    assert len(results) == 2
    assert len(picked) == 1
    assert isinstance(picked[0], dict), str(picked[0])


def test_resume_does_not_reserve_completed_task(tmp_path, monkeypatch):
    """Guards U14-C3/e01b1e57: removed networks of completed tasks consume no space."""
    plain, pinned = tasks(tmp_path)
    job = evaluation.Evaluation(
        tasks_dir=tmp_path,
        jobs_dir=tmp_path / "jobs",
        job_name="fixture",
        config=evaluation.EvaluationConfig(environment="docker"),
    )
    from benchflow.review import automatic, resume

    async def nothing(*args, **kwargs):
        return None

    class Stop(Exception):
        pass

    async def first_trial(remaining):
        assert remaining == [plain]
        allocator = subnets.SubnetAllocator([ipaddress.IPv4Network("172.20.0.0/16")])
        allocator.allocate([subnets.NetworkNeed("default")])
        raise Stop

    monkeypatch.setattr(job, "_get_task_dirs", lambda: [plain, pinned])
    monkeypatch.setattr(job, "_get_completed_tasks", lambda: {pinned.name: {}})
    monkeypatch.setattr(job, "_sweep_docker", nothing)
    monkeypatch.setattr(job, "_write_evaluation_record", lambda: None)
    monkeypatch.setattr(job, "_run_parallel_independent", first_trial)
    monkeypatch.setattr(evaluation, "_check_resume_mismatch", lambda *a, **k: None)
    monkeypatch.setattr(resume, "resume_pending_reviews", nothing)
    monkeypatch.setattr(automatic, "prepare_review", lambda *a: None)
    with pytest.raises(Stop):
        asyncio.run(job._run_unlocked())


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
async def test_shard_resume_reserves_only_unfinished_tasks(
    tmp_path, monkeypatch, backend
):
    """Guards U14-C3/e01b1e57 at the persisted worker-job resume boundary."""
    from benchflow import eval_sharding
    from benchflow.review import automatic

    plain, pinned = tasks(tmp_path / "tasks")
    jobs = tmp_path / "jobs"
    result_dir = jobs / "worker-shards" / "shard-000" / "jobs" / "fixture" / "pinned__1"
    result_dir.mkdir(parents=True)
    (result_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": pinned.name,
                "rollout_name": result_dir.name,
                "rewards": {"reward": 1},
                "agent": "nop",
            }
        )
    )
    monkeypatch.setattr(automatic, "prepare_review", lambda *a: None)
    payloads = []

    async def worker(payload_path, log_path, **kwargs):
        payload = json.loads(payload_path.read_text())
        payloads.append(payload)
        assert payload["docker_pinned_subnets"] == []
        allocator = subnets.SubnetAllocator(
            [ipaddress.IPv4Network("172.20.0.0/16")],
            daemon=subnets.backend_daemon_key(backend),
        )
        allocator.allocate([subnets.NetworkNeed("default")])
        return {"total": 1, "passed": 1, "failed": 0, "errored": 0}

    monkeypatch.setattr(eval_sharding, "_run_worker_with_retries", worker)
    await eval_sharding.run_sharded_evaluation(
        tasks_dir=plain.parent,
        jobs_dir=jobs,
        config=evaluation.EvaluationConfig(
            agent="nop", environment=backend, concurrency=2
        ),
        worker_concurrency=1,
        worker_retries=0,
        worker_start_stagger_sec=0,
    )
    assert len(payloads) == 2


def test_mixed_backends_still_share_pins_on_the_same_endpoint(tmp_path, monkeypatch):
    """Guards U14-C2/e01b1e57: backend names alone must not partition a daemon."""
    monkeypatch.setenv("DOCKER_HOST", "ssh://dummy@remote.invalid")
    _, pinned = tasks(tmp_path)
    subnets.pin_task_subnets(
        [pinned / "environment" / "docker-compose.yaml"], backend="remote-docker"
    )
    allocator = subnets.SubnetAllocator([ipaddress.IPv4Network("172.20.0.0/16")])
    with pytest.raises(subnets.SubnetRangeExhausted):
        allocator.allocate([subnets.NetworkNeed("default")])


async def test_worker_refuses_reservations_for_a_changed_daemon(tmp_path, monkeypatch):
    """Guards U14-C2/e01b1e57 across serialized parent/worker identities."""
    from benchflow import eval_worker

    payload = tmp_path / "payload.json"
    payload.write_text(
        json.dumps(
            {
                "config": {"environment": "docker"},
                "tasks_dir": str(tmp_path / "tasks"),
                "jobs_dir": str(tmp_path / "jobs"),
                "result_path": str(tmp_path / "result.json"),
                "docker_pinned_subnets": ["172.20.0.0/16"],
                "docker_subnet_daemon": "dummy-other-daemon",
            }
        )
    )

    class NoTrial:
        def __init__(self, **kwargs):
            pass

        async def run(self):
            return SimpleNamespace(
                job_name="fixture",
                total=1,
                passed=1,
                failed=0,
                errored=0,
                verifier_errored=0,
                elapsed_sec=0,
                score=1,
                score_excl_errors=1,
            )

    monkeypatch.setattr(eval_worker, "Evaluation", NoTrial)
    with pytest.raises(ValueError, match="daemon changed"):
        await eval_worker.run_worker(payload)


@pytest.mark.skipif(shutil.which("docker") is None, reason="Compose CLI required")
def test_unresolved_dynamic_pin_stops_admission(tmp_path):
    """Guards U14-C1/e01b1e57: missing required variables cannot silently lose pins."""
    _, pinned = tasks(tmp_path, "${U14_NET:?dummy-required}")
    with pytest.raises(ValueError, match=r"Cannot resolve.*network configuration"):
        subnets.pin_task_subnets([pinned / "environment" / "docker-compose.yaml"])


def manifest_tasks(tmp_path, monkeypatch):
    from benchflow.environment.manifest import EnvironmentManifest, ForwardEnv

    paths = tasks(tmp_path, "${U14_NET}")
    monkeypatch.setenv("U14_NET", "172.20.0.0/16")
    (paths[1] / "task.toml").write_text(
        '[sandbox]\nallow_internet = true\n[sandbox.env]\nU14_NET = "172.23.0.0/16"\n'
    )
    manifest = EnvironmentManifest(
        name="dummy", base_image="fixture", forward_env=ForwardEnv(keys=["U14_NET"])
    )
    return paths, manifest


@pytest.mark.skipif(shutil.which("docker") is None, reason="Compose CLI required")
@pytest.mark.parametrize("mode", ["as_completed", "arun_batch", "run_batch"])
async def test_batch_manifest_overlay_wins_over_task_env(tmp_path, monkeypatch, mode):
    """Guards U14-C1/cc428dad: admission needs the final per-config manifest overlay."""
    paths, manifest = manifest_tasks(tmp_path, monkeypatch)
    configs = [
        RolloutConfig.from_legacy(
            task_path=p,
            agent="nop",
            environment="docker",
            environment_manifest=manifest if p == paths[1] else None,
        )
        for p in paths
    ]
    monkeypatch.setattr(batch, "_check_configs", lambda items: None)
    monkeypatch.setattr(runtime, "check_host", lambda items: None)
    picked = []
    allocator = subnets.SubnetAllocator(
        [ipaddress.IPv4Network("172.20.0.0/15")], start=48
    )

    async def trial(config):
        picked.append(allocator.allocate([subnets.NetworkNeed("default")])["default"])
        return RolloutResult(
            task_name=config.task_path.name, agent="nop", rewards={"reward": 0}
        )

    monkeypatch.setattr(batch, "_run_one", trial)
    if mode == "as_completed":
        results = [r async for r in batch.as_completed(configs, concurrency=1)]
    elif mode == "arun_batch":
        results = await batch.arun_batch(configs, concurrency=1)
    else:
        results = batch.run_batch(configs, concurrency=1)
    assert len(results) == 2
    assert all(
        not net.overlaps(ipaddress.IPv4Network("172.20.0.0/16")) for net in picked
    )


@pytest.mark.skipif(shutil.which("docker") is None, reason="Compose CLI required")
@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
async def test_shard_manifest_overlay_reaches_all_payloads(
    tmp_path, monkeypatch, backend
):
    """Guards U14-C1/cc428dad at the manifest-to-parent-to-worker boundary."""
    from benchflow import eval_sharding
    from benchflow.review import automatic

    paths, manifest = manifest_tasks(tmp_path / "tasks", monkeypatch)
    monkeypatch.setattr(automatic, "prepare_review", lambda *a: None)
    seen = []

    async def worker(payload_path, log_path, **kwargs):
        payload = json.loads(payload_path.read_text())
        seen.append(payload["docker_pinned_subnets"])
        assert payload["docker_pinned_subnets"] == ["172.20.0.0/16"]
        return {"total": 1, "passed": 1, "failed": 0, "errored": 0}

    monkeypatch.setattr(eval_sharding, "_run_worker_with_retries", worker)
    await eval_sharding.run_sharded_evaluation(
        tasks_dir=paths[0].parent,
        jobs_dir=tmp_path / "jobs",
        config=evaluation.EvaluationConfig(
            agent="nop",
            environment=backend,
            concurrency=2,
            environment_manifest=manifest,
        ),
        worker_concurrency=1,
        worker_retries=0,
        worker_start_stagger_sec=0,
    )
    assert len(seen) == 2
