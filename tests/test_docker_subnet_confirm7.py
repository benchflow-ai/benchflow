"""Guards U14-C4/C1/C5 against the regressions at a479077c."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from benchflow import batch, eval_sharding, evaluation, runtime
from benchflow.models import RolloutResult
from benchflow.rollout import RolloutConfig
from benchflow.sandbox import _docker_subnets as subnets
from benchflow.sandbox._base import ExecResult


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    subnets.reset_allocators()
    for name in ("DOCKER_HOST", "DOCKER_CONTEXT", "U14_NET", "REPLICAS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BENCHFLOW_REMOTE_DOCKER_HOST", "ssh://dummy@remote.invalid")
    # Only daemon transport is faked; Compose still resolves the actual files.
    run = subprocess.run

    def docker(args, **kwargs):
        if args[:2] == ["docker", "info"]:
            return SimpleNamespace(stdout="dummy-daemon\n", returncode=0)
        return run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", docker)
    yield
    subnets.reset_allocators()


@pytest.mark.skipif(shutil.which("docker") is None, reason="Compose CLI required")
@pytest.mark.parametrize("source", ["scale", "deploy", "include", "extends"])
@pytest.mark.parametrize("network", ["implicit", "declared", "bridge", "named"])
def test_resolution_preserves_automatic_networks(tmp_path, source, network):
    """Guards U14-C4/a479077c: generated names cannot suppress allocation."""
    path = tmp_path / "environment" / "docker-compose.yaml"
    path.parent.mkdir()
    service = "    image: fixture\n"
    service += (
        '    deploy:\n      replicas: "${REPLICAS:-20}"\n'
        if source == "deploy"
        else '    scale: "${REPLICAS:-20}"\n'
    )
    declarations = {
        "implicit": "",
        "declared": "networks:\n  default: {}\n",
        "bridge": "networks:\n  default:\n    driver: bridge\n",
        # Even this exact name is explicit, and must stay task-controlled.
        "named": "networks:\n  default:\n    name: benchflow-subnet-admission_default\n",
    }[network]
    if source == "include":
        (path.parent / "part.yaml").write_text(
            "services:\n  worker:\n" + service + declarations
        )
        path.write_text(
            "include: [part.yaml]\nservices:\n  main:\n    image: fixture\n"
        )
    elif source == "extends":
        (path.parent / "part.yaml").write_text("services:\n  template:\n" + service)
        path.write_text(
            "services:\n  main:\n    extends:\n      file: part.yaml\n      service: template\n"
            + declarations
        )
    else:
        path.write_text("services:\n  main:\n" + service + declarations)
    plan = subnets.plan_networks(path)
    assert plan is not None
    assert plan.needs == (
        [] if network == "named" else [subnets.NetworkNeed("default", 27)]
    )


def tasks(tmp_path, source, monkeypatch):
    paths = [tmp_path / "plain", tmp_path / "pinned"]
    for path in paths:
        (path / "environment").mkdir(parents=True)
        (path / "instruction.md").write_text("Dummy network fixture.")
        (path / "task.toml").write_text(
            '[sandbox]\nallow_internet = true\n[sandbox.env]\nU14_NET = "172.23.0.0/16"\n'
        )
    (paths[1] / "environment" / "docker-compose.yaml").write_text(
        "services:\n  main:\n    image: fixture\n    networks: [fixed]\n"
        'networks:\n  fixed:\n    ipam:\n      config:\n        - subnet: "${U14_NET:-172.24.0.0/16}"\n'
    )
    if source == "task_manifest":
        monkeypatch.setenv("U14_NET", "172.20.0.0/16")
        monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_ENV", "U14_NET")
        (paths[1] / "task.md").write_text(
            "---\nname: pinned\nbenchflow:\n  environment:\n    manifest: manifest.yaml\n"
            "sandbox:\n  allow_internet: true\n  env:\n    U14_NET: 172.23.0.0/16\n---\nDummy fixture.\n"
        )
        (paths[1] / "manifest.yaml").write_text(
            "environment:\n  name: dummy\n  base_image: fixture\n  forward_env:\n    keys: [U14_NET]\n"
        )
    return paths


@pytest.mark.skipif(shutil.which("docker") is None, reason="Compose CLI required")
@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
@pytest.mark.parametrize(
    "entry", ["as_completed", "arun_batch", "run_batch", "evaluation", "shards"]
)
@pytest.mark.parametrize("source", ["override", "task_manifest"])
async def test_admission_matches_runtime_environment(
    tmp_path, monkeypatch, backend, entry, source
):
    """Guards U14-C1/a479077c across batch, evaluation and shard admission."""
    if source == "task_manifest" and entry not in ("evaluation", "shards"):
        # Batch receives explicit RolloutConfigs; it does not infer manifests.
        source = "explicit_manifest"
    paths = tasks(tmp_path / "tasks", source, monkeypatch)
    from benchflow.environment.manifest import EnvironmentManifest, ForwardEnv
    from benchflow.review import automatic, resume

    kwargs = {}
    if source == "override":
        kwargs["config_override"] = {"sandbox": {"env": {"U14_NET": "172.20.0.0/16"}}}
    elif source == "explicit_manifest":
        monkeypatch.setenv("U14_NET", "172.20.0.0/16")
        kwargs["environment_manifest"] = EnvironmentManifest(
            name="dummy", base_image="fixture", forward_env=ForwardEnv(keys=["U14_NET"])
        )
    cfg = evaluation.EvaluationConfig(
        agent="nop", environment=backend, concurrency=2, **kwargs
    )
    monkeypatch.setattr(automatic, "prepare_review", lambda *a: None)

    def assert_reserved():
        allocator = subnets.SubnetAllocator(
            [ipaddress.IPv4Network("172.20.0.0/15")],
            start=48,
            daemon=subnets.backend_daemon_key(backend),
        )
        chosen = allocator.allocate([subnets.NetworkNeed("default")])["default"]
        assert not chosen.overlaps(ipaddress.IPv4Network("172.20.0.0/16"))

    async def nothing(*args, **kwargs):
        pass

    if entry == "shards":
        monkeypatch.setattr(evaluation.Evaluation, "_get_task_dirs", lambda self: paths)
        seen = []

        async def worker(payload_path, log_path, **kwargs):
            payload = json.loads(payload_path.read_text())
            seen.append(payload)
            assert payload["docker_pinned_subnets"] == ["172.20.0.0/16"]
            assert_reserved()
            return {"total": 1, "passed": 1, "failed": 0, "errored": 0}

        monkeypatch.setattr(eval_sharding, "_run_worker_with_retries", worker)
        await eval_sharding.run_sharded_evaluation(
            tasks_dir=paths[0].parent,
            jobs_dir=tmp_path / "jobs",
            config=cfg,
            worker_concurrency=1,
            worker_retries=0,
            worker_start_stagger_sec=0,
        )
        assert len(seen) == 2
    elif entry == "evaluation":
        job = evaluation.Evaluation(
            tasks_dir=paths[0].parent, jobs_dir=tmp_path / "jobs", config=cfg
        )

        class Stop(Exception):
            pass

        async def first_trial(remaining):
            assert_reserved()
            raise Stop

        monkeypatch.setattr(job, "_get_task_dirs", lambda: paths)
        monkeypatch.setattr(job, "_get_completed_tasks", lambda: {})
        monkeypatch.setattr(job, "_sweep_docker", nothing)
        monkeypatch.setattr(job, "_write_evaluation_record", lambda: None)
        monkeypatch.setattr(job, "_run_parallel_independent", first_trial)
        monkeypatch.setattr(resume, "resume_pending_reviews", nothing)
        with pytest.raises(Stop):
            await job._run_unlocked()
    else:
        configs = [
            RolloutConfig.from_legacy(
                task_path=p, agent="nop", environment=backend, **kwargs
            )
            for p in paths
        ]
        monkeypatch.setattr(batch, "_check_configs", lambda items: None)
        monkeypatch.setattr(runtime, "check_host", lambda items: None)
        checks = []

        async def trial(config):
            try:
                assert_reserved()
                checks.append(True)
            except AssertionError:
                checks.append(False)
            return RolloutResult(
                task_name=config.task_path.name, agent="nop", rewards={"reward": 0}
            )

        monkeypatch.setattr(batch, "_run_one", trial)
        if entry == "as_completed":
            results = [r async for r in batch.as_completed(configs, concurrency=1)]
        elif entry == "arun_batch":
            results = await batch.arun_batch(configs, concurrency=1)
        else:
            results = await asyncio.to_thread(batch.run_batch, configs, concurrency=1)
        assert len(results) == 2
        assert checks == [True, True]


@pytest.mark.parametrize(
    "alias", ["context", "current_context", "host_alias", "ignored_host", "tls"]
)
def test_same_daemon_shares_reservations_across_client_aliases(monkeypatch, alias):
    """Guards U14-C5/a479077c: daemon identity, not its client spelling, owns pins."""
    first = dict(os.environ)
    second = dict(os.environ, DOCKER_HOST="ssh://dummy@remote.invalid")
    if alias in ("context", "ignored_host"):
        first["DOCKER_CONTEXT"] = "dummy-prod"
    if alias == "ignored_host":
        first["DOCKER_HOST"] = "ssh://dummy@ignored.invalid"
    if alias == "host_alias":
        first["DOCKER_HOST"] = "ssh://dummy@alias.invalid"
    if alias == "tls":
        first.update(
            DOCKER_HOST="tcp://dummy.invalid:2376",
            DOCKER_TLS_VERIFY="1",
            DOCKER_CERT_PATH="/tmp/dummy-certs",
        )
    subnets.pin_subnets(
        [ipaddress.IPv4Network("172.20.0.0/16")], key=subnets.daemon_key(first)
    )
    allocator = subnets.SubnetAllocator(
        [ipaddress.IPv4Network("172.20.0.0/16")], daemon=subnets.daemon_key(second)
    )
    with pytest.raises(subnets.SubnetRangeExhausted):
        allocator.allocate([subnets.NetworkNeed("default")])


async def test_aliases_share_allocator_and_worker_identity(monkeypatch):
    """Guards U14-C5/a479077c: aliases share live leases and payload identities."""
    monkeypatch.setenv(subnets.ENV_VAR, "172.20.0.0/16")
    first = subnets.daemon_key(dict(os.environ, DOCKER_CONTEXT="dummy-prod"))
    second = subnets.daemon_key(
        dict(os.environ, DOCKER_HOST="ssh://dummy@remote.invalid")
    )

    async def docker(args):
        return ExecResult(
            stdout=json.dumps({"name": "fixture", "pools": []})
            if args[0] == "info"
            else "",
            stderr="",
            return_code=0,
        )

    assert subnets.daemon_identity(first) == subnets.daemon_identity(second)
    assert await subnets.allocator_for(first, docker) is await subnets.allocator_for(
        second, docker
    )


def test_distinct_daemons_do_not_share_even_with_identical_client_host(monkeypatch):
    """Guards U14-C5/a479077c: TLS/config selection cannot alias distinct servers."""

    def docker(args, **kwargs):
        return SimpleNamespace(stdout=kwargs["env"]["DOCKER_CONFIG"], returncode=0)

    monkeypatch.setattr(subprocess, "run", docker)
    first = subnets.daemon_key(dict(os.environ, DOCKER_CONFIG="/tmp/dummy-a"))
    second = subnets.daemon_key(dict(os.environ, DOCKER_CONFIG="/tmp/dummy-b"))
    subnets.pin_subnets([ipaddress.IPv4Network("172.20.0.0/16")], key=first)
    allocator = subnets.SubnetAllocator(
        [ipaddress.IPv4Network("172.20.0.0/16")], daemon=second
    )
    assert allocator.allocate([subnets.NetworkNeed("default")])


@pytest.mark.parametrize("failure", ["missing", "timeout", "refused", "empty"])
def test_unresolved_daemon_never_gets_a_guessed_identity(monkeypatch, failure):
    """Guards U14-C5/a479077c: unavailable identity must stop admission."""

    def docker(args, **kwargs):
        if failure == "missing":
            raise FileNotFoundError("dummy")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args, 30)
        if failure == "refused":
            raise subprocess.CalledProcessError(1, args, stderr="dummy-private-value")
        return SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr(subprocess, "run", docker)
    with pytest.raises(
        subnets.SubnetConfigError, match="Cannot identify the Docker daemon"
    ) as error:
        subnets.pin_subnets([ipaddress.IPv4Network("172.20.0.0/16")])
    assert "dummy-private-value" not in str(error.value)
    assert subnets._PINNED == {}


async def test_unreadable_daemon_can_only_fall_back_without_reservations(monkeypatch):
    """Guards U14-C5/a479077c without breaking unavailable-daemon startup."""
    first = subnets.daemon_key(dict(os.environ, DOCKER_CONTEXT="dummy-other"))
    subnets.pin_subnets([ipaddress.IPv4Network("172.20.0.0/16")], key=first)

    def unavailable(*args, **kwargs):
        raise FileNotFoundError("dummy")

    async def docker(args):
        pytest.fail("must not allocate without identifying the daemon")

    monkeypatch.setattr(subprocess, "run", unavailable)
    with pytest.raises(subnets.SubnetConfigError, match="Cannot identify"):
        await subnets.allocator_for(subnets.daemon_key(None), docker)
    subnets.reset_allocators()
    assert await subnets.allocator_for(subnets.daemon_key(None), docker) is None
    assert not subnets._ALLOCATORS
