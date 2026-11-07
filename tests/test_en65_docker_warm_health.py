"""Fixes for warm identity (ccd6602a) and health timing (acaaefb9)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.evaluation import Evaluation, EvaluationConfig, _warm_image_key
from benchflow.rollout import Rollout
from benchflow.sandbox import _inert_bootstrap as bootstrap

IMMUTABLE = "example.invalid/task@sha256:" + "a" * 64


def task(tmp_path, image=IMMUTABLE):
    path = tmp_path / "task"
    path.mkdir()
    (path / "task.toml").write_text(
        'version = "1.0"\n[environment]\n'
        + (f'docker_image = "{image}"\n' if image else "")
    )
    (path / "instruction.md").write_text("Do nothing.\n")
    return path


def pool_for(path, monkeypatch, **kwargs):
    job = Evaluation(
        tasks_dir=path.parent,
        jobs_dir=path.parent / "jobs",
        config=EvaluationConfig(agent="nop", concurrency=1, warm_pool=1, **kwargs),
        preflight=False,
    )
    monkeypatch.setenv("BENCHFLOW_WARM_POOL", "on")
    pool = job._make_warm_pool([path, path], lambda: False)
    assert pool is not None
    monkeypatch.setattr(
        Rollout, "begin_prestart", lambda self: asyncio.create_task(asyncio.sleep(0))
    )
    monkeypatch.setattr(Rollout, "discard_prestart", AsyncMock())
    return pool


@pytest.mark.parametrize("image", ["example.invalid/task:latest", "", "sha256:short"])
async def test_unbound_image_cannot_be_prepared(tmp_path, monkeypatch, image):
    """Guards ccd6602a: tag ABA and mutable build bases must start in their slot."""
    path = task(tmp_path, image)
    pool = pool_for(path, monkeypatch)
    pool.ready()
    try:
        assert pool.made == 0
        assert pool.take(0) is None
        assert _warm_image_key(path, EvaluationConfig()) is None
    finally:
        await pool.drain()


@pytest.mark.parametrize("image", [IMMUTABLE, "sha256:" + "b" * 64])
async def test_immutable_image_reuses_prepared_rollout(tmp_path, monkeypatch, image):
    """Guards ccd6602a: immutable images retain the existing warm execution path."""
    path = task(tmp_path, image)
    original = (path / "task.toml").read_bytes()
    pool = pool_for(path, monkeypatch)
    pool.ready()
    try:
        assert pool.made == 1
        prepared = pool._entries[0].rollout
        assert pool.take(0) is prepared
        assert prepared._config.config_override is None
        assert pool.used == 1
        assert (path / "task.toml").read_bytes() == original
    finally:
        await pool.drain()


@pytest.mark.parametrize("kind", ["compose", "base_override"])
async def test_unbound_auxiliary_images_start_cold(tmp_path, monkeypatch, kind):
    """Guards ccd6602a: a fixed main ref does not bind Compose or build inputs."""
    path = task(tmp_path)
    kwargs = {}
    if kind == "compose":
        (path / "environment").mkdir()
        (path / "environment" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: mutable:latest\n"
        )
    else:
        kwargs["base_image_override"] = "mutable:latest"
    pool = pool_for(path, monkeypatch, **kwargs)
    pool.ready()
    try:
        assert pool.made == 0
        assert pool.take(0) is None
    finally:
        await pool.drain()


@pytest.mark.parametrize(
    ("health_extra", "expected_times"),
    [
        (
            {"StartPeriod": 2_000_000_000, "StartInterval": 500_000_000},
            [0, 0.5, 1, 1.5, 2, 10],
        ),
        ({"StartPeriod": 1_000_000_000}, [0, 5, 13]),
        ({"StartInterval": 500_000_000}, [0, 8]),
        ({}, [0, 8]),
    ],
)
async def test_startup_health_interval_and_retry_boundary(
    monkeypatch, health_extra, expected_times
):
    """Guards acaaefb9: startup cadence ends at StartPeriod; retries stay intact."""
    clock = SimpleNamespace(now=0.0)
    times = []

    async def sleep(delay):
        clock.now += delay

    async def wait_for(awaitable, timeout):
        return await awaitable

    async def probe(*args, **kwargs):
        times.append(clock.now)
        return SimpleNamespace(return_code=1)

    monkeypatch.setattr(
        bootstrap,
        "asyncio",
        SimpleNamespace(
            get_running_loop=lambda: SimpleNamespace(time=lambda: clock.now),
            sleep=sleep,
            wait_for=wait_for,
        ),
    )
    monkeypatch.setattr(
        bootstrap,
        "system_shell_route",
        AsyncMock(return_value=SimpleNamespace(kit=False)),
    )
    monkeypatch.setattr("benchflow.sandbox.docker.route_exec_argv", lambda *a, **k: [])
    env = SimpleNamespace(_docker_cli=probe)
    config = {
        "Healthcheck": {
            "Test": ["CMD", "/bin/false"],
            "Interval": 8_000_000_000,
            "Timeout": 1_000_000_000,
            "Retries": 2,
            **health_extra,
        }
    }
    with pytest.raises(RuntimeError, match="healthcheck failed"):
        await bootstrap.wait_healthy(env, "container", config)
    assert times == expected_times
