"""Guard K1-R15-2 B1 at 6bf2151da: factory results need admission before start."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from benchflow.rollout import Rollout, RolloutConfig
from benchflow.task import identity_switch
from benchflow.task.runtime_capabilities import UnsupportedTaskFeatureError
from tests.test_identity_switch_backends import _task


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
@pytest.mark.parametrize(
    "entry", ["lifecycle", "branch_child", "run_config", "run_batch"]
)
@pytest.mark.parametrize("capability", ["missing", False, "yes"])
def test_factory_barrier_refusal_precedes_start(
    tmp_path, monkeypatch, backend, entry, capability
):
    """Guards K1-R15-2 B1 at 6bf2151da through all four factory entry paths."""
    import benchflow as bf

    task = _task(tmp_path)
    (task / "solution").mkdir()
    (task / "solution/solve.sh").write_text("#!/bin/sh\nexit 0\n")
    (task / "tests/test.sh").write_text("#!/bin/sh\nexit 0\n")
    inner = SimpleNamespace(
        start=AsyncMock(side_effect=AssertionError("sandbox start reached")),
        stop=AsyncMock(),
    )
    if capability != "missing":
        inner.agent_container_no_new_privileges = capability
    planes = MagicMock()
    planes.resolve_locked_paths.return_value = []
    planes.resolve_agent_env.return_value = {}
    planes.create_environment.return_value = inner
    cfg = RolloutConfig.from_legacy(
        task_path=task,
        agent="oracle",
        environment=backend,
        jobs_dir=tmp_path / "jobs",
        planes=planes,
        usage_tracking="off",
    )
    monkeypatch.setenv("BENCHFLOW_SKIP_PREFLIGHT", "1")
    expected = identity_switch.refusal(backend)
    if entry in {"lifecycle", "branch_child"}:
        rollout = Rollout(cfg)
        if entry == "branch_child":
            from benchflow.rollout_branch import _make_sub_rollout

            node = rollout._tree.attach(rollout._tree.root)
            rollout = _make_sub_rollout(rollout, tmp_path / "fork", node)

        async def run():
            try:
                await rollout.setup()
                await rollout.start()
            finally:
                await rollout.cleanup()

        with pytest.raises(UnsupportedTaskFeatureError) as exc:
            asyncio.run(run())
        assert expected in str(exc.value)
    else:
        if entry == "run_config":
            result = asyncio.run(bf.run(cfg))
        else:
            result = bf.run_batch([cfg], concurrency=1)[0]
        assert expected in result.error
    planes.create_environment.assert_called_once()
    inner.start.assert_not_called()
    print(
        f"FACTORY REFUSAL backend={backend} entry={entry} "
        f"capability={capability!r} start_calls={inner.start.await_count}"
    )


@pytest.mark.asyncio
@pytest.mark.deterministic
async def test_capable_custom_factory_starts_once_and_runs(tmp_path, monkeypatch):
    """Guards K1-R15-2 B1 at 6bf2151da without rejecting a capable factory."""
    import benchflow as bf
    from benchflow.contracts import default_rollout_planes
    from tests.exec_diet_fakes import docker_available

    if not docker_available():
        pytest.skip("needs a Docker daemon")
    task = _task(tmp_path)
    config = task / "task.toml"
    config.write_text(config.read_text().replace("timeout_sec = 1", "timeout_sec = 60"))
    with config.open("a") as file:
        file.write(
            '[environment]\ndocker_image = "python:3.12-slim"\n'
            'workdir = "/app"\n'
            '[[environment.setup_commands]]\ncommand = "useradd -m worker"\n'
        )
    (task / "solution").mkdir()
    (task / "solution/solve.sh").write_text("#!/bin/sh\necho ran > /app/control\n")
    (task / "tests/test.sh").write_text(
        "#!/bin/sh\nset -eu\n"
        'test "$(cat /app/control)" = ran\n'
        "grep -q '^NoNewPrivs:[[:space:]]*1' /proc/self/status\n"
        "echo 1 > /logs/verifier/reward.txt\n"
    )
    planes = default_rollout_planes()
    original_factory = planes.create_environment
    created = []

    def create(*args, **kwargs):
        env = original_factory(*args, **kwargs)
        assert env.agent_container_no_new_privileges is True
        env.start = AsyncMock(wraps=env.start)
        created.append(env)
        return env

    factory = Mock(side_effect=create)
    monkeypatch.setattr(planes, "create_environment", factory)
    result = await bf.run(
        RolloutConfig.from_legacy(
            task_path=task,
            agent="oracle",
            environment="docker",
            jobs_dir=tmp_path / "jobs",
            rollout_name="k1r152r3-capable-factory",
            usage_tracking="off",
            planes=planes,
        )
    )
    factory.assert_called_once()
    [env] = created
    env.start.assert_awaited_once()
    assert result.error is None, result.error
    assert result.verifier_error is None, result.verifier_error
    assert result.reward == 1.0
    print(f"CAPABLE FACTORY start_calls={env.start.await_count} reward={result.reward}")
