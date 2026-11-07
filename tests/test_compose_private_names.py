"""Guards C10, the production-name mount regression left by 9d658dce."""

import json
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import docker as d
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.daytona_dind import _DaytonaDinD
from benchflow.task.config import SandboxConfig
from tests.test_compose_image_mounts import SPELLINGS
from tests.test_daytona_dind_build_isolation import resolved_services
from tests.test_daytona_dind_followup import strategy
from tests.test_docker_build_identity import sandbox
from tests.test_docker_prebuilt_choice import services, stub_start


@pytest.fixture(autouse=True)
def _docker29(monkeypatch):
    """Every backend reports Docker 29, whose image mounts hex-encode ID,
    source and destination (an unknown daemon gets the same rule). Without
    this the docker backend asked the host's daemon: on Docker 28 (GitHub's
    runner has 28.0.4) the long destinations below fit, so the refusals were
    never reached. Docker 28's rule: tests/test_compose_daemon_mount_bounds.py.
    """
    from tests.test_compose_daemon_mount_bounds import daemon_reply

    daemon_reply(monkeypatch, ExecResult(stdout="29.1.3\n", return_code=0))


async def prepare(tmp_path, monkeypatch, backend, document, prebuilt, trial):
    if backend == "dind":
        env, vm = strategy(tmp_path, monkeypatch, prebuilt)
        env.session_id = trial
        (env.environment_dir / "docker-compose.yaml").write_text(json.dumps(document))
        impl = _DaytonaDinD(env)
        impl._compose_up_with_retry = AsyncMock()
        await impl.start(False)
        return resolved_services(impl, vm), None
    # Exercise the real constructor with the production session id, including
    # the random project suffix. A shortened stand-in hid C10 in b28's repro.
    env = sandbox(
        tmp_path,
        backend,
        task_env_config=SandboxConfig(docker_image="trusted:v1" if prebuilt else None),
    )
    kwargs = dict(
        environment_dir=env.environment_dir,
        environment_name="task",
        session_id=trial,
        rollout_paths=env.rollout_paths,
        task_env_config=env.task_env_config,
    )
    if backend == "remote-docker":
        kwargs["docker_host"] = env.docker_host
    env = type(env)(**kwargs)
    (env.environment_dir / "docker-compose.yaml").write_text(json.dumps(document))
    stub_start(env, document)
    await d.DockerSandbox.start(env, False)
    return services(env), env


def document_for(producer, source, consumer, target):
    return {
        "services": {
            "main": {"image": "trusted:v1"},
            producer: {
                "image": source,
                "build": {"context": ".", "tags": ["extra:v1"]},
            },
            "grader": {
                "image": consumer,
                "volumes": [
                    {
                        "type": "image",
                        "source": consumer,
                        "target": target,
                        "read_only": True,
                    }
                ],
            },
        }
    }


@pytest.mark.parametrize("backend", ["docker", "remote-docker", "dind"])
@pytest.mark.parametrize(
    "producer,prebuilt",
    [("main", False), ("worker", True), ("worker_" + "x" * 200, False)],
)
@pytest.mark.parametrize("trial", ["trial-12345678", "trial-" + "x" * 200])
@pytest.mark.parametrize("source,consumer", SPELLINGS)
async def test_production_private_names_fit_mounts_and_cleanup(
    tmp_path, monkeypatch, backend, producer, prebuilt, trial, source, consumer
):
    document = document_for(producer, source, consumer, "/baseline")
    result, env = await prepare(
        tmp_path, monkeypatch, backend, document, prebuilt, trial
    )
    ref = result[producer]["image"]
    assert len(ref.encode()) <= 34
    mount = result["grader"]["volumes"][0]
    assert mount["source"] == result["grader"]["image"] == ref
    assert (
        len(("0" * 64 + ",src=" + ref + ",dst=" + mount["target"]).encode().hex())
        <= 255
    )
    if env:
        assert ref in env._built_image_tags == env._private_built_image_tags
        if producer == "main":
            assert ref == env._env_vars.main_image_name + ":latest"
    if prebuilt:
        assert result["main"]["image"] == "trusted:v1"


@pytest.mark.parametrize("backend", ["docker", "remote-docker", "dind"])
@pytest.mark.parametrize("target", ["/" + "x" * 19, "/" + "é" * 10, "/" + "x" * 250])
async def test_unrepresentable_private_mount_refuses_before_build(
    tmp_path, monkeypatch, backend, target
):
    document = document_for("worker", "task:v1", "task:v1", target)
    with pytest.raises(RuntimeError, match="Could not isolate Compose build") as caught:
        await prepare(tmp_path, monkeypatch, backend, document, True, "trial-12345678")
    assert "image mount" in str(caught.value.__cause__)


@pytest.mark.parametrize("backend", ["docker", "remote-docker", "dind"])
@pytest.mark.parametrize("target", ["/" + "x" * 18, "/" + "é" * 9])
async def test_private_mount_byte_boundary(tmp_path, monkeypatch, backend, target):
    document = document_for("worker", "task:v1", "task:v1", target)
    result, _ = await prepare(
        tmp_path, monkeypatch, backend, document, True, "trial-12345678"
    )
    assert result["grader"]["volumes"][0]["target"] == target


@pytest.mark.parametrize("backend", ["docker", "remote-docker", "dind"])
async def test_private_names_distinguish_services_and_trials(
    tmp_path, monkeypatch, backend
):
    document = document_for("worker", "task:v1", "task:v1", "/baseline")
    document["services"]["worker.dot"] = {"image": "second:v1", "build": "."}
    document["services"]["worker_dot"] = {"image": "third:v1", "build": "."}
    refs = []
    for index in range(2):
        result, _ = await prepare(
            tmp_path / str(index), monkeypatch, backend, document, True, "same-trial"
        )
        refs.extend(
            result[name]["image"] for name in ("worker", "worker.dot", "worker_dot")
        )
    assert len(set(refs)) == 6
