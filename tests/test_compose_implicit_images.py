"""Guards C8, the compatibility-name sibling missed by 67f9c75d's C6 fix."""

import json
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import docker as d
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.daytona_dind import _DaytonaDinD
from benchflow.task.config import SandboxConfig
from tests.test_daytona_dind_build_isolation import resolved_services
from tests.test_daytona_dind_followup import strategy
from tests.test_docker_build_identity import sandbox
from tests.test_docker_prebuilt_choice import services, stub_start


async def plan(tmp_path, monkeypatch, backend, config, bake, compatibility):
    # The operator explicitly opts this host Compose control into the plan.
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_ENV", "COMPOSE_COMPATIBILITY")
    if backend == "dind":
        env, vm = strategy(tmp_path, monkeypatch, True)
        env._persistent_env["COMPOSE_COMPATIBILITY"] = compatibility
        (env.environment_dir / "docker-compose.yaml").write_text(json.dumps(config))
        impl = _DaytonaDinD(env)
        original = impl._compose_exec
        calls = []

        async def execute(args, **kwargs):
            calls.append(args)
            if args[:2] == ["build", "--print"]:
                assert (
                    impl._compose_env_vars()["COMPOSE_COMPATIBILITY"] == compatibility
                )
                return bake
            return await original(args, **kwargs)

        impl._compose_exec = execute
        impl._compose_up_with_retry = AsyncMock()
        await impl.start(False)
        return resolved_services(impl, vm), calls
    env = sandbox(
        tmp_path, backend, task_env_config=SandboxConfig(docker_image="trusted:v1")
    )
    env._persistent_env["COMPOSE_COMPATIBILITY"] = compatibility
    (env.environment_dir / "docker-compose.yaml").write_text(json.dumps(config))
    stub_start(env, config)
    calls = []

    async def execute(args, **kwargs):
        calls.append(args)
        if args[:2] == ["build", "--print"]:
            assert env._docker_compose_env()["COMPOSE_COMPATIBILITY"] == compatibility
            return bake
        return ExecResult(return_code=0, stdout=json.dumps(config))

    env._run_docker_compose_command = execute
    await d.DockerSandbox.start(env, False)
    return services(env), calls


def baked(service, tags):
    # Compose's bake key may differ from the actual service name (e.g. dots).
    return ExecResult(
        return_code=0,
        stdout=json.dumps(
            {
                "target": {
                    "arbitrary-bake-key": {
                        "labels": {"com.docker.compose.service": service},
                        "tags": tags,
                    }
                }
            }
        ),
    )


@pytest.mark.parametrize("backend", ["docker", "remote-docker", "dind"])
@pytest.mark.parametrize(
    "worker", ["worker", "worker.dot", "worker-hyphen", "worker_under"]
)
@pytest.mark.parametrize("compatibility", ["1", "true", "0", "false"])
@pytest.mark.parametrize("form", ["dash", "underscore", "resolved"])
async def test_consumers_use_exact_compose_outputs(
    tmp_path, monkeypatch, backend, worker, compatibility, form
):
    alias = {
        "dash": f"trial-{worker}",
        "underscore": f"trial_{worker}",
        "resolved": "exact-output:v4",
    }[form]
    config = {
        "name": "trial",
        "services": {
            "main": {"image": "trusted:v1"},
            worker: {"build": {"context": ".", "tags": ["additional:v1"]}},
            "grader": {
                "image": "docker.io/library/" + alias,
                "volumes": [
                    {"type": "image", "source": alias, "target": "/baseline"},
                    {"type": "image", "source": "additional:v1", "target": "/extra"},
                ],
            },
        },
    }
    result, calls = await plan(
        tmp_path,
        monkeypatch,
        backend,
        config,
        baked(worker, [alias, "additional:v1"]),
        compatibility,
    )
    private = result[worker]["image"]
    assert result["grader"]["image"] == private
    assert all(mount["source"] == private for mount in result["grader"]["volumes"])
    assert calls.count(["build", "--print", "--", worker]) == 1
    assert result["main"]["image"] == "trusted:v1"
    assert not result["main"].get("build")


@pytest.mark.parametrize("backend", ["docker", "remote-docker", "dind"])
@pytest.mark.parametrize(
    "bad",
    [
        ExecResult(return_code=1, stdout="unsupported"),
        ExecResult(return_code=0, stdout="bad json"),
        ExecResult(return_code=0, stdout="{}"),
        ExecResult(return_code=0, stdout='{"target": []}'),
        baked("other", ["trial_worker"]),
        baked("worker", []),
        baked("worker", [None]),
        baked("worker", "trial_worker"),
        ExecResult(
            return_code=0,
            stdout=json.dumps(
                {
                    "target": {
                        "a": {
                            "labels": {"com.docker.compose.service": "worker"},
                            "tags": ["one"],
                        },
                        "b": {
                            "labels": {"com.docker.compose.service": "worker"},
                            "tags": ["two"],
                        },
                    }
                }
            ),
        ),
    ],
)
async def test_unknown_implicit_output_refuses_before_build(
    tmp_path, monkeypatch, backend, bad
):
    config = {
        "name": "trial",
        "services": {
            "main": {"image": "trusted:v1"},
            "worker": {"build": "."},
        },
    }
    with pytest.raises(RuntimeError, match="Could not isolate Compose build"):
        await plan(tmp_path, monkeypatch, backend, config, bad, "1")


@pytest.mark.parametrize("backend", ["docker", "remote-docker", "dind"])
async def test_all_implicit_profiles_use_service_metadata(
    tmp_path, monkeypatch, backend
):
    config = {
        "name": "trial",
        "services": {
            "main": {"image": "trusted:v1"},
            "worker.dot": {"build": ".", "profiles": ["optional"]},
            "worker_under": {"build": "."},
            "grader": {
                "image": "trial_worker.dot",
                "volumes": [
                    {
                        "type": "image",
                        "source": "trial_worker_under",
                        "target": "/baseline",
                    }
                ],
            },
        },
    }
    bake = ExecResult(
        return_code=0,
        stdout=json.dumps(
            {
                "target": {
                    "worker_under": {
                        "labels": {"com.docker.compose.service": "worker_under"},
                        "tags": ["trial_worker_under"],
                    },
                    "worker_dot": {
                        "labels": {"com.docker.compose.service": "worker.dot"},
                        "tags": ["trial_worker.dot"],
                    },
                }
            }
        ),
    )
    result, calls = await plan(tmp_path, monkeypatch, backend, config, bake, "1")
    assert result["grader"]["image"] == result["worker.dot"]["image"]
    assert result["grader"]["volumes"][0]["source"] == result["worker_under"]["image"]
    assert ["build", "--print", "--", "worker.dot", "worker_under"] in calls


@pytest.mark.parametrize("backend", ["docker", "remote-docker", "dind"])
@pytest.mark.parametrize("consumer", ["image", "mount"])
async def test_implicit_and_explicit_producers_cannot_ambiguously_feed_consumers(
    tmp_path, monkeypatch, backend, consumer
):
    grader = (
        {"image": "trial_worker"}
        if consumer == "image"
        else {
            "image": "grader:v1",
            "volumes": [
                {"type": "image", "source": "trial_worker", "target": "/baseline"}
            ],
        }
    )
    config = {
        "name": "trial",
        "services": {
            "main": {"image": "trusted:v1"},
            "worker": {"build": "."},
            "other": {"image": "docker.io/library/trial_worker:latest", "build": "."},
            "grader": grader,
        },
    }
    with pytest.raises(RuntimeError, match="Could not isolate Compose build"):
        await plan(
            tmp_path,
            monkeypatch,
            backend,
            config,
            baked("worker", ["trial_worker"]),
            "1",
        )
