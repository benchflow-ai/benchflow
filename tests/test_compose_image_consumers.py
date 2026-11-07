"""Guards FF-sw-docker C6, introduced by 283f9ccb and present at df17537e."""

import json
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import docker as d
from benchflow.sandbox.daytona_dind import _DaytonaDinD
from benchflow.task.config import SandboxConfig
from tests.test_daytona_dind_build_isolation import resolved_services
from tests.test_daytona_dind_followup import strategy
from tests.test_docker_build_identity import sandbox
from tests.test_docker_prebuilt_choice import services, stub_start

BACKENDS = ["docker", "remote-docker", "dind"]


async def start_plan(tmp_path, monkeypatch, backend, document, prebuilt=False):
    if backend == "dind":
        env, vm = strategy(tmp_path, monkeypatch, prebuilt)
        (env.environment_dir / "docker-compose.yaml").write_text(json.dumps(document))
        impl = _DaytonaDinD(env)
        impl._compose_up_with_retry = AsyncMock()
        await impl.start(False)
        return resolved_services(impl, vm)
    env = sandbox(
        tmp_path,
        backend,
        task_env_config=SandboxConfig(docker_image="trusted:v1" if prebuilt else None),
    )
    (env.environment_dir / "docker-compose.yaml").write_text(json.dumps(document))
    stub_start(env, document)
    await d.DockerSandbox.start(env, False)
    return services(env)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("producer", ["main", "worker"])
@pytest.mark.parametrize("additional", [False, True])
@pytest.mark.parametrize(
    "source,consumer",
    [
        ("task", "task:latest"),
        ("task:latest", "task"),
        ("task:v1", "docker.io/library/task:v1"),
        ("docker.io/task:v1", "library/task:v1"),
        ("index.docker.io/library/task:v1", "task:v1"),
        ("team/task", "docker.io/team/task:latest"),
        ("localhost:5000/task:v1", "localhost:5000/task:v1"),
        ("registry.invalid/team/task", "registry.invalid/team/task:latest"),
    ],
)
async def test_image_consumers_follow_their_private_producer(
    tmp_path, monkeypatch, backend, producer, additional, source, consumer
):
    spec = {
        "image": "different:primary" if additional else source,
        "build": {"context": ".", "tags": [source] if additional else []},
    }
    document = {
        "services": {
            "main": {"image": "unrelated:v1"},
            producer: spec,
            "grader": {"image": consumer, "pull_policy": "never"},
            "reader": {"image": consumer, "pull_policy": "always"},
            "external": {"image": "registry.invalid/external:v1"},
        }
    }
    result = await start_plan(tmp_path, monkeypatch, backend, document)
    assert result["grader"]["image"] == result[producer]["image"]
    assert result["reader"]["image"] == result[producer]["image"]
    assert result["reader"]["pull_policy"] == "never"
    assert not result[producer]["build"].get("tags")
    assert result["external"] == {
        **document["services"]["external"],
        "security_opt": ["no-new-privileges:true"],
    }


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("additional", [False, True])
async def test_selected_prebuilt_main_is_not_rebound_to_a_sidecar(
    tmp_path, monkeypatch, backend, additional
):
    document = {
        "services": {
            "main": {"image": "trusted:v1", "build": "."},
            "worker": {
                "image": "worker:v1" if additional else "docker.io/library/trusted:v1",
                "build": {"context": ".", "tags": ["trusted:v1"] if additional else []},
            },
            "reader": {"image": "index.docker.io/library/trusted:v1"},
        }
    }
    result = await start_plan(tmp_path, monkeypatch, backend, document, prebuilt=True)
    assert result["main"]["image"] == "trusted:v1"
    assert not result["main"].get("build")
    assert result["reader"]["image"] == result["worker"]["image"]
    assert result["reader"]["image"] != "trusted:v1"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("additional", [False, True])
async def test_ambiguous_image_consumer_refuses_startup(
    tmp_path, monkeypatch, backend, additional
):
    document = {
        "services": {
            "main": {"image": "task:v1", "build": "."},
            "worker": {
                "image": "worker:v1" if additional else "docker.io/library/task:v1",
                "build": {"context": ".", "tags": ["task:v1"] if additional else []},
            },
            "grader": {"image": "index.docker.io/library/task:v1"},
        }
    }
    with pytest.raises(RuntimeError, match="Could not isolate Compose build"):
        await start_plan(tmp_path, monkeypatch, backend, document)


@pytest.mark.parametrize("backend", BACKENDS)
async def test_other_tags_digests_namespaces_and_registries_remain_distinct(
    tmp_path, monkeypatch, backend
):
    external = {
        name: {"image": image}
        for name, image in {
            "tag": "task:v2",
            "registry": "registry.invalid/task:v1",
            "namespace": "team/task:v1",
            "nested": "docker.io/library/library/task:v1",
            "digest": "task@sha256:" + "a" * 64,
        }.items()
    }
    document = {"services": {"main": {"image": "task:v1", "build": "."}, **external}}
    result = await start_plan(tmp_path, monkeypatch, backend, document)
    assert {name: result[name] for name in external} == {
        name: {**spec, "security_opt": ["no-new-privileges:true"]}
        for name, spec in external.items()
    }


@pytest.mark.parametrize(
    "consumer", ["trial-worker", "docker.io/library/trial-worker:latest"]
)
def test_implicit_compose_build_name_has_the_same_consumers(consumer):
    import yaml

    from benchflow.sandbox._compose_runtime import private_build_override
    from tests.test_docker_prebuilt_choice import _Loader

    document, refs = private_build_override(
        {
            "name": "trial",
            "services": {"worker": {"build": "."}, "reader": {"image": consumer}},
        },
        "private-trial",
        prebuilt=False,
        implicit_aliases={"worker": ["trial-worker"]},
    )
    services = yaml.load(document, Loader=_Loader)["services"]
    assert services["worker"]["image"] == services["reader"]["image"]
    assert refs == {services["worker"]["image"]}


def test_repeated_aliases_of_one_producer_are_not_ambiguous():
    import yaml

    from benchflow.sandbox._compose_runtime import private_build_override
    from tests.test_docker_prebuilt_choice import _Loader

    document, _ = private_build_override(
        {
            "services": {
                "main": {
                    "image": "task",
                    "build": {
                        "context": ".",
                        "tags": ["task:latest", "docker.io/library/task"],
                    },
                },
                "reader": {"image": "index.docker.io/task"},
            }
        },
        "private-trial",
        prebuilt=False,
    )
    services = yaml.load(document, Loader=_Loader)["services"]
    assert services["main"]["image"] == services["reader"]["image"]
