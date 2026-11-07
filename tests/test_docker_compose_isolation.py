"""Guards FF-sw-docker C1/C2 against the gaps at e6151456 and 363d815b."""

import json

import pytest

from benchflow.sandbox import docker as d
from tests.test_docker_build_identity import sandbox
from tests.test_docker_prebuilt_choice import services, stub_start


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
@pytest.mark.parametrize(
    "image", [None, "shared:v1", "registry.invalid/team/shared:v1"]
)
@pytest.mark.parametrize(
    "build",
    [".", {"context": ".", "tags": ["shared:extra", "registry.invalid/extra:v2"]}],
)
@pytest.mark.parametrize("force", [False, True])
async def test_every_built_service_keeps_private_output_refs(
    tmp_path, backend, image, build, force
):
    observed = []
    for name in ("a", "b"):
        env = sandbox(tmp_path / name, backend)
        spec = {"build": build, "pull_policy": "always"}
        if image:
            spec["image"] = image
        document = {
            "services": {
                "main": dict(spec),
                "worker": dict(spec),
                "provider": {"image": "provider:v1"},
            }
        }
        (env.environment_dir / "docker-compose.yaml").write_text(json.dumps(document))
        stub_start(env, document)
        await d.DockerSandbox.start(env, force)
        current = services(env)
        refs = set()
        for service in ("main", "worker"):
            built = current[service]
            ref = built.get("image")
            assert ref and ref not in (
                image,
                "shared:extra",
                "registry.invalid/extra:v2",
            )
            assert ref not in refs, "two different service builds share an output"
            assert not built["build"].get("tags"), (
                "additional shared output tags survived"
            )
            assert built["pull_policy"] == "never", (
                "up could replace our completed build"
            )
            refs.add(ref)
        assert current["provider"] == {
            "image": "provider:v1",
            "security_opt": ["no-new-privileges:true"],
        }
        observed.append(refs)
    assert observed[0].isdisjoint(observed[1]), "interleaved trials share output tags"


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
@pytest.mark.parametrize("output", ["not json", "{}", '{"services": null}'])
async def test_unreadable_build_plan_refuses_startup(tmp_path, backend, output):
    env = sandbox(tmp_path, backend)
    document = {"services": {"main": {"build": "."}}}
    (env.environment_dir / "docker-compose.yaml").write_text(json.dumps(document))
    stub_start(env, document)
    from unittest.mock import AsyncMock

    from benchflow.sandbox._base import ExecResult

    env._run_docker_compose_command = AsyncMock(
        return_value=ExecResult(stdout=output, return_code=0)
    )
    with pytest.raises(RuntimeError, match="Could not isolate Compose build"):
        await d.DockerSandbox.start(env, False)
