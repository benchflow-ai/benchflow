"""Guards FF-sw-docker C1/C2 against the gaps at e6151456 and 363d815b."""

import json
from unittest.mock import AsyncMock

import pytest
import yaml

from benchflow.sandbox import docker as d
from benchflow.sandbox._base import ExecResult
from benchflow.task.config import SandboxConfig
from tests.test_docker_build_identity import sandbox


class _Reset:
    pass


class _Loader(yaml.SafeLoader):
    pass


_Loader.add_constructor("!reset", lambda loader, node: _Reset())
_Loader.add_constructor(
    "!override", lambda loader, node: loader.construct_sequence(node, deep=True)
)


def services(env):
    """Resolve the generated files' mapping/reset semantics without a daemon."""
    result = {}

    def merge(old, new):
        for key, value in new.items():
            if isinstance(value, _Reset):
                old.pop(key, None)
            elif isinstance(value, dict):
                if not isinstance(old.get(key, {}), dict):
                    old[key] = {"context": old[key]}
                merge(old.setdefault(key, {}), value)
            else:
                old[key] = value

    for path in env._docker_compose_paths:
        text = path.read_text()
        for key, value in env._docker_compose_env().items():
            text = text.replace("${" + key + "}", value)
        merge(result, yaml.load(text, Loader=_Loader) or {})
    return result["services"]


def stub_start(env, effective):
    # The planner fixture supplies resolved configuration without a Docker client.
    env.freeze_launch = lambda *args: None
    env._prepare_inert_bootstrap = AsyncMock()
    env._pin_or_verify_bind_targets = AsyncMock()
    env._release_inert_bootstrap = AsyncMock()

    async def execute(args, **kwargs):
        output = (
            bake_plan(effective, env.compose_project_name)
            if args[:2] == ["build", "--print"]
            else effective
        )
        return ExecResult(stdout=json.dumps(output), return_code=0)

    env._run_docker_compose_command = AsyncMock(side_effect=execute)
    env._run_docker_compose_build = AsyncMock()
    env._run_docker_compose_up = AsyncMock()
    env._absent_task_image = AsyncMock(return_value=None)
    env._note_task_image_pull = AsyncMock()
    env._choose_subnets = AsyncMock()
    env._name_trial_volumes = AsyncMock()
    env._learn_main_container = AsyncMock()
    env._prepare_log_dirs_after_up = AsyncMock()


def bake_plan(effective, project):
    """The fake Compose process's per-service build metadata."""
    return {
        "target": {
            name: {
                "labels": {"com.docker.compose.service": name},
                "tags": [spec.get("image", f"{project}-{name}")],
            }
            for name, spec in effective.get("services", {}).items()
            if spec.get("build")
        }
    }


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
@pytest.mark.parametrize("build", [".", {"context": ".", "tags": ["shared:extra"]}])
@pytest.mark.parametrize("policy", ["never", "build", "always", "missing"])
async def test_prebuilt_main_cannot_be_built_or_replaced(
    tmp_path, backend, build, policy
):
    env = sandbox(
        tmp_path, backend, task_env_config=SandboxConfig(docker_image="trusted:v1")
    )
    document = {
        "services": {
            "main": {"build": build, "image": "task:local", "pull_policy": policy},
            "worker": {"build": ".", "image": "worker:shared"},
        }
    }
    (env.environment_dir / "docker-compose.yaml").write_text(json.dumps(document))
    stub_start(env, document)
    await d.DockerSandbox.start(env, False)
    main = services(env)["main"]
    assert not main.get("build"), "selected prebuilt main was left buildable"
    assert main["image"] == "trusted:v1"
    assert main["pull_policy"] != "build"
    assert services(env)["worker"].get("build"), "sidecar build was lost"
    env._run_docker_compose_build.assert_awaited_once()


def test_plain_prebuilt_does_not_pin_the_task_over_install_layers(tmp_path):
    env = sandbox(tmp_path, task_env_config=SandboxConfig(docker_image="trusted:v1"))
    env._use_prebuilt = True
    env._write_prebuilt_compose_file()
    env._env_vars.prebuilt_image_name = "trusted-install-layer:private"
    main = services(env)["main"]
    assert main["image"] == "trusted-install-layer:private"


async def test_sidecars_do_not_pin_main_over_its_install_layer(tmp_path):
    env = sandbox(tmp_path)
    document = {"services": {"main": {"build": "."}, "worker": {"build": "."}}}
    (env.environment_dir / "docker-compose.yaml").write_text(json.dumps(document))
    stub_start(env, document)

    async def layer():
        env._use_prebuilt = True
        env._env_vars.prebuilt_image_name = "install-layer:private"

    env._start_from_layers = layer
    await d.DockerSandbox.start(env, False)
    main = services(env)["main"]
    assert main["image"] == "install-layer:private"
    assert not main.get("build")
    assert services(env)["worker"].get("build")
