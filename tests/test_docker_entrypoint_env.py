"""Guards the fix at integration commit 363d815b."""

import json

import pytest
import yaml

from benchflow.sandbox import docker as d
from tests.test_docker_build_identity import sandbox


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
@pytest.mark.parametrize("custom_compose", [False, True])
async def test_create_env_reaches_main_and_never_inherits_ambient_values(
    tmp_path, monkeypatch, backend, custom_compose
):
    """Guards 363d815b: manifest env is container config before PID 1 starts."""
    monkeypatch.setenv("UNRELATED_HOST_MARKER", "must not reach PID1")
    values = {"TASK_ID": "a$${b}: 'quoted'\nnext", "FORWARDED": "", "BOOL": "true"}
    env = sandbox(tmp_path, backend, persistent_env=values)
    if custom_compose:
        (env.environment_dir / "docker-compose.yaml").write_text(
            "services:\n  main:\n    environment:\n      TASK_ID: default\n  sidecar:\n    image: scratch\n"
        )
    # Exercise the real startup path up to container creation without a daemon.
    seen = []

    async def up():
        documents = [
            yaml.safe_load(path.read_text()) for path in env._docker_compose_paths
        ]
        merged = {}
        for document in documents:
            service = document.get("services", {}).get("main", {})
            merged.update(service.get("environment", {}))
        # Compose expands $$ to a literal $, independently of client env.
        seen.append({k: v.replace("$$", "$") for k, v in merged.items()})

    from unittest.mock import AsyncMock

    env._run_docker_compose_build = AsyncMock()
    env._run_docker_compose_up = up
    env._prepare_compose_builds = AsyncMock()
    env._choose_subnets = AsyncMock()
    env._name_trial_volumes = AsyncMock()
    env._learn_main_container = AsyncMock()
    env._prepare_log_dirs_after_up = AsyncMock()
    await d.DockerSandbox.start(env, False)
    assert seen == [values]
    assert "UNRELATED_HOST_MARKER" not in json.dumps(seen)
