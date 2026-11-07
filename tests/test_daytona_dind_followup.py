"""Guards FF-sw-docker C1/C3/C4 at e6151456, extending 363d815b coverage."""

import json
from unittest.mock import AsyncMock

import pytest
import yaml

from benchflow.sandbox.daytona_dind import _DaytonaDinD
from tests.test_daytona_dind_task_folder import _dind_sandbox, _FakeDaytonaVM
from tests.test_docker_prebuilt_choice import _Loader


def strategy(tmp_path, monkeypatch, prebuilt):
    # The follow-up tests construct a second strategy on the same fake VM.
    # Preserve the admission/proof seams installed by _dind_sandbox there too.
    monkeypatch.setattr(_DaytonaDinD, "_check_compose_project", AsyncMock())
    monkeypatch.setattr(_DaytonaDinD, "_prepare_inert_bootstrap", AsyncMock())
    monkeypatch.setattr(_DaytonaDinD, "_pin_or_verify_bind_targets", AsyncMock())
    directory = tmp_path / "task" / "environment"
    directory.mkdir(parents=True)
    (directory / "docker-compose.yaml").write_text(
        json.dumps(
            {
                "services": {
                    "main": {
                        "image": "task:local",
                        "build": {"context": ".", "no_cache": True},
                        "pull_policy": "build",
                        "environment": {"TASK_ID": "default", "KEEP": "yes"},
                    }
                }
            }
        )
    )
    vm = _FakeDaytonaVM(tmp_path / "vm")
    env = _dind_sandbox(directory, vm, monkeypatch)
    env.task_env_config.docker_image = "trusted:v1" if prebuilt else None
    env.task_env_config.build_timeout_sec = 5
    return env, vm


@pytest.mark.parametrize("prebuilt", [False, True])
@pytest.mark.parametrize("force", [False, True])
async def test_dind_pid1_gets_only_explicit_env(tmp_path, monkeypatch, prebuilt, force):
    monkeypatch.setenv("HOST_ONLY_MARKER", "private")
    env, vm = strategy(tmp_path, monkeypatch, prebuilt)
    env.task_env_config.env = {"TASK_ID": "task-default", "EMPTY": ""}
    env._persistent_env = {"TASK_ID": "chosen$${literal}\n'quoted'", "BOOL": "true"}
    impl = _DaytonaDinD(env)
    impl._compose_up_with_retry = AsyncMock()
    await impl.start(force)
    values = {}
    flags = impl._compose_file_flags()
    for path in flags[1::2]:
        document = yaml.load(vm.path(path).read_text(), Loader=_Loader)
        values.update(
            document.get("services", {}).get("main", {}).get("environment", {})
        )
    values = {key: value.replace("$$", "$") for key, value in values.items()}
    assert values == {
        "TASK_ID": "chosen$${literal}\n'quoted'",
        "EMPTY": "",
        "BOOL": "true",
        "KEEP": "yes",
    }
    assert "HOST_ONLY_MARKER" not in values


async def test_dind_selected_prebuilt_is_not_buildable(tmp_path, monkeypatch):
    env, vm = strategy(tmp_path, monkeypatch, True)
    impl = _DaytonaDinD(env)
    impl._compose_up_with_retry = AsyncMock()
    await impl.start(False)
    documents = [
        yaml.load(vm.path(path).read_text(), Loader=_Loader)
        for path in impl._compose_file_flags()[1::2]
    ]
    main = {}
    for document in documents:
        main.update(document.get("services", {}).get("main", {}))
    assert not isinstance(main.get("build"), (str, dict))
    assert main["image"] == "trusted:v1"
    assert main["pull_policy"] != "build"
