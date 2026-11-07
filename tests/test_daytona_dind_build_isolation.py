"""Guards FF-sw-docker C5 at df17537e and its pre-existing 7a2387ac path."""

import json
from unittest.mock import AsyncMock

import pytest
import yaml

from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.daytona_dind import _DaytonaDinD
from tests.test_daytona_dind_followup import strategy
from tests.test_docker_prebuilt_choice import _Loader, _Reset


def resolved_services(impl, vm):
    services = {}

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

    for path in impl._compose_file_flags()[1::2]:
        text = vm.path(path).read_text()
        for key, value in impl._compose_env_vars().items():
            text = text.replace("${" + key + "}", value)
        merge(services, yaml.load(text, Loader=_Loader).get("services", {}))
    return services


@pytest.mark.parametrize("prebuilt", [False, True])
@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("collision", ["image", "tags", "both"])
async def test_dind_builds_cannot_write_selected_or_shared_tags(
    tmp_path, monkeypatch, prebuilt, force, collision
):
    env, vm = strategy(tmp_path, monkeypatch, prebuilt)
    main = {"image": "trusted:v1", "build": {"context": "."}}
    worker = {
        "image": "trusted:v1" if collision != "tags" else "worker:v1",
        "build": {
            "context": ".",
            "tags": ["docker.io/library/trusted:v1"] if collision != "image" else [],
        },
        "pull_policy": "always",
    }
    document = {"services": {"main": main, "worker": worker}}
    (env.environment_dir / "docker-compose.yaml").write_text(json.dumps(document))
    impl = _DaytonaDinD(env)
    original = impl._compose_exec
    built = []

    async def execute(args, **kwargs):
        if args == ["config", "--format", "json"]:
            return ExecResult(
                return_code=0,
                stdout=json.dumps({"services": resolved_services(impl, vm)}),
            )
        if args == ["build"]:
            built.append(resolved_services(impl, vm))
            return ExecResult(return_code=0)
        return await original(args, **kwargs)

    impl._compose_exec = execute
    impl._compose_up_with_retry = AsyncMock()
    await impl.start(force)
    assert len(built) == 1
    current = built[0]
    refs = set()
    for service in ("main", "worker"):
        spec = current[service]
        if service == "main" and prebuilt and not force:
            assert spec["image"] == "trusted:v1"
            assert not spec.get("build")
            continue
        assert spec["image"] not in {"trusted:v1", "worker:v1"}
        assert spec["image"] not in refs
        assert not spec["build"].get("tags")
        assert spec["pull_policy"] == "never"
        refs.add(spec["image"])


@pytest.mark.parametrize(
    "output,code", [("bad json", 0), ("{}", 0), ('{"services":{}}', 1)]
)
async def test_dind_unknown_build_plan_refuses_before_build(
    tmp_path, monkeypatch, output, code
):
    env, _ = strategy(tmp_path, monkeypatch, True)
    impl = _DaytonaDinD(env)
    impl._compose_exec = AsyncMock(
        return_value=ExecResult(stdout=output, return_code=code)
    )
    impl._compose_up_with_retry = AsyncMock()
    with pytest.raises(RuntimeError, match="Could not isolate Compose build"):
        await impl.start(False)
    assert all(call.args[0] != ["build"] for call in impl._compose_exec.await_args_list)
    impl._compose_up_with_retry.assert_not_awaited()
