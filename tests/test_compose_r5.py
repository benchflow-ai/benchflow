"""Guards environment delivery against e7227c15's file-bind conversion."""

from __future__ import annotations

import json

import pytest

from tests.test_compose_launch import _from_task, _launched
from tests.test_compose_oracle import _compose_available, _compose_task, _gate

pytestmark = pytest.mark.skipif(not _compose_available(), reason="needs Compose config")


@pytest.fixture(autouse=True)
def _operator(monkeypatch, tmp_path):
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_ENV", "")
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "absent.env"))
    monkeypatch.setenv("BENCHFLOW_REMOTE_DOCKER_HOST", "ssh://s1@192.0.2.1")
    monkeypatch.delenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", raising=False)
    monkeypatch.delenv("S1_DATA", raising=False)
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
@pytest.mark.parametrize("top", ["secrets", "configs"])
@pytest.mark.parametrize("origin", ["task-env", "dotenv", "dotenv-loader", "allowed"])
@pytest.mark.parametrize("reference", ["short", "owned"])
def test_r5_environment_delivery_survives_freezing(
    monkeypatch, tmp_path, backend, top, origin, reference
):
    """Guards environment delivery: no client bind; native per-consumer metadata."""
    variable = "LD_PRELOAD" if origin == "dotenv-loader" else "S1_DATA"
    value = "task-value\nsecond=$literal ${STILL_LITERAL} é"
    toml = ""
    files = {}
    if origin == "task-env":
        toml = f"[environment.env]\n{variable} = {json.dumps(value)}\n"
    elif origin == "allowed":
        monkeypatch.setenv(variable, value)
        monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_ENV", variable)
    else:
        files[".env"] = f"{variable}='{value}'\n"
    mount = (
        "payload"
        if reference == "short"
        else '{source: payload, target: /tmp/owned-data, uid: "65534", gid: "65533", mode: 0440}'
    )
    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n"
        f"    {top}: [{mount}]\n"
        "  side:\n    image: alpine:3.20\n    user: '65534:65533'\n"
        f"    {top}: [payload]\n"
        f"{top}:\n  payload: {{environment: {variable}}}\n",
        toml,
        files=files,
    )
    _gate(task, sandbox_type=backend)
    sandbox = _from_task(task, backend).inner
    launch = sandbox._launch
    project = _launched(sandbox)
    source = project[top]["payload"]
    assert "file" not in source, "freezing must not introduce a daemon-local bind"
    name = source["environment"]
    assert name.startswith("BENCHFLOW_COMPOSE_SOURCE_")
    assert name != variable
    assert launch.env[name] == value
    if origin == "dotenv-loader":
        assert variable not in launch.env
    # Every consumer keeps its own delivery contract, even for a shared source.
    for service in ("main", "side"):
        ref = project["services"][service][top][0]
        assert ref["source"] == "payload"
        if reference == "owned" and service == "main":
            assert (ref["target"], ref["uid"], ref["gid"], ref["mode"]) == (
                "/tmp/owned-data",
                "65534",
                "65533",
                "0440",
            )
        else:
            assert not {"uid", "gid", "mode"} & ref.keys()
    # An unchanged restart reuses the frozen bytes, not the live dotenv file.
    (sandbox.environment_dir / ".env").write_text(f"{variable}=changed\n")
    sandbox.freeze_launch()
    assert sandbox._launch is launch
    assert _launched(sandbox) == project
    assert sandbox._launch.env[name] == value


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
@pytest.mark.parametrize("top", ["secrets", "configs"])
@pytest.mark.parametrize("value", ["", "single", "a\nb\n", "$x ${y} $$", "日本語"])
def test_r5_environment_delivery_keeps_exact_values(tmp_path, backend, top, value):
    """Guards environment delivery without newline, dollar or empty-value loss."""
    task = _compose_task(
        tmp_path / "task",
        f"services:\n  main:\n    {top}: [a, b]\n{top}:\n"
        "  a: {environment: S1_DATA}\n  b: {environment: S1_DATA}\n",
        f"[environment.env]\nS1_DATA = {json.dumps(value)}\n",
    )
    sandbox = _from_task(task, backend).inner
    project = _launched(sandbox)
    for source in project[top].values():
        assert "file" not in source
        assert sandbox._launch.env[source["environment"]] == value


@pytest.mark.parametrize("top", ["secrets", "configs"])
@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
def test_r5_an_unnamed_operator_value_is_still_unavailable(
    monkeypatch, tmp_path, top, backend
):
    """Guards the repair against reintroducing ambient reads."""
    from benchflow.task.env import TaskEnvNotAllowedError

    monkeypatch.setenv("S1_DATA", "dummy-operator-marker")
    task = _compose_task(
        tmp_path / "task",
        f"services:\n  main:\n    {top}: [payload]\n{top}:\n"
        "  payload: {environment: S1_DATA}\n",
    )
    with pytest.raises(TaskEnvNotAllowedError, match="S1_DATA"):
        _from_task(task, backend)
