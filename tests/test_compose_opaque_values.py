"""Guards S1-CONF-1 at 5539d4c6: path consumers cannot rewrite content."""

from __future__ import annotations

import pytest

from benchflow.task.compose_env import (
    _operator_project_paths,
    _ProjectCheck,
    compose_process_env,
    decode_dollars,
)
from tests.test_compose_launch import _from_task, _launched
from tests.test_compose_mutable_sources import _project
from tests.test_compose_oracle import _compose_available, _compose_task, _gate

PAYLOADS = [
    "/dummy//token/",
    "//dummy///token//",
    "/dummy/../token/",
    "/dummy/./token/",
    "/dummy/$literal/${LITERAL}//\n日本語/",
]


@pytest.fixture(autouse=True)
def operator(monkeypatch, tmp_path):
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_ENV", "S1_DATA")
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "absent.env"))
    monkeypatch.setenv("BENCHFLOW_REMOTE_DOCKER_HOST", "ssh://s1@192.0.2.1")
    monkeypatch.delenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize("value", PAYLOADS)
def test_process_environment_keeps_opaque_slash_values(value):
    """Guards S1-CONF-1 at 5539d4c6 for named values and equal task aliases."""
    env = compose_process_env(
        {"COPY": value}, environ={"S1_DATA": value}, allowed={"S1_DATA"}
    )
    assert env["S1_DATA"] == env["COPY"] == value


@pytest.mark.skipif(not _compose_available(), reason="needs Compose config")
@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
@pytest.mark.parametrize("top", ["secrets", "configs"])
@pytest.mark.parametrize("origin", ["named", "task-copy", "dotenv-copy"])
@pytest.mark.parametrize("payload", [*PAYLOADS, "host-alias"])
def test_frozen_content_keeps_exact_slash_values(
    monkeypatch, tmp_path, backend, top, origin, payload
):
    """Guards S1-CONF-1 at 5539d4c6 through preflight, freeze and restart."""
    if payload == "host-alias":
        physical = tmp_path / "physical"
        physical.mkdir()
        link = tmp_path / "logical"
        link.symlink_to(physical, target_is_directory=True)
        payload = str(link) + "/"
    monkeypatch.setenv("S1_DATA", payload)
    variable = "S1_DATA" if origin == "named" else "COPY"
    toml = '[environment.env]\nCOPY = "${S1_DATA}"\n' if origin == "task-copy" else ""
    files = {".env": "COPY=${S1_DATA}\n"} if origin == "dotenv-copy" else {}
    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n"
        f'    environment: {{TEXT: "${{{variable}}}"}}\n'
        f'    labels: {{text: "${{{variable}}}"}}\n'
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
    assert "file" not in source
    assert launch.env[source["environment"]] == payload
    # Compose escapes dollar signs in its printed project, not source bytes.
    printed = payload.replace("$", "$$")
    assert project["services"]["main"]["environment"]["TEXT"] == printed
    assert project["services"]["main"]["labels"]["text"] == printed
    assert launch.env["S1_DATA"] == payload
    if origin == "task-copy":
        assert launch.env["COPY"] == payload
    monkeypatch.setenv("S1_DATA", "/changed//later/")
    # The existing launch stays usable even after the operator changes a value.
    assert _launched(sandbox) == project
    assert launch.env[source["environment"]] == payload


@pytest.mark.parametrize(
    "consumer",
    [
        "bind",
        "secrets",
        "configs",
        "env_file",
        "label_file",
        "device",
        "context",
        "dockerfile",
        "additional_contexts",
        "oci_context",
        "cache_from",
        "cache_to",
        "ssh",
    ],
)
@pytest.mark.parametrize("spelling", ["plain", "dots", "slashes", "dollars"])
def test_only_host_path_consumers_resolve_named_roots(tmp_path, consumer, spelling):
    """Guards S1-CONF-1 at 5539d4c6 across every checked path consumer."""
    physical = tmp_path / "physical"
    physical.mkdir()
    link = tmp_path / ("logical$root" if spelling == "dollars" else "logical")
    link.symlink_to(physical, target_is_directory=True)
    raw = str(link) + {"dots": "/./", "slashes": "//"}.get(spelling, "")
    source = link / "data"
    project = _project(tmp_path / "work", source, "bind", consumer)
    if consumer == "oci_context":
        # Compose prints additional_contexts as a map, including list input.
        project["services"]["main"]["build"]["additional_contexts"] = {
            "data": f"oci-layout://{source}"
        }
    project["services"]["main"].update(
        environment={"TEXT": raw}, labels={"text": raw}, command=[raw]
    )
    project.setdefault("configs", {})["text"] = {"content": raw}

    # Model Compose's dollar escaping, which covers paths and text alike.
    def printed(value):
        if isinstance(value, dict):
            return {key: printed(item) for key, item in value.items()}
        if isinstance(value, list):
            return [printed(item) for item in value]
        return value.replace("$", "$$") if isinstance(value, str) else value

    before = printed(project)
    mapped = _operator_project_paths(before, {"S1_DATA"}, {"S1_DATA": raw})
    paths = _ProjectCheck(str(tmp_path), (str(tmp_path),)).paths(decode_dollars(mapped))
    assert str(physical / "data") in paths
    assert not any(str(link) in path for path in paths)
    main = mapped["services"]["main"]
    for key in ("environment", "labels", "command"):
        assert main[key] == before["services"]["main"][key]
    assert mapped["configs"]["text"] == before["configs"]["text"]
    assert before == printed(project), "mapping must not mutate its input"


@pytest.mark.parametrize(
    "driver,kind", [("local", "cifs"), ("local", "nfs"), ("plugin", "none")]
)
def test_non_host_volume_options_are_opaque(tmp_path, driver, kind):
    """Guards S1-CONF-1 at 5539d4c6 for volume-driver payload siblings."""
    value = "//dummy///share/"
    project = {
        "volumes": {
            "data": {"driver": driver, "driver_opts": {"type": kind, "device": value}}
        }
    }
    mapped = _operator_project_paths(project, {"S1_DATA"}, {"S1_DATA": value})
    assert mapped == project
