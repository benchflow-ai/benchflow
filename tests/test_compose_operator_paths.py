"""Guards 1043a78a's operator-symlink regression (s1-b8r2 review note 1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchflow.sandbox.docker import DockerSandbox
from benchflow.task.compose_env import benchflow_compose_values, compose_process_env
from benchflow.task.config import SandboxConfig
from benchflow.task.env import TaskEnvNotAllowedError
from benchflow.task.paths import RolloutPaths
from tests.test_compose_launch import _from_task, _launched
from tests.test_compose_oracle import _compose_available, _compose_task, _gate


@pytest.fixture(autouse=True)
def operator(monkeypatch, tmp_path):
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_ENV", "")
    monkeypatch.delenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", raising=False)
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "absent.env"))
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def alias(tmp_path):
    root = tmp_path / "physical"
    root.mkdir()
    (root / "data").mkdir()
    link = tmp_path / "logical"
    link.symlink_to(root, target_is_directory=True)
    return link, root


@pytest.mark.parametrize("suffix", ["", "/data", "/missing"])
def test_named_operator_values_stay_verbatim_until_used_as_paths(alias, suffix):
    """Guards S1-CONF-1 at 5539d4c6: only path consumers may use realpaths."""
    link, _root = alias
    value = str(link) + suffix
    env = compose_process_env(
        {"COPY": value, "TASK_PATH": str(link / "data") + "/child"},
        environ={"S1_PATH": value, "TEXT": "unchanged"},
        allowed={"S1_PATH", "TEXT"},
    )
    assert env["S1_PATH"] == env["COPY"] == value
    assert env["TEXT"] == "unchanged"
    assert env["TASK_PATH"] == str(link / "data") + "/child"


@pytest.mark.parametrize("with_rollout", [False, True])
def test_log_paths_are_canonical_even_without_rollout(
    monkeypatch, tmp_path, alias, with_rollout
):
    """Guards 1043a78a's /tmp fallback on hosts where /tmp is a symlink."""
    link, root = alias
    resolve = Path.resolve

    def host_resolve(path, *args, **kwargs):
        if str(path) in ("/tmp/verifier", "/tmp/agent", "/tmp/artifacts"):
            return root / path.name
        return resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", host_resolve)
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    sandbox = DockerSandbox(
        environment_dir=tmp_path,
        environment_name="operator-paths",
        session_id="operator-paths",
        rollout_paths=RolloutPaths(link) if with_rollout else None,
        task_env_config=SandboxConfig(),
    )
    # Integration keeps rollout paths as held handles until this boundary.
    # Check the actual Compose mount values, not a handle's diagnostic text.
    mounted = sandbox._env_vars.to_env_dict(include_os_env=False)
    for name in ("verifier", "agent", "artifacts"):
        key = (
            f"HOST_{name.upper()}_LOGS_PATH"
            if name != "artifacts"
            else "HOST_ARTIFACTS_PATH"
        )
        assert mounted[key] == str(root / name)
    values = benchflow_compose_values(
        tmp_path,
        SandboxConfig(),
        {"HOST_AGENT_LOGS_PATH": link / "agent"},
        image_name="fixture",
    )
    assert values["HOST_AGENT_LOGS_PATH"] == str(root / "agent")


compose = pytest.mark.skipif(
    not _compose_available(), reason="needs Compose config only"
)


@compose
@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
@pytest.mark.parametrize("indirect", [False, True])
@pytest.mark.parametrize("syntax", ["short", "long", "device"])
def test_named_paths_reach_frozen_project(
    monkeypatch, tmp_path, alias, backend, indirect, syntax
):
    """Guards 1043a78a through the gate and frozen launch; remote still refuses binds."""
    link, root = alias
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_ENV", "S1_PATH")
    monkeypatch.setenv("S1_PATH", str(link / "data"))
    variable = "COPY" if indirect else "S1_PATH"
    source = "${" + variable + "}"
    mounts = {
        "short": f'    volumes: ["{source}:/data:ro"]\n',
        "long": f'    volumes: [{{type: bind, source: "{source}", target: /data, read_only: true}}]\n',
        "device": '    volumes: ["data:/data:ro"]\n'
        f'volumes:\n  data:\n    driver_opts: {{type: none, o: bind, device: "{source}"}}\n',
    }
    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n"
        f'    environment: {{AS_TEXT: "{source}"}}\n' + mounts[syntax],
        '[environment.env]\nCOPY = "${S1_PATH}"\n' if indirect else "",
    )
    if backend == "remote-docker":
        with pytest.raises(TaskEnvNotAllowedError):
            _gate(task, sandbox_type=backend)
        return
    _gate(task)
    sandbox = _from_task(task).inner
    project = _launched(sandbox)
    if syntax == "device":
        assert project["volumes"]["data"]["driver_opts"]["device"] == str(root / "data")
    else:
        assert next(
            m["source"]
            for m in project["services"]["main"]["volumes"]
            if m["target"] == "/data"
        ) == str(root / "data")
    assert project["services"]["main"]["environment"]["AS_TEXT"] == str(link / "data")


@compose
@pytest.mark.parametrize("read_only", [False, True])
def test_operator_mount_override_is_canonical(tmp_path, alias, read_only):
    """Guards 1043a78a's --mount path before writing and freezing its override."""
    link, root = alias
    task = _compose_task(tmp_path / "task", "services:\n  main: {}\n")
    paths = RolloutPaths(tmp_path / "run")
    paths.mkdir()
    mount = {
        "type": "bind",
        "source": str(link / "data"),
        "target": "/data",
        "read_only": read_only,
    }
    sandbox = DockerSandbox(
        environment_dir=task / "environment",
        environment_name="operator-paths",
        session_id="operator-paths",
        rollout_paths=paths,
        task_env_config=SandboxConfig(),
        mounts_json=[mount],
    )
    sandbox.freeze_launch()
    project = _launched(sandbox)
    assert next(
        m["source"]
        for m in project["services"]["main"]["volumes"]
        if m["target"] == "/data"
    ) == str(root / "data")
    assert mount["source"] == str(link / "data")
    override = json.loads(sandbox._mounts_compose_path.read_text())
    assert override["services"]["main"]["volumes"][0]["source"] == str(root / "data")


@compose
@pytest.mark.parametrize("source", ["data", "alias"])
def test_named_paths_inside_agent_writers_stay_refused(
    monkeypatch, tmp_path, alias, source
):
    """Guards 1043a78a's invariant after resolving only operator aliases."""
    link, root = alias
    (root / "alias").symlink_to(root / "data", target_is_directory=True)
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_ENV", "S1_PATH")
    monkeypatch.setenv("S1_PATH", str(link))
    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n    volumes:\n"
        '      - "${S1_PATH}:/workspace"\n'
        f'      - "${{S1_PATH}}/{source}:/data:ro"\n',
    )
    with pytest.raises(TaskEnvNotAllowedError, match="writable"):
        _gate(task)
