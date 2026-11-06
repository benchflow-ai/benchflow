"""Guards C11, the Docker-28 startup regression introduced by f08fed68."""

import pytest

from benchflow.sandbox import docker as d
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.daytona_dind import _DaytonaDinD
from tests.test_compose_private_names import document_for, prepare


@pytest.fixture(autouse=True)
def _without_bfx(monkeypatch):
    # These check the task's own mounts and Compose files; bfx's read-only
    # overlay (present when the pinned binaries are built) is checked in
    # tests/test_bfx_*.py.
    monkeypatch.setenv("BENCHFLOW_BFX_DELIVERY", "off")


BACKENDS = ["docker", "remote-docker", "dind"]


def daemon_reply(monkeypatch, reply):
    """Stub only daemon version transport; retain the production planner."""
    calls = []
    original = _DaytonaDinD._vm_exec

    async def local(self, command, **kwargs):
        assert command[0] == "docker"
        assert command[-3:] == ["version", "--format", "{{.Server.Version}}"]
        assert kwargs["timeout_sec"] == 30
        assert kwargs["separate_stderr"]
        assert command[1:-3] == self._docker_compose_global_args()
        calls.append(command)
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def dind(self, command, **kwargs):
        if command.startswith("docker version "):
            assert kwargs["env"] == self._compose_env_vars()
            assert kwargs["timeout_sec"] == 30
            calls.append(command)
            if isinstance(reply, Exception):
                raise reply
            return reply
        return await original(self, command, **kwargs)

    monkeypatch.setattr(d.DockerSandbox, "_run_docker_process", local)
    monkeypatch.setattr(_DaytonaDinD, "_vm_exec", dind)
    return calls


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "version", ["28.0.0", "28.3.3", "28.3.3+vendor.1", "28.5.1-ce"]
)
@pytest.mark.parametrize(
    "target", ["/testbed/verify/base", "/" + "x" * 250, "/" + "é" * 80]
)
@pytest.mark.parametrize("location", ["main", "worker", "grader"])
@pytest.mark.parametrize("alias", ["task:v1", "docker.io/library/task:v1", "extra:v1"])
async def test_docker28_accepts_long_destinations(
    tmp_path, monkeypatch, backend, version, target, location, alias
):
    calls = daemon_reply(
        monkeypatch, ExecResult(stdout=version + "\n", stderr="warning", return_code=0)
    )
    document = document_for("worker", "task:v1", "task:v1", target)
    mounts = document["services"]["grader"].pop("volumes")
    mounts[0]["source"] = alias
    document["services"][location]["volumes"] = mounts
    result, env = await prepare(
        tmp_path, monkeypatch, backend, document, True, "trial-" + "x" * 200
    )
    mount = result[location]["volumes"][0]
    assert mount["target"] == target
    assert mount["read_only"] is True
    assert mount["source"] == result["worker"]["image"]
    assert len(mount["source"].encode()) == 34
    assert len(calls) == 1
    if env:
        assert mount["source"] in env._private_built_image_tags


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("version", ["29.0.0", "29.1.3", "29.1.3+vendor.1"])
@pytest.mark.parametrize(
    "target",
    ["/" + "x" * 18, "/" + "é" * 9, "//baseline/./discarded-long-component/../"],
)
async def test_docker29_accepts_normalized_byte_boundary(
    tmp_path, monkeypatch, backend, version, target
):
    calls = daemon_reply(monkeypatch, ExecResult(stdout=version, return_code=0))
    document = document_for("worker", "task:v1", "task:v1", target)
    result, _ = await prepare(tmp_path, monkeypatch, backend, document, True, "trial")
    assert result["grader"]["volumes"][0]["target"] == target
    assert len(calls) == 1


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "reply,unknown",
    [
        (ExecResult(stdout="29.1.3", return_code=0), False),
        (ExecResult(stdout="28.3.3", return_code=1), True),
        (ExecResult(stdout="", return_code=0), True),
        (ExecResult(stdout=None, return_code=0), True),
        (ExecResult(stdout='{"Client":"28.3.3"}', return_code=0), True),
        (ExecResult(stdout="28.3.3\n29.1.3", return_code=0), True),
        (ExecResult(stdout="30.0.0", return_code=0), True),
        (ExecResult(stdout="28.bad", return_code=0), True),
        (RuntimeError("unreachable"), True),
        (OSError("no client"), True),
        (TimeoutError(), True),
        (Exception("SDK transport failure"), True),
    ],
)
async def test_strict_bound_and_unknown_version_explanation(
    tmp_path, monkeypatch, backend, reply, unknown
):
    calls = daemon_reply(monkeypatch, reply)
    document = document_for("worker", "task:v1", "task:v1", "/testbed/verify/base")
    with pytest.raises(RuntimeError, match="Could not isolate Compose build") as caught:
        await prepare(tmp_path, monkeypatch, backend, document, True, "trial")
    assert len(calls) == 1
    text = str(caught.value)
    assert "255-byte" in text
    assert ("daemon version unavailable or unrecognized" in text) is unknown
    if unknown:
        assert "stricter Docker 29" in text


@pytest.mark.parametrize("backend", BACKENDS)
async def test_without_image_mounts_does_not_probe(tmp_path, monkeypatch, backend):
    calls = daemon_reply(monkeypatch, RuntimeError("should never query"))
    document = document_for("worker", "task:v1", "task:v1", "/baseline")
    document["services"]["grader"].pop("volumes")
    await prepare(tmp_path, monkeypatch, backend, document, True, "trial")
    assert calls == []


@pytest.mark.parametrize("backend", BACKENDS)
async def test_unknown_daemon_preserves_short_mounts(tmp_path, monkeypatch, backend):
    calls = daemon_reply(monkeypatch, OSError("unavailable"))
    document = document_for("worker", "task:v1", "task:v1", "/baseline")
    result, _ = await prepare(tmp_path, monkeypatch, backend, document, True, "trial")
    assert result["grader"]["volumes"][0]["source"] == result["worker"]["image"]
    assert len(calls) == 1


@pytest.mark.parametrize("backend", BACKENDS)
async def test_daemon_version_is_not_cached_across_trials(
    tmp_path, monkeypatch, backend
):
    document = document_for("worker", "task:v1", "task:v1", "/testbed/verify/base")
    for version in ["28.3.3", "29.1.3", "28.3.3+vendor"]:
        calls = daemon_reply(monkeypatch, ExecResult(stdout=version, return_code=0))
        if version.startswith("29."):
            with pytest.raises(RuntimeError, match="255-byte"):
                await prepare(
                    tmp_path / version, monkeypatch, backend, document, True, "trial"
                )
        else:
            await prepare(
                tmp_path / version, monkeypatch, backend, document, True, "trial"
            )
        assert len(calls) == 1
