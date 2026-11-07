"""Guard CONFIRM-taskmd-rewards-R2-2 against reviewed commit 037efeed."""

from types import SimpleNamespace

import pytest

from benchflow.taskmd import verify


@pytest.mark.parametrize("capability", [False, None, 1, "true", "missing"])
@pytest.mark.parametrize(
    "script",
    [
        "pwd -P",
        verify._probe_script("/app"),
        verify._copy_script("/app"),
        f"rm -rf {verify.COPY_DIR}",
        verify._offline_script(),
    ],
    ids=["workdir", "probe", "archive", "cleanup", "offline"],
)
async def test_root_exec_refuses_every_unprotected_shell(capability, script):
    """037efeed: no startup/loader/PATH spelling matters when launch is refused."""
    calls = []

    async def execute(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout="", return_code=0)

    sandbox = SimpleNamespace(exec=execute)
    if capability != "missing":
        sandbox.supports_system_shell = capability
    with pytest.raises(verify.TaskMdVerifierError, match=r"protected.*shell"):
        await verify.root_exec(sandbox, verify._as_root(script), timeout_sec=60)
    assert calls == []


async def test_protected_root_exec_requests_backend_contract():
    """037efeed: capable backends keep the absolute-shell, pre-launch sanitization route."""
    calls = []

    async def execute(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout="safe", return_code=0)

    result = await verify.root_exec(
        SimpleNamespace(exec=execute, supports_system_shell=True),
        verify._as_root("pwd -P"),
        timeout_sec=60,
        env={"ENV": ""},
    )
    assert result.stdout == "safe"
    assert calls[0][1] == {
        "user": "root",
        "timeout_sec": 60,
        "system_shell": True,
        "env": {"ENV": ""},
    }


@pytest.mark.parametrize(
    "module,name",
    [
        ("docker", "DockerSandbox"),
        ("remote_docker", "RemoteDockerSandbox"),
        ("daytona", "DaytonaSandbox"),
        ("apple_container", "AppleContainerSandbox"),
        ("modal_impl", "ModalSandbox"),
        ("agentcore", "AgentCoreSandbox"),
    ],
)
async def test_every_backend_obeys_capture_shell_capability(
    module, name, monkeypatch, tmp_path
):
    """037efeed: actual backend capability declarations govern privileged capture."""
    from importlib import import_module

    cls = getattr(import_module(f"benchflow.sandbox.{module}"), name)
    sandbox = cls.__new__(cls)
    calls = []

    async def execute(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(stdout="none", return_code=0)

    monkeypatch.setattr(sandbox, "exec", execute)
    if sandbox.supports_system_shell:
        await verify.root_exec(sandbox, verify._as_root("pwd -P"), timeout_sec=60)
        assert calls == [{"user": "root", "timeout_sec": 60, "system_shell": True}]
    else:
        with pytest.raises(verify.TaskMdVerifierError, match=r"protected.*shell"):
            await verify.copy_kept(sandbox, ["/app"], tmp_path)
        assert calls == []
