"""Guards the fixes for R4-1..3 reviewed at 5d0133a2."""

import logging
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.sandbox import lockdown
from benchflow.sandbox._base import ExecResult
from tests._receipt_fakes import receipt_aware


class Shell:
    # Root's PATH as the exec route pins it (C8): a caller's env never sets it.
    path = "/usr/local/bin:/usr/bin:/bin"

    @receipt_aware
    async def exec(self, command, env=None, **kwargs):
        assert "PATH" not in (env or {}), "a root helper set PATH itself"
        result = subprocess.run(
            ["/bin/sh", "-c", command],
            env={**(env or {}), "PATH": self.path},
            text=True,
            capture_output=True,
            timeout=30,
        )
        return ExecResult(
            stdout=result.stdout, stderr=result.stderr, return_code=result.returncode
        )


def task(value, service="main"):
    return SimpleNamespace(
        config=SimpleNamespace(
            verifier=SimpleNamespace(env={"BASH_ENV": value}, service=service)
        )
    )


def inspectors(folder: Path, *, python_only=False):
    folder.mkdir(mode=0o700)
    for name in ("python3",) if python_only else ("stat", "readlink"):
        source = shutil.which(name)
        assert source
        shutil.copy2(Path(source).resolve(), folder / name)
        (folder / name).chmod(0o700)


@pytest.mark.asyncio
@pytest.mark.parametrize("candidate", ["stat", "ls", "readlink", "python3"])
async def test_no_candidate_runs_before_inspection(tmp_path, monkeypatch, candidate):
    """R4-1: a single mutable candidate must never execute, even to inspect itself."""
    live = tmp_path / "live"
    live.mkdir()
    marker = tmp_path / "candidate-ran"
    python_only = candidate == "python3"
    names = ("python3",) if python_only else ("stat", "ls", "readlink")
    for name in names:
        source = shutil.which(name)
        assert source
        (live / name).symlink_to(source)
    (live / "sh").symlink_to("/bin/sh")
    (live / candidate).unlink()
    (live / candidate).write_text(
        "#!/bin/sh\n"
        f"printf ran > {shlex.quote(str(marker))}\n"
        f'exec {shlex.quote(shutil.which(candidate) or "")} "$@"\n'
    )
    (live / candidate).chmod(0o777)
    saved = tmp_path / "saved"
    inspectors(saved, python_only=python_only)
    env = Shell()
    env._benchflow_bash_env_inspector = str(saved)
    env.path = str(live)

    assert await lockdown._trusted_verifier_bash_env(env, task("/etc/passwd"), ()) == ""
    assert not marker.exists(), "the checker executed a mutable candidate"


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["/etc/passwd\n", ""])
async def test_missing_preagent_inspector_never_runs_live_tools(answer, caplog):
    """R4-1: missing setup/restore provenance refuses before any candidate query."""

    class NoInspector:
        calls = 0

        @receipt_aware
        async def exec(self, *args, **kwargs):
            self.calls += 1
            return ExecResult(stdout=answer, stderr="", return_code=0)

    env = NoInspector()
    with caplog.at_level(logging.WARNING):
        assert (
            await lockdown._trusted_verifier_bash_env(env, task("/etc/passwd"), ())
            == ""
        )
    assert env.calls == 0
    assert "pre-agent inspector" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["task", "image"])
@pytest.mark.parametrize("blocked", [("/app",), ("/transferred",)])
async def test_root_owned_shared_alias_remains_untrusted(source, blocked, caplog):
    """R4-2: freezing a workspace/transfer alias cannot erase its provenance."""
    value = "/opt/activation/start.sh"

    class SharedAlias:
        _benchflow_bash_env_inspector = "/etc/dummy-inspector"

        def __init__(self):
            self.calls = []

        @receipt_aware
        async def exec(self, command, **kwargs):
            self.calls.append((command, kwargs))
            # The image probe and genuine post-freeze ownership query both
            # answer the alias: the bytes written via /app are now root's.
            return ExecResult(stdout=value + "\n", stderr="", return_code=0)

    env = SharedAlias()
    config = task(value, "target")
    if source == "image":
        config.config.verifier.env = {}
    with caplog.at_level(logging.WARNING):
        assert await lockdown._trusted_verifier_bash_env(env, config, blocked) == ""
    assert "shared-mount provenance" in caplog.text
    assert len(env.calls) == (1 if source == "image" else 0)
    assert all(kwargs["service"] == "target" for _, kwargs in env.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("service", ["main", "target"])
@pytest.mark.parametrize("stdout", ["", "/etc/profile.d/activate.sh\n"])
@pytest.mark.parametrize("status", [1, 9])
async def test_failed_image_probe_warns_and_ignores_partial_output(
    service, stdout, status, caplog
):
    """R4-3: an unsuccessful ExecResult is an observed failure, including truncation."""

    class Truncated:
        @receipt_aware
        async def exec(self, command, **kwargs):
            assert kwargs.get("service", "main") == service
            return ExecResult(
                stdout=stdout, stderr="dummy truncated stream", return_code=status
            )

    with caplog.at_level(logging.WARNING):
        assert await lockdown._image_bash_env(Truncated(), service) == ""
    assert service in caplog.text
    assert f"status {status}" in caplog.text
    assert "dummy truncated stream" in caplog.text


@pytest.mark.asyncio
async def test_successful_empty_image_probe_is_quiet(caplog):
    """R4-3 positive control: an absent value from a successful probe is ordinary."""

    class Absent:
        @receipt_aware
        async def exec(self, *args, **kwargs):
            return ExecResult(stdout="", stderr="", return_code=0)

    assert await lockdown._image_bash_env(Absent()) == ""
    assert not caplog.records


@pytest.mark.asyncio
@pytest.mark.parametrize("service", ["main", "target"])
async def test_image_value_probe_never_executes_a_path_utility(tmp_path, service):
    """R4-1 sibling: reading BASH_ENV must not execute an agent's printenv."""
    marker = tmp_path / "printenv-ran"
    (tmp_path / "printenv").write_text(
        f"#!/bin/sh\nprintf ran > {shlex.quote(str(marker))}\n"
        "printf '/etc/profile.d/safe.sh\\n'\n"
    )
    (tmp_path / "printenv").chmod(0o755)

    class Image:
        @receipt_aware
        async def exec(self, command, **kwargs):
            assert kwargs.get("service", "main") == service
            result = subprocess.run(
                ["/bin/sh", "-c", command],
                env={
                    **os.environ,
                    "PATH": str(tmp_path),
                    "BASH_ENV": "/etc/profile.d/safe.sh",
                },
                capture_output=True,
                text=True,
                timeout=30,
            )
            return ExecResult(
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
            )

    assert await lockdown._image_bash_env(Image(), service) == "/etc/profile.d/safe.sh"
    assert not marker.exists()


@pytest.mark.asyncio
async def test_preagent_inspector_records_only_completed_setup(monkeypatch, tmp_path):
    """R4-1: capture bytes before solving, and fail closed on partial setup."""
    monkeypatch.setattr(lockdown, "_BASH_ENV_INSPECTOR_ROOT", str(tmp_path))

    class Setup:
        _benchflow_bash_env_inspector = "/etc/stale-inspector"

        @receipt_aware
        async def exec(self, command, **kwargs):
            assert "cp -L" in command and "chmod 700" in command
            # BASH_ENV, ENV and PATH are the exec route's alone (C8).
            assert "BASH_ENV" not in (kwargs.get("env") or {})
            assert kwargs["user"] == "root"
            return ExecResult(
                stdout="ready\n", stderr="dummy setup failure", return_code=1
            )

    env = Setup()
    await lockdown._prepare_bash_env_inspector(env)
    assert env._benchflow_bash_env_inspector is None


@pytest.mark.asyncio
async def test_restored_checkpoint_never_recaptures_inspectors():
    """R4-1: a missing baseline in a restored checkpoint cannot bless agent bytes."""

    class Restored:
        _benchflow_bash_env_capture_allowed = False
        _benchflow_bash_env_inspector = "/etc/stale-inspector"

        @receipt_aware
        async def exec(self, *args, **kwargs):
            raise AssertionError("checkpoint utilities must not execute")

    env = Restored()
    await lockdown._prepare_bash_env_inspector(env)
    assert env._benchflow_bash_env_inspector is None


@pytest.mark.asyncio
@pytest.mark.parametrize("restore", [False, True])
@pytest.mark.parametrize("permitted", ["", "/etc/profile.d/safe.sh"])
async def test_activation_is_decided_before_ownership_changes(
    tmp_path, monkeypatch, restore, permitted
):
    """R4-2: main's second workspace mount also retains pre-freeze provenance."""
    from tests.test_verifier_script_shell import _task, _without_image_probes

    config = _task(tmp_path, "echo 0 > /logs/verifier/reward.txt\n")
    config.config.verifier.env = {"BASH_ENV": "/opt/activation/start.sh"}
    _without_image_probes(monkeypatch, SimpleNamespace(bin="/usr/bin"))
    changed = False
    observations = []

    async def inspect(*args):
        observations.append(changed)
        return "/etc/profile.d/frozen-alias.sh" if changed else permitted

    async def mutate(*args):
        nonlocal changed
        changed = True

    class Sandbox:
        @receipt_aware
        async def exec(self, command, **kwargs):
            from tests.sandbox_mounts import ROOT_MOUNTINFO

            return ExecResult(
                stdout=ROOT_MOUNTINFO if "/proc/self/mountinfo" in command else "",
                stderr="",
                return_code=0,
            )

    monkeypatch.setattr(lockdown, "_trusted_verifier_bash_env", inspect)
    monkeypatch.setattr(lockdown, "_restore_workspace_state", mutate)
    monkeypatch.setattr(lockdown, "_freeze_workspace", mutate)
    await lockdown.harden_before_verify(
        Sandbox(), config, None, "/app", restore_workspace=restore
    )
    assert config.config.verifier.env["BASH_ENV"] == permitted
    assert observations == [False]
