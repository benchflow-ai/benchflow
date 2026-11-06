"""Behavioral guards for PY8 AG-R2-1..4 at 03e8bb89; fake credentials only.

The credential cases run their sandbox commands in disposable Docker
containers with no mounts and no network, as the agent user or root exactly as
production asks; the installer cases run the install snippet against a fake
download in a temporary folder.
"""

from __future__ import annotations

import hashlib
import io
import os
import shlex
import shutil
import subprocess
import tarfile
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.agents import antigravity_config, credentials, muse_config
from benchflow.sandbox import lockdown
from benchflow.sandbox.protocol import ExecResult
from tests.test_ag_credential_regressions import LOGIN, _rollout

BASE = "alpine@sha256:d9e853e87e55526f6b2917df91a2115c36dd7c696a35be12163d44e6e2a4b6bc"
LOGIN_PATH = "/home/agent/.config/muse/auth.json"


def _run(args, *, input=None):
    return subprocess.run(
        args, input=input, capture_output=True, text=True, timeout=300
    )


class _Box:
    """A container whose exec runs as the user production names."""

    def __init__(self, name):
        self.name = name

    async def exec(self, command, user=None, timeout_sec=None, **kwargs):
        done = _run(
            [
                "docker",
                "exec",
                "-u",
                "0" if user in (None, "root") else str(user),
                self.name,
                "/bin/sh",
                "-c",
                command,
            ]
        )
        return ExecResult(done.returncode, done.stdout, done.stderr)

    def sh(self, script, user="0"):
        done = _run(["docker", "exec", "-u", user, self.name, "/bin/sh", "-c", script])
        assert done.returncode == 0, done.stderr
        return done.stdout


@pytest.fixture
def box():
    if shutil.which("docker") is None or _run(["docker", "info"]).returncode:
        pytest.skip("Docker required")
    name = f"bf-ag-r2-{uuid.uuid4().hex[:12]}"
    made = _run(
        [
            "docker",
            "run",
            "-d",
            "--network",
            "none",
            "--name",
            name,
            BASE,
            "sleep",
            "600",
        ]
    )
    assert made.returncode == 0, made.stderr
    try:
        sandbox = _Box(name)
        sandbox.sh("adduser -D -u 1001 agent")
        yield sandbox
    finally:
        _run(["docker", "rm", "-f", name])


def _stage_login(box):
    box.sh(
        "mkdir -p /home/agent/.config/muse && "
        f"printf %s {shlex.quote(LOGIN)} > {LOGIN_PATH}",
        user="agent",
    )


def _login_exists(box):
    return box.sh(f"[ -e {LOGIN_PATH} ] && echo yes || echo no").strip() == "yes"


@pytest.mark.parametrize("mode", ["000", "600"])
async def test_unsearchable_parent_is_not_a_removed_login(box, mode):
    """Guards AG-R2-1 at 03e8bb89: the owner cannot see into a folder it
    cannot search, so the login looked absent and cleanup reported success."""
    _stage_login(box)
    box.sh(f"chmod {mode} /home/agent/.config", user="agent")

    with pytest.raises(RuntimeError, match="credential cleanup failed"):
        await credentials.remove_muse_credentials(box, "agent")
    assert _login_exists(box)


async def test_searchable_layout_still_removes_the_login(box):
    """Control for AG-R2-1: the ordinary layout is removed and confirmed."""
    _stage_login(box)

    await credentials.remove_muse_credentials(box, "agent")

    assert not _login_exists(box)


def _harden_with_real_cleanup(monkeypatch):
    """harden_before_verify with only its Muse clean-up left real."""
    monkeypatch.setattr(lockdown, "_kill_sandbox_user_procs", AsyncMock())
    for name in (
        "clear_verifier_output_dir",
        "ensure_legacy_app_dir",
        "_reclaim_disk",
    ):
        monkeypatch.setattr(lockdown, name, AsyncMock())
    monkeypatch.setattr(lockdown, "_read_hardening_config", lambda *a: {})
    monkeypatch.setattr(lockdown, "_build_verifier_env", AsyncMock(return_value={}))
    return SimpleNamespace(
        task_dir=None,
        config=SimpleNamespace(verifier=SimpleNamespace(env={}, service=None)),
    )


async def test_shared_hardening_refuses_an_unsearchable_muse_login(box, monkeypatch):
    """Guards AG-R2-1 at 03e8bb89 on the shared-verifier path: hardening
    repeated the same false success and admitted the root verifier."""
    _stage_login(box)
    box.sh("chmod 000 /home/agent/.config", user="agent")
    task = _harden_with_real_cleanup(monkeypatch)

    with pytest.raises(RuntimeError, match="credential cleanup failed"):
        await lockdown.harden_before_verify(box, task, "agent")
    assert _login_exists(box)


class _PastTheCleanup(Exception):
    """Hardening got past the credential clean-up to its next step."""


async def test_linked_config_without_a_muse_login_is_verified(box, monkeypatch):
    """Guards AG-R2-4 at 03e8bb89: an OpenCode or agy home whose .config is a
    link (dotfiles) failed verification on Muse's parent-link refusal.

    Hardening is stopped at its first step after the clean-up: the
    steps after it now need the pre-agent PATH inspector a real start makes,
    which this bare container has not, and they are not what this guards."""
    box.sh(
        "mkdir -p /home/agent/dotfiles/config/opencode && "
        "ln -s /home/agent/dotfiles/config /home/agent/.config",
        user="agent",
    )
    task = _harden_with_real_cleanup(monkeypatch)

    def past_the_cleanup(*args, **kwargs):
        raise _PastTheCleanup

    monkeypatch.setattr(lockdown, "_blocked_verifier_path_prefixes", past_the_cleanup)
    with pytest.raises(_PastTheCleanup):
        await lockdown.harden_before_verify(box, task, "agent")

    assert box.sh("readlink /home/agent/.config").strip() == (
        "/home/agent/dotfiles/config"
    )


async def test_linked_config_with_a_muse_login_is_still_refused(box, monkeypatch):
    """Control for AG-R2-4: a real Muse login keeps the strict parent rule."""
    box.sh(
        "mkdir -p /home/agent/dotfiles/config/muse && "
        "ln -s /home/agent/dotfiles/config /home/agent/.config && "
        f"printf %s {shlex.quote(LOGIN)} > /home/agent/dotfiles/config/muse/auth.json",
        user="agent",
    )
    task = _harden_with_real_cleanup(monkeypatch)

    with pytest.raises(RuntimeError, match="credential cleanup failed"):
        await lockdown.harden_before_verify(box, task, "agent")


async def test_cleanup_refusal_still_destroys_the_sandbox(tmp_path):
    """Guards AG-R2-3 at 03e8bb89: a disconnect that cannot remove the login
    aborted cleanup before env.stop(delete=True), leaving sandbox and login."""
    trial = _rollout(tmp_path)
    await trial.connect()
    trial._env.stop = AsyncMock()
    trial._planes.quiesce_agent.side_effect = RuntimeError("fake surviving process")

    with pytest.raises(RuntimeError, match="surviving process"):
        await trial.cleanup()

    trial._env.stop.assert_awaited_once_with(delete=True)
    assert trial._phase == "cleaned"


# The installers' reuse path: a binary already in place.

_TOOLS = (
    *("sh", "grep", "head", "tr", "uname", "mktemp", "mkdir", "mv", "chmod"),
    *("ln", "rm", "tar", "gzip", "cp", "cut", "cat", "sha256sum", "sha512sum"),
)


def _installer(tmp_path, monkeypatch, agent, good):
    """The install command for *agent* with a fake download of *good*."""
    tools = tmp_path / "tools"
    tools.mkdir()
    for command in _TOOLS:
        source = shutil.which(command)
        assert source, command
        (tools / command).symlink_to(source)
    payload = tmp_path / "download"
    if agent == "muse":
        payload.write_bytes(good)
        digest = hashlib.sha256(good).hexdigest()
    else:
        with tarfile.open(payload, "w:gz") as archive:
            entry = tarfile.TarInfo("antigravity")
            entry.size = len(good)
            entry.mode = 0o755
            archive.addfile(entry, io.BytesIO(good))
        digest = hashlib.sha512(payload.read_bytes()).hexdigest()
    curl = tools / "curl"
    curl.write_text(
        "#!/bin/sh\nwhile [ $# -gt 0 ]; do\n"
        'case "$1" in -o|-fsSLo) shift; out=$1;; esac\nshift\ndone\n'
        f"echo downloaded >> {shlex.quote(str(tmp_path / 'downloads'))}\n"
        f'cp {shlex.quote(str(payload))} "$out"\n'
    )
    curl.chmod(0o755)
    target = tmp_path / "installed"
    target.mkdir()
    if agent == "muse":
        monkeypatch.setattr(muse_config, "MUSE_DIR", str(target))
        monkeypatch.setattr(muse_config, "MUSE_BIN", str(target / "muse"))
        monkeypatch.setattr(
            muse_config,
            "MUSE_RELEASES",
            {
                arch: ("https://invalid.example/fake", digest, len(good))
                for arch in ("x86_64", "aarch64")
            },
        )
        command = muse_config.muse_install_cmd("exit 99")
        installed = target / f"muse-bin-{muse_config.MUSE_VERSION}"
        link = target / "muse"
    else:
        monkeypatch.setattr(antigravity_config, "AGY_BIN", str(target / "agy"))
        monkeypatch.setattr(
            antigravity_config,
            "AGY_RELEASES",
            {
                arch: ("https://invalid.example/fake", digest)
                for arch in ("x86_64", "aarch64")
            },
        )
        command = antigravity_config.agy_install_cmd("exit 99")
        installed = link = target / "agy"

    def run():
        return subprocess.run(
            ["/bin/sh", "-c", command],
            env={**os.environ, "PATH": str(tools), "TMPDIR": str(tmp_path)},
            capture_output=True,
            text=True,
        )

    return run, installed, link


def _version(agent):
    return (
        muse_config.MUSE_VERSION if agent == "muse" else antigravity_config.AGY_VERSION
    )


@pytest.mark.parametrize("agent", ["muse", "agy"])
def test_cached_unverified_binary_is_replaced(tmp_path, monkeypatch, agent):
    """Guards AG-R2-2 at 03e8bb89: a binary left by the pre-fix installer
    (wrong bytes that print the pinned version) was kept on reuse."""
    good = f"#!/bin/sh\necho '{_version(agent)}'\n".encode()
    run, installed, link = _installer(tmp_path, monkeypatch, agent, good)
    installed.write_text(f"#!/bin/sh\n# not the release\necho '{_version(agent)}'\n")
    installed.chmod(0o755)
    if link != installed:
        link.symlink_to(installed)

    result = run()

    assert result.returncode == 0, result.stderr
    assert installed.read_bytes() == good
    assert (tmp_path / "downloads").exists()


@pytest.mark.parametrize("agent", ["muse", "agy"])
def test_verified_install_is_reused_without_a_download(tmp_path, monkeypatch, agent):
    """Control for AG-R2-2: a binary this installer verified is kept."""
    good = f"#!/bin/sh\necho '{_version(agent)}'\n".encode()
    run, installed, _ = _installer(tmp_path, monkeypatch, agent, good)
    assert run().returncode == 0
    (tmp_path / "downloads").unlink()

    result = run()

    assert result.returncode == 0, result.stderr
    assert installed.read_bytes() == good
    assert not (tmp_path / "downloads").exists()
