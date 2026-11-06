"""The verifier's trusted PATH: judged in /bin/sh, root's folders kept, the
folders left out named (#1174, PLAN P5, Rust 0.8.2's P1.2c).

The check ran as ``python3 -c``, so on an image without Python (CRUSTBench's
Rust image, gson's Java image) it failed, was logged at debug level, and the
verifier ran with the system folders only: ``cargo: command not found``,
``mvnd: command not found``, and every such task unscored whatever the agent
did. It now runs in /bin/sh. Root's folders that group and others cannot
write are kept as before; another user's folder is kept for a non-root agent
user only when that user could never have changed it (another user's
``~/.cargo/bin``); every folder left out is logged with its reason, recorded
in ``verifier/trusted-path.json`` and, when the verifier could not find a
program, named in its error.

The container tests build a small image on the machine's Docker (skipped
without one): an Alpine image, which has no Python, and a Debian one, whose
/bin/sh is dash.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow._utils.scoring import NO_USABLE_REWARD
from benchflow._utils.trusted_path import PATH_QUERY_CMD
from benchflow.rollout._setup import _verify_rollout
from benchflow.sandbox import lockdown
from benchflow.sandbox.lockdown import (
    _SAFE_VERIFIER_PATH,
    _trusted_verifier_path,
)
from tests._receipt_fakes import receipt_aware

# The image's PATH: one folder of each kind, then the system folders.
IMAGE_PATH = ":".join(
    [
        "/home/sandbox/.cargo/bin",  # another user's, closed: kept for an agent user
        "/opt/root-tool/bin",  # root's, closed: kept
        "/opt/open/bin",  # root's, world-writable: out
        "/opt/mine/bin",  # the agent user's: out
        "/opt/held/bin",  # another user's, holding a file of the agent user's: out
        "/opt/groupw/bin",  # another user's, writable by the agent user's group: out
        "/opt/linkout/bin",  # another user's, with a link into the agent's home: out
        "/tmp/tools",  # a temp folder: out
        "/app/bin",  # the workspace: out
        "/nonexistent",  # no folder yet: excluded, including any late uploads
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    ]
)

DOCKERFILE = f"""\
FROM {{base}}
RUN (adduser -D -u 1000 sandbox 2>/dev/null || useradd -m -u 1000 sandbox) \\
 && mkdir -p /home/sandbox/.cargo/bin /opt/root-tool/bin /opt/open/bin \\
      /opt/mine/bin /opt/held/bin /opt/groupw/bin /opt/linkout/bin /tmp/tools /app/bin \\
 && printf '#!/bin/sh\\necho tool\\n' > /home/sandbox/.cargo/bin/uv \\
 && chmod 755 /home/sandbox /home/sandbox/.cargo /home/sandbox/.cargo/bin \\
      /home/sandbox/.cargo/bin/uv \\
 && chown -R sandbox:sandbox /home/sandbox/.cargo /opt/held /opt/groupw /opt/linkout \\
 && chmod 777 /opt/open/bin \\
 && touch /opt/held/bin/planted && ln -s /home/agent/evil /opt/linkout/bin/tool
ENV PATH={IMAGE_PATH}
"""

BASES = {
    # No Python; busybox sh, ls and id.
    "alpine-no-python": "alpine:3.20",
    # dash as /bin/sh, GNU coreutils.
    "debian-dash": "python:3.12-slim",
}


def _docker_ready() -> bool:
    try:
        probe = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
    except Exception:
        return False
    return probe.returncode == 0


def _dx(name: str, command: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "exec", "-u", "root", name, "sh", "-c", command],
        capture_output=True,
        text=True,
        timeout=120,
    )


class ContainerEnv:
    """``env.exec`` on a running container, as root."""

    # PT's PATH lock (setup_sandbox_user) runs only on a sandbox that starts
    # root's shell by absolute path; this one does (/bin/sh below).
    supports_system_shell = True

    def __init__(self, name: str) -> None:
        self.name = name
        self.commands: list[str] = []

    async def exec(self, command: str, user: str | None = None, **kwargs):
        self.commands.append(command)
        done = subprocess.run(
            [
                "docker",
                "exec",
                "-u",
                user or "root",
                self.name,
                "/bin/sh",
                "-c",
                command,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        return SimpleNamespace(
            return_code=done.returncode, stdout=done.stdout, stderr=done.stderr
        )


@pytest.fixture(params=sorted(BASES), ids=sorted(BASES))
def box(request):
    if shutil.which("docker") is None or not _docker_ready():
        pytest.skip("docker required")
    tag = f"bf-v3-path-{uuid.uuid4().hex[:8]}"
    build = subprocess.run(
        ["docker", "build", "-q", "-t", tag, "-"],
        input=DOCKERFILE.format(base=BASES[request.param]),
        capture_output=True,
        text=True,
        timeout=600,
    )
    if build.returncode != 0:
        pytest.skip(f"could not build the fixture image: {build.stderr[-300:]}")
    name = f"{tag}-c"
    run = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--network",
            "none",
            "--name",
            name,
            tag,
            "sleep",
            "600",
        ],
        capture_output=True,
        text=True,
    )
    try:
        if run.returncode != 0:
            pytest.skip(f"could not start the container: {run.stderr[:200]}")
        yield SimpleNamespace(name=name, base=request.param)
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)


async def _with_agent(box) -> ContainerEnv:
    """The container after BenchFlow made its agent user, with the folders
    that depend on that user's ids."""
    env = ContainerEnv(box.name)
    # Only the user (V3's uid rule): PT's setup_sandbox_user would also lock
    # these deliberately open PATH folders to root before the agent (C8),
    # and this file tests the judge on them (PT's lock has its own tests).
    made = _dx(box.name, lockdown._create_sandbox_user_cmd("agent"))
    assert made.returncode == 0, made.stderr
    setup = _dx(
        box.name,
        "chown agent /opt/mine/bin && chown agent /opt/held/bin/planted && "
        "chgrp agent /opt/groupw/bin && chmod 775 /opt/groupw/bin",
    )
    assert setup.returncode == 0, setup.stderr
    return env


@pytest.mark.asyncio
async def test_the_path_check_needs_no_python_and_keeps_closed_folders(box):
    """Guards #1174: root's closed tool folder stays on the verifier's PATH on
    an image without Python, and another user's folder the agent user could
    never change is kept for that agent user (P1.2c's stand-in)."""
    env = await _with_agent(box)
    if box.base == "alpine-no-python":
        assert _dx(box.name, "command -v python3").returncode != 0

    path = await _trusted_verifier_path(env, "agent", "/app")

    assert path == (
        "/home/sandbox/.cargo/bin:/opt/root-tool/bin:" + _SAFE_VERIFIER_PATH
    )
    assert not any("python3" in command for command in env.commands)


@pytest.mark.asyncio
async def test_a_root_agent_keeps_root_owned_folders_only(box):
    """With no sandbox user (the agent ran as root) only root's closed folders
    are kept: root could have changed anything else."""
    env = ContainerEnv(box.name)

    path = await _trusted_verifier_path(env, None, "/app")

    # No agent user here, so /opt/mine/bin is still root's.
    assert path == "/opt/root-tool/bin:/opt/mine/bin:" + _SAFE_VERIFIER_PATH


@pytest.mark.asyncio
async def test_each_folder_left_out_is_named_with_its_reason(box):
    from benchflow._utils.trusted_path import recording_trusted_path

    env = await _with_agent(box)
    with recording_trusted_path() as reports:
        await _trusted_verifier_path(env, "agent", "/app")

    (report,) = reports
    assert report.kept == ("/home/sandbox/.cargo/bin", "/opt/root-tool/bin")
    assert report.kept_other == ("/home/sandbox/.cargo/bin",)
    why = dict(report.dropped)
    assert set(why) == {
        "/opt/open/bin",
        "/opt/mine/bin",
        "/opt/held/bin",
        "/opt/groupw/bin",
        "/opt/linkout/bin",
        "/tmp/tools",
        "/app/bin",
        "/nonexistent",
    }
    assert why["/opt/open/bin"] == "root's, but writable by group or others"
    assert why["/opt/mine/bin"].endswith("/opt/mine/bin is the agent user's")
    assert why["/opt/held/bin"].endswith("it holds an entry the agent user owns")
    assert why["/opt/groupw/bin"].endswith(
        "/opt/groupw/bin is writable by a group of the agent user"
    )
    assert (
        "its link tool leads where the agent user can change" in why["/opt/linkout/bin"]
    )
    assert why["/tmp/tools"] == "under /tmp"
    assert why["/app/bin"] == "under /app"
    assert why["/nonexistent"] == "not present when the verifier PATH was checked"


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/tmp/tools", "/tmp"),
        ("/tmp", "/tmp"),
        ("/app/bin", "/app"),
        ("/app/x/y", "/app/x"),
        ("/home/a b/bin", "/home/a b"),
        ("/tmpx/bin", None),
        ("/opt/bin", None),
    ],
)
def test_the_exclusion_check_names_the_prefix_that_holds_the_path(path, expected):
    """A plain prefix list names the left-out folder holding the path (the
    longest), as the captured exclusions do: "/tmp/tools: under /tmp", never
    "under /tmp/tools", which repeats the path and says nothing."""
    predicate = lockdown._exclusion_predicate(("/tmp/", "/app", "/app/x", "/home/a b"))
    done = subprocess.run(
        ["/bin/sh", "-c", predicate + '\nbf_under "$1"', "sh", path],
        capture_output=True,
        text=True,
    )
    if expected is None:
        assert (done.returncode, done.stdout) == (1, "")
    else:
        assert (done.returncode, done.stdout) == (0, expected + "\n")


def test_a_root_prefix_names_the_root():
    predicate = lockdown._exclusion_predicate(("/",))
    done = subprocess.run(
        ["/bin/sh", "-c", predicate + '\nbf_under "$1"', "sh", "/usr/bin"],
        capture_output=True,
        text=True,
    )
    assert (done.returncode, done.stdout) == (0, "/\n")
    assert lockdown._exclusion_predicate(()) == "bf_under() { return 1; }"


@pytest.mark.asyncio
async def test_an_unreadable_agent_uid_keeps_root_owned_folders_only(box):
    """A sandbox user ``id`` cannot read gets root's folders only."""
    from benchflow._utils.trusted_path import recording_trusted_path

    env = ContainerEnv(box.name)
    with recording_trusted_path() as reports:
        path = await _trusted_verifier_path(env, "nosuchuser", "/app")

    assert path == "/opt/root-tool/bin:/opt/mine/bin:" + _SAFE_VERIFIER_PATH
    assert dict(reports[0].dropped)["/home/sandbox/.cargo/bin"] == (
        "not root's (uid 1000), and the agent user's uid could not be read"
    )


# The record and the error, through _verify_rollout with a scripted sandbox.

PROGRAM_OUTPUT = (
    "K /opt/root-tool/bin\n"
    "D /opt/vendor/bin\n"
    "W not root's (uid 1001), and /opt/vendor/bin is writable by others\n"
)


class _MissingProgramVerifier:
    """A verifier whose test.sh exited 127 without a reward."""

    execution_receipt = None

    def __init__(self, return_code: int, stdout: str, paths) -> None:
        self.test_return_code = return_code
        self.stdout = stdout
        self.paths = paths

    async def verify(self):
        from benchflow.task.verifier_core import RewardFileNotFoundError

        self.paths.test_stdout_path.write_text(self.stdout)
        raise RewardFileNotFoundError(
            f"verifier exited with rc={self.test_return_code}; no reward file found"
        )


def _scripted_env():
    async def exec_(command, **kwargs):
        if command == PATH_QUERY_CMD:
            stdout = "/opt/root-tool/bin:/opt/vendor/bin:/usr/bin\n"
        elif "bf-trusted-path" in command:
            stdout = PROGRAM_OUTPUT
        else:
            stdout = ""
        return SimpleNamespace(stdout=stdout, stderr="", return_code=0)

    return SimpleNamespace(exec=AsyncMock(side_effect=receipt_aware(exec_)))


def _rollout_paths(tmp_path: Path):
    verifier_dir = tmp_path / "verifier"
    return SimpleNamespace(
        verifier_dir=verifier_dir, test_stdout_path=verifier_dir / "test-stdout.txt"
    )


def _verify_planes(verifier):
    async def harden(env, task, sandbox_user, **kwargs):
        task.config.verifier.env = {
            "PATH": await _trusted_verifier_path(env, sandbox_user, "/app")
        }

    return SimpleNamespace(
        harden_before_verify=harden, verifier=lambda **kwargs: verifier
    )


def _task():
    return SimpleNamespace(
        name="path-task",
        task_dir=None,
        config=SimpleNamespace(
            verifier=SimpleNamespace(timeout_sec=60, reward_range=None, env={})
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("return_code", "stdout"),
    [
        (127, ""),
        (2, "/tests/test.sh: line 4: vendor-tool: command not found\n"),
        (2, "/tests/test.sh: 4: vendor-tool: not found\n"),
    ],
    ids=["exit-127", "bash-command-not-found", "dash-not-found"],
)
async def test_a_verifier_that_cannot_find_a_program_names_the_folders_left_out(
    tmp_path, return_code, stdout, caplog
):
    paths = _rollout_paths(tmp_path)
    verifier = _MissingProgramVerifier(return_code, stdout, paths)

    rewards, error, _ = await _verify_rollout(
        _scripted_env(), _task(), paths, {}, _verify_planes(verifier), "agent", "/app"
    )

    # No reward file: a verifier error that names the folders left out.
    assert rewards is None
    assert error is not None and error.startswith(NO_USABLE_REWARD)
    assert "the verifier's PATH left out these folders" in error
    assert "/opt/vendor/bin" in error
    record = json.loads((paths.verifier_dir / "trusted-path.json").read_text())
    assert record == {
        "path": "/opt/root-tool/bin:" + _SAFE_VERIFIER_PATH,
        "kept_not_root_owned": [],
        "dropped": [
            {
                "folder": "/opt/vendor/bin",
                "why": "not root's (uid 1001), and /opt/vendor/bin is writable "
                "by others",
            }
        ],
        "verifier_missed_a_program": True,
        "refused_programs_run": [],
    }


@pytest.mark.asyncio
async def test_other_verifier_failures_keep_their_message(tmp_path, caplog):
    """A verifier that failed for another reason keeps today's message; the
    record still lists the folders left out."""
    paths = _rollout_paths(tmp_path)
    verifier = _MissingProgramVerifier(1, "AssertionError\n", paths)

    _, error, _ = await _verify_rollout(
        _scripted_env(), _task(), paths, {}, _verify_planes(verifier), "agent", "/app"
    )

    assert error is not None and error.startswith(NO_USABLE_REWARD)
    assert "left out" not in error
    record = json.loads((paths.verifier_dir / "trusted-path.json").read_text())
    assert record["verifier_missed_a_program"] is False
    assert [d["folder"] for d in record["dropped"]] == ["/opt/vendor/bin"]


@pytest.mark.asyncio
async def test_nothing_is_recorded_when_no_folder_was_left_out(tmp_path):
    paths = _rollout_paths(tmp_path)
    verifier = MagicMock(execution_receipt=None)
    verifier.verify = AsyncMock(return_value=SimpleNamespace(rewards={"reward": 1.0}))

    async def exec_(command, **kwargs):
        stdout = "/usr/local/bin:/usr/bin\n" if command == PATH_QUERY_CMD else ""
        return SimpleNamespace(stdout=stdout, stderr="", return_code=0)

    rewards, error, _ = await _verify_rollout(
        SimpleNamespace(exec=AsyncMock(side_effect=receipt_aware(exec_))),
        _task(),
        paths,
        {},
        _verify_planes(verifier),
        "agent",
        "/app",
    )

    assert (rewards, error) == ({"reward": 1.0}, None)
    assert not (paths.verifier_dir / "trusted-path.json").exists()


# The verifier folder is the sandbox's /logs/verifier, which code the tests ran
# could write (V3-R1-B-1): what it left there must not make this host write
# through a link or read without end.


class _PlantingVerifier(_MissingProgramVerifier):
    """A verifier whose tests leave *plant* in the verifier folder."""

    def __init__(self, return_code, stdout, paths, plant) -> None:
        super().__init__(return_code, stdout, paths)
        self.plant = plant

    async def verify(self):
        self.paths.verifier_dir.mkdir(parents=True, exist_ok=True)
        self.plant(self.paths.verifier_dir)
        return await super().verify()


@pytest.mark.asyncio
async def test_a_planted_link_named_like_the_record_is_replaced_not_followed(
    tmp_path,
):
    """Guards V3-R1-B-1: ``trusted-path.json -> <a host file>`` left by the
    tests made the host truncate and overwrite that file."""
    victim = tmp_path / "victim-a.json"
    victim.write_text('{"keep": true}\n')
    paths = _rollout_paths(tmp_path)
    verifier = _PlantingVerifier(
        127,
        "",
        paths,
        lambda folder: (folder / "trusted-path.json").symlink_to(victim),
    )

    await _verify_rollout(
        _scripted_env(), _task(), paths, {}, _verify_planes(verifier), "agent", "/app"
    )

    assert victim.read_text() == '{"keep": true}\n'
    record = paths.verifier_dir / "trusted-path.json"
    assert not record.is_symlink()
    assert json.loads(record.read_text())["verifier_missed_a_program"] is True
    assert sorted(p.name for p in paths.verifier_dir.iterdir()) == [
        "test-stdout.txt",
        "trusted-path.json",
    ]


@pytest.mark.asyncio
async def test_record_staging_is_never_created_in_the_verifier_folder(
    tmp_path, monkeypatch
):
    """Guards V3-R2-A-1: the old reachable tempfile swap is inexpressible
    when the record is staged privately and published through a held folder."""
    from benchflow.rollout import _setup

    victim = tmp_path / "victim-c.env"
    victim.write_text("KEEP=1\n")
    victim.chmod(0o600)
    moved_away = tmp_path / "moved-away.tmp"
    real_mkstemp = tempfile.mkstemp

    def swapping_mkstemp(*args, **kwargs):
        fd, name = real_mkstemp(*args, **kwargs)
        os.rename(name, moved_away)
        os.symlink(victim, name)
        return fd, name

    monkeypatch.setattr(_setup, "tempfile", SimpleNamespace(mkstemp=swapping_mkstemp))
    paths = _rollout_paths(tmp_path)

    await _verify_rollout(
        _scripted_env(),
        _task(),
        paths,
        {},
        _verify_planes(_MissingProgramVerifier(127, "", paths)),
        "agent",
        "/app",
    )

    assert stat.S_IMODE(victim.stat().st_mode) == 0o600
    assert victim.read_text() == "KEEP=1\n"
    assert not moved_away.exists(), "record staging must not use a reachable tempfile"
    record = paths.verifier_dir / "trusted-path.json"
    assert not record.is_symlink()
    assert json.loads(record.read_text())["verifier_missed_a_program"] is True
    assert sorted(p.name for p in paths.verifier_dir.iterdir()) == [
        "test-stdout.txt",
        "trusted-path.json",
    ]


@pytest.mark.asyncio
async def test_a_test_stdout_link_is_not_followed(tmp_path, caplog):
    """Guards V3-R1-B-1: test-stdout.txt as a link is not read through, so a
    file outside the verifier folder cannot decide the error either."""
    elsewhere = tmp_path / "victim-b.txt"
    elsewhere.write_text("/tests/test.sh: line 4: vendor-tool: command not found\n")
    paths = _rollout_paths(tmp_path)

    class _LinkedStdout(_MissingProgramVerifier):
        async def verify(self):
            self.paths.verifier_dir.mkdir(parents=True, exist_ok=True)
            self.paths.test_stdout_path.symlink_to(elsewhere)
            from benchflow.task.verifier_core import RewardFileNotFoundError

            raise RewardFileNotFoundError("verifier exited with rc=2")

    _, error, _ = await _verify_rollout(
        _scripted_env(),
        _task(),
        paths,
        {},
        _verify_planes(_LinkedStdout(2, "", paths)),
        "agent",
        "/app",
    )

    assert error is not None and error.startswith(NO_USABLE_REWARD)
    assert "left out" not in error
    record = json.loads((paths.verifier_dir / "trusted-path.json").read_text())
    assert record["verifier_missed_a_program"] is False


_SCAN_IN_A_CHILD = """
import resource, sys
from pathlib import Path
from types import SimpleNamespace
from benchflow.rollout._setup import _verifier_missed_a_program
# What the process has mapped now, plus 512 MiB: the 4 MiB scan fits, an
# endless read does not.
size = next(l for l in open("/proc/self/status") if l.startswith("VmSize:"))
limit = int(size.split()[1]) * 1024 + (512 << 20)
resource.setrlimit(resource.RLIMIT_AS, (limit, resource.RLIM_INFINITY))
paths = SimpleNamespace(test_stdout_path=Path(sys.argv[1]))
print(_verifier_missed_a_program(SimpleNamespace(test_return_code=2), paths))
"""


@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS is enforced on Linux")
@pytest.mark.parametrize("kind", ["link-to-dev-zero", "fifo"])
def test_reading_test_stdout_ends_whatever_the_tests_left(tmp_path, kind):
    """Guards V3-R1-B-1: a link to /dev/zero grew host memory without end and
    a FIFO blocked the host at open. The scan runs in a child process with an
    address-space limit and a time limit, so a regression fails the test
    instead of the machine."""
    folder = tmp_path / "verifier"
    folder.mkdir()
    stdout_path = folder / "test-stdout.txt"
    if kind == "fifo":
        os.mkfifo(stdout_path)
    else:
        stdout_path.symlink_to("/dev/zero")

    done = subprocess.run(
        [sys.executable, "-c", _SCAN_IN_A_CHILD, str(stdout_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip() == "False"


def test_only_the_last_part_of_a_large_test_stdout_is_read(tmp_path, monkeypatch):
    """The scan reads at most _TEST_STDOUT_SCAN_BYTES, from the end."""
    from benchflow.rollout import _setup

    monkeypatch.setattr(_setup, "_TEST_STDOUT_SCAN_BYTES", 64)
    stdout_path = tmp_path / "test-stdout.txt"
    paths = SimpleNamespace(test_stdout_path=stdout_path)
    verifier = SimpleNamespace(test_return_code=2)

    stdout_path.write_text("x: command not found\n" + "y" * 200 + "\n")
    assert _setup._verifier_missed_a_program(verifier, paths) is False
    stdout_path.write_text("y" * 200 + "\nx: command not found\n")
    assert _setup._verifier_missed_a_program(verifier, paths) is True


@pytest.mark.asyncio
async def test_separate_image_path_keeps_nonroot_tools_and_blocks_transfers(box):
    """Fresh image tools stay; transferred paths stay out."""
    env = ContainerEnv(box.name)
    command = lockdown._trusted_path_extras_cmd(
        IMAGE_PATH, ("/app", "/tmp", "/home/agent"), fresh_image=True
    )
    done = await env.exec(command)
    assert done.return_code == 0, done.stderr
    kept, _, dropped = lockdown._parse_trusted_path_output(
        done.stdout, lockdown._path_entries_to_judge(IMAGE_PATH)
    )
    assert "/home/sandbox/.cargo/bin" in kept
    assert "/app/bin" not in kept
    assert "/tmp/tools" not in kept
    assert "/opt/linkout/bin" not in kept
    assert any(path == "/app/bin" for path, _ in dropped)


@pytest.mark.asyncio
async def test_fresh_image_path_can_reserve_missing_directories(box):
    """Absent image paths are safe, redirects are not."""
    env = ContainerEnv(box.name)
    setup = await env.exec(
        "mkdir -p /app/data/toolchain /outside; "
        "ln -s /app/output /outside/alias; "
        "printf '#!/bin/sh\\nexit 0\\n' > /app/data/toolchain/tool; "
        "chmod 755 /app/data/toolchain/tool"
    )
    assert setup.return_code == 0, setup.stderr
    entries = (
        "/app/data/toolchain:/home/coq/.local/bin:/app/output/bin:/outside/alias/bin"
    )
    command = lockdown._trusted_path_extras_cmd(
        entries, ("/app/output", "/logs"), fresh_image=True
    )
    done = await env.exec(command)
    assert done.return_code == 0, done.stderr
    kept, _, dropped = lockdown._parse_trusted_path_output(
        done.stdout, lockdown._path_entries_to_judge(entries)
    )
    assert kept == ["/app/data/toolchain", "/home/coq/.local/bin"]
    assert {p for p, _ in dropped} == {"/app/output/bin", "/outside/alias/bin"}
