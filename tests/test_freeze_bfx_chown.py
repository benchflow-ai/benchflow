"""The workspace freeze's ownership step on ``bfx chown`` (TRUSTED-RUNTIME v2
§4 hardening row, I5): the tree beneath the workspace goes to root, no link
is followed, no mount entered, and a regular file with another name keeps
its owner. Where bfx is not delivered the step is root's Python walk that
does the same, but enters mounts as ``chown -R`` did, or a ``find`` checked
by a second ``find`` without root Python (ACC-L1)."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import uuid
from types import SimpleNamespace

import pytest

from benchflow.sandbox import lockdown

PLAIN = lockdown._chown_workspace_cmd("/app")


def test_without_bfx_the_step_is_root_python_with_a_checked_find_fallback():
    assert shlex.quote(lockdown._WORKSPACE_OWNER_SCRIPT) in PLAIN
    assert " freeze /app)" in PLAIN
    python, _, fallback = PLAIN.rpartition("; else ")
    assert "__bf_hp_py" in python
    assert fallback.startswith("bf_workspace=$(cd -P /app")
    assert (
        '-exec chown -h root:root {} + || exit 1; bf_left=$(find "$bf_workspace" '
        "! -user 0 \\( ! -type f -o -links 1 \\) -print -quit) || exit 1;"
    ) in fallback
    step = next(
        s for s in lockdown._freeze_steps("/app") if s.name == "chown_workspace"
    )
    assert step.command == PLAIN


def test_with_bfx_the_step_runs_bfx_chown_and_keeps_today_as_fallback():
    command = lockdown._chown_workspace_cmd("/app", bfx=True)
    args = {"root": "/app", "path": "", "uid": 0, "gid": 0, "recursive": True}
    assert (
        "/.bfx/bfx chown --json " + shlex.quote(json.dumps(args, separators=(",", ":")))
        in command
    )
    assert PLAIN in command
    step = next(
        s
        for s in lockdown._freeze_steps("/app", bfx=True)
        if s.name == "chown_workspace"
    )
    assert step.command == command
    assert step.label == "Verifier hardening failed: freezing workspace ownership"


def test_bfx_in_scripts_needs_delivery_and_a_bfx_backend():
    class Docker:
        bfx_delivery = "bind"

        async def bfx_call(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

    assert lockdown._bfx_in_scripts(Docker())
    undelivered = Docker()
    undelivered.bfx_delivery = None
    assert not lockdown._bfx_in_scripts(undelivered)
    assert not lockdown._bfx_in_scripts(SimpleNamespace(bfx_delivery="upload"))


# Real containers: the pinned bfx bound read-only at /.bfx.

_TREE = r"""
set -e
useradd -m agent
mkdir -p /app/src /data
echo 1 > /app/src/a.txt
echo shared > /data/outside.txt
ln /data/outside.txt /app/hardlink.txt
ln -s /data /app/data-link
chown -R agent:agent /app /data
ln -s /app /work
"""


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


@pytest.fixture
def box():
    if not _docker_ready():
        pytest.skip("docker daemon not running")
    from benchflow._bfx import delivery
    from benchflow._bfx.locate import BfxUnavailable

    try:
        mount = delivery.run_args(
            delivery.BIND, source=str(delivery.host_dir("x86_64"))
        )
    except BfxUnavailable as exc:
        pytest.skip(f"no pinned bfx in this checkout: {exc}")
    import os

    name = os.environ.get("BENCHFLOW_TEST_CONTAINER_PREFIX", "trcore-chown-")
    name += uuid.uuid4().hex[:8]
    label = os.environ.get("BENCHFLOW_TEST_CONTAINER_LABEL", "target=TR-CORE-chown")
    key, _, value = label.partition("=")
    run = subprocess.run(
        ["docker", "run", "-d", "--name", name, "--label", f"{key}={value}",
         "--network", "none", *mount, "python:3.12-slim", "sleep", "600"],
        capture_output=True,
        text=True,
    )  # fmt: skip
    if run.returncode != 0:
        pytest.skip(f"cannot start the image: {run.stderr.strip()}")
    try:
        made = subprocess.run(
            ["docker", "exec", name, "/bin/sh", "-c", _TREE],
            capture_output=True,
            text=True,
        )
        assert made.returncode == 0, made.stderr
        yield name
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def _owners(name: str) -> dict[str, str]:
    done = subprocess.run(
        ["docker", "exec", name, "stat", "-c", "%n %U",
         "/app", "/app/src", "/app/src/a.txt", "/app/hardlink.txt",
         "/data/outside.txt", "/data"],
        capture_output=True,
        text=True,
        check=True,
    )  # fmt: skip
    return dict(line.split(" ", 1) for line in done.stdout.splitlines())


def _run(name: str, command: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "exec", name, "/bin/sh", "-c", command],
        capture_output=True,
        text=True,
    )


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker required")
def test_bfx_chown_gives_the_tree_to_root_and_keeps_a_shared_file(box):
    done = _run(box, lockdown._chown_workspace_cmd("/app", bfx=True))
    assert done.returncode == 0, done.stderr
    owners = _owners(box)
    assert owners["/app"] == owners["/app/src"] == owners["/app/src/a.txt"] == "root"
    # The file with another name, and what it shares, keep their owner.
    assert owners["/app/hardlink.txt"] == owners["/data/outside.txt"] == "agent"
    # The link's target folder is not followed.
    assert owners["/data"] == "agent"


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker required")
def test_a_workspace_path_through_a_link_takes_today_chown(box):
    done = _run(box, lockdown._chown_workspace_cmd("/work", bfx=True))
    assert done.returncode == 0, done.stderr
    owners = _owners(box)
    assert owners["/app"] == owners["/app/src/a.txt"] == "root"


def test_without_bfx_the_verifier_clear_is_unchanged():
    assert lockdown._clear_verifier_dir_cmd() == lockdown._CLEAR_VERIFIER_DIR_CMD
    step = lockdown._verifier_output_steps()[0]
    assert step.command == lockdown._CLEAR_VERIFIER_DIR_CMD


def test_with_bfx_the_verifier_clear_runs_bfx_clear_with_today_as_fallback():
    command = lockdown._clear_verifier_dir_cmd(bfx=True)
    args = {"root": "/logs/verifier", "path": ""}
    assert (
        "/.bfx/bfx clear --json " + shlex.quote(json.dumps(args, separators=(",", ":")))
        in command
    )
    assert lockdown._CLEAR_VERIFIER_DIR_CMD in command
    # Today's form is chosen before bfx runs, never after a bfx answer.
    before, _, after = command.partition("clear --json")
    assert lockdown._CLEAR_VERIFIER_DIR_CMD in before
    assert lockdown._CLEAR_VERIFIER_DIR_CMD not in after
    step = lockdown._verifier_output_steps(bfx=True)[0]
    assert step.command == command
    assert step.label == "Verifier hardening failed: clearing verifier output directory"


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker required")
def test_bfx_clear_empties_the_folder_and_leaves_what_a_link_names(box):
    made = _run(
        box,
        "mkdir -p /logs/verifier/sub && echo x > /logs/verifier/sub/f && "
        "echo y > /logs/verifier/.hidden && echo keep > /data/keep.txt && "
        "ln -s /data /logs/verifier/out",
    )
    assert made.returncode == 0, made.stderr
    done = _run(box, lockdown._clear_verifier_dir_cmd(bfx=True))
    assert done.returncode == 0, done.stderr
    listing = _run(
        box, "ls -A /logs/verifier; stat -c %a /logs/verifier; cat /data/keep.txt"
    )
    assert listing.stdout.split() == ["777", "keep"]


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker required")
def test_a_linked_verifier_folder_takes_today_clear(box):
    made = _run(box, "mkdir -p /logs && ln -s /data /logs/verifier")
    assert made.returncode == 0, made.stderr
    done = _run(box, lockdown._clear_verifier_dir_cmd(bfx=True))
    assert done.returncode == 0, done.stderr
    state = _run(
        box, "test -d /logs/verifier && test ! -L /logs/verifier && echo dir; ls /data"
    )
    assert state.stdout.split()[0] == "dir"
    assert "outside.txt" in state.stdout


# The plain step (no bfx) on GNU and BusyBox images (ACC-L1): root's Python
# walk, and the find form an image without root Python takes.

_PLAIN_TREE = r"""
set -e
id agent >/dev/null 2>&1 || useradd -m -u 1000 agent 2>/dev/null || adduser -D -u 1000 agent
mkdir -p /app/src /app/empty /data/dir
echo 1 > /app/src/a.txt
echo shared > /data/outside.txt
ln /data/outside.txt /app/hardlink.txt
echo two > /app/src/twice.txt
ln /app/src/twice.txt /app/src/twice-again.txt
echo t > /data/target.txt
ln -s /data/dir /app/dir-link
ln -s /data/target.txt /app/file-link
mkfifo /app/fifo
chown -R agent:agent /app /data
chown -h agent:agent /app/dir-link /app/file-link
"""

_PLAIN_NAMES = (
    "/app", "/app/src", "/app/src/a.txt", "/app/empty", "/app/fifo",
    "/app/dir-link", "/app/file-link", "/app/hardlink.txt",
    "/app/src/twice.txt", "/app/src/twice-again.txt",
    "/data", "/data/dir", "/data/outside.txt", "/data/target.txt",
)  # fmt: skip

_IMAGES = ("python:3.12-slim", "python:3.12-alpine")  # GNU and BusyBox find
# The root route names the judged interpreter (lockdown._prep_on_path).
_ROOT_PYTHON = "__bf_root_python=/usr/local/bin/python3\n"
_NO_ROOT_PYTHON = "__bf_root_python=\n"


def _start(image: str, *extra: str) -> str:
    import os

    name = os.environ.get("BENCHFLOW_TEST_CONTAINER_PREFIX", "trcore-chown-")
    name += "plain-" + uuid.uuid4().hex[:8]
    label = os.environ.get("BENCHFLOW_TEST_CONTAINER_LABEL", "target=TR-CORE-chown")
    key, _, value = label.partition("=")
    run = subprocess.run(
        ["docker", "run", "-d", "--name", name, "--label", f"{key}={value}",
         "--network", "none", *extra, image, "sleep", "600"],
        capture_output=True,
        text=True,
    )  # fmt: skip
    if run.returncode != 0:
        pytest.skip(f"cannot start the image: {run.stderr.strip()}")
    return name


@pytest.fixture(params=_IMAGES)
def plain_box(request):
    if not _docker_ready():
        pytest.skip("docker daemon not running")
    name = _start(request.param, "--tmpfs", "/ro:ro,uid=1000")
    try:
        made = _run(name, _PLAIN_TREE)
        assert made.returncode == 0, made.stderr
        yield name
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@pytest.fixture(params=[_ROOT_PYTHON, _NO_ROOT_PYTHON], ids=["python", "find"])
def route(request):
    return request.param


def _plain_owners(name: str) -> dict[str, str]:
    done = _run(name, "stat -c '%n %U' " + " ".join(_PLAIN_NAMES))
    assert done.returncode == 0, done.stderr
    return dict(line.split(" ", 1) for line in done.stdout.splitlines())


def _not_root(name: str, where: str = "/app") -> list[str]:
    """Workspace entries not root's, except regular files with another name
    (the end state the freeze must reach), read after the step."""
    done = _run(name, f"find {where} ! -user 0 \\( ! -type f -o -links 1 \\) -print")
    assert done.returncode == 0, done.stderr
    return done.stdout.splitlines()


def test_plain_chown_gives_the_tree_to_root_and_keeps_a_shared_file(plain_box, route):
    done = _run(plain_box, route + lockdown._chown_workspace_cmd("/app"))
    assert done.returncode == 0, done.stderr
    owners = _plain_owners(plain_box)
    # Folders, files, a FIFO and the links themselves are root's, as with
    # chown -R.
    for path in (
        "/app", "/app/src", "/app/src/a.txt", "/app/empty", "/app/fifo",
        "/app/dir-link", "/app/file-link",
    ):  # fmt: skip
        assert owners[path] == "root", (path, owners)
    # A regular file with another name keeps its owner at every name, in
    # the workspace or outside it (bfx chown's I5).
    for path in (
        "/app/hardlink.txt", "/data/outside.txt",
        "/app/src/twice.txt", "/app/src/twice-again.txt",
    ):  # fmt: skip
        assert owners[path] == "agent", (path, owners)
    # No link is followed.
    for path in ("/data", "/data/dir", "/data/target.txt"):
        assert owners[path] == "agent", (path, owners)
    assert _not_root(plain_box) == []


def test_plain_chown_fails_when_an_entry_cannot_be_given_to_root(plain_box, route):
    """A read-only mount in the tree fails the step, as chown -R did."""
    done = _run(plain_box, route + lockdown._chown_workspace_cmd("/ro"))
    assert done.returncode != 0


@pytest.mark.parametrize("image", _IMAGES)
def test_a_read_only_entry_in_an_early_batch_fails_the_freeze(image, route):
    """BusyBox find keeps only the last -exec ... + batch's status, so the
    step must not take it as proof (M2). The workspace root is a read-only
    mount, the first entry find hands to chown; a writable mount inside holds
    30,000 files, so later batches follow it."""
    if not _docker_ready():
        pytest.skip("docker daemon not running")
    ro, rw = (f"trcore-chown-vol-{uuid.uuid4().hex[:8]}" for _ in range(2))
    name = None
    try:
        for volume, fill in (
            (ro, "mkdir /v/big && chown -R 1000:1000 /v"),
            (rw, "cd /v && seq 30000 | xargs touch && chown -R 1000:1000 /v"),
        ):
            made = subprocess.run(
                ["docker", "run", "--rm", "--network", "none", "-v",
                 f"{volume}:/v", image, "sh", "-c", fill],
                capture_output=True,
                text=True,
            )  # fmt: skip
            assert made.returncode == 0, made.stderr
        name = _start(
            image,
            "--mount", f"type=volume,src={ro},dst=/app,readonly",
            "--mount", f"type=volume,src={rw},dst=/app/big",
        )  # fmt: skip
        done = _run(name, route + lockdown._chown_workspace_cmd("/app"))
        assert done.returncode != 0, (done.stdout, done.stderr)
        assert _not_root(name), "the read-only root should still be the agent's"
    finally:
        if name:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        subprocess.run(["docker", "volume", "rm", ro, rw], capture_output=True)


@pytest.mark.parametrize("image", _IMAGES)
def test_a_path_over_4096_bytes_does_not_fail_the_freeze(image):
    """chown -R walks by descriptor, so a 5,200-byte path passed with it on
    GNU; root's Python walk does too (L2)."""
    if not _docker_ready():
        pytest.skip("docker daemon not running")
    name = _start(image)
    try:
        # Python's chdir goes on past PATH_MAX, where the shell's cd and
        # BusyBox chown -R stop.
        made = _run(
            name,
            "python3 -c 'import os\n"
            'os.makedirs("/app"); os.chdir("/app")\n'
            'for _ in range(400): os.lchown(".", 1000, 1000); '
            'os.mkdir("b" * 12); os.chdir("b" * 12)\n'
            'open("leaf", "w").close(); os.lchown("leaf", 1000, 1000)\n'
            'os.lchown(".", 1000, 1000)\'',
        )
        assert made.returncode == 0, made.stderr
        done = _run(name, _ROOT_PYTHON + lockdown._chown_workspace_cmd("/app"))
        assert done.returncode == 0, done.stderr
        left = _run(
            name,
            "python3 -c 'import os\n"
            'os.chdir("/app"); n = 0\n'
            "while True:\n"
            ' n += sum(os.lstat(e).st_uid != 0 for e in os.listdir("."))\n'
            ' if not os.path.isdir("b" * 12): break\n'
            ' os.chdir("b" * 12)\n'
            "print(n)'",
        )
        assert left.stdout.strip() == "0", left.stderr
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def _close(name: str) -> subprocess.CompletedProcess:
    for command in (
        lockdown._chown_workspace_cmd("/app"),
        lockdown._close_workspace_cmd("/app"),
    ):
        done = _run(name, _ROOT_PYTHON + command)
        if done.returncode != 0:
            return done
    return done


def test_closing_for_kept_services_copies_a_shared_file_to_root(plain_box):
    """With kept services the close step gives a file with another name a
    fresh root-owned copy at its workspace name; the other name keeps the
    agent user's file (ACC-L1)."""
    chmod = _run(plain_box, "chmod 6777 /app/hardlink.txt")
    assert chmod.returncode == 0, chmod.stderr
    done = _close(plain_box)
    assert done.returncode == 0, done.stderr
    owners = _plain_owners(plain_box)
    for path in ("/app/hardlink.txt", "/app/src/twice.txt", "/app/src/twice-again.txt"):
        assert owners[path] == "root", (path, owners)
    assert owners["/data/outside.txt"] == "agent"
    facts = _run(
        plain_box,
        "stat -c '%n %h %a %i' /app/hardlink.txt /data/outside.txt "
        "/app/src/twice.txt /app/src/twice-again.txt; "
        "cat /app/hardlink.txt /app/src/twice.txt /app/src/twice-again.txt; "
        "ls -A /app /app/src",
    )
    assert facts.returncode == 0, facts.stderr
    lines = facts.stdout.splitlines()
    rows = {line.split()[0]: line.split()[1:] for line in lines[:4]}
    assert {row[0] for row in rows.values()} == {"1"}
    # setuid, setgid and group and other write are gone from the copy.
    assert rows["/app/hardlink.txt"][1] == "755"
    assert rows["/app/hardlink.txt"][2] != rows["/data/outside.txt"][2]
    assert rows["/app/src/twice.txt"][2] != rows["/app/src/twice-again.txt"][2]
    assert lines[4:7] == ["shared", "two", "two"]
    # No temporary name is left behind.
    assert not [name for name in lines[7:] if name.startswith(".bf")]
    assert _run(plain_box, "find /app ! -user 0").stdout == ""


def test_closing_needs_root_python(plain_box):
    done = _run(plain_box, _NO_ROOT_PYTHON + lockdown._close_workspace_cmd("/app"))
    assert done.returncode != 0


def _agent_shared(name: str, files: int, script: str) -> None:
    """*files* agent-owned workspace files, each with a second name in /cache."""
    made = _run(
        name,
        "set -e; python3 -c 'import os\n"
        f"for i in range({files}):\n"
        ' d = "/app/d%d" % (i % 3); os.makedirs(d, exist_ok=True)\n'
        ' os.makedirs("/cache", exist_ok=True)\n'
        ' open("/cache/%d" % i, "w").write("x" * (i % 4096))\n'
        ' os.link("/cache/%d" % i, "%s/%d" % (d, i))\'; '
        + script
        + "chown -R 1000:1000 /cache /app",
    )
    assert made.returncode == 0, made.stderr


def test_closing_7000_shared_files_and_a_250_byte_name(plain_box):
    """The close step either closes every file or fails; a 250-byte name
    leaves room for its short temporary name beside it (M2, L3)."""
    long = "L" * 250
    _agent_shared(
        plain_box,
        7000,
        f"echo agent > /app/d0/{long}; ln /app/d0/{long} /cache/long; ",
    )
    done = _close(plain_box)
    assert done.returncode == 0, done.stderr
    assert _run(plain_box, "find /app ! -user 0").stdout == ""
    assert _run(plain_box, f"cat /app/d0/{long}").stdout == "agent\n"
    outside = _run(plain_box, "find /cache -type f -user 1000 | wc -l")
    assert outside.stdout.strip() == "7001"


def test_closing_a_hard_linked_venv_of_10260_files(plain_box):
    """uv fills a venv from the agent user's cache with hard links: about
    10,000 such files (M1) take one walk, no process per file and no list
    in the environment. The bound is loose; results/ACC-L1.md has timings."""
    import time

    _agent_shared(plain_box, 10260, "")
    start = time.monotonic()
    done = _close(plain_box)
    took = time.monotonic() - start
    assert done.returncode == 0, done.stderr
    assert _run(plain_box, "find /app ! -user 0").stdout == ""
    assert took < 30, took
