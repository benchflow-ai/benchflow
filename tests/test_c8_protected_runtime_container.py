"""C8 by construction, in a real container: once the agent has had
a turn, nothing BenchFlow runs as root comes from a place the agent user can
write.

Every test drives BenchFlow's real code (``setup_sandbox_user``, a
``DockerSandbox``'s ``exec``, the review capture, the kill helper, the uv
install) against a container of ``python:3.12-slim`` whose image environment
names a preload library that does not exist: glibc says "cannot be preloaded"
for every program that starts with it, so a root step that kept the image's
loader settings shows on stderr. The agent's turn is replayed as the agent
user: it plants programs that leave a mark under ``/tmp/marks`` when they run
in the workspace folder the image puts first on PATH, and a ``json`` module
where the image's PYTHONPATH and WORKDIR point.
Skipped without a Docker daemon.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import subprocess
import uuid

import pytest

from benchflow.sandbox._base import ExecResult

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
]

IMAGE = "python:3.12-slim"
# Each container this file starts carries it; a build VM's owner label can
# replace it.
LABEL = os.environ.get("BENCHFLOW_TEST_CONTAINER_LABEL", "PT=1")
PRELOAD = "/usr/local/lib/bf-agent-profile.so"
# The agent's workspace. A workspace of /usr/local is refused since round 9
# (it holds /usr/local/lib, which the loader searches); instead the image
# puts the workspace first on PATH and PYTHONPATH, and root's exec starts in
# it (WORKDIR).
WORKSPACE = "/srv"
AGENT_BIN = f"{WORKSPACE}/bin"
IMAGE_PATH = f"{AGENT_BIN}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
TOOLS = (
    "sh", "bash", "python3", "tar", "rm", "kill", "ps", "env", "sed", "find",
    "cat", "id", "setpriv", "chmod", "chown", "readlink", "stat", "sleep",
)  # fmt: skip


def _docker_ready() -> bool:
    try:
        probe = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
    except Exception:
        return False
    return probe.returncode == 0


def _dx(name: str, command: str, user: str = "0") -> subprocess.CompletedProcess:
    """A command in the container as the test's harness (not BenchFlow)."""
    return subprocess.run(
        [
            "docker",
            "exec",
            "-u",
            user,
            "-e",
            "LD_PRELOAD=",
            "-e",
            "PATH=/usr/sbin:/usr/bin:/sbin:/bin",
            name,
            "/bin/sh",
            "-c",
            command,
        ],
        capture_output=True,
        text=True,
    )


def _sandbox(name: str):
    """A DockerSandbox whose ``main`` container is *name*: BenchFlow's own
    exec, argv and routes, run with the real ``docker`` CLI."""
    from benchflow.sandbox.docker import DockerSandbox

    sandbox = object.__new__(DockerSandbox)
    sandbox.default_user = None
    sandbox._persistent_env = {}
    sandbox._main_container = name
    sandbox.logger = logging.getLogger("test")
    # Downloads stream ``docker cp`` with Compose's process environment
    # here the real docker CLI, with this process's.
    sandbox._docker_compose_env = lambda: dict(os.environ)
    sandbox._compose_transport_env = lambda: {}

    async def run(argv, check=False, timeout_sec=None, stdin_data=None, **kwargs):
        # The exec's env arrives on stdin: pass it on.
        if isinstance(stdin_data, str):
            stdin_data = stdin_data.encode()
        done = await asyncio.to_thread(
            subprocess.run, ["docker", *argv], input=stdin_data, capture_output=True
        )
        result = ExecResult(
            return_code=done.returncode,
            stdout=done.stdout.decode(errors="replace"),
            stderr=done.stderr.decode(errors="replace"),
        )
        if check and result.return_code != 0:
            raise RuntimeError(result.stderr)
        return result

    sandbox._run_docker_container_command = run
    sandbox._docker_cli = run
    sandbox._docker_client_env = lambda: None
    return sandbox


@pytest.fixture
def box():
    if not _docker_ready():
        pytest.skip("docker daemon not running")
    name = "PT-c8-r8-" + uuid.uuid4().hex[:8]
    run = subprocess.run(
        [
            "docker", "run", "-d", "--name", name, "--label", LABEL,
            "--network", "none", "-e", f"LD_PRELOAD={PRELOAD}",
            "-e", f"PATH={IMAGE_PATH}", "-e", f"PYTHONPATH={WORKSPACE}",
            "-w", WORKSPACE, IMAGE, "sleep", "1200",
        ],
        capture_output=True,
        text=True,
    )  # fmt: skip
    if run.returncode != 0:
        pytest.skip(f"cannot start {IMAGE}: {run.stderr.strip()}")
    made = _dx(name, f"mkdir -p {AGENT_BIN}")
    assert made.returncode == 0, made.stderr
    try:
        yield name
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def _agent_turn(name: str) -> None:
    """The agent's turn: a mark-leaving program for every tool in the
    folder first on the image's PATH, a ``json`` module in the folder on
    its PYTHONPATH, which is also where root's exec starts."""
    plant = "; ".join(
        f"printf '#!/bin/sh\\nmkdir -p /tmp/marks; : > /tmp/marks/{tool}\\n' "
        f"> {AGENT_BIN}/{tool} && chmod 755 {AGENT_BIN}/{tool}"
        for tool in TOOLS
    )
    script = (
        f"{plant}; "
        f"mkdir -p {WORKSPACE}/work && echo 42 > {WORKSPACE}/work/answer.txt; "
        'echo \'import os; open("/tmp/marks/cwd-module", "w").close()\' '
        f"> {WORKSPACE}/json.py"
    )
    done = _dx(name, script, user="agent")
    assert done.returncode == 0, done.stderr


def _marks(name: str) -> list[str]:
    done = _dx(name, "ls /tmp/marks 2>/dev/null || true")
    return done.stdout.split()


async def _setup(name: str, workspace: str = WORKSPACE):
    from benchflow.sandbox.lockdown import setup_sandbox_user

    sandbox = _sandbox(name)
    await setup_sandbox_user(sandbox, "agent", workspace, guard_daemon=False)
    return sandbox


@pytest.mark.asyncio
async def test_root_plumbing_runs_nothing_the_agent_wrote_after_its_turn(box):
    """An image whose /usr/local/bin and stdlib everyone may write: root's
    ``python3``, shell and tools after the turn are never the agent's, and
    no root program starts with the image's preload."""
    sandbox = await _setup(box)
    _agent_turn(box)
    result = await sandbox.exec(
        "python3 -c 'print(1)' >/dev/null 2>&1; "
        "/opt/runtime/rt/py/python3 -c 'import json, sys; print(json.dumps(sys.prefix))'; "
        "sh -c 'echo shell-ok'; tar --version >/dev/null; env >/dev/null; "
        f"find {WORKSPACE}/work -maxdepth 0 >/dev/null; id -u",
        user="root",
    )
    assert result.return_code == 0, result.stderr
    assert "shell-ok" in result.stdout
    assert '"/opt/runtime/rt"' in result.stdout
    assert "cannot be preloaded" not in (result.stderr or "")
    assert _marks(box) == []


@pytest.mark.asyncio
async def test_review_capture_imports_no_agent_module_and_runs_no_agent_python(
    box, tmp_path
):
    """PT-C8-R3-1: the capture's Python is isolated and protected; the agent's
    ``json`` (in the folder Docker starts in, and in its stdlib) never loads."""
    from benchflow.review.evidence import capture_workspace

    sandbox = await _setup(box)
    _agent_turn(box)
    await capture_workspace(sandbox, f"{WORKSPACE}/work", tmp_path / "bundle")
    answer = tmp_path / "bundle" / "workspace" / "answer.txt"
    assert answer.read_text() == "42\n"
    assert _marks(box) == []


@pytest.mark.asyncio
async def test_kill_helper_runs_the_routes_tools_not_the_agents(box):
    """PT-C8-R3-3: the kill command keeps the route's PATH; /usr/local/bin,
    which everyone may write here, is never searched."""
    from benchflow.sandbox.lockdown import _kill_sandbox_user_procs

    sandbox = await _setup(box)
    _agent_turn(box)
    subprocess.Popen(
        ["docker", "exec", "-u", "agent", box, "/bin/sleep", "300"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    await asyncio.sleep(1)
    killed = await _kill_sandbox_user_procs(sandbox, "agent")
    assert killed >= 1
    assert _marks(box) == []


@pytest.mark.asyncio
async def test_an_opt_workspace_never_hands_over_benchflows_runtime_tree(box):
    """An honest ``WORKDIR /opt`` keeps
    /opt/runtime root's alone, so a later JS agent install's check passes,
    while the agent user can still add its own files to /opt."""
    from benchflow._root_only import root_only_tree_check

    _dx(box, "mkdir -p /opt/runtime/node/bin /opt/project")
    sandbox = await _setup(box, workspace="/opt")
    owned = _dx(box, "touch /opt/new-file && mv /opt/runtime /opt/x", user="agent")
    assert "new-file" in _dx(box, "ls /opt").stdout
    assert owned.returncode != 0
    check = await sandbox.exec(
        root_only_tree_check("/opt/runtime", "BenchFlow's agent runtime folder"),
        user="root",
    )
    assert check.return_code == 0, check.stderr


@pytest.mark.asyncio
async def test_a_link_through_a_folder_the_agent_writes_fails_the_runtime_check(box):
    """PT-C8-R3-9: a runtime reached through a link hop in an
    agent-writable folder is refused even though its target is root's."""
    from benchflow._root_only import root_only_tree_check

    _dx(
        box,
        "mkdir -p /srv/real-node /app && chmod 777 /app && "
        "ln -s /srv/real-node /app/node && mkdir -p /opt/runtime && "
        "ln -s /app/node /opt/runtime/node",
    )
    sandbox = _sandbox(box)
    check = await sandbox.exec(
        root_only_tree_check("/opt/runtime/node", "BenchFlow's Node"), user="root"
    )
    assert check.return_code != 0


@pytest.mark.asyncio
async def test_cached_uv_is_checked_before_its_version_probe_runs(box):
    """PT-C8-R3-5 / R3-4: a cached uv folder another user owns
    is refused before root runs ``uv --version``."""
    from benchflow._uv_pin import UV_DIR, uv_install_command

    _dx(
        box,
        f"mkdir -p {UV_DIR} && printf '#!/bin/sh\\nmkdir -p /tmp/marks; "
        f": > /tmp/marks/uv\\necho uv 0.13.4\\n' > {UV_DIR}/uv && "
        f"chmod 755 {UV_DIR}/uv && chown -R 1000:1000 {UV_DIR}",
    )
    sandbox = _sandbox(box)
    result = await sandbox.exec(uv_install_command(), user="root")
    assert result.return_code != 0
    assert _marks(box) == []


@pytest.mark.asyncio
async def test_root_steps_after_the_turn_take_the_kit_shell_and_path(box):
    """The route on the kit: root's shell is the kit's by absolute path and
    PATH is the kit's bin and proven system folders, never a folder
    another user may write."""
    from benchflow._runtime_tree import KIT_BIN, KIT_SH

    sandbox = await _setup(box)
    route = await sandbox.exec_route("root")
    assert route.shell == KIT_SH
    assert route.pin is not None and route.pin.split(":")[0] == KIT_BIN
    assert AGENT_BIN not in route.pin.split(":")
    result = await sandbox.exec('printf %s "$PATH"', user="root")
    assert result.stdout == route.pin


@pytest.mark.asyncio
async def test_a_restored_snapshot_starts_on_an_absolute_sleep_with_no_preload(box):
    """PT-C8-R3-6: the replacement's PID 1 is BenchFlow's sleep, its
    loader settings empty, the image's kept as data."""
    from benchflow._runtime_tree import KIT_BIN

    await _setup(box)
    _agent_turn(box)
    tag = f"pt-c8-r8-snap-{uuid.uuid4().hex[:8]}"
    restored = f"{box}-restored"
    try:
        commit = subprocess.run(
            ["docker", "commit", box, tag], capture_output=True, text=True
        )
        assert commit.returncode == 0, commit.stderr
        run = subprocess.run(
            [
                "docker", "run", "-d", "--name", restored, "--label", LABEL,
                "--network", "none", "--env", "LD_PRELOAD=",
                "--env", f"_BF_IMAGE_LD_PRELOAD={PRELOAD}",
                "--entrypoint", f"{KIT_BIN}/sleep", tag, "infinity",
            ],
            capture_output=True,
            text=True,
        )  # fmt: skip
        assert run.returncode == 0, run.stderr
        cmd = _dx(restored, "tr '\\000' ' ' < /proc/1/cmdline")
        assert cmd.stdout.startswith(f"{KIT_BIN}/sleep"), cmd.stdout
        assert _marks(restored) == []
    finally:
        subprocess.run(["docker", "rm", "-f", restored], capture_output=True)
        with contextlib.suppress(Exception):
            subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)


@pytest.mark.asyncio
async def test_a_changed_kit_refuses_every_root_step(box):
    """The kit is used only as BenchFlow made it: a file changed after the
    preparation (here by root, standing in for any change) refuses root's
    steps in a new sandbox generation."""
    from benchflow.sandbox._base import RuntimeKitBrokenError

    sandbox = await _setup(box)
    _dx(box, "echo changed >> /opt/runtime/rt/RECORD")
    sandbox._runtime_kit_unknown()  # as a restore leaves it
    with pytest.raises(RuntimeKitBrokenError):
        await sandbox.exec("true", user="root")


def test_kit_python_is_isolated_and_keeps_its_own_libraries():
    """A Python that needs its own libpython from
    /usr/local/lib works from the kit after the agent removed it."""
    if not _docker_ready():
        pytest.skip("docker daemon not running")
    name = "PT-c8-r8-py-" + uuid.uuid4().hex[:8]
    subprocess.run(
        ["docker", "run", "-d", "--name", name, "--label", LABEL, "--network",
         "none", IMAGE, "sleep", "600"],
        capture_output=True,
    )  # fmt: skip
    try:
        sandbox = _sandbox(name)
        asyncio.run(_setup_only(sandbox))
        _dx(name, "for f in /usr/local/lib/libpython3*.so*; do mv $f $f.gone; done")
        out = _dx(name, "cd /tmp && /opt/runtime/rt/py/python3 -c "
                  "'import sys, tarfile; print(sys.flags.isolated, sys.flags.no_site, sys.prefix)'")  # fmt: skip
        assert out.returncode == 0, out.stderr
        assert out.stdout.split() == ["1", "1", "/opt/runtime/rt"]
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


async def _setup_only(sandbox) -> None:
    from benchflow.sandbox.lockdown import setup_sandbox_user

    await setup_sandbox_user(sandbox, "agent", "/srv", guard_daemon=False)


@pytest.mark.asyncio
async def test_capture_from_an_app_workspace_never_imports_the_agents_json(tmp_path):
    """Docker starts root's exec in the image's
    WORKDIR, the agent's workspace; the capture's ``python3`` must not
    import the ``json`` module the agent left there."""
    from benchflow.review.evidence import capture_workspace

    if not _docker_ready():
        pytest.skip("docker daemon not running")
    name = "PT-c8-r8-app-" + uuid.uuid4().hex[:8]
    subprocess.run(
        ["docker", "run", "-d", "--name", name, "--label", LABEL, "--network",
         "none", "-w", "/app", IMAGE, "sleep", "600"],
        capture_output=True,
    )  # fmt: skip
    try:
        _dx(name, "mkdir -p /app")
        sandbox = _sandbox(name)
        from benchflow.sandbox.lockdown import setup_sandbox_user

        await setup_sandbox_user(sandbox, "agent", "/app", guard_daemon=False)
        planted = _dx(
            name,
            "echo 42 > /app/answer.txt; printf '%s\\n' 'import os' "
            "'os.makedirs(\"/tmp/marks\", exist_ok=True)' "
            '\'open("/tmp/marks/json", "w").close()\' '
            "'from _json import *' > /app/json.py",
            user="agent",
        )
        assert planted.returncode == 0, planted.stderr
        await capture_workspace(sandbox, "/app", tmp_path / "bundle")
        assert _marks(name) == []
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@pytest.mark.asyncio
async def test_a_chain_of_links_inside_the_tree_is_still_walked_where_it_leaves(box):
    """The tree check skips walking a link that stays in the tree only
    because every other link is walked: ``a -> b`` inside, ``b`` leaving
    through a folder others may write, is refused."""
    from benchflow._root_only import root_only_tree_check

    _dx(
        box,
        "mkdir -p /srv/real /app/hop /opt/runtime/t && chmod 777 /app/hop && "
        "ln -s /srv/real /app/hop/real && ln -s /app/hop/real /opt/runtime/t/b && "
        "ln -s b /opt/runtime/t/a && mkdir -p /opt/runtime/ok && "
        "ln -s ../ok/c /opt/runtime/ok/d && touch /opt/runtime/ok/c",
    )
    sandbox = _sandbox(box)
    bad = await sandbox.exec(root_only_tree_check("/opt/runtime/t", "t"), user="root")
    good = await sandbox.exec(
        root_only_tree_check("/opt/runtime/ok", "ok"), user="root"
    )
    assert bad.return_code != 0
    assert good.return_code == 0, good.stderr


# Round 9: the findings that need a real container.


@contextlib.contextmanager
def _fresh(prep: str = "", *, env: tuple[str, ...] = ()):
    """A fresh container of IMAGE with an ``agent`` user (uid 1000), *prep*
    run as root first (the image as built)."""
    if not _docker_ready():
        pytest.skip("docker daemon not running")
    name = "PT-c8-r9-" + uuid.uuid4().hex[:8]
    flags = [f for value in env for f in ("-e", value)]
    run = subprocess.run(
        ["docker", "run", "-d", "--name", name, "--label", LABEL, "--network",
         "none", *flags, "-w", WORKSPACE, IMAGE, "sleep", "900"],
        capture_output=True,
        text=True,
    )  # fmt: skip
    if run.returncode != 0:
        pytest.skip(f"cannot start {IMAGE}: {run.stderr.strip()}")
    try:
        made = _dx(name, f"mkdir -p {WORKSPACE}; {prep or 'true'}")
        assert made.returncode == 0, made.stderr
        yield name
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@pytest.mark.asyncio
async def test_a_runtime_root_the_image_made_a_link_is_refused():
    """/opt/runtime as a link uid 1000 owns in a sticky /opt
    its group may write (the kernel's protected_symlinks covers only a
    folder everyone may write), to a folder that is root's today: its owner
    could swap it later, so it is refused before anything is made through it."""
    prep = (
        "useradd -m -u 1000 agent && chgrp 1000 /opt && chmod 1775 /opt && "
        "mkdir -p /srv/rt && "
        "ln -s /srv/rt /opt/runtime && chown -h 1000 /opt/runtime"
    )
    with _fresh(prep) as name:
        with pytest.raises(RuntimeError, match="protected runtime"):
            await _setup(name)
        assert _dx(name, "ls -A /srv/rt").stdout == ""


@pytest.mark.asyncio
async def test_a_library_folder_another_user_may_write_refuses_the_kit():
    """A folder the loader searches that another
    user may write (or a loader file naming one) means no root step can
    rely on its libraries: the kit is refused, not built on them."""
    for prep in (
        # On the loader's list (ld.so.conf.d/libc.conf), not on PATH.
        "chmod o+w /usr/local/lib",
        "mkdir -p /srv/lib && chown 1000 /srv/lib && "
        "echo /srv/lib > /etc/ld.so.conf.d/zz-agent.conf",
    ):
        with _fresh(prep) as name, pytest.raises(RuntimeError, match="loader searches"):
            await _setup(name)


@pytest.mark.asyncio
async def test_a_usr_local_workspace_keeps_the_loaders_folders_roots():
    """A /usr/local workspace
    is granted again, without the loader's folders. The agent installs into
    site-packages and its own folders, never puts a library where the
    loader (or a later ldconfig) looks, never renames that folder, and
    root's PATH leaves the granted bin folders out."""
    with _fresh() as name:
        sandbox = await _setup(name, workspace="/usr/local")
        assert "/usr/local/bin" not in (sandbox.runtime_kit.path or "")
        assert sandbox.runtime_kit.python
        for command in (
            "touch /usr/local/lib/python3.12/site-packages/agent-file",
            "mkdir /usr/local/agent-folder",
        ):
            done = _dx(name, command, user="agent")
            assert done.returncode == 0, (command, done.stderr)
        for command in (
            "touch /usr/local/lib/libbf-agent.so",
            "mv /usr/local/lib /usr/local/lib.moved",
            "mkdir /usr/local/lib64/x",
        ):
            done = _dx(name, command, user="agent")
            assert done.returncode != 0, command


@pytest.mark.asyncio
async def test_one_foreign_link_in_a_system_folder_keeps_its_other_programs():
    """PT-C8-R8-6: a system folder holding one link into the agent's
    workspace stays usable through the kit's view of it: an honest later
    install still finds apt-get and dpkg, never that link."""
    prep = (
        f"printf '#!/bin/sh\\nmkdir -p /tmp/marks; : > /tmp/marks/foreign\\n' "
        f"> {WORKSPACE}/foreign && chmod 755 {WORKSPACE}/foreign && "
        f"ln -s {WORKSPACE}/foreign /usr/bin/bf-foreign"
    )
    with _fresh(prep) as name:
        sandbox = await _setup(name)
        result = await sandbox.exec(
            "command -v apt-get; dpkg --version >/dev/null && echo dpkg-ok; "
            "command -v bf-foreign || echo no-foreign; bf-foreign 2>/dev/null; true",
            user="root",
        )
        assert result.return_code == 0, result.stderr
        lines = result.stdout.split()
        assert lines[0].startswith("/opt/runtime/rt/view/") or lines[0] in (
            "/usr/sbin/apt-get",
            "/sbin/apt-get",
        ), lines
        assert "dpkg-ok" in lines and "no-foreign" in lines
        assert _marks(name) == []


@pytest.mark.asyncio
async def test_the_verifier_never_gets_a_loader_value_the_agent_can_replace():
    """The image's LD_AUDIT names a library in the agent's
    workspace; the verifier's root command (the task route on the kit)
    starts without it and says so."""
    audit = f"{WORKSPACE}/audit.so"
    with _fresh(env=(f"LD_AUDIT={audit}",)) as name:
        sandbox = await _setup(name)
        with sandbox.task_command():
            result = await sandbox.exec('printf %s "${LD_AUDIT:-unset}"', user="root")
        assert result.stdout == "unset", result
        assert "not restoring LD_AUDIT" in result.stderr


@pytest.mark.asyncio
async def test_root_plumbing_starts_from_nothing():
    """Round 9, requirement 3: a root plumbing process has only the
    route's names: no image value (a library, a Python path, a HOME the
    agent may be given) reaches it."""
    with _fresh(env=("PYTHONPATH=/srv", "NODE_OPTIONS=--require /srv/x.js")) as name:
        sandbox = await _setup(name)
        result = await sandbox.exec("env", user="root")
        names = {line.split("=", 1)[0] for line in result.stdout.splitlines()}
        assert "PYTHONPATH" not in names and "NODE_OPTIONS" not in names, names
        assert "HOME=/opt/runtime/run/home" in result.stdout.splitlines()


# Round 10: the b114 regressions that need a real container.


@pytest.mark.asyncio
async def test_commented_loader_configuration_starts_an_honest_sandbox():
    """PT-C8-R9-2: comments ldconfig accepts (leading whitespace,
    trailing) and an include of a relative pattern start the sandbox; an
    included folder another user may write still refuses it."""
    conf = (
        "printf '  # a comment\\n/usr/local/lib # trailing\\n\\n"
        "include extra.d/*.conf\\n' > /etc/ld.so.conf.d/zz-test.conf && "
        "mkdir -p /etc/ld.so.conf.d/extra.d /opt/vendor/lib && "
        "echo /opt/vendor/lib > /etc/ld.so.conf.d/extra.d/v.conf"
    )
    with _fresh(conf) as name:
        sandbox = await _setup(name)
        assert sandbox.runtime_kit is not None
    with (
        _fresh(
            conf + " && useradd -m -u 1000 agent && chown 1000 /opt/vendor/lib"
        ) as name,
        pytest.raises(RuntimeError, match="/opt/vendor/lib"),
    ):
        await _setup(name)


@pytest.mark.asyncio
async def test_a_grant_that_reaches_the_kit_is_undone_by_the_seal():
    """PT-C8-R9-1: a grant made after the kit's build that reaches its files
    or the folder above them (as main's grant does through a mount a sidecar
    shares) is undone by the seal, which runs after it, before any agent
    turn: the kit and its route are root's alone again."""
    from benchflow._runtime_tree import kit_prepare_command

    with _fresh("useradd -m -u 1000 agent") as name:
        built = _dx(name, kit_prepare_command("build"))
        assert built.returncode == 0, built.stderr
        _dx(name, "chown -R 1000:1000 /opt/runtime/rt")
        sealed = _dx(name, kit_prepare_command("seal"))
        assert sealed.returncode == 0, sealed.stderr
        owners = _dx(name, "find /opt/runtime/rt ! -user 0 | head -n 1")
        assert owners.stdout == ""
    with _fresh("useradd -m -u 1000 agent") as name:
        assert _dx(name, kit_prepare_command("build")).returncode == 0
        _dx(name, "chown 1000:1000 /opt/runtime && chmod 777 /opt/runtime")
        sealed = _dx(name, kit_prepare_command("seal"))
        assert sealed.returncode == 0, sealed.stderr
        mode = _dx(name, "stat -c '%u %a' /opt/runtime").stdout.split()
        assert mode[0] == "0" and int(mode[1], 8) & 0o022 == 0, mode


@pytest.mark.asyncio
async def test_a_root_home_the_agent_owns_gets_a_protected_one_for_ssh():
    """PT-C8-R9-3: with /root as the workspace, the SSH session's
    root entry is pointed at the protected HOME, and the terminal check
    then passes, so an honest Gemini task can launch."""
    from benchflow.sandbox._root_launch import (
        protect_root_terminal_home,
        require_protected_root_terminal,
    )

    with _fresh() as name:
        sandbox = await _setup(name, workspace="/root")
        with pytest.raises(RuntimeError):
            await require_protected_root_terminal(sandbox, "the SSH session")
        await protect_root_terminal_home(sandbox, "the SSH session")
        home = _dx(name, "getent passwd root | cut -d: -f6").stdout.strip()
        assert home == "/opt/runtime/run/home"
        await require_protected_root_terminal(sandbox, "the SSH session")


# Round 10, the b122 mediums.


@pytest.mark.asyncio
async def test_a_loader_folder_name_with_a_comma_or_space_is_kept_as_written():
    """PT-C8-R10-2: glibc reads the rest of an ld.so.conf line as
    one folder. ``/opt/lib,64`` and ``/opt/my libs`` (root's) are checked as
    themselves, not split into ``/opt/lib`` and ``/opt/my`` (writable
    shares here), so an honest sandbox starts."""
    prep = (
        "mkdir -p /opt/lib /opt/my '/opt/lib,64' '/opt/my libs' && "
        "chmod 777 /opt/lib /opt/my && "
        "printf '/opt/lib,64\\n/opt/my libs\\n' > /etc/ld.so.conf.d/zz-names.conf"
    )
    with _fresh(prep) as name:
        sandbox = await _setup(name)
        assert sandbox.runtime_kit is not None


@pytest.mark.asyncio
async def test_a_missing_ssh_folder_behind_a_root_link_keeps_the_terminal():
    """PT-C8-R10-2: root's .ssh is a root-owned link to a folder that
    does not exist under a root-only one: only root could create it, so
    the protected terminal is accepted."""
    with _fresh("ln -s /opt/bf-none/.ssh /root/.ssh") as name:
        sandbox = await _setup(name)
        from benchflow.sandbox._root_launch import require_protected_root_terminal

        await require_protected_root_terminal(sandbox, "the SSH session")


@pytest.mark.asyncio
async def test_a_writable_capability_folder_refuses_the_kit():
    """PT-C8-R9-4: a hardware-capability folder the loader searches
    (glibc-hwcaps) that another user may write refuses the kit."""
    prep = (
        "d=$(dirname $(readlink -f /bin/ls | xargs ldd | awk '/libc.so/{print $3}')); "
        'mkdir -p "$d/glibc-hwcaps/x86-64-v3" && chmod 777 "$d/glibc-hwcaps/x86-64-v3"'
    )
    with _fresh(prep) as name, pytest.raises(RuntimeError, match="glibc-hwcaps"):
        await _setup(name)


@pytest.mark.asyncio
async def test_an_audit_library_the_freeze_made_roots_is_not_restored():
    """PT-C8-R9-5 / PT-C8-R9-3: the image's LD_AUDIT names a
    file in the workspace; after the agent wrote it, verification's freeze
    made it root's. The verifier's root command still starts without it:
    the seal had not recorded that value before the agent's turn."""
    audit = f"{WORKSPACE}/audit.so"
    with _fresh(env=(f"LD_AUDIT={audit}",)) as name:
        sandbox = await _setup(name)
        assert _dx(name, f"echo x > {audit}", user="agent").returncode == 0
        _dx(name, f"chown -R 0:0 {WORKSPACE} && chmod -R go-w {WORKSPACE}")
        with sandbox.task_command():
            result = await sandbox.exec('printf %s "${LD_AUDIT:-unset}"', user="root")
        assert result.stdout == "unset", result
        assert "not restoring LD_AUDIT" in result.stderr


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "writable", ["/opt/vendor/lib", "/opt/vendor/lib/glibc-hwcaps/x86-64-v2"]
)
async def test_a_runpath_folder_another_user_can_write_drops_the_program(
    tmp_path, writable
):
    """PT-C8-R9-4: the image's tar has RUNPATH /opt/vendor/lib; a
    folder the loader tries there before the real library (the folder, or a
    capability subfolder of it) is the agent's. The kit drops tar and root's
    ``tar`` is the refusal, never the image's copy."""
    patchelf = shutil.which("patchelf")
    if patchelf is None:
        pytest.skip("needs patchelf on the test host")
    with _fresh() as name:
        tar = tmp_path / "tar"
        subprocess.run(["docker", "cp", f"{name}:/usr/bin/tar", str(tar)], check=True)
        subprocess.run(
            [patchelf, "--set-rpath", "/opt/vendor/lib", str(tar)], check=True
        )
        subprocess.run(["docker", "cp", str(tar), f"{name}:/usr/bin/tar"], check=True)
        # docker cp keeps the host user's uid: the image's tar is root's.
        made = _dx(
            name,
            "chown 0:0 /usr/bin/tar && chmod 755 /usr/bin/tar && "
            f"mkdir -p {writable} && chown 1000 {writable} && /usr/bin/tar --version",
        )
        assert made.returncode == 0, made.stderr
        sandbox = await _setup(name)
        result = await sandbox.exec("tar --version", user="root")
        assert result.return_code == 126, result
        assert "root does not run it" in result.stderr


def _stub_proxy_install(monkeypatch) -> None:
    """The in-trial proxy install with no download: the pinned uv, the
    managed Python and the package install answer at once. Every check and
    ownership step before them is BenchFlow's own."""
    from benchflow.providers import litellm_runtime as runtime

    monkeypatch.setattr(runtime, "uv_install_command", lambda: "true")
    monkeypatch.setattr(
        runtime, "managed_python_install_command", lambda *a, **k: "true"
    )
    monkeypatch.setattr(runtime, "install_commands", lambda **k: "true")


async def _uploaded_runtime(box: str):
    """A proxy runtime folder as a trial makes it, its files uploaded with
    BenchFlow's own ``docker cp`` and given the host user's uid, as a copy
    from a non-root host leaves them (Moby keeps the archive's uid)."""
    from benchflow.providers import litellm_runtime as runtime

    sandbox = _sandbox(box)
    runtime_dir = f"{runtime.LITELLM_SANDBOX_ROOT}/r1-{uuid.uuid4().hex[:8]}"
    paths = await runtime._upload_runtime_files_to_sandbox(
        sandbox, runtime_dir=runtime_dir, config={}
    )
    requirements = f"{runtime_dir}/litellm-proxy.txt"
    await runtime._upload_text(sandbox, "dummy==1\n", requirements, ".txt")
    host = _dx(box, f"find {runtime_dir} -mindepth 1 -exec chown 1000:1000 {{}} +")
    assert host.returncode == 0, host.stderr
    # The venv the stubbed install would leave: root's, and its Python
    # answers the import check.
    venv = paths["venv"]
    made = _dx(
        box,
        f"mkdir -p {venv}/bin && printf '#!/bin/sh\\ncat >/dev/null; echo ok\\n' "
        f"> {venv}/bin/python && chmod 755 {venv}/bin/python",
    )
    assert made.returncode == 0, made.stderr
    return sandbox, runtime_dir, venv, requirements


@pytest.mark.asyncio
async def test_an_in_trial_proxy_install_takes_the_hosts_uploads(box, monkeypatch):
    """FF-INSTALL-ON-PT-R1 at 2c60d34f: the install's tree check walked the
    runtime folder before any upload was root's, so every in-trial proxy
    install from a non-root host stopped at ``config.yaml``."""
    from benchflow.providers import litellm_runtime as runtime

    _stub_proxy_install(monkeypatch)
    sandbox, runtime_dir, venv, requirements = await _uploaded_runtime(box)

    python = await runtime._ensure_sandbox_litellm(
        sandbox, venv_dir=venv, requirements_path=requirements
    )

    assert python == f"{venv}/bin/python"
    owners = _dx(box, f"find {runtime_dir} ! -user 0 -print")
    assert owners.returncode == 0 and owners.stdout == "", owners.stdout


@pytest.mark.asyncio
async def test_a_runtime_folder_another_user_owns_is_refused_before_any_chown(
    box, monkeypatch
):
    """The folder is checked before its files are made root's: a runtime
    folder another user owns could hold anything, so nothing in it is taken."""
    from benchflow.providers import litellm_runtime as runtime

    _stub_proxy_install(monkeypatch)
    sandbox, runtime_dir, venv, requirements = await _uploaded_runtime(box)
    handed = _dx(box, f"chown 1000:1000 {runtime_dir}")
    assert handed.returncode == 0, handed.stderr

    with pytest.raises(RuntimeError, match="the proxy runtime folder"):
        await runtime._ensure_sandbox_litellm(
            sandbox, venv_dir=venv, requirements_path=requirements
        )

    config = _dx(box, f"stat -c %u {runtime_dir}/config.yaml")
    assert config.stdout.strip() == "1000"


def test_kit_python_supports_lib64_without_using_the_live_standard_library():
    """Fedora's isolated Python keeps its lib64 layout."""
    image = os.environ.get("BENCHFLOW_TEST_LIB64_IMAGE")
    if not image or not _docker_ready():
        pytest.skip(
            "BENCHFLOW_TEST_LIB64_IMAGE with Python and a Docker daemon required"
        )
    name = "en298-lib64-" + uuid.uuid4().hex[:8]
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--network",
            "none",
            image,
            "sleep",
            "600",
        ],
        capture_output=True,
        check=True,
    )
    try:
        asyncio.run(_setup_only(_sandbox(name)))
        moved = _dx(
            name,
            'for d in /usr/lib64/python3.*; do [ ! -d "$d" ] || mv "$d" "$d.saved"; done',
        )
        assert moved.returncode == 0, moved.stderr
        result = _dx(
            name,
            "/opt/runtime/rt/py/python3 -c 'import sys, json, tarfile, hashlib; print(sys.prefix, sys.flags.isolated, sys.flags.no_site)' ",
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.split() == ["/opt/runtime/rt", "1", "1"]
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@pytest.mark.asyncio
async def test_agent_visible_env_and_managed_paths_are_neutral(box):
    """Agent-visible env and managed paths stay neutral across the real runtime seal and privilege drop."""
    from benchflow.agents.visible_runtime import visible_env, visible_launch
    from benchflow.sandbox.lockdown import build_priv_drop_cmd

    sandbox = await _setup(box)
    command = visible_launch(
        "env; find /opt /run /var/lib -maxdepth 2 2>/dev/null; "
        "find / -maxdepth 1 2>/dev/null; true"
    )
    result = await sandbox.exec(
        build_priv_drop_cmd(command, "agent"),
        env=visible_env({"BENCHFLOW_PROVIDER_MODEL": "model-under-test"}),
    )
    assert result.return_code == 0, result.stderr
    assert "benchflow" not in result.stdout.lower()
    assert "RUNTIME_PROVIDER_MODEL=model-under-test" in result.stdout
    assert "/opt/runtime" in result.stdout
