"""PT round 13, in a real container (``python:3.12-slim``, VM Docker).

The batch-165 review inputs, run
through BenchFlow's real code with the harness of
``test_c8_protected_runtime_container``: honest images, dummy markers.

- PT-C8-CONF-6: a root task command (oracle, test.sh) gets the image's
  start-up values the route clears, not just the loader's thirteen:
  ``PYTHONPATH``, ``TAR_OPTIONS``, ... A value naming a file the agent
  could write still stays unset for root.
- PT-C8-CONF-1: a root-owned ``.pth`` adds the workspace to the image
  Python's search path; the agent's ``sitecustomize.py`` there never runs
  as root in verifier preparation, and the clean-up removes it.
- PT-C8-CONF-7: a typed ``ld.so.conf`` entry (``/opt/vendor/lib=libc6``)
  is the folder ``/opt/vendor/lib``.

Skipped without a Docker daemon.
"""

from __future__ import annotations

import base64
import contextlib
import shutil
import subprocess
import uuid

import pytest

from tests.test_c8_protected_runtime_container import (
    IMAGE,
    WORKSPACE,
    _docker_ready,
    _dx,
    _fresh,
    _marks,
    _sandbox,
    _setup,
)

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
]

VENDOR = (
    "mkdir -p /opt/vendor && "
    "printf 'MARK = \"bf-vendor-ok\"\\n' > /opt/vendor/bfvendor.py && "
    "chmod -R go-w /opt/vendor"
)
_REPORT = (
    "python3 -c 'import bfvendor; print(bfvendor.MARK)' 2>&1; "
    'printf "TAR_OPTIONS=%s\\n" "${TAR_OPTIONS-unset}"; '
    'printf "copies=%s\\n" "$(env | grep -c "^_BF_IMAGE_")"'
)


# --- PT-C8-CONF-6: every cleared start-up value comes back for root ---------


@contextlib.contextmanager
def _image_env_box(prep: str, env: tuple[str, ...], shape: str):
    """A container whose image sets *env*, in the two shapes BenchFlow meets.

    ``s1``: S1's bootstrap (a container with host binds): the exec's own
    environment lacks the values, PID 1 holds them, and the daemon-owned
    receipt hands them to task commands under ``_BF_IMAGE_`` names.
    ``plain``: a container S1 did not bootstrap: Docker passes them straight
    into every exec.
    """
    if not _docker_ready():
        pytest.skip("docker daemon not running")
    name = "PT-c8-r13-" + uuid.uuid4().hex[:8]
    flags = [f for v in env for f in ("-e", v)] if shape == "plain" else []
    pid1 = ["sleep", "900"] if shape == "plain" else ["env", *env, "sleep", "900"]
    run = subprocess.run(
        ["docker", "run", "-d", "--name", name, "--label", "PT=1", "--network",
         "none", *flags, "-w", WORKSPACE, IMAGE, *pid1],
        capture_output=True,
        text=True,
    )  # fmt: skip
    if run.returncode != 0:
        pytest.skip(f"cannot start {IMAGE}: {run.stderr.strip()}")
    try:
        made = _dx(name, f"mkdir -p {WORKSPACE}; {prep or 'true'}")
        assert made.returncode == 0, made.stderr
        sandbox = _sandbox(name)
        if shape == "s1":
            sandbox._bootstrap_configs = {"main": {"Env": list(env)}}
        yield name, sandbox
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["s1", "plain"])
async def test_a_root_task_command_gets_the_images_python_path_and_options(shape):
    """Fails on 7342d55c (s1): ModuleNotFoundError and TAR_OPTIONS unset; the
    route cleared them and only the loader's thirteen names came back."""
    from benchflow.sandbox.lockdown import setup_sandbox_user

    env = ("PYTHONPATH=/opt/vendor", "TAR_OPTIONS=--no-same-owner")
    with _image_env_box(VENDOR, env, shape) as (_name, sandbox):
        await setup_sandbox_user(sandbox, "agent", WORKSPACE, guard_daemon=False)
        with sandbox.task_command():
            result = await sandbox.exec(_REPORT, user="root")
    lines = (result.stdout or "").splitlines()
    assert lines == ["bf-vendor-ok", "TAR_OPTIONS=--no-same-owner", "copies=0"], (
        result.stdout,
        result.stderr,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["s1", "plain"])
async def test_a_python_path_the_agent_can_write_is_not_restored_for_root(shape):
    """The same trust rule as the loader's, for every name: the workspace on
    PYTHONPATH, or a module NODE_OPTIONS requires from it, never reaches
    root's command. Fails on 7342d55c (plain): Docker passed them through."""
    from benchflow.sandbox.lockdown import setup_sandbox_user

    env = ("PYTHONPATH=/srv/lib", "NODE_OPTIONS=--require /srv/x.js")
    prep = "useradd -m -u 1000 agent && mkdir -p /srv/lib && chown 1000 /srv/lib"
    with _image_env_box(prep, env, shape) as (_name, sandbox):
        await setup_sandbox_user(sandbox, "agent", WORKSPACE, guard_daemon=False)
        with sandbox.task_command():
            result = await sandbox.exec(
                'printf "%s|%s\\n" "${PYTHONPATH-unset}" "${NODE_OPTIONS-unset}"',
                user="root",
            )
    assert result.stdout == "unset|unset\n", result
    assert "not restoring PYTHONPATH" in result.stderr
    assert "not restoring NODE_OPTIONS" in result.stderr


# --- PT-C8-CONF-1: the image's .pth-added folders are covered ---------------

_PTH_PREP = (
    "useradd -m -u 1000 agent && mkdir -p /app && chown 1000 /app && "
    "echo /app > /usr/local/lib/python3.12/site-packages/bf-app.pth"
)
_HOOK = (
    "import os, sys\n"
    "if os.geteuid() == 0:\n"
    "    os.makedirs('/tmp/marks', exist_ok=True)\n"
    "    open('/tmp/marks/root-sitecustomize', 'w').write(' '.join(sys.orig_argv))\n"
)


@pytest.mark.asyncio
async def test_a_sitecustomize_on_a_pth_added_folder_never_runs_as_root():
    """PT-C8-CONF-1 (b165): plugin discovery and the guard's install do not
    start the image Python with site hooks, and the clean-up removes the
    hook the image's .pth puts on the path. The control shows the hook is
    live for an ordinary root python3."""
    from benchflow.sandbox.lockdown import (
        _discover_pytest_plugins_cmd,
        cleanup_verifier_python_hooks,
    )

    with _fresh(_PTH_PREP) as name:
        sandbox = await _setup(name, workspace="/app")
        hook = base64.b64encode(_HOOK.encode()).decode()
        planted = _dx(
            name, f"echo {hook} | base64 -d > /app/sitecustomize.py", user="agent"
        )
        assert planted.returncode == 0, planted.stderr
        # Control: the image's own interpreter, started as root, runs it.
        _dx(name, "/usr/local/bin/python3 -c pass")
        assert _marks(name) == ["root-sitecustomize"]
        _dx(name, "rm -rf /tmp/marks")

        discovered = await sandbox.exec(
            _discover_pytest_plugins_cmd(("/app",)), user="root"
        )
        assert discovered.return_code == 0, discovered.stderr
        assert _marks(name) == []
        await cleanup_verifier_python_hooks(
            sandbox, None, workspace="/app", user="root"
        )
        assert _marks(name) == []
        assert _dx(name, "test -e /app/sitecustomize.py").returncode == 1


# --- PT-C8-CONF-7: typed loader entries --------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agents", "starts"),
    [(None, True), ("/opt/vendor/lib=libc6", False), ("/opt/vendor/lib", False)],
)
async def test_a_typed_loader_entry_is_checked_in_both_readings(agents, starts):
    """PT-C8-CONF-7 (b165) with PT-R13-B2: glibc 2.36 and earlier read
    ``/opt/vendor/lib=libc6`` as ``/opt/vendor/lib``; 2.37 and later
    (Ubuntu 24.04, Debian 13) as the literal folder. Whichever the image's
    glibc is, both must be root's alone: an honest image whose two folders
    are root's starts; one where the agent owns either is refused, naming it.
    (Round 13's version of this test expected the literal agent-owned folder
    to pass, which is wrong from glibc 2.37 on.)"""
    prep = (
        "useradd -m -u 1000 agent && mkdir -p /opt/vendor/lib "
        "'/opt/vendor/lib=libc6' && "
        "echo '/opt/vendor/lib=libc6' > /etc/ld.so.conf.d/zz-typed.conf"
    )
    if agents:
        prep += f" && chown 1000 '{agents}'"
    with _fresh(prep) as name:
        if starts:
            sandbox = await _setup(name)
            assert sandbox.runtime_kit is not None
        else:
            with pytest.raises(RuntimeError, match=f"searches {agents}"):
                await _setup(name)
