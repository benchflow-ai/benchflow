"""PT round 14, in real containers (VM Docker): the b-review of round 13.

- PT-R13-B1: on an honest Debian distro-Python venv, the pytest plugin
  guard lands where the venv's own pytest imports it.
- PT-R13-B3: ``ENV PYTHONPATH=${PYTHONPATH}:/opt/vendor`` (PID 1 holds
  ``:/opt/vendor``) reaches a root task command as ``/opt/vendor``.

Skipped without a Docker daemon; B1's image is built here (network at
build time only) and skipped when it cannot be.
"""

from __future__ import annotations

import shutil
import subprocess
import uuid

import pytest

from tests.test_c8_protected_runtime_container import _docker_ready, _sandbox
from tests.test_c8_r13_container import VENDOR, _image_env_box

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
]

DEBIAN_VENV = "pt-r14-debian-venv:test"
DEBIAN_VENV_DOCKERFILE = """FROM debian:12
LABEL target="PT-r14"
RUN apt-get update && apt-get install -y --no-install-recommends python3-venv \\
 && rm -rf /var/lib/apt/lists/* \\
 && python3 -m venv /opt/venv && /opt/venv/bin/pip install -q pytest==8.3.3
"""


@pytest.fixture(scope="module")
def debian_venv_image():
    if not _docker_ready():
        pytest.skip("docker daemon not running")
    built = subprocess.run(
        ["docker", "build", "-q", "-t", DEBIAN_VENV, "-"],
        input=DEBIAN_VENV_DOCKERFILE,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if built.returncode != 0:
        pytest.skip(f"cannot build the debian:12 venv image: {built.stderr[-300:]}")
    yield DEBIAN_VENV


@pytest.mark.asyncio
async def test_the_guard_is_importable_by_a_debian_venvs_own_pytest(debian_venv_image):
    """Fails on e4ec66fa: under -I -S Debian's posix_local scheme and
    site.getsitepackages() name /usr/local/lib/python3.11/dist-packages,
    which the venv never searches; the venv's pytest then fails with
    "Error importing plugin" and the honest task is a verifier error."""
    from benchflow.sandbox.lockdown import (
        _install_pytest_plugin_guard,
        setup_sandbox_user,
    )

    name = "PT-c8-r14-" + uuid.uuid4().hex[:8]
    subprocess.run(
        ["docker", "run", "-d", "--name", name, "--label", "PT=1", "--network",
         "none", "-e", "PATH=/opt/venv/bin:/usr/sbin:/usr/bin:/sbin:/bin",
         "-w", "/srv", debian_venv_image, "sleep", "900"],
        check=True,
        capture_output=True,
    )  # fmt: skip
    try:
        subprocess.run(
            ["docker", "exec", name, "sh", "-c",
             "mkdir -p /srv /tmp/t && printf 'def test_ok():\\n    pass\\n' "
             "> /tmp/t/test_dummy.py"],
            check=True,
        )  # fmt: skip
        sandbox = _sandbox(name)
        await setup_sandbox_user(sandbox, "agent", "/srv", guard_daemon=False)
        _, flags, guarded, _ = await _install_pytest_plugin_guard(
            sandbox,
            "agent",
            "/srv",
            "",
            verifier_path="/opt/venv/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        )
        assert "/opt/venv/bin/python3" in guarded or "/opt/venv/bin/python" in guarded
        ran = subprocess.run(
            ["docker", "exec", "-w", "/tmp/t", name, "/opt/venv/bin/pytest",
             "-q", "-p", flags.split()[1], "test_dummy.py"],
            capture_output=True,
            text=True,
        )  # fmt: skip
        assert ran.returncode == 0, ran.stdout[-800:] + ran.stderr[-800:]
        assert "1 passed" in ran.stdout
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["s1", "plain"])
async def test_a_python_path_with_an_empty_entry_comes_back_without_it(shape):
    """PT-R13-B3, fails on e4ec66fa: the leading empty entry (the command's
    own folder) made the whole value refused, so the root oracle lost its
    vendor module. The empty entry is dropped; the rest comes back."""
    from benchflow.sandbox.lockdown import setup_sandbox_user

    env = ("PYTHONPATH=:/opt/vendor",)
    with _image_env_box(VENDOR, env, shape) as (_name, sandbox):
        await setup_sandbox_user(sandbox, "agent", "/srv", guard_daemon=False)
        with sandbox.task_command():
            result = await sandbox.exec(
                "python3 -c 'import bfvendor; print(bfvendor.MARK)' 2>&1; "
                'printf "PYTHONPATH=%s\\n" "${PYTHONPATH-unset}"',
                user="root",
            )
    assert result.stdout == "bf-vendor-ok\nPYTHONPATH=/opt/vendor\n", (
        result.stdout,
        result.stderr,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["s1", "plain"])
async def test_an_empty_loader_entry_is_dropped_wherever_it_is(shape):
    """PT-R13-A-1 then PT-R14-B1: ``ENV LD_LIBRARY_PATH=/opt/vendorlib:
    $LD_LIBRARY_PATH`` leaves a trailing ':' (the command's own folder for
    glibc). e4ec66fa passed it to root with the ':'; 899eca7e refused the
    whole value, so the honest library was lost. Root now gets the value
    without its empty entry, as for a language path list."""
    from benchflow.sandbox.lockdown import setup_sandbox_user

    env = ("LD_LIBRARY_PATH=/opt/vendorlib:", "PYTHONPATH=/opt/vendor:")
    prep = VENDOR + " && mkdir -p /opt/vendorlib"
    with _image_env_box(prep, env, shape) as (_name, sandbox):
        await setup_sandbox_user(sandbox, "agent", "/srv", guard_daemon=False)
        with sandbox.task_command():
            result = await sandbox.exec(
                'printf "%s|%s\\n" "${LD_LIBRARY_PATH-unset}" "${PYTHONPATH-unset}"',
                user="root",
            )
    assert result.stdout == "/opt/vendorlib|/opt/vendor\n", (
        result.stdout,
        result.stderr,
    )
    assert "not restoring" not in result.stderr


_VENV_PREP = (
    "/usr/local/bin/python3 -m venv --without-pip /opt/venv && "
    "echo /app > /opt/venv/lib/python3.12/site-packages/bf-app.pth && "
    "useradd -m -u 1000 agent && mkdir -p /app && chown 1000 /app"
)
_VENV_PATH = "PATH=/opt/venv/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_ROOT_HOOK = (
    "import os, sys\n"
    "if os.geteuid() == 0:\n"
    "    os.makedirs('/tmp/marks', exist_ok=True)\n"
    "    open('/tmp/marks/venv-sitecustomize', 'w').write(' '.join(sys.orig_argv))\n"
)


@pytest.mark.asyncio
async def test_a_hook_on_a_venvs_pth_folder_is_cleaned_for_the_venv_test_sh_runs():
    """PT-R13-A-2 (review a), fails on e4ec66fa: a root-owned .pth in the
    venv first on the image's PATH (an editable install) adds /app; the
    agent's /app/sitecustomize.py ran when test.sh's venv python started,
    as only root's python3 (/usr/local/bin) had its path captured. Now
    every interpreter on the image's PATH has a pre-agent record: the hook
    goes, the venv's own honest .pth stays, discovery on the verifier's
    PATH runs no hook."""
    import base64

    from benchflow.sandbox.lockdown import (
        _discover_pytest_plugins_cmd,
        cleanup_verifier_python_hooks,
        setup_sandbox_user,
    )
    from tests.test_c8_protected_runtime_container import _dx, _fresh, _marks

    with _fresh(_VENV_PREP, env=(_VENV_PATH,)) as name:
        sandbox = _sandbox(name)
        await setup_sandbox_user(sandbox, "agent", "/app", guard_daemon=False)
        hook = base64.b64encode(_ROOT_HOOK.encode()).decode()
        planted = _dx(
            name, f"echo {hook} | base64 -d > /app/sitecustomize.py", user="agent"
        )
        assert planted.returncode == 0, planted.stderr
        # Control: the venv's python, as test.sh starts it, runs the hook.
        _dx(name, "/opt/venv/bin/python -c pass")
        assert _marks(name) == ["venv-sitecustomize"]
        _dx(name, "rm -rf /tmp/marks")

        discovered = await sandbox.exec(
            _discover_pytest_plugins_cmd(
                ("/app",), verifier_path="/opt/venv/bin:/usr/bin:/bin"
            ),
            user="root",
        )
        assert discovered.return_code == 0, discovered.stderr
        await cleanup_verifier_python_hooks(
            sandbox, None, workspace="/app", user="root"
        )
        assert _dx(name, "test -e /app/sitecustomize.py").returncode == 1
        pth = "/opt/venv/lib/python3.12/site-packages/bf-app.pth"
        assert _dx(name, f"test -e {pth}").returncode == 0  # honest, root's
        _dx(name, "/opt/venv/bin/python -c pass")
        assert _marks(name) == []
