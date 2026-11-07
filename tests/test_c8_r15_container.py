"""PT round 15, in real containers (VM Docker): reviews of round 14.

The clean-up before the verifier now compares each start-up hook file
(``*.pth``, ``sitecustomize.py``, ``usercustomize.py``) with the pre-agent
record of every recorded search folder and removes only the new or changed
ones:

- PT-R14-B2 (review b): an honest uv project venv in the workspace (uv's
  documented Docker layout) keeps its editable-install ``.pth``.
- PT-R14-A-1 (review a): a conda env off PATH (SWE-bench's ``testbed``) and
  a ``/usr/local`` workspace lose the hook the agent wrote.

Each case has a live control: the agent's hook does run as root before the
clean-up. Skipped without a Docker daemon; the uv image is built here (the
test host's uv, network at build time only) and skipped when it cannot be.
"""

from __future__ import annotations

import base64
import shutil
import subprocess
import uuid

import pytest

from tests.test_c8_protected_runtime_container import (
    _docker_ready,
    _dx,
    _fresh,
    _marks,
    _sandbox,
)

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
]

_ROOT_HOOK = (
    "import os, sys\n"
    "if os.geteuid() == 0:\n"
    "    os.makedirs('/tmp/marks', exist_ok=True)\n"
    "    open('/tmp/marks/sitecustomize', 'w').write(' '.join(sys.orig_argv))\n"
)
# A .pth line that starts with "import" runs at start-up.
_ROOT_PTH = (
    "import os; os.geteuid() == 0 and os.makedirs('/tmp/marks/agent-pth', "
    "exist_ok=True)\n"
)


def _plant(name: str, path: str, text: str) -> None:
    data = base64.b64encode(text.encode()).decode()
    done = _dx(name, f"echo {data} | base64 -d > '{path}'", user="agent")
    assert done.returncode == 0, done.stderr


async def _clean(sandbox, workspace: str) -> None:
    from benchflow.sandbox.lockdown import cleanup_verifier_python_hooks

    await cleanup_verifier_python_hooks(sandbox, None, workspace=workspace, user="root")


# --- PT-R14-B2: uv's documented Docker layout --------------------------------

UV_IMAGE = "pt-r15-uv-project:test"
UV_DOCKERFILE = """FROM python:3.12-slim
LABEL target="PT-r15"
COPY uv /bin/uv
WORKDIR /app
RUN mkdir -p src/dummy_proj \\
 && printf 'MARK = "bf-proj-ok"\\n' > src/dummy_proj/__init__.py \\
 && printf '[project]\\nname = "dummy-proj"\\nversion = "0.1"\\nrequires-python = ">=3.12"\\n\\n[build-system]\\nrequires = ["hatchling"]\\nbuild-backend = "hatchling.build"\\n' > pyproject.toml \\
 && uv lock && uv sync --locked \\
 && useradd -m -u 1000 agent
ENV PATH="/app/.venv/bin:$PATH"
"""


@pytest.fixture(scope="module")
def uv_project_image(tmp_path_factory):
    if not _docker_ready():
        pytest.skip("docker daemon not running")
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("needs uv on the test host to build the image")
    context = tmp_path_factory.mktemp("uv-image")
    shutil.copy2(uv, context / "uv")
    (context / "Dockerfile").write_text(UV_DOCKERFILE)
    built = subprocess.run(
        ["docker", "build", "-q", "-t", UV_IMAGE, str(context)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if built.returncode != 0:
        pytest.skip(f"cannot build the uv project image: {built.stderr[-300:]}")
    yield UV_IMAGE


@pytest.mark.asyncio
async def test_a_uv_project_venv_in_the_workspace_keeps_its_editable_pth(
    uv_project_image,
):
    """PT-R14-B2, fails on 899eca7e: the clean-up removed every .pth in a
    venv under the workspace, the honest editable install's included, and
    ``import dummy_proj`` failed. Now only the agent's new hooks go."""
    from benchflow.sandbox.lockdown import setup_sandbox_user

    name = "PT-c8-r15-" + uuid.uuid4().hex[:8]
    subprocess.run(
        ["docker", "run", "-d", "--name", name, "--label", "PT=1", "--network",
         "none", uv_project_image, "sleep", "900"],
        check=True,
        capture_output=True,
    )  # fmt: skip
    site = "/app/.venv/lib/python3.12/site-packages"
    try:
        honest = _dx(name, f"ls {site} | grep -E '\\.pth$'").stdout.split()
        assert honest, "the uv sync wrote no editable .pth"
        sandbox = _sandbox(name)
        await setup_sandbox_user(sandbox, "agent", "/app", guard_daemon=False)
        _plant(name, "/app/src/sitecustomize.py", _ROOT_HOOK)
        _dx(name, f"chmod o+w {site}")  # so the agent can add a .pth too
        _plant(name, f"{site}/zz-agent.pth", _ROOT_PTH)
        # Control: the venv's python, as test.sh starts it, runs both.
        _dx(name, "/app/.venv/bin/python -c pass")
        assert sorted(_marks(name)) == ["agent-pth", "sitecustomize"]
        _dx(name, "rm -rf /tmp/marks")

        await _clean(sandbox, "/app")

        left = _dx(name, f"ls {site} | grep -E '\\.pth$'").stdout.split()
        assert left == honest, (left, honest)
        assert _dx(name, "test -e /app/src/sitecustomize.py").returncode == 1
        ran = _dx(
            name,
            "cd /tmp && /app/.venv/bin/python -c "
            "'import dummy_proj; print(dummy_proj.MARK)'",
        )
        assert ran.stdout.strip() == "bf-proj-ok", ran.stderr
        assert _marks(name) == []
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


# --- PT-R14-A-1: a conda env off PATH, and a /usr/local workspace -----------

_CONDA_PREP = (
    "/usr/local/bin/python3 -m venv --without-pip /opt/miniconda3 && "
    "mkdir /opt/miniconda3/conda-meta && "
    "/usr/local/bin/python3 -m venv --without-pip /opt/miniconda3/envs/testbed && "
    "echo /testbed > "
    "/opt/miniconda3/envs/testbed/lib/python3.12/site-packages/testbed.pth && "
    "useradd -m -u 1000 agent && mkdir -p /testbed && chown 1000 /testbed"
)
_CONDA_PATH = "PATH=/opt/miniconda3/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


@pytest.mark.asyncio
async def test_a_conda_env_off_path_loses_the_agents_hook():
    """PT-R14-A-1 (input 1), fails on 899eca7e: SWE-bench's testbed env is
    off PATH (test.sh runs ``conda activate testbed``), so its folders had
    no record and the agent's /testbed/sitecustomize.py ran as root. The
    conda base on PATH now brings its envs' records; the env's honest .pth
    naming /testbed stays."""
    from benchflow.sandbox.lockdown import setup_sandbox_user

    testbed = "/opt/miniconda3/envs/testbed/bin/python3"
    with _fresh(_CONDA_PREP, env=(_CONDA_PATH,)) as name:
        sandbox = _sandbox(name)
        await setup_sandbox_user(sandbox, "agent", "/testbed", guard_daemon=False)
        _plant(name, "/testbed/sitecustomize.py", _ROOT_HOOK)
        _dx(name, f"{testbed} -c pass")  # control
        assert _marks(name) == ["sitecustomize"]
        _dx(name, "rm -rf /tmp/marks")

        await _clean(sandbox, "/testbed")

        assert _dx(name, "test -e /testbed/sitecustomize.py").returncode == 1
        pth = "/opt/miniconda3/envs/testbed/lib/python3.12/site-packages/testbed.pth"
        assert _dx(name, f"test -e {pth}").returncode == 0
        _dx(name, f"{testbed} -c pass")
        assert _marks(name) == []


@pytest.mark.asyncio
async def test_a_usr_local_workspace_loses_the_agents_pth_and_keeps_the_images():
    """PT-R14-A-1 (input 2), fails on 899eca7e: root's own
    /usr/local/lib/python3.12/site-packages was skipped by prefix, so a .pth
    the agent wrote there (a /usr/local workspace hands it that folder) ran
    as root. It is now compared with its pre-agent record like any other."""
    from benchflow.sandbox.lockdown import setup_sandbox_user

    site = "/usr/local/lib/python3.12/site-packages"
    prep = (
        "useradd -m -u 1000 agent && mkdir -p /opt/vendor && "
        f"echo /opt/vendor > {site}/honest-vendor.pth"
    )
    with _fresh(prep) as name:
        sandbox = _sandbox(name)
        await setup_sandbox_user(sandbox, "agent", "/usr/local", guard_daemon=False)
        _dx(name, f"chmod o+w {site}")  # as the /usr/local grant can leave it
        _plant(name, f"{site}/zz-agent.pth", _ROOT_PTH)
        _dx(name, "/usr/local/bin/python3 -c pass")  # control
        assert _marks(name) == ["agent-pth"]
        _dx(name, "rm -rf /tmp/marks")

        await _clean(sandbox, "/usr/local")

        assert _dx(name, f"test -e {site}/zz-agent.pth").returncode == 1
        assert _dx(name, f"test -e {site}/honest-vendor.pth").returncode == 0
        _dx(name, "/usr/local/bin/python3 -c pass")
        assert _marks(name) == []
