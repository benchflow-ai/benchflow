"""PT round 16, in real containers (VM Docker): one rule for start-up hooks.

Honest images, BenchFlow's real kit build (``setup_sandbox_user``, which
takes the pre-agent record) and its real clean-up before the verifier
(``cleanup_verifier_python_hooks``). The agent's hooks are dummy markers:
run as root, they make a folder under ``/tmp/marks`` and nothing else.

- PT-R15-B1: every form of ``sitecustomize`` in a uv workspace venv (a
  package folder, a sourceless ``.pyc``, an extension-module name), and a
  ``usercustomize`` package in root's user site, are removed.
- PT-R15-B3: ``/app/model.pth`` saved by the oracle of a flat-layout
  editable project stays.
- PT-R15-B2: a kept service's descriptor on the honest editable ``.pth``
  writes into a file the verifier's Python no longer reads.
- Controls: the uv venv's editable ``.pth``, a conda env's ``.pth``, Debian
  system Python's ``sitecustomize.py`` and a ``/usr/local`` workspace's own
  ``.pth`` are kept.

Each removal case has a live control: the agent's hook runs as root first.
Skipped without a Docker daemon; the images are built here (the test host's
uv, network at build time only) and skipped when they cannot be.
"""

from __future__ import annotations

import base64
import contextlib
import shutil
import subprocess
import uuid

import pytest

from tests.test_c8_protected_runtime_container import _docker_ready, _dx, _sandbox

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
]

_ROOT_HOOK = (
    "import os\n"
    "if os.geteuid() == 0:\n"
    "    os.makedirs('/tmp/marks/{name}', exist_ok=True)\n"
)
SLIM = "python:3.12-slim"


def _marks(name: str) -> list[str]:
    return sorted(_dx(name, "ls /tmp/marks 2>/dev/null || true").stdout.split())


def _plant(name: str, path: str, text: str) -> None:
    data = base64.b64encode(text.encode()).decode()
    done = _dx(name, f"echo {data} | base64 -d > '{path}'", user="agent")
    assert done.returncode == 0, done.stderr


@contextlib.contextmanager
def _box(image: str, prep: str = "", *, env: tuple[str, ...] = ()):
    """A container of *image* named PT-R16-*, *prep* run as root first."""
    if not _docker_ready():
        pytest.skip("docker daemon not running")
    name = "PT-R16-" + uuid.uuid4().hex[:8]
    flags = [f for value in env for f in ("-e", value)]
    run = subprocess.run(
        ["docker", "run", "-d", "--name", name, "--label", "target=PT-R16",
         "--network", "none", *flags, image, "sleep", "900"],
        capture_output=True,
        text=True,
    )  # fmt: skip
    if run.returncode != 0:
        pytest.skip(f"cannot start {image}: {run.stderr.strip()}")
    try:
        if prep:
            made = _dx(name, prep)
            assert made.returncode == 0, made.stderr
        yield name
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


async def _setup(name: str, workspace: str):
    from benchflow.sandbox.lockdown import setup_sandbox_user

    sandbox = _sandbox(name)
    await setup_sandbox_user(sandbox, "agent", workspace, guard_daemon=False)
    return sandbox


async def _clean(sandbox, workspace: str) -> None:
    from benchflow.sandbox.lockdown import cleanup_verifier_python_hooks

    await cleanup_verifier_python_hooks(sandbox, None, workspace=workspace, user="root")


def _build(tmp_path_factory, tag: str, dockerfile: str, uv: bool = False) -> str:
    if not _docker_ready():
        pytest.skip("docker daemon not running")
    context = tmp_path_factory.mktemp(tag.split(":")[0])
    if uv:
        found = shutil.which("uv")
        if found is None:
            pytest.skip("needs uv on the test host to build the image")
        shutil.copy2(found, context / "uv")
    (context / "Dockerfile").write_text(dockerfile)
    built = subprocess.run(
        ["docker", "build", "-q", "-t", tag, str(context)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if built.returncode != 0:
        pytest.skip(f"cannot build {tag}: {built.stderr[-300:]}")
    return tag


# uv's documented Docker layout: `uv sync` into /app/.venv, the venv first on
# PATH. SRC is a src layout; FLAT puts the package at /app/dummy_proj, so the
# editable .pth names /app itself.
_UV = """FROM python:3.12-slim
LABEL target="PT-R16"
COPY uv /bin/uv
ENV UV_PYTHON_INSTALL_DIR=/app/.python UV_PYTHON_PREFERENCE=only-managed
WORKDIR /app
RUN mkdir -p {pkg} \\
 && printf 'MARK = "bf-proj-ok"\\n' > {pkg}/__init__.py \\
 && printf '[project]\\nname = "dummy-proj"\\nversion = "0.1"\\nrequires-python = ">=3.12"\\n\\n[build-system]\\nrequires = ["hatchling"]\\nbuild-backend = "hatchling.build"\\n{extra}' > pyproject.toml \\
 && uv python install 3.12 && uv lock && uv sync --locked \\
 && useradd -m -u 1000 agent
ENV PATH="/app/.venv/bin:$PATH"
"""
UV_SRC = _UV.format(pkg="src/dummy_proj", extra="")
UV_FLAT = _UV.format(
    pkg="dummy_proj",
    extra='\\n[tool.hatch.build.targets.wheel]\\npackages = ["dummy_proj"]\\n',
)
DEBIAN = """FROM debian:12
LABEL target="PT-R16"
RUN apt-get update && apt-get install -y --no-install-recommends python3 \\
 && rm -rf /var/lib/apt/lists/* && useradd -m -u 1000 agent && mkdir /work \\
 && chown 1000 /work
"""
VSITE = "/app/.venv/lib/python3.12/site-packages"


@pytest.fixture(scope="module")
def uv_src(tmp_path_factory):
    return _build(tmp_path_factory, "pt-r16-uv-src:test", UV_SRC, uv=True)


@pytest.fixture(scope="module")
def uv_flat(tmp_path_factory):
    return _build(tmp_path_factory, "pt-r16-uv-flat:test", UV_FLAT, uv=True)


@pytest.fixture(scope="module")
def debian(tmp_path_factory):
    return _build(tmp_path_factory, "pt-r16-debian:test", DEBIAN)


def _pths(name: str) -> list[str]:
    return sorted(_dx(name, f"ls {VSITE} | grep -E '\\.pth$'").stdout.split())


def _imports(name: str) -> str:
    ran = _dx(
        name,
        "cd /tmp && /app/.venv/bin/python -c 'import dummy_proj; print(dummy_proj.MARK)'",
    )
    return ran.stdout.strip() or ran.stderr


# --- PT-R15-B1 ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_sitecustomize_form_in_a_uv_venv_is_removed(uv_src):
    """PT-R15-B1, fails on daec118b: a sitecustomize package folder, a
    sourceless sitecustomize.pyc and an extension-module name in the venv's
    site-packages were not hooks to the clean-up, so the
    venv's Python ran the package and the .pyc as root. The honest editable
    .pth stays (control)."""
    with _box(uv_src) as name:
        honest = _pths(name)
        assert honest, "the uv sync wrote no editable .pth"
        sandbox = await _setup(name, "/app")
        _dx(name, f"chmod o+w {VSITE}")  # as the /app grant can leave it
        _dx(name, f"mkdir {VSITE}/sitecustomize", user="agent")
        _plant(
            name, f"{VSITE}/sitecustomize/__init__.py", _ROOT_HOOK.format(name="pkg")
        )
        _plant(name, "/tmp/hook.py", _ROOT_HOOK.format(name="pyc"))
        made = _dx(
            name,
            "/app/.venv/bin/python -I -c 'import py_compile; "
            f'py_compile.compile("/tmp/hook.py", cfile="{VSITE}/sitecustomize.pyc")\'',
            user="agent",
        )
        assert made.returncode == 0, made.stderr
        # Control: the package runs; with it gone, the sourceless .pyc runs.
        _dx(name, "/app/.venv/bin/python -c pass")
        _dx(name, f"mv {VSITE}/sitecustomize /tmp/pkg-aside")
        _dx(name, "/app/.venv/bin/python -c pass")
        _dx(name, f"mv /tmp/pkg-aside {VSITE}/sitecustomize")
        assert _marks(name) == ["pkg", "pyc"]
        # The import system tries an extension module before either (a
        # dummy here: no compiler in the image, so no live control).
        _plant(name, f"{VSITE}/sitecustomize.cpython-312-x86_64-linux-gnu.so", "dummy")
        _dx(name, "rm -rf /tmp/marks")

        await _clean(sandbox, "/app")

        for path in (
            f"{VSITE}/sitecustomize",
            f"{VSITE}/sitecustomize.pyc",
            f"{VSITE}/sitecustomize.cpython-312-x86_64-linux-gnu.so",
        ):
            assert _dx(name, f"test -e {path}").returncode == 1, path
        assert _pths(name) == honest
        assert _imports(name) == "bf-proj-ok"
        assert _marks(name) == []


@pytest.mark.asyncio
async def test_a_usercustomize_package_in_roots_user_site_is_removed():
    """PT-R15-B1, fails on daec118b: a usercustomize package in root's user
    site ran at root's Python start-up and survived the clean-up."""
    user_site = "/root/.local/lib/python3.12/site-packages"
    prep = "useradd -m -u 1000 agent && mkdir -p /srv && chown 1000 /srv"
    with _box(SLIM, prep) as name:
        sandbox = await _setup(name, "/srv")
        # As a workspace that holds root's HOME can leave it.
        _dx(
            name,
            f"mkdir -p {user_site} && chmod 755 /root && chown -R 1000 /root/.local",
        )
        _dx(name, f"mkdir {user_site}/usercustomize", user="agent")
        _plant(
            name,
            f"{user_site}/usercustomize/__init__.py",
            _ROOT_HOOK.format(name="user"),
        )
        _dx(name, "HOME=/root /usr/local/bin/python3 -c pass")  # control
        assert _marks(name) == ["user"]
        _dx(name, "rm -rf /tmp/marks")

        await _clean(sandbox, "/srv")

        assert _dx(name, f"test -e {user_site}/usercustomize").returncode == 1
        _dx(name, "HOME=/root /usr/local/bin/python3 -c pass")
        assert _marks(name) == []


# --- PT-R15-B3 ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_model_pth_saved_in_a_flat_editable_project_stays(uv_flat):
    """PT-R15-B3, fails on daec118b: the flat layout's editable .pth puts
    /app on the venv's search path, so /app was a recorded folder and the
    oracle's /app/model.pth (data: site never reads a .pth outside a site
    folder) was removed. A .pth the agent adds to site-packages still goes."""
    with _box(uv_flat) as name:
        assert _imports(name) == "bf-proj-ok"
        sandbox = await _setup(name, "/app")
        weights = base64.b64encode(b"PK\x03\x04 dummy weights").decode()
        _dx(name, f"echo {weights} | base64 -d > /app/model.pth", user="agent")
        _dx(name, f"chmod o+w {VSITE}")
        _plant(name, f"{VSITE}/zz-agent.pth", "import os; os.geteuid() == 0 and "
               "os.makedirs('/tmp/marks/agent-pth', exist_ok=True)\n")  # fmt: skip
        _dx(name, "/app/.venv/bin/python -c pass")  # control
        assert _marks(name) == ["agent-pth"]
        _dx(name, "rm -rf /tmp/marks")

        await _clean(sandbox, "/app")

        assert _dx(name, "base64 /app/model.pth").stdout.strip() == weights
        assert _dx(name, f"test -e {VSITE}/zz-agent.pth").returncode == 1
        assert _imports(name) == "bf-proj-ok"
        assert _marks(name) == []


# --- PT-R15-B2 ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_kept_services_descriptor_never_reaches_the_kept_pth(uv_src):
    """PT-R15-B2, fails on daec118b: the honest editable .pth matched its
    record and was kept as the same inode, so a process from the agent's
    turn holding it open wrote an import line the verifier's Python ran as
    root. The kept hook is now a fresh root-owned copy."""
    with _box(uv_src) as name:
        honest = _pths(name)
        sandbox = await _setup(name, "/app")
        pth = f"{VSITE}/{honest[0]}"
        before = _dx(name, f"cat {pth}").stdout
        line = "import os; os.makedirs('/tmp/marks/held-pth', exist_ok=True)\\n"
        held = subprocess.run(
            ["docker", "exec", "-d", "-u", "agent", name, "sh", "-c",
             f"exec 3<>'{pth}'; : > /tmp/held; "
             "while [ ! -e /tmp/go ]; do sleep 0.1; done; "
             f"printf '{line}' >&3; : > /tmp/wrote"],
            capture_output=True,
        )  # fmt: skip
        assert held.returncode == 0, held.stderr
        _dx(name, "for i in $(seq 50); do [ -e /tmp/held ] && break; sleep 0.1; done")
        assert _dx(name, "test -e /tmp/held").returncode == 0

        await _clean(sandbox, "/app")

        _dx(
            name,
            ": > /tmp/go; for i in $(seq 50); do [ -e /tmp/wrote ] && break; sleep 0.1; done",
        )
        assert _dx(name, "test -e /tmp/wrote").returncode == 0
        # What the verifier's Python reads is the honest file, root's.
        assert _imports(name) == "bf-proj-ok"
        assert _marks(name) == []
        assert _dx(name, f"cat {pth}").stdout == before
        assert _dx(name, f"stat -c %u {pth}").stdout.strip() == "0"


# --- controls ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_conda_env_keeps_its_pth_and_loses_the_agents_package():
    """Control (PT-R14-A-1's layout): SWE-bench's testbed env off PATH keeps
    its honest .pth naming /testbed; a sitecustomize package the agent puts
    in /testbed goes (PT-R15-A-1, fails on daec118b)."""
    prep = (
        "/usr/local/bin/python3 -m venv --without-pip /opt/miniconda3 && "
        "mkdir /opt/miniconda3/conda-meta && "
        "/usr/local/bin/python3 -m venv --without-pip /opt/miniconda3/envs/testbed && "
        "echo /testbed > "
        "/opt/miniconda3/envs/testbed/lib/python3.12/site-packages/testbed.pth && "
        "useradd -m -u 1000 agent && mkdir -p /testbed && chown 1000 /testbed"
    )
    path = "PATH=/opt/miniconda3/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    testbed = "/opt/miniconda3/envs/testbed/bin/python3"
    with _box(SLIM, prep, env=(path,)) as name:
        sandbox = await _setup(name, "/testbed")
        _dx(name, "mkdir /testbed/sitecustomize", user="agent")
        _plant(
            name, "/testbed/sitecustomize/__init__.py", _ROOT_HOOK.format(name="pkg")
        )
        _dx(name, f"{testbed} -c pass")  # control
        assert _marks(name) == ["pkg"]
        _dx(name, "rm -rf /tmp/marks")

        await _clean(sandbox, "/testbed")

        assert _dx(name, "test -e /testbed/sitecustomize").returncode == 1
        pth = "/opt/miniconda3/envs/testbed/lib/python3.12/site-packages/testbed.pth"
        assert _dx(name, f"cat {pth}").stdout.strip() == "/testbed"
        _dx(name, f"{testbed} -c pass")
        assert _marks(name) == []


@pytest.mark.asyncio
async def test_debian_system_pythons_sitecustomize_is_kept(debian):
    """Control: Debian's /usr/lib/python3.11/sitecustomize.py, a link to
    /etc/python3.11/sitecustomize.py, keeps its bytes for root's python3
    (as a fresh root-owned file); /etc's file is untouched."""
    hook = "/usr/lib/python3.11/sitecustomize.py"
    with _box(debian) as name:
        before = _dx(name, f"sha256sum < {hook}").stdout
        etc = _dx(name, "sha256sum < /etc/python3.11/sitecustomize.py").stdout
        assert before and before == etc
        sandbox = await _setup(name, "/work")
        _plant(name, "/work/model.pth", "dummy weights\n")

        await _clean(sandbox, "/work")

        assert _dx(name, f"sha256sum < {hook}").stdout == before
        assert _dx(name, "sha256sum < /etc/python3.11/sitecustomize.py").stdout == etc
        assert _dx(name, f"stat -c %u {hook}").stdout.strip() == "0"
        assert _dx(name, "python3 -c 'import sitecustomize'").returncode == 0
        assert _dx(name, "test -e /work/model.pth").returncode == 0


@pytest.mark.asyncio
async def test_a_usr_local_workspace_keeps_its_own_pth():
    """Control (PT-R14-A-1 input 2): the image's own .pth in
    /usr/local/lib/python3.12/site-packages stays; the agent's goes."""
    site = "/usr/local/lib/python3.12/site-packages"
    prep = (
        "useradd -m -u 1000 agent && mkdir -p /opt/vendor && "
        f"echo /opt/vendor > {site}/honest-vendor.pth"
    )
    with _box(SLIM, prep) as name:
        sandbox = await _setup(name, "/usr/local")
        _dx(name, f"chmod o+w {site}")
        _dx(name, f"mkdir {site}/sitecustomize", user="agent")
        _plant(name, f"{site}/sitecustomize/__init__.py", _ROOT_HOOK.format(name="pkg"))
        _dx(name, "/usr/local/bin/python3 -c pass")  # control
        assert _marks(name) == ["pkg"]
        _dx(name, "rm -rf /tmp/marks")

        await _clean(sandbox, "/usr/local")

        assert _dx(name, f"test -e {site}/sitecustomize").returncode == 1
        assert (
            _dx(name, f"cat {site}/honest-vendor.pth").stdout.strip() == "/opt/vendor"
        )
        _dx(name, "/usr/local/bin/python3 -c pass")
        assert _marks(name) == []


# --- round 2: a folder only PYTHONPATH puts on the search path (PT-R16-B-1) ---

PYTHONPATH_APP = """FROM python:3.12-slim
LABEL target="PT-R16"
ENV PYTHONPATH=/app
RUN useradd -m -u 1000 agent && mkdir /app && chown 1000 /app
"""


@pytest.fixture(scope="module")
def pythonpath_app(tmp_path_factory):
    return _build(tmp_path_factory, "pt-r16-pythonpath-app:test", PYTHONPATH_APP)


@pytest.mark.asyncio
async def test_an_image_pythonpath_folder_loses_the_agents_hook(pythonpath_app):
    """PT-R16-B-1/B-2, fails on 91c603c6: with ``ENV PYTHONPATH=/app`` the
    verifier's Python searches /app, but the kit's ``-I`` record and the
    clean-up's folders left it out, so a sitecustomize package the agent put
    there ran as root. The honest /app/model.pth stays (data: /app is no
    site folder)."""
    with _box(pythonpath_app) as name:
        sandbox = await _setup(name, "/app")
        _dx(name, "mkdir /app/sitecustomize", user="agent")
        _plant(name, "/app/sitecustomize/__init__.py", _ROOT_HOOK.format(name="pkg"))
        weights = base64.b64encode(b"PK\x03\x04 dummy weights").decode()
        _dx(name, f"echo {weights} | base64 -d > /app/model.pth", user="agent")
        _dx(name, "/usr/local/bin/python3 -c pass")  # control: PYTHONPATH=/app
        assert _marks(name) == ["pkg"]
        _dx(name, "rm -rf /tmp/marks")

        await _clean(sandbox, "/app")

        assert _dx(name, "test -e /app/sitecustomize").returncode == 1
        assert _dx(name, "base64 /app/model.pth").stdout.strip() == weights
        _dx(name, "/usr/local/bin/python3 -c pass")
        assert _marks(name) == []
