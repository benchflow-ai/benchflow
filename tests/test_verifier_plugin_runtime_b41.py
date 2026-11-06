"""Guards V2-REDESIGN-R3 (b41, Astra and Sol) through hardening, the guard and pytest.

At 3571f2c5 the guard judged plugin bytes that sit where the agent can write,
and each review found another way past that: a symlink plus a parent spelling
(R3-1), root installs into the system's site-packages (Astra R3-2), byte-code
caches of image code (R3-3), a shadowed hashlib (Sol R3-2), a ``.pth`` that
turns autoload back on (Astra R3-4). Now the verifier's plugins
come from BenchFlow's protected runtime and nothing test.sh installed after
the agent stopped loads. These tests use only what the guard's scoring
harness has had since 3571f2c5, so they run unchanged on that base.
"""

import importlib.util
import py_compile
import sys
from pathlib import Path

import pytest

from tests.test_verifier_plugin_published import venv_install
from tests.test_verifier_uv_state import (
    CTRF_PLUGIN,
    FAILING,
    hostile_ctrf,
    install_plugin,
    refused_scored_zero,
    score,
)

IMAGE_PLUGIN = "def pytest_configure(config):\n    config.image_plugin_loaded = True\n"


def cache_for(source: Path, prefix: Path) -> Path:
    """Where CPython reads the byte code of *source* under ``-X pycache_prefix``."""
    previous = sys.pycache_prefix
    sys.pycache_prefix = str(prefix)
    try:
        return Path(importlib.util.cache_from_source(str(source)))
    finally:
        sys.pycache_prefix = previous


def hostile_cache(tmp_path, proof, target, dfile, source_text=None):
    """An unchecked-hash cache at *target* holding a pass-everything plugin."""
    hostile = tmp_path / "compiled-from" / Path(dfile).name
    hostile.parent.mkdir(exist_ok=True)
    hostile.write_text(source_text or hostile_ctrf(proof))
    target.parent.mkdir(parents=True, exist_ok=True)
    py_compile.compile(
        str(hostile),
        cfile=str(target),
        dfile=str(dfile),
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
    )


def not_passed(rewards, error):
    """The hostile code never decided the score: refused, or an honest 0."""
    return refused_scored_zero(rewards, error) or (
        rewards == {"reward": 0.0} and error is None
    )


async def test_a_parent_spelling_through_a_link_never_loads_other_bytes(
    tmp_path, monkeypatch
):
    """R3-1 (both): the verifier's conftest puts ``/app/build/../site-packages``
    on sys.path and declares the plugin module; the agent made /app/build a
    link to /app/nested/build and left a hostile module in
    /app/nested/site-packages. ``abspath`` judged the genuine
    /app/site-packages copy and carried that trust to the file Python really
    opened."""
    proof = tmp_path / "hostile-imported"

    def leave(layout):
        workspace = layout.workspace
        install_plugin(workspace / "site-packages", CTRF_PLUGIN)
        (workspace / "nested/build").mkdir(parents=True)
        (workspace / "build").symlink_to(
            workspace / "nested/build", target_is_directory=True
        )
        install_plugin(workspace / "nested/site-packages", hostile_ctrf(proof))
        (workspace / "answer.txt").write_text("42\n")
        tests = layout.tmp / "verifier-tests"
        tests.mkdir()
        (tests / "conftest.py").write_text(
            "import sys\n"
            f"sys.path.insert(0, {str(workspace) + '/build/../site-packages'!r})\n"
            "pytest_plugins = ['ctrf_model']\n"
        )

    def nothing(env, layout):
        return []

    _, _, rewards, error = await score(
        tmp_path, monkeypatch, FAILING, nothing, before=leave, workspace_name="app"
    )

    assert not proof.exists()
    assert not_passed(rewards, error), (rewards, error)


@pytest.mark.parametrize("runtime", [True, False], ids=["runtime", "no-runtime"])
async def test_a_root_install_into_the_system_site_packages_never_loads(
    tmp_path, monkeypatch, runtime
):
    """Astra R3-2: test.sh's ``python3 -m pip install -r /app/requirements.txt``
    runs as root, and the agent's requirements name its own ctrf wheel; pip
    puts it in the system's site-packages, outside the agent's trees and
    root's. Ownership alone admitted it."""
    proof = tmp_path / "hostile-imported"

    def system_install(env, layout):
        site = layout.tmp / "system-site-packages"
        install_plugin(site, hostile_ctrf(proof))
        return [site]

    def leave(layout):
        (layout.workspace / "requirements.txt").write_text("./agent-ctrf.whl\n")

    _, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        FAILING,
        system_install,
        before=leave,
        workspace_name="app",
        published=runtime,
    )

    assert not proof.exists()
    assert not_passed(rewards, error), (rewards, error)


async def test_a_cache_the_agent_left_for_an_image_plugin_never_runs(
    tmp_path, monkeypatch, caplog
):
    """R3-3 (both): the image's plugin is unchanged, but test.sh runs pytest
    with ``PYTHONPYCACHEPREFIX=/app/bytecache`` and the agent left an
    unchecked-hash cache for the plugin's module there. Image trust skipped
    every cache check, and Python ran the agent's byte code."""
    proof = tmp_path / "hostile-imported"
    image = tmp_path / "image-site-packages"

    def leave(layout):
        install_plugin(
            image, IMAGE_PLUGIN, name="image_plugin", module="image_plugin_module"
        )
        prefix = layout.workspace / "bytecache"
        layout.pytest_env = {"PYTHONPYCACHEPREFIX": str(prefix)}
        layout.pytest_args = ["--assert=plain"]
        module = image / "image_plugin_module.py"
        hostile_cache(
            tmp_path,
            proof,
            cache_for(module, prefix),
            module,
            "from pathlib import Path\n"
            f"Path({str(proof)!r}).touch()\n"
            + IMAGE_PLUGIN
            + "def pytest_collection_modifyitems(items):\n"
            "    for item in items:\n"
            "        item.obj = lambda: None\n",
        )

    def image_python(env, layout):
        return [image]

    _, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        FAILING,
        image_python,
        before=leave,
        workspace_name="app",
        discovered=["image_plugin"],
    )

    assert not proof.exists()
    assert refused_scored_zero(rewards, error), (rewards, error)
    assert "pytest plugin guard not confirmed" in caplog.text


async def test_a_hashlib_the_agent_left_never_runs_inside_the_guard(
    tmp_path, monkeypatch
):
    """Sol R3-2: pytest runs with the workspace first on sys.path, and the
    agent left /app/hashlib.py. The guard imported hashlib lazily, from
    there, while it wrote its loading marker: the agent's code ran inside the
    guard and could switch its checks off."""
    proof = tmp_path / "hashlib-imported"

    def leave(layout):
        (layout.workspace / "hashlib.py").write_text(
            "import importlib.util, sysconfig\n"
            "from pathlib import Path\n"
            f"Path({str(proof)!r}).touch()\n"
            "spec = importlib.util.spec_from_file_location(\n"
            "    '_real_hashlib', sysconfig.get_paths()['stdlib'] + '/hashlib.py')\n"
            "real = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(real)\n"
            "globals().update(\n"
            "    {k: v for k, v in vars(real).items() if not k.startswith('__')})\n"
        )

    def workspace_first(env, layout):
        return [layout.workspace]

    _, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        FAILING,
        workspace_first,
        before=leave,
        workspace_name="app",
    )

    assert not proof.exists()
    assert not_passed(rewards, error), (rewards, error)


async def test_autoload_turned_back_on_loads_no_unchecked_plugin(tmp_path, monkeypatch):
    """Astra R3-4: a ``.pth`` in the environment test.sh built clears
    PYTEST_DISABLE_PLUGIN_AUTOLOAD at interpreter start (modelled here by the
    variable arriving empty), and a dependency the agent chose registers an
    unrequested pytest11 plugin. pytest autoloaded it unchecked."""
    proof = tmp_path / "hostile-imported"

    def leave(layout):
        layout.pytest_env = {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": ""}
        (layout.workspace / "answer.txt").write_text("42\n")

    def install(env, layout):
        sites = venv_install(CTRF_PLUGIN)(env, layout)
        install_plugin(sites[0], hostile_ctrf(proof), name="evil", module="evil_module")
        return sites

    _, _, rewards, error = await score(
        tmp_path, monkeypatch, FAILING, install, before=leave, workspace_name="app"
    )

    assert not proof.exists()
    assert not_passed(rewards, error), (rewards, error)
