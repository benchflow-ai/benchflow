"""Guards V2-REDESIGN-R1-1/2/3 at c1bd6fca through hardening, the guard and pytest.

R1-1: byte code Python reads for a published module from outside its package
(``PYTHONPYCACHEPREFIX``) ran in place of the verified source. R1-2: a task
had no way to have a plugin other than pytest-json-ctrf trusted by its bytes.
R1-3: uv's ``symlink`` link mode installs links, which the byte check refused.
"""

import importlib.util
import py_compile
import sys
from pathlib import Path

import pytest

from benchflow.task.config import TaskConfig
from tests import test_verifier_uv_state as uv_state
from tests.test_verifier_plugin_published import venv_install
from tests.test_verifier_uv_state import (
    CTRF_PLUGIN,
    FAILING,
    PASSING,
    hostile_ctrf,
    install_plugin,
    plugin_never_ran,
    refused_scored_zero,
    score,
)


def cache_for(source: Path, prefix: Path) -> Path:
    """Where CPython reads the byte code of *source* under ``-X pycache_prefix``."""
    previous = sys.pycache_prefix
    sys.pycache_prefix = str(prefix)
    try:
        return Path(importlib.util.cache_from_source(str(source)))
    finally:
        sys.pycache_prefix = previous


@pytest.mark.parametrize("case", ["hostile", "honest"])
async def test_byte_code_under_a_cache_prefix_cannot_replace_a_published_plugin(
    tmp_path, monkeypatch, case
):
    """V2-R1-1: test.sh runs pytest with ``PYTHONPYCACHEPREFIX=/app/bytecache``
    and ``--assert=plain``; the agent left an unchecked-hash cache there for
    the plugin's module. Python ran it instead of the verified source, and a
    failing solution passed. Byte code compiled from the source still loads."""
    proof = tmp_path / "hostile-imported"

    def leave(layout):
        prefix = layout.workspace / "bytecache"
        layout.pytest_env = {"PYTHONPYCACHEPREFIX": str(prefix)}
        layout.pytest_args = ["--assert=plain"]
        site = layout.workspace / ".venv/lib/python3.12/site-packages"
        source = tmp_path / "compiled-from" / "ctrf_model.py"
        source.parent.mkdir()
        source.write_text(hostile_ctrf(proof) if case == "hostile" else CTRF_PLUGIN)
        target = cache_for(site / "ctrf_model.py", prefix)
        target.parent.mkdir(parents=True)
        py_compile.compile(
            str(source),
            cfile=str(target),
            dfile=str(site / "ctrf_model.py"),
            invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
        )
        (layout.workspace / "answer.txt").write_text("42\n")

    _, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        FAILING,
        venv_install(CTRF_PLUGIN),
        before=leave,
        workspace_name="app",
    )

    assert not proof.exists()
    if case == "hostile":
        assert plugin_never_ran(rewards, error), (rewards, error)
    else:
        assert rewards == {"reward": 0.0} and error is None, (rewards, error)


def symlinked(where):
    """``uv --link-mode=symlink``: the environment's files are links into uv's
    cache, a folder of the verifier's uv state; *where* is the environment."""

    def place(env, layout):
        cache = Path(env["UV_CACHE_DIR"]) / "archive-v0" / "wheel-ctrf"
        site = (
            layout.workspace / ".venv/lib/python3.12/site-packages"
            if where == "workspace-venv"
            else Path(env["UV_CACHE_DIR"]) / "archive-v0/ctrf-env/site-packages"
        )
        install_plugin(cache, layout.plugin_source)
        for path in sorted(cache.rglob("*")):
            link = site / path.relative_to(cache)
            if path.is_dir():
                link.mkdir(parents=True, exist_ok=True)
            else:
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(path)
        if where == "workspace-venv":
            (layout.workspace / ".venv/pyvenv.cfg").write_text("home = /usr/bin\n")
        return [site]

    return place


@pytest.mark.parametrize("where", ["uvx", "workspace-venv"])
@pytest.mark.parametrize("case", ["oracle", "nop", "hostile"])
async def test_uv_symlink_link_mode_is_checked_by_the_linked_bytes(
    tmp_path, monkeypatch, where, case
):
    """V2-R1-3: ``uvx --link-mode=symlink --with pytest-json-ctrf`` after an
    honest answer: the published release's links were refused. They now
    count by the bytes they reach; links to anything else are still refused."""
    proof = tmp_path / "hostile-imported"

    def leave(layout):
        layout.plugin_source = hostile_ctrf(proof) if case == "hostile" else CTRF_PLUGIN
        (layout.workspace / "answer.txt").write_text("42\n")

    _, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        PASSING if case == "oracle" else FAILING,
        symlinked(where),
        before=leave,
        workspace_name="app",
    )

    assert not proof.exists()
    if case == "hostile":
        assert plugin_never_ran(rewards, error), (rewards, error)
    else:
        assert rewards == {"reward": float(case == "oracle")} and error is None, (
            rewards,
            error,
        )


@pytest.mark.parametrize("pinned", [True, False])
@pytest.mark.parametrize("case", ["oracle", "nop", "hostile"])
async def test_a_task_pins_the_other_plugin_releases_it_installs(
    tmp_path, monkeypatch, pinned, case, caplog
):
    """V2-R1-2: a plugin other than pytest-json-ctrf that test.sh installs with
    uvx was refused once the agent changed any file. A task that pins its
    release (``[verifier] pytest_plugin_releases``) has it trusted by the
    bytes PyPI publishes for that release; a hostile one never matches, and
    an unpinned one is still refused."""
    proof = tmp_path / "hostile-imported"
    original = uv_state.make_task

    def make_task(layout, env=None):
        task = original(layout, env=env)
        if pinned:
            task.config = TaskConfig.model_validate_toml(
                'version = "1.0"\n[verifier]\n'
                'pytest_plugin_releases = ["ctrf-model==0.3.5"]\n'
            )
            task.config.verifier.env = {"HOME": str(layout.home), **(env or {})}
        return task

    monkeypatch.setattr(uv_state, "make_task", make_task)

    def leave(layout):
        layout.plugin_source = hostile_ctrf(proof) if case == "hostile" else CTRF_PLUGIN
        (layout.workspace / "answer.txt").write_text("42\n")

    def uvx(env, layout):
        site = Path(env["UV_CACHE_DIR"]) / "archive-v0/ctrf-env/site-packages"
        install_plugin(site, layout.plugin_source)
        return [site]

    _, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        PASSING if case == "oracle" else FAILING,
        uvx,
        before=leave,
        published=False,
        pinned={"ctrf-model==0.3.5": {"ctrf": ("ctrf_model", CTRF_PLUGIN)}},
    )

    assert not proof.exists()
    if case == "hostile" and pinned:
        assert rewards == {"reward": 0.0}, (rewards, error)
        assert error is None
    elif not pinned:
        assert refused_scored_zero(rewards, error), (rewards, error)
        assert "pytest plugin guard not confirmed" in caplog.text
    else:
        assert rewards == {"reward": float(case == "oracle")}, (rewards, error)
        assert error is None
