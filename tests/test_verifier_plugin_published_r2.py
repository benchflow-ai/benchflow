"""Guards V2-REDESIGN-R2-1 at 1a2cc882 through hardening, the guard and pytest.

R2-1: a published release's module may be a link (uv's ``symlink`` link
mode). Its target was trusted by its own path too, so when the target's
folder came first on ``sys.path`` Python imported the target directly and
read byte code for that path, which no check covered: an unchecked-hash
cache under ``PYTHONPYCACHEPREFIX`` ran in place of the verified source.
"""

import py_compile

import pytest

from tests.test_verifier_plugin_published_r1 import cache_for
from tests.test_verifier_uv_state import (
    CTRF_PLUGIN,
    FAILING,
    hostile_ctrf,
    install_plugin,
    plugin_never_ran,
    score,
)


@pytest.mark.parametrize("imported", ["by-target", "through-site"])
async def test_a_link_target_imported_by_its_own_path_runs_no_unchecked_byte_code(
    tmp_path, monkeypatch, imported
):
    """test.sh: ``pip install --target /app/site-packages`` (which keeps what is
    there), then ``PYTHONPATH=/app/site-packages PYTHONPYCACHEPREFIX=/app/bytecache
    python3 -m pytest --assert=plain`` from /app. The agent left the genuine
    release there, its module a link to /app/ctrf_model.py (the genuine
    source), and a hostile unchecked-hash cache for /app/ctrf_model.py under
    the prefix. With /app first on sys.path (``by-target``), the target is
    imported by its own path and must be refused; imported through the
    site's link (``through-site``), Python reads the link's own cache, the
    hostile one never runs, and the failing solution scores 0."""
    proof = tmp_path / "hostile-imported"

    def leave(layout):
        workspace = layout.workspace
        prefix = workspace / "bytecache"
        layout.pytest_env = {"PYTHONPYCACHEPREFIX": str(prefix)}
        layout.pytest_args = ["--assert=plain"]
        site = workspace / "site-packages"
        install_plugin(site, CTRF_PLUGIN)
        target = workspace / "ctrf_model.py"
        target.write_text(CTRF_PLUGIN)
        (site / "ctrf_model.py").unlink()
        (site / "ctrf_model.py").symlink_to(target)
        hostile = tmp_path / "compiled-from" / "ctrf_model.py"
        hostile.parent.mkdir()
        hostile.write_text(hostile_ctrf(proof))
        cache = cache_for(target, prefix)
        cache.parent.mkdir(parents=True)
        py_compile.compile(
            str(hostile),
            cfile=str(cache),
            dfile=str(target),
            invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
        )
        (workspace / "answer.txt").write_text("42\n")

    def pip_target(env, layout):
        site = layout.workspace / "site-packages"
        return [layout.workspace, site] if imported == "by-target" else [site]

    _, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        FAILING,
        pip_target,
        before=leave,
        workspace_name="app",
    )

    assert not proof.exists()
    # Now ctrf comes from the protected runtime ahead of every
    # path: neither the link, its target nor the cache is ever read.
    assert plugin_never_ran(rewards, error), (rewards, error)
