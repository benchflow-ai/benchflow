"""The conftest sweep keeps the repository's own conftest.py when the agent
left it unchanged, and still removes one the agent planted or edited.

Guards this branch's fix for benchflow-ai/benchflow#1177 and #1171 against the
failure they report: verifier hardening removed every conftest.py outside
/tests, the workspace repository's own included, so a correct patch failed
every test that needs the repository's fixtures (``fixture 'app' not found``
on Sphinx and Flask, ``'pytester'`` on pytest, ``'long_df'`` on seaborn). On
a28f2b29 an oracle whose tests use a fixture from the workspace's
``tests/conftest.py`` scored 0 with ``fixture 'app' not found`` and 1 with
``cleanup_conftests = false`` (pytest 8.3.3 and 7.0.1 alike: the verifier's
``--confcutdir`` was not the cause).

A workspace conftest.py now stays when it is byte for byte the file at the same
place in the root-owned, closed pre-agent copy, and the agent ran as a user
other than root. The container case (root, real files, a link in the copy) is
in test_v2_conftest_sweep_container.py.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchflow._root_python import FLAGS, _script
from benchflow.sandbox import _hook_cleanup, lockdown
from tests.test_hook_cleanup_bfx import INPUTS, _FakeFs

REPO = b"import pytest\n\n@pytest.fixture\ndef app():\n    return 'repo'\n"
PLANTED = b"def pytest_runtest_makereport(item, call):\n    pass\n"
KEPT = b"/testbed/tests/conftest.py"
EDITED = b"/testbed/pkg/conftest.py"
NEW = b"/testbed/new/conftest.py"
OUTSIDE = b"/srv/conftest.py"


class _RootFs(_FakeFs):
    """The fake tree with root's owner on every entry, and modes to choose."""

    def __init__(self, tree: dict[bytes, object], modes: dict[bytes, int]):
        super().__init__(tree)
        self.modes = modes

    def facts(self, path: bytes, shown: bytes | None = None) -> dict:
        row = super().facts(path, shown)
        row["uid"] = 0
        if path in self.modes:
            row["mode"] = self.modes[path]
        return row


def _tree() -> dict[bytes, bytes]:
    return {
        KEPT: REPO,
        b"/testbed_verify/tests/conftest.py": REPO,
        EDITED: PLANTED,
        b"/testbed_verify/pkg/conftest.py": b"# the repository's\n",
        NEW: PLANTED,
        OUTSIDE: PLANTED,
    }


async def _sweep(fs: _FakeFs, *, agent_user: bool) -> None:
    loop = asyncio.get_running_loop()
    await _hook_cleanup.run(
        fs.call,
        _hook_cleanup.parse_inputs(INPUTS.replace("a:b", "")),
        cleanup_conftests=True,
        workspace="/testbed",
        workspace_aliases=(),
        verifier_dir="/opt/runtime/verifier",
        deadline=loop.time() + 30,
        agent_user=agent_user,
        baseline="/testbed_verify",
    )


@pytest.mark.asyncio
async def test_bfx_form_keeps_the_repositorys_unchanged_conftest_as_a_fresh_file():
    fs = _RootFs(_tree(), {b"/testbed_verify": 0o700})
    await _sweep(fs, agent_user=True)
    assert fs.tree[KEPT] == REPO
    writes = [args for cmd, args in fs.calls if cmd == "write"]
    assert [(w.get("root"), w.get("path"), w.get("uid")) for w in writes] == [
        ("/testbed/tests", "conftest.py", 0)
    ]
    for path in (EDITED, NEW, OUTSIDE):
        assert path not in fs.tree, path


@pytest.mark.asyncio
async def test_bfx_form_keeps_nothing_after_a_root_agent():
    """A root agent could have written the pre-agent copy as well."""
    fs = _RootFs(_tree(), {b"/testbed_verify": 0o700})
    await _sweep(fs, agent_user=False)
    for path in (KEPT, EDITED, NEW, OUTSIDE):
        assert path not in fs.tree, path


@pytest.mark.asyncio
async def test_bfx_form_needs_a_copy_closed_to_everyone_else():
    fs = _RootFs(_tree(), {b"/testbed_verify": 0o755})
    await _sweep(fs, agent_user=True)
    assert KEPT not in fs.tree


def _shell_sweep(tmp_path: Path, baseline: str) -> None:
    workspace = tmp_path / "testbed"
    found = []
    for path, data in _tree().items():
        real = tmp_path / os.fsdecode(path).lstrip("/")
        real.parent.mkdir(parents=True, exist_ok=True)
        real.write_bytes(data)
        found.append(real)
    ran = subprocess.run(
        [
            sys.executable,
            *FLAGS,
            "-c",
            _script(lockdown._CONFTEST_SWEEP_SCRIPT),
            f"baseline={baseline}",
            str(workspace),
        ],
        input=b"\0".join(os.fsencode(p) for p in found) + b"\0",
        capture_output=True,
        cwd="/",
    )
    assert ran.returncode == 0, ran.stderr


@pytest.mark.parametrize("given", ["", "copy"])
def test_shell_form_keeps_nothing_without_a_root_owned_baseline(tmp_path, given):
    """No baseline (a root agent), or a copy root does not own: every
    workspace conftest.py goes, as before the fix."""
    if given and os.geteuid() == 0:
        pytest.skip("root owns the copy here")
    baseline = str(tmp_path / "testbed_verify") if given else ""
    _shell_sweep(tmp_path, baseline)
    for path in (KEPT, EDITED, NEW):
        assert not (tmp_path / os.fsdecode(path).lstrip("/")).exists(), path


def test_the_sweep_passes_the_copy_only_for_an_agent_that_was_not_root():
    with_agent = lockdown._conftest_sweep_cmd(workspace="/testbed", agent_user=True)
    as_root = lockdown._conftest_sweep_cmd(workspace="/testbed", agent_user=False)
    assert "baseline=/testbed_verify" in with_agent
    assert "baseline=/testbed_verify" not in as_root and "baseline=" in as_root
    assert lockdown._conftest_sweep_cmd(agent_user=True).count("baseline=/") == 0
