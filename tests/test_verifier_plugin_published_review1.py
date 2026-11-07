"""Guards V2-R1-A-1/A-2/B-1/B-2 at 3571f2c5 through hardening, the guard and pytest.

A-2: the input proof compared every recorded workspace file with its
pre-agent copy inside the guard's install exec, so a workspace of many
gigabytes timed that exec out and a nop errored instead of scoring 0.
B-1: a pin PyPI could not serve was dropped, and a vouched release missing
from the table was refused as a plain refusal: the agent's failure. B-2: a
plugin test.sh installed itself, unpinned, was refused the same way, while
the messages told task authors to install it with uvx. (A-1, the 6 h skip of
the PyPI refresh, is in test_verifier_plugin_published_table.py.)
"""

import os
import time
from pathlib import Path

import pytest

from benchflow.sandbox import lockdown
from benchflow.sandbox._base import ExecResult
from tests.test_verifier_plugin_published import venv_install
from tests.test_verifier_uv_state import (
    CTRF_PLUGIN,
    FAILING,
    PASSING,
    Layout,
    hostile_ctrf,
    install_plugin,
    modelled,
    refused_scored_zero,
    score,
)


def uvx_install(env, layout):
    """``uvx --with <plugin>``: an environment below the verifier's uv state."""
    site = Path(env["UV_CACHE_DIR"]) / "archive-v0/plugin-env/site-packages"
    install_plugin(site, CTRF_PLUGIN)
    return [site]


PLACES = {"uvx": uvx_install, "workspace-venv": venv_install(CTRF_PLUGIN)}


def answer(layout):
    """What a real agent leaves: a file in its trees, so the proof is tainted."""
    (layout.workspace / "answer.txt").write_text("42\n")


def planted(source):
    """The agent leaves the plugin in a workspace venv; test.sh only runs it."""

    def leave(layout):
        venv_install(source)({}, layout)
        answer(layout)

    def keep(env, layout):
        return [layout.workspace / ".venv/lib/python3.12/site-packages"]

    return leave, keep


# ---------------------------------------------------------------------------
# V2-R1-B-2: a plugin test.sh installs, unpinned, is the task's problem.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("place", sorted(PLACES))
@pytest.mark.parametrize("solution", [PASSING, FAILING], ids=["oracle", "failing"])
async def test_an_unpinned_plugin_test_sh_installs_is_the_tasks_problem(
    tmp_path, monkeypatch, place, solution, caplog
):
    """test.sh installs a plugin BenchFlow does not vouch for, unpinned, after
    an agent that wrote its answer: the guard refuses it (the agent's files
    could have chosen the install). That refusal was blamed on the agent, and
    its message said to install the plugin with uvx, which is refused too.
    It still counts as a failure, but as the task's, with the pin advice."""
    _, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        solution,
        PLACES[place],
        before=answer,
        published=False,
    )

    assert refused_scored_zero(rewards, error), (rewards, error)
    assert "pytest plugin guard not confirmed" in caplog.text
    assert error is None


async def test_a_plugin_the_agent_left_is_still_the_agents(
    tmp_path, monkeypatch, caplog
):
    """Control: a plugin already in the agent's trees when hardening recorded
    them is the agent's, whatever test.sh does with it."""
    proof = tmp_path / "hostile-imported"
    leave, keep = planted(hostile_ctrf(proof))

    _, _, rewards, error = await score(
        tmp_path, monkeypatch, FAILING, keep, before=leave, published=False
    )

    assert not proof.exists()
    assert refused_scored_zero(rewards, error), (rewards, error)
    assert "pytest plugin guard not confirmed" in caplog.text
    assert error is None


# ---------------------------------------------------------------------------
# V2-R1-A-2: the proof ends inside the guard install's exec timeout.
# ---------------------------------------------------------------------------

# Each recorded file takes a quarter second to compare with its pre-agent
# copy, as gigabytes of them would.
SLOW_IMAGE_BYTES = (
    "import time as _slow\n"
    "def _image_bytes(path):\n"
    "    _slow.sleep(0.25)\n"
    "    return True\n"
)
FAST_IMAGE_BYTES = "def _image_bytes(path):\n    return True\n"


class TimedLayout(Layout):
    """A Layout whose exec fails, as a sandbox's does, past its timeout."""

    timeout = None

    async def exec(self, command, **kwargs):
        self.timeout = kwargs.get("timeout_sec")
        return await super().exec(command, **kwargs)

    def run(self, command):
        started = time.monotonic()
        result = super().run(command)
        if self.timeout is not None and time.monotonic() - started > self.timeout:
            return ExecResult(stdout="", stderr="timed out", return_code=124)
        return result


@pytest.mark.parametrize(
    ("image_bytes", "proof"),
    [(SLOW_IMAGE_BYTES, "tainted\n"), (FAST_IMAGE_BYTES, "clean\n")],
    ids=["slow-tainted", "fast-clean"],
)
async def test_the_input_proof_ends_inside_the_guard_install_timeout(
    tmp_path, monkeypatch, image_bytes, proof
):
    """A nop on a large workspace: every file is in the record and holds the
    image's bytes, and comparing them all takes longer than the exec's
    timeout. Hardening failed and the nop errored; the proof now stops at its
    budget as tainted. A walk that fits says clean, as before."""
    layout = TimedLayout(tmp_path, monkeypatch, workspace_name="app")
    for i in range(40):
        (layout.workspace / f"data-{i}.bin").write_text("DUMMY image data\n")
    monkeypatch.setattr(
        lockdown,
        "_DISCOVER_PYTEST_PLUGINS_SCRIPT",
        modelled(lockdown._DISCOVER_PYTEST_PLUGINS_SCRIPT, image_bytes),
    )
    monkeypatch.setattr(lockdown, "VERIFIER_SETUP_TIMEOUT_SEC", 5)
    monkeypatch.setattr(lockdown, "_INPUT_PROOF_RESERVE_SEC", 3, raising=False)

    await lockdown._install_pytest_plugin_guard(
        layout,
        "agent",
        str(layout.workspace),
        "",
        blocked=(str(layout.workspace),),
        record=True,
    )

    written = [
        Path(directory) / name
        for directory, _, names in os.walk(layout.fs)
        for name in names
        if name.endswith(".inputs")
    ]
    assert len(written) == 1, written
    assert written[0].read_text() == proof


def test_the_proof_budget_leaves_the_install_its_reserve():
    """The guard install's exec keeps time for its steps after the proof."""
    assert 0 < lockdown._INPUT_PROOF_RESERVE_SEC < lockdown.VERIFIER_SETUP_TIMEOUT_SEC


def test_a_proof_with_no_time_left_is_tainted(tmp_path, monkeypatch):
    """No budget left (the record walk took it, or the sandbox has no clock):
    tainted at once, never clean."""
    from benchflow.sandbox import _pytest_plugin_guard as guard

    path = tmp_path / "proof"
    monkeypatch.setattr(guard, "_BENCHFLOW_INPUT_PROOF", str(path))
    monkeypatch.setattr(guard, "_record", lambda: {})
    monkeypatch.setattr(guard, "_BENCHFLOW_BLOCKED", ())
    monkeypatch.setattr(guard, "_PROOF_DEADLINE", None, raising=False)
    for budget in ("0", "-3", ""):
        path.unlink(missing_ok=True)
        guard._write_input_proof(budget)
        assert path.read_text() == "tainted\n", budget
    path.unlink()
    monkeypatch.setattr(guard, "_PROOF_DEADLINE", None, raising=False)
    guard._write_input_proof("60")
    assert path.read_text() == "clean\n"


async def test_the_guard_install_goes_on_in_an_image_without_a_clock(
    tmp_path, monkeypatch
):
    """An image whose ``date`` fails: the proof has no budget and is never
    clean, and the guard install itself goes on."""
    layout = TimedLayout(tmp_path, monkeypatch, workspace_name="app")
    (layout.workspace / "data.bin").write_text("DUMMY image data\n")
    clock = layout.bindir / "date"
    clock.write_text("#!/bin/sh\nexit 127\n")
    clock.chmod(0o755)
    monkeypatch.setattr(
        lockdown,
        "_DISCOVER_PYTEST_PLUGINS_SCRIPT",
        modelled(lockdown._DISCOVER_PYTEST_PLUGINS_SCRIPT, FAST_IMAGE_BYTES),
    )

    await lockdown._install_pytest_plugin_guard(
        layout,
        "agent",
        str(layout.workspace),
        "",
        blocked=(str(layout.workspace),),
        record=True,
    )

    written = [
        (Path(directory) / name).read_text()
        for directory, _, names in os.walk(layout.fs)
        for name in names
        if name.endswith(".inputs")
    ]
    assert "clean\n" not in written, written
