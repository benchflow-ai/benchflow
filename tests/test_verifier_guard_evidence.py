"""A guard verdict needs evidence from a pytest of the run (GUARD-LOAD).

The verifier used to refuse every run in which no pytest registered the
plugin guard while the task's plugins came from BenchFlow's protected
runtime. On Terminal-Bench 2's qemu-startup and qemu-alpine-ssh, test.sh's
``apt-get install curl`` fails, so ``uvx`` and pytest never start and test.sh
writes 0; the oracle and nop were both refused, where other engines score 0.

Now a run is refused only on what a pytest of the run left: a signed marker
(refused, crashed, checking, loading, installed, ...) or pytest's own message
that it could not load the guard. These tests run the real hardening and
``_verify_rollout`` with the models of ``tests/test_verifier_uv_state.py``.
"""

from pathlib import Path

import pytest

from benchflow.sandbox._base import ExecResult
from tests.test_verifier_uv_state import (
    FAILING,
    PASSING,
    ScriptSandbox,
    hostile_ctrf,
    install_plugin,
    refused_scored_zero,
    score,
    uvx,
)


class NoPytest(ScriptSandbox):
    """test.sh never starts pytest: its install failed (qemu-startup), or it
    checks the answer another way. It writes the reward it decided."""

    reward = "0"

    async def exec(self, command, env=None, **kwargs):
        if "test-stdout.txt" not in command:
            return await super().exec(command, env=env, **kwargs)
        self.env = dict(env or {})
        self.layout.paths.test_stdout_path.write_text(
            "E: Unable to fetch some archives\n/tests/test.sh: line 19: uvx: "
            "command not found\n"
        )
        self.layout.paths.reward_text_path.write_text(self.reward)
        return ExecResult(stdout="", stderr="", return_code=0)


@pytest.mark.parametrize("reward", ["0", "1"])
async def test_a_test_sh_that_never_starts_pytest_is_scored(
    tmp_path, monkeypatch, reward
):
    """Fails on 57d28d656: "no pytest of this run registered the guard"."""

    class Sandbox(NoPytest):
        pass

    Sandbox.reward = reward
    monkeypatch.setattr("tests.test_verifier_uv_state.ScriptSandbox", Sandbox)

    layout, env, rewards, error = await score(
        tmp_path, monkeypatch, PASSING, workspace_name="app"
    )

    # The guard was armed: the task's ctrf comes from the protected runtime.
    assert "_benchflow_guard_" in env.get("PYTEST_ADDOPTS", ""), env
    assert error is None, error
    assert rewards == {"reward": float(reward)}
    assert list(layout.paths.verifier_dir.glob("_benchflow_guard_*")) == []


@pytest.mark.parametrize("published", [True, False], ids=["runtime", "no-runtime"])
@pytest.mark.parametrize("where", ["workspace-venv", "pythonpath"])
async def test_a_planted_plugin_in_a_pytest_the_guard_reaches_still_scores_nothing(
    tmp_path, monkeypatch, where, published, caplog
):
    """What the rule keeps: the agent leaves a ctrf that passes every test
    where test.sh's pytest imports it, and the solution fails. The guard
    registers in that pytest. With ctrf in the protected runtime the guard
    registers that one and the planted copy is never imported (0); without
    it the guard refuses the planted copy (a refusal, counted as a failure).
    Never the 1 the plugin would give."""
    proof = tmp_path / "hostile-imported"
    planted = {
        "workspace-venv": Path(".venv/lib/python3.12/site-packages"),
        "pythonpath": Path("."),
    }[where]

    def plant(layout):
        install_plugin(layout.workspace / planted, hostile_ctrf(proof))
        if where == "pythonpath":
            layout.image_pythonpath = str(layout.workspace)

    def loads_planted(env, layout):
        return [layout.workspace / planted, *uvx(env, layout)]

    _, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        FAILING,
        loads_planted,
        before=plant,
        workspace_name="app",
        published=published,
    )

    if published:
        assert rewards == {"reward": 0.0}, (rewards, error)
        assert error is None
    else:
        assert refused_scored_zero(rewards, error), (rewards, error)
        assert "pytest plugin guard not confirmed" in caplog.text
    assert not proof.exists()


async def test_a_pytest_that_cannot_load_the_guard_is_still_refused(
    tmp_path, monkeypatch, caplog
):
    """Evidence the rule keeps: pytest was handed the guard (its random name
    is in PYTEST_ADDOPTS) and says it cannot import it, so no test ran and
    test.sh's 0 scores the harness."""

    def leave(layout):
        # test.sh replaced PYTHONPATH: the guard's folder is gone.
        layout.pytest_env = {"PYTHONPATH": str(layout.tmp / "empty")}

    layout, _, rewards, error = await score(
        tmp_path, monkeypatch, PASSING, before=leave, workspace_name="app"
    )

    assert "Error importing plugin" in layout.paths.test_stdout_path.read_text()
    assert rewards == {"reward": 0.0}
    assert "pytest plugin guard not confirmed" in caplog.text
    assert error is None
