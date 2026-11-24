"""Held and provenance endings buy no rerun (: yes).

F4/F29 and Q2: an attempt whose verifier ended held (another user's processes kept writing in the output folder
while it was being closed) or ``provenance_incomplete`` is the trial's result. Another attempt would give the
agent a second turn after it defeated a check:

- 812-U2: a clear before test.sh that fails beside the agent's live processes is held when a root re-list shows a
  name another user owns still in the folder, whatever its stderr says (a name the agent chose can say "Directory
  not empty");
- 812-U3: the retry loop keeps such an attempt even when its agent-side error is one it retries (ACP, pipe,
  sandbox setup).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.models import RunResult

_HELD = (
    "verifier crashed: VerifierOutputsHeldError: agent processes kept writing in /logs/verifier while it was "
    "being closed (rc=1: rm: cannot remove '/logs/verifier/sub': Directory not empty)"
)
_AGENT_ERRORS = {
    "acp": ("ACP error -32603: Internal error: the agent process exited", "acp_error"),
    "install": ("Agent install failed (rc=1): boom", None),
    "setup": ("Sandbox startup failed: container exited during setup", None),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["held", "provenance_incomplete"])
@pytest.mark.parametrize("agent_error", list(_AGENT_ERRORS))
async def test_an_attempt_whose_verifier_ended_held_is_final(
    job_factory, ending, agent_error
):
    """812-U3 (R-812). The agent-side error alone would be retried (precondition); the verifier's ending makes the
    attempt final: one attempt.

    Mutant: the guard in the retry loop removed (the attempt retried, the agent's turn run again)."""
    job, tasks_dir = job_factory(n_tasks=1, max_retries=2)
    error, category = _AGENT_ERRORS[agent_error]
    assert job._config.retry.should_retry(error, category=category), (
        f"precondition: {agent_error} is retried on its own"
    )
    if ending == "held":
        result = RunResult(
            task_name="task-0",
            error=error,
            error_category=category,
            verifier_error=_HELD,
            verifier_error_category="verifier_failure",
        )
    else:
        result = RunResult(
            task_name="task-0",
            rewards={"reward": 0.0},
            error=error,
            error_category=category,
            verifier_error_category="provenance_incomplete",
        )
    job._run_single_task = AsyncMock(return_value=result)
    await job._run_task(tasks_dir / "task-0")
    assert job._run_single_task.call_count == 1, job._run_single_task.call_count


def _relist_env(*names):
    """A sandbox whose root re-list of /logs/verifier (bfx, depth 1) shows *names* as ``(name, uid)``."""

    async def bfx_call(cmd, args=None, *, timeout_sec=None):
        args = dict(args or {})
        assert cmd == "scan" and args.get("walks"), (cmd, args)
        # The output folder itself (the matrix as a run found the re-list reading lockdown's VERIFIER_DIR, the
        # protected verifier code folder /opt/runtime/verifier, and so never seeing the agent's names).
        if [w.get("root") for w in args["walks"]] != ["/logs/verifier"]:
            return {
                "entries": [
                    {"path": args["walks"][0].get("root"), "type": "dir", "uid": 0}
                ],
                "errors": [],
            }
        entries = [{"path": "/logs/verifier", "type": "dir", "uid": 0, "exists": True}]
        entries += [
            {"path": f"/logs/verifier/{n}", "type": "dir", "uid": uid, "exists": True}
            for n, uid in names
        ]
        return {"entries": entries, "errors": [], "mounts": []}

    return SimpleNamespace(
        bfx_call=AsyncMock(side_effect=bfx_call), bfx_delivery="bind", exec=AsyncMock()
    )


async def _clear_failing(env, message):
    from benchflow.sandbox import lockdown

    clearing = lockdown._outputs_clearing(env, "the clear before test.sh")
    if hasattr(clearing, "__aenter__"):
        async with clearing:
            raise RuntimeError(message)
    with clearing:
        raise RuntimeError(message)


_FIND_GONE = (
    "Verifier hardening failed: clearing verifier output directory exited with rc=1\n"
    "stderr: find: '/logs/verifier/flood': No such file or directory"
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("left", "message", "held"),
    [
        ([("flood", 1000)], _FIND_GONE, True),
        ([], _FIND_GONE, False),
        ([], _FIND_GONE + "\nstderr: rm: '/tmp/Directory not empty': gone", False),
    ],
    ids=["foreign-name-left", "nothing-left", "agent-text-says-not-empty"],
)
async def test_a_clear_that_fails_beside_a_live_writer_is_held(left, message, held):
    """812-U2. The clear's rc is non-zero. A root re-list (bfx, depth 1) decides: a name another user owns
    still there means the agent's processes held the folder (``VerifierOutputsHeldError``: counted, never retried);
    none means the clear failed on its own, and its error stands. The words in stderr decide nothing.

    Mutants: the re-list skipped (a held ending read as a crash); the "Directory not empty" text test kept."""
    from benchflow.task.verifier_core import VerifierOutputsHeldError

    env = _relist_env(*left)
    if held:
        with pytest.raises(VerifierOutputsHeldError):
            await _clear_failing(env, message)
    else:
        with pytest.raises(RuntimeError) as caught:
            await _clear_failing(env, message)
        assert not isinstance(caught.value, VerifierOutputsHeldError), caught.value
