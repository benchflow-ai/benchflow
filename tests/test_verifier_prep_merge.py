"""Regression checks for U5a's 85b87bc9 integration merge."""

from types import SimpleNamespace

import pytest

from benchflow.sandbox import lockdown

MARK = "BENCHFLOW-PREP"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output",
    ["", f"{MARK}-UNAVAILABLE\n", f"{MARK}-START\n{MARK}-STEP one 0 1 2\n{MARK}-END\n"],
)
async def test_unreceipted_preparation_never_replays_or_succeeds(output):
    """Guards the 85b87bc9 merge: lost/corrupted machine output is a failed
    read, never evidence that no step ran and it is safe to repeat cleanup."""
    calls = []

    async def execute(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(return_code=0, stdout=output, stderr="")

    env = SimpleNamespace(verifier_prep_script=True, exec=execute)
    with pytest.raises(RuntimeError, match="receipt"):
        await lockdown._run_prep_steps(
            env, [lockdown._PrepStep("one", "one", "true", 5)]
        )
    assert len(calls) == 1


def test_bash_fast_step_does_not_cancel_an_unready_watchdog(tmp_path):
    """Guards the U5a merge: a fast step can finish before Bash installs
    the watchdog's TERM handler. Cancellation must wait for that handler."""
    import subprocess

    parent = tmp_path / "sticky"
    parent.mkdir(mode=0o1700)
    parent.chmod(0o1700)
    steps = [lockdown._PrepStep(f"s{i}", f"s{i}", "true", 30) for i in range(4)]
    script = lockdown._prep_script(steps, str(parent / "prep"))
    # Before the child installs its handler, TERM may still be ignored by
    # the enclosing shell. Delay that installation to expose the lost cancel.
    script = script.replace("__bf_s=; __bf_e=", "sleep 0.05; __bf_s=; __bf_e=")
    done = subprocess.run(
        ["bash", "-c", "trap '' TERM\n" + script],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert "UNAVAILABLE" not in done.stdout, done.stdout
    assert f"{MARK}-END" in done.stdout, done.stdout
