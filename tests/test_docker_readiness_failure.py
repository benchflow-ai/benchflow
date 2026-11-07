"""Guards U11-C1: ea8cb7f2/17c37337 must not restart an observed failed init."""

import pytest

from benchflow.sandbox._base import ExecResult
from tests.test_docker_direct_exec import _sandbox


@pytest.mark.parametrize("failure", ["exited (42)", "is unhealthy", "invalid config"])
async def test_startup_failure_is_not_masked_by_a_second_up(tmp_path, failure):
    """A second up reuses the marker written by a failed entrypoint and succeeds."""
    sandbox = _sandbox(tmp_path)
    calls = []
    initialized = False

    async def compose(command, check=True, timeout_sec=None):
        nonlocal initialized
        calls.append(command)
        if command[0] == "up" and not initialized:
            initialized = True
            if "--wait" in command or failure == "invalid config":
                raise RuntimeError(f"container {failure}")
        # A subsequent up silently restarts the same partially initialized container.
        return ExecResult(stdout="", stderr="", return_code=0)

    sandbox._run_docker_compose_command = compose
    with pytest.raises(
        RuntimeError, match=failure.replace("(", r"\(").replace(")", r"\)")
    ):
        await sandbox._run_docker_compose_up()
    assert len([c for c in calls if c[0] == "up"]) == 1
