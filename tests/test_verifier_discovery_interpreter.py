"""V3-M129-C2: discovery answers for the Python the verifier runs.

On PT's route (merge wave batch 32b) discovery never starts that Python as
root: it runs on root's protected interpreter and reads the pre-agent search
path of the first recorded Python on the verifier's PATH as data
(PT-R13-A-2). V3's tests that executed the selected interpreter
are replaced by the check that the verifier's PATH reaches discovery.
"""

import shlex
import subprocess
from types import SimpleNamespace

import pytest

from benchflow.sandbox import lockdown
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox._exec_merge import StepAnswer
from benchflow.sandbox._shell import SYSTEM_PATH


@pytest.mark.asyncio
async def test_discovery_reads_the_verifier_path_as_data_on_the_protected_python():
    import json

    from tests._root_python_fakes import is_root_python

    seen = []

    from tests._receipt_fakes import receipt_aware

    @receipt_aware
    async def answer(command, **kwargs):
        seen.append(command)
        return ExecResult(
            stdout='{"plugins": ["image_plugin"], "rejected": []}', return_code=0
        )

    recording = SimpleNamespace(exec=answer)

    task = SimpleNamespace(
        task_dir=None,
        config=SimpleNamespace(verifier=SimpleNamespace(pytest_plugins=[])),
    )
    flags = await lockdown._discover_pytest_plugin_flags(
        recording,
        task,
        pythonpath="/opt/lib",
        verifier_path="/opt/image/bin:/usr/bin",
        agent_gids=(27, 1000),
    )
    assert flags == "-p image_plugin"
    (command,) = [c for c in seen if "import entry_points" in c]
    assert is_root_python(command)
    policies = [
        json.loads(word)
        for word in shlex.split(command)
        if word.startswith("{") and "verifier_path" in word
    ]
    assert policies == [
        {
            "ownership": True,
            "pythonpath": "/opt/lib",
            "verifier_path": "/opt/image/bin:/usr/bin",
            "agent_gids": [27, 1000],
        }
    ]


class Shell:
    async def exec(self, command, **kwargs):
        done = subprocess.run(
            ["/bin/sh", "-c", command],
            capture_output=True,
            text=True,
            env={"PATH": SYSTEM_PATH},
            timeout=15,
        )
        return ExecResult(
            stdout=done.stdout, stderr=done.stderr, return_code=done.returncode
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("cached_status", [0, 1])
async def test_absence_cache_cannot_cross_interpreter_selection(
    tmp_path, cached_status
):
    present = tmp_path / "present"
    absent = tmp_path / "absent"
    present.mkdir()
    absent.mkdir()
    python = present / "python3"
    python.symlink_to("/bin/true")
    env = Shell()
    token = lockdown._verifier_reads.set(
        (env, {"python3": StepAnswer("", cached_status)})
    )
    try:
        assert await lockdown._python3_absent(env, str(absent))
        assert not await lockdown._python3_absent(env, str(present))
    finally:
        lockdown._verifier_reads.reset(token)
