"""Guards FF-sw-other TRL verifier failure handling at 363d815b."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.integrations.trl.spec import (
    BashHarnessConfig,
    BenchFlowRuntimeEnvironment,
    benchflow_environment_reward,
)


@pytest.mark.parametrize(
    "reward,error,verifier_error",
    [
        (None, None, "timeout"),
        (None, "setup failed", None),
        (1.0, None, "malformed"),
        (None, None, None),
        (float("nan"), None, None),
        (True, None, None),
    ],
)
def test_trl_rejects_ungraded_sample_and_remembers_failure(
    reward, error, verifier_error
):
    class Runtime:
        closed = False

        async def verify(self):
            return SimpleNamespace(
                reward=reward,
                error=error,
                verifier_error=verifier_error,
                rollout_dir=Path("dummy"),
            )

        async def close(self):
            self.closed = True

    runtime = Runtime()
    env = BenchFlowRuntimeEnvironment(BashHarnessConfig())
    env._runtime = runtime
    for _ in range(2):
        with pytest.raises(RuntimeError, match=r"verif|reward|sample"):
            benchflow_environment_reward(["answer"], environments=[env])
    assert runtime.closed
