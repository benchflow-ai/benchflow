"""Portable tool-order resume checks; these never start a sandbox."""

import json

import pytest

from benchflow.evaluation import EvaluationConfig, _recorded_run_differs


@pytest.mark.parametrize(
    "backend",
    ["docker", "remote-docker", "daytona", "modal", "apple-container", "agentcore"],
)
@pytest.mark.parametrize("scope", [(), ("agent", "build"), ("mode", "build")])
def test_tool_order_resume_is_backend_independent(tmp_path, backend, scope):
    """Guards P2-CONFIRM-3 at d80576e9 using APIs also present at 8b03a596."""

    def config(switches):
        node = {"tools": switches}
        for key in reversed(scope):
            node = {key: node}
        return EvaluationConfig(
            agent="oracle",
            model="m1",
            environment=backend,
            agent_env={"OPENCODE_CONFIG_CONTENT": json.dumps(node)},
        )

    before = config({"write": True, "edit": False})
    after = config({"edit": False, "write": True})
    (tmp_path / "evaluation.json").write_text(
        json.dumps(
            {
                "config": {
                    "agent": "oracle",
                    "model": "m1",
                    "environment": backend,
                    "agent_env": before.agent_env,
                }
            }
        )
    )
    assert _recorded_run_differs(tmp_path, before, None) is None
    assert _recorded_run_differs(tmp_path, after, None) is not None
