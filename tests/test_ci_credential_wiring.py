"""Guards FF-sw-tests-ci's credential propagation fix against 363d815b."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("lane", ["scope", "final-review"])
def test_filter_and_execution_receive_the_same_provider_environment(lane):
    """Every admitted provider can execute."""
    workflow = yaml.safe_load(
        (ROOT / f".github/workflows/integration-{lane}.yml").read_text()
    )
    jobs = workflow["jobs"]
    filtering = next(
        step
        for step in jobs["plan"]["steps"]
        if "filter_credentialed_cells.py" in step.get("run", "")
    )
    execution = jobs["run-matrix"]
    run = next(
        step
        for step in execution["steps"]
        if "scenarios.run_eval(" in step.get("run", "")
    )
    effective = {**execution.get("env", {}), **run.get("env", {})}
    for key, mapping in filtering["env"].items():
        # Codex intentionally restores the real OpenAI key at execution,
        # after provider selection supplied the judge's OpenAI-compatible key.
        if key == "OPENAI_API_KEY":
            assert mapping == execution["env"][key]
            assert "secrets.OPENAI_API_KEY" in effective[key]
        else:
            assert effective.get(key) == mapping, key
