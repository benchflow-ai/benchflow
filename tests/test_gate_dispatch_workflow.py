"""gate-dispatch.yml is test.yml's jobs for one commit on the self-hosted gate
runners, run by hand (tools/ci/gen_gate_dispatch.py)."""

from __future__ import annotations

from pathlib import Path

import yaml

from tools.ci import gen_gate_dispatch as G

WF = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _load(name: str) -> dict:
    return yaml.safe_load((WF / name).read_text())


def test_the_file_is_what_the_generator_writes():
    assert G.main(["--check"]) == 0


def test_it_runs_only_by_hand():
    # PyYAML reads the key `on` as True.
    on = _load("gate-dispatch.yml")[True]
    assert set(on) == {"workflow_dispatch"}
    assert on["workflow_dispatch"]["inputs"]["sha"]["required"] is True


def test_its_jobs_are_test_yml_s_step_for_step():
    test = _load("test.yml")["jobs"]
    gate = _load("gate-dispatch.yml")["jobs"]
    assert set(gate) == set(test) - set(G.LEFT_OUT)
    for name, job in gate.items():
        assert job["runs-on"] == G.RUNNER, name
        want = test[name].get("steps", [])
        got = [s for s in job.get("steps", []) if s.get("name") != G.CHECK_STEP["name"]]
        stripped = []
        for step in got:
            if str(step.get("uses", "")).startswith(G.CHECKOUT):
                assert step["with"]["ref"] == "${{ inputs.sha }}", name
                rest = {k: v for k, v in step["with"].items() if k != "ref"}
                step = {k: v for k, v in step.items() if k != "with"}
                if rest:
                    step["with"] = rest
            stripped.append(step)
        assert stripped == want, name
        for key in set(test[name]) - {"runs-on", "steps", "needs"}:
            assert job[key] == test[name][key], (name, key)


def test_every_checkout_is_followed_by_the_commit_check():
    for name, job in _load("gate-dispatch.yml")["jobs"].items():
        steps = job.get("steps", [])
        for i, step in enumerate(steps):
            if str(step.get("uses", "")).startswith(G.CHECKOUT):
                assert steps[i + 1]["name"] == G.CHECK_STEP["name"], name
                assert "inputs.sha" not in steps[i + 1]["run"], name
