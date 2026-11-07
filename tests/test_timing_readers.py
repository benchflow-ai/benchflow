"""Old and new timing.json through every reader (round PY8, target T0).

T0 adds spans to timing.json, one of them (``harden_before_verify``) with a
sub-key per step. A job folder written before T0 (phases only) and one written
after (phases plus spans) must both read through ``bf.load_job``, the JSON
export, ``bench eval metrics``, the job summary's phase totals and the
viewer's data path, with the old keys meaning what they meant.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import benchflow as bf
from benchflow._utils.evaluation_results import phase_timing_summary
from benchflow.cli.main import app
from benchflow.job_export import job_export
from benchflow.trajectories.viewer import render_rollout
from tests.test_python_sdk_load_job import _trial

OLD = {
    "environment_setup": 2.1,
    "agent_setup": 3.0,
    "agent_execution": 10.0,
    "verifier": 4.2,
    "total": 20.0,
}
NEW = {
    "slot_wait": 0.004,
    "environment_setup": 2.0,
    "install_agent": 6.512,
    "litellm_install": 11.25,
    "proxy_health_wait": 2.75,
    "acp_connect": 1.5,
    "agent_setup": 16.0,
    "agent_execution": 9.0,
    "harden_before_verify": {
        "total": 1.2,
        "kill_sandbox_user_procs": 0.4,
        "clear_verifier_output": 0.1,
        "prepare_app_dir": 0.1,
        "reclaim_disk": 0.2,
        "freeze_workspace": 0.2,
        "cleanup_conftests": 0.1,
        "build_verifier_env": 0.1,
    },
    "verifier": 5.0,
    "cleanup": 1.75,
    "total": 40.0,
}


def _job(tmp_path: Path) -> Path:
    job = tmp_path / "jobs" / "j1"
    for task, timing in (("old-task", OLD), ("new-task", NEW)):
        trial = _trial(job, task)
        (trial / "timing.json").write_text(json.dumps(timing))
        result = json.loads((trial / "result.json").read_text())
        result["timing"] = timing
        (trial / "result.json").write_text(json.dumps(result))
    return job


def _payload_meta(page: str) -> dict:
    data = page.split('type="application/json">', 1)[1].split("</script>", 1)[0]
    return json.loads(data.replace("<\\/", "</"))["payload"]["meta"]


def test_load_job_reads_old_and_new_timing_as_written(tmp_path):
    job = bf.load_job(_job(tmp_path))
    by_task = {t.task_name: t for t in job.trials}
    assert by_task["old-task"].timing == OLD
    assert by_task["new-task"].timing == NEW


def test_the_export_sums_phases_as_before_and_spans_per_step(tmp_path):
    totals = job_export(bf.load_job(_job(tmp_path))).timing_totals
    for phase in OLD:
        assert totals[phase] == round(OLD[phase] + NEW[phase], 3)
    assert totals["install_agent"] == 6.512
    assert totals["harden_before_verify.kill_sandbox_user_procs"] == 0.4
    assert totals["harden_before_verify.total"] == 1.2
    assert all(isinstance(v, float) for v in totals.values())
    # The whole document stays JSON (trial timing kept as written).
    doc = bf.load_job(_job(tmp_path / "again")).to_json_dict()
    trial_timings = {t["task_name"]: t["timing"] for t in doc["trials"]}
    assert trial_timings == {"old-task": OLD, "new-task": NEW}


def test_bench_eval_metrics_reads_both(tmp_path):
    job = _job(tmp_path)
    result = CliRunner().invoke(app, ["eval", "metrics", str(job.parent), "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["total"] == 2


def test_the_job_summary_phase_totals_ignore_the_new_spans():
    summary = phase_timing_summary({"old": {"timing": OLD}, "new": {"timing": NEW}})
    assert summary["timing_coverage"] == 1.0
    assert summary["verifier_time_sec"] == round(OLD["verifier"] + NEW["verifier"], 3)
    assert summary["agent_setup_time_sec"] == round(
        OLD["agent_setup"] + NEW["agent_setup"], 3
    )


def test_the_viewer_shows_old_timing_unchanged_and_new_steps_one_value_each(
    tmp_path,
):
    job = _job(tmp_path)
    old_meta = _payload_meta(render_rollout(next(job.glob("old-task__*"))))
    new_meta = _payload_meta(render_rollout(next(job.glob("new-task__*"))))
    assert old_meta["timing"] == OLD
    assert old_meta["duration_sec"] == 20.0
    assert new_meta["duration_sec"] == 40.0
    assert new_meta["timing"]["agent_setup"] == 16.0
    assert new_meta["timing"]["acp_connect"] == 1.5
    assert new_meta["timing"]["harden_before_verify.reclaim_disk"] == 0.2
    assert "harden_before_verify" not in new_meta["timing"]
    assert all(isinstance(v, float) for v in new_meta["timing"].values())
