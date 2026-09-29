"""load_trial/load_job: rubric verdicts, results.jsonl-only jobs, comparable settings.

Three gaps are closed here: the ``bench review`` rubric verdict
(``review*/**/review_report.json``, up to four folders
above a trial) was not read; a job that has only ``results.jsonl`` rows (the
Verifiers-style export, or a copied-out results file) could not be loaded;
and ``bf.compare`` paired tasks without checking that both sides ran with
comparable settings. It now runs identity checks (task
digest, model, harness, dataset, reasoning effort, sandbox, sandbox user,
timeout, agent variable names, prompts hash) per paired task, warns (or
raises with ``on_mismatch="raise"``) on a difference the caller did not
declare with ``vary=``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

import benchflow as bf
from tests.test_python_sdk_load_job import _trial


def _review(root: Path, trial: Path, *, valid: bool = True) -> Path:
    report = root / "review-2026" / "run1" / "review_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "reviewer": {"model": "gpt-5.5"},
                "trials": [
                    {
                        "trial_name": trial.name,
                        "review_valid": valid,
                        "scoring": {"quality": 0.75} if valid else {},
                        "summary": "Correct file, terse explanation.",
                        "criterion_metadata": [{"name": "correct", "blocker": True}],
                        "checks": {"correct": {"outcome": "pass", "explanation": "ok"}},
                    }
                ],
            }
        )
    )
    return report


def test_load_trial_reads_the_rubric_review(tmp_path: Path) -> None:
    job = tmp_path / "jobs" / "j1"
    trial = _trial(job, "hello")
    report = _review(tmp_path / "jobs", trial)
    t = bf.load_trial(trial)
    assert t.review is not None
    assert t.review.reviewer_model == "gpt-5.5" and t.review.review_valid
    assert t.review.scoring == {"quality": 0.75}
    assert [c.name for c in t.review.criteria] == ["correct"]
    assert t.review.source == str(report)
    assert t.to_record()["review_valid"] is True


def test_no_review_is_none(tmp_path: Path) -> None:
    t = bf.load_trial(_trial(tmp_path / "job", "hello"))
    assert t.review is None and t.to_record()["review_valid"] is None


def _jsonl_job(root: Path) -> Path:
    rows = [
        # A clean scored run; the export marked it not training-ready, which
        # is not a run error.
        {
            "example_id": 0,
            "reward": 1.0,
            "info": {
                "task_name": "t1",
                "rollout_name": "t1__a",
                "agent": "codex-acp",
                "model": "gpt-5.5",
            },
            "error": {
                "error": "export_error",
                "error_chain_str": "missing llm trajectory",
            },
            "token_usage": {
                "input_tokens": 10.0,
                "output_tokens": 5.0,
                "total_tokens": 15.0,
            },
            "metrics": {"n_tool_calls": 3},
            "timing": {"total": 42.0},
        },
        {
            "example_id": 1,
            "reward": 0.0,
            "info": {"task_name": "t2", "agent": "codex-acp", "model": "gpt-5.5"},
            "error": {"error": "agent_error", "error_chain_str": "Agent timed out"},
        },
        {
            "example_id": 2,
            "reward": None,
            "info": {"task_name": "t3", "agent": "codex-acp", "model": "gpt-5.5"},
            "error": {"error": "verifier_error", "error_chain_str": "pytest crashed"},
        },
    ]
    root.mkdir(parents=True)
    (root / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root


def test_load_job_reads_a_results_jsonl_only_job(tmp_path: Path) -> None:
    job = bf.load_job(_jsonl_job(tmp_path / "exported"))
    assert [t.task_name for t in job.trials] == ["t1", "t2", "t3"]
    t1, t2, t3 = job.trials
    assert (
        t1.source == "results.jsonl"
        and t1.path == tmp_path / "exported" / "results.jsonl"
    )
    assert (t1.execution, t1.assessment, t1.reward) == ("completed", "scored", 1.0)
    assert (
        t1.result.error is None
        and t1.total_tokens == 15
        and t1.result.n_tool_calls == 3
    )
    assert t1.result.rollout_name == "t1__a" and t1.result.model == "gpt-5.5"
    assert (t2.execution, t2.assessment) == ("errored", "scored")
    assert (t3.execution, t3.assessment) == ("completed", "error")
    assert t1.trajectory == [] and t1.forks == [] and t1.review is None
    d = job.denominators()
    assert (d.attempted, d.scored, d.assessment_errors, d.passed) == (3, 2, 1, 1)


def test_result_json_trials_win_over_their_results_jsonl(tmp_path: Path) -> None:
    """Every normal job also has results.jsonl rows; they must not double-count."""
    job = tmp_path / "job"
    trial = _trial(job, "hello")
    (job / "results.jsonl").write_text(
        json.dumps({"reward": 1.0, "info": {"task_name": "hello"}}) + "\n"
    )
    (trial / "results.jsonl").write_text(
        json.dumps({"reward": 1.0, "info": {"task_name": "hello"}}) + "\n"
    )
    assert [t.source for t in bf.load_job(job).trials] == ["result.json"]


def _pair(tmp_path: Path, *, model_b: str = "claude-haiku-4-5", timeout_b: int = 300):
    a, b = tmp_path / "a", tmp_path / "b"
    for task in ("t1", "t2"):
        _trial(a, task, model="claude-haiku-4-5")
        tb = _trial(b, task, model=model_b, reward=0.0)
        cfg = json.loads((tb / "config.json").read_text())
        cfg["timeout_sec"] = timeout_b
        (tb / "config.json").write_text(json.dumps(cfg))
    return a, b


def test_comparable_runs_raise_no_warning(tmp_path: Path) -> None:
    a, b = _pair(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cmp = bf.compare(a, b)
    assert cmp.summary.setting_mismatches == 0 and cmp.mismatches == []
    checks = {c.setting: c for c in cmp.rows[0].checks}
    assert checks["model"].match is True
    assert checks["task_digest"].match is None  # not recorded in the fixture


def test_an_undeclared_difference_warns_and_is_reported(tmp_path: Path) -> None:
    a, b = _pair(tmp_path, model_b="claude-sonnet-4-6", timeout_b=900)
    with pytest.warns(UserWarning, match="model") as caught:
        cmp = bf.compare(a, b)
    assert "timeout_sec" in str(caught[0].message)
    assert cmp.summary.setting_mismatches == 2  # two paired tasks
    assert {(m.task, m.setting) for m in cmp.mismatches} >= {
        ("t1", "model"),
        ("t1", "timeout_sec"),
    }
    mismatch = next(m for m in cmp.mismatches if m.setting == "model")
    assert mismatch.a == ["claude-haiku-4-5"] and mismatch.b == ["claude-sonnet-4-6"]
    assert any("differ" in c for c in cmp.caveats)


def test_declared_differences_are_not_flagged(tmp_path: Path) -> None:
    a, b = _pair(tmp_path, model_b="claude-sonnet-4-6")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cmp = bf.compare(a, b, vary=("model",))
    assert cmp.summary.setting_mismatches == 0


def test_on_mismatch_raise_refuses(tmp_path: Path) -> None:
    a, b = _pair(tmp_path, timeout_b=900)
    with pytest.raises(ValueError, match="timeout_sec"):
        bf.compare(a, b, on_mismatch="raise")


def test_unknown_vary_names_are_refused(tmp_path: Path) -> None:
    a, b = _pair(tmp_path)
    with pytest.raises(ValueError, match="modle"):
        bf.compare(a, b, vary=("modle",))


def test_varying_the_harness_or_model_covers_its_agent_variables(
    tmp_path: Path,
) -> None:
    """Comparing an oracle job with a Claude job: each harness sets its
    own variables (ANTHROPIC_MODEL, ...), so vary=('harness', 'model') still
    reported agent_variable_names."""
    a, b = tmp_path / "a", tmp_path / "b"
    ta = _trial(a, "t1", agent="oracle", model=None)
    tb = _trial(b, "t1", agent="claude-agent-acp", model="claude-haiku-4-5")
    for trial, env in ((ta, {}), (tb, {"ANTHROPIC_MODEL": "claude-haiku-4-5"})):
        cfg = json.loads((trial / "config.json").read_text())
        cfg["agent_env"] = env
        (trial / "config.json").write_text(json.dumps(cfg))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cmp = bf.compare(a, b, vary=("harness", "model"), include_controls=True)
    assert cmp.mismatches == []
    with pytest.warns(UserWarning, match="agent_variable_names"):
        bf.compare(a, b, vary=("timeout_sec",), include_controls=True)
