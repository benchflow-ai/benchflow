"""Grouping for mixed-model jobs.

When each side held Claude and Codex runs, bf.compare
paired by task name only and averaged the two models; the warning said the
models 'differ' and suggested vary=, which silenced it but kept the mixed
average. Job.denominators() covered the whole job, so a
per-model pass rate meant regrouping by hand. by=("agent", "model") on
compare and Job.denominators_by(...) fix both.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

import benchflow as bf
from tests.test_python_sdk_load_job import _trial


def _side(root: Path, rewards: dict[tuple[str, str, str], float]) -> Path:
    for (task, agent, model), reward in rewards.items():
        _trial(root / f"{task}-{agent}", task, agent=agent, model=model, reward=reward)
    return root


def _pair(tmp_path: Path) -> tuple[Path, Path]:
    a = _side(
        tmp_path / "main",
        {
            ("t1", "claude-agent-acp", "sonnet"): 0.0,
            ("t1", "codex-acp", "gpt-5.5"): 1.0,
        },
    )
    b = _side(
        tmp_path / "target",
        {
            ("t1", "claude-agent-acp", "sonnet"): 1.0,
            ("t1", "codex-acp", "gpt-5.5"): 1.0,
        },
    )
    return a, b


def test_compare_by_agent_and_model_pairs_like_with_like(tmp_path: Path) -> None:
    a, b = _pair(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cmp = bf.compare(a, b, by=("agent", "model"))
    rows = {(r.task, r.group["agent"]): r for r in cmp.rows}
    assert rows[("t1", "claude-agent-acp")].delta == 1.0
    assert rows[("t1", "codex-acp")].delta == 0.0
    assert cmp.summary.paired == 2 and cmp.mismatches == []
    assert "agent" in cmp.to_markdown()


def test_a_side_that_mixes_models_is_named_as_such(tmp_path: Path) -> None:
    a, b = _pair(tmp_path)
    with pytest.warns(UserWarning, match="mix") as caught:
        cmp = bf.compare(a, b)
    assert "by=" in str(caught[0].message)
    assert cmp.rows[0].reward_a == 0.5  # the mixed average is still reported
    assert any("mix" in c for c in cmp.caveats)


def test_unknown_by_keys_are_refused(tmp_path: Path) -> None:
    a, b = _pair(tmp_path)
    with pytest.raises(ValueError, match="modle"):
        bf.compare(a, b, by=("modle",))


def test_denominators_by_agent_and_model(tmp_path: Path) -> None:
    job = bf.load_job(
        _side(
            tmp_path / "j",
            {
                ("t1", "claude-agent-acp", "sonnet"): 1.0,
                ("t2", "claude-agent-acp", "sonnet"): 0.0,
                ("t1", "codex-acp", "gpt-5.5"): 1.0,
                ("t1", "oracle", ""): 1.0,
            },
        )
    )
    groups = job.denominators_by(("agent", "model"))
    table = {(g.key["agent"], g.key["model"]): g.denominators for g in groups}
    assert table[("claude-agent-acp", "sonnet")].passed == 1
    assert table[("claude-agent-acp", "sonnet")].attempted == 2
    assert table[("codex-acp", "gpt-5.5")].pass_rate_scored == 1.0
    assert all(k[0] != "oracle" for k in table)  # controls left out by default
    with_controls = job.denominators_by(("agent",), include_controls=True)
    assert {g.key["agent"] for g in with_controls} >= {"oracle"}


def test_headline_rates_over_the_paired_tasks(tmp_path: Path) -> None:
    """'sonnet 1/3 (33%) vs haiku 0/2 (0%)' compared rates over
    different task sets; on the 2 paired tasks the sides were identical."""
    a = _side(
        tmp_path / "sonnet",
        {
            ("t1", "claude-agent-acp", "m"): 0.0,
            ("t2", "claude-agent-acp", "m"): 0.0,
            ("t3", "claude-agent-acp", "m"): 1.0,
        },
    )
    b = _side(
        tmp_path / "haiku",
        {
            ("t1", "claude-agent-acp", "m"): 0.0,
            ("t2", "claude-agent-acp", "m"): 0.0,
        },
    )
    cmp = bf.compare(a, b)
    assert (cmp.a_paired.attempted, cmp.a_paired.passed) == (2, 0)
    assert (cmp.b_paired.attempted, cmp.b_paired.passed) == (2, 0)
    md = cmp.to_markdown()
    assert "On the 2 tasks both sides ran" in md
    assert md.index("On the 2 tasks") < md.index("sonnet:")


def test_results_denominators(tmp_path: Path) -> None:
    """run_batch's Results printed 'n_passed 1 mean 1.0 len 2'
    with the errored run silently out of the mean and no scored/errored count."""
    rs = bf.Results(
        [
            bf.RolloutResult("t", rewards={"reward": 1.0}),
            bf.RolloutResult(
                "t", error="ACP initialize timed out", error_category="pipe_closed"
            ),
        ]
    )
    d = rs.denominators()
    assert (d.attempted, d.scored, d.passed, d.execution_errors) == (2, 1, 1, 1)
    assert (rs.n_scored, rs.n_errored) == (1, 1)


def test_trial_and_job_reprs_are_compact(tmp_path: Path) -> None:
    """Printing a job of many trials printed several KB."""
    job = tmp_path / "j"
    _trial(job, "hello")
    loaded = bf.load_job(job)
    trial = loaded.trials[0]
    assert repr(trial) == (
        "Trial(task='hello', agent='claude-agent-acp', model='claude-haiku-4-5', "
        "reward=1.0, execution='completed', assessment='scored')"
    )
    assert repr(loaded) == f"Job(path={str(job)!r}, trials=1, kind='directory')"
