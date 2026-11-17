"""Rel's branch readers keep an unknown count null and mark a total that leaves one out (EN-624).

A child whose agent died mid-turn keeps the native usage it started with:
every count None, ``usage_source`` ``unavailable``. A child whose agent's
report named no cache-write count has that one count None (EN-374). Neither
is 0: tree.json keeps them None, a sum with an unknown part is unknown, a
list-price estimate with an unknown part is unknown, and a token total that
leaves a child out is marked a floor. A scripted child (an oracle, a nop)
called no model, so its zeros are real.

The branch job's total (summary.json ``cost`` and ``bench branch``'s Total
line) leaves out a trial with no cost record only as a marked floor, and a
trial whose USD is unknown makes it unknown, never 0.

Two other readers rel never got #238's display fix for: the live progress
footer and the legacy trajectory viewer's RESULT line print ``$0`` for an
unknown cost.
"""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import jsonschema
import pytest
from rich.console import Console
from typer.testing import CliRunner

from benchflow import branch_run
from benchflow.branch_lineage import (
    branch_summary,
    child_cost,
    cost_totals,
    fork_cost,
)
from benchflow.branch_run import BranchPlan, BranchTrialOutcome, ChildSpec
from benchflow.branch_view import BRANCH_VIEW_SCHEMA, load_branch_view
from benchflow.cli._live_progress import LiveEvalProgress
from benchflow.cli.branch import _totals_line, _usd
from benchflow.cli.main import app
from benchflow.models import RolloutResult
from benchflow.trajectories.viewer.legacy import render_turn
from benchflow.usage_tracking import usage_unavailable

HELLO_TASK = Path(__file__).parent / "examples" / "hello-world-task"
COUNTERS = (
    "n_input_tokens",
    "n_output_tokens",
    "n_cache_read_tokens",
    "n_cache_creation_tokens",
    "total_tokens",
)
ZEROS = dict.fromkeys(COUNTERS, 0)
# What a child whose agent died before reporting any usage keeps.
DIED = {**usage_unavailable(), "usage_details": {"thought_tokens": None}}
PASSED = {
    "n_input_tokens": 2000,
    "n_output_tokens": 100,
    "n_cache_read_tokens": 0,
    "n_cache_creation_tokens": 0,
    "total_tokens": 2100,
    "cost_usd": None,
    "usage_source": "agent_native_acp",
}
# The agent's reports named no cache-write count: unknown, not 0 (EN-374).
NO_WRITES = {**PASSED, "n_cache_read_tokens": 300, "n_cache_creation_tokens": None}
PRICES = {
    "model": "m",
    "source": "test",
    "usd_per_token": {
        "input": 1e-6,
        "output": 5e-6,
        "cache_read": 1e-7,
        "cache_creation": 1.25e-6,
    },
}


# ── tree.json: the child's record ────────────────────────────────────────


def test_a_child_whose_usage_is_unavailable_has_unknown_usage_and_tokens():
    cost = child_cost(DIED, {"agent_execution": 12.0}, 18.133)
    assert cost["usage"] is None
    assert cost["cost"] == {"tokens": None, "usd": None, "sandbox_seconds": 18.133}
    # Nothing recorded at all is unknown too.
    assert child_cost(None, None)["usage"] is None


def test_a_child_whose_cache_write_count_is_unknown_keeps_it_null():
    cost = child_cost(NO_WRITES, {}, 11.0)
    assert cost["usage"] == {
        "n_input_tokens": 2000,
        "n_output_tokens": 100,
        "n_cache_read_tokens": 300,
        "n_cache_creation_tokens": None,
        "total_tokens": 2100,
    }
    assert cost["cost"]["tokens"] == 2100


def test_a_scripted_child_s_zero_usage_is_real():
    cost = child_cost(DIED, {}, 4.0, no_model_call=True)
    assert cost["usage"] == ZEROS
    assert cost["cost"]["tokens"] == 0


def test_a_counted_child_is_unchanged():
    cost = child_cost(PASSED, {}, 11.0)
    assert cost["usage"] == {key: PASSED[key] for key in COUNTERS}
    assert cost["cost"] == {"tokens": 2100, "usd": None, "sandbox_seconds": 11.0}


# ── tree.json and result.json: the fork's and the trial's totals ────────


def _child(node: str, label: str, usage: dict, seconds: float, **fields) -> dict:
    return {
        "node_id": node,
        "index": 0,
        "status": "failed",
        "reward": None,
        "reward_source": None,
        "intervention": {"label": label},
        "artifacts": {"status": "unavailable", "path": None},
        **fields,
        **child_cost(usage, {}, seconds),
    }


def _fork(*children: dict) -> dict:
    fork = {
        "id": "f" * 32,
        "rollout": "task__x",
        "status": "completed",
        "value": 1.0,
        "parent_node": "n0",
        "parent_restore": "restored",
        "children": list(children),
    }
    fork["cost"] = fork_cost(fork, wall_seconds=91.0, isolated=False)
    return fork


CRASH = _child("c-crash", "crash", DIED, 18.133)
PASS = _child(
    "c-pass", "pass", PASSED, 11.0, status="scored", reward=1.0, reward_source="v"
)
WRITES = _child(
    "c-nowrite",
    "nowrite",
    NO_WRITES,
    9.0,
    status="scored",
    reward=0.0,
    reward_source="v",
)


def test_a_fork_total_that_leaves_a_child_out_is_marked_a_floor():
    fork = _fork(CRASH, PASS)
    assert fork["cost"]["tokens"] == 2100
    assert fork["cost"]["children_without_usage"] == 1
    trial = cost_totals([fork])
    assert (trial["tokens"], trial["children_without_usage"]) == (2100, 1)
    assert "children_without_usage" not in _fork(PASS)["cost"]
    assert "children_without_usage" not in cost_totals([_fork(PASS)])


def test_result_json_child_usage_is_unknown_where_a_child_s_count_is():
    usage = branch_summary([_fork(CRASH, PASS)])["child_usage"]
    assert usage == dict.fromkeys(COUNTERS)
    usage = branch_summary([_fork(WRITES, PASS)])["child_usage"]
    assert usage["n_cache_creation_tokens"] is None
    assert usage["total_tokens"] == 4200 and usage["n_cache_read_tokens"] == 300


# ── the branch view and bench eval branches ──────────────────────────────


def _trial(tmp_path: Path, *children: dict) -> Path:
    fork = _fork(*children)
    trial = tmp_path / "job" / "task__x"
    trial.mkdir(parents=True)
    tree = {"kind": "benchflow-branch-tree", "schema_version": 1, "forks": [fork]}
    (trial / "tree.json").write_text(json.dumps(tree))
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task",
                "model": "claude-haiku-4-5",
                "branches": branch_summary([fork]),
            }
        )
    )
    return trial


def test_the_view_leaves_a_dead_child_s_and_the_trial_s_estimate_unknown(tmp_path):
    view = load_branch_view(_trial(tmp_path, CRASH, PASS), prices=PRICES)
    jsonschema.validate(view, BRANCH_VIEW_SCHEMA)
    crash, passed = view["forks"][0]["children"]
    assert crash["usage"] is None and crash["usd_estimate"] is None
    assert passed["usd_estimate"] == pytest.approx(2000 * 1e-6 + 100 * 5e-6)
    assert view["forks"][0]["usd_estimate"] is None
    assert view["forks"][0]["usage"] is None
    totals = view["totals"]
    assert totals["usage"] is None and totals["usd_estimate"] is None
    assert totals["per_scored_child"]["usd_estimate"] is None
    assert totals["per_scored_child"]["tokens"] is None
    assert totals["cost"]["children_without_usage"] == 1
    line = _totals_line(view)
    assert "2,100+ tokens (a floor: 1 child(ren) with no usage record)" in line, line
    assert "estimate unknown (1 child(ren) with no usage record)" in line, line
    assert "~USD" not in line and "no model calls" not in line, line


def test_the_view_leaves_an_unknown_cache_write_count_unknown(tmp_path):
    view = load_branch_view(_trial(tmp_path, WRITES, PASS), prices=PRICES)
    jsonschema.validate(view, BRANCH_VIEW_SCHEMA)
    nowrite, _ = view["forks"][0]["children"]
    assert nowrite["usage"]["n_cache_creation_tokens"] is None
    assert nowrite["usd_estimate"] is None
    assert view["forks"][0]["usage"] is None
    assert view["totals"]["usage"] is None
    assert view["totals"]["usd_estimate"] is None
    line = _totals_line(view)
    assert "cache write 0" not in line, line
    assert (
        "4,200 tokens (classes unknown: 1 child(ren) with an unreported cache count)"
    ) in line, line
    assert "estimate unknown (1 child(ren) with an unreported cache count)" in line


def _branches(trial: Path) -> list[list[str]]:
    result = CliRunner().invoke(
        app,
        ["eval", "branches", str(trial)],
        terminal_width=250,
        env={"NO_COLOR": "1"},
    )
    assert result.exit_code == 0, result.output
    rows = [line for line in result.output.splitlines() if "│" in line]
    return [[cell.strip() for cell in row.split("│") if cell.strip()] for row in rows]


def test_bench_eval_branches_prints_unknown_for_the_dead_child(tmp_path):
    pytest.importorskip("litellm")
    rows = _branches(_trial(tmp_path, CRASH, PASS))
    # The cost table's row (the first table has one too): In, Out, Cache and
    # USD unknown; the sandbox-seconds are known.
    crash = [row for row in rows if "crash" in row][-1]
    assert crash[2:] == ["-", "-", "-", "-", "18"], crash


def test_bench_eval_branches_prints_unknown_for_an_unknown_cache_count(tmp_path):
    pytest.importorskip("litellm")
    rows = _branches(_trial(tmp_path, WRITES, PASS))
    nowrite = [row for row in rows if "nowrite" in row][-1]
    # In and Out are known; Cache (reads + writes) and the estimate are not.
    assert nowrite[2:] == ["2,000", "100", "-", "-", "9"], nowrite


def test_an_older_tree_with_zeros_still_reads(tmp_path):
    """A tree written before this fix stored the dead child's usage as zeros."""
    old = {
        **CRASH,
        "cost": {"tokens": 0, "usd": None, "sandbox_seconds": 18.133},
        "usage": ZEROS,
    }
    fork = {**_fork(PASS), "children": [old, PASS]}
    fork["cost"] = {
        "tokens": 2100,
        "usd": None,
        "usd_known": False,
        "wall_seconds": 91.0,
        "parent_sandbox_seconds": 91.0,
        "children_sandbox_seconds": 0.0,
        "sandbox_seconds": 91.0,
    }
    trial = tmp_path / "old" / "task__x"
    trial.mkdir(parents=True)
    tree = {"kind": "benchflow-branch-tree", "schema_version": 1, "forks": [fork]}
    (trial / "tree.json").write_text(json.dumps(tree))
    (trial / "result.json").write_text(json.dumps({"task_name": "task"}))
    view = load_branch_view(trial, prices=PRICES)
    jsonschema.validate(view, BRANCH_VIEW_SCHEMA)
    assert view["totals"]["usage"]["total"] == 2100
    assert view["totals"]["cost"]["tokens"] == 2100
    assert "children_without_usage" not in view["totals"]["cost"]
    assert _totals_line(view).startswith("Totals: 2,100 tokens (in 2,000 ")


# ── the branch job's total: summary.json and bench branch's Total line ──


A = {"tokens": 1100, "usd": 0.5, "usd_known": True, "sandbox_seconds": 3.0}
B = {**A, "usd": 0.25}
UNKNOWN_USD = {"tokens": 10, "usd": None, "usd_known": False, "sandbox_seconds": 1.0}
DEAD_CHILD = {**UNKNOWN_USD, "tokens": 2100, "children_without_usage": 1}


@pytest.mark.parametrize("missing", [None, {}])
def test_a_trial_with_no_cost_record_makes_the_job_total_a_floor(missing):
    cost = branch_run.job_cost([A, missing])
    assert (cost["usd"], cost["usd_known"]) == (0.5, False)
    assert cost["total_cost_incomplete"] is True
    assert cost["trials_without_cost_record"] == cost["trials_without_cost"] == 1


def test_every_trial_priced_is_whole():
    cost = branch_run.job_cost([A, B])
    assert (cost["usd"], cost["usd_known"], cost["total_cost_incomplete"]) == (
        0.75,
        True,
        False,
    )
    assert cost["trials_without_cost"] == cost["trials_cost_incomplete"] == 0


def test_a_known_usd_of_none_is_never_0():
    cost = branch_run.job_cost([{"tokens": 5, "usd_known": True, "usd": None}])
    assert cost["usd"] is None and cost["total_cost_incomplete"] is True


def test_a_trial_with_an_unknown_count_flags_the_job_and_floors_its_tokens():
    cost = branch_run.job_cost([A, DEAD_CHILD])
    assert (cost["usd"], cost["usd_known"]) == (None, False)
    assert cost["total_cost_incomplete"] is True
    assert (cost["trials_without_cost"], cost["trials_without_cost_record"]) == (1, 0)
    assert (cost["tokens"], cost["children_without_usage"]) == (3200, 1)


def test_a_partial_child_makes_its_fork_trial_and_job_a_floor():
    def priced(usd: float, partial: bool = False) -> dict:
        usage = {"total_tokens": 550, "cost_usd": usd}
        if partial:
            usage.update(usage_incomplete=True, usage_incomplete_reason="cut")
        return {"node_id": f"n{usd}{partial}", **child_cost(usage, {}, 1.5)}

    fork = _fork(priced(0.25, True), priced(0.25))
    assert (fork["cost"]["usd"], fork["cost"].get("cost_incomplete")) == (0.5, True)
    assert _usd(fork["cost"]) == "0.5000+" and _usd(B) == "0.2500"
    trial = cost_totals([fork])
    job = branch_run.job_cost([trial, B])
    assert (job["usd"], job["total_cost_incomplete"]) == (0.75, True)
    assert job["trials_cost_incomplete"] == 1


def _plan(tmp_path: Path, agent: str) -> BranchPlan:
    return BranchPlan(
        task_paths=[tmp_path / "task"],
        agent=agent,
        model=None,
        sandbox="docker",
        children=[ChildSpec("a"), ChildSpec("b")],
        checkpoint_after=0 if agent in ("oracle", "nop") else 1,
        jobs_dir=tmp_path / "jobs",
        job_name="job",
    )


def _outcome(task: str, cost: dict | None) -> BranchTrialOutcome:
    forks = [{"id": f"{task}-f0", "from": "checkpoint", "children": 2, "cost": cost}]
    return BranchTrialOutcome(
        task=task,
        fork_status="completed" if cost is not None else None,
        value=0.5 if cost is not None else None,
        forks=forks if cost is not None else [],
        cost=cost,
        error=None if cost is not None else "RuntimeError: setup failed",
    )


NO_CALL = {"tokens": 0, "usd": None, "usd_known": False, "sandbox_seconds": 4.0}


def test_the_branch_job_summary_carries_the_flag_and_counts(tmp_path):
    job = branch_run.write_branch_job(
        _plan(tmp_path, "claude-agent-acp"), [_outcome("t1", A), _outcome("t2", None)]
    )
    cost = json.loads((job / "summary.json").read_text())["cost"]
    assert (cost["usd"], cost["usd_known"], cost["total_cost_incomplete"]) == (
        0.5,
        False,
        True,
    )
    assert cost["trials_without_cost_record"] == 1


@pytest.mark.parametrize("agent", ["nop", "script:run.sh", "oracle"])
def test_a_scripted_branch_job_s_summary_is_not_flagged(tmp_path, agent):
    job = branch_run.write_branch_job(_plan(tmp_path, agent), [_outcome("t1", NO_CALL)])
    cost = json.loads((job / "summary.json").read_text())["cost"]
    assert cost["usd"] is None
    assert (cost["total_cost_incomplete"], cost["trials_without_cost"]) == (False, 0)


def _bench_branch(
    tmp_path: Path, outcomes: dict[str, BranchTrialOutcome], monkeypatch, agent: str
) -> str:
    async def fake_trial(plan, task_path, **_):
        return outcomes[task_path.name]

    monkeypatch.setattr(branch_run, "run_branch_trial", fake_trial)
    monkeypatch.setattr("benchflow.cli.doctor.eval_preflight", lambda **_: None)
    root = tmp_path / "tasks"
    for name in outcomes:
        shutil.copytree(HELLO_TASK, root / name)
    result = CliRunner().invoke(
        app,
        [
            *("branch", "--tasks-dir", str(root), "--agent", agent),
            *(("--checkpoint-after-prompt", "0") if agent == "nop" else ()),
            *("--child", "label=a", "--child", "label=b"),
            *("--jobs-dir", str(tmp_path / "j"), "--job-name", "job"),
        ],
        terminal_width=240,
    )
    return " ".join(result.output.split())


def test_bench_branch_marks_a_total_that_leaves_a_trial_out(tmp_path, monkeypatch):
    out = _bench_branch(
        tmp_path,
        {"t1": _outcome("t1", A), "t2": _outcome("t2", None)},
        monkeypatch,
        "claude-agent-acp",
    )
    assert (
        "Total: 1,100 tokens, 3 sandbox-seconds, USD 0.5000+ "
        "(a floor: 1 of 2 trials have no cost record)"
    ) in out, out


def test_bench_branch_marks_a_token_total_that_leaves_a_child_out(
    tmp_path, monkeypatch
):
    out = _bench_branch(
        tmp_path, {"t1": _outcome("t1", DEAD_CHILD)}, monkeypatch, "claude-agent-acp"
    )
    assert (
        "Total: 2,100+ tokens (a floor: 1 child(ren) with no usage record), "
        "1 sandbox-seconds; USD not reported"
    ) in out, out
    assert "no model calls" not in out


def test_bench_branch_says_a_nop_job_made_no_model_calls(tmp_path, monkeypatch):
    out = _bench_branch(tmp_path, {"t1": _outcome("t1", NO_CALL)}, monkeypatch, "nop")
    assert "Total: 0 tokens, 4 sandbox-seconds; no model calls" in out, out
    assert "subscription" not in out


# ── the live progress footer (a review finding 1) ───────────────────────


def _footer(*results: RolloutResult) -> str:
    d = LiveEvalProgress(
        Console(), label="bench", agent="claude", model="m", sandbox="docker"
    )
    d.on_plan(total=len(results), done=0, remaining=len(results))
    for index, result in enumerate(results):
        d.on_result(f"t{index}", result)
    out = Console(file=io.StringIO(), width=200)
    out.print(d.__rich__())
    return out.file.getvalue()


def _run(cost: float | None, *, partial: bool = False) -> RolloutResult:
    result = RolloutResult(
        task_name="t",
        rewards={"reward": 1.0},
        total_tokens=1000,
        cost_usd=cost,
        usage_source="end_of_turn",
    )
    if partial:
        result.usage_incomplete = True
    return result


def test_the_footer_shows_an_unknown_cost_as_unknown_never_0():
    text = _footer(_run(None))
    assert "$—" in text and "$0.00" not in text, text


def test_the_footer_marks_a_total_that_leaves_an_unknown_cost_out_a_floor():
    assert "$0.50+ (floor)" in _footer(_run(0.5), _run(None))


def test_the_footer_marks_a_partial_cost_a_floor():
    assert "$0.50+ (floor)" in _footer(_run(0.5, partial=True))


def test_the_footer_shows_a_whole_total_whole():
    text = _footer(_run(0.5), _run(0.5))
    assert "$1.00" in text and "floor" not in text, text


# ── the legacy viewer's RESULT line (a review finding 2) ────────────────


@pytest.mark.parametrize(
    "event",
    [
        {"type": "result", "num_turns": 3, "result": "done"},
        {"type": "result", "num_turns": 3, "result": "done", "total_cost_usd": None},
    ],
)
def test_the_legacy_viewer_prints_an_unknown_cost_as_unknown(event):
    html = render_turn([event], 1)
    assert "cost=unknown" in html and "$0.0000" not in html, html


def test_the_legacy_viewer_prints_a_known_cost():
    event = {"type": "result", "num_turns": 1, "result": "ok", "total_cost_usd": 0.5}
    assert "cost=$0.5000" in render_turn([event], 1)
