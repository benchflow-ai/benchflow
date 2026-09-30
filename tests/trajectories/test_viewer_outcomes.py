"""The job-level views' data: grouping, attribution, splits and steps, the
Pareto frontier and intervals, the redacted export, and the browse server's
``/api/outcomes`` (benchflow.trajectories.viewer.outcomes / jobviews)."""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchflow.trajectories.viewer.catalog import BrowseRoots, root_labels
from benchflow.trajectories.viewer.jobviews import (
    build_for_roots,
    export_html,
    redact_for_export,
)
from benchflow.trajectories.viewer.outcomes import (
    OUTCOMES,
    bootstrap_ratios,
    build_outcomes,
    pareto_frontier,
)

FAKE_KEY = "sk-ant-api03-" + "Zx9" * 30  # key-shaped, not a real key


def _trial(
    folder: Path,
    task: str,
    *,
    reward: float | None = 1.0,
    model: str = "model-a",
    agent: str = "claude-agent-acp",
    trajectory: bool = False,
    config: dict | None = None,
    started: str = "2026-09-30 10:00:00",
    finished: str = "2026-09-30 10:01:40",
    **extra,
) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    result = {
        "task_name": task,
        "rollout_name": folder.name,
        "agent": agent,
        "agent_name": agent,
        "model": model,
        "rewards": {"reward": reward} if reward is not None else None,
        "n_tool_calls": 3,
        "started_at": started,
        "finished_at": finished,
        "timing": {"total": 120.0},
        "agent_result": {"total_tokens": 1000, "cost_usd": 0.02},
    }
    result.update(extra)
    (folder / "result.json").write_text(json.dumps(result))
    (folder / "config.json").write_text(
        json.dumps({"agent": agent, "model": model, **(config or {})})
    )
    if trajectory:
        (folder / "trajectory").mkdir(exist_ok=True)
        (folder / "trajectory" / "acp_trajectory.jsonl").write_text(
            json.dumps({"type": "agent_message", "text": "hi"}) + "\n"
        )
    return folder


def _values(doc: dict, dim: str) -> list[str]:
    d = doc["dims"][dim]
    return [d["values"][c] for c in d["codes"]]


def _by_name(doc: dict) -> dict[str, int]:
    return {name: i for i, name in enumerate(doc["columns"]["name"])}


@pytest.fixture
def two_models(tmp_path: Path) -> Path:
    job = tmp_path / "job"
    for model, rewards in (("model-a", [1.0, 1.0, 0.0]), ("model-b", [0.0, 0.5, 0.0])):
        for n, reward in enumerate(rewards):
            _trial(job / f"alpha__{model}{n}", "alpha", reward=reward, model=model)
            _trial(job / f"beta__{model}{n}", "beta", reward=1.0, model=model)
    _trial(job / "alpha__oracle", "alpha", agent="oracle", model="")
    return job


# ── grouping and statistics ───────────────────────────────────────────────


def test_trials_are_grouped_by_every_dimension(two_models: Path) -> None:
    doc = build_outcomes([("", two_models)], bootstrap_samples=20)
    assert doc["schema"] == "benchflow.outcomes/1"
    assert doc["n"] == 13
    names = _by_name(doc)
    assert _values(doc, "model")[names["alpha__model-b1"]] == "model-b"
    assert _values(doc, "task")[names["beta__model-a0"]] == "beta"
    # The oracle run is a control: its own role, left out of every statistic.
    roles = doc["columns"]["role"]
    assert doc["vocab"]["role"][roles[names["alpha__oracle"]]] == "control"
    task_stats = dict(
        zip(doc["dims"]["task"]["values"], doc["stats"]["task"], strict=True)
    )
    assert task_stats["alpha"]["trials"] == 6  # the oracle is not counted
    assert task_stats["alpha"]["solve_rate"] == pytest.approx(2 / 6)
    assert task_stats["alpha"]["mean_reward"] == pytest.approx(2.5 / 6)
    model_stats = dict(
        zip(doc["dims"]["model"]["values"], doc["stats"]["model"], strict=True)
    )
    a = model_stats["model-a"]
    assert a["solve_rate"] == pytest.approx(5 / 6)
    low, high = a["interval"]
    assert low < 5 / 6 < high
    # pass@k per row: the SDK's default ks (1 and powers of two) up to the
    # smallest per-task n (3 trials per task).
    ks = {row[0]: row for row in a["at_k"]}
    assert set(ks) == {1, 2}
    # alpha has 2 of 3 solved: pass@2 = 1 - C(1,2)/C(3,2) = 1, pass^2 = C(2,2)/C(3,2) = 1/3;
    # beta is always solved. Each is the mean over tasks.
    assert ks[2][1] == pytest.approx(1.0)
    assert ks[2][2] == pytest.approx((1 / 3 + 1) / 2)
    # Partial credit is kept, and graded as partial, not as a pass or a fail.
    i = names["alpha__model-b1"]
    assert doc["columns"]["reward"][i] == 0.5
    assert OUTCOMES[doc["columns"]["outcome"][i]] == "partial credit"


def test_several_jobs_are_told_apart_by_the_job_dimension(tmp_path: Path) -> None:
    a = _trial(tmp_path / "runs" / "one" / "alpha__1", "alpha", trajectory=True).parent
    b = _trial(
        tmp_path / "other" / "one" / "alpha__2", "alpha", reward=0.0, trajectory=True
    ).parent
    roots = BrowseRoots([a, b])
    assert roots.labels == ["one", "one-2"]
    doc = build_for_roots(roots, bootstrap_samples=20)
    assert sorted(set(_values(doc, "job"))) == ["one-2/.", "one/."]
    assert sorted(doc["columns"]["link"]) == ["one-2/alpha__2", "one/alpha__1"]
    assert root_labels([Path("/x/job"), Path("/y/job"), Path("/z/job")]) == [
        "job",
        "job-2",
        "job-3",
    ]


# ── attribution ───────────────────────────────────────────────────────────


def test_unscored_errors_and_scored_failures_stay_apart(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _trial(
        job / "crash__1",
        "crash",
        reward=None,
        error="ACP error -32603: the agent process exited",
        error_category="acp_error",
    )
    _trial(
        job / "verifier__1",
        "verifier",
        reward=None,
        verifier_error="RewardFileNotFoundError: no reward file",
        verifier_error_category="reward_missing",
    )
    _trial(
        job / "slow__1",
        "slow",
        reward=0.0,
        error="agent timed out after 60s",
        error_category="timeout",
    )
    _trial(job / "wrong__1", "wrong", reward=0.0)
    doc = build_outcomes([("", job)], bootstrap_samples=20)
    names = _by_name(doc)
    cols = doc["columns"]

    def cause(name: str) -> dict | None:
        index = cols["cause"][names[name]]
        return None if index < 0 else doc["causes"][index]

    # An unscored trial has no reward (never 0) and names its cause and fault.
    for name in ("crash__1", "verifier__1"):
        assert cols["reward"][names[name]] is None
        assert OUTCOMES[cols["outcome"][names[name]]] == "unscored"
    assert cause("crash__1")["fault"] == "agent"
    assert cause("verifier__1")["key"]
    assert cols["detail"][names["crash__1"]].startswith("ACP error")
    # A timeout the verifier scored stays a scored 0, marked as timed out.
    i = names["slow__1"]
    assert cols["reward"][i] == 0.0
    assert doc["vocab"]["execution"][cols["execution"][i]] == "timed_out"
    assert cause("slow__1")["key"] == "timeout_scored"
    # A plain wrong answer has no cause at all.
    assert cause("wrong__1") is None
    stats = dict(zip(doc["dims"]["task"]["values"], doc["stats"]["task"], strict=True))
    assert stats["crash"]["scored"] == 0 and stats["crash"]["unscored"] == 1
    # Cause text never carries a local path (no job or task folder filled in).
    assert str(tmp_path) not in json.dumps(doc["causes"])


def test_retries_list_every_attempt_and_sum_their_cost(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    (job / "evaluation.json").write_text(json.dumps({"tasks_dir": "/data/suite/tasks"}))
    _trial(
        job / "alpha__1",
        "alpha",
        reward=None,
        error="sandbox never started",
        error_category="sandbox_setup",
        started="2026-09-30 10:00:00",
        finished="2026-09-30 10:00:10",
    )
    _trial(
        job / "alpha__2",
        "alpha",
        reward=1.0,
        started="2026-09-30 10:01:00",
        finished="2026-09-30 10:02:00",
    )
    doc = build_outcomes([("", job)], bootstrap_samples=20)
    assert doc["n"] == 1  # one trial: the best attempt
    cols = doc["columns"]
    assert cols["attempts"] == [2]
    attempts = doc["attempt_outcomes"]["0"]
    assert [a[0] for a in attempts] == [3, 0]  # unscored, then passed
    assert cols["usd"][0] == pytest.approx(0.04)  # both attempts were paid for
    assert cols["wall_sec"][0] == pytest.approx(70.0)
    # The dataset comes from the job's tasks_dir when trials do not record one.
    assert _values(doc, "dataset") == ["suite"]


def test_integrity_verdicts_are_read_and_an_exploit_is_marked(tmp_path: Path) -> None:
    job = tmp_path / "job"
    hacked = _trial(job / "alpha__1", "alpha", reward=1.0)
    _trial(job / "alpha__2", "alpha", reward=1.0)
    (hacked / "integrity").mkdir()
    (hacked / "integrity" / "claim_verdict.json").write_text(
        json.dumps(
            {
                "core_verdict": "AgentViolation",
                "reason": "the agent wrote the reward file",
                "core": {"agent_evidence": ["e1"], "severity": "RewardRelevant"},
            }
        )
    )
    broken = _trial(job / "alpha__3", "alpha", reward=0.0)
    (broken / "integrity").mkdir()
    (broken / "integrity" / "claim_verdict.json").write_text("not json")
    doc = build_outcomes([("", job)], bootstrap_samples=20)
    names = _by_name(doc)
    verdicts = doc["columns"]["integrity"]
    assert doc["vocab"]["integrity"][verdicts[names["alpha__1"]]] == "AgentViolation"
    assert verdicts[names["alpha__2"]] == -1  # not audited
    assert verdicts[names["alpha__3"]] == -1  # an unreadable verdict degrades to none
    detail = doc["integrity_details"][str(names["alpha__1"])]
    assert detail["exploited"] is True
    assert "reward file" in detail["reason"]
    # A verdict never changes the reward.
    assert doc["columns"]["reward"][names["alpha__1"]] == 1.0


# ── splits and steps ──────────────────────────────────────────────────────


def _hillclimb(tmp_path: Path) -> Path:
    run = tmp_path / "climb"
    run.mkdir()
    (run / "hillclimb.json").write_text(
        json.dumps(
            {
                "kind": "hillclimb-demo",
                "split": {"train": ["alpha"], "test": ["beta"]},
                "rounds": [{"candidate": {"id": "r01"}}, {"candidate": {"id": "r02"}}],
                "best": {"version": "v001"},
                "stop": {"reason": "stalled"},
            }
        )
    )
    rewards = {"baseline": 0.0, "r01": 1.0, "r02": 0.5}
    for version, reward in rewards.items():
        for split, task in (("train", "alpha"), ("test", "beta")):
            for repeat in (1, 2):
                job = run / "evals" / version / split / f"trial-0{repeat}" / "job"
                job.mkdir(parents=True, exist_ok=True)
                (job / "summary.json").write_text("{}")
                _trial(job / f"{task}__{version}{repeat}", task, reward=reward)
    _trial(run / "proposer" / "r01" / "job" / "task__p1", "task", model="model-big")
    return run


def test_hillclimb_splits_rounds_and_repeats(tmp_path: Path) -> None:
    run = _hillclimb(tmp_path)
    doc = build_outcomes([("", run)], bootstrap_samples=20)
    names = _by_name(doc)
    i = names["beta__r011"]
    assert _values(doc, "split")[i] == "test"
    assert _values(doc, "step")[i] == "r01"
    assert _values(doc, "seed")[i] == "trial-01"
    assert doc["sources"]["split"] == ["hillclimb.json split"]
    # The optimizer's own rollout is not an agent trial of the benchmark.
    p = names["task__p1"]
    assert doc["vocab"]["role"][doc["columns"]["role"][p]] == "optimizer"
    training = doc["training"]
    assert training["steps"] == ["baseline", "r01", "r02"]  # hill-climb order
    assert training["group_dim"] == "split"
    series = {s["group"]: s["points"] for s in training["series"]}
    assert [p["mean_reward"] for p in series["train"]] == [0.0, 1.0, 0.5]
    assert training["heldout_split"] == ["test"]
    assert training["heldout"] == [
        {"task": "beta", "mean": [0.0, 1.0, 0.5], "n": [2, 2, 2]}
    ]
    assert any("best version v001" in note for note in doc["notes"])


def test_recorded_steps_are_ordered_numerically(tmp_path: Path) -> None:
    job = tmp_path / "job"
    for step in (10, 2, 1):
        _trial(
            job / f"alpha__s{step}",
            "alpha",
            reward=step / 10,
            config={"policy_version": step},
        )
    doc = build_outcomes([("", job)], bootstrap_samples=20)
    assert doc["training"]["steps"] == ["1", "2", "10"]
    assert doc["sources"]["step"] == ["recorded policy_version/step/checkpoint"]


def test_rollouts_jsonl_steps_are_joined_by_rollout_folder(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    run = jobs / "2026-09-30__10-00-00"
    for step in (0, 1):
        _trial(run / f"alpha__s{step}", "alpha", reward=float(step))
    (jobs / "rollouts.jsonl").write_text(
        "".join(
            json.dumps({"rollout_dir": str(run / f"alpha__s{s}"), "step": s}) + "\n"
            for s in (0, 1)
        )
    )
    doc = build_outcomes([("", jobs)], bootstrap_samples=20)
    assert doc["training"]["steps"] == ["0", "1"]


def test_no_steps_means_no_training_view(two_models: Path) -> None:
    doc = build_outcomes([("", two_models)], bootstrap_samples=20)
    assert doc["training"] is None
    assert any("no training steps" in note for note in doc["notes"])


# ── Pareto ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "points, frontier",
    [
        ([], []),
        ([(1.0, 0.5)], [0]),
        # (2, 0.4) is dominated by (1, 0.5): costlier and worse.
        ([(1.0, 0.5), (2.0, 0.4), (3.0, 0.9)], [0, 2]),
        # Same cost, lower reward: dominated.
        ([(1.0, 0.5), (1.0, 0.3)], [0]),
        # Exact ties both stay on the frontier.
        ([(1.0, 0.5), (1.0, 0.5)], [0, 1]),
        # Cheaper but worse is still on the frontier (a trade-off).
        ([(5.0, 0.9), (0.5, 0.1), (2.0, 0.1)], [1, 0]),
    ],
)
def test_pareto_frontier(points, frontier) -> None:
    assert pareto_frontier(points) == frontier


def test_bootstrap_intervals_are_deterministic_and_bracket_the_estimate() -> None:
    clusters = [{"y": (float(k % 3), 3), "x": (10.0 * k, 3)} for k in range(12)]
    first = bootstrap_ratios(clusters, ["x", "y", "z"], seed=7, samples=300)
    again = bootstrap_ratios(clusters, ["x", "y", "z"], seed=7, samples=300)
    assert first == again
    for m in ("x", "y"):
        assert first[m]["lo"] <= first[m]["v"] <= first[m]["hi"]
        assert first[m]["lo"] < first[m]["hi"]
    assert first["z"] == {"v": None, "lo": None, "hi": None, "n": 0}
    one = bootstrap_ratios(clusters[:1], ["y"], seed=7)
    assert one["y"]["lo"] is None  # one cluster: no interval, never a zero-width one


def test_pareto_points_and_frontiers_per_partition(two_models: Path) -> None:
    doc = build_outcomes([("", two_models)], bootstrap_samples=50)
    block = doc["pareto"]["model"]["none"]
    points = {p["group"]: p for p in block["points"]}
    assert set(points) == {"model-a", "model-b"}
    a, b = points["model-a"], points["model-b"]
    assert a["m"]["mean_reward"][0] == pytest.approx(5 / 6)
    assert b["m"]["mean_reward"][0] == pytest.approx(3.5 / 6)
    assert a["m"]["usd"][0] == pytest.approx(0.02)
    assert a["trials"] == 6 and a["tasks"] == 2
    # Same cost, better reward: model-a alone is on the usd frontier.
    frontier = block["frontiers"]["usd|mean_reward"]["all"]
    assert [block["points"][j]["group"] for j in frontier] == ["model-a"]


# ── export ────────────────────────────────────────────────────────────────


def test_export_masks_secrets_paths_and_links(tmp_path: Path) -> None:
    job = tmp_path / "private-job"
    _trial(
        job / "alpha__1",
        "alpha",
        reward=None,
        error=f"provider said 401 for key {FAKE_KEY} at {job}/alpha__1",
        error_category="provider_auth",
        trajectory=True,
    )
    _trial(job / "beta__1", "beta", trajectory=True)
    out, categories, n = export_html([job], tmp_path / "share" / "outcomes.html")
    html = out.read_text()
    assert n == 2
    assert FAKE_KEY not in html
    assert str(tmp_path) not in html
    assert sum(categories.values()) >= 1
    assert '"mode":"export"' in html.replace(" ", "")
    boot = json.loads(
        html.split('<script id="bf-payload" type="application/json">', 1)[1].split(
            "</script>", 1
        )[0]
    )
    shared = boot["outcomes"]
    assert shared["columns"]["link"] == [None, None]
    assert "timing" not in shared
    assert shared["redaction"]


def test_redaction_keeps_the_document_usable(two_models: Path) -> None:
    doc = build_outcomes([("", two_models)], bootstrap_samples=20)
    shared, categories = redact_for_export(doc, [two_models])
    assert not categories
    assert shared["columns"]["tokens"] == doc["columns"]["tokens"]
    assert shared["dims"] == doc["dims"]
    assert shared["pareto"] == doc["pareto"]


def test_cli_export(tmp_path: Path) -> None:
    from benchflow.cli.main import app

    job = tmp_path / "job"
    _trial(job / "alpha__1", "alpha")
    out = tmp_path / "o.html"
    result = CliRunner().invoke(app, ["eval", "view", str(job), "--export", str(out)])
    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "Wrote" in output and "(1 trial)" in output
    assert "Masked for you: nothing" in output
    assert out.is_file()


# ── server ────────────────────────────────────────────────────────────────


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_browse_server_serves_outcomes_for_several_jobs(tmp_path: Path, capsys) -> None:
    from benchflow.trajectories.viewer.server import serve

    one = tmp_path / "one"
    two = tmp_path / "two"
    _trial(one / "alpha__1", "alpha", trajectory=True)
    _trial(two / "alpha__2", "alpha", reward=0.0, trajectory=True)
    port = _free_port()
    thread = threading.Thread(
        target=serve,
        args=(str(one), port),
        kwargs={"more_paths": [str(two)]},
        daemon=True,
    )
    thread.start()
    base = f"http://localhost:{port}/"
    deadline = time.monotonic() + 20
    while True:
        try:
            status, body = _get(base)
            break
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.1)
    assert status == 200
    assert b'"jobviews":true' in body.replace(b" ", b"")
    status, body = _get(base + "api/outcomes")
    assert status == 200
    doc = json.loads(body)
    assert doc["n"] == 2
    assert sorted(doc["columns"]["link"]) == ["one/alpha__1", "two/alpha__2"]
    status, _ = _get(base + "api/rollout?id=two/alpha__2")
    assert status == 200
    status, _ = _get(base + "api/rollout?id=two/../one/alpha__1")
    assert status == 404
