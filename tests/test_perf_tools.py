"""The perf instrument in tools/perf (round PY8, target T0).

The statistics every speed claim is reported with (median, bootstrap
interval, paired difference, Wilson interval), how a trial's timing.json
becomes a CSV row, and the task the fake-provider route runs.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pytest

from tools.perf import setup_trials, summarise
from tools.perf._common import append_rows, interleaved, parse_arm


def test_wilson_interval_matches_the_textbook_values():
    low, high = summarise.wilson_interval(3, 3)
    assert low == pytest.approx(0.4385, abs=1e-4)
    assert high == 1.0
    low, high = summarise.wilson_interval(0, 3)
    assert low == pytest.approx(0.0, abs=1e-12)
    assert high == pytest.approx(0.5615, abs=1e-4)
    low, high = summarise.wilson_interval(5, 10)
    assert (low, high) == pytest.approx((0.2366, 0.7634), abs=1e-4)


def test_the_bootstrap_interval_is_repeatable_and_brackets_the_median():
    values = [10.2, 11.0, 9.8, 10.5, 10.1, 12.3, 9.9]
    first = summarise.bootstrap_interval(values)
    assert first == summarise.bootstrap_interval(values)
    assert first[0] <= summarise.median(values) <= first[1]
    assert summarise.bootstrap_interval([4.0] * 5) == (4.0, 4.0)


def test_paired_differences_pair_trials_of_the_same_round():
    rows = [
        {"arm": "base", "round": "0", "x": "10"},
        {"arm": "tip", "round": "0", "x": "7"},
        {"arm": "base", "round": "1", "x": "12"},
        {"arm": "tip", "round": "1", "x": "8"},
        {"arm": "base", "round": "2", "x": "11"},  # tip missed round 2
    ]
    diffs = summarise.paired_differences(
        rows,
        metric="x",
        arm_column="arm",
        baseline="base",
        other="tip",
        pair_by=["round"],
    )
    assert diffs == [-3.0, -4.0]


def test_summarise_reports_absent_spans_and_counts_unscored_as_not_passed(
    tmp_path, capsys
):
    path = tmp_path / "trials.csv"
    append_rows(
        path,
        ("arm", "round", "reward", "litellm_install", "total"),
        [
            {"arm": "base", "round": 0, "reward": 1.0, "total": 30.0},
            {"arm": "tip", "round": 0, "reward": 1.0, "total": 20.0},
            {"arm": "base", "round": 1, "reward": None, "total": 31.0},
            {"arm": "tip", "round": 1, "reward": 1.0, "total": 21.0},
        ],
    )
    argv = [str(path), "--metric", "litellm_install", "--metric", "total"]
    assert summarise.main([*argv, "--pass-column", "reward"]) == 0
    out = capsys.readouterr().out
    assert "litellm_install | base | n=0 (absent)" in out
    assert "total | tip - base (paired, n=2) | median -10.000" in out
    assert "| base | 1/2 " in out and "| tip | 2/2 " in out


def test_a_timing_json_becomes_a_row_with_unnamed_time():
    row = setup_trials.timing_row(
        {
            "slot_wait": 0.01,
            "environment_setup": 2.0,
            "install_agent": 5.0,
            "litellm_install": 9.0,
            "proxy_health_wait": 2.0,
            "acp_connect": 1.0,
            "agent_setup": 13.0,
            "agent_execution": 4.0,
            "verifier": 3.0,
            "harden_before_verify": {"total": 1.0, "reclaim_disk": 0.5},
            "cleanup": 1.5,
            "total": 30.0,
        }
    )
    assert row["install_agent"] == 5.0
    assert row["harden_before_verify.reclaim_disk"] == 0.5
    assert row["harden_before_verify.restore_workspace"] is None
    # 30 - (2 + 5 + 13 + 4 + 3 + 1.5): the nested spans are not subtracted.
    assert row["unnamed"] == pytest.approx(1.5)
    old = setup_trials.timing_row({"environment_setup": 2.0, "total": 9.0})
    assert old["install_agent"] is None and old["unnamed"] == pytest.approx(7.0)


def test_the_route_task_carries_the_egress_policy_for_the_sandbox_proxy(tmp_path):
    task = setup_trials.materialize_task(
        tmp_path, variant="deterministic", proxy="sandbox"
    )
    text = (task / "task.md").read_text()
    assert "  network_mode: allowlist\n  allowed_hosts:\n  - example.com" in text
    assert (task / "environment.toml").is_file()
    host = setup_trials.materialize_task(
        tmp_path, variant="deterministic", proxy="host"
    )
    assert "network_mode" not in (host / "task.md").read_text()


def test_the_no_python_task_has_no_python_and_no_in_sandbox_fake(tmp_path):
    task = setup_trials.materialize_task(tmp_path, variant="no-python", proxy="sandbox")
    dockerfile = (task / "environment" / "Dockerfile").read_text()
    assert "apt-get install" in dockerfile and "python3" not in dockerfile
    assert not (task / "environment.toml").exists()
    assert "manifest" not in (task / "task.md").read_text()


def test_model_agents_reach_the_fake_and_controls_get_no_model():
    assert setup_trials.agent_args("nop", None) == ["--agent", "nop"]
    codex = setup_trials.agent_args("codex-acp", "http://172.17.0.1:8901")
    assert "BENCHFLOW_PROVIDER_BASE_URL=http://172.17.0.1:8901/v1" in codex
    claude = setup_trials.agent_args("claude-agent-acp", "http://172.17.0.1:8901")
    assert claude[:4] == ["--agent", "claude-agent-acp", "--model", "claude-haiku-4-5"]
    assert "BENCHFLOW_PROVIDER_BASE_URL=http://172.17.0.1:8901" in claude


def test_arms_parse_and_rotate():
    base = parse_arm("base=2ea2ad00:/tmp/venv-a")
    tip = parse_arm("tip=abc12345:/tmp/venv-b")
    assert (base.name, base.commit, base.venv) == (
        "base",
        "2ea2ad00",
        Path("/tmp/venv-a"),
    )
    assert [a.name for a in interleaved([base, tip], 0)] == ["base", "tip"]
    assert [a.name for a in interleaved([base, tip], 1)] == ["tip", "base"]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_arm("base:/tmp/venv-a")


def test_rows_append_under_one_header(tmp_path):
    path = tmp_path / "rows.csv"
    append_rows(path, ("a", "b"), [{"a": 1, "b": None}])
    append_rows(path, ("a", "b"), [{"a": 2.5, "b": "x"}])
    with path.open(newline="") as handle:
        assert list(csv.DictReader(handle)) == [
            {"a": "1", "b": ""},
            {"a": "2.5", "b": "x"},
        ]


# A stand-in for an arm's ``bench``: it writes one trial named after its agent
# under --jobs-dir (oracle scores 1, anything else 0) and talks to nothing.
STUB_BENCH = """#!/bin/sh
jobs=""; agent=""
while [ $# -gt 0 ]; do
  case "$1" in
    --jobs-dir) jobs="$2"; shift ;;
    --agent) agent="$2"; shift ;;
  esac
  shift
done
reward=0; [ "$agent" = oracle ] && reward=1
mkdir -p "$jobs/job/$agent-trial"
printf '{"rewards": {"reward": %s}, "timing": {"total": 1.0}}' "$reward" \\
  > "$jobs/job/$agent-trial/result.json"
echo "$agent ran"
"""


def _stub_arms(tmp_path: Path) -> list[str]:
    specs = []
    for name in ("base", "tip"):
        exe = tmp_path / f"venv-{name}" / "bin" / "bench"
        exe.parent.mkdir(parents=True)
        exe.write_text(STUB_BENCH)
        exe.chmod(0o755)
        specs += ["--arm", f"{name}=0000000{name[0]}:{exe.parent.parent}"]
    return specs


@pytest.fixture
def stub_trials(tmp_path, monkeypatch):
    """setup_trials.main with stub arms, no Docker, no provenance probes."""
    task = tmp_path / "task"
    task.mkdir()
    monkeypatch.setattr(setup_trials, "materialize_task", lambda *a, **k: task)
    monkeypatch.setattr(
        setup_trials, "provenance", lambda script, **k: {"script": script}
    )
    arms = _stub_arms(tmp_path)

    def run(agent: str, work: str) -> int:
        # Daytona settings keep Docker (the base image pull) out of the test;
        # the stub never reaches a sandbox.
        return setup_trials.main(
            [
                *arms,
                "--agent",
                agent,
                "--backend",
                "daytona",
                "--count",
                "1",
                "--work",
                str(tmp_path / work),
                "--out",
                str(tmp_path / "controls.csv"),
            ]
        )

    return run


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_the_nop_control_keeps_the_oracle_controls_trials(tmp_path, stub_trials):
    # Both controls into one --work, as an operator may well do.
    assert stub_trials("oracle", "ctl") == 0
    assert stub_trials("nop", "ctl") == 0
    rows = _rows(tmp_path / "controls.csv")
    assert [(r["agent"], r["arm"], r["reward"]) for r in rows] == [
        ("oracle", "base", "1"),
        ("oracle", "tip", "1"),
        ("nop", "base", "0"),
        ("nop", "tip", "0"),
    ]
    # Every row's trial is still there, and each its own.
    for row in rows:
        assert (Path(row["trial_dir"]) / "result.json").is_file(), row
    assert len({row["trial_dir"] for row in rows}) == 4
    logs = sorted((tmp_path / "ctl" / "logs").rglob("*.txt"))
    assert [log.read_text() for log in logs] == ["nop ran\n"] * 2 + ["oracle ran\n"] * 2


def test_a_rerun_into_a_used_work_folder_refuses_and_keeps_the_trials(
    tmp_path, stub_trials
):
    assert stub_trials("oracle", "ctl") == 0
    trials = sorted((tmp_path / "ctl" / "jobs").rglob("result.json"))
    assert len(trials) == 2
    stamps = [trial.stat().st_mtime_ns for trial in trials]
    with pytest.raises(SystemExit, match="already exists"):
        stub_trials("oracle", "ctl")
    assert [trial.stat().st_mtime_ns for trial in trials] == stamps
    assert len(_rows(tmp_path / "controls.csv")) == 2
    # A fresh --work runs.
    assert stub_trials("oracle", "ctl-2") == 0


@pytest.mark.parametrize("name", ["..", ".", "../..", "a/b", ".hidden", ""])
def test_arm_names_cannot_leave_the_work_folder(name):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_arm(f"{name}=2ea2ad00:/tmp/venv-a")


def test_summarise_refuses_to_pair_two_runs_mixed_in_one_csv(tmp_path, capsys):
    path = tmp_path / "controls.csv"
    rows = [
        {"arm": arm, "round": 0, "agent": agent, "reward": reward}
        for agent, reward in (("oracle", 1.0), ("nop", 0.0))
        for arm in ("base", "tip")
    ]
    append_rows(path, ("arm", "round", "agent", "reward"), rows)
    with pytest.raises(SystemExit, match="arm base has two rows for round = 0"):
        summarise.main([str(path), "--pass-column", "reward"])
    capsys.readouterr()
    assert summarise.main([str(path), "--by", "agent", "--pass-column", "reward"]) == 0
    out = capsys.readouterr().out
    assert out.count("| base | 1/1 ") == 1 and out.count("| base | 0/1 ") == 1
