"""`bench eval inspect` and `bench eval compare` mirror bf.load_job / bf.compare.

Both commands call the same Python functions and, with --json, print the same
versioned documents (docs/reference/json-export.md). compare exits 0 when the
sides are comparable, 1 on an undeclared setting difference with
--on-mismatch raise, 2 for a usage error.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import benchflow as bf
from benchflow.cli.main import app
from tests.test_python_sdk_load_job import _job, _trial


def _run(*args: str):
    return CliRunner().invoke(app, ["eval", *args])


def test_inspect_json_is_the_job_document(tmp_path: Path) -> None:
    job = _job(tmp_path, "a", {"t1": 1.0, "t2": 0.0})
    result = _run("inspect", str(job), "--json")
    assert result.exit_code == 0, result.output
    doc = json.loads(result.stdout)
    assert doc == bf.load_job(job).to_json_dict()


def test_inspect_table_names_the_denominators(tmp_path: Path) -> None:
    job = _job(tmp_path, "a", {"t1": 1.0, "t2": 0.0})
    result = _run("inspect", str(job))
    assert result.exit_code == 0, result.output
    assert "1/2 scored passed" in result.output and "t2" in result.output
    assert "2 control runs left out" in result.output


def test_inspect_a_trial_prints_the_trial_document(tmp_path: Path) -> None:
    trial = _trial(tmp_path / "job", "hello")
    doc = json.loads(_run("inspect", str(trial), "--json", "--trajectories").stdout)
    assert doc["kind"] == "benchflow.trial" and doc["trajectory"]


def test_inspect_writes_to_a_file(tmp_path: Path) -> None:
    job = _job(tmp_path, "a", {"t1": 1.0})
    out = tmp_path / "out.json"
    result = _run("inspect", str(job), "--json", "--no-verifier", "--out", str(out))
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["trials"][0]["verifier"] is None


def test_compare_json_is_the_comparison_document(tmp_path: Path) -> None:
    a = _job(tmp_path, "a", {"t1": 1.0, "t2": 0.0})
    b = _job(tmp_path, "b", {"t1": 1.0, "t2": 1.0})
    result = _run("compare", str(a), str(b), "--json", "--labels", "a", "b")
    assert result.exit_code == 0, result.output
    doc = json.loads(result.stdout)
    expected = bf.compare(a, b, labels=("a", "b")).to_json_dict()
    assert doc == expected


def test_compare_markdown_by_default(tmp_path: Path) -> None:
    a = _job(tmp_path, "a", {"t1": 1.0})
    b = _job(tmp_path, "b", {"t1": 0.0})
    result = _run("compare", str(a), str(b))
    assert result.exit_code == 0
    assert "| Task | a | b |" in result.output


def test_compare_refuses_an_undeclared_difference_when_asked(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _trial(a, "t1", model="claude-haiku-4-5")
    _trial(b, "t1", model="claude-sonnet-4-6")
    refused = _run("compare", str(a), str(b), "--on-mismatch", "raise")
    assert refused.exit_code == 1 and "model" in refused.output
    warned = _run("compare", str(a), str(b))
    assert warned.exit_code == 0 and "Settings differ" in warned.output
    declared = _run(
        "compare", str(a), str(b), "--vary", "model", "--on-mismatch", "raise"
    )
    assert declared.exit_code == 0, declared.output


def test_compare_accepts_globs_for_each_side(tmp_path: Path) -> None:
    for task in ("t1", "t2"):
        _trial(tmp_path / task / "side-a" / "ts", task)
        _trial(tmp_path / task / "side-b" / "ts", task, reward=0.0)
    result = _run(
        "compare",
        str(tmp_path / "*" / "side-a"),
        str(tmp_path / "*" / "side-b"),
        "--json",
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["summary"]["b_lower"] == 2


def test_missing_paths_are_a_clean_error(tmp_path: Path) -> None:
    result = _run("compare", str(tmp_path / "nope"), str(tmp_path / "nope2"))
    assert result.exit_code == 2 and "no trial" in result.output
    assert "Traceback" not in result.output


def test_inspect_keeps_task_names_whole_on_a_narrow_terminal(tmp_path: Path) -> None:
    """At 80 columns the table cut every task name to
    'hello-w…', the one column needed to find a trial."""
    job = tmp_path / "job"
    _trial(job, "synthetic-long-task-name-for-narrow-terminal-check")
    result = CliRunner().invoke(app, ["eval", "inspect", str(job)], terminal_width=80)
    assert result.exit_code == 0, result.output
    first_column = "".join(
        line.split("│")[1].strip()
        for line in result.output.splitlines()
        if line.startswith("│")
    )
    assert "synthetic-long-task-name-for-narrow-terminal-check" in first_column


def test_inspect_a_json_file_is_a_clean_usage_error(tmp_path: Path) -> None:
    """The error must stay on one line at 80 columns (a
    wrapped path cannot be copied or grepped from a CI log)."""
    (tmp_path / "summary.json").write_text('{\n  "total": 1\n}\n')
    result = _run("inspect", str(tmp_path / "summary.json"))
    assert result.exit_code == 2 and "pass a job directory" in result.output
    assert "Traceback" not in result.output


def test_inspect_by_agent_and_model_prints_one_line_per_group(tmp_path: Path) -> None:
    """The one headline pooled two agents and two builds."""
    job = tmp_path / "job"
    _trial(job, "t1", agent="claude-agent-acp", model="sonnet", suffix="00000001")
    _trial(job, "t1", agent="codex-acp", model="gpt-5.5", reward=0.0, suffix="00000002")
    result = CliRunner().invoke(
        app, ["eval", "inspect", str(job), "--by", "agent,model"], terminal_width=200
    )
    assert result.exit_code == 0, result.output
    assert "agent=claude-agent-acp model=sonnet: 1/1 scored passed" in result.output
    assert "agent=codex-acp model=gpt-5.5: 0/1 scored passed" in result.output


def test_inspect_include_controls_and_a_controls_only_job(tmp_path: Path) -> None:
    """An oracle-only job read '0/0 scored passed, 0 attempted
    … N control runs left out', and inspect had no --include-controls."""
    job = tmp_path / "job"
    _trial(job, "t1", agent="oracle", model=None)
    plain = CliRunner().invoke(app, ["eval", "inspect", str(job)], terminal_width=200)
    assert "only control runs" in plain.output and "1/1 scored passed" in plain.output
    with_controls = CliRunner().invoke(
        app, ["eval", "inspect", str(job), "--include-controls"], terminal_width=200
    )
    assert "1/1 scored passed" in with_controls.output


def test_inspect_csv_writes_the_records(tmp_path: Path) -> None:
    """No CSV path from the CLI."""
    import csv

    job = _job(tmp_path, "a", {"t1": 1.0, "t2": 0.0})
    out = tmp_path / "trials.csv"
    result = _run("inspect", str(job), "--csv", str(out))
    assert result.exit_code == 0, result.output
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 4 and {"task_name", "reward", "execution"} <= set(rows[0])


def test_compare_by_and_cli_wording_and_parent_labels(tmp_path: Path) -> None:
    """Default labels were timestamp folder names, and the
    warning spoke Python ('Pass vary=(...)')."""
    for side, model in (("claude-cli", "sonnet"), ("codex-cli", "gpt-5.5")):
        _trial(
            tmp_path / side / "2026-01-01__12-00-00",
            "t1",
            agent="claude-agent-acp" if side == "claude-cli" else "codex-acp",
            model=model,
        )
    a = tmp_path / "claude-cli" / "2026-01-01__12-00-00"
    b = tmp_path / "codex-cli" / "2026-01-01__12-00-00"
    result = CliRunner().invoke(
        app, ["eval", "compare", str(a), str(b)], terminal_width=200
    )
    assert result.exit_code == 0, result.output
    assert "| Task | claude-cli | codex-cli |" in result.output
    assert "--vary" in result.output and "vary=(" not in result.output
    grouped = CliRunner().invoke(
        app,
        [
            "eval",
            "compare",
            str(a),
            str(b),
            "--by",
            "task_digest",
            "--vary",
            "harness",
            "--vary",
            "model",
        ],
        terminal_width=200,
    )
    assert grouped.exit_code == 0, grouped.output


def test_inspect_drops_an_empty_cost_column(tmp_path: Path) -> None:
    """At 80 columns an all-blank Cost column squeezed Task."""
    job = tmp_path / "j"
    _trial(job, "synthetic-long-task-name-cost-check", cost=None)
    unpriced = _run("inspect", str(job))
    assert unpriced.exit_code == 0, unpriced.output
    assert "Cost" not in unpriced.output
    _trial(tmp_path / "k", "t", cost=0.01)
    assert "Cost" in _run("inspect", str(tmp_path / "k")).output


def test_default_labels_use_each_sides_own_name(tmp_path: Path) -> None:
    """A named job compared with a timestamped one keeps its own name.

    Regression test: comparing
    ``jobs/matrix/fake/trial-01/2026-01-01__12-00-00`` with ``jobs/batch-oracle``
    labelled the second side ``jobs`` (its parent folder), because one
    timestamp side sent both sides to their parents.
    """
    from benchflow.jobs import _default_labels

    stamped = tmp_path / "trial-01" / "2026-01-01__12-00-00"
    named = tmp_path / "jobs" / "batch-oracle"
    assert _default_labels(stamped, named) == ("trial-01", "batch-oracle")
    assert _default_labels(named, stamped) == ("batch-oracle", "trial-01")
    # Same names still fall back to the parents, then to A/B.
    assert _default_labels(tmp_path / "x" / "run", tmp_path / "y" / "run") == ("x", "y")
    assert _default_labels(tmp_path / "x" / "run", tmp_path / "x" / "run") == ("A", "B")
