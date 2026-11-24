"""The oracle-parity tally (tools/parity/tally.py), R732B's method as code.

The rows below are the 0.10.0rc4 gate-1 record's own (attempt 3c4c79a1de2e):
its five mismatches against Harbor 0.26.0, each re-run alone, and a sample of
the unscored rows. The record's reading of each is the expected class: two
flakes, one accepted hardware fact, two verifier caps under load. A planted
regression, a broken must-show and a changed reward each block.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.parity import tally as T

PARITY = Path(__file__).resolve().parents[1] / "tools" / "parity"


@pytest.fixture(scope="module")
def ledger():
    return T.load_ledger(PARITY / "ledger")


@pytest.fixture(scope="module")
def ref():
    return T.load_reference(PARITY / "harbor-ref-0.26.0.tsv")


def _row(key, end="SCORED", reward="", why="", **kw):
    return T.Row(key=key, end=end, reward=reward, why=why, **kw)


def test_the_reference_has_the_404(ref):
    assert len(ref) == 404
    assert ref["algotune/algotune-robust-kalman-filter"] == 1.0079
    assert ref["sldbench/easy_question_scaling_law"] == -1.0


def test_a_match_passes(ledger, ref):
    v = T.classify(_row("hello-world/hello-world", reward="1.0"), ref, ledger, [])
    assert (v.cls, v.blocking) == ("MATCH", False)


def test_a_mismatch_that_matches_alone_is_a_flake(ledger, ref):
    row = _row("ade-bench/ade-bench-asana002", reward="0.0")
    v = T.classify(row, ref, ledger, [_row(row.key, reward="1.0")])
    assert (v.cls, v.blocking) == ("FLAKE", False)


def test_the_flaky_ledger_allows_more_reruns(ledger, ref):
    key = "svc-openthoughts-tblite/cpp-daemon-sighup-segfault"
    reruns = [_row(key, reward="0.0"), _row(key, reward="1.0")]
    v = T.classify(_row(key, reward="0.0"), ref, ledger, reruns)
    assert (v.cls, v.blocking) == ("FLAKE", False)
    assert T.reruns_allowed(key, ledger) == 3
    assert T.reruns_allowed("ade-bench/ade-bench-asana002", ledger) == 1


def test_a_hardware_fact_on_the_ledger_is_accepted(ledger, ref):
    key = "algotune/algotune-robust-kalman-filter"
    v = T.classify(_row(key, reward="1.0"), ref, ledger, [_row(key, reward="1.0")])
    assert (v.cls, v.blocking) == ("ACCEPTED", False)


def test_a_verifier_at_its_cap_under_load_is_named(ledger, ref):
    key = "swtbench-verified/scikit-learn__scikit-learn-15100"
    row = _row(key, reward="0.0", verifier_secs=1219.7, verifier_cap=1200.0)
    rerun = _row(key, reward="0.0", verifier_secs=1190.0, verifier_cap=1200.0)
    pr = T.classify(row, ref, ledger, [rerun], mode="pr")
    assert (pr.cls, pr.blocking) == ("TIMEOUT_UNDER_LOAD", False)
    nightly = T.classify(row, ref, ledger, [rerun], mode="nightly")
    assert (nightly.cls, nightly.blocking) == ("TIMEOUT_UNDER_LOAD", True)


@pytest.mark.parametrize(
    ("key", "why"),
    [
        (
            "bfcl/bfcl-parallel-93",
            "TaskConfigKeyError: /w/bfcl-parallel-93/task.toml: - unknown task.toml key "
            "'sandbox.resources.cpus' in the [sandbox] table",
        ),
        (
            "binary-audit/radare2-decompile",
            "FileNotFoundError: Oracle requires oracle/solve.sh or legacy solution/solve.sh: /w/x",
        ),
        (
            "svc-seta-env/1243",
            "TaskRefusedError: task refused before the turn: seta-env 1243: this task's "
            "reference solution fails under BenchFlow's root-exec limit; not scored",
        ),
    ],
)
def test_an_unscored_row_with_an_accepted_reason_passes(ledger, ref, key, why):
    v = T.classify(_row(key, end="UNSCORED", why=why), ref, ledger, [])
    assert (v.cls, v.blocking) == ("ACCEPTED_UNSCORED", False)


def test_an_unscored_row_with_another_reason_blocks(ledger, ref):
    row = _row(
        "hello-world/hello-world", end="UNSCORED", why="RuntimeError: something new"
    )
    v = T.classify(row, ref, ledger, [])
    assert (v.cls, v.blocking) == ("REGRESSION", True)


def test_a_planted_regression_blocks(ledger, ref):
    key = "hello-world/hello-world"
    v = T.classify(_row(key, reward="0.0"), ref, ledger, [_row(key, reward="0.0")])
    assert (v.cls, v.blocking) == ("REGRESSION", True)


def test_a_broken_must_show_blocks(ledger, ref):
    key = "sldbench/easy_question_scaling_law"
    v = T.classify(_row(key, reward="0.0"), ref, ledger, [_row(key, reward="0.0")])
    assert v.blocking and v.cls == "MUST_SHOW"
    algotune = "algotune/algotune-robust-kalman-filter"
    v = T.classify(
        _row(algotune, reward="0.0"), ref, ledger, [_row(algotune, reward="0.0")]
    )
    assert v.blocking and v.cls == "MUST_SHOW"


def test_the_refused_sentinel_must_be_refused_before_its_turn(ledger, ref):
    v = T.classify(_row("svc-seta-env/1243", reward="1.0"), ref, ledger, [])
    assert v.blocking and v.cls == "MUST_SHOW"


@pytest.mark.parametrize(
    "extra",
    [
        {"reward_changed": "True"},
        {"provenance_outcome": "provenance_incomplete"},
        {"error_category": "integrity_unavailable"},
        {"walker": "find"},
    ],
)
def test_the_counts_block(ledger, ref, extra):
    v = T.classify(
        _row("hello-world/hello-world", reward="1.0", **extra), ref, ledger, []
    )
    assert (v.cls, v.blocking) == ("COUNT", True)


def test_a_row_harbor_never_scored_is_context(ledger, ref):
    key = "mlgym-bench/mlgym-regression-kaggle-house-price"
    v = T.classify(
        _row(key, end="UNSCORED", why="verifier wrote no usable reward"),
        ref,
        ledger,
        [],
    )
    assert v.blocking is False


def test_the_summary_blocks_only_on_blocking_rows(ledger, ref, tmp_path):
    rows = [
        _row("hello-world/hello-world", reward="1.0"),
        _row("ade-bench/ade-bench-asana002", reward="0.0"),
    ]
    reruns = [_row("ade-bench/ade-bench-asana002", reward="1.0")]
    verdicts = T.tally(rows, reruns, ref, ledger)
    assert T.exit_code(verdicts) == 0
    text = T.render(verdicts)
    assert "FLAKE" in text and "MATCH" in text
    verdicts.append(
        T.classify(_row("hello-world/hello-world", reward="0.0"), ref, ledger, [])
    )
    assert T.exit_code(verdicts) == 1


def test_an_excluded_task_in_the_rows_is_named_not_blocking(ledger, ref):
    key = "reasoning-gym/reasoning-gym-algorithmic-letter-jumble-hard0292e21f-seed45-2"
    row = _row(key, end="UNSCORED", why="verifier wrote no usable reward")
    v = T.classify(row, ref, ledger, [])
    assert (v.cls, v.blocking) == ("EXCLUDED", False)


def test_a_task_bigger_than_the_runner_is_named_not_a_regression(ledger, ref):
    # 0.10.0's CI: otel-bench/ruby-microservices asks for 4 CPUs on a 2-vCPU
    # runner, Docker refused it, and the tally called it a regression.
    why = "the task asks for 4 CPUs; this runner has 2"
    row = _row("hello-world/hello-world", end=T.RUNNER_CAPACITY, why=why)
    pr = T.classify(row, ref, ledger, [], mode="pr")
    assert (pr.cls, pr.blocking, pr.note) == ("RUNNER_CAPACITY", False, why)
    nightly = T.classify(row, ref, ledger, [], mode="nightly")
    assert (nightly.cls, nightly.blocking) == ("RUNNER_CAPACITY", True)


def test_the_runner_does_not_start_a_task_bigger_than_itself(tmp_path, monkeypatch):
    from tools.parity import run as R

    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text("[environment]\ncpus = 4\n")
    monkeypatch.setattr(R.os, "cpu_count", lambda: 2)

    def never(*a, **kw):
        raise AssertionError("bench must not run")

    monkeypatch.setattr(R.subprocess, "run", never)
    row = R.run_one("bench", "otel-bench/ruby-microservices", task, tmp_path / "jobs")
    assert row.end == T.RUNNER_CAPACITY
    assert row.why == "the task asks for 4 CPUs; this runner has 2"
    (task / "task.toml").write_text("[environment]\ncpus = 2\n")
    with pytest.raises(AssertionError, match="bench must not run"):
        R.run_one("bench", "otel-bench/ruby-microservices", task, tmp_path / "jobs")
