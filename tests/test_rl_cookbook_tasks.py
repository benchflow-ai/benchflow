"""The RL cookbook task family (docs/examples/rl/tasks) and its verifier.

The generator and verifier are imported by path, like the other docs
examples. Sandboxed checks (oracle 1 and do-nothing 0 on Docker and Daytona)
live in the cookbook's README; these tests pin the parts that run anywhere.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchflow._utils.task_authoring import check_task

TASKS = Path(__file__).resolve().parents[1] / "docs" / "examples" / "rl" / "tasks"
sys.path.insert(0, str(TASKS))
import family  # noqa: E402
import generate  # noqa: E402
import verify  # noqa: E402

DATA_SEEDS = [0, 1, 2, 4, 5, 6, 900000, 900001, 900002]


def test_train_and_test_seed_ranges_are_disjoint() -> None:
    _, train_end = generate.SPLITS["train"]
    test_first, test_end = generate.SPLITS["test"]
    assert train_end <= test_first
    assert test_end <= family.CONTROL_SEED


def test_generated_split_is_balanced_and_valid(tmp_path: Path) -> None:
    generate.main(["--split", "train", "--out", str(tmp_path), "--count", "8"])
    rows = [
        json.loads(line)
        for line in (tmp_path / "manifest.jsonl").read_text().splitlines()
    ]
    assert [r["kind"] for r in rows] == ["sql", "log", "csv", "bugfix"] * 2
    for row in rows[:4]:
        task = tmp_path / row["task"]
        assert check_task(task) == []
        assert (task / "verifier" / "expected.json").is_file()
        assert (task / "oracle" / "solve.sh").stat().st_mode & 0o111


def test_instances_are_deterministic(tmp_path: Path) -> None:
    for seed in (0, 3, 900001):
        first, second = family.build(seed), family.build(seed)
        assert first.prompt == second.prompt
        assert first.expected == second.expected
        a, b = tmp_path / f"{seed}-a", tmp_path / f"{seed}-b"
        family.materialize(first.kind, seed, a)
        family.materialize(first.kind, seed, b)
        for path in a.iterdir():
            assert path.read_bytes() == (b / path.name).read_bytes(), path.name


def test_policy_view_holds_no_answers(tmp_path: Path) -> None:
    """Only the control task leaks: its answer file is the one planted flaw."""

    for seed in [*DATA_SEEDS, 3, 7]:
        out = tmp_path / str(seed)
        family.materialize(family.kind_for_seed(seed), seed, out)
        names = {p.name for p in out.rglob("*")}
        assert "expected.json" not in names
        assert ".grader" not in names
    control = tmp_path / "control"
    family.materialize("sql", family.CONTROL_SEED, control, leak_answer=True)
    leaked = json.loads((control / ".grader" / "expected.json").read_text())
    assert leaked == family.control_instance().expected


@pytest.mark.skipif(shutil.which("sqlite3") is None, reason="needs the sqlite3 CLI")
def test_oracles_reproduce_the_expected_answers(tmp_path: Path, monkeypatch) -> None:
    for seed in DATA_SEEDS:
        instance = family.build(seed)
        work = tmp_path / str(seed)
        family.materialize(instance.kind, seed, work)
        script = instance.oracle.replace("/workdir", str(work))
        subprocess.run(["bash", "-c", script], check=True, capture_output=True)
        reward, passed, _ = _score(monkeypatch, work / "answer.txt", instance.expected)
        assert (reward, passed) == (1.0, True), (seed, instance.kind)


def _score(monkeypatch, answer: Path, expected: dict):
    monkeypatch.setattr(verify, "ANSWER", answer)
    return verify.check_answers(expected)


def test_answers_score_partial_credit(tmp_path: Path, monkeypatch) -> None:
    """Guards the answer-numbering fix: '267.32' is an answer, not '267.' + '32'."""

    expected = {
        "type": "answers",
        "answers": [
            {"type": "int", "answer": "20", "tolerance": 0.0},
            {"type": "money", "answer": "267.32", "tolerance": 0.005},
            {"type": "text", "answer": "10.0.3.17", "tolerance": None},
        ],
    }
    answer = tmp_path / "answer.txt"
    cases = {
        "20\n267.32\n10.0.3.17\n": (1.0, True),
        "1. 20\n2) 267.32\n3: 10.0.3.17": (1.0, True),
        "20\n$267.32\n": (2 / 3, False),
        "20\n267.30\n10.0.3.1\n": (1 / 3, False),
        "20\n267.32\n10.0.3.17\nextra guess\n": (0.0, False),
    }
    for text, (want_reward, want_passed) in cases.items():
        answer.write_text(text)
        reward, passed, _ = _score(monkeypatch, answer, expected)
        assert reward == pytest.approx(want_reward), text
        assert passed is want_passed, text
    answer.unlink()
    assert _score(monkeypatch, answer, expected)[:2] == (0.0, False)


def test_averages_accept_either_rounding_of_a_tie(tmp_path: Path, monkeypatch) -> None:
    """An exact 3.125 rounds to 3.12 in Python and 3.13 in SQLite; both are right."""

    expected = {
        "type": "answers",
        "answers": [{"type": "money", "answer": "3.125", "tolerance": 0.005}],
    }
    answer = tmp_path / "answer.txt"
    for text, want in (("3.12", 1.0), ("3.13", 1.0), ("3.14", 0.0)):
        answer.write_text(text)
        assert _score(monkeypatch, answer, expected)[0] == want


def test_bugfix_credit_counts_repaired_and_kept_cases(monkeypatch) -> None:
    instance = next(
        family.build(s) for s in range(3, 400, 4) if family.build(s).kind == "bugfix"
    )
    expected = instance.expected
    broken = {tuple(pair) for pair in expected["broken"]}
    assert broken, "the bug must break at least one hidden case"
    good = [check["outputs"] for check in expected["checks"]]

    def run_with(outputs):
        monkeypatch.setattr(verify, "_run_module", lambda _expected: (outputs, "ran"))
        return verify.check_bugfix(expected)

    assert run_with(good)[:2] == (1.0, True)
    # Doing nothing: every broken case still fails, the rest still pass.
    unfixed = [
        [
            {"__raised__": "Bug"} if (i, c) in broken else out
            for c, out in enumerate(check)
        ]
        for i, check in enumerate(good)
    ]
    assert run_with(unfixed)[:2] == (0.0, False)
    # A rewrite that repairs the bug but breaks every other case earns nothing.
    regressed = [
        [
            out if (i, c) in broken else {"__raised__": "Bug"}
            for c, out in enumerate(check)
        ]
        for i, check in enumerate(good)
    ]
    kept_cases = sum(len(check) for check in good) - len(broken)
    assert run_with(regressed)[0] == (0.0 if kept_cases else 1.0)
