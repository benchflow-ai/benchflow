"""The differential-test runner's pieces that run no engine.

Round PY8, target DT1 (tools/dt/run.py): the tree hash it checks fetched
trees against, the argv adapters, the ``env -i`` allowlist, the refused
caller variables, the interleaved schedule, and the corpus.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.dt import run

DT = Path(__file__).parents[1] / "tools" / "dt"


def test_tree_hash_matches_gits_own_hashes(tmp_path):
    # git's empty tree, and a tree with one file "hello\n" named a.txt
    # (`git write-tree` gives these, checked on the build VM; they are fixed by
    # git's object format).
    empty = tmp_path / "empty"
    empty.mkdir()
    assert run.tree_hash(empty) == "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
    one = tmp_path / "one"
    one.mkdir()
    (one / "a.txt").write_text("hello\n")
    (one / "a.txt").chmod(0o644)
    assert run.tree_hash(one) == "2e81171448eb9f2ee3821e3d447aa6b2fe3ddba1"


def test_tree_hash_sees_the_executable_bit_and_folder_order(tmp_path):
    root = tmp_path / "t"
    (root / "a").mkdir(parents=True)
    (root / "a" / "x").write_text("x")
    (root / "a.txt").write_text("y")
    first = run.tree_hash(root)
    (root / "a.txt").chmod(0o755)
    assert run.tree_hash(root) != first


def test_the_python_argv_runs_the_copy_from_its_own_parent(tmp_path):
    copy = tmp_path / "cell" / "tasks" / "hello"
    argv = run.cell_argv(
        "python", Path("/opt/arm/bench"), copy, "oracle", tmp_path / "jobs"
    )
    assert argv == [
        "/opt/arm/bench",
        "eval",
        "run",
        "--tasks-dir",
        str(copy.parent),
        "--agent",
        "oracle",
        "--sandbox",
        "docker",
        "--concurrency",
        "1",
        "--jobs-dir",
        str(tmp_path / "jobs"),
    ]


def test_the_rust_argv_is_fresh_with_no_task_skills(tmp_path):
    copy = tmp_path / "hello"
    argv = run.cell_argv("rust", Path("/opt/arm/bench"), copy, "nop", tmp_path / "jobs")
    assert argv == [
        "/opt/arm/bench",
        "run",
        str(copy),
        "-a",
        "nop",
        "-s",
        "docker",
        "-j",
        "1",
        "--jobs-dir",
        str(tmp_path / "jobs"),
        "--fresh",
        "--no-task-skills",
    ]


def test_cells_get_only_the_allowlisted_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    argv = run.cell_env(tmp_path)
    assert argv[:2] == ["env", "-i"]
    assert sorted(a.split("=", 1)[0] for a in argv[2:]) == ["HOME", "LANG", "PATH"]
    monkeypatch.setenv("DOCKER_HOST", "unix:///tmp/dt-docker.sock")
    assert "DOCKER_HOST=unix:///tmp/dt-docker.sock" in run.cell_env(tmp_path)


@pytest.mark.parametrize("name", run.REFUSED_ENV)
def test_a_lenient_caller_environment_is_refused(monkeypatch, name):
    monkeypatch.setenv(name, "1")
    with pytest.raises(SystemExit, match=name):
        run.check_caller_env()


def test_the_schedule_rotates_arms_every_round_and_starts_slow_tasks_first():
    tasks = [{"name": "a"}, {"name": "slow", "weight": 3}]
    rounds = list(run.schedule(["x", "y"], tasks, ["oracle"], 2))
    (r1, groups1), (r2, groups2) = rounds
    assert (r1, r2) == (1, 2)
    assert [g[0][1]["name"] for g in groups1] == ["slow", "a"]
    assert [arm for arm, *_ in groups1[0]] == ["x", "y"]
    assert [arm for arm, *_ in groups2[0]] == ["y", "x"]


def test_the_corpus_pins_every_tree_and_names_each_once():
    corpus = run._toml(DT / "corpus.toml")
    tasks = run.corpus_tasks(corpus, "smoke")
    assert 35 <= len(tasks) <= 45
    for task in tasks:
        assert len(task["tree"]) == 40 and int(task["tree"], 16) >= 0
        assert task["commit"] == corpus["source"][task["source"]]["commit"]
    names = {t["name"] for t in tasks}
    for required in (
        "hello-split",
        "sidecar",
        "snapshot-hook",
        "acp_smoke",
        "net-blocklist",
        "net-ip-blocklist",
        "reward-fraction",
        "reward-json",
        "reward-out-of-range",
        "reward-negative",
    ):
        assert required in names, required


def test_a_corpus_id_must_name_source_path_and_tree():
    with pytest.raises(ValueError):
        run.parse_task_id("examples/tasks/hello")
    assert run.parse_task_id("p1/examples/tasks/hello@abc") == (
        "p1",
        "examples/tasks/hello",
        "abc",
    )


def test_every_arm_names_a_kind_and_a_commit_and_its_patch_exists():
    arms = run._toml(DT / "arms.toml")["arm"]
    assert {"base", "base-b", "rust-0.8.2"} <= set(arms)
    for name, arm in arms.items():
        assert arm["kind"] in ("python", "rust"), name
        assert len(arm["commit"]) == 40, name
        if arm.get("patch"):
            assert (DT / arm["patch"]).is_file(), name


def test_run_py_imports_nothing_from_benchflow():
    import ast

    tree = ast.parse((DT / "run.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("benchflow"), node.module
        if isinstance(node, ast.Import):
            assert all(not a.name.startswith("benchflow") for a in node.names)
    assert os.path.basename(run.__file__) == "run.py"


def _cell(work: Path, name: str, agent: str, log: str = "") -> Path:
    cell = run.cell_folder(work, "base", 1, name, agent)
    (cell / "jobs").mkdir(parents=True)
    (cell / "run.log").write_text(log)
    return cell


def test_read_cell_takes_the_last_attempt_and_counts_attempts(tmp_path):
    import json

    from tests.test_dt_normalise import materialise

    cell = _cell(tmp_path, "hello-split", "oracle")
    job = materialise("py-0.8__hello-split", cell / "jobs")
    (trial,) = [p for p in job.iterdir() if p.is_dir()]
    retry = job / "hello-split__00000000"
    retry.mkdir()
    data = json.loads((trial / "result.json").read_text())
    data["started_at"] = "2026-10-16 03:00:00"  # earlier than the fixture's
    data["rewards"] = None
    data["error"] = "RuntimeError: Command timed out after 10 seconds"
    data["error_category"] = "infra_failure"
    (retry / "result.json").write_text(json.dumps(data))
    row = {"cell": {"name": "hello-split", "agent": "oracle"}, "dir": str(cell)}

    final = run.read_cell(row, kind="python")

    assert final == trial
    assert row["record"]["outcome"] == "passed" and row["record"]["attempts"] == 2
    assert row["error"] is None


def test_read_cell_records_a_planner_refusal_or_an_error(tmp_path):
    log = (
        "WARNING Skipping task 'stages': stages: BenchFlow cannot honor these task.md "
        "draft 2 fields yet, so it refuses the task\n  - [stage]: stages are not run\n"
    )
    row = {
        "cell": {"name": "stages", "agent": "nop"},
        "dir": str(_cell(tmp_path, "stages", "nop", log)),
    }
    assert run.read_cell(row, kind="python") is None
    assert row["record"]["outcome"] == "refused_at_load"
    row = {
        "cell": {"name": "hello", "agent": "nop"},
        "dir": str(_cell(tmp_path, "hello", "nop", "boom\n")),
    }
    run.read_cell(row, kind="python")
    assert row["record"] is None and row["error"].startswith("no trial folder")
    # A Rust cell never reads a planner log line as a refusal.
    row = {
        "cell": {"name": "stages", "agent": "nop"},
        "dir": str(tmp_path / "base" / "r1" / "stages__nop"),
    }
    run.read_cell(row, kind="rust")
    assert row["record"] is None and row["error"]
