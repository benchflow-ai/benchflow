"""The hill-climb demo (docs/examples/hillclimb) does what the post relies on.

The demo is imported by path, like the other docs examples. The agent under
test and the optimizer are scripted (tests/_hillclimb_example_fakes.py); the
demo's ``bf.Evaluation`` jobs, ``bf.load_job`` reads, workspace, mount
checks, statistics, record and report run for real. The split is fixed:
t0..t5 train, t6..t9 test.
"""

from __future__ import annotations

import ast
import asyncio
import json
import random
import sys
from pathlib import Path

import pytest

import benchflow as bf
from tests._hillclimb_example_fakes import (
    FakeAgent,
    FakeOptimizer,
    append_rule,
    instruction_of,
    make_tasks,
    read_tree,
)

DEMO = Path(__file__).resolve().parents[1] / "docs" / "examples" / "hillclimb"
sys.path.insert(0, str(DEMO))
import hillclimb  # noqa: E402
import hillclimb_stats  # noqa: E402
from hillclimb_proposer import ProposerSettings  # noqa: E402

TRAIN = [f"t{i}" for i in range(6)]
TEST = [f"t{i}" for i in range(6, 10)]


def settings(tmp_path: Path, **overrides) -> hillclimb.Settings:
    tasks = make_tasks(tmp_path / "tasks", TRAIN + TEST)
    skill = tmp_path / "skills" / "flood"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: flood\ndescription: Flood reports.\n---\nUse pandas.\n"
    )
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": TRAIN, "test": TEST}))
    values = dict(
        tasks_dir=tasks,
        skills=tmp_path / "skills",
        out=tmp_path / "run",
        split_file=split,
        sandbox="modal",
        trials=2,
        rounds=1,
        min_gain=0.1,
        retry_attempts=0,
        bootstrap_samples=200,
        proposer=ProposerSettings(sandbox="modal"),
    )
    return hillclimb.Settings(**{**values, **overrides})


def climb(s: hillclimb.Settings) -> dict:
    doc = asyncio.run(hillclimb.climb(s))
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((DEMO / "hillclimb.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(
        json.loads((s.out / "hillclimb.json").read_text()),
        schema,
        cls=jsonschema.Draft202012Validator,
    )
    return doc


def test_a_patch_that_lifts_train_but_not_test_is_reverted(tmp_path, monkeypatch):
    FakeAgent(
        lambda task, skills, trial: float("TRICK" in skills and task in TRAIN[:4])
    ).install(monkeypatch)
    FakeOptimizer(
        [append_rule("TRICK: check the station ids first")], root=tmp_path
    ).install(monkeypatch)
    doc = climb(settings(tmp_path))
    entry = doc["rounds"][0]
    assert entry["decision"] == "revert"
    assert (
        entry["train_delta"]["value"] == pytest.approx(4 / 6)
        and entry["test_delta"]["value"] == 0
    )
    assert any("overfitting" in r for r in entry["reasons"])
    assert (
        doc["best"]["version"] == "v000"
        and not doc["best"]["verdict"]["recommend_merge"]
    )


def test_a_patch_that_lifts_train_and_test_is_kept(tmp_path, monkeypatch):
    FakeAgent(
        lambda task, skills, trial: float("FIX" in skills and task != "t0")
    ).install(monkeypatch)
    FakeOptimizer(
        [
            append_rule("FIX: validate the output file before finishing"),
            append_rule("A remark that changes nothing."),
        ],
        root=tmp_path,
    ).install(monkeypatch)
    s = settings(tmp_path, rounds=2)
    doc = climb(s)
    first, second = doc["rounds"]
    assert first["decision"] == "keep" and first["version_after"] == "v001"
    assert (
        first["train_delta"]["value"] == pytest.approx(5 / 6)
        and first["test_delta"]["value"] == 1.0
    )
    assert second["base_version"] == "v001" and second["decision"] == "revert"
    assert (
        doc["best"]["version"] == "v001" and doc["best"]["verdict"]["recommend_merge"]
    )
    assert "+FIX: validate" in first["candidate"]["diff"]
    # Every evaluation is a normal BenchFlow job per split and trial.
    assert len(bf.load_job(s.out / "evals" / "r01" / "test").trials) == len(TEST) * 2
    assert "Gain exceeds noise" in (s.out / "report.html").read_text()


def test_the_noise_gate_refuses_when_min_gain_is_within_noise(tmp_path, monkeypatch):
    FakeAgent(
        lambda task, skills, trial: float(
            random.Random(f"{task}{trial}").random() < 0.5
        )
    ).install(monkeypatch)
    optimizer = FakeOptimizer([append_rule("never used")]).install(monkeypatch)
    doc = climb(settings(tmp_path, min_gain=0.05, rounds=3))
    gate = doc["noise_gate"]
    assert doc["status"] == "refused" and not gate["passed"] and not gate["forced"]
    assert gate["splits"]["train"]["noise_95"] > 0.05
    assert f"--trials {gate['suggestion']['trials']} (now 2)" in gate["message"]
    assert optimizer.runs == [] and doc["rounds"] == []


def test_the_gate_needs_two_trials_and_passes_quiet_evals():
    one = {"a": [1.0], "b": [0.0]}
    assert not hillclimb_stats.noise_gate(one, one, min_gain=0.5, trials=1)["passed"]
    steady = {f"t{i}": [1.0, 1.0, 1.0] if i % 2 else [0.0, 0.0, 0.0] for i in range(20)}
    assert hillclimb_stats.noise_gate(
        steady, steady, min_gain=0.05, trials=3, samples=200
    )["passed"]


def test_the_optimizer_never_sees_the_test_split(tmp_path, monkeypatch):
    """The sandbox walk: everything uploaded to the optimizer is searched for
    the test tasks' names, instructions and grader output, and the paths
    where they would be if they had been mounted are tried."""
    looked = []

    def probe(sandbox: Path) -> None:
        tree = read_tree(sandbox)
        looked.append(tree)
        for task in TEST:
            for rel, text in tree.items():
                assert task not in rel.split("/"), rel
                assert (
                    instruction_of(task) not in text
                    and f"GRADER-OUTPUT-{task}" not in text
                ), rel
        for where in (
            "hillclimb/test",
            "hillclimb/train/tasks/t6",
            "evals",
            "hillclimb.json",
        ):
            assert not (sandbox / where).exists()
        scores = json.loads((sandbox / "hillclimb" / "scores.json").read_text())
        assert set(scores["current"]["test"]) == {"score", "ci95", "tasks"}

    FakeAgent(lambda task, skills, trial: 0.0).install(monkeypatch)
    optimizer = FakeOptimizer(
        [append_rule("an idea")], probe=probe, root=tmp_path
    ).install(monkeypatch)
    s = settings(tmp_path)
    doc = climb(s)
    assert len(looked) == 1
    assert all(f"hillclimb/train/tasks/{t}/instruction.md" in looked[0] for t in TRAIN)
    assert set(optimizer.runs[0]["uploads"].values()) == {"/hillclimb", "/app/surface"}
    seen = doc["rounds"][0]["candidate"]["mounted"]
    assert [(m["sandbox_path"], m["read_only"]) for m in seen["mounts"]] == [
        ("/hillclimb", True),
        ("/app/surface", False),
    ]
    assert (
        seen["network"] == "none"
        and seen["train_tasks"] == TRAIN
        and seen["test_tasks"] == len(TEST)
    )
    assert (
        seen["test_tasks_in_paths"] == [] and seen["test_instructions_in_files"] == []
    )
    assert len(json.loads(Path(seen["manifest"]).read_text())["files"]) == sum(
        m["files"] for m in seen["mounts"]
    )
    assert (
        "allow_internet: false"
        in (s.out / "proposer" / "r01" / "task" / "task.md").read_text()
    )
    assert "Test split never mounted: 0 of 4" in (s.out / "report.html").read_text()


def test_a_stall_ends_with_a_root_cause_analysis(tmp_path, monkeypatch):
    FakeAgent(lambda task, skills, trial: 0.0).install(monkeypatch)
    analysis = {
        "summary": "Mostly capability gaps.",
        "recommendations": ["fix t1's grader"],
        "failures": [
            {"id": "t0/trial-01", "category": "capability_gap", "explanation": "units"},
            {
                "id": "t1/trial-01",
                "category": "grader_bug",
                "explanation": "rejects csv",
            },
            {"id": "t2/trial-01", "category": "bogus", "explanation": "dropped"},
        ],
    }
    optimizer = FakeOptimizer(
        [append_rule(f"idea {i}") for i in range(5)], analysis=analysis, root=tmp_path
    ).install(monkeypatch)
    doc = climb(settings(tmp_path, rounds=5, stall_rounds=2))
    assert doc["stop"]["reason"] == "stalled" and len(doc["rounds"]) == 2
    assert [r["mode"] for r in optimizer.runs] == ["propose", "propose", "analyze"]
    assert doc["analysis"]["counts"] == {
        "ambiguous_task": 0,
        "grader_bug": 1,
        "infrastructure": 0,
        "capability_gap": 1,
    }


def test_graders_and_infrastructure_errors(tmp_path, monkeypatch):
    """Grader checks drop broken tasks; trials without a score are left out of
    every score, counted, and stop the climb when there are too many."""
    FakeAgent(
        lambda task, skills, trial: None if trial == 2 else 1.0,
        oracle=lambda task: 0.0 if task == "t1" else 1.0,
        nop=lambda task: 1.0 if task == "t7" else 0.0,
    ).install(monkeypatch)
    FakeOptimizer().install(monkeypatch)
    doc = climb(settings(tmp_path))
    assert doc["controls"]["excluded"] == ["t1", "t7"]
    assert "t1" not in doc["split"]["train"] and "t7" not in doc["split"]["test"]
    train = doc["baseline"]["train"]
    assert train["infra_errors"] == 5 and train["score"]["value"] == 1.0
    assert doc["status"] == "stopped" and doc["stop"]["reason"] == "infra"


def test_the_demo_uses_only_public_benchflow_names():
    for path in DEMO.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "benchflow"
            ):
                pytest.fail(
                    f"{path.name} imports {node.module}; use `import benchflow as bf`"
                )
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "bf"
            ):
                assert node.attr in bf.__all__, f"{path.name} uses bf.{node.attr}"
