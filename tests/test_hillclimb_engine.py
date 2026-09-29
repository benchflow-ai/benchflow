"""``bench hillclimb`` end to end on scripted sandboxes.

The agent under test and the optimizer are scripted (tests/_hillclimb_fakes.py):
the real ``Evaluation`` jobs, ``bf.load_job``, the proposer's workspace,
wrapper task and output collection, the statistics, ``hillclimb.json``, the
surface history and the report all run for real. Only the two calls that
would start a sandbox are replaced.

The split is fixed (t0..t5 train, t6..t9 test) so each scenario controls
which tasks a patch helps.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from benchflow.hillclimbing import HillclimbConfig, ProposerSettings, hillclimb
from benchflow.hillclimbing.engine import decide
from tests._hillclimb_fakes import (
    FakeAgent,
    FakeProposer,
    append_to_skill,
    instruction_of,
    make_tasks,
    read_tree,
)

REPO = Path(__file__).resolve().parents[1]
SCHEMA = REPO / "docs/reference/schemas/benchflow-hillclimb.v1.schema.json"
TRAIN = ["t0", "t1", "t2", "t3", "t4", "t5"]
TEST = ["t6", "t7", "t8", "t9"]


def _setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    tasks = make_tasks(tmp_path / "tasks", TRAIN + TEST)
    skill = tmp_path / "skills" / "flood-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: flood-skill\ndescription: Flood report workflow.\n---\n"
        "Read the gauge CSVs with pandas.\n"
    )
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": TRAIN, "test": TEST}))
    return tasks, tmp_path / "skills", split


def _config(tmp_path: Path, **overrides) -> HillclimbConfig:
    tasks, skills, split = _setup(tmp_path)
    values = {
        "tasks": tasks,
        "surface": skills,
        "out": tmp_path / "run",
        "split_file": split,
        "environment": "modal",
        "model": "claude-haiku-4-5",
        "trials": 2,
        "rounds": 1,
        "min_gain": 0.1,
        "retry_attempts": 0,
        "bootstrap_samples": 200,
        "proposer": ProposerSettings(environment="modal"),
    }
    values.update(overrides)
    return HillclimbConfig(**values)


def assert_valid(run_dir: Path) -> dict:
    jsonschema = pytest.importorskip("jsonschema")
    doc = json.loads((run_dir / "hillclimb.json").read_text())
    schema = json.loads(SCHEMA.read_text())
    jsonschema.validate(doc, schema, cls=jsonschema.Draft202012Validator)
    return doc


def _git_subjects(repo: Path) -> list[str]:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    out = subprocess.run(
        ["git", "-C", str(repo), "log", "--format=%s"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return out.splitlines()[::-1]


def test_a_patch_that_lifts_train_but_not_test_is_reverted(tmp_path, monkeypatch):
    def reward(task, surface, trial):
        return 1.0 if "TRAIN-TRICK" in surface and task in TRAIN[:4] else 0.0

    FakeAgent(reward).install(monkeypatch)
    FakeProposer(
        [append_to_skill("TRAIN-TRICK: check the station ids first")],
        root=tmp_path / "sandboxes",
    ).install(monkeypatch)
    result = hillclimb(_config(tmp_path))

    doc = result.record
    assert doc.status == "finished" and doc.stop.reason == "rounds"
    cand = doc.rounds[0].candidates[0]
    assert cand.decision == "revert"
    assert cand.train_delta.value == pytest.approx(4 / 6)
    assert cand.test_delta.value == 0
    assert any("overfitting" in r for r in cand.reasons)
    assert doc.rounds[0].kept is None
    assert doc.best.version == "v000" and doc.best.candidate is None
    assert doc.best.verdict.recommend_merge is False
    assert _git_subjects(result.run_dir / "surface-history") == ["baseline surface"]
    assert_valid(result.run_dir)


def test_a_patch_that_lifts_train_and_test_is_kept(tmp_path, monkeypatch):
    def reward(task, surface, trial):
        return 1.0 if "GENERAL-FIX" in surface and task != "t0" else 0.0

    FakeAgent(reward).install(monkeypatch)
    FakeProposer(
        [
            append_to_skill("GENERAL-FIX: validate the output file before finishing"),
            append_to_skill("A remark that changes nothing."),
        ],
        root=tmp_path / "sandboxes",
    ).install(monkeypatch)
    result = hillclimb(_config(tmp_path, rounds=2))

    doc = result.record
    first, second = (rd.candidates[0] for rd in doc.rounds)
    assert first.decision == "keep"
    assert first.train_delta.value == pytest.approx(5 / 6)
    assert first.test_delta.value == pytest.approx(1.0)
    assert doc.rounds[0].kept == "r01-c1" and doc.rounds[0].version_after == "v001"
    # Round 2 starts from the kept version and gains nothing on train.
    assert second.base_version == "v001"
    assert second.decision == "revert"
    assert any("below --min-gain" in r for r in second.reasons)
    best = doc.best
    assert best.version == "v001" and best.candidate == "r01-c1" and best.round == 1
    assert best.test_delta_vs_baseline.value == pytest.approx(1.0)
    assert best.verdict.exceeds_noise and best.verdict.recommend_merge
    assert (
        "GENERAL-FIX"
        in (result.run_dir / "surfaces/v001/skills/flood-skill/SKILL.md").read_text()
    )
    assert "GENERAL-FIX" in first.diff and first.diff_stats.added == 2
    history = result.run_dir / "surface-history"
    assert _git_subjects(history)[0] == "baseline surface"
    assert _git_subjects(history)[1].startswith("r01-c1:")
    assert "GENERAL-FIX" in (history / "skills/flood-skill/SKILL.md").read_text()
    assert best.git_commit is not None
    report = (result.run_dir / "report.html").read_text()
    assert "Gain exceeds noise" in report and "r01-c1" in report
    # Every evaluation is a normal BenchFlow job folder per trial.
    for split, names in (("train", TRAIN), ("test", TEST)):
        for k in (1, 2):
            folder = (
                result.run_dir / "evals" / "r01-c1" / split / f"trial-{k:02d}" / "job"
            )
            assert (folder / "summary.json").is_file()
            ran = sorted(
                r.parent.name.split("__")[0] for r in folder.glob("*/result.json")
            )
            assert ran == names
    assert_valid(result.run_dir)


def test_the_noise_gate_refuses_when_min_gain_is_within_noise(tmp_path, monkeypatch):
    def reward(task, surface, trial):
        return float(random.Random(f"{task}:{trial}").random() < 0.5)

    FakeAgent(reward).install(monkeypatch)
    proposer = FakeProposer([append_to_skill("never used")]).install(monkeypatch)
    result = hillclimb(_config(tmp_path, min_gain=0.05, rounds=3))

    doc = result.record
    assert doc.status == "refused" and doc.stop.reason == "noise_gate"
    gate = doc.noise_gate
    assert gate.passed is False and gate.forced is False
    assert gate.train.noise_95 > 0.05
    assert "--trials" in gate.message and "--min-gain" in gate.message
    assert gate.suggestion.trials > 2
    assert gate.suggestion.min_gain >= gate.train.noise_95 - 1e-3
    assert proposer.runs == [] and doc.rounds == []
    assert doc.best.version == "v000"
    assert "Refused by the noise gate" in (result.run_dir / "report.html").read_text()
    assert_valid(result.run_dir)


def test_a_proposer_that_looks_for_the_test_split_finds_none_of_it(
    tmp_path, monkeypatch
):
    """The structural guarantee: nothing of the test split is in the sandbox.

    The scripted optimizer searches everything uploaded to it for the test
    tasks' names, instructions and verifier output, and tries the paths where
    they would be if they had been mounted.
    """

    def reward(task, surface, trial):
        return 0.0

    def verifier_text(task):
        return f"TEST-ONLY-OUTPUT-{task}\n" if task in TEST else f"FAILED {task}\n"

    looked: list[dict[str, str]] = []

    def probe(sandbox: Path) -> None:
        tree = read_tree(sandbox)
        looked.append(tree)
        for task in TEST:
            assert not (sandbox / "hillclimb" / "train" / "tasks" / task).exists()
            assert not (sandbox / "hillclimb" / "train" / "failures" / task).exists()
            for rel, text in tree.items():
                assert task not in rel.split("/"), rel
                assert instruction_of(task) not in text, rel
                assert f"TEST-ONLY-OUTPUT-{task}" not in text, rel
        for where in ("hillclimb/test", "hillclimb/evals", "evals", "hillclimb.json"):
            assert not (sandbox / where).exists()
        assert not any(rel.endswith("hillclimb.json") for rel in tree)
        scores = json.loads((sandbox / "hillclimb" / "scores.json").read_text())
        assert set(scores["current"]["test"]) == {"score", "ci95", "tasks"}
        assert set(scores["baseline"]["test"]) == {"score", "ci95", "tasks"}

    FakeAgent(reward, verifier_text=verifier_text).install(monkeypatch)
    proposer = FakeProposer(
        [append_to_skill("an idea")], probe=probe, root=tmp_path / "sandboxes"
    ).install(monkeypatch)
    result = hillclimb(_config(tmp_path))

    assert len(looked) == 1
    tree = looked[0]
    # It did get the train split: failures, instructions, verifier output.
    for task in TRAIN:
        assert f"hillclimb/train/tasks/{task}/instruction.md" in tree
        assert tree[
            f"hillclimb/train/failures/{task}/trial-01/verifier/test-stdout.txt"
        ]
    assert "app/surface/skills/flood-skill/SKILL.md" in tree
    # Exactly two uploads, both from this round's workspace.
    uploads = proposer.runs[0]["uploads"]
    workspace = result.run_dir / "proposer" / "r01-c1" / "workspace"
    assert uploads == {
        str(workspace / "evidence"): "/hillclimb",
        str(workspace / "surface"): "/app/surface",
    }
    task_md = (
        result.run_dir / "proposer/r01-c1/task/hillclimb-propose/task.md"
    ).read_text()
    assert "allow_internet: false" in task_md
    # hillclimb.json records what was mounted, checked against the test split.
    seen = result.record.rounds[0].candidates[0].proposer.exposure
    assert [(m.sandbox_path, m.read_only) for m in seen.mounts] == [
        ("/hillclimb", True),
        ("/app/surface", False),
    ]
    assert seen.network == "none" and seen.test_tasks == len(TEST)
    assert seen.test_tasks_in_paths == [] and seen.test_instructions_in_files == []
    assert seen.train_tasks == TRAIN
    assert seen.failures == [f"{t}/trial-01" for t in TRAIN] + [
        f"{t}/trial-02" for t in TRAIN
    ]
    # The audit copy stays, and the run folder can still be deleted.
    evidence = result.run_dir / "proposer" / "r01-c1" / "workspace" / "evidence"
    assert (evidence / "BRIEF.md").stat().st_mode & 0o200
    manifest = json.loads((result.run_dir / seen.manifest).read_text())
    assert sum(m.files for m in seen.mounts) == len(manifest["files"])
    assert {f["mount"] for f in manifest["files"]} == {"/hillclimb", "/app/surface"}
    report = (result.run_dir / "report.html").read_text()
    assert (
        "Test split never mounted: 0 of 4 test tasks in 1 optimizer sandbox" in report
    )
    assert "What the optimizer saw" in report
    assert_valid(result.run_dir)


def test_the_exposure_check_finds_test_material_in_an_upload(tmp_path):
    from benchflow.hillclimbing.proposer import Workspace, exposure

    evidence = tmp_path / "evidence"
    (evidence / "train" / "tasks" / "t6").mkdir(parents=True)
    (evidence / "notes.txt").write_text("copied: " + instruction_of("t7"))
    surface = tmp_path / "surface"
    surface.mkdir()
    ws = Workspace(
        root=tmp_path,
        evidence=evidence,
        surface=surface,
        uploads={str(evidence): "/hillclimb", str(surface): "/app/surface"},
        failures=[],
        infra=[],
    )
    seen = exposure(
        ws,
        test_instructions={t: instruction_of(t) for t in TEST},
        open_network=False,
        manifest_path=tmp_path / "mounted.json",
    )
    assert seen["test_tasks_in_paths"] == ["t6"]
    assert seen["test_instructions_in_files"] == ["t7"]


def test_a_stall_ends_the_climb_with_a_root_cause_analysis(tmp_path, monkeypatch):
    FakeAgent(lambda task, surface, trial: 0.0).install(monkeypatch)
    analysis = {
        "summary": "Most failures are capability gaps; one grader looks wrong.",
        "failures": [
            {
                "id": "t0/trial-01",
                "category": "capability_gap",
                "explanation": "wrong units",
            },
            {
                "id": "t1/trial-01",
                "category": "grader_bug",
                "explanation": "rejects a valid csv",
            },
            {"id": "t2/trial-01", "category": "bogus", "explanation": "not a category"},
        ],
        "recommendations": ["Fix t1's grader."],
    }
    proposer = FakeProposer(
        [append_to_skill(f"idea {i}") for i in range(5)],
        analysis=analysis,
        root=tmp_path / "sandboxes",
    ).install(monkeypatch)
    result = hillclimb(_config(tmp_path, rounds=5, stall_rounds=2))

    doc = result.record
    assert doc.status == "finished" and doc.stop.reason == "stalled"
    assert len(doc.rounds) == 2
    assert [r["mode"] for r in proposer.runs] == ["propose", "propose", "analyze"]
    a = doc.analysis
    assert a is not None and a.status == "ok" and a.trigger == "stall"
    assert a.counts == {
        "ambiguous_task": 0,
        "grader_bug": 1,
        "infrastructure": 0,
        "capability_gap": 1,
    }
    assert [f.id for f in a.failures] == ["t0/trial-01", "t1/trial-01"]
    assert a.recommendations == ["Fix t1's grader."]
    assert any("malformed" in w for w in doc.warnings)
    report = (result.run_dir / "report.html").read_text()
    assert "Stall analysis" in report and "rejects a valid csv" in report
    # The analysis sandbox got the train split only, like the proposer's.
    assert not any(
        t in rel.split("/") for rel in read_tree(proposer.sandboxes[-1]) for t in TEST
    )
    assert_valid(result.run_dir)


def test_infrastructure_errors_are_excluded_and_stop_the_climb(tmp_path, monkeypatch):
    def reward(task, surface, trial):
        return None if task in ("t0", "t1", "t6") or trial == 2 else 1.0

    FakeAgent(reward).install(monkeypatch)
    proposer = FakeProposer([append_to_skill("x")]).install(monkeypatch)
    result = hillclimb(_config(tmp_path))

    doc = result.record
    assert doc.status == "stopped" and doc.stop.reason == "infra"
    assert "ended without a score" in doc.stop.detail
    train = doc.baseline.train
    assert train.infra_errors == 2 + 2 + 4 and train.infra_error_categories == {
        "sandbox_setup": 8
    }
    # Errored trials are left out of the score, not counted as zero.
    assert train.score.value == pytest.approx(1.0) and train.score.tasks == 4
    assert proposer.runs == []
    assert_valid(result.run_dir)


def test_a_candidate_that_makes_trials_crash_is_reverted(tmp_path, monkeypatch):
    def reward(task, surface, trial):
        if "CRASHY" in surface:
            return None if task in ("t0", "t1", "t2") else 1.0
        return 0.0

    FakeAgent(reward).install(monkeypatch)
    FakeProposer([append_to_skill("CRASHY: give up on hard stations")]).install(
        monkeypatch
    )
    result = hillclimb(_config(tmp_path, max_infra_error_rate=0.5))

    cand = result.record.rounds[0].candidates[0]
    # Scored on the tasks it did not crash, it looks like a big win...
    assert cand.train_delta.value == pytest.approx(1.0)
    # ...but its crashes rose, so it is reverted.
    assert cand.decision == "revert"
    assert any("infrastructure errors rose from 0 to 6" in r for r in cand.reasons)
    assert_valid(result.run_dir)


def test_grader_checks_flag_and_can_exclude_broken_tasks(tmp_path, monkeypatch):
    agent = FakeAgent(
        lambda task, surface, trial: 0.0,
        oracle=lambda task: 0.0 if task == "t1" else 1.0,
        nop=lambda task: 1.0 if task == "t7" else 0.0,
    ).install(monkeypatch)
    FakeProposer().install(monkeypatch)
    result = hillclimb(_config(tmp_path, rounds=0, exclude_broken_tasks=True))

    doc = result.record
    controls = doc.controls
    assert controls.ran and controls.grader_bugs == ["t1", "t7"]
    assert controls.excluded == ["t1", "t7"]
    flags = {t.task: t.flags for t in controls.tasks}
    assert flags["t1"] == ["oracle_fails"] and flags["t7"] == ["nop_passes"]
    assert "t1" not in doc.split.train and "t7" not in doc.split.test
    ran = {c["task"] for c in agent.calls if c["agent"] == "claude-agent-acp"}
    assert ran == set(TRAIN + TEST) - {"t1", "t7"}
    assert any("grader check: t1" in w for w in doc.warnings)
    assert_valid(result.run_dir)


def test_the_cost_objective_keeps_a_cheaper_surface_that_holds_the_score(
    tmp_path, monkeypatch
):
    FakeAgent(
        lambda task, surface, trial: 1.0,
        cost=lambda task, surface, trial: 0.01 if "CHEAPER" in surface else 0.02,
    ).install(monkeypatch)
    FakeProposer(
        [append_to_skill("CHEAPER: stop exploring once the answer file validates")]
    ).install(monkeypatch)
    result = hillclimb(_config(tmp_path, objective="cost", min_gain=0.2))

    doc = result.record
    cand = doc.rounds[0].candidates[0]
    assert cand.decision == "keep", cand.reasons
    assert cand.cost_change.train == pytest.approx(-0.5)
    assert doc.best.cost_change_vs_baseline.test == pytest.approx(-0.5)
    assert doc.best.verdict.recommend_merge
    assert "fell 50.0%" in doc.best.verdict.text
    assert_valid(result.run_dir)


def test_a_patch_that_pastes_a_train_instruction_is_rejected(tmp_path, monkeypatch):
    agent = FakeAgent(lambda task, surface, trial: 0.0).install(monkeypatch)

    def paste(surface: Path) -> dict:
        skill = surface / "skills" / "flood-skill" / "SKILL.md"
        skill.write_text(
            skill.read_text() + "\nFor example: " + instruction_of("t2") + "\n"
        )
        return {"root_cause": "a", "change": "b", "rationale": "c"}

    FakeProposer([paste]).install(monkeypatch)
    result = hillclimb(_config(tmp_path))

    cand = result.record.rounds[0].candidates[0]
    assert cand.decision == "invalid"
    assert cand.leak_check.status == "flagged"
    assert cand.leak_check.matches[0]["source"] == "train task instruction: t2"
    assert cand.evaluation is None
    assert not (result.run_dir / "evals" / "r01-c1").exists()
    assert all("r01-c1" not in c["jobs_dir"] for c in agent.calls)
    assert_valid(result.run_dir)


def test_the_budget_stops_the_climb_before_a_round_it_cannot_afford(
    tmp_path, monkeypatch
):
    FakeAgent(lambda task, surface, trial: 0.0).install(monkeypatch)
    proposer = FakeProposer([append_to_skill("x")]).install(monkeypatch)
    result = hillclimb(_config(tmp_path, max_cost_usd=0.25))

    doc = result.record
    # The baseline spent 10 tasks x 2 trials x $0.01; a round would cost as much.
    assert doc.cost.agent_usd == pytest.approx(0.2)
    assert doc.status == "stopped" and doc.stop.reason == "budget"
    assert doc.cost.budget_stopped and proposer.runs == []
    assert_valid(result.run_dir)


def test_a_candidate_the_budget_cannot_cover_is_skipped_not_judged(
    tmp_path, monkeypatch
):
    FakeAgent(lambda task, surface, trial: 0.0).install(monkeypatch)
    FakeProposer([append_to_skill("x")], cost_usd=10.0).install(monkeypatch)
    result = hillclimb(_config(tmp_path, max_cost_usd=0.5))

    doc = result.record
    cand = doc.rounds[0].candidates[0]
    assert cand.decision == "skipped" and cand.evaluation is None
    assert "--max-cost-usd" in cand.reasons[0]
    assert doc.status == "stopped" and doc.stop.reason == "budget"
    assert doc.cost.proposer_usd == pytest.approx(10.0)
    assert_valid(result.run_dir)


def test_a_prompt_surface_reaches_the_agent_through_the_config_overlay(
    tmp_path, monkeypatch
):
    prompt = tmp_path / "prompt.md"
    prompt.write_text("Work carefully.\n")

    def reward(task, surface, trial):
        return 1.0 if "CHECK-UNITS" in surface else 0.0

    agent = FakeAgent(reward).install(monkeypatch)

    def edit(surface: Path) -> dict:
        path = surface / "prompt.md"
        path.write_text(path.read_text() + "CHECK-UNITS before writing the answer.\n")
        return {"root_cause": "unit errors", "change": "check units", "rationale": "r"}

    FakeProposer([edit]).install(monkeypatch)
    result = hillclimb(_config(tmp_path, surface=[prompt]))

    doc = result.record
    assert doc.config.surfaces[0].kind == "prompt"
    assert doc.rounds[0].candidates[0].decision == "keep"
    assert any("Work carefully." in c["surface"] for c in agent.calls)
    assert (
        (result.run_dir / "surfaces/v001/prompt.md")
        .read_text()
        .endswith("CHECK-UNITS before writing the answer.\n")
    )
    assert_valid(result.run_dir)


def test_a_run_folder_is_never_reused(tmp_path, monkeypatch):
    FakeAgent(lambda task, surface, trial: 0.0).install(monkeypatch)
    FakeProposer().install(monkeypatch)
    hillclimb(_config(tmp_path, rounds=0))
    with pytest.raises(ValueError, match="already holds a hillclimb run"):
        hillclimb(_config(tmp_path / "again", rounds=0, out=tmp_path / "run"))


@pytest.mark.parametrize(
    ("train", "test", "keep", "reason"),
    [
        (0.2, 0.1, True, "train +0.200"),
        (0.2, 0.0, False, "overfitting"),
        (0.2, -0.1, False, "test regressed"),
        (0.05, 0.3, False, "below --min-gain"),
        (-0.1, 0.3, False, "train regressed"),
        (0.0, 0.0, False, "test is flat"),
    ],
)
def test_the_keep_rule(train, test, keep, reason):
    d = decide(objective="score", min_gain=0.1, train_delta=train, test_delta=test)
    assert d.keep is keep
    assert any(reason in r for r in d.reasons), d.reasons


def test_the_keep_rule_for_cost_holds_the_score_within_noise():
    ok = decide(
        objective="cost",
        min_gain=0.1,
        train_delta=-0.02,
        test_delta=-0.01,
        train_cost_change=-0.3,
        test_cost_change=-0.2,
        train_noise=0.05,
        test_noise=0.05,
    )
    assert ok.keep
    worse = decide(
        objective="cost",
        min_gain=0.1,
        train_delta=-0.2,
        test_delta=0.0,
        train_cost_change=-0.3,
        test_cost_change=-0.2,
        train_noise=0.05,
        test_noise=0.05,
    )
    assert not worse.keep and "beyond the noise band" in worse.reasons[0]
