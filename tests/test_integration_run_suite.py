from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.integration.check_adapter_evidence import (
    check_continuallearningbench,
    check_hilbench,
    check_programbench,
    check_skillsbench_result,
)
from tests.integration.check_adapter_evidence import (
    main as adapter_evidence_main,
)
from tests.integration.run_suite import (
    expand_lane,
    load_suite,
    main,
    run_adapter_evidence,
    select_lanes,
)

SUITE_PATH = Path("tests/integration/suites/release.yaml")


def test_release_suite_loads_and_has_only_release_blocker_lanes() -> None:
    suite = load_suite(SUITE_PATH)

    assert suite["suite"] == "release"
    assert suite["lanes"]
    assert suite["run_tracking"]["future_system"] == "Linear"
    assert "near-term" in suite["execution_profiles"]
    assert all(lane["release_blocker"] for lane in suite["lanes"])


def test_select_lanes_rejects_unknown_lane() -> None:
    suite = load_suite(SUITE_PATH)

    with pytest.raises(ValueError, match="unknown lane"):
        select_lanes(suite, ["not-a-lane"])


def test_expand_shared_sandbox_smoke_resolves_axis_references() -> None:
    suite = load_suite(SUITE_PATH)
    lane = select_lanes(suite, ["shared-sandbox-smoke"])[0]

    expanded = expand_lane(suite, lane)

    assert expanded["matrix"]["agents"] == ["gemini"]
    assert expanded["matrix"]["sandboxes"] == [
        "docker",
        "daytona",
        "modal",
        "firecracker",
        "k8s",
    ]
    assert expanded["todos"] == [
        "Select one boring representative task that runs everywhere."
    ]


def test_dry_run_prints_selected_lane(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(
        ["--suite", str(SUITE_PATH), "--lane", "security-dind-smoke", "--dry-run"]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "security-dind-smoke" in out
    assert "firecracker, k8s" in out


def test_near_term_profile_prints_small_daytona_plan(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["--suite", str(SUITE_PATH), "--profile", "near-term", "--dry-run"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Profile: near-term" in out
    assert "Future tracker: Linear" in out
    assert "Benchmark suites: SkillsBench" in out
    assert "Preferred sandboxes: daytona" in out
    assert "benchflow-ai/skillsbench/tasks@main (9 tasks)" in out
    assert "adapter-release-set" in out
    assert "Task budget:" in out
    assert "per_adapter: 1" in out
    assert "terminal-bench-smoke" not in out
    assert "shared-sandbox-smoke" not in out


def test_requires_dry_run_until_execution_exists(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        main(["--suite", str(SUITE_PATH), "--lane", "shared-sandbox-smoke"])

    err = capsys.readouterr().err
    assert "--dry-run or --execute-adapter-evidence" in err


def test_adapter_evidence_execution_requires_adapter_lane(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Guards ENG-89 release runner execution is scoped to adapter evidence."""
    rc = main(
        [
            "--suite",
            str(SUITE_PATH),
            "--profile",
            "near-term",
            "--execute-adapter-evidence",
        ]
    )

    assert rc == 2
    err = capsys.readouterr().err
    assert "unsupported selected lane(s): skillsbench-agent-matrix" in err


def test_adapter_evidence_execution_invokes_checker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Guards ENG-89 adapter-release-set execution plumbing."""
    captured = {}

    def fake_checker(argv: list[str]) -> int:
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(
        "tests.integration.run_suite._run_adapter_evidence_checker", fake_checker
    )
    result = tmp_path / "result.json"
    result.write_text("{}")

    rc = main(
        [
            "--suite",
            str(SUITE_PATH),
            "--lane",
            "adapter-release-set",
            "--execute-adapter-evidence",
            "--adapter-evidence-repo-root",
            str(tmp_path),
            "--skillsbench-result",
            str(result),
            "--open-pr-root",
            f"ContinualLearningBench={tmp_path}",
            "--allow-blocked",
        ]
    )

    assert rc == 0
    assert captured["argv"] == [
        "--repo-root",
        str(tmp_path),
        "--skillsbench-result",
        str(result),
        "--open-pr-root",
        f"ContinualLearningBench={tmp_path}",
        "--allow-blocked",
    ]


def test_run_adapter_evidence_rejects_empty_selection() -> None:
    """Guards ENG-89 adapter-release-set execution rejects missing lane."""
    args = SimpleNamespace(
        adapter_evidence_repo_root=Path.cwd(),
        skillsbench_result=None,
        open_pr_root=[],
        allow_blocked=False,
    )

    with pytest.raises(ValueError, match="requires lane adapter-release-set"):
        run_adapter_evidence([], args)


def test_adapter_evidence_checker_validates_programbench_fixture() -> None:
    """Guards ENG-89 adapter-release-set evidence for merged ProgramBench."""
    finding = check_programbench(Path.cwd())

    assert finding.status == "pass"
    assert "pipeline parity" in finding.message


def test_adapter_evidence_checker_accepts_skillsbench_result(tmp_path: Path) -> None:
    """Guards ENG-89 adapter-release-set evidence for SkillsBench smoke runs."""
    result = tmp_path / "result.json"
    result.write_text(
        """{
  "task_name": "jax-computing-basics",
  "rewards": {"reward": 1.0},
  "agent": "oracle",
  "error": null,
  "verifier_error": null
}
"""
    )

    finding = check_skillsbench_result(result)

    assert finding.status == "pass"
    assert "reward=1" in finding.message


def test_adapter_evidence_checker_marks_hilbench_eval_blocked(tmp_path: Path) -> None:
    """Guards ENG-89 adapter-release-set blocker reporting for HILBench."""
    evidence = tmp_path / "benchmarks" / "hilbench"
    evidence.mkdir(parents=True)
    (evidence / "parity_experiment.json").write_text(
        """{
  "structural_parity": {
    "results_summary": {"passed": 3, "failed": 0}
  },
  "eval_parity": {
    "status": "blocked",
    "blocker": "gated image access"
  }
}
"""
    )

    finding = check_hilbench(tmp_path)

    assert finding.status == "blocked"
    assert "gated image access" in finding.message


def test_adapter_evidence_checker_requires_continuallearningbench_dogfood(
    tmp_path: Path,
) -> None:
    """Guards ENG-89 adapter-release-set evidence for ContinualLearningBench dogfood."""
    evidence = tmp_path / "benchmarks" / "continuallearningbench"
    evidence.mkdir(parents=True)
    (evidence / "parity_experiment.json").write_text(
        """{
  "structural_parity": {"tasks_tested": 3, "passed": 3},
  "eval_parity": {"tasks_tested": 3, "passed": 3},
  "e2e_parity": {"tasks_tested": 10, "passed": 10}
}
"""
    )

    finding = check_continuallearningbench(tmp_path)

    assert finding.status == "fail"
    assert "dogfooding" in finding.message


def test_adapter_evidence_main_fails_without_skillsbench_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Guards ENG-89 adapter-release-set evidence CLI failure behavior."""
    rc = adapter_evidence_main(["--repo-root", str(Path.cwd())])

    assert rc == 1
    out = capsys.readouterr().out
    assert "SkillsBench" in out
    assert "representative result.json is required" in out
