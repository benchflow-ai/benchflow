from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow._utils.task_authoring import check_task
from tests.integration.check_adapter_evidence import (
    check_clbench,
    check_hilbench,
    check_programbench,
    check_skillsbench_result,
)
from tests.integration.check_adapter_evidence import (
    main as adapter_evidence_main,
)
from tests.integration.run_suite import (
    collect_lane_blockers,
    collect_lane_todos,
    expand_lane,
    load_suite,
    main,
    run_adapter_evidence,
    run_hosted_env_evidence,
    run_trace_evidence,
    select_lanes,
)

SUITE_PATH = Path("tests/integration/suites/release.yaml")
TERMINAL_BENCH_SMOKE_TASK = Path("tests/examples/terminal-bench-smoke-task")


def test_release_suite_loads_and_tracks_backlog_lanes() -> None:
    suite = load_suite(SUITE_PATH)
    lanes_by_id = {lane["id"]: lane for lane in suite["lanes"]}
    full_release_lane_ids = suite["execution_profiles"]["full-release"]["lanes"]

    assert suite["suite"] == "release"
    assert suite["lanes"]
    assert suite["run_tracking"]["future_system"] == "Linear"
    assert "near-term" in suite["execution_profiles"]
    assert "v0.4-release" in suite["execution_profiles"]
    assert "hosted-envs" in suite["execution_profiles"]
    assert "backlog" in suite["execution_profiles"]
    assert suite["execution_profiles"]["backlog"]["lanes"] == ["security-dind-smoke"]
    assert "security-dind-smoke" not in full_release_lane_ids
    assert lanes_by_id["security-dind-smoke"]["status"] == "backlog"
    assert lanes_by_id["security-dind-smoke"]["release_blocker"] is False
    assert all(
        lanes_by_id[lane_id]["release_blocker"] for lane_id in full_release_lane_ids
    )


def test_release_suite_benchmarks_have_source_uids() -> None:
    """Guards ENG-92 benchmark identity uses source repo/path/ref, not display names."""
    suite = load_suite(SUITE_PATH)

    seen = set()
    for group in suite["axes"]["benchmarks"].values():
        for benchmark in group:
            source = benchmark["source"]
            expected_uid = f"{source['repo']}:{source['path']}@{source['ref']}"
            assert benchmark["uid"] == expected_uid
            assert benchmark["uid"] not in seen
            seen.add(benchmark["uid"])

    assert "benchflow-ai/skillsbench:tasks@main" in seen
    assert "benchflow-ai/benchmarks:datasets/harvey-lab/tasks@main" in seen


def test_release_suite_hosted_env_hubs_have_hub_urls() -> None:
    """Guards ENG-92 hosted env hubs are tracked as env sources, not benchmarks."""
    suite = load_suite(SUITE_PATH)

    hubs = suite["axes"]["hosted_env_hubs"]["current"]
    hubs_by_platform = {hub["platform"]: hub for hub in hubs}

    assert {hub["platform"] for hub in hubs} == {
        "openreward",
        "harbor",
        "primeintellect",
    }
    assert all(hub["hub_url"].startswith("https://") for hub in hubs)
    assert all("env_uid_pattern" in hub for hub in hubs)
    assert [
        env["env_uid"] for env in hubs_by_platform["openreward"]["selected_envs"]
    ] == [
        ("openreward:GeneralReasoning/KellyBench@be14865a-3c70-422e-a2ba-f45c132cd29a"),
        ("openreward:GeneralReasoning/CTF@fcfcd0ef-1298-40e9-9492-83628fd98a1c"),
    ]
    assert [
        env["env_uid"] for env in hubs_by_platform["primeintellect"]["selected_envs"]
    ] == [
        "primeintellect:primeintellect/reverse-text@0.1.4",
        "primeintellect:primeintellect/math-python@0.1.10",
    ]
    assert [env["env_uid"] for env in hubs_by_platform["harbor"]["selected_envs"]] == [
        (
            "harbor:terminal-bench/adaptive-rejection-sampler@"
            "69671fbaac6d67a7ef0dfec016cc38a64ef7a77c"
        ),
        (
            "harbor:binary-audit/caddy-backdoor-detect@"
            "75f3e6e331776b80f77faa3d2ff80627b8b5d069"
        ),
    ]
    assert "todo" not in hubs_by_platform["openreward"]
    assert "todo" not in hubs_by_platform["primeintellect"]
    assert "todo" not in hubs_by_platform["harbor"]


def test_select_lanes_rejects_unknown_lane() -> None:
    suite = load_suite(SUITE_PATH)

    with pytest.raises(ValueError, match="unknown lane"):
        select_lanes(suite, ["not-a-lane"])


def test_expand_shared_sandbox_smoke_resolves_release_gated_sandboxes() -> None:
    """Guards ENG-92 keeps v0.4 release sandboxes aligned with live CLI support."""
    suite = load_suite(SUITE_PATH)
    lane = select_lanes(suite, ["shared-sandbox-smoke"])[0]

    expanded = expand_lane(suite, lane)

    assert expanded["matrix"]["agents"] == ["gemini"]
    assert expanded["matrix"]["sandboxes"] == ["docker", "daytona"]
    assert expanded["matrix"]["task_sets"] == [
        {
            "source": {
                "kind": "local_task",
                "path": "tests/examples/hello-world-task",
            },
            "scope": "v0.4 supported sandbox smoke",
        }
    ]
    assert expanded["todos"] == []


def test_collect_lane_todos_groups_unresolved_manifest_todos() -> None:
    """Guards ENG-92 manifest TODOs remain machine-detectable."""
    suite = load_suite(SUITE_PATH)
    lanes = select_lanes(
        suite,
        ["shared-sandbox-smoke", "skillsbench-agent-matrix", "terminal-bench-smoke"],
    )

    lane_todos = collect_lane_todos(suite, lanes)

    assert lane_todos == {}


def test_security_dind_lane_has_concrete_task_but_remains_blocked() -> None:
    """Guards ENG-92 security DinD has a task without claiming runnable support."""
    suite = load_suite(SUITE_PATH)
    lane = select_lanes(suite, ["security-dind-smoke"])[0]

    expanded = expand_lane(suite, lane)

    assert lane["status"] == "backlog"
    assert lane["release_blocker"] is False
    assert "Firecracker/K8s" in lane["backlog_reason"]
    assert "bench eval create --sandbox firecracker" in lane["activation_criteria"][0]
    task_set = expanded["matrix"]["task_sets"][0]
    assert task_set["source"]["env_uid"] == (
        "harbor:termigen-environments/docker_escape_privileged_container_medium@"
        "dc329464161db64b0c670f46fa39b62e4719dddd"
    )
    assert expanded["todos"] == []
    assert collect_lane_blockers([lane]) == {"security-dind-smoke": lane["blocked_by"]}


def test_terminal_bench_smoke_task_is_valid() -> None:
    """Guards ENG-92 terminal smoke uses a real local shell-verifier task."""
    assert check_task(TERMINAL_BENCH_SMOKE_TASK) == []


def test_dry_run_prints_selected_lane(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(
        ["--suite", str(SUITE_PATH), "--lane", "security-dind-smoke", "--dry-run"]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "security-dind-smoke" in out
    assert "Status: backlog" in out
    assert "Backlog reason: Firecracker/K8s are not included" in out
    assert "Activation criteria:" in out
    assert "firecracker, k8s" in out
    assert "Blocked by:" in out
    assert "not exposed by the current v0.4 CLI" in out


def test_full_release_dry_run_passes_current_release_gate(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Guards ENG-92 full-release excludes future Firecracker/K8s backlog lanes."""
    rc = main(
        [
            "--suite",
            str(SUITE_PATH),
            "--profile",
            "full-release",
            "--dry-run",
            "--fail-on-todo",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert "Profile: full-release" in captured.out
    assert "shared-sandbox-smoke" in captured.out
    assert "hosted-env-compatibility-board" in captured.out
    assert "terminal-bench-smoke" in captured.out
    assert "trace-to-task-e2e" in captured.out
    assert "security-dind-smoke" not in captured.out
    assert captured.err == ""


def test_backlog_profile_tracks_security_dind_blocker(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Guards ENG-92 backlog keeps future Firecracker/K8s DinD evidence concrete."""
    rc = main(
        [
            "--suite",
            str(SUITE_PATH),
            "--profile",
            "backlog",
            "--dry-run",
            "--fail-on-todo",
        ]
    )

    assert rc == 1
    captured = capsys.readouterr()
    assert "Profile: backlog" in captured.out
    assert "Status: backlog" in captured.out
    assert (
        "harbor:termigen-environments/docker_escape_privileged_container_medium@"
        in captured.out
    )
    assert "bench eval create --sandbox firecracker" in captured.out
    assert "unresolved TODOs or blocked lanes in selected lane(s):" in captured.err
    assert "security-dind-smoke (blocked: 2)" in captured.err


def test_trace_to_task_e2e_prints_concrete_sources_and_evidence(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Guards ENG-92 trace-to-task release planning is evidence-backed."""
    rc = main(
        [
            "--suite",
            str(SUITE_PATH),
            "--lane",
            "trace-to-task-e2e",
            "--dry-run",
            "--fail-on-todo",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    out = captured.out
    assert "Blocked by:" not in out
    assert "jsonl:tests/examples/traces/minimal-claude.jsonl (claude-code)" in out
    assert (
        "opentraces:tests/examples/traces/minimal-opentraces.jsonl (opentraces)" in out
    )
    assert (
        "generated_trace_tasks (2 sources); "
        "evidence=dogfood/2026-05-19-trace-to-task-e2e"
    ) in out
    assert captured.err == ""


def test_v04_release_dry_run_passes_fail_on_todo(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Guards ENG-92 v0.4 release planning is distinct from backlog sandboxes."""
    rc = main(
        [
            "--suite",
            str(SUITE_PATH),
            "--profile",
            "v0.4-release",
            "--dry-run",
            "--fail-on-todo",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert "Profile: v0.4-release" in captured.out
    assert "Preferred sandboxes: docker, daytona" in captured.out
    assert "local:tests/examples/hello-world-task" in captured.out
    assert "local:tests/examples/terminal-bench-smoke-task" in captured.out
    assert "firecracker, k8s" not in captured.out
    assert captured.err == ""


def test_near_term_dry_run_passes_fail_on_todo(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Guards ENG-92 near-term profile is TODO-free in the manifest."""
    rc = main(
        [
            "--suite",
            str(SUITE_PATH),
            "--profile",
            "near-term",
            "--dry-run",
            "--fail-on-todo",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert "Profile: near-term" in captured.out
    assert captured.err == ""


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
    assert "benchmarks:" in out
    assert "SkillsBench [benchflow-ai/skillsbench:tasks@main]" in out
    assert (
        "HILBench [benchflow-ai/benchmarks:datasets/hilbench/tasks@main] (PR #279)"
        in out
    )
    assert "Task budget:" in out
    assert "per_adapter: 1" in out
    assert "terminal-bench-smoke" not in out
    assert "shared-sandbox-smoke" not in out


def test_hosted_env_profile_prints_hub_level_plan(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Guards ENG-92 hosted env compatibility starts with hub-level coverage."""
    rc = main(
        [
            "--suite",
            str(SUITE_PATH),
            "--profile",
            "hosted-envs",
            "--dry-run",
            "--fail-on-todo",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    out = captured.out
    assert "Profile: hosted-envs" in out
    assert "hosted-env-compatibility-board" in out
    assert "OpenReward environments [https://openreward.ai/environments]" in out
    assert "openreward:GeneralReasoning/KellyBench@" in out
    assert "openreward:GeneralReasoning/CTF@" in out
    assert "Harbor Hub [https://hub.harborframework.com/]" in out
    assert "PrimeIntellect Environments Hub" in out
    assert "primeintellect:primeintellect/reverse-text@0.1.4" in out
    assert "primeintellect:primeintellect/math-python@0.1.10" in out
    assert "harbor:terminal-bench/adaptive-rejection-sampler@" in out
    assert "harbor:binary-audit/caddy-backdoor-detect@" in out
    assert "Evidence dir: dogfood/2026-05-19-release-gate/hosted-envs" in out
    assert "--execute-hosted-env-evidence" in out
    assert "TODOs:" not in out
    assert captured.err == ""


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
            f"CLBench={tmp_path}",
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
        f"CLBench={tmp_path}",
        "--allow-blocked",
    ]


def test_trace_evidence_execution_requires_trace_lane(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Guards ENG-93 trace evidence execution is scoped to trace-to-task."""
    rc = main(
        [
            "--suite",
            str(SUITE_PATH),
            "--profile",
            "full-release",
            "--execute-trace-evidence",
        ]
    )

    assert rc == 2
    err = capsys.readouterr().err
    assert "unsupported selected lane(s): shared-sandbox-smoke" in err


def test_trace_evidence_execution_invokes_checker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Guards ENG-93 trace-to-task-e2e execution plumbing."""
    captured = {}

    def fake_checker(argv: list[str]) -> int:
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(
        "tests.integration.run_suite._run_trace_evidence_checker", fake_checker
    )

    rc = main(
        [
            "--suite",
            str(SUITE_PATH),
            "--lane",
            "trace-to-task-e2e",
            "--execute-trace-evidence",
            "--trace-evidence-repo-root",
            str(tmp_path),
            "--trace-evidence-dir",
            str(tmp_path / "evidence"),
            "--trace-evidence-sandbox",
            "docker",
            "--run-trace-eval",
        ]
    )

    assert rc == 0
    assert captured["argv"] == [
        "--suite",
        str(SUITE_PATH),
        "--repo-root",
        str(tmp_path),
        "--sandbox",
        "docker",
        "--evidence-dir",
        str(tmp_path / "evidence"),
        "--run-eval",
    ]


def test_hosted_env_evidence_execution_requires_hosted_lane(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Guards ENG-92 hosted-env evidence execution is scoped to the board."""
    rc = main(
        [
            "--suite",
            str(SUITE_PATH),
            "--profile",
            "full-release",
            "--execute-hosted-env-evidence",
        ]
    )

    assert rc == 2
    err = capsys.readouterr().err
    assert "unsupported selected lane(s): shared-sandbox-smoke" in err


def test_hosted_env_evidence_execution_invokes_checker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Guards ENG-92 hosted-env evidence execution plumbing."""
    captured = {}

    def fake_checker(argv: list[str]) -> int:
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(
        "tests.integration.run_suite._run_hosted_env_evidence_checker", fake_checker
    )

    rc = main(
        [
            "--suite",
            str(SUITE_PATH),
            "--lane",
            "hosted-env-compatibility-board",
            "--execute-hosted-env-evidence",
            "--hosted-env-evidence-dir",
            str(tmp_path / "hosted-envs"),
            "--harbor-inventory-limit",
            "3",
        ]
    )

    assert rc == 0
    assert captured["argv"] == [
        "--suite",
        str(SUITE_PATH),
        "--evidence-dir",
        str(tmp_path / "hosted-envs"),
        "--harbor-inventory-limit",
        "3",
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


def test_run_trace_evidence_rejects_empty_selection() -> None:
    """Guards ENG-93 trace evidence rejects missing lane."""
    args = SimpleNamespace(
        suite=SUITE_PATH,
        trace_evidence_repo_root=Path.cwd(),
        trace_evidence_dir=None,
        run_trace_eval=False,
        trace_evidence_sandbox="docker",
    )

    with pytest.raises(ValueError, match="requires lane trace-to-task-e2e"):
        run_trace_evidence([], args)


def test_run_hosted_env_evidence_rejects_empty_selection() -> None:
    """Guards ENG-92 hosted-env evidence rejects missing lane."""
    args = SimpleNamespace(
        suite=SUITE_PATH,
        hosted_env_evidence_dir=None,
        harbor_inventory_limit=2,
    )

    with pytest.raises(
        ValueError, match="requires lane hosted-env-compatibility-board"
    ):
        run_hosted_env_evidence([], args)


def test_hosted_env_evidence_main_writes_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Guards ENG-92 hosted-env evidence writes durable hub artifacts."""
    from tests.integration import check_hosted_env_evidence

    def fake_harbor_registry(*args, **kwargs):
        out = kwargs["out"]
        record = {
            "framework": "harbor",
            "env_uid": "harbor:dataset/task@ref",
            "hub_url": "https://hub.harborframework.com/",
            "status": "pass",
        }
        out.write_text(f"{json.dumps(record)}\n")
        return [record]

    monkeypatch.setattr(
        check_hosted_env_evidence,
        "check_harbor_registry",
        fake_harbor_registry,
    )

    rc = check_hosted_env_evidence.main(
        [
            "--suite",
            str(SUITE_PATH),
            "--evidence-dir",
            str(tmp_path),
            "--harbor-inventory-limit",
            "1",
        ]
    )

    assert rc == 0
    assert (tmp_path / "hosted-env-evidence.json").exists()
    assert (tmp_path / "harbor-registry-inventory.jsonl").exists()
    out = capsys.readouterr().out
    assert "openreward" in out
    assert "primeintellect" in out
    assert "harbor_inventory" in out


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


def test_adapter_evidence_checker_marks_hilbench_bucket_download_bug(
    tmp_path: Path,
) -> None:
    """Guards ENG-89 adapter-release-set bucket download reporting for HILBench."""
    evidence = tmp_path / "benchmarks" / "hilbench"
    evidence.mkdir(parents=True)
    (evidence / "parity_experiment.json").write_text(
        """{
  "structural_parity": {
    "results_summary": {"passed": 3, "failed": 0}
  },
  "eval_parity": {
    "status": "blocked",
    "blocker": "HUGGINGFACE_TOKEN returns 404 for ScaleAI/hil-bench-swe-images"
  }
}
"""
    )

    finding = check_hilbench(tmp_path)

    assert finding.status == "fail"
    assert "bucket objects" in finding.message
    assert "resolve/images/<uid>.tar.zst" in finding.message


def test_adapter_evidence_checker_requires_hilbench_eval_parity_pass(
    tmp_path: Path,
) -> None:
    """Guards ENG-89 HILBench release evidence does not pass on stale eval parity."""
    evidence = tmp_path / "benchmarks" / "hilbench"
    evidence.mkdir(parents=True)
    (evidence / "parity_experiment.json").write_text(
        """{
  "structural_parity": {
    "results_summary": {"passed": 3, "failed": 0}
  },
  "eval_parity": {
    "status": "needs_refresh",
    "blocker": null
  }
}
"""
    )

    finding = check_hilbench(tmp_path)

    assert finding.status == "fail"
    assert "expected 'passed'" in finding.message


def test_adapter_evidence_checker_requires_clbench_dogfood(tmp_path: Path) -> None:
    """Guards ENG-89 adapter-release-set evidence for CLBench dogfood."""
    evidence = tmp_path / "benchmarks" / "clbench"
    evidence.mkdir(parents=True)
    (evidence / "parity_experiment.json").write_text(
        """{
  "structural_parity": {"tasks_tested": 3, "passed": 3},
  "eval_parity": {"tasks_tested": 3, "passed": 3},
  "e2e_parity": {"tasks_tested": 10, "passed": 10}
}
"""
    )

    finding = check_clbench(tmp_path)

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
    assert "HILBench" in out
    assert "--open-pr-root HILBench=/path/to/worktree" in out


def test_adapter_evidence_main_requires_open_pr_roots(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Guards ENG-89 open adapter PRs are release-blocking evidence."""
    result = tmp_path / "skillsbench-result.json"
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

    rc = adapter_evidence_main(
        [
            "--repo-root",
            str(Path.cwd()),
            "--skillsbench-result",
            str(result),
        ]
    )

    assert rc == 1
    out = capsys.readouterr().out
    assert "SkillsBench" in out
    assert "reward=1" in out
    assert "HILBench" in out
    assert "OpaqueToolsBench" in out
    assert "CLBench" in out
    assert "--open-pr-root HILBench=/path/to/worktree" in out
