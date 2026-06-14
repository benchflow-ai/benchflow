"""Coverage for the official Stagehand parity driver."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from benchflow.agent_router_parity import build_verify_report


def _load_stagehand_parity_module() -> ModuleType:
    script = (
        Path(__file__).parents[1] / "benchmarks" / "stagehand-smoke" / "parity_test.py"
    )
    # Each smoke benchmark ships its own ``import_upstream`` module; loaded by
    # bare name they collide in ``sys.modules`` across benchmarks. Evict any
    # stale entry (and our own module name) so ``parity_test.py``'s
    # ``from import_upstream import import_tasks`` binds to this benchmark's
    # copy regardless of test-collection order.
    stale_import_upstream = sys.modules.pop("import_upstream", None)
    sys.modules.pop("stagehand_parity_test", None)
    sys.path.insert(0, str(script.parent))
    try:
        spec = importlib.util.spec_from_file_location("stagehand_parity_test", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(script.parent))
        sys.modules.pop("import_upstream", None)
        if stale_import_upstream is not None:
            sys.modules["import_upstream"] = stale_import_upstream


def test_stagehand_parity_script_writes_loop_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_stagehand_parity_module()
    repo = tmp_path / "stagehand"
    source_dir = repo / "packages" / "evals" / "tasks" / "bench" / "agent"
    source_dir.mkdir(parents=True)
    (source_dir / "sign_in.ts").write_text(
        """\
import { defineBenchTask } from "../../../framework/defineTask.js";

export default defineBenchTask(
  { name: "agent/sign_in" },
  async ({ agent, v3 }) => {
    const page = v3.context.pages()[0];
    await page.goto("https://v0-modern-login-flow.vercel.app/");
    await agent.execute({
      instruction:
        "Sign in with the email address 'test@browserbaser.com' and the password 'stagehand=goated' ",
      maxSteps: Number(process.env.AGENT_EVAL_MAX_STEPS) || 15,
    });
    const url = page.url();
    return { _success: url === "https://v0-modern-login-flow.vercel.app/authorized", observations: url };
  },
);
"""
    )
    runner = repo / "packages" / "evals" / "dist" / "esm" / "framework"
    runner.mkdir(parents=True)
    (runner / "runner.js").write_text("export {};\n")
    monkeypatch.setattr(module.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(cmd, **kwargs):
        command = [str(part) for part in cmd]
        if command[:3] == ["docker", "container", "ls"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["docker", "network", "ls"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:5] == ["node", "--import", "tsx", "--input-type=module", "-e"]:
            payload = {
                "summary": {"passed": 1, "failed": 0, "total": 1},
                "results": [
                    {
                        "name": "agent/sign_in",
                        "score": 1,
                        "output": {
                            "_success": True,
                            "observations": (
                                "https://v0-modern-login-flow.vercel.app/authorized"
                            ),
                            "logs": ["started", "done"],
                            "metrics": {"total_ms": {"value": 1200}},
                        },
                    }
                ],
            }
            stdout = (
                "stagehand logs\n"
                f"{module._ORIGINAL_RESULT_MARKER}{json.dumps(payload)}\n"
            )
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")
        if command[:4] == ["uv", "run", "bench", "eval"]:
            jobs_dir = Path(command[command.index("--jobs-dir") + 1])
            result_dir = jobs_dir / "run" / "agent-sign_in__fake"
            artifacts = result_dir / "artifacts"
            artifacts.mkdir(parents=True)
            (result_dir / "result.json").write_text(
                json.dumps(
                    {
                        "task_name": "agent-sign_in",
                        "agent": "stagehand-agent",
                        "rewards": {"reward": 1.0},
                        "trajectory_summary": {
                            "steps": 3,
                            "tool_call_steps": 1,
                        },
                        "n_tool_calls": 1,
                        "timing": {"total": 12.0},
                        "error": None,
                        "verifier_error": None,
                    }
                )
            )
            (artifacts / "browser-use-smoke-trace.json").write_text(
                json.dumps(
                    {
                        "framework": "benchflow-stagehand-agent",
                        "steps": [{"i": 1}, {"i": 2}],
                        "screenshots_b64": ["abc"],
                        "stagehand_current_url": (
                            "https://v0-modern-login-flow.vercel.app/authorized"
                        ),
                        "duration_sec": 3.0,
                    }
                )
            )
            (artifacts / "stagehand-url-verifier.json").write_text(
                json.dumps(
                    {
                        "reward": 1.0,
                        "current_url": (
                            "https://v0-modern-login-flow.vercel.app/authorized"
                        ),
                        "expected_url": (
                            "https://v0-modern-login-flow.vercel.app/authorized"
                        ),
                    }
                )
            )
            stdout = json.dumps(
                {
                    "status": "completed",
                    "ok": True,
                    "result": {
                        "total": 1,
                        "errored": 0,
                        "verifier_errored": 0,
                        "elapsed_sec": 12.0,
                    },
                    "summary": {
                        "total": 1,
                        "errored": 0,
                        "verifier_errored": 0,
                        "agent": "stagehand-agent",
                        "total_trajectory_steps": 3,
                        "elapsed_sec": 12.0,
                    },
                    "summary_path": str(jobs_dir / "summary.json"),
                }
            )
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")
        raise AssertionError(f"unexpected command: {command}")

    parity_out = tmp_path / "evidence" / "parity_experiment.json"
    summary = module.run_parity(
        stagehand_repo=repo,
        task="agent/sign_in",
        model="google/gemini-3.5-flash",
        provider="google",
        agent_mode="dom",
        benchflow_agent="stagehand-agent",
        sandbox="docker",
        parity_out=parity_out,
        upstream_commit="stagehand-test",
        run_fn=fake_run,
    )

    assert summary["ok"] is True
    parity = json.loads(parity_out.read_text())
    adoption = json.loads(parity_out.with_name("adoption_report.json").read_text())
    assert parity["status"] == "parity-confirmed"
    assert parity["adapter_parity"]["benchmark_adapter"] == "stagehand-evals"
    assert adoption["schema"] == "benchflow.environment-adapter-adoption-report.v1"
    assert adoption["planes"]["agent_adapter"] == "stagehand-agent"
    assert (
        adoption["parity"]["criteria_agreed"] == adoption["parity"]["criteria_compared"]
    )


def _write_stagehand_repo(repo: Path) -> None:
    """Materialize a minimal built Stagehand repo for the parity importer."""

    source_dir = repo / "packages" / "evals" / "tasks" / "bench" / "agent"
    source_dir.mkdir(parents=True)
    (source_dir / "sign_in.ts").write_text(
        """\
import { defineBenchTask } from "../../../framework/defineTask.js";

export default defineBenchTask(
  { name: "agent/sign_in" },
  async ({ agent, v3 }) => {
    const page = v3.context.pages()[0];
    await page.goto("https://v0-modern-login-flow.vercel.app/");
    await agent.execute({
      instruction:
        "Sign in with the email address 'test@browserbaser.com' and the password 'stagehand=goated' ",
      maxSteps: Number(process.env.AGENT_EVAL_MAX_STEPS) || 15,
    });
    const url = page.url();
    return { _success: url === "https://v0-modern-login-flow.vercel.app/authorized", observations: url };
  },
);
"""
    )
    runner = repo / "packages" / "evals" / "dist" / "esm" / "framework"
    runner.mkdir(parents=True)
    (runner / "runner.js").write_text("export {};\n")


def _make_fake_run(
    module: ModuleType,
    *,
    original_score: float = 1.0,
    bench_reward: float = 1.0,
    write_trace_artifact: bool = True,
    eval_status: str = "completed",
    eval_ok: bool = True,
):
    """Build a ``run_fn`` whose BenchFlow side is tunable for divergence cases."""

    authorized = "https://v0-modern-login-flow.vercel.app/authorized"

    def fake_run(cmd, **kwargs):
        command = [str(part) for part in cmd]
        if command[:3] == ["docker", "container", "ls"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["docker", "network", "ls"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:5] == ["node", "--import", "tsx", "--input-type=module", "-e"]:
            payload = {
                "summary": {"passed": 1, "failed": 0, "total": 1},
                "results": [
                    {
                        "name": "agent/sign_in",
                        "score": original_score,
                        "output": {
                            "_success": original_score >= 1.0,
                            "observations": authorized,
                            "logs": ["started", "done"],
                            "metrics": {"total_ms": {"value": 1200}},
                        },
                    }
                ],
            }
            stdout = (
                "stagehand logs\n"
                f"{module._ORIGINAL_RESULT_MARKER}{json.dumps(payload)}\n"
            )
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")
        if command[:4] == ["uv", "run", "bench", "eval"]:
            jobs_dir = Path(command[command.index("--jobs-dir") + 1])
            result_dir = jobs_dir / "run" / "agent-sign_in__fake"
            artifacts = result_dir / "artifacts"
            artifacts.mkdir(parents=True)
            (result_dir / "result.json").write_text(
                json.dumps(
                    {
                        "task_name": "agent-sign_in",
                        "agent": "stagehand-agent",
                        "rewards": {"reward": bench_reward},
                        "trajectory_summary": {
                            "steps": 3,
                            "tool_call_steps": 1,
                        },
                        "n_tool_calls": 1,
                        "timing": {"total": 12.0},
                        "error": None,
                        "verifier_error": None,
                    }
                )
            )
            if write_trace_artifact:
                (artifacts / "browser-use-smoke-trace.json").write_text(
                    json.dumps(
                        {
                            "framework": "benchflow-stagehand-agent",
                            "steps": [{"i": 1}, {"i": 2}],
                            "screenshots_b64": ["abc"],
                            "stagehand_current_url": authorized,
                            "duration_sec": 3.0,
                        }
                    )
                )
                (artifacts / "stagehand-url-verifier.json").write_text(
                    json.dumps(
                        {
                            "reward": bench_reward,
                            "current_url": authorized,
                            "expected_url": authorized,
                        }
                    )
                )
            stdout = json.dumps(
                {
                    "status": eval_status,
                    "ok": eval_ok,
                    "result": {
                        "total": 1,
                        "errored": 0,
                        "verifier_errored": 0,
                        "elapsed_sec": 12.0,
                    },
                    "summary": {
                        "total": 1,
                        "errored": 0,
                        "verifier_errored": 0,
                        "agent": "stagehand-agent",
                        "total_trajectory_steps": 3,
                        "elapsed_sec": 12.0,
                    },
                    "summary_path": str(jobs_dir / "summary.json"),
                }
            )
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")
        raise AssertionError(f"unexpected command: {command}")

    return fake_run


def _run_parity(module: ModuleType, repo: Path, parity_out: Path, fake_run) -> Any:
    return module.run_parity(
        stagehand_repo=repo,
        task="agent/sign_in",
        model="google/gemini-3.5-flash",
        provider="google",
        agent_mode="dom",
        benchflow_agent="stagehand-agent",
        sandbox="docker",
        parity_out=parity_out,
        upstream_commit="stagehand-test",
        run_fn=fake_run,
    )


def test_stagehand_parity_records_no_trace_divergence_without_crashing(
    tmp_path: Path, monkeypatch
) -> None:
    """No trace / no URL but eval completed with reward 0 -> recorded divergence.

    Mirrors the real anti-bot case (original 1.0, BenchFlow 0.0, no trace): the
    harness must RECORD the divergence as evidence (parity-recorded with a
    reward delta) instead of raising AssertionError.
    """

    module = _load_stagehand_parity_module()
    repo = tmp_path / "stagehand"
    _write_stagehand_repo(repo)
    monkeypatch.setattr(module.shutil, "which", lambda name: f"/usr/bin/{name}")
    fake_run = _make_fake_run(
        module,
        original_score=1.0,
        bench_reward=0.0,
        write_trace_artifact=False,
    )

    parity_out = tmp_path / "evidence" / "parity_experiment.json"
    summary = _run_parity(module, repo, parity_out, fake_run)

    assert summary["ok"] is False
    divergence = summary["divergence"]
    assert divergence["status"] == "divergent"
    assert divergence["original_reward"] == 1.0
    assert divergence["benchflow_reward"] == 0.0
    assert divergence["reward_delta"] == 1.0
    assert any("no-trace" in note for note in divergence["notes"])
    assert any("reward-mismatch" in note for note in divergence["notes"])

    parity = json.loads(parity_out.read_text())
    assert parity["status"] == "parity-recorded"
    reward_sample = parity["agent_parity"]["results"][0]
    assert reward_sample["legacy_reward"] == 1.0
    assert reward_sample["converted_reward"] == 0.0
    assert reward_sample["reward_delta"] == 1.0
    report = build_verify_report("stagehand-smoke", parity)
    assert report.verdict == "parity-divergent"


def test_stagehand_parity_records_reward_mismatch_without_crashing(
    tmp_path: Path, monkeypatch
) -> None:
    """Reward mismatch (trace present) -> recorded divergence, not an exception."""

    module = _load_stagehand_parity_module()
    repo = tmp_path / "stagehand"
    _write_stagehand_repo(repo)
    monkeypatch.setattr(module.shutil, "which", lambda name: f"/usr/bin/{name}")
    fake_run = _make_fake_run(
        module,
        original_score=1.0,
        bench_reward=0.0,
        write_trace_artifact=True,
    )

    parity_out = tmp_path / "evidence" / "parity_experiment.json"
    summary = _run_parity(module, repo, parity_out, fake_run)

    assert summary["ok"] is False
    assert summary["divergence"]["reward_delta"] == 1.0
    assert any("reward-mismatch" in note for note in summary["divergence"]["notes"])
    # The trace criteria all agree (full trace, matching URL), so the raw
    # experiment status stays parity-confirmed -- exactly as the browser-use
    # harness records a reward-only divergence. The honest delta is preserved in
    # agent_parity and the downstream verify verdict is parity-divergent.
    parity = json.loads(parity_out.read_text())
    reward_sample = parity["agent_parity"]["results"][0]
    assert reward_sample["legacy_reward"] == 1.0
    assert reward_sample["converted_reward"] == 0.0
    assert reward_sample["reward_delta"] == 1.0
    report = build_verify_report("stagehand-smoke", parity)
    assert report.verdict == "parity-divergent"


def test_stagehand_parity_hard_errors_when_eval_did_not_complete(
    tmp_path: Path, monkeypatch
) -> None:
    """Eval status != completed is an infra failure -> still a hard error."""

    module = _load_stagehand_parity_module()
    repo = tmp_path / "stagehand"
    _write_stagehand_repo(repo)
    monkeypatch.setattr(module.shutil, "which", lambda name: f"/usr/bin/{name}")
    fake_run = _make_fake_run(
        module,
        eval_status="failed",
        eval_ok=False,
    )

    parity_out = tmp_path / "evidence" / "parity_experiment.json"
    with pytest.raises(AssertionError, match="did not complete cleanly"):
        _run_parity(module, repo, parity_out, fake_run)


def test_stagehand_original_runner_prefers_built_tasks_without_tsx(
    tmp_path: Path,
) -> None:
    """Guards the 0.7 Stagehand parity loop against fragile source-loader deps."""

    module = _load_stagehand_parity_module()
    repo = tmp_path / "stagehand"
    runner = repo / "packages" / "evals" / "dist" / "esm" / "framework"
    runner.mkdir(parents=True)
    (runner / "runner.js").write_text("export {};\n")
    task_dir = (
        repo / "packages" / "evals" / "dist" / "esm" / "tasks" / "bench" / "agent"
    )
    task_dir.mkdir(parents=True)
    (task_dir / "sign_in.js").write_text("export default {};\n")

    seen_commands: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        command = [str(part) for part in cmd]
        seen_commands.append(command)
        assert command[:3] == ["node", "--input-type=module", "-e"]
        assert "--import" not in command
        payload = {
            "summary": {"passed": 1, "failed": 0, "total": 1},
            "results": [
                {
                    "name": "agent/sign_in",
                    "score": 1,
                    "output": {
                        "_success": True,
                        "observations": (
                            "https://v0-modern-login-flow.vercel.app/authorized"
                        ),
                        "logs": ["started"],
                    },
                }
            ],
        }
        stdout = f"{module._ORIGINAL_RESULT_MARKER}{json.dumps(payload)}\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    result = module.run_original_stagehand(
        stagehand_repo=repo,
        task="agent/sign_in",
        model="google/gemini-3.5-flash",
        provider="google",
        agent_mode="dom",
        run_fn=fake_run,
    )

    assert seen_commands
    assert result["score"] == 1.0
    assert result["duration_sec"] is not None
    assert result["final_url"] == "https://v0-modern-login-flow.vercel.app/authorized"


def test_stagehand_original_result_extracts_final_url_from_logs() -> None:
    module = _load_stagehand_parity_module()

    result = module._normalize_original_result(
        {
            "summary": {"passed": 1, "failed": 0, "total": 1},
            "results": [
                {
                    "name": "agent/steam_games",
                    "score": 1,
                    "output": {
                        "_success": True,
                        "logs": [
                            {
                                "message": "performing action",
                                "parsedAuxiliary": {
                                    "url": "https://store.steampowered.com/charts"
                                },
                            }
                        ],
                    },
                }
            ],
        },
        task="agent/steam_games",
        fallback_duration_sec=1.2,
    )

    assert result["final_url"] == "https://store.steampowered.com/charts"
    assert result["duration_sec"] == 1.2


def test_stagehand_artifact_manifest_uses_contains_for_contains_url_checks() -> None:
    module = _load_stagehand_parity_module()

    manifest = module._artifact_manifest(
        {
            "success_check": {
                "type": "url_contains",
                "value": "https://store.steampowered.com/",
            }
        }
    )

    current_url = next(
        item for item in manifest if item["id"] == "stagehand-current-url"
    )
    assert current_url["contains"] == "https://store.steampowered.com/"
    assert "equals" not in current_url
    assert module._url_satisfies_success_check(
        "https://store.steampowered.com/charts/mostplayed",
        {
            "success_check": {
                "type": "url_contains",
                "value": "https://store.steampowered.com/",
            }
        },
    )


def test_stagehand_leaked_resources_tolerates_preexisting_and_detects_leak() -> None:
    # Per-run scoping: the stagehand parity script's cleanup check must tolerate
    # pre-existing/concurrent benchflow-owned containers (they are in ``before``)
    # while still flagging a container/network THIS run created and left behind.
    module = _load_stagehand_parity_module()

    before = {
        "available": True,
        "containers": ["hello-world-task-main-1", "skillsbench-citation-main-1"],
        "networks": [],
    }

    # Clean run: pre-existing survivors persist, nothing new -> no leak.
    clean = module._leaked_benchflow_resources(before=before, after=dict(before))
    assert clean["containers"] == []
    assert clean["networks"] == []

    # Real leak: a new benchflow container/network from THIS run remains.
    after = {
        "available": True,
        "containers": [*before["containers"], "benchflow-agent-sign-in-main-1"],
        "networks": ["benchflow-agent-sign-in_default"],
    }
    leaked = module._leaked_benchflow_resources(before=before, after=after)
    assert leaked["containers"] == ["benchflow-agent-sign-in-main-1"]
    assert leaked["networks"] == ["benchflow-agent-sign-in_default"]
