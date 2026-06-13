"""Coverage for importing official Stagehand eval task slices."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from benchflow.adapters.stagehand import StagehandEvalAdapter


def test_import_upstream_stagehand_task_writes_descriptor_package(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "stagehand"
    task_dir = repo / "packages" / "evals" / "tasks" / "bench" / "agent"
    task_dir.mkdir(parents=True)
    (task_dir / "sign_in.ts").write_text(
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
    return { _success: url === "https://v0-modern-login-flow.vercel.app/authorized" };
  },
);
"""
    )
    out_dir = tmp_path / "tasks"
    script = (
        Path(__file__).parents[1]
        / "benchmarks"
        / "stagehand-smoke"
        / "import_upstream.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--stagehand-repo",
            str(repo),
            "--out-dir",
            str(out_dir),
            "--tasks",
            "agent/sign_in",
            "--upstream-commit",
            "stagehand-test",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["unsupported"] == []
    imported = Path(payload["task_dirs"][0])
    descriptor = json.loads((imported / "stagehand-task.json").read_text())
    assert descriptor["benchmark"] == "stagehand-evals"
    assert descriptor["upstream_commit"] == "stagehand-test"
    assert descriptor["success_check"]["type"] == "url_exact"
    assert (imported / "environment" / "Dockerfile").is_file()

    inbound = StagehandEvalAdapter.from_task_dir(imported)
    assert inbound.config.metadata["stagehand"]["task_id"] == "agent/sign_in"
    assert "tests/test.sh" in inbound.generated_files


def test_import_upstream_stagehand_task_writes_support_report(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "stagehand"
    task_dir = repo / "packages" / "evals" / "tasks" / "bench" / "agent"
    task_dir.mkdir(parents=True)
    (task_dir / "sign_in.ts").write_text(
        """\
import { defineBenchTask } from "../../../framework/defineTask.js";

export default defineBenchTask(
  { name: "agent/sign_in" },
  async ({ agent, v3 }) => {
    const page = v3.context.pages()[0];
    await page.goto("https://v0-modern-login-flow.vercel.app/");
    await agent.execute({ instruction: "Sign in.", maxSteps: 15 });
    const url = page.url();
    return { _success: url === "https://v0-modern-login-flow.vercel.app/authorized" };
  },
);
"""
    )
    (task_dir / "expected_answer.ts").write_text(
        """\
import { defineBenchTask } from "../../../framework/defineTask.js";
import { runWithVerifier } from "../../../framework/verifierAdapter.js";

export default defineBenchTask(
  { name: "agent/expected_answer" },
  async ({ agent, v3 }) => {
    const instruction = "Find the answer.";
    const expected = "42";
    await runWithVerifier({
      v3,
      agent,
      taskSpec: { id: "agent/expected_answer", instruction, expectedAnswer: expected },
    });
    return { _success: true };
  },
);
"""
    )
    out_dir = tmp_path / "tasks"
    report_path = tmp_path / "stagehand-support.json"
    script = (
        Path(__file__).parents[1]
        / "benchmarks"
        / "stagehand-smoke"
        / "import_upstream.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--stagehand-repo",
            str(repo),
            "--out-dir",
            str(out_dir),
            "--tasks",
            "agent/sign_in,agent/expected_answer",
            "--upstream-commit",
            "stagehand-test",
            "--support-report-out",
            str(report_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout = json.loads(result.stdout)
    assert stdout["support_report"] == str(report_path)
    report = json.loads(report_path.read_text())
    assert report["schema"] == "benchflow.stagehand-import-support.v1"
    assert report["supported_count"] == 1
    assert report["unsupported_count"] == 1
    assert report["supported"][0]["task_id"] == "agent/sign_in"
    assert report["unsupported"][0]["task_id"] == "agent/expected_answer"
    assert report["unsupported"][0]["details"]["issue"] == (
        "stagehand-expected-answer-verifier-not-mapped"
    )


def test_import_upstream_stagehand_tasks_all_discovers_official_sources(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "stagehand"
    task_dir = repo / "packages" / "evals" / "tasks" / "bench" / "agent"
    task_dir.mkdir(parents=True)
    (task_dir / "a.ts").write_text(
        """\
import { defineBenchTask } from "../../../framework/defineTask.js";
export default defineBenchTask(
  { name: "agent/a" },
  async ({ agent, v3 }) => {
    const page = v3.context.pages()[0];
    await page.goto("https://example.com/");
    await agent.execute({ instruction: "Open example.", maxSteps: 1 });
    const url = page.url();
    return { _success: url === "https://example.com/" };
  },
);
"""
    )
    (task_dir / "b.ts").write_text(
        """\
import { defineBenchTask } from "../../../framework/defineTask.js";
export default defineBenchTask(
  { name: "agent/b" },
  async () => {
    return { _success: false };
  },
);
"""
    )
    script = (
        Path(__file__).parents[1]
        / "benchmarks"
        / "stagehand-smoke"
        / "import_upstream.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--stagehand-repo",
            str(repo),
            "--out-dir",
            str(tmp_path / "tasks"),
            "--tasks",
            "all",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    report = payload["support_report"]
    assert report["tasks_requested"] == ["agent/a", "agent/b"]
    assert report["supported_count"] == 1
    assert report["unsupported_count"] == 1
    assert report["unsupported"][0]["details"]["issue"] == (
        "missing-static-instruction"
    )
