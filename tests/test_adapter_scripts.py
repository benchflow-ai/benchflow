import asyncio
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from benchflow._utils.task_authoring import check_task, migrate_task_to_task_md
from benchflow.task import export_task_to_split_layout
from benchflow.task.document import TaskDocument
from benchflow.task.verifier_document import VerifierDocument


def _load_module_from_path(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_minimal_opaquetoolsbench_fixture(tmp_path: Path) -> Path:
    otb_dir = tmp_path / "OpaqueToolsBench"
    configs_dir = otb_dir / "src" / "datasets" / "bfcl" / "tool_configs"
    configs_dir.mkdir(parents=True)
    (configs_dir / "executable_simple_base_config.json").write_text(
        json.dumps(
            {
                "tests": [
                    {
                        "test_id": 0,
                        "question": "Call the function.",
                        "tools": [
                            {
                                "name": "lookup_city",
                                "description": "Look up a city.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "city": {
                                            "type": "string",
                                            "description": "City name",
                                        }
                                    },
                                    "required": ["city"],
                                },
                            }
                        ],
                        "ground_truth": ['lookup_city(city="Paris")'],
                        "name_mapping": {},
                        "execution_result_type": ["exact_match"],
                    }
                ]
            }
        )
    )
    return otb_dir


def _write_minimal_programbench_fixture(tmp_path: Path) -> Path:
    tasks_dir = tmp_path / "programbench-tasks"
    task_dir = tasks_dir / "example__tool.abcdef0"
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text(
        yaml.safe_dump(
            {
                "repository": "benchflow-ai/example-tool",
                "commit": "abcdef0123456789",
                "language": "c",
                "difficulty": "easy",
                "eval_clean_hashes": [
                    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                ],
            }
        )
    )
    (task_dir / "tests.json").write_text(
        json.dumps(
            {
                "branches": {
                    "smoke": {
                        "ignored": False,
                    }
                }
            }
        )
    )
    return tasks_dir


def _generate_native_adapter_task(
    adapter: str,
    tmp_path: Path,
) -> tuple[Path, tuple[str, ...]]:
    repo = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / f"{adapter}-native"

    if adapter == "opaquetoolsbench":
        fixture = _write_minimal_opaquetoolsbench_fixture(tmp_path)
        result = subprocess.run(
            [
                sys.executable,
                "benchmarks/opaquetoolsbench/benchflow.py",
                "--opaquetoolsbench-dir",
                str(fixture),
                "--output-dir",
                str(output_dir),
                "--task-format",
                "task-md",
            ],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return output_dir / "executable-simple-0", (
            "verifier/evaluate.py",
            "verifier/ground_truth.json",
        )

    if adapter == "programbench":
        fixture = _write_minimal_programbench_fixture(tmp_path)
        result = subprocess.run(
            [
                sys.executable,
                "benchmarks/programbench/benchflow.py",
                "--programbench-dir",
                str(fixture),
                "--output-dir",
                str(output_dir),
                "--task-format",
                "task-md",
            ],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return output_dir / "example__tool.abcdef0", (
            "verifier/verify.py",
            "verifier/tests.json",
        )

    if adapter == "hilbench":
        module = _load_module_from_path(
            repo / "benchmarks" / "hilbench" / "benchflow.py",
            "hilbench_roundtrip_converter_under_test",
        )
        task_dir = module.generate_task(
            _minimal_hilbench_task(module),
            output_dir,
            task_format="task-md",
        )
        return task_dir, (
            "verifier/verify.py",
            "oracle/solve.patch",
        )

    raise AssertionError(f"unknown adapter fixture: {adapter}")


def _write_minimal_harvey_lab_fixture(tmp_path: Path) -> tuple[Path, str]:
    harvey_dir = tmp_path / "harvey-labs"
    task_id = "corporate-ma/analyze-cim-deal-teaser/scenario-01"
    task_dir = harvey_dir / "tasks" / Path(*task_id.split("/"))
    docs_dir = task_dir / "documents"
    docs_dir.mkdir(parents=True)
    (docs_dir / "teaser.txt").write_text("TargetCo teaser facts.\n")
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "title": "Analyze CIM Deal Teaser",
                "work_type": "analyze",
                "tags": ["corporate-ma", "analysis"],
                "instructions": "Review the teaser and prepare a concise memo.",
                "deliverables": {"memo": "memo.md"},
                "criteria": [
                    {
                        "id": "c1",
                        "title": "Addresses target business",
                        "match_criteria": "The memo identifies TargetCo.",
                        "deliverables": ["memo.md"],
                    }
                ],
            }
        )
    )
    return harvey_dir, task_id


_HILBENCH_PASSING_PATCH = """\
diff --git a/test_app.py b/test_app.py
new file mode 100644
index 0000000..902a596
--- /dev/null
+++ b/test_app.py
@@ -0,0 +1,4 @@
+from app import VALUE
+
+def test_ok():
+    assert VALUE == "fixed"
"""

_HILBENCH_GOLD_PATCH = """\
diff --git a/app.py b/app.py
index 38f0c92..8aa1a7e 100644
--- a/app.py
+++ b/app.py
@@ -1 +1 @@
-VALUE = "broken"
+VALUE = "fixed"
"""


def _minimal_hilbench_task(module: ModuleType):
    return module.HILBenchTask(
        task_id="public_swe_0",
        task_type="swe",
        repo_name="example/repo",
        download_link=(
            "hf://buckets/ScaleAI/hil-bench-swe-images/images/example.tar.zst"
        ),
        problem="Fix the example repository.",
        test_patch=_HILBENCH_PASSING_PATCH,
        tests_to_pass=["test_app.py::test_ok"],
        test_files=["test_app.py"],
        ground_truth_answer=_HILBENCH_GOLD_PATCH,
        uid="example-uid",
    )


def test_harvey_lab_run_script_help_works():
    """Guards ENG-81: Harvey run script is invokable by path."""
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "benchmarks/harvey-lab/run_harvey_lab.py", "--help"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Run Harvey LAB via BenchFlow" in result.stdout
    assert "--task-format" in result.stdout


def test_harvey_lab_runner_uses_format_specific_jobs_dir() -> None:
    """Guards PR #1 against legacy/task-md Harvey LAB resume contamination."""
    repo = Path(__file__).resolve().parents[1]
    module = _load_module_from_path(
        repo / "benchmarks" / "harvey-lab" / "run_harvey_lab.py",
        "harvey_lab_runner_under_test",
    )
    jobs_dir = Path("jobs/harvey-lab-gemini-flash-lite")

    assert module._jobs_dir_for_task_format(jobs_dir, "legacy") == jobs_dir
    assert module._jobs_dir_for_task_format(jobs_dir, "task-md") == Path(
        "jobs/harvey-lab-gemini-flash-lite-task-md"
    )
    assert module._jobs_dir_for_task_format(
        Path("jobs/harvey-lab-gemini-flash-lite-task-md"),
        "task-md",
    ) == Path("jobs/harvey-lab-gemini-flash-lite-task-md")


def test_harvey_lab_converter_emits_native_task_md(tmp_path: Path) -> None:
    """Guards PR #1's Harvey LAB native LLM-judge verifier package path."""
    repo = Path(__file__).resolve().parents[1]
    module = _load_module_from_path(
        repo / "benchmarks" / "harvey-lab" / "benchflow.py",
        "harvey_lab_converter_under_test",
    )
    harvey_dir, _task_id = _write_minimal_harvey_lab_fixture(tmp_path)
    task_info = module._discover_tasks(harvey_dir)[0]

    output_dir = tmp_path / "tasks"
    task_dir = module.generate_task(task_info, output_dir, task_format="task-md")

    assert (task_dir / "task.md").exists()
    assert (task_dir / "environment" / "Dockerfile").exists()
    assert (task_dir / "environment" / "documents" / "teaser.txt").exists()
    assert not (task_dir / "environment" / "rubric.json").exists()
    assert (task_dir / "verifier" / "test.sh").exists()
    assert (task_dir / "verifier" / "evaluate.py").exists()
    assert (task_dir / "verifier" / "verifier.md").exists()
    assert (task_dir / "verifier" / "rubrics" / "rubric.json").exists()
    assert (task_dir / "verifier" / "rubrics" / "verifier.md").exists()
    assert (task_dir / "oracle" / "README.md").exists()
    assert not (task_dir / "oracle" / "solve.sh").exists()
    assert not (task_dir / "task.toml").exists()
    assert not (task_dir / "instruction.md").exists()
    assert not (task_dir / "tests").exists()
    assert not (task_dir / "solution").exists()
    assert check_task(task_dir, validation_level="publication-grade") == []

    document = TaskDocument.from_path(task_dir / "task.md")
    assert document.config.schema_version == "1.3"
    assert document.config.task is not None
    assert document.config.task.name.startswith("harvey-lab/")
    assert "Review the teaser" in document.instruction
    assert document.benchflow["oracle"]["static_solution"] is False
    assert document.benchflow["verifier"]["implementation"]["type"] == "script"

    verifier = VerifierDocument.from_verifier_dir(task_dir / "verifier")
    assert verifier.selected_strategy.command == "./test.sh"
    assert verifier.strategies["llm_judge"].rubric_path == "rubrics/rubric.json"
    assert verifier.rubric["dimensions"]["criteria_pass_rate"]["source"] == (
        verifier.default_strategy
    )
    assert verifier.outputs.reward_json == "/logs/verifier/reward.json"
    rubric = json.loads((task_dir / "verifier" / "rubrics" / "rubric.json").read_text())
    assert rubric["criteria"][0]["files"] == ["memo.md"]

    parity = subprocess.run(
        [
            sys.executable,
            "benchmarks/harvey-lab/parity_test.py",
            "--mode",
            "subset",
            "--harvey-root",
            str(harvey_dir),
            "--task-format",
            "task-md",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert parity.returncode == 0, parity.stderr

    empty_output = tmp_path / "empty-output"
    empty_output.mkdir()
    log_dir = tmp_path / "logs" / "verifier"
    smoke = subprocess.run(
        ["bash", str(task_dir / "verifier" / "test.sh")],
        cwd=repo,
        env={
            **os.environ,
            "BENCHFLOW_VERIFIER_DIR": str(task_dir / "verifier"),
            "BENCHFLOW_OUTPUT_DIR": str(empty_output),
            "BENCHFLOW_VERIFIER_LOG": str(log_dir / "verifier.log"),
            "BENCHFLOW_REWARD_TEXT": str(log_dir / "reward.txt"),
            "BENCHFLOW_REWARD_JSON": str(log_dir / "reward.json"),
            "BENCHFLOW_REWARD_DETAILS_JSON": str(log_dir / "reward-details.json"),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert smoke.returncode == 0, smoke.stderr
    assert (log_dir / "reward.txt").read_text().strip() == "0"
    assert json.loads((log_dir / "reward.json").read_text()) == {"reward": 0.0}
    assert json.loads((log_dir / "reward-details.json").read_text())["reward"] == 0.0

    judged_output = tmp_path / "judged-output"
    judged_output.mkdir()
    (judged_output / "memo.md").write_text("TargetCo is a logistics company.\n")
    judge_log_dir = tmp_path / "judge-logs" / "verifier"
    judge_smoke = subprocess.run(
        ["bash", str(task_dir / "verifier" / "test.sh")],
        cwd=repo,
        env={
            **os.environ,
            "ANTHROPIC_API_KEY": "",
            "BENCHFLOW_VERIFIER_DIR": str(task_dir / "verifier"),
            "BENCHFLOW_OUTPUT_DIR": str(judged_output),
            "BENCHFLOW_VERIFIER_LOG": str(judge_log_dir / "verifier.log"),
            "BENCHFLOW_REWARD_TEXT": str(judge_log_dir / "reward.txt"),
            "BENCHFLOW_REWARD_JSON": str(judge_log_dir / "reward.json"),
            "BENCHFLOW_REWARD_DETAILS_JSON": str(judge_log_dir / "reward-details.json"),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert judge_smoke.returncode == 0, judge_smoke.stderr
    assert (judge_log_dir / "reward.txt").read_text().strip() == "0.0"
    judge_details = json.loads((judge_log_dir / "reward-details.json").read_text())
    assert judge_details["results"][0]["verdict"] == "fail"
    assert "Judge error:" in judge_details["results"][0]["reasoning"]


def test_harvey_lab_converter_rejects_mismatched_existing_layout(
    tmp_path: Path,
) -> None:
    """Guards PR #1 against direct Harvey LAB legacy/task-md output contamination."""
    repo = Path(__file__).resolve().parents[1]
    module = _load_module_from_path(
        repo / "benchmarks" / "harvey-lab" / "benchflow.py",
        "harvey_lab_layout_guard_under_test",
    )
    harvey_dir, _task_id = _write_minimal_harvey_lab_fixture(tmp_path)
    task_info = module._discover_tasks(harvey_dir)[0]
    output_dir = tmp_path / "tasks"
    module.generate_task(task_info, output_dir, task_format="legacy")

    with pytest.raises(ValueError, match=r"already contains task\.toml"):
        module.generate_task(task_info, output_dir, task_format="task-md")


def test_programbench_run_script_help_works():
    """Guards ENG-81: ProgramBench run script is invokable by path."""
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "benchmarks/programbench/run_programbench.py", "--help"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Run ProgramBench via BenchFlow" in result.stdout
    assert "--task-format" in result.stdout
    assert "--prepare-only" in result.stdout


def _programbench_readme_run_script_flags(readme: str) -> set[str]:
    flags: set[str] = set()
    lines = readme.splitlines()
    for index, line in enumerate(lines):
        if "benchmarks/programbench/run_programbench.py" not in line:
            continue
        for command_index, command_line in enumerate(lines[index:], start=index):
            stripped = command_line.strip()
            if (
                command_index != index
                and stripped.startswith("python ")
                and "run_programbench.py" not in stripped
            ):
                break
            if not stripped:
                break
            flags.update(part for part in stripped.split() if part.startswith("--"))
    return flags


def test_programbench_readme_runner_flags_match_help() -> None:
    """Guards PR #1 against README examples naming nonexistent runner flags."""
    repo = Path(__file__).resolve().parents[1]
    readme = (repo / "benchmarks" / "programbench" / "README.md").read_text()
    documented_flags = _programbench_readme_run_script_flags(readme)
    result = subprocess.run(
        [sys.executable, "benchmarks/programbench/run_programbench.py", "--help"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert documented_flags
    for flag in documented_flags:
        assert flag in result.stdout


def test_programbench_runner_prepare_only_exits_before_eval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Guards PR #1's documented ProgramBench task-md prepare-only workflow."""
    repo = Path(__file__).resolve().parents[1]
    module = _load_module_from_path(
        repo / "benchmarks" / "programbench" / "run_programbench.py",
        "programbench_prepare_only_under_test",
    )
    prepared_dir = tmp_path / "programbench-task-md"

    def fake_ensure_converted_tasks(task_format: str = "legacy") -> Path:
        assert task_format == "task-md"
        prepared_dir.mkdir()
        return prepared_dir

    monkeypatch.setattr(module, "ensure_converted_tasks", fake_ensure_converted_tasks)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_programbench.py",
            str(tmp_path / "missing-config.yaml"),
            "--task-format",
            "task-md",
            "--prepare-only",
        ],
    )

    asyncio.run(module.main())

    assert capsys.readouterr().out.strip() == (
        f"Prepared ProgramBench task-md tasks at {prepared_dir}"
    )


def test_hilbench_run_script_help_works():
    """Guards ENG-81: HILBench run script is invokable by path."""
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "benchmarks/hilbench/run_hilbench.py", "--help"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Run HILBench via BenchFlow" in result.stdout
    assert "--task-format" in result.stdout


def test_hilbench_runner_uses_native_metadata_and_jobs_dir(tmp_path: Path) -> None:
    """Guards PR #1 against HILBench legacy/task-md resume contamination."""
    repo = Path(__file__).resolve().parents[1]
    module = _load_module_from_path(
        repo / "benchmarks" / "hilbench" / "run_hilbench.py",
        "hilbench_runner_under_test",
    )
    task_dir = tmp_path / "public-swe-0"
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "tests" / "task_metadata.json").write_text('{"layout": "legacy"}')
    (task_dir / "verifier").mkdir()
    native_meta = task_dir / "verifier" / "task_metadata.json"
    native_meta.write_text('{"layout": "native"}')
    jobs_dir = Path("jobs/hilbench-gemini-flash-lite")

    assert module._task_metadata_file(task_dir) == native_meta
    assert module._jobs_dir_for_task_format(jobs_dir, "legacy") == jobs_dir
    assert module._jobs_dir_for_task_format(jobs_dir, "task-md") == Path(
        "jobs/hilbench-gemini-flash-lite-task-md"
    )
    assert module._jobs_dir_for_task_format(
        Path("jobs/hilbench-gemini-flash-lite-task-md"),
        "task-md",
    ) == Path("jobs/hilbench-gemini-flash-lite-task-md")


def test_hilbench_converter_emits_native_task_md(tmp_path: Path) -> None:
    """Guards PR #1's HILBench task.md adapter dogfood path."""
    repo = Path(__file__).resolve().parents[1]
    module = _load_module_from_path(
        repo / "benchmarks" / "hilbench" / "benchflow.py",
        "hilbench_converter_under_test",
    )
    task = _minimal_hilbench_task(module)

    output_dir = tmp_path / "tasks"
    task_dir = module.generate_task(task, output_dir, task_format="task-md")

    assert task_dir == output_dir / "public-swe-0"
    assert (task_dir / "task.md").exists()
    assert (task_dir / "verifier" / "test.sh").exists()
    assert (task_dir / "verifier" / "verify.py").exists()
    assert (task_dir / "verifier" / "verifier.md").exists()
    assert (task_dir / "verifier" / "rubrics" / "verifier.md").exists()
    assert (task_dir / "oracle" / "solve.sh").exists()
    assert (task_dir / "oracle" / "solve.patch").exists()
    assert not (task_dir / "task.toml").exists()
    assert not (task_dir / "instruction.md").exists()
    assert check_task(task_dir, validation_level="publication-grade") == []

    document = TaskDocument.from_path(task_dir / "task.md")
    assert document.config.schema_version == "1.3"
    assert document.config.task is not None
    assert document.config.task.name == "hilbench/public-swe-0"
    assert "Fix the example repository." in document.instruction

    parity = subprocess.run(
        [
            sys.executable,
            "benchmarks/hilbench/parity_test.py",
            "--tasks-dir",
            str(output_dir),
            "--task-format",
            "task-md",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert parity.returncode == 0, parity.stderr

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text('VALUE = "broken"\n')
    log_dir = tmp_path / "logs" / "verifier"
    oracle = subprocess.run(
        ["bash", str(task_dir / "oracle" / "solve.sh")],
        cwd=repo,
        env={
            **os.environ,
            "BENCHFLOW_ORACLE_DIR": str(task_dir / "oracle"),
            "BENCHFLOW_WORKSPACE": str(workspace),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert oracle.returncode == 0, oracle.stderr

    smoke = subprocess.run(
        ["bash", str(task_dir / "verifier" / "test.sh")],
        cwd=repo,
        env={
            **os.environ,
            "BENCHFLOW_VERIFIER_DIR": str(task_dir / "verifier"),
            "BENCHFLOW_WORKSPACE": str(workspace),
            "BENCHFLOW_VERIFIER_LOG": str(log_dir / "verifier.log"),
            "BENCHFLOW_REWARD_TEXT": str(log_dir / "reward.txt"),
            "BENCHFLOW_REWARD_JSON": str(log_dir / "reward.json"),
            "BENCHFLOW_REWARD_DETAILS_JSON": str(log_dir / "reward-details.json"),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert smoke.returncode == 0, smoke.stderr
    assert (log_dir / "reward.txt").read_text().strip() == "1.000000"
    assert json.loads((log_dir / "reward.json").read_text()) == {"reward": 1.0}


def test_hilbench_converter_keeps_legacy_layout_with_oracle(tmp_path: Path) -> None:
    """Guards PR #1's HILBench legacy output while adding oracle support."""
    repo = Path(__file__).resolve().parents[1]
    module = _load_module_from_path(
        repo / "benchmarks" / "hilbench" / "benchflow.py",
        "hilbench_legacy_converter_under_test",
    )
    task = _minimal_hilbench_task(module)

    output_dir = tmp_path / "tasks"
    task_dir = module.generate_task(task, output_dir, task_format="legacy")

    assert (task_dir / "task.toml").exists()
    assert (task_dir / "instruction.md").exists()
    assert (task_dir / "tests" / "test.sh").exists()
    assert (task_dir / "tests" / "verify.py").exists()
    assert (task_dir / "solution" / "solve.sh").exists()
    assert (task_dir / "solution" / "solve.patch").exists()
    assert not (task_dir / "task.md").exists()
    assert not (task_dir / "verifier").exists()
    assert not (task_dir / "oracle").exists()
    assert check_task(task_dir) == []

    parity = subprocess.run(
        [
            sys.executable,
            "benchmarks/hilbench/parity_test.py",
            "--tasks-dir",
            str(output_dir),
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert parity.returncode == 0, parity.stderr


def test_hilbench_converter_rejects_mismatched_existing_layout(
    tmp_path: Path,
) -> None:
    """Guards PR #1 against direct HILBench legacy/task-md output contamination."""
    repo = Path(__file__).resolve().parents[1]
    module = _load_module_from_path(
        repo / "benchmarks" / "hilbench" / "benchflow.py",
        "hilbench_layout_guard_under_test",
    )
    task = _minimal_hilbench_task(module)
    output_dir = tmp_path / "tasks"
    module.generate_task(task, output_dir, task_format="legacy")

    with pytest.raises(ValueError, match=r"already contains task\.toml"):
        module.generate_task(task, output_dir, task_format="task-md")


def test_programbench_runner_uses_format_specific_jobs_dir() -> None:
    """Guards PR #1 against legacy/task-md ProgramBench resume contamination."""
    repo = Path(__file__).resolve().parents[1]
    module = _load_module_from_path(
        repo / "benchmarks" / "programbench" / "run_programbench.py",
        "programbench_runner_under_test",
    )
    jobs_dir = Path("jobs/programbench-gemini-flash-lite")

    assert module._jobs_dir_for_task_format(jobs_dir, "legacy") == jobs_dir
    assert module._jobs_dir_for_task_format(jobs_dir, "task-md") == Path(
        "jobs/programbench-gemini-flash-lite-task-md"
    )
    assert module._jobs_dir_for_task_format(
        Path("jobs/programbench-gemini-flash-lite-task-md"),
        "task-md",
    ) == Path("jobs/programbench-gemini-flash-lite-task-md")


def test_programbench_converter_emits_native_task_md(tmp_path: Path) -> None:
    """Guards PR #1's ProgramBench task.md adapter dogfood path."""
    repo = Path(__file__).resolve().parents[1]
    tasks_dir = _write_minimal_programbench_fixture(tmp_path)

    output_dir = tmp_path / "tasks"
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/programbench/benchflow.py",
            "--programbench-dir",
            str(tasks_dir),
            "--output-dir",
            str(output_dir),
            "--task-format",
            "task-md",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    task_dir = output_dir / "example__tool.abcdef0"
    assert (task_dir / "task.md").exists()
    assert (task_dir / "verifier" / "test.sh").exists()
    assert (task_dir / "verifier" / "verify.py").exists()
    assert (task_dir / "verifier" / "tests.json").exists()
    assert (task_dir / "verifier" / "verifier.md").exists()
    assert (task_dir / "verifier" / "rubrics" / "verifier.md").exists()
    assert (task_dir / "oracle" / "solve.sh").exists()
    assert not (task_dir / "task.toml").exists()
    assert not (task_dir / "instruction.md").exists()
    assert not (task_dir / "tests").exists()
    assert not (task_dir / "solution").exists()
    assert check_task(task_dir, validation_level="publication-grade") == []

    document = TaskDocument.from_path(task_dir / "task.md")
    assert document.config.schema_version == "1.3"
    assert document.config.task is not None
    assert document.config.task.name == "programbench/example__tool.abcdef0"
    assert "Compiled executable" in document.instruction

    parity = subprocess.run(
        [
            sys.executable,
            "benchmarks/programbench/parity_test.py",
            "--tasks-dir",
            str(output_dir),
            "--task-ids",
            "example__tool.abcdef0",
            "--task-format",
            "task-md",
            "--mode",
            "structural",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert parity.returncode == 0, parity.stderr + parity.stdout
    assert "All tasks passed structural validation." in parity.stdout

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    log_dir = tmp_path / "logs" / "verifier"
    smoke = subprocess.run(
        ["bash", str(task_dir / "verifier" / "test.sh")],
        cwd=repo,
        env={
            **os.environ,
            "BENCHFLOW_VERIFIER_DIR": str(task_dir / "verifier"),
            "BENCHFLOW_WORKSPACE": str(workspace),
            "BENCHFLOW_VERIFIER_LOG": str(log_dir / "verifier.log"),
            "BENCHFLOW_REWARD_TEXT": str(log_dir / "reward.txt"),
            "BENCHFLOW_REWARD_JSON": str(log_dir / "reward.json"),
            "BENCHFLOW_REWARD_DETAILS_JSON": str(log_dir / "reward-details.json"),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert smoke.returncode == 0, smoke.stderr
    assert (log_dir / "reward.txt").read_text().strip() == "0"
    assert json.loads((log_dir / "reward.json").read_text()) == {"reward": 0.0}


def test_programbench_task_md_structural_parity_requires_tests_json(
    tmp_path: Path,
) -> None:
    """Guards PR #1 against reward-zero masking of missing ProgramBench tests."""
    repo = Path(__file__).resolve().parents[1]
    module = _load_module_from_path(
        repo / "benchmarks" / "programbench" / "benchflow.py",
        "programbench_structural_parity_under_test",
    )
    task = module.load_tasks(_write_minimal_programbench_fixture(tmp_path))[0]
    output_dir = tmp_path / "tasks"
    task_dir = module.generate_task(task, output_dir, task_format="task-md")
    (task_dir / "verifier" / "tests.json").unlink()

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/programbench/parity_test.py",
            "--tasks-dir",
            str(output_dir),
            "--task-ids",
            "example__tool.abcdef0",
            "--task-format",
            "task-md",
            "--mode",
            "structural",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Missing file: verifier/tests.json" in result.stdout + result.stderr


def test_programbench_converter_keeps_legacy_layout(tmp_path: Path) -> None:
    """Guards ProgramBench legacy split output now that task-md is the default."""
    repo = Path(__file__).resolve().parents[1]
    module = _load_module_from_path(
        repo / "benchmarks" / "programbench" / "benchflow.py",
        "programbench_legacy_converter_under_test",
    )
    task = module.load_tasks(_write_minimal_programbench_fixture(tmp_path))[0]

    output_dir = tmp_path / "tasks"
    task_dir = module.generate_task(task, output_dir, task_format="legacy")

    assert (task_dir / "task.toml").exists()
    assert (task_dir / "instruction.md").exists()
    assert (task_dir / "tests" / "test.sh").exists()
    assert (task_dir / "tests" / "verify.py").exists()
    assert (task_dir / "tests" / "tests.json").exists()
    assert (task_dir / "solution" / "solve.sh").exists()
    assert not (task_dir / "task.md").exists()
    assert not (task_dir / "verifier").exists()
    assert not (task_dir / "oracle").exists()
    assert check_task(task_dir) == []


def test_programbench_converter_rejects_mismatched_existing_layout(
    tmp_path: Path,
) -> None:
    """Guards PR #1 against direct ProgramBench legacy/task-md output contamination."""
    repo = Path(__file__).resolve().parents[1]
    module = _load_module_from_path(
        repo / "benchmarks" / "programbench" / "benchflow.py",
        "programbench_layout_guard_under_test",
    )
    task = module.load_tasks(_write_minimal_programbench_fixture(tmp_path))[0]
    output_dir = tmp_path / "tasks"
    module.generate_task(task, output_dir, task_format="legacy")

    with pytest.raises(ValueError, match=r"already contains task\.toml"):
        module.generate_task(task, output_dir, task_format="task-md")


def test_opaquetoolsbench_run_script_help_works():
    """Guards ENG-89: OpaqueToolsBench run script is invokable by path."""
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/opaquetoolsbench/run_opaquetoolsbench.py",
            "--help",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Run OpaqueToolsBench via BenchFlow" in result.stdout
    assert "--task-format" in result.stdout


def test_opaquetoolsbench_converter_emits_oracle_solution(tmp_path):
    """Guards ENG-101 and commit 7eade93b's legacy verifier fallback."""
    repo = Path(__file__).resolve().parents[1]
    otb_dir = _write_minimal_opaquetoolsbench_fixture(tmp_path)

    output_dir = tmp_path / "tasks"
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/opaquetoolsbench/benchflow.py",
            "--opaquetoolsbench-dir",
            str(otb_dir),
            "--output-dir",
            str(output_dir),
            "--task-format",
            "legacy",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    solve_sh = output_dir / "executable-simple-0" / "solution" / "solve.sh"
    assert solve_sh.exists()
    assert "/app/output/response.json" in solve_sh.read_text()

    task_dir = output_dir / "executable-simple-0"
    response_path = tmp_path / "response.json"
    response_path.write_text(
        json.dumps([{"function": "lookup_city", "args": {"city": "Paris"}}])
    )
    log_dir = tmp_path / "legacy-logs" / "verifier"
    empty_verifier_dir = tmp_path / "empty-verifier"
    empty_verifier_dir.mkdir()
    fallback_smoke = subprocess.run(
        ["bash", str(task_dir / "tests" / "test.sh")],
        cwd=repo,
        env={
            **os.environ,
            "BENCHFLOW_VERIFIER_DIR": str(empty_verifier_dir),
            "BENCHFLOW_LEGACY_TESTS_DIR": str(task_dir / "tests"),
            "BENCHFLOW_RESPONSE_PATH": str(response_path),
            "BENCHFLOW_VERIFIER_LOG": str(log_dir / "verifier.log"),
            "BENCHFLOW_REWARD_TEXT": str(log_dir / "reward.txt"),
            "BENCHFLOW_REWARD_JSON": str(log_dir / "reward.json"),
            "BENCHFLOW_REWARD_DETAILS_JSON": str(log_dir / "reward-details.json"),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert fallback_smoke.returncode == 0, fallback_smoke.stderr
    assert (log_dir / "reward.txt").read_text().strip() == "1.000000"


def test_opaquetoolsbench_converter_emits_native_task_md(tmp_path):
    """Guards commit 7eade93b's task.md adapter dogfood path."""
    repo = Path(__file__).resolve().parents[1]
    otb_dir = _write_minimal_opaquetoolsbench_fixture(tmp_path)

    output_dir = tmp_path / "tasks"
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/opaquetoolsbench/benchflow.py",
            "--opaquetoolsbench-dir",
            str(otb_dir),
            "--output-dir",
            str(output_dir),
            "--task-format",
            "task-md",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    task_dir = output_dir / "executable-simple-0"
    assert (task_dir / "task.md").exists()
    assert (task_dir / "verifier" / "test.sh").exists()
    assert (task_dir / "verifier" / "verifier.md").exists()
    assert (task_dir / "verifier" / "rubrics" / "verifier.md").exists()
    assert (task_dir / "oracle" / "solve.sh").exists()
    assert not (task_dir / "task.toml").exists()
    assert not (task_dir / "instruction.md").exists()
    assert check_task(task_dir, validation_level="publication-grade") == []
    assert TaskDocument.from_path(task_dir / "task.md").config.schema_version == "1.3"

    parity = subprocess.run(
        [
            sys.executable,
            "benchmarks/opaquetoolsbench/parity_test.py",
            "--tasks-dir",
            str(output_dir),
            "--task-format",
            "task-md",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert parity.returncode == 0, parity.stderr

    response_path = tmp_path / "response.json"
    response_path.write_text(
        json.dumps([{"function": "lookup_city", "args": {"city": "Paris"}}])
    )
    log_dir = tmp_path / "logs" / "verifier"
    smoke = subprocess.run(
        ["bash", str(task_dir / "verifier" / "test.sh")],
        cwd=repo,
        env={
            **os.environ,
            "BENCHFLOW_VERIFIER_DIR": str(task_dir / "verifier"),
            "BENCHFLOW_RESPONSE_PATH": str(response_path),
            "BENCHFLOW_VERIFIER_LOG": str(log_dir / "verifier.log"),
            "BENCHFLOW_REWARD_TEXT": str(log_dir / "reward.txt"),
            "BENCHFLOW_REWARD_JSON": str(log_dir / "reward.json"),
            "BENCHFLOW_REWARD_DETAILS_JSON": str(log_dir / "reward-details.json"),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert smoke.returncode == 0, smoke.stderr
    assert (log_dir / "reward.txt").read_text().strip() == "1.000000"
    assert json.loads((log_dir / "reward.json").read_text()) == {"reward": 1.0}


@pytest.mark.parametrize(
    "adapter",
    ["opaquetoolsbench", "programbench", "hilbench"],
)
def test_adapter_task_md_export_migrate_preserves_benchmark_assets(
    tmp_path: Path,
    adapter: str,
) -> None:
    """Guards PR #1's adapter task.md -> Harbor -> task.md dogfood path."""
    task_dir, expected_native_assets = _generate_native_adapter_task(adapter, tmp_path)
    exported = tmp_path / f"{adapter}-harbor"

    export_task_to_split_layout(task_dir, exported, target="harbor")
    for asset in expected_native_assets:
        split_asset = asset.replace("verifier/", "tests/").replace(
            "oracle/", "solution/"
        )
        assert (exported / split_asset).is_file()

    migration = migrate_task_to_task_md(exported, remove_legacy=True)

    assert migration.removed_legacy is True
    assert (exported / "task.md").is_file()
    assert not (exported / "task.toml").exists()
    assert not (exported / "instruction.md").exists()
    assert not (exported / "tests").exists()
    assert not (exported / "solution").exists()
    for asset in expected_native_assets:
        assert (exported / asset).is_file()
    assert check_task(exported, validation_level="publication-grade") == []


def test_opaquetoolsbench_runner_uses_format_specific_jobs_dir() -> None:
    """Guards commit 7eade93b against legacy/task-md resume contamination."""
    repo = Path(__file__).resolve().parents[1]
    module = _load_module_from_path(
        repo / "benchmarks" / "opaquetoolsbench" / "run_opaquetoolsbench.py",
        "opaquetoolsbench_runner_under_test",
    )
    jobs_dir = Path("jobs/opaquetoolsbench-gemini-flash-lite")

    assert module._jobs_dir_for_task_format(jobs_dir, "legacy") == jobs_dir
    assert module._jobs_dir_for_task_format(jobs_dir, "task-md") == Path(
        "jobs/opaquetoolsbench-gemini-flash-lite-task-md"
    )
    assert module._jobs_dir_for_task_format(
        Path("jobs/opaquetoolsbench-gemini-flash-lite-task-md"),
        "task-md",
    ) == Path("jobs/opaquetoolsbench-gemini-flash-lite-task-md")


def test_opaquetoolsbench_converter_rejects_mismatched_existing_layout(
    tmp_path: Path,
) -> None:
    """Guards commit 7eade93b against direct legacy/task-md output contamination."""
    repo = Path(__file__).resolve().parents[1]
    module = _load_module_from_path(
        repo / "benchmarks" / "opaquetoolsbench" / "benchflow.py",
        "opaquetoolsbench_layout_guard_under_test",
    )
    task = module.load_tasks(_write_minimal_opaquetoolsbench_fixture(tmp_path))[0]
    output_dir = tmp_path / "tasks"
    module.generate_task(task, output_dir, task_format="legacy")

    with pytest.raises(ValueError, match=r"already contains task\.toml"):
        module.generate_task(task, output_dir, task_format="task-md")


def test_opaquetoolsbench_parity_reports_missing_task_name(tmp_path: Path) -> None:
    """Guards commit 7eade93b against malformed task.md parity crashes."""
    repo = Path(__file__).resolve().parents[1]
    module = _load_module_from_path(
        repo / "benchmarks" / "opaquetoolsbench" / "parity_test.py",
        "opaquetoolsbench_parity_under_test",
    )
    task_dir = tmp_path / "missing-task-name"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "environment" / "Dockerfile").write_text(
        "FROM python:3.13-slim\nRUN mkdir -p /logs/verifier\n"
    )
    verifier_dir = task_dir / "verifier"
    verifier_dir.mkdir()
    test_sh = verifier_dir / "test.sh"
    test_sh.write_text("#!/bin/bash\nexit 0\n")
    test_sh.chmod(0o755)
    (verifier_dir / "evaluate.py").write_text("print('ok')\n")
    (verifier_dir / "ground_truth.json").write_text(
        json.dumps({"ground_truth": ['lookup_city(city="Paris")']})
    )
    (verifier_dir / "verifier.md").write_text(
        """---
document_version: "0.3"
verifier:
  default_strategy: deterministic
  strategies:
    deterministic:
      type: script
      command: ./test.sh
  rubric:
    dimensions:
      function_name: {weight: 1.0, source: deterministic}
  outputs:
    reward_json: /logs/verifier/reward.json
---
"""
    )
    (verifier_dir / "rubrics").mkdir()
    (verifier_dir / "rubrics" / "verifier.md").write_text("# Rubric\n")
    oracle_dir = task_dir / "oracle"
    oracle_dir.mkdir()
    solve_sh = oracle_dir / "solve.sh"
    solve_sh.write_text("#!/bin/bash\nexit 0\n")
    solve_sh.chmod(0o755)
    (task_dir / "task.md").write_text(
        """---
schema_version: "1.3"
metadata:
  category: function-calling
agent:
  timeout_sec: 600
verifier:
  timeout_sec: 120
environment:
  cpus: 1
  memory_mb: 1024
---

## prompt

## Query

Call the function.

## Available Functions

Write to `/app/output/response.json`.
"""
    )

    errors = module._validate_task(task_dir, task_format="task-md")

    assert any("missing opaquetoolsbench/ prefix" in error for error in errors)


# Descriptor ↔ converter alignment (ENG-#369)
#
# benchmark.yaml advertises capabilities (CLI flags, has_oracle_solutions).
# These tests guard against drift between the descriptor and the actual
# converter behavior.


_BENCHMARKS_WITH_DESCRIPTOR_CLI = [
    "programbench",
    "opaquetoolsbench",
    "harvey-lab",
    "hilbench",
    "continuallearningbench",
]


@pytest.mark.parametrize("benchmark", _BENCHMARKS_WITH_DESCRIPTOR_CLI)
def test_benchmark_descriptor_advertises_task_md(benchmark: str) -> None:
    """Guards PR #1's adapted benchmark task.md descriptor contract."""
    repo = Path(__file__).resolve().parents[1]
    descriptor = yaml.safe_load(
        (repo / "benchmarks" / benchmark / "benchmark.yaml").read_text()
    )
    conversion = descriptor["conversion"]

    assert conversion["script"] == "benchflow.py"
    assert "task-md" in conversion["task_formats"]
    assert conversion["default_task_format"] in conversion["task_formats"]
    # task.md is the standard default; legacy is emitted only on request.
    assert conversion["default_task_format"] == "task-md"


@pytest.mark.parametrize("benchmark", _BENCHMARKS_WITH_DESCRIPTOR_CLI)
def test_benchflow_py_help_works(benchmark: str) -> None:
    """benchmark.yaml's `conversion.script` must expose a usable --help.

    Guards #369: ProgramBench's benchflow.py was an import-only module
    while its descriptor advertised CLI flags.
    """
    repo = Path(__file__).resolve().parents[1]
    script = repo / "benchmarks" / benchmark / "benchflow.py"
    assert script.exists(), f"missing converter: {script}"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"benchflow.py --help failed for {benchmark}: "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    # An argparse help screen always prints "usage:".
    assert "usage:" in result.stdout.lower(), (
        f"benchflow.py --help did not print argparse usage for {benchmark}: "
        f"stdout={result.stdout!r}"
    )
    assert "--task-format" in result.stdout


def test_opaquetoolsbench_descriptor_matches_oracle_emission(tmp_path: Path) -> None:
    """benchmark.yaml's has_oracle_solutions must match converter output.

    Guards #369: descriptor said False while the converter emits
    solution/solve.sh.
    """
    repo = Path(__file__).resolve().parents[1]
    descriptor = yaml.safe_load(
        (repo / "benchmarks" / "opaquetoolsbench" / "benchmark.yaml").read_text()
    )
    advertised = descriptor["conversion"]["has_oracle_solutions"]

    otb_dir = tmp_path / "OpaqueToolsBench"
    configs_dir = otb_dir / "src" / "datasets" / "bfcl" / "tool_configs"
    configs_dir.mkdir(parents=True)
    (configs_dir / "executable_simple_base_config.json").write_text(
        json.dumps(
            {
                "tests": [
                    {
                        "test_id": 0,
                        "question": "Call the function.",
                        "tools": [
                            {
                                "name": "f",
                                "description": "",
                                "parameters": {
                                    "type": "object",
                                    "properties": {},
                                    "required": [],
                                },
                            }
                        ],
                        "ground_truth": ["f()"],
                        "name_mapping": {},
                        "execution_result_type": ["exact_match"],
                    }
                ]
            }
        )
    )

    output_dir = tmp_path / "tasks"
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/opaquetoolsbench/benchflow.py",
            "--opaquetoolsbench-dir",
            str(otb_dir),
            "--output-dir",
            str(output_dir),
            "--task-format",
            "legacy",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    solve_sh = output_dir / "executable-simple-0" / "solution" / "solve.sh"
    emits_oracle = solve_sh.exists()
    assert emits_oracle == advertised, (
        f"opaquetoolsbench descriptor says has_oracle_solutions={advertised} "
        f"but converter emits oracle solve.sh={emits_oracle}"
    )
