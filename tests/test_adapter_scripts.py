import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


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


def test_opaquetoolsbench_converter_emits_oracle_solution(tmp_path):
    """Guards ENG-101: converted tasks can run with oracle evidence."""
    repo = Path(__file__).resolve().parents[1]
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

    output_dir = tmp_path / "tasks"
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/opaquetoolsbench/benchflow.py",
            "--opaquetoolsbench-dir",
            str(otb_dir),
            "--output-dir",
            str(output_dir),
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


# ── Descriptor ↔ converter alignment (ENG-#369) ──────────────────────
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
