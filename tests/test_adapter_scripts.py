import subprocess
import sys
from pathlib import Path


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
