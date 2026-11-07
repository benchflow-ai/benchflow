"""The Python SDK examples stay runnable and on the public API.

Each script under docs/examples/python-sdk runs on Docker or Daytona. This
keeps them importable, their --help working, and their imports on
the public ``benchflow`` namespace rather than private modules.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import benchflow as bf

EXAMPLES = sorted(
    (Path(__file__).resolve().parents[1] / "docs/examples/python-sdk").glob("*.py")
)


def test_examples_exist() -> None:
    assert {p.name for p in EXAMPLES} >= {
        "run-oracle.py",
        "run-agent.py",
        "run-batch.py",
        "run-with-manifest.py",
        "run-many.py",
        "run-branch.py",
        "quickstart.py",
        "compare-jobs.py",
    }


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_uses_only_public_benchflow_names(path: Path) -> None:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "benchflow"
        ):
            assert node.module == "benchflow", f"{path.name} imports {node.module}"
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "bf"
        ):
            assert node.attr in bf.__all__, f"{path.name} uses bf.{node.attr}"


def test_the_gallery_lists_every_example() -> None:
    """docs/examples/python-sdk/README.md is the gallery index."""
    index = (EXAMPLES[0].parent / "README.md").read_text()
    missing = [p.name for p in EXAMPLES if f"`{p.name}`" not in index]
    assert missing == []


def test_the_quickstart_runs_its_offline_cells(tmp_path: Path) -> None:
    """With BF_QUICKSTART_OFFLINE=1 the quickstart runs every cell that needs
    no sandbox or credentials (reading and comparing a job on disk)."""
    import os

    env = {**os.environ, "BF_QUICKSTART_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES[0].parent / "quickstart.py")],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "offline" in proc.stdout


@pytest.mark.parametrize(
    "path",
    [p for p in EXAMPLES if p.name.startswith(("run-", "compare-"))],
    ids=lambda p: p.name,
)
def test_example_help_runs(path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(path), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    if path.name.startswith("run-"):
        assert "--sandbox" in proc.stdout


def _docker_ready() -> bool:
    import shutil

    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
    except Exception:
        return False
    return probe.returncode == 0


@pytest.mark.deterministic
def test_the_quickstart_oracle_scores_1_on_docker(tmp_path: Path) -> None:
    """The cells a newcomer runs, for real (DF-p1-first-run-1): before, every
    oracle rollout ended "Command timed out after 0 seconds", was filed as
    infra_failure, and the script still exited 0. No model credential is
    passed, so only the oracle and batch cells run."""
    import os

    if not _docker_ready():
        pytest.skip("Docker is not available")
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY"}
    }
    env.update(
        {
            "BF_SANDBOX": "docker",
            "BF_QUICKSTART_OFFLINE": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES[0].parent / "quickstart.py")],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=tmp_path,
        env=env,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out[-3000:]
    assert "timed out after 0 seconds" not in out
    assert "reward 1.0 passed True" in proc.stdout
    assert "oracle scored 1, as it should" in proc.stdout
    assert "2 of 2 passed" in proc.stdout
