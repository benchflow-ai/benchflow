"""Tests for the BenchFlow example, which lives outside this tree.

``EXAMPLE_DIR`` is put on sys.path by conftest so the modules import by bare
name, the way a rollout loads them (``--custom-agent-function-path``).
"""

from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parents[5] / "examples" / "experimental" / "benchflow"
