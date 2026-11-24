"""The grading matrix runs as a run (map-H1 (d); DESIGN.md §2.6).

``tests/conftest.py`` turns exec channels and concurrent steps off (``no_exec_channel``) for every module that
doesn't set ``DOCKER_RUN_DEFAULTS = True``, and direct Docker off (``no_direct_docker``) for every module. A real run
has all three on, and the rc4 bfx-off hang hid behind that difference. The matrix module sets the flag and turns
direct Docker back on itself; this static pin keeps both.
"""

from __future__ import annotations

import ast
from pathlib import Path

MATRIX = Path(__file__).parent / "test_grading_integrity_matrix.py"


def test_the_grading_matrix_runs_as_a_run():
    """813-U3. Mutant: the flag, or the direct-Docker fixture, removed."""
    tree = ast.parse(MATRIX.read_text())
    flag = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "DOCKER_RUN_DEFAULTS" for t in node.targets)
    ]
    assert flag and getattr(flag[0].value, "value", None) is True, (
        "DOCKER_RUN_DEFAULTS is not True"
    )
    fixtures = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any("autouse=True" in ast.unparse(d) for d in node.decorator_list)
    }
    direct = [
        name
        for name, node in fixtures.items()
        if "direct_docker" in ast.unparse(node) and "True" in ast.unparse(node)
    ]
    assert direct, (
        f"no autouse fixture turns direct Docker on (autouse fixtures: {sorted(fixtures)})"
    )
