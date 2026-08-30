from __future__ import annotations

import tomllib
from pathlib import Path

from benchflow.continue_run.sandbox_proxy import sandbox_replay_runtime_source
from benchflow.trajectories.redaction import canonical_redaction_module_source


def test_wheel_packages_canonical_redactor_as_data() -> None:
    """Guards PR #1057 against source-stripped installs losing the redactor."""

    root = Path(__file__).parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text())
    build = config["tool"]["hatch"]["build"]

    assert build["hooks"]["custom"] == {"path": "hatch_build.py"}
    assert "hatch_build.py" in build["targets"]["sdist"]["only-include"]
    assert (
        canonical_redaction_module_source()
        == (root / "src/benchflow/trajectories/redaction.py").read_text()
    )
    assert (
        sandbox_replay_runtime_source()
        == (root / "src/benchflow/continue_run/sandbox_replay_runtime.py").read_text()
    )
