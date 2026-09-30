"""The Prime cookbook's Verifiers taskset uses the shared RL cookbook harness.

benchflow-taskset cannot import BenchFlow (it lives next to Verifiers, whose mcp
pin conflicts with BenchFlow's), so it carries a copy of the harness settings
that evaluate.py and the TRL cookbook use. A policy trained there is evaluated
here: these tests keep the copy identical to the source of truth, without
importing the Verifiers side.
"""

from __future__ import annotations

import ast
import contextlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs" / "examples" / "rl" / "common"))
import harness  # noqa: E402

from benchflow.integrations.trl import bash_tool_schemas  # noqa: E402

PACKAGE = (
    ROOT
    / "docs"
    / "examples"
    / "rl"
    / "prime"
    / "benchflow_taskset"
    / "benchflow_taskset"
)


def constants(path: Path) -> dict[str, object]:
    values: dict[str, object] = {}
    for node in ast.parse(path.read_text()).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            with contextlib.suppress(ValueError):
                values[node.targets[0].id] = ast.literal_eval(node.value)
    return values


def test_the_prompt_and_limits_match_the_shared_harness() -> None:
    copied = constants(PACKAGE / "taskset.py")
    assert copied["HARNESS_MESSAGE"] == harness.HARNESS_MESSAGE
    assert copied["BASH_TIMEOUT_SEC"] == harness.BASH_TIMEOUT_SEC
    assert copied["MAX_OUTPUT_CHARS"] == harness.MAX_OUTPUT_CHARS
    assert copied["MAX_TURNS"] == harness.MAX_TURNS
    assert copied["SUBMIT_PATH"] == harness.harness_config().submit_path


def test_the_tools_match_the_trl_tool_schemas() -> None:
    copied = constants(PACKAGE / "tools.py")
    schemas = {
        tool["function"]["name"]: tool["function"] for tool in bash_tool_schemas()
    }
    assert copied["RUN_BASH_DESCRIPTION"] == schemas["run_bash"]["description"]
    assert (
        copied["RUN_BASH_COMMAND"]
        == schemas["run_bash"]["parameters"]["properties"]["command"]["description"]
    )
    assert copied["SUBMIT_DESCRIPTION"] == schemas["submit"]["description"]
    assert (
        copied["SUBMIT_ANSWER"]
        == schemas["submit"]["parameters"]["properties"]["answer"]["description"]
    )
    assert schemas["submit"]["parameters"]["required"] == ["answer"]


def test_the_truncation_marker_matches_the_trl_harness() -> None:
    from benchflow.integrations.trl.spec import _truncate

    copied = constants(PACKAGE / "session.py")
    assert (
        _truncate("x" * 50, 40)
        == "x" * (40 - len(copied["TRUNCATION_MARKER"])) + copied["TRUNCATION_MARKER"]
    )
