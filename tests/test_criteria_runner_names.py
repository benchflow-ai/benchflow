"""The criteria-runner verifier strategy has a neutral name and no alias.

These tests use only APIs that exist under both names (``load_verifier_document``,
``SandboxPaths``, the authoring profile table), so they fail on behaviour
rather than on import.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path, PurePosixPath
from textwrap import dedent

import pytest

from benchflow.task._document_profiles import _TASK_AUTHORING_PROFILES
from benchflow.task.paths import SandboxPaths
from benchflow.task.verifier_document import (
    VerifierDocumentParseError,
    load_verifier_document,
)

_STRATEGIES = "agent-judge, criteria-runner, llm-judge, ors-episode, script, taskmd"


def _write_verifier_md(tmp_path: Path, strategy_type: str) -> Path:
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    (verifier / "verifier.md").write_text(
        dedent(
            f"""\
            ---
            verifier:
              default_strategy: scorer
              strategies:
                scorer:
                  type: {strategy_type}
                  root: criteria_runner
            ---
            """
        )
    )
    return verifier


def test_criteria_runner_is_a_strategy_type(tmp_path: Path) -> None:
    document = load_verifier_document(_write_verifier_md(tmp_path, "criteria-runner"))

    assert document is not None
    assert document.strategies["scorer"].type == "criteria-runner"


def test_an_unknown_strategy_type_lists_the_strategies_there_are(
    tmp_path: Path,
) -> None:
    verifier = _write_verifier_md(tmp_path, "no-such-runner")

    with pytest.raises(VerifierDocumentParseError) as raised:
        load_verifier_document(verifier)

    assert str(raised.value) == (
        "verifier.strategies.scorer.type 'no-such-runner' is an unknown value; "
        f"choose one of: {_STRATEGIES}"
    )


def test_the_criteria_runner_is_the_only_runner_strategy(tmp_path: Path) -> None:
    """No other spelling of the runner strategy is accepted (no alias)."""
    from benchflow.task.verifier_document import _KNOWN_STRATEGY_TYPES

    assert sorted(_KNOWN_STRATEGY_TYPES) == _STRATEGIES.split(", ")


def test_the_authoring_profile_is_named_criteria_runner() -> None:
    profile = _TASK_AUTHORING_PROFILES["criteria-runner"]
    verifier = profile["benchflow"]["verifier"]

    assert verifier["entrypoint"] == "verifier/criteria_runner/reward.py"
    assert verifier["implementation"] == {"type": "criteria-runner"}
    assert [name for name in _TASK_AUTHORING_PROFILES if "runner" in name] == [
        "criteria-runner"
    ]


def test_the_runner_manifest_is_named_for_the_criteria_runner() -> None:
    manifests = [
        value
        for value in dataclasses.asdict(SandboxPaths()).values()
        if str(value).endswith("-manifest.json")
    ]

    assert manifests == [PurePosixPath("/logs/verifier/criteria-runner-manifest.json")]
