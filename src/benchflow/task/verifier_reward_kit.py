"""Reward Kit strategy resolution and manifest building.

Extracted from ``benchflow.task.verifier`` as a pure leaf cluster. Resolves the
reward-kit root/runner/criteria from a verifier strategy and builds the
verifier-scoped manifest passed to Reward Kit runners. ``_safe_strategy_relative_path``
also guards relative strategy paths used elsewhere in the verifier.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

from benchflow.rewards.rubric_config import criteria_aggregate_policy_from_rubric
from benchflow.task.paths import SandboxPaths
from benchflow.task.verifier_document import VerifierDocument, VerifierStrategy
from benchflow.task.verifier_errors import VerifierOutputParseError


def _reward_kit_root(strategy: VerifierStrategy) -> PurePosixPath:
    root = strategy.root_path
    if root is None:
        raise VerifierOutputParseError(
            f"reward-kit strategy {strategy.name!r} is missing root"
        )
    return _safe_strategy_relative_path(root, strategy=strategy, field="root")


def _llm_judge_input_dir(strategy: VerifierStrategy | None, judge: Any) -> str:
    if strategy is not None and strategy.input_dir is not None:
        return strategy.input_dir
    return str(judge.input_dir)


def _reward_kit_runner(
    strategy: VerifierStrategy,
    *,
    verifier_dir: Path,
) -> str:
    root = _reward_kit_root(strategy)
    entrypoint = strategy.entrypoint or "reward.py"
    runner = root / _safe_strategy_relative_path(
        entrypoint,
        strategy=strategy,
        field="entrypoint",
    )
    local_runner = verifier_dir / Path(*runner.parts)
    if not local_runner.is_file():
        raise VerifierOutputParseError(
            f"reward-kit strategy {strategy.name!r} expected runner at {local_runner}"
        )
    return str(runner)


def _reward_kit_criteria(
    strategy: VerifierStrategy,
    *,
    verifier_dir: Path,
) -> str | None:
    criteria = strategy.criteria_path
    if criteria is None:
        return None
    path = _safe_strategy_relative_path(criteria, strategy=strategy, field="criteria")
    local_criteria = verifier_dir / Path(*path.parts)
    if not local_criteria.is_file():
        raise VerifierOutputParseError(
            f"reward-kit strategy {strategy.name!r} expected criteria at "
            f"{local_criteria}"
        )
    return str(path)


def _reward_kit_criteria_policy(
    criteria: str,
    *,
    strategy: VerifierStrategy,
    verifier_dir: Path,
) -> dict[str, Any]:
    path = _safe_strategy_relative_path(criteria, strategy=strategy, field="criteria")
    local_criteria = verifier_dir / Path(*path.parts)
    try:
        return criteria_aggregate_policy_from_rubric(local_criteria)
    except ValueError as e:
        raise VerifierOutputParseError(
            f"reward-kit strategy {strategy.name!r} has invalid criteria: {e}"
        ) from e


def _reward_kit_manifest_json(
    *,
    document: VerifierDocument,
    strategy: VerifierStrategy,
    root: PurePosixPath,
    criteria: str | None,
    criteria_policy: dict[str, Any] | None,
    sandbox_paths: SandboxPaths,
    verifier_code_dir: PurePosixPath,
) -> str:
    """Build the verifier-scoped manifest passed to Reward Kit runners."""

    criteria_path = (
        str(
            verifier_code_dir
            / _safe_strategy_relative_path(
                criteria,
                strategy=strategy,
                field="criteria",
            )
        )
        if criteria is not None
        else None
    )
    payload = {
        "version": "benchflow.reward-kit.v1",
        "verifier": {
            "name": document.name,
            "document_version": document.document_version,
            "default_strategy": document.default_strategy,
        },
        "strategy": {
            "name": strategy.name,
            "type": strategy.type,
            "root": str(root),
            "entrypoint": strategy.entrypoint or "reward.py",
            "criteria": criteria,
            "config": _jsonable(strategy.config),
        },
        "paths": {
            "verifier_dir": str(verifier_code_dir),
            "reward_kit_root": str(verifier_code_dir / root),
            "criteria": criteria_path,
            "reward_text": str(sandbox_paths.reward_text_path),
            "reward_json": str(sandbox_paths.reward_json_path),
            "reward_details_json": str(sandbox_paths.reward_details_json_path),
        },
        "rubric": _jsonable(document.rubric),
        "criteria_policy": _jsonable(criteria_policy),
        "outputs": _jsonable(asdict(document.outputs)),
    }
    return json.dumps(payload, indent=2, allow_nan=False)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _safe_strategy_relative_path(
    value: str,
    *,
    strategy: VerifierStrategy,
    field: str,
) -> PurePosixPath:
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise VerifierOutputParseError(
            f"verifier strategy {strategy.name!r} {field} must be a safe relative path"
        )
    return path
