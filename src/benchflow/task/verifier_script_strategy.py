"""Script-strategy command building and reward-aggregate helpers.

Extracted from ``benchflow.task.verifier`` as a pure leaf cluster. Builds the
sandbox command (and optional chmod) for a script verifier strategy, validating
that the command stays inside the verifier directory.
"""

from __future__ import annotations

import shlex
from pathlib import PurePosixPath
from typing import Any

from benchflow.task.verifier_document import VerifierStrategy
from benchflow.task.verifier_errors import VerifierOutputParseError


def _script_strategy_command(
    strategy: VerifierStrategy,
    verifier_code_dir: PurePosixPath,
) -> str:
    command = strategy.command
    if command is None:
        raise VerifierOutputParseError(
            f"script verifier strategy {strategy.name!r} is missing command"
        )
    _script_strategy_first_token(strategy)
    return f"cd {shlex.quote(str(verifier_code_dir))} && {command}"


def _script_strategy_chmod_command(
    strategy: VerifierStrategy,
    verifier_code_dir: PurePosixPath,
) -> str | None:
    first = _script_strategy_first_token(strategy)
    if "/" not in first:
        return None
    script_path = _relative_posix_path(first, strategy=strategy)
    return f"chmod +x {shlex.quote(str(verifier_code_dir / script_path))}"


def _script_strategy_first_token(strategy: VerifierStrategy) -> str:
    command = strategy.command
    if command is None:
        raise VerifierOutputParseError(
            f"script verifier strategy {strategy.name!r} is missing command"
        )
    try:
        tokens = shlex.split(command)
    except ValueError as e:
        raise VerifierOutputParseError(
            f"script verifier strategy {strategy.name!r} has invalid command"
        ) from e
    if not tokens:
        raise VerifierOutputParseError(
            f"script verifier strategy {strategy.name!r} has empty command"
        )
    first = tokens[0]
    if first.startswith("/"):
        raise VerifierOutputParseError(
            f"script verifier strategy {strategy.name!r} command must be relative "
            "to the verifier directory"
        )
    if "/" in first:
        _relative_posix_path(first, strategy=strategy)
    return first


def _relative_posix_path(
    value: str,
    *,
    strategy: VerifierStrategy,
) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise VerifierOutputParseError(
            f"script verifier strategy {strategy.name!r} command must stay inside "
            "the verifier directory"
        )
    return path


def _has_aggregate_declaration(
    rewards: dict[str, Any],
    aggregate_policy: dict[str, Any] | None,
) -> bool:
    return isinstance(rewards.get("aggregate"), dict) or bool(aggregate_policy)
