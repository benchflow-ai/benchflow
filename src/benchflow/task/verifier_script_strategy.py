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

# How a verifier script starts: under the shell its task was written for. A
# first-line ``#!`` wins (the kernel reads it, so ``#!/usr/bin/env python3``
# runs Python); a script without one runs under bash when the image has bash,
# and under ``sh`` only when it does not. Starting the script by path alone
# left a script without a first-line shebang to the calling shell: Docker and
# Daytona-in-Docker call through ``sh -c``, which on Debian and Ubuntu images
# is dash, so SWE-bench Verified's test.sh (``#!/bin/bash`` on line 2, after
# a comment) died on ``set -o pipefail`` and every trial was an unscored
# verifier crash, the oracle's included. Daytona and Modal call through
# ``bash -c`` and ran the same script under bash. The function and its call
# sit in one brace group, so a ``cd dir && `` before them keeps the script
# from starting when the ``cd`` fails. A program the kernel runs itself, an
# ELF binary (``./bin/grader`` in a verifier.md strategy), is started by path
# too: bash would refuse it as a "binary file". The first line is read with
# the shell's own ``read``, so no tool beyond the shell is needed, and only
# POSIX constructs are used, so it runs in dash, busybox ash and bash alike.
_RUN_SCRIPT_FUNCTION = "_bf_run_script"
_RUN_SCRIPT_DEFINITION = (
    f"{_RUN_SCRIPT_FUNCTION}() {{ "
    '_bf_first=; IFS= read -r _bf_first 2>/dev/null < "$1"; '
    "case $_bf_first in "
    "'#!'*|\"$(printf '\\177')ELF\"*) \"$@\" ;; "
    '*) if command -v bash >/dev/null 2>&1; then bash "$@"; else sh "$@"; fi ;; '
    "esac; }"
)


def script_run_command(script: str, rest: str = "") -> str:
    """Shell text that runs *script* (a path with a ``/``) under its shell.

    *script* is shell text for the script's path, already quoted; *rest* is
    the shell text that follows it (arguments, ``&& more``), kept as written.
    See :data:`_RUN_SCRIPT_DEFINITION` for how the shell is chosen. The group
    closes on its own line, so a *rest* that ends in ``&``, ``;`` or a
    ``# comment`` still parses.
    """
    tail = f" {rest}" if rest else ""
    return f"{{ {_RUN_SCRIPT_DEFINITION}; {_RUN_SCRIPT_FUNCTION} {script}{tail}\n}}"


def _script_strategy_command(
    strategy: VerifierStrategy,
    verifier_code_dir: PurePosixPath,
) -> str:
    command = strategy.command
    if command is None:
        raise VerifierOutputParseError(
            f"script verifier strategy {strategy.name!r} is missing command"
        )
    first = _script_strategy_first_token(strategy)
    stripped = command.strip()
    # A command that starts a script of the package by its path (``./test.sh``,
    # which imported tasks get) runs it under its shell, like the default
    # test.sh. Only when the command's text starts with that path as written:
    # a quoted path, an assignment (``A=b/c ./x``) or anything cleverer runs
    # as the task wrote it.
    if "/" in first and "=" not in first and stripped.startswith(first):
        rest = stripped[len(first) :]
        if not rest or rest[0] in " \t":
            command = script_run_command(first, rest.strip())
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
