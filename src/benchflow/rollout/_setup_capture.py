"""A setup command's capture, held to the variables the operator lists.

A task's ``[[environment.setup_commands]]`` entry can archive a sandbox
folder after its command (``capture_dir``) into a variable of BenchFlow's own
environment on this machine (``capture_dir_b64_env``), and upsert it into the
dotenv file whose path another of the operator's variables holds
(``capture_dir_b64_env_file_var``). The task names both, so on 0.10.0 it could
overwrite any variable of the ``bench`` process and rewrite the file behind
any of the operator's variables.

Now the capture takes effect only for a variable the operator lists with
``--allow-task-env`` (``BENCHFLOW_ALLOW_TASK_ENV``, ``allow_task_env()`` in the
SDK), and the dotenv write only when the file's variable is listed too. A
``BENCHFLOW_*`` name is never allowed, listed or not. Otherwise the trial is
refused before anything starts, naming each variable; the capture checks
again before it writes.
"""

from __future__ import annotations

from typing import Any

from benchflow.errors import UserError


class SetupCaptureRefused(ValueError, UserError):
    """A setup command's capture would write a variable (or a dotenv file)
    the operator did not list, or a ``BENCHFLOW_*`` name: refused before
    anything starts."""

    fault = "task"


def captures(task_config: Any) -> list[tuple[int, str, str | None]]:
    """(index, variable, dotenv file variable) for each setup command that
    captures a folder."""
    sandbox = getattr(task_config, "sandbox", None)
    found = []
    for index, command in enumerate(getattr(sandbox, "setup_commands", None) or ()):
        name = getattr(command, "capture_dir_b64_env", None)
        if getattr(command, "capture_dir", None) and name:
            found.append(
                (
                    index,
                    str(name),
                    getattr(command, "capture_dir_b64_env_file_var", None),
                )
            )
    return found


def unlisted(
    index: int, name: str, file_var: str | None, allowed: frozenset[str]
) -> list[str]:
    """Why one capture can't write: each variable it names that is a
    ``BENCHFLOW_*`` name or one the operator did not list."""
    reasons = []
    for what, variable in (
        ("writes the variable", name),
        ("upserts into the dotenv file named by", file_var),
    ):
        if variable is None:
            continue
        if variable.upper().startswith("BENCHFLOW_"):
            reasons.append(
                f"setup command {index}'s capture {what} {variable}: never a "
                "BENCHFLOW_ name"
            )
        elif variable not in allowed:
            reasons.append(
                f"setup command {index}'s capture {what} {variable}: needs "
                f"--allow-task-env {variable}"
            )
    return reasons


def refusal(task_name: str, reasons: list[str]) -> SetupCaptureRefused:
    error = SetupCaptureRefused(
        f"{task_name}: a setup command's capture writes into BenchFlow's own "
        "environment on this machine, which a task can do only for a variable "
        f"the operator lists: {'; '.join(reasons)}. Nothing was started."
    )
    error.hint = (
        "list each variable with --allow-task-env NAME (BENCHFLOW_ALLOW_TASK_ENV, "
        "allow_task_env() in the SDK), or drop the capture from the task; a "
        "BENCHFLOW_ name is never allowed"
    )
    return error


def refuse_unlisted(task_config: Any, *, task_name: str) -> list[dict[str, Any]]:
    """Refuse a task whose capture the operator did not list
    (:class:`SetupCaptureRefused`); otherwise the records of the captures the
    run allows (result.json's ``isolation_opt_ins``)."""
    from benchflow.task.env import allowed_task_env

    allowed = allowed_task_env()
    found = captures(task_config)
    reasons = [r for c in found for r in unlisted(*c, allowed)]
    if reasons:
        raise refusal(task_name, reasons)
    return [
        {
            "setting": "capture_dir_b64_env",
            "task": f"setup_commands[{index}] writes {name}"
            + (f" and the dotenv file at ${file_var}" if file_var else ""),
            "run": f"--allow-task-env {name}"
            + (f" --allow-task-env {file_var}" if file_var else ""),
            "lifts": "a variable of BenchFlow's own environment"
            + (" and the operator's dotenv file" if file_var else ""),
        }
        for index, name, file_var in found
    ]


def check_before_write(name: str, file_var: str | None) -> None:
    """The capture's own check, right before it writes (the trial was
    refused before setup otherwise)."""
    from benchflow.task.env import allowed_task_env

    reasons = unlisted(0, name, file_var, allowed_task_env())
    if reasons:
        raise refusal("the setup command", reasons)
