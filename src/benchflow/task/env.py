"""Task env templates: ``${VAR}`` values filled from the operator's machine.

A task's env tables (``[environment] env`` in task.toml, ``sandbox.env``,
``[verifier] env``, ``[solution] env``, each setup command's ``env``) may
hold ``${VAR}`` or ``${VAR:-default}`` as a whole value. BenchFlow fills
such a value from the operator's environment or BenchFlow's ``.env`` only
for variables the operator names, with ``--allow-task-env NAME`` or
``BENCHFLOW_ALLOW_TASK_ENV`` (a comma list). Otherwise a task downloaded
from a registry could ask for ``ANTHROPIC_API_KEY`` or ``DAYTONA_API_KEY``
and hand it to its own setup, oracle or verifier scripts.

* A named variable is filled from the environment, then ``.env``, then
  its default; a named variable that is set nowhere and has no default
  is an error.
* An unnamed ``${VAR:-default}`` gets its default, never the operator's
  value; the launch gate warns, naming the variable, when the operator has
  it set (:func:`defaulted_task_env`).
* An unnamed ``${VAR}`` refuses the task before any sandbox starts
  (:func:`check_task_env`), naming the variable and the opt-in.

A value with other text around the template (``Bearer ${KEY}``) is
literal: it was never filled.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

from benchflow._dotenv import load_dotenv_env
from benchflow.errors import UserError

if TYPE_CHECKING:
    from benchflow.task.config import TaskConfig

_TEMPLATE_PATTERN = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")

#: The environment variable that names the variables tasks may read.
ALLOW_TASK_ENV_VAR = "BENCHFLOW_ALLOW_TASK_ENV"
#: The CLI flag that names one variable a task may read (repeatable).
ALLOW_TASK_ENV_FLAG = "--allow-task-env"

_VARIABLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Names allowed by the code that runs the job (``allow_task_env``), on top
# of BENCHFLOW_ALLOW_TASK_ENV. A context variable, so concurrent jobs in one
# process keep their own lists.
_ALLOWED_IN_CONTEXT: ContextVar[frozenset[str]] = ContextVar(
    "benchflow_allowed_task_env", default=frozenset()
)


def parse_allowed_task_env(text: str | None) -> frozenset[str]:
    """The names in a ``BENCHFLOW_ALLOW_TASK_ENV`` value (commas or spaces)."""
    if not text:
        return frozenset()
    return frozenset(name for name in re.split(r"[,\s]+", text) if name)


def allowed_task_env() -> frozenset[str]:
    """The variables the operator lets tasks read, in this context."""
    return _ALLOWED_IN_CONTEXT.get() | parse_allowed_task_env(
        os.environ.get(ALLOW_TASK_ENV_VAR)
    )


@contextmanager
def allow_task_env(names: Iterable[str]) -> Iterator[None]:
    """Let tasks read ``names`` from the operator's environment in this block.

    The SDK's form of ``--allow-task-env``; it adds to
    ``BENCHFLOW_ALLOW_TASK_ENV``. Tasks the block starts (asyncio tasks
    inherit the context) see the names too.
    """
    token = _ALLOWED_IN_CONTEXT.set(_ALLOWED_IN_CONTEXT.get() | frozenset(names))
    try:
        yield
    finally:
        _ALLOWED_IN_CONTEXT.reset(token)


def invalid_task_env_names(names: Iterable[str]) -> list[str]:
    """The entries of ``names`` that are not environment variable names."""
    return [name for name in names if not _VARIABLE_NAME.fullmatch(name)]


@dataclass(frozen=True)
class TaskEnvRequest:
    """One env value of a task that reads ``variable`` from the operator.

    A request with a ``remedy`` is not about a variable (a Compose file
    that mounts a folder outside the task, a host hook): naming a variable
    does not clear it, and the remedy says what does.
    """

    field: str
    variable: str
    default: str | None
    #: How the task reads it, when not an env table's ``field = template``.
    shown: str | None = None
    #: What clears the refusal, when naming ``variable`` does not.
    remedy: str | None = None

    @property
    def template(self) -> str:
        if self.default is None:
            return f"${{{self.variable}}}"
        return f"${{{self.variable}:-{self.default}}}"

    def describe(self) -> str:
        return self.shown or f"{self.field} = {self.template!r}"

    def hint(self) -> str:
        """How to clear this one refusal."""
        return self.remedy or refusal_reason([self.variable])


def _template(value: object) -> tuple[str, str | None] | None:
    if not isinstance(value, str):
        return None
    match = _TEMPLATE_PATTERN.fullmatch(value)
    if match is None:
        return None
    return match.group(1), match.group(2)


def env_requests(env: Mapping[str, str] | None, field: str) -> list[TaskEnvRequest]:
    """The templated values of one env table, ``field`` naming the table."""
    requests: list[TaskEnvRequest] = []
    for key, value in (env or {}).items():
        parsed = _template(value)
        if parsed is not None:
            requests.append(TaskEnvRequest(f"{field}.{key}", *parsed))
    return requests


def task_env_requests(config: TaskConfig) -> list[TaskEnvRequest]:
    """Every env value of a task config that reads the operator's environment.

    Covers each table BenchFlow fills with :func:`resolve_env_vars`:
    ``sandbox.env`` (task.toml ``[environment] env``), each
    ``sandbox.setup_commands`` entry's ``env``, ``verifier.env``,
    ``solution.env`` and each step's ``verifier.env``.
    """
    requests = env_requests(config.sandbox.env, "sandbox.env")
    for index, command in enumerate(config.sandbox.setup_commands):
        requests += env_requests(command.env, f"sandbox.setup_commands[{index}].env")
    requests += env_requests(config.verifier.env, "verifier.env")
    requests += env_requests(config.solution.env, "solution.env")
    for step in config.steps or ():
        requests += env_requests(step.verifier.env, f"steps.{step.name}.verifier.env")
    return requests


def refused_task_env(
    requests: Iterable[TaskEnvRequest], allowed: Iterable[str] | None = None
) -> list[TaskEnvRequest]:
    """The requests with no default for a variable the operator did not name."""
    names = allowed_task_env() if allowed is None else frozenset(allowed)
    return [
        r
        for r in requests
        if r.remedy is not None or (r.default is None and r.variable not in names)
    ]


def defaulted_task_env(
    requests: Iterable[TaskEnvRequest], allowed: Iterable[str] | None = None
) -> list[TaskEnvRequest]:
    """The ``${VAR:-default}`` requests that get their default although the
    operator has ``VAR`` set (in the environment or BenchFlow's ``.env``)
    without naming it.

    Before the allowlist such a task got the operator's value; now it runs
    with the default, which can quietly drop a credential the task needs
    (a token a setup command writes into a config), so the launch gate
    warns, naming the variable and never its value.
    """
    names = allowed_task_env() if allowed is None else frozenset(allowed)
    candidates = [
        r
        for r in requests
        if r.remedy is None
        and r.default is not None
        and r.variable
        and r.variable not in names
    ]
    if not candidates:
        return []
    present = set(os.environ) | set(load_dotenv_env())
    return [r for r in candidates if r.variable in present]


def defaulted_task_env_warning(
    defaulted: Mapping[str, list[TaskEnvRequest]],
) -> str | None:
    """One warning for every task that gets a default instead of a value the
    operator has set, or ``None`` when there is none."""
    tasks = sorted(task for task, found in defaulted.items() if found)
    variables = sorted({r.variable for task in tasks for r in defaulted[task]})
    if not variables:
        return None
    shown = ", ".join(tasks[:3]) + (
        f" and {len(tasks) - 3} more" if len(tasks) > 3 else ""
    )
    if len(tasks) > 1:
        who = f"tasks ({shown}) get their default"
    else:
        who = f"the task {shown}" if tasks[0] else "the task"
        who += " gets its default"
    return (
        f"{', '.join(variables)} {'is' if len(variables) == 1 else 'are'} set "
        f"in your environment or .env, but {who} instead, "
        f"since you did not name {'it' if len(variables) == 1 else 'them'}. "
        f"{refusal_reason(variables)}"
    )


def refusal_reason(variables: Iterable[str]) -> str:
    """How to name ``variables``, for a refusal message."""
    names = sorted(set(variables))
    flags = " ".join(f"{ALLOW_TASK_ENV_FLAG} {name}" for name in names)
    return (
        "BenchFlow passes a variable from your environment into a task only "
        f"when you name it: {flags}, or {ALLOW_TASK_ENV_VAR}={','.join(names)}"
    )


class TaskEnvNotAllowedError(ValueError, UserError):
    """A task asks for variables of the operator's environment nobody named."""

    def __init__(self, refused: Mapping[str, list[TaskEnvRequest]]) -> None:
        self.refused = dict(refused)
        requests = [r for found in self.refused.values() for r in found]
        variables = sorted({r.variable for r in requests if r.remedy is None})
        remedies = list(dict.fromkeys(r.remedy for r in requests if r.remedy))
        lines = [
            f"{task or 'the task'}: {r.describe()}"
            for task, found in self.refused.items()
            for r in found
        ]
        one = len(self.refused) == 1
        what = []
        if variables:
            what.append(
                f"{'asks' if one else 'ask'} for {', '.join(variables)} "
                "from your environment"
            )
        if remedies:
            what.append(f"{'reaches' if one else 'reach'} outside the task")
        self.variables = variables
        self.hint = "\n".join(
            ([refusal_reason(variables)] if variables else []) + remedies
        )
        super().__init__(
            f"refused: {'the task' if one else 'tasks'} {' and '.join(what)}\n"
            + "".join(f"- {line}\n" for line in lines)
            + self.hint
        )


def check_task_env(
    config: TaskConfig,
    *,
    task: str | None = None,
    allowed: Iterable[str] | None = None,
) -> None:
    """Refuse a task that reads a variable the operator did not name.

    Raises :class:`TaskEnvNotAllowedError`; run before any sandbox starts.
    """
    refused = refused_task_env(task_env_requests(config), allowed)
    if refused:
        raise TaskEnvNotAllowedError({task or "": refused})


def resolve_env_vars(
    env_dict: Mapping[str, str],
    *,
    allowed: Iterable[str] | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Fill a task env table's ``${VAR}`` and ``${VAR:-default}`` values.

    A variable in ``allowed`` (default :func:`allowed_task_env`) is read from
    ``os.environ``, then BenchFlow's local ``.env``, then its default. Any
    other variable gets its default. Literal values pass through unchanged.
    ``extra_env`` supplies outputs of earlier setup commands without reading
    or mutating the host environment; these phase-local values take priority.

    Raises:
        TaskEnvNotAllowedError: a ``${VAR}`` with no default names a
            variable the operator did not allow.
        ValueError: an allowed variable is set nowhere and has no default.
    """
    names = allowed_task_env() if allowed is None else frozenset(allowed)
    source_env: dict[str, str] | None = None
    resolved: dict[str, str] = {}
    refused: list[TaskEnvRequest] = []
    for key, value in env_dict.items():
        parsed = _template(value)
        if parsed is None:
            resolved[key] = value
            continue
        var_name, default = parsed
        if extra_env is not None and var_name in extra_env:
            resolved[key] = extra_env[var_name]
            continue
        if var_name in names:
            if source_env is None:
                source_env = {**load_dotenv_env(), **os.environ}
            if var_name in source_env:
                resolved[key] = source_env[var_name]
                continue
        if default is not None:
            resolved[key] = default
        elif var_name in names:
            raise ValueError(
                f"Environment variable '{var_name}' not found in host environment or .env"
            )
        else:
            refused.append(TaskEnvRequest(key, var_name, None))
    if refused:
        raise TaskEnvNotAllowedError({"": refused})
    return resolved
