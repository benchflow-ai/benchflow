"""Several sandbox commands in one exec, each with its own exit status.

Every exec is a round trip: on Docker a ``docker compose exec`` process (about
0.1 s), on Daytona a session plus its create, execute, status and logs
requests. Some commands BenchFlow runs one after another have the same user,
service, environment and working directory, and no step needs the host
between them. :func:`merged_script` runs such commands in one POSIX ``sh``
script: each command, unchanged, in its own subshell (an ``exit`` in it ends
only that step), its output between two marker lines and its exit status on
the second. :func:`parse_merged` splits the exec's stdout back into what each
command's own exec would have returned.

The markers carry a random tag made for each exec, so a command's output
cannot forge another step's answer without knowing it; a tag that appears
anywhere else, a missing or repeated marker, or a step out of order makes the
whole answer malformed. Callers then do what the separate execs did: a
read-only step runs its own exec again, a step with side effects is never
replayed unless it is idempotent and its answer is missing.

Budgets: a merged exec gets the smallest budget of its steps
(:func:`merged_budget`). When it finishes in time, every step finished within
its own budget; when it does not, the caller falls back to the separate execs
with their full budgets. That is only sound for read-only or idempotent steps,
which are the only ones merged with different budgets (PY8 U12).
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from benchflow.sandbox._exec_output import exec_raw

logger = logging.getLogger("benchflow")

# The marker lines start with this; the random part follows.
_TAG_PREFIX = "@@benchflow-merged-"


@dataclass(frozen=True)
class MergedStep:
    """One command of a merged exec.

    ``capture`` keeps the command's stdout in the shell variable
    ``__bf_out<i>`` (``<i>`` its position) for later steps to read, instead of
    streaming it; the step's answer is then that variable's value (trailing
    newlines removed, as ``$(...)`` does). ``gate`` stops the script after this
    step unless it exited 0, so the later steps never run. ``stderr_in_answer``
    sends the step's stderr to stdout (``2>&1`` on its subshell), so it lands
    in this step's own answer, in order: an exec whose stdout and stderr reach
    the host on separate pipes (Docker without a TTY) cannot put it inside
    another step's answer. Not with ``capture``.
    """

    command: str
    capture: bool = False
    gate: bool = False
    stderr_in_answer: bool = False

    def __post_init__(self) -> None:
        if self.capture and self.stderr_in_answer:
            raise ValueError("a captured step cannot send stderr to its answer")


@dataclass(frozen=True)
class StepAnswer:
    """What one step printed (stdout, and on Docker stderr too) and its exit
    status."""

    output: str
    return_code: int


def new_tag() -> str:
    return _TAG_PREFIX + secrets.token_hex(12)


def merged_script(steps: Sequence[MergedStep | str], tag: str) -> str:
    """The ``sh`` script that runs *steps* in order, as described above."""
    lines: list[str] = []
    for index, step in enumerate(steps):
        if isinstance(step, str):
            step = MergedStep(step)
        lines.append(f"printf '%s\\n' '{tag}:{index}:begin'")
        if step.capture:
            lines.append(f"__bf_out{index}=$(\n{step.command}\n)")
            lines.append("__bf_rc=$?")
            lines.append(f"printf '%s' \"$__bf_out{index}\"")
        else:
            redirect = " 2>&1" if step.stderr_in_answer else ""
            lines.append(f"(\n{step.command}\n){redirect}")
            lines.append("__bf_rc=$?")
        lines.append(f"printf '\\n%s:%s:end:%s\\n' '{tag}' '{index}' \"$__bf_rc\"")
        if step.gate:
            lines.append('[ "$__bf_rc" -eq 0 ] || exit "$__bf_rc"')
    return "\n".join(lines)


def parse_merged(stdout: object, tag: str, count: int) -> dict[int, StepAnswer]:
    """Each step's answer, by position, from the merged exec's stdout.

    Steps that did not run (after a gate, or when the exec died) are missing.
    A malformed answer (a marker repeated, out of order or garbled, or the tag
    anywhere outside the markers) gives an empty dict: nothing in it can be
    trusted.
    """
    if not isinstance(stdout, str) or not tag.startswith(_TAG_PREFIX):
        return {}
    answers: dict[int, StepAnswer] = {}
    position = 0
    for index in range(count):
        begin = f"{tag}:{index}:begin\n"
        start = stdout.find(begin, position)
        if start < 0:
            break
        if start != 0 and stdout[start - 1] != "\n":
            return {}
        if stdout[position:start].find(tag) >= 0:
            return {}
        body_start = start + len(begin)
        end_prefix = f"\n{tag}:{index}:end:"
        end = stdout.find(end_prefix, body_start)
        if end < 0:
            return {}
        line_end = stdout.find("\n", end + len(end_prefix))
        status = stdout[end + len(end_prefix) : line_end if line_end >= 0 else None]
        if line_end < 0 or not status.isdigit() or not status.isascii():
            return {}
        body = stdout[body_start:end]
        if tag in body:
            return {}
        answers[index] = StepAnswer(output=body, return_code=int(status))
        position = line_end + 1
    if stdout.find(tag, position) >= 0:
        return {}
    return answers


def merged_budget(*budgets: int | None) -> int | None:
    """The smallest positive budget (``None`` and non-positive ones are no
    budget, as for a single exec)."""
    bounded = [b for b in budgets if b is not None and b > 0]
    return min(bounded) if bounded else None


@dataclass(frozen=True)
class KeptIdentity:
    """The default user's working directory and uid, as the start probe saw
    them (PY8 U12).

    Docker's and Daytona's start already runs one exec as the default user
    (the log folders' mode); it now also prints that user's ``pwd`` and
    ``id -u``, so ``_resolve_agent_cwd`` and ``_verifier_uid`` need no exec of
    their own. The answers hold only for an exec like the probe's: the same
    sandbox generation (a start, a stop or a restore begins a new one), the
    same default user and the same environment. ``pwd`` and ``uid`` are the
    steps' raw output, ``None`` when that step did not answer with status 0
    or its answer is not one absolute path (``pwd``) or one number (``uid``).
    """

    generation: int
    user: str | int | None
    env_key: tuple[tuple[str, str], ...] | None
    pwd: str | None
    uid: str | None


def env_key(env: dict[str, str] | None) -> tuple[tuple[str, str], ...] | None:
    """An exec environment as a comparable key (``None`` for no variables)."""
    return tuple(sorted(env.items())) if env else None


def kept_default_identity(sandbox: Any) -> KeptIdentity | None:
    """What *sandbox*'s start probe kept, for an exec as its default user
    with no extra environment; ``None`` when there is no probe answer, it is
    from another generation, or that exec would not run as the probe's did.
    Sandboxes without the hook (other backends, test doubles) have none."""
    method = getattr(type(sandbox), "kept_identity", None)
    if method is None:
        return None
    kept = method(sandbox)
    return kept if isinstance(kept, KeptIdentity) else None


# The default user's working directory and uid, as ``_resolve_agent_cwd`` and
# ``_verifier_uid`` read them; the commands are theirs, unchanged.
PWD_COMMAND = "pwd"
UID_COMMAND = "id -u"
# Their budgets (``_resolve_agent_cwd``: 10 s; ``_verifier_uid``: the
# verifier setup budget, 180 s). The start command has none.
PWD_BUDGET_SEC = 10


def _one_absolute_path(output: str) -> bool:
    """Whether *output* is what ``pwd`` prints: one line holding an absolute
    path. Anything else (another command's stderr that reached the host
    inside this step's markers, PY8 U12-R1-B-1) is not kept."""
    line = output.rstrip("\n")
    return line.startswith("/") and "\n" not in line and "\r" not in line


def _one_number(output: str) -> bool:
    """Whether *output* is what ``id -u`` prints: one ASCII number."""
    value = output.strip()
    return value.isdigit() and value.isascii()


async def start_with_identity_probe(
    sandbox: Any,
    command: str,
    run: Callable[..., Awaitable[Any]],
    *,
    user: str | int | None,
    env: dict[str, str] | None,
) -> None:
    """Run the start's own *command* (as the default user, no budget) in one
    exec with that user's ``pwd`` and ``id -u``, and keep their answers on
    *sandbox* (``_keep_identity``) for this generation.

    *run* is the backend's exec (``run(script, timeout_sec=...)``); *user* and
    *env* are what that exec runs with, the key the answers are kept under.
    The merged exec has the smallest budget of the three (``pwd``'s 10 s); if
    it raises, or *command*'s answer is missing or malformed, *command* runs
    again on its own exactly as before (it only makes folders and sets their
    mode) and nothing is kept, so ``pwd`` and ``id -u`` run as their readers'
    own execs. ``pwd`` is a shell builtin and ``id -u`` reads the process's
    own uid with no name lookup, so neither can need more than that budget.

    The start command's stderr goes into its own answer (``stderr_in_answer``),
    which nothing reads, and ``pwd`` is kept only as one absolute path and
    ``id -u`` only as one number: on Docker the exec's stderr and stdout
    arrive on separate pipes in no fixed order, so text the backend itself
    writes to stderr can still fall between a step's markers (PY8 U12-R1-B-1).
    """
    tag = new_tag()
    script = merged_script(
        [MergedStep(command, stderr_in_answer=True), PWD_COMMAND, UID_COMMAND], tag
    )
    try:
        result = await exec_raw(run, script, timeout_sec=PWD_BUDGET_SEC)
    except Exception as exc:
        logger.debug("Start probe failed, running the start command alone: %s", exc)
        answers: dict[int, StepAnswer] = {}
    else:
        answers = parse_merged(getattr(result, "stdout", None), tag, 3)
    if 0 not in answers:
        await exec_raw(run, command)
        return
    pwd = answers.get(1)
    uid = answers.get(2)
    sandbox._keep_identity(
        user=user,
        env=env,
        pwd=pwd.output
        if pwd is not None and pwd.return_code == 0 and _one_absolute_path(pwd.output)
        else None,
        uid=uid.output
        if uid is not None and uid.return_code == 0 and _one_number(uid.output)
        else None,
    )
