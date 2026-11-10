"""benchflow CLI — agent benchmarking framework.

This module owns the top-level Typer ``app``, the global callback/version flag,
and the ``eval`` command group (``eval run`` / ``eval list``). ``eval run``
is defined here on purpose: tests pin its callback ``__module__`` to
``benchflow.cli.main`` and import it (plus the Daytona helpers) from here.
(``eval create`` remains as a deprecated alias of ``eval run``.)

Every other command lives in a sibling ``cli/<group>.py`` module with a
``register_<group>(app)`` function. ``app`` and ``eval_app`` list them as
:class:`~benchflow.cli._lazy.LazyCommand` entries (``_ROOT_LAYOUT`` and
``_EVAL_LAYOUT`` below), so a module is imported only when its command runs or
shows its own help: ``bench --version`` and ``bench --help`` import none of
them. For the same reason this module imports only what defining its own
commands needs; everything a command uses is imported inside it. The shared
console + display helpers live in :mod:`benchflow.cli._shared`; the names this
module used to import still resolve from it (``__getattr__`` at the bottom).

``bench --help`` lists the verb set Rust's bench 0.8.2 shares (``bench run``,
``check``, ``results``, ...) in Rust's groups, then the commands only this
bench has under "More". The 0.8 spellings keep working and print one note
naming the new one; that map, and the next step of a mistyped word, an
unknown command or an unknown flag, live in :mod:`benchflow.cli.verbs`, which
``_RootGroup`` reads.
"""

import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, NoReturn

import click
import typer
from typer.core import TyperCommand

from benchflow._utils.log_context import TaskPrefixFormatter, quiet_http_logs
from benchflow._utils.sandbox_providers import providers_phrase
from benchflow.cli._errors import install_recent_log
from benchflow.cli._lazy import (
    LazyCommand,
    LazyGroup,
    NoSuchOption,
    UsageError,
    listing_help,
)
from benchflow.cli._options import (
    AgentOption,
    AllowCloudMetadataOption,
    AllowTaskEnvOption,
    FreshInstallOption,
    HarnessOption,
    ModelOption,
    SkillModeOption,
)
from benchflow.cli.reviewer_options import (
    ReviewerAgentOption,
    ReviewerConcurrencyOption,
    ReviewerEffortOption,
    ReviewerEnvOption,
    ReviewerEvidenceTimeoutOption,
    ReviewerIdleTimeoutOption,
    ReviewerImageOption,
    ReviewerMemoryOption,
    ReviewerModelOption,
    ReviewerNetworkOption,
    ReviewerPdfToolsOption,
    ReviewerSandboxOption,
    ReviewerSetupTimeoutOption,
    ReviewerStorageOption,
    ReviewerTimeoutOption,
)
from benchflow.skill_policy import SKILL_MODE_NO_SKILL

if TYPE_CHECKING:
    from benchflow.eval_plan import EvalCreateRequest, EvalPlan
    from benchflow.evaluation import Evaluation, EvaluationConfig


# Public surface that tests and downstream callers import from
# ``benchflow.cli.main``. The Daytona helpers are defined here (not in a sibling
# module) so tests that monkeypatch ``_daytona_client_or_exit`` on ``cli.main``
# redirect the call ``_cleanup_daytona_sandboxes`` makes through this namespace.
__all__ = [
    "_cleanup_daytona_sandboxes",
    "_daytona_client_or_exit",
    "app",
    "eval_app",
    "eval_run",
    "eval_create",  # deprecated import alias of eval_run
]


class _OmittedSandboxUser(str):
    """Keep the displayed CLI default while distinguishing an omitted keyword."""

    def __str__(self) -> str:
        # Click converts defaults with str(); retain this omission sentinel.
        return self


_SANDBOX_USER_OMITTED = _OmittedSandboxUser("agent")


def _log_settings(environ: Mapping[str, str]) -> tuple[int, str]:
    """Log level and format: progress messages (INFO, message only) by default.

    ``BENCHFLOW_LOG_LEVEL`` (DEBUG, INFO, WARNING, ERROR) changes the level and
    ``BENCHFLOW_LOG_FORMAT=time`` adds a timestamp, the level and the logger
    name to each line, for CI logs.
    """
    name = environ.get("BENCHFLOW_LOG_LEVEL", "").strip().upper()
    level = logging.getLevelNamesMapping().get(name, logging.INFO)
    if environ.get("BENCHFLOW_LOG_FORMAT", "").strip().lower() == "time":
        return level, "%(asctime)s %(levelname)s %(name)s: %(message)s"
    return level, "%(message)s"


_level, _format = _log_settings(os.environ)
_handlers_before = list(logging.getLogger().handlers)
logging.basicConfig(level=_level, format=_format)
# A trial's lines start with its task ([fix-git] Running verifier...), so
# the lines of trials running at once can be told apart. Only the handler
# basicConfig added: one a host process installed keeps its own format.
for _handler in logging.getLogger().handlers:
    if _handler not in _handlers_before:
        _handler.setFormatter(TaskPrefixFormatter(_format))
# httpx logs one "HTTP Request: ..." line per request at INFO (the verifier
# runtime's PyPI fetches, the doctor's probes): DEBUG shows them, INFO not.
quiet_http_logs()
# The last log lines, for the crash log of an unexpected error (cli/_errors.py).
install_recent_log()

_TAGLINE = "The universal environment framework — run, author, and adopt agent benchmarks across any environment."

# Root command setup

# The commands other modules define, as `bench --help` and `bench eval --help`
# list them (in this order). Each module is imported when its command is used;
# tests/test_cli_lazy_commands.py checks every listed fact against the real
# command, so a changed help line there fails until it is copied here.
#
# The root lists the verb set Rust's bench 0.8.2 shares (cli/verbs.py, which a
# test keeps equal to this layout) in Rust's groups, then the commands only
# this bench has under "More". `bench new` is `bench tasks init`, `bench
# check` is `bench tasks check`, and so on: the same command under the verb's
# name (LazyCommand.source). `embodied` is hidden from the list.
_CLI = "benchflow.cli"
_TASKS, _RUNS, _MACHINE, _MORE = "Tasks", "Runs", "This machine", "More"
_ROOT_LAYOUT: tuple[LazyCommand | str, ...] = (
    LazyCommand(
        "new",
        f"{_CLI}.tasks",
        "register_tasks",
        "Scaffold a new benchmark task.",
        rich_help_panel=_TASKS,
        source=("tasks", "init"),
    ),
    LazyCommand(
        "check",
        f"{_CLI}.tasks",
        "register_tasks",
        "Validate task structure and the run's plan without starting a sandbox.",
        rich_help_panel=_TASKS,
        source=("tasks", "check"),
    ),
    LazyCommand(
        "import",
        f"{_CLI}.tasks",
        "register_tasks",
        "Convert task.toml + instruction.md into the unified task.md format.",
        rich_help_panel=_TASKS,
        source=("tasks", "migrate"),
    ),
    LazyCommand(
        "run",
        f"{_CLI}.new_verbs",
        "register_run",
        "Run an agent on tasks and score them; without -a, the oracle.",
        rich_help_panel=_RUNS,
    ),
    LazyCommand(
        "branch",
        f"{_CLI}.branch",
        "register_eval_branch",
        "Fork an agent run at a checkpoint into labelled children.",
        rich_help_panel=_RUNS,
    ),
    LazyCommand(
        "results",
        f"{_CLI}.new_verbs",
        "register_results",
        "Summarize jobs; with two, compare them.",
        rich_help_panel=_RUNS,
    ),
    "view",
    LazyCommand(
        "regrade",
        f"{_CLI}.eval_regrade",
        "register_eval_regrade",
        "Re-run the task's current verifier on each trial's frozen workspace.",
        rich_help_panel=_RUNS,
    ),
    LazyCommand(
        "export",
        f"{_CLI}.tasks",
        "register_tasks",
        "Export a task to a compatibility split layout with a loss report.",
        rich_help_panel=_RUNS,
        source=("tasks", "export"),
    ),
    LazyCommand(
        "doctor",
        f"{_CLI}.doctor",
        "register_doctor",
        "Check this machine can run evals: Python/uv, sandbox, agent credentials, network.",
        rich_help_panel=_MACHINE,
    ),
    LazyCommand(
        "agents",
        f"{_CLI}.new_verbs",
        "register_agents",
        "List the agents, or show one in detail.",
        rich_help_panel=_MACHINE,
    ),
    LazyCommand(
        "explain",
        f"{_CLI}.explain",
        "register_explain",
        "Explain why a trial has no score: what happened, whose problem, what next.",
        rich_help_panel=_MACHINE,
    ),
    LazyCommand(
        "cache",
        f"{_CLI}.cache",
        "register_cache",
        "List or clean what BenchFlow keeps: cached layers, build records, snapshots.",
        rich_help_panel=_MACHINE,
    ),
    "eval",
    LazyCommand(
        "tasks",
        f"{_CLI}.tasks",
        "register_tasks",
        "Task authoring commands.",
        rich_help_panel=_MORE,
    ),
    LazyCommand(
        "agent",
        f"{_CLI}.agent",
        "register_agent",
        "Agent management commands.",
        rich_help_panel=_MORE,
    ),
    LazyCommand(
        "review",
        f"{_CLI}.review",
        "register_review",
        "Grade finished rollouts against a rubric with a reviewer agent.",
        rich_help_panel=_MORE,
    ),
    LazyCommand(
        "assess",
        f"{_CLI}.assessment",
        "register_assessment",
        "Blind pairwise human assessments of saved outputs, kept in an opt-in sidecar.",
        rich_help_panel=_MORE,
    ),
    LazyCommand(
        "preferences",
        f"{_CLI}.preferences",
        "register_preferences",
        "Human preferences and rubric scores on finished trials.",
        rich_help_panel=_MORE,
    ),
    LazyCommand(
        "bundle",
        f"{_CLI}.eval_bundle",
        "register_eval_bundle",
        "Write one zip of a run's evidence to share, redacted.",
        rich_help_panel=_MORE,
    ),
    LazyCommand(
        "skills",
        f"{_CLI}.skills",
        "register_skills",
        "Skill discovery and evaluation.",
        rich_help_panel=_MORE,
    ),
    LazyCommand(
        "traj",
        f"{_CLI}.traj",
        "register_traj",
        "Trajectory commands.",
        rich_help_panel=_MORE,
    ),
    LazyCommand(
        "train",
        f"{_CLI}.train",
        "register_train",
        "Training data commands.",
        rich_help_panel=_MORE,
    ),
    LazyCommand(
        "hub",
        f"{_CLI}.hub",
        "register_hub",
        "Environment hubs: browse a hub's environments (list/show/inspect), "
        "check registry tasks (check), and publish datasets and push runs "
        "to a BenchFlow Hub (publish/push).",
        rich_help_panel=_MORE,
    ),
    LazyCommand(
        "daemon",
        f"{_CLI}.daemon",
        "register_daemon",
        "Manage the opt-in warm local runtime.",
        rich_help_panel=_MORE,
    ),
    LazyCommand(
        "sandbox",
        f"{_CLI}.sandbox",
        "register_sandbox",
        "Daytona sandboxes: create, list and clean up (cleanup also removes leftover Docker containers).",
        rich_help_panel=_MORE,
    ),
    LazyCommand(
        "embodied",
        f"{_CLI}.embodied",
        "register_embodied",
        "Embodied rollouts (robots and simulators): reports, evidence checks, export, specs.",
        hidden=True,
    ),
    LazyCommand(
        "continue",
        f"{_CLI}.continue_cmd",
        "register_continue",
        "Resume a previous unfinished (timed-out) OpenHands run to completion.",
        hidden=True,
        deprecated=True,
        alias=True,
    ),
    LazyCommand(
        "continue-batch",
        f"{_CLI}.continue_cmd",
        "register_continue",
        "Continue all timed-out OpenHands runs under a directory tree.",
        hidden=True,
        deprecated=True,
        alias=True,
    ),
    LazyCommand(
        "adopt",
        f"{_CLI}.adopt",
        "register_adopt_deprecated",
        "deprecated — use `bench eval adopt`.",
        hidden=True,
    ),
    LazyCommand(
        "environment",
        f"{_CLI}.environment",
        "register_environment",
        "Deprecated; use `bench sandbox` / `bench hub`.",
        hidden=True,
    ),
    LazyCommand(
        "monitor",
        f"{_CLI}.monitor",
        "register_monitor",
        "Monitor mode — score a rollout in production (#386). API surface "
        "scaffold; runtime not yet implemented.",
        hidden=True,
    ),
)
# `bench eval --help` lists commands in this order: smoke, then run, the order
# the README and getting-started use (doctor, smoke, run).
_EVAL_LAYOUT: tuple[LazyCommand | str, ...] = (
    LazyCommand(
        "smoke",
        f"{_CLI}.doctor",
        "register_eval_smoke",
        "Run a bundled hello-world task once per credentialed agent, one at a time.",
    ),
    "run",
    "create",
    # Canonical single `bench eval adopt` command (eval is the universal
    # benchmark entry point; adopt makes a foreign benchmark runnable).
    LazyCommand(
        "adopt",
        f"{_CLI}.adopt",
        "register_eval_adopt",
        "Adopt an external benchmark into benchflow.",
    ),
    LazyCommand(
        "compare-lift",
        f"{_CLI}.eval_lift",
        "register_eval_lift",
        "Compare paired baseline vs trained eval job directories.",
    ),
    LazyCommand(
        "score",
        f"{_CLI}.rescore",
        "register_eval_score",
        "Retry a failed reviewer using the saved workspace and trajectory.",
    ),
    LazyCommand(
        "regrade",
        f"{_CLI}.eval_regrade",
        "register_eval_regrade",
        "Re-run the task's current verifier on each trial's frozen workspace.",
    ),
    "list",
    "metrics",
    "view",
    LazyCommand(
        "continue",
        f"{_CLI}.continue_cmd",
        "register_continue",
        "Resume a previous unfinished (timed-out) OpenHands run to completion.",
    ),
    LazyCommand(
        "continue-batch",
        f"{_CLI}.continue_cmd",
        "register_continue",
        "Continue all timed-out OpenHands runs under a directory tree.",
        hidden=True,
    ),
    LazyCommand(
        "branch",
        f"{_CLI}.branch",
        "register_eval_branch",
        "Fork an agent run at a checkpoint into labelled children.",
    ),
    LazyCommand(
        "inspect",
        f"{_CLI}.results",
        "register_eval_results",
        "Read a finished job or trial: denominators, one row per trial, or --json.",
    ),
    LazyCommand(
        "compare",
        f"{_CLI}.results",
        "register_eval_results",
        "Pair two jobs by task: rewards, deltas, fair denominators, setting checks.",
    ),
    LazyCommand(
        "resume",
        f"{_CLI}.results",
        "register_eval_resume",
        "Finish an interrupted job from its folder: rerun only the unfinished tasks.",
    ),
    LazyCommand(
        "branches",
        f"{_CLI}.branch",
        "register_eval_branches",
        "Show every fork and child of a job's (or one trial's) branches:\n"
        "request, status, reward, advantage, tokens, USD and sandbox-seconds.",
    ),
)


class UsageErrorWithNext(UsageError):
    """A usage error (exit 2) that names its next step, kept apart for a JSON form.

    ``detail`` is printed only in the JSON form (cli/_errors.py).
    """

    def __init__(
        self,
        message: str,
        next_step: str,
        ctx: click.Context | None = None,
        *,
        detail: str | None = None,
    ) -> None:
        super().__init__(f"{message}\n  next: {next_step}", ctx=ctx)
        self.bench_message = message
        self.bench_next = next_step
        self.bench_detail = detail


def _with_next_step(exc: UsageError) -> UsageError | None:
    """``exc`` with the next step an unknown command, flag or mistyped word gets.

    None for any other usage error, and for one that already names its next
    step. The text and the step come from cli/verbs.py.
    """
    if isinstance(exc, UsageErrorWithNext):
        return None
    from benchflow.cli import verbs

    path = exc.ctx.command_path if exc.ctx is not None else "bench"
    # typer adds its "Did you mean" to an unknown command's message; click 8.4
    # adds the same again in format_message(). Say it once.
    message = exc.message if "Did you mean" in exc.message else exc.format_message()
    if isinstance(exc, NoSuchOption):
        next_step = verbs.unknown_flag_next(path, exc.option_name)
    elif message.startswith("No such command"):
        word = getattr(exc, "command_name", None)
        if word is None:  # click before 8.4 raises a plain "No such command 'x'."
            word = message.split("'")[1] if message.count("'") >= 2 else ""
        known = verbs.known_word(word)
        if known is not None and path == "bench":
            message, next_step = known
        else:
            next_step = verbs.unknown_command_next(path)
    elif isinstance(exc, click.BadParameter) and not isinstance(
        exc, click.MissingParameter
    ):
        # A value Click refused (-j abc): dogfood p5-9. A missing argument
        # keeps the pointer the JSON error gives it.
        next_step = verbs.bad_value_next(path)
    else:
        return None
    return UsageErrorWithNext(message, next_step, ctx=exc.ctx)


class _RootGroup(LazyGroup):
    """``bench``: the verb set, the 0.8 spellings' notes and maps, next steps.

    A 0.8 spelling with a new one prints one note on stderr before it runs;
    ``bench eval``, ``bench tasks`` or ``bench agent`` alone prints where its
    commands went and exits 2; an unknown command or flag, at any depth, exits
    2 with a next step. All of it is read from cli/verbs.py, imported only
    then, never for ``--version`` or ``--help``.
    """

    layout = _ROOT_LAYOUT

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        try:
            return super().parse_args(ctx, args)
        except UsageError as exc:
            raise _with_next_step(exc) or exc from None

    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str | None, click.Command | None, list[str]]:
        if args and not ctx.resilient_parsing:
            from benchflow.cli import verbs

            group = verbs.group_typed_alone(args)
            if group is not None:
                from benchflow.cli._errors import json_requested

                if json_requested(args):
                    # One JSON error on stdout, the map its detail (as Rust's).
                    raise UsageErrorWithNext(
                        f"`bench {group}` was BenchFlow 0.8's; name one of its "
                        "commands",
                        "the detail maps each 0.8 command to its spelling here",
                        ctx=ctx,
                        detail=verbs.group_map(group).rstrip("\n"),
                    )
                typer.echo(verbs.group_map(group), err=True, nl=False)
                ctx.exit(2)
            note = verbs.note_for(args)
            if note is not None and self._parses(ctx, args):
                typer.echo(f"note: {note}", err=True)
        return super().resolve_command(ctx, args)

    def _parses(self, ctx: click.Context, args: list[str]) -> bool:
        """Whether the 0.8 command line parses. One that does not (a mistyped
        flag) gets its usage error and no note: the note would carry the
        typo into the new spelling (dogfood p5-20)."""
        group = self.get_command(ctx, args[0])
        command = (
            group.get_command(ctx, args[1]) if isinstance(group, click.Group) else None
        )
        if command is None:
            return True
        try:
            # No parameter of these commands has a callback: parsing only
            # reads the words.
            command.make_context(args[1], list(args[2:]), parent=ctx).close()
        except click.ClickException:
            return False
        except Exception:
            return True
        return True

    def invoke(self, ctx: click.Context) -> Any:
        try:
            return super().invoke(ctx)
        except UsageError as exc:
            raise _with_next_step(exc) or exc from None


class _EvalGroup(LazyGroup):
    layout = _EVAL_LAYOUT

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        command = super().get_command(ctx, cmd_name)
        if not self._listing or command is None:
            return command
        from benchflow.cli import verbs

        new = verbs.new_spelling("eval", cmd_name)
        if new is None:
            return command
        # `bench eval --help` marks a command that has a new spelling: it
        # still works, with a note (dogfood p5-11).
        return TyperCommand(
            cmd_name,
            help=f"{listing_help(command)} (old name of `bench {new}`)",
            hidden=command.hidden,
            rich_help_panel=getattr(command, "rich_help_panel", None),
        )


app = typer.Typer(
    name="benchflow",
    cls=_RootGroup,
    help=_TAGLINE,
    epilog=(
        "New here? Run bench doctor, then follow "
        "https://github.com/benchflow-ai/benchflow/blob/main/docs/start/quick-start.md"
    ),
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        from benchflow import __version__

        typer.echo(f"benchflow {__version__}")
        raise typer.Exit()


def _quiet_run() -> None:
    """``--quiet``: no live progress and no per-trial log lines.

    Warnings, errors and the score block still print, so a quiet run never
    prints more than a plain one (dogfood p5-2). ``-v`` keeps its debug lines.
    """
    # Explicit off beats the session layer's auto-gate; see
    # benchflow.acp.session._console_progress_enabled.
    os.environ["BENCHFLOW_PROGRESS"] = "off"
    # The heartbeat lines and the TTY Rich dashboard (see
    # cli._live_progress.progress_enabled).
    os.environ["BENCHFLOW_NO_PROGRESS"] = "1"
    root = logging.getLogger()
    if root.level > logging.DEBUG:
        root.setLevel(logging.WARNING)


def _verbose_callback(value: bool) -> None:
    if value:
        # The same as BENCHFLOW_LOG_LEVEL=DEBUG: debug lines and the
        # tracebacks of failed trials, which the default level leaves out.
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("httpx").setLevel(logging.NOTSET)


@app.callback()
def _cli_main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            callback=_verbose_callback,
            is_eager=True,
            help="Print debug logs, with the traceback of each failed trial "
            "(the same as BENCHFLOW_LOG_LEVEL=DEBUG). Goes before the command: "
            "bench -v run ...",
        ),
    ] = False,
) -> None:
    """The universal environment framework — run, author, and adopt agent benchmarks."""


def _normalize_eval_agent_or_exit(agent_spec: str) -> str:
    from benchflow.agents.registry import is_operator_script, parse_agent_spec
    from benchflow.cli._shared import print_error
    from benchflow.eval_plan import EvalPlanError, _normalize_eval_agent

    if is_operator_script(agent_spec):
        try:
            return _normalize_eval_agent(agent_spec)
        except EvalPlanError as exc:
            print_error(str(exc), next_step="check the script path given to --agent")
            raise typer.Exit(1) from None
    protocol, canonical_agent = parse_agent_spec(agent_spec)
    if protocol not in ("acp", "acpx"):
        print_error(
            f"Unsupported eval agent protocol: {protocol}",
            next_step="name an ACP agent; `bench agents` lists them",
        )
        raise typer.Exit(1)
    if protocol == "acpx":
        return f"acpx/{canonical_agent}"
    return canonical_agent


def _daytona_client_or_exit(*, unconfigured_ok: bool = False):
    # Canonical sync-client bootstrap (anyio compat + client build) lives in
    # benchflow.sandbox.daytona; reuse it instead of re-deriving it here.
    from benchflow._utils.sandbox_providers import extra_install_hint
    from benchflow.cli._shared import print_error
    from benchflow.sandbox.daytona import build_sync_client

    try:
        return build_sync_client()
    except ModuleNotFoundError as exc:
        if exc.name == "daytona":
            print_error(
                "daytona SDK not installed",
                next_step=f"install it with `{extra_install_hint('sandbox-daytona')}`",
            )
        else:
            print_error(
                f"daytona SDK import failed: {exc}",
                next_step="`bench doctor --sandbox daytona` checks the SDK and the key",
            )
        raise typer.Exit(1) from None
    except Exception as exc:
        if unconfigured_ok and "credentials not found" in str(exc).lower():
            from benchflow.cli._shared import DaytonaUnconfigured

            raise DaytonaUnconfigured(str(exc)) from None
        print_error(
            f"Could not open a Daytona client: {exc}",
            next_step="`bench doctor --sandbox daytona` checks the SDK and the key",
        )
        raise typer.Exit(1) from None


def _cleanup_daytona_sandboxes(
    dry_run: bool,
    max_age_minutes: int,
    all_mine: bool = False,
    names: frozenset[str] = frozenset(),
) -> None:
    """Clean up orphaned Daytona sandboxes (display wrapper over the library reaper)."""
    from collections import Counter

    from benchflow.cli._shared import console
    from benchflow.sandbox.daytona import reap_stale_sandboxes
    from benchflow.sandbox.daytona_reaper import _REAP_MIN_IDLE_MIN, AUTO_STOP_MIN

    d = _daytona_client_or_exit(unconfigured_ok=True)
    _cleanup_daytona_leftovers(d, dry_run=dry_run)
    if not all_mine:
        console.print(
            "[yellow]Warning:[/yellow] a sandbox with a live run lease is kept "
            "(name it with --id to delete it). One Daytona stopped whose lease "
            "lapsed is deleted. Any other (no lease, or STARTED with a lapsed "
            f"lease) is selected by age ({max_age_minutes}m) and activity "
            f"({_REAP_MIN_IDLE_MIN}m idle) alone, which cannot tell a quiet "
            "live run from an abandoned one: use --dry-run first.",
            soft_wrap=True,
        )

    def _show(sb, age_minutes, will_delete):
        verdict = "[red](delete)[/red]" if will_delete else "[green](skip)[/green]"
        if dry_run or not will_delete:
            console.print(
                f"  [dim]{sb.id}[/dim] state={sb.state} age={age_minutes:.0f}m {verdict}"
            )

    reasons: Counter[str] = Counter()
    counts = reap_stale_sandboxes(
        d,
        max_age_minutes=max_age_minutes,
        failed_max_age_minutes=max_age_minutes,
        dry_run=dry_run,
        on_decision=_show,
        ignore_age=all_mine,
        on_skip=lambda sb, why: reasons.update([why]),
        names=names,
    )
    # The real reason for each skip: not yours, active, young.
    parts = []
    if reasons["young"]:
        parts.append(f"{reasons['young']} younger than {max_age_minutes}m")
    if reasons["active"]:
        parts.append(
            f"{reasons['active']} active in the last {_REAP_MIN_IDLE_MIN}m "
            "(use --all to delete your own that are not STARTED)"
        )
    if reasons["leased"]:
        parts.append(
            f"{reasons['leased']} held by a live run's lease "
            "(name one with --id to delete it anyway)"
        )
    if reasons["started"]:
        parts.append(
            f"{reasons['started']} STARTED (--all keeps a running sandbox; "
            "Daytona stops an abandoned one after "
            f"{AUTO_STOP_MIN} min without activity; name one with --id to "
            "delete it now)"
        )
    if reasons["kept"]:
        parts.append(
            f"{reasons['kept']} kept with --keep-sandbox "
            "(`bench sandbox release <trial>` deletes one)"
        )
    if reasons["building"]:
        parts.append(
            f"{reasons['building']} still building (deleting one now would "
            "make Daytona keep its build as failed; run cleanup again once "
            "it has started)"
        )
    if reasons["no_created_at"]:
        parts.append(f"{reasons['no_created_at']} with no creation time")
    if reasons["foreign"]:
        parts.append(f"{reasons['foreign']} another owner's (ignored)")
    detail = f" ({'; '.join(parts)})" if parts else ""
    if dry_run:
        console.print(
            f"\n[bold]{counts['found']} sandboxes found, {counts['deleted']} would be deleted[/bold]{detail} (use without --dry-run to delete)",
            soft_wrap=True,
        )
    else:
        console.print(
            f"\n[bold green]{counts['deleted']} sandboxes deleted[/bold green]{detail}"
            + (f", {counts['failed']} failed" if counts["failed"] else ""),
            soft_wrap=True,
        )
    _cleanup_daytona_snapshots(d, dry_run=dry_run, max_age_minutes=max_age_minutes)


def _cleanup_daytona_leftovers(client, *, dry_run: bool) -> None:
    """Delete what this owner scope's runs recorded as not confirmed deleted
    (the Daytona leftovers file, DAYTONA-START round 3 rule D)."""
    from benchflow.cli._shared import console
    from benchflow.sandbox.daytona_ledger import cleanup_leftovers, leftovers_path
    from benchflow.sandbox.daytona_reaper import _benchflow_managed_value

    counts = cleanup_leftovers(client, _benchflow_managed_value(), dry_run=dry_run)
    if not counts["open"] and not counts["other_owner"]:
        return
    verb = "would delete" if dry_run else "deleted"
    line = (
        f"Recorded leftovers ({leftovers_path()}): {counts['open']} open, "
        f"{counts['deleted']} {verb}"
    )
    if counts["ids"]:
        line += f" ({', '.join(counts['ids'])})"
    if counts["closed"]:
        line += f", {counts['closed']} already gone"
    if counts["kept"]:
        line += f", {counts['kept']} create attempts not found yet (kept)"
    if counts["failed"]:
        line += f", {counts['failed']} failed (kept)"
    if counts["building"]:
        line += f", {counts['building']} still building (kept until started)"
    if counts["other_owner"]:
        line += f"; {counts['other_owner']} of another owner scope left alone"
    console.print(line, soft_wrap=True)


def _cleanup_daytona_snapshots(client, *, dry_run: bool, max_age_minutes: int) -> None:
    """Reap this owner's stale branch snapshots (bf-snap2-<owner hash>-*)."""
    from benchflow.cli._shared import console, print_error
    from benchflow.sandbox.daytona import reap_stale_snapshots
    from benchflow.sandbox.daytona_reaper import owner_snapshot_prefix

    prefix = owner_snapshot_prefix()
    if prefix is None:
        console.print(
            "[dim]Branch snapshots: set BENCHFLOW_DAYTONA_OWNER to clean up "
            "your own branch snapshots (without an owner, none are touched).[/dim]"
        )
        return

    def _show(snap, age_minutes, will_delete):
        verdict = "[red](delete)[/red]" if will_delete else "[green](skip)[/green]"
        if dry_run or not will_delete:
            console.print(f"  [dim]{snap.name}[/dim] age={age_minutes:.0f}m {verdict}")

    try:
        counts = reap_stale_snapshots(
            client,
            max_age_minutes=max_age_minutes,
            dry_run=dry_run,
            on_decision=_show,
        )
    except Exception as exc:
        print_error(
            f"Skipping branch snapshot cleanup: {exc}",
            next_step="`bench doctor --sandbox daytona` checks the Daytona key",
        )
        return
    verb = "would be deleted" if dry_run else "deleted"
    console.print(
        f"[bold]{counts['deleted']} {prefix}* snapshots {verb}[/bold]"
        f" (older than {max_age_minutes}m; {counts['failed']} failed)"
    )


# Evaluation execution


eval_app = typer.Typer(
    cls=_EvalGroup,
    help="Run agents on tasks (smoke, run) and read the results (metrics, view, inspect, compare).",
)
app.add_typer(eval_app, name="eval", rich_help_panel=_MORE)


def _print_dry_run(plan: Any, tasks_dir: Path | None) -> None:
    """``--dry-run``: the plan the run would run, and its tasks; nothing runs.

    Read from the Evaluation the run builds (a ``--config`` file with the
    command line's overrides, or the task folder), and its tasks from the
    run's own discovery, so a single task folder is one task and a config's
    agent, sandbox and concurrency are the config's (dogfood p5-2).
    """
    import json as _json

    from benchflow.cli._errors import expected_error_parts
    from benchflow.cli._shared import console, print_error

    req = plan.request
    evaluation = None
    refusal: Exception | None = None
    if req.config_file is not None:
        job_plan = _job_plan(plan)
        if job_plan is not None and job_plan.is_matrix:
            _print_job_matrix_dry_run(plan, job_plan)
            return
        evaluation = _config_file_evaluation(plan)
    elif tasks_dir is not None and Path(tasks_dir).is_dir():
        from benchflow.evaluation import Evaluation
        from benchflow.task.discovery import resolve_task_collection_root

        evaluation = Evaluation(
            tasks_dir=str(resolve_task_collection_root(Path(tasks_dir))),
            jobs_dir=plan.output_jobs_dir,
            config=plan.make_eval_config(),
            job_name=plan.job_name,
            preflight=False,
        )
    if evaluation is not None:
        config = evaluation._config
        out: dict[str, Any] = {
            "agent": config.agent,
            "model": config.model,
            "environment": config.environment,
            "concurrency": config.concurrency,
            "seeds": config.seeds,
            "include": sorted(config.include_tasks),
            "exclude": sorted(config.exclude_tasks),
            "n_tasks": config.n_tasks,
            "sample_seed": config.sample_seed,
            "timeout_multiplier": config.timeout_multiplier,
            "timeout_multipliers": (config.config_override or {}).get(
                "timeout_multipliers"
            ),
            "extra_instruction": config.extra_instruction,
            "verifier_sandbox": config.verifier_sandbox or "task",
            "dataset": req.dataset,
            "jobs_dir": str(evaluation._jobs_dir),
        }
        # The job when it is known now: named, or the one a rerun resumes (a
        # new one is named by the time it starts).
        job = evaluation._job_name
        named = plan.job_name or evaluation._auto_job_name is None
        if named or (
            (evaluation._jobs_dir / job).is_dir()
            and not evaluation._unselected_results(evaluation._jobs_dir / job)
        ):
            out["job"] = job
        try:
            chosen = evaluation._get_task_dirs()
        except (OSError, ValueError) as exc:
            err_text, err_hint = expected_error_parts(exc)
            print_error(err_text, next_step=err_hint or "fix the task folder")
            raise typer.Exit(1) from None
        out["tasks"] = [d.name for d in chosen]
        # --trials N runs every task N times, each trial a job of its own.
        trials = max(1, int(getattr(req, "trials", 1) or 1))
        out["rollouts"] = len(chosen) * trials
        from benchflow.run_card import RunCardError, build_run_card, check_card

        evaluation._config.planned_trials = trials
        try:
            card = build_run_card(evaluation, chosen)
        except RunCardError as exc:
            err_text, err_hint = expected_error_parts(exc)
            print_error(err_text, next_step=err_hint)
            raise typer.Exit(1) from None
        # The plan, not when it was printed: two dry runs of one plan match.
        card.pop("written_at", None)
        out["run_card"] = card
        try:
            check_card(card)
        except RunCardError as exc:
            out["run_card_refusal"] = str(exc)
            refusal = exc
    else:
        out = {
            "agent": plan.eval_agent,
            "model": req.model,
            "environment": plan.eval_environment,
            "concurrency": plan.eval_concurrency,
            "seeds": plan.eval_seeds,
            "include": sorted(plan.include_tasks),
            "exclude": sorted(plan.exclude_tasks),
            "n_tasks": req.n_tasks,
            "sample_seed": req.sample_seed,
            "timeout_multiplier": req.timeout_multiplier,
            "extra_instruction": req.extra_instruction,
            "verifier_sandbox": req.verifier_sandbox or "task",
            "dataset": req.dataset,
            "jobs_dir": plan.output_jobs_dir,
            # A repository or dataset target is fetched when the run starts:
            # its card, and a strict reference's refusal, come then.
            "run_card": None,
            "run_card_note": "built when the run fetches the tasks (a repository "
            "or dataset target); a dry run of a local folder or run config "
            "prints it",
        }
    if req.matrix is not None or req.trials > 1:
        # --trials and --matrix repeat every task; without them in the plan a
        # 3-trial run of 4 tasks planned "rollouts": 4 and ran 12.
        out["trials"] = req.trials
        entries = 1
        if req.matrix is not None:
            from benchflow.cli.eval_artifacts import _load_eval_matrix

            try:
                models = _load_eval_matrix(req.matrix)
            except (OSError, ValueError) as exc:
                print_error(
                    str(exc),
                    next_step="fix the --matrix file; docs/reference/pass-at-k.md shows one",
                )
                raise typer.Exit(1) from None
            out["matrix"] = {
                alias: {
                    "agent": str(raw.get("agent") or plan.eval_agent),
                    "model": raw.get("model"),
                }
                for alias, raw in models.items()
            }
            entries = len(models)
        task_names = out.get("tasks")
        if isinstance(task_names, list):
            out["rollouts"] = len(task_names) * req.trials * entries
    version_issue = None
    if req.dataset:
        from benchflow._utils.dataset_registry import (
            DatasetResolutionError,
            bench_version_issue,
            dataset_registry_entry,
        )

        try:
            entry = dataset_registry_entry(req.dataset, registry=req.registry or None)
        except DatasetResolutionError as exc:
            err_text, err_hint = expected_error_parts(exc)
            print_error(err_text, next_step=err_hint)
            raise typer.Exit(1) from None
        pin = entry.get("bench_version")
        version_issue = bench_version_issue(pin)
        out["bench_version"] = pin
        out["bench_version_issue"] = version_issue
        out["ignore_bench_version"] = req.ignore_bench_version
    console.print_json(_json.dumps(out, default=str))
    card = out.get("run_card")
    if card is None and "run_card_note" in out:
        from benchflow.cli._shared import err_console

        err_console.print(
            "Run card: none yet. A repository or dataset target is fetched when the "
            "run starts; its card, and a strict reference's refusal, come then.",
            markup=False,
            highlight=False,
            soft_wrap=True,
        )
    if isinstance(card, dict):
        # The card as a person reads it, on stderr: the plan above stays JSON.
        from benchflow.cli._shared import err_console
        from benchflow.run_card import card_lines, reference_findings

        lines = card_lines(card, Path(out["jobs_dir"]))
        lines[0] = "Run card (a dry run: nothing ran, and no job folder was made):"
        findings = reference_findings(card)
        if findings:
            lines += ["The run differs from its reference setup:"]
            lines += [f"  {finding}" for finding in findings]
        for line in lines:
            err_console.print(line, markup=False, highlight=False, soft_wrap=True)
    if version_issue and not req.ignore_bench_version:
        _refuse_dataset_bench_version(version_issue)
    if refusal is not None:
        # The real run would stop here, so the dry run exits as it would.
        err_text, err_hint = expected_error_parts(refusal)
        print_error(err_text, next_step=err_hint)
        raise typer.Exit(1)


def _print_job_matrix_dry_run(plan: Any, job_plan: Any) -> None:
    """``--dry-run`` of a YAML job config of several jobs: each cell's agent,
    model and tasks, its trials, and the rollouts in all."""
    import json as _json

    from benchflow.cli._shared import console

    root = _job_matrix_root(plan, job_plan)
    jobs: list[dict[str, Any]] = []
    # Refuses a cell with no task, as the run does (review L-A3).
    evaluations = _job_cell_evaluations(plan, job_plan, root)
    for cell in job_plan.cells:
        evaluation = evaluations[cell.alias]
        config = evaluation._config
        chosen = evaluation._get_task_dirs()
        jobs.append(
            {
                "cell": cell.alias,
                "jobs_dir": str(root / cell.alias),
                "agent": config.agent,
                "model": config.model,
                "dataset": cell.dataset,
                "tasks_dir": str(evaluation._tasks_dir),
                "tasks": [d.name for d in chosen],
                "include": sorted(config.include_tasks),
                "exclude": sorted(config.exclude_tasks),
                "n_tasks": config.n_tasks,
                "environment": config.environment,
                "concurrency": config.concurrency,
                "timeout_multiplier": config.timeout_multiplier,
                "timeout_multipliers": (config.config_override or {}).get(
                    "timeout_multipliers"
                ),
                "extra_instruction": config.extra_instruction,
                "trials": job_plan.attempts,
                "rollouts": len(chosen) * job_plan.attempts,
            }
        )
    out = {
        "job_matrix": job_plan.describe(),
        "jobs_dir": str(root),
        "trials": job_plan.attempts,
        "jobs": jobs,
        "trial_jobs": len(jobs) * job_plan.attempts,
        "rollouts": sum(job["rollouts"] for job in jobs),
    }
    console.print_json(_json.dumps(out, default=str))


def _refuse_dataset_bench_version(version_issue: str) -> None:
    """One message and next step for a run and a dry run outside the pin."""
    from benchflow.cli._shared import print_error

    print_error(
        version_issue,
        next_step="pick a dataset version validated for this version of "
        "BenchFlow, or re-run with --ignore-bench-version to proceed anyway",
    )
    raise typer.Exit(1)


@eval_app.command("run")
def eval_run(
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "--run-config",
            help="YAML run-config file (the whole run spec)",
        ),
    ] = None,
    tasks_dir: Annotated[
        Path | None,
        typer.Option("--tasks-dir", help="Local tasks directory"),
    ] = None,
    source_repo: Annotated[
        str | None,
        typer.Option(
            "--source-repo",
            help="GitHub repo of the tasks as org/repo (e.g. benchflow-ai/skillsbench)",
        ),
    ] = None,
    source_path: Annotated[
        str | None,
        typer.Option(
            "--source-path",
            help=(
                "Folder in the repo: one task (tasks/econ-detrending-correlation) or a folder "
                "of tasks (tasks). Only this folder is downloaded"
            ),
        ),
    ] = None,
    source_ref: Annotated[
        str | None,
        typer.Option(
            "--source-ref", help="Branch, tag or full commit SHA to fetch (e.g. main)"
        ),
    ] = None,
    source_env: Annotated[
        str | None,
        typer.Option(
            "--source-env",
            help="Hosted environment source (e.g. primeintellect/general-agent)",
        ),
    ] = None,
    source_env_version: Annotated[
        str | None,
        typer.Option("--source-env-version", help="Hosted environment version"),
    ] = None,
    source_env_arg: Annotated[
        list[str] | None,
        typer.Option(
            "--source-env-arg",
            help="Hosted environment arg as KEY=VALUE; repeatable",
        ),
    ] = None,
    source_env_num_examples: Annotated[
        int,
        typer.Option("--source-env-num-examples", help="Number of env examples"),
    ] = 1,
    source_env_rollouts_per_example: Annotated[
        int,
        typer.Option(
            "--source-env-rollouts-per-example",
            help="Rollouts per hosted env example",
        ),
    ] = 1,
    source_env_max_tokens: Annotated[
        int,
        typer.Option("--source-env-max-tokens", help="Max tokens for hosted env run"),
    ] = 1024,
    source_env_temperature: Annotated[
        float | None,
        typer.Option(
            "--source-env-temperature",
            help="Temperature for hosted env run (default: none sent, so the "
            "provider's own)",
        ),
    ] = None,
    source_env_sampling_arg: Annotated[
        list[str] | None,
        typer.Option(
            "--source-env-sampling-arg",
            help="Hosted env sampling arg as KEY=VALUE; repeatable (e.g. reasoning_effort=minimal)",
        ),
    ] = None,
    source_env_verifiers_version: Annotated[
        str | None,
        typer.Option(
            "--source-env-verifiers-version",
            help=(
                "Pin the verifiers release installed with the hosted env "
                "(default: the env package's own requirement)"
            ),
        ),
    ] = None,
    source_env_base_url: Annotated[
        str | None,
        typer.Option(
            "--source-env-base-url",
            help=(
                "Model endpoint for the hosted env run (default: verifiers' "
                "own, Prime inference with PRIME_API_KEY)"
            ),
        ),
    ] = None,
    source_env_api_key_var: Annotated[
        str | None,
        typer.Option(
            "--source-env-api-key-var",
            help="Name of the environment variable holding that endpoint's API key",
        ),
    ] = None,
    agent: Annotated[
        str | None,
        typer.Option(
            "--agent",
            "-a",
            help=(
                "Agent: claude, codex, gemini, ... (bench agents), oracle (the "
                "task's reference solution) or nop (does nothing)"
            ),
        ),
    ] = None,
    model: ModelOption = None,
    reasoning_effort: Annotated[
        str | None,
        typer.Option(
            "--reasoning-effort",
            help="Agent reasoning/thinking effort when the agent exposes one (e.g. max)",
        ),
    ] = None,
    harness: HarnessOption = None,
    environment: Annotated[
        str | None,
        typer.Option("--sandbox", "-s", help=f"Sandbox: {providers_phrase()}"),
    ] = None,
    usage_tracking: Annotated[
        str | None,
        typer.Option(
            "--usage-tracking",
            help=(
                "Telemetry-enforcement policy: auto, required, or off. The "
                "LiteLLM proxy is always used for routable agents (usage, cost, "
                "and llm_trajectory.jsonl are always captured); this flag only "
                "controls whether trusted telemetry is required."
            ),
        ),
    ] = None,
    environment_manifest: Annotated[
        Path | None,
        typer.Option(
            "--environment-manifest",
            help=(
                "Environment-plane manifest applied to every rollout: a path to "
                "an environment.toml, OR a 'name@version' registry spec "
                "resolved via $BENCHFLOW_ENV_REGISTRY when set, else the "
                "built-in registry shipped with benchflow (env0@prod, "
                "env0@outage). The manifest-declared stateful environment is "
                "provisioned, gated on readiness, and torn down."
            ),
        ),
    ] = None,
    state: Annotated[
        str | None,
        typer.Option(
            "--state",
            help=(
                "The stateful environment to bind, apart from the task. Inline JSON "
                'with an optional tool subset (e.g. {"name": "env0", "tools": '
                '["gmail", "gcal"]}), OR a name@version spec, OR a manifest path. '
                "Takes precedence over --environment-manifest."
            ),
        ),
    ] = None,
    prompt: Annotated[
        list[str] | None,
        typer.Option(
            "--prompt",
            help="Prompt(s) to send (default: the task prompt: the task.md body, or instruction.md for older tasks)",
        ),
    ] = None,
    config_override: Annotated[
        str | None,
        typer.Option(
            "--config-override",
            help=(
                "Deep-merge a config patch into each task's resolved config for "
                "this run. Inline JSON/YAML/TOML or an @file ref, e.g. "
                '--config-override \'{"agent":{"timeout_sec":120}}\'. Varies one '
                "knob (budget, skills, stopping rules) while the tasks, agent, "
                "model and environment stay fixed; recorded by content hash for "
                "replay."
            ),
        ),
    ] = None,
    concurrency: Annotated[
        int | None,
        typer.Option(
            "--concurrency",
            "-j",
            help=(
                "Trials running at a time. On Docker a job also prepares up to "
                "--warm-pool sandboxes ahead (default 1 when this is 2 or more)"
            ),
        ),
    ] = None,
    build_concurrency: Annotated[
        int | None,
        typer.Option(
            "--build-concurrency",
            help=(
                "Max concurrent docker image builds. Defaults to --concurrency. "
                "Set lower (e.g. 8) when --concurrency is high to avoid "
                "overwhelming the docker daemon with parallel builds."
            ),
        ),
    ] = None,
    warm_pool: Annotated[
        int | None,
        typer.Option(
            "--warm-pool",
            help=(
                "Docker: how many sandboxes to prepare ahead of their trial's turn, "
                "beyond the --concurrency running ones (0 to --concurrency; "
                "default 1, or 0 when --concurrency is 1)"
            ),
        ),
    ] = None,
    worker_concurrency: Annotated[
        int | None,
        typer.Option(
            "--worker-concurrency",
            help=(
                "Run batch eval through isolated worker subprocesses, each with "
                "at most this many concurrent tasks; --concurrency remains the "
                "aggregate target."
            ),
        ),
    ] = None,
    worker_retries: Annotated[
        int,
        typer.Option(
            "--worker-retries",
            help="Retry a crashed worker shard this many times, resuming its jobs_dir.",
        ),
    ] = 1,
    worker_start_stagger_sec: Annotated[
        float,
        typer.Option(
            "--worker-start-stagger-sec",
            help="Seconds to stagger worker starts to avoid Daytona connection storms.",
        ),
    ] = 1.0,
    agent_idle_timeout: Annotated[
        str | None,
        typer.Option(
            "--agent-idle-timeout",
            "--idle-timeout",
            help=(
                "Abort a prompt when the agent is silent this long: seconds, or "
                "90s, 10m, 1h; 0, none or null turns it off and leaves the "
                "task's wall-clock timeout (default: 10m, 30m at effort xhigh "
                "or max)"
            ),
        ),
    ] = None,
    checkpoints: Annotated[
        str | None,
        typer.Option(
            "--checkpoints",
            help=(
                "Keep a sandbox snapshot after these prompts: every-prompt or "
                "prompt:N[,M]. Later `bench branch --from-checkpoint "
                "<trial>` forks from one. Docker, or Daytona direct; each is a "
                "full filesystem image. See docs/train/environment.md"
            ),
        ),
    ] = None,
    checkpoint_keep: Annotated[
        int | None,
        typer.Option(
            "--checkpoint-keep",
            min=1,
            help=(
                "Keep at most this many checkpoints per trial (default 3, or a "
                "run config's checkpoint_keep); the oldest is deleted."
            ),
        ),
    ] = None,
    freeze_workspace: Annotated[
        bool,
        typer.Option(
            "--freeze-workspace",
            help=(
                "Save each trial's final workspace and declared artifacts "
                "(evidence/, with a manifest) before the verifier runs, so "
                "`bench regrade` can re-score it with a changed verifier."
            ),
        ),
    ] = False,
    keep_sandbox: Annotated[
        str | None,
        typer.Option(
            "--keep-sandbox",
            help=(
                "Keep a trial's sandbox: on-failure, always or never (default). "
                "`bench sandbox attach <trial>` opens a shell in it; `bench "
                "sandbox release <trial>` deletes it."
            ),
        ),
    ] = None,
    accept_deviations: Annotated[
        str | None,
        typer.Option(
            "--accept-deviations",
            help=(
                "Why this run differs from its task set's run-reference.toml "
                "(a strict one refuses the run without it); kept in run-card.json."
            ),
        ),
    ] = None,
    fresh_install: FreshInstallOption = False,
    allow_cloud_metadata: AllowCloudMetadataOption = False,
    integrity: Annotated[
        str | None,
        typer.Option(
            "--integrity",
            help=(
                "off, audit (record what the agent did and check it against the "
                "task) or strict (audit, and the verifier in a separate sandbox). "
                "The verdict is in integrity/claim_verdict.json; rewards never change."
            ),
        ),
    ] = None,
    verifier_sandbox: Annotated[
        str | None,
        typer.Option(
            "--verifier-sandbox",
            help=(
                "task (default: each task's own choice) or separate: every "
                "verifier in a fresh sandbox with only the workspace and the "
                "declared artifacts; a task that cannot run there refuses the run."
            ),
        ),
    ] = None,
    retry_from_checkpoint: Annotated[
        str | None,
        typer.Option(
            "--retry-from-checkpoint",
            help=(
                "on-failure, on-timeout, or both: when a trial fails or times out, "
                "fork one retry from its last --checkpoints snapshot, verified the "
                "same way. The trial's reward is kept; the retry's is reported "
                "next to it (result.json retry block, summary checkpoint_retries)."
            ),
        ),
    ] = None,
    retry_prompt: Annotated[
        str | None,
        typer.Option(
            "--retry-prompt",
            help=(
                "Prompt for the retry, replacing the prompts after the "
                "checkpoint (the default). '@instruction' is the task "
                "instruction, '@verifier_feedback' the failed trial's reward "
                "and verifier output tail."
            ),
        ),
    ] = None,
    max_cost_usd: Annotated[
        float | None,
        typer.Option(
            "--max-cost-usd",
            help=(
                "Job budget in USD, checked between starts, so a job can pass it "
                "by about one rollout's cost; at the cap running rollouts are "
                "cancelled. With --trials, each repeat has its own cap"
            ),
        ),
    ] = None,
    max_sandbox_seconds: Annotated[
        float | None,
        typer.Option(
            "--max-sandbox-seconds",
            help=(
                "Job budget in trial wall-clock seconds, running trials "
                "included and re-checked as they run; same stop/cancel "
                "behaviour as --max-cost-usd"
            ),
        ),
    ] = None,
    max_tokens: Annotated[
        int | None,
        typer.Option(
            "--max-tokens",
            help=(
                "Job budget in tokens, checked between starts like "
                "--max-cost-usd. With --trials, each repeat has its own cap"
            ),
        ),
    ] = None,
    max_rollouts: Annotated[
        int | None,
        typer.Option(
            "--max-rollouts",
            help=(
                "Job budget in rollouts started, retries included: at the cap "
                "no rollout starts and running ones finish"
            ),
        ),
    ] = None,
    token_budget: Annotated[
        int | None,
        typer.Option(
            "--token-budget",
            help=(
                "Per-trial budget in tokens (input, cached included, plus output), "
                "subagents' calls included: BenchFlow's model proxy refuses the "
                "next model call once it is spent, the agent is stopped and its "
                "work graded. Replaces a task's [agent] token_budget"
            ),
        ),
    ] = None,
    turn_budget: Annotated[
        int | None,
        typer.Option(
            "--turn-budget",
            help=(
                "Per-trial budget in turns, one turn being one model call through "
                "BenchFlow's model proxy (subagents' and side calls included); "
                "enforced like --token-budget. Replaces a task's [agent] turn_budget"
            ),
        ),
    ] = None,
    retry_resume_session: Annotated[
        bool,
        typer.Option(
            "--retry-resume-session",
            help=(
                "The retry resumes the failed trial's conversation at the "
                "checkpoint (ACP session/load; Claude Code) instead of a fresh "
                "session."
            ),
        ),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            help=(
                "Print only warnings, errors and the final score: no Rich "
                "dashboard on a TTY, no progress heartbeat (the '… Nmin, K "
                "tool calls' lines) and no per-trial log lines"
            ),
        ),
    ] = False,
    jobs_dir: Annotated[
        str | None,
        typer.Option(
            "--jobs-dir",
            help="Output directory (default jobs); a plain run resumes its latest job",
        ),
    ] = None,
    fresh: Annotated[
        bool,
        typer.Option(
            "--fresh",
            help="Start a new timestamped job instead of resuming the latest job "
            "in --jobs-dir (the default reuses its finished tasks)",
        ),
    ] = False,
    job_name: Annotated[
        str | None,
        typer.Option(
            "--job-name",
            help="Job folder name under --jobs-dir (resumed if it exists)",
        ),
    ] = None,
    restart_changed: Annotated[
        bool,
        typer.Option(
            "--restart-changed",
            help=(
                "When the named job ran with other settings or task files, set "
                "those trials aside and run them again (default: refuse)"
            ),
        ),
    ] = False,
    fail_under: Annotated[
        float | None,
        typer.Option(
            "--fail-under",
            help=(
                "Exit 1 when the pass rate (passed / planned tasks; one a budget "
                "kept from running counts as not passed) is below this, e.g. 0.8"
            ),
        ),
    ] = None,
    fail_on: Annotated[
        list[str] | None,
        typer.Option(
            "--fail-on",
            help="Exit 1 when any trial ended this way: timeout, error, verifier-error "
            "(comma-separated or repeated)",
        ),
    ] = None,
    summary_out: Annotated[
        Path | None,
        typer.Option(
            "--summary-out",
            help="Write the run's benchflow.run-summary JSON (job dir, counts, gate) here",
        ),
    ] = None,
    codex_apps_policy: Annotated[
        str | None,
        typer.Option(
            "--codex-apps-policy",
            help="Codex account Apps: disabled (scored-task default) or inherit (explicit opt-in).",
        ),
    ] = None,
    sandbox_user: Annotated[
        str | None,
        typer.Option("--sandbox-user", help="Sandbox user (null for root)"),
    ] = _SANDBOX_USER_OMITTED,
    sandbox_setup_timeout: Annotated[
        int | None,
        typer.Option(
            "--sandbox-setup-timeout",
            help="Timeout (seconds) for sandbox user setup inside the environment (default: 120; keeps the file's value with --config).",
        ),
    ] = None,
    context_root: Annotated[
        Path | None,
        typer.Option(
            "--context-root",
            help=(
                "Repo/build-context root used to stage Dockerfile COPY sources "
                "for monorepo-authored local tasks."
            ),
        ),
    ] = None,
    base_image_override: Annotated[
        str | None,
        typer.Option(
            "--base-image-override",
            help=(
                "Rewrite task Dockerfile FROM images on the runtime task copy. "
                "Use only for reproducing runs whose base image moved namespaces."
            ),
        ),
    ] = None,
    separate_verifier: Annotated[
        bool,
        typer.Option(
            "--separate-verifier",
            help=(
                "Run each verifier in a fresh sandbox built from the verifier image "
                "that receives only the frozen workspace, declared artifacts and "
                "/logs/artifacts, as if every task set "
                "\\[verifier].sandbox_mode = 'separate'. Docker, remote-docker, "
                "Daytona and Modal."
            ),
        ),
    ] = False,
    separate_verifier_include: Annotated[
        list[str] | None,
        typer.Option(
            "--separate-verifier-include",
            help=(
                "A path outside the workspace that a separate verifier sandbox "
                "receives too, at the same path, for tasks whose agent writes "
                "graded files there (e.g. /root when the workspace is /app). "
                "Repeatable."
            ),
        ),
    ] = None,
    skills_dir: Annotated[
        Path | None,
        typer.Option("--skills-dir", help="Skills directory to deploy"),
    ] = None,
    skill_mode: SkillModeOption = SKILL_MODE_NO_SKILL,
    skill_creator_dir: Annotated[
        Path | None,
        typer.Option(
            "--skill-creator-dir",
            help="Path to skill-creator or a skills root containing it",
        ),
    ] = None,
    self_gen_no_internet: Annotated[
        bool,
        typer.Option(
            "--self-gen-no-internet",
            help="Disable web tools for the self-generated run",
        ),
    ] = False,
    loop_strategy: Annotated[
        str | None,
        typer.Option(
            "--loop-strategy",
            help=(
                "Harness loop strategy, e.g. 'verify-retry:k=3,feedback=names', "
                "'self-review:k=3', or 'single-shot' (default). verify-retry "
                "re-prompts with filtered soft-verifier feedback; self-review "
                "re-prompts the agent to critique its OWN work (no verifier "
                "signal) — both loop until the soft verifier passes or k retries "
                "are spent."
            ),
        ),
    ] = None,
    reviewer_agent: ReviewerAgentOption = None,
    reviewer_model: ReviewerModelOption = None,
    reviewer_reasoning_effort: ReviewerEffortOption = None,
    reviewer_sandbox: ReviewerSandboxOption = None,
    reviewer_timeout_sec: ReviewerTimeoutOption = None,
    reviewer_idle_timeout_sec: ReviewerIdleTimeoutOption = None,
    reviewer_storage_mb: ReviewerStorageOption = None,
    reviewer_memory_mb: ReviewerMemoryOption = None,
    reviewer_setup_timeout_sec: ReviewerSetupTimeoutOption = None,
    reviewer_evidence_timeout_sec: ReviewerEvidenceTimeoutOption = None,
    reviewer_concurrency: ReviewerConcurrencyOption = None,
    reviewer_image: ReviewerImageOption = None,
    reviewer_agent_env: ReviewerEnvOption = None,
    reviewer_open_network: ReviewerNetworkOption = None,
    reviewer_pdf_tools: ReviewerPdfToolsOption = None,
    agent_env: Annotated[
        list[str] | None,
        typer.Option("--agent-env", help="Agent env var (KEY=VALUE)"),
    ] = None,
    allow_task_env: AllowTaskEnvOption = None,
    include: Annotated[
        list[str] | None,
        typer.Option(
            "--include",
            "-i",
            help="Only run these task names; repeatable (e.g. --include jax-computing-basics --include data-to-d3)",
        ),
    ] = None,
    exclude: Annotated[
        list[str] | None,
        typer.Option(
            "--exclude",
            "-x",
            help="Skip these task names; repeatable (e.g. --exclude quantum-numerical-simulation)",
        ),
    ] = None,
    dataset: Annotated[
        str | None,
        typer.Option(
            "--dataset",
            "-d",
            help="Registry dataset to run, as <name>@<version> (e.g. skillsbench@1.1). "
            "Resolves the pinned snapshot and verifies task content digests.",
        ),
    ] = None,
    registry: Annotated[
        str | None,
        typer.Option(
            "--registry",
            help="Dataset registry JSON URL or local file, or a BenchFlow Hub as "
            "hub:<url> (default: BenchFlow's registry, then $BENCHFLOW_HUB_URL). "
            "Only valid with --dataset.",
        ),
    ] = None,
    ignore_bench_version: Annotated[
        bool,
        typer.Option(
            "--ignore-bench-version",
            help="Run a dataset even when this bench version is outside the "
            "range it was validated against. Only valid with --dataset.",
        ),
    ] = False,
    task_manifest_out: Annotated[
        Path | None,
        typer.Option(
            "--task-manifest-out", help="Write selected task-set manifest JSON"
        ),
    ] = None,
    run_config_out: Annotated[
        Path | None,
        typer.Option(
            "--run-config-out", help="Write redacted normalized run config JSON"
        ),
    ] = None,
    health_summary_out: Annotated[
        Path | None,
        typer.Option(
            "--health-summary-out", help="Write trajectory health summary JSON"
        ),
    ] = None,
    expected_tasks: Annotated[
        int | None,
        typer.Option(
            "--expected-tasks", help="Fail unless the selected task count matches"
        ),
    ] = None,
    canonicalize: Annotated[
        str,
        typer.Option(
            "--canonicalize",
            help="Canonicalization policy: none or one-healthy-per-task",
        ),
    ] = "none",
    canonical_selection_out: Annotated[
        Path | None,
        typer.Option(
            "--canonical-selection-out",
            help="Write canonical rollout-selection JSON",
        ),
    ] = None,
    canonical_jobs_dir: Annotated[
        Path | None,
        typer.Option(
            "--canonical-jobs-dir",
            help="Materialize selected rollout dirs for trainer conversion",
        ),
    ] = None,
    retry_policy: Annotated[
        str,
        typer.Option("--retry-policy", help="Retry policy: default or unscored-only"),
    ] = "default",
    retry_attempts: Annotated[
        int | None,
        typer.Option(
            "--retry-attempts",
            help="Retries per task after a retryable failure (overrides the default of 2)",
        ),
    ] = None,
    retry_concurrency: Annotated[
        int | None,
        typer.Option("--retry-concurrency", help="Reserved retry concurrency"),
    ] = None,
    publish_hf: Annotated[
        str | None,
        typer.Option(
            "--publish-hf", help="Upload final eval artifacts to a HF dataset repo"
        ),
    ] = None,
    hf_prefix: Annotated[
        str | None,
        typer.Option("--hf-prefix", help="Path prefix in the HF repo"),
    ] = None,
    hf_public_read_check: Annotated[
        bool,
        typer.Option(
            "--hf-public-read-check", help="Verify public HF reads after upload"
        ),
    ] = False,
    publish_bucket: Annotated[
        str | None,
        typer.Option(
            "--publish-bucket", help="Upload final eval artifacts to a HF bucket"
        ),
    ] = None,
    eval_results_model: Annotated[
        str | None,
        typer.Option(
            "--eval-results-model",
            help="HF model repo to open a community eval-results PR on",
        ),
    ] = None,
    eval_results_dataset: Annotated[
        str | None,
        typer.Option(
            "--eval-results-dataset", help="HF benchmark dataset id, e.g. org/benchmark"
        ),
    ] = None,
    eval_results_task: Annotated[
        str | None,
        typer.Option(
            "--eval-results-task", help="Benchmark task_id from the dataset's eval.yaml"
        ),
    ] = None,
    matrix: Annotated[
        Path | None,
        typer.Option(
            "--matrix",
            help=(
                "YAML file whose models: maps each alias to a model, or to "
                "{model, agent, agent_env}; {agent: oracle} for a control "
                "(docs/audit/compare-runs.md#choose-the-metric)"
            ),
        ),
    ] = None,
    trials: Annotated[
        int,
        typer.Option(
            "--trials",
            help=(
                "Run every task N times (pass@k); each repeat is a job of its "
                "own, in <jobs dir>/<agent>/trial-NN"
            ),
        ),
    ] = 1,
    n_tasks: Annotated[
        int | None,
        typer.Option(
            "--n-tasks",
            "-n",
            "--limit",
            help="Run at most N tasks after --include/--exclude: the first N, or a seeded "
            "random sample with --sample-seed",
        ),
    ] = None,
    sample_seed: Annotated[
        int | None,
        typer.Option("--sample-seed", help="Seed for the --n-tasks random sample"),
    ] = None,
    timeout_multiplier: Annotated[
        float | None,
        typer.Option(
            "--timeout-multiplier",
            help="Scale every task's agent time budget (task.md agent.timeout_sec), e.g. 2.0",
        ),
    ] = None,
    extra_instruction: Annotated[
        str | None,
        typer.Option(
            "--extra-instruction",
            help="Text appended to every task prompt (prompt ablations; recorded per rollout)",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Resolve the plan (agent, model, selected tasks, seeds, budgets) and print it without running; exits 1 when the run card would refuse the run",
        ),
    ] = False,
    seeds: Annotated[
        str | None,
        typer.Option(
            "--seeds",
            help=(
                "Seeded rollouts: run every task once per seed, e.g. 0-4 or 0,3,7 "
                "(tasks in a task format that writes seeded variants, such as "
                "embodied tasks). summary.json gets pass@k and variance per task."
            ),
        ),
    ] = None,
) -> None:
    # The supported --sandbox values are rendered from the provider registry
    # into that option's own help text. This docstring used to hand-copy them
    # ("docker, daytona, modal, or apple-container") and went stale the moment
    # a backend was added, so it no longer repeats them.
    """Run an evaluation — single task or batch."""
    from rich.markup import escape

    from benchflow.cli._errors import expected_error_parts
    from benchflow.cli._shared import (
        _apply_allow_task_env,
        _apply_dotenv_to_process_env,
        _parse_agent_env,
        console,
        exit_if_missing,
        print_error,
        print_hint,
    )
    from benchflow.cli.doctor import eval_preflight, start_docker_preflight
    from benchflow.cli.eval_artifacts import run_matrix_eval
    from benchflow.cli.new_verbs import idle_seconds
    from benchflow.cli.reviewer_options import reviewer_from_cli, reviewer_idle_timeout

    _apply_dotenv_to_process_env()
    _apply_allow_task_env(allow_task_env)
    if (
        dataset
        and registry
        and not registry.startswith(("http://", "https://", "hub:"))
    ):
        exit_if_missing(
            Path(registry),
            next_step="pass an existing registry JSON file or URL to --registry",
        )
    # The preflight's Docker check runs while the engine imports (below).
    docker_preflight = (
        None
        if dry_run or config_file or source_env
        else start_docker_preflight(environment)
    )
    from benchflow.eval_plan import EvalCreateRequest, EvalPlanError, build_eval_plan
    from benchflow.evaluation import effective_model
    from benchflow.task.verifier_sandbox import normalize_include_paths

    if quiet:
        _quiet_run()

    try:
        include_paths = normalize_include_paths(separate_verifier_include or ())
    except ValueError as exc:
        print_error(f"{exc}", next_step="fix the --separate-verifier-include path")
        raise typer.Exit(2) from None

    # All CLI spellings and both exported Python names reach this callback.
    # The sentinel survives Click's string conversion, so identity distinguishes
    # omission from an explicit "agent", empty string or None without a context.
    sandbox_user_explicit = sandbox_user is not _SANDBOX_USER_OMITTED
    if not sandbox_user_explicit:
        sandbox_user = "agent"
    request = EvalCreateRequest(
        config_file=config_file,
        tasks_dir=tasks_dir,
        source_repo=source_repo,
        source_env=source_env,
        agent=agent,
        model=model,
        reasoning_effort=reasoning_effort,
        harness=harness,
        environment=environment,
        usage_tracking=usage_tracking,
        environment_manifest=environment_manifest,
        state=state,
        config_override=config_override,
        prompt=prompt,
        concurrency=concurrency,
        build_concurrency=build_concurrency,
        warm_pool=warm_pool,
        worker_concurrency=worker_concurrency,
        worker_retries=worker_retries,
        worker_start_stagger_sec=worker_start_stagger_sec,
        # 90s, 10m and 1h as bench run takes them (dogfood p5-11).
        agent_idle_timeout=idle_seconds(agent_idle_timeout)
        if agent_idle_timeout is not None
        else None,
        checkpoints=checkpoints,
        checkpoint_keep=checkpoint_keep,
        freeze_workspace=freeze_workspace,
        keep_sandbox=keep_sandbox,
        accept_deviations=accept_deviations,
        fresh_install=fresh_install,
        allow_cloud_metadata=allow_cloud_metadata,
        integrity=integrity,
        verifier_sandbox=verifier_sandbox,
        retry_from_checkpoint=retry_from_checkpoint,
        retry_prompt=retry_prompt,
        retry_resume_session=retry_resume_session,
        max_cost_usd=max_cost_usd,
        max_sandbox_seconds=max_sandbox_seconds,
        max_tokens=max_tokens,
        max_rollouts=max_rollouts,
        token_budget=token_budget,
        turn_budget=turn_budget,
        jobs_dir=jobs_dir,
        fresh=fresh,
        job_name=job_name,
        restart_changed=restart_changed,
        fail_under=fail_under,
        fail_on=fail_on,
        summary_out=summary_out,
        codex_apps_policy=codex_apps_policy,
        sandbox_user=sandbox_user,
        sandbox_user_explicit=sandbox_user_explicit,
        sandbox_setup_timeout=sandbox_setup_timeout,
        context_root=context_root,
        base_image_override=base_image_override,
        separate_verifier=separate_verifier,
        separate_verifier_include=include_paths,
        skills_dir=skills_dir,
        skill_mode=skill_mode,
        skill_creator_dir=skill_creator_dir,
        self_gen_no_internet=self_gen_no_internet,
        loop_strategy=loop_strategy,
        agent_env=_parse_agent_env(agent_env),
        reviewer=reviewer_from_cli(
            agent=reviewer_agent,
            model=reviewer_model,
            reasoning_effort=reviewer_reasoning_effort,
            environment=reviewer_sandbox,
            timeout_sec=reviewer_timeout_sec,
            idle_timeout_sec=reviewer_idle_timeout(reviewer_idle_timeout_sec),
            storage_mb=reviewer_storage_mb,
            memory_mb=reviewer_memory_mb,
            setup_timeout_sec=reviewer_setup_timeout_sec,
            evidence_timeout_sec=reviewer_evidence_timeout_sec,
            concurrency=reviewer_concurrency,
            image=reviewer_image,
            agent_env=(
                _parse_agent_env(reviewer_agent_env)
                if reviewer_agent_env is not None
                else None
            ),
            open_network=reviewer_open_network,
            pdf_tools=reviewer_pdf_tools,
        ),
        include=include,
        exclude=exclude,
        dataset=dataset,
        registry=registry,
        ignore_bench_version=ignore_bench_version,
        task_manifest_out=task_manifest_out,
        run_config_out=run_config_out,
        health_summary_out=health_summary_out,
        expected_tasks=expected_tasks,
        canonicalize=canonicalize,
        canonical_selection_out=canonical_selection_out,
        canonical_jobs_dir=canonical_jobs_dir,
        retry_policy=retry_policy,
        retry_attempts=retry_attempts,
        retry_concurrency=retry_concurrency,
        publish_hf=publish_hf,
        hf_prefix=hf_prefix,
        hf_public_read_check=hf_public_read_check,
        publish_bucket=publish_bucket,
        eval_results_model=eval_results_model,
        eval_results_dataset=eval_results_dataset,
        eval_results_task=eval_results_task,
        matrix=matrix,
        trials=trials,
        seeds=seeds,
        n_tasks=n_tasks,
        sample_seed=sample_seed,
        timeout_multiplier=timeout_multiplier,
        extra_instruction=extra_instruction,
    )
    # --source-path/--source-ref only apply to --source-repo; otherwise they're
    # silently ignored (e.g. `--dataset X --source-ref abc` drops the ref).
    if (source_path or source_ref) and not source_repo:
        print_error(
            "--source-path/--source-ref require --source-repo",
            next_step="add --source-repo, or drop --source-path and --source-ref",
        )
        raise typer.Exit(2)
    if integrity is not None:
        from benchflow.integrity.trial import normalize_integrity_mode

        try:
            normalize_integrity_mode(integrity)
        except ValueError as exc:
            print_error(
                str(exc),
                next_step="`bench run --help` lists the --integrity modes",
            )
            raise typer.Exit(2) from None
    if checkpoints or retry_from_checkpoint:
        from benchflow.checkpoint_retry import parse_retry_policy
        from benchflow.checkpoints import parse_checkpoint_policy

        try:
            if checkpoints:
                parse_checkpoint_policy(
                    checkpoints, keep=3 if checkpoint_keep is None else checkpoint_keep
                )
            if retry_from_checkpoint:
                # A run config may set checkpoints: checked once it is loaded.
                if not checkpoints and not config_file:
                    raise ValueError(
                        "--retry-from-checkpoint needs --checkpoints: a trial can "
                        "only be retried from a checkpoint it kept"
                    )
                parse_retry_policy(retry_from_checkpoint, prompt=retry_prompt)
        except ValueError as exc:
            print_error(
                str(exc),
                next_step="`bench run --help` describes --checkpoints and "
                "--retry-from-checkpoint",
            )
            raise typer.Exit(2) from None
    try:
        plan = build_eval_plan(request)
    except EvalPlanError as exc:
        err_text, err_hint = expected_error_parts(exc)
        print_error(
            err_text, next_step=err_hint or "`bench run --help` lists the flags"
        )
        raise typer.Exit(exc.exit_code) from None
    if not (config_file or source_env or source_repo or tasks_dir or dataset):
        # Before the default-agent warning and the sandbox and credential
        # checks: with no task, a missing key is not the problem
        # (DF-p1-first-run-4).
        _exit_no_tasks()
    if agent is None and not config_file and not source_env and matrix is None:
        # 0.8 still runs its default agent and model here, but says
        # so once (_run_config_file_eval warns for a --config file without an
        # agent once its overrides are applied; a matrix names its models).
        _warn_default_agent(effective_model(plan.eval_agent, model, plan.parsed_env))
    # A dry run prints the plan and starts nothing, so it needs no sandbox or
    # credential checks.
    if dry_run:
        _print_dry_run(plan, tasks_dir)
        return
    # Hosted source envs own their harness; a run config names its sandbox in
    # the YAML, so _run_config_file_eval checks it once the file is loaded.
    if not config_file and not source_env:
        # A matrix runs its entries' agents and models, not the run's own: a
        # matrix of {agent: oracle} needs no model login.
        for pre_agent, pre_model, pre_env in _preflight_targets(plan):
            eval_preflight(
                sandbox=plan.eval_environment,
                agent=pre_agent,
                model=pre_model,
                agent_env=pre_env,
                docker=docker_preflight,
            )

    if config_file:
        _run_config_file_eval(plan)
    elif source_env:
        _run_source_env_eval(
            plan,
            source_env_version=source_env_version,
            source_env_arg=source_env_arg,
            source_env_num_examples=source_env_num_examples,
            source_env_rollouts_per_example=source_env_rollouts_per_example,
            source_env_max_tokens=source_env_max_tokens,
            source_env_temperature=source_env_temperature,
            source_env_sampling_arg=source_env_sampling_arg,
            source_env_verifiers_version=source_env_verifiers_version,
            source_env_base_url=source_env_base_url,
            source_env_api_key_var=source_env_api_key_var,
        )
    elif source_repo:
        import subprocess

        from benchflow._utils.benchmark_repos import (
            GitCloneError,
            resolve_source_with_metadata,
        )
        from benchflow.adapters.source import adapt_resolved_source_if_needed
        from benchflow.errors import UserError

        try:
            resolved = resolve_source_with_metadata(
                source_repo, path=source_path, ref=source_ref
            )
            resolved = adapt_resolved_source_if_needed(resolved)
        except (subprocess.CalledProcessError, OSError, ValueError, RuntimeError) as e:
            if isinstance(
                e, (subprocess.CalledProcessError, GitCloneError)
            ) and not isinstance(e, UserError):
                raise
            # A clone failure (missing/private repo, bad --source-ref, auth,
            # network) otherwise escapes as a raw traceback — unlike the sibling
            # config/source-env/dataset branches which all map to a clean error.
            hint = getattr(e, "hint", None)
            print_error(
                f"Could not resolve --source-repo {source_repo!r}: {e}",
                next_step=hint
                if isinstance(hint, str) and hint
                else "check the repository name, --source-path and "
                "--source-ref, and that this machine can clone it",
            )
            raise typer.Exit(1) from None
        if matrix is not None or plan.request.trials > 1:
            # Repeated trials of a repository's tasks, as for a task folder.
            run_matrix_eval(
                plan,
                resolved.path,
                run_batch_eval,
                config_overrides={"source_provenance": resolved.provenance},
            )
        else:
            run_batch_eval(
                plan,
                resolved.path,
                plan.make_eval_config(source_provenance=resolved.provenance),
            )
    elif tasks_dir:
        # Single-task and batch share one orchestration path. Evaluation
        # handles both layouts (Evaluation._get_task_dirs detects when
        # tasks_dir itself contains task.toml) and applies include/exclude
        # filters uniformly (#400, #401, #407).
        if not Path(tasks_dir).is_dir():
            # Without this guard a file (or missing path) reaches iterdir() and
            # dumps a raw NotADirectoryError, unlike sandbox create / eval metrics.
            print_error(
                f"Not a directory: {tasks_dir}",
                next_step="pass a task folder or a folder of tasks to --tasks-dir",
            )
            raise typer.Exit(2)
        if matrix is not None or plan.request.trials > 1:
            run_matrix_eval(plan, tasks_dir, run_batch_eval)
        else:
            run_batch_eval(plan, tasks_dir, plan.make_eval_config())
    elif dataset:
        from benchflow._utils.dataset_registry import (
            bench_version_issue,
            resolve_dataset,
        )
        from benchflow.errors import UserError

        try:
            with console.status(f"Resolving dataset {dataset}…"):
                resolved_dataset = resolve_dataset(dataset, registry=registry or None)
        except UserError as e:
            # DatasetResolutionError, and a snapshot fetch that failed (no
            # network, a repository git cannot reach): its message and hint.
            err_text, err_hint = expected_error_parts(e)
            print_error(
                err_text,
                next_step=err_hint or "check the --dataset name@version and --registry",
            )
            raise typer.Exit(getattr(e, "exit_code", 1)) from None
        version_issue = bench_version_issue(resolved_dataset.bench_version)
        if version_issue and not ignore_bench_version:
            # Hard gate: published results must come from a harness the
            # dataset version was validated against. The escape hatch keeps
            # local experimentation possible without weakening the default.
            _refuse_dataset_bench_version(version_issue)
        if version_issue:
            console.print(
                f"[yellow]Warning:[/yellow] {escape(str(version_issue))} (--ignore-bench-version)"
            )
        unpinned = len(resolved_dataset.unpinned_tasks)
        total = len(resolved_dataset.task_names)
        if not unpinned:
            checked = "digests verified"
        elif unpinned == total:
            checked = "no digests pinned by the registry, so none checked"
        else:
            checked = (
                f"{total - unpinned} digests verified, {unpinned} not pinned "
                "by the registry"
            )
        console.print(
            f"[green]✓[/green] {escape(str(resolved_dataset.spec))}: "
            f"{total} tasks, {checked} "
            f"({str(resolved_dataset.provenance.get('resolved_sha', '')).removeprefix('sha256:')[:12]})"
        )
        from benchflow.cli._hub_client import run_controls_notice

        run_controls_notice(resolved_dataset, agent=agent)
        # tasks_dir is the resolved source checkout (a superset); restrict to
        # the dataset's pinned task set, narrowed by --include and --exclude.
        # An empty set would mean "no filter" to Evaluation and run the whole
        # checkout, so a selection that matches nothing stops here, with the
        # same refusal as --tasks-dir (DF-p3-eval-runner-1).
        from benchflow.cli._shared import command_arg
        from benchflow.evaluation import (
            empty_selection_message,
            task_name_selected,
            unmatched_selection,
        )

        dataset_names = resolved_dataset.task_names
        dataset_include = {
            name
            for name in dataset_names
            if task_name_selected(name, plan.include_tasks, plan.exclude_tasks)
        }
        if not dataset_include:
            print_error(
                empty_selection_message(
                    f"dataset={resolved_dataset.spec}",
                    plan.include_tasks,
                    plan.exclude_tasks,
                    dataset_names,
                ),
                next_step=(
                    f"`bench hub show {resolved_dataset.spec}`"
                    if registry is None
                    else "`bench check "
                    f"{command_arg(resolved_dataset.tasks_dir, '<dataset checkout>')}`"
                )
                + " lists the dataset's tasks; check --include and --exclude "
                "against their names",
            )
            # A usage mistake: -i or -x names no task (dogfood p5-8).
            raise typer.Exit(2)
        for line in unmatched_selection(
            dataset_names, plan.include_tasks, plan.exclude_tasks
        ):
            print_hint(
                f"warning: {line} in {resolved_dataset.spec}",
                style="yellow",
                stderr=True,
            )
        dataset_config = {
            "source_provenance": resolved_dataset.provenance,
            "dataset_name": resolved_dataset.name,
            "dataset_version": resolved_dataset.version,
            "dataset_task_digests": resolved_dataset.task_digests,
            "include_tasks": dataset_include,
            # Already applied above, against the dataset's own names.
            "exclude_tasks": set(),
        }
        if matrix is not None or plan.request.trials > 1:
            # Repeated trials of a dataset, for pass@k (DF-p3-eval-runner-3).
            run_matrix_eval(
                plan,
                resolved_dataset.tasks_dir,
                run_batch_eval,
                config_overrides=dataset_config,
            )
        else:
            run_batch_eval(
                plan,
                resolved_dataset.tasks_dir,
                plan.make_eval_config(**dataset_config),
            )
    else:
        _exit_no_tasks()


def _exit_no_tasks() -> NoReturn:
    from benchflow.cli._shared import print_error

    print_error(
        "No tasks to run: no --config, --tasks-dir, --source-repo, "
        "--source-env or --dataset",
        next_step="name a task, for example `bench run "
        "benchflow-ai/skillsbench/tasks/econ-detrending-correlation`; "
        "`bench run --help` describes each source",
    )
    raise typer.Exit(2)


# Batch and config-file execution helpers


def _preflight_targets(
    plan: "EvalPlan",
) -> list[tuple[str, str | None, dict[str, str]]]:
    """``(agent, model, agent_env)`` for each agent the run will start: the
    run's own, or each ``--matrix`` entry's (its agent, else the run's)."""
    from benchflow.cli._shared import print_error
    from benchflow.cli.eval_artifacts import (
        _load_eval_matrix,
        _matrix_agent_env,
    )

    req = plan.request
    if req.matrix is None:
        return [(plan.eval_agent, req.model, dict(plan.parsed_env))]
    try:
        models = _load_eval_matrix(req.matrix)
        targets = [
            (
                str(raw.get("agent") or plan.eval_agent),
                None if raw["model"] is None else str(raw["model"]),
                {**plan.parsed_env, **_matrix_agent_env(raw.get("agent_env"))},
            )
            for raw in models.values()
        ]
    except (OSError, ValueError) as exc:
        print_error(
            str(exc),
            next_step="fix the --matrix file; docs/audit/compare-runs.md#choose-the-metric shows one",
        )
        raise typer.Exit(1) from None
    # Each distinct agent, model and variables once.
    seen: set[tuple[str, str | None, tuple[tuple[str, str], ...]]] = set()
    distinct = []
    for agent, model, env in targets:
        key = (agent, model, tuple(sorted(env.items())))
        if key not in seen:
            seen.add(key)
            distinct.append((agent, model, env))
    return distinct


def _eval_label(plan: "EvalPlan", tasks_dir: Path) -> str:
    """A short source descriptor for the live dashboard header."""
    req = plan.request
    for attr in ("source_repo", "dataset", "source_env"):
        val = getattr(req, attr, None)
        if val:
            return str(val)
    return tasks_dir.name


def _finish_run(req: "EvalCreateRequest", result: Any, job_dir: Path | None) -> None:
    """Apply --fail-under/--fail-on, write --summary-out, and exit.

    Exit 1 when a trial ended unscored with an error (as before), a gate
    failed or a run check failed (summary.json's ``run_checks``); otherwise
    return (exit 0).
    """
    from rich.markup import escape

    from benchflow.cli._shared import err_console
    from benchflow.eval_plan import split_fail_on

    results = getattr(result, "results", {}) or {}
    counts = {
        "timeout": sum(r.error_category == "timeout" for r in results.values()),
        "error": int(getattr(result, "errored", 0) or 0),
        "verifier-error": int(getattr(result, "verifier_errored", 0) or 0),
    }
    failed: list[str] = []
    # The gate's denominator is the plan: a task a budget or usage limit kept
    # from running counts as not passed, so a job capped after 3 of 10 tasks
    # cannot pass --fail-under 0.95 on 3/3 (DF-p3-eval-runner-5).
    not_run = int(getattr(result, "not_run", 0) or 0)
    planned_score = (
        result.score_of_plan
        if not_run and hasattr(result, "score_of_plan")
        else result.score
    )
    if req.fail_under is not None and planned_score < req.fail_under:
        failed.append(
            f"pass rate {planned_score:.2f} < --fail-under {req.fail_under:g}"
            + (
                f" ({result.passed} passed of {result.total + not_run} planned tasks; "
                f"{not_run} not run)"
                if not_run
                else ""
            )
        )
    for kind in split_fail_on(req.fail_on):
        if counts[kind]:
            failed.append(f"{counts[kind]} trial(s) with {kind} (--fail-on {kind})")
    for line in failed:
        err_console.print(
            f"[red]Gate failed:[/red] {escape(line)}", highlight=False, soft_wrap=True
        )
    # A run check refused the run (benchflow.run_checks): its trials were cut
    # at the output cap, or the calls did not carry what the run asked for.
    from benchflow.cli._shared import print_run_check_failures

    failed += [f"run check: {line}" for line in print_run_check_failures(result)]
    exit_code = 1 if (failed or counts["error"] or counts["verifier-error"]) else 0
    _write_summary_out(req, result, job_dir, failed=failed, exit_code=exit_code)
    if exit_code:
        raise typer.Exit(exit_code)


def _write_summary_out(
    req: "EvalCreateRequest",
    result: Any,
    job_dir: Path | None,
    *,
    failed: list[str],
    exit_code: int,
) -> None:
    """Write --summary-out (a ``benchflow.run-summary``), whole or not at all."""
    import json

    from benchflow.eval_plan import split_fail_on

    if req.summary_out is None:
        return
    # Only here: the export models take about 25 ms to import.
    from benchflow._utils.disk_space import write_text_atomic
    from benchflow.job_export import run_summary_export

    results = getattr(result, "results", {}) or {}
    document = run_summary_export(
        result,
        job_dir=job_dir,
        timeouts=sum(r.error_category == "timeout" for r in results.values()),
        fail_under=req.fail_under,
        fail_on=split_fail_on(req.fail_on),
        gate_failed=failed,
        exit_code=exit_code,
    ).model_dump(mode="json")
    req.summary_out.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(
        req.summary_out, json.dumps(document, indent=2, allow_nan=False) + "\n"
    )


def _write_stopped_summary_out(
    req: "EvalCreateRequest", evaluation: Any, stop: Any
) -> None:
    """After Ctrl-C or SIGTERM: --summary-out from the partial result the
    job wrote to summary.json (DFR-p3-11). No gate is applied to a stopped
    run; its exit code is the stop's."""
    result = getattr(evaluation, "result", None) if evaluation is not None else None
    if result is None or not getattr(result, "stopped", False):
        return
    _write_summary_out(
        req, result, evaluation.job_dir, failed=[], exit_code=stop.exit_code
    )


def run_batch_eval(
    plan: "EvalPlan",
    resolved_tasks_dir: Path,
    eval_config: "EvaluationConfig",
):
    """Run the source-repo / tasks-dir batch path and report its result.

    One trial job of a matrix (``plan.request.matrix_trial``) reports the job
    but applies no gate and never exits on its errors: the matrix gates once,
    after every trial has run (DX-F-MATRIX-STOP).

    Promoted from the ``eval_run`` ``_run_batch_eval`` closure: the worker /
    jobs-dir / manifest knobs it used to capture now ride in on ``plan``.
    """
    from benchflow.agents.errors import UsageLimitError
    from benchflow.cli._errors import expected_error_parts
    from benchflow.cli._live_progress import (
        LiveEvalProgress,
        live_session,
        progress_enabled,
    )
    from benchflow.cli._shared import (
        RUN_FAILED_NEXT,
        _report_eval_result,
        console,
        print_error,
        report_stopped,
        stopped_run_result,
    )
    from benchflow.cli._termination import RunStopped, run_until_terminated
    from benchflow.cli.eval_artifacts import postprocess_eval_artifacts
    from benchflow.eval_sharding import ShardWorkerError
    from benchflow.evaluation import (
        EmptyTaskSelectionError,
        Evaluation,
        ResumeMismatchError,
    )
    from benchflow.task.discovery import resolve_task_collection_root

    task_collection_dir = resolve_task_collection_root(resolved_tasks_dir)
    if eval_config.source_provenance is None:
        from benchflow._utils.hf_datasets import load_source_sidecar

        eval_config.source_provenance = load_source_sidecar(task_collection_dir)

    evaluation: Evaluation | None = None
    started = time.time()
    try:
        if plan.request.retry_attempts is not None:
            eval_config.retry.max_retries = plan.request.retry_attempts
        if plan.request.worker_concurrency is None:
            # One Evaluation construction; the live dashboard contributes hooks +
            # a context manager on a TTY, and a null context + no hooks otherwise
            # (CI/pipes keep the plain logger stream).
            hooks: dict = {}
            run_ctx: AbstractContextManager = nullcontext()
            if progress_enabled(console):
                live = LiveEvalProgress(
                    console,
                    label=_eval_label(plan, task_collection_dir),
                    agent=eval_config.agent,
                    model=eval_config.model,
                    sandbox=eval_config.environment,
                )
                hooks = {
                    "on_plan": live.on_plan,
                    "on_task_start": live.on_task_start,
                    "on_result": live.on_result,
                }
                run_ctx = live_session(live)
            evaluation = Evaluation(
                tasks_dir=str(task_collection_dir),
                jobs_dir=plan.output_jobs_dir,
                config=eval_config,
                job_name=plan.job_name,
                # eval_preflight already ran for this command.
                preflight=False,
                restart_changed=plan.request.restart_changed,
                **hooks,
            )
            with run_ctx:
                result = run_until_terminated(evaluation.run(), announce=False)
        else:
            from benchflow.eval_sharding import run_sharded_evaluation

            result = run_until_terminated(
                run_sharded_evaluation(
                    tasks_dir=task_collection_dir,
                    jobs_dir=Path(plan.output_jobs_dir),
                    config=eval_config,
                    worker_concurrency=plan.request.worker_concurrency,
                    worker_retries=plan.request.worker_retries,
                    worker_start_stagger_sec=plan.request.worker_start_stagger_sec,
                ),
                announce=False,
            )
    except RunStopped as stop:
        # Said after the live display has stopped, so the line stays last.
        if plan.request.worker_concurrency is None:
            _write_stopped_summary_out(plan.request, evaluation, stop)
            report_stopped(
                stop,
                evaluation.job_dir if evaluation is not None else None,
                since=started,
            )
        else:
            # Shards resume by running the same command again (L-A4).
            report_stopped(
                stop,
                Path(plan.output_jobs_dir) / "worker-shards",
                since=started,
                resume="run the same command again with the same --jobs-dir: "
                "each shard picks up where it stopped",
            )
        raise
    except UsageLimitError as e:
        # The job stopped on its login's usage limit: report what finished;
        # the summary names the login, the window and the reset.
        result = stopped_run_result(e)
    except EmptyTaskSelectionError as e:
        err_text, err_hint = expected_error_parts(e)
        print_error(
            err_text,
            next_step=err_hint
            or "`bench check <folder>` lists the tasks it finds; check "
            "--include and --exclude against their names",
        )
        raise typer.Exit(
            # An --include or --exclude that names no task is a usage
            # mistake (dogfood p5-8); a selection the tasks themselves
            # emptied is not.
            2 if "matches no task" in err_text else 1
        ) from None
    except ResumeMismatchError as e:
        # 2 for a named job that ran with other settings: nothing ran.
        err_text, err_hint = expected_error_parts(e)
        print_error(err_text, next_step=err_hint or RUN_FAILED_NEXT)
        raise typer.Exit(e.exit_code) from None
    except (ValueError, RuntimeError, ShardWorkerError) as e:
        err_text, err_hint = expected_error_parts(e)
        print_error(err_text, next_step=err_hint or RUN_FAILED_NEXT)
        raise typer.Exit(1) from None

    job_name = getattr(result, "job_name", None)
    job_dir = Path(plan.output_jobs_dir) / job_name if job_name else None
    postprocess_eval_artifacts(plan, task_collection_dir, eval_config, job_dir)
    _report_eval_result(result, job_dir)
    if not plan.request.matrix_trial:
        _finish_run(plan.request, result, job_dir)
    return result


def _warn_default_agent(model: str | None) -> None:
    """Print the one warning for a run that names no agent."""
    from rich.markup import escape

    from benchflow.cli._shared import err_console
    from benchflow.evaluation import default_agent_warning

    err_console.print(
        "[yellow]Warning:[/yellow] " + escape(default_agent_warning(model)),
        highlight=False,
        soft_wrap=True,
    )


def _config_file_evaluation(
    plan: "EvalPlan", *, cell: int | None = None, jobs_dir: Path | None = None
) -> "Evaluation":
    """The Evaluation a ``--config`` run runs: the YAML with the command
    line's overrides applied. The run and its ``--dry-run`` both take it, so
    the plan printed is the plan run (dogfood p5-2).

    ``cell`` picks one job of a YAML job config of several, and ``jobs_dir``
    is that trial's folder (``<jobs dir>/<cell>/trial-NN``).
    """
    import subprocess

    import yaml

    from benchflow._utils.benchmark_repos import GitCloneError
    from benchflow._utils.config import normalize_sandbox_user
    from benchflow._utils.config_keys import ConfigKeyError
    from benchflow.cli._shared import (
        print_error,
        print_hint,
    )
    from benchflow.errors import UserError
    from benchflow.evaluation import (
        Evaluation,
        default_oracle_note,
        effective_model,
    )

    req = plan.request
    config_file = req.config_file
    assert (
        config_file is not None
    )  # config-file source: build_eval_plan guarantees this
    # from_yaml + the override overlay run BEFORE j.run(); a malformed config
    # surfaces here (bad YAML, missing source.repo, a list where a mapping is
    # expected, a non-existent/invalid environment_manifest, …). Guard the
    # whole block so the CLI prints one clean error instead of ~10 distinct raw
    # tracebacks. The exception type is kept in the message for diagnosability.
    try:
        # eval_preflight runs below, once the YAML's config is known.
        # The default-agent warning waits until --agent/--model are applied:
        # from_yaml would name an agent or model this run does not use.
        j = Evaluation.from_yaml(
            config_file, preflight=False, warn_default_agent=False, job_cell=cell
        )
        if req.agent is not None:
            j._config.agent = plan.eval_agent
        else:
            j._config.agent = _normalize_eval_agent_or_exit(j._config.agent)
        if req.agent is None and j._default_oracle:
            # a YAML `tasks:` file with no `agents:` runs the oracle.
            print_hint(default_oracle_note(config_file), stderr=True)
        # An --agent on the command line names the agent even when the YAML
        # does not, so the job must not record it as defaulted.
        j._config.agent_defaulted = j._config.agent_defaulted and req.agent is None
        if req.reasoning_effort is not None:
            j._config.reasoning_effort = plan.eval_reasoning_effort
        if req.harness is not None:
            from benchflow.native_harness.harnesses import check_harness

            check_harness(plan.eval_harness, [j._config.agent])
            j._config.harness = plan.eval_harness
        if req.environment is not None:
            j._config.environment = plan.eval_environment
        if req.codex_apps_policy is not None:
            j._config.codex_apps_policy = req.codex_apps_policy
        # Every value a string, as EvaluationConfig makes them: this
        # assignment skips its __post_init__.
        from benchflow._utils.env_values import normalize_env_mapping

        j._config.agent_env = normalize_env_mapping(
            {**j._config.agent_env, **plan.parsed_env}
        )
        # The model once the command line's agent and variables are applied:
        # --model, else the file's own model, else the one the merged
        # agent_env names (ANTHROPIC_MODEL=... on --agent-env counts), else
        # the agent's default. from_yaml resolved a model the file did not
        # name against the file's agent and env alone.
        named = req.model or (j._config.model if j._config.model_given else None)
        j._config.model = effective_model(j._config.agent, named, j._config.agent_env)
        j._config.model_given = bool(named) and j._config.model is not None
        if req.reviewer is not None:
            from benchflow.review.options import ReviewerConfig

            reviewer_fields = req.reviewer.model_dump(exclude_unset=True)
            if "agent_env" in reviewer_fields:
                reviewer_fields["agent_env"] = {
                    **j._config.reviewer.agent_env,
                    **req.reviewer.agent_env,
                }
            j._config.reviewer = ReviewerConfig.model_validate(
                {
                    **j._config.reviewer.to_dict(),
                    **reviewer_fields,
                }
            )
        j._config.sandbox_user = normalize_sandbox_user(
            req.sandbox_user if req.sandbox_user_explicit else j._config.sandbox_user
        )
        if req.jobs_dir is not None or plan.job_name is not None:
            # Resolve the job folder against the CLI's --jobs-dir, not the
            # YAML's: from_yaml already picked a name under the YAML's jobs dir.
            if req.jobs_dir is not None:
                j._jobs_dir = Path(req.jobs_dir)
            # A job the file names (job_name) keeps its name.
            named = plan.job_name or (j._job_name if j._auto_job_name is None else None)
            j._job_name = named or Evaluation._resolve_job_name(
                j._jobs_dir, j._config, j._tasks_dir
            )
            # run() checks an unnamed job again once the flags below are applied.
            j._auto_job_name = None if named else j._job_name
            if j._config.job_mode == "sequential-shared":
                j.learner_store = j._load_or_init_learner_store()
        if jobs_dir is not None:
            # One trial job of a job matrix: the job name the file or
            # --job-name gives, else the trial folder's own (resumed when rerun).
            named = plan.job_name or (j._job_name if j._auto_job_name is None else None)
            j._jobs_dir = jobs_dir
            j._job_name = named or Evaluation._resolve_job_name(
                jobs_dir, j._config, j._tasks_dir
            )
            j._auto_job_name = None if named else j._job_name
            if j._config.job_mode == "sequential-shared":
                j.learner_store = j._load_or_init_learner_store()
        j._restart_changed = req.restart_changed
        if req.concurrency is not None:
            j._config.concurrency = req.concurrency
        if req.build_concurrency is not None:
            j._config.build_concurrency = req.build_concurrency
        if req.warm_pool is not None:
            j._config.warm_pool = req.warm_pool
        if req.fresh_install:
            j._config.fresh_install = True
        if req.allow_cloud_metadata:
            j._config.allow_cloud_metadata = True
        if req.prompt is not None:
            j._config.prompts = plan.eval_prompts
        if req.timeout_multiplier is not None:
            j._config.timeout_multiplier = req.timeout_multiplier
        if req.sandbox_setup_timeout is not None:
            j._config.sandbox_setup_timeout = req.sandbox_setup_timeout
        if req.base_image_override is not None:
            j._config.base_image_override = req.base_image_override
        # Never a run of more tasks, or fewer retries, than the line asks for
        # (review A, pre-existing): flags win over the file.
        if req.n_tasks is not None:
            j._config.n_tasks = req.n_tasks
        if req.sample_seed is not None:
            j._config.sample_seed = req.sample_seed
        if req.retry_attempts is not None:
            j._config.retry.max_retries = req.retry_attempts
        # Each flag on its own, only when given (review B2-M2): a policy
        # flag keeps the file's retention, and the reverse.
        if req.checkpoints:
            j._config.checkpoints = req.checkpoints
        if req.checkpoint_keep is not None:
            j._config.checkpoint_keep = req.checkpoint_keep
        if req.freeze_workspace:
            j._config.freeze_workspace = True
        if req.keep_sandbox is not None:
            from benchflow.rollout._keep_sandbox import check_backend, normalize_policy

            j._config.keep_sandbox = normalize_policy(req.keep_sandbox)
            check_backend(j._config.keep_sandbox, j._config.environment)
        if req.accept_deviations is not None:
            j._config.accept_deviations = req.accept_deviations
        if req.retry_from_checkpoint:
            j._config.retry_from_checkpoint = req.retry_from_checkpoint
        if req.retry_prompt is not None:
            j._config.retry_prompt = req.retry_prompt
        if req.retry_resume_session:
            j._config.retry_resume_session = True
        if j._config.retry_from_checkpoint and not j._config.checkpoints:
            raise ValueError(
                "--retry-from-checkpoint needs checkpoints (--checkpoints, or "
                "checkpoints: in the run config): a trial can only be retried "
                "from a checkpoint it kept"
            )
        if req.extra_instruction is not None:
            j._config.extra_instruction = req.extra_instruction
        if req.agent_idle_timeout is not None:
            j._config.agent_idle_timeout = plan.eval_agent_idle_timeout
            j._config.agent_idle_timeout_given = True
        # BenchFlow's default idle limit follows the effort the line set.
        j._config.settle_idle_timeout()
        if plan.usage_tracking_overridden:
            j._config.usage_tracking = j._config.usage_tracking.overlay(
                plan.eval_usage_tracking
            )
        if j._config_task_list is not None and (
            plan.include_tasks or plan.exclude_tasks
        ):
            # The file's `tasks:` list names the tasks (review L-R3-2).
            print_error(
                f"{config_file}: its tasks: list names the tasks to run "
                f"({', '.join(j._config_task_list)}); --include and --exclude "
                "would change that list",
                next_step="drop --include/--exclude, or edit the tasks: list",
            )
            raise typer.Exit(2)
        if plan.include_tasks:
            j._config.include_tasks = plan.include_tasks
        if plan.exclude_tasks:
            j._config.exclude_tasks = plan.exclude_tasks
        # CLI --environment-manifest wins over whatever the YAML carried
        # so an operator can override a baseline manifest without editing
        # the config file.
        if plan.eval_env_manifest is not None:
            j._config.environment_manifest = plan.eval_env_manifest
        # CLI --config / --config-override (the C axis) likewise wins over the
        # YAML's own config_override. Parsed + allowlist-validated at plan time
        # and applied per task at the rollout layer. Without this the
        # file-config path silently dropped the overlay (it threaded every
        # other override but never this one), so a --config-override on a
        # run-config file was a no-op.
        if plan.eval_config_override is not None:
            # A YAML job config's timeout multipliers stay.
            config_scales = (j._config.config_override or {}).get("timeout_multipliers")
            cli_scales = plan.eval_config_override.get("timeout_multipliers", {})
            explicit = j._config_explicit_timeout_phases
            if explicit is None:
                explicit = set(config_scales or {})
            if explicit.intersection(cli_scales):
                raise ValueError(
                    "--config-override timeout_multipliers and the file's timeout "
                    "multipliers both scale the timeouts; keep one"
                )
            j._config.config_override = {
                **plan.eval_config_override,
                **(
                    {"timeout_multipliers": {**config_scales, **cli_scales}}
                    if config_scales
                    else {}
                ),
            }
        # --integrity likewise wins over the YAML. It is a safety flag, so a
        # silent drop fails in the unsafe direction: the operator believes the
        # trials are audited and they are not.
        if req.integrity is not None:
            from benchflow.integrity.trial import normalize_integrity_mode

            j._config.integrity = normalize_integrity_mode(req.integrity)
        # --verifier-sandbox too: it only ever moves verifiers into the
        # separate sandbox, and a dropped flag would fail in the unsafe direction.
        if req.verifier_sandbox is not None:
            from benchflow.task.verifier_sandbox import normalize_verifier_sandbox

            j._config.verifier_sandbox = normalize_verifier_sandbox(
                req.verifier_sandbox
            )
        if plan.eval_budget is not None:
            j._config.budget = plan.eval_budget
        # --separate-verifier only ever strengthens isolation, so it wins over
        # a config file that leaves it off.
        from benchflow.task.verifier_sandbox import normalize_include_paths

        if plan.request.separate_verifier:
            j._config.separate_verifier = True
        if plan.request.separate_verifier_include:
            j._config.separate_verifier_include = normalize_include_paths(
                [
                    *j._config.separate_verifier_include,
                    *plan.request.separate_verifier_include,
                ]
            )
    except (subprocess.CalledProcessError, GitCloneError) as e:
        if not isinstance(e, UserError):
            raise
        # A source.repo clone/fetch failure (git exits non-zero) otherwise escapes
        # as a raw traceback — it is not a config-parse error, so give it its own
        # clean message. Mirrors the --source-repo guard's CalledProcessError catch.
        hint = getattr(e, "hint", None)
        print_error(
            f"Failed to fetch source repo for {config_file}: {e}",
            next_step=hint
            if isinstance(hint, str) and hint
            else "check source.repo and its ref in the config, and that "
            "this machine can clone it",
        )
        raise typer.Exit(1) from None
    except ConfigKeyError as e:
        # A key the run would not read, by file and line (dogfood p5-3): a
        # usage error, before anything runs.
        print_error(str(e), next_step=e.hint)
        raise typer.Exit(2) from None
    except (yaml.YAMLError, ValueError, TypeError, LookupError, OSError) as e:
        # LookupError covers missing source.repo (KeyError) and empty legacy
        # agents:/datasets: lists (IndexError). Type-mismatch cases (e.g. a
        # non-string source.repo) are converted to ValueError at the parse
        # source rather than catching AttributeError here, which would mask
        # genuine bugs.
        print_error(
            f"Invalid eval config {config_file}: {type(e).__name__}: {e}",
            next_step="fix the config; its keys are in "
            "https://github.com/benchflow-ai/benchflow/blob/main/docs/scale/configs.md",
        )
        raise typer.Exit(1) from None
    return j


def _job_plan(plan: "EvalPlan") -> Any:
    """The ``--config`` file's job plan (None for a BenchFlow run config),
    with the load errors _config_file_evaluation prints, and the overrides a
    YAML job config of several agents cannot take."""
    import yaml

    from benchflow._utils.config_keys import ConfigKeyError
    from benchflow.cli._shared import print_error
    from benchflow.job_config import job_plan as load_job_plan

    req = plan.request
    config_file = req.config_file
    assert config_file is not None
    try:
        job_plan = load_job_plan(config_file, agent=req.agent, model=req.model)
    except ConfigKeyError as e:
        print_error(str(e), next_step=e.hint)
        raise typer.Exit(2) from None
    except (yaml.YAMLError, ValueError, TypeError, LookupError, OSError):
        # _config_file_evaluation reports it, as for any run config.
        return None
    if job_plan is None:
        return None
    if job_plan.n_agents > 1:
        flags = [
            flag
            for flag, value in (("--agent", req.agent), ("--model", req.model))
            if value is not None
        ]
        if flags:
            print_error(
                f"{config_file}: its agents: list names {job_plan.n_agents} agents, "
                f"and {'/'.join(flags)} would run one agent in place of all of them",
                next_step="drop the flag, or edit the agents: list",
            )
            raise typer.Exit(2)
    if job_plan.quiet:
        # quiet: as --quiet does.
        _quiet_run()
    return job_plan


def _job_matrix_root(plan: "EvalPlan", job_plan: Any) -> Path:
    """Where a job matrix's cells go: --jobs-dir, else the file's jobs_dir."""
    req = plan.request
    return Path(req.jobs_dir) if req.jobs_dir is not None else job_plan.jobs_dir


def _run_config_file_eval(plan: "EvalPlan") -> None:
    """Apply CLI overrides onto a YAML-loaded Evaluation, then run and report it."""
    from benchflow.cli.doctor import eval_preflight

    def run_job(
        plan: "EvalPlan", j: "Evaluation", resume: str | None = None
    ) -> tuple[Any, Path | None]:
        """Run one --config job, write its artifacts and print its result;
        ``resume`` is the line a stop prints (a job matrix's own)."""
        from benchflow.agents.errors import UsageLimitError
        from benchflow.cli._errors import expected_error_parts
        from benchflow.cli._shared import (
            RUN_FAILED_NEXT,
            _report_eval_result,
            print_error,
            report_stopped,
            stopped_run_result,
        )
        from benchflow.cli._termination import RunStopped, run_until_terminated
        from benchflow.cli.eval_artifacts import postprocess_eval_artifacts
        from benchflow.evaluation import EmptyTaskSelectionError, ResumeMismatchError

        started = time.time()
        try:
            result = run_until_terminated(j.run(), announce=False)
        except RunStopped as stop:
            _write_stopped_summary_out(plan.request, j, stop)
            # The resume line, as for a plain run (L-A5).
            report_stopped(stop, j.job_dir, since=started, resume=resume)
            raise
        except UsageLimitError as e:
            result = stopped_run_result(e)
        except (EmptyTaskSelectionError, ValueError) as e:
            err_text, err_hint = expected_error_parts(e)
            print_error(err_text, next_step=err_hint or RUN_FAILED_NEXT)
            # 2 for a named job that ran with other settings: nothing ran.
            code = e.exit_code if isinstance(e, ResumeMismatchError) else 1
            raise typer.Exit(code) from None
        job_name = getattr(result, "job_name", None)
        job_dir = Path(j._jobs_dir) / job_name if job_name else None
        # The artifact flags (--run-config-out, --task-manifest-out, ...) apply to
        # a --config run too.
        postprocess_eval_artifacts(plan, j._tasks_dir, j._config, job_dir)
        _report_eval_result(result, job_dir)
        return result, job_dir

    req = plan.request
    job_plan = _job_plan(plan)
    if job_plan is not None and job_plan.is_matrix:
        _run_job_matrix(plan, job_plan, run_job)
        return
    j = _config_file_evaluation(plan)
    if j._config.agent_defaulted:
        _warn_default_agent(j._config.model)
    eval_preflight(
        sandbox=j._config.environment,
        agent=j._config.agent,
        model=j._config.model,
        agent_env=j._config.agent_env,
    )
    result, job_dir = run_job(plan, j)
    _finish_run(req, result, job_dir)


def _run_job_matrix(
    plan: "EvalPlan",
    job_plan: Any,
    run_config_job: Callable[..., tuple[Any, Path | None]],
) -> None:
    """Every job of a YAML job config: each agent with each dataset,
    n_attempts times."""
    from benchflow.cli.doctor import eval_preflight
    from benchflow.cli.eval_artifacts import run_job_matrix

    root = _job_matrix_root(plan, job_plan)
    # Each cell loads (and its overrides apply) before anything runs, and
    # each distinct agent, model and variables is checked once.
    seen: set[tuple[Any, ...]] = set()
    evaluations = _job_cell_evaluations(plan, job_plan, root)
    for j in evaluations.values():
        if j._config.agent_defaulted:
            _warn_default_agent(j._config.model)
        key = (
            j._config.environment,
            j._config.agent,
            j._config.model,
            tuple(sorted(j._config.agent_env.items())),
        )
        if key not in seen:
            seen.add(key)
            eval_preflight(
                sandbox=j._config.environment,
                agent=j._config.agent,
                model=j._config.model,
                agent_env=j._config.agent_env,
            )

    def load(cell: int, trial_dir: Path, trial_plan: "EvalPlan") -> "Evaluation":
        return _config_file_evaluation(trial_plan, cell=cell, jobs_dir=trial_dir)

    def run_job(trial_plan: "EvalPlan", j: "Evaluation", resume: str) -> Any:
        return run_config_job(trial_plan, j, resume)[0]

    run_job_matrix(plan, job_plan, root, load, run_job, _rerun_command(plan))


def _job_cell_evaluations(
    plan: "EvalPlan", job_plan: Any, root: Path
) -> dict[str, "Evaluation"]:
    """Each cell's first trial job, loaded with the command line's
    overrides, by cell. A cell that selects no task is refused before
    anything runs (review L-A3)."""
    from benchflow.cli._errors import expected_error_parts
    from benchflow.cli._shared import print_error

    out: dict[str, Evaluation] = {}
    empty: list[str] = []
    for cell in job_plan.cells:
        j = _config_file_evaluation(
            plan, cell=cell.index, jobs_dir=root / cell.alias / "trial-01"
        )
        try:
            if not j._get_task_dirs():
                empty.append(cell.alias)
        except (OSError, ValueError) as exc:
            err_text, err_hint = expected_error_parts(exc)
            print_error(err_text, next_step=err_hint or "fix the task folder")
            raise typer.Exit(1) from None
        out[cell.alias] = j
    if empty:
        print_error(
            f"{plan.request.config_file}: {len(empty)} of {len(job_plan.cells)} "
            f"cells select no task ({', '.join(empty)}), so they would run "
            "nothing",
            next_step="check --include/--exclude and each dataset's task_names "
            "against its tasks (bench check <folder> lists them)",
        )
        raise typer.Exit(1)
    return out


def _rerun_command(plan: "EvalPlan") -> str:
    """The command that resumes a whole job matrix: this run's own line
    (finished trial jobs are kept), secrets hidden; or ``bench run --config``
    with the job-folder flags when the line is not known (the Python API)."""
    import shlex

    from benchflow.cli._errors import _CLI_ARGV, _secrets

    words = _CLI_ARGV.get()
    if words:
        secrets = [s for s in _secrets(list(words)) if s]
        shown = [
            "<hidden>" if any(s in word for s in secrets) else word for word in words
        ]
        return "bench " + shlex.join(shown)
    req = plan.request
    line = ["run", "--config", str(req.config_file)]
    if req.jobs_dir is not None:
        line += ["--jobs-dir", str(req.jobs_dir)]
    if req.job_name:
        line += ["--job-name", str(req.job_name)]
    return "bench " + shlex.join(line)


def _run_source_env_eval(
    plan: "EvalPlan",
    *,
    source_env_version: str | None,
    source_env_arg: list[str] | None,
    source_env_num_examples: int,
    source_env_rollouts_per_example: int,
    source_env_max_tokens: int,
    source_env_temperature: float | None,
    source_env_sampling_arg: list[str] | None,
    source_env_verifiers_version: str | None = None,
    source_env_base_url: str | None = None,
    source_env_api_key_var: str | None = None,
) -> None:
    """Warn about ignored ACP-only flags, then run a hosted Verifiers env."""
    from rich.markup import escape

    from benchflow.cli._shared import command_arg, console, print_error
    from benchflow.evaluation import DEFAULT_AGENT
    from benchflow.hosted_env import (
        HostedEnvError,
        HostedEnvRef,
        HostedEnvRunConfig,
        parse_sampling_args,
        parse_source_env_args,
        run_hosted_env,
    )

    req = plan.request
    if plan.parsed_env:
        console.print(
            "[yellow]--agent-env is for BenchFlow ACP agents; source-env runs inherit the process environment.[/yellow]"
        )
    if plan.eval_environment != "docker":
        console.print(
            f"[yellow]--sandbox {escape(repr(plan.eval_environment))} is not used by source-env runs; "
            "the hosted Verifiers environment owns its harness/sandbox.[/yellow]"
        )
    if plan.eval_agent != DEFAULT_AGENT:
        console.print(
            f"[dim]source-env records --agent {escape(repr(plan.eval_agent))}, but executes the model endpoint through Verifiers.[/dim]"
        )
    if plan.eval_env_manifest is not None:
        console.print(
            "[yellow]--environment-manifest is for benchflow Environment-plane rollouts; "
            "the hosted Verifiers environment owns its own provisioning. Ignoring.[/yellow]"
        )
    if plan.usage_tracking_overridden:
        console.print(
            "[yellow]--usage-tracking is for BenchFlow ACP rollouts; "
            "source-env runs own their provider calls. Ignoring.[/yellow]"
        )
    if req.prompt is not None:
        console.print(
            "[yellow]--prompt is for BenchFlow ACP rollouts; "
            "source-env runs own their prompts. Ignoring.[/yellow]"
        )

    source_env = req.source_env
    assert source_env is not None  # source-env source: build_eval_plan guarantees this
    try:
        ref = HostedEnvRef.parse(source_env, version=source_env_version)
        run_result = run_hosted_env(
            HostedEnvRunConfig(
                source_env=ref,
                model=req.model or "",
                env_args=parse_source_env_args(source_env_arg),
                agent=plan.eval_agent,
                jobs_dir=Path(plan.output_jobs_dir),
                concurrency=plan.eval_concurrency,
                num_examples=source_env_num_examples,
                rollouts_per_example=source_env_rollouts_per_example,
                max_tokens=source_env_max_tokens,
                temperature=source_env_temperature,
                sampling_args=parse_sampling_args(source_env_sampling_arg),
                verifiers_version=source_env_verifiers_version,
                api_base_url=source_env_base_url,
                api_key_var=source_env_api_key_var,
            )
        )
    except HostedEnvError as e:
        print_error(
            f"{e}",
            next_step="`bench run --help` describes --source-env and its flags",
        )
        raise typer.Exit(1) from None

    console.print(f"\n[bold]Environment:[/bold] {run_result.source_env.env_uid}")
    console.print(f"[bold]Hub:[/bold] {escape(str(run_result.source_env.hub_url))}")
    console.print(
        f"[bold]Model:[/bold] {escape(str(run_result.normalized_model))}"
        + (
            f" [dim](from {run_result.model})[/dim]"
            if run_result.normalized_model != run_result.model
            else ""
        )
    )
    console.print(f"[bold]Run dir:[/bold] {escape(str(run_result.run_dir))}")
    console.print(f"[bold]Reward:[/bold] {run_result.reward}")
    if run_result.total_tool_calls is not None:
        console.print(f"[bold]Tool calls:[/bold] {run_result.total_tool_calls}")
    if run_result.error:
        print_error(
            f"Error: {run_result.error}",
            next_step=f"the run's output and logs are in "
            f"{command_arg(run_result.run_dir, '<run folder>')}",
        )
        raise typer.Exit(1)


# `bench eval create` was renamed to `bench eval run`, and both are now
# `bench run`. Keep the old name as a hidden alias so existing scripts,
# configs, and downstream repos (e.g. benchflow-ai/skillsbench) keep working;
# like every 0.8 spelling it prints one note naming `bench run` (cli/verbs.py).
eval_app.command("create", hidden=True)(eval_run)

# Back-compat for the Python symbol: `eval_create` was part of this module's
# public surface (``__all__``). Keep it as an import alias of ``eval_run`` so any
# `from benchflow.cli.main import eval_create` keeps resolving.
eval_create = eval_run


@eval_app.command("list")
# Evaluation inspection commands


def eval_list(
    jobs_dir: Annotated[
        Path | None,
        typer.Argument(help="Jobs directory to list (default: ./jobs)"),
    ] = None,
) -> None:
    """List the jobs in a jobs folder, one row each, with their scores."""
    from rich.markup import escape
    from rich.table import Table

    # None means the argument was omitted: the default ./jobs simply not existing
    # yet is a benign first-run state (exit 0). An *explicit* path that doesn't
    # exist is a typo and should fail like `eval metrics` does, so scripts don't
    # read it as success. (A literal `eval list jobs` is then treated as explicit,
    # which is correct — the user named a path.)
    from benchflow._utils.command_hints import looked_in
    from benchflow.cli._shared import JOBS_FOLDER_NEXT, console, print_error

    explicit = jobs_dir is not None
    jobs_dir = jobs_dir or Path("jobs")
    if not jobs_dir.exists():
        if not explicit:
            # Where it looked, so a run from the wrong folder shows as one
            # (DF-p1-first-run-9).
            console.print(
                f"[yellow]No jobs yet[/yellow] in {escape(looked_in(jobs_dir))}: "
                "`bench run` writes them there, or pass the jobs folder",
                highlight=False,
                soft_wrap=True,
            )
            return
        print_error(
            f"No such jobs directory: {looked_in(jobs_dir)}",
            next_step=JOBS_FOLDER_NEXT,
        )
        raise typer.Exit(2)
    if not jobs_dir.is_dir():
        # exists() is True for a file; iterdir() below would NotADirectoryError.
        print_error(f"Not a directory: {jobs_dir}", next_step=JOBS_FOLDER_NEXT)
        raise typer.Exit(2)

    from benchflow.cli._job_listing import job_rows

    rows = job_rows(jobs_dir)
    table = Table(title="Jobs")
    table.add_column("Job", style="cyan")
    table.add_column("Tasks", justify="right")
    table.add_column("Summary")
    # Memory only when a listed job has a memory score: an empty column on
    # every other job (DFR-p1-17); "—" for a job without one beside it.
    memory = any(row.memory is not None for row in rows)
    if memory:
        table.add_column("Memory")
    for row in rows:
        cells = [escape(row.label), row.tasks, row.summary]
        if memory:
            cells.append("—" if row.memory is None else f"{row.memory:.1%}")
        table.add_row(*cells)
    console.print(table)
    for row in rows:
        if row.next_step:
            console.print(
                f"[dim]{escape(row.label)}: {escape(row.next_step)}[/dim]",
                highlight=False,
                soft_wrap=True,
            )


def _tokens_row(tasks: list[Any]) -> str | None:
    """The job's token use, and where it came from."""
    known = [
        t for t in tasks if t.input_tokens is not None or t.output_tokens is not None
    ]
    if not known:
        return None
    text = (
        f"{sum(t.input_tokens or 0 for t in known):,} in, "
        f"{sum(t.output_tokens or 0 for t in known):,} out "
        f"over {len(known)} of {len(tasks)} trials"
    )
    sources = sorted({t.usage_source for t in known if t.usage_source})
    return text + (f" (from {', '.join(sources)})" if sources else "")


def _model_calls_row(job: Any) -> str | None:
    """The job's model calls from each trial's model-calls.json: how many, how
    many were cut at the output cap, refused or failed at the provider, and
    which source recorded them (trajectories/model_calls.py)."""
    import json as _json
    from collections import Counter

    from benchflow.trajectories.model_calls import SUMMARY_FILE

    totals = []
    for trial in getattr(job, "trials", None) or []:
        try:
            totals.append(_json.loads((Path(trial.path) / SUMMARY_FILE).read_text()))
        except (OSError, ValueError):
            continue
    totals = [t for t in totals if isinstance(t, dict)]
    if not totals:
        return None
    calls = sum(t.get("calls") or 0 for t in totals)
    parts = [f"{calls:,} over {len(totals)} trials"]
    cut = [
        t["truncated_calls"]
        for t in totals
        if isinstance(t.get("truncated_calls"), int)
    ]
    if cut:
        parts.append(f"{sum(cut)} cut at the output cap")
    for key, label in (
        ("refused_calls", "refused"),
        ("http_errors", "provider errors"),
    ):
        count = sum(t.get(key) or 0 for t in totals)
        if count:
            parts.append(f"{count} {label}")
    sources = Counter(
        t["source"] if isinstance(t.get("source"), str) else "mixed" for t in totals
    )
    parts.append(
        "from " + ", ".join(f"{name} ({n})" for name, n in sorted(sources.items()))
    )
    return "; ".join(parts)


# bench eval metrics' pooled figures, none of which a folder of several
# settings gets (benchflow.setting_groups).
_POOLED_FIGURES = (
    "score",
    "score_ratio",
    "score_excl_errors",
    "score_excl_errors_ratio",
    "mean_reward",
    "mean_reward_scored_only",
    "mean_reward_caveat",
    "score_of_plan",
)


def _print_setting_rows(settings: Any, summary: dict[str, Any]) -> None:
    """``bench eval metrics``' By setting block: one line per setting, its
    pass@k, and its warnings; then the mixed-settings warning."""
    from rich.markup import escape

    from benchflow.cli._shared import console, err_console

    console.print(
        f"By setting, each counted by the scoring rule {summary['scoring']['rule']}:",
        highlight=False,
        soft_wrap=True,
    )
    blocks = summary["scoring"]["settings"]["rows"]
    for row, block in zip(settings.rows, blocks, strict=True):
        console.print(f"  {escape(row.line())}", highlight=False, soft_wrap=True)
        rates = block.get("solve_rates") or {}
        at_k = [
            f"pass@{k} {value:.1%}"
            for k, value in (rates.get("pass_at_k") or {}).items()
            if value is not None
        ]
        if at_k:
            console.print(
                f"    {', '.join(at_k)}", highlight=False, soft_wrap=True, style="dim"
            )
        for warning in row.score().warnings():
            console.print(
                f"    [yellow]{escape(warning)}[/yellow]",
                highlight=False,
                soft_wrap=True,
            )
    err_console.print(
        f"[yellow]Warning:[/yellow] {escape(settings.warning() or '')}",
        highlight=False,
        soft_wrap=True,
    )


def _row_solve_rates(
    row: Any, ks: list[int] | None, solve_threshold: float | None
) -> dict[str, Any]:
    """One setting's pass@k, pass^k and solve rate (``--json``)."""
    from benchflow.cli.results import cli_wording
    from benchflow.pass_at_k import solve_rates

    rates = solve_rates(
        (t.solve_sample() for t in row.trials),
        ks=ks or None,
        solve_threshold=solve_threshold,
    ).to_dict()
    rates["caveats"] = [cli_wording(c) for c in rates.get("caveats") or []]
    return rates


@eval_app.command("metrics")
def eval_metrics(
    jobs_dir: Annotated[
        Path,
        typer.Argument(help="Jobs directory to analyze"),
    ],
    benchmark: Annotated[
        str,
        typer.Option("--benchmark", help="Benchmark name"),
    ] = "",
    agent: AgentOption = "",
    model: Annotated[
        str,
        typer.Option("--model", help="Model name"),
    ] = "",
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON"),
    ] = False,
    k: Annotated[
        list[int] | None,
        typer.Option(
            "--k",
            help=(
                "k for pass@k and pass^k (repeatable; default 1, powers of two "
                "and multiples of five, using both trial and scored-only counts)"
            ),
        ),
    ] = None,
    solve_threshold: Annotated[
        float | None,
        typer.Option(
            "--solve-threshold",
            help=(
                "Count a scored trial as solved when reward >= this value "
                "(partial credit); default: solved = passed (reward 1)"
            ),
        ),
    ] = None,
    include_controls: Annotated[
        bool,
        typer.Option(
            "--include-controls",
            help=(
                "Count control runs (oracle, empty/nop) in the solve rates and "
                "pass@k; a folder of only control runs counts them anyway"
            ),
        ),
    ] = False,
) -> None:
    """Collect and display metrics from a jobs directory.

    One row per trial, as bf.load_job counts them: a retried task of an
    Evaluation job counts once, and repeated rollouts (a bf.run_batch
    folder, the trial-NN folders of --matrix --trials) count as separate
    trials. pass@k / pass^k pool a task's trials across every job folder
    under the directory. Every figure counts every trial, an unscored one as
    a fail; with unscored trials, the scored-only figures (unscored trials
    left out, as BenchFlow 0.7's mean_reward was) follow with their own n. Control runs are left out of the solve rates
    unless --include-controls, or unless the folder holds nothing else. The
    score follows the scoring rule benchflow.score.v1, with a 95% interval
    over tasks. The trials are split by what ran: agent, harness version,
    served model, route, effort sent, task set and BenchFlow version. A
    folder of several settings gets one line per setting and no pooled
    score, since one number would mix them. When the pooled trials differ
    in agent, model or a setting bench eval compare checks, a warning names
    it and a table splits the trials by agent.
    """
    from rich.markup import escape
    from rich.table import Table

    import benchflow as bf
    from benchflow.cli._shared import JOBS_FOLDER_NEXT, console, print_error
    from benchflow.metrics import collect_metrics

    if not Path(jobs_dir).is_dir():
        # Without this, collect_metrics rglobs nothing and reports a green
        # all-zeros table with exit 0 — a silent trap for scripted collectors.
        from benchflow._utils.command_hints import looked_in

        print_error(
            f"Not a directory: {looked_in(jobs_dir)}", next_step=JOBS_FOLDER_NEXT
        )
        raise typer.Exit(2)
    if any(v < 1 for v in k or ()):
        print_error("--k must be >= 1", next_step="pass --k 1 or more")
        raise typer.Exit(2)
    m = collect_metrics(str(jobs_dir), benchmark=benchmark, agent=agent, model=model)
    summary = m.summary()
    # Absent version denotes the old scored-only metric semantics.
    summary["schema_version"] = 2
    job = None
    differences: list = []
    only_controls = False
    try:
        job = bf.load_job(jobs_dir)
        # A folder of only control runs is counted, as inspect and compare
        # count it: an oracle-stability check is what such a folder is for
        # (DF-p3-eval-runner-3).
        only_controls = bool(job.trials) and not job.agents()
        rates = job.solve_rates(
            ks=k or None,
            solve_threshold=solve_threshold,
            include_controls=include_controls or only_controls,
        )
        differences = job.pooled_differences()
    except FileNotFoundError:
        rates = None
    except ValueError as exc:
        print_error(
            str(exc),
            next_step="`bench eval metrics --help` describes --k and --solve-threshold",
        )
        raise typer.Exit(2) from None
    # The scoring rule over every trial, and the trials split by what ran
    # (benchflow.scoring, benchflow.setting_groups): over several settings
    # no pooled figure is given, only one row per setting.
    from benchflow.scoring import rule_record, score
    from benchflow.setting_groups import job_summaries, setting_rows

    settings = (
        setting_rows(job.trials, summaries=job_summaries(Path(jobs_dir), job.trials))
        if job is not None
        else None
    )
    rule = score(
        [r for row in settings.rows for r in row.records()] if settings else []
    )
    summary["scoring_rule"] = rule_record()
    settings_block: dict[str, Any] | None = (
        settings.to_dict() if settings is not None else None
    )
    summary["scoring"] = {
        **rule.to_dict(),
        "settings": settings_block,
        # Trial folders that could not be read: in no figure, and named.
        "unreadable": [str(p) for p in job.unreadable] if job is not None else [],
    }
    several = settings is not None and settings.split
    if several and settings is not None and settings_block is not None:
        for row, block in zip(settings.rows, settings_block["rows"], strict=True):
            block["solve_rates"] = _row_solve_rates(row, k, solve_threshold)
    summary["solve_rates"] = rates.to_dict() if rates is not None else None
    if summary["solve_rates"] is not None:
        from benchflow.cli.results import cli_wording

        # The CLI's JSON names its flags too, as its text does (dx-f's
        # cli_wording; DF-p1-first-run-11): the library's caveats name the
        # Python keyword.
        summary["solve_rates"]["caveats"] = [
            cli_wording(c) for c in summary["solve_rates"].get("caveats") or []
        ]
    # What the pooled trials differ in (agent, model, compare's settings),
    # and each agent's own counts when there are several (DF-p3-eval-runner-8).
    # The job's plan beside its score: only for one job
    # folder (a folder of jobs has no single plan), from its evaluation.json
    # or summary.json; None when unknown.
    summary["planned"] = summary["ran"] = summary["score_of_plan"] = None
    plan: tuple[int, int, int] | None = None  # planned, ran, passed tasks
    if job is not None and job.evaluation is not None:
        recorded = (job.summary or {}).get("planned")
        planned_tasks = job.planned_tasks
        planned_n = (
            len(planned_tasks)
            if planned_tasks is not None
            else recorded
            if isinstance(recorded, int)
            else None
        )
        if planned_n:
            ran_tasks = {t.task_name for t in job.trials}
            passed_tasks = {t.task_name for t in job.trials if t.passed}
            plan = (planned_n, len(ran_tasks), len(passed_tasks))
            summary["planned"] = planned_n
            summary["ran"] = len(ran_tasks)
            summary["score_of_plan"] = len(passed_tasks) / planned_n
    summary["settings_differ"] = [
        {"setting": d.setting, "task": d.task, "values": d.values} for d in differences
    ]
    groups = (
        job.denominators_by(("agent", "model"), include_controls=True)
        if job is not None
        else []
    )
    # Per agent and model, unless one of them ran several settings: that row
    # would pool them (efforts, routes, versions), so the By setting lines
    # stand instead.
    pools = settings is not None and any(
        len(
            {
                row.setting.key()
                for row in settings.rows
                if (row.setting.agent, row.trials[0].model)
                == (g.key["agent"], g.key["model"])
            }
        )
        > 1
        for g in groups
    )
    summary["by_agent"] = (
        [
            {
                "agent": g.key["agent"],
                "model": g.key["model"],
                "trials": g.denominators.attempted,
                "passed": g.denominators.passed,
                "pass_rate": g.denominators.pass_rate_attempted,
            }
            for g in groups
        ]
        if len(groups) > 1 and not pools
        else []
    )
    if several and settings is not None:
        # One number over several settings would mix them (mixed settings):
        # each row has its own figures under scoring.settings.rows.
        withheld = settings.warning()
        for key in _POOLED_FIGURES:
            summary[key] = None
        summary["headline_withheld"] = withheld
        summary["solve_rates"] = None
        summary["solve_rates_withheld"] = withheld
        summary["scoring"] = {
            "rule": rule.rule,
            "withheld": withheld,
            "settings": settings_block,
            "unreadable": summary["scoring"]["unreadable"],
        }
        rates = None
    if differences and not output_json:
        from benchflow.cli._shared import err_console

        def shown(d) -> str:
            values = " vs ".join(repr(v) for v in d.values)
            if d.task is None:
                return f"{d.setting} ({', '.join(str(v) for v in d.values)})"
            return f"{d.setting} (on {d.task}: {values})"

        err_console.print(
            "[yellow]Warning:[/yellow] "
            + escape(
                "these trials differ in "
                + ", ".join(shown(d) for d in differences)
                + ", so the pooled figures mix them; "
                + (
                    "the By agent table below splits them by agent, and "
                    if summary["by_agent"]
                    else ""
                )
                + "`bench eval metrics <job folder>` reads one job"
            ),
            highlight=False,
            soft_wrap=True,
        )

    if output_json:
        # Plain echo: Rich wraps long lines, which breaks strings in the JSON.
        typer.echo(json.dumps(summary, indent=2))
        return

    table = Table(title=f"Results: {escape(str(jobs_dir))}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")

    table.add_row("Total", str(summary["total"]))
    table.add_row("Passed", f"[green]{summary['passed']}[/green]")
    table.add_row("Failed", f"[red]{summary['failed']}[/red]")
    table.add_row("Errored", f"[yellow]{summary['errored']}[/yellow]")
    broken = summary.get("integration_failures") or {}
    if broken.get("total"):
        causes = ", ".join(f"{k} {v}" for k, v in broken["by_cause"].items())
        table.add_row(
            "Integration failures",
            f"[yellow]{broken['total']}[/yellow] (unscored, in Errored: {escape(causes)})",
        )
    # Every trial counted, then the figures over scored trials only (0.7's
    # mean_reward rule) when some trial is unscored (otherwise they are the same numbers).
    scored = summary["passed"] + summary["failed"]
    both = summary["total"] > scored and not several
    if several and settings is not None:
        table.add_row(
            "Score",
            f"[yellow]none for the whole folder: it holds {len(settings.counted())} "
            "settings, and one number would mix them (By setting, below)[/yellow]",
        )
    else:
        if settings is not None and settings.rows:
            table.add_row("Setting", escape(settings.rows[0].setting.label()))
        table.add_row(
            "Score",
            f"[bold]{summary['score']}[/bold] ({summary['passed']} of "
            f"{summary['total']} trials); {escape(rule.interval_text())}",
        )
        table.add_row(
            "Scoring rule",
            f"{rule.rule}: every finished trial counts, and one with an error or "
            "no score counts 0",
        )
    if both:
        # The scoring rule's error-excluding figure, over its own n: a
        # refusal stays in as a failure, so it is named in the count.
        refused = rule.outcomes.get("refused", 0)
        table.add_row(
            "Score, scored only",
            f"{summary['score_excl_errors']} ({rule.passed} of {rule.n_excl_errors} "
            + ("scored or refused trials)" if refused else "scored trials)")
            if rule.n_excl_errors
            else "n/a (no scored trial)",
        )
    if summary["mean_reward"] is not None:
        table.add_row(
            "Mean reward",
            f"{summary['mean_reward']:.3f} (over {summary['mean_reward_n']} trials)",
        )
    if both and summary["mean_reward"] is not None:
        table.add_row(
            "Mean reward, scored only",
            f"{summary['mean_reward_scored_only']:.3f} (over "
            f"{summary['mean_reward_scored_only_n']} scored trials)"
            if summary["mean_reward_scored_only"] is not None
            else "n/a (no scored trial)",
        )
    if summary.get("memory_score") is not None:
        scored = (summary.get("memory") or {}).get("scored", 0)
        table.add_row(
            "Memory score",
            f"{summary['memory_score']:.1%} ({scored}/{summary['total']})",
        )
    if plan is not None and plan[0] > plan[1]:
        # Planned tasks a budget, a usage limit or an interrupt kept from
        # running: the figures above are over what ran. A trial that started
        # and wrote no result is interrupted, or running while a live process
        # holds the job (DFR-p3-10); the rest never started.
        started = len(job.unfinished()[0]) if job is not None else 0
        never = max(plan[0] - plan[1] - started, 0)
        live = job is not None and job.running() is not None
        parts = [f"{plan[0]} tasks", f"{plan[1]} ran"]
        if started:
            parts.append(f"{started} {'running' if live else 'interrupted'}")
        if never:
            parts.append(f"{never} not started{' yet' if live else ''}")
        table.add_row("Planned", f"[yellow]{', '.join(parts)}[/yellow]")
        if not several:
            table.add_row(
                "Score of plan",
                f"{plan[2] / plan[0]:.1%} ({plan[2]} of {plan[0]} planned tasks "
                "passed)",
            )
    table.add_row("Avg tool calls", f"{summary['avg_tool_calls']:.1f}")
    table.add_row("Avg duration", f"{summary['avg_duration_sec']:.0f}s")
    tokens = _tokens_row(m.tasks)
    if tokens:
        table.add_row("Tokens", escape(tokens))
        if job is not None:
            # report.md's cost line: says which figures are estimates.
            from benchflow.jobs import _cost_line

            table.add_row("Cost", escape(_cost_line(job.trials).removeprefix("Cost: ")))
    calls = _model_calls_row(job)
    if calls:
        table.add_row("Model calls", escape(calls))
    if rates is not None:
        table.add_row("Success rule", escape(rates.success_rule))
        for label, value in rates.rows():
            table.add_row(escape(label), escape(value))

    console.print(table)
    if summary["by_agent"]:
        split = Table(title="By agent")
        for column in ("Agent", "Model", "Trials", "Passed", "Score"):
            split.add_column(column)
        for row in summary["by_agent"]:
            rate = row["pass_rate"]
            split.add_row(
                escape(str(row["agent"])),
                escape(str(row["model"] or "no model")),
                str(row["trials"]),
                str(row["passed"]),
                "n/a" if rate is None else f"{rate:.1%}",
            )
        console.print(split)
    if several and settings is not None:
        _print_setting_rows(settings, summary)
    elif rule.n:
        for warning in rule.warnings():
            console.print(
                f"[yellow]Score:[/yellow] {escape(warning)}",
                highlight=False,
                soft_wrap=True,
            )
        if settings is not None and settings.mixed:
            # One setting now, but a restart set earlier trials aside.
            from benchflow.cli._shared import err_console

            err_console.print(
                f"[yellow]Warning:[/yellow] {escape(settings.warning() or '')}",
                highlight=False,
                soft_wrap=True,
            )
    for warning in settings.effort_warnings() if settings is not None else []:
        console.print(
            f"[yellow]Warning:[/yellow] {escape(warning)}",
            highlight=False,
            soft_wrap=True,
        )
    if job is not None and job.unreadable:
        from benchflow.jobs import unreadable_warning

        console.print(
            f"[yellow]Warning:[/yellow] {escape(unreadable_warning(job.unreadable))}",
            highlight=False,
            soft_wrap=True,
        )
    from benchflow.cli.results import cli_wording

    caveats = [summary["mean_reward_caveat"]] if summary["mean_reward_caveat"] else []
    caveats += rates.caveats if rates is not None else []
    if rates is not None and only_controls and not include_controls:
        caveats.append(
            "Every trial is a control run (oracle, empty/nop), so they are counted."
        )
    for caveat in caveats:
        console.print(f"[dim]{escape(cli_wording(caveat))}[/dim]")

    if summary["passed_tasks"]:
        console.print(
            f"\n[green]Passed:[/green] {escape(', '.join(summary['passed_tasks']))}"
        )
    if summary["failed_tasks"]:
        # A task with several trials is listed under each outcome it had.
        console.print(
            f"[red]Failed:[/red] {escape(', '.join(summary['failed_tasks']))}"
        )
    if summary["errored_tasks"]:
        console.print(
            f"[yellow]Errors:[/yellow] {escape(', '.join(summary['errored_tasks']))}"
        )
    if summary["error_breakdown"]:
        console.print(
            f"[yellow]Error breakdown:[/yellow] {escape(str(summary['error_breakdown']))}"
        )
        from benchflow._utils.scoring import REPORTING_CATEGORIES
        from benchflow.failures import cause_of, describe

        # Each trial's own cause: one category can have several, each with
        # its own fix (a freezer the provider or the Engine cannot give).
        causes = {}
        for trial in job.trials if job is not None else ():
            if trial.raw.get("error_category") in REPORTING_CATEGORIES:
                cause = cause_of(trial.raw)
                if cause is not None:
                    causes.setdefault(cause.key, cause)
        if not causes:
            for category in summary["error_breakdown"]:
                cause = describe(category)
                if category in REPORTING_CATEGORIES and cause is not None:
                    causes[category] = cause
        for key, cause in causes.items():
            console.print(
                f"{escape(key)} — next: {escape(cause.next_step)} "
                f"(bench explain {escape(cause.code)})"
            )


@eval_app.command("view")
def eval_view(
    rollout_dir: Annotated[
        # A str, not a Path: hf:// specs must arrive verbatim — Path
        # normalization would collapse the double slash into hf:/.
        str,
        typer.Argument(
            help=(
                "Rollout directory, job directory, trajectory JSONL file, or "
                "an hf://<org>/<dataset> trajectory dataset "
                "(optionally with /subpath)"
            )
        ),
    ],
    port: Annotated[int, typer.Option(help="Server port")] = 8888,
    confirm: Annotated[
        bool,
        typer.Option(
            "--confirm",
            help=(
                "Show an Approve & submit / Not this one bar on the page; "
                "print DECISION: approved|rejected and exit 0 on approve, "
                "3 on reject."
            ),
        ),
    ] = False,
    redaction_summary: Annotated[
        str | None,
        typer.Option(
            "--redaction-summary",
            help=(
                "Masked-secret breakdown shown in the --confirm bar, e.g. "
                "'2 API keys, 1 bearer token' (lift it from "
                "`bench traj upload PATH --dry-run`). Display-only; no effect "
                "without --confirm."
            ),
        ),
    ] = None,
) -> None:
    """View a trial or session trajectory in the browser."""
    from benchflow.trajectories import viewer

    decision = viewer.serve(
        rollout_dir, port, confirm=confirm, redaction_summary=redaction_summary
    )
    # Exit-code contract for --confirm: 0 approved, 3 rejected. 3 is chosen
    # so a rejection never collides with the CLI's existing error exits
    # (1 = errors, 2 = Typer usage errors).
    if decision == "rejected":
        raise typer.Exit(3)


# `bench view` is `bench eval view` (cli/verbs.py): the same function, listed
# under Runs in `bench --help`.
app.command("view", rich_help_panel=_RUNS)(eval_view)


# Names this module imported eagerly before its commands became lazy. Code that
# imported them from here (``from benchflow.cli.main import console``) still
# gets the same objects, imported on first use.
_FORMER_IMPORTS: dict[str, str] = {
    "__version__": "benchflow",
    "escape": "rich.markup",
    "Table": "rich.table",
    "normalize_sandbox_user": "benchflow._utils.config",
    "parse_agent_spec": "benchflow.agents.registry",
    "LiveEvalProgress": "benchflow.cli._live_progress",
    "live_session": "benchflow.cli._live_progress",
    "progress_enabled": "benchflow.cli._live_progress",
    "_apply_dotenv_to_process_env": "benchflow.cli._shared",
    "_parse_agent_env": "benchflow.cli._shared",
    "_report_eval_result": "benchflow.cli._shared",
    "console": "benchflow.cli._shared",
    "err_console": "benchflow.cli._shared",
    "print_error": "benchflow.cli._shared",
    "stopped_run_result": "benchflow.cli._shared",
    "run_until_terminated": "benchflow.cli._termination",
    "register_adopt_deprecated": "benchflow.cli.adopt",
    "register_eval_adopt": "benchflow.cli.adopt",
    "register_agent": "benchflow.cli.agent",
    "register_eval_branch": "benchflow.cli.branch",
    "register_eval_branches": "benchflow.cli.branch",
    "register_continue": "benchflow.cli.continue_cmd",
    "eval_preflight": "benchflow.cli.doctor",
    "register_doctor": "benchflow.cli.doctor",
    "register_eval_smoke": "benchflow.cli.doctor",
    "register_embodied": "benchflow.cli.embodied",
    "register_environment": "benchflow.cli.environment",
    "postprocess_eval_artifacts": "benchflow.cli.eval_artifacts",
    "run_matrix_eval": "benchflow.cli.eval_artifacts",
    "register_eval_lift": "benchflow.cli.eval_lift",
    "register_eval_bundle": "benchflow.cli.eval_bundle",
    "register_eval_regrade": "benchflow.cli.eval_regrade",
    "register_hub": "benchflow.cli.hub",
    "register_monitor": "benchflow.cli.monitor",
    "register_eval_score": "benchflow.cli.rescore",
    "register_eval_results": "benchflow.cli.results",
    "register_eval_resume": "benchflow.cli.results",
    "register_review": "benchflow.cli.review",
    "reviewer_from_cli": "benchflow.cli.reviewer_options",
    "reviewer_idle_timeout": "benchflow.cli.reviewer_options",
    "register_sandbox": "benchflow.cli.sandbox",
    "register_skills": "benchflow.cli.skills",
    "register_tasks": "benchflow.cli.tasks",
    "register_train": "benchflow.cli.train",
    "register_traj": "benchflow.cli.traj",
    "EvalCreateRequest": "benchflow.eval_plan",
    "EvalPlanError": "benchflow.eval_plan",
    "build_eval_plan": "benchflow.eval_plan",
    "DEFAULT_AGENT": "benchflow.evaluation",
    "effective_model": "benchflow.evaluation",
    "extra_install_hint": "benchflow._utils.sandbox_providers",
}


def __getattr__(name: str) -> Any:
    module = _FORMER_IMPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)


def main() -> None:
    """The ``bench`` / ``benchflow`` console script (see ``benchflow.cli._errors``)."""
    from benchflow.cli._errors import run_cli

    if os.environ.get("BENCHFLOW_DAEMON"):
        import sys

        from benchflow._daemon import try_attach

        code = try_attach(sys.argv[1:])
        if code is not None:
            raise SystemExit(code)

    run_cli(app)


if __name__ == "__main__":
    main()
