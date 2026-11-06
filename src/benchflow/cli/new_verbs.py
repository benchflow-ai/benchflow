"""``bench run``, ``bench results`` and ``bench agents``: Rust 0.8.2's spelling.

The other verbs of the set (``bench new``, ``check``, ``import``, ``branch``,
``view``, ``regrade``, ``export``, ``doctor``) are the Python commands that
already did the job, listed under a new name (``cli/main.py``'s layout).
These three take Rust's arguments, so they are commands of their own, each
running the Python command that does the work:

* ``bench run [TARGET]`` is ``bench eval run`` with Rust's flags; without
  ``-a`` it runs the oracle, and it refuses a flag only a model agent reads
  (``-m``, ``--effort``, ``--harness``) without ``-a``. Every
  other ``bench eval run`` flag is a ``bench run`` flag too (``--config``,
  ``--allow-task-env``, budgets, matrices: :class:`RunCommand`), and the
  eval spellings of its own flags (``--job-name``, ``--freeze-workspace``)
  are aliases, so one verb takes every run the docs show (dogfood p5-4).
* ``bench results [JOBS]...`` is ``bench eval list`` with no job or a folder
  of jobs, ``bench eval inspect`` with one job, ``bench eval compare`` with two.
* ``bench agents [NAME]`` is ``bench agent list``, or ``bench agent show NAME``.

Each imports what it runs only when it runs (``bench --help`` stays light).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Any, ClassVar

import click
import typer
from typer.core import TyperCommand, TyperGroup

from benchflow._utils.sandbox_providers import providers_phrase
from benchflow.cli._lazy import UsageError, get_current_context
from benchflow.cli._options import HARNESS_HELP


def _usage(message: str, next_step: str) -> UsageError:
    """A usage error (exit 2) that names its next step, as every bench error does."""
    from benchflow.cli.main import UsageErrorWithNext

    return UsageErrorWithNext(message, next_step, ctx=get_current_context(silent=True))


def _command(module: str, register: str, *path: str) -> click.Command:
    """The command ``register`` of ``module`` builds at ``path``, built fresh."""
    import importlib

    host = typer.Typer()
    getattr(importlib.import_module(module), register)(host)
    command: click.Command = typer.main.get_group(host)
    for part in path:
        assert isinstance(command, TyperGroup)
        command = command.commands[part]
    return command


def _call(command: click.Command, **params: Any) -> None:
    """Run ``command``'s function with ``params``, its own defaults for the rest.

    Not ``ctx.invoke(command, ...)``: from typer 0.26 (its own click) that
    calls the command object as a whole program, not with these parameters.
    """
    from inspect import unwrap

    assert command.callback is not None, command.name
    unwrap(command.callback)(**params)


# A GitHub repository as a target: org/repo[/path][@ref], or its web address.
_NAME = r"[A-Za-z0-9_.-]+"
_WEB = re.compile(
    rf"https://github\.com/(?P<org>{_NAME})/(?P<repo>{_NAME}?)(?:\.git)?"
    r"(?:/tree/(?P<ref>[^/]+)(?:/(?P<path>.*?))?)?/?"
)


def parse_target(
    text: str,
) -> tuple[Path | None, tuple[str, str | None, str | None] | None]:
    """A target as (task folder, None) or (None, (repo, path, ref)).

    A folder that exists is a folder. Otherwise ``org/repo[/path][@ref]`` (or
    ``https://github.com/org/repo/tree/<ref>/<path>``) is a repository, unless
    its first part is a folder here, which makes it a mistyped folder.
    """
    local = Path(text).expanduser()
    if local.exists():
        return local, None
    web = _WEB.fullmatch(text)
    if web:
        return None, (
            f"{web['org']}/{web['repo']}",
            (web["path"] or "").strip("/") or None,
            web["ref"],
        )
    spec, at, ref = text.rpartition("@") if "@" in text else (text, "", "")
    parts = spec.strip("/").split("/")
    looks_like_repo = (
        len(parts) >= 2
        and not text.startswith((".", "/", "~"))
        and all(re.fullmatch(_NAME, p) for p in parts[:2])
        and all(parts)
        and not Path(parts[0]).exists()
    )
    if not looks_like_repo or (at and not ref):
        raise _usage(
            f"no such task folder: {text}",
            "pass a task folder, a folder of tasks, or a repository as "
            "org/repo[/path][@ref]",
        )
    return None, (f"{parts[0]}/{parts[1]}", "/".join(parts[2:]) or None, ref or None)


def idle_seconds(text: str) -> str:
    """``--idle-timeout`` as the whole seconds ``bench eval run`` takes.

    Seconds (``600``), or a number with ``s``, ``m`` or ``h`` (``90s``,
    ``10m``, ``1h``); ``0``, ``none``, ``null`` or empty turns the watchdog
    off, as on ``bench eval run``.
    """
    value = text.strip().lower()
    if value in ("", "none", "null", "0"):
        return value
    match = re.fullmatch(r"(\d+)([smh]?)", value)
    if not match:
        raise _usage(
            f"--idle-timeout {text!r} is not a duration",
            "pass seconds (600) or a number with s, m or h (90s, 10m, 1h); "
            "0, none or null turns it off",
        )
    scale = {"": 1, "s": 1, "m": 60, "h": 3600}[match[2]]
    return str(int(match[1]) * scale)


def _split_patterns(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    from benchflow.cli.verbs import glob_bracket_end

    patterns: list[str] = []
    for value in values:
        start = index = 0
        while index < len(value):
            if value[index] == "[":
                closing = glob_bracket_end(value, index)
                if closing is not None:
                    index = closing + 1
                    continue
            if value[index] == ",":
                if pattern := value[start:index].strip():
                    patterns.append(pattern)
                start = index + 1
            index += 1
        if pattern := value[start:].strip():
            patterns.append(pattern)
    return patterns


def run(
    targets: Annotated[
        list[str] | None,
        typer.Argument(
            help=(
                "A task folder, a folder of tasks, or a GitHub repository "
                "(org/repo, org/repo/path or org/repo/path@ref); one per run"
            ),
            show_default=False,
        ),
    ] = None,
    agent: Annotated[
        str | None,
        typer.Option(
            "--agent",
            "-a",
            help=(
                "The agent: a name `bench agents` lists, or nop (does nothing) "
                "(default: oracle, the task's reference solution)"
            ),
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="The model, in the agent's own naming"),
    ] = None,
    effort: Annotated[
        str | None,
        typer.Option(
            "--effort",
            "-e",
            "--reasoning-effort",
            help="Reasoning effort, when the agent exposes one (low, medium, high, max, ...)",
        ),
    ] = None,
    harness: Annotated[
        str | None,
        typer.Option("--harness", help=HARNESS_HELP),
    ] = None,
    env: Annotated[
        list[str] | None,
        typer.Option(
            "--env",
            "--agent-env",
            help="Environment variable for the agent, KEY=VALUE (repeatable)",
        ),
    ] = None,
    task_skills: Annotated[
        bool,
        typer.Option(
            "--task-skills/--no-task-skills",
            help=(
                "Install the task's own skills (environment/skills) for the agent; "
                "left out by default, as in BenchFlow 0.8"
            ),
        ),
    ] = False,
    dataset: Annotated[
        str | None,
        typer.Option(
            "--dataset",
            "-d",
            help="A registry dataset, name@version, instead of a target",
        ),
    ] = None,
    registry: Annotated[
        str | None,
        typer.Option(
            "--registry",
            help="The registry for --dataset: a URL or a file (default: the skillsbench registry)",
        ),
    ] = None,
    include: Annotated[
        list[str] | None,
        typer.Option(
            "--include",
            "-i",
            help=(
                "Only tasks whose names match: a name, or a glob pattern such as "
                "'hard-*' (comma-separated or repeated)"
            ),
        ),
    ] = None,
    exclude: Annotated[
        list[str] | None,
        typer.Option(
            "--exclude",
            "-x",
            help="Skip tasks whose names match (same patterns as --include)",
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            "-n",
            "--n-tasks",
            help=(
                "Run at most N tasks: the first N by name, or a seeded random "
                "sample with --sample-seed"
            ),
        ),
    ] = None,
    sandbox: Annotated[
        str | None,
        typer.Option("--sandbox", "-s", help=f"Where trials run: {providers_phrase()}"),
    ] = None,
    concurrency: Annotated[
        int | None,
        typer.Option(
            "--concurrency",
            "-j",
            help=(
                "Trials running at a time (on Docker, up to --warm-pool more are "
                "prepared ahead)"
            ),
        ),
    ] = None,
    warm_pool: Annotated[
        int | None,
        typer.Option(
            "--warm-pool",
            help=(
                "Docker: sandboxes prepared ahead of their trial's turn, beyond "
                "the running ones (0 to --concurrency; default 1, or 0 at -j 1)"
            ),
        ),
    ] = None,
    jobs_dir: Annotated[
        str | None,
        typer.Option(
            "--jobs-dir",
            help="Where job folders go (default: jobs); a plain rerun resumes its latest job",
        ),
    ] = None,
    timeout_multiplier: Annotated[
        float | None,
        typer.Option(
            "--timeout-multiplier",
            help="Scale every task's agent timeout, e.g. 2.0",
        ),
    ] = None,
    idle_timeout: Annotated[
        str | None,
        typer.Option(
            "--idle-timeout",
            "--agent-idle-timeout",
            help=(
                "Abort a prompt when the agent is silent this long: seconds, or "
                "90s, 10m, 1h; 0, none or null turns it off (default: 10m)"
            ),
        ),
    ] = None,
    fail_under: Annotated[
        float | None,
        typer.Option(
            "--fail-under",
            help="Exit 1 when the pass rate over the planned tasks is below this (0 to 1)",
        ),
    ] = None,
    save_outputs: Annotated[
        bool,
        typer.Option(
            "--save-outputs",
            "--freeze-workspace",
            help=(
                "Save each trial's final workspace before the verifier runs, so "
                "`bench regrade` can score it again"
            ),
        ),
    ] = False,
    extra_instruction: Annotated[
        str | None,
        typer.Option(
            "--extra-instruction", help="Text appended to every task's instruction"
        ),
    ] = None,
    verifier_sandbox: Annotated[
        str | None,
        typer.Option(
            "--verifier-sandbox",
            help=(
                "task or separate. separate runs every task's verifier in a "
                "fresh sandbox that gets only the workspace and declared "
                "artifacts; a task it cannot run there refuses the whole run "
                "before anything starts (default: task, each task's own choice)"
            ),
        ),
    ] = None,
    job: Annotated[
        str | None,
        typer.Option(
            "--job",
            "--job-name",
            help="Job name under --jobs-dir (an existing job is resumed)",
        ),
    ] = None,
    fresh: Annotated[
        bool,
        typer.Option(
            "--fresh",
            help="Start a new job instead of resuming the latest one in --jobs-dir",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the plan and run nothing"),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            "-q",
            help="Print only warnings, errors and the final score",
        ),
    ] = False,
) -> None:
    """Run an agent on tasks and score them; without -a, the oracle.

    The oracle is the task's reference solution: no model and no key. Rerunning
    the same command resumes its job. A run config (--config) is the whole run:
    its agent runs unless -a names another, and flags win over its keys.
    """
    from benchflow.cli.main import eval_run

    # The bench eval run flags given (RunCommand puts them here).
    more: dict[str, Any] = dict(get_current_context().meta.get(_MORE_KEY, {}))
    sources = [n for n in ("tasks_dir", "source_repo", "source_env") if n in more]
    whole_run = "config_file" in more or "matrix" in more or bool(sources)

    if len(targets or []) > 1:
        raise _usage(
            f"bench run takes one target; {len(targets or [])} were given",
            "run each target on its own, or put the tasks in one folder",
        )
    target = (targets or [None])[0]
    if agent is not None and not agent.strip():
        raise _usage(
            "--agent must not be empty or whitespace-only",
            "pass -a <agent> (`bench agents` lists them); to run the oracle, "
            "omit -a and model-only flags",
        )
    if agent is not None:
        # A padded name would skip the registry check as a raw ACP command.
        agent = agent.strip()
    # Tasks alone run the oracle by default; a run config, a hosted
    # environment or a matrix names its own agents (the config's agent runs),
    # so -m, --effort and --harness apply to those (review F1).
    oracle_default = not any(n in more for n in ("config_file", "source_env", "matrix"))
    if agent is None and oracle_default:
        given = [
            f"-m {model}" if model else None,
            f"--effort {effort}" if effort else None,
            f"--harness {harness}" if harness else None,
        ]
        first = next((g for g in given if g), None)
        if first is not None:
            flag = first.split(" ", 1)[0]
            what = "names a model" if flag == "-m" else "is for model agents"
            raise _usage(
                f"{first} {what}, but no agent runs it: without -a, bench run "
                "runs the oracle (the task's reference solution)",
                f"add -a <agent> (`bench agents` lists them), or drop {flag} to "
                "run the oracle",
            )
    if target is not None and dataset is not None:
        raise _usage(
            "bench run takes a target or --dataset, not both",
            "drop the target to run the dataset, or drop --dataset",
        )
    if target is not None and sources:
        flag = "--" + sources[0].replace("_", "-")
        raise _usage(
            f"bench run takes a target or {flag}, not both",
            f"drop {flag} (the target names the tasks) or drop the target",
        )
    if target is None and dataset is None and not whole_run:
        raise _usage(
            "bench run needs a target: a task folder, a folder of tasks, or a repository",
            "bench run tasks/ (or --dataset <name>@<version>, or --config run.yaml)",
        )
    if task_skills and "skill_mode" in more:
        raise _usage(
            "--task-skills and --skill-mode both set the skill mode",
            "keep one: --task-skills is --skill-mode with-skill",
        )
    # One value set for both verbs (dogfood p5-9): native. headless was this
    # command's own name for it and still works.
    if harness == "headless":
        harness = "native"
    if harness not in (None, "acp", "native"):
        raise _usage(f"--harness {harness} is not a harness", "pass acp or native")
    kwargs: dict[str, Any] = {}
    if target is not None:
        folder, repo = parse_target(target)
        if folder is not None:
            kwargs["tasks_dir"] = folder
        else:
            assert repo is not None
            kwargs["source_repo"], kwargs["source_path"], kwargs["source_ref"] = repo
    if "skill_mode" not in more:
        more["skill_mode"] = "with-skill" if task_skills else "no-skill"
    eval_run(
        **kwargs,
        **more,
        dataset=dataset,
        registry=registry,
        agent=("oracle" if oracle_default else None) if agent is None else agent,
        model=model,
        reasoning_effort=effort,
        harness=harness,
        agent_env=env,
        include=_split_patterns(include),
        exclude=_split_patterns(exclude),
        n_tasks=limit,
        environment=sandbox,
        concurrency=concurrency,
        warm_pool=warm_pool,
        jobs_dir=jobs_dir,
        timeout_multiplier=timeout_multiplier,
        agent_idle_timeout=idle_seconds(idle_timeout)
        if idle_timeout is not None
        else None,
        fail_under=fail_under,
        freeze_workspace=save_outputs,
        extra_instruction=extra_instruction,
        verifier_sandbox=verifier_sandbox,
        job_name=job,
        fresh=fresh,
        dry_run=dry_run,
        quiet=quiet,
    )


def _is_trial(folder: Path) -> bool:
    # A trial writes config.json when it starts and result.json when it ends.
    return (folder / "result.json").is_file() or (folder / "config.json").is_file()


def _has_job_record(folder: Path) -> bool:
    # A job writes evaluation.json when it starts and summary.json when it ends
    # (older jobs only the latter); its jobs folder gets a copy of summary.json.
    return (folder / "evaluation.json").is_file() or (folder / "summary.json").is_file()


def _is_export(folder: Path) -> bool:
    """Rows in a results.jsonl with no trial anywhere below, as inspect reads them.

    A vf-eval or hosted-env output dir, or a copied-out results file, keeps
    its rows there and has no trial folders. `load_job` reads results.jsonl
    only when it finds no trial, and a finished job writes a job-level
    results.jsonl next to its trials, so one trial anywhere below means this
    is not an export (a folder of finished jobs is never pooled into one).
    """
    found = False
    for root, _dirs, files in os.walk(folder):
        if root != str(folder) and _is_trial(Path(root)):
            return False
        found = found or "results.jsonl" in files
    return found


def _jobs_folder(path: str) -> bool:
    """A folder of jobs, which `bench eval list` lists, rather than one job.

    Read from the layout, not from whether the jobs finished: a folder whose
    jobs were interrupted (no summary.json) is still a folder of jobs, never
    one job pooling every agent's trials into one pass rate. A child with a
    job record makes it a folder of jobs; a child that is a trial makes it a
    job. With neither, a folder with no job record of its own is one job when
    it holds a results.jsonl and no trial anywhere below (an export, which
    `bench eval inspect` reads) and a folder of jobs otherwise; a folder with
    a job record is one job unless it has a trial two levels down.
    """
    folder = Path(path)
    if not folder.is_dir() or _is_trial(folder):
        return False
    # A hidden folder is never a job of it: a job 0.9 resumed from 0.7 keeps
    # its old files, summary.json included, in .pre-0.8/.
    children = [
        child
        for child in folder.iterdir()
        if child.is_dir() and not child.name.startswith(".")
    ]
    if any(_has_job_record(child) for child in children):
        return True
    if any(_is_trial(child) for child in children):
        return False
    if not _has_job_record(folder):
        return not _is_export(folder)
    return any(
        _is_trial(grandchild)
        for child in children
        for grandchild in child.iterdir()
        if grandchild.is_dir()
    )


def results(
    jobs: Annotated[
        list[str] | None,
        typer.Argument(
            help=(
                "Job folders, or a folder of jobs (default: ./jobs); one job is "
                "read trial by trial, two are compared task by task"
            ),
            show_default=False,
        ),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help=(
                "Print the benchflow.job document (one job) or the "
                "benchflow.comparison document (two)"
            ),
        ),
    ] = False,
    attempts: Annotated[
        str,
        typer.Option(
            "--attempts", help="best (one attempt per task, as resume picks) or all"
        ),
    ] = "best",
    include_controls: Annotated[
        bool,
        typer.Option(
            "--include-controls",
            help="Count control runs (oracle, empty) in the denominators",
        ),
    ] = False,
    by: Annotated[
        str | None,
        typer.Option(
            "--by",
            help="One job: also count per group, e.g. agent,model; two: pair on these keys too",
        ),
    ] = None,
) -> None:
    """Summarize jobs; with two, compare them.

    With no job, or a folder of jobs, it lists the jobs (`bench eval list`);
    with one, it reads the job (`bench eval inspect`); with two, it pairs them
    by task (`bench eval compare`).
    """
    names = jobs or []
    if len(names) > 2:
        raise _usage(
            f"bench results reads one job or compares two; {len(names)} were given",
            "bench results <job> <job>",
        )
    shared = {
        "as_json": as_json,
        "attempts": attempts,
        "include_controls": include_controls,
        "by": by,
    }
    if len(names) == 2:
        compare = _command("benchflow.cli.results", "register_eval_results", "compare")
        _call(compare, job_a=names[0], job_b=names[1], **shared)
        return
    if len(names) == 1 and not _jobs_folder(names[0]):
        inspect = _command("benchflow.cli.results", "register_eval_results", "inspect")
        _call(inspect, path=names[0], **shared)
        return
    if as_json or by or include_controls or attempts != "best":
        raise _usage(
            "--json, --attempts, --include-controls and --by read one job or compare two",
            "name the job: bench results <job> --json",
        )
    from benchflow.cli.main import eval_list

    eval_list(Path(names[0]) if names else None)


def agents(
    name: Annotated[
        str | None,
        typer.Argument(help="Show this agent in detail", show_default=False),
    ] = None,
) -> None:
    """List the agents, or show one in detail."""
    command = _command(
        "benchflow.cli.agent", "register_agent", "agent", "show" if name else "list"
    )
    if name:
        _call(command, name=name)
    else:
        _call(command)


_MORE_KEY = "benchflow.bench_run.more"
# Help panel of the bench eval run flags bench run takes as they are.
MORE_RUN_PANEL = "More run settings"


def _eval_run_function() -> Any:
    """The function ``bench eval`` registered as ``run`` (whatever
    ``main.eval_run`` names now: a test may replace that one)."""
    from benchflow.cli.main import eval_app

    (info,) = [c for c in eval_app.registered_commands if c.name == "run"]
    assert info.callback is not None
    return info.callback


def _eval_run_command() -> click.Command:
    """``bench eval run``'s command, built as its group builds it."""
    host = typer.Typer()
    host.command("run")(_eval_run_function())
    host.command("unused")(lambda: None)  # two commands: get_group keeps names
    group = typer.main.get_group(host)
    assert isinstance(group, TyperGroup)
    return group.commands["run"]


class RunCommand(TyperCommand):
    """``bench run``: its own options, then every other ``bench eval run`` option.

    An eval option whose parameter ``bench run`` already has (``--agent``,
    ``--concurrency``, ...) or calls by another name (``--job-name`` is
    ``--job``, an alias there) is not added again. The rest keep their names,
    help and defaults, in their own help panel; the ones given go to ``bench
    eval run`` as given (:func:`run` reads them from the context).
    """

    # bench eval run parameters bench run has under its own name.
    SAME: ClassVar[dict[str, str]] = {
        "reasoning_effort": "effort",
        "environment": "sandbox",
        "agent_idle_timeout": "idle_timeout",
        "freeze_workspace": "save_outputs",
        "job_name": "job",
        "agent_env": "env",
        "n_tasks": "limit",
    }
    # bench eval run flags that --help shows too: the README's first command
    # uses the source flags (dogfood p5-1), and --config is a whole run.
    SHOWN: ClassVar[frozenset[str]] = frozenset(
        {"config_file", "source_repo", "source_path"}
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        import copy

        have = {p.name for p in self.params}
        taken = {o for p in self.params for o in (*p.opts, *p.secondary_opts)}
        # The hidden ones (--help-all shows them), and all that pass through.
        self.more: list[str] = []
        self.given: list[str] = []
        # Typer's own conversions of eval run's values (a Path, a list), as
        # its command applies them before it calls eval_run.
        _, self.convert, _ = (
            typer.main.get_params_convertors_ctx_param_name_from_function(
                _eval_run_function()
            )
        )
        for param in _eval_run_command().params:
            if param.name in have or param.name in self.SAME:
                continue
            names = (*param.opts, *param.secondary_opts)
            assert not set(names) & taken, (param.name, names)
            extra = copy.copy(param)
            assert extra.name is not None
            if extra.name in self.SHOWN:
                # In the main panel, beside the target they stand in for.
                self.params.append(extra)
                self.given.append(extra.name)
                continue
            if hasattr(extra, "rich_help_panel"):
                setattr(extra, "rich_help_panel", MORE_RUN_PANEL)  # noqa: B010
            # --help stays the short page; --help-all lists these too.
            if isinstance(extra, click.Option):
                extra.hidden = True
            self.params.append(extra)
            self.more.append(extra.name)
            self.given.append(extra.name)
        self.params.append(
            click.Option(
                ["--help-all"],
                is_flag=True,
                is_eager=True,
                expose_value=False,
                callback=self._help_all,
                help="Show every flag, the run settings --help leaves out included",
            )
        )
        epilog = (
            f"{len(self.more)} more run settings (run configs, budgets, hosted "
            "environments, checkpoints, publishing): bench run --help-all"
        )
        # --help-all shows every setting, so its page leaves this line out.
        self.all_epilog = self.epilog
        self.epilog = f"{self.epilog}\n\n{epilog}" if self.epilog else epilog

    def _help_all(self, ctx: click.Context, _param: Any, value: bool) -> None:
        if not value or ctx.resilient_parsing:
            return
        for param in self.params:
            if isinstance(param, click.Option) and param.name in self.more:
                param.hidden = False
        self.epilog = self.all_epilog
        click.echo(ctx.get_help(), color=ctx.color)
        ctx.exit()

    def invoke(self, ctx: click.Context) -> Any:
        given: dict[str, Any] = {}
        for name in self.given:
            value = ctx.params.pop(name, None)
            source = ctx.get_parameter_source(name)
            if source not in (None, click.core.ParameterSource.DEFAULT):
                convert = self.convert.get(name)
                given[name] = convert(value) if convert is not None else value
        ctx.meta[_MORE_KEY] = given
        return super().invoke(ctx)


def register_run(app: typer.Typer) -> None:
    app.command("run", cls=RunCommand)(run)


def register_results(app: typer.Typer) -> None:
    app.command("results")(results)


def register_agents(app: typer.Typer) -> None:
    app.command("agents")(agents)
