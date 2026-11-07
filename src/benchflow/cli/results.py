"""``bench eval inspect`` and ``bench eval compare``: the CLI face of
``bf.load_job`` / ``bf.load_trial`` / ``bf.compare``.

Both call the Python functions directly; ``--json`` prints the versioned
documents of ``benchflow.job_export`` (docs/train/formats.md#schema-versions), the
same bytes ``Job.to_json()`` / ``Comparison.to_json()`` produce. Warnings go
to stderr so ``--json`` output stays machine-readable.
"""

from __future__ import annotations

import glob
import json
import time
import warnings
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import typer
from rich.markup import escape
from rich.table import Table

from benchflow.cli._shared import JOBS_FOLDER_NEXT, console, err_console, print_error
from benchflow.cli._termination import run_until_terminated


def _paths(spec: str) -> list[Path]:
    """A path, or every folder a glob matches (one side of a paired run)."""
    if any(ch in spec for ch in "*?["):
        matches = sorted(glob.glob(spec))
        if not matches:
            raise FileNotFoundError(
                f"no trial (a folder with result.json) matches {spec}"
            )
        return [Path(m) for m in matches]
    return [Path(spec)]


def _emit(text: str, out: Path | None) -> None:
    if out is None:
        # Plain print keeps JSON unwrapped and unstyled.
        typer.echo(text)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    err_console.print(f"Wrote {escape(str(out))}", highlight=False)


def cli_wording(message: str) -> str:
    """bf.compare's and bf.load_job's messages name Python keywords; say the
    CLI flags instead. Every read command (inspect, compare, metrics) prints
    library text through this."""
    import re

    message = message.replace("bf.compare: ", "")
    message = message.replace(
        "Pass vary=(...) to declare an intended difference, or on_mismatch='raise' "
        "to refuse.",
        "Pass --vary SETTING (repeatable) to declare an intended difference, or "
        "--on-mismatch raise to refuse.",
    )
    message = re.sub(
        r"Pass vary=\(([^)]*)\)",
        lambda m: (
            "Pass "
            + " ".join(
                f"--vary {n.strip().strip(chr(39))}"
                for n in m.group(1).split(",")
                if n.strip()
            )
        ),
        message,
    )
    message = message.replace("by=('agent', 'model')", "--by agent,model")
    message = message.replace("include_controls=True", "--include-controls")
    return message.replace("in by:", "in --by:").replace("in vary:", "in --vary:")


def _rate(d, *, controls_counted: bool = False) -> str:
    # Every attempted run counted first (an unscored one as a fail), then the
    # scored runs only, each with its n. ``controls_counted``: the trials are
    # control runs and are the count (a job of only control runs).
    from benchflow.scoring import no_score_text

    pa, pr = d.pass_rate_attempted, d.pass_rate_scored
    controls = (
        ", only control runs, so they are counted"
        if controls_counted
        else f", {d.controls_excluded} control runs left out"
    )
    # The runs with no score in the scoring rule's classes, the words the
    # run's own Outcomes line uses (scoring.no_score_text).
    return (
        f"{d.passed}/{d.attempted} attempted passed"
        + (f" ({pa:.0%})" if pa is not None else "")
        + f", {d.passed}/{d.scored} scored passed"
        + (f" ({pr:.0%})" if pr is not None else "")
        + ", "
        + no_score_text(
            d.outcomes,
            errored_notes=[
                f"{d.integration_failures} integration failure"
                f"{'' if d.integration_failures == 1 else 's'}: the agent did nothing"
            ]
            if d.integration_failures
            else [],
        )
        + controls
    )


def _print_scoring(job: Any) -> None:
    """The job's score by the scoring rule, with its interval over tasks and
    its warnings; one line per setting instead when its trials ran several
    (benchflow.setting_groups), with the mixed-settings warning."""
    from benchflow.setting_groups import job_summaries, setting_rows

    settings = setting_rows(job.trials, summaries=job_summaries(job.path, job.trials))
    rows = settings.counted()
    if not rows:
        return
    for warning in settings.effort_warnings():
        err_console.print(
            f"[yellow]Warning:[/yellow] {escape(warning)}",
            highlight=False,
            soft_wrap=True,
        )
    if job.unreadable:
        from benchflow.jobs import unreadable_warning

        err_console.print(
            f"[yellow]Warning:[/yellow] {escape(unreadable_warning(job.unreadable))}",
            highlight=False,
            soft_wrap=True,
        )
    if not settings.split:
        figures = rows[0].score()
        console.print(
            f"  Score by {figures.rule}: {figures.score or 0.0:.1%} ({figures.passed} "
            f"of {figures.n} trials; {escape(figures.interval_text())}), "
            f"{figures.n_tasks} task{'' if figures.n_tasks == 1 else 's'}",
            highlight=False,
            soft_wrap=True,
        )
        for text in (figures.model_refusals_text(), figures.budget_stops_text()):
            if text:
                console.print(f"  {escape(text)}", highlight=False, soft_wrap=True)
        for warning in figures.warnings():
            console.print(
                f"  [yellow]{escape(warning)}[/yellow]", highlight=False, soft_wrap=True
            )
        if settings.mixed:
            err_console.print(
                f"[yellow]Warning:[/yellow] {escape(settings.warning() or '')}",
                highlight=False,
                soft_wrap=True,
            )
        return
    console.print(
        f"  By setting, each counted by {rows[0].score().rule}:",
        highlight=False,
        soft_wrap=True,
    )
    for row in rows:
        console.print(f"    {escape(row.line())}", highlight=False, soft_wrap=True)
    err_console.print(
        f"[yellow]Warning:[/yellow] {escape(settings.warning() or '')}",
        highlight=False,
        soft_wrap=True,
    )


def unfinished_line(job: Any, where: str) -> str | None:
    """What a job that stopped early left undone, and the command that
    finishes it (DF-p3-eval-runner-6), or, while a live process runs the
    job, what is still running (DFR-p3-10); None when nothing is left."""
    interrupted, not_started = job.unfinished()
    if not interrupted and not not_started:
        return None
    planned = job.planned_tasks
    finished = (
        f"{len({t.task_name for t in job.trials} & set(planned))} of "
        f"{len(planned)} planned tasks finished, "
        if planned
        else ""
    )
    pid = job.running()
    if pid is not None:
        # A live process holds the job's lock: its started trials are
        # running, not interrupted, and there is nothing to resume yet.
        parts = [f"{len(interrupted)} running"] if interrupted else []
        if not_started:
            parts.append(f"{len(not_started)} not started yet")
        return f"Running (process {pid}): {finished}{', '.join(parts)}"
    parts = [f"{len(interrupted)} interrupted"] if interrupted else []
    if not_started:
        parts.append(f"{len(not_started)} not started")
    return (
        f"Not finished: {finished}{', '.join(parts)}; "
        f"`bench eval resume {where}` runs them"
    )


def register_eval_results(eval_app: typer.Typer) -> None:
    """Attach ``inspect`` and ``compare`` to ``bench eval``."""

    @eval_app.command("inspect")
    def inspect(
        path: Annotated[
            str,
            typer.Argument(
                help="A job folder, a trial folder, a results.jsonl file, or a glob of folders"
            ),
        ],
        as_json: Annotated[
            bool,
            typer.Option(
                "--json",
                help="Print the benchflow.job (or benchflow.trial) JSON document",
            ),
        ] = False,
        trajectories: Annotated[
            bool,
            typer.Option("--trajectories", help="Include trajectories in the JSON"),
        ] = False,
        verifier: Annotated[
            bool,
            typer.Option(
                "--verifier/--no-verifier",
                help="Include verifier output (test logs) in the JSON",
            ),
        ] = True,
        attempts: Annotated[
            str,
            typer.Option(
                "--attempts", help="best (one per task, as resume picks) or all"
            ),
        ] = "best",
        out: Annotated[
            Path | None, typer.Option("--out", help="Write the output to this file")
        ] = None,
        by: Annotated[
            str | None,
            typer.Option(
                "--by",
                help="Also print denominators per group, e.g. agent,model",
            ),
        ] = None,
        include_controls: Annotated[
            bool,
            typer.Option(
                "--include-controls",
                help="Count control runs (oracle, empty) in the denominators",
            ),
        ] = False,
        csv_out: Annotated[
            Path | None,
            typer.Option("--csv", help="Also write one CSV row per trial to this file"),
        ] = None,
    ) -> None:
        """Read a finished job or trial: denominators, one row per trial, or --json.

        The same data as bf.load_job / bf.load_trial; control runs (oracle,
        empty) are counted apart. Reads files only.
        """
        import benchflow as bf

        if attempts not in ("best", "all"):
            print_error(
                f"--attempts {attempts!r} is not a choice",
                next_step="pass --attempts best or --attempts all",
            )
            raise typer.Exit(2)
        keys = tuple(k.strip() for k in (by or "").split(",") if k.strip())
        from benchflow.jobs import InterruptedJobError

        try:
            roots = _paths(path)
            single_trial = len(roots) == 1 and (roots[0] / "result.json").is_file()
            if single_trial and as_json:
                trial = bf.load_trial(roots[0])
                document = trial.to_json_dict(
                    include_trajectory=trajectories, include_verifier=verifier
                )
                _emit(json.dumps(document, indent=2, allow_nan=False), out)
                return
            job = bf.load_job(roots, attempts=cast(Literal["best", "all"], attempts))
        except InterruptedJobError as exc:
            # A job, stopped before its first result: resume it.
            print_error(str(exc), next_step=exc.hint)
            raise typer.Exit(1) from None
        except (FileNotFoundError, ValueError) as exc:
            print_error(str(exc), next_step=JOBS_FOLDER_NEXT)
            raise typer.Exit(2) from None
        try:
            groups = (
                job.denominators_by(keys, include_controls=include_controls)
                if keys
                else []
            )
        except ValueError as exc:
            print_error(
                str(exc).replace("in by", "in --by"),
                next_step="pass --by with the keys the message lists",
            )
            raise typer.Exit(2) from None
        if csv_out is not None:
            job.to_csv(csv_out)
            err_console.print(f"Wrote {escape(str(csv_out))}", highlight=False)
        if as_json:
            document = job.to_json_dict(
                include_trajectories=trajectories, include_verifier=verifier
            )
            _emit(json.dumps(document, indent=2, allow_nan=False), out)
            return
        only_controls = bool(job.trials) and not job.agents()
        counted = include_controls or only_controls
        denominators = job.denominators(include_controls=counted)
        headline = _rate(
            denominators,
            controls_counted=only_controls and not include_controls,
        )
        from benchflow.setting_groups import job_summaries, setting_rows

        split = setting_rows(
            job.trials, summaries=job_summaries(job.path, job.trials)
        ).counted()
        if len(split) > 1:
            # One rate over several settings would mix them: each setting's
            # line follows (mixed settings).
            headline = (
                f"{sum(len(row.trials) for row in split)} trials in {len(split)} "
                "settings, so no pooled score: one line per setting below"
            )
            if failures := denominators.integration_failures:
                # A count of trials whose agent did nothing, not a score.
                headline += (
                    f"; {failures} integration failure"
                    f"{'' if failures == 1 else 's'}: the agent did nothing"
                )
        console.print(
            f"{escape(str(job.path))} ({job.kind}): {headline}",
            highlight=False,
            soft_wrap=True,
        )
        _print_scoring(job)
        from benchflow._utils.command_hints import command_arg

        left = unfinished_line(job, command_arg(job.path, "<job dir>"))
        if left is not None:
            console.print(
                f"[yellow]{escape(left)}[/yellow]", highlight=False, soft_wrap=True
            )
        for group in groups:
            label = " ".join(f"{k}={v}" for k, v in group.key.items())
            console.print(
                f"  {escape(label)}: {_rate(group.denominators)}",
                highlight=False,
                soft_wrap=True,
            )
        # Names fold onto several lines instead of being cut: the task is the
        # column a reader needs whole. Short headers leave it the room.
        table = Table()
        table.add_column("Task", overflow="fold", ratio=3)
        for column in ("Reward", "Exec", "Assess", "Control"):
            table.add_column(column, no_wrap=True)
        table.add_column("Agent", overflow="fold", ratio=1)
        table.add_column("Model", overflow="fold", ratio=1)
        # A Cost column with no costs (subscription logins report none) only
        # takes width from Task.
        priced = any(t.cost_usd is not None for t in job.trials)
        if priced:
            table.add_column("Cost", no_wrap=True)
        # Why a trial did nothing, when its agent integration broke
        # (agent_auth, agent_install, …); the column appears only when needed.
        broken = any(t.integration_failure is not None for t in job.trials)
        if broken:
            table.add_column("Cause", no_wrap=True)
        for trial in job.trials:
            cells = [
                escape(trial.task_name),
                "" if trial.reward is None else f"{trial.reward:g}",
                # "broken" keeps the column narrow; the Cause column says why.
                "broken"
                if trial.execution == "integration_failed"
                else trial.execution,
                trial.assessment,
                trial.control or "",
                escape(trial.result.agent or ""),
                escape(trial.result.model or ""),
            ]
            if priced:
                # "~": an estimate from the agent's own session log.
                estimate = trial.result.price_source == "agent_session_log"
                cells.append(
                    ""
                    if trial.cost_usd is None
                    else f"{'~' if estimate else ''}${trial.cost_usd:.4f}"
                )
            if broken:
                failure = trial.integration_failure
                cells.append(escape(str(failure.get("cause"))) if failure else "")
            table.add_row(*cells)
        console.print(table)
        if out is not None:
            _emit(job.to_json(include_verifier=verifier), out)

    @eval_app.command("compare")
    def compare(
        job_a: Annotated[
            str,
            typer.Argument(help="Side A: a job folder, a trial, or a glob of folders"),
        ],
        job_b: Annotated[str, typer.Argument(help="Side B")],
        as_json: Annotated[
            bool,
            typer.Option("--json", help="Print the benchflow.comparison JSON document"),
        ] = False,
        labels: Annotated[
            tuple[str, str] | None,
            typer.Option("--labels", help="Names for the two sides"),
        ] = None,
        vary: Annotated[
            list[str] | None,
            typer.Option(
                "--vary",
                help="A setting the comparison is about (repeatable), e.g. model",
            ),
        ] = None,
        on_mismatch: Annotated[
            str,
            typer.Option(
                "--on-mismatch",
                help="warn (default), raise (exit 1) or ignore an undeclared setting difference",
            ),
        ] = "warn",
        include_controls: Annotated[
            bool,
            typer.Option(
                "--include-controls", help="Keep control runs (oracle, empty)"
            ),
        ] = False,
        attempts: Annotated[
            str, typer.Option("--attempts", help="best or all")
        ] = "best",
        out: Annotated[
            Path | None, typer.Option("--out", help="Write the output to this file")
        ] = None,
        by: Annotated[
            str | None,
            typer.Option(
                "--by",
                help="Pair on these keys as well as the task, e.g. agent,model",
            ),
        ] = None,
        k: Annotated[
            list[int] | None,
            typer.Option("--k", help="k for pass@k and pass^k (repeatable)"),
        ] = None,
        solve_threshold: Annotated[
            float | None,
            typer.Option(
                "--solve-threshold",
                help="Solved = reward >= this value (partial credit); default: passed",
            ),
        ] = None,
    ) -> None:
        """Pair two jobs by task: rewards, deltas, fair denominators, setting checks.

        The same code as bf.compare. Each paired task is checked for comparable
        settings (task digest, model, harness, dataset, reasoning effort,
        sandbox, sandbox user, timeout, agent variables, prompts); declare the
        settings the comparison is about with --vary. Reads files only.
        """
        import benchflow as bf

        if on_mismatch not in ("warn", "raise", "ignore") or attempts not in (
            "best",
            "all",
        ):
            print_error(
                "--on-mismatch is warn, raise or ignore; --attempts is best or all",
                next_step="pass one of those values",
            )
            raise typer.Exit(2)
        mode = cast(Literal["best", "all"], attempts)
        from benchflow.jobs import InterruptedJobError

        try:
            a = bf.load_job(_paths(job_a), attempts=mode)
            b = bf.load_job(_paths(job_b), attempts=mode)
        except InterruptedJobError as exc:
            print_error(str(exc), next_step=exc.hint)
            raise typer.Exit(1) from None
        except (FileNotFoundError, ValueError) as exc:
            print_error(str(exc), next_step=JOBS_FOLDER_NEXT)
            raise typer.Exit(2) from None
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", UserWarning)
                result = bf.compare(
                    a,
                    b,
                    labels=tuple(labels) if labels else None,
                    vary=tuple(vary or ()),
                    on_mismatch=cast(Literal["warn", "raise", "ignore"], on_mismatch),
                    include_controls=include_controls,
                    by=tuple(
                        key.strip() for key in (by or "").split(",") if key.strip()
                    ),
                    ks=k or None,
                    solve_threshold=solve_threshold,
                )
        except ValueError as exc:
            print_error(
                cli_wording(str(exc)),
                next_step="`bench eval compare --help` describes --vary, --by and "
                "--on-mismatch",
            )
            raise typer.Exit(1 if "differ" in str(exc) else 2) from None
        for warning in caught:
            if issubclass(warning.category, UserWarning):
                err_console.print(
                    f"[yellow]Warning:[/yellow] {escape(cli_wording(str(warning.message)))}",
                    highlight=False,
                    soft_wrap=True,
                )
        _emit(
            json.dumps(result.to_json_dict(), indent=2, allow_nan=False)
            if as_json
            else cli_wording(result.to_markdown().rstrip("\n")),
            out,
        )


def register_eval_resume(eval_app: typer.Typer) -> None:
    """Attach ``bench eval resume``: the CLI face of ``Evaluation.resume``."""

    @eval_app.command("resume")
    def resume(
        job_dir: Annotated[Path, typer.Argument(help="The job folder to finish")],
        agent_env: Annotated[
            list[str] | None,
            typer.Option(
                "--agent-env",
                help=(
                    "KEY=VALUE for the agent (repeatable); values that look "
                    "secret are never stored in the job"
                ),
            ),
        ] = None,
        concurrency: Annotated[
            int | None,
            typer.Option("--concurrency", help="Override the job's concurrency"),
        ] = None,
        restart_changed: Annotated[
            bool,
            typer.Option(
                "--restart-changed",
                help=(
                    "When the job's trials ran with other settings or task files "
                    "than this resume, set them aside in <job>/.set-aside/ and run "
                    "them again (default: refuse)"
                ),
            ),
        ] = False,
    ) -> None:
        """Finish an interrupted job from its folder: rerun only the unfinished tasks.

        Reads the tasks directory and config the job recorded in evaluation.json
        (no agent_env value that looks secret, so pass those again with
        --agent-env), then runs the tasks with no finished result. A job that is still running is
        refused. Exit codes as for bench eval run; 2 when the folder is not a job,
        or when its trials ran with other settings or task files than this
        resume has (an --agent-env value, an edited task): --restart-changed
        runs them again.
        """
        from benchflow.agents.errors import UsageLimitError
        from benchflow.cli._errors import expected_error_parts
        from benchflow.cli._shared import (
            RUN_FAILED_NEXT,
            _exit_if_evaluation_had_errors,
            _parse_agent_env,
            _report_eval_result,
            stopped_run_result,
        )
        from benchflow.evaluation import Evaluation, ResumeMismatchError

        overrides: dict[str, Any] = (
            {"concurrency": concurrency} if concurrency is not None else {}
        )
        try:
            evaluation = Evaluation.resume(
                job_dir,
                agent_env=_parse_agent_env(agent_env) or None,
                restart_changed=restart_changed,
                **overrides,
            )
        except (FileNotFoundError, ValueError) as exc:
            message, hint = expected_error_parts(exc)
            print_error(
                message,
                next_step=hint
                or "pass the job folder of a run that stopped before it finished",
            )
            raise typer.Exit(2) from None
        from benchflow.cli._shared import report_stopped
        from benchflow.cli._termination import RunStopped

        started = time.time()
        try:
            result = run_until_terminated(evaluation.run(), announce=False)
        except RunStopped as stop:
            # The resume line, as for a plain run (L-A5).
            report_stopped(stop, evaluation.job_dir, since=started)
            raise
        except UsageLimitError as exc:
            result = stopped_run_result(exc)
        except (RuntimeError, ValueError) as exc:
            message, hint = expected_error_parts(exc)
            print_error(message, next_step=hint or RUN_FAILED_NEXT)
            # 2 for trials that ran with other settings: nothing ran.
            code = exc.exit_code if isinstance(exc, ResumeMismatchError) else 1
            raise typer.Exit(code) from None
        _report_eval_result(result, Path(job_dir))
        _exit_if_evaluation_had_errors(result)
