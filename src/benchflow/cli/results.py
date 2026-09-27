"""``bench eval inspect`` and ``bench eval compare``: the CLI face of
``bf.load_job`` / ``bf.load_trial`` / ``bf.compare``.

Both call the Python functions directly; ``--json`` prints the versioned
documents of ``benchflow.job_export`` (docs/reference/json-export.md), the
same bytes ``Job.to_json()`` / ``Comparison.to_json()`` produce. Warnings go
to stderr so ``--json`` output stays machine-readable.
"""

from __future__ import annotations

import glob
import json
import warnings
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import typer
from rich.markup import escape
from rich.table import Table

from benchflow.cli._shared import console, err_console, print_error
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


def _cli_wording(message: str) -> str:
    """bf.compare's messages name Python keywords; say the CLI flags instead."""
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
    return message.replace("in by:", "in --by:").replace("in vary:", "in --vary:")


def _rate(d) -> str:
    pr = d.pass_rate_scored
    return (
        f"{d.passed}/{d.scored} scored passed"
        + (f" ({pr:.0%})" if pr is not None else "")
        + f", {d.attempted} attempted, {d.assessment_errors} verifier errors, "
        f"{d.unscored} unscored, {d.controls_excluded} control runs left out"
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
            print_error("--attempts is best or all")
            raise typer.Exit(2)
        keys = tuple(k.strip() for k in (by or "").split(",") if k.strip())
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
        except (FileNotFoundError, ValueError) as exc:
            print_error(str(exc))
            raise typer.Exit(2) from None
        try:
            groups = (
                job.denominators_by(keys, include_controls=include_controls)
                if keys
                else []
            )
        except ValueError as exc:
            print_error(str(exc).replace("in by", "in --by"))
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
        headline = _rate(job.denominators(include_controls=counted))
        if only_controls and not include_controls:
            headline += " (only control runs, so they are counted)"
        console.print(
            f"{escape(str(job.path))} ({job.kind}): {headline}",
            highlight=False,
            soft_wrap=True,
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
        for trial in job.trials:
            cells = [
                escape(trial.task_name),
                "" if trial.reward is None else f"{trial.reward:g}",
                trial.execution,
                trial.assessment,
                trial.control or "",
                escape(trial.result.agent or ""),
                escape(trial.result.model or ""),
            ]
            if priced:
                cells.append("" if trial.cost_usd is None else f"${trial.cost_usd:.4f}")
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
                "--on-mismatch is warn, raise or ignore; --attempts is best or all"
            )
            raise typer.Exit(2)
        mode = cast(Literal["best", "all"], attempts)
        try:
            a = bf.load_job(_paths(job_a), attempts=mode)
            b = bf.load_job(_paths(job_b), attempts=mode)
        except (FileNotFoundError, ValueError) as exc:
            print_error(str(exc))
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
            print_error(_cli_wording(str(exc)))
            raise typer.Exit(1 if "differ" in str(exc) else 2) from None
        for warning in caught:
            if issubclass(warning.category, UserWarning):
                err_console.print(
                    f"[yellow]Warning:[/yellow] {escape(_cli_wording(str(warning.message)))}",
                    highlight=False,
                    soft_wrap=True,
                )
        _emit(
            json.dumps(result.to_json_dict(), indent=2, allow_nan=False)
            if as_json
            else result.to_markdown().rstrip("\n"),
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
                help="KEY=VALUE for the agent (repeatable); values are never stored in the job",
            ),
        ] = None,
        concurrency: Annotated[
            int | None,
            typer.Option("--concurrency", help="Override the job's concurrency"),
        ] = None,
    ) -> None:
        """Finish an interrupted job from its folder: rerun only the unfinished tasks.

        Reads the tasks directory and config the job recorded in evaluation.json
        (agent_env keys only, so pass their values again with --agent-env), then
        runs the tasks with no finished result. A job that is still running is
        refused. Exit codes as for bench eval run; 2 when the folder is not a job.
        """
        from benchflow.cli._shared import (
            _exit_if_evaluation_had_errors,
            _parse_agent_env,
            _report_eval_result,
        )
        from benchflow.evaluation import Evaluation

        overrides: dict[str, Any] = (
            {"concurrency": concurrency} if concurrency is not None else {}
        )
        try:
            evaluation = Evaluation.resume(
                job_dir, agent_env=_parse_agent_env(agent_env) or None, **overrides
            )
        except (FileNotFoundError, ValueError) as exc:
            print_error(str(exc))
            raise typer.Exit(2) from None
        try:
            result = run_until_terminated(evaluation.run())
        except (RuntimeError, ValueError) as exc:
            print_error(str(exc))
            raise typer.Exit(1) from None
        _report_eval_result(result, Path(job_dir))
        _exit_if_evaluation_had_errors(result)
