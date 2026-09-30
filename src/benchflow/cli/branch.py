"""``bench eval branch``: fork an agent run at a checkpoint into labelled children.

The engine is :mod:`benchflow.branch_run`; this module parses options, prints
progress and the per-child table, and sets the exit code (0 when every child
was scored, 1 otherwise, 2 for an inconsistent request).
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, cast

import typer
from rich.markup import escape
from rich.table import Table

from benchflow.cli._options import ModelOption
from benchflow.cli._shared import (
    _apply_dotenv_to_process_env,
    _parse_agent_env,
    console,
    print_error,
)

_EXAMPLE = (
    "bench eval branch --tasks-dir tests/examples/hello-world-task "
    "--agent claude-agent-acp --model claude-sonnet-4-6 --sandbox daytona "
    '--prompt "Write draft.txt containing: Hello world" '
    "--prompt @instruction --checkpoint-after-prompt 1 "
    '--child "label=baseline" '
    '--child "label=hint,prompt=Rename draft.txt to hello.txt."'
)


def _select_tasks(tasks_dir: Path, include: list[str] | None) -> list[Path]:
    from benchflow.branch_run import BranchPlanError
    from benchflow.evaluation import _is_task_dir
    from benchflow.task.formats import detect_task_format, materialize_task_dir

    def is_task(path: Path) -> bool:
        return detect_task_format(path) is not None or _is_task_dir(path)

    if is_task(tasks_dir):
        tasks = [tasks_dir]
    else:
        tasks = sorted(p for p in tasks_dir.iterdir() if p.is_dir() and is_task(p))
    if include:
        tasks = [task for task in tasks if task.name in set(include)]
    # A folder in a registered task format runs as the native package it
    # materializes, as in bench eval run.
    natives = []
    for task in tasks:
        try:
            natives.append(materialize_task_dir(task))
        except (ValueError, RuntimeError) as exc:
            raise BranchPlanError(f"{task}: {exc}") from None
    return natives


def _policy(spec: str | None, keep: int):
    from benchflow.branch_run import BranchPlanError
    from benchflow.checkpoints import parse_checkpoint_policy

    if not spec:
        return None
    try:
        return parse_checkpoint_policy(spec, keep=keep)
    except ValueError as exc:
        raise BranchPlanError(str(exc)) from None


def _trial_header(plan: Any, source: Any, task: Path, index: int) -> str:
    """``[i/n] task: agent on sandbox, N children …``. A kept checkpoint is
    named by its id and trial, never by its snapshot ref (a capability)."""
    where = (
        f"from checkpoint {source.fork_id} of {source.trial_dir.name}"
        if source is not None
        else f"after prompt {plan.checkpoint_after}"
    )
    return escape(
        f"[{index}/{len(plan.task_paths)}] {task.name}: {plan.agent} on "
        f"{plan.sandbox}, {len(plan.children)} children {where}"
    )


def _reward(value: float | None) -> str:
    return "-" if value is None else f"{value:g}"


def _usd(cost: dict | None) -> str:
    if not cost or not cost.get("usd_known"):
        return "-"
    return f"{cost['usd']:.4f}"


def _count(usage: dict, *keys: str) -> str:
    if not usage:
        return "-"
    return f"{sum(int(usage.get(key) or 0) for key in keys):,}"


def _totals_line(view: dict) -> str:
    """Tokens by class, reported and estimated USD, sandbox-seconds, and the
    same per scored child (branch-view 1.1 ``totals``)."""
    totals = view["totals"]
    cost = totals.get("cost") or {}
    usage = totals.get("usage")
    parts = []
    if usage is not None:
        parts.append(
            f"{usage['total']:,} tokens (in {usage['input']:,} · out "
            f"{usage['output']:,} · cache read {usage['cache_read']:,} · cache "
            f"write {usage['cache_creation']:,})"
        )
    no_model = usage is not None and usage["total"] == 0
    if cost.get("usd_known"):
        parts.append(f"USD {cost['usd']:.4f} reported")
    elif no_model:
        parts.append("no model calls")
    else:
        parts.append("USD not reported (subscription or no provider price)")
    pricing = view.get("pricing")
    if totals.get("usd_estimate") is not None and not no_model and pricing:
        parts.append(
            f"estimate ~USD {totals['usd_estimate']:.4f} at list price for "
            f"{pricing['model']} ({pricing['source']})"
        )
    if cost.get("sandbox_seconds") is not None:
        parts.append(f"{cost['sandbox_seconds']:.0f} sandbox-s")
    per = totals.get("per_scored_child") or {}
    if per.get("scored"):
        each = (
            [f"{per['sandbox_seconds']:.0f} sandbox-s"]
            if per.get("sandbox_seconds") is not None
            else []
        )
        if per.get("usd") is not None:
            each.insert(0, f"USD {per['usd']:.4f}")
        elif per.get("usd_estimate") is not None and not no_model:
            each.insert(0, f"~USD {per['usd_estimate']:.4f}")
        if each:
            parts.append(f"per scored child ({per['scored']}): " + ", ".join(each))
    return "Totals: " + "; ".join(parts) + "."


def _print_cost_table(outcomes) -> None:
    """Per fork: children, V, tokens, USD, sandbox-seconds and wall seconds."""
    cost_table = Table(title="Cost per fork")
    many = len(outcomes) > 1
    columns = ("Task",) * many + (
        "Fork",
        "From",
        "Kids",
        "V",
        "Tokens",
        "USD",
        "Sandbox-s",
        "Wall-s",
    )
    for column in columns:
        cost_table.add_column(
            column,
            justify="left" if column in {"Task", "Fork", "From"} else "right",
            no_wrap=True,
        )
    unknown_usd = False
    for outcome in outcomes:
        for fork in outcome.forks:
            cost = fork.get("cost") or {}
            unknown_usd = unknown_usd or not cost.get("usd_known")
            cost_table.add_row(
                *([escape(outcome.task)] if many else []),
                escape(str(fork["id"])[:8]),
                escape(str(fork.get("from") or "-")),
                str(fork.get("children", "-")),
                _reward(fork.get("value")),
                f"{cost.get('tokens', 0):,}",
                _usd(cost),
                f"{cost.get('sandbox_seconds') or 0:.0f}",
                f"{cost.get('wall_seconds') or 0:.0f}",
            )
    if cost_table.row_count:
        console.print(cost_table)
        totals = [o.cost for o in outcomes if o.cost]
        console.print(
            f"Total: {sum(c.get('tokens') or 0 for c in totals):,} tokens, "
            f"{sum(c.get('sandbox_seconds') or 0 for c in totals):.0f} sandbox-seconds"
            + (
                "; no model calls"
                if not any(c.get("tokens") for c in totals)
                else "; USD not reported (subscription or no provider price; "
                "bench eval branches shows a list-price estimate)"
                if unknown_usd
                else f", USD {sum(c.get('usd') or 0 for c in totals):.4f}"
            ),
            highlight=False,
            soft_wrap=True,
        )


def register_eval_branch(eval_app: typer.Typer) -> None:
    """Attach ``bench eval branch`` to the ``eval`` group."""

    @eval_app.command("branch")
    def eval_branch(
        tasks_dir: Annotated[
            Path | None,
            typer.Option(
                "--tasks-dir",
                help="A task directory, or a directory of tasks (run one after another).",
            ),
        ] = None,
        agent: Annotated[
            str | None,
            typer.Option(
                "--agent",
                help="Agent name, or 'oracle' (children run the task's solve.sh). "
                "Default: the checkpoint's agent with --from-checkpoint, else claude-agent-acp.",
            ),
        ] = None,
        model: ModelOption = None,
        reasoning_effort: Annotated[
            str | None,
            typer.Option(
                "--reasoning-effort", help="Agent reasoning effort (e.g. max)"
            ),
        ] = None,
        sandbox: Annotated[
            str | None,
            typer.Option(
                "--sandbox",
                help="docker or daytona (Daytona direct mode snapshots). "
                "Default: docker, or the checkpoint's provider.",
            ),
        ] = None,
        prompt: Annotated[
            list[str] | None,
            typer.Option(
                "--prompt",
                help="Parent prompt; repeatable, sent in order. '@instruction' is "
                "the task's instruction. Default: the task's prompts.",
            ),
        ] = None,
        checkpoint_after_prompt: Annotated[
            int | None,
            typer.Option(
                "--checkpoint-after-prompt",
                help="Branch after this many parent prompts (0: before the first). "
                "Default 1; 0 for the oracle and with --from-checkpoint.",
            ),
        ] = None,
        child: Annotated[
            list[str] | None,
            typer.Option(
                "--child",
                help="One child, as label=NAME[,parent=LABEL][,prompt=TEXT|,prompt-file=PATH]; "
                "give at least two. prompt= takes the rest of the option (commas allowed). "
                "parent=LABEL forks this child from child LABEL's state after LABEL "
                "has run its prompts (a nested fork of at least two children). "
                "Without a prompt the child runs the parent's remaining prompts, or "
                "the task's prompts when the checkpoint is after the last one.",
            ),
        ] = None,
        snapshot_layers: Annotated[
            str,
            typer.Option(
                "--snapshot-layers",
                help="What the checkpoint captures: sandbox (container filesystem), "
                "environment (declared database state, needs a manifest), or both.",
            ),
        ] = "sandbox",
        parent: Annotated[
            str | None,
            typer.Option(
                "--parent",
                help="continue: restore the parent world after the children, send the "
                "remaining prompts and verify. discard: skip that restore (one fewer), "
                "leave the parent unverified. Default: continue, or discard with "
                "--from-checkpoint.",
            ),
        ] = None,
        retain_snapshots: Annotated[
            bool,
            typer.Option(
                "--retain-snapshots",
                help="Keep the checkpoint's sandbox snapshot so a later "
                "--from-checkpoint can branch from it again (otherwise deleted). "
                "`bench sandbox cleanup` removes kept Daytona snapshots once stale.",
            ),
        ] = False,
        from_checkpoint: Annotated[
            Path | None,
            typer.Option(
                "--from-checkpoint",
                help="A trial folder from an earlier `bench eval branch "
                "--retain-snapshots` run: branch again from its kept sandbox "
                "snapshot. Needs --tasks-dir holding the same task.",
            ),
        ] = None,
        fork: Annotated[
            str | None,
            typer.Option(
                "--fork",
                "--checkpoint",
                help="With --from-checkpoint: a fork id from tree.json or an automatic "
                "checkpoint as prompt:N (default: the last kept fork, else the last "
                "kept checkpoint).",
            ),
        ] = None,
        checkpoints: Annotated[
            str | None,
            typer.Option(
                "--checkpoints",
                help="Also keep a sandbox snapshot after these parent prompts "
                "(every-prompt or prompt:N[,M]) for a later --from-checkpoint.",
            ),
        ] = None,
        checkpoint_keep: Annotated[
            int,
            typer.Option(
                "--checkpoint-keep",
                min=1,
                help="Keep at most this many --checkpoints per trial.",
            ),
        ] = 3,
        resume_session: Annotated[
            bool,
            typer.Option(
                "--resume-session",
                help="Children resume the parent's agent conversation (ACP "
                "session/load) instead of starting fresh, so a child prompt can say "
                "'continue'. Needs an agent that advertises loadSession and keeps "
                "its session on disk (Claude Code), and a checkpoint after prompt 1+.",
            ),
        ] = False,
        child_retries: Annotated[
            int,
            typer.Option(
                "--child-retries",
                min=0,
                help="Retry a child this many times when it failed before its agent "
                "did anything (a provider hiccup such as a connect timeout).",
            ),
        ] = 1,
        stop_on_child_failure: Annotated[
            bool,
            typer.Option(
                "--stop-on-child-failure",
                help="Stop a fork at the first child that fails for good (default: "
                "the other children still run and the fork is partial).",
            ),
        ] = False,
        concurrency: Annotated[
            int,
            typer.Option(
                "--concurrency",
                min=1,
                help="Run up to this many children of a fork at once, each in its own "
                "sandbox created from the snapshot (implies --isolate-children).",
            ),
        ] = 1,
        isolate_children: Annotated[
            bool,
            typer.Option(
                "--isolate-children",
                help="Run each child in its own sandbox created from the snapshot "
                "instead of restoring the parent's sandbox between children. "
                "Implied by --concurrency > 1 and by nested children.",
            ),
        ] = False,
        agent_env: Annotated[
            list[str] | None,
            typer.Option("--agent-env", help="Agent env var (KEY=VALUE); repeatable"),
        ] = None,
        include: Annotated[
            list[str] | None,
            typer.Option("--include", help="Only these task names; repeatable"),
        ] = None,
        jobs_dir: Annotated[
            Path, typer.Option("--jobs-dir", help="Output directory")
        ] = Path("jobs"),
        job_name: Annotated[
            str | None,
            typer.Option("--job-name", help="Job folder name (default: branch-<time>)"),
        ] = None,
    ) -> None:
        """Fork an agent run at a checkpoint into labelled children.

        Each task runs once up to the checkpoint; the sandbox (and, when asked,
        declared environment state) is snapshotted; every --child starts from
        that snapshot with a fresh agent session (child prompts must stand
        alone unless --resume-session reloads the parent's conversation) and is
        scored by the task's verifier. The job folder has the usual layout plus, per trial, tree.json
        and branches/<fork>/children/<node>/observation.json.

        Example:

            bench eval branch --tasks-dir tests/examples/hello-world-task
            --agent claude-agent-acp --sandbox daytona
            --prompt "Write draft.txt containing: Hello world"
            --prompt @instruction --checkpoint-after-prompt 1
            --child "label=baseline" --child "label=hint,prompt=Rename draft.txt."
        """
        from benchflow import branch_run
        from benchflow.cli.doctor import eval_preflight
        from benchflow.cli.main import _normalize_eval_agent_or_exit

        try:
            source = (
                branch_run.load_checkpoint_source(from_checkpoint, fork)
                if from_checkpoint is not None
                else None
            )
            if fork is not None and source is None:
                raise branch_run.BranchPlanError("--fork needs --from-checkpoint")
            if tasks_dir is None:
                raise branch_run.BranchPlanError(
                    f"--tasks-dir is required, e.g.\n  {_EXAMPLE}"
                )
            if not tasks_dir.is_dir():
                raise branch_run.BranchPlanError(f"Not a directory: {tasks_dir}")
            tasks = _select_tasks(tasks_dir, include)
            if source is not None:
                tasks = [task for task in tasks if task.name == source.task_name]
                if not tasks:
                    raise branch_run.BranchPlanError(
                        f"--tasks-dir {tasks_dir} has no task named "
                        f"{source.task_name!r} (the checkpoint's task)"
                    )
            recorded = {}
            if from_checkpoint is not None:
                import json

                recorded = json.loads((from_checkpoint / "config.json").read_text())
            agent_name = _normalize_eval_agent_or_exit(
                agent or recorded.get("agent") or "claude-agent-acp"
            )
            parent_mode = parent or ("discard" if source is not None else "continue")
            if parent_mode not in ("continue", "discard"):
                raise branch_run.BranchPlanError("--parent is continue or discard")
            plan = branch_run.BranchPlan(
                task_paths=tasks,
                agent=agent_name,
                model=model or (recorded.get("model") if agent is None else None),
                reasoning_effort=reasoning_effort,
                sandbox=sandbox or (source.provider if source else "docker"),
                children=[branch_run.parse_child_spec(spec) for spec in (child or [])],
                checkpoint_after=checkpoint_after_prompt
                if checkpoint_after_prompt is not None
                else (0 if agent_name == "oracle" or source is not None else 1),
                snapshot_layers=branch_run.parse_snapshot_layers(snapshot_layers),
                parent_mode=cast(branch_run.ParentMode, parent_mode),
                retain_snapshots=retain_snapshots,
                prompts=prompt,
                agent_env=_parse_agent_env(agent_env),
                jobs_dir=jobs_dir,
                job_name=job_name
                or f"branch-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                source=source,
                concurrency=concurrency,
                isolate_children=isolate_children,
                checkpoints=_policy(checkpoints, checkpoint_keep),
                resume_session=resume_session,
                child_retries=child_retries,
                continue_after_child_failure=not stop_on_child_failure,
            )
            plan.validate()
        except branch_run.BranchPlanError as exc:
            print_error(str(exc))
            raise typer.Exit(2) from None
        eval_preflight(
            sandbox=plan.sandbox,
            agent=plan.agent,
            model=plan.model,
            agent_env=plan.agent_env,
        )

        _apply_dotenv_to_process_env()
        outcomes = []
        for index, task in enumerate(plan.task_paths, start=1):
            console.print(
                _trial_header(plan, source, task, index),
                highlight=False,
                soft_wrap=True,
            )
            outcomes.append(asyncio.run(branch_run.run_branch_trial(plan, task)))
        job_dir = branch_run.write_branch_job(plan, outcomes)

        table = Table(title=f"Branches: {escape(str(job_dir))}")
        for column in ("Task", "Child", "From", "Status", "Reward", "Source"):
            table.add_column(column)
        for outcome in outcomes:
            for row in outcome.children:
                table.add_row(
                    escape(outcome.task),
                    escape(str(row["label"])),
                    escape(str(row.get("parent_label") or "checkpoint")),
                    escape(str(row["status"])),
                    _reward(row["reward"]),
                    escape(str(row["reward_source"] or "-")),
                )
        console.print(table)
        _print_cost_table(outcomes)
        for outcome in outcomes:
            line = (
                f"{escape(outcome.task)}: V = {_reward(outcome.value)}, fork "
                f"{escape(str(outcome.fork_status))}, parent "
                f"{escape(str(outcome.parent_restore))}"
            )
            if plan.parent_mode == "continue":
                line += f", parent reward {_reward(outcome.parent_reward)}"
            if outcome.kept_snapshot:
                line += f", kept snapshot {escape(outcome.kept_snapshot)}"
            if outcome.error:
                line += f"\n  [red]error:[/red] {escape(outcome.error)}"
            console.print(line, highlight=False, soft_wrap=True)
        console.print(
            f"Next: [cyan]bench eval view {escape(str(job_dir))}[/cyan]",
            highlight=False,
            soft_wrap=True,
        )
        if any(
            outcome.error or outcome.fork_status != "completed" for outcome in outcomes
        ):
            raise typer.Exit(1)


def register_eval_branches(eval_app: typer.Typer) -> None:
    """Attach ``bench eval branches`` (read a job's or trial's branches)."""

    @eval_app.command("branches")
    def eval_branches(
        path: Annotated[Path, typer.Argument(help="A trial folder or a job folder.")],
        as_json: Annotated[
            bool,
            typer.Option(
                "--json",
                help="Print the benchflow.branch-view/1 documents (a JSON list, one "
                "per branched trial); see docs/reference/branch-view.md.",
            ),
        ] = False,
    ) -> None:
        """Show every fork and child of a job's (or one trial's) branches:
        request, status, reward, advantage, tokens, USD and sandbox-seconds."""
        import json

        import benchflow as bf

        try:
            views = bf.load_job(path).branch_views()
        except FileNotFoundError as exc:
            # A usage error, like inspect and compare (exit 2).
            print_error(str(exc))
            raise typer.Exit(2) from None
        if as_json:
            typer.echo(json.dumps(views, indent=2))
            return
        if not views:
            console.print("No branched trials here.")
            return
        for view in views:
            table = Table(
                title=f"{view['trial']['name']}: {view['totals']['forks']} fork(s)"
            )
            costs = Table(title="Cost per child (~ = estimate at list price)")
            for column in ("Fork", "Child", "Status", "Reward", "Adv."):
                table.add_column(column, no_wrap=True)
            for column in ("Fork", "Child", "In", "Out", "Cache", "USD", "Sandbox-s"):
                costs.add_column(
                    column,
                    no_wrap=True,
                    justify="left" if column in {"Fork", "Child"} else "right",
                )
            for fork in view["forks"]:
                where = (
                    str(fork["id"])[:8]
                    + (" retry" if fork["kind"] == "retry" else "")
                    + (f" d{fork['depth']}" if fork["depth"] > 1 else "")
                )
                for child in fork["children"]:
                    cost = child["cost"] or {}
                    usage = child.get("usage") or {}
                    label = escape(str(child["label"]))
                    table.add_row(
                        escape(where),
                        label,
                        escape(str(child["status"])),
                        _reward(child["reward"]),
                        _reward(child["advantage"]),
                    )
                    costs.add_row(
                        escape(where),
                        label,
                        *(
                            _count(usage, *keys)
                            for keys in (
                                ("n_input_tokens",),
                                ("n_output_tokens",),
                                ("n_cache_read_tokens", "n_cache_creation_tokens"),
                            )
                        ),
                        # reported, else "~" + the list-price estimate
                        f"{cost['usd']:.4f}"
                        if cost.get("usd") is not None
                        else "-"
                        if child.get("usd_estimate") is None
                        else f"~{child['usd_estimate']:.4f}",
                        f"{cost['sandbox_seconds']:.0f}"
                        if cost.get("sandbox_seconds") is not None
                        else "-",
                    )
            console.print(table)
            console.print(costs)
            console.print(_totals_line(view), highlight=False, soft_wrap=True)
