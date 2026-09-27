"""``bench train`` — training data conversion commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import typer
from rich.markup import escape

from benchflow.cli._shared import console, print_error


def _ensure_training_format(format_name: str) -> None:
    if format_name not in {"prime-sft", "trl-sft", "branch-tree"}:
        print_error("--format must be 'prime-sft', 'trl-sft' or 'branch-tree'")
        raise typer.Exit(1)


def _has_branch_trees(jobs_dir: Path) -> bool:
    try:
        return next(Path(jobs_dir).rglob("tree.json"), None) is not None
    except OSError:
        return False


def register_train(app: typer.Typer) -> None:
    """Attach the ``train`` command group to the top-level benchflow app."""
    train_app = typer.Typer(help="Training data commands.")
    app.add_typer(train_app, name="train", rich_help_panel="Core")
    run_app = typer.Typer(help="Launch training jobs.")
    train_app.add_typer(run_app, name="run")

    @train_app.command("convert")
    def train_convert(
        jobs_dir: Annotated[
            Path,
            typer.Argument(help="BenchFlow rollout or jobs directory"),
        ],
        output: Annotated[
            Path,
            typer.Option("--out", "-o", help="Output JSONL path"),
        ],
        format_name: Annotated[
            str,
            typer.Option(
                "--format",
                help="Trainer format: prime-sft, trl-sft, or branch-tree (one row "
                "per branch child: shared prefix, continuation, reward, advantage)",
            ),
        ] = "prime-sft",
        pairs: Annotated[
            Path | None,
            typer.Option(
                "--pairs",
                help="branch-tree only: also write one row per sibling pair whose "
                "rewards differ (chosen/rejected, margin) to this JSONL path",
            ),
        ] = None,
        pairs_any_request: Annotated[
            bool,
            typer.Option(
                "--pairs-any-request",
                help="branch-tree only: also pair siblings that were asked different "
                "things (default: only siblings with the same first user message)",
            ),
        ] = False,
        include_oracle: Annotated[
            bool,
            typer.Option(
                "--include-oracle",
                help="branch-tree only: keep oracle children as rows (skipped by "
                "default; their only message is the solution script's exit code)",
            ),
        ] = False,
        min_reward: Annotated[
            float | None,
            typer.Option("--min-reward", help="Only include rows with reward >= value"),
        ] = None,
        row_mode: Annotated[
            Literal["rollout", "exchange"],
            typer.Option(
                "--row-mode",
                help="rollout writes one row per rollout; exchange writes one row per LLM exchange",
            ),
        ] = "rollout",
        manifest: Annotated[
            Path | None,
            typer.Option("--manifest", help="Optional conversion stats JSON path"),
        ] = None,
        expected_rows: Annotated[
            int | None,
            typer.Option(
                "--expected-rows",
                help=(
                    "Fail (before writing the output file) unless exactly this "
                    "many rows would be exported"
                ),
            ),
        ] = None,
        canonical_selection: Annotated[
            Path | None,
            typer.Option(
                "--canonical-selection",
                help="Restrict conversion to rows selected by canonical-selection.json",
            ),
        ] = None,
        redact: Annotated[
            bool,
            typer.Option(
                "--redact/--no-redact",
                help=(
                    "Redact secrets while exporting trainer JSONL. Use --no-redact "
                    "only for private local SFT data when preserving exact tool-call "
                    "argument tokens is required."
                ),
            ),
        ] = True,
        context_policy: Annotated[
            Literal["full", "message-window"],
            typer.Option(
                "--context-policy",
                help="TRL context handling: full or tokenizer-aware message-window",
            ),
        ] = "full",
        tokenizer_id: Annotated[
            str | None,
            typer.Option(
                "--tokenizer",
                help="Tokenizer/model ID for TRL message-window conversion",
            ),
        ] = None,
        tokenizer_revision: Annotated[
            str | None,
            typer.Option(
                "--tokenizer-revision",
                help="Immutable tokenizer revision for TRL conversion",
            ),
        ] = None,
        max_length: Annotated[
            int | None,
            typer.Option(
                "--max-length",
                help="Maximum rendered length for TRL message-window conversion",
            ),
        ] = None,
        subagent_rows: Annotated[
            bool,
            typer.Option(
                "--subagent-rows",
                help=(
                    "Also write each Claude subagent conversation as separate "
                    "rows tagged with the spawning tool call. By default rows "
                    "contain the parent agent's conversation only."
                ),
            ),
        ] = False,
        reward_vector: Annotated[
            bool,
            typer.Option(
                "--reward-vector",
                help=(
                    "prime-sft/trl-sft: add reward_vector to every row: the "
                    "test gate and each rubric criterion (names, kinds, weights, "
                    "0-1 values) for rubric-scored trials, else every numeric "
                    "key the verifier wrote. Unscored rollouts get null."
                ),
            ),
        ] = False,
        group_advantage: Annotated[
            str | None,
            typer.Option(
                "--group-advantage",
                help=(
                    "prime-sft/trl-sft: add advantage and group to every row, "
                    "normalised over the scored rollouts of the same group: "
                    "grpo = (r - mean) / (std + 1e-4), sample std; loo = r - "
                    "mean of the others. Unscored rollouts get null, never 0."
                ),
            ),
        ] = None,
        group_by: Annotated[
            str,
            typer.Option(
                "--group-by",
                help=(
                    "Grouping key for --group-advantage, comma-separated from "
                    "task, agent, model, task_digest"
                ),
            ),
        ] = "task,agent,model",
    ) -> None:
        """Convert BenchFlow rollout artifacts into trainer-ready data."""
        _ensure_training_format(format_name)
        from benchflow.trajectories.training_signal import check_options

        try:
            group_keys = check_options(
                group_advantage=group_advantage, group_by=group_by
            )
        except ValueError as exc:
            flag = (
                "--group-advantage" if "group_advantage" in str(exc) else "--group-by"
            )
            print_error(f"{flag}: {exc}")
            raise typer.Exit(1) from None
        for flag, used in (
            ("--pairs", pairs is not None),
            ("--pairs-any-request", pairs_any_request),
            ("--include-oracle", include_oracle),
        ):
            if used and format_name != "branch-tree":
                print_error(f"{flag} needs --format branch-tree")
                raise typer.Exit(1)
        if format_name == "branch-tree":
            from benchflow.trajectories.export_branch import export_branch_jsonl

            for flag, used in (
                ("--row-mode", row_mode != "rollout"),
                ("--canonical-selection", canonical_selection is not None),
                ("--context-policy", context_policy != "full"),
                ("--tokenizer", tokenizer_id is not None),
                ("--tokenizer-revision", tokenizer_revision is not None),
                ("--max-length", max_length is not None),
                ("--subagent-rows", subagent_rows),
                ("--reward-vector", reward_vector),
                ("--group-advantage", group_advantage is not None),
            ):
                if used:
                    print_error(
                        f"{flag} does not apply to --format branch-tree (one row per "
                        "branch child from the ACP events, with advantage = reward "
                        "- value; see docs/branching.md)"
                    )
                    raise typer.Exit(1)
            try:
                branch_stats = export_branch_jsonl(
                    jobs_dir,
                    output,
                    pairs_out=pairs,
                    redact=redact,
                    min_reward=min_reward,
                    expected_rows=expected_rows,
                    manifest=manifest,
                    include_oracle=include_oracle,
                    any_request_pairs=pairs_any_request,
                )
            except ValueError as exc:
                print_error(str(exc))
                raise typer.Exit(1) from None
            console.print(
                f"[green]Converted {branch_stats.child_rows} child row(s)[/green] "
                f"from {branch_stats.forks} fork(s) in {branch_stats.trials} "
                f"trial(s) -> {escape(str(output))}"
                + (
                    f"; {branch_stats.pair_rows} pair row(s) -> {escape(str(pairs))}"
                    if pairs is not None
                    else ""
                ),
                soft_wrap=True,
            )
            if branch_stats.unscored_children or branch_stats.missing_observations:
                console.print(
                    f"{branch_stats.unscored_children} child(ren) unscored (kept, no "
                    f"reward or advantage); {branch_stats.missing_observations} "
                    "without an observation.json (empty continuation)."
                )
            skipped = [
                f"{n} {what}"
                for n, what in (
                    (
                        branch_stats.skipped_oracle,
                        "oracle child(ren) (--include-oracle)",
                    ),
                    (branch_stats.below_min_reward, "below --min-reward"),
                    (
                        branch_stats.pairs_skipped_different_request,
                        "pair(s) of siblings asked different things "
                        "(--pairs-any-request)",
                    ),
                )
                if n
            ]
            if skipped:
                console.print("Skipped: " + "; ".join(skipped) + ".")
            if manifest is not None:
                console.print(f"Stats: {escape(str(manifest))}")
            return
        try:
            if format_name == "prime-sft":
                if (
                    context_policy != "full"
                    or tokenizer_id is not None
                    or tokenizer_revision is not None
                    or max_length is not None
                ):
                    raise ValueError(
                        "--context-policy/--tokenizer/--tokenizer-revision/"
                        "--max-length are supported only with --format trl-sft"
                    )
                from benchflow.trajectories.export_prime_sft import (
                    export_prime_sft_jsonl,
                )

                stats = export_prime_sft_jsonl(
                    jobs_dir,
                    output,
                    min_reward=min_reward,
                    row_mode=row_mode,
                    expected_rows=expected_rows,
                    manifest=manifest,
                    canonical_selection=canonical_selection,
                    redact=redact,
                    subagent_rows=subagent_rows,
                    reward_vector=reward_vector,
                    group_advantage=cast(Any, group_advantage),
                    group_by=group_keys,
                )
            else:
                from benchflow.trajectories.export_trl_sft import (
                    export_trl_sft_jsonl,
                )

                stats = export_trl_sft_jsonl(
                    jobs_dir,
                    output,
                    min_reward=min_reward,
                    row_mode=row_mode,
                    expected_rows=expected_rows,
                    manifest=manifest,
                    canonical_selection=canonical_selection,
                    redact=redact,
                    context_policy=context_policy,
                    tokenizer_id=tokenizer_id,
                    tokenizer_revision=tokenizer_revision,
                    max_length=max_length,
                    subagent_rows=subagent_rows,
                    reward_vector=reward_vector,
                    group_advantage=cast(Any, group_advantage),
                    group_by=group_keys,
                )
        except (OSError, ValueError) as exc:
            message = str(exc)
            if "llm_trajectory.jsonl" in message and _has_branch_trees(jobs_dir):
                message = (
                    "this is a branch run: runs that branch record the ACP "
                    "trajectory, not an LLM-proxy llm_trajectory.jsonl; use "
                    f"--format branch-tree ({message})"
                )
            print_error(message)
            raise typer.Exit(1) from None

        console.print(
            f"[green]Converted {stats.rows_written} row(s)[/green] "
            f"from {stats.rollouts_seen} rollout(s) -> {escape(str(output))}"
        )
        subagents = stats.subagents
        if subagents.rollouts_with_subagents:
            hint = "" if subagent_rows else " (--subagent-rows emits them as rows)"
            console.print(
                f"Subagents: {subagents.rollouts_with_subagents} rollout(s) made "
                f"{subagents.subagent_calls_seen} subagent call(s); included "
                f"{subagents.subagent_rows_written} subagent row(s), excluded "
                f"{subagents.subagent_exchanges_excluded} subagent LLM "
                f"exchange(s){hint}."
            )
        signal = stats.training_signal
        if signal is not None and signal["group_advantage"] is not None:
            groups = signal["groups"]
            unscored = sum(
                1
                for g in groups
                for m in g["members"]
                if m.get("excluded") == "unscored"
            )
            single = sum(1 for g in groups if g["scored"] == 1)
            console.print(
                f"Groups ({signal['group_advantage']}, by "
                f"{','.join(signal['group_by'])}): {len(groups)}; "
                f"{unscored} unscored rollout(s) left out of every baseline"
                + (
                    f"; {single} group(s) with one scored rollout (no advantage)"
                    if single
                    else ""
                )
                + "."
            )
        if manifest is not None:
            console.print(f"Stats: {escape(str(manifest))}")

    @train_app.command("token-coverage")
    def train_token_coverage(
        path: Annotated[
            Path,
            typer.Argument(help="BenchFlow rollout or jobs directory"),
        ],
        as_json: Annotated[
            bool, typer.Option("--json", help="Print the report as JSON")
        ] = False,
    ) -> None:
        """Report whether each rollout's model calls carry token ids and logprobs.

        Reads trajectory/llm_trajectory.jsonl token_capture blocks (see
        docs/reference/token-capture.md). A rollout is training-grade when
        every call has prompt token ids, sampled token ids and logprobs, and
        each prompt extends the previous call's prompt and sampled tokens.
        """
        from benchflow.trajectories.token_capture import (
            summarize_rollout_token_capture,
        )

        if not path.exists():
            print_error(f"No such path: {path}")
            raise typer.Exit(2)
        rollouts = sorted(
            {p.parent for p in path.rglob("result.json") if p.parent.is_dir()}
        )
        per_rollout = [summarize_rollout_token_capture(r) for r in rollouts]
        report = {
            "rollouts": len(per_rollout),
            "training_grade_rollouts": sum(
                1 for r in per_rollout if r["training_grade"]
            ),
            "per_rollout": per_rollout,
        }
        if as_json:
            typer.echo(json.dumps(report, indent=2))
            return
        for r in per_rollout:
            if r["status"] == "captured":
                prefix = r["prefix"]
                detail = (
                    f"{r['complete_calls']}/{r['calls']} calls complete, "
                    f"{prefix['extends_previous_call']}/{prefix['pairs']} prompts "
                    "extend the previous call"
                )
                if r["unavailable"]:
                    detail += "; missing: " + ", ".join(
                        f"{k} in {v} calls" for k, v in r["unavailable"].items()
                    )
            else:
                detail = r["reason"]
            mark = "yes" if r["training_grade"] else "no "
            route = f" [{r['path']}]" if r.get("path") else ""
            typer.echo(f"{mark}  {r['rollout']}{route}: {detail}")
        typer.echo(
            f"{report['training_grade_rollouts']} of {report['rollouts']} rollouts "
            "are training-grade (complete token ids and logprobs, token-in/token-out)"
        )

    @train_app.command("stream")
    def train_stream(
        job_dir: Annotated[
            Path,
            typer.Argument(
                help="Job folder (or a --jobs-dir holding one job); may not exist yet"
            ),
        ],
        format_name: Annotated[
            str, typer.Option("--format", help="Output format: jsonl")
        ] = "jsonl",
        follow: Annotated[
            bool,
            typer.Option(
                "--follow/--no-follow",
                help="Keep polling until the job finishes (default), or scan once",
            ),
        ] = True,
        poll_interval: Annotated[
            float, typer.Option("--poll-interval", help="Seconds between scans")
        ] = 1.0,
        timeout: Annotated[
            float | None,
            typer.Option("--timeout", help="Stop after this many seconds (exit 3)"),
        ] = None,
        group_size: Annotated[
            int | None,
            typer.Option(
                "--group-size",
                help="Hold rollouts until N of a group finished, then emit them "
                "with a GRPO advantage",
            ),
        ] = None,
        group_by: Annotated[
            str | None,
            typer.Option(
                "--group-by",
                help="Grouping key, comma-separated from task, agent, model, "
                "task_digest (default task,agent,model)",
            ),
        ] = None,
    ) -> None:
        """Print each finished rollout of a (running) job as one JSON line.

        For a trainer that consumes a job while it runs: every line is a
        benchflow.rollout-stream.v1 record with the reward, group id and,
        when the gateway captured them, per-call token ids and logprobs
        (docs/reference/rollout-stream.md). Status goes to stderr. Exit 0
        when the job finished, 1 when its process died first, 2 on bad
        input, 3 on --timeout.
        """
        from benchflow.trajectories.rollout_stream import (
            JobNotFound,
            JobProcessGone,
            StreamTimeout,
            stream_rollouts,
        )
        from benchflow.trajectories.training_signal import parse_group_by

        def fail(message: str, code: int) -> typer.Exit:
            typer.echo(f"bench train stream: {message}", err=True)
            return typer.Exit(code)

        if format_name != "jsonl":
            raise fail("--format must be 'jsonl'", 2)
        if group_size is not None and group_size < 1:
            raise fail("--group-size must be at least 1", 2)
        try:
            group_key = parse_group_by(group_by)
        except ValueError as exc:
            raise fail(f"--group-by: {exc}", 2) from None
        count = 0
        try:
            for record in stream_rollouts(
                job_dir,
                follow=follow,
                poll_interval=poll_interval,
                timeout=timeout,
                group_by=group_key,
                group_size=group_size,
            ):
                typer.echo(record.to_json())
                count += 1
        except JobNotFound as exc:
            raise fail(str(exc), 2) from None
        except JobProcessGone as exc:
            raise fail(f"{exc}; {count} rollouts streamed", 1) from None
        except StreamTimeout as exc:
            raise fail(str(exc), 3) from None
        typer.echo(f"bench train stream: {count} rollouts streamed", err=True)

    @train_app.command("validate")
    def train_validate(
        jsonl: Annotated[
            Path,
            typer.Argument(help="Trainer JSONL path to validate"),
        ],
        format_name: Annotated[
            str,
            typer.Option("--format", help="Trainer format"),
        ] = "prime-sft",
        expected_rows: Annotated[
            int | None,
            typer.Option(
                "--expected-rows", help="Fail unless this many rows are present"
            ),
        ] = None,
        source_jobs: Annotated[
            Path | None,
            typer.Option("--source-jobs", help="Source BenchFlow jobs dir to audit"),
        ] = None,
        source_canonical_selection: Annotated[
            Path | None,
            typer.Option(
                "--source-canonical-selection",
                help="Canonical selection JSON used for this trainer data",
            ),
        ] = None,
        task_manifest: Annotated[
            Path | None,
            typer.Option("--task-manifest", help="Task manifest for source rows"),
        ] = None,
        require_llm_trajectory: Annotated[
            bool,
            typer.Option(
                "--require-llm-trajectory",
                help="Fail unless source selected rows have valid llm_trajectory.jsonl",
            ),
        ] = False,
        require_tool_calls: Annotated[
            bool,
            typer.Option(
                "--require-tool-calls",
                help="Fail unless trainer rows and source rows include tool calls",
            ),
        ] = False,
        tokenizer_id: Annotated[
            str | None,
            typer.Option(
                "--tokenizer",
                help="Tokenizer/model ID for TRL render and assistant-mask validation",
            ),
        ] = None,
        tokenizer_revision: Annotated[
            str | None,
            typer.Option(
                "--tokenizer-revision",
                help="Immutable tokenizer revision for TRL validation",
            ),
        ] = None,
        max_length: Annotated[
            int | None,
            typer.Option(
                "--max-length",
                help="Fail TRL validation when a rendered row exceeds this length",
            ),
        ] = None,
    ) -> None:
        """Validate trainer-ready data."""
        _ensure_training_format(format_name)
        try:
            if format_name == "branch-tree":
                for flag, used in (
                    ("--tokenizer", tokenizer_id is not None),
                    ("--tokenizer-revision", tokenizer_revision is not None),
                    ("--max-length", max_length is not None),
                    ("--require-llm-trajectory", require_llm_trajectory),
                ):
                    if used:
                        raise ValueError(
                            f"{flag} does not apply to --format branch-tree "
                            "(branch rows come from the ACP trajectory)"
                        )
                from benchflow.trajectories.export_branch import (
                    validate_branch_jsonl,
                )

                result = validate_branch_jsonl(jsonl, expected_rows=expected_rows)
                result["format"] = "branch-tree"
            elif format_name == "prime-sft":
                if (
                    tokenizer_id is not None
                    or tokenizer_revision is not None
                    or max_length is not None
                ):
                    raise ValueError(
                        "--tokenizer/--tokenizer-revision/--max-length "
                        "are supported only with --format trl-sft"
                    )
                from benchflow.trajectories.export_prime_sft import (
                    validate_prime_sft_jsonl,
                )

                result = validate_prime_sft_jsonl(
                    jsonl,
                    expected_rows=expected_rows,
                )
            else:
                from benchflow.trajectories.export_trl_sft import (
                    validate_trl_sft_jsonl,
                )

                result = validate_trl_sft_jsonl(
                    jsonl,
                    expected_rows=expected_rows,
                    tokenizer_id=tokenizer_id,
                    tokenizer_revision=tokenizer_revision,
                    max_length=max_length,
                )
            if require_tool_calls and result["rows_with_tool_calls"] != result["rows"]:
                raise ValueError(
                    "not all trainer rows contain tool calls: "
                    f"{result['rows_with_tool_calls']} / {result['rows']}"
                )
            if source_jobs is not None:
                from benchflow.eval_artifacts import build_health_summary

                health = build_health_summary(
                    source_jobs, canonical_selection=source_canonical_selection
                )
                if require_llm_trajectory and (
                    health["missing_llm_trajectory"]
                    or health["malformed_llm_trajectory"]
                ):
                    raise ValueError(
                        "source jobs contain missing/malformed llm_trajectory.jsonl"
                    )
                if (
                    require_tool_calls
                    and health["rows_with_tool_calls"] != health["total_rows"]
                ):
                    raise ValueError(
                        "not all source rows contain tool calls: "
                        f"{health['rows_with_tool_calls']} / {health['total_rows']}"
                    )
                result["source_health"] = {
                    key: health[key]
                    for key in (
                        "total_rows",
                        "scored_rows",
                        "unscored_rows",
                        "rows_with_tool_calls",
                        "missing_llm_trajectory",
                        "malformed_llm_trajectory",
                    )
                }
            if source_canonical_selection is not None:
                data = json.loads(source_canonical_selection.read_text())
                selected = (
                    data.get("selected", data.get("selection"))
                    if isinstance(data, dict)
                    else None
                )
                if not isinstance(selected, list):
                    raise ValueError(
                        f"{source_canonical_selection}: selected or selection must be a list"
                    )
                result["canonical_selected_rows"] = len(selected)
            if task_manifest is not None:
                data = json.loads(task_manifest.read_text())
                tasks = data.get("tasks") if isinstance(data, dict) else None
                if not isinstance(tasks, list):
                    raise ValueError(f"{task_manifest}: tasks must be a list")
                result["task_manifest_rows"] = len(tasks)
        except (OSError, ValueError) as exc:
            print_error(str(exc))
            raise typer.Exit(1) from None
        console.print(json.dumps(result, sort_keys=True))

    @run_app.command("sft")
    def train_run_sft(
        config: Annotated[
            Path,
            typer.Option("--config", help="Prime-RL SFT TOML config"),
        ],
        backend: Annotated[
            Literal["prime-rl"],
            typer.Option("--backend", help="Training backend"),
        ] = "prime-rl",
        data: Annotated[
            str | None,
            typer.Option(
                "--data",
                help="Optional dataset override passed to Prime-RL as --data.name",
            ),
        ] = None,
        output_dir: Annotated[
            Path | None,
            typer.Option(
                "--output-dir",
                help="Prime-RL trainer output dir. Defaults to <work-dir>/prime-rl-output.",
            ),
        ] = None,
        compat_profile: Annotated[
            str | None,
            typer.Option(
                "--compat-profile",
                help=(
                    "Named BenchFlow Prime-RL SFT compatibility profile. "
                    "Currently supports env0-mobile300-pr828."
                ),
            ),
        ] = None,
        work_dir: Annotated[
            Path,
            typer.Option("--work-dir", help="BenchFlow training run directory"),
        ] = Path("train-runs/sft"),
        prime_rl_dir: Annotated[
            Path | None,
            typer.Option(
                "--prime-rl-dir",
                help="Prime-RL checkout to run uv from. Defaults to the current directory.",
            ),
        ] = None,
        dry_run: Annotated[
            bool,
            typer.Option("--dry-run", help="Pass --dry-run through to Prime-RL"),
        ] = False,
        follow: Annotated[
            bool,
            typer.Option("--follow", help="Stream trainer stdout while writing logs"),
        ] = False,
        uv_no_sync: Annotated[
            bool,
            typer.Option(
                "--uv-no-sync",
                help=(
                    "Run Prime-RL with `uv run --no-sync`, useful after backend "
                    "post-install steps such as flash-attn."
                ),
            ),
        ] = False,
        override: Annotated[
            list[str] | None,
            typer.Option(
                "--override",
                help="Prime-RL config override as KEY=VALUE; repeatable",
            ),
        ] = None,
        target_examples: Annotated[
            int | None,
            typer.Option(
                "--target-examples",
                help=(
                    "Derive Prime-RL max_steps from a target number of training "
                    "examples using data.batch_size, rounding up"
                ),
            ),
        ] = None,
        target_micro_steps: Annotated[
            int | None,
            typer.Option(
                "--target-micro-steps",
                help=(
                    "Derive Prime-RL max_steps from custom-trainer batch-size-1 "
                    "microsteps, dropping the final partial accumulation"
                ),
            ),
        ] = None,
        sync_scheduler_to_max_steps: Annotated[
            bool,
            typer.Option(
                "--sync-scheduler-to-max-steps/--no-sync-scheduler-to-max-steps",
                help=(
                    "When --target-examples or --target-micro-steps is set, "
                    "also derive scheduler.decay_steps from the computed max_steps"
                ),
            ),
        ] = True,
        sync_ckpt_to_max_steps: Annotated[
            bool,
            typer.Option(
                "--sync-ckpt-to-max-steps/--no-sync-ckpt-to-max-steps",
                help=(
                    "When deriving max_steps, also derive ckpt.interval and "
                    "ckpt.keep_interval from the computed max_steps"
                ),
            ),
        ] = False,
        pack_function: Annotated[
            str | None,
            typer.Option(
                "--pack-function",
                help="Optional first-class Prime-RL data.pack_function override: cat or stack",
            ),
        ] = None,
        loss_mask: Annotated[
            str | None,
            typer.Option(
                "--loss-mask",
                help=(
                    "Optional first-class Prime-RL data.loss_mask override: "
                    "'assistant', 'all', or comma-separated roles"
                ),
            ),
        ] = None,
        loss_normalization: Annotated[
            str | None,
            typer.Option(
                "--loss-normalization",
                help=(
                    "Prime-RL SFT loss normalization: token_mean for native "
                    "Prime-RL behavior, or sample_mean to match the historical "
                    "custom trainer's per-row mean loss"
                ),
            ),
        ] = None,
        model_attn: Annotated[
            str | None,
            typer.Option(
                "--model-attn",
                help="Optional first-class Prime-RL model.attn override, e.g. sdpa",
            ),
        ] = None,
        renderer_mode: Annotated[
            str | None,
            typer.Option(
                "--renderer-mode",
                help=(
                    "Optional Prime-RL renderer mode override. Use 'none' to "
                    "fall back to tokenizer.apply_chat_template tokenization."
                ),
            ),
        ] = None,
        tool_defs_mode: Annotated[
            str,
            typer.Option(
                "--tool-defs-mode",
                help=(
                    "How local training JSONL exposes tool schemas to Prime-RL: "
                    "preserve or omit"
                ),
            ),
        ] = "preserve",
        chat_template_kwarg: Annotated[
            list[str] | None,
            typer.Option(
                "--chat-template-kwarg",
                help=(
                    "Apply KEY=VALUE to every local Prime-SFT row's "
                    "chat_template_kwargs before Prime-RL loads it; repeatable. "
                    "Values are parsed as JSON literals when possible."
                ),
            ),
        ] = None,
        message_tail_truncation: Annotated[
            str,
            typer.Option(
                "--message-tail-truncation",
                help=(
                    "Local Prime-SFT row truncation before Prime-RL tokenizes it: "
                    "off, keep-first-user, or custom-trainer-pretokenized. "
                    "The keep-first-user mode keeps the initial user instruction "
                    "plus the longest final message suffix that fits "
                    "data.seq_len * data.micro_batch_size. The "
                    "custom-trainer-pretokenized mode renders rows like the "
                    "historical custom trainer, keeps the exact token tail, "
                    "and stages shifted input_ids/target_ids/loss_mask tensors."
                ),
            ),
        ] = "off",
        allow_unsafe_stack_flash_attn: Annotated[
            bool,
            typer.Option(
                "--allow-unsafe-stack-flash-attn",
                help=(
                    "Allow Qwen3.5 stack packing with flash attention despite "
                    "known Prime-RL varlen-kernel risk"
                ),
            ),
        ] = False,
        force: Annotated[
            bool,
            typer.Option(
                "--force",
                help="Overwrite an existing <work-dir>/train-run.json manifest",
            ),
        ] = False,
        publish_model: Annotated[
            str | None,
            typer.Option(
                "--publish-model", help="Upload trainer output to this HF model repo"
            ),
        ] = None,
        model_tag: Annotated[
            str | None,
            typer.Option(
                "--model-tag", help="Path prefix/tag for --publish-model upload"
            ),
        ] = None,
        model_card: Annotated[
            str | None,
            typer.Option(
                "--model-card", help="Model card mode; currently accepts 'auto'"
            ),
        ] = None,
        publish_artifacts: Annotated[
            str | None,
            typer.Option(
                "--publish-artifacts",
                help="Upload BenchFlow train run artifacts to this HF dataset repo",
            ),
        ] = None,
        hf_prefix: Annotated[
            str | None,
            typer.Option("--hf-prefix", help="Path prefix for --publish-artifacts"),
        ] = None,
        hf_public_read_check: Annotated[
            bool,
            typer.Option(
                "--hf-public-read-check", help="Verify public HF reads after upload"
            ),
        ] = False,
    ) -> None:
        """Run a Prime-RL SFT job and record a BenchFlow manifest."""
        del backend  # Typer validates the single supported backend for now.
        from benchflow.training.backends.prime_rl import (
            PrimeRlSftSpec,
            run_prime_rl_sft,
        )

        try:
            result = run_prime_rl_sft(
                PrimeRlSftSpec(
                    config=config,
                    work_dir=work_dir,
                    data=data,
                    output_dir=output_dir,
                    compat_profile=compat_profile,
                    dry_run=dry_run,
                    follow=follow,
                    uv_no_sync=uv_no_sync,
                    overrides=tuple(override or ()),
                    target_examples=target_examples,
                    target_micro_steps=target_micro_steps,
                    sync_scheduler_to_max_steps=sync_scheduler_to_max_steps,
                    sync_ckpt_to_max_steps=sync_ckpt_to_max_steps,
                    pack_function=pack_function,
                    loss_mask=loss_mask,
                    loss_normalization=loss_normalization,
                    model_attn=model_attn,
                    renderer_mode=renderer_mode,
                    tool_defs_mode=tool_defs_mode,
                    chat_template_kwargs=tuple(chat_template_kwarg or ()),
                    message_tail_truncation=message_tail_truncation,
                    allow_unsafe_stack_flash_attn=allow_unsafe_stack_flash_attn,
                    force=force,
                    cwd=prime_rl_dir,
                    publish_model=publish_model,
                    model_tag=model_tag,
                    model_card=model_card,
                    publish_artifacts=publish_artifacts,
                    hf_prefix=hf_prefix,
                    hf_public_read_check=hf_public_read_check,
                )
            )
        except ValueError as exc:
            print_error(str(exc))
            raise typer.Exit(1) from None

        if result.returncode != 0:
            print_error(
                f"Prime-RL SFT failed with exit code {result.returncode}; "
                f"see {result.manifest_path}"
            )
            raise typer.Exit(result.returncode)
        console.print(
            "[green]Prime-RL SFT completed[/green] "
            f"(manifest: {escape(str(result.manifest_path))})"
        )
