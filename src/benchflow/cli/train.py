"""``bench train`` — training data conversion commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.markup import escape

from benchflow.cli._shared import console, print_error


def _ensure_prime_sft(format_name: str) -> None:
    if format_name != "prime-sft":
        print_error("--format currently supports only 'prime-sft'")
        raise typer.Exit(1)


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
            typer.Option("--format", help="Trainer format"),
        ] = "prime-sft",
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
    ) -> None:
        """Convert BenchFlow rollout artifacts into trainer-ready data."""
        _ensure_prime_sft(format_name)
        from benchflow.trajectories.export_prime_sft import export_prime_sft_jsonl

        try:
            stats = export_prime_sft_jsonl(
                jobs_dir,
                output,
                min_reward=min_reward,
                row_mode=row_mode,
                expected_rows=expected_rows,
                manifest=manifest,
                canonical_selection=canonical_selection,
            )
        except ValueError as exc:
            print_error(str(exc))
            raise typer.Exit(1) from None

        console.print(
            f"[green]Converted {stats.rows_written} row(s)[/green] "
            f"from {stats.rollouts_seen} rollout(s) -> {escape(str(output))}"
        )
        if manifest is not None:
            console.print(f"Stats: {escape(str(manifest))}")

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
    ) -> None:
        """Validate trainer-ready data."""
        _ensure_prime_sft(format_name)
        from benchflow.trajectories.export_prime_sft import validate_prime_sft_jsonl

        try:
            result = validate_prime_sft_jsonl(jsonl, expected_rows=expected_rows)
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
                    "off, keep-first-user, or custom-trainer-token-suffix. "
                    "The keep-first-user mode keeps the initial user instruction "
                    "plus the longest final message suffix that fits "
                    "data.seq_len * data.micro_batch_size. The "
                    "custom-trainer-token-suffix mode renders rows like the "
                    "historical custom trainer, keeps the exact token suffix, "
                    "and stages a passthrough Prime-RL chat template."
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
