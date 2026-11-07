"""``bench tasks`` — task authoring commands (init / check / migrate /
normalize / export), plus the trace-import generators wired from
:mod:`benchflow.cli.trace_import`.

Registered onto the top-level app by :func:`register_tasks`; ``cli/main.py``
only wires the call.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal, cast

import typer
from rich.markup import escape

from benchflow.cli._options import AllowTaskEnvOption
from benchflow.cli._shared import (
    _apply_allow_task_env,
    command_arg,
    console,
    err_console,
    exit_if_missing,
    print_error,
    print_hint,
)
from benchflow.cli.trace_import import register_tasks_generate
from benchflow.sandbox.providers import is_known_provider, providers_phrase

# The `bench check` levels that check against a sandbox; without --sandbox
# they use the one _assumed_sandbox picks for each task.
_SANDBOX_LEVELS = frozenset({"runtime-capability", "equivalence", "acceptance-live"})


def _assumed_sandbox(task_dir: Path) -> str:
    """The backend a task is checked against when --sandbox is not given: one
    that can run it (docker, or modal for a task that asks for GPUs:
    runtime_capabilities.sandbox_for_task). A task whose config cannot be
    read gets docker, and the checks below report why it cannot be read."""
    from benchflow.task.runtime_capabilities import sandbox_for_task
    from benchflow.task.task import Task

    try:
        return sandbox_for_task(Task(task_dir).config)
    except Exception:
        return "docker"


def _check_separate(
    task_dir: Path, sandbox: str, *, report_success: bool = True
) -> bool:
    """``--verifier-sandbox separate``: print what the run's plan gate would
    say about this task; True when it would refuse it. The refusal lines and
    the next step are the run's own text
    (:func:`benchflow.integrity.trial.separate_task_check`)."""
    from benchflow.integrity.trial import separate_task_check

    refusal, image_source = separate_task_check(task_dir, sandbox=sandbox)
    if refusal is None:
        if not report_success:
            return False
        where = f"; verifier image from {image_source}" if image_source else ""
        console.print(
            f"[green]✓[/green] {escape(task_dir.name)} — --verifier-sandbox "
            f"separate can run it on {escape(sandbox)}{escape(where)}",
            soft_wrap=True,
        )
        return False
    console.print(
        f"[red]✗[/red] {escape(task_dir.name)} — --verifier-sandbox separate "
        f"refuses it on {escape(sandbox)}:",
        soft_wrap=True,
    )
    message = str(refusal)
    for line in message.splitlines()[1:]:
        console.print(f"  {escape(line)}", soft_wrap=True)
    return True


def _keep_services_check(task_dir: Path) -> tuple[str | None, list[str]]:
    """What ``bench tasks check`` says about the services a task keeps
    (``x-keep-services``): a line naming them and where the names came from
    (the task, or BenchFlow's Terminal-Bench 2 list), and the key's problems.

    task.toml values are not otherwise checked here (only its syntax, #379),
    but a run refuses a bad key, so the check names each problem at its place.
    task.md front matter is checked with the rest of the document.
    """
    import tomllib

    from benchflow.task import Task
    from benchflow.task import keep_services as ks

    toml_path = task_dir / "task.toml"
    if toml_path.is_file() and not (task_dir / "task.md").exists():
        try:
            metadata = tomllib.loads(toml_path.read_text()).get("metadata")
        except Exception:
            metadata = None  # the syntax error is an issue already
        if isinstance(metadata, dict) and ks.KEY in metadata:
            _services, problems = ks.read(metadata[ks.KEY])
            if problems:
                return None, [
                    f"metadata.{ks.KEY}{path} {message}" for path, message in problems
                ]
    try:
        kept = ks.for_task(Task(task_dir))
    except Exception:
        return None, []
    if kept is None:
        return None, []
    ports = ", ".join(s.label() for s in kept.services)
    return (
        f"keeps the agent's services on port {ports} for grading "
        f"(x-keep-services, from {kept.describe_source()}); while graded they "
        "cannot write under /logs, nor in the workspace except to a file held "
        "open since before the kill, so they keep their state under /tmp or "
        "the agent user's home; they can read every command line in the "
        "sandbox, so test.sh must keep keys off its commands' arguments",
        [],
    )


def register_tasks(app: typer.Typer) -> None:
    """Attach the ``tasks`` command group to the top-level benchflow app."""
    tasks_app = typer.Typer(help="Task authoring commands.")
    app.add_typer(tasks_app, name="tasks", rich_help_panel="Core")

    register_tasks_generate(tasks_app)

    @tasks_app.command("init")
    def tasks_init(
        name: Annotated[str, typer.Argument(help="Task name")],
        parent_dir: Annotated[
            Path,
            typer.Option("--dir", help="Parent directory (default: tasks/)"),
        ] = Path("tasks"),
        no_pytest: Annotated[
            bool, typer.Option("--no-pytest", help="Skip pytest template")
        ] = False,
        no_oracle: Annotated[
            bool,
            typer.Option(
                "--no-oracle",
                "--no-solution",
                help="Skip oracle template",
            ),
        ] = False,
        task_format: Annotated[
            str,
            typer.Option(
                "--format",
                help=(
                    "Task format. New tasks use task-md; legacy scaffolding is retired."
                ),
            ),
        ] = "task-md",
    ) -> None:
        """Scaffold a new benchmark task."""
        from benchflow._utils.task_authoring import scaffold_task

        try:
            if task_format == "legacy":
                print_error(
                    "bench new no longer scaffolds the legacy split layout.",
                    next_step="`bench new <name>` scaffolds a task.md package; "
                    "`bench import <dir> --remove-legacy` converts an existing "
                    "split task",
                )
                raise typer.Exit(1)

            result = scaffold_task(
                name,
                parent_dir=parent_dir,
                no_pytest=no_pytest,
                no_oracle=no_oracle,
                task_format=cast(Literal["legacy", "task-md"], task_format),
            )
            console.print(f"[green]Created:[/green] {escape(str(result.task_dir))}/")
            # List every file actually written, derived from the scaffold itself
            # so the summary can never under-report (e.g. omit
            # verifier/test_outputs.py or verifier/rubrics/verifier.toml, both of
            # which `bench tasks check` validates).
            for rel in result.files:
                console.print(f"  {rel}")
            # The scaffold fails on purpose until edited; say what comes next.
            # Its own --jobs-dir, so it cannot resume another experiment's job,
            # and --fresh, because this is the authoring loop: without it a
            # re-run after an edit resumes its own last job and reports the
            # cached result of the code the author just changed.
            task_dir = command_arg(result.task_dir, "<task folder>")
            jobs_dir = command_arg(
                f"jobs/{result.task_dir.name}-oracle", "<jobs folder>"
            )
            for line in (
                "Next: replace every [REPLACE: ...] placeholder (bench check "
                "lists them), then prove the task with its reference solution:",
                f"bench check {task_dir}",
                f"bench run {task_dir} --sandbox docker --jobs-dir {jobs_dir} --fresh",
                "Guide: https://github.com/benchflow-ai/benchflow/blob/main/docs/tasks/write-a-task.md",
            ):
                print_hint(line)
        except (OSError, ValueError) as e:
            # OSError covers FileExistsError plus the NotADirectoryError /
            # PermissionError that mkdir() raises for `--dir <file>` or a
            # read-only parent — siblings (migrate/normalize/export) already
            # degrade gracefully; init was the outlier that dumped a traceback.
            # escape(): the OSError message echoes the user-supplied path.
            print_error(
                str(e), next_step="pick another name, or another parent with --dir"
            )
            raise typer.Exit(1) from None

    @tasks_app.command("check")
    def tasks_check(
        task_dir: Annotated[
            Path, typer.Argument(help="Task directory or folder of tasks")
        ],
        validation_level: Annotated[
            Literal[
                "schema",
                "structural",
                "runtime-capability",
                "publication-grade",
                "acceptance",
                "acceptance-live",
                "equivalence",
            ],
            typer.Option(
                "--level",
                help=(
                    "Validation level: schema, structural, runtime-capability, "
                    "publication-grade, acceptance, acceptance-live, or "
                    "equivalence (runs the oracle in --sandbox, then grades "
                    "wrong and value-equivalent variants of its output)"
                ),
            ),
        ] = "structural",
        sandbox: Annotated[
            str | None,
            typer.Option(
                "--sandbox",
                help=(
                    f"Validate runtime semantics for {providers_phrase()} "
                    "(default: a sandbox that can run the task: docker, or "
                    "modal for a task that asks for GPUs)"
                ),
            ),
        ] = None,
        report_output: Annotated[
            Path | None,
            typer.Option(
                "--report-output",
                help=(
                    "Write the acceptance-live report to this host path instead "
                    "of the task-declared report path; with --level equivalence, "
                    "write the battery's JSON report here"
                ),
            ),
        ] = None,
        no_report_write: Annotated[
            bool,
            typer.Option(
                "--no-report-write",
                help=(
                    "Validate acceptance-live without writing the declared report "
                    "or its .sha256 sidecar (report-only dogfood; leaves the task "
                    "package unmodified). Takes precedence over --report-output."
                ),
            ),
        ] = False,
        allow_task_env: AllowTaskEnvOption = None,
        verifier_sandbox: Annotated[
            str | None,
            typer.Option(
                "--verifier-sandbox",
                help=(
                    "With separate: also check what `bench run "
                    "--verifier-sandbox separate` checks before it starts (on "
                    "--sandbox, default docker), with the same refusals, and "
                    "name where the verifier image comes from"
                ),
            ),
        ] = None,
        agent: Annotated[
            str, typer.Option("--agent", help="Agent to validate, as for eval run")
        ] = "claude-agent-acp",
        model: Annotated[
            str | None,
            typer.Option("--model", help="Model to validate, as for eval run"),
        ] = None,
        sandbox_user: Annotated[
            str,
            typer.Option(
                "--sandbox-user", help="Agent user to validate, as for eval run"
            ),
        ] = "agent",
    ) -> None:
        """Validate task structure and the run's plan without starting a sandbox.

        Warnings (e.g. task.toml keys BenchFlow does not know) are printed
        but do not fail the check.
        """
        from benchflow._utils.config import normalize_agent_name, normalize_sandbox_user
        from benchflow.cli._shared import _apply_dotenv_to_process_env
        from benchflow.evaluation import effective_model
        from benchflow.task.formats import detect_task_format
        from benchflow.task.verifier_sandbox import normalize_verifier_sandbox

        if sandbox is not None and not is_known_provider(sandbox):
            print_error(
                f"Unknown sandbox: {sandbox!r}",
                next_step=f"pass --sandbox with one of: {providers_phrase()}",
            )
            raise typer.Exit(2)
        try:
            separate = normalize_verifier_sandbox(verifier_sandbox) == "separate"
        except ValueError as exc:
            print_error(f"--verifier-sandbox: {exc}", next_step="pass task or separate")
            raise typer.Exit(2) from None

        _apply_allow_task_env(allow_task_env)
        # Without --sandbox each task is checked against a sandbox that can
        # run it (check_one, _assumed_sandbox): the default the help gives
        # (dogfood p5-14).

        _apply_dotenv_to_process_env()
        try:
            if validation_level != "schema":
                agent = normalize_agent_name(agent)
                model = effective_model(agent, model)
            sandbox_user_value = normalize_sandbox_user(sandbox_user)
        except (KeyError, ValueError) as exc:
            print_error(str(exc), next_step="check --agent, --model and --sandbox-user")
            raise typer.Exit(2) from None

        def is_task(path: Path) -> bool:
            return (
                (path / "task.md").exists()
                or (path / "task.toml").exists()
                or detect_task_format(path) is not None
            )

        from benchflow.task.discovery import resolve_task_collection_root

        task_dir = resolve_task_collection_root(task_dir)
        if not task_dir.is_dir():
            # A usage mistake, with its next step (dogfood p5-8, p5-9).
            print_error(
                f"Not a directory: {task_dir}",
                next_step="pass a task folder or a folder of tasks; "
                "`bench new <name>` creates a task",
            )
            raise typer.Exit(2)
        candidates = [task_dir]
        if task_dir.is_dir() and not is_task(task_dir):
            children = [
                child
                for child in sorted(task_dir.iterdir())
                if child.is_dir() and is_task(child)
            ]
            if children:
                candidates = children
        failed = False
        for candidate in candidates:
            try:
                check_one(
                    candidate,
                    validation_level=validation_level,
                    sandbox=sandbox,
                    report_output=report_output,
                    no_report_write=no_report_write,
                    agent=agent,
                    model=model,
                    sandbox_user=sandbox_user_value,
                    separate=separate,
                )
            except typer.Exit:
                failed = True
            except (OSError, ValueError, RuntimeError) as exc:
                print_error(str(exc), next_step=f"fix {candidate} and check it again")
                failed = True
        if failed:
            raise typer.Exit(1)

    def check_one(
        task_dir: Path,
        *,
        validation_level,
        sandbox: str | None,
        report_output: Path | None,
        no_report_write: bool,
        agent: str,
        model: str | None,
        sandbox_user: str | None,
        separate: bool,
    ) -> None:
        from benchflow._utils.task_authoring import (
            check_task,
            check_task_warnings,
            task_symlink_issues,
        )
        from benchflow.task.formats import detect_task_format, materialize_task_dir
        from benchflow.task.plan_validation import validate_task_plan
        from benchflow.task.runtime_capabilities import UnsupportedTaskFeatureError

        linked = (
            task_symlink_issues(task_dir)
            if task_dir.is_dir() and validation_level != "schema"
            else []
        )
        if linked:
            raise ValueError(linked[0])
        # Inspect the source package before an adapter materializes a copy:
        # copying macOS metadata can contaminate an image's task data.
        if task_dir.is_dir():
            for sidecar in sorted(task_dir.rglob("._*")):
                if sidecar.is_file():
                    relative = sidecar.relative_to(task_dir).as_posix()
                    console.print(
                        "  [yellow]warning:[/yellow] AppleDouble metadata file "
                        f"{escape(relative)}: remove it before staging the task; "
                        "on macOS, COPYFILE_DISABLE=1 prevents tar from adding it"
                    )
        fmt = detect_task_format(task_dir)
        refused_before_native = False
        if fmt is not None and fmt.name == "taskmd":
            # task.md draft 2: the reference checker's report, then BenchFlow's
            # decisions field by field, then the native package's own checks.
            from benchflow.taskmd.check import check_package

            report = check_package(
                task_dir,
                execute_family=validation_level in {"equivalence", "acceptance-live"},
                schema_only=validation_level == "schema",
            )
            console.print("reference checker (task-md tools/taskmd.py check):")
            for line in report.reference:
                console.print(f"  {escape(line)}")
            console.print("BenchFlow:")
            for line in report.benchflow:
                console.print(f"  {escape(line)}")
            refused_before_native = not report.ok
            if report.native_dir is None:
                if refused_before_native:
                    # Preserve the run's full refusal as well as the field report.
                    materialize_task_dir(task_dir)
                    raise typer.Exit(1)
                return
            task_dir = report.native_dir
            console.print(
                f"{escape(fmt.name)} task format: checking the materialized package "
                f"{escape(str(task_dir))}"
            )
        elif fmt is not None:
            task_dir = materialize_task_dir(task_dir)
            console.print(
                f"{escape(fmt.name)} task format: checking the materialized package "
                f"{escape(str(task_dir))}"
            )
        for warning in check_task_warnings(task_dir) if task_dir.is_dir() else ():
            console.print(f"  [yellow]warning:[/yellow] {escape(warning)}")
        # Without --sandbox, a sandbox that can run the task: docker, or
        # modal for a task that asks for GPUs, which docker refuses (0.7
        # checked those without a sandbox at all). --sandbox docker still
        # refuses one.
        checked = sandbox if sandbox is not None else _assumed_sandbox(task_dir)
        if sandbox is None and checked != "docker" and validation_level != "schema":
            console.print(
                f"  no --sandbox given: checking against {escape(checked)}, the "
                "sandbox that runs a task asking for GPUs"
            )
        if sandbox is None and validation_level in _SANDBOX_LEVELS:
            sandbox = checked
        # The explicit isolation switch promises the same refusal as the
        # evaluation plan. Check it before the general runtime gate, which
        # can reject the same feature with a different diagnostic (SV2).
        if (
            separate
            and task_dir.is_dir()
            and _check_separate(task_dir, checked, report_success=False)
        ):
            raise typer.Exit(1)
        if validation_level != "schema" and (
            (task_dir / "task.md").is_file() or (task_dir / "task.toml").is_file()
        ):
            try:
                validate_task_plan(
                    task_dir,
                    sandbox=checked,
                    agent=agent,
                    model=model,
                    sandbox_user=sandbox_user,
                )
            except UnsupportedTaskFeatureError as exc:
                # Keep the run's complete error verbatim on stderr and the
                # existing field-level check diagnostics on stdout. Both
                # come from the same refusal, not a second validation pass.
                print_error(
                    str(exc),
                    next_step="pick a --sandbox that runs these features, or "
                    "change the task",
                )
                for feature in exc.features:
                    console.print(
                        "  [yellow]→[/yellow] Unsupported runtime feature: "
                        + escape(feature.format())
                    )
                raise typer.Exit(1) from None
        kept, keep_issues = _keep_services_check(task_dir)
        if kept is not None:
            console.print(f"  {escape(kept)}")
        issues = check_task(
            task_dir,
            sandbox_type=sandbox,
            validation_level=validation_level,
            acceptance_live_report_output=(
                None if validation_level == "equivalence" else report_output
            ),
            acceptance_live_write_report=not no_report_write,
            equivalence_report_output=(
                report_output if validation_level == "equivalence" else None
            ),
        )
        if task_dir.is_dir():
            env_issues, env_warning = _task_env_issues(task_dir, checked)
            if env_warning:
                console.print(f"  [yellow]warning:[/yellow] {escape(env_warning)}")
            issues.extend(env_issues)
        failed = refused_before_native

        issues = [*keep_issues, *issues]
        if not issues:
            console.print(
                f"[green]✓[/green] {escape(task_dir.name)} — valid ({validation_level})"
            )
        else:
            failed = True
            console.print(
                f"[red]✗[/red] {escape(task_dir.name)} — {len(issues)} issue(s):"
            )
            for issue in issues:
                # Escape Rich markup so literal section names like "[agent]"
                # render verbatim instead of being parsed as styling (#379).
                console.print(f"  [yellow]→[/yellow] {escape(issue)}")
        if separate and task_dir.is_dir():
            failed = _check_separate(task_dir, checked) or failed
        if failed:
            raise typer.Exit(1)

    def _task_env_issues(
        task_dir: Path, sandbox: str | None
    ) -> tuple[list[str], str | None]:
        """What `bench eval run` refuses: env values reading unnamed variables,
        and Compose paths and hooks outside the task's environment folder;
        and its warning for ``${VAR:-default}`` values that get the default
        although you have the variable set."""
        from benchflow.evaluation import _task_env_findings
        from benchflow.task.env import defaulted_task_env_warning

        refused, defaulted = _task_env_findings(task_dir, sandbox=sandbox or "docker")
        issues = [
            f"{request.describe()}: "
            + (
                f"reads {request.variable} from your environment, "
                if request.remedy is None
                else ""
            )
            + f"so bench run refuses the task. {request.hint()}"
            for request in refused
        ]
        return issues, defaulted_task_env_warning({task_dir.name: defaulted})

    @tasks_app.command("overlap")
    def tasks_overlap(
        left_manifest: Annotated[
            Path,
            typer.Argument(help="Left task manifest JSON"),
        ],
        right_manifest: Annotated[
            Path,
            typer.Argument(help="Right task manifest JSON"),
        ],
        output: Annotated[
            Path | None,
            typer.Option("--out", "-o", help="Optional JSON output path"),
        ] = None,
    ) -> None:
        """Compare two task manifests for exact task-id and digest overlap."""
        from benchflow.eval_artifacts import task_overlap

        try:
            result = task_overlap(left_manifest, right_manifest)
        except (OSError, ValueError) as exc:
            print_error(
                str(exc),
                next_step="pass two task manifests (`bench run "
                "--task-manifest-out` writes one)",
            )
            raise typer.Exit(1) from None
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            console.print(f"[green]Wrote:[/green] {escape(str(output))}")
        console.print(
            "Task-id overlap: "
            f"{result['task_id_overlap_count']} "
            f"({result['left_count']} left, {result['right_count']} right)"
        )
        console.print(f"Digest overlap: {result['digest_overlap_count']}")
        console.print(f"[dim]{escape(result['caveat'])}[/dim]")

    @tasks_app.command("migrate")
    def tasks_migrate(
        task_dir: Annotated[Path, typer.Argument(help="Legacy task directory")],
        overwrite: Annotated[
            bool,
            typer.Option("--overwrite", help="Replace an existing task.md"),
        ] = False,
        remove_legacy: Annotated[
            bool,
            typer.Option(
                "--remove-legacy",
                help=(
                    "Delete split files and promote tests/solution aliases after "
                    "task.md is verified"
                ),
            ),
        ] = False,
    ) -> None:
        """Convert task.toml + instruction.md into the unified task.md format."""
        from benchflow._utils.task_authoring import migrate_task_to_task_md

        exit_if_missing(task_dir, next_step="pass a task.toml task folder")
        try:
            result = migrate_task_to_task_md(
                task_dir,
                overwrite=overwrite,
                remove_legacy=remove_legacy,
            )
        except (
            FileExistsError,
            FileNotFoundError,
            NotADirectoryError,
            ValueError,
        ) as e:
            print_error(
                f"{e}",
                next_step="pass a task.toml task folder; --overwrite replaces an "
                "existing task.md",
            )
            raise typer.Exit(1) from None

        console.print(f"[green]Created:[/green] {result.task_md}")
        if result.removed_legacy:
            console.print("  removed task.toml and instruction.md")
            for migrated_dir in result.migrated_legacy_dirs:
                console.print(f"  promoted {migrated_dir}")
        else:
            console.print("  kept task.toml and instruction.md")

    @tasks_app.command("normalize")
    def tasks_normalize(
        task_dir: Annotated[Path, typer.Argument(help="Task directory with task.md")],
        output: Annotated[
            Path | None,
            typer.Option(
                "--output",
                "-o",
                help="Write normalized task.md to this path instead of stdout",
            ),
        ] = None,
        write: Annotated[
            bool,
            typer.Option(
                "--write",
                help="Replace task.md in place with the normalized canonical form",
            ),
        ] = False,
    ) -> None:
        """Expand minimal task.md authoring profiles into canonical task.md."""
        from benchflow._utils.task_authoring import normalize_task_md

        exit_if_missing(task_dir, next_step="pass a task.md task folder")
        try:
            result = normalize_task_md(task_dir, output_path=output, write=write)
        except (OSError, ValueError) as e:
            # OSError (covers IsADirectoryError when --output is an existing dir,
            # plus FileNotFoundError/NotADirectoryError) → clean error, no
            # traceback. Mirrors `tasks init`'s (OSError, ValueError) handler.
            print_error(
                f"{e}",
                next_step="pass a task.md task folder, and an --output file path "
                "(not a folder)",
            )
            raise typer.Exit(1) from None

        if result.output_path is None:
            typer.echo(result.normalized_text, nl=False)
        else:
            console.print(f"[green]Normalized:[/green] {result.output_path}")

    @tasks_app.command("export")
    def tasks_export(
        task_dir: Annotated[Path, typer.Argument(help="Task directory to export")],
        output_dir: Annotated[
            Path | None,
            typer.Argument(
                help="Destination split-layout directory (omit with --report-only)",
            ),
        ] = None,
        target: Annotated[
            str,
            typer.Option("--target", help="Layout to write: task-toml"),
        ] = "task-toml",
        overwrite: Annotated[
            bool,
            typer.Option("--overwrite", help="Replace an existing export directory"),
        ] = False,
        report_only: Annotated[
            bool,
            typer.Option(
                "--report-only",
                help="Print the compatibility loss report without writing files",
            ),
        ] = False,
    ) -> None:
        """Export a task to a compatibility split layout with a loss report."""
        from benchflow.task import (
            build_compatibility_export_report,
            export_task_to_split_layout,
        )
        from benchflow.task.export import check_export_target

        exit_if_missing(task_dir, next_step="pass a task folder to export")
        try:
            layout = check_export_target(target)
        except ValueError as e:
            print_error(f"--{e}", next_step="pass one of those to --target")
            raise typer.Exit(2) from None

        try:
            if report_only:
                report = build_compatibility_export_report(task_dir, target=layout)
                typer.echo(report.to_json(), nl=False)
                return
            if output_dir is None:
                print_error(
                    "Missing output_dir",
                    next_step="pass one, or use --report-only",
                )
                raise typer.Exit(2)
            report = export_task_to_split_layout(
                task_dir,
                output_dir,
                target=layout,
                overwrite=overwrite,
            )
        except (
            FileExistsError,
            FileNotFoundError,
            NotADirectoryError,
            ValueError,
        ) as e:
            print_error(
                f"{e}",
                next_step="pass a task folder and a new output folder; "
                "--overwrite replaces an existing export",
            )
            raise typer.Exit(1) from None

        console.print(f"[green]Exported:[/green] {escape(str(output_dir))}")
        console.print(f"  target: {report.target}")
        console.print(f"  status: {report.status}")
        console.print(f"  losses: {len(report.losses)}")
        console.print("  report: compatibility/export-report.json")

    @tasks_app.command("snapshot-hf")
    def tasks_snapshot_hf(
        repo_id: Annotated[
            str,
            typer.Argument(help="Hugging Face dataset repo ID, e.g. org/name"),
        ],
        output_dir: Annotated[
            Path,
            typer.Argument(help="Local output directory for the materialized snapshot"),
        ],
        revision: Annotated[
            str | None,
            typer.Option(
                "--revision",
                "--ref",
                help="Dataset revision, branch, tag, or commit",
            ),
        ] = None,
        path: Annotated[
            str | None,
            typer.Option(
                "--path",
                help="Optional subpath inside the dataset repo, e.g. tasks",
            ),
        ] = None,
        cache_dir: Annotated[
            Path | None,
            typer.Option("--cache-dir", help="Optional Hugging Face cache directory"),
        ] = None,
        overwrite: Annotated[
            bool,
            typer.Option("--overwrite", help="Replace an existing output directory"),
        ] = False,
        include_task: Annotated[
            list[str] | None,
            typer.Option(
                "--include-task",
                help="Download only this task directory; repeat for multiple tasks",
            ),
        ] = None,
    ) -> None:
        """Materialize a Hugging Face dataset task tree with provenance."""
        from benchflow._utils.hf_datasets import SOURCE_SIDECAR, snapshot_hf_dataset
        from benchflow.task.discovery import is_task_dir, resolve_task_collection_root

        try:
            snapshot = snapshot_hf_dataset(
                repo_id,
                output_dir=output_dir,
                revision=revision,
                path=path,
                cache_dir=cache_dir,
                overwrite=overwrite,
                include_tasks=include_task or (),
            )
        except (ImportError, OSError, ValueError) as e:
            print_error(
                f"{e}",
                next_step="check the dataset id, --revision and --path; "
                "--overwrite replaces an existing output folder",
            )
            raise typer.Exit(1) from None

        task_root = resolve_task_collection_root(snapshot.path)
        if is_task_dir(task_root):
            task_count = 1
        else:
            task_count = sum(
                1
                for child in task_root.iterdir()
                if child.is_dir() and is_task_dir(child)
            )
        console.print(f"[green]Snapshot:[/green] {escape(str(snapshot.path))}")
        console.print(f"  tasks: {task_count} under {escape(str(task_root))}")
        console.print(
            f"  source: {escape(repo_id)}@{escape(str(snapshot.provenance['resolved_revision']))}"
        )
        console.print(f"  provenance: {SOURCE_SIDECAR}")

    @tasks_app.command("digest")
    def tasks_digest(
        path: Annotated[
            Path,
            typer.Argument(
                help="Path to a task directory, or a directory of task directories"
            ),
        ],
    ) -> None:
        """Compute the content digest that pins a task's files, independent of git.

        Matches the digests in the skillsbench dataset registry (registry.json).
        Given a single task directory (a legacy ``task.toml`` or a native
        ``task.md`` task), prints the digest; given a directory of tasks, prints
        one "<name> <digest>" line per task.
        """
        from benchflow._utils.task_authoring import task_digest

        # A task directory is either a legacy task.toml task or a native task.md
        # task (the universal-adapter format) — recognize both, not just legacy.
        def _is_task_dir(p: Path) -> bool:
            # is_file() can raise (e.g. PermissionError) when p is an unreadable
            # directory — stat'ing p/task.toml needs +x on p. Treat anything we
            # cannot stat as "not a task dir" so the directory scan below skips it
            # instead of dumping a raw traceback.
            try:
                return (p / "task.toml").is_file() or (p / "task.md").is_file()
            except OSError:
                return False

        if not path.is_dir():
            print_error(
                f"Not a directory: {path}",
                next_step="pass a task folder or a folder of tasks",
            )
            raise typer.Exit(2)

        if _is_task_dir(path):
            # typer.echo, not console.print: Rich wraps lines at terminal width,
            # which would corrupt piped machine-readable output.
            try:
                digest = task_digest(path)
            except OSError as e:
                # An unreadable file (permissions) otherwise dumps a raw traceback.
                print_error(
                    f"Cannot read task files under {path}: {e}",
                    next_step="make the task's files readable by this user",
                )
                raise typer.Exit(1) from None
            typer.echo(digest)
            return

        task_dirs = sorted(d for d in path.iterdir() if d.is_dir() and _is_task_dir(d))
        if not task_dirs:
            print_error(
                f"No tasks under {path}",
                next_step="pass a folder with task.toml or task.md in it or in its "
                "immediate subfolders",
            )
            raise typer.Exit(1)
        for task_dir in task_dirs:
            # Batch: warn-and-skip an unreadable task (to stderr so the digest
            # lines on stdout stay machine-readable) instead of aborting the run.
            try:
                digest = task_digest(task_dir)
            except OSError as e:
                err_console.print(
                    f"[yellow]Skipping[/yellow] {escape(task_dir.name)}: {escape(str(e))}"
                )
                continue
            typer.echo(f"{task_dir.name} {digest}")
