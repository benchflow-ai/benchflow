"""``bench doctor`` and ``bench eval smoke`` — first-run checks.

``bench doctor`` prints one pass / warn / fail / skip line per check, each with
a concrete fix, and exits 1 when anything required fails. ``bench eval smoke``
reruns the checks, then runs the bundled hello-world task once per agent with a
working credential, one at a time, and prints a compact result table.

The logic lives in :mod:`benchflow.doctor` and :mod:`benchflow.doctor_smoke`;
this module renders it. Both are looked up through their modules at call time
so tests can monkeypatch ``run_doctor`` and ``subprocess_runner``.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.markup import escape

from benchflow.cli._shared import console, err_console, print_error
from benchflow.sandbox.providers import is_known_provider, providers_phrase

if TYPE_CHECKING:
    from collections.abc import Mapping

    from benchflow.doctor import Check, DoctorProbes, DoctorReport
    from benchflow.doctor_smoke import SmokeOutcome, SmokeTarget

_GROUP_TITLES = {
    "runtime": "Runtime",
    "sandbox": "Sandbox",
    "agents": "Agent credentials",
    "versions": "Agent versions",
    "proxy": "Model proxy",
    "network": "Network",
}
_GETTING_STARTED_URL = (
    "https://github.com/benchflow-ai/benchflow/blob/main/docs/getting-started.md"
)
_STATUS_STYLE = {
    "pass": "green",
    "warn": "yellow",
    "fail": "red",
    "skip": "dim",
    "error": "red",
}


def _status_tag(status: str) -> str:
    style = _STATUS_STYLE.get(status, "white")
    return f"[{style}]{status.upper():<5}[/{style}]"


def _check_line(check: Check, width: int) -> str:
    return f"  {_status_tag(check.status)} {escape(check.name):<{width}}  {escape(check.summary)}"


def _fix_line(check: Check, width: int) -> str:
    return f"  {'':<5} {'':<{width}}  [cyan]fix:[/cyan] {escape(check.fix)}"


def render_report(report: DoctorReport, *, only_problems: bool = False) -> None:
    """Print the report grouped by area; ``only_problems`` hides pass/skip lines."""
    checks = [
        c for c in report.checks if not only_problems or c.status in ("warn", "fail")
    ]
    # Wide enough for network host names (cloudcode-pa.googleapis.com, the
    # Bedrock runtime host); only a longer custom host breaks the column.
    width = min(40, max((len(c.name) for c in checks), default=8))
    group = None
    for check in checks:
        if check.group != group:
            group = check.group
            console.print(f"[bold]{_GROUP_TITLES.get(group, group.title())}[/bold]")
        console.print(_check_line(check, width), highlight=False, soft_wrap=True)
        if check.fix and check.status != "pass":
            console.print(_fix_line(check, width), highlight=False, soft_wrap=True)


def _counts_line(report: DoctorReport) -> str:
    counts = report.counts()
    warnings = "warning" if counts["warn"] == 1 else "warnings"
    return (
        f"[green]{counts['pass']} passed[/green], "
        f"[yellow]{counts['warn']} {warnings}[/yellow], "
        f"[red]{counts['fail']} failed[/red], "
        f"[dim]{counts['skip']} skipped[/dim]"
    )


def _validate_sandbox(sandbox: str) -> None:
    if not is_known_provider(sandbox):
        print_error(f"Unknown sandbox {sandbox!r}; expected {providers_phrase()}")
        raise typer.Exit(2)


# ── bench eval run preflight ────────────────────────────────────────────

PREFLIGHT_OPT_OUT_ENV = "BENCHFLOW_SKIP_PREFLIGHT"
_OPT_OUT_VALUES = frozenset({"1", "true", "yes", "on"})


def _print_checks(checks: list[Check]) -> None:
    width = max(len(c.name) for c in checks)
    for check in checks:
        err_console.print(_check_line(check, width), highlight=False, soft_wrap=True)
        if check.fix:
            err_console.print(_fix_line(check, width), highlight=False, soft_wrap=True)


def _expired_claude_login(
    probes: DoctorProbes,
    *,
    agent: str,
    model: str | None,
    agent_env: Mapping[str, str],
) -> Check | None:
    """Doctor's Claude line when the run would fall back to an expired login file.

    Only for agents whose subscription login is ``~/.claude/.credentials.json``
    and models that need Anthropic auth. Credentials passed with
    ``--agent-env`` count, as they do for the run.
    """
    from dataclasses import replace

    from benchflow import doctor as doctor_mod
    from benchflow.agents.registry import AGENTS, infer_env_key_for_model

    config = AGENTS.get(agent)
    login = config.subscription_auth if config else None
    if login is None or not login.detect_file.endswith(".claude/.credentials.json"):
        return None
    if model and infer_env_key_for_model(model) != login.replaces_env:
        return None
    extra = {k: v for k, v in agent_env.items() if v.strip()}
    auth = doctor_mod.claude_auth(replace(probes, environ={**probes.environ, **extra}))
    effective = auth.effective
    if effective is None or effective.kind != "login-file" or effective.usable:
        return None
    return doctor_mod.check_agent_auth(auth, probes.now())


def eval_preflight(
    *,
    sandbox: str,
    agent: str | None = None,
    model: str | None = None,
    agent_env: Mapping[str, str] | None = None,
) -> None:
    """Checks ``bench eval run`` makes before it creates a job.

    Reuses ``bench doctor``'s checks, looked up through :mod:`benchflow.doctor`
    at call time:

    - ``--sandbox docker``: a failed Docker check (no CLI, or no daemon) exits
      1 here, before a job directory exists, instead of failing every rollout
      with a ``docker compose`` traceback.
    - Claude: when the only credential is ``~/.claude/.credentials.json`` and
      its access token has expired, print doctor's warning and fix. It is a
      warning, not a refusal: Claude refreshes an expired access token when
      the file's refresh token is still valid (the usual case on Linux), and
      fails after the image build when it is not (the usual case on macOS).

    - An agent name that closely misspells a registered one exits 1 with a
      suggestion (the check ``bf.run`` and ``Evaluation.run`` make).

    Set ``BENCHFLOW_SKIP_PREFLIGHT=1`` to skip the Docker and Claude checks
    (the test suite does); the agent name is checked regardless.
    """
    if agent:
        from benchflow.runtime import check_agent_names

        try:
            check_agent_names([agent])
        except ValueError as exc:
            print_error(f"{exc} No job was created.")
            raise typer.Exit(1) from None
    if os.environ.get(PREFLIGHT_OPT_OUT_ENV, "").strip().lower() in _OPT_OUT_VALUES:
        return
    if agent:
        from types import SimpleNamespace

        from benchflow.errors import MissingCredentialError, user_message
        from benchflow.evaluation import effective_model
        from benchflow.runtime import check_credentials

        try:
            # The model the job's trials will use (bench eval run resolves the
            # agent's default the same way).
            trial_model = effective_model(agent, model)
        except ValueError:
            trial_model = None
        try:
            check_credentials(
                [SimpleNamespace(agent=agent, model=trial_model, agent_env=agent_env)]
            )
        except MissingCredentialError as exc:
            print_error(f"{user_message(exc)}\nNo job was created.")
            raise typer.Exit(1) from None
    from benchflow import doctor as doctor_mod

    probes = doctor_mod.DoctorProbes.from_host()
    if sandbox == "docker":
        failed = [
            c
            for c in doctor_mod.check_docker(probes, required=True)
            if c.status == "fail"
        ]
        if failed:
            print_error("Docker is not ready for --sandbox docker; no job was created.")
            _print_checks(failed)
            err_console.print(
                "Run [cyan]bench doctor[/cyan] for the full report, or set "
                f"[cyan]{PREFLIGHT_OPT_OUT_ENV}=1[/cyan] to skip this check.",
                highlight=False,
                soft_wrap=True,
            )
            raise typer.Exit(1)
    if sandbox == "remote-docker":
        from benchflow.sandbox.remote_docker import (
            probe_remote_docker,
            resolve_remote_docker_host,
        )

        try:
            probe_remote_docker(resolve_remote_docker_host())
        except (ValueError, RuntimeError) as exc:
            print_error(
                "The remote Docker host is not ready for --sandbox remote-docker; "
                f"no job was created.\n  {exc}"
            )
            err_console.print(
                f"Set [cyan]{PREFLIGHT_OPT_OUT_ENV}=1[/cyan] to skip this check.",
                highlight=False,
                soft_wrap=True,
            )
            raise typer.Exit(1) from None
    if sandbox == "daytona":
        # The same live key check doctor makes, before a job exists: a bad key
        # otherwise created the job and retried sandbox creation.
        daytona = doctor_mod.check_daytona(probes, required=True, offline=False)
        if daytona.status == "fail":
            print_error(
                "Daytona is not ready for --sandbox daytona; no job was created."
            )
            _print_checks([daytona])
            err_console.print(
                "Run [cyan]bench doctor --sandbox daytona[/cyan] for the full report, "
                f"or set [cyan]{PREFLIGHT_OPT_OUT_ENV}=1[/cyan] to skip this check.",
                highlight=False,
                soft_wrap=True,
            )
            raise typer.Exit(1)
    if agent:
        claude = _expired_claude_login(
            probes, agent=agent, model=model, agent_env=agent_env or {}
        )
        if claude is not None:
            err_console.print(
                "[yellow]Warning:[/yellow] Claude will use a login file whose "
                "access token has expired. Unless its refresh token still works, "
                "the run fails after the image build and agent install.",
                highlight=False,
                soft_wrap=True,
            )
            _print_checks([claude])


def _agent_start_checks(
    agents: list[str], sandbox: str, *, quiet: bool = False
) -> list[Check]:
    """Run ``bench doctor --agent-start`` probes one at a time."""
    from benchflow import agent_start as agent_start_mod

    checks = []
    for agent in dict.fromkeys(a.strip() for a in agents if a.strip()):
        if not quiet:
            err_console.print(
                f"Starting {escape(agent)} in a fresh {escape(sandbox)} sandbox "
                "(install + ACP handshake, no prompt)…",
                highlight=False,
            )
        outcome = agent_start_mod.probe_agent_start(agent, sandbox=sandbox)
        checks.append(agent_start_mod.agent_start_check(outcome))
    return checks


def register_doctor(app: typer.Typer) -> None:
    """Attach ``bench doctor`` to the top-level app."""

    @app.command("doctor", rich_help_panel="Core")
    def doctor(
        sandbox: Annotated[
            str,
            typer.Option(
                "--sandbox",
                help=(
                    "Sandbox the checks treat as required: docker (default) or "
                    "daytona. Others are listed but not checked yet."
                ),
            ),
        ] = "docker",
        offline: Annotated[
            bool,
            typer.Option(
                "--offline",
                help="Skip network probes and the live Daytona credential check.",
            ),
        ] = False,
        output_json: Annotated[
            bool,
            typer.Option("--json", help="Print the report as JSON."),
        ] = False,
        agent_start: Annotated[
            list[str] | None,
            typer.Option(
                "--agent-start",
                help=(
                    "Also install this agent in a fresh --sandbox and open its "
                    "ACP connection without a prompt (no model call), to catch "
                    "a broken install or login before a batch (repeatable)."
                ),
            ),
        ] = None,
    ) -> None:
        """Check this machine can run evals: Python/uv, sandbox, agent credentials, network.

        Credentials are reported by name, source and expiry only; values are
        never printed. Exits 1 when a required check fails.
        """
        from benchflow import doctor as doctor_mod

        _validate_sandbox(sandbox)
        report = doctor_mod.run_doctor(sandbox=sandbox, offline=offline)
        starts = _agent_start_checks(agent_start or [], sandbox, quiet=output_json)
        failed = not report.ok or any(c.status == "fail" for c in starts)
        if output_json:
            data = report.to_dict()
            data["checks"] = [*data.get("checks", []), *(asdict(c) for c in starts)]
            if starts:
                data["ok"] = not failed
            typer.echo(json.dumps(data, indent=2))
            raise typer.Exit(1 if failed else 0)

        from benchflow import __version__

        console.print(
            f"[bold]BenchFlow doctor[/bold] - benchflow {__version__}, "
            f"{doctor_mod.host_summary()}, sandbox {escape(sandbox)}\n",
            highlight=False,
        )
        render_report(report)
        if starts:
            console.print()
            _print_checks(starts)
        console.print()
        console.print(_counts_line(report))
        if not failed:
            console.print("Next: [cyan]bench eval smoke[/cyan]")
            return
        console.print(
            "Fix the failed checks above, then run [cyan]bench doctor[/cyan] again."
        )
        raise typer.Exit(1)


# ── bench eval smoke ────────────────────────────────────────────────────


def _rel(path: Path | None) -> str:
    if path is None:
        return "-"
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _render_outcomes(outcomes: list[SmokeOutcome]) -> None:
    """Plain aligned columns, trajectory last and unwrapped so it stays copyable.

    Auth is on each run's progress line above, so it is not repeated here.
    """
    header = ("Agent", "Model", "Result", "Reward", "Time", "Trajectory")
    rows = [
        (
            o.target.agent,
            o.target.model,
            o.status.upper(),
            "-" if o.reward is None else f"{o.reward:g}",
            f"{o.seconds:.0f}s",
            _rel(o.trajectory),
        )
        for o in outcomes
    ]
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(5)]

    def _line(row: tuple[str, ...], status: str | None = None) -> str:
        cells = [
            escape(cell.ljust(width)) for cell, width in zip(row, widths, strict=False)
        ]
        if status is not None:
            style = _STATUS_STYLE.get(status, "white")
            cells[2] = f"[{style}]{cells[2]}[/{style}]"
        return "  " + "  ".join(cells) + "  " + escape(row[5])

    console.print("[bold]Smoke results[/bold]")
    console.print(_line(header), style="bold", highlight=False, soft_wrap=True)
    for outcome, row in zip(outcomes, rows, strict=True):
        console.print(_line(row, outcome.status), highlight=False, soft_wrap=True)
    for o in outcomes:
        if o.status != "pass":
            console.print(
                f"  [red]{escape(o.target.agent)}[/red]: {escape(o.reason)}\n"
                f"    log: {escape(_rel(o.log_path))}",
                highlight=False,
                soft_wrap=True,
            )


def register_eval_smoke(eval_app: typer.Typer) -> None:
    """Attach ``bench eval smoke`` to the ``eval`` group."""

    @eval_app.command("smoke")
    def eval_smoke(
        agent: Annotated[
            list[str] | None,
            typer.Option(
                "--agent",
                help=(
                    "Agent to smoke, as NAME or NAME=MODEL; repeatable. Default: "
                    "every agent `bench doctor` found a working credential for "
                    "(claude, codex, gemini)."
                ),
            ),
        ] = None,
        sandbox: Annotated[
            str,
            typer.Option("--sandbox", help="Sandbox: docker (default) or daytona."),
        ] = "docker",
        jobs_dir: Annotated[
            Path,
            typer.Option(
                "--jobs-dir",
                help="Parent directory; each smoke writes a timestamped folder here.",
            ),
        ] = Path("jobs/smoke"),
        timeout_sec: Annotated[
            int,
            typer.Option(
                "--timeout-sec",
                min=60,
                help="Kill a run that takes longer than this many seconds.",
            ),
        ] = 900,
    ) -> None:
        """Run a bundled hello-world task once per credentialed agent, one at a time.

        Checks the machine first (as `bench doctor`), then runs each agent in
        its own `bench eval run`, prints agent, reward, time and trajectory
        path, and exits 1 unless every run scores 1.0.
        """
        from benchflow import doctor as doctor_mod
        from benchflow import doctor_smoke

        _validate_sandbox(sandbox)
        console.print("Checking this machine (bench doctor)...", highlight=False)
        report = doctor_mod.run_doctor(sandbox=sandbox)
        blocking = [
            c for c in report.checks if c.status == "fail" and c.group != "agents"
        ]
        if any(c.status in ("warn", "fail") for c in report.checks):
            render_report(report, only_problems=True)
        console.print(_counts_line(report))
        if blocking:
            print_error(
                "Fix the failed checks above before running the smoke "
                "(`bench doctor` shows the full report)."
            )
            raise typer.Exit(1)

        try:
            targets, skipped = doctor_smoke.plan_smoke(report, agent or [])
        except ValueError as exc:
            print_error(str(exc))
            raise typer.Exit(2) from None
        for skip in skipped:
            console.print(
                f"[dim]skip {escape(skip.agent)}: {escape(skip.reason)}[/dim]",
                highlight=False,
                soft_wrap=True,
            )
        if not targets:
            print_error(
                "No agent to smoke: no working credential found. "
                "Run `bench doctor` for fixes, or pass --agent NAME[=MODEL]."
            )
            raise typer.Exit(1)

        root = jobs_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
        total = len(targets)

        def _on_start(index: int, target: SmokeTarget) -> None:
            console.print(
                f"{escape(f'[{index + 1}/{total}]')} {escape(target.agent)} "
                f"({escape(target.model)}, auth {escape(target.auth)}) on "
                f"{escape(sandbox)}...",
                highlight=False,
                soft_wrap=True,
            )

        def _on_done(index: int, outcome: SmokeOutcome) -> None:
            reward = "-" if outcome.reward is None else f"{outcome.reward:g}"
            console.print(
                f"      {_status_tag(outcome.status).strip()} reward {reward} "
                f"in {outcome.seconds:.0f}s",
                highlight=False,
            )

        outcomes = doctor_smoke.run_smoke(
            targets,
            root=root,
            sandbox=sandbox,
            timeout_sec=timeout_sec,
            runner=doctor_smoke.subprocess_runner,
            on_start=_on_start,
            on_done=_on_done,
        )
        console.print()
        _render_outcomes(outcomes)
        console.print(
            f"Summary: {escape(_rel(root / 'smoke-summary.json'))}",
            highlight=False,
            soft_wrap=True,
        )
        if all(o.status == "pass" for o in outcomes):
            # soft_wrap: a wrapped command or URL cannot be copied.
            for line in (
                f"Next: [cyan]bench eval view {escape(_rel(root))}[/cyan] "
                "to read the smoke trajectories,",
                "      then [cyan]bench eval run --tasks-dir <task> --agent "
                "<agent> --model <model>[/cyan] on a real task.",
                f"      Guide: {_GETTING_STARTED_URL}",
            ):
                console.print(line, highlight=False, soft_wrap=True)
            return
        raise typer.Exit(1)
