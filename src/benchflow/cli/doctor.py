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

import os
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.markup import escape

from benchflow.cli._shared import (
    console,
    err_console,
    local_path_arg,
    print_error,
    print_hint,
)
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
    "https://github.com/benchflow-ai/benchflow/blob/main/docs/start/quick-start.md"
)


def oracle_quickstart(sandbox: str = "docker") -> str:
    """The README's step 2: one SkillsBench task with the oracle, no login."""
    return (
        "bench run benchflow-ai/skillsbench/tasks/econ-detrending-correlation "
        f"--agent oracle --sandbox {sandbox} --jobs-dir jobs/oracle"
    )


def _doctor_next_lines(report: DoctorReport, sandbox: str) -> list[str]:
    """What a passing doctor sends you to: the oracle first, as the README does.

    With no working credential the smoke has nothing to run (DF-p1-first-run-3:
    doctor said ``bench eval smoke`` and smoke said ``bench doctor``), so the
    oracle run is the step, and the login comes before the smoke.
    """
    head = f"Next: {oracle_quickstart(sandbox)}"
    if any(c.id == "auth.any" for c in report.checks):
        return [
            head,
            "      runs one task with the oracle, which needs no login.",
            "      For a model agent, log in (the agent credentials fix above), "
            "then run bench eval smoke.",
        ]
    logins = {
        c.id for c in report.checks if c.id.startswith("auth.") and c.status == "pass"
    }
    if logins == {"auth.providers"}:
        # Only provider keys: a bare smoke tries the logged-in agents and has
        # none, so name an agent that runs on the key.
        providers = next(c for c in report.checks if c.id == "auth.providers")
        names = [
            str(s.get("name", ""))
            for s in (providers.details or {}).get("sources", [])
            if s.get("usable", True)
        ]
        prefix = next(
            (
                n.removesuffix("_API_KEY").lower()
                for n in names
                if n.endswith("_API_KEY")
            ),
            "<provider>",
        )
        return [
            head,
            "      runs one task with the oracle (no model); then "
            f"bench eval smoke --agent openhands={prefix}/<model> runs a model "
            "on your provider key.",
        ]
    return [
        head,
        "      runs one task with the oracle (no model); then bench eval smoke "
        "runs each agent you are logged in to.",
    ]


_STATUS_STYLE = {
    "pass": "green",
    "warn": "yellow",
    "fail": "red",
    "skip": "dim",
    "error": "red",
}


def _print(
    output: Console,
    message: str = "",
    *,
    fields: Sequence[str] = (),
    render: Callable[[Sequence[str]], str] | None = None,
    **kwargs,
) -> None:
    """All doctor/smoke terminal output goes through the shared redactor last."""
    from benchflow._diagnostics import render_diagnostic
    from benchflow.cli._errors import _redact, _secrets

    text = (
        render_diagnostic(fields, _secrets(), render)
        if render is not None
        else _redact(message, _secrets())
    )
    output.print(text, **kwargs)


def _status_tag(status: str) -> str:
    style = _STATUS_STYLE.get(status, "white")
    return f"[{style}]{status.upper():<5}[/{style}]"


def _check_line(check: Check, width: int) -> str:
    return f"  {_status_tag(check.status)} {escape(check.name):<{width}}  {escape(check.summary)}"


def _fix_line(check: Check, width: int) -> str:
    return f"  {'':<5} {'':<{width}}  fix: {check.fix}"


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
            _print(console, f"[bold]{_GROUP_TITLES.get(group, group.title())}[/bold]")
        _print_check(console, check, width)
        if check.fix and check.status != "pass":
            print_hint(_fix_line(check, width))


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
        print_error(
            f"Unknown sandbox {sandbox!r}",
            next_step=f"pass --sandbox {providers_phrase()}",
        )
        raise typer.Exit(2)


# ── bench eval run preflight ────────────────────────────────────────────

PREFLIGHT_OPT_OUT_ENV = "BENCHFLOW_SKIP_PREFLIGHT"
_OPT_OUT_VALUES = frozenset({"1", "true", "yes", "on"})


def _print_checks(checks: list[Check]) -> None:
    width = max(len(c.name) for c in checks)
    for check in checks:
        _print_check(err_console, check, width)
        if check.fix:
            print_hint(_fix_line(check, width), stderr=True)


def _print_check(output: Console, check: Check, width: int) -> None:
    from dataclasses import replace

    _print(
        output,
        fields=(check.name, check.summary),
        render=lambda fields: _check_line(
            replace(check, name=fields[0], summary=fields[1]), width
        ),
        highlight=False,
        soft_wrap=True,
    )


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


def _check_provider_key(
    agent: str, model: str | None, agent_env: Mapping[str, str] | None
) -> None:
    """Stop before a job exists when the provider refuses the run's key.

    See :mod:`benchflow.key_check`: only a direct Anthropic or OpenAI key is
    asked about; a model missing from the provider's list gets a note.
    """
    from benchflow.agents.registry import is_scripted_agent
    from benchflow.key_check import KEY_CHECK_OPT_OUT_ENV, check_key, key_check_enabled

    # The oracle and nop call no model: a key in the host's environment is
    # none of their business.
    if not key_check_enabled() or not model or is_scripted_agent(agent):
        return
    try:
        from benchflow.agents.env import resolve_agent_env

        resolved = resolve_agent_env(agent, model, dict(agent_env or {}))
    except Exception:
        return  # the trial reports what the resolution rejects
    result = check_key(resolved, model)
    if result is None:
        return
    if result.state == "rejected":
        print_error(
            f"{result.provider} refused the key in {result.env_name} "
            f"(HTTP {result.status}); no job was created.",
            next_step="check the key with `bench doctor`, or set "
            f"{KEY_CHECK_OPT_OUT_ENV}=0 to skip this check",
        )
        raise typer.Exit(1)
    note = result.model_note(model)
    if note:
        err_console.print(f"[yellow]Note:[/yellow] {escape(note)}", soft_wrap=True)


class DockerPreflight:
    """``eval_preflight``'s Docker check, running in a thread from the start.

    ``bench eval run`` imports the engine (about 0.25 s) before it reaches
    the preflight; the check's two docker calls (``context show``, ``info``,
    about 50 ms) run meanwhile instead of after it. The result is the same
    list of checks ``check_docker`` returns inline.
    """

    def __init__(self, probes: DoctorProbes) -> None:
        import threading

        self._probes = probes
        self._checks: list[Check] = []
        self._error: BaseException | None = None
        # A daemon thread: a command that exits before the preflight (a bad
        # flag) does not wait for docker.
        self._thread = threading.Thread(
            target=self._run, name="bench-docker-preflight", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        from benchflow import doctor as doctor_mod

        try:
            self._checks = doctor_mod.check_docker(
                self._probes, required=True, buildx=False
            )
        except BaseException as exc:  # handed to the caller in checks()
            self._error = exc

    def checks(self) -> list[Check]:
        self._thread.join()
        if self._error is not None:
            raise self._error
        return self._checks


def start_docker_preflight(sandbox: str | None) -> DockerPreflight | None:
    """Start the Docker check ``eval_preflight`` will make, or None.

    Only for ``--sandbox docker`` (the default) without the opt-out, and
    without ``DOCKER_HOST`` (whose check imports the remote sandbox module,
    which the main thread may be importing at the same time): those run the
    check inline, as before.
    """
    if (sandbox if sandbox is not None else "docker") != "docker":
        return None
    if os.environ.get(PREFLIGHT_OPT_OUT_ENV, "").strip().lower() in _OPT_OUT_VALUES:
        return None
    from benchflow import doctor as doctor_mod

    probes = doctor_mod.DoctorProbes.from_host()
    if probes.get("DOCKER_HOST"):
        return None
    return DockerPreflight(probes)


def eval_preflight(
    *,
    sandbox: str,
    agent: str | None = None,
    model: str | None = None,
    agent_env: Mapping[str, str] | None = None,
    docker: DockerPreflight | None = None,
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
    - An agent/model pair no provider route serves (``--agent claude`` with
      an OpenAI-only provider's model) exits 1 naming the protocols the
      provider serves (the same check).

    Set ``BENCHFLOW_SKIP_PREFLIGHT=1`` to skip the Docker and Claude checks
    (the test suite does); the agent name and route are checked regardless.

    ``docker``: the Docker check already started (``start_docker_preflight``);
    its result stands in for running it here.
    """
    if agent:
        from benchflow.runtime import check_agent_names

        try:
            check_agent_names([agent])
        except ValueError as exc:
            # The message ends with its own next step (`bench agent list`).
            print_error(f"No job was created. {exc}")
            raise typer.Exit(1) from None
        from benchflow.agents.env import unroutable_agent_model
        from benchflow.evaluation import effective_model

        try:
            route_model = effective_model(agent, model, dict(agent_env or {}))
        except ValueError:
            # No --model and no default: reported below, with its fix.
            route_model = None
        reason = unroutable_agent_model(agent, route_model)
        if reason is not None:
            print_error(
                f"{reason}\nNo job was created.",
                next_step=(
                    "pass a --model whose provider serves this agent's protocol, "
                    "or an --agent that speaks one the provider supports"
                ),
            )
            raise typer.Exit(1)
    if os.environ.get(PREFLIGHT_OPT_OUT_ENV, "").strip().lower() in _OPT_OUT_VALUES:
        return
    if agent:
        from types import SimpleNamespace

        from benchflow.cli._errors import expected_error_parts
        from benchflow.errors import MissingCredentialError
        from benchflow.evaluation import effective_model
        from benchflow.runtime import check_credentials

        try:
            # The model the job's trials will use (bench eval run resolves the
            # agent's default the same way).
            trial_model = effective_model(agent, model, dict(agent_env or {}))
        except ValueError as exc:
            # An agent with no default model and no --model: the job would
            # fail on the same error once it started, so say it here.
            print_error(
                f"{exc}\nNo job was created.",
                next_step="pass --model with a model this agent supports",
            )
            raise typer.Exit(1) from None
        try:
            check_credentials(
                [SimpleNamespace(agent=agent, model=trial_model, agent_env=agent_env)]
            )
        except MissingCredentialError as exc:
            text, hint = expected_error_parts(exc)
            print_error(
                f"{text}\nNo job was created.",
                next_step=hint,
            )
            raise typer.Exit(1) from None
        _check_provider_key(agent, trial_model, agent_env)
    from benchflow import doctor as doctor_mod
    from benchflow._bfx.delivery import DELIVERY_ENV

    probes = doctor_mod.DoctorProbes.from_host()
    if (
        sandbox in ("docker", "remote-docker")
        and os.environ.get(DELIVERY_ENV, "").strip().lower() != "off"
    ):
        # A wheel built from source without `make bfx` runs without it; say so
        # at the start of the run, not only in a debug log (REL r4 M7).
        bfx = doctor_mod.check_bfx()
        if bfx.status != "pass":
            _print(
                err_console,
                f"[yellow]Warning:[/yellow] {bfx.summary}; {bfx.fix}.",
                highlight=False,
                soft_wrap=True,
            )
    if sandbox == "docker":
        failed = [
            c
            # Only failures stop a run: the warn-only buildx check is left out.
            for c in (
                docker.checks()
                if docker is not None
                else doctor_mod.check_docker(probes, required=True, buildx=False)
            )
            if c.status == "fail"
        ]
        if failed:
            # The failed checks first, so the error and its next step end it.
            _print_checks(failed)
            print_error(
                "Docker is not ready for --sandbox docker; no job was created.",
                next_step="run `bench doctor` for the full report, or set "
                f"{PREFLIGHT_OPT_OUT_ENV}=1 to skip this check",
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
                f"no job was created.\n  {exc}",
                next_step="check BENCHFLOW_REMOTE_DOCKER_HOST (or DOCKER_HOST) and that "
                "`docker info` works against that host, or set "
                f"{PREFLIGHT_OPT_OUT_ENV}=1 to skip this check",
            )
            raise typer.Exit(1) from None
    if sandbox == "daytona":
        # The same live key check doctor makes, before a job exists, so a
        # rejected key makes no job. Only that refuses: an API (or a machine)
        # too busy to answer lets the run go on to sandbox creation, which
        # retries within its budget (DAYTONA-START).
        daytona = doctor_mod.check_daytona(
            probes, required=True, offline=False, for_run=True
        )
        if daytona.status == "fail":
            _print_checks([daytona])
            print_error(
                "Daytona is not ready for --sandbox daytona; no job was created.",
                next_step="run `bench doctor --sandbox daytona` for the full "
                f"report, or set {PREFLIGHT_OPT_OUT_ENV}=1 to skip this check",
            )
            raise typer.Exit(1)
    if agent:
        claude = _expired_claude_login(
            probes, agent=agent, model=model, agent_env=agent_env or {}
        )
        if claude is not None:
            _print(
                err_console,
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
            _print(
                err_console,
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
                    "Sandbox the checks treat as required: docker (default), "
                    "remote-docker or daytona. Others are listed but not "
                    "checked yet."
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
            from benchflow._diagnostics import diagnostic_json
            from benchflow.cli._errors import _secrets

            typer.echo(diagnostic_json(data, _secrets(), indent=2))
            raise typer.Exit(1 if failed else 0)

        from benchflow import __version__

        _print(
            console,
            f"[bold]BenchFlow doctor[/bold] - benchflow {__version__}, "
            f"{doctor_mod.host_summary()}, sandbox {escape(sandbox)}\n",
            highlight=False,
        )
        render_report(report)
        if starts:
            _print(
                console,
            )
            _print_checks(starts)
        _print(
            console,
        )
        _print(console, _counts_line(report))
        if not failed:
            for line in _doctor_next_lines(report, sandbox):
                print_hint(line)
            return
        _print(
            console,
            "Fix the failed checks above, then run [cyan]bench doctor[/cyan] again.",
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

    def _line(row: Sequence[str], status: str | None = None) -> str:
        cells = [
            escape(cell.ljust(width)) for cell, width in zip(row, widths, strict=False)
        ]
        if status is not None:
            style = _STATUS_STYLE.get(status, "white")
            cells[2] = f"[{style}]{cells[2]}[/{style}]"
        return "  " + "  ".join(cells) + "  " + escape(row[5])

    _print(console, "[bold]Smoke results[/bold]")
    _print(console, _line(header), style="bold", highlight=False, soft_wrap=True)
    for outcome, row in zip(outcomes, rows, strict=True):
        _print(
            console,
            fields=row,
            render=lambda fields, status=outcome.status: _line(fields, status),
            highlight=False,
            soft_wrap=True,
        )
    for o in outcomes:
        if o.status != "pass":
            _print(
                console,
                fields=(o.target.agent, o.reason, _rel(o.log_path)),
                render=lambda fields: (
                    f"  [red]{escape(fields[0])}[/red]: {escape(fields[1])}\n"
                    f"    log: {escape(fields[2])}"
                ),
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
        _print(console, "Checking this machine (bench doctor)...", highlight=False)
        report = doctor_mod.run_doctor(sandbox=sandbox)
        blocking = [
            c for c in report.checks if c.status == "fail" and c.group != "agents"
        ]
        if any(c.status in ("warn", "fail") for c in report.checks):
            render_report(report, only_problems=True)
        _print(console, _counts_line(report))
        if blocking:
            print_error(
                "This machine failed checks the smoke needs; no smoke ran.",
                next_step="fix the failed checks above (`bench doctor` shows the "
                "full report), then run the smoke again",
            )
            raise typer.Exit(1)

        try:
            targets, skipped = doctor_smoke.plan_smoke(report, agent or [])
        except ValueError as exc:
            print_error(
                str(exc), next_step="`bench eval smoke --help` describes --agent"
            )
            raise typer.Exit(2) from None
        for skip in skipped:
            _print(
                console,
                fields=(skip.agent, skip.reason),
                render=lambda fields: (
                    f"[dim]skip {escape(fields[0])}: {escape(fields[1])}[/dim]"
                ),
                highlight=False,
                soft_wrap=True,
            )
        if not targets:
            print_error(
                "No agent to smoke: no working credential found.",
                next_step="the oracle needs no login: "
                f"`{oracle_quickstart(sandbox)}`; to smoke a model agent, log in "
                "(`bench doctor` shows how) or pass --agent NAME[=MODEL]",
            )
            raise typer.Exit(1)

        root = jobs_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
        total = len(targets)

        def _on_start(index: int, target: SmokeTarget) -> None:
            _print(
                console,
                fields=(target.agent, target.model, target.auth, sandbox),
                render=lambda fields: (
                    f"{escape(f'[{index + 1}/{total}]')} {escape(fields[0])} "
                    f"({escape(fields[1])}, auth {escape(fields[2])}) on "
                    f"{escape(fields[3])}..."
                ),
                highlight=False,
                soft_wrap=True,
            )

        def _on_done(index: int, outcome: SmokeOutcome) -> None:
            reward = "-" if outcome.reward is None else f"{outcome.reward:g}"
            _print(
                console,
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
        _print(
            console,
        )
        _render_outcomes(outcomes)
        _print(
            console,
            f"Summary: {escape(_rel(root / 'smoke-summary.json'))}",
            highlight=False,
            soft_wrap=True,
        )
        if all(o.status == "pass" for o in outcomes):
            for line in (
                f"Next: bench view "
                f"{local_path_arg(_rel(root), '<jobs folder>')} "
                "to read the smoke trajectories,",
                "      then bench run <task> --agent <agent> --model <model> "
                "on a real task.",
                f"      Guide: {_GETTING_STARTED_URL}",
            ):
                print_hint(line)
            return
        passed = sum(o.status == "pass" for o in outcomes)
        print_error(
            f"{len(outcomes) - passed} of {len(outcomes)} smoke runs did not pass.",
            next_step=f"`bench view {local_path_arg(_rel(root), '<jobs folder>')}` "
            "shows what each agent did",
        )
        raise typer.Exit(1)
