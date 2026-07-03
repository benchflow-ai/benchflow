"""`bench init` (guided onboarding wizard) + `bench doctor` (health checks).

Thin typer glue over :mod:`benchflow.onboarding`. Every prompt has a flag
mirror, so a fully-flagged invocation never blocks on stdin (CI mode); the
same check functions back both the wizard's closing smoke test and the
standalone doctor command.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click
import typer

from benchflow import onboarding
from benchflow.cli._shared import console

BENCHFLOW_HOME_ENV = "BENCHFLOW_HOME"


def benchflow_home() -> Path:
    return Path(os.environ.get(BENCHFLOW_HOME_ENV) or Path.home() / ".benchflow")


def _echo_results(results: list[onboarding.CheckResult]) -> bool:
    all_ok = True
    for r in results:
        mark = "○" if r.skipped else ("✅" if r.ok else "❌")
        typer.echo(f"  {mark} {r.name}: {r.detail}")
        all_ok &= r.ok
    return all_ok


def _skip_note(results: list[onboarding.CheckResult]) -> str:
    n = sum(1 for r in results if r.skipped)
    return f" ({n} check(s) skipped — not verifiable before run time)" if n else ""


def _choose(title: str, options: list[tuple[str, str]], default: int | None = 1) -> int:
    """Numbered menu: print options, return the chosen 1-based index.

    Enter accepts the default. Selection beats free-form typing for
    discoverability; the ◆/│ step visuals follow the OpenClaw/clack wizard
    look (rich degrades to plain text on non-tty). Free-text escape hatches
    are modeled as an explicit "other" option by the caller.
    """
    from rich.markup import escape

    console.print(f"\n[bold cyan]◆[/] [bold]{escape(title)}[/]")
    for i, (label, desc) in enumerate(options, 1):
        suffix = f"  [dim]— {escape(desc)}[/]" if desc else ""
        console.print(f"[dim]│[/]  [cyan]{i})[/] {escape(label)}{suffix}")
    return typer.prompt("Select", type=click.IntRange(1, len(options)), default=default)


def _step(label: str, value: str) -> None:
    """Confirmed-step rail line (the wizard's running summary)."""
    from rich.markup import escape

    console.print(f"[dim]│[/] [green]●[/] {escape(label)}  [bold]{escape(value)}[/]")


def _isatty() -> bool:
    """Interactive-session check (module-level seam so tests can drive the
    tty-only menus through CliRunner, which swaps sys.stdin)."""
    return sys.stdin.isatty()


def _warn_shadow(auth_env: str, value: str) -> None:
    """The shell export wins over anything init saves — say so."""
    exported = os.environ.get(auth_env)
    if exported and exported != value:
        typer.echo(
            f"warning: {auth_env} is exported in your shell with a different"
            " value — the exported variable will shadow this choice at run"
            " time unless you unset it.",
            err=True,
        )


def _fingerprint(value: str | None) -> str:
    return f"…{value[-4:]}" if value and len(value) > 4 else "…"


def _auth_option(source: str, value: str | None, agent: str, auth_env: str):
    if source == "subscription":
        return (
            f"{agent}'s subscription login",
            f"host login detected — no {auth_env} needed",
        )
    if source == "environment":
        return (
            f"{auth_env} from your environment ({_fingerprint(value)})",
            "exported variable or saved setup",
        )
    return (
        f"{auth_env} from {Path.cwd() / '.env'} ({_fingerprint(value)})",
        "will be saved to your bench setup",
    )


def _wizard_auth_step(home: Path, agent: str, auth_env: str) -> None:
    """OpenClaw-style auth step: every detected credential source is a listed
    choice (subscription login included), manual entry is the escape hatch.

    Interactive (tty): a menu, defaulting to what the run path would use.
    Non-interactive: the run-path-preferred source is taken automatically —
    CI never blocks on stdin.
    """
    sources = onboarding.detect_key_sources(auth_env, agent=agent)
    use: tuple[str, str | None] | None = None
    if sources and _isatty():
        options = [_auth_option(s, v, agent, auth_env) for s, v in sources]
        options.append(("enter an API key", "typed hidden, stored for future runs"))
        pick = _choose("Credentials:", options, default=1)
        if pick <= len(sources):
            use = sources[pick - 1]
    elif sources:
        use = sources[0]
        label, _ = _auth_option(use[0], use[1], agent, auth_env)
        typer.echo(f"✓ Using {label}.")
    if use is None:
        key = typer.prompt(f"{auth_env}", hide_input=True)
        _warn_shadow(auth_env, key)
        onboarding.write_env_file(home / ".env", {auth_env: key})
        os.environ[auth_env] = key  # the explicit choice wins in-process
    elif use[0] == "./.env":
        value = use[1] or ""  # detect_key_sources only lists ./.env with a value
        _warn_shadow(auth_env, value)
        onboarding.write_env_file(home / ".env", {auth_env: value})
        os.environ[auth_env] = value  # the explicit choice wins in-process
        typer.echo(f"✓ {auth_env} ({_fingerprint(value)}) saved to {home / '.env'}.")
    elif use[0] == "environment":
        typer.echo(
            f"✓ Using {auth_env} from your environment ({_fingerprint(use[1])})."
        )
    elif use[0] == "subscription":
        if os.environ.get(auth_env):
            # Honor the choice for this process (so the smoke verifies the
            # subscription setup, not the declined key) and warn that the
            # shell export will shadow it at run time.
            typer.echo(
                f"warning: {auth_env} is exported in your shell — it will"
                " shadow the subscription login at run time unless you unset"
                " it.",
                err=True,
            )
            os.environ.pop(auth_env, None)
        typer.echo(f"✓ Using {agent}'s subscription login (no {auth_env} needed).")


def _osc52_copy(text: str) -> None:
    """Best-effort clipboard copy via OSC52; harmless where unsupported."""
    import base64

    if sys.stdout.isatty():
        payload = base64.b64encode(text.encode()).decode()
        sys.stdout.write(f"\x1b]52;c;{payload}\x07")
        sys.stdout.flush()


def register_init(app: typer.Typer) -> None:
    @app.command("init", rich_help_panel="Core")
    def init(
        model: str = typer.Option(
            None, "--model", "-m", help="Model id, e.g. deepseek/deepseek-v4-flash."
        ),
        agent: str = typer.Option(None, "--agent", "-a", help="Agent to run."),
        dataset: str = typer.Option(
            None,
            "--dataset",
            "-d",
            help="Dataset spec (e.g. skillsbench@1.1) or a tasks dir path.",
        ),
        sandbox: str = typer.Option(
            None, "--sandbox", help="Sandbox provider (docker, daytona, ...)"
        ),
        api_key: str = typer.Option(
            None,
            "--api-key",
            help="Provider API key to store (falls back to subscription"
            " login, then env var, then a hidden prompt).",
        ),
        skill_mode: str = typer.Option("with-skill", "--skill-mode"),
        skip_smoke: bool = typer.Option(
            False, "--skip-smoke", help="Skip the post-setup smoke test."
        ),
        full_smoke: bool = typer.Option(
            False,
            "--full-smoke",
            help="Also run the credential-free oracle agent on one task in the"
            " chosen sandbox (a real eval — takes minutes).",
        ),
        smoke_task: str = typer.Option(
            None, "--smoke-task", help="Task name for --full-smoke's oracle run."
        ),
    ) -> None:
        """Guided first-run setup: model → agent → tasks → sandbox → creds → smoke."""
        home = benchflow_home()

        if full_smoke and not smoke_task:
            typer.echo("--full-smoke requires --smoke-task <task-name>.", err=True)
            raise typer.Exit(2)

        console.print("[bold cyan]◇ bench init[/] [dim]— first-run setup[/]")

        # Agent first (the thing the user came to benchmark). The menu lists
        # locally-registered ACP agents only — populating it never touches the
        # network. "other" (and the --agent flag) reach the full catalog, and
        # the repo is cloned lazily there, only when an agent is actually
        # resolved for a run.
        if not agent:
            agents = onboarding.acp_agents()
            default = agents.index("pi-acp") + 1 if "pi-acp" in agents else 1
            options = [(a, "") for a in agents]
            options.append(
                ("other", "browse the full catalog (acp / ai-sdk / omnigent)")
            )
            pick = _choose("Agent:", options, default=default)
            if pick == len(options):
                # Browse, don't type: fetch the full catalog HERE (the one
                # lazy point) and offer path -> agent menus.
                console.print("[dim]│ fetching the agent catalog…[/]")
                paths = onboarding.catalog_paths()
                path_names = list(paths)
                ppick = _choose(
                    "Agent path:",
                    [(p, f"{len(paths[p])} agents") for p in path_names],
                    default=path_names.index("acp") + 1 if "acp" in path_names else 1,
                )
                cat_agents = paths[path_names[ppick - 1]]
                apick = _choose("Agent:", [(a, "") for a in cat_agents], default=1)
                agent = cat_agents[apick - 1]
            else:
                agent = agents[pick - 1]
        # Resolve like `bench eval run` does: aliases + the miss-driven catalog
        # autoload (the only place the agents repo is cloned). A menu pick is
        # already local, so this is a no-op for it; a typed/flagged catalog
        # name resolves here.
        from benchflow.agents.registry import resolve_agent

        try:
            agent = resolve_agent(agent).name
        except KeyError:
            typer.echo(f"Unknown agent {agent!r} — see `bench agent list`.", err=True)
            raise typer.Exit(1) from None
        _step("agent", agent)

        if not model:
            from benchflow.agents.providers import PROVIDERS
            from benchflow.agents.registry import AGENTS as _AGENTS

            names = onboarding.compatible_providers(agent)
            _acfg = _AGENTS.get(agent)
            _aproto = (_acfg.api_protocol or "") if _acfg else ""

            def _label(n: str) -> str:
                cfg = PROVIDERS[n]
                # a lone prefix equal to the provider name adds nothing
                prefixes = [p for p in cfg.model_prefixes if p != n]
                if prefixes:
                    return ", ".join(prefixes)
                if not cfg.base_url:
                    return "BYO base URL"
                # the endpoint this AGENT will use, not the provider's primary
                return _aproto if _aproto in cfg.all_endpoints else cfg.api_protocol

            # Some agents have no registry endpoint but a first-class run
            # path via a well-known inferred key: anthropic-native agents
            # (subscription login / ANTHROPIC_API_KEY) and gemini (Google's
            # native wire / GEMINI_API_KEY). Offer that as a synthetic entry,
            # listed first + default, or those auth paths can never be
            # reached from the menus.
            synthetic = None  # (label, desc, default model)
            if _acfg and (
                (
                    _acfg.subscription_auth
                    and _acfg.subscription_auth.replaces_env == "ANTHROPIC_API_KEY"
                )
                or _aproto == "anthropic-messages"
            ):
                synthetic = (
                    "anthropic",
                    "claude-* via subscription login or ANTHROPIC_API_KEY",
                    "claude-sonnet-4-6",
                )
            elif agent == "gemini":
                # gemini never routes through the provider registry — the
                # 21-provider list would be a lie for it.
                synthetic = ("google", "gemini-* via GEMINI_API_KEY", "gemini-3-pro")
                names = []
            options = [(n, _label(n)) for n in names]
            if synthetic:
                options.insert(0, (synthetic[0], synthetic[1]))
            options.append(("other", "type a full model id yourself"))
            if synthetic:
                default = 1
            elif "deepseek" in names:
                default = names.index("deepseek") + 1
            elif "openai" in names:
                default = names.index("openai") + 1
            else:
                default = 1
            pick = _choose(f"Provider (routable by {agent}):", options, default=default)
            if synthetic and pick == 1:
                model = typer.prompt("Model id", default=synthetic[2])
            elif pick == len(options):  # other
                model = typer.prompt("Model id (provider/model or bare)")
            else:
                if synthetic:
                    pick -= 1  # past the synthetic entry
                prov = PROVIDERS[names[pick - 1]]
                catalog = [
                    str(m.get("id") or m.get("name"))
                    for m in (prov.models or [])
                    if m.get("id") or m.get("name")
                ]
                if catalog:
                    mp = _choose(
                        f"Model ({names[pick - 1]}):",
                        [(m, "") for m in catalog] + [("other", "type a model id")],
                    )
                    bare = (
                        catalog[mp - 1]
                        if mp <= len(catalog)
                        else typer.prompt("Model id")
                    )
                else:
                    hint = (
                        f" (e.g. {prov.model_prefixes[0]}-...)"
                        if prov.model_prefixes
                        else ""
                    )
                    bare = typer.prompt(f"Model id{hint}")
                model = bare if "/" in bare else f"{names[pick - 1]}/{bare}"
        _step("model", model)
        resolved = onboarding.resolve_provider(model)
        if resolved:
            prov_name, prov_cfg = resolved
            auth_type, auth_env = prov_cfg.auth_type, prov_cfg.auth_env
        else:
            # Well-known model families (claude-*, gpt-*, gemini-*) run via
            # their inferred key even without a registered provider endpoint.
            from benchflow.agents.registry import infer_env_key_for_model

            inferred = infer_env_key_for_model(model)
            if not inferred:
                typer.echo(f"No registered provider recognizes {model!r}.", err=True)
                raise typer.Exit(1)
            prov_name, prov_cfg = None, None
            auth_type, auth_env = "api_key", inferred

        # Consistency gate (also covers flag combinations): the run path
        # would reject a protocol mismatch, so init must too. Native-wire
        # agents (gemini) never route through the provider registry — for
        # them the check is the model FAMILY, not the wire protocol.
        if agent == "gemini":
            from benchflow.agents.registry import infer_env_key_for_model

            if infer_env_key_for_model(model) != "GEMINI_API_KEY":
                typer.echo(
                    f"Agent 'gemini' runs gemini-* models only; {model!r} is not one.",
                    err=True,
                )
                raise typer.Exit(1)
            offered = [agent]
        else:
            offered = onboarding.compatible_agents(model)
        if agent not in offered:
            typer.echo(
                f"Agent {agent!r} cannot route {model!r} ({prov_name or 'provider'}"
                " wire protocol mismatch) — the run would reject it. Compatible"
                " agents:\n  " + ", ".join(offered),
                err=True,
            )
            raise typer.Exit(1)

        if not dataset:
            choices = onboarding.dataset_choices()
            if choices:
                opts = [*choices, ("a local tasks dir", "path to your own tasks")]
                pick = _choose("Task set:", opts, default=1)
                if pick == len(opts):
                    d = typer.prompt("Tasks dir path")
                    dataset = onboarding.normalize_dataset_input(
                        d, local_tasks_dir=True
                    )
                else:
                    dataset = choices[pick - 1][0]
            else:
                dataset = typer.prompt(
                    "Task set (dataset spec or tasks dir; registry unreachable)",
                    default="skillsbench@1.1",
                )
        dataset = onboarding.normalize_dataset_input(dataset)
        if "/" not in dataset and not dataset.startswith("."):
            # Registry-style name: must parse as <name>@<version> or the
            # printed command will not run.
            from benchflow._utils.dataset_registry import parse_dataset_spec

            try:
                parse_dataset_spec(dataset)
            except Exception as exc:
                typer.echo(
                    f"Dataset {dataset!r} is not a valid spec: {exc}\n"
                    "Use <name>@<version> (e.g. skillsbench@1.1) or a tasks"
                    " dir path.",
                    err=True,
                )
                raise typer.Exit(1) from exc
        _step("task set", dataset)
        if not sandbox:
            from benchflow.sandbox.providers import SANDBOX_PROVIDERS

            pick = _choose("Sandbox:", [(s, "") for s in SANDBOX_PROVIDERS])
            sandbox = SANDBOX_PROVIDERS[pick - 1]
        _step("sandbox", sandbox)

        # Credentials: explicit --api-key > auto-detection (./.env in the
        # working folder, then the environment incl. the saved setup, then
        # subscription login) > hidden prompt as the last resort. Stored keys
        # land in the private env file future runs auto-load.
        if auth_type == "api_key" and auth_env:
            try:
                if api_key:
                    _warn_shadow(auth_env, api_key)
                    onboarding.write_env_file(home / ".env", {auth_env: api_key})
                    os.environ.setdefault(auth_env, api_key)
                else:
                    _wizard_auth_step(home, agent, auth_env)
            except OSError as exc:
                typer.echo(
                    f"Could not save credentials to {home / '.env'}: {exc}",
                    err=True,
                )
                raise typer.Exit(1) from exc
        elif api_key:
            typer.echo(
                f"--api-key is not used by provider {prov_name!r} (auth:"
                f" {auth_type}) — configure {auth_type} credentials instead.",
                err=True,
            )
            raise typer.Exit(1)

        prefs = {
            "agent": agent,
            "model": model,
            "dataset": dataset,
            "sandbox": sandbox,
            "skill_mode": skill_mode,
        }
        onboarding.save_prefs(home / "config.toml", prefs)

        if not skip_smoke:
            if full_smoke and smoke_task:
                # Stage 1 (Harbor's oracle pattern): prove install + sandbox
                # plumbing with NO credentials involved.
                import subprocess

                argv = onboarding.smoke_argv(prefs, task=smoke_task)
                typer.echo(
                    "\nStage-1 smoke (oracle, no credentials): "
                    f"{onboarding.shell_join(argv)}"
                )
                try:
                    oracle = subprocess.run(argv)
                except FileNotFoundError as exc:
                    typer.echo(
                        "Cannot run the oracle smoke: `bench` is not on PATH"
                        f" ({exc}). Add your install's bin directory to PATH"
                        " and re-run, or use `bench doctor`.",
                        err=True,
                    )
                    raise typer.Exit(1) from exc
                mark = "✅" if oracle.returncode == 0 else "❌"
                typer.echo(f"  {mark} oracle sandbox run (rc={oracle.returncode})")
                if oracle.returncode != 0:
                    typer.echo(
                        "\nSetup saved, but the stage-1 oracle smoke failed —"
                        " the sandbox plumbing is broken independent of your"
                        " credentials.",
                        err=True,
                    )
                    raise typer.Exit(1)
            typer.echo("\nSmoke test:")
            env = dict(os.environ)
            if api_key and auth_env:
                # Verify the key that was just SAVED, not whatever happened to
                # be exported before init ran.
                env[auth_env] = api_key
            results = onboarding.run_doctor(model, sandbox, env, agent=agent)
            if not _echo_results(results):
                typer.echo(
                    "\nSetup saved, but the smoke test failed — fix the ❌ rows"
                    " above and re-check with `bench doctor`.",
                    err=True,
                )
                raise typer.Exit(1)

        cmd = onboarding.final_command(prefs)
        console.print("\n[bold green]◆ Ready.[/] Run your first eval with:\n")
        typer.echo(f"  {cmd}\n")
        _osc52_copy(cmd)

    @app.command("doctor", rich_help_panel="Core")
    def doctor() -> None:
        """Re-validate the saved setup: sandbox, provider key, LiteLLM route,
        model ping."""
        home = benchflow_home()
        try:
            prefs = onboarding.load_prefs(home / "config.toml")
        except Exception:
            prefs = None  # corrupt TOML — same remediation as missing keys
        if not prefs or not {"model", "sandbox"} <= prefs.keys():
            typer.echo(
                "Saved setup is missing or incomplete — run `bench init` first.",
                err=True,
            )
            raise typer.Exit(1)
        onboarding.load_env_file(home / ".env")
        results = onboarding.run_doctor(
            prefs["model"],
            prefs["sandbox"],
            dict(os.environ),
            agent=prefs.get("agent"),
        )
        ok = _echo_results(results)
        note = _skip_note(results)
        typer.echo(f"\nAll checks passed.{note}" if ok else "\nSome checks failed.")
        raise typer.Exit(0 if ok else 1)
