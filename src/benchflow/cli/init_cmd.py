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
    discoverability (the Hermes-style wizard pattern); free-text escape
    hatches are modeled as an explicit "other" option by the caller.
    """
    typer.echo(f"\n{title}")
    for i, (label, desc) in enumerate(options, 1):
        suffix = f"  — {desc}" if desc else ""
        typer.echo(f"  {i}) {label}{suffix}")
    return typer.prompt("Select", type=click.IntRange(1, len(options)), default=default)


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

        # Agent first (the thing the user came to benchmark), then the
        # provider menu narrows to what that agent can route.
        if not agent:
            agents = onboarding.compatible_agents()
            default = agents.index("pi-acp") + 1 if "pi-acp" in agents else 1
            pick = _choose("Agent:", [(a, "") for a in agents], default=default)
            agent = agents[pick - 1]

        if not model:
            from benchflow.agents.providers import PROVIDERS

            names = onboarding.compatible_providers(agent)
            options = [
                (n, ", ".join(PROVIDERS[n].model_prefixes) or PROVIDERS[n].api_protocol)
                for n in names
            ]
            options.append(("other", "type a full model id yourself"))
            default = names.index("deepseek") + 1 if "deepseek" in names else 1
            pick = _choose(f"Provider (routable by {agent}):", options, default=default)
            if pick == len(options):  # other
                model = typer.prompt("Model id (provider/model or bare)")
            else:
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
        # would reject a protocol mismatch, so init must too.
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
                opts = choices + [("a local tasks dir", "path to your own tasks")]
                pick = _choose("Task set:", opts, default=1)
                if pick == len(opts):
                    d = typer.prompt("Tasks dir path")
                    # bare relative names must route to --tasks-dir, not the
                    # registry-spec validation below
                    dataset = d if "/" in d or d.startswith(".") else f"./{d}"
                else:
                    dataset = choices[pick - 1][0]
            else:
                dataset = typer.prompt(
                    "Task set (dataset spec or tasks dir; registry unreachable)",
                    default="skillsbench@1.1",
                )
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
        if not sandbox:
            from benchflow.sandbox.providers import SANDBOX_PROVIDERS

            pick = _choose("Sandbox:", [(s, "") for s in SANDBOX_PROVIDERS])
            sandbox = SANDBOX_PROVIDERS[pick - 1]

        # Credentials: explicit --api-key > auto-detection (subscription
        # login, then the environment incl. the saved setup, then ./.env in
        # the working folder) > hidden prompt as the last resort. Stored keys
        # land in the private env file future runs auto-load.
        if auth_type == "api_key" and auth_env:
            try:
                if api_key:
                    exported = os.environ.get(auth_env)
                    if exported and exported != api_key:
                        typer.echo(
                            f"warning: {auth_env} is exported with a different"
                            " value — the exported variable will shadow the"
                            " saved key at run time.",
                            err=True,
                        )
                    onboarding.write_env_file(home / ".env", {auth_env: api_key})
                    os.environ.setdefault(auth_env, api_key)
                else:
                    source, value = onboarding.detect_key(auth_env, agent=agent)
                    if source == "subscription":
                        typer.echo(
                            f"✓ Using {agent}'s host subscription login"
                            f" (no {auth_env} needed)."
                        )
                    elif source == "environment":
                        typer.echo(
                            f"✓ {auth_env} found in your environment (or saved"
                            " setup) — using it."
                        )
                    elif source == "./.env":
                        src = Path.cwd() / ".env"
                        tail = value[-4:] if len(value) > 4 else "****"
                        if sys.stdin.isatty() and not typer.confirm(
                            f"Use {auth_env}=…{tail} from {src} and save it"
                            f" to {home / '.env'}?",
                            default=True,
                        ):
                            key = typer.prompt(f"{auth_env}", hide_input=True)
                            onboarding.write_env_file(home / ".env", {auth_env: key})
                            os.environ.setdefault(auth_env, key)
                        else:
                            typer.echo(
                                f"✓ {auth_env} (…{tail}) from {src} — saved"
                                f" to {home / '.env'}."
                            )
                            onboarding.write_env_file(home / ".env", {auth_env: value})
                            os.environ.setdefault(auth_env, value)
                    else:
                        key = typer.prompt(f"{auth_env}", hide_input=True)
                        onboarding.write_env_file(home / ".env", {auth_env: key})
                        if not os.environ.get(auth_env):
                            os.environ[auth_env] = key
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
                    f"\nStage-1 smoke (oracle, no credentials): {' '.join(argv)}"
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
        typer.echo("\nReady. Run your first eval with:\n")
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
