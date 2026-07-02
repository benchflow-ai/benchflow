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

import typer

from benchflow import onboarding

BENCHFLOW_HOME_ENV = "BENCHFLOW_HOME"


def benchflow_home() -> Path:
    return Path(os.environ.get(BENCHFLOW_HOME_ENV) or Path.home() / ".benchflow")


def _echo_results(results: list[onboarding.CheckResult]) -> bool:
    all_ok = True
    for r in results:
        mark = "✅" if r.ok else "❌"
        typer.echo(f"  {mark} {r.name}: {r.detail}")
        all_ok &= r.ok
    return all_ok


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
            help="Dataset name (e.g. skillsbench) or a tasks dir path.",
        ),
        sandbox: str = typer.Option(None, "--sandbox", help="docker | daytona"),
        api_key: str = typer.Option(
            None, "--api-key", help="Provider API key to store (else prompted/hidden)."
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

        model = model or typer.prompt("Model (provider/model)")
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

        offered = onboarding.compatible_agents(model)
        if not agent:
            typer.echo(f"Agents able to route {model} ({len(offered)}):")
            typer.echo("  " + ", ".join(offered))
            agent = typer.prompt(
                "Agent", default="pi-acp" if "pi-acp" in offered else offered[0]
            )
        if agent not in offered:
            typer.echo(
                f"Agent {agent!r} cannot route {model!r} ({prov_name or 'provider'}"
                " wire protocol mismatch) — the run would reject it. Compatible"
                " agents:\n  " + ", ".join(offered),
                err=True,
            )
            raise typer.Exit(1)

        dataset = dataset or typer.prompt(
            "Task set (dataset name or tasks dir)", default="skillsbench"
        )
        sandbox = sandbox or typer.prompt("Sandbox", default="docker")

        # Credentials: subscription login > already-set env var > given/prompted
        # key stored in the private env file future runs auto-load.
        if auth_type == "api_key" and auth_env:
            from benchflow.agents.env import check_subscription_auth

            if api_key:
                onboarding.write_env_file(home / ".env", {auth_env: api_key})
                os.environ.setdefault(auth_env, api_key)
            elif check_subscription_auth(agent, auth_env):
                typer.echo(
                    f"Using {agent}'s host subscription login (no {auth_env} needed)."
                )
            elif os.environ.get(auth_env):
                typer.echo(f"Using {auth_env} already set in your environment.")
            else:
                key = typer.prompt(f"{auth_env}", hide_input=True)
                onboarding.write_env_file(home / ".env", {auth_env: key})
                os.environ.setdefault(auth_env, key)
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
                oracle = subprocess.run(argv)
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
            if not _echo_results(onboarding.run_doctor(model, sandbox, env)):
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
        """Re-validate the saved setup: sandbox, provider key, model ping."""
        home = benchflow_home()
        prefs = onboarding.load_prefs(home / "config.toml")
        if not prefs:
            typer.echo("No saved setup found — run `bench init` first.", err=True)
            raise typer.Exit(1)
        onboarding.load_env_file(home / ".env")
        results = onboarding.run_doctor(
            prefs["model"], prefs["sandbox"], dict(os.environ)
        )
        ok = _echo_results(results)
        typer.echo("\nAll checks passed." if ok else "\nSome checks failed.")
        raise typer.Exit(0 if ok else 1)
