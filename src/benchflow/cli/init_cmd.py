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
    ) -> None:
        """Guided first-run setup: model → agent → tasks → sandbox → creds → smoke."""
        home = benchflow_home()

        model = model or typer.prompt("Model (provider/model)")
        resolved = onboarding.resolve_provider(model)
        if not resolved:
            typer.echo(f"No registered provider recognizes {model!r}.", err=True)
            raise typer.Exit(1)
        prov_name, prov_cfg = resolved

        offered = onboarding.compatible_agents(model)
        if not agent:
            typer.echo(f"Agents able to route {model} ({len(offered)}):")
            typer.echo("  " + ", ".join(offered))
            agent = typer.prompt(
                "Agent", default="pi-acp" if "pi-acp" in offered else offered[0]
            )
        if agent not in offered:
            typer.echo(
                f"Agent {agent!r} cannot route {model!r} ({prov_name} wire protocol"
                " mismatch) — the run would reject it. Compatible agents:\n  "
                + ", ".join(offered),
                err=True,
            )
            raise typer.Exit(1)

        dataset = dataset or typer.prompt(
            "Task set (dataset name or tasks dir)", default="skillsbench"
        )
        sandbox = sandbox or typer.prompt("Sandbox", default="docker")

        # Credentials: reuse an already-set env var, else store the given/prompted
        # key in the private env file future runs auto-load.
        if prov_cfg.auth_type == "api_key" and prov_cfg.auth_env:
            if api_key:
                onboarding.write_env_file(home / ".env", {prov_cfg.auth_env: api_key})
                os.environ.setdefault(prov_cfg.auth_env, api_key)
            elif os.environ.get(prov_cfg.auth_env):
                typer.echo(
                    f"Using {prov_cfg.auth_env} already set in your environment."
                )
            else:
                key = typer.prompt(f"{prov_cfg.auth_env}", hide_input=True)
                onboarding.write_env_file(home / ".env", {prov_cfg.auth_env: key})
                os.environ.setdefault(prov_cfg.auth_env, key)

        prefs = {
            "agent": agent,
            "model": model,
            "dataset": dataset,
            "sandbox": sandbox,
            "skill_mode": skill_mode,
        }
        onboarding.save_prefs(home / "config.toml", prefs)

        if not skip_smoke:
            typer.echo("\nSmoke test:")
            env = dict(os.environ)
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
