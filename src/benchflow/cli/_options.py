"""Reusable Typer option aliases shared across benchflow CLI commands.

Each alias carries only the flag name, type, and help text. Per-command
defaults stay at the parameter declaration (``param: Alias = <default>``) so
commands keep their own defaults while sharing one definition of each flag.
Only flag/type/help combinations that recur identically across commands are
factored here; one-off variants stay inline in ``main.py``.
"""

from pathlib import Path
from typing import Annotated

import typer

from benchflow.sandbox.providers import providers_phrase

AgentOption = Annotated[str, typer.Option("--agent", help="Agent name")]
EnvironmentManifestOption = Annotated[
    Path | None,
    typer.Option(
        "--environment-manifest",
        help=(
            "Environment-plane manifest applied to every rollout: a path to "
            "an environment.toml, OR a 'name@version' registry spec (the S "
            "axis) resolved via $BENCHFLOW_ENV_REGISTRY when set, else the "
            "built-in registry shipped with benchflow (env0@prod, "
            "env0@outage). The manifest-declared stateful environment is "
            "provisioned, gated on readiness, and torn down."
        ),
    ),
]
ModelOption = Annotated[str | None, typer.Option("--model", help="Model")]
SandboxOption = Annotated[
    str, typer.Option("--sandbox", help=f"Sandbox: {providers_phrase()}")
]
ConcurrencyOption = Annotated[
    int, typer.Option("--concurrency", help="Max concurrent tasks")
]
JobsDirOption = Annotated[
    str, typer.Option("--jobs-dir", help="Output directory for results")
]
MonitorJobsDirOption = Annotated[
    str, typer.Option("--jobs-dir", help="Output root for monitor artifacts.")
]
SkillModeOption = Annotated[
    str,
    typer.Option("--skill-mode", help="Skill mode: no-skill, with-skill, or self-gen"),
]
