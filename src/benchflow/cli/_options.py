"""Reusable Typer option aliases shared across benchflow CLI commands.

Each alias carries only the flag name, type, and help text. Per-command
defaults stay at the parameter declaration (``param: Alias = <default>``) so
commands keep their own defaults while sharing one definition of each flag.
Only flag/type/help combinations that recur identically across commands are
factored here; one-off variants stay inline in ``main.py``.
"""

from typing import Annotated

import typer

AgentOption = Annotated[str, typer.Option("--agent", help="Agent name")]
ModelOption = Annotated[str | None, typer.Option("--model", help="Model")]
SandboxOption = Annotated[
    str, typer.Option("--sandbox", help="Sandbox: docker, daytona, modal, or cua")
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
