"""Reusable Typer option aliases shared across benchflow CLI commands.

Each alias carries only the flag name, type, and help text. Per-command
defaults stay at the parameter declaration (``param: Alias = <default>``) so
commands keep their own defaults while sharing one definition of each flag.
Only flag/type/help combinations that recur identically across commands are
factored here; one-off variants stay inline in ``main.py``.
"""

from typing import Annotated

import typer

from benchflow.sandbox.providers import providers_phrase

AgentOption = Annotated[str, typer.Option("--agent", help="Agent name")]
ModelOption = Annotated[
    str | None,
    typer.Option(
        "--model",
        help="Model id for the agent, e.g. claude-haiku-4-5-20251001 (none for oracle and nop)",
    ),
]
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
HarnessOption = Annotated[
    str | None,
    typer.Option(
        "--harness",
        help=(
            "How the agent runs: acp (default, its ACP adapter) or native (its "
            "own CLI in headless JSON mode; claude-agent-acp and codex-acp)"
        ),
    ),
]
SkillModeOption = Annotated[
    str,
    typer.Option("--skill-mode", help="Skill mode: no-skill, with-skill, or self-gen"),
]
