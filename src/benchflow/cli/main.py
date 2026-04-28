"""benchflow CLI — agent benchmarking framework.

Root Typer app shell. Each sub-app lives in its own ``cli/<name>.py``
module per PLAN_V2_impl §13.4 / §13.6. main.py is the only file that
knows the full command tree; sub-app modules are independent.
"""

from __future__ import annotations

import typer

from benchflow.cli import legacy as _legacy
from benchflow.cli import run as _run_module
from benchflow.cli.agent import agent_app
from benchflow.cli.env import env_app
from benchflow.cli.eval import eval_app
from benchflow.cli.skills import skills_app
from benchflow.cli.tasks import tasks_app
from benchflow.cli.train import train_app

app = typer.Typer(
    name="benchflow",
    help="ACP-native agent benchmarking framework.",
    no_args_is_help=True,
)

# Root command: bench run (the primary entry point).
_run_module.register(app)

# Resource-verb sub-apps (the 0.3 CLI shape).
app.add_typer(skills_app, name="skills")
app.add_typer(tasks_app, name="tasks")
app.add_typer(agent_app, name="agent")
app.add_typer(eval_app, name="eval")
app.add_typer(train_app, name="train")
app.add_typer(env_app, name="environment")

# Hidden, deprecated top-level commands (removed when callers stop using them).
_legacy.register(app)


if __name__ == "__main__":
    app()
