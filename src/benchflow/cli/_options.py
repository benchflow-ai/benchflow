"""Reusable Typer option aliases shared across benchflow CLI commands.

Each alias carries only the flag name, type, and help text. Per-command
defaults stay at the parameter declaration (``param: Alias = <default>``) so
commands keep their own defaults while sharing one definition of each flag.
Only flag/type/help combinations that recur identically across commands are
factored here; one-off variants stay inline in ``main.py``.
"""

from typing import Annotated

import typer

from benchflow._utils.sandbox_providers import providers_phrase

AgentOption = Annotated[str, typer.Option("--agent", help="Agent name")]
ModelOption = Annotated[
    str | None,
    typer.Option(
        "--model",
        "-m",
        help="Model id for the agent, e.g. claude-haiku-5-5 (none for oracle and nop)",
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
# One help line for --harness on every command that has it (bench run, bench
# eval run, bench eval branch): one set of values (dogfood p5-9).
HARNESS_HELP = (
    "How the agent runs: acp (default, its ACP adapter) or native (its "
    "own CLI in headless JSON mode; claude-agent-acp and codex-acp)"
)
HarnessOption = Annotated[
    str | None,
    typer.Option("--harness", help=HARNESS_HELP),
]
SkillModeOption = Annotated[
    str,
    typer.Option("--skill-mode", help="Skill mode: no-skill, with-skill, or self-gen"),
]
AllowTaskEnvOption = Annotated[
    list[str] | None,
    typer.Option(
        "--allow-task-env",
        help=(
            "Let tasks read this variable of your environment (a ${NAME} "
            "env value, or their docker-compose.yaml); repeatable. A task "
            "reading an unnamed one is refused; ${NAME:-default} gets the default"
        ),
    ),
]

FreshInstallOption = Annotated[
    bool,
    typer.Option(
        "--fresh-install",
        help=(
            "Install the model proxy and the agent in every trial instead of "
            "starting from their cached image layers (same as "
            "BENCHFLOW_FRESH_INSTALL=1)"
        ),
    ),
]
AllowCloudCredentialFileOption = Annotated[
    bool,
    typer.Option(
        "--allow-cloud-credential-file",
        help=(
            "Let a cloud credential file (the gcloud application default "
            "credentials) be staged in the sandbox: the Gemini CLI's own Vertex "
            "route, or a Vertex model through a proxy that runs in the sandbox. "
            "Each trial records it. Without it such a run is refused before it "
            "starts; the proxy on the host (Docker, `public`) keeps the file out "
            "of the sandbox and needs no flag"
        ),
    ),
]
AllowTaskProcessesWithCredentialsOption = Annotated[
    bool,
    typer.Option(
        "--allow-task-processes-with-credentials",
        help=(
            "Start a model agent whose operator credentials go into the sandbox "
            "(a subscription login, its own API key, or a key the proxy in the "
            "sandbox holds) while the task's own processes run in main, where "
            "they could read them. Each trial records it. Without it such a "
            "trial is refused before anything is staged; the proxy on the host "
            "(Docker, `public`) keeps an API key out of the sandbox"
        ),
    ),
]
ScopedCredentialOption = Annotated[
    list[str] | None,
    typer.Option(
        "--scoped-credential",
        help=(
            "Name a credential variable that is this run's own (a per-run "
            "token, not one of your logins or keys): repeatable. It still goes "
            "into the sandbox and each trial records it, but a task process in "
            "main is not refused for it. Refused before the run starts when the "
            "agent's env does not set it, or when it is staged as a file"
        ),
    ),
]
AllowCloudMetadataOption = Annotated[
    bool,
    typer.Option(
        "--allow-cloud-metadata",
        help=(
            "Let the agent reach the cloud metadata endpoints under an open "
            "network (no block; each trial records the opt-out), and grant a "
            "task's own allow_cloud_metadata key, which lifts the block for "
            "every process of its trials. Without it the agent starts only once "
            "the block is proven on, and a task that sets that key is refused "
            "before it starts"
        ),
    ),
]
