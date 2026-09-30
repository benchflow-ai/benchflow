"""The run_bash/submit harness settings every RL cookbook shares.

Training (TRL on HF Jobs, and the other cookbooks) and held-out evaluation
(``evaluate.py``) build their harness from :func:`harness_config`, so a policy
is evaluated with the same tools, instructions, and limits it trained with.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from benchflow.integrations.trl import BashHarnessConfig

# Appended to each task prompt, as TRL appends an environment's reset() text.
HARNESS_MESSAGE = (
    "\n\nYou are working in a Linux sandbox. Use the run_bash tool to run shell "
    "commands; each call starts in /workdir. When you are done, call the submit tool "
    "once with your final answer: it writes the answer to /workdir/answer.txt for you "
    "and ends the task. For a code fix, submit the word done."
)

BASH_TIMEOUT_SEC = 30
MAX_OUTPUT_CHARS = 2000
# Tool-calling turns per episode (TRL: max_tool_calling_iterations).
MAX_TURNS = 10


def harness_config(
    *,
    environment: str = "daytona",
    jobs_dir: str | Path = "jobs/rl",
    background_start: bool = False,
    **overrides: Any,
) -> BashHarnessConfig:
    """The shared harness, with the sandbox backend and jobs folder of the caller."""

    settings: dict[str, Any] = {
        "environment": environment,
        "sandbox_user": "agent",
        "jobs_dir": jobs_dir,
        "bash_timeout_sec": BASH_TIMEOUT_SEC,
        "max_output_chars": MAX_OUTPUT_CHARS,
        "submit_path": "/workdir/answer.txt",
        "reset_message": HARNESS_MESSAGE,
        "background_start": background_start,
    }
    settings.update(overrides)
    return BashHarnessConfig(**settings)


# ---------------------------------------------------------------------------
# Sandbox lifetime and cleanup for training and evaluation runs

DAYTONA_LIFETIME_MINS = 30


def shorten_daytona_lifetimes(minutes: int = DAYTONA_LIFETIME_MINS) -> None:
    """Let Daytona stop idle sandboxes, and delete stopped ones, after ``minutes``.

    A run killed before it closes its sandboxes then leaves nothing running
    for long (BenchFlow's default is a day). Rollouts here last minutes, so
    half an hour of idleness means the run is gone. An explicit setting in the
    environment wins.
    """

    os.environ.setdefault("BENCHFLOW_DAYTONA_AUTO_STOP_MINS", str(minutes))
    os.environ.setdefault("BENCHFLOW_DAYTONA_AUTO_DELETE_MINS", str(minutes))


def run_owner(prefix: str) -> str:
    """A Daytona owner label unique to this run, so its cleanup touches nothing else."""

    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"


def sweep_daytona(owner: str) -> dict[str, int]:
    """Delete every Daytona sandbox labelled with ``owner``, whatever its state.

    Uses BenchFlow's reaper, which never touches another owner's sandboxes.
    Call it when a run ends or is interrupted, and after a hard kill.
    """

    from daytona import Daytona

    from benchflow.sandbox.daytona import reap_stale_sandboxes

    previous = os.environ.get("BENCHFLOW_DAYTONA_OWNER")
    os.environ["BENCHFLOW_DAYTONA_OWNER"] = owner
    try:
        return reap_stale_sandboxes(
            Daytona(),
            max_age_minutes=0,
            failed_max_age_minutes=0,
            dry_run=False,
            ignore_age=True,
        )
    finally:
        if previous is None:
            os.environ.pop("BENCHFLOW_DAYTONA_OWNER", None)
        else:
            os.environ["BENCHFLOW_DAYTONA_OWNER"] = previous
