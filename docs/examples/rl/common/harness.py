"""The run_bash/submit harness settings every RL cookbook shares.

Training (TRL on HF Jobs, and the other cookbooks) and held-out evaluation
(``evaluate.py``) build their harness from :func:`harness_config`, so a policy
is evaluated with the same tools, instructions, and limits it trained with.
"""

from __future__ import annotations

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
