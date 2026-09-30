#!/usr/bin/env python3
"""A stand-in native CLI for tests: replays a recorded sample on stdout.

Reads the prompt from stdin like the real CLIs and writes it to
``REPLAY_PROMPT_OUT`` and its arguments to ``REPLAY_ARGV_OUT`` (when set),
prints ``REPLAY_SAMPLE`` without its last ``REPLAY_CUT`` lines (default 0),
then either sleeps (``REPLAY_HANG=1``; with ``REPLAY_CHILD=1`` it first
starts a ``sleep`` in its own session, as a detached tool process would be)
or exits with ``REPLAY_EXIT`` (default 0).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time


def main() -> None:
    prompt = sys.stdin.read()
    if os.environ.get("REPLAY_PROMPT_OUT"):
        with open(os.environ["REPLAY_PROMPT_OUT"], "w", encoding="utf-8") as f:
            f.write(prompt)
    if os.environ.get("REPLAY_ARGV_OUT"):
        with open(os.environ["REPLAY_ARGV_OUT"], "a", encoding="utf-8") as f:
            f.write(json.dumps(sys.argv[1:]) + "\n")
    with open(os.environ["REPLAY_SAMPLE"], encoding="utf-8") as f:
        lines = f.readlines()
    cut = int(os.environ.get("REPLAY_CUT", "0"))
    for line in lines[: len(lines) - cut]:
        sys.stdout.write(line)
        sys.stdout.flush()
    if os.environ.get("REPLAY_HANG") == "1":
        if os.environ.get("REPLAY_CHILD") == "1":
            subprocess.Popen(["sleep", "3600"], start_new_session=True)
        time.sleep(3600)
    sys.exit(int(os.environ.get("REPLAY_EXIT", "0")))


if __name__ == "__main__":
    main()
