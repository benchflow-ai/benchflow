#!/usr/bin/env python3
"""Run a task inside an Environment-plane manifest that starts a service.

An environment manifest (docs/environment-plane.md) declares the stateful
world a task runs in: images, services BenchFlow starts, readiness probes and
restorable state. Here the manifest starts a small HTTP service; the task asks
the agent to fetch a file from it, and the verifier checks the copy.

The script writes the task to a temporary directory, so it runs as is. The
oracle needs no credentials; pass --agent/--model to run a real agent.

Usage:
  uv run python docs/examples/python-sdk/run-with-manifest.py
  uv run python docs/examples/python-sdk/run-with-manifest.py --sandbox daytona
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
from pathlib import Path

import benchflow as bf

MANIFEST = """
[environment]
name = "notes-service"
base_image = "python:3.12-slim"   # the task's Dockerfile builds FROM this
owns_lifecycle = false            # BenchFlow starts the services below

[[environment.services]]
name = "notes"
command = "python3 -m http.server 8080 --directory /srv/notes"
port = 8080
health_path = "/note.txt"

[environment.readiness]
timeout_sec = 60
"""

TASK_FILES = {
    "task.toml": """version = "1.0"

[verifier]
timeout_sec = 60

[agent]
timeout_sec = 300

[environment]
allow_internet = true
""",
    "instruction.md": (
        "A notes service is running at http://localhost:8080. Download "
        "http://localhost:8080/note.txt and save it unchanged as /app/note.txt.\n"
    ),
    "environment/Dockerfile": """FROM python:3.12-slim
RUN apt-get update -qq && apt-get install -y -qq curl && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /srv/notes /logs/verifier /logs/agent /logs/artifacts \\
    && echo "served by the manifest" > /srv/notes/note.txt
WORKDIR /app
""",
    "solution/solve.sh": "#!/bin/bash\ncurl -sf http://localhost:8080/note.txt -o /app/note.txt\n",
    "tests/test.sh": """#!/bin/bash
if [ "$(cat /app/note.txt 2>/dev/null)" = "served by the manifest" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
""",
}


def write_task(root: Path) -> Path:
    task = root / "fetch-note"
    for rel, text in TASK_FILES.items():
        path = task / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        if rel.endswith(".sh"):
            path.chmod(0o755)
    return task


async def main(args: argparse.Namespace) -> int:
    # Or bf.load_manifest("environment.toml") for a manifest file.
    manifest = bf.EnvironmentManifest.model_validate_toml(MANIFEST)
    with tempfile.TemporaryDirectory() as tmp:
        result = await bf.run(
            bf.RolloutConfig(
                task_path=write_task(Path(tmp)),
                agent=args.agent,
                model=args.model,
                environment=args.sandbox,
                environment_manifest=manifest,
                jobs_dir=args.jobs_dir,
            )
        )
    print(result)
    print(
        f"reward {result.reward}  passed {result.passed}  artifacts {result.rollout_dir}"
    )
    if result.error:
        print(f"error [{result.error_category}]: {result.error}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sandbox", default="docker", choices=["docker", "daytona"])
    parser.add_argument("--agent", default="oracle")
    parser.add_argument("--model", default=None)
    parser.add_argument("--jobs-dir", default="jobs/python-sdk-examples")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
