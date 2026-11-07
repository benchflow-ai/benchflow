"""`python -m benchflow.embodied.serve`: run one episode server for a task.

The benchmark provides a factory ``module:function`` with the signature

    factory(task_dir: Path, *, run_dir: Path, workspace: Path | None, seed: int | None,
            max_wall_s: float | None) -> EpisodeServer

that loads the task, builds its simulator backend and returns the (reset) episode server. The simulator sidecar's
entry point runs this module with ``--factory $ROBO_EPISODE_FACTORY``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


def load_factory(spec: str):
    mod, _, fn = spec.partition(":")
    if not mod or not fn:
        raise SystemExit(f"--factory must be module:function, got {spec!r}")
    return getattr(importlib.import_module(mod), fn)


def main(argv: list[str] | None = None) -> int:
    from .server import serve

    ap = argparse.ArgumentParser(prog="python -m benchflow.embodied.serve")
    ap.add_argument(
        "--factory", required=True, help="module:function that builds the EpisodeServer"
    )
    ap.add_argument("--task", required=True, help="task folder (task.md)")
    ap.add_argument(
        "--run-dir", required=True, help="where the episode record is written"
    )
    ap.add_argument("--socket", required=True, help="Unix socket path to listen on")
    ap.add_argument("--workspace", help="agent-visible folder for camera images")
    ap.add_argument("--seed", type=int, help="override the task's seed")
    ap.add_argument("--max-wall-s", type=float, help="wall-clock budget in seconds")
    ap.add_argument("--ready-file", help="file written once the server listens")
    a = ap.parse_args(argv)
    factory = load_factory(a.factory)
    server = factory(
        Path(a.task),
        run_dir=Path(a.run_dir),
        workspace=Path(a.workspace) if a.workspace else None,
        seed=a.seed,
        max_wall_s=a.max_wall_s,
    )
    result = serve(server, a.socket, ready_file=a.ready_file)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
