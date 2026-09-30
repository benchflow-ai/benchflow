"""``python -m benchflow.integrations.miles serve|prepare``.

serve    run the environment server the Miles agent function calls
prepare  write Miles prompt data (JSONL) for a folder of BenchFlow tasks
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from benchflow.integrations.miles.data import dataset_rows, write_dataset
from benchflow.integrations.miles.episode import (
    BASH_TIMEOUT_SEC,
    MAX_OUTPUT_CHARS,
    MAX_TURNS,
    SUBMIT_PATH,
    EpisodeSettings,
)
from benchflow.integrations.trl.spec import BashHarnessConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchflow.integrations.miles", description=__doc__
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the environment server")
    serve.add_argument("--tasks-dir", type=Path, required=True)
    serve.add_argument("--sandbox", choices=["daytona", "docker"], default="daytona")
    serve.add_argument("--jobs-dir", type=Path, default=Path("jobs/miles"))
    serve.add_argument("--job-name", help="one job folder for every episode")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=12100)
    serve.add_argument("--max-sandboxes", type=int, default=16)
    serve.add_argument("--max-turns", type=int, default=MAX_TURNS)
    serve.add_argument("--bash-timeout", type=int, default=BASH_TIMEOUT_SEC)
    serve.add_argument("--max-output-chars", type=int, default=MAX_OUTPUT_CHARS)
    serve.add_argument("--sandbox-user", default="agent")
    serve.add_argument("--submit-path", default=SUBMIT_PATH)
    serve.add_argument("--model", default="default")
    serve.add_argument(
        "--extra-body",
        default="{}",
        help='JSON merged into every chat request, e.g. \'{"chat_template_kwargs": '
        '{"enable_thinking": false}}\'',
    )
    serve.add_argument("--episode-timeout", type=float, default=1800.0)
    serve.add_argument("--request-timeout", type=float, default=300.0)
    serve.add_argument("--retries", type=int, default=2)
    serve.add_argument("--integrity", choices=["off", "audit", "strict"], default="off")
    serve.add_argument(
        "--token-file",
        type=Path,
        help="bearer token clients must send; required to bind a non-loopback host",
    )
    serve.add_argument("--owner", help="Daytona owner label (BENCHFLOW_DAYTONA_OWNER)")

    prepare = sub.add_parser("prepare", help="write Miles prompt data")
    prepare.add_argument("--tasks-dir", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--split", help="recorded in each row's metadata")
    prepare.add_argument(
        "--include", action="append", default=[], help="task id to include"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if args.command == "prepare":
        rows = dataset_rows(
            args.tasks_dir, split=args.split, include_tasks=tuple(args.include)
        )
        out = write_dataset(rows, args.out)
        print(f"wrote {len(rows)} rows to {out}")
        return 0

    if args.owner:
        os.environ["BENCHFLOW_DAYTONA_OWNER"] = args.owner
    token = None
    if args.token_file is not None:
        token = args.token_file.read_text().strip()
        if not token:
            print(f"error: {args.token_file} is empty", file=sys.stderr)
            return 2
    from benchflow.integrations.miles.server import serve

    settings = EpisodeSettings(
        tasks_dir=args.tasks_dir,
        harness=BashHarnessConfig(
            environment=args.sandbox,
            sandbox_user=args.sandbox_user,
            jobs_dir=args.jobs_dir,
            bash_timeout_sec=args.bash_timeout,
            max_output_chars=args.max_output_chars,
            submit_path=args.submit_path,
        ),
        max_turns=args.max_turns,
        model=args.model,
        extra_body=json.loads(args.extra_body),
        job_name=args.job_name,
        request_timeout_sec=args.request_timeout,
        transport_retries=args.retries,
        episode_timeout_sec=args.episode_timeout,
        integrity=args.integrity,
    )
    serve(
        settings,
        host=args.host,
        port=args.port,
        max_sandboxes=args.max_sandboxes,
        token=token,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
