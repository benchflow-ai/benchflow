"""Check the evaluator end to end with a policy whose score is known.

    python docs/examples/rl/common/positive_control.py \\
        --tasks-dir tasks/v2/hard/test --sandbox daytona --limit 8

Before trusting evaluate.py's numbers for a model, run it against a scripted
policy through the same path: a local OpenAI-compatible endpoint, the
run_bash/submit harness, BenchFlow sandboxes, and the task verifiers. The
``oracle`` policy submits each task's expected answers (for a bug fix, it runs
the task's reference fix with run_bash first) and must score exactly 1.0; the
``nothing`` policy submits an empty answer and must score exactly 0.0. Any
other result means the evaluation path loses or invents credit (answers cut
off, submissions dropped, verifier failures), and model scores measured
through it cannot be trusted yet.

The endpoint reads answers from the task folders on this machine; nothing
secret is involved, and it listens only on 127.0.0.1.
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import secrets
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate

from benchflow.integrations.trl import BenchFlowSpec

# The task's files, hashed the same way here and in the sandbox. SQLite files are
# hashed as a text dump: their header records the library version that wrote them.
FINGERPRINT = (
    'cd /workdir && for f in $(ls -A | sort); do case "$f" in '
    "*.db) python3 -c 'import sqlite3, sys; "
    'print(chr(10).join(sqlite3.connect(sys.argv[1]).iterdump()))\' "$f" ;; '
    '*) [ -f "$f" ] && cat "$f" ;; esac; done | sha256sum | cut -c1-32'
)


def _task_table(
    tasks_dir: Path, rows: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Each task's file fingerprint, mapped to its expected answers and reference fix.

    Prompts are not unique (two bug-fix tasks can share a module name), so the
    scripted policy first fingerprints the files it finds and looks them up.
    The files are rebuilt here with the task's own generator and setup command.
    """

    table = {}
    for row in rows:
        task = Path(row["benchflow_task_dir"])
        front = (task / "task.md").read_text().split("---")[1]
        sandbox = next(
            json.loads(line.split(":", 1)[1])
            for line in front.splitlines()
            if line.startswith("sandbox:")
        )
        setup = sandbox["setup_commands"][0]["command"].split(" && ")[0].split()
        with tempfile.TemporaryDirectory() as work:
            subprocess.run(
                [
                    sys.executable,
                    str(task / "environment" / "family.py"),
                    *setup[2:-1],
                    work,
                ],
                check=True,
            )
            fingerprint = subprocess.run(
                ["bash", "-c", FINGERPRINT.replace("/workdir", work)],
                check=True, capture_output=True, text=True,
            ).stdout.strip()  # fmt: skip
        table[fingerprint] = {
            "expected": json.loads((task / "verifier" / "expected.json").read_text()),
            "fix": (task / "oracle" / "solve.sh").read_text(),
        }
    return table


def _reply(table: dict[str, dict[str, Any]], mode: str, messages: list[dict]) -> dict:
    tool_results = [
        str(m.get("content", "")) for m in messages if m.get("role") == "tool"
    ]
    if mode == "nothing":
        call = ("submit", {"answer": ""})
    elif not tool_results:
        call = ("run_bash", {"command": FINGERPRINT})
    else:
        task = table.get(tool_results[0].strip())
        if task is None:
            call = ("submit", {"answer": "unknown task"})
        elif task["expected"]["type"] != "bugfix":
            answers = "\n".join(a["answer"] for a in task["expected"]["answers"])
            call = ("submit", {"answer": answers})
        elif len(tool_results) == 1:
            call = ("run_bash", {"command": task["fix"]})
        else:
            call = ("submit", {"answer": "done"})
    name, arguments = call
    return {
        "id": "chatcmpl-control",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call-{secrets.token_hex(4)}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }


def serve(
    table: dict[str, dict[str, Any]], mode: str, key: str
) -> http.server.HTTPServer:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.headers.get("Authorization") != f"Bearer {key}":
                self.send_error(401)
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            data = json.dumps(_reply(table, mode, body["messages"])).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: Any) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--sandbox", choices=["daytona", "docker"], default="daytona")
    parser.add_argument("--out", type=Path, default=Path("positive-control"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args(argv)

    rows = list(BenchFlowSpec(tasks_dir=args.tasks_dir).train_dataset_rows)
    if args.limit:
        rows = rows[: args.limit]
    table = _task_table(args.tasks_dir, rows)
    if len(table) != len(rows):
        print(
            "error: two tasks have the same files; they cannot be told apart",
            file=sys.stderr,
        )
        return 2
    key = secrets.token_hex(16)
    os.environ["POSITIVE_CONTROL_KEY"] = key
    failures = []
    for mode, want in (("oracle", 1.0), ("nothing", 0.0)):
        server = serve(table, mode, key)
        out = args.out / mode
        command = [
            "--tasks-dir", str(args.tasks_dir), "--sandbox", args.sandbox,
            "--base-url", f"http://127.0.0.1:{server.server_address[1]}/v1",
            "--model", f"control-{mode}", "--api-key-env", "POSITIVE_CONTROL_KEY",
            "--concurrency", str(args.concurrency), "--out", str(out),
        ]  # fmt: skip
        if args.limit:
            command += ["--limit", str(args.limit)]
        try:
            status = evaluate.main(command)
        finally:
            server.shutdown()
        if status != 0:
            print(f"{mode}: evaluate.py exited with {status}")
            return status
        summary = json.loads((out / "summary.json").read_text())
        got = summary["mean_reward"]
        verdict = "ok" if got == want and not summary["dropped"] else "FAILED"
        print(
            f"{mode}: mean reward {got} (want {want}), pass rate {summary['solve_rate']}, "
            f"dropped {summary['drop_reasons']}: {verdict}"
        )
        if verdict != "ok":
            failures.append(mode)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
