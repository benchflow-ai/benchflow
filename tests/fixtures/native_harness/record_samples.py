#!/usr/bin/env python3
"""Record the pinned CLIs' headless JSON output for the native-harness tests.

Run this when a native harness pin moves (``_CLAUDE_CODE_PACKAGE``,
``_CODEX_CLI_PACKAGE``), on a machine with the pinned CLIs installed under
``--npm-prefix`` (``npm install -g --prefix DIR <pins>``). It starts the
deterministic fake model (``tests/integration/deterministic``) on the host,
launches each CLI exactly as the harness does (its command builder), and
writes four samples per CLI into ``<cli>-<version>/``: a first turn, a resumed
second turn, a failing tool call, and a turn cancelled with SIGINT while its
tool runs (40 s in, so Claude Code's tool heartbeat is in it). Paths and ids
that change per run are replaced with placeholders;
``tests/test_native_harness_parsers.py`` reads the result.

No model credentials are involved: the fake answers every request.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from benchflow.agents.registry import pinned_npm_package  # noqa: E402
from benchflow.native_harness.claude_code import claude_code_launch  # noqa: E402
from benchflow.native_harness.codex import codex_launch  # noqa: E402
from benchflow.native_harness.spec import NativeTurn  # noqa: E402
from tests.integration.deterministic import harness as det  # noqa: E402

_UUID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)


def _fake(scripts: dict) -> tuple[ThreadingHTTPServer, str]:
    fake = det._load_fake_llm_module()
    fake._Handler.scripts = scripts
    fake._Handler.log_path = None
    server = ThreadingHTTPServer(("127.0.0.1", 0), fake._Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _run(
    argv: list[str], prompt: str, env: dict, cwd: Path, cancel_after: float | None
):
    with tempfile.TemporaryFile("w+") as stdin:
        stdin.write(prompt)
        stdin.seek(0)
        proc = subprocess.Popen(
            argv,
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=cwd,
            text=True,
            start_new_session=True,
        )
        if cancel_after is not None:
            time.sleep(cancel_after)
            os.killpg(proc.pid, signal.SIGINT)
        out, err = proc.communicate(timeout=180)
    return out, err, proc.returncode


def _scrub(text: str, replacements: dict[str, str]) -> str:
    for old, new in replacements.items():
        text = text.replace(old, new)
    return _UUID.sub("00000000-0000-4000-8000-000000000000", text)


def record(cli: str, prefix: Path, out_root: Path) -> None:
    scripts = json.loads((HERE / "scripts.json").read_text())
    server, url = _fake(scripts)
    work = Path(tempfile.mkdtemp(prefix="bf-native-work-"))
    # Codex refuses helper aliases under a temporary directory; keep its home out.
    home = prefix / f"sample-home-{cli}"
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir(parents=True)
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(home), "LANG": "C.UTF-8"}
    if cli == "claude-code":
        executable = str(prefix / "bin" / "claude")
        version = pinned_npm_package("claude-code")[1]
        env.update(
            ANTHROPIC_BASE_URL=url,
            ANTHROPIC_AUTH_TOKEN="fake-deterministic-key",
            ANTHROPIC_MODEL="claude-haiku-4-5",
            CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
        )

        def argv(turn: NativeTurn) -> list[str]:
            return [executable, *claude_code_launch(turn).argv]

        first = NativeTurn(
            cwd=str(work), new_session_id="11111111-2222-4333-8444-555555555555"
        )
        resume = NativeTurn(cwd=str(work), resume_id=first.new_session_id)
    else:
        executable = str(prefix / "bin" / "codex")
        version = pinned_npm_package("codex")[1]
        env["OPENAI_API_KEY"] = "sk-benchflow-fake"
        config = {
            "model_provider": "benchflow-litellm",
            "model": "gpt-5.4",
            "model_providers": {
                "benchflow-litellm": {
                    "name": "litellm",
                    "base_url": f"{url}/v1",
                    "env_key": "OPENAI_API_KEY",
                    "wire_api": "responses",
                    "supports_websockets": False,
                }
            },
        }

        def argv(turn: NativeTurn) -> list[str]:
            return [executable, *codex_launch(turn, config).argv]

        first = NativeTurn(cwd=str(work))
        resume = None
    out_dir = out_root / f"{cli}-{version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    replacements = {
        str(work): "/work",
        str(home): "/home/agent",
        url: "http://fake-llm",
    }
    samples = {}
    text, err, rc = _run(
        argv(first), "Create hello.txt. [[fake-llm:hello]]", env, work, None
    )
    samples["turn"] = (text, err, rc)
    if resume is None:
        thread = json.loads(text.splitlines()[0])["thread_id"]
        resume = NativeTurn(cwd=str(work), resume_id=thread)
    samples["resumed"] = _run(
        argv(resume), "Append a line. [[fake-llm:second]]", env, work, None
    )
    samples["tool-error"] = _run(
        argv(NativeTurn(cwd=str(work))), "Fail. [[fake-llm:fail]]", env, work, None
    )
    # Cancelled 40 s in: past the first of Claude Code's 30 s heartbeats for
    # a running tool (tool_progress), which a long tool call produces.
    samples["cancelled"] = _run(
        argv(NativeTurn(cwd=str(work))), "Sleep. [[fake-llm:sleep]]", env, work, 40.0
    )
    for name, (text, err, rc) in samples.items():
        (out_dir / f"{name}.jsonl").write_text(_scrub(text, replacements))
        print(
            f"{cli} {name}: exit {rc}, {len(text.splitlines())} lines"
            + (f", stderr: {err.strip()[:200]}" if err.strip() else "")
        )
    server.shutdown()
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(home, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--npm-prefix", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--cli", choices=("claude-code", "codex"), action="append")
    args = parser.parse_args()
    for cli in args.cli or ("claude-code", "codex"):
        record(cli, args.npm_prefix.resolve(), args.out)


if __name__ == "__main__":
    main()
