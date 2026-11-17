"""The agent's user is unchanged by its record, on real Docker.

Deterministic tier (``tests/integration/README.md``): real ``bench run`` with
``claude-agent-acp`` and the scripted fake model, no credentials. The agent's
Bash tool writes ``/app/hello.txt`` and its own ``id -un`` and ``id -u`` (the
workspace is root's again by grading time, so the file's owner cannot tell);
root's test.sh keeps them. Each case runs the agent as it ran before
that change, and its result.json's ``agent_user`` names that user beside the
task's own harness user, with the line in the run's output only where they
differ (sandbox/agent_user.py).
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from tests.integration.deterministic import harness as h

pytestmark = pytest.mark.deterministic

SANDBOX, SKIP_REASON = h.select_sandbox()
REQUESTED = os.environ.get(h.SANDBOX_ENV, "").strip().lower() == "docker"

ID_PROBE = "agent-user-id"
ID_PROBE_SCRIPT = [
    {
        "text": "Writing hello.txt and who I am.",
        "tool": "Bash",
        "input": {
            "command": "printf 'Hello, world!\\n' > /app/hello.txt; "
            "{ id -un; id -u; } > /app/whoami.txt",
            "description": "Write hello.txt",
        },
    },
    {"text": "Created /app/hello.txt."},
]
OWNER_TEST_SH = """#!/bin/bash
cat /app/whoami.txt > /logs/verifier/owner.txt 2>&1 || echo missing > /logs/verifier/owner.txt
REWARD=0
if [ -f /app/hello.txt ] && [ "$(tr -d '\\n' < /app/hello.txt)" = "Hello, world!" ]; then
    REWARD=1
fi
echo "$REWARD" > /logs/verifier/reward.txt
"""
# 0.10's claude-agent-acp refuses a root sandbox user (its connector
# restrictions), so no flag matches a root task for it there.
CLAUDE_REFUSES_ROOT = (
    importlib.util.find_spec("benchflow.agents.claude_connector_policy") is not None
)
ROOT_LINE = (
    "This task's own harness runs the agent as root; BenchFlow runs it as `agent`. "
    + (
        "No --sandbox-user matches it: claude-agent-acp's connector restrictions "
        "need a non-root sandbox user."
        if CLAUDE_REFUSES_ROOT
        else "To match it, pass `--sandbox-user null`."
    )
)
# codex-acp as root: its Codex Apps policy inherit (the default refuses root)
# and the full-access mode BenchFlow gives it as a non-root user (agents/
# codex_config.py), as its own bwrap sandbox cannot start in a container.
CODEX_ROOT = [
    "--sandbox-user",
    "null",
    "--codex-apps-policy",
    "inherit",
    "--agent-env",
    "INITIAL_AGENT_MODE=agent-full-access",
]
# The script agent's whole turn: the probe's command, no model.
PROBE_SH = (
    "#!/bin/bash\nprintf 'Hello, world!\\n' > /app/hello.txt\n"
    "{ id -un; id -u; } > /app/whoami.txt\n"
)


AUTO = object()


def _rec(
    task: str,
    source: str,
    ran: str,
    same: bool,
    flag: str | None,
    workdir: str | None = "/app",
    tree: object = AUTO,
) -> dict:
    """The record; the template image works in /app, which BenchFlow hands
    to the agent user (root writes every file)."""
    return {
        "task_harness": task,
        "task_harness_source": source,
        "ran_as": ran,
        "image_workdir": workdir,
        "working_tree": ("root" if ran == "root" else "handed")
        if tree is AUTO
        else tree,
        "same": same,
        "matching_flag": flag,
    }


# name: (image USER line, [agent].user, extra flags, route, the user the
# agent's tool ran as, the record)
CASES: dict[str, tuple[str | None, str | None, list[str], str, str, dict]] = {
    "root-image": (
        None,
        None,
        [],
        "proxy",
        "agent",
        _rec(
            "root",
            "image USER",
            "agent",
            False,
            None if CLAUDE_REFUSES_ROOT else "--sandbox-user null",
        ),
    ),
    # claude-agent-acp's install runs as the image's USER and cannot make
    # /opt/runtime as nobody (as before that change): the script agent
    # installs nothing.
    "nobody-image": (
        "USER nobody",
        None,
        [],
        "script",
        "agent",
        _rec("nobody", "image USER", "agent", False, "--sandbox-user nobody"),
    ),
    "declared-agent": (
        None,
        "agent",
        [],
        "proxy",
        "agent",
        _rec("agent", "task.toml [agent].user", "agent", True, "--sandbox-user agent"),
    ),
    # codex-acp runs as root once its Codex Apps policy is inherit.
    "null-on-root": (
        None,
        None,
        CODEX_ROOT,
        "codex",
        "root",
        _rec("root", "image USER", "root", True, "--sandbox-user null"),
    ),
}
if not CLAUDE_REFUSES_ROOT:
    CASES["claude-null-on-root"] = (
        None,
        None,
        ["--sandbox-user", "null"],
        "proxy",
        "root",
        _rec("root", "image USER", "root", True, "--sandbox-user null"),
    )


@pytest.fixture(scope="session")
def au_root(tmp_path_factory) -> Path:
    kept = os.environ.get(h.KEEP_JOBS_ENV)
    if kept:
        root = Path(kept) / "agent-user"
        root.mkdir(parents=True, exist_ok=True)
        return root
    return tmp_path_factory.mktemp("agent-user")


@contextmanager
def _host_fake(log_path: Path) -> Iterator[str]:
    """The tier's host fake provider with one script more (ID_PROBE); the
    shared scripts.json is unchanged."""
    fake = h._load_fake_llm_module()
    scripts = json.loads((h.FAKE_LLM_DIR / "scripts.json").read_text())
    scripts[ID_PROBE] = ID_PROBE_SCRIPT
    fake._Handler.scripts = scripts
    fake._Handler.log_path = log_path
    server = ThreadingHTTPServer(("127.0.0.1", 0), fake._Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture(scope="session")
def au_fake_url(au_root) -> Any:
    if SANDBOX != "docker":
        yield None
        return
    with _host_fake(au_root / "host-fake-llm.jsonl") as url:
        yield url


_runs: dict[str, h.CliRun] = {}


def _case(au_root: Path, fake_url: str | None, name: str) -> h.CliRun:
    if SANDBOX != "docker":
        (pytest.fail if REQUESTED else pytest.skip)(SKIP_REASON or "Docker only")
    if name in _runs:
        return _runs[name]
    user_line, declared, flags, route, _, _ = CASES[name]
    tasks = au_root / name / "tasks"
    tasks.mkdir(parents=True, exist_ok=True)
    task = h.materialize_task(h.TaskVariant(name, ID_PROBE), tasks)
    if user_line:
        dockerfile = task / "environment" / "Dockerfile"
        dockerfile.write_text(dockerfile.read_text() + f"\n{user_line}\n")
        # The verifier would run as the image's USER too, which cannot read
        # its own launch record (as before that change): root's test.sh
        # reads the probe.
        doc = task / "task.md"
        text, n = re.subn(
            r"(?m)^(verifier:\n  type: test-script)$",
            r"\g<1>\n  user: root",
            doc.read_text(),
        )
        assert n == 1
        doc.write_text(text)
    if declared:
        doc = task / "task.md"
        text, n = re.subn(
            r"(?m)^(agent:\n  timeout_sec: [0-9.]+)$",
            rf"\g<1>\n  user: {declared}",
            doc.read_text(),
        )
        assert n == 1
        doc.write_text(text)
    test_sh = task / "tests" / "test.sh"
    test_sh.write_text(OWNER_TEST_SH)
    test_sh.chmod(0o755)
    jobs = au_root / name / "jobs"
    if route == "script":
        _runs[name] = _script_run(au_root / name, task, jobs, flags)
    else:
        _runs[name] = h.run_bench(
            ["eval", "run"],
            tasks_dir=task,
            jobs_dir=jobs,
            sandbox=SANDBOX,
            host_fake_url=fake_url,
            route=route,
            extra=["--retry-attempts", "0", *flags],
        )
    return _runs[name]


def _script_run(root: Path, task: Path, jobs: Path, flags: list[str]) -> h.CliRun:
    """``bench eval run --agent script:PROBE_SH``: no model, no provider."""
    probe = root / "probe.sh"
    probe.write_text(PROBE_SH)
    probe.chmod(0o755)
    args = [
        h.bench_executable(),
        "eval",
        "run",
        "--tasks-dir",
        str(task),
        "--agent",
        f"script:{probe}",
        "--sandbox",
        "docker",
        "--jobs-dir",
        str(jobs),
        "--retry-attempts",
        "0",
        *flags,
    ]
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("ANTHROPIC_", "CLAUDE_CODE_", "OPENAI_", "CODEX_"))
        and k not in {"BENCHFLOW_PROVIDER_BASE_URL", "BENCHFLOW_PROVIDER_API_KEY"}
    }
    env["BENCHFLOW_SKIP_UPDATE_CHECK"] = "1"
    jobs.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        args,
        cwd=h.REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=1500,
    )
    (jobs / "cli-output.txt").write_text(proc.stdout)
    # Not timed: nothing here reads how long it took.
    return h.CliRun(args, proc.returncode, proc.stdout, 0.0, jobs)


def _trial(run: h.CliRun) -> Path:
    (trial,) = run.trial_dirs()
    return trial


@pytest.mark.parametrize("name", list(CASES))
def test_the_agent_runs_as_the_same_user(au_root, au_fake_url, name):
    """Who the agent's own tool ran as: ``agent`` by default and under a
    task's own ``[agent].user = "agent"``, root under ``--sandbox-user null``
    (codex-acp with its Codex Apps policy inherit; 0.9.x's claude-agent-acp
    too); the image's USER changes nothing."""
    run = _case(au_root, au_fake_url, name)
    trial = _trial(run)
    owner = (trial / "verifier" / "owner.txt").read_text().split()
    assert owner and owner[0] == CASES[name][4], (owner, run.output[-3000:])
    if owner[0] == "root":
        assert owner[1] == "0"
    else:
        assert owner[1] != "0"
    assert h.read_json(trial / "result.json")["rewards"] == {"reward": 1.0}


@pytest.mark.parametrize("name", list(CASES))
def test_the_record_names_both_users(au_root, au_fake_url, name):
    run = _case(au_root, au_fake_url, name)
    result = h.read_json(_trial(run) / "result.json")
    assert result.get("agent_user") == CASES[name][5], result.get("agent_user")
    said = "own harness runs the agent as" in run.output
    assert said is (CASES[name][5]["same"] is False), run.output[-3000:]
    if name == "root-image":
        assert ROOT_LINE in run.output


SLASH_TOML = """version = "1.0"
[agent]
timeout_sec = 600
[verifier]
timeout_sec = 120
[environment]
cpus = 1
memory_mb = 1024
"""
SLASH_TEST_SH = """#!/bin/bash
cat /testbed/f > /logs/verifier/owner.txt 2>&1
if grep -qx agent /testbed/f; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
"""
SLASH_PROBE_SH = "#!/bin/bash\necho agent >> /testbed/f\n"


def test_an_image_that_works_in_slash_hands_the_agent_no_tree(au_root):
    """FeatureBench's shape: the image works in /, its repository (/testbed
    here) is root's, and the task names no [environment].workdir. BenchFlow
    starts the agent user in /root and hands it nothing of the task's, so its
    write to /testbed fails and the trial scores 0; the run says so outright
    before the agent starts, and the record says "not handed"."""
    if SANDBOX != "docker":
        (pytest.fail if REQUESTED else pytest.skip)(SKIP_REASON or "Docker only")
    root = au_root / "slash"
    task = root / "tasks" / "featurebench-like"
    for sub in ("environment", "tests"):
        (task / sub).mkdir(parents=True, exist_ok=True)
    (task / "task.toml").write_text(SLASH_TOML)
    (task / "instruction.md").write_text("Append agent to /testbed/f.\n")
    (task / "environment" / "Dockerfile").write_text(
        "FROM python:3.12-slim\n"
        "RUN mkdir -p /testbed /logs/verifier /logs/agent && echo x > /testbed/f\n"
        "WORKDIR /\n"
    )
    (task / "tests" / "test.sh").write_text(SLASH_TEST_SH)
    (task / "tests" / "test.sh").chmod(0o755)
    probe = root / "probe.sh"
    probe.write_text(SLASH_PROBE_SH)
    probe.chmod(0o755)
    jobs = root / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    args = [
        h.bench_executable(),
        "eval",
        "run",
        "--tasks-dir",
        str(task),
        "--agent",
        f"script:{probe}",
        "--sandbox",
        "docker",
        "--jobs-dir",
        str(jobs),
        "--retry-attempts",
        "0",
    ]
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("ANTHROPIC_", "CLAUDE_CODE_", "OPENAI_", "CODEX_"))
    }
    env["BENCHFLOW_SKIP_UPDATE_CHECK"] = "1"
    proc = subprocess.run(
        args,
        cwd=h.REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=1500,
    )
    (jobs / "cli-output.txt").write_text(proc.stdout)
    trial = _trial(h.CliRun(args, proc.returncode, proc.stdout, 0.0, jobs))
    result = h.read_json(trial / "result.json")
    # The agent's write did not land: the tree was never its to write.
    assert (trial / "verifier" / "owner.txt").read_text().split() == ["x"]
    assert result["rewards"] == {"reward": 0.0}, result.get("error")
    record = result["agent_user"]
    assert record["image_workdir"] == "/"
    assert record["working_tree"] == "not handed"
    assert record["ran_as"] == "agent"
    said = [ln for ln in proc.stdout.splitlines() if "no working tree" in ln]
    assert len(said) == 1, proc.stdout[-3000:]
    assert "featurebench-like: BenchFlow hands `agent` no working tree" in said[0]
    assert "/testbed" in said[0]
