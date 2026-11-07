"""Deterministic tier: no copy of the solution is in the sandbox while a
non-oracle agent runs, even one run as root (GAP-ORACLE-READABLE).

The lockdown makes ``/oracle`` and ``/solution`` root-only (chmod 700), which
keeps them from the ``agent`` user but not from an agent run as root
(``--sandbox-user null``). Before the fix the task's ``solution/`` was uploaded
at the start of every trial; a scripted-user run with ``oracle_access`` moved
it to ``/solution_oracle_backup`` (``/oracle`` to ``/oracle_backup``) for the
agent's rounds. Now only an oracle run uploads it.

The fixture task's ``solve.sh`` carries a marker. The scripted agent's one
Bash call (run as root) lists the four paths and greps the whole sandbox for
the marker, which it builds at run time so its own command never matches;
the verifier's ``test.sh`` does the same after the agent's turn. Three runs:
``claude-agent-acp`` as root, ``nop`` as root, and a scripted-user run
(``FunctionUser`` with ``oracle_access=True``, two rounds) as root.
"""

from __future__ import annotations

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

MARK = "BF_GAPS1_" + "SOLUTION_MARK"
PATHS = ("/oracle", "/solution", "/oracle_backup", "/solution_oracle_backup")

# The marker in two halves, so neither the command nor a log of it holds it.
_PROBE = (
    'm=BF_GAPS1_; m="${m}SOLUTION_MARK"; '
    "for p in " + " ".join(PATHS) + "; do "
    'if [ -e "$p" ]; then echo "solprobe present $p"; '
    'else echo "solprobe absent $p"; fi; done; '
    "timeout 120 grep -rlsI --exclude-dir=proc --exclude-dir=sys "
    '--exclude-dir=dev "$m" / 2>/dev/null | sed "s/^/solprobe copy /"; '
    'echo "solprobe uid $(id -u)"; echo "solprobe done"'
)
_PROBE_SCRIPT = [
    {
        "text": "Looking for the solution, then writing the answer.",
        "tool": "Bash",
        "input": {
            "command": _PROBE + "; printf 'Hello, world!\\n' > /app/hello.txt",
            "description": "look for the solution, then write the answer",
        },
    },
    {"text": "Done."},
]
_TEST_SH = """#!/bin/bash
({probe}) > /logs/verifier/solprobe.txt 2>&1
REWARD=0
if [ -f /app/hello.txt ] && [ "$(tr -d '\\n' < /app/hello.txt)" = "Hello, world!" ]; then
    REWARD=1
fi
echo "$REWARD" > /logs/verifier/reward.txt
"""


@pytest.fixture(scope="module")
def require_docker() -> None:
    if SANDBOX != "docker":
        pytest.skip(SKIP_REASON if SANDBOX is None else "Docker only")


def _task(dest_root: Path, name: str = "solution-hidden") -> Path:
    dest = h.materialize_task(h.TaskVariant(name, "solution-probe"), dest_root)
    solve = dest / "solution" / "solve.sh"
    solve.write_text(solve.read_text() + f"# {MARK}\n")
    test_sh = dest / "tests" / "test.sh"
    test_sh.write_text(_TEST_SH.replace("{probe}", _PROBE))
    test_sh.chmod(0o755)
    return dest


@contextmanager
def _host_fake(log_path: Path) -> Iterator[str]:
    """The host fake provider, serving the fixture scripts plus the probe."""
    fake = h._load_fake_llm_module()
    scripts = json.loads((h.FAKE_LLM_DIR / "scripts.json").read_text())
    scripts["solution-probe"] = _PROBE_SCRIPT
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


def _parse(text: str) -> dict[str, Any]:
    lines = re.findall(r"(?m)^solprobe (.*)$", text)
    return {
        "present": [ln[8:] for ln in lines if ln.startswith("present ")],
        "absent": [ln[7:] for ln in lines if ln.startswith("absent ")],
        "copies": [ln[5:] for ln in lines if ln.startswith("copy ")],
        "uid": next((ln[4:] for ln in lines if ln.startswith("uid ")), ""),
        "done": "done" in lines,
    }


def _agent_probes(trial: Path) -> list[dict[str, Any]]:
    """Every probe the agent's Bash calls printed, in order."""
    return [
        _parse(str(event.get("raw_output") or ""))
        for event in h.normalized_acp(trial)
        if event.get("type") == "tool_call" and event.get("tool_name") == "Bash"
    ]


def _verifier_probe(trial: Path) -> dict[str, Any]:
    found = list((trial / "verifier").rglob("solprobe.txt"))
    assert found, sorted(p.name for p in (trial / "verifier").rglob("*"))
    return _parse(found[0].read_text(errors="replace"))


def _assert_hidden(probe: dict[str, Any], *, root: bool = True) -> None:
    assert probe["done"], probe
    if root:
        assert probe["uid"] == "0", probe
    assert probe["present"] == [], probe
    assert sorted(probe["absent"]) == sorted(PATHS), probe
    assert probe["copies"] == [], probe


@pytest.fixture(scope="module")
def cli_runs(tmp_path_factory, require_docker) -> dict[str, Path]:
    """``claude-agent-acp`` and ``nop``, each as root, on the probe task."""
    root = tmp_path_factory.mktemp("solution-hidden")
    tasks = root / "tasks"
    _task(tasks)
    trials: dict[str, Path] = {}
    with _host_fake(root / "host-fake-llm.jsonl") as url:
        run = h.run_bench(
            ["eval", "run"],
            tasks_dir=tasks,
            jobs_dir=root / "jobs-agent",
            sandbox="docker",
            host_fake_url=url,
            extra=["--sandbox-user", "null", "--retry-attempts", "0"],
        )
        trials["agent"] = run.trial("solution-hidden")
    # nop runs no model, so no provider route (as tests/test_integrity_deterministic.py
    # runs the oracle).
    args = [
        h.bench_executable(),
        "eval",
        "run",
        "--tasks-dir",
        str(tasks),
        "--agent",
        "nop",
        "--sandbox",
        "docker",
        "--sandbox-user",
        "null",
        "--retry-attempts",
        "0",
        "--jobs-dir",
        str(root / "jobs-nop"),
    ]
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("ANTHROPIC_", "CLAUDE_CODE_", "OPENAI_"))
    }
    env["BENCHFLOW_SKIP_UPDATE_CHECK"] = "1"
    (root / "jobs-nop").mkdir()
    proc = subprocess.run(
        args,
        cwd=h.REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=1500,
    )
    nop = h.CliRun(args, proc.returncode, proc.stdout, 0.0, root / "jobs-nop")
    trials["nop"] = nop.trial("solution-hidden")
    return trials


def test_an_agent_run_as_root_finds_no_solution(cli_runs):
    trial = cli_runs["agent"]
    result = h.read_json(trial / "result.json")
    probes = _agent_probes(trial)
    assert len(probes) == 1, (probes, result.get("error"))
    _assert_hidden(probes[0])
    assert result["rewards"] == {"reward": 1.0}, result.get("error")


def test_the_verifier_after_a_root_agent_finds_no_solution(cli_runs):
    _assert_hidden(_verifier_probe(cli_runs["agent"]))


def test_a_nop_run_as_root_has_no_solution(cli_runs):
    trial = cli_runs["nop"]
    result = h.read_json(trial / "result.json")
    assert result["rewards"] == {"reward": 0.0}, result.get("error")
    _assert_hidden(_verifier_probe(trial))


@pytest.mark.usefixtures("require_docker")
async def test_a_scripted_user_run_with_oracle_access_has_no_solution(tmp_path):
    """The user reads solve.sh on the host; the sandbox holds no copy under
    any name in either round, nor at the soft verify between them."""
    from benchflow.contracts import RoundResult
    from benchflow.contracts.user import BaseUser
    from benchflow.rollout import Rollout, RolloutConfig, Scene

    class TwoRounds(BaseUser):
        solution: str | None = None

        async def setup(self, instruction: str, solution: str | None = None) -> None:
            self.solution = solution

        async def run(
            self,
            round: int,
            instruction: str,
            round_result: RoundResult | None = None,
        ) -> str | None:
            return instruction if round < 2 else None

    task = _task(tmp_path / "tasks")
    user = TwoRounds()
    with _host_fake(tmp_path / "host-fake-llm.jsonl") as url:
        rollout = Rollout(
            RolloutConfig(
                task_path=task,
                scenes=[Scene.single(agent=h.AGENT, model=h.MODEL)],
                environment="docker",
                sandbox_user=None,
                user=user,
                max_user_rounds=2,
                oracle_access=True,
                agent_env=h.route_env("proxy", "docker", url),
                jobs_dir=tmp_path / "jobs",
                job_name="solution-hidden-user",
                rollout_name="solution-hidden__user",
            )
        )
        result = await rollout.run()
    assert user.solution is not None and MARK in user.solution
    trial = rollout._rollout_dir
    assert trial is not None
    probes = _agent_probes(trial)
    assert len(probes) == 2, (probes, result.error)
    for probe in probes:
        _assert_hidden(probe)
    _assert_hidden(_verifier_probe(trial))
    assert result.error is None, result.error
    assert result.rewards == {"reward": 1.0}
