"""Run the first-run journeys' commands as a new user would.

The ``docs`` CI job builds the wheel from this checkout, serves it from a
local package index, and runs this file with ``BENCHFLOW_DOCS_E2E=1`` and
``BENCHFLOW_DOCS_INDEX_URL`` set. Each journey runs in bash, in an empty
HOME, with the commands read from its page as written, so a page and its
test cannot drift apart:

- README: install, ``bench doctor``, the oracle run, reading the results;
- ``docs/tasks/write-a-task.md``: scaffold, the files the page gives (its
  data download served from this checkout, which stands in for ``main``), check,
  then the oracle (1/1) and ``nop`` (0/1) runs;
- ``docs/train/export.md``: convert and validate, on a model trial written
  here, because CI has no model key.

Commands that need a model key, a subscription login or Daytona are checked
by ``test_docs_journeys.py`` instead, which parses them against the CLI.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from tests.docs_pages import REPO, Block, bench_argvs, blocks, read, shell_lines

pytestmark = pytest.mark.skipif(
    os.environ.get("BENCHFLOW_DOCS_E2E") != "1",
    reason="runs in the docs CI job (BENCHFLOW_DOCS_E2E=1 and a package index)",
)

TIMEOUT = 1800
# A page's downloads from main; the journeys read them from this checkout, as
# they install the wheel from a local index rather than PyPI.
RAW_MAIN = "https://raw.githubusercontent.com/benchflow-ai/benchflow/main/"


@pytest.fixture
def user(tmp_path: Path) -> dict[str, str]:
    """A new user's environment: an empty HOME, no keys, the built wheel."""
    home = tmp_path / "home"
    home.mkdir()
    env = {
        "HOME": str(home),
        "PATH": f"{home}/.local/bin:" + os.environ["PATH"],
        "LANG": "C.UTF-8",
        "TERM": "dumb",
        "NO_COLOR": "1",
        "UV_INDEX": os.environ["BENCHFLOW_DOCS_INDEX_URL"],
        "PIP_EXTRA_INDEX_URL": os.environ["BENCHFLOW_DOCS_INDEX_URL"],
    }
    for name in ("DOCKER_HOST", "UV_PYTHON_INSTALL_DIR"):
        if name in os.environ:
            env[name] = os.environ[name]
    return env


def run(script: str, env: dict[str, str], cwd: Path, *, ok: bool = True) -> str:
    proc = subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + script],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        check=False,
    )
    out = proc.stdout + proc.stderr
    if ok:
        assert proc.returncode == 0, (
            f"exit {proc.returncode} running:\n{script}\n---\n{out[-6000:]}"
        )
    return out


def block_with(page_blocks: list[Block], *needles: str) -> Block:
    for block in page_blocks:
        if all(n in block.text for n in needles):
            return block
    raise AssertionError(f"no code block contains {needles!r}")


def under(page_blocks: list[Block], heading: str) -> list[Block]:
    found = [b for b in page_blocks if b.heading == heading]
    assert found, f"no code block under the heading {heading!r}"
    return found


def install(env: dict[str, str], cwd: Path) -> str:
    """README step 1, as written: install and check this machine."""
    readme = blocks(read("README.md"))
    step = block_with(readme, "uv tool install", "bench doctor")
    out = run("\n".join(shell_lines(step)), env, cwd)
    version = run("bench --version", env, cwd)
    assert re.search(r"benchflow \d+\.\d+", version), version
    return out


def test_readme_first_score_with_no_model_and_no_key(user, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    install(user, work)
    readme = blocks(read("README.md"))
    oracle = block_with(readme, "bench run", "--agent oracle")
    out = run("\n".join(shell_lines(oracle)), user, work)
    assert "Score: 1/1" in out, out[-3000:]
    # Step 4: read the results. The README says to use jobs/oracle when the
    # model run was skipped; `bench view` serves a page, so it is not run here.
    results = block_with(readme, "bench results jobs/", "bench eval metrics")
    lines = [
        line.replace("jobs/claude", "jobs/oracle")
        for line in shell_lines(results)
        if not line.startswith("bench view")
    ]
    out = run("\n".join(lines), user, work)
    assert "1/1" in out, out


def test_write_a_task_tutorial_scores_one_with_the_oracle_and_zero_with_nop(
    user, tmp_path
):
    work = tmp_path / "work"
    work.mkdir()
    install(user, work)
    page = blocks(read("docs/tasks/write-a-task.md"))

    scaffold, cleanup = [b for b in under(page, "Scaffold it") if b.lang == "bash"]
    ((_, name),) = bench_argvs("\n".join(shell_lines(scaffold)))
    task = work / "tasks" / name
    run("\n".join(shell_lines(scaffold)), user, work)
    run("\n".join(shell_lines(cleanup)), user, work)
    check, controls = [
        b for b in under(page, "Check it and run the controls") if b.lang == "bash"
    ]
    # bench new's Next: line on the page: bench check lists the placeholders
    # left, so the unfilled task fails its check.
    out = run("\n".join(shell_lines(check)), user, work, ok=False)
    assert "unreplaced placeholder" in out, out

    environment = under(page, "Build the environment and add the data")
    (dockerfile,) = [b for b in environment if b.lang == "dockerfile"]
    (task / "environment" / "Dockerfile").write_text(dockerfile.text + "\n")
    (download,) = [b for b in environment if b.lang == "bash"]
    fetch = "\n".join(shell_lines(download))
    assert RAW_MAIN in fetch, fetch
    run(fetch.replace(RAW_MAIN, f"file://{REPO}/"), user, work)

    (task_md,) = under(page, "Write the instruction")
    assert task_md.text.startswith("---\n"), task_md.text[:200]
    (task / "task.md").write_text(task_md.text + "\n")
    (verifier,) = under(page, "Write the test")
    (task / "verifier" / "test.sh").write_text(verifier.text + "\n")
    (oracle,) = under(page, "Write the reference solution")
    (task / "oracle" / "solve.sh").write_text(oracle.text + "\n")

    out = run("\n".join(shell_lines(check)), user, work)
    assert "valid (structural)" in out, out

    oracle_line, nop_line = shell_lines(controls)
    out = run(oracle_line, user, work)
    assert "Score: 1/1" in out, out[-3000:]
    out = run(nop_line, user, work)
    assert "nop control: 0/1" in out, out[-3000:]


def _model_trial(job: Path) -> Path:
    """One finished model trial, as `bench run` writes it, with a tool call."""
    trial = job / "hello__0a1b2c3d"
    (trial / "trajectory").mkdir(parents=True)
    (trial / "verifier").mkdir()
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "hello",
                "trial_name": "hello__0a1b2c3d",
                "agent": "openhands",
                "agent_name": "openhands",
                "model": "deepseek/deepseek-flash",
                "rewards": {"reward": 1.0},
                "n_tool_calls": 1,
                "agent_result": {
                    "n_input_tokens": 120,
                    "n_output_tokens": 30,
                    "n_tool_calls": 1,
                },
            }
        )
    )
    (trial / "verifier" / "reward.txt").write_text("1\n")
    tool = {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
            },
        },
    }
    user_turn = {
        "role": "user",
        "content": "Create /app/hello.txt containing Hello, world!",
    }
    call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps(
                        {"command": "echo 'Hello, world!' > /app/hello.txt"}
                    ),
                },
            }
        ],
    }
    exchanges = [
        {
            "request": {
                "body": {
                    "model": "deepseek-flash",
                    "messages": [user_turn],
                    "tools": [tool],
                }
            },
            "response": {"status_code": 200, "body": {"choices": [{"message": call}]}},
        },
        {
            "request": {
                "body": {
                    "model": "deepseek-flash",
                    "messages": [
                        user_turn,
                        call,
                        {
                            "role": "tool",
                            "tool_call_id": "call_1",
                            "content": "(no output)",
                        },
                    ],
                    "tools": [tool],
                }
            },
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [{"message": {"role": "assistant", "content": "Done."}}]
                },
            },
        },
    ]
    (trial / "trajectory" / "llm_trajectory.jsonl").write_text(
        "\n".join(json.dumps(e) for e in exchanges) + "\n"
    )
    return trial


def test_export_page_writes_rows_with_reward_and_messages(user, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    install(user, work)
    page = blocks(read("docs/train/export.md"))
    (export,) = [b for b in under(page, "Export a job") if b.lang == "bash"]
    convert, validate = shell_lines(export)
    ((_, _, job, *flags),) = bench_argvs(convert)
    _model_trial(work / job)
    out = run(convert, user, work)
    assert "Converted 1 row(s) from 1 rollout(s)" in out, out
    out = run(validate, user, work)
    assert '"rows_with_tool_calls": 1' in out, out
    rows = [
        json.loads(line)
        for line in (work / flags[flags.index("--out") + 1]).read_text().splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["reward"] == 1.0
    roles = [m["role"] for m in rows[0]["messages"]]
    assert roles[:2] == ["user", "assistant"] and "tool" in roles, roles
