"""The no-sandbox acceptance shim refuses what ``bench eval run`` refuses.

Round PY8, target DT1 (tools/dt/accept_py.py). It loads a task through the
launch loader (task/formats.py ``materialize_task_dir`` -> taskmd
``load_plan``) and the launch gate's runtime check
(``validate_task_runtime_support``), and starts nothing.
``net-blocklist`` (integ/p1:examples/tasks/net-blocklist) is refused at
load, unconditionally: ``[agent] network = { block = [...] }`` falls to the
refusal at taskmd/plan.py:931-936 and raises TaskMdError at
taskmd/materialize.py:133-138.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.dt import accept_py

ORACLE = "#!/bin/bash\nprintf 'Hello, world!' > /app/hello.txt\n"
VERIFIER = (
    "#!/bin/bash\n"
    "if printf 'Hello, world!' | cmp -s - /app/hello.txt; then echo 1; else echo 0; fi"
    " > /logs/verifier/reward.txt\n"
)


def _taskmd(root: Path, name: str, toml: str) -> Path:
    task = root / name
    (task / "sandbox").mkdir(parents=True)
    (task / "sandbox" / "Dockerfile").write_text("FROM ubuntu:24.04\nWORKDIR /app\n")
    (task / "oracle").mkdir()
    (task / "oracle" / "solve.sh").write_text(ORACLE)
    (task / "verifier").mkdir()
    (task / "verifier" / "test.sh").write_text(VERIFIER)
    (task / "task.md").write_text(
        "Create a file `/app/hello.txt` that contains exactly `Hello, world!`.\n\n"
        f"```toml task\n{toml}```\n"
    )
    return task


HELLO = """\
name = "examples/hello"
title = "Hello, world (task.md)"
version = "1.0.0"

[agent]
timeout = "2m"

[verifier]
timeout = "1m"
"""

# integ/p1:examples/tasks/net-blocklist/task.md's task block.
NET_BLOCKLIST = """\
name = "examples/net-blocklist"
title = "Network policy: net-blocklist"
version = "1.0.0"

[agent]
timeout = "2m"
network = { block = ["blocked.bench.invalid", "127.0.0.2/32"] }

[sandbox]
network = "open"

[verifier]
timeout = "1m"
network = "none"
"""


@pytest.fixture(autouse=True)
def _format_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_TASK_FORMAT_CACHE", str(tmp_path / "format-cache"))


def test_net_blocklist_is_refused_at_load(tmp_path):
    row = accept_py.accept(_taskmd(tmp_path, "net-blocklist", NET_BLOCKLIST))

    assert row["verdict"] == "refused_at_load"
    assert any(reason.startswith("[agent] network") for reason in row["reasons"])
    assert "cannot honor" in row["message"]


def test_a_plain_taskmd_package_is_accepted(tmp_path):
    assert accept_py.accept(_taskmd(tmp_path, "hello", HELLO))["verdict"] == "accepted"


def test_a_runtime_refusal_lists_its_features_with_code_r401(tmp_path):
    task = tmp_path / "collect-hook"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task / "tests").mkdir()
    (task / "tests" / "test.sh").write_text(
        "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"
    )
    (task / "instruction.md").write_text("Do nothing.\n")
    (task / "task.toml").write_text(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60.0\n\n'
        '[[steps]]\nname = "one"\n\n'
        '[[steps.verifier.collect]]\ncommand = "true"\ntimeout_sec = 30.0\n'
    )

    row = accept_py.accept(task)

    assert row["verdict"] == "refused"
    assert ("steps[0].verifier.collect", "R401") in [
        (f["path"], f["code"]) for f in row["features"]
    ]
