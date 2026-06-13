"""Coverage for importing official encrypted Browser Use benchmark slices."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from cryptography.fernet import Fernet

from benchflow.adapters.browser_use import BrowserUseAdapter


def test_import_upstream_browser_use_task_writes_llm_judge_package(
    tmp_path: Path,
) -> None:
    encrypted = tmp_path / "BU_Bench_V1.enc"
    tasks = [
        {
            "task_id": "66c6641b-f949-46a2-8bcc-6d9dd388b534",
            "category": "WebBenchREAD",
            "confirmed_task": "Browse https://stackexchange.com and list the top communities.",
        }
    ]
    key = base64.urlsafe_b64encode(hashlib.sha256(b"BU_Bench_V1").digest())
    encrypted.write_text(
        base64.b64encode(Fernet(key).encrypt(json.dumps(tasks).encode())).decode()
    )
    out_dir = tmp_path / "tasks"
    script = (
        Path(__file__).parents[1]
        / "benchmarks"
        / "browser-use-smoke"
        / "import_upstream.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--encrypted-file",
            str(encrypted),
            "--out-dir",
            str(out_dir),
            "--task-indices",
            "0",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    task_dir = Path(payload["task_dirs"][0])
    descriptor = json.loads((task_dir / "browser-use-task.json").read_text())

    assert descriptor["benchmark"] == "BU_Bench_V1"
    assert descriptor["verifier"]["type"] == "llm-judge"
    assert sorted(path.name for path in (task_dir / "tests").iterdir()) == [
        "context.md",
        "rubric.toml",
        "verifier.md",
    ]

    inbound = BrowserUseAdapter.from_task_dir(task_dir)
    assert inbound.config.verifier.type == "llm-judge"
    assert inbound.config.verifier.judge.input_dir == "/logs/artifacts"
    assert inbound.files["tests/verifier.md"] == task_dir / "tests" / "verifier.md"
