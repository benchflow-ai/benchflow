from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_run_cell_archives_stale_jobs_before_collecting_result(tmp_path: Path) -> None:
    """Guards PR #641 against requeued cells reusing old rollout results."""
    repo = Path(__file__).resolve().parents[1]
    script = repo / "experiments" / "skillsbench-fill" / "run_cell.sh"
    bench_root = tmp_path / "bench"
    skills_root = tmp_path / "skillsbench"
    jobs_root = tmp_path / "jobs"
    state_dir = tmp_path / "state"
    cell = "gemini-3.5-flash__without__citation-check__t1"
    old_result = jobs_root / cell / "2000-01-01__00-00-00" / "citation-check__old" / "result.json"
    _write_json(
        old_result,
        {
            "agent_result": {
                "total_tokens": 999,
                "n_input_tokens": 999,
                "n_output_tokens": 0,
                "usage_source": "provider_response",
            },
            "rewards": {"reward": 0.0},
            "timing": {"total": 999.0},
            "error": "Agent timed out after 900s",
            "partial_trajectory": True,
        },
    )
    old_epoch = 946684800
    os.utime(old_result, (old_epoch, old_epoch))

    bench = bench_root / ".venv" / "bin" / "bench"
    bench.parent.mkdir(parents=True)
    bench.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
jobs=""
while [ "$#" -gt 0 ]; do
  if [ "$1" = "--jobs-dir" ]; then jobs="$2"; shift 2; else shift; fi
done
mkdir -p "$jobs/2026-06-07__00-00-00/citation-check__new"
python3 - "$jobs/2026-06-07__00-00-00/citation-check__new/result.json" <<'PY'
import json, sys
json.dump({
  "agent_result": {
    "total_tokens": 3,
    "n_input_tokens": 1,
    "n_output_tokens": 2,
    "usage_source": "provider_response"
  },
  "rewards": {"reward": 1.0},
  "timing": {"total": 4.2},
  "error": None,
  "partial_trajectory": False
}, open(sys.argv[1], "w"))
PY
"""
    )
    bench.chmod(bench.stat().st_mode | stat.S_IXUSR)
    (tmp_path / "keys.env").write_text("GEMINI_API_KEY=dummy\n")
    (skills_root / "tasks").mkdir(parents=True)

    result = subprocess.run(
        [
            "bash",
            str(script),
            "gemini-3.5-flash",
            "without",
            "citation-check",
            "1",
            "daytona",
            str(jobs_root),
            str(state_dir),
        ],
        env={
            **os.environ,
            "HOME": str(tmp_path),
            "BENCHFLOW_BENCH_ROOT": str(bench_root),
            "BENCHFLOW_SKILLSBENCH_ROOT": str(skills_root),
            "BENCHFLOW_KEYS_ENV": str(tmp_path / "keys.env"),
        },
        text=True,
        capture_output=True,
        check=True,
    )

    assert "completed" in result.stdout
    archived = list((jobs_root / ".archived-stale").glob(f"{cell}.*"))
    assert len(archived) == 1
    state = json.loads((state_dir / f"{cell}.json").read_text())
    assert state["status"] == "completed"
    assert state["reward"] == 1.0
    assert state["partial"] is False
    assert state["tokens"] == {"total": 3, "input": 1, "output": 2}
    assert "citation-check__new" in state["rollout_dir"]
