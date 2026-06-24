"""End-to-end vertical slice: ClawsBench manifest → scored trajectory →
Verifiers/ORS export.

The data-path test proves the thread with no Docker. A full live rollout is
run manually:

    bench eval run --tasks-dir benchmarks/clawsbench/tasks/<task> \\
      --environment-manifest benchmarks/clawsbench/environment.toml \\
      --agent claude-agent-acp --model claude-haiku-4-5
"""

import json
from pathlib import Path

from benchflow.environment.manifest import load_manifest
from benchflow.rewards.protocol import VerifyResult
from benchflow.trajectories.export import (
    export_trajectories_to_jsonl,
    trajectory_to_verifiers_record,
)

MANIFEST = Path("benchmarks/clawsbench/environment.toml")


def test_manifest_drives_a_scored_exportable_record(tmp_path):
    """The ClawsBench manifest loads, and a scored trajectory from it
    exports to a valid Verifiers dataset line — the slice's data path."""
    manifest = load_manifest(MANIFEST)
    assert manifest.name == "clawsbench"

    # A scored trajectory, as Rollout.verify() would produce one.
    messages = [
        {"role": "user", "content": "Archive Alice's email."},
        {"role": "assistant", "content": "Archived 1 email."},
    ]
    result = VerifyResult(reward=1.0, items={"exact_match": 1.0})

    record = trajectory_to_verifiers_record(
        task_id="clawsbench/archive-alice",
        messages=messages,
        verify_result=result,
        model="claude-haiku-4-5",
        environment=manifest.name,
    )
    out = tmp_path / "clawsbench_dataset.jsonl"
    export_trajectories_to_jsonl([record], out)

    parsed = json.loads(out.read_text().strip())
    assert parsed["reward"] == 1.0
    assert parsed["info"]["environment"] == "clawsbench"
    assert parsed["prompt"] and parsed["completion"]
    assert parsed["metrics"] == {"exact_match": 1.0}
