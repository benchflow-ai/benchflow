"""Write expected/ for tests/trace_export.rs with Harbor's own exporters.

Run with Harbor installed and harbor-atif2otel importable, from this folder:

    PYTHONPATH=<harbor>/packages/harbor-atif2otel/src python make_expected.py

For every trial of every job below it reads the trial's ATIF document
(trainer/atif.json, else agent/trajectory.json; trials with neither are left
to the Rust test), then writes:

- expected/<job>.hf.jsonl: Harbor's rows (every episode, or the last for the
  jobs marked so below) (extract_conversations_from_trajectory,
  openai_to_sharegpt), with the run metadata BenchFlow maps from result.json.
- expected/<job>.otel.jsonl: Harbor's OTLP JSON (convert_trajectory with the
  trial's folder name as the trace seed), with service.name "benchflow" and the
  resource and scope naming BenchFlow, whose version is written as VERSION.
"""

import json
import sys
from pathlib import Path

from harbor.utils.traces_utils import (
    MultimodalExportError,
    extract_conversations_from_trajectory,
    openai_to_sharegpt,
)
from harbor_atif2otel.convert import convert_trajectory, resource_spans_to_otlp_json

HERE = Path(__file__).resolve().parent
# Each job, and whether to keep every episode or only the last (Harbor's
# --episodes); the long 0.8 Claude job keeps the last to stay small.
JOBS = {
    "synthetic": (HERE / "jobs/synthetic", "all"),
    "rust-claude": (HERE / "jobs/rust-claude", "all"),
    "rb-claude-opus5-3": (HERE / "../trainer-0.8/rb-claude-opus5-3", "last"),
    "rb-codex-astra3": (HERE / "../trainer-0.8/rb-codex-astra3", "all"),
}


def atif_path(trial: Path) -> Path | None:
    for rel in ("trainer/atif.json", "agent/trajectory.json"):
        if (trial / rel).is_file():
            return trial / rel
    return None


def run_meta(result: dict, job: str) -> dict:
    model = result.get("model") or None
    reward = (result.get("rewards") or {}).get("reward")
    if reward is not None:
        outcome = str(float(reward))
    elif result.get("problem"):
        outcome = result["problem"]["code"]
    elif result.get("status") == "errored":
        outcome = "errored"
    else:
        outcome = None
    return {
        "agent_name": result.get("agent") or "unknown-agent",
        "model_name": model or "unknown-model",
        "model_provider": model.split("/", 1)[0] if model and "/" in model else "unknown-provider",
        "start_time": result.get("started_at") or "",
        "run_id": job,
        "task_name": result["task_name"],
        "trial_name": result["rollout_name"],
        "result": outcome,
    }


def main() -> None:
    out = HERE / "expected"
    out.mkdir(exist_ok=True)
    for job, (root, episodes) in JOBS.items():
        hf_lines, otel_lines = [], []
        for trial in sorted(p for p in root.iterdir() if (p / "result.json").is_file()):
            path = atif_path(trial)
            if path is None:
                continue
            result = json.loads((trial / "result.json").read_text())
            meta = run_meta(result, job)
            try:
                rows = extract_conversations_from_trajectory(path, meta)
            except MultimodalExportError:
                rows = []
            if episodes == "last":
                rows = rows[-1:]
            for row in rows:
                row["task"] = meta["task_name"]
                row["trial_name"] = meta["trial_name"]
                row["result"] = meta["result"]
                row["conversations_sharegpt"] = openai_to_sharegpt(row["conversations"])
                hf_lines.append(json.dumps(row, ensure_ascii=False))
            traj = json.loads(path.read_text())
            rs = convert_trajectory(traj, trace_seed=trial.name, service_name="benchflow")
            for kv in rs.resource.attributes:
                if kv.key == "telemetry.sdk.language":
                    kv.value.string_value = "rust"
                elif kv.key == "telemetry.sdk.name":
                    kv.value.string_value = "benchflow"
                elif kv.key == "telemetry.sdk.version":
                    kv.value.string_value = "VERSION"
            for ss in rs.scope_spans:
                ss.scope.name = "benchflow"
                ss.scope.version = "VERSION"
            otel_lines.append(json.dumps(resource_spans_to_otlp_json(rs)))
        (out / f"{job}.hf.jsonl").write_text("".join(l + "\n" for l in hf_lines))
        (out / f"{job}.otel.jsonl").write_text("".join(l + "\n" for l in otel_lines))
        print(f"{job}: {len(hf_lines)} rows, {len(otel_lines)} traces", file=sys.stderr)


if __name__ == "__main__":
    main()
