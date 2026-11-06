"""Training export of embodied episodes.

For a trial whose verifier downloaded an episode record (``verifier/episode/``), write next to BenchFlow's other
trainer artifacts:

  trainer/embodied_steps.jsonl    one row per simulator step: obs (state before the step, filtered to what the agent
                                  could see in vision mode), action by group, reward, success, done, truncated,
                                  op (act / skill:<name> / settle), role, frame (video frame index), t
  trainer/embodied_episode.json   the embodiment spec, the video index (path relative to the trial folder), the
                                  return, the outcome, the seed and the ATIF session id of the trial

The next observation of row i is the obs of row i+1; the last row carries ``final_obs``. `export_job` collects all
trials of a job into ``steps.jsonl`` + ``episodes.jsonl``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

EPISODE_RELPATH = Path("verifier") / "episode"
STEPS_RELPATH = Path("trainer") / "embodied_steps.jsonl"
EPISODE_META_RELPATH = Path("trainer") / "embodied_episode.json"
TRUNCATED_OUTCOMES = {"budget_exhausted", "wall_time_exhausted", "agent_exited"}


def episode_dir(rollout_dir: str | Path) -> Path | None:
    d = Path(rollout_dir) / EPISODE_RELPATH
    return (
        d if (d / "steps.jsonl").is_file() and (d / "episode.json").is_file() else None
    )


def _read_jsonl(path: Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except (
                    json.JSONDecodeError
                ):  # a torn last line (server killed mid-write)
                    return


def episode_rows(ep_dir: Path) -> tuple[dict, list[dict]]:
    """(episode header + result, per-step rows) of one episode record."""
    header = json.loads((ep_dir / "episode.json").read_text())
    result_path = ep_dir / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    vision = header.get("observation_mode") == "vision"
    visible = set(header.get("visible_fields") or [])

    def view(st: Any) -> Any:
        if vision and isinstance(st, dict):
            return {k: v for k, v in st.items() if k in visible}
        return st

    obs = view(header.get("initial_state"))
    rows: list[dict] = []
    for rec in _read_jsonl(ep_dir / "steps.jsonl"):
        row = {
            "index": rec.get("i", len(rows)),
            "step": rec.get("step"),
            "t": rec.get("t"),
            "op": rec.get("op"),
            "obs": obs,
            "action": rec.get("action"),
            "reward": rec.get("reward", 0.0),
            "success": bool(rec.get("success")),
            "done": False,
            "truncated": False,
        }
        for k in ("role", "frame", "info"):
            if k in rec:
                row[k] = rec[k]
        rows.append(row)
        if "state" in rec:
            obs = view(rec["state"])
    if rows:
        rows[-1]["done"] = True
        rows[-1]["truncated"] = result.get("outcome") in TRUNCATED_OUTCOMES
        rows[-1]["final_obs"] = obs
    return {**header, "result": result}, rows


def write_rollout_embodied(
    rollout_dir: str | Path,
    *,
    trajectory_id: str | None = None,
    task_name: str | None = None,
    rewards: dict | None = None,
) -> dict | None:
    """Write the embodied trainer artifacts of one trial. Returns the episode summary, or None without a record."""
    rollout_dir = Path(rollout_dir)
    ep = episode_dir(rollout_dir)
    if ep is None:
        return None
    header, rows = episode_rows(ep)
    out = rollout_dir / STEPS_RELPATH
    out.parent.mkdir(parents=True, exist_ok=True)
    episode_id = trajectory_id or rollout_dir.name
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps({"episode_id": episode_id, **r}) + "\n")
    video_index = {}
    vi_path = ep / "video_index.json"
    if vi_path.exists():
        video_index = json.loads(vi_path.read_text())
        if video_index.get("video"):
            video_index["video"] = str(EPISODE_RELPATH / video_index["video"])
    result = header.get("result") or {}
    summary = {
        "episode_id": episode_id,
        "atif_session_id": trajectory_id,
        "task": task_name or header.get("task"),
        "source_task": header.get("task"),
        "seed": header.get("seed"),
        "backend": header.get("backend"),
        "embodiment": header.get("embodiment"),
        "observation_mode": header.get("observation_mode"),
        "visible_fields": header.get("visible_fields"),
        "success_mode": header.get("success_mode"),
        "initial_state_sha256": header.get("initial_state_sha256"),
        "n_steps": len(rows),
        "return": round(sum(float(r.get("reward") or 0.0) for r in rows), 6),
        "reward": (rewards or {}).get("reward", 1.0 if result.get("success") else 0.0),
        "success": result.get("success"),
        "outcome": result.get("outcome"),
        "steps_used": result.get("steps_used"),
        "video_index": video_index,
        "steps_path": str(STEPS_RELPATH),
    }
    (rollout_dir / EPISODE_META_RELPATH).write_text(json.dumps(summary, indent=2))
    return summary


def export_job(job_dir: str | Path, out_dir: str | Path) -> dict:
    """Collect every trial's embodied export of a job into ``out_dir/steps.jsonl`` and ``out_dir/episodes.jsonl``."""
    job_dir, out_dir = Path(job_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_eps = n_steps = 0
    with (
        open(out_dir / "steps.jsonl", "w") as fs,
        open(out_dir / "episodes.jsonl", "w") as fe,
    ):
        for rj in sorted(job_dir.glob("*/result.json")):
            trial = rj.parent
            try:
                res = json.loads(rj.read_text())
            except json.JSONDecodeError:
                continue
            summary = write_rollout_embodied(
                trial,
                trajectory_id=trial.name,
                task_name=res.get("task_name"),
                rewards=res.get("rewards"),
            )
            if summary is None:
                continue
            summary["trial"] = trial.name
            if summary["video_index"].get("video"):
                summary["video_index"]["video"] = str(
                    Path(trial.name) / summary["video_index"]["video"]
                )
            fe.write(json.dumps(summary) + "\n")
            with open(trial / STEPS_RELPATH) as f:
                for line in f:
                    fs.write(line)
                    n_steps += 1
            n_eps += 1
    return {"episodes": n_eps, "steps": n_steps, "out": str(out_dir)}
