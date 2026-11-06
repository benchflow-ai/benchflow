"""Training export of embodied episodes.

For a trial whose verifier downloaded an episode record (``verifier/episode/``), place the episode's video and camera
images in the trial's ``artifacts/`` (:func:`place_episode_artifacts`), then write next to BenchFlow's other trainer
artifacts:

  trainer/embodied_steps.jsonl    one row per simulator step: obs (state before the step, filtered to what the agent
                                  could see in vision mode), action by group, reward, success, done, truncated,
                                  op (act / skill:<name> / settle), role, frame (video frame index), t
  trainer/embodied_episode.json   the embodiment spec, the video index (path relative to the trial folder), the
                                  return, the outcome, the seed, the ATIF session id of the trial, and `evidence`:
                                  the record's sha256s from the verifier and whether the trial's video files still
                                  match them (`video_matches_verdict`; None when the verifier recorded no video)

The next observation of row i is the obs of row i+1; the last row carries ``final_obs``. `export_job` collects all
trials of a job into ``steps.jsonl`` + ``episodes.jsonl``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

EPISODE_RELPATH = Path("verifier") / "episode"
STEPS_RELPATH = Path("trainer") / "embodied_steps.jsonl"
EPISODE_META_RELPATH = Path("trainer") / "embodied_episode.json"
REWARD_DETAILS_RELPATH = Path("verifier") / "reward-details.json"
ARTIFACTS_RELPATH = Path("artifacts")
ARTIFACTS_MANIFEST = "artifacts-manifest.json"
# The trial's copies of the episode video: the record's own and the copy placed in artifacts/.
VIDEO_RELPATHS = (
    EPISODE_RELPATH / "recording.mp4",
    ARTIFACTS_RELPATH / "recording.mp4",
)


def _clear(path: Path) -> None:
    """Remove what is at ``path`` (a file, a link or a folder) without following a link."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif os.path.lexists(path):  # a socket or a fifo
        path.unlink()


def _place(src: Path, dst: Path) -> None:
    """Copy ``src`` to a fresh ``dst``: what is at that name goes first, and the new file is created with
    O_EXCL | O_NOFOLLOW, so a link left there is never followed."""
    _clear(dst)
    fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, "wb") as out, open(src, "rb") as inp:
        shutil.copyfileobj(inp, out)


def place_episode_artifacts(rollout_dir: str | Path) -> list[dict]:
    """Put the episode's video and camera images in ``artifacts/``, from the record the verifier downloaded.

    On Docker the agent's container mounts ``artifacts/``, so it can leave files and links there during its turn.
    Whatever is at ``recording.mp4`` and ``observations/`` is replaced (never followed), and when the episode
    recorded no video, a ``recording.mp4`` there is removed: only the record's files carry those names. The
    placed files are added to ``artifacts-manifest.json`` as an ``embodied`` collection. Returns their entries.
    """
    rollout_dir = Path(rollout_dir)
    episode = rollout_dir / EPISODE_RELPATH
    artifacts = rollout_dir / ARTIFACTS_RELPATH
    if not episode.is_dir() or episode.is_symlink():
        return []
    if artifacts.is_symlink():
        artifacts.unlink()
    artifacts.mkdir(parents=True, exist_ok=True)
    placed: list[tuple[Path, Path]] = []
    video = episode / "recording.mp4"
    if video.is_file() and not video.is_symlink():
        _place(video, artifacts / "recording.mp4")
        placed.append((video, Path("recording.mp4")))
    else:
        _clear(artifacts / "recording.mp4")
    saved = episode / "observations"
    images = []
    if saved.is_dir() and not saved.is_symlink():
        images = sorted(
            f for f in saved.glob("*.png") if f.is_file() and not f.is_symlink()
        )
    _clear(artifacts / "observations")
    if images:
        (artifacts / "observations").mkdir()
        for f in images:
            _place(f, artifacts / "observations" / f.name)
            placed.append((f, Path("observations") / f.name))
    entries = [
        {
            "path": rel.as_posix(),
            "kind": "file",
            "size": src.stat().st_size,
            "sha256": _sha256(src).removeprefix("sha256:"),
        }
        for src, rel in placed
    ]
    _record_in_manifest(rollout_dir / ARTIFACTS_MANIFEST, entries)
    return entries


def _record_in_manifest(manifest_path: Path, entries: list[dict]) -> None:
    """Add (or replace) the ``embodied`` collection of the trial's artifacts manifest, when it has one."""
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("collections"), list
    ):
        return
    collections = [
        c
        for c in manifest["collections"]
        if not (isinstance(c, dict) and c.get("kind") == "embodied")
    ]
    index = len(collections)
    collections.append(
        {
            "kind": "embodied",
            "source": str(EPISODE_RELPATH),
            "destination": ".",
            "status": "collected" if entries else "empty",
        }
    )
    names = {e["path"] for e in entries}
    files = [
        f
        for f in manifest.get("files") or []
        if isinstance(f, dict)
        and f.get("path") not in names
        and f.get("path") != "observations"
        and not str(f.get("path", "")).startswith("observations/")
    ]
    files += [{**e, "collection": index} for e in entries]
    manifest["collections"], manifest["files"] = collections, files
    manifest["total_files"] = sum(1 for f in files if f.get("kind") == "file")
    manifest["total_bytes"] = sum(
        int(f.get("size") or 0) for f in files if f.get("kind") == "file"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))


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
    place_episode_artifacts(rollout_dir)
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
        "evidence": verdict_evidence(rollout_dir),
    }
    (rollout_dir / EPISODE_META_RELPATH).write_text(json.dumps(summary, indent=2))
    return summary


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def verdict_evidence(rollout_dir: str | Path) -> dict:
    """The verifier's record hashes (reward-details.json ``evidence``) and a check of the trial's video files.

    ``video_matches_verdict`` is True when every copy of the video in the trial folder hashes to what the verifier
    read in the trusted container, False when one differs or is missing, and None when the verifier recorded no
    video hash (no video, or a verifier from before 0.10). Run after :func:`place_episode_artifacts`.
    """
    rollout_dir = Path(rollout_dir)
    try:
        details = json.loads((rollout_dir / REWARD_DETAILS_RELPATH).read_text())
    except (OSError, json.JSONDecodeError):
        details = {}
    recorded = details.get("evidence") if isinstance(details, dict) else None
    recorded = recorded if isinstance(recorded, dict) else {}
    expected = recorded.get("recording.mp4")
    copies = {}
    for rel in VIDEO_RELPATHS:
        path = rollout_dir / rel
        if path.is_file() and not path.is_symlink():
            copies[str(rel)] = _sha256(path)
    matches = None
    if isinstance(expected, str):
        matches = all(copies.get(str(rel)) == expected for rel in VIDEO_RELPATHS)
    return {
        "recorded": recorded,
        "video_copies": copies,
        "video_matches_verdict": matches,
    }


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
