"""Check an embodied run against the evidence its verdicts carry, offline: ``bench embodied verify <job>``.

For each trial with an episode record, recompute now, from the files on disk:

- every file the verdict names in ``evidence`` still hashes to what the verifier read in the trusted container.
  Since 0.10 that is every file of the record (``evidence_scope: record``), so a file added to the record later is
  named too; older verdicts named five files (the video, trace, steps, result and episode header);
- every copy in the trial folder of the video and of the camera images (``artifacts/``) matches those hashes;
- the reward agrees with the record: ``verifier/reward.txt``, the verdict and the trial's ``result.json`` give 1
  exactly when the episode's ``result.json`` says success.

``trainer/embodied_episode.json``'s ``video_matches_verdict`` was computed when the trial finished; this checks the
files as they are now, so a copy edited later fails here.

What it can't show: it doesn't re-run the simulator, and it checks a folder against the verdict stored in the same
folder, so it catches edits and mix-ups made after the trial, not a folder rewritten whole by whoever holds it.

A trial whose verifier left no verdict is reported as ``no verdict``. BenchFlow scored it 0: a verifier that writes
no reward lowers the trial's score and never removes the trial, so no agent can reroll its attempt by breaking the
grader. This names those trials so an auditor can count them apart from the agent's failures.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .export import EPISODE_RELPATH, REWARD_DETAILS_RELPATH, VIDEO_RELPATHS

STATUS_OK = "ok"
STATUS_MISMATCH = "mismatch"
STATUS_NO_VERDICT = "no verdict"
STATUS_NO_EVIDENCE = "no evidence"  # a verifier from before 0.10 recorded no hashes


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _reward_text(path: Path) -> float | None:
    try:
        return float(path.read_text().strip())
    except (OSError, ValueError):
        return None


def is_embodied_trial(trial: Path) -> bool:
    return (trial / EPISODE_RELPATH).is_dir() or (
        trial / REWARD_DETAILS_RELPATH
    ).is_file()


def verify_trial(trial: str | Path) -> dict[str, Any]:
    """The checks of one trial and its status (ok, mismatch, no verdict, no evidence)."""
    trial = Path(trial)
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": ok, "detail": detail})

    result = _json(trial / "result.json") or {}
    task = result.get("task_name") or trial.name.rsplit("__", 1)[0]
    out = {"trial": trial.name, "task": task, "checks": checks}
    details = _json(trial / REWARD_DETAILS_RELPATH)
    if not isinstance(details, dict):
        warnings = [
            w
            for w in result.get("warnings") or []
            if isinstance(w, str) and "Verifier scored 0" in w
        ]
        reason = (
            # result.json marks paths as <reachable:...>; the report names them plainly
            re.sub(r"<reachable:([^>]*)>", r"\1", warnings[0])
            if warnings
            else "the verifier wrote no reward-details.json"
        )
        check("verdict", False, reason)
        return {**out, "status": STATUS_NO_VERDICT}
    evidence = details.get("evidence")
    if not isinstance(evidence, dict) or not evidence:
        check(
            "evidence",
            False,
            "the verdict records no hashes (a verifier from before 0.10)",
        )
        return {**out, "status": STATUS_NO_EVIDENCE}

    episode = trial / EPISODE_RELPATH
    for name in sorted(evidence):
        recorded = evidence[name]
        path = episode / name
        if not path.is_file() or path.is_symlink():
            check(f"record {name}", False, "missing")
            continue
        now = _sha256(path)
        check(
            f"record {name}",
            now == recorded,
            ""
            if now == recorded
            else f"now {now[:19]}…, verdict {str(recorded)[:19]}…",
        )
    if details.get("evidence_scope") == "record" and episode.is_dir():
        # The verdict hashed the whole record, so a file or link it doesn't name was added afterwards.
        for path in sorted(episode.rglob("*")):
            if path.is_dir() and not path.is_symlink():
                continue
            name = path.relative_to(episode).as_posix()
            if name not in evidence:
                check(f"record {name}", False, "not in the verdict: added later")

    video = evidence.get("recording.mp4")
    if isinstance(video, str):
        for rel in VIDEO_RELPATHS:
            path = trial / rel
            if not path.is_file() or path.is_symlink():
                check(f"video {rel}", False, "missing")
                continue
            now = _sha256(path)
            check(
                f"video {rel}",
                now == video,
                "" if now == video else "differs from the verdict's video",
            )
    observations = trial / "artifacts" / "observations"
    if observations.is_dir():
        # Each camera image BenchFlow placed in artifacts/ is a copy of one the verdict hashed.
        for path in sorted(observations.iterdir()):
            rel = f"artifacts/observations/{path.name}"
            recorded = evidence.get(f"observations/{path.name}")
            if recorded is None:
                if details.get("evidence_scope") == "record":
                    check(f"image {rel}", False, "not in the verdict")
                continue
            ok = path.is_file() and not path.is_symlink() and _sha256(path) == recorded
            check(f"image {rel}", ok, "" if ok else "differs from the verdict's image")

    episode_result = _json(episode / "result.json")
    if isinstance(episode_result, dict) and "success" in episode_result:
        expected = 1.0 if episode_result.get("success") else 0.0
        reward_json = _json(trial / "verifier" / "reward.json")
        given = {
            "verifier/reward.txt": _reward_text(trial / "verifier" / "reward.txt"),
            # reward.json carries the same reward with the metrics; a reader of it gets the same check
            "verifier/reward.json": reward_json.get("reward")
            if isinstance(reward_json, dict)
            else None,
            "verdict": details.get("reward"),
            "result.json": (result.get("rewards") or {}).get("reward")
            if isinstance(result.get("rewards"), dict)
            else None,
        }
        for where, value in given.items():
            if value is None:
                continue
            ok = float(value) == expected
            check(
                f"reward {where}",
                ok,
                ""
                if ok
                else f"{value} but the episode's result says success={episode_result.get('success')}",
            )
    else:
        check("reward", False, "the episode's result.json is missing or unreadable")

    status = STATUS_OK if all(c["ok"] for c in checks) else STATUS_MISMATCH
    return {
        **out,
        "status": status,
        "outcome": details.get("outcome"),
        "reward": details.get("reward"),
    }


def _trial_dirs(job_dir: Path) -> list[Path]:
    """The embodied trial folders of a job folder (``jobs/<job>``, every run in it) or of one run
    (``jobs/<job>/<time>``): every one, retries and reruns included."""
    found: list[Path] = []
    for p in sorted(job_dir.iterdir()):
        if not p.is_dir() or p.is_symlink():
            continue
        if is_embodied_trial(p):
            found.append(p)
            continue
        found.extend(
            c
            for c in sorted(p.iterdir())
            if c.is_dir() and not c.is_symlink() and is_embodied_trial(c)
        )
    return found


def verify_job(job_dir: str | Path) -> dict[str, Any]:
    """Every embodied trial of a job, checked; counts by status."""
    job_dir = Path(job_dir)
    trials = []
    for t in _trial_dirs(job_dir):
        checked = verify_trial(t)
        if t.parent != job_dir:
            checked["run"] = t.parent.name
        trials.append(checked)
    counts: dict[str, int] = {}
    for t in trials:
        counts[t["status"]] = counts.get(t["status"], 0) + 1
    return {
        "job": str(job_dir),
        "trials": trials,
        "counts": counts,
        "ok": bool(trials) and counts.get(STATUS_OK, 0) == len(trials),
    }


def format_report(report: dict[str, Any]) -> str:
    """A fixed-width table: one line per trial, then the failed checks, then the counts."""
    rows = [f"{'trial':<44} {'status':<12} {'reward':>6}  checks"]
    runs = {t.get("run") for t in report["trials"]}
    run = None
    for t in report["trials"]:
        if len(runs) > 1 and t.get("run") != run:
            run = t.get("run")
            rows.append(f"run {run}")
        passed = sum(c["ok"] for c in t["checks"])
        reward = t.get("reward")
        rows.append(
            f"{t['trial'][:44]:<44} {t['status']:<12} {'' if reward is None else f'{float(reward):.1f}':>6}  "
            f"{passed}/{len(t['checks'])}"
        )
        for c in t["checks"]:
            if not c["ok"]:
                rows.append(f"    {c['check']}: {c['detail']}")
    n = len(report["trials"])
    counts = report["counts"]
    rows.append(
        f"{counts.get(STATUS_OK, 0)} of {n} trials check out; {counts.get(STATUS_MISMATCH, 0)} mismatch, "
        f"{counts.get(STATUS_NO_VERDICT, 0)} without a verdict (scored 0), {counts.get(STATUS_NO_EVIDENCE, 0)} without evidence"
    )
    return "\n".join(rows)
