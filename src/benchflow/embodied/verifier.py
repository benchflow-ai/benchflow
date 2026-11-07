"""The physical verifier. Runs inside the trusted simulator service (``verifier.service: simulator``), never in the
agent container. `verifier/test.sh` runs ``python3 -m benchflow.embodied.verifier``.

1. Close the episode. If the agent ended with `robo done` / `robo give-up`, or a budget ran out, the episode server
   has already judged it and exits on its own. Otherwise (the agent stopped without `robo done`) send `shutdown`;
   the server scores that as outcome `agent_exited` after the usual settle period.
2. Wait for the server to exit (it writes the video on the way out; ``serve.exit`` marks the end).
3. Copy the episode record (the video and the camera images included) to ``/logs/verifier/episode`` and write
   ``reward.txt`` (1 if the server judged success, else 0), ``reward.json`` (reward, success_ever, budget_used) and
   ``reward-details.json``. The details carry ``evidence``: the sha256 of every file of the record (the video, the
   request trace, the steps, the result, the camera images), read here in the trusted container, so a copy of any of
   them can be checked against the verdict it came with; and ``runtime``: the Python, OS and package versions the
   verdict was computed with.

The verifier writes nothing anywhere the agent's container can reach. BenchFlow places the trial's
``artifacts/recording.mp4`` and ``artifacts/observations/`` on the host afterwards, from the record this verifier
wrote (:func:`benchflow.embodied.export.place_episode_artifacts`), on every backend.

A missing result.json is an infrastructure error (exit 1, no reward), not a zero score. Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

from .protocol import send


def _env(name: str, default: str) -> str:
    return (
        os.environ.get(f"ROBO_{name}") or os.environ.get(f"ROBOUSE_{name}") or default
    )


# The record files every episode writes; evidence covers these and every other file of the record.
EVIDENCE_FILES = (
    "recording.mp4",
    "trace.jsonl",
    "steps.jsonl",
    "result.json",
    "episode.json",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def evidence(episode: Path) -> dict[str, str]:
    """The sha256 of every regular file of the episode record, by path relative to it (the video, the trace, the
    steps, the result, the camera images, ...). Links are left out: the record holds none of its own."""
    files = sorted(p for p in episode.rglob("*") if p.is_file() and not p.is_symlink())
    return {p.relative_to(episode).as_posix(): _sha256(p) for p in files}


def runtime() -> dict:
    """What the verdict was computed with: Python, the OS image, and every installed distribution's version."""
    import platform
    from importlib import metadata

    os_name = None
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                os_name = line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    packages = {
        d.metadata["Name"].lower(): d.version
        for d in metadata.distributions()
        if d.metadata["Name"]
    }
    return {
        "python": platform.python_version(),
        "os": os_name,
        "packages": dict(sorted(packages.items())),
    }


def rewards_from_result(r: dict) -> dict:
    """The reward.json map: the judged reward plus bounded episode metrics."""
    success = bool(r.get("success"))
    max_steps = max(1, int(r.get("max_steps") or 1))
    return {
        "reward": 1.0 if success else 0.0,
        "success_ever": 1.0 if r.get("success_ever") else 0.0,
        "budget_used": round(min(1.0, int(r.get("steps_used") or 0) / max_steps), 6),
    }


def main() -> int:
    episode = Path(_env("EPISODE_DIR", "/episode"))
    sock = _env("SOCKET", "/rpc/episode.sock")
    out = Path(_env("VERIFIER_DIR", "/logs/verifier"))
    wait_s = float(_env("VERIFIER_WAIT_S", "240"))
    out.mkdir(parents=True, exist_ok=True)
    exit_marker = episode / "serve.exit"
    closed_by = "episode_server"
    if not exit_marker.exists():
        status = send(sock, {"op": "status"}, timeout=60)
        print("status before close:", status)
        if status.get("ok") and not (status.get("result") or {}).get("finished"):
            closed_by = "verifier_shutdown"
        # shutdown is harmless if the episode already finished (the server is only lingering)
        print("shutdown:", send(sock, {"op": "shutdown"}, timeout=60))
    deadline = time.time() + wait_s
    while not exit_marker.exists() and time.time() < deadline:
        time.sleep(0.5)
    result_path = episode / "result.json"
    if episode.exists():
        shutil.copytree(episode, out / "episode", dirs_exist_ok=True, symlinks=True)
    if not result_path.exists():
        print(
            f"error: no episode result at {result_path} (episode server did not finish)",
            file=sys.stderr,
        )
        return 1
    r = json.loads(result_path.read_text())
    rewards = rewards_from_result(r)
    (out / "reward.txt").write_text("1" if rewards["reward"] == 1.0 else "0")
    (out / "reward.json").write_text(json.dumps(rewards))
    details = {
        **rewards,
        "closed_by": closed_by,
        "server_exit_code": exit_marker.read_text().strip()
        if exit_marker.exists()
        else None,
        **{k: v for k, v in r.items() if k != "reward"},
        "evidence": evidence(episode),
        "evidence_scope": "record",  # every file of the record; older verdicts named five
        "runtime": runtime(),
    }
    (out / "reward-details.json").write_text(json.dumps(details, indent=2))
    print(
        ("success" if rewards["reward"] else "fail"),
        r.get("outcome"),
        f"steps={r.get('steps_used')}/{r.get('max_steps')}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
