"""The physical verifier. Runs inside the trusted simulator service (``verifier.service: simulator``), never in the
agent container. `verifier/test.sh` runs ``python3 -m benchflow.embodied.verifier``.

1. Close the episode. If the agent ended with `robo done` / `robo give-up`, or a budget ran out, the episode server
   has already judged it and exits on its own. Otherwise (the agent stopped without `robo done`) send `shutdown`;
   the server scores that as outcome `agent_exited` after the usual settle period.
2. Wait for the server to exit (it writes the video on the way out; ``serve.exit`` marks the end).
3. Copy the episode record to ``/logs/verifier/episode`` and write ``reward.txt`` (1 if the server judged success,
   else 0), ``reward.json`` (reward, success_ever, budget_used) and ``reward-details.json``.
4. Copy the video and the camera images the agent saw to ``/logs/artifacts``.

A missing result.json is an infrastructure error (exit 1, no reward), not a zero score. Standard library only.
"""

from __future__ import annotations

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


def _copy_artifacts(episode: Path, workspace: Path, artifacts: Path) -> None:
    if not artifacts.is_dir():
        return
    try:
        video = episode / "recording.mp4"
        if video.exists():
            shutil.copy2(video, artifacts / "recording.mp4")
        images = sorted((workspace / "observations").glob("*.png"))
        if images:
            (artifacts / "observations").mkdir(exist_ok=True)
            for f in images:
                shutil.copy2(f, artifacts / "observations" / f.name)
    except OSError as e:  # artifacts are a convenience copy; the evidence is in /logs/verifier/episode
        print(f"warning: could not copy artifacts: {e}", file=sys.stderr)


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
    artifacts = Path(_env("ARTIFACTS_DIR", "/logs/artifacts"))
    workspace = Path(_env("WORKSPACE_DIR", "/rpc/ws"))
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
        shutil.copytree(episode, out / "episode", dirs_exist_ok=True)
    _copy_artifacts(episode, workspace, artifacts)
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
