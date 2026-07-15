"""Live Casino Floor viewer: poll a running World + rebuild the browser HTML.

Polls the shared World's /_admin endpoints every few seconds, writes a run dir,
rebuilds casinobench's `casino-town.html` from it (with a meta-refresh so the
browser auto-reloads), into the directory served by a Cloudflare tunnel.

    uv run python examples/casino/live_viewer.py <world_url> <serve_dir>
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

CASINOBENCH = "/home/liu.10379/casinobench"


def main() -> None:
    world = sys.argv[1].rstrip("/")
    serve_dir = Path(sys.argv[2])
    serve_dir.mkdir(parents=True, exist_ok=True)
    run = serve_dir / "_run"
    run.mkdir(exist_ok=True)
    out = serve_dir / "index.html"

    while True:
        try:
            ev = httpx.get(f"{world}/_admin/events", timeout=8).json().get("jsonl", "")
            state = httpx.get(f"{world}/_admin/state", timeout=8).json()
            standings = httpx.get(f"{world}/_admin/standings", timeout=8).json()
            (run / "events.jsonl").write_text(ev)
            (run / "standings.json").write_text(json.dumps(standings))
            (run / "run.json").write_text(
                json.dumps(
                    {
                        "final_bankrolls": standings,
                        "game_config": state.get("game_config") or {"stake": 50},
                        "players": sorted(standings.keys()),
                        "starting_bankroll": int(state.get("starting_bankroll", 1000)),
                        "subject": state.get("subject", "agent"),
                    }
                )
            )
            r = subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    "viewer/build.py",
                    "--from",
                    str(run),
                    "--out",
                    str(out),
                ],
                cwd=CASINOBENCH,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if out.exists():
                html = out.read_text()
                if 'http-equiv="refresh"' not in html:
                    html = html.replace(
                        "<head>", '<head><meta http-equiv="refresh" content="6">', 1
                    )
                    out.write_text(html)
            else:
                print("build:", (r.stderr or r.stdout or "")[-200:], flush=True)
        except Exception as exc:
            print("live_viewer:", type(exc).__name__, str(exc)[:120], flush=True)
        time.sleep(6)


if __name__ == "__main__":
    main()
