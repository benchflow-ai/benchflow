"""Snapshot writer for the custom Casino Floor viewer.

Polls the running World (standings + event log) and reads each seat's live
acp_trajectory.jsonl, writing `state.json` + `traj/<seat>.json` into the served
directory for floor.html. Robust: when the World is down (run ended) it falls back
to the persisted floor.json, so the viewer keeps working after the run finishes.

    uv run python examples/casino/snapshot.py <world_url> <run_dir> <serve_dir> [--once]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx


def _roster(run_dir: Path) -> dict:
    f = run_dir / "roster.json"
    if f.exists():
        return {r["seat"]: r for r in json.loads(f.read_text())}
    return {}


def _events(jsonl: str) -> list[dict]:
    out = []
    for line in jsonl.splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        p = e.get("payload") or {}
        kind = p.get("type") or e.get("type") or e.get("kind") or ""
        if "games" in p:
            text = "table opened: " + ", ".join(p["games"])
        else:
            text = p.get("action") or p.get("note") or p.get("game") or json.dumps(p)[:90]
        out.append({"seat": e.get("actor") or "", "text": f"{kind} {text}".strip()[:140]})
    return out


def _traj_rows(run_dir: Path, seat: str) -> list[dict]:
    p = run_dir / seat / "trajectory" / "acp_trajectory.jsonl"
    rows = []
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def snapshot(world: str, run_dir: Path, serve: Path) -> None:
    (serve / "traj").mkdir(parents=True, exist_ok=True)
    roster = _roster(run_dir)
    running = True
    standings, events, statuses = {}, [], {}
    try:
        standings = httpx.get(f"{world}/_admin/standings", timeout=5).json()
        events = _events(httpx.get(f"{world}/_admin/events", timeout=5).json().get("jsonl", ""))
    except Exception:  # noqa: BLE001 — World down (run ended) → use persisted floor.json
        running = False
        fj = run_dir / "floor.json"
        if fj.exists():
            d = json.loads(fj.read_text())
            standings = d.get("standings", {})
            for r in d.get("results", []):
                statuses[r["seat"]] = r.get("status")
                roster.setdefault(r["seat"], {}).update(agent=r["agent"], model=r["model"])
        ev = run_dir / "events.jsonl"  # persisted at teardown — narration for finished runs
        if ev.exists():
            events = _events(ev.read_text())

    seats = list(roster) or list(standings)
    agents = []
    for seat in seats:
        rows = _traj_rows(run_dir, seat)
        (serve / "traj" / f"{seat}.json").write_text(json.dumps(rows))
        meta = roster.get(seat, {})
        agents.append({
            "seat": seat, "agent": meta.get("agent", ""), "model": meta.get("model", ""),
            "chips": standings.get(seat),
            "status": statuses.get(seat, "playing" if running else "ended"),
            "tool_calls": sum(1 for r in rows if r.get("type") == "tool_call"),
        })
    state = {"running": running, "updated": time.strftime("%H:%M:%S"),
             "standings": standings, "agents": agents, "events": events}
    (serve / "state.json").write_text(json.dumps(state))


def main() -> None:
    world, run_dir, serve = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    once = "--once" in sys.argv
    while True:
        try:
            snapshot(world, run_dir, serve)
        except Exception as exc:  # noqa: BLE001
            print("snapshot:", type(exc).__name__, str(exc)[:120], flush=True)
        if once:
            return
        time.sleep(3)


if __name__ == "__main__":
    main()
