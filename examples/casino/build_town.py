"""Build the Stanford-Town-style casino viewer for a concurrent floor run.

Unlike casinobench's build.py (single --trajectory), this merges EVERY seat's
acp_trajectory.jsonl into one seq-keyed thinking map, so the town floor shows all
agents moving between tables WITH each one's reasoning overlaid on its actions.

    uv run python examples/casino/build_town.py <floor_run_dir> <out.html>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from casinobench.catalog import default_registry
from casinobench.event_log import EventLog
from casinobench.thinking import thinking_for_run
from casinobench.viewer_data import to_viewer_data
from casinobench.viewer_html import render_html


def main() -> int:
    run_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    events = list(EventLog.from_jsonl((run_dir / "events.jsonl").read_text()).events)

    fj = json.loads((run_dir / "floor.json").read_text())
    standings = fj.get("standings", {})
    players = sorted(standings) or sorted({e.actor for e in events if e.actor})
    starting = int(fj.get("starting_bankroll", 1000))
    game_config = fj.get("game_config") if isinstance(fj.get("game_config"), dict) else {"stake": 50}

    merged: dict[int, str] = {}
    for seat in players:
        tp = run_dir / seat / "trajectory" / "acp_trajectory.jsonl"
        if tp.exists():
            merged.update(thinking_for_run(events, tp, subject=seat))

    run = to_viewer_data(events, players, starting, default_registry(),
                         game_config=game_config, thinking=merged)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(run))
    print(f"wrote {out}: {len(run['events'])} events, {len(run['games'])} games, "
          f"{len(players)} players, thinking={len(merged)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
