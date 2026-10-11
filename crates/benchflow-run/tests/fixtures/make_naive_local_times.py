"""Write python-naive-local-times.json: naive local times as CPython reads them.

BenchFlow 0.8 writes a trial's started_at and finished_at as naive local
times (str(datetime.now())), and reads them back with
datetime.fromisoformat(); turning one into an instant (.timestamp()) uses
the machine's time zone with fold=0. Around a clock change a naive time can
be skipped (the clocks jump over it) or repeated. This records, for times
every 5 minutes around clock changes in four zones, the instant CPython
gives: a 1-hour change (America/Los_Angeles), a 30-minute one
(Australia/Lord_Howe), a 2-hour one (Antarctica/Troll) and a skipped day
(Pacific/Apia, 2011-12-30).

Each case: zone, the naive text, and CPython's .timestamp() split into
whole seconds since the epoch and microseconds.

Run it with 0.8's Python (from any directory; it sets TZ itself):

    uv run --project /path/to/benchflow-0.8.0 python -I make_naive_local_times.py

It rewrites the JSON next to this script; `--check` compares instead.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

OUT = Path(__file__).resolve().parent / "python-naive-local-times.json"

# Zone, then (first naive time, last naive time) windows around its changes.
ZONES = {
    "America/Los_Angeles": [
        ("2026-03-08 00:00", "2026-03-08 05:00"),  # 02:00 -> 03:00
        ("2026-11-01 00:00", "2026-11-01 05:00"),  # 02:00 -> 01:00
    ],
    "Australia/Lord_Howe": [
        ("2026-04-05 00:00", "2026-04-05 05:00"),  # 02:00 -> 01:30
        ("2026-10-04 00:00", "2026-10-04 05:00"),  # 02:00 -> 02:30
    ],
    "Antarctica/Troll": [
        ("2026-03-29 00:00", "2026-03-29 05:00"),  # 01:00 -> 03:00
        ("2026-10-25 00:00", "2026-10-25 05:00"),  # 03:00 -> 01:00
    ],
    "Pacific/Apia": [
        ("2011-12-29 22:00", "2011-12-31 02:00"),  # Dec 30 skipped
    ],
}
STEP = timedelta(minutes=5)


def main() -> int:
    cases = []
    for zone, windows in ZONES.items():
        os.environ["TZ"] = zone
        time.tzset()
        for first, last in windows:
            t, end = datetime.fromisoformat(first), datetime.fromisoformat(last)
            while t <= end:
                # 0.8's text: whole seconds, or with microseconds.
                for text in (str(t), str(t + timedelta(seconds=7, microseconds=250000))):
                    ts = datetime.fromisoformat(text).timestamp()
                    whole = datetime.fromisoformat(text).replace(microsecond=0).timestamp()
                    micros = datetime.fromisoformat(text).microsecond
                    assert ts == whole + micros / 1e6
                    cases.append(
                        {"zone": zone, "naive": text, "seconds": int(whole), "micros": micros}
                    )
                t += STEP
    rows = ",\n".join(json.dumps(c) for c in cases)
    text = f'{{"python": "{sys.version.split()[0]}", "cases": [\n{rows}\n]}}\n'
    if "--check" in sys.argv[1:]:
        same = OUT.read_text(encoding="utf-8") == text
        print(f"{OUT.name}: {'same' if same else 'DIFFERENT'} ({len(cases)} cases)")
        return 0 if same else 1
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {len(cases)} cases to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
