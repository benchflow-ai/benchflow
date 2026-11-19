"""Refuse a release whose price snapshot is past a vendor's announced change.

``src/benchflow/providers/pricing_snapshot.py`` dates the rates a vendor has
said will change (``VALID_THROUGH``): Gemini 4 Argon's double on 2026-12-01.
A release cut after that day would price those calls at the old rates, so
both release workflows run this before they publish; it exits 1 and names
each such entry. The unit tests read the snapshot as of its own date and
never fail on the calendar. Refresh an entry from the vendor's
pricing page, with its new date, rather than moving the date alone.

Run with ``--no-project``: it loads the snapshot module by its path, which
imports nothing but the standard library.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from types import ModuleType

SNAPSHOT = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "benchflow"
    / "providers"
    / "pricing_snapshot.py"
)


def _snapshot(path: Path = SNAPSHOT) -> ModuleType:
    spec = importlib.util.spec_from_file_location("_pricing_snapshot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Refuse a release whose price snapshot is past a vendor's "
        "announced change."
    )
    parser.add_argument(
        "--today",
        type=date.fromisoformat,
        default=datetime.now(UTC).date(),
        help="the day to check against (default: today, UTC)",
    )
    args = parser.parse_args(argv)
    snapshot = _snapshot()
    past = snapshot.rates_past_their_change(args.today)
    if past:
        for (family, model), last_day in sorted(past.items()):
            print(
                f"{family}/{model}: its snapshot rates ended {last_day}; read the "
                "vendor's pricing page and update pricing_snapshot.PRICES and "
                "VALID_THROUGH",
                file=sys.stderr,
            )
        return 1
    print(
        f"Every dated rate in the price snapshot ({snapshot.SNAPSHOT_DATE}) is "
        f"current on {args.today}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
