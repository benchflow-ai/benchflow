#!/usr/bin/env python3
"""Refresh pricing source metadata.

This script updates only ``retrieved_at`` and ``source_hash`` in
``src/benchflow/trajectories/pricing.py``. It intentionally does not scrape or
rewrite price values; pricing pages are not stable APIs, so numerical changes
should be reviewed by a human before editing the table.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import urllib.request
from datetime import UTC, date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRICING_PATH = REPO_ROOT / "src" / "benchflow" / "trajectories" / "pricing.py"
SOURCE_URL_RE = re.compile(r'source_url="([^"]+)"')


def _fetch_source_hash(url: str, timeout: float) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "benchflow-pricing-source-refresh/1.0",
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    normalized = b" ".join(payload.split())
    return hashlib.sha256(normalized).hexdigest()


def _source_urls(text: str) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for match in SOURCE_URL_RE.finditer(text):
        url = match.group(1)
        if url not in seen:
            urls.append(url)
            seen.add(url)
    return urls


def _replace_metadata(text: str, *, hashes: dict[str, str], retrieved_at: str) -> str:
    current_url: str | None = None
    lines = text.splitlines()
    updated: list[str] = []
    for line in lines:
        source_match = SOURCE_URL_RE.search(line)
        if source_match:
            current_url = source_match.group(1)
            updated.append(line)
            continue
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if current_url in hashes and stripped.startswith("retrieved_at="):
            updated.append(f'{indent}retrieved_at="{retrieved_at}",')
            continue
        if current_url in hashes and stripped.startswith("source_hash="):
            updated.append(f'{indent}source_hash="{hashes[current_url]}",')
            current_url = None
            continue
        updated.append(line)
    return "\n".join(updated) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh pricing.py source hashes from provider pricing pages."
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="retrieved_at date to write, default: today in local date",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="HTTP timeout per pricing source URL",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without modifying pricing.py",
    )
    args = parser.parse_args()

    text = PRICING_PATH.read_text()
    urls = _source_urls(text)
    if not urls:
        print(f"No source_url entries found in {PRICING_PATH}", file=sys.stderr)
        return 1

    hashes: dict[str, str] = {}
    for url in urls:
        print(f"Fetching {url}", file=sys.stderr)
        hashes[url] = _fetch_source_hash(url, args.timeout)

    updated = _replace_metadata(text, hashes=hashes, retrieved_at=args.date)
    for url, digest in hashes.items():
        print(f"{url} -> {args.date} sha256:{digest[:12]}", file=sys.stderr)

    if args.dry_run:
        print(updated)
        return 0

    PRICING_PATH.write_text(updated)
    print(
        f"Updated {PRICING_PATH.relative_to(REPO_ROOT)} at "
        f"{datetime.now(UTC).isoformat(timespec='seconds')}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
