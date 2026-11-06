#!/bin/bash
set -euo pipefail

WORKSPACE="${BENCHFLOW_WORKSPACE:-/root}"
INPUT_BIB="$WORKSPACE/test.bib"
ANSWER_JSON="$WORKSPACE/answer.json"

mkdir -p "$WORKSPACE"

python3 - "$INPUT_BIB" "$ANSWER_JSON" <<'PY'
import json
import re
import sys
from pathlib import Path

input_bib = Path(sys.argv[1])
answer_json = Path(sys.argv[2])

# The oracle is a deterministic ground-truth solver for the checked-in fixture.
# Agents may use live academic APIs, but CI oracle determinism must not depend on
# CrossRef/Semantic Scholar availability or changing external metadata.
FAKE_CITATION_KEYS = {
    "patel2023blockchain",
    "smith2020ai",
    "wilson2021neural",
}


def clean_title(title: str) -> str:
    title = re.sub(r"[{}\\]", "", title)
    return " ".join(title.split())


def iter_bibtex_entries(text: str):
    pattern = re.compile(
        r"@\w+\s*\{\s*(?P<key>[^,\s]+)\s*,(?P<body>.*?)(?=\n@\w+\s*\{|\Z)",
        re.DOTALL,
    )
    yield from pattern.finditer(text)


def title_from_body(body: str) -> str | None:
    match = re.search(
        r"^\s*title\s*=\s*(?:\{(?P<braced>.*?)\}|\"(?P<quoted>.*?)\")\s*,?\s*$",
        body,
        re.IGNORECASE | re.MULTILINE,
    )
    if match is None:
        return None
    value = match.group("braced") if match.group("braced") is not None else match.group("quoted")
    return clean_title(value)


text = input_bib.read_text(encoding="utf-8")
titles_by_key: dict[str, str] = {}

for entry in iter_bibtex_entries(text):
    key = entry.group("key")
    if key not in FAKE_CITATION_KEYS:
        continue
    title = title_from_body(entry.group("body"))
    if title:
        titles_by_key[key] = title

missing = sorted(FAKE_CITATION_KEYS - titles_by_key.keys())
if missing:
    raise SystemExit(f"missing expected fake citation entries: {', '.join(missing)}")

fake_titles = sorted(titles_by_key.values())
answer_json.write_text(
    json.dumps({"fake_citations": fake_titles}, indent=2) + "\n",
    encoding="utf-8",
)

print(f"Wrote {len(fake_titles)} fake citation titles to {answer_json}")
PY
