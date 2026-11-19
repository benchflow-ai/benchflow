"""Print the model catalog a Codex binary bundles, as src/benchflow/agents/codex_models.py holds it.

    curl -fsSO https://registry.npmjs.org/@openai/codex/-/codex-<version>-linux-x64.tgz
    tar xzf codex-<version>-linux-x64.tgz
    python tools/codex_model_catalog.py package/vendor/x86_64-unknown-linux-musl/bin/codex

Codex embeds its ``models.json`` in the binary. This finds the JSON document
that holds the first ``"supported_reasoning_levels": [`` entry, parses it, and
prints its size, sha256 and each model's default and supported efforts.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def _document(data: bytes) -> bytes:
    at = data.find(b'"supported_reasoning_levels": [')
    if at < 0:
        raise SystemExit("no model catalog found in this file")
    start = data.rfind(b"{", 0, data.rfind(b'"models"', 0, at))
    depth, index, in_string, escaped = 0, start, False, False
    while index < len(data):
        char = data[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == 0x5C:
                escaped = True
            elif char == 0x22:
                in_string = False
        elif char == 0x22:
            in_string = True
        elif char == 0x7B:
            depth += 1
        elif char == 0x7D:
            depth -= 1
            if depth == 0:
                return data[start : index + 1]
        index += 1
    raise SystemExit("the model catalog does not end")


def main(path: str) -> None:
    raw = _document(Path(path).read_bytes())
    catalog = json.loads(raw)
    print(f"# {len(raw)} bytes, sha256 {hashlib.sha256(raw).hexdigest()}")
    for model in catalog["models"]:
        levels = tuple(level["effort"] for level in model["supported_reasoning_levels"])
        print(
            f'    "{model["slug"]}": ("{model["default_reasoning_level"]}", {levels!r}),'
        )


if __name__ == "__main__":
    main(sys.argv[1])
