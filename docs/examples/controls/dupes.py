"""Near-duplicate tasks in a folder of tasks.

Compares every pair of task instructions by the share of five-word runs they have in
common (Jaccard similarity). Reads each task's instruction.md, or its task.md when there
is no instruction.md.

    python3 dupes.py <folder of task folders>
"""

import itertools
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
runs = {}
for task in sorted(p for p in root.iterdir() if p.is_dir()):
    text_file = task / "instruction.md"
    if not text_file.exists():
        text_file = task / "task.md"
    if not text_file.exists():
        continue
    words = re.findall(r"[a-z0-9]+", text_file.read_text(errors="replace").lower())
    runs[task.name] = {tuple(words[i : i + 5]) for i in range(max(len(words) - 4, 1))}

pairs = sorted(
    ((len(a & b) / len(a | b), x, y) for (x, a), (y, b) in itertools.combinations(runs.items(), 2)),
    reverse=True,
)
print(f"{len(runs)} instructions, {len(pairs)} pairs")
for share, x, y in pairs[:5]:
    print(f"{share:.2f}  {x}  {y}")
print("pairs at 0.5 or more:", sum(1 for share, *_ in pairs if share >= 0.5))
