"""Write python-pass-at-k.json: BenchFlow 0.8's pass@k and Wilson interval.

- "pass_at_k": benchflow.pass_at_k.pass_at_k(n, c, k) for every n up to 40,
  every c <= n and every k <= n + 1: the value, null when n < k, or
  "error" when 0.8 refuses the arguments (k = 0).
- "wilson": benchflow.pass_at_k.wilson_interval(k, n, z) for every k <= n
  up to 40 (null for n = 0), the function behind the solve_rate_interval
  in 0.8's summary.json, called with "wilson_z", the z bench's
  summary.json score_interval uses (the 97.5% normal quantile). 0.8's
  own default is z = 1.96; with it the endpoints move by at most the
  "z_1_96_max_shift" this file records.

Run it with 0.8's Python (from any directory):

    uv run --project /path/to/benchflow-0.8.0 python -I make_pass_at_k.py

It rewrites python-pass-at-k.json next to this script; `--check` compares
instead and exits 1 on a difference.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from benchflow.pass_at_k import pass_at_k, wilson_interval

OUT = Path(__file__).resolve().parent / "python-pass-at-k.json"
Z = 1.959963984540054
MAX_N = 40


def main() -> int:
    pk = []
    for n in range(MAX_N + 1):
        for c in range(n + 1):
            for k in range(n + 2):
                try:
                    value = pass_at_k(n, c, k)
                except ValueError:
                    value = "error"
                pk.append({"n": n, "c": c, "k": k, "pass_at_k": value})
    wilson = []
    shift = 0.0
    for n in range(MAX_N + 1):
        for k in range(n + 1):
            if n == 0:
                wilson.append({"k": k, "n": n, "lo": None, "hi": None})
                continue
            lo, hi = wilson_interval(k, n, Z)
            lo96, hi96 = wilson_interval(k, n)
            shift = max(shift, abs(lo - lo96), abs(hi - hi96))
            wilson.append({"k": k, "n": n, "lo": lo, "hi": hi})
    doc = {
        "wilson_z": Z,
        "z_1_96_max_shift": shift,
        "pass_at_k": pk,
        "wilson": wilson,
    }
    text = json.dumps(doc) + "\n"
    if "--check" in sys.argv[1:]:
        same = OUT.read_text(encoding="utf-8") == text
        print(f"{OUT.name}: {'same' if same else 'DIFFERENT'}")
        return 0 if same else 1
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {len(pk)} pass@k and {len(wilson)} Wilson cases to {OUT}; z=1.96 shifts by at most {shift:.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
