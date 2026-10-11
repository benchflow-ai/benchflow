"""Write python-float-repr.jsonl: floats with the repr() Python gives them.

BenchFlow 0.8 writes floats with json.dumps, which uses repr(float):
the shortest digits that read back as the same float, and when two
candidates are equally short, the one nearer the exact value, ties going
to the even digit (755133721037486.25 is 755133721037486.2). Each line is
a float's IEEE-754 bits (as an unsigned integer) and its repr.

The floats are a hand-picked list, then seeded groups:

- random bit patterns (finite only),
- decimals rounded to 0-12 places,
- a digit string times a power of ten, over the whole finite range,
- neighbours of the points where repr switches notation (1e16, 1e-4),
- ties: floats whose exact value lies halfway between two equally short
  digit strings that both read back as it, half of them rounding down
  (where breaking ties upward gives the wrong digits) and half up.

Run it with 0.8's Python (from any directory):

    uv run --project /path/to/benchflow-0.8.0 python -I make_float_repr.py

It rewrites python-float-repr.jsonl next to this script; `--check` compares
instead and exits 1 on a difference.
"""

from __future__ import annotations

import json
import math
import random
import struct
import sys
from decimal import ROUND_HALF_DOWN, ROUND_HALF_EVEN, ROUND_HALF_UP, Context, Decimal
from pathlib import Path

SEED = 20261011
OUT = Path(__file__).resolve().parent / "python-float-repr.jsonl"

PICKED = [
    0.0, -0.0, 1.0, -1.0, 0.1, 1e16, 1e15, 9999999999999998.0, 1e-4, 1e-5,
    1.2345678901234568e17, 5e-324, 1.7976931348623157e308, 2.5, 1 / 3, 100.0,
    1e22, 1e21, 0.5, 0.1 + 0.2, 755133721037486.25, 755133721037486.75,
    2.0**53, 2.0**53 + 2, 2.2250738585072014e-308, 2.225073858507201e-308,
]


def bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def from_bits(b: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", b))[0]


def significant_digits(r: str) -> int:
    mantissa = r.lstrip("-").split("e")[0].replace(".", "").lstrip("0")
    return len(mantissa.rstrip("0")) or 1


def tie(x: float) -> str | None:
    """'down' or 'up' when x lies exactly halfway between two equally
    short digit strings that both read back as x (repr takes the even one)."""
    d = significant_digits(repr(x))
    exact = Decimal(x)
    up = Context(prec=d, rounding=ROUND_HALF_UP).plus(exact)
    down = Context(prec=d, rounding=ROUND_HALF_DOWN).plus(exact)
    if up == down or float(up) != x or float(down) != x:
        return None
    even = Context(prec=d, rounding=ROUND_HALF_EVEN).plus(exact)
    return "down" if even == down else "up"


def ties(rng: random.Random, want: int) -> list[float]:
    """Floats m * 2**(e-52) with m given t trailing zero bits, for binades e
    and t where ties occur; `want` of them, half rounding each way."""
    found = {"down": [], "up": []}
    seen = set()
    for _ in range(400_000):
        if min(len(v) for v in found.values()) >= want // 2:
            break
        e = rng.randrange(-30, 53)
        t = rng.randrange(0, 53)
        m = (rng.getrandbits(52 - t) << t) | (1 << 52)
        x = math.ldexp(m, e - 52)
        if rng.random() < 0.5:
            x = -x
        if x in seen:
            continue
        seen.add(x)
        side = tie(x)
        if side and len(found[side]) < want // 2:
            found[side].append(x)
    return found["down"] + found["up"]


def main() -> int:
    rng = random.Random(SEED)
    floats: list[float] = list(PICKED)
    while len(floats) < 2_000 + len(PICKED):
        x = from_bits(rng.getrandbits(64))
        if math.isfinite(x):
            floats.append(x)
    for _ in range(1_500):
        floats.append(round(rng.uniform(-1e6, 1e6) * 10.0 ** rng.randrange(-8, 9), rng.randrange(13)))
    wide = 0
    while wide < 1_500:
        digits = rng.randrange(1, 10 ** rng.randrange(1, 18))
        x = float(f"{digits}e{rng.randrange(-340, 300)}")
        if math.isfinite(x):
            floats.append(x)
            wide += 1
    for edge in (1e16, 1e-4, 1e-5, 1e15, 1e17):
        x = edge
        for _ in range(8):
            x = math.nextafter(x, 0.0)
        for _ in range(17):
            floats.append(x)
            x = math.nextafter(x, math.inf)
    tie_floats = ties(rng, 4_000)
    floats += tie_floats
    lines = [json.dumps({"bits": bits(x), "repr": repr(x)}) for x in floats]
    text = "\n".join(lines) + "\n"
    down = sum(tie(x) == "down" for x in tie_floats)
    summary = f"{len(floats)} floats, {len(tie_floats)} ties ({down} rounding down)"
    if "--check" in sys.argv[1:]:
        same = OUT.read_text(encoding="utf-8") == text
        print(f"{OUT.name}: {'same' if same else 'DIFFERENT'} ({summary})")
        return 0 if same else 1
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {summary} to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
