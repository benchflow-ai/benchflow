"""Write pydantic-lax.jsonl: how BenchFlow 0.8's task.toml model reads values.

Each line is one value (a string, or a JSON number or boolean) with what
0.8's pydantic models (pydantic 2.12.5, pydantic-core 2.41.5, lax mode)
returned for it in an integer field, a float field and a boolean field:
the value, or null when pydantic refused it ("inf", "-inf" and "nan" stand
for those floats). The fields are real 0.8 task.toml fields:

- int: [environment.healthcheck] retries (HealthcheckConfig.retries: int)
- float: [environment.healthcheck] interval_sec (HealthcheckConfig.interval_sec: float)
- bool: [verifier.hardening] cleanup_conftests (VerifierHardeningConfig.cleanup_conftests: bool)

The values are a hand-picked list (spellings found by earlier reviews first),
then random strings over digits, "_", ".", "e", signs, whitespace of several
kinds (tab, newline, NBSP, ideographic space, ...), inf/nan spellings,
leading zeros and non-ASCII digits, then number-shaped strings with
underscores, whitespace and signs in random places, then boolean words in
random case, then JSON numbers. The
random part is seeded, so a rerun writes the same file.

Run it with 0.8's Python (from any directory):

    uv run --project /path/to/benchflow-0.8.0 python -I make_fixture.py

It rewrites pydantic-lax.jsonl next to this script; `--check` compares
instead and exits 1 on a difference.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import pydantic
import pydantic_core
from benchflow.task.config import HealthcheckConfig, VerifierHardeningConfig
from pydantic import ValidationError

SEED = 20261010
OUT = Path(__file__).resolve().parent / "pydantic-lax.jsonl"

# Spellings from D5's first fixture and its review, then more by hand.
PICKED: list[object] = [
    "", "\t3\n", "  ", " 12 ", " true ", "+0", "+1", "-0", "-1",
    "-9223372036854775809", "-inf", "..1", ".5", "0", "00012", "0b1", "0o7",
    "0x10", "1", "1 000", "1,000", "1.", "1.0", "1.5", "1E3", "1_", "1_000",
    "1__0", "1e3", "1e400", "2.0e0", "5.", "9223372036854775807",
    "9223372036854775808", "NaN", "True", "YES", "_1", "f", "false", "inf",
    "infinity", "n", "nan", "no", "off", "on", "t", "true", "true ", "y",
    "yes", "١", "１", 0, 1, 2, -1, 0.0, 1.0, 1.5, 2.0, -0.0, 1e20,
    True, False,
    # The review of D5's fix: underscores are removed from the untrimmed
    # text, and pydantic-core's integer cleanup skips leading zeros.
    "\t48_1", " 0_3", "5_47 ", "0__1", "0-495", "1_.0", "1._0", "1.0_0",
    "+_1", "-_1", "1_000.0", "1_000.5", "0_0", "0_", "_0", "00_1", "0_.0",
    "0._0", "00.00", "-00.0", "+00", "-0_1", "0-0", "0+1", "00-1", "-0-1",
    "--1", "+-1", "-+1", "++1", "+", "-", ".", "_", "e", "1e", "e1", "1e+",
    "1e_3", "1_e3", "1e3_0", "1.5_5", "1_1.5", " 1_1.5", "1_1.5 ",
    "\xa01", "1\xa0", "　1　", " " "2", "​1", "﻿1",
    "\x851", "1\x0b", "\x0c1", "1\r\n", "\n\n7\n\n", "1 _0", "1_ 0",
    "inf_", "in_f", "i_nf", "Infinity", "INFINITY", "+inf", "+nan", "-nan",
    "-NaN", "nan_", "+Infinity", "infinit", "infinityy", "1e308", "1e309",
    "-1e309", "4.9e-324", "2e-324", "0.1e1", "10.0", "10.00_0", "1.00",
    "1.000000000000000000000001", "100_000_000_000_000_000_000",
    "9_223_372_036_854_775_807", "9223372036854775807.0", "0x_10", "0_x10",
    "٣", "৩", "\U0001d7cf", "１２", "1١",
    "TRUE", "FALSE", "Yes", "No", "ON", "OFF", "T", "F", "Y", "N", "tRuE",
    "yes ", " no", "1.0", "01", "00", "1 ", " 0", "on\n", "\toff",
    "true\x00", "ı", "tru", "truee", "yess", "nope", "none", "null",
    -1.0, 3, 0.5, 2.5, 1e-300, 1e15, 1e19, -1e19, 2.0**63, -(2.0**63),
    2.0**63 - 1024, -(2.0**63) + 1024, 9.2e18, 9.21e18, -9.22e18,
    9223372036854775807, -9223372036854775808, 4611686018427387904,
]

DIGITS = "0123456789"
SPACES = [" ", "\t", "\n", "\r", "\x0b", "\x0c", "\x85", "\xa0", " ",
          " ", " ", " ", "　", "​", "﻿"]
OTHER_DIGITS = ["١", "٣", "৩", "０", "５", "\U0001d7cf"]
WORDS = ["inf", "Inf", "INF", "infinity", "Infinity", "iNfInItY", "nan",
         "NaN", "NAN", "nAn", "true", "false", "yes", "no", "on", "off",
         "0x", "0o", "0b", "t", "f", "y", "n"]


def random_text(rng: random.Random) -> str:
    """A short random string, mostly digits and number punctuation."""
    out = []
    for _ in range(rng.randint(0, 8)):
        roll = rng.random()
        if roll < 0.45:
            out.append(rng.choice(DIGITS))
        elif roll < 0.62:
            out.append("_")
        elif roll < 0.70:
            out.append(".")
        elif roll < 0.75:
            out.append(rng.choice("eE"))
        elif roll < 0.82:
            out.append(rng.choice("+-"))
        elif roll < 0.92:
            out.append(rng.choice(SPACES))
        elif roll < 0.95:
            out.append(rng.choice(OTHER_DIGITS))
        else:
            out.append(rng.choice(WORDS))
    return "".join(out)


def sprinkle(rng: random.Random, digits: str) -> str:
    """Digits with underscores put between, around or doubled at random."""
    out = []
    for ch in digits:
        if rng.random() < 0.2:
            out.append(rng.choice(["_", "_", "__"]))
        out.append(ch)
    if rng.random() < 0.1:
        out.append("_")
    return "".join(out)


def number_text(rng: random.Random) -> str:
    """A number-shaped string with odd whitespace, signs, zeros, underscores."""

    def ws() -> str:
        if rng.random() < 0.6:
            return ""
        return "".join(rng.choice(SPACES) for _ in range(rng.randint(1, 2)))

    sign = rng.choice(["", "", "", "", "+", "-", "-", "--", "+-", "-+", "_"])
    if rng.random() < 0.08:
        body = rng.choice(WORDS[:10])
        if rng.random() < 0.3:
            i = rng.randint(0, len(body))
            body = body[:i] + "_" + body[i:]
        return ws() + sign + body + ws()
    zeros = rng.choice(["", "", "", "0", "00", "0_0", "0_", "_0", "000"])
    whole = "".join(rng.choice(DIGITS) for _ in range(rng.randint(0, 5)))
    body = zeros + sprinkle(rng, whole)
    if rng.random() < 0.45:
        frac_digits = rng.choice(
            ["", "0", "00", "000"]
            + ["".join(rng.choice(DIGITS) for _ in range(rng.randint(1, 4)))]
        )
        body += rng.choice([".", ".", "._", "_."]) + sprinkle(rng, frac_digits)
    if rng.random() < 0.2:
        exp_digits = "".join(rng.choice(DIGITS) for _ in range(rng.randint(0, 3)))
        body += rng.choice("eE") + rng.choice(["", "+", "-", "_"]) + sprinkle(rng, exp_digits)
    if rng.random() < 0.04:
        i = rng.randint(0, len(body))
        body = body[:i] + rng.choice(OTHER_DIGITS + ["-", "+", " "]) + body[i:]
    return ws() + sign + body + ws()


def bool_text(rng: random.Random) -> str:
    """A boolean word in random case, sometimes with whitespace around it."""
    word = rng.choice(["1", "0", "t", "f", "y", "n", "on", "no", "off", "yes",
                       "true", "false", "tru", "of", "ye", "1.0", "01", "2"])
    word = "".join(c.upper() if rng.random() < 0.5 else c for c in word)
    if rng.random() < 0.3:
        word = rng.choice(["", rng.choice(SPACES)]) + word + rng.choice(["", rng.choice(SPACES)])
    return word


def json_number(rng: random.Random) -> object:
    """An int or float as a TOML or JSON file holds them."""
    roll = rng.random()
    if roll < 0.3:
        return rng.randint(-(2**63), 2**63 - 1) >> rng.randint(0, 63)
    if roll < 0.5:
        return float(rng.randint(-(10**6), 10**6))
    if roll < 0.7:
        return round(rng.uniform(-10, 10), rng.randint(0, 3))
    if roll < 0.9:
        # Whole floats near the ends of the 64-bit integers.
        return float(rng.choice([1, -1]) * (2**63 - rng.randint(0, 2**12) * 1024))
    return rng.choice([0.0, -0.0, 1.0, -1.0, 0.5, 1e16, 1e300, -1e300])


def read(field: str, model: type, value: object, **extra: object) -> object:
    try:
        return getattr(model.model_validate({**extra, field: value}), field)
    except ValidationError:
        return None


def as_json_float(f: float | None) -> object:
    if f is None or math.isfinite(f):
        return f
    if math.isnan(f):
        return "nan"
    return "inf" if f > 0 else "-inf"


def cases() -> list[object]:
    rng = random.Random(SEED)
    values = list(PICKED)
    values += [random_text(rng) for _ in range(6000)]
    values += [number_text(rng) for _ in range(6000)]
    values += [bool_text(rng) for _ in range(400)]
    values += [json_number(rng) for _ in range(400)]
    seen, out = set(), []
    for v in values:
        key = json.dumps(v)
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


def main() -> int:
    assert pydantic.VERSION == "2.12.5", pydantic.VERSION
    assert pydantic_core.__version__ == "2.41.5", pydantic_core.__version__
    lines = []
    for v in cases():
        i = read("retries", HealthcheckConfig, v, command="true")
        f = read("interval_sec", HealthcheckConfig, v, command="true")
        b = read("cleanup_conftests", VerifierHardeningConfig, v)
        case = {"value": v, "int": i, "float": as_json_float(f), "bool": b}
        lines.append(json.dumps(case) + "\n")
    text = "".join(lines)
    if "--check" in sys.argv[1:]:
        same = OUT.read_text(encoding="utf-8") == text
        print(f"{OUT.name}: {'same' if same else 'DIFFERENT'} ({len(lines)} cases)")
        return 0 if same else 1
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {len(lines)} cases to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
