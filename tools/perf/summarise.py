"""Summarise a perf CSV: medians with 95% bootstrap intervals, paired
differences between two arms, and Wilson intervals for pass counts.

For every metric column and every arm: n, median, and a 95% percentile
bootstrap interval of the median (10,000 resamples, fixed seed). With two or
more arms, each other arm against ``--baseline`` (the first arm by default):
the median of the paired differences (other minus baseline), paired on
``--pair-by`` (default ``round``: the trials of one interleaved round), with
its bootstrap interval. With ``--pass-column``: passes per arm and a Wilson
95% interval. Rows can be filtered (``--where warmup=0``) and split
(``--by command``). The provenance lines beside the CSV are printed first.

    uv run python -m tools.perf.summarise $W/perf/b4d.csv --pass-column reward
    uv run python -m tools.perf.summarise $W/perf/b2.csv --metric seconds \\
        --by command --where warmup=0
    uv run python -m tools.perf.summarise $W/perf/b9.csv --arm-column tool \\
        --metric seconds
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path

RESAMPLES = 10_000
SEED = 20261015
Z95 = 1.959963984540054

# Columns that describe a row rather than measure it.
NOT_METRICS = {
    "arm",
    "commit",
    "round",
    "position",
    "agent",
    "proxy",
    "backend",
    "task_variant",
    "exit_code",
    "error",
    "trial_dir",
    "command",
    "warmup",
    "tool",
    "vertex",
    "reward",
}


def median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def bootstrap_interval(
    values: Sequence[float],
    stat: Callable[[Sequence[float]], float] = median,
    *,
    resamples: int = RESAMPLES,
    seed: int = SEED,
) -> tuple[float, float]:
    """95% percentile bootstrap interval of ``stat`` (fixed seed: repeatable)."""
    if not values:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    n = len(values)
    estimates = sorted(stat(rng.choices(values, k=n)) for _ in range(resamples))
    low = estimates[int(0.025 * (resamples - 1))]
    high = estimates[math.ceil(0.975 * (resamples - 1))]
    return (low, high)


def wilson_interval(passes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a pass rate."""
    if n == 0:
        return (math.nan, math.nan)
    p = passes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_differences(
    rows: Sequence[dict[str, str]],
    *,
    metric: str,
    arm_column: str,
    baseline: str,
    other: str,
    pair_by: Sequence[str],
) -> list[float]:
    """other minus baseline, for every pair key both arms measured."""
    by_key: dict[tuple[str, ...], dict[str, float]] = {}
    for row in rows:
        value = _float(row.get(metric))
        if value is None:
            continue
        key = tuple(row.get(column, "") for column in pair_by)
        by_key.setdefault(key, {})[row[arm_column]] = value
    return [
        arms[other] - arms[baseline]
        for _key, arms in sorted(by_key.items())
        if baseline in arms and other in arms
    ]


def check_one_run(
    rows: Sequence[dict[str, str]], *, arm_column: str, pair_by: Sequence[str]
) -> None:
    """Stop when one arm has two rows under one pair key: the group holds two
    runs (say the oracle and the nop control), so its pass rate would mix them
    and pairing would keep only the later row."""
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for row in rows:
        key = tuple(row.get(column, "") for column in pair_by)
        if (row[arm_column], key) in seen:
            raise SystemExit(
                f"arm {row[arm_column]} has two rows for {', '.join(pair_by)} = "
                f"{', '.join(key)}: split the rows with --by (agent, proxy, ...) "
                "or pair by more columns with --pair-by"
            )
        seen.add((row[arm_column], key))


def _float(cell: str | None) -> float | None:
    if cell is None or cell == "":
        return None
    try:
        value = float(cell)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def metric_columns(
    fieldnames: Sequence[str], rows: Sequence[dict[str, str]]
) -> list[str]:
    return [
        name
        for name in fieldnames
        if name not in NOT_METRICS
        and any(_float(row.get(name)) is not None for row in rows)
    ]


def _fmt(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.3f}"


def summarise(
    rows: Sequence[dict[str, str]],
    *,
    metrics: Sequence[str],
    arm_column: str,
    baseline: str | None,
    pair_by: Sequence[str],
    pass_column: str | None,
    pass_threshold: float,
) -> list[str]:
    """The report's lines for one group of rows."""
    arms = list(dict.fromkeys(row[arm_column] for row in rows))
    if not arms:
        return ["(no rows)"]
    check_one_run(rows, arm_column=arm_column, pair_by=pair_by)
    base = baseline or arms[0]
    lines = []
    for metric in metrics:
        for arm in arms:
            values = [
                v
                for row in rows
                if row[arm_column] == arm and (v := _float(row.get(metric))) is not None
            ]
            if not values:
                lines.append(f"{metric} | {arm} | n=0 (absent)")
                continue
            low, high = bootstrap_interval(values)
            lines.append(
                f"{metric} | {arm} | n={len(values)} | median {_fmt(median(values))} "
                f"[{_fmt(low)}, {_fmt(high)}]"
            )
        for other in arms:
            if other == base:
                continue
            diffs = paired_differences(
                rows,
                metric=metric,
                arm_column=arm_column,
                baseline=base,
                other=other,
                pair_by=pair_by,
            )
            if not diffs:
                continue
            low, high = bootstrap_interval(diffs)
            lines.append(
                f"{metric} | {other} - {base} (paired, n={len(diffs)}) | median "
                f"{_fmt(median(diffs))} [{_fmt(low)}, {_fmt(high)}]"
            )
    if pass_column:
        for arm in arms:
            scored = [
                v
                for row in rows
                if row[arm_column] == arm
                and (v := _float(row.get(pass_column))) is not None
            ]
            all_rows = sum(1 for row in rows if row[arm_column] == arm)
            passes = sum(1 for v in scored if v >= pass_threshold)
            low, high = wilson_interval(passes, all_rows)
            lines.append(
                f"pass ({pass_column} >= {pass_threshold:g}) | {arm} | {passes}/{all_rows} "
                f"[{_fmt(low)}, {_fmt(high)}] (unscored counted as not passed: "
                f"{all_rows - len(scored)})"
            )
    return lines


def _where(spec: str) -> tuple[str, str]:
    column, eq, value = spec.partition("=")
    if not eq or not column:
        raise argparse.ArgumentTypeError(f"--where {spec!r}: expected COLUMN=VALUE")
    return column, value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csv", type=Path)
    parser.add_argument("--metric", action="append", help="default: every number")
    parser.add_argument("--arm-column", default="arm")
    parser.add_argument("--baseline", help="the arm others are paired against")
    parser.add_argument("--pair-by", action="append", help="default: round")
    parser.add_argument("--by", action="append", default=[], help="split rows by")
    parser.add_argument("--where", type=_where, action="append", default=[])
    parser.add_argument("--pass-column")
    parser.add_argument("--pass-threshold", type=float, default=1.0)
    args = parser.parse_args(argv)

    sidecar = args.csv.with_name(args.csv.name + ".provenance.jsonl")
    if sidecar.is_file():
        for line in sidecar.read_text().splitlines():
            if line.strip():
                print("# provenance " + json.dumps(json.loads(line), sort_keys=True))
    with args.csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [
            row
            for row in reader
            if all(row.get(column) == value for column, value in args.where)
        ]
    metrics = args.metric or metric_columns(fieldnames, rows)
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(c, "") for c in args.by), []).append(row)
    print(
        f"# {len(rows)} rows; median with 95% bootstrap interval "
        f"({RESAMPLES} resamples, seed {SEED}); Wilson 95% for passes"
    )
    for key, group in groups.items():
        if args.by:
            print(
                "## " + ", ".join(f"{c}={v}" for c, v in zip(args.by, key, strict=True))
            )
        for line in summarise(
            group,
            metrics=metrics,
            arm_column=args.arm_column,
            baseline=args.baseline,
            pair_by=args.pair_by or ["round"],
            pass_column=args.pass_column,
            pass_threshold=args.pass_threshold,
        ):
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
