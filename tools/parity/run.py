"""Run BenchFlow's oracle on the sampled gate-1 tasks, then each mismatch again alone.

    python -m tools.parity.run --sample sample.tsv --tasks DIR --jobs DIR --out rows.tsv --reruns reruns.tsv
        [-P N] [--bench bench]

Each task runs as gate 1 ran it: ``bench run <task> -a oracle -s docker`` in its own jobs folder, ``P`` at a time
(P=6 on the gate's 32-vCPU VMs; CI uses about half its vCPUs). Then every task whose reward differs from Harbor's
runs again alone (P=1), up to the times ``ledger/flaky.toml`` allows, stopping at the first match: R732B's method,
since a mismatch under load is often the load's. One row per run, the fields ``tools/parity/tally.py`` reads. No
value is a secret: the oracle makes no model call, and no key is set. ``HARBOR_TELEMETRY=off`` is set for each run.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Any, cast

from tools.parity import tally

TIMEOUT = 3600


def _dict(value: object) -> dict[str, Any]:
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def _json(path: Path) -> dict[str, Any]:
    try:
        return _dict(json.loads(path.read_text()))
    except (OSError, ValueError):
        return {}


def _cap(task: Path) -> float | None:
    try:
        cfg = tomllib.loads((task / "task.toml").read_text())
    except (OSError, ValueError):
        return None
    value = (cfg.get("verifier") or {}).get("timeout_sec")
    return float(value) if isinstance(value, (int, float)) else None


def _cpus(task: Path) -> float | None:
    """The CPUs the task asks for (``[environment] cpus``, or ``[sandbox]``'s)."""
    try:
        cfg = tomllib.loads((task / "task.toml").read_text())
    except (OSError, ValueError):
        return None
    for table in ("environment", "sandbox"):
        value = (cfg.get(table) or {}).get("cpus")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _verifier_secs(trial: Path) -> float | None:
    timing = _json(trial / "timing.json")
    secs = [
        float(v)
        for k, v in timing.items()
        if "verifier" in str(k).lower() and isinstance(v, (int, float))
    ]
    return max(secs) if secs else None


def read_trial(key: str, task: Path, jobs: Path, rc: int, secs: float) -> tally.Row:
    results = sorted(jobs.glob("*/*/result.json"), key=lambda p: p.stat().st_mtime)
    if not results:
        log = (
            (jobs / "bench.log").read_text(errors="replace").splitlines()
            if (jobs / "bench.log").exists()
            else []
        )
        said = [
            ln
            for ln in log
            if any(w in ln.lower() for w in ("error", "refus", "failed", "unscored"))
        ]
        return tally.Row(
            key,
            end="NO-RESULT",
            why=(said[-1] if said else f"bench exited {rc}")[:400],
            secs=secs,
        )
    trial = results[-1].parent
    r = _json(results[-1])
    rewards = _dict(r.get("rewards"))
    reward = rewards.get("reward")
    inputs = _json(trial / "integrity" / "verifier-inputs.json")
    provenance = _dict(inputs.get("provenance"))
    walker = _dict(provenance.get("walker"))
    why = str(
        r.get("verifier_error") or r.get("error") or r.get("unscored_reason") or ""
    )
    return tally.Row(
        key=key,
        end="SCORED" if reward is not None else "UNSCORED",
        reward="" if reward is None else str(reward),
        why=" ".join(why.split())[:400],
        error_category=str(
            r.get("error_category") or r.get("verifier_error_category") or ""
        ),
        reward_changed=str(inputs.get("reward_changed", "")),
        provenance_outcome=str(provenance.get("outcome") or ""),
        walker=str(walker.get("engine") or ""),
        verifier_secs=_verifier_secs(trial),
        verifier_cap=_cap(task),
        secs=secs,
    )


def run_one(bench: str, key: str, task: Path, jobs: Path) -> tally.Row:
    # A task that asks for more CPUs than this runner has is refused by Docker
    # ("range of CPUs is from 0.01 to N"): that is the runner's size, not
    # BenchFlow's, so it is not run and the tally says so (0.10.0's CI run on
    # otel-bench/ruby-microservices, cpus = 4 on a 2-vCPU runner).
    asked, have = _cpus(task), os.cpu_count() or 1
    if asked is not None and asked > have:
        return tally.Row(
            key,
            end=tally.RUNNER_CAPACITY,
            why=f"the task asks for {asked:g} CPUs; this runner has {have}",
            secs=0.0,
        )
    if jobs.exists():
        shutil.rmtree(jobs)
    jobs.mkdir(parents=True)
    env = {**os.environ, "HARBOR_TELEMETRY": "off", "COLUMNS": "200"}
    for name in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "DEEPSEEK_API_KEY",
    ):
        env.pop(name, None)
    start = time.monotonic()
    with open(jobs / "bench.log", "w") as log:
        try:
            rc = subprocess.run(
                [
                    bench,
                    "run",
                    str(task),
                    "-a",
                    "oracle",
                    "-s",
                    "docker",
                    "--jobs-dir",
                    str(jobs),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=TIMEOUT,
            ).returncode
        except subprocess.TimeoutExpired:
            rc = 124
    return read_trial(key, task, jobs, rc, round(time.monotonic() - start, 1))


def _write(path: Path, rows: list[tally.Row]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(tally.ROW_FIELDS)
        for r in rows:
            w.writerow(
                [
                    "" if getattr(r, f) is None else getattr(r, f)
                    for f in tally.ROW_FIELDS
                ]
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reruns", type=Path, required=True)
    ap.add_argument("-P", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--bench", default=shutil.which("bench") or "bench")
    args = ap.parse_args(argv)
    with open(args.sample, newline="") as fh:
        keys = [r["key"] for r in csv.DictReader(fh, delimiter="\t")]
    ref = tally.load_reference()
    ledger = tally.load_ledger()

    def one(key: str, jobs: Path) -> tally.Row:
        row = run_one(args.bench, key, args.tasks / key, jobs)
        print(f"{key}\t{row.end}\t{row.reward}\t{row.secs}s", flush=True)
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.P) as pool:
        rows = list(pool.map(lambda k: one(k, args.jobs / k.replace("/", "__")), keys))
    _write(args.out, rows)
    reruns: list[tally.Row] = []
    for row in rows:
        harbor = ref.get(row.key)
        if harbor is None or row.value is None or abs(row.value - harbor) < 1e-9:
            continue
        for attempt in range(1, tally.reruns_allowed(row.key, ledger) + 1):
            again = one(
                row.key, args.jobs / f"{row.key.replace('/', '__')}.alone{attempt}"
            )
            reruns.append(again)
            if again.value is not None and abs(again.value - harbor) < 1e-9:
                break
    _write(args.reruns, reruns)
    return 0


if __name__ == "__main__":
    sys.exit(main())
