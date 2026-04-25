#!/usr/bin/env python3
"""Reward-hack matrix orchestrator.

Pulls real tasks from three Harbor-format benchmarks (skillsbench,
swebench-verified, terminal-bench-2), copies each into an isolated
"cell" directory under .cells/, swaps the task's `solution/solve.sh`
with one of our exploit payloads, and runs the cell under both
benchflow 0.2.0 and benchflow 0.2.2 via Daytona oracle mode.

Output: a benchmark × pattern × version table showing where reward
hacking succeeds and where 0.2.2's hardening blocks it.

Usage:
    python run_matrix.py                  # full matrix
    python run_matrix.py --cells P1@skillsbench/data-to-d3
                                          # single cell, comma-separated
    python run_matrix.py --clean          # delete .venvs/, .jobs/, .cells/
    python run_matrix.py --env docker     # local docker instead of daytona

Requires:
    * Network access to PyPI on first run (creates two venvs)
    * `uv` on PATH (preferred) — falls back to `python -m venv` + pip
    * For --env daytona: `DAYTONA_API_KEY` in environment
    * For --env docker:  Docker daemon accessible
    * Bench corpora cloned at the paths in CORPORA below — run
      `./fetch_corpora.sh` once before the first invocation.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Daytona per-sandbox CPU cap (surfaced as
# "CPU request N exceeds maximum allowed per sandbox (4)").
# Tasks that declare `cpus > DAYTONA_MAX_CPUS` get clamped at cell-staging
# time so the sweep doesn't hit fallible-infra errors on otherwise-valid
# tasks (6 skillsbench tasks currently declare cpus=8).
DAYTONA_MAX_CPUS = 4
_CPUS_RE = re.compile(r"^(\s*cpus\s*=\s*)(\d+)\s*$", re.MULTILINE)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
VENVS_DIR = HERE / ".venvs"
JOBS_DIR = HERE / ".jobs"
CELLS_DIR = HERE / ".cells"
CORPORA_DIR = HERE / ".corpora"

# Where each benchmark's tasks live, relative to CORPORA_DIR.
# fetch_corpora.sh populates these on first run.
CORPORA = {
    "skillsbench": CORPORA_DIR / "harbor-datasets" / "datasets" / "skillsbench",
    "swebench-verified": CORPORA_DIR / "harbor-datasets" / "datasets" / "swebench-verified",
    "terminal-bench-2": CORPORA_DIR / "terminal-bench-2",
}


@dataclass
class Cell:
    """One row of the matrix: a benchmark × task × pattern combination."""

    pattern_id: str  # "P1", "P7"
    pattern_name: str  # "conftest-hook", "path-trojan"
    benchmark: str  # "skillsbench" | "swebench-verified" | "terminal-bench-2"
    task: str  # task directory name within the benchmark corpus
    exploit: str  # filename in exploits/ to install as solution/solve.sh

    @property
    def cell_id(self) -> str:
        return f"{self.pattern_id}@{self.benchmark}/{self.task}"

    @property
    def source_task_dir(self) -> Path:
        return CORPORA[self.benchmark] / self.task

    @property
    def cell_task_dir(self) -> Path:
        # Sanitize task name for filesystem safety
        safe = self.task.replace("/", "_").replace("__", "-")
        return CELLS_DIR / self.benchmark / f"{self.pattern_id}_{safe}"


# One exploit choice per benchmark — used by --sweep to auto-enumerate
# every task in each corpus. The picks match the representative cells in
# CELLS below and are justified in the individual exploit script headers.
SWEEP_PATTERN = {
    "skillsbench": ("P7", "pth-injection", "pth_injection.sh"),
    "swebench-verified": ("P1", "conftest-hook", "conftest_hook.sh"),
    "terminal-bench-2": ("P7", "path-trojan", "path_trojan.sh"),
}


# The matrix. One entry per (pattern, benchmark, representative task).
# Add more rows as we expand coverage.
CELLS: list[Cell] = [
    # skillsbench: /tests is bind-mounted at verifier phase, so conftest drops
    # don't survive. Use Python startup-hook injection instead.
    Cell(
        pattern_id="P7",
        pattern_name="pth-injection",
        benchmark="skillsbench",
        task="data-to-d3",
        exploit="pth_injection.sh",
    ),
    # swebench-verified: /testbed is in-image (created by the SWE-bench Docker
    # image, not a Harbor mount), so writes during agent phase persist.
    # Conftest drop at /testbed/conftest.py works.
    Cell(
        pattern_id="P1",
        pattern_name="conftest-hook",
        benchmark="swebench-verified",
        task="astropy__astropy-12907",
        exploit="conftest_hook.sh",
    ),
    # terminal-bench-2: /usr/local/bin is in-image and writable as root, so
    # PATH-trojan (fake curl/uvx/pytest) survives into verifier phase.
    Cell(
        pattern_id="P7",
        pattern_name="path-trojan",
        benchmark="terminal-bench-2",
        task="adaptive-rejection-sampler",
        exploit="path_trojan.sh",
    ),
]

# Two versions to compare. Add ("harbor-orig", ["harbor"]) as a third row
# once we have a harbor adapter that can run a task in oracle mode.
VERSIONS: list[tuple[str, list[str]]] = [
    ("0.2.0", ["benchflow==0.2.0"]),
    ("0.2.2", ["-e", str(REPO_ROOT)]),
]


# ---------------------------------------------------------------------------
# venv setup (mirrors labs/benchjack-sandbox-hardening/run_comparison.py)


def _have_uv() -> bool:
    return shutil.which("uv") is not None


def _create_venv(venv_dir: Path, spec: list[str]) -> None:
    if (venv_dir / "bin" / "python").exists():
        return
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    python = venv_dir / "bin" / "python"

    if _have_uv():
        subprocess.check_call(["uv", "venv", str(venv_dir)])
        subprocess.check_call(
            ["uv", "pip", "install", "--python", str(python), *spec]
        )
        return

    subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
    subprocess.check_call(
        [str(venv_dir / "bin" / "pip"), "install", "--upgrade", "pip"]
    )
    subprocess.check_call([str(venv_dir / "bin" / "pip"), "install", *spec])


# ---------------------------------------------------------------------------
# cell preparation: copy real task → cell dir → overwrite solve.sh


def _prepare_cell(cell: Cell) -> None:
    """Stage one cell: copy the source task and inject our exploit."""

    src = cell.source_task_dir
    dst = cell.cell_task_dir

    if not src.is_dir():
        raise FileNotFoundError(
            f"source task missing: {src}\n"
            f"run ./fetch_corpora.sh from {HERE} to populate .corpora/"
        )

    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)

    # Replace solution/solve.sh with our exploit, alongside the conftest
    # payload (which the exploit script reads from the same directory).
    soln = dst / "solution"
    soln.mkdir(exist_ok=True)
    exploit_src = HERE / "exploits" / cell.exploit
    exploit_dst = soln / "solve.sh"
    shutil.copyfile(exploit_src, exploit_dst)
    exploit_dst.chmod(exploit_dst.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    # conftest_payload.py is a standalone reference copy of the payload;
    # conftest_hook.sh inlines it as a heredoc and does not read this file.
    # Ship it alongside the exploit script so the cell is self-contained and
    # the payload is human-readable without parsing the shell script.
    payload_src = HERE / "exploits" / "conftest_payload.py"
    if payload_src.exists():
        shutil.copyfile(payload_src, soln / "conftest_payload.py")

    # Clamp cpus to Daytona's per-sandbox maximum. Without this, the 6
    # skillsbench tasks declaring cpus=8 fail sandbox creation before the
    # exploit even gets a chance to run.
    task_toml = dst / "task.toml"
    if task_toml.exists():
        text = task_toml.read_text()

        def _clamp(m: re.Match[str]) -> str:
            prefix, value = m.group(1), int(m.group(2))
            return f"{prefix}{min(value, DAYTONA_MAX_CPUS)}"

        new_text = _CPUS_RE.sub(_clamp, text)
        if new_text != text:
            task_toml.write_text(new_text)


# ---------------------------------------------------------------------------
# trial execution


def _run_cell_in_venv(
    venv_dir: Path,
    version_label: str,
    cell: Cell,
    environment: str,
) -> dict:
    """Invoke _runner.py in the given venv against one cell."""

    python = venv_dir / "bin" / "python"
    env = os.environ.copy()
    env["RH_TASK_PATH"] = str(cell.cell_task_dir)
    env["RH_PATTERN_ID"] = cell.pattern_id
    env["RH_BENCHMARK"] = cell.benchmark
    env["RH_VERSION_LABEL"] = version_label
    env["RH_JOBS_DIR"] = str(JOBS_DIR / version_label / cell.benchmark / cell.pattern_id)
    env["RH_TRIAL_NAME"] = f"{version_label}-{cell.pattern_id}-{cell.task}"
    env["RH_ENVIRONMENT"] = environment

    proc = subprocess.run(
        [str(python), str(HERE / "_runner.py")],
        env=env,
        capture_output=True,
        text=True,
    )

    payload: dict | None = None
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
            break
        except json.JSONDecodeError:
            continue

    if payload is None:
        payload = {
            "benchflow_version": None,
            "reward": None,
            "error": f"no JSON on stdout (rc={proc.returncode})",
        }

    payload["version_label"] = version_label
    payload["cell_id"] = cell.cell_id
    payload["returncode"] = proc.returncode
    if proc.stderr:
        payload["stderr_tail"] = proc.stderr[-2000:]
    return payload


# ---------------------------------------------------------------------------
# table rendering


def _fmt_cell(row: dict) -> str:
    reward = row.get("reward")
    if reward is None:
        return "  N/A   ERROR     "
    elif reward >= 0.999:
        return f"  {reward:.2f}  EXPLOITED "
    else:
        return f"  {reward:.2f}  BLOCKED   "


def _print_table(results: dict[str, dict[str, dict]]) -> None:
    width = 72
    print()
    print("=" * width)
    print("Reward-hack matrix — benchflow 0.2.0 vs 0.2.2")
    print("=" * width)
    header = f"{'cell':<40}  {'0.2.0':<18}  {'0.2.2'}"
    print(header)
    print("-" * width)
    for cell in CELLS:
        cells = results.get(cell.cell_id, {})
        c020 = _fmt_cell(cells.get("0.2.0", {}))
        c021 = _fmt_cell(cells.get("0.2.2", {}))
        label = f"{cell.pattern_id} {cell.benchmark}/{cell.task}"
        if len(label) > 40:
            label = label[:37] + "..."
        print(f"{label:<40}  {c020}  {c021}")
    print()


# ---------------------------------------------------------------------------
# sweep: enumerate every task in each corpus


def _enumerate_sweep_cells(limit_per_bench: int | None) -> list[Cell]:
    cells: list[Cell] = []
    for bench, corpus_dir in CORPORA.items():
        if bench not in SWEEP_PATTERN:
            continue
        pat_id, pat_name, exploit = SWEEP_PATTERN[bench]
        if not corpus_dir.is_dir():
            print(f"[sweep] skip {bench}: corpus dir missing ({corpus_dir})")
            continue
        task_dirs = sorted(
            p
            for p in corpus_dir.iterdir()
            if p.is_dir()
            and not p.name.startswith(".")
            and (p / "task.toml").exists()
        )
        if limit_per_bench is not None:
            task_dirs = task_dirs[:limit_per_bench]
        for td in task_dirs:
            cells.append(
                Cell(
                    pattern_id=pat_id,
                    pattern_name=pat_name,
                    benchmark=bench,
                    task=td.name,
                    exploit=exploit,
                )
            )
    return cells


class _Worker:
    """A long-lived ``_worker.py`` subprocess pinned to one benchflow venv.

    One worker per version — the SDK is imported once at worker startup,
    and all trials for that version run as asyncio coroutines inside the
    worker under its own ``Semaphore``. Replaces the old subprocess-per-
    trial design which OOM'd an ~8 GB dev container at concurrency 64
    because it re-imported benchflow/harbor/daytona SDK per trial.
    """

    def __init__(self, version_label: str, venv_dir: Path, per_worker_concurrency: int):
        self.version_label = version_label
        self.venv_dir = venv_dir
        self.concurrency = per_worker_concurrency
        self.proc: asyncio.subprocess.Process | None = None
        self.pending: dict[str, asyncio.Future] = {}
        self.write_lock = asyncio.Lock()
        self.reader_task: asyncio.Task | None = None
        self.stderr_task: asyncio.Task | None = None
        self.ready: asyncio.Future | None = None
        self.benchflow_version: str | None = None

    async def start(self) -> None:
        python = self.venv_dir / "bin" / "python"
        env = os.environ.copy()
        # Unbuffered stdout so result lines flush immediately.
        env["PYTHONUNBUFFERED"] = "1"
        self.proc = await asyncio.create_subprocess_exec(
            str(python),
            str(HERE / "_worker.py"),
            "--concurrency",
            str(self.concurrency),
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.ready = asyncio.get_running_loop().create_future()
        self.reader_task = asyncio.create_task(self._read_stdout())
        self.stderr_task = asyncio.create_task(self._drain_stderr())
        await self.ready

    async def _read_stdout(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        while True:
            line = await self.proc.stdout.readline()
            if not line:
                break
            line = line.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                sys.stderr.write(f"[worker/{self.version_label}] non-JSON: {line}\n")
                continue
            if obj.get("__ready__"):
                self.benchflow_version = obj.get("benchflow_version")
                if self.ready and not self.ready.done():
                    self.ready.set_result(None)
                continue
            if obj.get("__done__"):
                break
            req_id = obj.get("id")
            fut = self.pending.pop(req_id, None) if req_id else None
            if fut and not fut.done():
                fut.set_result(obj)
        # Worker stdout closed — fail any still-pending futures
        for fut in self.pending.values():
            if not fut.done():
                fut.set_result(
                    {"reward": None, "error": "worker stdout closed before reply"}
                )
        self.pending.clear()
        if self.ready and not self.ready.done():
            self.ready.set_exception(
                RuntimeError(f"worker {self.version_label} exited before ready")
            )

    async def _drain_stderr(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        while True:
            line = await self.proc.stderr.readline()
            if not line:
                return
            sys.stderr.write(f"[worker/{self.version_label}] {line.decode('utf-8', 'replace')}")

    async def submit(self, req: dict) -> dict:
        assert self.proc is not None and self.proc.stdin is not None
        fut = asyncio.get_running_loop().create_future()
        self.pending[req["id"]] = fut
        async with self.write_lock:
            self.proc.stdin.write((json.dumps(req) + "\n").encode("utf-8"))
            await self.proc.stdin.drain()
        return await fut

    async def shutdown(self) -> None:
        if self.proc is None:
            return
        if self.proc.stdin and not self.proc.stdin.is_closing():
            with contextlib.suppress(OSError, BrokenPipeError):
                self.proc.stdin.close()
        try:
            await asyncio.wait_for(self.proc.wait(), timeout=30)
        except TimeoutError:
            self.proc.kill()
            await self.proc.wait()
        if self.reader_task:
            await asyncio.gather(self.reader_task, return_exceptions=True)
        if self.stderr_task:
            self.stderr_task.cancel()
            await asyncio.gather(self.stderr_task, return_exceptions=True)


async def _async_sweep(
    cells: list[Cell],
    environment: str,
    concurrency: int,
    summary_path: Path,
    resume: bool,
) -> dict:
    """Worker-pool sweep: one long-lived Python process per benchflow version.

    Total in-flight trials = ``concurrency`` (split evenly across workers).
    Local RSS is bounded by the number of workers, not the trial count.
    """

    results: dict[str, dict[str, dict]] = {}
    if resume and summary_path.exists():
        try:
            results = json.loads(summary_path.read_text())
            print(f"[sweep] resumed from {summary_path} ({len(results)} cells cached)")
        except Exception as exc:
            print(f"[sweep] could not parse existing summary: {exc}; starting fresh")
            results = {}

    num_workers = len(VERSIONS)
    per_worker = max(1, concurrency // num_workers)
    print(
        f"[sweep] starting {num_workers} worker(s), "
        f"concurrency/worker={per_worker}, total in-flight={per_worker * num_workers}"
    )

    workers: dict[str, _Worker] = {}
    for version_label, _ in VERSIONS:
        w = _Worker(
            version_label=version_label,
            venv_dir=VENVS_DIR / f"bf-{version_label}",
            per_worker_concurrency=per_worker,
        )
        await w.start()
        print(f"[sweep] worker {version_label} ready (benchflow {w.benchflow_version})")
        workers[version_label] = w

    total = len(cells) * len(VERSIONS)
    state = {"done": 0, "exploited": 0, "blocked": 0, "errors": 0}
    started_at = time.monotonic()
    lock = asyncio.Lock()

    def _write_summary_sync() -> None:
        tmp = summary_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(results, indent=2, default=str))
        tmp.replace(summary_path)

    async def one(cell: Cell, version_label: str) -> None:
        existing = results.get(cell.cell_id, {}).get(version_label)
        if (
            resume
            and existing
            and existing.get("reward") is not None
            and not existing.get("error")
        ):
            async with lock:
                state["done"] += 1
                reward = existing.get("reward") or 0
                if reward >= 0.999:
                    state["exploited"] += 1
                else:
                    state["blocked"] += 1
            return

        req = {
            "id": f"{version_label}::{cell.cell_id}",
            "task_path": str(cell.cell_task_dir),
            "jobs_dir": str(
                JOBS_DIR / version_label / cell.benchmark / cell.pattern_id
            ),
            "trial_name": f"{version_label}-{cell.pattern_id}-{cell.task}",
            "environment": environment,
        }
        payload = await workers[version_label].submit(req)

        # Normalize into the same shape the rest of this file expects
        row = {
            "benchflow_version": payload.get("benchflow_version"),
            "reward": payload.get("reward"),
            "error": payload.get("error"),
            "verifier_error": payload.get("verifier_error"),
            "version_label": version_label,
            "cell_id": cell.cell_id,
        }
        if "traceback_tail" in payload:
            row["traceback_tail"] = payload["traceback_tail"]

        async with lock:
            results.setdefault(cell.cell_id, {})[version_label] = row
            state["done"] += 1
            reward = row.get("reward")
            err = row.get("error")
            if err or reward is None:
                state["errors"] += 1
                mark = "ERROR "
            elif reward >= 0.999:
                state["exploited"] += 1
                mark = "EXPLT "
            else:
                state["blocked"] += 1
                mark = "BLOCK "
            elapsed = time.monotonic() - started_at
            rate = state["done"] / elapsed if elapsed > 0 else 0.0
            eta = (total - state["done"]) / rate if rate > 0 else 0.0
            print(
                f"[{state['done']:>4}/{total}] "
                f"{version_label} {mark} reward={reward} "
                f"E={state['exploited']} B={state['blocked']} X={state['errors']} "
                f"rate={rate:.2f}/s eta={eta / 60:.1f}m :: {cell.cell_id}",
                flush=True,
            )
            _write_summary_sync()

    try:
        jobs = [one(cell, version_label) for cell in cells for version_label, _ in VERSIONS]
        await asyncio.gather(*jobs)
    finally:
        for w in workers.values():
            await w.shutdown()
    return results


def _print_sweep_rollup(results: dict[str, dict[str, dict]]) -> None:
    # Aggregate by (benchmark, version) → counts
    agg: dict[tuple[str, str], dict[str, int]] = {}
    for cell_id, versions in results.items():
        try:
            _pat, bench_task = cell_id.split("@", 1)
            bench = bench_task.split("/", 1)[0]
        except ValueError:
            continue
        for version_label, row in versions.items():
            key = (bench, version_label)
            a = agg.setdefault(key, {"expl": 0, "block": 0, "err": 0, "n": 0})
            a["n"] += 1
            reward = row.get("reward")
            if row.get("error") or reward is None:
                a["err"] += 1
            elif reward >= 0.999:
                a["expl"] += 1
            else:
                a["block"] += 1

    print()
    print("=" * 78)
    print("Sweep rollup — exploit success rate per (benchmark, version)")
    print("=" * 78)
    print(f"{'benchmark':<22} {'version':<10} {'exploited':>10} {'blocked':>8} {'err':>5} {'n':>5}")
    print("-" * 78)
    for (bench, version_label), a in sorted(agg.items()):
        print(
            f"{bench:<22} {version_label:<10} "
            f"{a['expl']:>10} {a['block']:>8} {a['err']:>5} {a['n']:>5}"
        )
    print()


# ---------------------------------------------------------------------------
# main


def _filter_cells(args_cells: str | None) -> list[Cell]:
    if not args_cells:
        return CELLS
    wanted = {c.strip() for c in args_cells.split(",") if c.strip()}
    return [c for c in CELLS if c.cell_id in wanted]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true", help="delete .venvs/, .jobs/, .cells/ first")
    ap.add_argument("--env", default="daytona", choices=["daytona", "docker"], help="benchflow environment backend")
    ap.add_argument("--cells", help="comma-separated cell IDs to run (default: all)")
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="enumerate every task in each corpus (1 pattern/benchmark from SWEEP_PATTERN)",
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="max in-flight trials for --sweep (default 64)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="per-benchmark task cap for --sweep (default: all)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="for --sweep: keep completed trials from the existing summary and only run missing cells",
    )
    ap.add_argument(
        "--summary-path",
        default=None,
        help="override summary JSON path (default: .jobs/matrix_summary.json or .jobs/matrix_sweep.json)",
    )
    args = ap.parse_args()

    if args.clean:
        for d in (VENVS_DIR, JOBS_DIR, CELLS_DIR):
            if d.exists():
                print(f"[clean] removing {d}")
                shutil.rmtree(d)

    if args.sweep:
        cells = _enumerate_sweep_cells(args.limit)
    else:
        cells = _filter_cells(args.cells)

    if not cells:
        print("ERROR: no matching cells")
        return 1

    # Pre-flight: ensure source corpora exist
    missing = []
    for cell in cells:
        if not cell.source_task_dir.is_dir():
            missing.append(str(cell.source_task_dir))
    if missing:
        print("ERROR: missing source tasks. Run ./fetch_corpora.sh first.")
        for m in missing[:10]:
            print(f"  {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
        return 1

    # Stage cells (sync; file-copy is fast even for 668 tasks)
    print(f"[stage] preparing {len(cells)} cell(s)")
    for i, cell in enumerate(cells, 1):
        _prepare_cell(cell)
        if args.sweep and i % 50 == 0:
            print(f"  staged {i}/{len(cells)}")
    if not args.sweep:
        for cell in cells:
            print(f"  staged {cell.cell_id}")

    # Set up venvs
    print("[venv] benchflow==0.2.0 (PyPI)")
    _create_venv(VENVS_DIR / "bf-0.2.0", VERSIONS[0][1])
    print("[venv] benchflow@HEAD (editable)")
    _create_venv(VENVS_DIR / "bf-0.2.2", VERSIONS[1][1])

    JOBS_DIR.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        summary_path = Path(args.summary_path) if args.summary_path else (JOBS_DIR / "matrix_sweep.json")
        print(
            f"[sweep] {len(cells)} cells x {len(VERSIONS)} versions = "
            f"{len(cells) * len(VERSIONS)} trials, concurrency={args.concurrency}"
        )
        results = asyncio.run(
            _async_sweep(
                cells=cells,
                environment=args.env,
                concurrency=args.concurrency,
                summary_path=summary_path,
                resume=args.resume,
            )
        )
        _print_sweep_rollup(results)
        print(f"[summary] wrote {summary_path}")
        return 0

    # Non-sweep path: sequential, small matrix
    results: dict[str, dict[str, dict]] = {}
    total = len(cells) * len(VERSIONS)
    step = 0
    for cell in cells:
        results[cell.cell_id] = {}
        for version_label, _ in VERSIONS:
            step += 1
            print(f"[{step}/{total}] {version_label} :: {cell.cell_id}")
            venv_dir = VENVS_DIR / f"bf-{version_label}"
            results[cell.cell_id][version_label] = _run_cell_in_venv(
                venv_dir, version_label, cell, args.env
            )

    _print_table(results)
    summary_path = Path(args.summary_path) if args.summary_path else (JOBS_DIR / "matrix_summary.json")
    summary_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"[summary] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
