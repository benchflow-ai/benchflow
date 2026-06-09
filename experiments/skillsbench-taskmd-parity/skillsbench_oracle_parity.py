"""SkillsBench oracle E2E parity: legacy path vs native task.md path.

For each sampled task, run `bench eval create --agent oracle --sandbox docker`
twice — once on the legacy layout, once on the migrated native task.md layout —
and assert reward_legacy == reward_taskmd (run-parity). Absolute value is a
property of the task's oracle, not of task.md.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

from benchflow._utils.task_authoring import migrate_task_to_task_md

PRIV = Path("/Users/lixiangyi/benchflow/bf-task-standard-private")
SB = Path("/Users/lixiangyi/benchflow/skillsbench/tasks")
WORK = Path("/tmp/sbp_parity")
JOBS = PRIV / "jobs"

SAMPLE = [
    "3d-scan-calc",
    "tictoc-unnecessary-abort-detection",
    "llm-prefix-cache-replay",
    "travel-planning",
    "parallel-tfidf-search",
    "grid-dispatch-operator",
]


def newest_result_after(ts: float) -> dict | None:
    cands = [p for p in JOBS.rglob("result.json") if p.stat().st_mtime >= ts]
    if not cands:
        return None
    return json.loads(max(cands, key=lambda p: p.stat().st_mtime).read_text())


def run_oracle(parent: Path) -> tuple[float | None, str]:
    import time
    ts = time.time() - 1
    proc = subprocess.run(
        ["uv", "run", "bench", "eval", "create", "--tasks-dir", str(parent),
         "--agent", "oracle", "--sandbox", "docker", "--concurrency", "1"],
        cwd=PRIV, capture_output=True, text=True, timeout=1200,
    )
    res = newest_result_after(ts)
    if res is None:
        return None, f"no result (rc={proc.returncode}); tail={proc.stdout[-200:]}"
    reward = (res.get("rewards") or {}).get("reward")
    err = res.get("error_category") or res.get("verifier_error_category") or ""
    return reward, err


def main() -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    rows = []
    for name in SAMPLE:
        src = SB / name
        if not src.is_dir():
            rows.append((name, None, None, "MISSING", ""))
            continue
        # legacy
        lparent = WORK / f"L_{name}"
        shutil.copytree(src, lparent / name)
        r_legacy, e_legacy = run_oracle(lparent)
        # native task.md
        mparent = WORK / f"M_{name}"
        shutil.copytree(src, mparent / name)
        migrate_task_to_task_md(mparent / name, overwrite=True, remove_legacy=True)
        r_taskmd, e_taskmd = run_oracle(mparent)
        parity = (r_legacy == r_taskmd)
        rows.append((name, r_legacy, r_taskmd, "PARITY" if parity else "DIFF",
                     f"{e_legacy}|{e_taskmd}".strip("|")))
        print(f"  {name:42s} legacy={r_legacy}  taskmd={r_taskmd}  "
              f"{'OK' if parity else 'MISMATCH'}  {rows[-1][4]}")

    print("\n=== SkillsBench oracle E2E parity ===")
    par = sum(1 for _, l, m, s, _ in rows if s == "PARITY")
    diff = sum(1 for _, l, m, s, _ in rows if s == "DIFF")
    print(f"tasks={len(rows)}  PARITY(legacy==taskmd)={par}  MISMATCH={diff}")
    print(f"{'task':42s} {'legacy':>8} {'taskmd':>8}  verdict")
    for name, l, m, s, e in rows:
        print(f"{name:42s} {str(l):>8} {str(m):>8}  {s}  {e}")
    sys.exit(0)


if __name__ == "__main__":
    main()
