"""SkillsBench task.md conversion-parity sweep.

For every SkillsBench task, migrate legacy (task.toml+instruction.md+solution/+tests/)
-> task.md -> export back to split, and assert the supported compatibility surface
survives: canonical TaskConfig, normalized prompt, and environment/solution/tests
file-map hashes. No execution (deterministic, no Docker).
"""
import sys
from pathlib import Path
from benchflow.task.export import build_harbor_roundtrip_conformance_report

tasks_root = Path("/Users/lixiangyi/benchflow/skillsbench/tasks")
dirs = sorted(d for d in tasks_root.iterdir() if d.is_dir())

results = []
for d in dirs:
    try:
        rep = build_harbor_roundtrip_conformance_report(d)
        mm = list(getattr(rep, "mismatches", []) or [])
        results.append((d.name, "PASS" if not mm else "MISMATCH",
                        [f"{m.path}: {m.reason}" for m in mm]))
    except Exception as e:  # noqa: BLE001
        results.append((d.name, "ERROR", [f"{type(e).__name__}: {e}"]))

npass = sum(1 for _, s, _ in results if s == "PASS")
nmis = sum(1 for _, s, _ in results if s == "MISMATCH")
nerr = sum(1 for _, s, _ in results if s == "ERROR")
print(f"\n=== SkillsBench conversion-parity: {len(results)} tasks | "
      f"PASS={npass}  MISMATCH={nmis}  ERROR={nerr} ===\n")
for name, status, det in results:
    if status != "PASS":
        print(f"  [{status}] {name}")
        for line in det[:4]:
            print(f"      {line}")
sys.exit(0)
