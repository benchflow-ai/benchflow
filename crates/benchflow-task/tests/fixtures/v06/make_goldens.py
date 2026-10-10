"""What BenchFlow 0.8 reads from each fixture package, for tests/reference_08.rs.

usage: python make_goldens.py <crate tests/fixtures dir> <BenchFlow 0.8 src dir>

Loads every package under v06/, split/, and convert/v06/ with 0.8's own loader (benchflow.task.Task, which reads
task.md frontmatter strictly and task.toml leniently) and writes expected.json: the instruction and the settings the
run engine uses, or the error 0.8 refuses the package with. Needs BenchFlow 0.8's dependencies (the repository's own
virtual environment has them).
"""

import json
import sys
from pathlib import Path

fixtures, src = (Path(a).resolve() for a in sys.argv[1:3])
sys.path.insert(0, str(src))
from benchflow.task import Task  # noqa: E402


def pick(model, keys):
    data = model.model_dump(mode="json")
    return {k: data.get(k) for k in keys}


expected = {}
for group in ("v06", "split", "convert/v06"):
    for pkg in sorted((fixtures / group).iterdir()):
        if not pkg.is_dir():
            continue
        key = f"{group}/{pkg.name}"
        try:
            task = Task(pkg)
        except Exception as exc:  # 0.8 refuses the package
            expected[key] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc).splitlines()[0]}"}
            continue
        cfg = task.config
        expected[key] = {
            "ok": True,
            "instruction": task.instruction,
            "name": cfg.task.name if cfg.task else None,
            "agent": pick(cfg.agent, ["timeout_sec", "user", "network_mode", "allowed_hosts"]),
            "sandbox": pick(
                cfg.sandbox,
                ["docker_image", "cpus", "memory_mb", "storage_mb", "gpus", "workdir", "env", "network_mode",
                 "allowed_hosts", "build_timeout_sec"],
            ),
            "verifier": pick(cfg.verifier, ["timeout_sec", "user", "env"]),
            "solution": pick(cfg.solution, ["env"]),
        }

(fixtures / "v06" / "expected.json").write_text(json.dumps(expected, indent=2, ensure_ascii=False) + "\n")
print(f"wrote {len(expected)} packages")
