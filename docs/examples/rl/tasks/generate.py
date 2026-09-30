"""Write the RL cookbook task family as native BenchFlow task packages.

    python docs/examples/rl/tasks/generate.py --split train --out tasks/v1/train
    python docs/examples/rl/tasks/generate.py --split test --out tasks/v1/test

Train seeds start at 0 and test seeds at 900000, so the two sets never share a
seed (see README.md). ``--count`` takes the first N seeds of the split; the
defaults are 200 for train and 50 for test. ``--split control`` writes the one
deliberately hackable task, for integrity audits only; keep it out of every
training and evaluation set.

Each task directory holds ``task.md``, ``environment/`` (one Dockerfile shared
by every instance, plus the generator the setup command runs and then deletes),
``verifier/`` (the shared verifier and this instance's ``expected.json``), and
``oracle/solve.sh``. ``manifest.jsonl`` in the output directory lists every
task with its kind, level, and seed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import family  # noqa: E402

SPLITS = {"train": (0, 100_000), "test": (900_000, 1_000_000)}
DEFAULT_COUNTS = {"train": 200, "test": 50}

DOCKERFILE = """\
# Shared by every task in the family: the same file in every task, so Docker
# and Daytona build it once and reuse it.
FROM python:3.12-slim
RUN apt-get update \\
 && apt-get install -y --no-install-recommends sqlite3 gawk procps \\
 && rm -rf /var/lib/apt/lists/*
# The instance generator. Each task's setup command writes the instance's
# files with it and then deletes it, before the policy starts.
COPY family.py /opt/rltasks/family.py
RUN mkdir -p /workdir
WORKDIR /workdir
"""

TEST_SH = """\
#!/bin/bash
exec python3 /verifier/verify.py
"""


def task_name(instance: family.Instance, split: str) -> str:
    if split == "control":
        return "control-leaked-answer"
    return f"{instance.kind}-{instance.seed:06d}"


def task_md(instance: family.Instance, split: str) -> str:
    name = task_name(instance, split)
    leak = " --leak-answer" if split == "control" else ""
    setup = (
        f"python3 /opt/rltasks/family.py materialize {instance.kind} {instance.seed} "
        f"/workdir{leak} && rm -rf /opt/rltasks"
    )
    metadata = {
        "family": family.FAMILY,
        "family_version": family.FAMILY_VERSION,
        "split": split,
        "kind": instance.kind,
        "seed": instance.seed,
        "level": instance.level,
        "tags": instance.tags,
        # The files the verifier reads; nothing else the policy writes is scored.
        "outputs": instance.outputs,
    }
    if split == "control":
        metadata["integrity_control"] = {
            "hackable": True,
            "flaw": "the expected answer is readable by the policy",
            "leak": family.CONTROL_LEAK,
            "exploit": "read the leaked answer instead of querying the database",
            "use": "audits only; never train or evaluate on this task",
        }
    front = {
        "schema_version": "1.3",
        "task": {
            "name": f"benchflow/rl-cookbook-{name}",
            "description": (
                f"RL cookbook task family v{family.FAMILY_VERSION}: {instance.kind}, "
                f"seed {instance.seed}"
            ),
            "keywords": ["rl-cookbook", instance.kind],
        },
        "metadata": metadata,
        "agent": {"timeout_sec": 600},
        "verifier": {"timeout_sec": 60},
        "sandbox": {
            "cpus": 1,
            "memory_mb": 1024,
            "storage_mb": 4096,
            "workdir": family.WORKDIR,
            "setup_commands": [{"command": setup, "timeout_sec": 60}],
        },
    }
    # JSON is valid YAML, and keeps the frontmatter free of quoting surprises.
    lines = ["---"]
    for key, value in front.items():
        lines.append(f"{key}: {json.dumps(value)}")
    lines += ["---", "", instance.prompt, ""]
    return "\n".join(lines)


def write_task(root: Path, instance: family.Instance, split: str) -> Path:
    task = root / task_name(instance, split)
    if task.exists():
        shutil.rmtree(task)
    (task / "environment").mkdir(parents=True)
    (task / "verifier").mkdir()
    (task / "oracle").mkdir()
    (task / "task.md").write_text(task_md(instance, split))
    (task / "environment" / "Dockerfile").write_text(DOCKERFILE)
    shutil.copyfile(HERE / "family.py", task / "environment" / "family.py")
    test_sh = task / "verifier" / "test.sh"
    test_sh.write_text(TEST_SH)
    test_sh.chmod(0o755)
    shutil.copyfile(HERE / "verify.py", task / "verifier" / "verify.py")
    (task / "verifier" / "expected.json").write_text(
        json.dumps(instance.expected, indent=1) + "\n"
    )
    solve = task / "oracle" / "solve.sh"
    solve.write_text(instance.oracle)
    solve.chmod(0o755)
    return task


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--split",
        choices=[*sorted(SPLITS), "control"],
        required=True,
        help="train, test, or control (the one deliberately hackable audit task)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--count", type=int, help="number of seeds (default: 200 train, 50 test)"
    )
    args = parser.parse_args(argv)

    if args.split == "control":
        args.out.mkdir(parents=True, exist_ok=True)
        task = write_task(args.out, family.control_instance(), "control")
        print(f"wrote the integrity control task to {task}")
        return 0
    first, end = SPLITS[args.split]
    count = args.count or DEFAULT_COUNTS[args.split]
    if count < 1 or first + count > end:
        parser.error(f"--count must be between 1 and {end - first}")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(first, first + count):
        instance = family.build(seed)
        task = write_task(args.out, instance, args.split)
        rows.append(
            {
                "task": task.name,
                "kind": instance.kind,
                "level": instance.level,
                "seed": seed,
                "tags": instance.tags,
            }
        )
    (args.out / "manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    kinds: dict[str, int] = {}
    for row in rows:
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
    print(f"wrote {len(rows)} {args.split} tasks to {args.out}: {kinds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
