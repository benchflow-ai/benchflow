"""Assemble the throwaway wrapper task that carries one rubric review.

Each review of a finished rollout runs as an ordinary single rollout of a
synthetic task built here on the host:

- the wrapper uses a pinned prebuilt image (no Dockerfile, no image build on
  any backend);
- the rollout evidence and, when available, the original task definition are
  uploaded read-only into the sandbox after start (``RolloutConfig.uploads``);
- the instruction body is the rendered review prompt plus the structured
  output contract;
- ``tests/`` holds a stdlib-only validator plus the criterion-name list, so
  the wrapper's own reward means exactly "the reviewer produced a
  structurally valid result file" — never "the reviewed run was good".

The rubric file itself never enters the sandbox.  Only its derivatives do:
guidance lines inside the instruction, criterion names inside the output
schema, and the same names inside ``tests/criteria.json``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from benchflow.review.config import (
    REVIEW_RESULT_FILENAME,
    Rubric,
    build_review_response_model,
)
from benchflow.review.prompts import (
    TASK_MOUNT,
    TRIAL_MOUNT,
    render_review_instruction,
)

REVIEWER_IMAGE = "python:3.13-slim"
REVIEWER_AGENT_TIMEOUT_SEC = 1800
REVIEWER_VERIFIER_TIMEOUT_SEC = 120

#: Rollout-side entries never copied into the evidence snapshot.  Excluding
#: prior review output means a re-review can never read an earlier verdict.
_EVIDENCE_EXCLUDES = (
    ".git",
    "review",
    REVIEW_RESULT_FILENAME,
    "review_report.json",
)

_TEST_SCRIPT = """#!/bin/bash
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p /logs/verifier
cp /app/{result_filename} /logs/verifier/{result_filename} 2>/dev/null || true
if python3 "$DIR/validate.py" /app/{result_filename} "$DIR/criteria.json"; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""

_VALIDATOR = '''"""Structural check of the reviewer's result file.

Runs inside the wrapper's verifier with only the standard library. Verifies
shape, not judgment quality: reward 1 means "a well-formed review exists".
Prints one reason per line on failure; exit code 0 means valid.
"""

import json
import sys
from pathlib import Path

OUTCOMES = {"pass", "fail", "not_applicable"}


def main() -> int:
    result_path = Path(sys.argv[1])
    names = set(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")))

    if not result_path.is_file():
        print(f"result file not found: {result_path}")
        return 1
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"result file is not valid JSON: {exc}")
        return 1
    if not isinstance(data, dict):
        print("result must be a JSON object")
        return 1

    problems = []
    if not isinstance(data.get("trial_name"), str) or not data["trial_name"].strip():
        problems.append("trial_name must be a non-empty string")
    if not isinstance(data.get("summary"), str) or not data["summary"].strip():
        problems.append("summary must be a non-empty string")

    checks = data.get("checks")
    if not isinstance(checks, dict):
        problems.append("checks must be an object keyed by criterion name")
        checks = {}
    for name in sorted(names - checks.keys()):
        problems.append(f"checks is missing criterion: {name}")
    for name in sorted(checks.keys() - names):
        problems.append(f"checks has unexpected key: {name}")
    for name in sorted(names & checks.keys()):
        check = checks[name]
        if not isinstance(check, dict):
            problems.append(f"{name}: value must be an object")
            continue
        if check.get("outcome") not in OUTCOMES:
            problems.append(f"{name}: outcome must be one of {sorted(OUTCOMES)}")
        explanation = check.get("explanation")
        if not isinstance(explanation, str) or not explanation.strip():
            problems.append(f"{name}: explanation must be a non-empty string")

    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
'''

_TASK_FRONTMATTER = """---
schema_version: '1.3'
metadata:
  category: rubric-review
verifier:
  type: test-script
  timeout_sec: {verifier_timeout}
agent:
  timeout_sec: {agent_timeout}
environment:
  docker_image: {image}
  network_mode: public
  cpus: 1
  memory_mb: 2048
  storage_mb: 4096
---

"""


def copy_evidence(source: Path, destination: Path) -> None:
    """Copy one evidence tree, dropping VCS metadata and prior review output."""

    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(*_EVIDENCE_EXCLUDES),
        symlinks=False,
        ignore_dangling_symlinks=True,
    )


def assemble_review_task(
    rollout_dir: Path,
    task_dir: Path | None,
    rubric: Rubric,
    dest: Path,
    *,
    template: str | None = None,
    image: str = REVIEWER_IMAGE,
    agent_timeout_sec: int = REVIEWER_AGENT_TIMEOUT_SEC,
) -> tuple[Path, dict[str, str]]:
    """Assemble one wrapper task under ``dest``.

    Returns the wrapper path plus the upload map (host evidence directory →
    absolute sandbox path) to pass through ``RolloutConfig.uploads``.
    """

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    evidence = dest / "evidence"
    copy_evidence(rollout_dir, evidence / "trial")
    uploads = {str(evidence / "trial"): TRIAL_MOUNT}
    task_mount: str | None = None
    if task_dir is not None and task_dir.is_dir():
        copy_evidence(task_dir, evidence / "task")
        uploads[str(evidence / "task")] = TASK_MOUNT
        task_mount = TASK_MOUNT

    response_model = build_review_response_model(rubric)
    instruction = render_review_instruction(
        rubric,
        template=template,
        trial_path=TRIAL_MOUNT,
        task_path=task_mount,
        result_filename=REVIEW_RESULT_FILENAME,
        output_schema=response_model.model_json_schema(),
    )
    frontmatter = _TASK_FRONTMATTER.format(
        verifier_timeout=float(REVIEWER_VERIFIER_TIMEOUT_SEC),
        agent_timeout=float(agent_timeout_sec),
        image=image,
    )
    (dest / "task.md").write_text(frontmatter + instruction, encoding="utf-8")

    tests_dir = dest / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text(
        _TEST_SCRIPT.format(result_filename=REVIEW_RESULT_FILENAME),
        encoding="utf-8",
    )
    (tests_dir / "validate.py").write_text(_VALIDATOR, encoding="utf-8")
    (tests_dir / "criteria.json").write_text(
        json.dumps([criterion.name for criterion in rubric.criteria], indent=2),
        encoding="utf-8",
    )
    return dest, uploads
