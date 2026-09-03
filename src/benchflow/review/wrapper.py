"""Assemble the throwaway wrapper task that carries one rubric review.

Each review of a finished rollout runs as an ordinary single rollout of a
synthetic task built here on the host:

- the wrapper uses a digest-pinned base image and no task-authored Dockerfile;
  backends may still derive their own runtime image;
- the rollout evidence and, when available, the original task definition are
  uploaded after start (``RolloutConfig.uploads``) to ``/evidence`` — outside
  the agent workdir, so the sandbox-user chown never touches them and they
  stay root-owned and unwritable by the reviewer (the wrapper declares
  ``allow_internet: false``, which engages the sandbox-local model proxy,
  the agent-UID egress firewall, and the no-web agent policy end to end);
- the instruction body is the rendered review prompt plus the structured
  output contract;
- ``tests/`` holds a stdlib-only validator plus the rubric contract
  metadata, so the wrapper's own reward means exactly "the reviewer produced a
  structurally valid result file" — never "the reviewed run was good".

The rubric file itself never enters the sandbox.  Only its derivatives do:
guidance lines inside the instruction, criterion names inside the output
schema, and non-prompt contract metadata inside ``tests/criteria.json``. Task
evidence is sanitized: skills and any shipped ``rubric.json`` are excluded,
and symlinks anywhere in the evidence are dropped rather than dereferenced.
"""

from __future__ import annotations

import json
import shutil
import stat
from pathlib import Path

from benchflow.review.config import (
    REVIEW_RESULT_FILENAME,
    Rubric,
    build_review_response_model,
)
from benchflow.review.policy import DEFAULT_REVIEWER_TIMEOUT_SEC
from benchflow.review.prompts import (
    TASK_MOUNT,
    TRIAL_MOUNT,
    render_review_instruction,
)

#: Pinned by digest: ``python:3.13-slim`` is a mutable tag, so a bare tag
#: would silently change the reviewer environment between runs. Override
#: with ``--image`` (e.g. to an internal mirror or a newer digest).
REVIEWER_IMAGE = (
    "python@sha256:6771159cd4fa5d9bba1258caf0b82e6b73458c694d178ad97c5e925c2d0e1a91"
)
REVIEWER_AGENT_TIMEOUT_SEC = DEFAULT_REVIEWER_TIMEOUT_SEC
REVIEWER_VERIFIER_TIMEOUT_SEC = 120

#: Rollout-side entries never copied into the evidence snapshot.  Excluding
#: prior review output means a re-review can never read an earlier verdict.
_EVIDENCE_EXCLUDES = (
    ".git",
    "llm_trajectory.jsonl",
    "review",
    REVIEW_RESULT_FILENAME,
    "review_report.json",
    "workspace-baseline.json",
)

#: Task-side material the reviewer must not see: skills would break no-skill
#: resource isolation, and the task's own rubric must never enter a sandbox.
_TASK_EVIDENCE_EXCLUDES = (
    "skills",
    "rubric.json",
)

_TEST_SCRIPT = """#!/bin/bash
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p /logs/verifier
cp /app/{result_filename} /logs/verifier/{result_filename} 2>/dev/null || true
if python3 "$DIR/validate.py" /app/{result_filename} "$DIR/criteria.json" "$DIR/trial_name.txt"; then
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

LEGACY_OUTCOMES = {"pass", "fail", "not_applicable"}
BLOCKER_OUTCOMES = {"pass", "fail"}


def main() -> int:
    result_path = Path(sys.argv[1])
    rubric = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    criteria = rubric["criteria"]
    specs = {criterion["name"]: criterion for criterion in criteria}
    names = set(specs)
    weighted = rubric.get("contract") == "v0.2"

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
    expected_top_level = {"trial_name", "summary", "checks"}
    for name in sorted(expected_top_level - data.keys()):
        problems.append(f"result is missing key: {name}")
    if weighted:
        for name in sorted(data.keys() - expected_top_level):
            problems.append(f"result has unexpected key: {name}")
    expected_trial = Path(sys.argv[3]).read_text(encoding="utf-8")
    if data.get("trial_name") != expected_trial:
        problems.append(
            f"trial_name must be {expected_trial!r}, got {data.get('trial_name')!r}"
        )
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
        blocker = specs[name].get("blocker")
        if blocker is None:
            expected_keys = {"outcome", "explanation"}
            outcome = check.get("outcome")
            if not isinstance(outcome, str) or outcome not in LEGACY_OUTCOMES:
                problems.append(
                    f"{name}: outcome must be one of {sorted(LEGACY_OUTCOMES)}"
                )
        elif blocker == 1:
            expected_keys = {"outcome", "explanation"}
            outcome = check.get("outcome")
            if not isinstance(outcome, str) or outcome not in BLOCKER_OUTCOMES:
                problems.append(
                    f"{name}: blocker outcome must be one of "
                    f"{sorted(BLOCKER_OUTCOMES)}"
                )
        else:
            expected_keys = {"score", "explanation"}
            score = check.get("score")
            if isinstance(score, bool) or not isinstance(score, int) or score not in {0, 1, 2}:
                problems.append(f"{name}: score must be the integer 0, 1, or 2")
        if weighted:
            for key in sorted(expected_keys - check.keys()):
                problems.append(f"{name}: check is missing key: {key}")
            for key in sorted(check.keys() - expected_keys):
                problems.append(f"{name}: check has unexpected key: {key}")
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
sandbox:
  docker_image: {image}
  workdir: /app{network_line}
  cpus: 1
  memory_mb: 2048
  storage_mb: 4096
---

"""

#: Default network posture: the wrapper declares no-internet, which engages
#: the no-web pipeline (web tools disabled, sandbox-local model proxy,
#: agent-UID egress firewall) and makes backends that cannot enforce network
#: isolation fail closed at launch. ``open_network=True`` is an explicit
#: operator override for those backends; the report records it.
_NETWORK_LINE_ISOLATED = "\n  allow_internet: false"
_NETWORK_LINE_OPEN = ""


def _evidence_ignore(extra_excludes: tuple[str, ...]):
    """Build a copytree ignore callback: name excludes plus symlink rejection.

    ``shutil.copytree(symlinks=False)`` would *dereference* links, so a
    source-controlled symlink could materialize an arbitrary host file into
    the uploaded evidence. Links are dropped entirely instead.
    """

    patterns = shutil.ignore_patterns(*(_EVIDENCE_EXCLUDES + extra_excludes))

    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored = set(patterns(directory, names))
        for name in names:
            if (Path(directory) / name).is_symlink():
                ignored.add(name)
        return ignored

    return ignore


def copy_evidence(
    source: Path,
    destination: Path,
    *,
    extra_excludes: tuple[str, ...] = (),
) -> None:
    """Copy one evidence tree; drops VCS data, prior reviews, and symlinks."""

    shutil.copytree(
        source,
        destination,
        ignore=_evidence_ignore(extra_excludes),
        symlinks=True,  # never dereference; links are filtered by ignore()
    )
    _strip_write_bits(destination)


def _strip_write_bits(root: Path) -> None:
    """Clear every write bit under *root*.

    ``copytree`` preserves source permissions, so a rollout file recorded
    as 0666 (or a 0777 directory) would stay writable — and the upload
    backends preserve modes — letting the reviewer mutate the evidence it
    is supposed to be grading. Root ownership alone does not prevent that,
    so the modes are normalized here, before upload.
    """

    write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    for path in [root, *root.rglob("*")]:
        if path.is_symlink():
            continue
        current = path.stat().st_mode
        path.chmod(stat.S_IMODE(current) & ~write_bits)


def assemble_review_task(
    rollout_dir: Path,
    task_dir: Path | None,
    rubric: Rubric,
    dest: Path,
    *,
    template: str | None = None,
    image: str = REVIEWER_IMAGE,
    agent_timeout_sec: int = REVIEWER_AGENT_TIMEOUT_SEC,
    open_network: bool = False,
    net_admin_overlay: bool = False,
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
        copy_evidence(
            task_dir,
            evidence / "task",
            extra_excludes=_TASK_EVIDENCE_EXCLUDES,
        )
        uploads[str(evidence / "task")] = TASK_MOUNT
        task_mount = TASK_MOUNT

    response_model = build_review_response_model(rubric)
    instruction = render_review_instruction(
        rubric,
        template=template,
        trial_path=TRIAL_MOUNT,
        task_path=task_mount,
        result_path=f"/app/{REVIEW_RESULT_FILENAME}",
        trial_name=rollout_dir.name,
        output_schema=response_model.model_json_schema(),
    )
    frontmatter = _TASK_FRONTMATTER.format(
        verifier_timeout=float(REVIEWER_VERIFIER_TIMEOUT_SEC),
        agent_timeout=float(agent_timeout_sec),
        image=image,
        network_line=_NETWORK_LINE_OPEN if open_network else _NETWORK_LINE_ISOLATED,
    )
    (dest / "task.md").write_text(frontmatter + instruction, encoding="utf-8")

    if net_admin_overlay:
        # Docker only: the agent-UID egress firewall programs iptables inside
        # the container, which needs NET_ADMIN. The docker backend stacks
        # this overlay on top of its generated compose files; other backends
        # must not receive it (a task compose file changes their strategy).
        environment_dir = dest / "environment"
        environment_dir.mkdir()
        (environment_dir / "docker-compose.yaml").write_text(
            "services:\n  main:\n    cap_add:\n      - NET_ADMIN\n",
            encoding="utf-8",
        )

    tests_dir = dest / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text(
        _TEST_SCRIPT.format(result_filename=REVIEW_RESULT_FILENAME),
        encoding="utf-8",
    )
    (tests_dir / "validate.py").write_text(_VALIDATOR, encoding="utf-8")
    (tests_dir / "criteria.json").write_text(
        json.dumps(rubric.metadata(), indent=2),
        encoding="utf-8",
    )
    # Bind the verdict to the rollout under review: a reviewer that reports
    # some other run's name produces an unattributable review.
    (tests_dir / "trial_name.txt").write_text(rollout_dir.name, encoding="utf-8")
    return dest, uploads
