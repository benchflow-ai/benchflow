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
definitions are sanitized: skills and any shipped ``rubric.json`` are excluded.
Generic run/task copies drop symlinks. A rollout that already holds a verdict
gives a re-review only its rollout layout (``_REVIEW_LAYOUT``, an allow-list),
a ``result.json`` from before any review, and a record of the names it left
out. Automatic workspace bundles preserve
manifest-validated internal links through an archive and verify every file after
upload, before the reviewer starts. Available provider traces cross the canonical
redaction boundary before inclusion.
"""

from __future__ import annotations

import json
import math
import shutil
import stat
from pathlib import Path

from benchflow._utils.raw_evidence import is_raw_evidence, refuse_raw_evidence
from benchflow.review.config import (
    REVIEW_RESULT_FILENAME,
    Rubric,
    build_review_response_model,
)
from benchflow.review.options import (
    REVIEWER_AGENT_TIMEOUT_SEC,
    REVIEWER_IMAGE,
    REVIEWER_MEMORY_MB,
    REVIEWER_STORAGE_MB,
)
from benchflow.review.prompts import (
    TASK_MOUNT,
    TRIAL_MOUNT,
    WORKSPACE_MOUNT,
    render_review_instruction,
)

REVIEWER_VERIFIER_TIMEOUT_SEC = 120
REVIEWER_ARTIFACT_PACKAGES = ("numpy==2.2.6", "pypdf==5.9.0")
#: What ``ReviewerConfig.pdf_tools`` adds: PyMuPDF extracts a PDF's text and
#: renders its pages to images. It installs with pip, so it works on every
#: sandbox backend; poppler's pdftotext needs an image that holds it
#: (:data:`REVIEWER_PDF_IMAGE`).
REVIEWER_PDF_PACKAGES = ("PyMuPDF==2.1",)
#: The image ``docker/reviewer-pdf/Dockerfile`` builds: :data:`REVIEWER_IMAGE`
#: plus poppler-utils and the packages above. A local tag, so a backend that
#: pulls from a registry (Daytona) cannot start it; its suffix is the first 12
#: hex of the Dockerfile's sha256 (tests/test_reviewer_pdf_tools.py).
REVIEWER_PDF_IMAGE = "benchflow-reviewer-pdf:1e2f39cd619c"
#: The reviewer images BenchFlow knows, each with the artifact packages it
#: already holds; the reviewer's setup installs the ones it lacks
#: (:func:`reviewer_setup_packages`).
KNOWN_REVIEWER_IMAGES: dict[str, tuple[str, ...]] = {
    REVIEWER_IMAGE: (),
    REVIEWER_PDF_IMAGE: REVIEWER_ARTIFACT_PACKAGES + REVIEWER_PDF_PACKAGES,
}


def reviewer_setup_packages(image: str, *, pdf_tools: bool = False) -> tuple[str, ...]:
    """The pip pins the reviewer sandbox's setup installs into *image*.

    A known image (:data:`KNOWN_REVIEWER_IMAGES`) gets the artifact packages
    it lacks, and with *pdf_tools* the PDF packages it lacks. Any other image
    gets nothing, as before, unless *pdf_tools* asks for both sets; pip leaves
    a pin the image already satisfies as it is.
    """
    wanted = REVIEWER_ARTIFACT_PACKAGES + (REVIEWER_PDF_PACKAGES if pdf_tools else ())
    held = KNOWN_REVIEWER_IMAGES.get(image)
    if held is None:
        return wanted if pdf_tools else ()
    return tuple(package for package in wanted if package not in held)


#: Rollout-side entries never copied into the evidence snapshot.  Excluding
#: prior review output means a re-review can never read an earlier verdict.
#: A hosted run's raw vf-eval output (``is_raw_evidence``) never enters a
#: reviewer sandbox either.
_EVIDENCE_EXCLUDES = (
    ".git",
    "llm_trajectory.jsonl",
    "review",
    "reviews",
    "scoring",
    REVIEW_RESULT_FILENAME,
    "review_report.json",
)

#: A ``result.json`` key holding a verdict of record: BenchFlow's ``scoring``,
#: the ``review``/``reviews`` of older writers and a run owner's
#: ``operator_verdict``.
_VERDICT_KEYS = ("scoring", "review", "reviews", "operator_verdict")
#: The top-level names a re-review's copy keeps when the rollout already holds
#: a verdict. Everything else at the top of the rollout stays out.
#:
#: The copy is a blind re-review's input, so it is an allow-list: default-deny
#: is the safe side. A verdict can be recorded under any name beside
#: ``result.json`` (BenchFlow's score exports and regrade outputs, and the
#: backups, re-score and publication records other tools keep there), and no
#: deny-list can name them all. A name missing here costs the reviewer one
#: record; a verdict let through costs the review. So a name is added here by
#: a reviewed change, and the copy's skip record (``_LEFT_OUT_RECORD``) names
#: whatever a rollout had that the copy left out.
_REVIEW_LAYOUT = frozenset(
    {
        # The rollout layout (task.paths.RolloutPaths); result.json is
        # written anew as the record from before any review.
        "agent",
        "artifacts",
        "trajectory",
        "verifier",
        "config.json",
        "result.json",
        # The trial's records docs/evaluate/read-results.md lists, without
        # the score exports (rewards.jsonl, results.jsonl, trainer/).
        "timing.json",
        "prompts.json",
        "artifacts-manifest.json",
        "cloud-metadata.json",
        "firewall-power.json",
        "locked-system-path.json",
        "kill-before-grading.json",
        "kill-after-grading.json",
        "trust-state.json",
        # bench bundle's trial files and folders (evidence_bundle.TRIAL_FILES
        # and TRIAL_FOLDERS), without its reviews and regrades.
        "steps.json",
        "verification.json",
        "rounds",
        # The solver's record, and the verifier revisions verification_source
        # admits with it.
        "solver.json",
        "verifier-recovery",
        "verifier-sandbox",
        # What the rollout writes before grading: the sandbox, a user loop's
        # rounds, a failed rollout's traceback, the agent's network policies.
        "sandbox.json",
        "user_rounds.jsonl",
        "error-traceback.txt",
        "native-oauth-network.json",
        "codex_apps_policy.json",
        # Records a caller's own tooling writes into a rollout before its
        # verifier runs, as a reported rollout's own allow-list keeps them.
        "workspace",
        "provider_overrides",
        "runtime.json",
        "evidence_capture_policy.json",
        "runtime_facts.json",
        "pre_verify_removed.json",
    }
)
#: In a folder the layout leaves out, the files a re-review's copy takes all
#: the same, by their path: the masking record, which a withheld file's stub
#: points to (integrity/'s other records stay out).
_REVIEW_LAYOUT_FILES = frozenset({"integrity/masking.json"})
#: Beside a re-review's copy (``evidence/trial-left-out.json``, uploaded to
#: ``/evidence/``): the top-level names of the rollout that the copy does not
#: hold, and their count, never their contents. Outside the copy, so the copy
#: itself holds only the rollout layout.
_LEFT_OUT_RECORD = "trial-left-out.json"

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
        # A reviewer that copies the schema it was given can copy its
        # definitions too ($defs, $schema); they carry no verdict.
        schema_keys = {"$defs", "$schema"}
        for name in sorted(data.keys() - expected_top_level - schema_keys):
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
  memory_mb: {memory_mb}
  storage_mb: {storage_mb}{setup_commands}
---

"""

#: The reviewer sandbox's disk beyond its evidence: the image, the reviewer
#: agent, the model proxy and the artifact packages.
REVIEWER_DISK_MARGIN_MB = 2048


def reviewer_storage_mb(evidence_bytes: int, environment: str) -> int:
    """The reviewer sandbox's disk for *evidence_bytes* of evidence: that plus
    :data:`REVIEWER_DISK_MARGIN_MB`, never below :data:`REVIEWER_STORAGE_MB`,
    at most what the backend gives one sandbox (Daytona's ceiling)."""
    wanted = max(
        REVIEWER_STORAGE_MB,
        math.ceil(evidence_bytes / 1024**2) + REVIEWER_DISK_MARGIN_MB,
    )
    if environment == "daytona":
        from benchflow.sandbox.setup import _DAYTONA_MAX_STORAGE_MB

        return min(wanted, _DAYTONA_MAX_STORAGE_MB)
    return wanted


#: Default network posture: the wrapper declares no-internet, which engages
#: the no-web pipeline (web tools disabled, sandbox-local model proxy,
#: agent-UID egress firewall) and makes backends that cannot enforce network
#: isolation fail closed at launch. ``open_network=True`` is an explicit
#: operator override for those backends; the report records it.
_NETWORK_LINE_ISOLATED = "\n  allow_internet: false"
_NETWORK_LINE_OPEN = ""


def _evidence_ignore(
    extra_excludes: tuple[str, ...],
    root: Path | None = None,
    root_names: frozenset[str] | None = None,
):
    """Build a copytree ignore callback: name excludes (and, directly in
    *root*, every name not in *root_names*) plus symlink rejection.

    ``shutil.copytree(symlinks=False)`` would *dereference* links, so a
    source-controlled symlink could materialize an arbitrary host file into
    the uploaded evidence. Links are dropped entirely instead.
    """

    patterns = shutil.ignore_patterns(*(_EVIDENCE_EXCLUDES + extra_excludes))

    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored = set(patterns(directory, names))
        if root_names is not None and Path(directory) == root:
            ignored |= {name for name in names if name not in _tops(root_names)}
        elif root_names is not None and Path(directory).parent == root:
            # A folder the layout names files of (integrity/masking.json):
            # those files only.
            inside = _inside(root_names, Path(directory).name)
            if inside:
                ignored |= {name for name in names if name not in inside}
        for name in names:
            if (Path(directory) / name).is_symlink() or is_raw_evidence(
                Path(directory), name
            ):
                ignored.add(name)
        return ignored

    return ignore


def _tops(layout: frozenset[str]) -> set[str]:
    return {name.split("/", 1)[0] for name in layout}


def _inside(layout: frozenset[str], folder: str) -> set[str]:
    """The names in *folder* the layout names one by one (``folder/name``);
    empty for a folder it names whole."""
    return {name.split("/", 1)[1] for name in layout if name.startswith(f"{folder}/")}


def copy_evidence(
    source: Path,
    destination: Path,
    *,
    extra_excludes: tuple[str, ...] = (),
    root_names: frozenset[str] | None = None,
) -> None:
    """Copy one evidence tree; drops VCS data, prior reviews, and symlinks.
    With *root_names*, only those names are copied from the top of *source*.
    A *source* that is a hosted run's raw evidence or sits inside it is
    refused (``RawEvidenceError``)."""
    refuse_raw_evidence(source)
    shutil.copytree(
        source,
        destination,
        ignore=_evidence_ignore(extra_excludes, Path(source), root_names),
        symlinks=True,  # never dereference; links are filtered by ignore()
    )
    _strip_write_bits(destination)


def _held_verdict(rollout_dir: Path) -> bool:
    """Whether the rollout's ``result.json`` already holds a verdict.

    A linked ``result.json`` counts as one: the copy drops the link unread,
    so nothing shows what it holds, and what is copied beside it is kept to
    the rollout layout."""
    path = rollout_dir / "result.json"
    if path.is_symlink():
        return True
    try:
        result = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return isinstance(result, dict) and any(
        result.get(key) is not None for key in _VERDICT_KEYS
    )


def shows_final_result(rollout_dir: Path) -> bool:
    """Whether a review of *rollout_dir* shows its reviewer the run's final
    ``result.json``: a finished run that no review has scored. The in-run
    review's rollout has no result yet, and a re-review of a scored one
    shows the record from before any review (:func:`_pre_review_record`)."""
    return (rollout_dir / "result.json").is_file() and not _held_verdict(rollout_dir)


def _pre_review_record(trial: Path) -> dict:
    """The copy's ``result.json`` as the first review saw it: the solver's
    record with its admitted verifier revision, no verdict of record. Read
    from the copy, which holds no link, never from the rollout.

    Without a readable solver record (a rollout from before BenchFlow wrote
    ``solver.json``, or one whose ``solver.json`` was a link) it is the
    copy's ``result.json`` without its verdict keys and with only the
    verifier's reward: ``rewards.verifier_reward`` as ``reward``, and no
    ``rewards`` when it names none, since the scored reward is the verdict's.
    With no ``result.json`` either (a linked one is not copied) it is ``{}``.
    """
    from benchflow.rollout._verifier_recovery import verification_source

    try:
        record = verification_source(trial)
    except (OSError, ValueError, KeyError, TypeError):
        record = None
    if not isinstance(record, dict):
        try:
            record = json.loads((trial / "result.json").read_text())
        except (OSError, ValueError):
            record = {}
        if not isinstance(record, dict):
            record = {}
        rewards = record.pop("rewards", None)
        if isinstance(rewards, dict) and rewards.get("verifier_reward") is not None:
            record["rewards"] = {"reward": rewards["verifier_reward"]}
    for key in _VERDICT_KEYS:
        record.pop(key, None)
    return record


def _write_pre_review_result(trial: Path) -> None:
    """Replace the copy's ``result.json`` with :func:`_pre_review_record`."""
    result = _pre_review_record(trial)
    target = trial / "result.json"
    trial.chmod(trial.stat().st_mode | stat.S_IWUSR)
    try:
        target.unlink(missing_ok=True)
        target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        target.chmod(0o444)
    finally:
        trial.chmod(stat.S_IMODE(trial.stat().st_mode) & ~stat.S_IWUSR)


def _write_left_out_record(rollout_dir: Path, trial: Path) -> Path:
    """Write ``_LEFT_OUT_RECORD`` beside *trial*: each top-level name of
    *rollout_dir* the copy does not hold, and their count. Names only: the
    reviewer can see that a record was left out, never what it held."""
    held = {entry.name for entry in trial.iterdir()}
    left_out = [entry.name for entry in rollout_dir.iterdir() if entry.name not in held]
    # In a folder the layout copies file by file, the names the copy does not
    # hold there: the others, and a listed one the copy skipped (a link).
    for folder in sorted(held):
        source = rollout_dir / folder
        if _inside(_REVIEW_LAYOUT_FILES, folder) and source.is_dir():
            copied = {entry.name for entry in (trial / folder).iterdir()}
            left_out += [
                f"{folder}/{entry.name}"
                for entry in source.iterdir()
                if entry.name not in copied
            ]
    left_out.sort()
    path = trial.parent / _LEFT_OUT_RECORD
    record = {
        "about": (
            "Top-level names of the reviewed rollout that are not in its "
            "evidence copy (/evidence/trial), and in integrity/ the records "
            "other than masking.json. A rollout that already holds a "
            "verdict is copied with its rollout layout only, and its "
            "result.json is the record from before any review."
        ),
        "count": len(left_out),
        "left_out": left_out,
    }
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    path.chmod(0o444)
    return path


def _copy_provider_trace(source: Path, destination: Path) -> None:
    """Include canonical provider capture only after its established redactor."""
    from benchflow.trajectories.types import redact_trajectory_obj

    relative = Path("trajectory") / "llm_trajectory.jsonl"
    original = source / relative
    if original.is_symlink() or original.parent.is_symlink():
        raise ValueError("provider trajectory must be a regular captured file")
    if not original.exists():
        return
    target = destination / relative
    target.parent.chmod(target.parent.stat().st_mode | stat.S_IWUSR)
    try:
        with (
            original.open(encoding="utf-8") as reader,
            target.open("w", encoding="utf-8") as writer,
        ):
            for line in reader:
                if line.strip():
                    writer.write(
                        json.dumps(redact_trajectory_obj(json.loads(line))) + "\n"
                    )
    finally:
        _strip_write_bits(target.parent)


def remove_review_task(root: Path) -> None:
    """Remove read-only upload copies without following evidence symlinks."""
    if not root.exists():
        return
    for path in [root, *root.rglob("*")]:
        if not path.is_symlink():
            path.chmod(stat.S_IMODE(path.stat().st_mode) | stat.S_IWUSR)
    shutil.rmtree(root)


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
    workspace_bundle: Path | None = None,
    memory_mb: int = REVIEWER_MEMORY_MB,
    storage_mb: int | None = None,
    environment: str = "docker",
    pdf_tools: bool = False,
    submitted_file_rule: bool = False,
    criterion_evidence: bool = False,
    final_result: bool = False,
) -> tuple[Path, dict[str, str]]:
    """Assemble one wrapper task under ``dest``.

    Returns the wrapper path plus the upload map (host evidence directory →
    absolute sandbox path) to pass through ``RolloutConfig.uploads``. Without
    *storage_mb* the reviewer sandbox's disk is sized for the evidence it
    holds (:func:`reviewer_storage_mb` on *environment*): the uploads, and
    each workspace unpacked beside its archive. The setup installs
    :func:`reviewer_setup_packages` for *image* and *pdf_tools*; the
    instruction takes the rule with *submitted_file_rule*
    (:func:`~benchflow.review.prompts.review_template`), each criterion's
    Evidence line with *criterion_evidence*, and with *final_result*
    (:func:`shows_final_result`) does not say that the trial has no final
    result yet.
    """

    if dest.exists():
        remove_review_task(dest)
    dest.mkdir(parents=True)

    evidence = dest / "evidence"
    # A re-review never reads the verdict its rollout already holds: it gets
    # the rollout layout, the record from before any review, and the names
    # of what was left out. A first review gets the rollout as it is.
    held = _held_verdict(rollout_dir)
    copy_evidence(
        rollout_dir,
        evidence / "trial",
        extra_excludes=("evidence",),
        root_names=_REVIEW_LAYOUT | _REVIEW_LAYOUT_FILES if held else None,
    )
    uploads = {str(evidence / "trial"): TRIAL_MOUNT}
    if held:
        _write_pre_review_result(evidence / "trial")
        left_out = _write_left_out_record(rollout_dir, evidence / "trial")
        uploads[str(left_out)] = f"/evidence/{_LEFT_OUT_RECORD}"
    if workspace_bundle is not None:
        from benchflow.review.evidence import (
            EvidenceManifest,
            prepare_workspace_upload,
            refuse_linked_bundle,
        )

        manifest = EvidenceManifest.model_validate_json(
            (workspace_bundle / "manifest.json").read_text()
        )
        # Before any bundle is read: a link below it may lead to another
        # capture, which would validate and be uploaded as this one's.
        refuse_linked_bundle(workspace_bundle, manifest)
        bundles = [(workspace_bundle, "workspace")]
        bundles.extend(
            (workspace_bundle / artifact.bundle_path, artifact.bundle_path)
            for artifact in manifest.artifacts
            if artifact.status == "captured" and artifact.bundle_path is not None
        )
        for bundle, relative in bundles:
            archive = evidence / f"{relative}.tar"
            archive.parent.mkdir(parents=True, exist_ok=True)
            prepare_workspace_upload(bundle, archive)
            manifest_copy = evidence / f"{relative}-manifest.json"
            shutil.copyfile(bundle / "manifest.json", manifest_copy)
            uploads[str(archive)] = f"/evidence/{relative}.tar"
            uploads[str(manifest_copy)] = f"/evidence/{relative}-manifest.json"
        _copy_provider_trace(rollout_dir, evidence / "trial")
    task_mount: str | None = None
    if task_dir is not None and task_dir.is_dir():
        copy_evidence(
            task_dir,
            evidence / "task",
            extra_excludes=_TASK_EVIDENCE_EXCLUDES,
        )
        uploads[str(evidence / "task")] = TASK_MOUNT
        task_mount = TASK_MOUNT
    if storage_mb is None:
        held = sum(
            path.stat().st_size for path in evidence.rglob("*") if path.is_file()
        )
        if workspace_bundle is not None:
            from benchflow.review.evidence import evidence_bytes

            held += evidence_bytes(workspace_bundle)
        storage_mb = reviewer_storage_mb(held, environment)

    response_model = build_review_response_model(rubric)
    instruction = render_review_instruction(
        rubric,
        template=template,
        trial_path=TRIAL_MOUNT,
        task_path=task_mount,
        result_path=f"/app/{REVIEW_RESULT_FILENAME}",
        trial_name=rollout_dir.name,
        output_schema=response_model.model_json_schema(),
        workspace_path=WORKSPACE_MOUNT if workspace_bundle is not None else None,
        submitted_file_rule=submitted_file_rule,
        criterion_evidence=criterion_evidence,
        final_result=final_result,
    )
    packages = reviewer_setup_packages(image, pdf_tools=pdf_tools)
    frontmatter = _TASK_FRONTMATTER.format(
        verifier_timeout=float(REVIEWER_VERIFIER_TIMEOUT_SEC),
        agent_timeout=float(agent_timeout_sec),
        image=image,
        network_line=_NETWORK_LINE_OPEN if open_network else _NETWORK_LINE_ISOLATED,
        memory_mb=memory_mb,
        storage_mb=storage_mb,
        setup_commands=(
            "\n  setup_commands:"
            "\n    - command: python3 -m pip install --disable-pip-version-check --no-cache-dir "
            + " ".join(packages)
            + "\n      user: root\n      timeout_sec: 600\n      cwd: /tmp"
            if packages
            else ""
        ),
    )
    (dest / "task.md").write_text(frontmatter + instruction, encoding="utf-8")

    # No task compose file: it would flip Daytona and other backends into
    # compose strategies. The agent-UID egress firewall's NET_ADMIN comes from
    # the backend's no-web rule (``compose_needs_net_admin`` on Docker).

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
