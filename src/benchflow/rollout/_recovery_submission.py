"""Transport and immutable receipts for task-declared submission files."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any

from benchflow._root_python import root_python_command
from benchflow.review.persistence import write_json_atomic
from benchflow.sandbox import _recovery_submission as files
from benchflow.sandbox._exec_output import exec_raw
from benchflow.trust.state import IngressKind, admit


async def _command(
    env: Any, action: str, remote: str, paths: list[str], digest: str | None = None
) -> str:
    args = [action, remote, json.dumps(paths)]
    if digest is not None:
        args.append(digest)
    # The protected Python, isolated (C8, PY8-PT-R8-5).
    return await root_python_command(env, Path(files.__file__).read_text(), *args)


def _identity(rollout: Any) -> dict:
    baseline = getattr(rollout, "_docker_recovery_baseline", None)
    if baseline is None:
        raise files.SubmissionError(
            "Explicit submission recovery requires the original Docker image lease"
        )
    if baseline.task_digest != rollout._config.task_digest:
        raise files.SubmissionError("Original image lease task identity changed")
    return {
        "task_digest": rollout._config.task_digest,
        "effective_config_digest": baseline.effective_config_digest,
        "image_id": baseline.image_id,
        "paths": list(
            files.validate_contract(rollout._task.config.verifier.submission_files)
        ),
    }


def admitted_submission(rollout: Any) -> tuple[Path, str]:
    root = rollout._require_rollout_dir()
    receipt = json.loads((root / "submission.json").read_text())
    expected = _identity(rollout)
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise files.SubmissionError(
            "Submission checkpoint task/runtime contract changed"
        )
    digest = receipt["manifest_sha256"]
    bundle = root.resolve() / "submission-evidence"
    files.validate_submission(
        bundle, expected_manifest_sha256=digest, paths=expected["paths"]
    )
    return bundle, digest


async def capture_submission(rollout: Any) -> None:
    """Caller has disconnected/quiesced writers. Never replace an existing receipt."""
    root = rollout._require_rollout_dir()
    if (root / "submission.json").exists():
        await asyncio.to_thread(admitted_submission, rollout)
        return
    identity = _identity(rollout)
    bundle = root.resolve() / "submission-evidence"
    bundle.mkdir(mode=0o700)  # An incomplete previous capture fails closed.
    remote = "/tmp/benchflow-submission-" + uuid.uuid4().hex
    result = await exec_raw(
        rollout._env.exec,
        await _command(rollout._env, "capture", remote, identity["paths"]),
        user="root",
        timeout_sec=120,
    )
    if result.return_code:
        raise files.SubmissionError("Explicit submission capture failed")
    digest = (result.stdout or "").strip()
    await rollout._env.download_dir(remote, bundle)
    await asyncio.to_thread(
        files.validate_submission,
        bundle,
        expected_manifest_sha256=digest,
        paths=identity["paths"],
    )
    record = {"schema_version": 1, **identity, "manifest_sha256": digest}
    temporary = root / (".submission-" + uuid.uuid4().hex + ".json")
    try:
        write_json_atomic(temporary, record)
        os.link(temporary, root / "submission.json")  # atomic, exclusive publication
    finally:
        temporary.unlink(missing_ok=True)


async def restore_submission(rollout: Any, env: Any) -> None:
    bundle, digest = await asyncio.to_thread(admitted_submission, rollout)
    paths = list(
        files.validate_contract(rollout._task.config.verifier.submission_files)
    )
    remote = "/tmp/benchflow-submission-" + uuid.uuid4().hex
    admission = admit(env, "main", IngressKind.RECOVERY_TRANSFER)
    await env.upload_dir(bundle, remote, admission=admission)
    result = await exec_raw(
        env.exec,
        await _command(env, "restore", remote, paths, digest),
        user="root",
        timeout_sec=120,
    )
    if result.return_code:
        raise files.SubmissionError(
            "Explicit submission restore failed; verifier must not run"
        )
