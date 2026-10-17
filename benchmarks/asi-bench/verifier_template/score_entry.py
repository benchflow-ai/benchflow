#!/usr/bin/env python3
"""ASI-Bench BenchFlow verifier: staging, scoring, and reward reporting.

Invoked by test.sh inside the BenchFlow sandbox after the agent finishes.
Reads from /verifier/ (reference, evaluator code, instance.json, output_specs.json)
and /workspace/ (agent outputs), writes to /logs/verifier/.

Contract:
  - Submission failure (missing/bad outputs): write reward (may be 0.0), exit 0.
  - Evaluator/infrastructure error: do NOT write reward, write asi_error.json, exit 1.
    BenchFlow records this as verifier_failure, not a 0-score.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import site
import stat
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

# ── layout constants ────────────────────────────────────────────────────────

VERIFIER_DIR = Path("/verifier")
WORKSPACE_DIR = Path("/workspace")
LOGS_DIR = Path("/logs/verifier")

EVALUATOR_DIR = VERIFIER_DIR / "evaluator"
REFERENCE_DIR = VERIFIER_DIR / "reference"
INSTANCE_JSON = VERIFIER_DIR / "instance.json"
OUTPUT_SPECS_JSON = VERIFIER_DIR / "output_specs.json"
SCORE_DIVISOR_JSON = VERIFIER_DIR / "score_divisor.json"


class MissingEvaluatorInputError(RuntimeError):
    """Immutable input required by the evaluator is absent or unsafe."""

    failure_kind = "missing_evaluator_input"


# ── path safety ─────────────────────────────────────────────────────────────

def _safe_rel(value: str) -> Path:
    """Reject absolute paths, traversal, and null bytes."""
    if not value or value.startswith("/") or "\\" in value or "\x00" in value:
        raise ValueError(f"unsafe output path: {value!r}")
    parts = value.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"unsafe output path: {value!r}")
    return Path(value)


def _is_safe_regular(path: Path) -> bool:
    """Return True iff path is a regular file (not a symlink)."""
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── output staging ──────────────────────────────────────────────────────────

def _stage_outputs(
    workspace: Path,
    output_specs: list[dict],
    staging: Path,
) -> dict[str, str]:
    """Copy declared output files from workspace into staging/.

    Returns a dict mapping relative output path → SHA-256 for present files.
    Missing files are noted in the return value with None-like absence
    (they are not copied) and will be caught by gates/scorers.
    """
    present: dict[str, str] = {}
    for spec in output_specs:
        rel_str = spec.get("name", "")
        if not rel_str:
            continue
        try:
            rel = _safe_rel(rel_str)
        except ValueError:
            continue  # invalid spec → skip, scorer will fail

        src = workspace / rel
        if rel.parts[0] == "data":
            raise MissingEvaluatorInputError(
                f"Declared prediction overlaps evaluator input: {rel}"
            )
        if any((workspace / Path(*rel.parts[:i])).is_symlink() for i in range(1, len(rel.parts))):
            # A nested path through an agent-authored symlink is not a safe output.
            continue
        if src.is_symlink():
            # Reject symlinks; scorer will see the file missing
            continue
        if not _is_safe_regular(src):
            continue  # missing; scorer handles it as submission failure

        dst = staging / rel
        if dst.exists() or dst.is_symlink():
            raise MissingEvaluatorInputError(f"Prediction would overwrite evaluator input: {rel}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        present[rel_str] = _sha256(dst)
    return present


def _stage_data_inputs(data_src: Path, staging: Path, *, required: bool) -> None:
    """Copy the verifier-owned instance data tree into the scorer workspace."""
    if not data_src.exists() and not data_src.is_symlink():
        if required:
            raise MissingEvaluatorInputError(f"Required evaluator input is missing: {data_src}")
        return
    if data_src.is_symlink() or not data_src.is_dir():
        raise MissingEvaluatorInputError(f"Evaluator input is not a real directory: {data_src}")
    data_dst = staging / "data"
    data_dst.mkdir()
    for path in data_src.rglob("*"):
        relative = path.relative_to(data_src)
        target = data_dst / relative
        if path.is_symlink():
            raise MissingEvaluatorInputError(f"Evaluator input contains a symlink: {relative}")
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        else:
            raise MissingEvaluatorInputError(f"Evaluator input contains unsupported file: {relative}")


def _persist_prediction_outputs(staging: Path, artifact_shas: dict[str, str]) -> None:
    """Keep scored predictions in host-mounted verifier logs for parity checks."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    destination = LOGS_DIR / "predictions"
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"Prediction archive already exists: {destination}")
    destination.mkdir()
    for rel_str, expected_sha in sorted(artifact_shas.items()):
        rel = _safe_rel(rel_str)
        source = staging / rel
        if not _is_safe_regular(source):
            raise RuntimeError(f"Staged prediction disappeared: {rel_str}")
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if _sha256(target) != expected_sha:
            raise RuntimeError(f"Persisted prediction checksum mismatch: {rel_str}")


# ── evaluator bootstrap ──────────────────────────────────────────────────────

def _bootstrap_evaluator(evaluator_dir: Path) -> None:
    """Prepend evaluator_dir to sys.path so ai4sci_bench imports cleanly.

    Also wires the task_eval.yaml path into the evaluator environment so
    resolve_task_sources() finds it without needing a git checkout.
    """
    ev = str(evaluator_dir.resolve())
    if ev not in sys.path:
        sys.path.insert(0, ev)


def _activate_task_runtime(evaluation: dict[str, Any]) -> None:
    """Let task scorers launch workers using the prebuilt verifier runtime."""
    if evaluation.get("runtime") != "task":
        return
    os.environ["AI4SCI_TASK_RUNTIME_ACTIVE"] = "1"
    os.environ["AI4SCI_TASK_RUNTIME_PYTHON"] = sys.executable
    os.environ["AI4SCI_TASK_RUNTIME_BIN"] = str(Path(sys.executable).parent)
    os.environ["AI4SCI_TRUSTED_SITE_PACKAGES"] = os.pathsep.join(site.getsitepackages())


# ── scoring ──────────────────────────────────────────────────────────────────

def _detail_dict(detail: Any) -> dict[str, Any]:
    """Convert a ScoreDetail dataclass into a JSON-serializable dict."""
    import dataclasses

    if dataclasses.is_dataclass(detail) and not isinstance(detail, type):
        return dataclasses.asdict(detail)
    if isinstance(detail, dict):
        return detail
    return {"value": str(detail)}


def _load_instance() -> dict[str, Any]:
    if not INSTANCE_JSON.is_file():
        raise FileNotFoundError(f"instance.json not found at {INSTANCE_JSON}")
    with INSTANCE_JSON.open(encoding="utf-8") as fh:
        return json.load(fh)


def _load_output_specs() -> list[dict]:
    if not OUTPUT_SPECS_JSON.is_file():
        return []
    with OUTPUT_SPECS_JSON.open(encoding="utf-8") as fh:
        return json.load(fh)


def _load_score_divisor() -> float:
    if not SCORE_DIVISOR_JSON.is_file():
        return 1.0
    with SCORE_DIVISOR_JSON.open(encoding="utf-8") as fh:
        data = json.load(fh)
    divisor = float(data.get("score_divisor", 1.0))
    return divisor if divisor > 0 else 1.0


def _run_scoring(
    evaluator_dir: Path,
    pred_dir: Path,
    ref_dir: Path,
    instance: dict,
) -> dict[str, Any]:
    """Import and call the ASI scoring framework.

    Mirrors ``ai4sci_bench.benchflow._score_one`` so BenchFlow rewards stay
    bit-identical to ``asibench score``.

    Returns a dict with: final_score, max_score, hard_gates_passed,
    soft_gate_failures, gate_results, score_results, scorer_internal_error.
    """
    import importlib

    # Load task evaluation config
    task_eval_path = evaluator_dir / "task_eval.yaml"
    if not task_eval_path.is_file():
        raise FileNotFoundError(f"task_eval.yaml not found in {evaluator_dir}")

    import yaml
    with task_eval_path.open(encoding="utf-8") as fh:
        task_eval: dict = yaml.safe_load(fh) or {}

    evaluation = task_eval.get("evaluation") or {}
    _activate_task_runtime(evaluation)

    # Register scorers
    scorers_pkg = evaluator_dir / "ai4sci_bench" / "scorers"
    for scorer_file in sorted(scorers_pkg.glob("*.py")):
        if not scorer_file.stem.startswith("_") and scorer_file.stem != "custom":
            importlib.import_module(f"ai4sci_bench.scorers.{scorer_file.stem}")

    # Load custom scorer if present
    custom_path = evaluator_dir / "custom_scorer.py"
    if custom_path.is_file():
        custom_mod = importlib.import_module("ai4sci_bench.scorers.custom")
        custom_mod.load_custom_scorer(evaluator_dir)

    # The shared scoring core. Dependency-light by construction so it imports
    # cleanly in the minimal /opt/asi-eval environment.
    scoring_mod = importlib.import_module("ai4sci_bench.core.scoring")
    evaluate_fn = scoring_mod._evaluate_gates_and_scores

    # max_score and the task divisor come from the framework, not from local
    # arithmetic, so rewards stay bit-identical to `asibench score`.
    scorer_mod = importlib.import_module("ai4sci_bench.core.scorer")
    scoring_max_score = scorer_mod.scoring_max_score
    normalize_task_score = scorer_mod.normalize_task_score

    params = instance.get("parameters") or {}
    # The framework uses lowercase level values ("b1"); never upper-case here.
    prompt_level = instance.get("prompt_level") or None

    # ``prompt_level`` is keyword-only; ``_evaluate_gates_and_scores`` returns a
    # 5-tuple, not a dict.
    gates, hard_ok, soft_failures, scores, final_score = evaluate_fn(
        evaluation,
        pred_dir,
        ref_dir,
        params,
        prompt_level=prompt_level,
    )

    # Same two steps as ai4sci_bench.benchflow: raw weight sum, then normalize
    # BOTH score and maximum by the task divisor (the ratio is divisor-invariant).
    raw_max_score = scoring_max_score(evaluation)
    final_score, max_score = normalize_task_score(
        evaluation, final_score, raw_max_score
    )

    # An evaluator fault surfaces per-ScoreDetail, not as a top-level flag.
    all_details = [*gates, *scores]
    internal_error = any(
        isinstance(detail.details, dict)
        and detail.details.get("scorer_internal_error") is True
        for detail in all_details
    )

    return {
        "final_score": None if final_score is None else float(final_score),
        "max_score": float(max_score),
        "raw_max_score": float(raw_max_score),
        "hard_gates_passed": bool(hard_ok),
        "soft_gate_failures": int(soft_failures),
        "gate_results": [_detail_dict(detail) for detail in gates],
        "score_results": [_detail_dict(detail) for detail in scores],
        "scorer_internal_error": internal_error,
    }


# ── reward normalization ─────────────────────────────────────────────────────

def _normalize_reward(result: dict[str, Any]) -> float:
    """Compute the [0,1] reward from an ASI scoring result.

    ``_run_scoring`` already passed both ``final_score`` and ``max_score``
    through the framework's ``normalize_task_score``, which divides BOTH by the
    task's ``score_divisor``. The divisor therefore cancels in this ratio and
    must NOT be applied a second time here — doing so would under-report every
    task with a divisor != 1.0 (e.g. ising at 1.05) by that factor.
    """
    final_score = float(result.get("final_score") or 0.0)
    max_score = float(result.get("max_score") or 0.0)
    if max_score <= 0.0:
        # A task whose scoring weights sum to zero cannot produce a meaningful
        # reward. Treat it as an evaluator/config fault, never as a 0 score.
        raise ValueError(
            f"task declares non-positive max_score ({max_score}); "
            "cannot normalize reward"
        )
    return max(0.0, min(1.0, final_score / max_score))


def _json_default(value: Any) -> Any:
    """Coerce numpy scalars / Paths that scorers put in ScoreDetail.details."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


# ── output writers ───────────────────────────────────────────────────────────

def _write_reward(
    reward: float,
    result: dict,
    instance: dict,
    artifact_shas: dict[str, str],
    score_divisor: float,
) -> None:
    """Write reward.txt and reward.json to /logs/verifier/."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # reward.txt — plain scalar
    (LOGS_DIR / "reward.txt").write_text(f"{reward:.10f}\n", encoding="utf-8")

    # reward.json — BenchFlow format: scalar reward + metadata/details
    # Flat non-[0,1] fields are rejected by BenchFlow → put everything in metadata/details
    reward_json = {
        "reward": reward,
        "metadata": {
            "benchmark": "ASI-Bench",
            "seed": 31415,
            "official": False,
            "task_id": instance.get("task_id"),
            "instance_id": instance.get("instance_id"),
            "prompt_level": instance.get("prompt_level"),
            "asi_revision": instance.get("asi_revision"),
            "hf_revision": instance.get("hf_revision"),
            "final_score": result.get("final_score"),
            "max_score": result.get("max_score"),
            "raw_max_score": result.get("raw_max_score"),
            "score_divisor": score_divisor,
        },
        "details": {
            "hard_gates_passed": result.get("hard_gates_passed", False),
            "soft_gate_failures": result.get("soft_gate_failures", 0),
            "gate_results": result.get("gate_results") or [],
            "score_results": result.get("score_results") or [],
        },
        "artifacts": dict(artifact_shas),
    }
    (LOGS_DIR / "reward.json").write_text(
        json.dumps(reward_json, indent=2, ensure_ascii=False, default=_json_default)
        + "\n",
        encoding="utf-8",
    )

    # asi_score.json — full detail without BenchFlow schema restrictions
    asi_json = dict(reward_json)
    asi_json["scorer_internal_error"] = result.get("scorer_internal_error", False)
    (LOGS_DIR / "asi_score.json").write_text(
        json.dumps(asi_json, indent=2, ensure_ascii=False, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def _write_error(exc: Exception, context: str) -> None:
    """Write asi_error.json (no reward) for infrastructure failures."""
    error_doc = {
        "scorer_internal_error": True,
        "failure_kind": getattr(exc, "failure_kind", "evaluator_runtime_error"),
        "error_type": type(exc).__name__,
        "error_context": context,
        "detail": traceback.format_exc(),
    }
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        (LOGS_DIR / "asi_error.json").write_text(
            json.dumps(error_doc, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        # Explicitly ensure no reward files are present
        for name in ("reward.txt", "reward.json"):
            p = LOGS_DIR / name
            if p.exists():
                p.unlink()
        shutil.rmtree(LOGS_DIR / "predictions", ignore_errors=True)
    except OSError:
        # Can't write to /logs (e.g. running outside the sandbox); stderr only.
        pass


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    # 1. Load metadata (evaluator-infrastructure failures → exit 1, no reward)
    try:
        instance = _load_instance()
        output_specs = _load_output_specs()
        # Recorded for provenance only. The reward ratio is divisor-invariant
        # because the framework normalizes score and maximum together.
        score_divisor = _load_score_divisor()
    except Exception as exc:
        _write_error(exc, "loading instance metadata")
        print(f"[verifier] evaluator error (metadata): {exc}", file=sys.stderr)
        return 1

    # 2. Validate seed (fail-closed)
    if instance.get("seed") != 31415:
        exc = ValueError(f"Unsupported seed: {instance.get('seed')!r}; only seed31415 is supported")
        _write_error(exc, "seed validation")
        print(f"[verifier] evaluator error (seed): {exc}", file=sys.stderr)
        return 1

    # 3. Bootstrap evaluator sys.path
    try:
        _bootstrap_evaluator(EVALUATOR_DIR)
    except Exception as exc:
        _write_error(exc, "evaluator bootstrap")
        print(f"[verifier] evaluator error (bootstrap): {exc}", file=sys.stderr)
        return 1

    # 4. Stage outputs in a temp directory
    staging = Path(tempfile.mkdtemp(prefix="asi-verifier-"))
    try:
        _stage_data_inputs(
            VERIFIER_DIR / "instance_data" / "data", staging,
            required=bool(instance.get("requires_instance_data")),
        )
        artifact_shas = _stage_outputs(WORKSPACE_DIR, output_specs, staging)

        # 5. Run scoring
        try:
            result = _run_scoring(EVALUATOR_DIR, staging, REFERENCE_DIR, instance)
        except Exception as exc:
            # Fail closed. `_evaluate_gates_and_scores` already converts every
            # scorer-level exception into a ScoreDetail carrying
            # scorer_internal_error, and a genuine submission failure (missing
            # or malformed outputs) is reported by the scorers as a low score —
            # not as a raised exception. Anything that escapes to here is
            # therefore an evaluator/runtime fault and must never be recorded
            # as a 0 score.
            _write_error(exc, "evaluator scoring")
            print(f"[verifier] evaluator error (scoring): {exc}", file=sys.stderr)
            return 1

        # An evaluator fault detected inside the scoring details: same rule.
        if result.get("scorer_internal_error"):
            failing = [
                detail.get("scorer_name")
                for detail in [*result.get("gate_results", []),
                               *result.get("score_results", [])]
                if isinstance(detail.get("details"), dict)
                and detail["details"].get("scorer_internal_error") is True
            ]
            exc = RuntimeError(
                f"scorer reported an internal error: {failing or 'unknown scorer'}"
            )
            _write_error(exc, "scorer_internal_error flag")
            print(f"[verifier] evaluator error (internal flag): {exc}", file=sys.stderr)
            return 1

        # 6. Normalize reward (divisor already applied by the framework)
        try:
            reward = _normalize_reward(result)
        except Exception as exc:
            _write_error(exc, "reward normalization")
            print(f"[verifier] evaluator error (normalization): {exc}", file=sys.stderr)
            return 1

        # 7. Preserve the exact scored files before the sandbox is destroyed.
        # BenchFlow mounts /logs/verifier on the host; /workspace is ephemeral.
        _persist_prediction_outputs(staging, artifact_shas)

        # 8. Write outputs (submission result → exit 0)
        _write_reward(reward, result, instance, artifact_shas, score_divisor)
        print(f"[verifier] reward={reward:.6f}", file=sys.stdout)
        return 0

    except Exception as exc:
        _write_error(exc, "verifier outer")
        print(f"[verifier] evaluator error (outer): {exc}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
