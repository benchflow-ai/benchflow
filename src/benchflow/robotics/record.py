"""The embodied trial record: provenance, honest outcome and trace index.

``write_trial_record(trial_dir)`` reads what a trial left on disk and writes
two files beside the trial's normal BenchFlow artifacts:

``trial-record.json``
    Embodiment and restoration boundary, provenance (which runtime drove the
    arm, arm and camera mapping, controller identity), the execution and
    assessment states (the execution carries its raw ``evidence``: host agent
    exit code and wall time, time budget), and the synchronized trace index.
``result.json``
    A BenchFlow result for scored physical trials, so ``collect_metrics``,
    evaluation resume and rescoring see one result per trial (the nested SDK
    rollout becomes an artifact of it). It carries ``assessment``; the shared
    scoring helpers keep it unscored until a reviewer has assessed it.

It runs in the runner's ``finally`` block and offline on saved trials. It
never modifies a recorded stream.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from benchflow._utils.scoring import ASSESSED_STATUSES
from benchflow.embodiment import TRIAL_RECORD_FILENAME, Embodiment

from .bridge import write_json
from .outcome import assessment_outcome, execution_outcome
from .trace import build_trace_index, utc_iso

RECORD_SCHEMA_VERSION = 1
#: Who drove the embodiment during the trial.
RUNTIMES = frozenset({"benchflow", "inspect_robots", "raw_adapter", "unknown"})
CAMERA_MOUNTS = frozenset({"wrist", "fixed"})
PHYSICAL = Embodiment(kind="physical", world_restore=False, action_replay=False)
_ERROR_EXECUTIONS = frozenset(
    {"agent_error", "timed_out", "cancelled", "infrastructure_error", "unfinalized"}
)


def _source(url: Any) -> str | None:
    """Camera feed identity without credentials or query strings."""
    if not isinstance(url, str):
        return None
    parts = urlsplit(url)
    host = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{host}{port}{parts.path}" if parts.scheme else parts.path


def camera_mapping(
    cameras: Mapping[str, Any],
    declared: Mapping[str, Any] | None,
    arms: Mapping[str, Any],
) -> dict[str, Any]:
    """Map each logical camera to its feed, mount and arm.

    Nothing is inferred from a camera's name: on some setups ``wrist``
    has been a different camera.
    Undeclared cameras are recorded with ``mount``/``arm`` null. Logical
    cameras that share one feed are reported, since two such "views" are one.
    """
    declared = declared or {}
    if not isinstance(declared, Mapping):
        raise ValueError("camera_mapping must map camera names to mounts")
    unknown = set(declared) - set(cameras)
    if unknown:
        raise ValueError(f"camera_mapping names unknown cameras: {sorted(unknown)}")
    mapping: dict[str, Any] = {}
    for name, url in cameras.items():
        spec = declared.get(name)
        if spec is not None:
            if not isinstance(spec, Mapping) or spec.get("mount") not in CAMERA_MOUNTS:
                raise ValueError(f"camera {name!r} mount must be wrist or fixed")
            arm = spec.get("arm")
            if spec["mount"] == "wrist" and arm not in arms:
                raise ValueError(f"wrist camera {name!r} must name one of the arms")
            if spec["mount"] == "fixed" and arm is not None:
                raise ValueError(f"fixed camera {name!r} cannot name an arm")
        mapping[name] = {
            "source": _source(url),
            "mount": spec.get("mount") if spec else None,
            "arm": spec.get("arm") if spec else None,
            "declared": spec is not None,
        }
    feeds: dict[str, list[str]] = {}
    for name, entry in mapping.items():
        if entry["source"]:
            feeds.setdefault(entry["source"], []).append(name)
    return {
        "cameras": mapping,
        "shared_feeds": sorted(names for names in feeds.values() if len(names) > 1),
    }


def controller_identity(declared: Any) -> dict[str, Any]:
    """The arm controller's declared identity; undeclared stays unknown."""
    if declared is None:
        return {"declared": False, "name": None, "revision": None, "patches": None}
    if not isinstance(declared, Mapping):
        raise ValueError("controller must be a mapping with name and revision")
    name, revision = declared.get("name"), declared.get("revision")
    patches = declared.get("patches", [])
    if not isinstance(name, str) or not name or not isinstance(revision, str):
        raise ValueError("controller needs a nonempty name and a revision string")
    if not isinstance(patches, list) or not all(isinstance(p, str) for p in patches):
        raise ValueError("controller.patches must be a list of strings")
    return {"declared": True, "name": name, "revision": revision, "patches": patches}


def runtime_identity(name: str, **fields: Any) -> dict[str, Any]:
    if name not in RUNTIMES:
        raise ValueError(f"runtime must be one of {sorted(RUNTIMES)}, got {name!r}")
    return {"name": name, **fields}


def setup_provenance(setup: Mapping[str, Any]) -> dict[str, Any]:
    """Arm, camera and controller provenance from a loaded setup."""
    arms = {
        name: {"robot_id": spec["robot_id"], "primary": name == setup["primary_arm"]}
        for name, spec in setup["arms"].items()
    }
    return {
        "arms": arms,
        **camera_mapping(setup["cameras"], setup.get("camera_mapping"), arms),
        "controller": controller_identity(setup.get("controller")),
    }


def legacy_provenance(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Provenance recoverable from a manifest written before provenance existed."""
    version = manifest.get("benchflow_version")
    arms = manifest.get("arms")
    if not isinstance(arms, Mapping) and manifest.get("robot_id"):
        arms = {"arm": manifest["robot_id"]}
    primary = manifest.get("primary_arm") or (next(iter(arms)) if arms else None)
    return {
        "runtime": runtime_identity(
            "benchflow" if version else "unknown",
            version=version,
            adapter="benchflow.robotics" if version else None,
            adapter_sha256=manifest.get("adapter_sha256"),
            backend=manifest.get("backend"),
        ),
        "arms": {
            name: {"robot_id": robot_id, "primary": name == primary}
            for name, robot_id in (arms or {}).items()
        },
        "cameras": None,
        "shared_feeds": None,
        "controller": controller_identity(None),
    }


def inspect_robots_provenance(eval_spec: Mapping[str, Any]) -> dict[str, Any]:
    """Provenance of a trial driven by Inspect Robots, from its EvalLog ``eval``.

    ``resettable`` in Inspect Robots means the embodiment can drive to a home
    pose, possibly after human confirmation. It is not a world restore, so a
    physical (``is_simulated: false``) embodiment stays non-restorable.
    """
    info = eval_spec.get("embodiment_info")
    info = info if isinstance(info, Mapping) else {}
    simulated = info.get("is_simulated")
    policy = eval_spec.get("policy_config")
    policy = policy if isinstance(policy, Mapping) else {}
    return {
        "runtime": runtime_identity(
            "inspect_robots",
            version=eval_spec.get("inspect_robots_version"),
            git_commit=eval_spec.get("git_commit"),
        ),
        "embodiment": {
            "name": eval_spec.get("embodiment"),
            "kind": "simulated"
            if simulated is True
            else "physical"
            if simulated is False
            else None,
            "capabilities": sorted(info.get("capabilities") or []),
            "control_hz": info.get("control_hz"),
            "environment_id": info.get("environment_id"),
            "environment_revision": info.get("environment_revision"),
        },
        "policy": {
            "name": eval_spec.get("policy"),
            "requested_model": policy.get("model"),
            "reasoning_effort": policy.get("effort"),
        },
    }


def _read(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    return data if isinstance(data, dict) else None


def execution_evidence(trial: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Recorded facts shown beside the execution status, never used to set it.

    A host agent killed at its deadline was recorded by older runners as
    ``agent_error`` with no error text; only its exit code and wall time,
    read against the time budget, show what happened. They are copied as
    written: ``host_agent`` from ``host-agent-summary.json`` (null when the
    trial has none) and ``agent_timeout_s`` from the manifest.
    """
    summary = _read(trial / "host-agent-summary.json")
    host_agent = None
    if summary is not None:
        host_agent = {
            "source": "host-agent-summary.json",
            "exit_code": summary.get("exit_code"),
            "agent_wall_s": summary.get("agent_wall_s"),
        }
        if "agent_timeout_s" in summary:
            host_agent["agent_timeout_s"] = summary["agent_timeout_s"]
    return {
        "host_agent": host_agent,
        "agent_timeout_s": manifest.get("agent_timeout_s"),
    }


def build_trial_record(trial: Path) -> dict[str, Any]:
    """Assemble the trial record from ``trial``'s files, reading only."""
    trial = Path(trial)
    manifest = _read(trial / "manifest.json")
    if manifest is None:
        raise FileNotFoundError(f"{trial / 'manifest.json'} is missing")
    metrics = _read(trial / "metrics.json")
    review = _read(trial / "assessment.json")
    provenance = manifest.get("provenance") or legacy_provenance(manifest)
    arms = provenance.get("arms") or {}
    primary = next((name for name, arm in arms.items() if arm.get("primary")), None)
    index = build_trace_index(
        trial,
        cameras=list(provenance["cameras"]) if provenance.get("cameras") else None,
        arms=arms,
        primary_arm=primary,
        clock_anchor=manifest.get("clock_anchor"),
        expect_agent=manifest.get("kind") != "read_only_smoke",
    )
    execution = execution_outcome(manifest, metrics=metrics, actions=index["actions"])
    execution["evidence"] = execution_evidence(trial, manifest)
    assessment = assessment_outcome(manifest, review, execution=execution["status"])
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "kind": "benchflow-embodied-trial",
        "trial_id": manifest.get("trial_id", trial.name),
        "trial_kind": manifest.get("kind"),
        "task": manifest.get("task"),
        "reset_id": manifest.get("reset_id"),
        "embodiment": PHYSICAL.restoration_record(),
        "provenance": {
            **provenance,
            "policy": provenance.get("policy")
            or {
                "agent": manifest.get("agent"),
                "requested_model": manifest.get("requested_model"),
                "reasoning_effort": manifest.get("requested_reasoning_effort"),
            },
            "task_sha256": manifest.get("task_sha256"),
            "setup": {
                "id": manifest.get("setup_id"),
                "sha256": manifest.get("setup_sha256"),
            },
        },
        "outcome": {"execution": execution, "assessment": assessment},
        **index,
    }


def trial_result(record: Mapping[str, Any], trial: Path) -> dict[str, Any] | None:
    """BenchFlow result for a scored physical trial; None for probes/smokes."""
    if record["trial_kind"] != "physical_trial":
        return None
    manifest = _read(trial / "manifest.json") or {}
    metrics = _read(trial / "metrics.json") or {}
    execution = record["outcome"]["execution"]
    assessment = record["outcome"]["assessment"]
    reward = assessment["reward"] if assessment["status"] in ASSESSED_STATUSES else None
    error = None
    if execution["status"] in _ERROR_EXECUTIONS:
        error = execution["error"] or f"physical trial {execution['status']}"
    return {
        "task_name": record["task"],
        "rollout_name": record["trial_id"],
        "kind": "embodied-trial",
        "embodiment": record["embodiment"]["kind"],
        "agent": manifest.get("agent"),
        "model": manifest.get("requested_model"),
        "rewards": {"reward": reward} if reward is not None else None,
        "assessment": {
            "status": assessment["status"],
            "reason": assessment["reason"],
        },
        "execution": execution,
        "error": error,
        "error_category": execution["error_category"],
        "verifier_error": None,
        "n_tool_calls": metrics.get("n_tool_calls") or 0,
        "agent_result": {
            key: metrics.get(key)
            for key in (
                "n_input_tokens",
                "n_output_tokens",
                "n_cache_read_tokens",
                "n_cache_creation_tokens",
                "total_tokens",
                "cost_usd",
                "usage_source",
            )
        },
        "started_at": utc_iso(manifest.get("started_utc_epoch")),
        "finished_at": utc_iso(manifest.get("finished_utc_epoch")),
        "trial_record": TRIAL_RECORD_FILENAME,
    }


def write_trial_record(trial: Path) -> dict[str, Any]:
    """Write ``trial-record.json`` (and ``result.json`` for scored trials)."""
    trial = Path(trial)
    record = build_trial_record(trial)
    record["generated_utc"] = utc_iso(time.time())
    write_json(trial / TRIAL_RECORD_FILENAME, record)
    result = trial_result(record, trial)
    if result is not None:
        write_json(trial / "result.json", result)
    return record
