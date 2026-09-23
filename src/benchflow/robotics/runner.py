"""Physical-trial orchestration around the BenchFlow SDK, not a new agent loop."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib.metadata
import json
import os
import shutil
import time
import uuid
from pathlib import Path

from .bridge import HarnessTransport, TrialBridge, TrialLease, write_json
from .recording import RecordingSidecar

PROBE_PROMPT = """This is a read-only connectivity and vision probe, not a manipulation trial.
Read /app/setup.json. Run `python /app/robot.py observe`, open both returned
camera images using your image-viewing tool, and describe what you can see
and what remains uncertain. Then run `python /app/robot.py finish`.
Do not move any joint or the gripper. Motion is disabled by the host bridge.
"""


def digest_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            digest.update(path.relative_to(root).as_posix().encode() + b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


ARM_FIELDS = ("robot_id", "socket_path", "frames_root", "episode_log")


def load_setup(path: Path) -> dict:
    """Read a setup file. Either the legacy single-arm layout (robot_id,
    socket_path, frames_root, episode_log at the top level) or a multi-arm
    layout with an ``arms`` mapping of name -> those fields plus an optional
    ``offset_m`` (that arm's base expressed in the primary arm's frame).
    After loading, ``setup["arms"]`` and ``setup["primary_arm"]`` are always
    present and the legacy top-level fields describe the primary arm."""
    setup = json.loads(path.read_text())
    arms = setup.get("arms")
    if arms is not None:
        if not isinstance(arms, dict) or not arms:
            raise ValueError("Setup arms must be a nonempty mapping")
        for name, spec in arms.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError("Arm names must be identifiers")
            missing = set(ARM_FIELDS) - set(spec)
            if missing:
                raise ValueError(f"Arm {name!r} missing fields: {sorted(missing)}")
            if not spec["robot_id"]:
                raise ValueError("Robot IDs must be nonempty")
        primary = setup.get("primary_arm") or next(iter(arms))
        if primary not in arms:
            raise ValueError("primary_arm must name one of the arms")
        for field in ARM_FIELDS:
            setup.setdefault(field, arms[primary][field])
    required = {"setup_id", "arm_type", "cameras", "public_facts", *ARM_FIELDS}
    if not required <= setup.keys():
        raise ValueError(f"Setup missing fields: {sorted(required - setup.keys())}")
    if setup["arm_type"] != "metal" or not {"wrist", "side"} <= set(setup["cameras"]):
        raise ValueError(
            "This first adapter requires a Metal arm with wrist and side cameras"
        )
    if not setup["setup_id"] or not setup["robot_id"]:
        raise ValueError("Setup and robot IDs must be nonempty")
    if arms is None:
        setup["arms"] = {"arm": {field: setup[field] for field in ARM_FIELDS}}
        setup["primary_arm"] = "arm"
    else:
        setup["primary_arm"] = primary
    return setup


def task_prompt(task_path: Path, fallback: str) -> str:
    """The agent prompt is the task.md ``## prompt`` section when present, so host
    and container trials read the same frozen task text."""
    task_md = task_path / "task.md"
    if task_md.exists():
        text = task_md.read_text()
        marker = "## prompt"
        if marker in text:
            body = text.split(marker, 1)[1]
            body = body.split("\n## ", 1)[0]
            if body.strip():
                return body.strip() + "\n"
    return fallback


def provider_environment(env_file: Path | None) -> dict[str, str]:
    """Load only into the host process; never serialize keys into trial manifests."""
    allowed_prefixes = ("OPENAI_", "ANTHROPIC_", "AWS_", "CLAUDE_CODE_")
    candidates = dict(os.environ)
    if env_file is not None:
        from dotenv import dotenv_values

        candidates.update(
            {k: v for k, v in dotenv_values(env_file).items() if v is not None}
        )
    return {k: v for k, v in candidates.items() if k.startswith(allowed_prefixes)}


async def run_trial(
    *,
    setup_path: Path,
    task_path: Path,
    output_root: Path,
    agent: str,
    model: str,
    reasoning_effort: str,
    reset_id: str,
    operator: str,
    env_file: Path | None = None,
    allow_motion: bool = False,
    smoke: bool = False,
    probe: bool = False,
    backend: str = "docker",
    timeout: int = 1800,
    bind: str = "127.0.0.1",
    advertised_host: str = "host.docker.internal",
    lock_root: Path | None = None,
) -> Path:
    if not reset_id.strip() or not operator.strip():
        raise ValueError(
            "A physical reset ID and operator are required for every trial"
        )
    if not smoke and (not model.strip() or not reasoning_effort.strip()):
        raise ValueError("Exact model and reasoning effort are required")
    if smoke and probe:
        raise ValueError("Select either a hardware smoke or an agent probe")
    if (smoke or probe) and allow_motion:
        raise ValueError("Read-only checks cannot enable motion")
    if not smoke and not probe and not allow_motion:
        raise ValueError("Physical agent trials require explicit --execute")
    if backend not in {"docker", "host"}:
        raise ValueError("Backend must be docker or host")
    if not smoke and backend == "docker":
        from benchflow.sandbox.docker import DockerSandbox

        DockerSandbox.preflight()
        if bind == "127.0.0.1":
            raise ValueError(
                "Docker needs a reachable host bridge; specify its --bind address"
            )
    setup = load_setup(setup_path)
    scenario = json.loads((task_path / "scenario.json").read_text())
    trial_id = (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:10]
    )
    output = output_root.resolve() / trial_id
    output.mkdir(parents=True, exist_ok=False)
    arms = setup["arms"]
    primary = setup["primary_arm"]
    transports = {
        name: HarnessTransport(Path(spec["socket_path"])) for name, spec in arms.items()
    }
    recorder = RecordingSidecar(
        output / "cameras", setup["cameras"],
        fps=float(os.environ.get("BENCHFLOW_RECORD_FPS", "2")),
    )
    bridge = TrialBridge(
        output=output,
        arms={
            name: {
                "transport": transports[name],
                "frames_root": Path(spec["frames_root"]),
                "offset_m": spec.get("offset_m", (0.0, 0.0, 0.0)),
            }
            for name, spec in arms.items()
        },
        robot_id=setup["arm_type"],
        allow_motion=allow_motion,
        timeout=timeout + 600,
        recording_healthy=recorder.healthy,
    )
    manifest = {
        "schema_version": 1,
        "trial_id": trial_id,
        "kind": "read_only_smoke"
        if smoke
        else "agent_probe"
        if probe
        else "physical_trial",
        "setup_id": setup["setup_id"],
        "robot_id": setup["robot_id"],
        "reset_id": reset_id,
        "operator": operator,
        "task": task_path.name,
        "task_sha256": digest_tree(task_path),
        "setup_sha256": hashlib.sha256(setup_path.read_bytes()).hexdigest(),
        "expected": scenario["expected"],
        "agent": agent,
        "backend": backend,
        "requested_model": model,
        "requested_reasoning_effort": reasoning_effort,
        "agent_timeout_s": timeout,
        "benchflow_version": importlib.metadata.version("benchflow"),
        "adapter_sha256": digest_tree(Path(__file__).parent),
        "started_utc_epoch": time.time(),
        "status": "preparing",
        "assessment": "pending",
        "public_facts": setup["public_facts"],
        "arms": {name: spec["robot_id"] for name, spec in arms.items()},
        "primary_arm": primary,
    }
    write_json(output / "manifest.json", manifest)
    shutil.copyfile(task_path / "task.md", output / "task.md")
    if probe:
        (output / "probe-prompt.md").write_text(PROBE_PROMPT)
    episodes = {name: Path(spec["episode_log"]) for name, spec in arms.items()}
    wall_start = time.monotonic()
    episode_starts = {name: path.stat().st_size for name, path in episodes.items()}
    result = None
    lease_root = lock_root or Path.home() / ".cache/benchflow/robot-leases"
    with contextlib.ExitStack() as leases:
        for spec in arms.values():
            leases.enter_context(TrialLease(lease_root, spec["robot_id"], trial_id))
        try:
            # Start footage before any trial inspection or agent bootstrap.
            await asyncio.to_thread(recorder.start)
            initial_states = {}
            for name in arms:
                initial = await asyncio.to_thread(transports[name], "observe", [])
                if initial.get("arm") != setup["arm_type"] or not initial.get("ok"):
                    raise RuntimeError(
                        f"Wrong arm or failed initial observation ({name})"
                    )
                if allow_motion and (not initial.get("armed") or initial.get("faults")):
                    raise RuntimeError(
                        f"Commissioned arm must be armed and fault-free ({name})"
                    )
                bridge._images(initial, name)
                bridge.note_state(name, initial)
                initial_states[name] = {k: v for k, v in initial.items() if k != "text"}
            write_json(output / "initial-state.json", initial_states[primary])
            if len(arms) > 1:
                write_json(output / "initial-states.json", initial_states)
            port = bridge.serve(bind)
            manifest.update(status="running", bridge_port=port)
            write_json(output / "manifest.json", manifest)
            if smoke:
                bridge.activate()
                # Exercise HTTP authentication/serialization, not just direct Python calls.
                import urllib.request

                def read_only_request():
                    req = urllib.request.Request(
                        f"http://127.0.0.1:{port}/command",
                        json.dumps(
                            {
                                "request_id": "smoke-observe",
                                "command": "observe",
                                "args": [],
                            }
                        ).encode(),
                        {
                            "Authorization": "Bearer " + bridge.token,
                            "Content-Type": "application/json",
                        },
                    )
                    with urllib.request.build_opener(
                        urllib.request.ProxyHandler({})
                    ).open(req, timeout=120) as response:
                        return json.load(response)

                receipt = await asyncio.to_thread(read_only_request)
                if not receipt.get("ok") or len(receipt.get("images", [])) != 2:
                    raise RuntimeError("Read-only bridge check failed")
                await asyncio.sleep(3)
                manifest["status"] = "smoke_passed"
            else:
                import benchflow as bf
                from benchflow import RolloutConfig
                from benchflow.models import RolloutResult
                from benchflow.usage_tracking import UsageTrackingConfig

                async def connect(environment):
                    connection = output / ".connection.json"
                    write_json(
                        connection,
                        {
                            "url": f"http://{advertised_host}:{port}",
                            "token": bridge.token,
                        },
                    )
                    connection.chmod(0o600)
                    facts = output / ".public-setup.json"
                    write_json(
                        facts, {"setup_id": setup["setup_id"], **setup["public_facts"]}
                    )
                    try:
                        await environment.upload_file(
                            connection, "/app/robot-connection.json"
                        )
                        await environment.upload_file(facts, "/app/setup.json")
                    finally:
                        connection.unlink(missing_ok=True)
                        facts.unlink(missing_ok=True)
                    # Readable by the dedicated agent user; expires at bridge.close().
                    await environment.exec(
                        "chmod 644 /app/robot-connection.json /app/setup.json",
                        user="root",
                    )
                    bridge.activate()

                config = RolloutConfig(
                    task_path=task_path,
                    agent=agent,
                    model=model,
                    prompts=[PROBE_PROMPT] if probe else None,
                    reasoning_effort=reasoning_effort,
                    environment="docker",
                    agent_env=provider_environment(env_file),
                    jobs_dir=output / "benchflow",
                    job_name=trial_id,
                    rollout_name="agent",
                    timeout=timeout,
                    concurrency=1,
                    skill_mode="no-skill",
                    usage_tracking=UsageTrackingConfig("required"),
                    sandbox_setup_timeout=600,
                    pre_agent_hooks=[connect],
                    skip_verify=True,
                )
                if backend == "host":
                    from .host import run_host_agent
                    from .tasks import CONTROL_INSTRUCTIONS

                    result = await run_host_agent(
                        output=output,
                        workspace=Path("/tmp") / ("benchflow-host-" + trial_id),
                        bridge=bridge,
                        setup=setup,
                        task_path=task_path,
                        agent=agent,
                        model=model,
                        effort=reasoning_effort,
                        timeout=timeout,
                        prompt=PROBE_PROMPT
                        if probe
                        else task_prompt(
                            task_path, scenario["instruction"] + CONTROL_INSTRUCTIONS
                        ),
                        provider_env=provider_environment(env_file),
                    )
                else:
                    result = await bf.run(config)
                if not isinstance(result, RolloutResult):
                    raise TypeError("Unexpected BenchFlow SDK result type")
                manifest["status"] = (
                    ("probe_completed" if probe else "awaiting_assessment")
                    if result.success
                    else "agent_error"
                )
                fields = (
                    "agent",
                    "agent_name",
                    "model",
                    "n_tool_calls",
                    "n_input_tokens",
                    "n_output_tokens",
                    "n_cache_read_tokens",
                    "n_cache_creation_tokens",
                    "total_tokens",
                    "cost_usd",
                    "usage_source",
                    "price_source",
                    "usage_details",
                    "error",
                    "error_category",
                    "partial_trajectory",
                    "trajectory_source",
                )
                metrics = {key: getattr(result, key, None) for key in fields}
                if backend == "host":
                    host_summary = json.loads(
                        (output / "host-agent-summary.json").read_text()
                    )
                    metrics.update(
                        usage_source=host_summary["usage_source"],
                        agent_wall_s=host_summary["agent_wall_s"],
                        price_source="agent_native_cli"
                        if result.cost_usd is not None
                        else None,
                    )
                write_json(output / "metrics.json", metrics)
        except BaseException as exc:
            manifest.update(
                status="interrupted"
                if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError))
                else "infrastructure_error",
                error_type=type(exc).__name__,
            )
            raise
        finally:
            await asyncio.to_thread(bridge.close)
            # Final evidence is taken after the agent capability has been revoked.
            try:
                final_states = {}
                for name in arms:
                    final = await asyncio.to_thread(transports[name], "observe", [])
                    bridge.count += 1
                    bridge._images(final, name)
                    final_states[name] = {k: v for k, v in final.items() if k != "text"}
                write_json(output / "final-state.json", final_states[primary])
                if len(arms) > 1:
                    write_json(output / "final-states.json", final_states)
            except Exception as exc:
                manifest["final_observation_error"] = type(exc).__name__
            manifest["execution_wall_s"] = time.monotonic() - wall_start
            capture = await asyncio.to_thread(recorder.stop)
            manifest["footage_complete"] = capture.get("complete", False)
            manifest["video_exports_complete"] = all(
                capture.get("exports", {}).get(name, {}).get("ok", False)
                for name in setup["cameras"]
            )
            if (
                smoke
                and manifest["status"] == "smoke_passed"
                and (
                    not manifest["footage_complete"]
                    or not manifest["video_exports_complete"]
                    or "final_observation_error" in manifest
                )
            ):
                manifest["status"] = "smoke_failed"
            manifest["bridge_halted"] = bridge.halted
            manifest["bridge_halt_reason"] = bridge.halt_reason
            manifest["finished_utc_epoch"] = time.time()
            for name, path in episodes.items():
                if not path.exists():
                    continue
                target = (
                    "encoders.jsonl" if name == primary else f"encoders-{name}.jsonl"
                )
                with (
                    path.open("rb") as source,
                    (output / target).open("wb") as destination,
                ):
                    source.seek(episode_starts[name])
                    shutil.copyfileobj(source, destination)
            write_json(output / "manifest.json", manifest)
    if smoke and manifest["status"] != "smoke_passed":
        raise RuntimeError(
            f"Read-only smoke failed; inspect {output / 'manifest.json'}"
        )
    return output


def score_trial(
    output: Path,
    *,
    placements: dict[str, str],
    reviewer: str,
    interventions: int,
    cups_upright: bool,
    evidence: str,
    external_interruption: str | None = None,
) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["kind"] != "physical_trial" or manifest["status"] in {
        "preparing",
        "running",
    }:
        raise ValueError("Only finished physical trials can be scored")
    expected = manifest["expected"]
    if set(placements) != set(expected):
        raise ValueError("Give an observed outcome for every target block")
    if not reviewer.strip() or not evidence.strip() or interventions < 0:
        raise ValueError(
            "Reviewer, evidence references, and nonnegative interventions are required"
        )
    if (output / "assessment.json").exists():
        raise FileExistsError(
            "An assessment already exists; preserve it rather than silently replacing it"
        )
    accuracy = sum(placements[k] == target for k, target in expected.items()) / len(
        expected
    )
    metrics_path = output / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    valid = (
        manifest["footage_complete"]
        and not external_interruption
        and not manifest.get("final_observation_error")
        and manifest["status"] != "infrastructure_error"
        and manifest.get("bridge_halt_reason")
        not in {"recording_lost", "uncertain_outcome"}
        and metrics.get("usage_source")
        in {"provider_response", "agent_native_acp", "agent_native_cli"}
        and metrics.get("error_category") not in {"api_error", "suspected_api_error"}
    )
    assessment = {
        "reviewer": reviewer,
        "evidence": evidence,
        "observed": placements,
        "interventions": interventions,
        "external_interruption": external_interruption,
        "cups_upright": cups_upright,
        "object_accuracy": accuracy,
        "task_success": accuracy == 1 and cups_upright,
        "autonomous_success": accuracy == 1 and cups_upright and interventions == 0,
        "benchmark_valid": bool(valid),
        "scored_utc_epoch": time.time(),
    }
    write_json(output / "assessment.json", assessment)
    if valid:
        (output / "reward.txt").write_text(
            str(float(assessment["autonomous_success"])) + "\n"
        )
    return assessment


def report_trials(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        row = {
            k: manifest.get(k)
            for k in (
                "trial_id",
                "kind",
                "setup_id",
                "reset_id",
                "task",
                "agent",
                "backend",
                "requested_model",
                "status",
                "execution_wall_s",
                "footage_complete",
            )
        }
        for name in ("metrics", "host-derived-metrics", "assessment"):
            source = path.parent / f"{name}.json"
            if source.exists():
                row.update(json.loads(source.read_text()))
        row["directory"] = str(path.parent)
        rows.append(row)
    return rows
