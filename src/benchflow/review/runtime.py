"""Separate-sandbox runtime and evidence preparation for rubric review."""

from __future__ import annotations

import json
import logging
import re
import secrets
import shlex
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchflow.acp.client import ACPError
from benchflow.review.config import ReviewParams
from benchflow.trajectories._capture import TrajectoryWriter
from benchflow.trajectories.types import redact_trajectory_obj, redact_trajectory_text

_MAX_SNAPSHOT_BYTES = 1024 * 1024 * 1024
_SECRET_NAMES = re.compile(
    r"(^|[._-])(auth|credential|secret|token|password|cookie|session|api[_-]?key)($|[._-])",
    re.IGNORECASE,
)
_PRIVATE_KEY_NAMES = re.compile(r"\.(pem|key|p12|pfx)$", re.IGNORECASE)
_EXCLUDED_TOP_LEVEL = {
    ".aws",
    ".cache",
    ".claude",
    ".codex",
    ".config",
    ".gemini",
    ".gnupg",
    ".npm",
    ".openclaw",
    ".ssh",
    ".benchflow-review",
}
_STRUCTURAL_MARKER_RE = re.compile(
    r"=====+\s*(?:BEGIN|END)\s+(?:EVIDENCE|SYSTEM|CRITERIA)\s*=====+",
    re.IGNORECASE,
)
_ROLE_TAG_RE = re.compile(r"</?(system|assistant|user|tool)(?=[\s>])", re.IGNORECASE)

logger = logging.getLogger(__name__)


def neutralize_structural_delimiters(text: str) -> str:
    """Defang prompt/control delimiters in attacker-controlled evidence text."""

    text = re.sub(r"`{3,}", "``\u200b`", text)
    text = _STRUCTURAL_MARKER_RE.sub("[neutralized-structural-marker]", text)
    return _ROLE_TAG_RE.sub(
        lambda match: match.group(0).replace("<", "<\u200b", 1), text
    )


def _sanitize_value(value: Any) -> Any:
    value = redact_trajectory_obj(value)
    if isinstance(value, dict):
        return {key: _sanitize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, str):
        return neutralize_structural_delimiters(value)
    return value


def _write_sanitized_jsonl(source: Path, destination: Path) -> None:
    rows: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append(json.dumps(_sanitize_value(payload), ensure_ascii=False))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def _jsonl_line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as source:
        return sum(1 for _ in source)


def _recover_final_reply(path: Path, *, after_line: int) -> str | None:
    """Recover a completed provider reply when an ACP wrapper drops it.

    Gemini CLI can emit an ACP 500 after LiteLLM has already captured a valid
    terminal response. Recovery is deliberately narrow: only new, successful,
    non-tool terminal responses from this prompt are eligible.
    """

    if not path.is_file():
        return None
    recovered: str | None = None
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if index < after_line or not line.strip():
            continue
        try:
            payload = json.loads(line)
            response = payload["response"]
            if response.get("status_code") != 200:
                continue
            choice = response["body"]["choices"][0]
            if choice.get("finish_reason") != "stop":
                continue
            message = choice.get("message", {})
            content = message.get("content")
            if (
                isinstance(content, str)
                and content.strip()
                and not message.get("tool_calls")
            ):
                recovered = content
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            continue
    return recovered


def _provider_tool_events(path: Path, *, after_line: int) -> list[dict[str, Any]]:
    """Return normalized tool calls from new successful provider responses.

    Some ACP agents render search calls as human-friendly titles that omit the
    exact file or directory arguments. The provider trace retains those
    arguments, so it is the authoritative source for checking the reviewer's
    evidence citations. Only tool calls emitted after this prompt started are
    included.
    """

    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if index < after_line or not line.strip():
            continue
        try:
            payload = json.loads(line)
            response = payload["response"]
            if response.get("status_code") != 200:
                continue
            tool_calls = response["body"]["choices"][0]["message"].get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                name = function.get("name")
                arguments = function.get("arguments")
                if not isinstance(name, str) or not name:
                    continue
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        continue
                if not isinstance(arguments, dict):
                    continue
                events.append(
                    {
                        "type": "provider_tool_call",
                        "name": name,
                        "arguments": arguments,
                    }
                )
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            continue
    return events


def _is_secret_path(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if relative.parts and relative.parts[0] in _EXCLUDED_TOP_LEVEL:
        return True
    name = path.name
    return name == ".env" or bool(
        _SECRET_NAMES.search(name) or _PRIVATE_KEY_NAMES.search(name)
    )


def _sanitize_tree(root: Path) -> None:
    """Remove links/credential files and sanitize every UTF-8 text leaf."""

    paths = sorted(root.rglob("*"), key=lambda path: len(path.parts), reverse=True)
    for path in paths:
        if path.is_symlink() or _is_secret_path(path, root):
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
            continue
        if not path.is_file():
            continue
        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        sanitized = neutralize_structural_delimiters(redact_trajectory_text(text))
        path.write_text(sanitized, encoding="utf-8")


def _safe_extract_regular_files(archive: Path, destination: Path) -> None:
    """Extract only bounded regular files/directories; never materialize links."""

    destination.mkdir(parents=True, exist_ok=True)
    total = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"unsafe workspace archive path: {member.name!r}")
            if member.issym() or member.islnk() or member.isdev():
                continue
            target = destination / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            total += member.size
            if total > _MAX_SNAPSHOT_BYTES:
                raise RuntimeError(
                    "review workspace snapshot exceeds the 1 GiB safety limit"
                )
            source = tar.extractfile(member)
            if source is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


@dataclass
class EvidenceSnapshot:
    """Host-side sanitized evidence ready for upload to the reviewer."""

    tempdir: tempfile.TemporaryDirectory[str]
    workspace: Path
    trajectory: Path
    artifacts: Path
    control: Path
    trajectory_files: list[str]
    control_token: str

    def cleanup(self) -> None:
        self.tempdir.cleanup()


async def capture_evidence_snapshot(solver: Any) -> EvidenceSnapshot:
    """Capture solver evidence without changing ownership of its workspace."""

    tempdir = tempfile.TemporaryDirectory(prefix="benchflow-review-evidence-")
    root = Path(tempdir.name)
    workspace = root / "workspace" / "root"
    trajectory = root / "trajectory"
    artifacts = root / "artifacts"
    control = root / "control"
    archive_remote = f"/tmp/benchflow-review-{secrets.token_hex(8)}.tar.gz"
    archive_host = root / "workspace.tar.gz"

    workspace_path = solver._agent_cwd
    excludes = " ".join(
        f"--exclude={shlex.quote('./' + name)}" for name in sorted(_EXCLUDED_TOP_LEVEL)
    )
    command = (
        f"tar -C {shlex.quote(workspace_path)} {excludes} "
        f"-czf {shlex.quote(archive_remote)} ."
    )
    result = await solver._env.exec(command, user="root", timeout_sec=300)
    if result.return_code != 0:
        tempdir.cleanup()
        raise RuntimeError(
            "failed to capture reviewer workspace snapshot: "
            f"{(result.stderr or result.stdout or '').strip()}"
        )
    try:
        await solver._env.download_file(archive_remote, archive_host)
    finally:
        await solver._env.exec(
            f"rm -f {shlex.quote(archive_remote)}", user="root", timeout_sec=30
        )
    _safe_extract_regular_files(archive_host, workspace)
    archive_host.unlink(missing_ok=True)
    _sanitize_tree(workspace)

    trajectory_files: list[str] = []
    solver_trajectory_dir = solver._require_rollout_dir() / "trajectory"
    for source in sorted(solver_trajectory_dir.glob("*.jsonl")):
        destination = trajectory / source.name
        _write_sanitized_jsonl(source, destination)
        if destination.stat().st_size:
            trajectory_files.append(source.name)
    if solver._trajectory:
        events_path = trajectory / "solver_events.jsonl"
        events_path.parent.mkdir(parents=True, exist_ok=True)
        events_path.write_text(
            "\n".join(
                json.dumps(_sanitize_value(event), ensure_ascii=False)
                for event in solver._trajectory
            )
            + "\n",
            encoding="utf-8",
        )
        if events_path.name not in trajectory_files:
            trajectory_files.append(events_path.name)

    source_artifacts = solver._require_rollout_dir() / "artifacts"
    if source_artifacts.is_dir():
        shutil.copytree(source_artifacts, artifacts, dirs_exist_ok=True, symlinks=True)
        _sanitize_tree(artifacts)
    else:
        artifacts.mkdir(parents=True)

    control_token = secrets.token_hex(16)
    control.mkdir(parents=True)
    (control / "integrity-token.txt").write_text(control_token, encoding="utf-8")
    return EvidenceSnapshot(
        tempdir=tempdir,
        workspace=workspace,
        trajectory=trajectory,
        artifacts=artifacts,
        control=control,
        trajectory_files=trajectory_files,
        control_token=control_token,
    )


@dataclass
class ReviewerTurn:
    reply: str
    events: list[dict[str, Any]]
    n_tool_calls: int
    evidence_trace: str
    recovered_from_provider_trace: bool = False


class IsolatedReviewerRuntime:
    """Reviewer agent running in its own no-network BenchFlow sandbox."""

    def __init__(
        self,
        solver: Any,
        *,
        harness: str,
        model: str | None,
        timeout_sec: float,
        review_dir: Path,
    ) -> None:
        self._solver = solver
        self.harness = harness
        self.model = model
        self.timeout_sec = timeout_sec
        self.review_dir = review_dir
        self._task_tmp: tempfile.TemporaryDirectory[str] | None = None
        self._rollout: Any = None
        self._tool_calls = 0
        self._provider_trace_recoveries = 0
        self._upstream_agent_env: dict[str, str] | None = None

    def _build_task(self) -> Path:
        self._task_tmp = tempfile.TemporaryDirectory(prefix="benchflow-review-task-")
        task = Path(self._task_tmp.name) / "isolated-reviewer"
        environment = task / "environment"
        environment.mkdir(parents=True)
        (task / "task.md").write_text(
            "---\n"
            "schema_version: '1.3'\n"
            "agent:\n"
            f"  timeout_sec: {int(self.timeout_sec)}\n"
            "environment:\n"
            "  network_mode: no-network\n"
            "  workdir: /review\n"
            "  cpus: 1\n"
            "  memory_mb: 2048\n"
            "  storage_mb: 4096\n"
            "---\n\n"
            "## prompt\n\n"
            "Wait for an explicit rubric-review request.\n",
            encoding="utf-8",
        )
        (environment / "Dockerfile").write_text(
            "FROM ubuntu:24.04\n\n"
            "RUN apt-get update -qq && apt-get install -y -qq curl ca-certificates "
            "python3 tar xz-utils && rm -rf /var/lib/apt/lists/*\n\n"
            "WORKDIR /review\n"
            "RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts /review\n",
            encoding="utf-8",
        )
        if self._solver._config.environment == "docker":
            # Docker drops NET_ADMIN by default. The root-owned local model
            # gateway needs outbound access while the reviewer UID is denied
            # all non-loopback egress with owner-matched iptables rules.
            (environment / "docker-compose.yaml").write_text(
                "services:\n  main:\n    cap_add:\n      - NET_ADMIN\n",
                encoding="utf-8",
            )
        return task

    async def start(self, snapshot: EvidenceSnapshot, rubric_path: Path) -> None:
        from benchflow.rollout import Rollout, RolloutConfig

        task = self._build_task()
        config = RolloutConfig(
            task_path=task,
            environment=self._solver._config.environment,
            sandbox_user="reviewer",
            sandbox_setup_timeout=self._solver._config.sandbox_setup_timeout,
            jobs_dir=self.review_dir / "runtime",
            job_name="isolated",
            rollout_name="reviewer",
            agent=self.harness,
            model=self.model,
            agent_env=self._solver._config.agent_env,
            timeout=int(self.timeout_sec),
            agent_idle_timeout=self._solver._config.agent_idle_timeout,
            usage_tracking=self._solver._config.usage_tracking,
            self_gen_no_internet=True,
            review=ReviewParams(enabled=False),
        )
        self._rollout = await Rollout.create(config)
        await self._rollout.setup()
        await self._rollout.start()
        await self._rollout.install_agent()
        self._upstream_agent_env = dict(self._rollout._agent_env)

        env = self._rollout.env
        await env.upload_dir(snapshot.workspace, "/review/workspace/root")
        await env.upload_dir(snapshot.trajectory, "/review/trajectory")
        await env.upload_dir(snapshot.artifacts, "/review/artifacts")
        await env.upload_dir(snapshot.control, "/review/control")
        await env.upload_file(rubric_path, "/review/rubric.json")
        lock = await env.exec(
            "chown -R root:root /review/workspace /review/trajectory "
            "/review/artifacts /review/control /review/rubric.json && "
            "find /review/workspace /review/trajectory /review/artifacts "
            "/review/control -type d -exec chmod 0555 {} + && "
            "find /review/workspace /review/trajectory /review/artifacts "
            "/review/control -type f -exec chmod 0444 {} + && "
            "chmod 0444 /review/rubric.json && chmod 0555 /review",
            user="root",
            timeout_sec=60,
        )
        if lock.return_code != 0:
            raise RuntimeError(
                "failed to lock reviewer evidence read-only: "
                f"{(lock.stderr or lock.stdout or '').strip()}"
            )
        await self._rollout.connect()

    async def fresh_session(self) -> None:
        await self._rollout.disconnect()
        if self._upstream_agent_env is not None:
            # ``connect()`` rewrites provider credentials to the local proxy
            # master key. A fresh session must resolve/reuse the gateway from
            # the original upstream environment, never treat that master key
            # as an upstream provider credential.
            self._rollout._agent_env = dict(self._upstream_agent_env)
        await self._rollout.connect()

    async def prompt(self, message: str) -> ReviewerTurn:
        before_events = len(self._rollout.trajectory)
        before_tools = self._rollout._n_tool_calls
        llm_trace = (
            self._rollout._require_rollout_dir() / "trajectory" / "llm_trajectory.jsonl"
        )
        before_llm_lines = _jsonl_line_count(llm_trace)
        recovered_reply: str | None = None
        try:
            await self._rollout.execute(prompts=[message])
        except ACPError:
            self._rollout._capture_partial_acp_trajectory()
            recovered_reply = _recover_final_reply(
                llm_trace,
                after_line=before_llm_lines,
            )
            if recovered_reply is None:
                raise
            self._provider_trace_recoveries += 1
            logger.warning(
                "Recovered completed reviewer reply from the provider trace "
                "after an ACP wrapper error"
            )
        events = self._rollout.trajectory[before_events:]
        n_tool_calls = max(0, self._rollout._n_tool_calls - before_tools)
        self._tool_calls += n_tool_calls
        reply = recovered_reply or "\n".join(
            str(event["text"])
            for event in events
            if event.get("type") == "agent_message" and event.get("text")
        )
        evidence_events = [
            event for event in events if event.get("type") == "tool_call"
        ]
        evidence_events.extend(
            _provider_tool_events(llm_trace, after_line=before_llm_lines)
        )
        evidence_trace = json.dumps(
            _sanitize_value(evidence_events), ensure_ascii=False, default=str
        )
        return ReviewerTurn(
            reply=reply,
            events=events,
            n_tool_calls=n_tool_calls,
            evidence_trace=evidence_trace,
            recovered_from_provider_trace=recovered_reply is not None,
        )

    async def close(self) -> dict[str, Any]:
        if self._rollout is None:
            if self._task_tmp is not None:
                self._task_tmp.cleanup()
            return {}
        sandbox_id = self._rollout._current_sandbox_id()
        try:
            await self._rollout.disconnect()
        finally:
            await self._rollout.cleanup()
        TrajectoryWriter(self.review_dir / "reviewer_trajectory.jsonl").write_final(
            self._rollout.trajectory
        )
        metadata = {
            "harness": self.harness,
            "model": self.model,
            "agent_name": self._rollout._agent_name,
            "n_events": len(self._rollout.trajectory),
            "n_tool_calls": self._tool_calls,
            "provider_trace_recoveries": self._provider_trace_recoveries,
            "usage": self._rollout._usage_metrics,
            "sandbox_id": sandbox_id,
            "runtime_artifacts": "review/runtime/isolated/reviewer",
        }
        if self._task_tmp is not None:
            self._task_tmp.cleanup()
        return metadata
