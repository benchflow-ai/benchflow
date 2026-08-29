"""Lifecycle orchestration for the always-present LLM trajectory artifact."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shlex
import tempfile
from contextlib import suppress
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from benchflow.agents.env import uses_native_subscription_auth
from benchflow.agents.registry import AGENTS
from benchflow.trajectories.llm_capture_manifest import (
    LLM_TRAJECTORY_FILENAME,
    AuthMode,
    CaptureFidelity,
    CaptureSource,
    CaptureStatus,
    LLMRoleCapture,
    initialize_llm_trajectory_artifacts,
    write_llm_trajectory_manifest,
)
from benchflow.trajectories.llm_capture_records import (
    CaptureTarget as _CaptureTarget,
)
from benchflow.trajectories.llm_capture_records import (
    NativeCaptureBundle as _NativeCaptureBundle,
)
from benchflow.trajectories.llm_capture_records import (
    assemble_capture,
    load_provider_wire_records,
    role_captures_for_targets,
    write_exchange_records,
)
from benchflow.trajectories.native_capture_parsers import (
    parse_claude_raw_capture,
    parse_claude_sessions,
    parse_codex_sessions,
    project_acp_trajectory,
)
from benchflow.trajectories.types import redact_trajectory_text

logger = logging.getLogger(__name__)

_REMOTE_CAPTURE_PREFIX = "/tmp/benchflow-llm-capture-"
_OTEL_SINK_SOURCE = r"""
import { createServer } from 'node:http';
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const [outputDir, portFile] = process.argv.slice(2);
mkdirSync(outputDir, { recursive: true });
let sequence = 0;
const server = createServer((request, response) => {
  const chunks = [];
  let size = 0;
  request.on('data', chunk => {
    size += chunk.length;
    if (size <= 64 * 1024 * 1024) chunks.push(chunk);
  });
  request.on('end', () => {
    if (size <= 64 * 1024 * 1024) {
      const name = `${Date.now()}-${String(sequence++).padStart(6, '0')}.json`;
      writeFileSync(join(outputDir, name), Buffer.concat(chunks));
    }
    response.writeHead(200, { 'content-type': 'application/json' });
    response.end('{}');
  });
});
server.listen(0, '127.0.0.1', () => {
  const address = server.address();
  writeFileSync(portFile, `${address.port}\n`);
});
process.on('SIGTERM', () => server.close(() => process.exit(0)));
""".strip()


class LLMTrajectoryCapture:
    """Own capture initialization, sandbox instrumentation, and finalization."""

    def __init__(
        self,
        rollout_dir: Path,
        *,
        agent: str,
        model: str | None,
        session_id: str,
        started_at: datetime,
    ) -> None:
        self.rollout_dir = rollout_dir
        self.agent = agent
        self.model = model
        self.session_id = session_id
        self.started_at = started_at
        capture_suffix = hashlib.sha256(session_id.encode()).hexdigest()[:12]
        self._remote_capture_root = f"{_REMOTE_CAPTURE_PREFIX}{capture_suffix}"
        self.manifest = initialize_llm_trajectory_artifacts(
            rollout_dir,
            agent=agent,
            model=model,
            session_id=session_id,
            started_at=started_at,
        )
        self._targets: dict[tuple[str, str, str | None, str], _CaptureTarget] = {}
        self._collector_started = False
        self._capture_root_prepared = False
        self._preparation_errors: list[str] = []

    @property
    def trajectory_path(self) -> Path:
        return self.rollout_dir / "trajectory" / LLM_TRAJECTORY_FILENAME

    def configure(self, agent_env: dict[str, str]) -> None:
        self.manifest.auth_mode = _resolve_auth_mode(
            self.agent,
            self.model,
            agent_env,
        )
        write_llm_trajectory_manifest(self.rollout_dir, self.manifest)

    async def prepare_agent(
        self,
        env: Any,
        *,
        agent: str,
        model: str | None,
        agent_env: dict[str, str],
        credential_home: str,
        sandbox_user: str | None,
        role_name: str | None = None,
    ) -> dict[str, str]:
        """Return the agent environment augmented for native capture."""

        prepared = dict(agent_env)
        native = uses_native_subscription_auth(agent, model, prepared)
        auth_mode = _resolve_auth_mode(agent, model, prepared)
        target = _CaptureTarget(
            agent=agent,
            model=model,
            credential_home=credential_home,
            auth_mode=auth_mode,
            native=native,
            role=role_name or "primary",
        )
        primary_key = ("primary", agent, model, credential_home)
        if role_name is None:
            self._targets[primary_key] = target
        else:
            self._targets.pop(primary_key, None)
            self._targets[(role_name, agent, model, credential_home)] = target
        self._refresh_manifest_auth_mode()
        if not native:
            write_llm_trajectory_manifest(self.rollout_dir, self.manifest)
            return prepared
        if _is_claude_code_agent(agent):
            raw_dir = f"{self._remote_capture_root}/raw"
            prepared.update(
                {
                    "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
                    "OTEL_LOG_RAW_API_BODIES": f"file:{raw_dir}",
                }
            )
            try:
                port = await self._ensure_otel_sink(env, sandbox_user=sandbox_user)
            except Exception as exc:
                warning = _sanitized_error(exc)
                self._preparation_errors.append(warning)
                logger.warning(
                    "Claude OTel correlation unavailable; raw/session fallback remains "
                    "enabled: %s",
                    warning,
                )
            else:
                prepared.update(
                    {
                        "OTEL_LOGS_EXPORTER": "otlp",
                        "OTEL_EXPORTER_OTLP_PROTOCOL": "http/json",
                        "OTEL_EXPORTER_OTLP_LOGS_PROTOCOL": "http/json",
                        "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT": (
                            f"http://127.0.0.1:{port}/v1/logs"
                        ),
                        "OTEL_LOGS_EXPORT_INTERVAL": "500",
                    }
                )
        write_llm_trajectory_manifest(self.rollout_dir, self.manifest)
        return prepared

    def _native_targets(self) -> list[_CaptureTarget]:
        return [target for target in self._targets.values() if target.native]

    def _refresh_manifest_auth_mode(self) -> None:
        modes = {target.auth_mode for target in self._targets.values()}
        if len(modes) == 1:
            self.manifest.auth_mode = next(iter(modes))
        elif len(modes) > 1:
            self.manifest.auth_mode = AuthMode.MIXED

    async def finalize(
        self,
        env: Any,
        *,
        acp_events: list[dict[str, Any]],
        model_call_seen: bool,
    ) -> None:
        """Publish the highest-fidelity available capture and terminal sidecar."""

        self.manifest.finished_at = datetime.now()
        provider_records: list[dict[str, Any]] = []
        if self.trajectory_path.stat().st_size > 0:
            try:
                provider_records = load_provider_wire_records(
                    self.trajectory_path,
                    targets=[
                        target for target in self._targets.values() if not target.native
                    ],
                    fallback_agent=self.agent,
                    fallback_model=self.model,
                    fallback_auth=self.manifest.auth_mode,
                )
            except Exception:
                if env is not None:
                    await self._cleanup_remote_capture(env)
                raise

        native_bundles: list[_NativeCaptureBundle] = []
        collection_errors: list[str] = list(self._preparation_errors)
        native_targets = self._native_targets()
        if native_targets and env is not None:
            try:
                native_bundles = await self._collect_native_results(env)
            except Exception as exc:
                collection_errors.append(_sanitized_error(exc))
                logger.warning("Native LLM trajectory collection failed: %s", exc)
            finally:
                await self._cleanup_remote_capture(env)

        if not provider_records and not native_bundles and acp_events:
            projected = project_acp_trajectory(
                acp_events,
                agent=self.agent,
                session_id=self.session_id,
                started_at=self.started_at,
                auth_mode=self.manifest.auth_mode.value,
            )
            if projected is not None:
                native_bundles.append(
                    _NativeCaptureBundle(
                        targets=(
                            tuple(native_targets)
                            or (self._fallback_target(native=True),)
                        ),
                        result=projected,
                    )
                )

        prepared_targets = list(self._targets.values())
        assembly = assemble_capture(
            provider_records=provider_records,
            native_bundles=native_bundles,
            targets=prepared_targets,
            collection_errors=collection_errors,
            model_call_seen=model_call_seen,
            fallback_auth=self.manifest.auth_mode,
        )
        write_exchange_records(self.trajectory_path, assembly.records)
        self.manifest.auth_mode = assembly.auth_mode
        self._finish_manifest(
            status=assembly.status,
            source=assembly.source,
            fidelity=assembly.fidelity,
            exchange_count=len(assembly.records),
            request_complete=assembly.request_complete,
            response_complete=assembly.response_complete,
            missing_fields=assembly.missing_fields,
            errors=assembly.errors,
            role_captures=assembly.role_captures,
        )

    def _fallback_target(self, *, native: bool) -> _CaptureTarget:
        return _CaptureTarget(
            agent=self.agent,
            model=self.model,
            credential_home="",
            auth_mode=self.manifest.auth_mode,
            native=native,
            role="primary",
        )

    def record_failure(self, error: object, *, model_call_seen: bool) -> None:
        """Leave a terminal, truthful sidecar when finalization itself fails."""

        self.manifest.finished_at = datetime.now()
        _atomic_replace_text(self.trajectory_path, "")
        self._finish_manifest(
            status=(
                CaptureStatus.CAPTURE_FAILED
                if model_call_seen
                else CaptureStatus.NO_MODEL_CALL
            ),
            source=CaptureSource.NONE,
            fidelity=CaptureFidelity.NONE,
            exchange_count=0,
            request_complete=False,
            response_complete=False,
            missing_fields=(
                ["provider_request", "provider_response"] if model_call_seen else []
            ),
            errors=[_sanitized_error(error)],
            role_captures=role_captures_for_targets(list(self._targets.values())),
        )

    async def _ensure_otel_sink(self, env: Any, *, sandbox_user: str | None) -> int:
        if self._collector_started:
            return await self._read_collector_port(env)
        capture_owner = shlex.quote(sandbox_user or "root")
        setup = await env.exec(
            f"mkdir -p {self._remote_capture_root}/raw "
            f"{self._remote_capture_root}/otel\n"
            f"chown -R {capture_owner} {self._remote_capture_root}\n"
            f"chmod 700 {self._remote_capture_root} "
            f"{self._remote_capture_root}/raw {self._remote_capture_root}/otel",
            user="root",
            timeout_sec=10,
        )
        if setup.return_code != 0:
            detail = (setup.stderr or setup.stdout or "capture directory setup failed")[
                :300
            ]
            raise RuntimeError(f"Claude capture directory setup failed: {detail}")
        self._capture_root_prepared = True
        with tempfile.TemporaryDirectory(prefix="benchflow-otel-sink-") as temporary:
            source = Path(temporary) / "otel_sink.mjs"
            source.write_text(_OTEL_SINK_SOURCE + "\n")
            await env.upload_file(
                source,
                f"{self._remote_capture_root}/otel_sink.mjs",
                mode="755",
            )
        command = f"""
find {self._remote_capture_root} -maxdepth 1 -type f -name port -delete
find {self._remote_capture_root} -maxdepth 1 -type f -name pid -delete
node_bin=/opt/benchflow/node/bin/node
if ! test -x "$node_bin"; then
  node_bin=$(command -v node || true)
fi
if test -z "$node_bin"; then
  echo "node runtime not found" >&2
  exit 1
fi
nohup "$node_bin" {self._remote_capture_root}/otel_sink.mjs \
  {self._remote_capture_root}/otel {self._remote_capture_root}/port \
  >{self._remote_capture_root}/collector.stdout \
  2>{self._remote_capture_root}/collector.stderr </dev/null &
echo $! > {self._remote_capture_root}/pid
for attempt in $(seq 1 50); do
  if test -s {self._remote_capture_root}/port; then
    cat {self._remote_capture_root}/port
    exit 0
  fi
  sleep 0.1
done
tail -c 300 {self._remote_capture_root}/collector.stderr >&2 2>/dev/null || true
exit 1
"""
        result = await env.exec(
            command,
            user=sandbox_user or "root",
            timeout_sec=10,
        )
        if result.return_code != 0:
            detail = (result.stderr or result.stdout or "collector did not start")[:300]
            raise RuntimeError(f"Claude OTel sink failed to start: {detail}")
        self._collector_started = True
        return _parse_port(result.stdout)

    async def _read_collector_port(self, env: Any) -> int:
        result = await env.exec(
            f"cat {self._remote_capture_root}/port",
            user="root",
            timeout_sec=5,
        )
        if result.return_code != 0:
            raise RuntimeError("Claude OTel sink port file is unavailable")
        return _parse_port(result.stdout)

    async def _collect_native_results(self, env: Any) -> list[_NativeCaptureBundle]:
        if self._collector_started:
            await env.exec(
                f"if test -s {self._remote_capture_root}/pid; then "
                f"kill -TERM $(cat {self._remote_capture_root}/pid) "
                f"2>/dev/null || true; fi",
                user="root",
                timeout_sec=5,
            )
        bundles: list[_NativeCaptureBundle] = []
        native_targets = self._native_targets()
        claude_targets = tuple(
            target for target in native_targets if _is_claude_code_agent(target.agent)
        )
        with tempfile.TemporaryDirectory(prefix="benchflow-native-llm-") as temporary:
            local_root = Path(temporary)
            capture_dir = local_root / "capture"
            raw_claude_captured = False
            if self._capture_root_prepared:
                await env.download_dir(self._remote_capture_root, capture_dir)
                result = parse_claude_raw_capture(
                    capture_dir,
                    agent=(claude_targets[0].agent if claude_targets else self.agent),
                    session_id=self.session_id,
                    started_at=self.started_at,
                )
                if result is not None:
                    bundles.append(
                        _NativeCaptureBundle(targets=claude_targets, result=result)
                    )
                    raw_claude_captured = True
            credential_homes = sorted(
                {target.credential_home for target in native_targets}
            )
            for index, credential_home in enumerate(credential_homes):
                home_targets = tuple(
                    target
                    for target in native_targets
                    if target.credential_home == credential_home
                )
                home_claude_targets = tuple(
                    target
                    for target in home_targets
                    if _is_claude_code_agent(target.agent)
                )
                home_codex_targets = tuple(
                    target for target in home_targets if target.agent == "codex-acp"
                )
                claude_local = local_root / f"home-{index}" / "claude-projects"
                codex_local = local_root / f"home-{index}" / "codex-sessions"
                if (
                    home_claude_targets
                    and not raw_claude_captured
                    and await _download_recent_session_files(
                        env,
                        f"{credential_home}/.claude/projects",
                        claude_local,
                        started_at=self.started_at,
                    )
                ):
                    target = home_claude_targets[0]
                    result = parse_claude_sessions(
                        claude_local,
                        agent=target.agent,
                        session_id=self.session_id,
                        started_at=self.started_at,
                    )
                    if result is not None:
                        bundles.append(
                            _NativeCaptureBundle(
                                targets=home_claude_targets,
                                result=result,
                            )
                        )
                if home_codex_targets and await _download_recent_session_files(
                    env,
                    f"{credential_home}/.codex/sessions",
                    codex_local,
                    started_at=self.started_at,
                ):
                    target = home_codex_targets[0]
                    result = parse_codex_sessions(
                        codex_local,
                        agent=target.agent,
                        session_id=self.session_id,
                        started_at=self.started_at,
                        configured_model=target.model,
                        auth_mode=target.auth_mode.value,
                    )
                    if result is not None:
                        bundles.append(
                            _NativeCaptureBundle(
                                targets=home_codex_targets,
                                result=result,
                            )
                        )
        return bundles

    async def _cleanup_remote_capture(self, env: Any) -> None:
        if not self._capture_root_prepared:
            return
        try:
            result = await env.exec(
                f"find {self._remote_capture_root} -depth -delete",
                user="root",
                timeout_sec=10,
            )
            if result.return_code != 0:
                detail = (result.stderr or result.stdout or "unknown error")[:300]
                logger.warning("Sandbox LLM capture cleanup failed: %s", detail)
        except Exception as exc:
            logger.warning("Sandbox LLM capture cleanup failed: %s", exc)

    def _finish_manifest(
        self,
        *,
        status: CaptureStatus,
        source: CaptureSource,
        fidelity: CaptureFidelity,
        exchange_count: int,
        request_complete: bool,
        response_complete: bool,
        missing_fields: list[str] | None = None,
        errors: list[str] | None = None,
        role_captures: list[LLMRoleCapture] | None = None,
    ) -> None:
        self.manifest.status = status
        self.manifest.capture_source = source
        self.manifest.capture_fidelity = fidelity
        self.manifest.exchange_count = exchange_count
        self.manifest.request_complete = request_complete
        self.manifest.response_complete = response_complete
        self.manifest.missing_fields = sorted(set(missing_fields or []))
        self.manifest.errors = [_sanitized_error(item) for item in errors or []]
        self.manifest.role_captures = role_captures or []
        write_llm_trajectory_manifest(self.rollout_dir, self.manifest)


async def _download_recent_session_files(
    env: Any,
    remote: str,
    local: Path,
    *,
    started_at: datetime,
) -> bool:
    boundary = started_at.timestamp() - 1.0
    remote_root = shlex.quote(remote)
    result = await env.exec(
        f"if test -d {remote_root}; then "
        f"find {remote_root} -type f -name '*.jsonl' "
        f"-newermt {shlex.quote(f'@{boundary}')} -printf '%P\\n' | head -n 1001; "
        "fi",
        user="root",
        timeout_sec=10,
    )
    if result.return_code != 0:
        detail = (result.stderr or result.stdout or "session discovery failed")[:300]
        raise RuntimeError(f"Native session discovery failed: {detail}")
    relative_paths = [line for line in result.stdout.splitlines() if line]
    if len(relative_paths) > 1000:
        raise RuntimeError("Native session discovery exceeded the 1000-file limit")
    if not relative_paths:
        return False
    downloads: list[tuple[str, Path]] = []
    for value in relative_paths:
        relative = PurePosixPath(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Native session discovery returned an unsafe path")
        destination = local.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        downloads.append((f"{remote}/{relative.as_posix()}", destination))
    await asyncio.gather(
        *(env.download_file(source, destination) for source, destination in downloads)
    )
    return True


def _is_claude_code_agent(agent: str) -> bool:
    config = AGENTS.get(agent)
    subscription = config.subscription_auth if config is not None else None
    return bool(
        subscription is not None and subscription.replaces_env == "ANTHROPIC_API_KEY"
    )


def _resolve_auth_mode(
    agent: str,
    model: str | None,
    agent_env: dict[str, str],
) -> AuthMode:
    if not uses_native_subscription_auth(agent, model, agent_env):
        return AuthMode.API_KEY
    if agent != "codex-acp":
        return AuthMode.OAUTH_SUBSCRIPTION
    raw_auth = agent_env.get("CODEX_AUTH_JSON")
    if raw_auth is None and agent_env.get("_BENCHFLOW_SUBSCRIPTION_AUTH") == "1":
        host_auth = Path.home() / ".codex" / "auth.json"
        with suppress(OSError):
            raw_auth = host_auth.read_text()
    if raw_auth:
        try:
            auth = json.loads(raw_auth)
        except json.JSONDecodeError:
            auth = None
        if isinstance(auth, dict):
            auth_name = str(auth.get("auth_mode") or "").casefold()
            if auth_name == "chatgpt":
                return AuthMode.OAUTH_SUBSCRIPTION
            if auth_name in {"api_key", "apikey", "api-key"} or auth.get(
                "OPENAI_API_KEY"
            ):
                return AuthMode.API_KEY
            if isinstance(auth.get("tokens"), dict):
                return AuthMode.OAUTH_SUBSCRIPTION
    return AuthMode.OAUTH_SUBSCRIPTION


def _parse_port(value: str) -> int:
    try:
        port = int(value.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError("Claude OTel sink returned an invalid port") from exc
    if not 1 <= port <= 65535:
        raise RuntimeError("Claude OTel sink returned an out-of-range port")
    return port


def _sanitized_error(error: object) -> str:
    text = redact_trajectory_text(str(error)).replace("\n", " ").strip()
    return text[:500] or type(error).__name__


def _atomic_replace_text(path: Path, payload: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)


def model_call_seen_from_evidence(
    usage_metrics: dict[str, Any] | None,
    acp_events: list[dict[str, Any]],
) -> bool:
    for value in (usage_metrics or {}).values():
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value > 0
        ):
            return True
    return any(
        event.get("type") in {"agent_message", "agent_thought", "tool_call"}
        for event in acp_events
    )
