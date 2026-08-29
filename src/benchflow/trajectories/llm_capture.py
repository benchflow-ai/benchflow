"""Lifecycle orchestration for the always-present LLM trajectory artifact."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shlex
import tempfile
from contextlib import suppress
from dataclasses import dataclass, replace
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
    NativeParseResult,
    parse_claude_raw_capture,
    parse_claude_sessions,
    parse_codex_sessions,
    project_acp_trajectory,
    retain_uncovered_claude_session_exchanges,
)
from benchflow.trajectories.types import redact_trajectory_text

logger = logging.getLogger(__name__)

_REMOTE_CAPTURE_PREFIX = "/tmp/benchflow-llm-capture-"
_MAX_NATIVE_SESSION_FILES = 1000
_SAFE_NATIVE_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
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


@dataclass(frozen=True)
class _NativeCollection:
    """Native bundles retained alongside isolated collection errors."""

    bundles: tuple[_NativeCaptureBundle, ...] = ()
    errors: tuple[str, ...] = ()


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
        self._provisional_target_key: tuple[str, str, str | None, str] | None = None
        self._collector_started = False
        self._collector_owned = False
        self._capture_root_prepared = False
        self._preparation_errors: list[str] = []
        self._otel_setup_error: str | None = None

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
        target_key = _capture_target_key(
            agent=agent,
            model=model,
            credential_home=credential_home,
            role_name=role_name,
        )
        previous_target = self._targets.get(target_key)
        if (
            previous_target is not None
            and previous_target.native
            and target.native
            and previous_target.auth_mode is target.auth_mode
        ):
            target = replace(
                target,
                native_session_ids=previous_target.native_session_ids,
            )
        primary_key = _capture_target_key(
            agent=agent,
            model=model,
            credential_home=credential_home,
            role_name=None,
        )
        if role_name is None:
            if (
                self._provisional_target_key is not None
                and self._provisional_target_key != primary_key
            ):
                self._targets.pop(self._provisional_target_key, None)
            self._targets[primary_key] = target
            self._provisional_target_key = primary_key
        else:
            if self._provisional_target_key is not None:
                self._targets.pop(self._provisional_target_key, None)
                self._provisional_target_key = None
            self._targets[target_key] = target
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
                prepared.pop("CLAUDE_CODE_ENABLE_TELEMETRY", None)
                prepared.pop("OTEL_LOG_RAW_API_BODIES", None)
                warning = _sanitized_error(exc)
                self._otel_setup_error = warning
                logger.warning(
                    "Claude OTel correlation unavailable; session fallback remains "
                    "enabled: %s",
                    warning,
                )
            else:
                self._otel_setup_error = None
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

    def bind_native_session(
        self,
        *,
        agent: str,
        model: str | None,
        credential_home: str,
        native_session_id: str,
        role_name: str | None = None,
    ) -> None:
        """Bind an ACP session ID to its prepared native capture target."""

        key = _capture_target_key(
            agent=agent,
            model=model,
            credential_home=credential_home,
            role_name=role_name,
        )
        target = self._targets.get(key)
        if target is None:
            return
        if key == self._provisional_target_key:
            self._provisional_target_key = None
        if not target.native:
            return
        if _SAFE_NATIVE_SESSION_ID.fullmatch(native_session_id) is None:
            warning = "native ACP session identifier was unsafe for file scoping"
            self._preparation_errors.append(warning)
            logger.warning("Native LLM capture disabled: %s", warning)
            return
        session_ids = tuple(sorted({*target.native_session_ids, native_session_id}))
        if len(session_ids) > _MAX_NATIVE_SESSION_FILES:
            warning = "native ACP session count exceeded the capture limit"
            self._preparation_errors.append(warning)
            logger.warning("Native LLM capture disabled: %s", warning)
            return
        self._targets[key] = replace(target, native_session_ids=session_ids)

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
        capture_errors: list[str] | None = None,
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
                    if self._collector_owned:
                        await self._stop_otel_sink(env)
                    await self._cleanup_remote_capture(env)
                raise

        native_bundles: list[_NativeCaptureBundle] = []
        preparation_errors = [*self._preparation_errors]
        if self._otel_setup_error is not None:
            preparation_errors.append(self._otel_setup_error)
        collection_errors = list(
            dict.fromkeys([*preparation_errors, *(capture_errors or [])])
        )
        native_targets = self._native_targets()
        native_resources_exist = self._collector_owned or self._capture_root_prepared
        cleanup_failed = False
        if (native_targets or native_resources_exist) and env is not None:
            try:
                if native_targets:
                    collection = await self._collect_native_results(env)
                    native_bundles.extend(collection.bundles)
                    collection_errors.extend(collection.errors)
                elif self._collector_owned:
                    await self._stop_otel_sink(env)
            except Exception as exc:
                collection_errors.append(_sanitized_error(exc))
                logger.warning("Native LLM trajectory collection failed: %s", exc)
            finally:
                try:
                    await self._cleanup_remote_capture(env)
                except Exception as exc:
                    collection_errors.append(_sanitized_error(exc))
                    logger.error("Sandbox LLM capture cleanup failed: %s", exc)
                    cleanup_failed = True

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
            status=(
                CaptureStatus.CAPTURE_FAILED if cleanup_failed else assembly.status
            ),
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
        await self._stop_otel_sink(env)
        capture_owner = shlex.quote(sandbox_user or "root")
        setup = await env.exec(
            f"find {self._remote_capture_root} -depth -mindepth 1 -delete "
            "2>/dev/null || true\n"
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
        self._collector_owned = True
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

    async def _stop_otel_sink(self, env: Any) -> None:
        command = f"""
if ! test -s {self._remote_capture_root}/pid; then
  exit 0
fi
read -r old_pid < {self._remote_capture_root}/pid || true
case "$old_pid" in
  ''|*[!0-9]*) exit 0 ;;
esac
old_command=$(ps -p "$old_pid" -o command= 2>/dev/null || true)
case "$old_command" in
  *{self._remote_capture_root}/otel_sink.mjs*) ;;
  *) exit 0 ;;
esac
kill -TERM "$old_pid" 2>/dev/null || true
for attempt in $(seq 1 20); do
  if ! kill -0 "$old_pid" 2>/dev/null; then
    exit 0
  fi
  sleep 0.05
done
old_command=$(ps -p "$old_pid" -o command= 2>/dev/null || true)
case "$old_command" in
  *{self._remote_capture_root}/otel_sink.mjs*) ;;
  *) exit 0 ;;
esac
kill -KILL "$old_pid" 2>/dev/null || true
for attempt in $(seq 1 20); do
  if ! kill -0 "$old_pid" 2>/dev/null; then
    exit 0
  fi
  sleep 0.05
done
echo "previous Claude telemetry collector did not stop" >&2
exit 1
"""
        result = await env.exec(command, user="root", timeout_sec=5)
        if result.return_code != 0:
            detail = (result.stderr or result.stdout or "collector did not stop")[:300]
            raise RuntimeError(f"Claude OTel sink shutdown failed: {detail}")
        self._collector_started = False
        self._collector_owned = False

    async def _read_collector_port(self, env: Any) -> int:
        result = await env.exec(
            f"cat {self._remote_capture_root}/port",
            user="root",
            timeout_sec=5,
        )
        if result.return_code != 0:
            raise RuntimeError("Claude OTel sink port file is unavailable")
        return _parse_port(result.stdout)

    async def _collect_native_results(self, env: Any) -> _NativeCollection:
        bundles: list[_NativeCaptureBundle] = []
        errors: list[str] = []
        if self._collector_owned:
            try:
                await self._stop_otel_sink(env)
            except Exception as exc:
                errors.append(_sanitized_error(exc))
                logger.warning("Claude OTel collector shutdown failed: %s", exc)

        native_targets = self._native_targets()
        claude_targets = tuple(
            target for target in native_targets if _is_claude_code_agent(target.agent)
        )
        with tempfile.TemporaryDirectory(prefix="benchflow-native-llm-") as temporary:
            local_root = Path(temporary)
            raw_claude_result = await self._collect_claude_raw_capture(
                env,
                local_root=local_root,
                claude_targets=claude_targets,
                bundles=bundles,
                errors=errors,
            )
            for index, target in enumerate(native_targets):
                try:
                    if _is_claude_code_agent(target.agent):
                        target_bundles = await self._collect_claude_session_fallback(
                            env,
                            local_root=local_root,
                            index=index,
                            target=target,
                            raw_result=raw_claude_result,
                        )
                        bundles.extend(target_bundles)
                        bundle = None
                    elif target.agent == "codex-acp":
                        bundle = await self._collect_codex_session(
                            env,
                            local_root=local_root,
                            index=index,
                            target=target,
                        )
                    else:
                        bundle = None
                    if bundle is not None:
                        bundles.append(bundle)
                except Exception as exc:
                    warning = (
                        f"native capture failed for role {target.role}: "
                        f"{_sanitized_error(exc)}"
                    )
                    errors.append(warning)
                    logger.warning("%s", warning)
        return _NativeCollection(bundles=tuple(bundles), errors=tuple(errors))

    async def _collect_claude_raw_capture(
        self,
        env: Any,
        *,
        local_root: Path,
        claude_targets: tuple[_CaptureTarget, ...],
        bundles: list[_NativeCaptureBundle],
        errors: list[str],
    ) -> NativeParseResult | None:
        if not self._capture_root_prepared:
            return None
        capture_dir = local_root / "capture"
        try:
            await env.download_dir(self._remote_capture_root, capture_dir)
            result = parse_claude_raw_capture(
                capture_dir,
                agent=(claude_targets[0].agent if claude_targets else self.agent),
                session_id=self.session_id,
                started_at=self.started_at,
            )
        except Exception as exc:
            errors.append(_sanitized_error(exc))
            logger.warning("Claude raw LLM capture collection failed: %s", exc)
            return None
        if result is None:
            return None
        bundles.append(_NativeCaptureBundle(targets=claude_targets, result=result))
        return result

    async def _collect_claude_session_fallback(
        self,
        env: Any,
        *,
        local_root: Path,
        index: int,
        target: _CaptureTarget,
        raw_result: NativeParseResult | None,
    ) -> tuple[_NativeCaptureBundle, ...]:
        bundles: list[_NativeCaptureBundle] = []
        for session_index, native_session_id in enumerate(target.native_session_ids):
            local = local_root / f"target-{index}" / f"claude-session-{session_index}"
            downloaded = await _download_bound_session_files(
                env,
                f"{target.credential_home}/.claude/projects",
                local,
                started_at=self.started_at,
                session_ids=(native_session_id,),
            )
            if not downloaded:
                continue
            result = parse_claude_sessions(
                local,
                agent=target.agent,
                session_id=self.session_id,
                started_at=self.started_at,
            )
            if result is None:
                continue
            uncovered = retain_uncovered_claude_session_exchanges(
                raw_result,
                result,
                native_session_id=native_session_id,
            )
            if uncovered is not None:
                bundles.append(
                    _NativeCaptureBundle(targets=(target,), result=uncovered)
                )
        return tuple(bundles)

    async def _collect_codex_session(
        self,
        env: Any,
        *,
        local_root: Path,
        index: int,
        target: _CaptureTarget,
    ) -> _NativeCaptureBundle | None:
        local = local_root / f"target-{index}" / "codex-sessions"
        downloaded = await _download_bound_session_files(
            env,
            f"{target.credential_home}/.codex/sessions",
            local,
            started_at=self.started_at,
            session_ids=target.native_session_ids,
        )
        if not downloaded:
            return None
        result = parse_codex_sessions(
            local,
            agent=target.agent,
            session_id=self.session_id,
            started_at=self.started_at,
            configured_model=target.model,
            auth_mode=target.auth_mode.value,
        )
        return (
            _NativeCaptureBundle(targets=(target,), result=result)
            if result is not None
            else None
        )

    async def _cleanup_remote_capture(self, env: Any) -> None:
        if not self._capture_root_prepared:
            return
        if self._collector_owned:
            raise RuntimeError(
                "Refusing to remove Claude capture ownership files while its "
                "collector may still be running"
            )
        result = await env.exec(
            "for attempt in 1 2 3; do\n"
            f"  if ! test -e {self._remote_capture_root} || "
            f"find {self._remote_capture_root} -depth -delete; then\n"
            "    exit 0\n"
            "  fi\n"
            "  sleep 0.1\n"
            "done\n"
            "exit 1",
            user="root",
            timeout_sec=10,
        )
        if result.return_code != 0:
            detail = (result.stderr or result.stdout or "unknown error")[:300]
            raise RuntimeError(f"Sandbox LLM capture cleanup failed: {detail}")
        self._capture_root_prepared = False

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


async def _download_bound_session_files(
    env: Any,
    remote: str,
    local: Path,
    *,
    started_at: datetime,
    session_ids: tuple[str, ...],
) -> bool:
    if not session_ids:
        return False
    if any(_SAFE_NATIVE_SESSION_ID.fullmatch(value) is None for value in session_ids):
        raise RuntimeError("Native session discovery received an unsafe session ID")
    boundary = started_at.timestamp() - 1.0
    remote_root = shlex.quote(remote)
    filename_patterns = [
        pattern
        for session_id in session_ids
        for pattern in (f"{session_id}.jsonl", f"*-{session_id}.jsonl")
    ]
    filename_filter = (
        r"\( "
        + " -o ".join(f"-name {shlex.quote(pattern)}" for pattern in filename_patterns)
        + r" \)"
    )
    result = await env.exec(
        f"if test -d {remote_root}; then "
        f"find {remote_root} -type f -name '*.jsonl' "
        f"-newermt {shlex.quote(f'@{boundary}')} {filename_filter} "
        f"-printf '%P\\n' | head -n {_MAX_NATIVE_SESSION_FILES + 1}; "
        "fi",
        user="root",
        timeout_sec=10,
    )
    if result.return_code != 0:
        detail = (result.stderr or result.stdout or "session discovery failed")[:300]
        raise RuntimeError(f"Native session discovery failed: {detail}")
    relative_paths = [line for line in result.stdout.splitlines() if line]
    if len(relative_paths) > _MAX_NATIVE_SESSION_FILES:
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


def _capture_target_key(
    *,
    agent: str,
    model: str | None,
    credential_home: str,
    role_name: str | None,
) -> tuple[str, str, str | None, str]:
    return (role_name or "primary", agent, model, credential_home)


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
