"""Lifecycle orchestration for the always-present LLM trajectory artifact."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from contextlib import suppress
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from benchflow.agents.env import uses_native_subscription_auth
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
from benchflow.trajectories.native_capture_collection import (
    MAX_NATIVE_SESSION_FILES,
    ClaudeOtelCollector,
    NativeSessionCollector,
    is_claude_code_agent,
    native_session_id_is_safe,
    sanitized_capture_error,
)
from benchflow.trajectories.native_capture_parsers import project_acp_trajectory

logger = logging.getLogger(__name__)

_REMOTE_CAPTURE_PREFIX = "/tmp/benchflow-llm-capture-"


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
        self._otel_collector = ClaudeOtelCollector(self._remote_capture_root)
        self._native_collector = NativeSessionCollector(
            agent=agent,
            session_id=session_id,
            started_at=started_at,
            otel=self._otel_collector,
        )
        self.manifest = initialize_llm_trajectory_artifacts(
            rollout_dir,
            agent=agent,
            model=model,
            session_id=session_id,
            started_at=started_at,
        )
        self._targets: dict[
            tuple[str, str, str | None, str, AuthMode], _CaptureTarget
        ] = {}
        self._active_target_keys: dict[
            tuple[str, str, str | None, str],
            tuple[str, str, str | None, str, AuthMode],
        ] = {}
        self._provisional_target_key: (
            tuple[str, str, str | None, str, AuthMode] | None
        ) = None
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
            auth_mode=auth_mode,
        )
        previous_target = self._targets.get(target_key)
        if previous_target is not None and not previous_target.provider_capture_trusted:
            target = replace(target, provider_capture_trusted=False)
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
            auth_mode=auth_mode,
        )
        base_key = _capture_target_base_key(
            agent=agent,
            model=model,
            credential_home=credential_home,
            role_name=role_name,
        )
        if role_name is None:
            self._targets[primary_key] = target
            self._provisional_target_key = primary_key
        else:
            if self._provisional_target_key is not None:
                removed_key = self._provisional_target_key
                self._targets.pop(removed_key, None)
                for active_base, active_key in list(self._active_target_keys.items()):
                    if active_key == removed_key:
                        self._active_target_keys.pop(active_base, None)
                self._provisional_target_key = None
            self._targets[target_key] = target
        self._active_target_keys[base_key] = target_key
        self._refresh_manifest_auth_mode()
        if not native:
            write_llm_trajectory_manifest(self.rollout_dir, self.manifest)
            return prepared
        if is_claude_code_agent(agent):
            raw_dir = f"{self._remote_capture_root}/raw"
            prepared.update(
                {
                    "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
                    "OTEL_LOG_RAW_API_BODIES": f"file:{raw_dir}",
                }
            )
            try:
                port = await self._otel_collector.ensure(env, sandbox_user=sandbox_user)
            except Exception as exc:
                prepared.pop("CLAUDE_CODE_ENABLE_TELEMETRY", None)
                prepared.pop("OTEL_LOG_RAW_API_BODIES", None)
                warning = sanitized_capture_error(exc)
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

        base_key = _capture_target_base_key(
            agent=agent,
            model=model,
            credential_home=credential_home,
            role_name=role_name,
        )
        key = self._active_target_keys.get(base_key)
        if key is None:
            return
        target = self._targets.get(key)
        if target is None:
            return
        if key == self._provisional_target_key:
            self._provisional_target_key = None
        if not target.native:
            return
        if not native_session_id_is_safe(native_session_id):
            warning = "native ACP session identifier was unsafe for file scoping"
            self._preparation_errors.append(warning)
            logger.warning("Native LLM capture disabled: %s", warning)
            return
        session_ids = tuple(sorted({*target.native_session_ids, native_session_id}))
        if len(session_ids) > MAX_NATIVE_SESSION_FILES:
            warning = "native ACP session count exceeded the capture limit"
            self._preparation_errors.append(warning)
            logger.warning("Native LLM capture disabled: %s", warning)
            return
        self._targets[key] = replace(target, native_session_ids=session_ids)

    def bind_provider_capture_trust(
        self,
        *,
        agent: str,
        model: str | None,
        credential_home: str,
        trusted: bool,
        role_name: str | None = None,
    ) -> None:
        """Bind the gateway custody boundary to its prepared capture target."""

        base_key = _capture_target_base_key(
            agent=agent,
            model=model,
            credential_home=credential_home,
            role_name=role_name,
        )
        key = self._active_target_keys.get(base_key)
        if key is None:
            return
        target = self._targets.get(key)
        if target is None or target.native:
            return
        self._targets[key] = replace(
            target,
            # Once any provider rows for this target were collected under
            # shared root custody, later trusted placement cannot recover the
            # target-level training claim without per-row custody evidence.
            provider_capture_trusted=(target.provider_capture_trusted and trusted),
        )

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
        preparation_errors = [*self._preparation_errors]
        if self._otel_setup_error is not None:
            preparation_errors.append(self._otel_setup_error)
        collection_errors = list(
            dict.fromkeys([*preparation_errors, *(capture_errors or [])])
        )
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
            except Exception as exc:
                warning = (
                    f"provider capture parse failed: {sanitized_capture_error(exc)}"
                )
                collection_errors.append(warning)
                logger.warning("%s", warning)

        native_bundles: list[_NativeCaptureBundle] = []
        native_targets = self._native_targets()
        native_resources_exist = (
            self._otel_collector.owned or self._otel_collector.root_prepared
        )
        cleanup_failed = False
        if (native_targets or native_resources_exist) and env is not None:
            try:
                if native_targets:
                    collection = await self._native_collector.collect(
                        env, targets=native_targets
                    )
                    native_bundles.extend(collection.bundles)
                    collection_errors.extend(collection.errors)
                elif self._otel_collector.owned:
                    await self._otel_collector.stop(env)
            except Exception as exc:
                collection_errors.append(sanitized_capture_error(exc))
                logger.warning("Native LLM trajectory collection failed: %s", exc)
            finally:
                try:
                    await self._otel_collector.cleanup(env)
                except Exception as exc:
                    collection_errors.append(sanitized_capture_error(exc))
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
        exchange_count = _valid_jsonl_row_count(self.trajectory_path)
        if exchange_count is None:
            _atomic_replace_text(self.trajectory_path, "")
            exchange_count = 0
        rows_preserved = exchange_count > 0
        self._finish_manifest(
            status=(
                CaptureStatus.CAPTURE_FAILED
                if model_call_seen or rows_preserved
                else CaptureStatus.NO_MODEL_CALL
            ),
            source=(
                self.manifest.capture_source if rows_preserved else CaptureSource.NONE
            ),
            fidelity=(
                self.manifest.capture_fidelity
                if rows_preserved
                else CaptureFidelity.NONE
            ),
            exchange_count=exchange_count,
            request_complete=False,
            response_complete=False,
            missing_fields=(
                sorted(
                    {
                        *self.manifest.missing_fields,
                        *(
                            ["provider_request", "provider_response"]
                            if model_call_seen or rows_preserved
                            else []
                        ),
                    }
                )
            ),
            errors=[*self.manifest.errors, sanitized_capture_error(error)],
            role_captures=(
                self.manifest.role_captures
                or role_captures_for_targets(list(self._targets.values()))
            ),
        )

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
        self.manifest.errors = [sanitized_capture_error(item) for item in errors or []]
        self.manifest.role_captures = role_captures or []
        write_llm_trajectory_manifest(self.rollout_dir, self.manifest)


def _capture_target_key(
    *,
    agent: str,
    model: str | None,
    credential_home: str,
    role_name: str | None,
    auth_mode: AuthMode,
) -> tuple[str, str, str | None, str, AuthMode]:
    return (
        *_capture_target_base_key(
            agent=agent,
            model=model,
            credential_home=credential_home,
            role_name=role_name,
        ),
        auth_mode,
    )


def _capture_target_base_key(
    *,
    agent: str,
    model: str | None,
    credential_home: str,
    role_name: str | None,
) -> tuple[str, str, str | None, str]:
    return (role_name or "primary", agent, model, credential_home)


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


def _atomic_replace_text(path: Path, payload: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)


def _valid_jsonl_row_count(path: Path) -> int | None:
    """Count valid JSON objects, distinguishing an empty artifact from corruption."""

    count = 0
    try:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            if not isinstance(json.loads(line), dict):
                return None
            count += 1
    except (OSError, json.JSONDecodeError):
        return None
    return count


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
