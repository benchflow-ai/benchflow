"""Sandbox-owned native telemetry and session collection components."""

from __future__ import annotations

import asyncio
import logging
import re
import shlex
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from benchflow.agents.registry import AGENTS
from benchflow.trajectories.llm_capture_records import (
    CaptureTarget,
    NativeCaptureBundle,
)
from benchflow.trajectories.native_capture_parsers import (
    NativeParseResult,
    parse_claude_raw_capture,
    parse_claude_sessions,
    parse_codex_sessions,
    retain_uncovered_claude_session_exchanges,
)
from benchflow.trajectories.redaction import redact_trajectory_text

logger = logging.getLogger(__name__)

MAX_NATIVE_SESSION_FILES = 1000
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
class NativeCollection:
    """Native bundles retained alongside isolated collection errors."""

    bundles: tuple[NativeCaptureBundle, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass
class ClaudeOtelCollector:
    """Own the sandbox OTel sink and its private capture directory."""

    remote_root: str
    started: bool = False
    owned: bool = False
    root_prepared: bool = False

    async def ensure(self, env: Any, *, sandbox_user: str | None) -> int:
        if self.started:
            return await self.read_port(env)
        await self.stop(env)
        capture_owner = shlex.quote(sandbox_user or "root")
        setup = await env.exec(
            f"find {self.remote_root} -depth -mindepth 1 -delete "
            "2>/dev/null || true\n"
            f"mkdir -p {self.remote_root}/raw {self.remote_root}/otel\n"
            f"chown root:root {self.remote_root} {self.remote_root}/otel\n"
            f"chown {capture_owner} {self.remote_root}/raw\n"
            f"chmod 711 {self.remote_root}\n"
            f"chmod 700 {self.remote_root}/raw {self.remote_root}/otel",
            user="root",
            timeout_sec=10,
        )
        if setup.return_code != 0:
            detail = (setup.stderr or setup.stdout or "capture directory setup failed")[
                :300
            ]
            raise RuntimeError(f"Claude capture directory setup failed: {detail}")
        self.root_prepared = True
        with tempfile.TemporaryDirectory(prefix="benchflow-otel-sink-") as temporary:
            source = Path(temporary) / "otel_sink.mjs"
            source.write_text(_OTEL_SINK_SOURCE + "\n")
            await env.upload_file(
                source, f"{self.remote_root}/otel_sink.mjs", mode="755"
            )
        command = f"""
find {self.remote_root} -maxdepth 1 -type f -name port -delete
find {self.remote_root} -maxdepth 1 -type f -name pid -delete
node_bin=/opt/benchflow/node/bin/node
if ! test -x "$node_bin"; then
  node_bin=$(command -v node || true)
fi
if test -z "$node_bin"; then
  echo "node runtime not found" >&2
  exit 1
fi
nohup "$node_bin" {self.remote_root}/otel_sink.mjs \
  {self.remote_root}/otel {self.remote_root}/port \
  >{self.remote_root}/collector.stdout \
  2>{self.remote_root}/collector.stderr </dev/null &
echo $! > {self.remote_root}/pid
for attempt in $(seq 1 50); do
  if test -s {self.remote_root}/port; then
    cat {self.remote_root}/port
    exit 0
  fi
  sleep 0.1
done
tail -c 300 {self.remote_root}/collector.stderr >&2 2>/dev/null || true
exit 1
"""
        self.owned = True
        # The sink is BenchFlow infrastructure, not part of the agent process
        # tree. Keep it root-owned so role teardown and the next API-proxy
        # custody proof can require zero live sandbox-user processes. Claude
        # still writes its raw-body files into the separately agent-owned
        # ``raw`` directory and exports OTel to this loopback listener.
        result = await env.exec(command, user="root", timeout_sec=10)
        if result.return_code != 0:
            detail = (result.stderr or result.stdout or "collector did not start")[:300]
            raise RuntimeError(f"Claude OTel sink failed to start: {detail}")
        self.started = True
        return _parse_port(result.stdout)

    async def stop(self, env: Any) -> None:
        command = f"""
if ! test -s {self.remote_root}/pid; then
  exit 0
fi
read -r old_pid < {self.remote_root}/pid || true
case "$old_pid" in
  ''|*[!0-9]*) exit 0 ;;
esac
old_command=$(ps -p "$old_pid" -o command= 2>/dev/null || true)
case "$old_command" in
  *{self.remote_root}/otel_sink.mjs*) ;;
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
  *{self.remote_root}/otel_sink.mjs*) ;;
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
        self.started = False
        self.owned = False

    async def read_port(self, env: Any) -> int:
        result = await env.exec(
            f"cat {self.remote_root}/port", user="root", timeout_sec=5
        )
        if result.return_code != 0:
            raise RuntimeError("Claude OTel sink port file is unavailable")
        return _parse_port(result.stdout)

    async def cleanup(self, env: Any) -> None:
        if not self.root_prepared:
            return
        if self.owned:
            raise RuntimeError(
                "Refusing to remove Claude capture ownership files while its "
                "collector may still be running"
            )
        result = await env.exec(
            "for attempt in 1 2 3; do\n"
            f"  if ! test -e {self.remote_root} || "
            f"find {self.remote_root} -depth -delete; then\n"
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
        self.root_prepared = False


@dataclass
class NativeSessionCollector:
    """Collect session-bound Claude/Codex evidence after the agent exits."""

    agent: str
    session_id: str
    started_at: datetime
    otel: ClaudeOtelCollector

    async def collect(
        self, env: Any, *, targets: list[CaptureTarget]
    ) -> NativeCollection:
        bundles: list[NativeCaptureBundle] = []
        errors: list[str] = []
        if self.otel.owned:
            try:
                await self.otel.stop(env)
            except Exception as exc:
                errors.append(sanitized_capture_error(exc))
                logger.warning("Claude OTel collector shutdown failed: %s", exc)

        claude_targets = tuple(
            target for target in targets if is_claude_code_agent(target.agent)
        )
        with tempfile.TemporaryDirectory(prefix="benchflow-native-llm-") as temporary:
            local_root = Path(temporary)
            try:
                raw_claude_bundle = await self._collect_claude_raw_capture(
                    env,
                    local_root=local_root,
                    claude_targets=claude_targets,
                )
            except Exception as exc:
                warning = sanitized_capture_error(exc)
                errors.append(warning)
                logger.warning("Claude raw LLM capture collection failed: %s", exc)
                raw_claude_bundle = None
            if raw_claude_bundle is not None:
                bundles.append(raw_claude_bundle)
            raw_claude_result = (
                raw_claude_bundle.result if raw_claude_bundle is not None else None
            )
            for index, target in enumerate(targets):
                try:
                    if is_claude_code_agent(target.agent):
                        bundles.extend(
                            await self._collect_claude_session_fallback(
                                env,
                                local_root=local_root,
                                index=index,
                                target=target,
                                raw_result=raw_claude_result,
                            )
                        )
                        bundle = None
                    elif target.agent == "codex-acp":
                        bundle = await self._collect_codex_session(
                            env, local_root=local_root, index=index, target=target
                        )
                    else:
                        bundle = None
                    if bundle is not None:
                        bundles.append(bundle)
                except Exception as exc:
                    warning = (
                        f"native capture failed for role {target.role}: "
                        f"{sanitized_capture_error(exc)}"
                    )
                    errors.append(warning)
                    logger.warning("%s", warning)
        return NativeCollection(bundles=tuple(bundles), errors=tuple(errors))

    async def _collect_claude_raw_capture(
        self,
        env: Any,
        *,
        local_root: Path,
        claude_targets: tuple[CaptureTarget, ...],
    ) -> NativeCaptureBundle | None:
        if not self.otel.root_prepared:
            return None
        capture_dir = local_root / "capture"
        await env.download_dir(self.otel.remote_root, capture_dir)
        result = parse_claude_raw_capture(
            capture_dir,
            agent=(claude_targets[0].agent if claude_targets else self.agent),
            session_id=self.session_id,
            started_at=self.started_at,
        )
        if result is None:
            return None
        return NativeCaptureBundle(targets=claude_targets, result=result)

    async def _collect_claude_session_fallback(
        self,
        env: Any,
        *,
        local_root: Path,
        index: int,
        target: CaptureTarget,
        raw_result: NativeParseResult | None,
    ) -> tuple[NativeCaptureBundle, ...]:
        bundles: list[NativeCaptureBundle] = []
        for session_index, native_session_id in enumerate(target.native_session_ids):
            local = local_root / f"target-{index}" / f"claude-session-{session_index}"
            downloaded = await download_bound_session_files(
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
                raw_result, result, native_session_id=native_session_id
            )
            if uncovered is not None:
                bundles.append(NativeCaptureBundle(targets=(target,), result=uncovered))
        return tuple(bundles)

    async def _collect_codex_session(
        self,
        env: Any,
        *,
        local_root: Path,
        index: int,
        target: CaptureTarget,
    ) -> NativeCaptureBundle | None:
        local = local_root / f"target-{index}" / "codex-sessions"
        downloaded = await download_bound_session_files(
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
            NativeCaptureBundle(targets=(target,), result=result)
            if result is not None
            else None
        )


async def download_bound_session_files(
    env: Any,
    remote: str,
    local: Path,
    *,
    started_at: datetime,
    session_ids: tuple[str, ...],
) -> bool:
    """Download only explicitly bound, rollout-fresh native session files."""

    if not session_ids:
        return False
    if any(not native_session_id_is_safe(value) for value in session_ids):
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
        f"-printf '%P\\n' | head -n {MAX_NATIVE_SESSION_FILES + 1}; "
        "fi",
        user="root",
        timeout_sec=10,
    )
    if result.return_code != 0:
        detail = (result.stderr or result.stdout or "session discovery failed")[:300]
        raise RuntimeError(f"Native session discovery failed: {detail}")
    relative_paths = [line for line in result.stdout.splitlines() if line]
    if len(relative_paths) > MAX_NATIVE_SESSION_FILES:
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


def native_session_id_is_safe(value: str) -> bool:
    return _SAFE_NATIVE_SESSION_ID.fullmatch(value) is not None


def is_claude_code_agent(agent: str) -> bool:
    config = AGENTS.get(agent)
    subscription = config.subscription_auth if config is not None else None
    return bool(
        subscription is not None and subscription.replaces_env == "ANTHROPIC_API_KEY"
    )


def sanitized_capture_error(error: object) -> str:
    text = redact_trajectory_text(str(error)).replace("\n", " ").strip()
    return text[:500] or type(error).__name__


def _parse_port(value: str) -> int:
    try:
        port = int(value.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError("Claude OTel sink returned an invalid port") from exc
    if not 1 <= port <= 65535:
        raise RuntimeError("Claude OTel sink returned an out-of-range port")
    return port
