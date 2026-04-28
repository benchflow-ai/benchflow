"""Agent-tester L0/L1 per PLAN_V2_byoa.md §6.

Catches the cold-start failure modes that today only surface after a
five-hour run_batch dispatches and dies. Three levels:

  L0 — install + version_cmd from [smoke_test]. Local exec, ~$0, <2s.
  L1 — L0 + provider auth ping (ping_cmd). ~$0.000004, <30s.
  L2 — L1 + 30s end-to-end built-in task smoke. Deferred (PLAN_V2_byoa §8).

Current implementation runs commands on the **host process**, not inside
the sandbox. This is the simpler scaffolding to land first; sandbox-fidelity
L0 (which actually validates ``install_cmd`` ran) is deferred to PR7
follow-up. Documented in the AgentTestResult.fidelity field so callers
can surface the gap.
"""

from __future__ import annotations

import asyncio
import re
import shlex
import time
from dataclasses import dataclass, field
from typing import Literal

from benchflow.agents.discovery import BUILTINS_DIR
from benchflow.agents.loader import AgentManifest, load_agent_toml

Outcome = Literal["pass", "fail", "skipped", "error"]
Fidelity = Literal["host", "sandbox"]


@dataclass(frozen=True)
class AgentTestResult:
    """Result of one ``bf agent test`` invocation."""

    name: str
    level: int
    outcome: Outcome
    latency_ms: int
    cost_usd: float = 0.0
    stderr_tail: str = ""
    version_detected: str = ""
    fidelity: Fidelity = "host"
    detail: str = ""
    rules_failed: tuple[str, ...] = field(default_factory=tuple)


def _load_manifest(name: str) -> AgentManifest:
    return load_agent_toml(BUILTINS_DIR / name)


async def _run(cmd: str, timeout: int) -> tuple[int, str, str]:
    """Run *cmd* with /bin/sh -c, return (rc, stdout, stderr)."""
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "", "(timeout)"
    return (
        proc.returncode if proc.returncode is not None else -1,
        out_b.decode(errors="replace"),
        err_b.decode(errors="replace"),
    )


def _tail(s: str, n: int = 400) -> str:
    return s[-n:] if len(s) > n else s


async def run_l0_async(manifest: AgentManifest) -> AgentTestResult:
    """L0: shell out to ``[smoke_test].version_cmd`` and parse the version.

    Returns ``skipped`` when the manifest doesn't declare a smoke_test —
    a deliberate non-failure so unannotated agents don't block run_batch.
    """
    smoke = manifest.smoke_test
    start = time.monotonic()
    if not smoke.version_cmd:
        return AgentTestResult(
            name=manifest.name,
            level=0,
            outcome="skipped",
            latency_ms=0,
            detail="no smoke_test.version_cmd declared",
            rules_failed=("smoke_test.version_cmd_missing",),
        )
    rc, stdout, stderr = await _run(smoke.version_cmd, timeout=10)
    elapsed = int((time.monotonic() - start) * 1000)
    if rc != 0:
        return AgentTestResult(
            name=manifest.name,
            level=0,
            outcome="fail",
            latency_ms=elapsed,
            stderr_tail=_tail(stderr),
            detail=f"version_cmd exited {rc}",
            rules_failed=("smoke.version_cmd_nonzero",),
        )
    detected = ""
    if smoke.version_regex:
        m = re.search(smoke.version_regex, stdout) or re.search(smoke.version_regex, stderr)
        if m is None:
            return AgentTestResult(
                name=manifest.name,
                level=0,
                outcome="fail",
                latency_ms=elapsed,
                stderr_tail=_tail(stdout + stderr),
                detail=f"version_regex {smoke.version_regex!r} did not match",
                rules_failed=("smoke.version_regex_no_match",),
            )
        detected = m.group(1) if m.groups() else m.group(0)
    return AgentTestResult(
        name=manifest.name,
        level=0,
        outcome="pass",
        latency_ms=elapsed,
        version_detected=detected,
    )


async def run_l1_async(
    manifest: AgentManifest,
    provider: str = "",
    model: str = "",
) -> AgentTestResult:
    """L1: L0 + ``[smoke_test].ping_cmd`` against the configured provider.

    *provider* and *model* override the manifest defaults; both are passed
    in via env (``BENCHFLOW_PROVIDER``, ``BENCHFLOW_MODEL``) for the
    ping_cmd to consume.
    """
    l0 = await run_l0_async(manifest)
    if l0.outcome != "pass":
        return l0
    smoke = manifest.smoke_test
    if not smoke.ping_cmd:
        return AgentTestResult(
            name=manifest.name,
            level=1,
            outcome="skipped",
            latency_ms=l0.latency_ms,
            version_detected=l0.version_detected,
            detail="no smoke_test.ping_cmd declared",
            rules_failed=("smoke_test.ping_cmd_missing",),
        )
    start = time.monotonic()
    cmd = smoke.ping_cmd
    if provider:
        cmd = f"BENCHFLOW_PROVIDER={shlex.quote(provider)} " + cmd
    if model:
        cmd = f"BENCHFLOW_MODEL={shlex.quote(model)} " + cmd
    rc, stdout, stderr = await _run(cmd, timeout=smoke.ping_timeout or 30)
    elapsed = int((time.monotonic() - start) * 1000) + l0.latency_ms
    if rc != 0:
        return AgentTestResult(
            name=manifest.name,
            level=1,
            outcome="fail",
            latency_ms=elapsed,
            stderr_tail=_tail(stderr),
            version_detected=l0.version_detected,
            detail=f"ping_cmd exited {rc}",
            rules_failed=("smoke.ping_cmd_nonzero",),
        )
    return AgentTestResult(
        name=manifest.name,
        level=1,
        outcome="pass",
        latency_ms=elapsed,
        version_detected=l0.version_detected,
    )


def _run_sync(coro):
    """Run *coro* in a fresh event loop so we don't perturb the caller's
    default loop policy. ``asyncio.run`` sets None as the running loop on
    exit, which breaks any later ``asyncio.get_event_loop()`` caller in the
    same process (e.g. legacy test fixtures)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def run_l0(manifest_or_name: AgentManifest | str) -> AgentTestResult:
    manifest = (
        manifest_or_name
        if isinstance(manifest_or_name, AgentManifest)
        else _load_manifest(manifest_or_name)
    )
    return _run_sync(run_l0_async(manifest))


def run_l1(
    manifest_or_name: AgentManifest | str,
    provider: str = "",
    model: str = "",
) -> AgentTestResult:
    manifest = (
        manifest_or_name
        if isinstance(manifest_or_name, AgentManifest)
        else _load_manifest(manifest_or_name)
    )
    return _run_sync(run_l1_async(manifest, provider, model))


# ── per-session L1 cache ───────────────────────────────────────────────────

_L1_CACHE: dict[tuple[str, str, str], AgentTestResult] = {}


def cached_l1(name: str, provider: str = "", model: str = "") -> AgentTestResult:
    """Run L1 once per ``(name, provider, model)`` per process; cache after."""
    key = (name, provider, model)
    if key not in _L1_CACHE:
        _L1_CACHE[key] = run_l1(name, provider, model)
    return _L1_CACHE[key]


def clear_l1_cache() -> None:
    """Reset the per-session L1 cache (test helper)."""
    _L1_CACHE.clear()
