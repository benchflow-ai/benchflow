"""A pool of Claude subscriptions for the hill-climb demo: which account runs each job.

A pool file (``--oauth-pool FILE``) holds Claude Code OAuth tokens, one
``CC_OAUTH_<NAME>=<token>`` per line. Before each evaluation job and each
optimizer run, the demo leases the account with the most 5-hour headroom that
no running job holds, and runs the job with that token as its
``CLAUDE_CODE_OAUTH_TOKEN`` (``EvaluationConfig.agent_env``).

Headroom comes from one tiny Messages API request per token (the Claude Code
system prompt, ``claude-haiku-4-5-20251001``, 8 output tokens, the OAuth
headers), cached for five minutes. On a 200 and a 429 alike, the reply's
``anthropic-ratelimit-unified-*`` headers give the share of the 5-hour and
7-day windows already used (the enforced ``7d_oi`` window when present) and
when each resets. An account is left out when the probe is refused, when less
than 15% of its 7-day window is left, when more than 70% of its 5-hour window
is used, or when the job would take it past 80% at the rate the demo's
earlier jobs used accounts. A job that ends on the usage limit ("You've hit
your ... limit") marks its account spent until the window resets.

Token values stay inside this module's objects: every record, warning and
report names the account (the part after ``CC_OAUTH_``), never the secret.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

API_URL = "https://api.anthropic.com/v1/messages"
PROBE_MODEL = "claude-haiku-4-5-20251001"
HEADER = "anthropic-ratelimit-unified-"
PREFIX = "CC_OAUTH_"
# What Claude Code says when a subscription's window is used up; a trial that
# ends with it failed on the account, not on the task.
QUOTA = re.compile(r"hit your [^.\n]{0,40}?limit", re.IGNORECASE)


class NoHeadroom(RuntimeError):
    """No account can take the job, and none will free up soon."""


@dataclass
class Limits:
    probe_ttl_sec: float = 300.0
    max_5h: float = 0.70  # leave out an account with more of its 5-hour window used
    target_5h: float = 0.80  # nor start a job that would take it past this
    min_7d_left: float = 0.15  # nor one with less of its 7-day window left
    max_wait_sec: float = 2700.0  # wait this long at most for a window to reset
    # Share of a 5-hour window one rollout uses, until jobs have measured it:
    # a Haiku trial, and an Opus optimizer run.
    rate: dict[str, float] = field(
        default_factory=lambda: {"agent": 0.01, "optimizer": 0.10}
    )


@dataclass(eq=False)
class Account:
    name: str
    token: str = field(repr=False)
    probe: dict = field(default_factory=dict)  # the last probe's headers, no secret
    probed_at: float | None = None
    out_until: float | None = None  # spent or refused until then (inf: for good)
    why_out: str | None = None
    held_by: str | None = None
    held_util: float | None = None  # its 5-hour use when the current job started
    jobs: int = 0
    rollouts: int = 0
    optimizer_runs: int = 0
    quota_failures: int = 0


def read_pool(path: str | Path) -> dict[str, str]:
    """``{name: token}`` from ``NAME=token`` lines (``export`` and quotes allowed)."""
    tokens = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip().strip("'\"")
        if value:
            tokens[key.removeprefix(PREFIX)] = value
    return tokens


def _number(value: Any) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_headers(headers: Any) -> dict:
    """The unified rate-limit headers as numbers; ``util_7d`` is the enforced window."""
    h = {str(k).lower(): v for k, v in dict(headers or {}).items()}
    out = {
        f"{kind}_{w}": _number(h.get(f"{HEADER}{w}-{name}"))
        for w in ("5h", "7d", "7d_oi")
        for kind, name in (("util", "utilization"), ("reset", "reset"))
    }
    if out["util_7d_oi"] is not None:
        out["util_7d"], out["reset_7d"] = out["util_7d_oi"], out["reset_7d_oi"]
    out["status"] = h.get(f"{HEADER}status")
    out["rejected"] = sorted(
        m[1]
        for k, v in h.items()
        if v == "rejected" and (m := re.fullmatch(rf"{HEADER}(.+)-status", k))
    )
    return out


def probe(token: str, *, opener: Callable | None = None, timeout: float = 20.0) -> dict:
    """One 8-token request with ``token``; its headroom, whatever the HTTP status."""
    body = {
        "model": PROBE_MODEL,
        "max_tokens": 8,
        "system": "You are Claude Code, Anthropic's official CLI for Claude.",
        "messages": [{"role": "user", "content": "Reply with ready."}],
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "anthropic-beta": "oauth-2025-04-20",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    try:
        reply = (opener or urllib.request.urlopen)(request, timeout=timeout)
        status, headers = int(reply.status), reply.headers
        reply.close()
    except urllib.error.HTTPError as exc:
        status, headers = int(exc.code), exc.headers
    except Exception as exc:  # network, TLS, timeout: headroom unknown
        return {"http_status": None, "error": f"probe failed ({type(exc).__name__})"}
    record = {"http_status": status, **parse_headers(headers)}
    if status not in (200, 429) or record["status"] == "rejected":
        record["error"] = f"HTTP {status}" + (
            f", {record['status']}" if record["status"] else ""
        )
    return record


class Pool:
    """The accounts a climb may run on, probed on demand and leased one job at a time."""

    def __init__(
        self,
        tokens: dict[str, str],
        limits: Limits | None = None,
        *,
        opener: Callable | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.accounts = {n: Account(n, t) for n, t in sorted(tokens.items())}
        self.limits = limits or Limits()
        self.rate = dict(self.limits.rate)
        self._opener, self._clock = opener, clock
        self._cond: asyncio.Condition | None = None

    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> Pool:
        return cls(read_pool(path), **kwargs)

    # -- probing ---------------------------------------------------------------

    def _probe(self, account: Account) -> None:
        account.probe = probe(account.token, opener=self._opener)
        account.probed_at = self._clock()
        util_7d = account.probe.get("util_7d")
        if account.probe.get("http_status") in (401, 403):
            account.out_until, account.why_out = math.inf, account.probe["error"]
        elif util_7d is not None and util_7d > 1 - self.limits.min_7d_left:
            account.out_until = account.probe.get("reset_7d") or math.inf
            account.why_out = f"7-day window {util_7d:.0%} used"

    def refresh(self, *, fresh: Account | None = None) -> None:
        """Probe every account whose probe is older than the TTL (and ``fresh``)."""
        now = self._clock()
        stale = [
            a
            for a in self.accounts.values()
            if a is fresh
            or (
                not self._out(a, now)
                and (
                    a.probed_at is None or now - a.probed_at > self.limits.probe_ttl_sec
                )
            )
        ]
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(self._probe, stale))

    def _out(self, account: Account, now: float) -> bool:
        """Whether ``account`` is spent or refused; forgets a window that has reset."""
        if account.out_until is None:
            return False
        if account.out_until > now:
            return True
        account.out_until = account.why_out = None
        return False

    def why_not(self, account: Account, kind: str, rollouts: int) -> str | None:
        """Why ``account`` cannot take a job of ``rollouts`` rollouts now, or None."""
        p, lim = account.probe, self.limits
        if self._out(account, self._clock()):
            return account.why_out
        if account.held_by:
            return f"held by {account.held_by}"
        if p.get("error"):
            return p["error"]
        util_5h = p.get("util_5h") or 0.0
        if util_5h > lim.max_5h:
            return f"5-hour window {util_5h:.0%} used"
        if util_5h + self.rate[kind] * rollouts > lim.target_5h:
            return f"5-hour window {util_5h:.0%} used, the job would pass {lim.target_5h:.0%}"
        return None

    # -- leasing ---------------------------------------------------------------

    async def lease(self, label: str, *, kind: str, rollouts: int) -> Account:
        """The free account with the most 5-hour headroom; waits while running jobs
        hold the fitting ones, or up to ``max_wait_sec`` for a window to reset."""
        self._cond = self._cond or asyncio.Condition()
        async with self._cond:
            while True:
                await asyncio.to_thread(self.refresh)
                fit = [
                    a
                    for a in self.accounts.values()
                    if self.why_not(a, kind, rollouts) is None
                ]
                if fit:
                    best = min(
                        fit,
                        key=lambda a: (
                            a.probe.get("util_5h") or 0.0,
                            a.probe.get("util_7d") or 0.0,
                            a.name,
                        ),
                    )
                    best.held_by, best.held_util = label, best.probe.get("util_5h")
                    return best
                if any(a.held_by for a in self.accounts.values()):
                    await self._cond.wait()
                    continue
                wait = self._next_reset() - self._clock()
                if wait > self.limits.max_wait_sec:
                    raise NoHeadroom(self.describe(kind, rollouts))
                await asyncio.sleep(max(wait, 0) + 60)

    def _next_reset(self) -> float:
        now, resets = self._clock(), []
        for a in self.accounts.values():
            if a.out_until and a.out_until != math.inf:
                resets.append(a.out_until)
            elif not self._out(a, now) and a.probe.get("reset_5h"):
                resets.append(a.probe["reset_5h"])
        return min(resets, default=math.inf)

    async def release(
        self, account: Account, *, kind: str, rollouts: int, quota: bool
    ) -> None:
        """Free ``account``; learn how much of a window a rollout used, and mark
        it spent until the window resets when the job ended on the usage limit."""
        await asyncio.to_thread(self.refresh, fresh=account)
        after, before = account.probe.get("util_5h"), account.held_util
        if after is not None and before is not None and after >= before and rollouts:
            learned = 0.5 * self.rate[kind] + 0.5 * (after - before) / rollouts
            # no lower than a fifth of the first guess: the headers are coarse
            self.rate[kind] = max(learned, self.limits.rate[kind] / 5)
        if kind == "optimizer":
            account.optimizer_runs += 1
        else:
            account.jobs += 1
        account.rollouts += rollouts
        if quota:
            account.quota_failures += 1
            p = account.probe
            windows = p.get("rejected") or ["5h"]
            resets = [p.get(f"reset_{w}") for w in windows if p.get(f"reset_{w}")]
            account.out_until = max(resets, default=self._clock() + 1800)
            account.why_out = f"hit its {'/'.join(windows)} limit"
        account.held_by = account.held_util = None
        self._cond = self._cond or asyncio.Condition()
        async with self._cond:
            self._cond.notify_all()

    # -- records ---------------------------------------------------------------

    def secrets(self) -> list[str]:
        """The tokens, only to scrub them from what the demo keeps."""
        return [a.token for a in self.accounts.values()]

    def describe(self, kind: str = "agent", rollouts: int = 1) -> str:
        return "; ".join(
            f"{a.name}: {self.why_not(a, kind, rollouts) or 'free'}"
            for a in self.accounts.values()
        )

    def usage(self) -> dict:
        """Per account: jobs, rollouts, quota failures and the last probe (no secret)."""
        return {
            a.name: {
                "jobs": a.jobs,
                "optimizer_runs": a.optimizer_runs,
                "rollouts": a.rollouts,
                "quota_failures": a.quota_failures,
                "util_5h": a.probe.get("util_5h"),
                "util_7d": a.probe.get("util_7d"),
                "reset_5h": a.probe.get("reset_5h"),
                "reset_7d": a.probe.get("reset_7d"),
                "probed_at": a.probed_at,
                "out": a.why_out,
            }
            for a in self.accounts.values()
        }
