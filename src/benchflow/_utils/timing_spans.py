"""Timing spans: the seconds each setup and teardown step takes, in timing.json.

A trial's ``timing.json`` has always held its phases (``environment_setup``,
``agent_setup``, ``agent_execution``, ``verifier``, ``total``). The spans here
add the steps inside and between them that speed work changes, so a before
and an after run are read with the same instrument:

- ``slot_wait``: from taking the job's concurrency slot to the trial start
  (before the start pacer it held the start jitter's in-slot sleep);
  ``start_pace_wait``: above ``--concurrency 16``, the seconds the trial
  waited for its start token, with no slot held (absent at 16 or less).
- ``install_agent``: the agent install itself (not counted in any phase before:
  it runs before the ``agent_setup`` timer starts).
- ``litellm_install``: the in-sandbox LiteLLM proxy's install (inside
  ``agent_setup``).
- ``proxy_health_wait``: from launching the LiteLLM proxy to its first healthy
  answer (inside ``agent_setup``).
- ``acp_connect``: starting the ACP agent and opening its session (inside
  ``agent_setup``).
- ``cloud_metadata_block``: refusing the cloud metadata endpoints to the agent
  user under an open network, an iptables install included (inside
  ``agent_setup``).
- ``firewall_power``: checking that nothing in the image hands the agent user
  the firewall a rule of BenchFlow's holds it with, and dropping NET_ADMIN from
  its commands (inside ``agent_setup``).
- ``harden_before_verify``: one sub-key per step of
  :func:`benchflow.sandbox.lockdown.harden_before_verify` plus ``total``
  (inside ``verifier``).
- ``cleanup``: the trial's teardown (sandbox stop included). In separate
  verifier mode the agent's release moved out of it (below), so it holds
  only what is left.
- ``agent_stop_to_reward``: from the solver's completion (``verify()``'s
  entry) to the rewards being set, in shared and separate mode.
- Separate verifier mode, when the agent's sandbox stops before the transfer:
  ``agent_release``, the teardown steps that read the agent's sandbox or its
  usage proxy (skill export, the usage runtime's stop, the egress proxy's
  stop, usage finalization, session-log pricing) plus the stop itself;
  ``agent_sandbox_stop_before_verify``, that stop alone;
  ``verifier_sandbox_ready_wait``, how long scoring waited for the verifier
  sandbox once the transfer was packed; ``verifier_sandbox_overlap``, the
  seconds both sandboxes existed at once.
- ``daemon_guard``: keeping the agent user from Daytona's root toolbox
  daemon (``lockdown.guard_sandbox_daemon``), an iptables install included,
  during the sandbox user's setup (inside ``sandbox_ready``; only on a
  single-container Daytona sandbox with a sandbox user who runs code).
- ``kept_freezer_probe``: proving the sandbox can hold the services a task
  keeps (x-keep-services) in a frozen cgroup, before the agent installs or
  runs (only for a trial that keeps services).
- ``sandbox_ready``: from the trial start to its sandbox started and its
  agent installed (setup, start and the install), as the trial waited for
  them: with a warm-pool sandbox only the wait below.
- Warm pool (Docker jobs with more trials than ``--concurrency``):
  ``warm_pool_wait``, from the trial start to its prepared sandbox being
  ready (0 when it was ready first); ``warm_pool_idle``, the seconds the
  ready sandbox waited for the trial's slot. Its setup, start and install
  phases and spans keep their own seconds, spent before the trial start;
  ``total`` counts from its setup, so it includes the idle seconds.
- ``oracle_network_policy``: arming and removing a task's network policy
  around the oracle's solve.sh (inside ``agent_execution``; only for an
  oracle on an ``allowlist`` or ``denylist`` task).

A span is absent when its step did not run (the oracle starts no proxy, nop
installs no agent), never 0. A step that runs more than once in a trial (a
second scene's connect) adds to its span. Spans record on failure too: the
seconds spent before an install failed are still spent.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator, Mapping
from contextvars import ContextVar
from typing import Any

SLOT_WAIT = "slot_wait"
START_PACE_WAIT = "start_pace_wait"
INSTALL_AGENT = "install_agent"
LITELLM_INSTALL = "litellm_install"
PROXY_HEALTH_WAIT = "proxy_health_wait"
ACP_CONNECT = "acp_connect"
CLOUD_METADATA_BLOCK = "cloud_metadata_block"
FIREWALL_POWER = "firewall_power"
HARDEN_BEFORE_VERIFY = "harden_before_verify"
CLEANUP = "cleanup"
ORACLE_NETWORK_POLICY = "oracle_network_policy"
KEPT_FREEZER_PROBE = "kept_freezer_probe"
DAEMON_GUARD = "daemon_guard"
AGENT_STOP_TO_REWARD = "agent_stop_to_reward"
VERIFIER_SANDBOX_READY_WAIT = "verifier_sandbox_ready_wait"
AGENT_RELEASE = "agent_release"
AGENT_SANDBOX_STOP_BEFORE_VERIFY = "agent_sandbox_stop_before_verify"
VERIFIER_SANDBOX_OVERLAP = "verifier_sandbox_overlap"
SANDBOX_READY = "sandbox_ready"
WARM_POOL_WAIT = "warm_pool_wait"
WARM_POOL_IDLE = "warm_pool_idle"

#: The keys this module adds to timing.json. Older keys keep their meaning.
SPAN_KEYS = (
    SLOT_WAIT,
    START_PACE_WAIT,
    INSTALL_AGENT,
    LITELLM_INSTALL,
    PROXY_HEALTH_WAIT,
    ACP_CONNECT,
    CLOUD_METADATA_BLOCK,
    FIREWALL_POWER,
    HARDEN_BEFORE_VERIFY,
    CLEANUP,
    ORACLE_NETWORK_POLICY,
    KEPT_FREEZER_PROBE,
    DAEMON_GUARD,
    AGENT_STOP_TO_REWARD,
    VERIFIER_SANDBOX_READY_WAIT,
    AGENT_RELEASE,
    AGENT_SANDBOX_STOP_BEFORE_VERIFY,
    VERIFIER_SANDBOX_OVERLAP,
    SANDBOX_READY,
    WARM_POOL_WAIT,
    WARM_POOL_IDLE,
)

#: The spans recorded as a mapping of sub-keys rather than one number.
NESTED_SPANS = frozenset({HARDEN_BEFORE_VERIFY})

#: ``harden_before_verify``'s sub-keys, one per step, in the order they run.
#: Stable names: a later one-script version of the steps fills the same keys.
HARDEN_STEPS = (
    "kill_sandbox_user_procs",
    "restore_disarmed_delegates",
    "clear_verifier_output",
    "prepare_app_dir",
    "reclaim_disk",
    "restore_workspace",
    "freeze_workspace",
    "cleanup_conftests",
    "build_verifier_env",
)

# Spans keep millisecond precision; the older phases keep their 0.1 s.
_SPAN_DIGITS = 3
_PHASE_DIGITS = 1

_current: ContextVar[dict[str, Any] | None] = ContextVar(
    "benchflow_timing_spans", default=None
)
_pending: ContextVar[dict[str, float] | None] = ContextVar(
    "benchflow_pending_trial_timing", default=None
)


def _clock() -> float:
    return time.monotonic()


def clock() -> float:
    """The monotonic clock spans use (tests replace ``_clock``)."""
    return _clock()


def _add(timing: dict[str, Any], name: str, step: str | None, seconds: float) -> None:
    if step is None:
        timing[name] = timing.get(name, 0.0) + seconds
        return
    steps = timing.get(name)
    if not isinstance(steps, dict):
        steps = timing[name] = {}
    steps[step] = steps.get(step, 0.0) + seconds


@contextlib.contextmanager
def span(
    name: str, step: str | None = None, *, into: dict[str, Any] | None = None
) -> Iterator[None]:
    """Add the seconds the block takes to ``timing[name]`` (or its ``step``).

    ``into`` names the timing dict; without it the span records into the one
    :func:`recording_into` set for this task, and does nothing when none is
    set (a library call outside a trial).
    """
    timing = into if into is not None else _current.get()
    if timing is None:
        yield
        return
    started = _clock()
    try:
        yield
    finally:
        _add(timing, name, step, _clock() - started)


def add(name: str, step: str | None, seconds: float) -> None:
    """Add *seconds* measured elsewhere to ``timing[name]`` (or its ``step``).

    For a step timed where :func:`span` cannot wrap it: the verifier
    preparation script's steps, timed inside the sandbox from its status
    lines. Records into the timing :func:`recording_into` set for this task,
    and does nothing when none is set, as :func:`span` does.
    """
    timing = _current.get()
    if timing is not None:
        _add(timing, name, step, seconds)


def add_into(timing: dict[str, Any], name: str, seconds: float) -> None:
    """Add *seconds* measured elsewhere to ``timing[name]`` of a known trial."""
    _add(timing, name, None, seconds)


@contextlib.contextmanager
def recording_into(timing: dict[str, Any]) -> Iterator[None]:
    """Let spans in code below (proxy, hardening) record into ``timing``."""
    token = _current.set(timing)
    try:
        yield
    finally:
        _current.reset(token)


def set_pending_trial_timing(**seconds: float) -> None:
    """Hand ``slot_wait`` (and ``start_pace_wait``) to the trial about to start.

    The job knows them before the rollout exists; the rollout takes them with
    :func:`take_pending_trial_timing` when it is made, in the same task.
    """
    _pending.set(dict(seconds))


def take_pending_trial_timing() -> dict[str, float]:
    """The waits the job handed over, once: a retried attempt gets none."""
    pending = _pending.get()
    if not pending:
        return {}
    taken = dict(pending)
    pending.clear()
    return taken


def rounded(timing: Mapping[str, Any]) -> dict[str, Any]:
    """timing.json's values: phases to 0.1 s as before, spans to 1 ms."""
    out: dict[str, Any] = {}
    for key, value in timing.items():
        if isinstance(value, dict):
            out[key] = {
                step: round(seconds, _SPAN_DIGITS) for step, seconds in value.items()
            }
        elif key in SPAN_KEYS:
            out[key] = round(value, _SPAN_DIGITS)
        else:
            out[key] = round(value, _PHASE_DIGITS)
    return out


def flat_items(timing: Mapping[str, Any]) -> Iterator[tuple[str, Any]]:
    """Every (key, value) of a timing block, ``harden_before_verify``'s steps as
    ``harden_before_verify.<step>``.

    For readers that want one value per key (totals, the viewer). Any other
    key, an old timing.json's included, comes back as it is, whatever its shape.
    """
    for key, value in timing.items():
        if key in NESTED_SPANS and isinstance(value, Mapping):
            for step, seconds in value.items():
                yield f"{key}.{step}", seconds
        else:
            yield str(key), value
