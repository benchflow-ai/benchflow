"""Live smoke test for SDK.run() against a real environment.

Run:
    pytest -m live tests/test_smoke.py

Bare ``pytest tests/test_smoke.py`` will silently report "1 deselected" because
the ``addopts = "-m 'not live'"`` filter in pyproject.toml applies to direct
file invocation too.

Importing ``benchflow.sdk`` triggers ``_patch_harbor_dind()`` at sdk.py:135.
That patch is gated on ``/.dockerenv`` and runs ``docker info`` with a 5s
timeout, swallowing all exceptions — safe but worth flagging.

Cost / runtime budget (for the green path against claude-agent-acp + Haiku 4.5):
- Cold: 90-180s (apt + node 22 + npm install @zed-industries/claude-agent-acp,
  plus ubuntu:24.04 pull, plus model latency)
- Warm: 30-60s
- ~$0.005 per run on Haiku 4.5
- 1-3% flake rate from model variability against the strict-equality verifier
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from benchflow import SDK
from benchflow._env_setup import _detect_dind_mount

HELLO_TASK = Path(__file__).parent / "examples" / "hello-world-task"
SMOKE_JOBS_BASE = Path(__file__).parent / ".smoke-jobs"


def _smoke_skip_reason() -> str | None:
    """Return a skip reason or None.

    Pure function — must not be evaluated at decorator time. The fixture below
    defers the docker subprocess until the test is actually selected.

    Checks:
    - docker CLI present (cheap, no subprocess)
    - docker daemon reachable (3s timeout to kill hangs on misconfigured DOCKER_HOST)
    - ANTHROPIC_API_KEY env var OR ~/.claude/.credentials.json (matches what
      resolve_agent_env at _agent_env.py accepts for claude-agent-acp's
      subscription_auth path)

    Deliberately does NOT call resolve_agent_env — the test exercises that code
    path; skipping when it raises would mask real regressions.
    """
    if shutil.which("docker") is None:
        return "docker CLI not installed"
    try:
        r = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            timeout=3,
            capture_output=True,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        return f"docker daemon unreachable: {e}"
    if r.returncode != 0:
        return "docker daemon unreachable"
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_login = Path("~/.claude/.credentials.json").expanduser().is_file()
    if not (has_key or has_login):
        return "no ANTHROPIC_API_KEY and no ~/.claude/.credentials.json"
    return None


@pytest.fixture(scope="session")
def smoke_prereqs() -> bool:
    """Session-scoped prereq check.

    Cached so the docker subprocess fires at most once per pytest session, and
    only when a live test is actually selected. Replaces the naive
    ``@pytest.mark.skipif(_smoke_skip_reason() is not None, ...)`` pattern,
    which evaluates at decorator (collection) time on every pytest invocation.
    """
    reason = _smoke_skip_reason()
    if reason:
        pytest.skip(reason)
    return True


@pytest.fixture
def smoke_jobs_dir(tmp_path: Path) -> Iterator[Path]:
    """A jobs_dir whose host docker daemon can bind-mount it.

    Outside DinD: ``tmp_path`` is fine — pytest's tmp lives on a real host fs.

    Inside DinD (devcontainer that shares the host docker socket): pytest's
    ``tmp_path`` is on the container's overlay/tmpfs and has no host-side
    equivalent, so Harbor's ``HOST_VERIFIER_LOGS_PATH`` bind mount silently
    maps to nothing — verifier writes to the bind, the host loses them, and
    ``reward.txt`` never appears. ``_patch_harbor_dind`` only translates paths
    under the workspace mount, so we use a workspace-rooted directory in that
    case.

    Cleanup is best-effort: trial files written from the container as root
    may not be removable by our (non-root) test user.
    """
    if _detect_dind_mount() is None:
        yield tmp_path
        return

    SMOKE_JOBS_BASE.mkdir(exist_ok=True)
    d = SMOKE_JOBS_BASE / f"run-{uuid.uuid4().hex[:8]}"
    d.mkdir()
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.mark.live
@pytest.mark.asyncio
async def test_hello_world_smoke(smoke_prereqs: bool, smoke_jobs_dir: Path) -> None:
    """End-to-end: claude-agent-acp + Haiku 4.5 solves hello-world-task.

    Asserts the minimal set that proves the orchestration pipeline ran:
    - Verifier produced reward 1.0 (strict equality on "Hello, world!")
    - No infra error and no verifier error
    - Agent used at least one tool (n_tool_calls is ACP-sourced and never
      overwritten by scraped fallback — see sdk.py:83-84,540)
    - Trajectory file exists and is non-empty
    """
    result = await SDK().run(
        task_path=HELLO_TASK,
        agent="claude-agent-acp",
        model="claude-haiku-4-5-20251001",
        jobs_dir=smoke_jobs_dir,
    )

    assert result.rewards is not None
    assert result.rewards.get("reward") == 1.0
    assert result.error is None
    assert result.verifier_error is None
    assert result.n_tool_calls > 0

    # trial_dir = jobs_dir / job_name / trial_name (sdk.py:166).
    # job_name is an auto-generated timestamp, so glob for it.
    matches = list(
        smoke_jobs_dir.glob(f"*/{result.trial_name}/trajectory/acp_trajectory.jsonl")
    )
    assert len(matches) == 1, f"expected exactly one trajectory, found {matches}"
    assert matches[0].stat().st_size > 0
