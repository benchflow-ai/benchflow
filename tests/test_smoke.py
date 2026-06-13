"""Live smoke test for SDK.run() against a real environment.

Run:
    pytest -m live tests/test_smoke.py

Bare ``pytest tests/test_smoke.py`` will silently report "1 deselected" because
the ``addopts = "-m 'not live'"`` filter in pyproject.toml applies to direct
file invocation too.

Default agent/model is ``claude-agent-acp`` + Haiku 4.5. A contributor whose
only credential is non-Anthropic can point the smoke at an agent/model they can
actually authenticate via two env vars (proven combo: openhands + deepseek):

    export BENCHFLOW_SMOKE_AGENT=openhands
    export BENCHFLOW_SMOKE_MODEL=deepseek/deepseek-chat
    export DEEPSEEK_API_KEY=...   DEEPSEEK_BASE_URL=https://api.deepseek.com
    pytest -m live tests/test_smoke.py

The skip reason is always specific (which credential is missing for the chosen
model) so a release gate can grep the pytest summary and refuse to call a
skipped live smoke "green" — see ``docs/release.md`` and the launch-prep skill.

Importing ``benchflow.sdk`` triggers ``_patch_dind()`` at sdk.py:135.
That patch is gated on ``/.dockerenv`` and runs ``docker info`` with a 5s
timeout, swallowing all exceptions — safe but worth flagging.

Cost / runtime budget (for the green path against claude-agent-acp + Haiku 4.5):
- Cold: 90-180s (apt + node 22 + npm install @agentclientprotocol/claude-agent-acp,
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
from benchflow.sandbox.setup import _detect_dind_mount

HELLO_TASK = Path(__file__).parent / "examples" / "hello-world-task"
SMOKE_JOBS_BASE = Path(__file__).parent / ".smoke-jobs"

DEFAULT_SMOKE_AGENT = "claude-agent-acp"
DEFAULT_SMOKE_MODEL = "claude-haiku-4-5-20251001"

SMOKE_AGENT_ENV = "BENCHFLOW_SMOKE_AGENT"
SMOKE_MODEL_ENV = "BENCHFLOW_SMOKE_MODEL"


def resolve_smoke_target() -> tuple[str, str]:
    """Return the (agent, model) the live smoke will run.

    Defaults to claude-agent-acp + Haiku 4.5. ``BENCHFLOW_SMOKE_AGENT`` and
    ``BENCHFLOW_SMOKE_MODEL`` override both (the escape hatch for contributors
    without an Anthropic credential). Both must be set together; setting only
    one is a configuration error, not a silent fall-back to the Anthropic
    default the contributor cannot authenticate.
    """
    agent = os.environ.get(SMOKE_AGENT_ENV)
    model = os.environ.get(SMOKE_MODEL_ENV)
    if (agent is None) != (model is None):
        missing = SMOKE_MODEL_ENV if agent is not None else SMOKE_AGENT_ENV
        raise RuntimeError(
            f"{SMOKE_AGENT_ENV} and {SMOKE_MODEL_ENV} must be set together; "
            f"{missing} is unset."
        )
    if agent and model:
        return agent, model
    return DEFAULT_SMOKE_AGENT, DEFAULT_SMOKE_MODEL


def _missing_model_credentials(model: str) -> str | None:
    """Return a reason string if the chosen model has no usable credential.

    The default Anthropic model accepts either ``ANTHROPIC_API_KEY`` or
    ``~/.claude/.credentials.json`` (subscription OAuth) — the dual path
    ``resolve_agent_env`` honors for claude-agent-acp. Any other model is
    validated against the credential env vars its registered provider needs:
    the inferred API key plus every ``url_params`` env var (e.g. deepseek's
    ``DEEPSEEK_BASE_URL``), so the skip reason names the missing var.
    """
    from benchflow.agents.providers import find_provider
    from benchflow.agents.registry import infer_env_key_for_model

    provider = find_provider(model)
    if provider is None:
        # No registered provider: only the built-in Anthropic dual path is
        # special-cased; the heuristic key still applies as a fallback.
        required = infer_env_key_for_model(model)
        if required == "ANTHROPIC_API_KEY":
            has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
            has_login = Path("~/.claude/.credentials.json").expanduser().is_file()
            if has_key or has_login:
                return None
            return "no ANTHROPIC_API_KEY and no ~/.claude/.credentials.json"
        if required and not os.environ.get(required):
            return f"no {required} for model {model!r}"
        return None

    _, cfg = provider
    needed: list[str] = []
    if cfg.auth_type == "api_key" and cfg.auth_env:
        needed.append(cfg.auth_env)
    needed.extend(cfg.url_params.values())
    missing = [var for var in needed if not os.environ.get(var)]
    if missing:
        return f"missing {', '.join(missing)} for model {model!r}"
    return None


def _smoke_skip_reason() -> str | None:
    """Return a skip reason or None.

    Pure function — must not be evaluated at decorator time. The fixture below
    defers the docker subprocess until the test is actually selected.

    Checks:
    - docker CLI present (cheap, no subprocess)
    - docker daemon reachable (3s timeout to kill hangs on misconfigured DOCKER_HOST)
    - credentials for the resolved smoke model (the Anthropic dual key/login
      path for the default, or the provider's required env vars for an
      escape-hatch model — see ``_missing_model_credentials``)

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
    _, model = resolve_smoke_target()
    return _missing_model_credentials(model)


@pytest.fixture(scope="session")
def smoke_prereqs() -> bool:
    """Session-scoped prereq check.

    Cached so the docker subprocess fires at most once per pytest session, and
    only when a live test is actually selected. Replaces the naive
    ``@pytest.mark.skipif(_smoke_skip_reason() is not None, ...)`` pattern,
    which evaluates at decorator (collection) time on every pytest invocation.

    A skip here is intentionally loud at the gate level: ``docs/release.md`` and
    the launch-prep skill run the live smoke with ``-ra`` and fail the gate if
    the summary reports this test as skipped, so a missing credential cannot
    false-green the e2e step on a run that never executed.
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
    equivalent, so the ``HOST_VERIFIER_LOGS_PATH`` bind mount silently
    maps to nothing — verifier writes to the bind, the host loses them, and
    ``reward.txt`` never appears. ``_patch_dind`` only translates paths
    under the workspace mount, so we use a workspace-rooted directory in that
    case.

    Cleanup is best-effort: trial files written from the container as root
    may not be removable by our (non-root) test user.
    """
    if _detect_dind_mount() is None:
        if Path("/.dockerenv").exists():
            pytest.fail(
                "Running inside DinD (/.dockerenv present) but cwd is not under any "
                "container bind mount. Move your checkout under a bind-mounted path "
                "(e.g. /workspace) so host docker can see it."
            )
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
@pytest.mark.usefixtures("smoke_prereqs")
async def test_hello_world_smoke(smoke_jobs_dir: Path) -> None:
    """End-to-end: the resolved agent + model solves hello-world-task.

    Defaults to claude-agent-acp + Haiku 4.5; ``BENCHFLOW_SMOKE_AGENT`` /
    ``BENCHFLOW_SMOKE_MODEL`` redirect it at any other ACP agent/model the
    contributor can authenticate (e.g. openhands + deepseek).

    Asserts the minimal set that proves the orchestration pipeline ran:
    - Verifier produced reward 1.0 (strict equality on "Hello, world!")
    - No infra error and no verifier error
    - Agent used at least one tool (n_tool_calls is ACP-sourced and never
      overwritten by scraped fallback — see sdk.py:83-84,540)
    - Trajectory file exists and is non-empty
    """
    agent, model = resolve_smoke_target()
    result = await SDK().run(
        task_path=HELLO_TASK,
        agent=agent,
        model=model,
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
        smoke_jobs_dir.glob(f"*/{result.rollout_name}/trajectory/acp_trajectory.jsonl")
    )
    assert len(matches) == 1, f"expected exactly one trajectory, found {matches}"
    assert matches[0].stat().st_size > 0
