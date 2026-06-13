from __future__ import annotations

import os

import pytest

from benchflow.providers.litellm_bedrock_patch import (
    BEDROCK_ADAPTIVE_THINKING_RE,
    BEDROCK_THINKING_EFFORT_ENV,
)
from benchflow.providers.litellm_config import resolve_litellm_route


@pytest.mark.parametrize(
    "model",
    [
        "us.anthropic.claude-opus-4-8",
        "global.anthropic.claude-sonnet-4-9",
        "anthropic.claude-haiku-4-10",
        "us.anthropic.claude-fable-5",
    ],
)
def test_provider_patch_matcher_covers_bedrock_claude_4_8_plus(model):
    assert BEDROCK_ADAPTIVE_THINKING_RE.search(model)


@pytest.mark.parametrize(
    "model",
    [
        "us.anthropic.claude-opus-4-7",
        "claude-3-7-sonnet",
        "gemini-3.5-flash",
    ],
)
def test_provider_patch_matcher_rejects_older_or_non_claude_models(model):
    assert BEDROCK_ADAPTIVE_THINKING_RE.search(model) is None


def test_bedrock_thinking_effort_is_threaded_into_route_params():
    route = resolve_litellm_route(
        "aws-bedrock/us.anthropic.claude-opus-4-8",
        {
            "AWS_BEARER_TOKEN_BEDROCK": "token",
            "AWS_REGION": "us-west-2",
            BEDROCK_THINKING_EFFORT_ENV: "medium",
        },
    )

    assert route.upstream_model == "bedrock/us.anthropic.claude-opus-4-8"
    assert route.litellm_params["reasoning_effort"] == "medium"


@pytest.mark.parametrize("requested", ["xhigh", "max"])
def test_bedrock_effort_clamps_unsupported_to_litellm_ceiling(requested):
    """Guards #737: LiteLLM rejects xhigh/max for opus-4-8 with a
    BadRequestError, so BenchFlow clamps a requested xhigh/max down to the
    accepted ceiling (`high`) — a MAX-thinking config runs at the real maximum
    instead of erroring mid-run, matching the garbage->high default behavior."""
    route = resolve_litellm_route(
        "aws-bedrock/us.anthropic.claude-opus-4-8",
        {
            "AWS_BEARER_TOKEN_BEDROCK": "token",
            "AWS_REGION": "us-west-2",
            BEDROCK_THINKING_EFFORT_ENV: requested,
        },
    )
    assert route.litellm_params["reasoning_effort"] == "high"


def test_bedrock_fable5_thinking_effort_is_threaded_into_route_params():
    route = resolve_litellm_route(
        "aws-bedrock/us.anthropic.claude-fable-5",
        {
            "AWS_BEARER_TOKEN_BEDROCK": "token",
            "AWS_REGION": "us-west-2",
            BEDROCK_THINKING_EFFORT_ENV: "max",
        },
    )

    assert route.upstream_model == "bedrock/us.anthropic.claude-fable-5"
    assert route.litellm_params["reasoning_effort"] == "max"


def test_bedrock_thinking_effort_defaults_to_high_and_rejects_garbage():
    base_env = {"AWS_BEARER_TOKEN_BEDROCK": "token", "AWS_REGION": "us-west-2"}

    default_route = resolve_litellm_route(
        "aws-bedrock/us.anthropic.claude-opus-4-8", base_env
    )
    assert default_route.litellm_params["reasoning_effort"] == "high"

    garbage_route = resolve_litellm_route(
        "aws-bedrock/us.anthropic.claude-opus-4-8",
        {**base_env, BEDROCK_THINKING_EFFORT_ENV: "turbo"},
    )
    assert garbage_route.litellm_params["reasoning_effort"] == "high"


# --------------------------------------------------------------------------- #
# End-to-end run-level effort -> Bedrock wire payload, Docker/Daytona parity   #
# (issue #599: the old host BedrockProxyServer stored per-run env but the      #
# translators read process-global os.environ, so Docker silently fell back to #
# `high`. PR #613 deleted that proxy; this pins the run-level behavior in the  #
# replacement LiteLLM runtime so the regression cannot return.)               #
# --------------------------------------------------------------------------- #

# Versioned Bedrock inference-profile ID — the real wire form. Stock litellm
# 1.88.0rc1 does not classify it as adaptive-thinking; the bedrock patch's
# anthropic gate does, which is why output_config.effort is emitted at all.
_BEDROCK_OPUS_48_WIRE_MODEL = "bedrock/us.anthropic.claude-opus-4-8-20251101-v1:0"
_BEDROCK_BASE_ENV = {"AWS_BEARER_TOKEN_BEDROCK": "token", "AWS_REGION": "us-west-2"}


def _wire_output_config_effort(reasoning_effort: str, env_override: str | None) -> str:
    """Run the REAL litellm Bedrock Converse transform (with the benchflow
    patch applied) and return the ``output_config.effort`` it emits.

    ``reasoning_effort`` models the value baked into config.yaml from the route
    (``litellm_config`` reads it from the run-level agent env). ``env_override``
    models the proxy *process* environment, which is launched as
    ``os.environ + agent_env`` — so a run-level value reaches it too. Either
    path landing the effort in the wire payload proves the #599 fix end to end.
    """
    from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig

    import benchflow.providers.litellm_bedrock_patch  # noqa: F401 — applies patch

    saved = os.environ.get(BEDROCK_THINKING_EFFORT_ENV)
    os.environ.pop(BEDROCK_THINKING_EFFORT_ENV, None)
    if env_override is not None:
        os.environ[BEDROCK_THINKING_EFFORT_ENV] = env_override
    try:
        optional_params: dict = {}
        AmazonConverseConfig()._handle_reasoning_effort_parameter(
            _BEDROCK_OPUS_48_WIRE_MODEL, reasoning_effort, optional_params
        )
    finally:
        os.environ.pop(BEDROCK_THINKING_EFFORT_ENV, None)
        if saved is not None:
            os.environ[BEDROCK_THINKING_EFFORT_ENV] = saved
    cfg = optional_params.get("output_config")
    assert isinstance(cfg, dict), (
        f"adaptive-thinking output_config not emitted (patch inactive?): "
        f"{optional_params!r}"
    )
    return cfg["effort"]


def test_run_level_effort_from_route_lands_in_bedrock_wire_payload(monkeypatch):
    """Guards PR #613 against #599's run-env regression on the route path.

    A run-level effort baked into config.yaml reaches the Bedrock Converse
    ``output_config.effort`` with the host process env empty, proving it is
    sourced from the run, not os.environ.

    Uses ``medium`` (litellm-accepted and distinct from the ``high`` default)
    so the assertion fails if the effort silently falls back.
    """
    monkeypatch.delenv(BEDROCK_THINKING_EFFORT_ENV, raising=False)
    assert _wire_output_config_effort("medium", env_override=None) == "medium"


def test_run_level_effort_via_proxy_process_env_overrides_stale_route(monkeypatch):
    """Guards PR #613 against #599's run-env regression on the proxy-env path.

    Even if config.yaml carried a stale default (``high``), the run-level value
    present in the proxy process env (os.environ + agent_env) overrides it in
    the wire payload, matching the divergence the old Docker translator got
    wrong.
    """
    monkeypatch.delenv(BEDROCK_THINKING_EFFORT_ENV, raising=False)
    assert _wire_output_config_effort("high", env_override="medium") == "medium"


def test_docker_and_daytona_resolve_identical_bedrock_effort_from_run_env(monkeypatch):
    """Guards PR #613 against #599's Docker/Daytona run-env parity regression.

    Docker (host proxy) and Daytona (sandbox proxy) build the route from the
    SAME ``resolve_litellm_route(model, agent_env)`` call, so the effort in
    config.yaml is identical and independent of the host's os.environ.

    The old bug was Docker-only because a host-side translator read process
    os.environ; here, with that env scrubbed, the run-level value still flows
    through — so neither lane can diverge.
    """
    monkeypatch.delenv(BEDROCK_THINKING_EFFORT_ENV, raising=False)
    run_env = {**_BEDROCK_BASE_ENV, BEDROCK_THINKING_EFFORT_ENV: "medium"}

    # Identical resolution path for both execution environments.
    docker_route = resolve_litellm_route(
        "aws-bedrock/us.anthropic.claude-opus-4-8-20251101-v1:0", run_env
    )
    daytona_route = resolve_litellm_route(
        "aws-bedrock/us.anthropic.claude-opus-4-8-20251101-v1:0", dict(run_env)
    )
    assert (
        docker_route.litellm_params["reasoning_effort"]
        == daytona_route.litellm_params["reasoning_effort"]
        == "medium"
    )
    # And the host env being empty did not strip it — it came from the run env.
    assert BEDROCK_THINKING_EFFORT_ENV not in os.environ


# --------------------------------------------------------------------------- #
# Clamp end-to-end + drift guard (issue #737)                                 #
# --------------------------------------------------------------------------- #


def test_proxy_env_max_override_clamps_in_wire_payload(monkeypatch):
    """#737 (patch path): a run-level `max` in the proxy process env is clamped
    to the LiteLLM ceiling in the actual wire payload — it must NOT raise."""
    monkeypatch.delenv(BEDROCK_THINKING_EFFORT_ENV, raising=False)
    assert _wire_output_config_effort("high", env_override="max") == "high"


def test_every_requestable_bedrock_effort_survives_real_litellm_transform(monkeypatch):
    """Drift guard for #737: every effort a user may request must, after the
    route clamp, be accepted by the REAL litellm Converse transform (no
    BadRequestError). If a future litellm changes its accepted set, this fails
    and forces _BEDROCK_LITELLM_MAX_EFFORT / the ladder to be revisited rather
    than silently shipping a value the provider rejects mid-run.
    """
    from benchflow.providers.litellm_config import _BEDROCK_EFFORT_LADDER

    monkeypatch.delenv(BEDROCK_THINKING_EFFORT_ENV, raising=False)
    for requested in _BEDROCK_EFFORT_LADDER:
        route = resolve_litellm_route(
            "aws-bedrock/us.anthropic.claude-opus-4-8-20251101-v1:0",
            {**_BEDROCK_BASE_ENV, BEDROCK_THINKING_EFFORT_ENV: requested},
        )
        clamped = route.litellm_params["reasoning_effort"]
        # Must not raise — this is the whole point of the clamp.
        effort = _wire_output_config_effort(clamped, env_override=None)
        assert effort, f"{requested!r} -> {clamped!r} produced no effort"
