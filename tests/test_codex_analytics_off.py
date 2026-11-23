"""Codex's own analytics are off in every BenchFlow launch, ACP and native.

Competitor telemetry is off by rule. The native harness always passed
``-c analytics.enabled=false`` before the run's CODEX_CONFIG; codex-acp got it
only under no-web, so an ordinary ACP run left Codex's analytics on unless
the user's CODEX_CONFIG said otherwise. Both now take it from the config the
launch renders, and a value the user sets explicitly is kept, on both.
"""

from __future__ import annotations

import json

import pytest

from benchflow.agents.codex_config import (
    CODEX_CONFIG_ENV,
    apply_codex_launch_config,
    apply_codex_provider_config,
)
from benchflow.native_harness.codex import codex_launch
from benchflow.native_harness.spec import NativeTurn


def _provider_env(user: dict | None = None) -> dict[str, str]:
    env: dict[str, str] = {"OPENAI_API_KEY": "proxy-key"}
    if user is not None:
        env[CODEX_CONFIG_ENV] = json.dumps(user)
    apply_codex_provider_config(
        env,
        base_url="http://172.17.0.1:4000/v1",
        model="deepseek-flash",
        provider_name="litellm",
        strict=True,
    )
    return env


def _launch(env: dict[str, str]) -> dict:
    launched, _ = apply_codex_launch_config(
        "codex-acp", env, model="deepseek/deepseek-flash", reasoning_effort=None
    )
    return json.loads(launched.get(CODEX_CONFIG_ENV) or "{}")


def _native_analytics(config: dict) -> str:
    """The value native ``codex exec`` runs with: its last ``-c`` wins."""
    argv = codex_launch(NativeTurn(cwd="/app"), config).argv
    values = [a.split("=", 1)[1] for a in argv if a.startswith("analytics.enabled=")]
    nested = [a for a in argv if a.startswith("analytics=")]
    assert values and not nested, argv
    return values[-1]


def test_an_acp_launch_turns_codex_analytics_off():
    assert _launch(_provider_env())["analytics"] == {"enabled": False}


def test_an_acp_launch_with_no_codex_config_turns_them_off_too():
    assert _launch({}) == {"analytics": {"enabled": False}}


def test_other_analytics_settings_the_user_wrote_are_kept():
    config = _launch(_provider_env({"analytics": {"enabled": False, "x": 1}}))
    assert config["analytics"] == {"enabled": False, "x": 1}


@pytest.mark.parametrize("enabled", [True, False])
def test_a_users_explicit_choice_is_kept_by_both_harnesses(enabled):
    config = _launch(_provider_env({"analytics": {"enabled": enabled}}))
    assert config["analytics"] == {"enabled": enabled}
    assert _native_analytics(config) == ("true" if enabled else "false")


def test_acp_and_native_run_with_the_same_default():
    config = _launch(_provider_env())
    assert config["analytics"]["enabled"] is False
    assert _native_analytics(config) == "false"
