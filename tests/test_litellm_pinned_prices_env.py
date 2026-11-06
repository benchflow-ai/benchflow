"""Guards PR #63: gateway startup uses the installed, pinned price table."""

import pytest

from benchflow.providers.litellm_runtime import _litellm_proxy_env, _proxy_process_env


@pytest.mark.parametrize("value", [None, "False"])
def test_proxy_process_pins_prices(value):
    source = {} if value is None else {"LITELLM_LOCAL_MODEL_COST_MAP": value}
    assert (
        _proxy_process_env(source, "/runtime")["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"
    )


def test_proxy_launch_configuration_pins_prices():
    env = _litellm_proxy_env(agent="codex-acp", agent_env={}, required_skill_names=())
    assert env["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"
