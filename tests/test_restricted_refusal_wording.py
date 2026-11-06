"""Guards 7fefe2b0: restricted-web refusals lead with the verdict and next step.

A refusal used to open with the policy ("With web access restricted, BenchFlow
runs only models it can identify ...") and name the refused model only in its
second sentence. Now the first words say what was refused, and ``next:`` says
what to do.
"""

import pytest

from benchflow.providers import litellm_runtime as runtime
from benchflow.providers import restricted_models as rm


def test_an_unidentified_model_is_refused_by_name_first():
    model = "openai/mimo-v2.6-flash"
    with pytest.raises(rm.RestrictedModelRefused) as refused:
        rm.admit_restricted_model(model, environ={})
    text = str(refused.value)
    assert text.startswith(f"Refused {model!r} with web access restricted: ")
    assert (
        f"next: if it runs without web access, set {rm.OPERATOR_ENV}={model} " in text
    )


def test_a_self_searching_model_is_refused_by_name_first():
    model = "openrouter/openai/gpt-4o:online"
    with pytest.raises(rm.RestrictedModelRefused) as refused:
        rm.admit_restricted_model(model, environ={rm.OPERATOR_ENV: model})
    text = str(refused.value)
    assert text.startswith(f"Refused {model!r} with web access restricted: ")
    assert text.endswith("next: choose a model without hosted web search.")


def test_a_cloud_alias_names_the_variable_and_never_its_value():
    env = {"AWS_BEARER_TOKEN_BEDROCK": "dummy", "AWS_REGION": "us-east-1"}
    route = runtime.resolve_litellm_route(
        "aws-bedrock/invoke/amazon.nova-pro-v1:0", env
    )
    alias = {**env, "AWS_REGION": "private-alias.example/us-east-1"}
    with pytest.raises(rm.RestrictedModelRefused) as refused:
        rm.admit_restricted_route(route, alias)
    text = str(refused.value)
    assert text.startswith("Refused the bedrock route with web access restricted: ")
    assert "AWS_REGION must be a plain name" in text
    assert "private-alias" not in text
