"""Known routing variants pass admission.

Launch admission looked up metadata with the ":nitro" still on the id, so an
honest OpenRouter route with a routing-only variant failed to start. Known
routing-only variants are separated before lookup, the full id is still what
the gateway dispatches, and unknown or web-enabling variants stay refused.
"""

import pytest

from benchflow.providers import restricted_models as rm
from benchflow.providers.litellm_request_guard import MODEL_VARIANTS

ROUTING_ONLY = sorted(v for v, why in MODEL_VARIANTS.items() if why is None)


@pytest.mark.parametrize(
    "suffix", [*(f":{v}" for v in ROUTING_ONLY), ":nitro:floor", ":free:nitro"]
)
@pytest.mark.parametrize(
    "model,base",
    [
        ("openrouter/openai/gpt-4o", "openai/gpt-4o"),
        ("openrouter/anthropic/claude-sonnet-4.5", "anthropic/claude-sonnet-4.5"),
        ("openai/gpt-4.1", "gpt-4.1"),
    ],
)
def test_routing_only_variants_are_admitted_by_their_base(model, base, suffix):
    # C24/C25 (85b87bc9): a gateway requires the operator entry even when
    # its fixed model and routing-only variants are known. Keep proving the
    # C26 base parsing and exact dispatch, under the narrower launch policy.
    gateway = model.startswith("openrouter/")
    env = {rm.OPERATOR_ENV: model + suffix} if gateway else {}
    admission = rm.admit_restricted_model(model + suffix, environ=env)
    assert admission.admitted_by == ("operator" if gateway else "metadata")
    assert admission.base == base
    # The admitted model is the full configured id: it is what is dispatched.
    assert admission.model == model + suffix


@pytest.mark.parametrize(
    "model",
    ["openrouter/openai/gpt-4o:turbo", "openrouter/openai/gpt-4o:nitro:turbo"],
)
def test_unknown_variants_still_need_the_operator_entry(model):
    with pytest.raises(rm.RestrictedModelRefused, match=rm.OPERATOR_ENV):
        rm.admit_restricted_model(model, environ={})
    admission = rm.admit_restricted_model(model, environ={rm.OPERATOR_ENV: model})
    assert admission.admitted_by == "operator"


@pytest.mark.parametrize(
    "model",
    ["openrouter/openai/gpt-4o:online", "openrouter/openai/gpt-4o:nitro:online"],
)
def test_web_enabling_variants_stay_refused_even_with_the_entry(model):
    for environ in ({}, {rm.OPERATOR_ENV: model}):
        with pytest.raises(rm.RestrictedModelRefused, match="web search"):
            rm.admit_restricted_model(model, environ=environ)
