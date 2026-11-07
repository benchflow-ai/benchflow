"""The no-web model gate names, and accepts, the model id the user typed.

`--model xiaomi/mimo-v2.6-pro` goes upstream as `openai/mimo-v2.6-pro`; the
refusal used to name only the upstream id and told the user to allow it,
an id they never typed (seen in a 0.10.0rc1 acceptance run).
"""

from __future__ import annotations

import pytest

from benchflow.providers import restricted_models as rm


def test_the_refusal_names_the_id_the_user_typed():
    with pytest.raises(rm.RestrictedModelRefused) as refused:
        rm.admit_restricted_model(
            "openai/mimo-v2.6-pro",
            provider_name="xiaomi",
            requested="xiaomi/mimo-v2.6-pro",
            environ={},
        )
    message = str(refused.value)
    assert message.startswith(
        "Refused 'xiaomi/mimo-v2.6-pro' (sent to the provider as 'openai/mimo-v2.6-pro') "
        "with web access restricted: "
    )
    assert f"{rm.OPERATOR_ENV}=xiaomi/mimo-v2.6-pro " in message


@pytest.mark.parametrize(
    "entry", ["xiaomi/mimo-v2.6-pro", "openai/mimo-v2.6-pro", "xiaomi/*"]
)
def test_either_id_or_the_provider_admits_it(entry):
    admission = rm.admit_restricted_model(
        "openai/mimo-v2.6-pro",
        provider_name="xiaomi",
        requested="xiaomi/mimo-v2.6-pro",
        environ={rm.OPERATOR_ENV: entry},
    )
    assert admission.admitted_by == "operator"
