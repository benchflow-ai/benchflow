"""result.json error text keeps its trajectory redactor.

``_masked`` masks the error text a result.json keeps: in
``benchflow._utils.evaluation_results`` for the result.json of a trial that
left no folder and the summary rows, and in ``benchflow.rollout._results``
for a trial's own result.json. Both call ``redact_trajectory_text``,
imported at the top of the module. A merge that resolves a conflict in that
import block by taking a side without ``redact_trajectory_text`` still
imports cleanly: ``_masked`` breaks only when an error string reaches it,
and the error text goes unmasked or the write fails. These tests fail on
that, by behaviour and by identity.

Every value here is made up; the key-shaped ones are built at run time.
"""

from __future__ import annotations

import pytest

from benchflow._utils import evaluation_results
from benchflow.models import RolloutResult
from benchflow.rollout import _results as rollout_results
from benchflow.trajectories import types as trajectory_types

MASK = "***REDACTED***"

ANTHROPIC_KEY = "sk-ant-api03-" + "FAKE" * 4 + "1234"
GEMINI_KEY = "AIzaSy" + "FAKE" * 5 + "x9"
TOKEN = "FAKEtok" + "0123456789abcdef"

# (error text, the secret in it): a prefixed key, a NAME=value carrier, a
# header and a URL credential, as a provider's error or a traceback quotes one.
ERRORS = [
    pytest.param(
        f"Error code: 401 - invalid x-api-key {ANTHROPIC_KEY}",
        ANTHROPIC_KEY,
        id="anthropic-key",
    ),
    pytest.param(
        f"400 API key not valid: key={GEMINI_KEY}", GEMINI_KEY, id="gemini-key"
    ),
    pytest.param(f"export GITHUB_TOKEN={TOKEN} && git push", TOKEN, id="env-var"),
    pytest.param(
        f"request failed: Authorization: Bearer {TOKEN}", TOKEN, id="bearer-header"
    ),
    pytest.param(
        f"could not connect to https://user:{TOKEN}@db.example.com/x",
        TOKEN,
        id="url-password",
    ),
]

MODULES = [
    pytest.param(evaluation_results, id="evaluation_results"),
    pytest.param(rollout_results, id="rollout._results"),
]


@pytest.mark.parametrize("module", MODULES)
def test_the_module_masks_with_the_trajectory_redactor(module):
    assert module.redact_trajectory_text is trajectory_types.redact_trajectory_text


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize("text,secret", ERRORS)
def test_masked_masks_a_secret_in_error_text(module, text, secret):
    masked = module._masked(text)

    assert secret not in masked
    assert MASK in masked


@pytest.mark.parametrize("module", MODULES)
def test_masked_keeps_none_and_plain_text(module):
    assert module._masked(None) is None
    assert module._masked("verifier timed out after 600s") == (
        "verifier timed out after 600s"
    )


def test_the_no_folder_result_masks_its_error_text(tmp_path):
    """The payload ``Evaluation._record_trial_without_folder`` writes as the
    result.json of a trial that failed before its folder existed."""
    text, secret = ERRORS[0].values
    result = RolloutResult(
        task_name="typo",
        agent="claude-agent-acp",
        error=text,
        verifier_error=f"verifier crashed: {text}",
        export_error=f"export failed: {text}",
    )

    payload = evaluation_results.rollout_result_payload(
        result, source_provenance=None, tasks_dir=tmp_path, task_name="typo"
    )

    for field in ("error", "verifier_error", "export_error"):
        assert secret not in payload[field], field
        assert MASK in payload[field], field
    # The run keeps the text it classified.
    assert result.error == text


def test_a_hosted_run_masks_with_the_same_function():
    from benchflow import hosted_env

    assert hosted_env._masked is evaluation_results._masked
