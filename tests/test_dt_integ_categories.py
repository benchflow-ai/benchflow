"""Guard DT3-CU2 at integ 7b4907c8: every exported category is readable.

Only tests import the engine catalog. The pinned reader keeps its independent
allowlist, and an unknown category must still fail rather than become "other".
"""

import json

import pytest

from benchflow._utils.scoring import ERROR_CATEGORIES, VERIFIER_ERROR_CATEGORIES
from tools.dt import normalise

ERROR_FAMILIES = {
    "unsupported_task_feature": "refusal",
    "kept_service_freezer": "refusal",
    "runtime_sealing": "refusal",
    "guard_timeout": "sandbox",
    "invalid_provider_response": "provider",
    "sandbox_setup": "sandbox",
    "install_failure": "agent_install",
    "pipe_closed": "agent_error",
    "acp_error": "agent_error",
    "idle_timeout": "agent_idle",
    "timeout": "agent_timeout",
    "provider_auth": "provider",
    "provider_rate_limit": "provider",
    "provider_rejected": "provider",
    "api_error": "provider",
    "suspected_api_error": "provider",
    "agent_integration": "integration",
    "usage_limit": "usage_limit",
    "infra_failure": "infra",
    "image_build": "sandbox",
    "registry_auth": "sandbox",
    "task_config": "other",
    "disk_full": "other",
    "trial_budget": "agent_budget",
    "other": "other",
}
VERIFIER_FAMILIES = {
    "verifier_failure": "verifier_error",
    "verifier_other": "verifier_error",
    "verifier_dep_install": "verifier_error",
    "verifier_infra": "infra",
    "verifier_timeout": "verifier_timeout",
    "reward_out_of_range": "verifier_error",
    "disk_full": "verifier_error",
    "infra_failure": "infra",
}
CASES = [
    (channel, category, family)
    for channel, families in (
        ("error", ERROR_FAMILIES),
        ("verifier_error", VERIFIER_FAMILIES),
    )
    for category, family in families.items()
]


def test_pinned_allowlist_covers_the_complete_integ_catalog():
    """Guards DT3-CU2, including V4's range and P4's fallback categories."""
    assert set(ERROR_FAMILIES) == ERROR_CATEGORIES
    assert set(VERIFIER_FAMILIES) == VERIFIER_ERROR_CATEGORIES
    assert normalise.load_categories()["python"]["error_category"] == ERROR_FAMILIES
    assert (
        normalise.load_categories()["python"]["verifier_error_category"]
        == VERIFIER_FAMILIES
    )


@pytest.mark.parametrize("channel,category,family", CASES)
def test_each_category_reads_in_trials_and_job_summaries(
    tmp_path, channel, category, family
):
    """Guards DT3-CU2 on both persisted category channels and read paths."""
    trial = tmp_path / "trial"
    trial.mkdir()
    data = {
        "task_name": "category-control",
        "agent": "nop",
        channel: "DummyFailure: category control",
        f"{channel}_category": category,
        "rewards": None,
    }
    (trial / "result.json").write_text(json.dumps(data))
    (tmp_path / "summary.json").write_text(
        json.dumps({"total": 1, f"{channel}_categories": {category: 1}})
    )
    record = normalise.normalise_trial(trial)
    assert record["reward"] is None
    assert record["outcome"] == ("errored" if channel == "error" else "unscored")
    assert record["family"] == family
    assert record[f"{channel}_category"] == category
    summary = normalise.summary_counts(tmp_path)
    assert summary["families"] == {family: 1}
    assert summary["raw"][f"{channel}_categories"] == {category: 1}
    assert normalise.normalise(tmp_path)[0]["summary"] == summary


@pytest.mark.parametrize("channel", ["error", "verifier_error"])
@pytest.mark.parametrize("path", ["trial", "summary"])
@pytest.mark.parametrize(
    "category", ["new_category", "REWARD_OUT_OF_RANGE", "reward-out-of-range"]
)
def test_unknown_or_misspelled_categories_still_fail(tmp_path, channel, path, category):
    """Guards DT3-CU2 without weakening the unknown-category boundary."""
    if path == "trial":
        (tmp_path / "result.json").write_text(
            json.dumps({channel: "DummyFailure", f"{channel}_category": category})
        )
        read = normalise.normalise_trial
    else:
        (tmp_path / "summary.json").write_text(
            json.dumps({f"{channel}_categories": {category: 1}})
        )
        read = normalise.summary_counts
    with pytest.raises(normalise.NormaliseError, match="place it in a family first"):
        read(tmp_path)
