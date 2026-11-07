"""Guards PY8 taskmd/rewards sweep fixes against base commit 363d815b."""

import pytest

from benchflow.rewards.llm import parse_verdict


@pytest.mark.parametrize(
    "reason",
    [
        "Handles } correctly",
        "Handles { correctly",
        'An escaped \\" } brace',
        "Nested { braces } and \\ backslashes",
    ],
)
@pytest.mark.parametrize(
    "wrapper",
    ["{}", "```json\n{}\n```", "Response: {} Done.", "Ignore {{broken prefix. {}"],
)
def test_json_strings_do_not_change_verdict_boundaries(reason, wrapper):
    """Decode JSON strings, nesting and escapes with the JSON grammar."""
    import json

    verdict = {"verdict": "pass", "reasoning": reason, "extra": {"items": [1, 2]}}
    assert parse_verdict(wrapper.format(json.dumps(verdict))) == verdict
