"""Guards Devin review on validate_reward_map metadata passthrough."""

from benchflow.rewards.validation import validate_reward_map


def test_validate_reward_map_preserves_non_numeric_metadata_fields() -> None:
    """Non-score metadata must not fail validation when reward is valid."""
    parsed = validate_reward_map(
        {
            "reward": 1.0,
            "explanation": "passed all checks",
            "details": {"mode": "oracle"},
        }
    )

    assert parsed["reward"] == 1.0
    assert parsed["explanation"] == "passed all checks"
    assert parsed["details"] == {"mode": "oracle"}
