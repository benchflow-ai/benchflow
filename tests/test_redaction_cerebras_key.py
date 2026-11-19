"""/ R207: a bare Cerebras key (`csk-...`) is masked wherever it stands.

On 0.10 the generic `sk-` rule's left boundary skips a key whose `c` is part of
it, so a Cerebras key showed whole in a provider's error echo, after
`--api-key` and as a positional argument. It is a provider family of its own,
beside gsk_ (Groq), xai-, r8_, hf_ and fw_: `csk-` and twenty or more letters
or digits. Made-up keys only.
"""

from __future__ import annotations

import json

import pytest

from benchflow.publish.redact import redact_value
from benchflow.trajectories.types import (
    redact_acp_trajectory_jsonl,
    redact_trajectory_obj,
    redact_trajectory_text,
)

KEY = "csk-FAKEcerebras0123456789abcdefghijklmn"


def _six_layers(text: str) -> dict[str, str]:
    from benchflow.rollout._results import _masked

    return {
        "text": redact_trajectory_text(text),
        "_masked": _masked(text),
        "obj": json.dumps(redact_trajectory_obj({"output": text})),
        "acp": redact_acp_trajectory_jsonl(
            [{"type": "tool_call_update", "content": [{"type": "text", "text": text}]}]
        ),
        "record text": redact_trajectory_text(json.dumps({"output": text})),
        "publish": json.dumps(redact_value({"output": text})[0]),
    }


@pytest.mark.parametrize(
    "text",
    [
        f"cerebras.AuthenticationError: Error code: 401 - {{'message': 'Wrong API Key: {KEY}'}}",
        f"openai.AuthenticationError: Incorrect API key provided: {KEY}.",
        f"cerebras-cli chat --api-key {KEY} --model llama-4",
        f"cerebras-cli chat --api-key={KEY}",
        f"python3 run.py {KEY} 'hello'",
        f"$ ./bench run {KEY}",
        f'client = make_client("{KEY}")',
        f"$ echo $CEREBRAS_API_KEY\n{KEY}",
        KEY,
        f'{{"key": "{KEY}"}}',
    ],
)
def test_a_cerebras_key_is_masked_in_every_layer(text):
    shown = [layer for layer, out in _six_layers(text).items() if "FAKEcerebras" in out]
    assert shown == []


def test_a_short_csk_word_is_kept():
    """Twenty letters or digits at least, as for the other families."""
    for text in ("csk-1234", "the csk-2 cluster", "task-notification"):
        assert redact_trajectory_text(text) == text
