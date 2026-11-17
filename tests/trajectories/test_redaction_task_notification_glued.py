"""A key next to the word ``task-notification`` keeps its mask in every view.

A view's key mask that starts at the ``sk-`` of ``task-notification`` stops
at the word, as 0.9.2's mask of the text does. That cut applies only
to the word itself (review R307B on #307):

- ``subtask-notification`` is another name, which the sk- rule masks as a key
  in the text and in every view, the escape the view reads included;
- a key glued straight after the word (``task-notificationsk-proj-...``, the
  ``s`` either the word's or the key's) is masked as it is anywhere else.

Every value here is made up. The keys are built from split literals so no
whole key shape sits in the source.
"""

from __future__ import annotations

import pytest

from tests.trajectories.test_redaction_task_notification import _layers

BODY_42 = "FX307Cmadeup" + "0123456789" + "abcdefghij" + "KLMNOPqrst"
BODY_8 = "Qz8Wm" + "3Rt"
TAIL = "Yp1Xk4Vn" + "7Jb2Hd5L" + "c9Fg0SaW"
PREFIXES = {"sk-proj": "sk-" + "proj-", "sk-ant": "sk-" + "ant-" + "api03-"}
# The letter A, escaped as a view decodes it.
ESCAPES = {
    "json": "\\u0041",
    "json written": "\\\\u0041",
    "entity": "&#65;",
    "entity escaped": "&amp;#65;",
}


def _shown(value: str, body: str) -> list[str]:
    """The parts of the made-up key that survive in *value*."""
    parts = {
        "body": body[-8:],
        "after the escape": TAIL[:12],
        "end": TAIL[-12:],
    }
    return [name for name, part in parts.items() if part in value]


@pytest.mark.parametrize("escape", list(ESCAPES))
@pytest.mark.parametrize("prefix", list(PREFIXES))
@pytest.mark.parametrize("word", ["subtask-notification", "subtask-notifications"])
def test_a_key_after_subtask_notification_is_masked_in_every_layer(
    tmp_path, word, prefix, escape
):
    text = f"see {word}{PREFIXES[prefix]}{BODY_42}{ESCAPES[escape]}{TAIL} here"
    for layer, value in _layers(text, tmp_path).items():
        assert not _shown(value, BODY_42), (layer, value)


@pytest.mark.parametrize("escape", list(ESCAPES))
@pytest.mark.parametrize("prefix", list(PREFIXES))
@pytest.mark.parametrize("body", [BODY_8, BODY_42], ids=["8", "42"])
@pytest.mark.parametrize("word", ["task-notification", "task-notifications"])
def test_a_key_glued_after_the_word_keeps_its_mask_in_every_layer(
    tmp_path, word, body, prefix, escape
):
    text = f"see {word}{PREFIXES[prefix]}{body}{ESCAPES[escape]}{TAIL} here"
    for layer, value in _layers(text, tmp_path).items():
        assert not _shown(value, body), (layer, value)
