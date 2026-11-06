"""Guards S1-CONF-10 at 694245fd: literal dollars in snapshot receipts."""

import json

import pytest


@pytest.mark.parametrize(
    "text", ["$X", "${X}", "$$", "a$X/b", 'space $X "quote" \\ newline\n非']
)
def test_checkpoint_receipt_can_be_written_without_docker_expansion(text):
    """CONF-10: the Dockerfile value cannot express a variable expansion."""
    from benchflow.sandbox import _bind_targets

    # Use the real snapshot writer below in the native commit/restore test;
    # this invariant enumerates the entire problematic text class cheaply.
    value = {"main": {"/data" + text: "1:2"}}
    encode = getattr(_bind_targets, "encode", lambda v: json.dumps(v))
    encoded = encode(value)
    assert "$" not in encoded
    assert _bind_targets.decode(encoded) == value


@pytest.mark.parametrize("value", ["base64:!", "base64:/w==", "base64:e30=", None, 1])
def test_invalid_checkpoint_receipt_refuses_reuse(value):
    """Guards CONF-10 at 694245fd: damaged encodings cannot become empty proof."""
    from benchflow.sandbox import _bind_targets

    with pytest.raises(ValueError, match="fresh checkpoint"):
        _bind_targets.decode(value)


def test_legacy_receipts_keep_their_literal_paths():
    """Guards CONF-10 at 694245fd: do not reinterpret older valid JSON labels."""
    from benchflow.sandbox import _bind_targets

    value = {"main": {"/data$X": "1:2"}}
    assert _bind_targets.decode(json.dumps(value)) == value
