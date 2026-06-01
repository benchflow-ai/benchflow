"""Tests for verifier dependency-install classification and stdout surfacing.

Covers the ENG-151 / PR #540 path where a missing reward file surfaces a
redacted tail of ``test-stdout.txt`` through ``RewardFileNotFoundError`` so
``classify_verifier_error`` can detect dependency-install failures, plus the
PR #572 review fix that redacts secrets from that untrusted subprocess output
before it lands in ``verifier_error`` / summaries / dashboards.
"""

import pytest

from benchflow._utils.scoring import (
    VERIFIER_DEP_INSTALL,
    VERIFIER_FAILED,
    classify_verifier_error,
)
from benchflow.task.verifier import _redact_secrets, _tail_file

# ---------------------------------------------------------------------------
# dep_install classification (PR #540) — markers reach the classifier only
# because verifier.py tails test-stdout.txt into the exception message.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_str,expected",
    [
        (
            "verifier crashed: verifier exited with rc=1; no reward file found\n"
            "--- test-stdout.txt (last 30 lines) ---\n"
            "x No solution found when resolving tool dependencies: torch==2.1.2+cpu",
            VERIFIER_DEP_INSTALL,
        ),
        (
            "verifier crashed: verifier exited with rc=1\n"
            "Could not find a version that satisfies the requirement foo==9.9.9",
            VERIFIER_DEP_INSTALL,
        ),
        (
            "verifier crashed: verifier exited with rc=1\nERROR: dependency install failed",
            VERIFIER_DEP_INSTALL,
        ),
        (
            "verifier crashed: verifier exited with rc=1\nresolution impossible for bar",
            VERIFIER_DEP_INSTALL,
        ),
    ],
)
def test_classify_dep_install_from_stdout_tail(input_str, expected):
    """PR #540: dep-install markers in the surfaced stdout tail classify."""
    assert classify_verifier_error(input_str) == expected


def test_dep_install_takes_precedence_over_generic_crash():
    """PR #540: dep_install wins over verifier_failure when both markers present."""
    msg = (
        "verifier crashed: verifier exited with rc=1; no reward file found\n"
        "--- test-stdout.txt (last 30 lines) ---\n"
        "x No solution found when resolving tool dependencies: torch==2.1.2+cpu"
    )
    assert classify_verifier_error(msg) == VERIFIER_DEP_INSTALL


# ---------------------------------------------------------------------------
# _tail_file: bounded streaming read + secret redaction (PR #572 review)
# ---------------------------------------------------------------------------


class TestTailFile:
    def test_returns_redacted_tail(self, tmp_path):
        """PR #540: tail surfaces the dep-install marker for the classifier."""
        stdout = tmp_path / "test-stdout.txt"
        stdout.write_text(
            "Installing dependencies...\n"
            "x No solution found when resolving tool dependencies: torch==2.1.2+cpu\n"
        )
        assert "no solution found" in _tail_file(stdout).lower()

    def test_missing_file_returns_empty(self, tmp_path):
        assert _tail_file(tmp_path / "nope.txt") == ""

    def test_only_keeps_last_n_lines(self, tmp_path):
        stdout = tmp_path / "test-stdout.txt"
        stdout.write_text("".join(f"line{i}\n" for i in range(100)))
        tail = _tail_file(stdout, n=5)
        assert tail.splitlines() == [f"line{i}" for i in range(95, 100)]

    def test_redacts_secrets_in_tail(self, tmp_path):
        """PR #572: untrusted stdout (env dumps, tokens) must be redacted.

        An obviously-fake key is used as the fixture (never a real one).
        """
        stdout = tmp_path / "test-stdout.txt"
        stdout.write_text(
            "GEMINI_API_KEY=AIzaSyFAKEKEYFORTESTSONLYxxxxxxxxxxxxxxx\n"
            "Authorization: Bearer fake-token-aaaaaaaaaaaaaaaa\n"
        )
        tail = _tail_file(stdout)
        assert "AIzaSyFAKEKEYFORTESTSONLYxxxxxxxxxxxxxxx" not in tail
        assert "fake-token-aaaaaaaaaaaaaaaa" not in tail
        assert "***REDACTED***" in tail


class TestRedactSecrets:
    """PR #572: redaction patterns mirror the trajectory set (#537).

    All fixtures use obviously-fake credential values, never real ones.
    """

    @pytest.mark.parametrize(
        "raw,leaked",
        [
            (
                "key=AIzaSyFAKEKEYFORTESTSONLYxxxxxxxxxxxxxxx",
                "FORTESTSONLYxxxxxxxxxxxxxxx",
            ),
            ("key=sk-ant-api03-FAKEFAKEFAKEdeadbeefcafe1234", "deadbeefcafe1234"),
            ("key=AKIAFAKEFAKEFAKE1234", "FAKEFAKE1234"),
            ("key=dtn_FAKEFAKEFAKEFAKEFAKE12345678", "FAKEFAKE12345678"),
            ("x-api-key: abc123secretvalue456", "abc123secretvalue456"),
        ],
    )
    def test_redacts_pattern(self, raw, leaked):
        out = _redact_secrets(raw)
        assert leaked not in out
        assert "***REDACTED***" in out

    @pytest.mark.parametrize(
        "raw",
        [
            "region=ASIAPACIFIC",  # English word, not an AWS key
            "queue=task-sk-us-east-1-foo-bar",  # slug containing sk-
            "label=dtn_v2_0",  # short identifier
            "just normal verifier output, nothing secret here",
        ],
    )
    def test_preserves_non_secret(self, raw):
        assert _redact_secrets(raw) == raw

    def test_classifier_still_fires_after_redaction(self, tmp_path):
        """Redaction must not eat the dep-install marker the classifier needs."""
        stdout = tmp_path / "test-stdout.txt"
        stdout.write_text(
            "GEMINI_API_KEY=AIzaSyFAKEKEYFORTESTSONLYxxxxxxxxxxxxxxx\n"
            "x No solution found when resolving tool dependencies: torch==2.1.2+cpu\n"
        )
        verifier_error = (
            "verifier crashed: verifier exited with rc=1\n"
            f"--- test-stdout.txt (last 30 lines) ---\n{_tail_file(stdout)}"
        )
        assert classify_verifier_error(verifier_error) == VERIFIER_DEP_INSTALL
        assert "AIzaSyFAKEKEYFORTESTSONLYxxxxxxxxxxxxxxx" not in verifier_error


def test_no_dep_markers_stays_verifier_failure(tmp_path):
    """PR #540: without dep-install markers the classifier returns verifier_failure."""
    stdout = tmp_path / "test-stdout.txt"
    stdout.write_text("some random test output\nassertion failed\n")
    verifier_error = (
        "verifier crashed: verifier exited with rc=1\n"
        f"--- test-stdout.txt (last 30 lines) ---\n{_tail_file(stdout)}"
    )
    assert classify_verifier_error(verifier_error) == VERIFIER_FAILED
