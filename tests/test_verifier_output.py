"""Tests for verifier dependency-install classification (PR #540 / #572).

PR #540 made dep-install failures classifiable: a missing reward file whose
``test-stdout.txt`` shows a resolver failure surfaces a diagnostic through
``RewardFileNotFoundError`` so ``classify_verifier_error`` returns
``VERIFIER_DEP_INSTALL``.

PR #572 review hardened this: the raw stdout is NEVER persisted into
``verifier_error`` (it can carry env dumps / tokens / credentialed install
URLs). Instead the verifier scans stdout for dep-install markers and, on a
hit, appends only a FIXED secret-free diagnostic. The raw resolver output
stays in the downloaded ``verifier/test-stdout.txt`` artifact.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow._utils.scoring import (
    VERIFIER_DEP_INSTALL,
    VERIFIER_FAILED,
    classify_verifier_error,
)
from benchflow.task.verifier import (
    _DEP_INSTALL_DIAGNOSTIC,
    RewardFileNotFoundError,
    Verifier,
    _has_dep_install_failure,
)

# ---------------------------------------------------------------------------
# Classifier recognises the fixed diagnostic (PR #540 / #572)
# ---------------------------------------------------------------------------


def test_fixed_diagnostic_classifies_as_dep_install():
    """PR #572: the fixed diagnostic the verifier appends must classify as
    dep_install once rollout wraps it with 'verifier crashed:'."""
    verifier_error = (
        f"verifier crashed: ...no reward file...\n{_DEP_INSTALL_DIAGNOSTIC}"
    )
    assert classify_verifier_error(verifier_error) == VERIFIER_DEP_INSTALL


def test_diagnostic_carries_no_stdout():
    """PR #572: the fixed diagnostic must not contain raw resolver output —
    only a marker the classifier needs and a pointer to the log artifact."""
    assert "dependency install failed" in _DEP_INSTALL_DIAGNOSTIC
    assert "test-stdout.txt" in _DEP_INSTALL_DIAGNOSTIC


# ---------------------------------------------------------------------------
# _has_dep_install_failure: boolean verdict only, never returns stdout
# ---------------------------------------------------------------------------


class TestHasDepInstallFailure:
    @pytest.mark.parametrize(
        "stdout",
        [
            "Installing...\nx No solution found when resolving torch==2.1.2+cpu\n",
            "Could not find a version that satisfies the requirement foo==9.9\n",
            "ERROR: dependency install failed\n",
            "resolution impossible for package bar\n",
            "NO SOLUTION FOUND\n",  # case-insensitive
        ],
    )
    def test_detects_marker(self, tmp_path, stdout):
        p = tmp_path / "test-stdout.txt"
        p.write_text(stdout)
        assert _has_dep_install_failure(p) is True

    def test_no_marker_is_false(self, tmp_path):
        p = tmp_path / "test-stdout.txt"
        p.write_text("running tests...\nassertion failed: 2 != 3\n")
        assert _has_dep_install_failure(p) is False

    def test_missing_file_is_false(self, tmp_path):
        assert _has_dep_install_failure(tmp_path / "nope.txt") is False

    def test_single_huge_line_is_bounded(self, tmp_path):
        """PR #572 finding 3: a single huge line must not blow up — we read a
        bounded window and still detect the marker if present."""
        p = tmp_path / "test-stdout.txt"
        p.write_text("x" * (256 * 1024) + " no solution found\n")
        assert _has_dep_install_failure(p) is True


# ---------------------------------------------------------------------------
# Production wiring: _verify_test_script raises a redacted, classifiable error
# (PR #572 finding 2 — exercise the real verifier, not synthesized strings)
# ---------------------------------------------------------------------------


def _make_verifier(tmp_path, *, stdout_text):
    """Build a Verifier over a fake mounted sandbox whose test.sh writes the
    given stdout and produces NO reward file."""
    verifier_dir = tmp_path / "verifier"
    verifier_dir.mkdir(parents=True, exist_ok=True)

    rollout_paths = MagicMock()
    rollout_paths.verifier_dir = verifier_dir
    rollout_paths.test_stdout_path = verifier_dir / "test-stdout.txt"
    rollout_paths.reward_text_path = verifier_dir / "reward.txt"
    rollout_paths.reward_json_path = verifier_dir / "reward.json"

    task = MagicMock()
    task.config.verifier.type = "test-script"
    task.config.verifier.service = "main"
    task.config.verifier.user = "root"
    task.config.verifier.env = None
    task.config.verifier.timeout_sec = 60
    task.paths.tests_dir = tmp_path / "tests"
    (tmp_path / "tests").mkdir(exist_ok=True)
    task.paths.test_path = tmp_path / "tests" / "test.sh"

    sandbox = MagicMock()
    sandbox.is_mounted = True
    sandbox.upload_dir = AsyncMock()

    def _cmd(args, kwargs):
        return args[0] if args else kwargs.get("command", "")

    async def fake_exec(*args, **kwargs):
        cmd = _cmd(args, kwargs)
        # Setup commands (chmod/mkdir) succeed; the test.sh run writes stdout,
        # produces no reward file, and exits nonzero.
        if "chmod" in cmd or cmd.startswith("mkdir"):
            return MagicMock(exit_code=0, returncode=0)
        rollout_paths.test_stdout_path.write_text(stdout_text)
        return MagicMock(exit_code=1, returncode=1)

    sandbox.exec = AsyncMock(side_effect=fake_exec)
    return Verifier(task=task, rollout_paths=rollout_paths, sandbox=sandbox)


@pytest.mark.asyncio
async def test_verify_test_script_surfaces_classifiable_dep_install(tmp_path):
    """PR #572 finding 2: the REAL verifier path must raise an error that, once
    wrapped as 'verifier crashed: {e}', classifies as VERIFIER_DEP_INSTALL."""
    verifier = _make_verifier(
        tmp_path,
        stdout_text=(
            "GEMINI_API_KEY=AIzaSyFAKEKEYFORTESTSONLYxxxxxxxxxxxxxxx\n"
            "PIP_INDEX_URL=https://user:p4ssw0rd@pypi.example/simple\n"
            "x No solution found when resolving torch==2.1.2+cpu\n"
        ),
    )
    with pytest.raises(RewardFileNotFoundError) as exc:
        await verifier._verify_test_script()

    msg = str(exc.value)
    wrapped = f"verifier crashed: {msg}"  # how rollout.py builds verifier_error
    assert classify_verifier_error(wrapped) == VERIFIER_DEP_INSTALL
    # Finding 1: no raw stdout / secrets leaked into the surfaced error.
    assert "AIzaSyFAKEKEYFORTESTSONLYxxxxxxxxxxxxxxx" not in msg
    assert "p4ssw0rd" not in msg
    assert "PIP_INDEX_URL" not in msg


@pytest.mark.asyncio
async def test_verify_test_script_non_dep_failure_is_not_dep_install(tmp_path):
    """A missing reward file with no dep-install marker must NOT be misclassified
    as dep_install (stays a generic verifier failure)."""
    verifier = _make_verifier(
        tmp_path,
        stdout_text="running pytest...\n3 failed, 0 passed\nassertion error\n",
    )
    with pytest.raises(RewardFileNotFoundError) as exc:
        await verifier._verify_test_script()

    wrapped = f"verifier crashed: {exc.value}"
    assert classify_verifier_error(wrapped) == VERIFIER_FAILED
    assert _DEP_INSTALL_DIAGNOSTIC not in str(exc.value)
