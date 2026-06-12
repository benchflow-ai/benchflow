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

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow._utils.scoring import (
    VERIFIER_DEP_INSTALL,
    VERIFIER_DEP_INSTALL_MARKERS,
    VERIFIER_FAILED,
    classify_verifier_error,
    contains_verifier_dep_install_marker,
)
from benchflow.task.paths import RolloutPaths
from benchflow.task.verifier import (
    _DEP_INSTALL_DIAGNOSTIC,
    RewardFileNotFoundError,
    Verifier,
    _has_dep_install_failure,
)

# Classifier recognises the fixed diagnostic (PR #540 / #572)


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
    assert contains_verifier_dep_install_marker(_DEP_INSTALL_DIAGNOSTIC)
    assert "test-stdout.txt" in _DEP_INSTALL_DIAGNOSTIC


def test_canonical_markers_drive_classifier_and_stdout_scan(tmp_path):
    """Guards PR #572 against marker drift between verifier stdout scanning and
    verifier-error classification."""
    p = tmp_path / "test-stdout.txt"
    for marker in VERIFIER_DEP_INSTALL_MARKERS:
        assert (
            classify_verifier_error(f"verifier crashed: {marker}")
            == VERIFIER_DEP_INSTALL
        )

        p.write_text(f"running setup...\n{marker.upper()}\n")
        assert _has_dep_install_failure(p) is True


# _has_dep_install_failure: boolean verdict only, never returns stdout


class TestHasDepInstallFailure:
    @pytest.mark.parametrize(
        "stdout",
        [
            "Installing...\nx No solution found when resolving torch==2.1.2+cpu\n",
            "Could not find a version that satisfies the requirement foo==9.9\n",
            "ERROR: dependency install failed\n",
            "resolution impossible for package bar\n",
            (
                "  x Failed to download `azure-identity==1.25.3`\n"
                "  +- Request failed after 3 retries\n"
                "  +- Failed to fetch:\n"
                "  |  `https://files.pythonhosted.org/packages/example.whl`\n"
                "  +- error sending request for url\n"
                "  +- dns error\n"
                "  `- failed to lookup address information: Try again\n"
            ),
            "NO SOLUTION FOUND\n",  # case-insensitive
        ],
    )
    def test_detects_marker(self, tmp_path, stdout):
        """Guards PR #1's no-network uv resolver diagnostic."""
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

    def test_early_marker_with_long_trailing_output(self, tmp_path):
        """PR #572 / issue #540: dep-install runs at the START of test.sh, so a
        resolver-failure marker on an EARLY line can be followed by far more than
        the old 30-line tail window. Scanning the whole file must still detect
        it (a tail-only scan silently misclassified it as a generic failure)."""
        p = tmp_path / "test-stdout.txt"
        early = "Resolving dependencies...\nx No solution found resolving torch\n"
        trailing = "".join(f"falling back / cleanup line {i}\n" for i in range(500))
        p.write_text(early + trailing)
        assert _has_dep_install_failure(p) is True

    def test_marker_spanning_chunk_boundary(self, tmp_path):
        """PR #572: a marker straddling a fixed-chunk read boundary must still be
        caught (the chunked scan carries a small overlap across reads)."""
        from benchflow.task.verifier import _SCAN_CHUNK_BYTES

        p = tmp_path / "test-stdout.txt"
        marker = "no solution found"
        # Place the marker so it begins a few bytes before a chunk boundary and
        # ends after it, then bury it under lots of trailing output.
        prefix = "a" * (_SCAN_CHUNK_BYTES - 5)
        p.write_text(prefix + marker + "\n" + "trailing\n" * 1000)
        assert _has_dep_install_failure(p) is True


# Production wiring: _verify_test_script raises a redacted, classifiable error
# (PR #572 finding 2 — exercise the real verifier, not synthesized strings)


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


@pytest.mark.asyncio
async def test_verify_test_script_recovers_reward_when_dir_download_fails(tmp_path):
    """Guards private PR #1's paratransit Daytona verifier export regression."""
    rollout_paths = RolloutPaths(tmp_path / "rollout")
    rollout_paths.mkdir()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_path = tests_dir / "test.sh"
    test_path.write_text("#!/bin/sh\nexit 0\n")

    task = MagicMock()
    task.config.verifier.type = "test-script"
    task.config.verifier.service = "main"
    task.config.verifier.user = "root"
    task.config.verifier.env = None
    task.config.verifier.timeout_sec = 60
    task.paths.tests_dir = tests_dir
    task.paths.test_path = test_path

    sandbox = MagicMock()
    sandbox.is_mounted = False
    sandbox.upload_dir = AsyncMock()
    sandbox.exec = AsyncMock(return_value=MagicMock(exit_code=0, returncode=0))
    sandbox.download_dir = AsyncMock(side_effect=RuntimeError("provider export failed"))

    async def download_file(source: str, target: str | Path) -> None:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        if source.endswith("/reward.txt"):
            path.write_text("0")
        elif source.endswith("/test-stdout.txt"):
            path.write_text("pytest output")
        elif source.endswith("/ctrf.json"):
            path.write_text("{}")
        else:
            raise FileNotFoundError(source)

    sandbox.download_file = AsyncMock(side_effect=download_file)
    verifier = Verifier(task=task, rollout_paths=rollout_paths, sandbox=sandbox)

    result = await verifier._verify_test_script()

    assert result.rewards == {"reward": 0.0}
    sandbox.download_dir.assert_awaited_once()
    assert rollout_paths.reward_text_path.read_text() == "0"
    assert rollout_paths.test_stdout_path.read_text() == "pytest output"


# Legacy verifier-mount compat (family-2 / opaquetoolsbench parity finding)
#
# ``bench tasks migrate`` mounts a task.md verifier at /verifier instead of the
# legacy /tests. Verifier scripts carried over from a legacy benchmark often
# hardcode ``python3 /tests/evaluate.py``; without a /tests -> /verifier alias
# every converted verifier crashes rc=2 "no reward file" while the legacy
# variant passes. These guard the alias that keeps such conversions working.


def _make_script_verifier(tmp_path, *, uses_native_verifier_dir):
    """Verifier over a mounted sandbox driving the default test.sh path, with a
    captured exec log. test.sh writes a passing reward so verify() completes."""
    verifier_dir = tmp_path / "verifier"
    verifier_dir.mkdir(parents=True, exist_ok=True)

    rollout_paths = MagicMock()
    rollout_paths.verifier_dir = verifier_dir
    rollout_paths.test_stdout_path = verifier_dir / "test-stdout.txt"
    rollout_paths.reward_text_path = verifier_dir / "reward.txt"
    rollout_paths.reward_json_path = verifier_dir / "reward.json"

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(exist_ok=True)
    (tests_dir / "test.sh").write_text("#!/bin/sh\npython3 /tests/evaluate.py\n")

    task = MagicMock()
    task.config.verifier.type = "test-script"
    task.config.verifier.service = "main"
    task.config.verifier.user = "root"
    task.config.verifier.env = None
    task.config.verifier.timeout_sec = 60
    task.paths.tests_dir = tests_dir
    task.paths.test_path = tests_dir / "test.sh"
    task.paths.uses_native_verifier_dir = uses_native_verifier_dir

    calls: list[dict] = []

    async def fake_exec(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("command", "")
        calls.append({"command": cmd, "user": kwargs.get("user")})
        if "ln -s" in cmd or "chmod" in cmd or cmd.startswith("mkdir") or "[ -e" in cmd:
            return MagicMock(exit_code=0, returncode=0)
        # The test.sh run: a passing reward + stdout, mounted in place.
        rollout_paths.test_stdout_path.write_text("ok")
        rollout_paths.reward_text_path.write_text("1")
        return MagicMock(exit_code=0, returncode=0)

    sandbox = MagicMock()
    sandbox.is_mounted = True
    sandbox.upload_dir = AsyncMock()
    sandbox.exec = AsyncMock(side_effect=fake_exec)
    verifier = Verifier(task=task, rollout_paths=rollout_paths, sandbox=sandbox)
    return verifier, calls


@pytest.mark.asyncio
async def test_native_verifier_aliases_legacy_tests_mount(tmp_path):
    """A native (task.md) verifier mounts at /verifier and must alias /tests so
    legacy-convention verifier scripts (hardcoded /tests/...) still resolve."""
    verifier, calls = _make_script_verifier(tmp_path, uses_native_verifier_dir=True)

    result = await verifier._verify_test_script()

    assert result.rewards == {"reward": 1.0}
    link = [c for c in calls if "ln -s" in c["command"]]
    assert link, (
        f"expected a /tests->/verifier alias; got {[c['command'] for c in calls]}"
    )
    cmd = link[0]["command"]
    assert "ln -s /verifier /tests" in cmd
    # Guarded so real /tests content is never clobbered.
    assert "[ -e /tests ]" in cmd
    # Created as root so the verifier user (any) can traverse it.
    assert link[0]["user"] == "root"


@pytest.mark.asyncio
async def test_legacy_verifier_does_not_alias_tests_mount(tmp_path):
    """A legacy verifier already mounts at /tests, so no alias is created."""
    verifier, calls = _make_script_verifier(tmp_path, uses_native_verifier_dir=False)

    result = await verifier._verify_test_script()

    assert result.rewards == {"reward": 1.0}
    assert not [c for c in calls if "ln -s" in c["command"]]
