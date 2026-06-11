"""Non-live coverage for the live-smoke skip wiring.

These tests guard the false-green hazard: the live smoke ``skipif``s (exit 0)
when Docker is down or the chosen model has no credential. They pin down the
pure helpers so a skip is always specific and attributable to a named missing
credential, and they assert the JUnit-summary gate used by ``docs/release.md``
/ launch-prep would treat such a skip as RED, not green.

They never touch Docker or any model — ``_smoke_skip_reason`` (which shells out
to ``docker version``) is only exercised through ``monkeypatch`` of its docker
probe, so the suite stays in the ``not live`` default selection.
"""

from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from tests import test_smoke as smoke

ALL_SMOKE_CRED_VARS = [
    "ANTHROPIC_API_KEY",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    smoke.SMOKE_AGENT_ENV,
    smoke.SMOKE_MODEL_ENV,
]


@pytest.fixture
def clean_env(monkeypatch):
    """Start from a known-empty credential environment.

    The autouse ``isolate_local_dotenv`` fixture only redirects the dotenv path;
    real ``ANTHROPIC_API_KEY`` etc. on the developer machine still leak through
    ``os.environ``. Clearing them makes the missing-credential reasons
    deterministic regardless of who runs the suite.
    """
    for var in ALL_SMOKE_CRED_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_resolve_smoke_target_defaults_to_claude(clean_env):
    assert smoke.resolve_smoke_target() == (
        smoke.DEFAULT_SMOKE_AGENT,
        smoke.DEFAULT_SMOKE_MODEL,
    )


def test_resolve_smoke_target_env_overrides_both(clean_env):
    clean_env.setenv(smoke.SMOKE_AGENT_ENV, "openhands")
    clean_env.setenv(smoke.SMOKE_MODEL_ENV, "deepseek/deepseek-chat")
    assert smoke.resolve_smoke_target() == ("openhands", "deepseek/deepseek-chat")


@pytest.mark.parametrize("present", [smoke.SMOKE_AGENT_ENV, smoke.SMOKE_MODEL_ENV])
def test_resolve_smoke_target_half_set_is_an_error(clean_env, present):
    # Setting only one of the pair must NOT silently fall back to the Anthropic
    # default the contributor cannot authenticate — that would re-introduce a
    # false-green via an unauthenticatable default.
    clean_env.setenv(present, "x")
    with pytest.raises(RuntimeError, match="must be set together"):
        smoke.resolve_smoke_target()


def test_missing_credentials_anthropic_default_no_creds(clean_env):
    reason = smoke._missing_model_credentials(smoke.DEFAULT_SMOKE_MODEL)
    assert reason == "no ANTHROPIC_API_KEY and no ~/.claude/.credentials.json"


def test_missing_credentials_anthropic_satisfied_by_api_key(clean_env):
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert smoke._missing_model_credentials(smoke.DEFAULT_SMOKE_MODEL) is None


def test_missing_credentials_anthropic_satisfied_by_login(clean_env, monkeypatch):
    real_is_file = Path.is_file

    def fake_is_file(self):
        if self == Path("~/.claude/.credentials.json").expanduser():
            return True
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", fake_is_file)
    assert smoke._missing_model_credentials(smoke.DEFAULT_SMOKE_MODEL) is None


def test_missing_credentials_deepseek_names_each_missing_var(clean_env):
    reason = smoke._missing_model_credentials("deepseek/deepseek-chat")
    assert reason is not None
    # Both the API key and the provider's url_params var must be named so the
    # skip message is actionable, not just "missing credential".
    assert "DEEPSEEK_API_KEY" in reason
    assert "DEEPSEEK_BASE_URL" in reason
    assert "deepseek/deepseek-chat" in reason


def test_missing_credentials_deepseek_satisfied(clean_env):
    clean_env.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test")
    clean_env.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    assert smoke._missing_model_credentials("deepseek/deepseek-chat") is None


def test_missing_credentials_deepseek_partial_still_skips(clean_env):
    # API key alone is not enough — base_url is also required by the provider.
    clean_env.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test")
    reason = smoke._missing_model_credentials("deepseek/deepseek-chat")
    assert reason is not None
    assert "DEEPSEEK_BASE_URL" in reason
    assert "DEEPSEEK_API_KEY" not in reason


def _patch_docker_ok(monkeypatch):
    """Make the docker probe in _smoke_skip_reason report a reachable daemon."""
    monkeypatch.setattr(smoke.shutil, "which", lambda _name: "/usr/bin/docker")

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=b"28.0")

    monkeypatch.setattr(smoke.subprocess, "run", fake_run)


def test_skip_reason_credential_gate_after_docker_ok(clean_env):
    # Docker present and reachable, but no credential for the default model:
    # the skip reason must be the credential one, not a docker one.
    _patch_docker_ok(clean_env)
    assert (
        smoke._smoke_skip_reason()
        == "no ANTHROPIC_API_KEY and no ~/.claude/.credentials.json"
    )


def test_skip_reason_none_when_docker_and_creds_present(clean_env):
    _patch_docker_ok(clean_env)
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert smoke._smoke_skip_reason() is None


def test_skip_reason_uses_escape_hatch_model(clean_env):
    # With the escape hatch pointing at deepseek and its creds present, the
    # smoke is runnable even with zero Anthropic credentials on the box.
    _patch_docker_ok(clean_env)
    clean_env.setenv(smoke.SMOKE_AGENT_ENV, "openhands")
    clean_env.setenv(smoke.SMOKE_MODEL_ENV, "deepseek/deepseek-chat")
    clean_env.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test")
    clean_env.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    assert smoke._smoke_skip_reason() is None


def test_skip_reason_docker_missing_short_circuits(clean_env):
    clean_env.setattr(smoke.shutil, "which", lambda _name: None)
    assert smoke._smoke_skip_reason() == "docker CLI not installed"


# --- The gate's own skip detection (mirrors the launch-prep Step 3 predicate) -


def _junit_xml(tests: int, skipped: int) -> str:
    """Build JUnit XML the way pytest does: a <testsuites> wrapper whose nested
    <testsuite> carries the tests/skipped counts. A predicate that reads the
    counts off the <testsuites> root instead of the child would see them as 0
    and false-RED a passing run — these fixtures pin that down."""
    return (
        '<testsuites name="pytest tests">'
        f'<testsuite name="pytest" tests="{tests}" skipped="{skipped}"'
        ' errors="0" failures="0"></testsuite>'
        "</testsuites>"
    )


def _gate_is_green(junit_xml: str) -> bool:
    """Replicate the launch-prep gate predicate: green iff a test ran clean.

    Sums tests/skipped over every <testsuite> element so it reads the counts at
    the level pytest actually writes them (not the <testsuites> root)."""
    suites = list(ET.fromstring(junit_xml).iter("testsuite"))
    tests = sum(int(e.get("tests", 0)) for e in suites)
    skipped = sum(int(e.get("skipped", 0)) for e in suites)
    return bool(tests) and not skipped


def test_gate_detects_skipped_live_smoke_as_red():
    assert _gate_is_green(_junit_xml(tests=1, skipped=1)) is False


def test_gate_detects_unrun_live_smoke_as_red():
    # Deselected / collected-nothing run reports zero tests — also not green.
    assert _gate_is_green(_junit_xml(tests=0, skipped=0)) is False


def test_gate_passes_executed_live_smoke():
    # The counts live on the nested <testsuite>; a root-only reader would
    # false-RED this passing run, so this case is the regression guard.
    assert _gate_is_green(_junit_xml(tests=1, skipped=0)) is True


def test_gate_predicate_matches_real_pytest_junit_shape():
    # Sanity: the shape we assert is the shape pytest emits — counts on the
    # child <testsuite>, not the <testsuites> root.
    root = ET.fromstring(_junit_xml(tests=1, skipped=0))
    assert root.tag == "testsuites"
    assert "tests" not in root.attrib
    (child,) = list(root.iter("testsuite"))
    assert child.get("tests") == "1"
