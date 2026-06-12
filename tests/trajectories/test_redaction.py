"""Tests for trajectory secret redaction patterns.

Guards PR #585 (extends #537): redaction must cover raw-text header/kv forms
(agents print env, curl -v, request logs into trajectories), and audit-sensitive
token families (Google/Daytona/AWS) must be redacted whole — prefix included —
so the v0.5 secret-leak audit grep sees no live-key shape.
"""

import pytest

from benchflow.trajectories.types import (
    LLMExchange,
    LLMRequest,
    LLMResponse,
    Trajectory,
    redact_acp_trajectory_jsonl,
    redact_trajectory_text,
)

# Fake token fixtures assembled from split literals so the full token string
# never appears verbatim in source (GitHub secret-scanning push protection
# scans raw text). The runtime value still matches the redaction patterns.
_FAKE_GHP = "ghp" + "_" + "16C7e42F292c6912E7710c838347Ae178B4abcde"
_FAKE_GH_PAT = "github" + "_pat_" + "11ABCDE0Y0abcdefghij_klmnopqrstuvwxyz0123456789"
_FAKE_XOXB = "xox" + "b-" + "1234567890-0987654321-abcDEFghiJKLmnoPQR"


def _jsonl_for_request_body(body: dict) -> str:
    traj = Trajectory(
        session_id="t",
        exchanges=[LLMExchange(request=LLMRequest(body=body), response=LLMResponse())],
    )
    return traj.to_jsonl(redact_keys=True)


@pytest.mark.parametrize(
    "label,raw,must_not_contain",
    [
        pytest.param(
            "anthropic sk-ant-",
            '{"key": "sk-ant-api03-abc123XYZ_defghijklmnopqrstuvwxyz0123456789"}',
            "defghijklmnopqrstuvwxyz0123456789",
            id="anthropic",
        ),
        pytest.param(
            "openai sk-proj-",
            '{"key": "sk-proj-abc123XYZ_defghijklmnopqrstuvwxyz0123456789"}',
            "defghijklmnopqrstuvwxyz0123456789",
            id="openai-proj",
        ),
        pytest.param(
            "openai sk-",
            '{"key": "sk-abc1234567defghijklmnopqrstuvwxyz"}',
            "defghijklmnopqrstuvwxyz",
            id="openai-generic",
        ),
        pytest.param(
            "google AIzaSy",
            '{"key": "AIzaSyFAKEKEYFORTESTSONLYxxxxxxxxxxxxxxx"}',
            "FORTESTSONLYxxxxxxxxxxxxxxx",
            id="google",
        ),
        pytest.param(
            "aws AKIA",
            '{"key": "AKIAIOSFODNN7EXAMPLE"}',
            "IOSFODNN7EXAMPLE",
            id="aws-akia",
        ),
        pytest.param(
            "aws ASIA (STS)",
            '{"key": "ASIAQWERTYUIOPASDFGH"}',
            "QWERTYUIOPASDFGH",
            id="aws-asia",
        ),
        pytest.param(
            "daytona dtn_",
            '{"key": "dtn_abcdefghijklmnop1234567890"}',
            "ghijklmnop1234567890",
            id="daytona",
        ),
        pytest.param(
            "bearer header",
            '{"authorization": "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.secret"}',
            "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.secret",
            id="bearer",
        ),
        pytest.param(
            "x-api-key header",
            '{"x-api-key": "AIzaSyFAKEKEYFORTESTSONLYxxxxxxxxxxxxxxx"}',
            "AIzaSyFAKEKEYFORTESTSONLYxxxxxxxxxxxxxxx",
            id="x-api-key",
        ),
        pytest.param(
            "api-key header (Azure)",
            '{"api-key": "abc123secret456value"}',
            "abc123secret456value",
            id="api-key-azure",
        ),
        pytest.param(
            "github classic PAT ghp_",
            '{"key": "' + _FAKE_GHP + '"}',
            "16C7e42F292c6912E7710c838347Ae178B4abcde",
            id="github-ghp",
        ),
        pytest.param(
            "github fine-grained PAT",
            '{"key": "' + _FAKE_GH_PAT + '"}',
            "11ABCDE0Y0abcdefghij_klmnopqrstuvwxyz0123456789",
            id="github-pat",
        ),
        pytest.param(
            "slack bot token xoxb-",
            '{"key": "' + _FAKE_XOXB + '"}',
            "1234567890-0987654321-abcDEFghiJKLmnoPQR",
            id="slack-xoxb",
        ),
        pytest.param(
            "generic TOKEN carrier",
            "GITHUB_TOKEN=plainvaluewithnoprefix12345",
            "plainvaluewithnoprefix12345",
            id="generic-token",
        ),
    ],
)
def test_redacts_secret_pattern(label, raw, must_not_contain):
    result = redact_trajectory_text(raw)
    assert "***REDACTED***" in result, f"{label}: no redaction applied"
    assert must_not_contain not in result, f"{label}: secret suffix survived"


def test_preserves_non_secret_content():
    raw = '{"role": "user", "content": "Write a Python script to process data"}'
    assert redact_trajectory_text(raw) == raw


@pytest.mark.parametrize(
    "raw",
    [
        # AWS prefix in English words / identifiers (issue: ASIA matched as substring)
        '{"region": "ASIAPACIFIC"}',
        '{"id": "ASIANEWS2024UPDATED"}',
        # Hyphenated slugs containing "sk-" should not be flagged
        '{"queue": "task-sk-us-east-1-foo-bar-baz"}',
        '{"job": "workspace-sk-us-east-1-extra"}',
        # Short Daytona-prefixed identifiers
        '{"label": "dtn_v2_0"}',
        '{"name": "dtn_test"}',
        # Short Google-prefixed values that aren't keys
        '{"name": "AIzaSy"}',
    ],
)
def test_does_not_redact_non_secret_values(raw):
    """Patterns must not corrupt legitimate identifiers that share a prefix."""
    assert redact_trajectory_text(raw) == raw


def test_redacts_multiple_patterns_in_one_string():
    raw = (
        '{"env": "GEMINI_API_KEY=AIzaSyFAKEKEYFORTESTSONLYxxxxxxxxxxxxxxx '
        'ANTHROPIC_API_KEY=sk-ant-api03-abc123XYZ_longSecretValueHere"}'
    )
    result = redact_trajectory_text(raw)
    assert "FORTESTSONLYxxxxxxxxxxxxxxx" not in result
    assert "longSecretValueHere" not in result
    assert result.count("***REDACTED***") >= 2


def test_to_jsonl_uses_redaction(tmp_path):
    """End-to-end: Trajectory.to_jsonl applies redact_trajectory_text."""
    from benchflow.trajectories.types import (
        LLMExchange,
        LLMRequest,
        LLMResponse,
        Trajectory,
    )

    trajectory = Trajectory(
        session_id="test",
        exchanges=[
            LLMExchange(
                request=LLMRequest(
                    headers={"x-api-key": "AIzaSyFAKEKEYFORTESTSONLYxxxxxxxxxxxxxxx"},
                ),
                response=LLMResponse(),
            ),
        ],
    )
    jsonl = trajectory.to_jsonl(redact_keys=True)
    assert "***REDACTED***" in jsonl
    assert "FORTESTSONLYxxxxxxxxxxxxxxx" not in jsonl


def test_to_jsonl_no_redaction_preserves_keys():
    from benchflow.trajectories.types import (
        LLMExchange,
        LLMRequest,
        LLMResponse,
        Trajectory,
    )

    trajectory = Trajectory(
        session_id="test",
        exchanges=[
            LLMExchange(
                request=LLMRequest(
                    headers={"x-api-key": "AIzaSyFAKEKEYFORTESTSONLYxxxxxxxxxxxxxxx"},
                ),
                response=LLMResponse(),
            ),
        ],
    )
    jsonl = trajectory.to_jsonl(redact_keys=False)
    assert "AIzaSyFAKEKEYFORTESTSONLYxxxxxxxxxxxxxxx" in jsonl


# PR #585 finding 1: raw-text header / key-value forms (not only JSON keys)


@pytest.mark.parametrize(
    "raw,leaked",
    [
        ("x-api-key: abc123secret456value", "abc123secret456value"),
        ("api-key=abc123secret456value", "abc123secret456value"),
        ("api_key: abc123secret456value", "abc123secret456value"),
        (
            "x-goog-api-key: AIzaSyFAKEKEYFORTESTSONLYxxxxxxxxxxxxxxx",
            "FORTESTSONLYxxxxxxxxxxxxxxx",
        ),
        ("authorization: Token abc123secret456value", "abc123secret456value"),
        ("authorization: Basic dXNlcjpwYXNzd29yZGZ4eA==", "dXNlcjpwYXNz"),
        ("Authorization: bare-token-value-aaaa", "bare-token-value-aaaa"),
    ],
)
def test_redacts_raw_text_header_forms(raw, leaked):
    """PR #585 finding 1: raw header/kv text (env dumps, curl -v) must redact,
    not only JSON-shaped ``"x-api-key": "..."`` keys."""
    result = redact_trajectory_text(raw)
    assert "***REDACTED***" in result
    assert leaked not in result


def test_to_jsonl_redacts_raw_env_dump_in_message():
    """PR #585: a shell env dump pasted into trajectory content must redact."""
    body = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "GEMINI_API_KEY=AIzaSyFAKEKEYFORTESTSONLYxxxxxxxxxxxxxxx\n"
                    "DAYTONA_API_KEY=dtn_FAKEFAKEFAKEFAKEFAKE12345678\n"
                    "GITHUB_TOKEN=" + _FAKE_GHP + "\n"
                    "SLACK_TOKEN=" + _FAKE_XOXB + "\n"
                    "x-api-key: abc123secret456value"
                ),
            }
        ]
    }
    jsonl = _jsonl_for_request_body(body)
    # The live-key *values* must be gone; variable names may remain.
    assert "AIzaSyFAKEKEYFORTESTSONLYxxxxxxxxxxxxxxx" not in jsonl
    assert "dtn_FAKEFAKEFAKEFAKEFAKE12345678" not in jsonl
    assert "abc123secret456value" not in jsonl
    assert _FAKE_GHP not in jsonl
    assert _FAKE_XOXB not in jsonl
    # Env var names are preserved so the dump stays readable/auditable.
    assert "GITHUB_TOKEN" in jsonl
    assert "SLACK_TOKEN" in jsonl


# PR #585 finding 2: audit-sensitive tokens redacted whole (no live prefix)


@pytest.mark.parametrize(
    "prefix_token,prefix",
    [
        ("AIzaSyFAKEKEYFORTESTSONLYxxxxxxxxxxxxxxx", "AIzaSy"),
        ("dtn_FAKEFAKEFAKEFAKEFAKE12345678", "dtn_"),
        ("AKIAIOSFODNN7EXAMPLE", "AKIA"),
    ],
)
def test_audit_prefix_not_preserved(prefix_token, prefix):
    """PR #585 finding 2: the v0.5 leak audit greps for raw prefixes like
    ``AIzaSy``/``dtn_``; a kept prefix (``AIzaSy***``) would still trip it, so
    the whole token — prefix included — must be redacted."""
    out = redact_trajectory_text(f'{{"k": "{prefix_token}"}}')
    assert prefix not in out
    assert "***REDACTED***" in out


def test_leak_audit_intent_passes_end_to_end():
    """PR #585 finding 2: reproduce the §14 audit — a trajectory body with
    GEMINI/DAYTONA env keys, written to JSONL, must leave no live-key shape
    (the audit greps AIzaSy/dtn_ key *values*, not variable names)."""
    import re

    body = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "export GEMINI_API_KEY=AIzaSyFAKEKEYFORTESTSONLYxxxxxxxxxxxxxxx\n"
                    "export DAYTONA_API_KEY=dtn_FAKEFAKEFAKEFAKEFAKE12345678"
                ),
            }
        ]
    }
    jsonl = _jsonl_for_request_body(body)
    # No live AIzaSy.../dtn_... token shape survives (bare prefixes excluded).
    residual = [m for m in re.findall(r"AIzaSy[A-Za-z0-9_-]+|dtn_[A-Za-z0-9_]+", jsonl)]
    assert residual == [], f"live key shapes survived: {residual}"


# PR #585 bug 1: acp_trajectory.jsonl (the file the PR claims to protect) was
# written with raw json.dumps and no redaction. rollout.py and hosted_env.py
# now serialize ACP events through redact_acp_trajectory_jsonl.


def test_acp_trajectory_jsonl_redacts_event_content(tmp_path):
    """PR #585 (issue #537): an ACP trajectory event whose content carries a
    secret (an AIzaSy... live key plus a GEMINI_API_KEY=... env dump) must not
    appear raw in the written ``acp_trajectory.jsonl`` file."""
    import re

    trajectory = [
        {
            "type": "agent_message",
            "content": (
                "Here is the environment:\n"
                "GEMINI_API_KEY=AIzaSyFAKEKEYFORTESTSONLYxxxxxxxxxxxxxxx\n"
                "export DAYTONA_API_KEY=dtn_FAKEFAKEFAKEFAKEFAKE12345678\n"
                "export GITHUB_TOKEN=" + _FAKE_GHP + "\n"
                "export SLACK_TOKEN=" + _FAKE_XOXB
            ),
        },
        {"type": "tool_call", "command": "env | grep API"},
    ]

    traj_dir = tmp_path / "trajectory"
    traj_dir.mkdir()
    out = traj_dir / "acp_trajectory.jsonl"
    out.write_text(redact_acp_trajectory_jsonl(trajectory))

    written = out.read_text()
    assert "***REDACTED***" in written
    # The live-key *values* and any AIzaSy/dtn_/ghp_/xoxb- token shape must be gone.
    assert "AIzaSyFAKEKEYFORTESTSONLYxxxxxxxxxxxxxxx" not in written
    assert "dtn_FAKEFAKEFAKEFAKEFAKE12345678" not in written
    assert _FAKE_GHP not in written
    assert _FAKE_XOXB not in written
    assert "GITHUB_TOKEN" in written  # var name preserved
    residual = re.findall(
        r"AIzaSy[A-Za-z0-9_-]+|dtn_[A-Za-z0-9_]+"
        r"|ghp_[A-Za-z0-9]+|xox[baprs]-[A-Za-z0-9-]+",
        written,
    )
    assert residual == [], f"live key shapes survived: {residual}"
    # Non-secret event structure is preserved (one line per event).
    assert len(written.splitlines()) == 2
    assert "env | grep API" in written


# PR #585 bug 2: namespaced *_API_KEY=value env vars were not redacted because
# the underscore api_key rule had a (?<![A-Za-z0-9_]) lookbehind that the `_`
# in GEMINI_API_KEY tripped. The lookbehind was removed; the \1 capture keeps
# the key name so no over-redaction is introduced.


@pytest.mark.parametrize(
    "raw,leaked,kept_name",
    [
        (
            "GEMINI_API_KEY=plainSecretNoPrefix123",
            "plainSecretNoPrefix123",
            "GEMINI_API_KEY=",
        ),
        (
            '{"openai_api_key": "plainSecretNoPrefix123"}',
            "plainSecretNoPrefix123",
            "openai_api_key",
        ),
        (
            "AZURE_OPENAI_API_KEY=anotherPlainSecret456",
            "anotherPlainSecret456",
            "AZURE_OPENAI_API_KEY=",
        ),
    ],
)
def test_redacts_namespaced_api_key_env_vars(raw, leaked, kept_name):
    """PR #585 (issue #537): namespaced ``*_API_KEY=value`` env dumps and
    ``"*_api_key"`` JSON keys must redact their values even when the secret has
    no recognizable token prefix; the key name is preserved by the \\1 capture."""
    result = redact_trajectory_text(raw)
    assert "***REDACTED***" in result
    assert leaked not in result
    # The key name itself must survive (over-redaction guard).
    assert kept_name in result
