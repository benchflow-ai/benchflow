"""Tests for trajectory secret redaction patterns.

Guards PR #585 (extends #537): redaction must cover raw-text header/kv forms
(agents print env, curl -v, request logs into trajectories), and audit-sensitive
token families (Google/Daytona/AWS) must be redacted whole — prefix included —
so the v0.5 secret-leak audit grep sees no live-key shape.
"""

import json

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


# Issue #830 fix#2: surfacing the upstream provider cause (Option A) routes the
# FULL provider exception string into error.message for ALL failure types, so
# redact_trajectory_text must cover the credential shapes that appear in real
# provider error bodies. These vectors were all verified leaking before this fix
# (adversarial review), across URL userinfo, query params, hyphenated key
# families, and the json.dumps-escaped JSON that to_jsonl produces.

# Split literals so the verbatim secret never appears in source (push protection).
_FAKE_MASTER_KEY = "sk-" + "benchflow-" + "FAKEmasterKEYvalue1234567890abcd"
_FAKE_SVCACCT = "sk-" + "svcacct-" + "T3BlbkFJFAKEsvcacct0123456789ab"
_FAKE_ADMIN = "sk-" + "admin-" + "T3BlbkFJFAKEadminkey0123456789ab"
_FAKE_OPENROUTER = "sk-" + "or-v1-" + "0123fakeOPENROUTER456789abcdef0123456789"
_FAKE_URL_PW = "s3cret" + "P4sswordVALUE"
_FAKE_DB_PW = "Db" + "Pr0dPassValue123"
_FAKE_BEARER_TOK = "bearersecret" + "value1234567890"
_FAKE_QUERY_SECRET = "GENERICQUERY" + "SECRETvalue12345"
_FAKE_BEDROCK = "bedrocksecret" + "value123456"
_FAKE_GSK = "gsk_" + "FAKEgroq0123456789ABCDEF01234567"
_FAKE_XAI = "xai-" + "FAKExai0123456789ABCDEF012345678"
_FAKE_R8 = "r8_" + "FAKErepl0123456789ABCDEF01234567"
_FAKE_HF = "hf_" + "FAKEhugg0123456789ABCDEF01234567"
_FAKE_FW = "fw_" + "FAKEfire0123456789ABCDEF01234567"
_FAKE_JWT = (
    "eyJ"
    + "hbGciOiJIUzI1NiJ9"
    + "."
    + "eyJzdWIiOiIxMjM0NTY3ODkwIn0"
    + "."
    + "dQw4w9WgXcQfakeSIGNATURE12"
)


@pytest.mark.parametrize(
    "label,raw,leaked,kept",
    [
        pytest.param(
            "https userinfo (user:pass@host)",
            f"connecting to https://admin:{_FAKE_URL_PW}@proxy.internal/v1",
            _FAKE_URL_PW,
            "proxy.internal",
            id="url-userinfo-https",
        ),
        pytest.param(
            "postgres userinfo (DSN in a failure body)",
            f"could not connect: postgres://litellm:{_FAKE_DB_PW}@db.internal:5432/x",
            _FAKE_DB_PW,
            "db.internal",
            id="url-userinfo-postgres",
        ),
        pytest.param(
            "redis userinfo, empty username (:pass@)",
            f"ConnectionError redis://:{_FAKE_DB_PW}@cache.prod:6379/0",
            _FAKE_DB_PW,
            "cache.prod",
            id="url-userinfo-redis-emptyuser",
        ),
        pytest.param(
            "benchflow master key (sk-benchflow- family)",
            f'{{"master_key": "{_FAKE_MASTER_KEY}"}}',
            "FAKEmasterKEYvalue1234567890abcd",
            None,
            id="sk-benchflow",
        ),
        pytest.param(
            "OpenAI service-account key (sk-svcacct-)",
            f"Incorrect API key provided: {_FAKE_SVCACCT}",
            "T3BlbkFJFAKEsvcacct0123456789ab",
            None,
            id="sk-svcacct",
        ),
        pytest.param(
            "OpenAI admin key (sk-admin-)",
            f"AuthenticationError: {_FAKE_ADMIN}",
            "T3BlbkFJFAKEadminkey0123456789ab",
            None,
            id="sk-admin",
        ),
        pytest.param(
            "OpenRouter key (sk-or-v1-)",
            f"OpenrouterException - invalid api key {_FAKE_OPENROUTER}",
            "0123fakeOPENROUTER456789abcdef0123456789",
            None,
            id="sk-or-v1",
        ),
        pytest.param(
            "master key inside a Bearer header",
            f"Authorization: Bearer {_FAKE_MASTER_KEY}",
            "FAKEmasterKEYvalue1234567890abcd",
            "Bearer",
            id="master-key-bearer",
        ),
        pytest.param(
            "AWS_BEARER_TOKEN_BEDROCK env dump (TOKEN not at name end)",
            f"AWS_BEARER_TOKEN_BEDROCK={_FAKE_BEDROCK} set",
            _FAKE_BEDROCK,
            "AWS_BEARER_TOKEN_BEDROCK",
            id="bedrock-bearer-token",
        ),
        pytest.param(
            "Python dict-repr single-quoted Authorization",
            f"headers={{'Authorization': 'Bearer {_FAKE_BEARER_TOK}'}}",
            _FAKE_BEARER_TOK,
            "Bearer",
            id="dict-repr-authorization",
        ),
        pytest.param(
            "Python dict-repr single-quoted x-api-key",
            f"headers={{'x-api-key': '{_FAKE_BEARER_TOK}'}}",
            _FAKE_BEARER_TOK,
            None,
            id="dict-repr-x-api-key",
        ),
        pytest.param(
            "secret in URL query param (?apikey=)",
            f"GET https://api/v1/x?apikey={_FAKE_QUERY_SECRET}&page=2",
            _FAKE_QUERY_SECRET,
            "page=2",
            id="query-apikey",
        ),
        pytest.param(
            "secret in URL query param (?access_token=)",
            f"https://api/v1/x?access_token={_FAKE_QUERY_SECRET}&z=1",
            _FAKE_QUERY_SECRET,
            "z=1",
            id="query-access-token",
        ),
        pytest.param(
            "Azure SAS signature (?sig=)",
            f"blob fetch failed https://acct.blob/c/f?sv=2023-01-01&sig={_FAKE_QUERY_SECRET}&se=x",
            _FAKE_QUERY_SECRET,
            "sv=2023-01-01",
            id="query-sas-sig",
        ),
        pytest.param(
            "session id in query (?sessionid=)",
            f"https://api?sessionid={_FAKE_QUERY_SECRET}&b=2",
            _FAKE_QUERY_SECRET,
            "b=2",
            id="query-sessionid",
        ),
        pytest.param(
            "Groq key (gsk_)",
            f"groq AuthenticationError: {_FAKE_GSK}",
            "FAKEgroq0123456789ABCDEF01234567",
            None,
            id="provider-gsk",
        ),
        pytest.param(
            "xAI key (xai-)",
            f"xai exception: {_FAKE_XAI}",
            "FAKExai0123456789ABCDEF012345678",
            None,
            id="provider-xai",
        ),
        pytest.param(
            "Replicate token (r8_)",
            f"replicate 401: {_FAKE_R8}",
            "FAKErepl0123456789ABCDEF01234567",
            None,
            id="provider-r8",
        ),
        pytest.param(
            "HuggingFace token (hf_)",
            f"hf inference error {_FAKE_HF}",
            "FAKEhugg0123456789ABCDEF01234567",
            None,
            id="provider-hf",
        ),
        pytest.param(
            "Fireworks key (fw_)",
            f"fireworks auth fail {_FAKE_FW}",
            "FAKEfire0123456789ABCDEF01234567",
            None,
            id="provider-fw",
        ),
        pytest.param(
            "JWT session/bearer token",
            f"token expired: {_FAKE_JWT}",
            "dQw4w9WgXcQfakeSIGNATURE12",
            None,
            id="jwt",
        ),
        pytest.param(
            "AWS_SECRET_ACCESS_KEY (SECRET mid-name, ends ACCESS_KEY)",
            "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMIK7MDENGbPx" + "RfiCYsecretKEYval",
            "wJalrXUtnFEMIK7MDENGbPx" + "RfiCYsecretKEYval",
            "AWS_SECRET_ACCESS_KEY",
            id="aws-secret-access-key",
        ),
        pytest.param(
            "Azure storage account key (ends ACCOUNT_KEY)",
            "AZURE_STORAGE_ACCOUNT_KEY=base64acct" + "AAAABBBBCCCCDDDDEEEE==",
            "base64acct" + "AAAABBBBCCCCDDDDEEEE==",
            "AZURE_STORAGE_ACCOUNT_KEY",
            id="azure-account-key",
        ),
        pytest.param(
            "GCP/Vertex service-account PEM private key block",
            '{"private_key": "-----BEGIN PRIVATE KEY-----\\nMIIEvQIBADANBgkq'
            + "PEMfakeSECRETmaterial12345"
            + '\\n-----END PRIVATE KEY-----"}',
            "PEMfakeSECRETmaterial12345",
            None,
            id="gcp-pem-private-key",
        ),
        pytest.param(
            "URL password containing an un-encoded @",
            "redis://admin:p@ss" + "w0rdSECRETval@cache.internal:6379/0",
            "p@ss" + "w0rdSECRETval",
            "cache.internal",
            id="url-literal-at-password",
        ),
        pytest.param(
            "OpenAI admin key with uppercase (sk-admin-)",
            "AuthError: sk-" + "admin-" + "T3BlbkFJFAKEadminKEY0123456789ab",
            "T3BlbkFJFAKEadminKEY0123456789ab",
            None,
            id="sk-admin-uppercase",
        ),
    ],
)
def test_redacts_proxy_failure_leak_vectors(label, raw, leaked, kept):
    """Issue #830 fix#2: each input is a real leak shape from provider failure
    text that survived redaction before this fix (adversarial review)."""
    result = redact_trajectory_text(raw)
    assert "***REDACTED***" in result, f"{label}: nothing redacted"
    assert leaked not in result, f"{label}: secret survived"
    if kept is not None:
        assert kept in result, f"{label}: over-redacted (dropped {kept!r})"


@pytest.mark.parametrize(
    "carrier_json,leaked",
    [
        pytest.param(
            'AzureException 401 - {"sent_headers": {"api-key": "AZ0123456789abcdefSECRET"}}',
            "AZ0123456789abcdefSECRET",
            id="escaped-api-key",
        ),
        pytest.param(
            'OpenAIException {"request": {"headers": {"authorization": "Bearer ESCbearerSECRETtok123456"}}}',
            "ESCbearerSECRETtok123456",
            id="escaped-authorization",
        ),
        pytest.param(
            'ProxyException {"master_key": "ESCmasterKEYnoprefixSECRET12"}',
            "ESCmasterKEYnoprefixSECRET12",
            id="escaped-master-key",
        ),
    ],
)
def test_redacts_escaped_json_provider_body(carrier_json, leaked):
    """Issue #830 fix#2 (the central new surface): Option A surfaces the provider
    error body into error.message; Trajectory.to_jsonl runs json.dumps over the
    record (escaping inner quotes to \\") BEFORE redacting. The carriers must fire
    on that backslash-escaped form, not just raw/JSON/dict-repr."""
    # Exactly what to_jsonl feeds redact_trajectory_text: json.dumps of a record
    # whose error.message embeds the provider's JSON body.
    serialized = json.dumps({"event": "failure", "error": {"message": carrier_json}})
    assert leaked in serialized  # sanity: the secret is present pre-redaction
    result = redact_trajectory_text(serialized)
    assert "***REDACTED***" in result
    assert leaked not in result, "escaped-JSON carrier bypass — secret survived"


def test_redacts_double_escaped_json_carrier():
    """Issue #830: a body that is json.dumps'd twice has THREE backslashes before
    each quote; _ESCQ absorbs a run of backslashes, so even double-escaped carriers
    redact (guards against a single-backslash-only escape atom)."""
    secret = "DBL" + "771b8b5e9f2a4c6d8e0f1a2b"
    double = json.dumps(json.dumps(f'{{"api-key": "{secret}"}}'))
    assert secret in double
    out = redact_trajectory_text(double)
    assert secret not in out
    assert "***REDACTED***" in out


def test_redaction_preserves_json_when_code_mentions_token_label():
    """Guards the expanded integration run: ordinary code text like ``class Token:``
    must not be treated as a secret carrier after JSON serialization."""

    serialized = json.dumps(
        {
            "request": {
                "body": {
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "class Token:\n"
                                '    """Immutable token representation."""\n'
                                "    value: str\n"
                            ),
                        }
                    ]
                }
            }
        }
    )

    redacted = redact_trajectory_text(serialized)

    json.loads(redacted)
    assert "class Token" in redacted
    assert "***REDACTED***" not in redacted


@pytest.mark.parametrize(
    "raw",
    [
        # host:port in a URL has no userinfo — must not be touched.
        '{"url": "https://example.com:8080/v1/chat"}',
        "redis://cache.internal:6379/0",
        # An email/'@' in a query param must not be mistaken for userinfo.
        "https://localhost:8080?login_hint=user@example.com",
        "GET https://auth.example.com:8443?login_hint=alice@corp.com&state=xyz",
        "https://h:9000?a=1&b=2&c=x@d.com&keepme=yes",
        # Non-secret query params keep their values (bare key/token/auth excluded).
        "https://api/v1/list?page=2&limit=10&sortkey=name",
        "https://api/search?monkey=banana&donkey=kong",
        "https://api/v1?author=jane&keyboard=qwerty&pwd2=keepme",
        "https://api/items?key=name&order=asc",
        "https://api/login?auth=basic&user=bob",
        "https://x.com/cb?sig=v2&id=42",  # short ?sig= is a scheme-version flag
        # Names that merely CONTAIN token/secret/key but aren't secrets.
        "TOKENIZER_PATH=/models/tokenizer.json",
        '{"sortkey": "created_at", "pagekey": "next"}',
        # Common non-secret *_key DB/ORM identifiers (NOT a secret key suffix).
        "primary_key=id_column",
        "{'foreign_key': 'user_id'}",
        "sort_key=created_at",
        # Words that merely CONTAIN a secret marker but don't end in one.
        "SECRETARY_NAME=Jane",
        "MONKEY=banana",
        # Single-quoted non-secret dict reprs.
        "{'role': 'user', 'content': 'hello world'}",
        # sk-proj-/sk-admin- kebab slugs (lowercase → no uppercase → not a key).
        "git checkout sk-proj-refactor-auth-module",
        "kubectl create sa sk-admin-cluster-operator-prod",
        # eyJ-prefixed identifiers / dotted base64 wire formats that aren't JWTs
        # (3rd segment < 20 chars).
        "result = eyJsonObject.parseColumns.getValuesFrom(input)",
        "msg=eyJ0eXBlIjoiUElORyJ9.eyJzZXEiOjEwMDB9.eyJhY2siOnRydWV9",
        # Bespoke app deep-link with :...@ that isn't a credential (scheme not in
        # the credential-bearing allowlist).
        "vscode://file:line@/path",
    ],
)
def test_proxy_failure_vectors_do_not_over_redact(raw):
    """The leak-vector patterns must not corrupt legitimate URLs, query params,
    env names, or dict reprs that merely resemble a secret carrier."""
    assert redact_trajectory_text(raw) == raw


def test_carrier_value_stops_at_ampersand():
    """A token-named query param redacts only its value, not the sibling params
    after `&` (the *token* carrier value class excludes `&`)."""
    out = redact_trajectory_text(
        "https://h/v1?refresh_token=SEKRETvalue123456&model=gpt-4&n=2"
    )
    assert "SEKRETvalue123456" not in out
    assert "model=gpt-4" in out
    assert "n=2" in out


@pytest.mark.parametrize(
    "label,pathological",
    [
        # Long colon-rich, '@'-less URL — userinfo username class excludes ':'.
        ("colon-rich-url", "https://" + "x:" * 4000),
        # A large base64 image field is the realistic DoS: an uncapped name-prefix
        # `[A-Za-z0-9_]*` before a marker backtracks O(n²) on it. The {0,64} cap
        # plus the (?<![A-Za-z0-9]) left-anchor on the variable-prefix carriers
        # keep it linear. (Pre-fix this stalled the redactor for ~45 s.)
        ("base64-image", '{"image":"' + "iVBORw0KGgoAAAANSUhEUgAA" * 8000 + '"}'),
        # A long contiguous alnum/underscore run (near-miss for the *TOKEN* carrier).
        ("alnum-run", "Aa0_" * 4000),
    ],
)
def test_redactor_no_redos(label, pathological):
    """Guards ReDoS across the new/modified patterns: redaction of a large benign
    blob must stay near-linear, not catastrophically backtrack."""
    import time

    start = time.perf_counter()
    redact_trajectory_text(pathological)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 750, f"possible ReDoS on {label}: {elapsed_ms:.0f} ms"


# Regression (PR #849): the redactor used to run over the already-serialized JSON
# (``json.dumps`` then ``re.sub``). When a redacted secret sat next to a ``\``
# escape, the substitution could split it and leave an invalid ``\X``, corrupting
# llm_trajectory.jsonl / acp_trajectory.jsonl so they no longer parsed and
# ``bench train convert`` hard-failed. The writers now redact string *values*
# before serialization (``redact_trajectory_obj``), so output is always valid JSON.

_ADVERSARIAL_CONTENTS = [
    # secret immediately after a backslash
    "auth header:\\" + "sk-abc1234567defghijklmnop987654",
    # carrier-tripping code (``token =`` / ``if not token:``) with escaped quotes,
    # newlines, and a secret in a comment — mirrors a real agent code edit that
    # reproduced the corruption in the wild.
    'token = form.get("token")\n'
    "    if not token:\n"
    '        raise HTTPException(400, "invalid")  '
    "# key sk-abc1234567defghijklmnop987654\n",
    # windows-style path backslashes adjacent to an AWS key
    "C:\\Users\\app\\AKIAIOSFODNN7EXAMPLE",
    # a JSON blob (escaped quotes once serialized) carrying a Google key
    '{"x-api-key": "AIzaSyFAKEKEYFORTESTSONLYxxxxxxxxxxxxxxx"}',
]

_ADVERSARIAL_LEAKS = (
    "abc1234567defghijklmnop987654",
    "IOSFODNN7EXAMPLE",
    "FORTESTSONLYxxxxxxxxxxxxxxx",
)


@pytest.mark.parametrize("content", _ADVERSARIAL_CONTENTS)
def test_to_jsonl_emits_valid_json_with_secret_adjacent_escapes(content):
    """PR #849: to_jsonl round-trips to valid JSON even when content puts a secret
    next to a backslash escape, and still strips the secret."""
    traj = Trajectory(
        session_id="t",
        exchanges=[
            LLMExchange(
                request=LLMRequest(
                    body={"messages": [{"role": "user", "content": content}]}
                ),
                response=LLMResponse(
                    status_code=200,
                    body={
                        "choices": [
                            {"message": {"role": "assistant", "content": content}}
                        ]
                    },
                ),
            )
        ],
    )
    jsonl = traj.to_jsonl(redact_keys=True)
    for line in jsonl.splitlines():
        json.loads(line)  # must round-trip — no corrupt escapes
    for leaked in _ADVERSARIAL_LEAKS:
        assert leaked not in jsonl


@pytest.mark.parametrize("content", _ADVERSARIAL_CONTENTS)
def test_acp_jsonl_emits_valid_json_with_secret_adjacent_escapes(content):
    """PR #849: redact_acp_trajectory_jsonl emits valid JSON lines for the same
    secret-adjacent-escape content."""
    out = redact_acp_trajectory_jsonl(
        [{"type": "agent_message", "content": content}, {"type": "tool_call"}]
    )
    lines = out.splitlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)
    for leaked in _ADVERSARIAL_LEAKS:
        assert leaked not in out


def test_redact_trajectory_obj_redacts_nested_string_leaves():
    """PR #849: redact_trajectory_obj scrubs string leaves at any depth, leaving
    non-string values and structure intact."""
    from benchflow.trajectories.types import redact_trajectory_obj

    obj = {
        "a": "sk-abc1234567defghijklmnop987654",
        "b": ["plain", {"c": "AIzaSyFAKEKEYFORTESTSONLYxxxxxxxxxxxxxxx"}],
        "n": 5,
        "ok": True,
    }
    out = redact_trajectory_obj(obj)
    dumped = json.dumps(out)  # re-serializes to valid JSON
    for leaked in ("abc1234567defghijklmnop987654", "FORTESTSONLYxxxxxxxxxxxxxxx"):
        assert leaked not in dumped
    # non-string leaves and structure are preserved
    assert out["n"] == 5 and out["ok"] is True
    assert out["b"][0] == "plain"


# PR #849 review (discussion_r3494557495): object-level redaction must keep the
# field-name context, or the ``name: value`` carriers stop stripping *prefixless*
# secrets held in structured fields (LLM request/response headers + body, ACP
# structured events). A prefixless value has no token prefix and no inline
# ``name: value`` text, so only key context can flag it.

_PREFIXLESS_SECRET = "prefixlessSecretValue1234567890"


def test_to_jsonl_redacts_prefixless_secret_in_structured_headers_and_body():
    """PR #849 review: prefixless secrets in request/response headers or body must
    still redact via key context, with valid-JSON output."""
    traj = Trajectory(
        session_id="t",
        exchanges=[
            LLMExchange(
                request=LLMRequest(
                    headers={"x-api-key": _PREFIXLESS_SECRET},
                    body={
                        "api_key": _PREFIXLESS_SECRET,
                        "authorization": "Bearer " + _PREFIXLESS_SECRET,
                    },
                ),
                response=LLMResponse(
                    status_code=200,
                    headers={"x-api-key": _PREFIXLESS_SECRET},
                    body={
                        "choices": [{"message": {"role": "assistant", "content": "ok"}}]
                    },
                ),
            )
        ],
    )
    jsonl = traj.to_jsonl(redact_keys=True)
    parsed = json.loads(jsonl)  # single exchange -> one valid line
    assert _PREFIXLESS_SECRET not in jsonl
    assert parsed["request"]["headers"]["x-api-key"] == "***REDACTED***"
    assert parsed["request"]["body"]["api_key"] == "***REDACTED***"
    assert parsed["request"]["body"]["authorization"] == "Bearer ***REDACTED***"
    assert parsed["response"]["headers"]["x-api-key"] == "***REDACTED***"


def test_acp_jsonl_redacts_prefixless_secret_in_structured_event():
    """PR #849 review: a prefixless secret in a structured ACP event field (key
    separate from value, possibly nested) must redact via key context."""
    out = redact_acp_trajectory_jsonl(
        [
            {
                "type": "tool_result",
                "headers": {"authorization": _PREFIXLESS_SECRET},
                "api_key": _PREFIXLESS_SECRET,
            }
        ]
    )
    event = json.loads(out)  # valid JSON
    assert _PREFIXLESS_SECRET not in out
    assert event["headers"]["authorization"] == "***REDACTED***"
    assert event["api_key"] == "***REDACTED***"


def test_redact_trajectory_obj_preserves_key_context_for_structured_secret():
    """PR #849 review: object-level redaction keys off the field name (parity with
    the prior serialized-text path) without touching non-secret fields."""
    from benchflow.trajectories.types import redact_trajectory_obj

    out = redact_trajectory_obj({"x-api-key": _PREFIXLESS_SECRET, "note": "plain text"})
    assert out["x-api-key"] == "***REDACTED***"
    assert out["note"] == "plain text"
