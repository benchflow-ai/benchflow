"""A long credential is redacted like a short one, and its trial counted.

Review of PR #122 (F2b-R122C, 2026-11-14, F1): the one pattern for every
form of every credential was built by recursion, a level per character, so a
value over about 950 characters (an inline CODEX_AUTH_JSON, a ChatGPT access
token, a Google service-account JSON, a Bedrock bearer token) raised
RecursionError in both session-log copies. Neither copy was kept and the
trial lost its count: a Codex trial recorded its last call's 13,584 of
38,834 tokens, a lost sandbox nothing. The trie is now built with loops. The
longest value here is 64 KB. Every value is made up.
"""

from __future__ import annotations

import asyncio
import base64
import json

import pytest

from benchflow.rollout._session_log import _redact
from benchflow.rollout._session_mirror import _clean
from tests.test_codex_acp_tokens import THREE_CALLS, _codex_log, _summed
from tests.test_en374_salvage import OPUS, _claude_session_log, _tokens, _usage
from tests.test_en374_session_mirror import _acp_rollout, _turn_lost_after


def _b64(n: int, seed: int) -> str:
    raw = bytes((i * 7919 + seed) % 256 for i in range(n))
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _jwt(n: int, seed: int) -> str:
    return "eyJhbGciOiJSUzI1NiJ9." + _b64(n, seed) + "." + _b64(256, seed + 1)


def _auth_json(size: int) -> str:
    """A Codex auth.json of about *size* characters, as CODEX_AUTH_JSON."""
    pad = max(0, (size - 900) * 3 // 8)
    return json.dumps(
        {
            "OPENAI_API_KEY": None,
            "tokens": {
                "id_token": _jwt(pad, 3),
                "access_token": _jwt(pad, 5),
                "refresh_token": "rt_FAKE" + _b64(60, 9),
                "account_id": "acct-FAKE-0001",
            },
            "last_refresh": "2026-11-13T00:00:00Z",
        }
    )


def _service_account() -> str:
    pem = "\n".join(
        [("--" + "---BEGIN" + " PRIVATE" + " KEY----" + "-")]
        + [_b64(48, i) for i in range(26)]
        + ["-----END PRIVATE KEY-----", ""]
    )
    return json.dumps(
        {
            "type": "service_account",
            "project_id": "fake-proj",
            "private_key_id": "f" * 40,
            "private_key": pem,
            "client_email": "fake@fake-proj.iam.gserviceaccount.com",
        }
    )


def _with_downloads(sandbox):
    """The local sandbox, able to hand files back, so the full copy at
    teardown is made (the test sandbox alone cannot, and the copy made while
    the turns ran is placed instead)."""
    import shutil

    async def download_file(src, dst):
        shutil.copyfile(src, dst)

    sandbox.download_file = download_file
    return sandbox


CODEX = {
    "access token, 1.3 KB": ("CODEX_ACCESS_TOKEN", _jwt(700, 2)),
    "auth.json, 4 KB": ("CODEX_AUTH_JSON", _auth_json(4096)),
    "auth.json, 64 KB": ("CODEX_AUTH_JSON", _auth_json(64 * 1024)),
}
OTHERS = {
    "Bedrock bearer, 1.4 KB": "bedrock-api-key-" + _b64(1040, 6),
    "service account, 2 KB": _service_account(),
}


def test_the_values_are_as_long_as_their_names_say():
    lengths = {name: len(value) for name, (_, value) in CODEX.items()}
    assert 1200 < lengths["access token, 1.3 KB"] < 1400
    assert 3800 < lengths["auth.json, 4 KB"] < 4400
    assert 60_000 < lengths["auth.json, 64 KB"] < 70_000
    assert len(OTHERS["Bedrock bearer, 1.4 KB"]) > 1300


@pytest.mark.parametrize(
    "name",
    [*CODEX, *OTHERS],
    ids=lambda name: name.replace(", ", "-").replace(" ", "-"),
)
def test_both_copies_redact_a_long_value(tmp_path, name):
    value = CODEX[name][1] if name in CODEX else OTHERS[name]
    line = json.dumps({"type": "user", "message": {"content": f"auth: {value} end"}})
    mirrored = _clean(line, [value])
    copies = {"mirror": mirrored}
    for suffix, text in ((".jsonl", line), (".txt", f"auth: {value} end")):
        copy = tmp_path / f"s{suffix}"
        copy.write_text(text + "\n")
        _redact(copy, [value])
        copies[f"teardown {suffix}"] = copy.read_text()
    escaped = json.dumps(value)[1:-1]
    for copy, text in copies.items():
        assert value not in text and escaped not in text, copy
        assert "[redacted]" in text, copy


@pytest.mark.asyncio
@pytest.mark.parametrize("name", list(CODEX))
async def test_a_codex_trial_lost_mid_turn_is_counted(tmp_path, monkeypatch, name):
    rollout, _trial, sandbox, home = _acp_rollout(tmp_path, monkeypatch, "codex-acp")
    rollout._agent_env = {**(rollout._agent_env or {}), CODEX[name][0]: CODEX[name][1]}
    await _turn_lost_after(
        rollout, monkeypatch, sandbox, lambda: _codex_log(home, THREE_CALLS)
    )
    rollout._env = None
    await rollout._cleanup_before_teardown()
    usage = rollout._usage_metrics
    assert _tokens(usage) == _summed(THREE_CALLS)
    assert usage["usage_source"] == "session_log"


@pytest.mark.asyncio
@pytest.mark.parametrize("name", list(CODEX))
async def test_a_codex_trial_that_ends_is_counted(tmp_path, monkeypatch, name):
    """The turn ends with the sandbox up: the full copy at teardown counts it."""
    rollout, trial, sandbox, home = _acp_rollout(tmp_path, monkeypatch, "codex-acp")
    _with_downloads(sandbox)
    rollout._agent_env = {**(rollout._agent_env or {}), CODEX[name][0]: CODEX[name][1]}

    async def prompts(_client, session, prompt_list, *_args, **_kwargs):
        session.record_user_prompt(prompt_list[0])
        _codex_log(home, THREE_CALLS)
        await asyncio.sleep(0.2)
        # codex-acp reports each turn's last call only.
        session.record_prompt_usage(
            {"inputTokens": 13500, "outputTokens": 84, "totalTokens": 13584}
        )
        return [], 0

    monkeypatch.setattr(rollout._planes, "execute_prompts", prompts)
    await rollout.execute()
    await rollout._cleanup_before_teardown()
    usage = rollout._usage_metrics
    assert (trial / "agent" / "codex-sessions").is_dir()
    assert _tokens(usage) == _summed(THREE_CALLS)
    assert usage["usage_source"] == "session_log"
    assert not usage.get("usage_incomplete")
    assert rollout._session_log_kept is True
    copied = "".join(
        path.read_text()
        for path in (trial / "agent" / "codex-sessions").rglob("*.jsonl")
    )
    assert CODEX[name][1] not in copied


@pytest.mark.asyncio
async def test_claude_on_bedrock_lost_mid_turn_is_counted(tmp_path, monkeypatch):
    rollout, _trial, sandbox, home = _acp_rollout(
        tmp_path, monkeypatch, "claude-agent-acp"
    )
    rollout._agent_env = {
        **(rollout._agent_env or {}),
        "AWS_BEARER_TOKEN_BEDROCK": OTHERS["Bedrock bearer, 1.4 KB"],
    }
    first = _usage(1500, 0, 3000, 120)
    await _turn_lost_after(
        rollout,
        monkeypatch,
        sandbox,
        lambda: _claude_session_log(home, [(OPUS, first)]),
    )
    rollout._env = None
    await rollout._cleanup_before_teardown()
    usage = rollout._usage_metrics
    assert usage["usage_source"] == "session_log"
    assert _tokens(usage)["n_input_tokens"] == 1500
    assert _tokens(usage)["n_cache_creation_tokens"] == 3000
