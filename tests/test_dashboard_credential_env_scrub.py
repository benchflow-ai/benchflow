"""Guards dashboard ``data.json`` from embedding credential-rule env var
names verbatim in artifact content (#494).

Secret scanners trip on names like ``GEMINI_API_KEY`` and ``DAYTONA_API_KEY``
even when no value is present. We never ship values, but the names showed up
in rollout artifact payloads (configs, logs, etc.), creating noise in
artifact-sharing pipelines. Names must be masked to a stable placeholder.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dashboard import generate


def _walk_strings(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _walk_strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _walk_strings(v)


def test_scrub_credential_env_names_masks_known_rule_names() -> None:
    text = (
        "Configured with GEMINI_API_KEY and DAYTONA_API_KEY from env.\n"
        "GOOGLE_API_KEY mirror set.\n"
    )
    scrubbed = generate._scrub_credential_env_names(text)
    assert "GEMINI_API_KEY" not in scrubbed
    assert "DAYTONA_API_KEY" not in scrubbed
    assert "GOOGLE_API_KEY" not in scrubbed
    assert scrubbed.count("<CREDENTIAL_ENV>") == 3


def test_scrub_credential_env_names_preserves_unrelated_text() -> None:
    text = "MY_OTHER_VAR=hello and PUBLIC_FLAG=on"
    assert generate._scrub_credential_env_names(text) == text


def test_file_payload_scrubs_credential_env_names(tmp_path: Path) -> None:
    """``_file_payload`` embeds artifact text into data.json; that text
    must not carry credential-rule env var names verbatim."""
    artifact = tmp_path / "agent_config.json"
    artifact.write_text(
        json.dumps(
            {
                "agent": "gemini",
                "required_env": ["GEMINI_API_KEY", "DAYTONA_API_KEY"],
                "note": "GOOGLE_API_KEY is a mirror of GEMINI_API_KEY",
            }
        )
    )

    content, _lines, _truncated, lang = generate._file_payload(artifact)

    assert lang == "json"
    assert content is not None
    assert "GEMINI_API_KEY" not in content
    assert "DAYTONA_API_KEY" not in content
    assert "GOOGLE_API_KEY" not in content
    assert "<CREDENTIAL_ENV>" in content


def test_collect_jobs_payload_has_no_credential_env_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: a rollout whose artifact files mention credential-rule env
    var names produces a data.json payload with no such names anywhere."""
    jobs_root = tmp_path / "jobs"
    rollout = jobs_root / "2026-05-22__01-30-00" / "task-a__abc123"
    (rollout / "verifier").mkdir(parents=True)
    (rollout / "trajectory").mkdir(parents=True)
    (rollout / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "agent": "codex",
                "rewards": {"reward": 1.0},
                "timing": {},
            }
        )
    )
    (rollout / "config.json").write_text(
        json.dumps(
            {
                "agent": "codex",
                # Credential rule names appear in the config artifact — this
                # is the #494 leak vector.
                "agent_env": ["GEMINI_API_KEY", "DAYTONA_API_KEY"],
                "comment": "GOOGLE_API_KEY auto-mirrored from GEMINI_API_KEY",
            }
        )
    )
    (rollout / "verifier" / "reward.txt").write_text("1.0\n")
    (rollout / "trajectory" / "acp_trajectory.jsonl").write_text(
        json.dumps({"type": "step"}) + "\n"
    )

    monkeypatch.setenv(generate.JOBS_ROOT_ENV, str(jobs_root))
    payload = generate.collect_jobs()

    forbidden = ("GEMINI_API_KEY", "DAYTONA_API_KEY", "GOOGLE_API_KEY")
    for s in _walk_strings(payload):
        for name in forbidden:
            assert name not in s, (
                f"credential env var name {name!r} leaked into data.json: {s!r}"
            )
