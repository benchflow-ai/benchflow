"""SPEED-FIX (0.8.3 harness time): round-one changes fail at dc509087a.

- The direct Docker path takes BenchFlow's own no-restart file,
  which every trial with the cloud metadata block stacks; at dc509087a every
  such trial fell back to Compose ("service key restart"). The fixture
  ``fixtures/direct_docker/compose-2.40.3-no-restart.json`` is Compose's own
  printed model, hash and ContainerCreate body for it, recorded on VM 22.
- Trajectory redaction redacts each distinct text once per trial, with the
  same result.
- The host proxy's health poll makes one HTTP client, not one per poll.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from benchflow.sandbox import _direct_docker as dd
from benchflow.task.compose_env import decode_dollars

NO_RESTART = json.loads(
    (
        Path(__file__).parent / "fixtures/direct_docker/compose-2.40.3-no-restart.json"
    ).read_text()
)


def _no_restart_project() -> dict:
    return decode_dollars(NO_RESTART["printed"])


# --- The direct path and the no-restart file ----------------------------------


def test_the_no_restart_file_takes_the_direct_path():
    project = _no_restart_project()
    assert project["services"]["main"]["restart"] == "no"
    dd.check_project(project, "speedfixrp")


def test_the_no_restart_body_and_hash_are_composes():
    project = _no_restart_project()
    service = project["services"]["main"]
    assert dd.service_hash(service) == NO_RESTART["hash"]
    got = dd.container_body(
        project,
        service,
        image_id=NO_RESTART["image_id"],
        compose_version="2.40.3",
        config_files=NO_RESTART["files"],
        working_dir=NO_RESTART["working_dir"],
        replace=False,
    )
    expected = NO_RESTART["requests"]["container"]
    assert got["HostConfig"]["RestartPolicy"] == {"Name": "no", "MaximumRetryCount": 0}
    assert dd.order_rules(got) == dd.order_rules(expected)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda s: s.update(restart="always"), "restart policy other than no"),
        (lambda s: s.update(restart="on-failure:3"), "restart policy other than no"),
        (
            lambda s: s["deploy"]["restart_policy"].update(condition="any"),
            "restart_policy other than none",
        ),
        (
            lambda s: s["deploy"]["restart_policy"].update(max_attempts=2),
            "restart_policy other than none",
        ),
    ],
)
def test_any_other_restart_policy_goes_to_compose(change, reason):
    project = _no_restart_project()
    change(project["services"]["main"])
    with pytest.raises(dd.Fallback, match=reason):
        dd.check_project(project, "speedfixrp")


# --- Redaction runs once per distinct text -------------------------------------


def test_redaction_of_a_repeated_text_runs_the_patterns_once(monkeypatch):
    from benchflow.trajectories import types

    calls = 0
    real = types._REDACTION_PATTERNS

    class Counting:
        def __init__(self, pattern):
            self.pattern = pattern

        def subn(self, replacement, text):
            nonlocal calls
            calls += 1
            return self.pattern.subn(replacement, text)

    monkeypatch.setattr(
        types, "_REDACTION_PATTERNS", [(Counting(p), r, c) for p, r, c in real]
    )
    text = "token sk-ant-api03-" + "a" * 40 + " and GITHUB_TOKEN=ghp_" + "b" * 36
    with types.trial_redaction_memo():
        first = types.redact_trajectory_text_with_categories(text)
        once = calls
        for _ in range(5):
            assert types.redact_trajectory_text_with_categories(text) == first
        assert calls == once == len(real)


def test_remembered_redaction_equals_a_fresh_one(monkeypatch):
    from benchflow.trajectories import types

    samples = [
        "plain words",
        "Authorization: Bearer abcdefghijklmnopqrstuvwx",
        '{"x-api-key": "prefixless-secret-value-1234567890"}',
        "AWS_SECRET_ACCESS_KEY=" + "A" * 40,
        "https://user:hunter2hunter2@example.com/x",
    ]
    with types.trial_redaction_memo():
        fresh = [types.redact_trajectory_text_with_categories(s) for s in samples]
        again = [types.redact_trajectory_text_with_categories(s) for s in samples]
        assert again == fresh
        # The caller's category dict is its own: changing it changes no later answer.
        again[1][1].clear()
        assert types.redact_trajectory_text_with_categories(samples[1]) == fresh[1]
        obj = {
            "headers": {"x-api-key": "prefixless-secret-value-1234567890"},
            "t": samples,
        }
        assert types.redact_trajectory_obj(obj) == types.redact_trajectory_obj(obj)


def test_the_redaction_memo_is_bounded(monkeypatch):
    from benchflow.trajectories import types

    monkeypatch.setattr(types, "_REDACTED_MAX_BYTES", 2000)
    with types.trial_redaction_memo() as memo:
        for i in range(20):
            types.redact_trajectory_text_with_categories(f"text number {i:04d}")
        assert memo.size <= 2000
        assert len(memo.entries) < 20
        big = "x" * 2000
        assert types.redact_trajectory_text_with_categories(big) == (big, {})
        assert big not in memo.entries


# --- The host proxy's health poll ---------------------------------------------


def test_the_health_poll_makes_one_client(monkeypatch):
    from unittest.mock import MagicMock

    import benchflow.providers.litellm_runtime as rt
    from benchflow._proxy_zygote import ProxyProcess

    made = 0
    answers = iter([ConnectionError("refused")] * 6)

    class Client:
        def __init__(self, *args, **kwargs):
            nonlocal made
            made += 1

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url):
            answer = next(answers, None)
            if answer is not None:
                raise answer
            return MagicMock(status_code=200)

    monkeypatch.setattr(rt.httpx, "AsyncClient", Client)
    process = MagicMock()
    process.process = MagicMock(spec=ProxyProcess)
    process.process.poll.return_value = None
    process.endpoint.local_base_url = "http://127.0.0.1:9"
    asyncio.run(rt._poll_host_health(process, deadline_s=30.0))
    assert made == 1


# --- The ACP SDK is imported beside the sandbox setup ---------------------------


def test_the_acp_sdk_is_imported_in_a_thread_once(monkeypatch):
    from benchflow.acp import types

    started = []

    class Thread:
        def __init__(self, *, target, name, daemon):
            started.append((target, name, daemon))

        def start(self):
            pass

    monkeypatch.setattr(types.threading, "Thread", Thread)
    monkeypatch.setattr(types, "_sdk_import_started", False)
    monkeypatch.delitem(types.sys.modules, "acp.schema", raising=False)
    imported = []
    monkeypatch.setattr(types.importlib, "import_module", imported.append)
    types.prewarm_sdk()
    types.prewarm_sdk()
    assert [(name, daemon) for _, name, daemon in started] == [("acp-import", True)]
    started[0][0]()
    assert imported == ["acp.schema", "acp.meta"]


@pytest.mark.parametrize(
    ("agent", "harness", "prewarmed"),
    [
        ("mini-swe", "acp", True),
        ("claude", "native", False),
        ("oracle", "acp", False),
        ("nop", "acp", False),
        (None, "acp", False),
    ],
)
def test_an_acp_run_prewarms_the_sdk_once_its_sandbox_is_up(
    monkeypatch, agent, harness, prewarmed
):
    from types import SimpleNamespace

    from benchflow.acp import types
    from benchflow.rollout import Rollout

    calls = []
    monkeypatch.setattr(types, "prewarm_sdk", lambda: calls.append(1))
    rollout = SimpleNamespace(
        _config=SimpleNamespace(primary_agent=agent, harness=harness)
    )
    Rollout._prewarm_acp(rollout)
    assert bool(calls) is prewarmed
