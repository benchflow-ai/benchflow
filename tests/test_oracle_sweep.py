"""Guards the oracle policy regressions introduced by 4adfb00f."""

import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.rollout import _setup
from benchflow.task import Task
from tests.test_oracle_network_policy import ALLOWLIST, FakeSandbox, _task


@pytest.mark.asyncio
@pytest.mark.parametrize("disk_policy", ["", ALLOWLIST])
async def test_oracle_uses_effective_policy_not_the_disk_task(
    tmp_path, monkeypatch, disk_policy
):
    """Overrides apply to open and already-filtered tasks."""
    path = _task(tmp_path / "task", disk_policy)
    effective = Task(path)
    effective.config.sandbox.network_mode = "allowlist"
    effective.config.sandbox.allowed_hosts = ["pypi.org", "example.com"]
    arm = AsyncMock(return_value=SimpleNamespace(env={}, gid=None))
    monkeypatch.setattr(_setup, "arm_oracle_egress", arm)
    monkeypatch.setattr(_setup, "disarm_oracle_egress", AsyncMock())
    kwargs = (
        {"task": effective}
        if "task" in inspect.signature(_setup._run_oracle).parameters
        else {}
    )
    await _setup._run_oracle(FakeSandbox(), path, timeout=30, **kwargs)
    arm.assert_awaited_once()
    assert arm.await_args.args[1].allowed_hosts == ("pypi.org", "example.com")


@pytest.mark.asyncio
async def test_oracle_proxy_receives_resolved_diagnostic_secrets(tmp_path, monkeypatch):
    """Quiet solution env secrets still need redaction."""
    marker = "synthetic-oracle-credential"
    path = _task(
        tmp_path / "task",
        ALLOWLIST,
        '[solution.env]\nDOWNLOAD_TOKEN = "${DOWNLOAD_TOKEN}"\n',
    )
    monkeypatch.setenv("DOWNLOAD_TOKEN", marker)
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_ENV", "DOWNLOAD_TOKEN")
    arm = AsyncMock(return_value=SimpleNamespace(env={}, gid=None))
    monkeypatch.setattr(_setup, "arm_oracle_egress", arm)
    monkeypatch.setattr(_setup, "disarm_oracle_egress", AsyncMock())
    await _setup._run_oracle(FakeSandbox(), path, timeout=30)
    policy = arm.await_args.args[1]
    assert marker in getattr(policy, "diagnostic_secrets", ())


@pytest.mark.asyncio
async def test_block_response_and_log_do_not_expose_solution_secrets(tmp_path):
    """Guards the fix at the raw proxy's log and response boundaries."""
    import json

    from benchflow.sandbox import _egress_denylist_proxy as proxy

    marker = "synthetic-oracle-credential"
    kwargs = (
        {"diagnostic_secrets": [marker]}
        if "diagnostic_secrets" in inspect.signature(proxy.Policy).parameters
        else {}
    )
    policy = proxy.Policy([], ["downloads.example.test"], **kwargs)
    handler = object.__new__(proxy.Handler)
    log = tmp_path / "blocked.jsonl"
    handler.server = SimpleNamespace(policy=policy, log=proxy.Log(str(log)))
    sock = SimpleNamespace(sendall=lambda data: responses.append(data))
    responses = []
    handler._deny(
        sock,
        "GET",
        f"https://user:{marker}@downloads.example.test/private/{marker}?token={marker}&opaque=other-private-value",
        f"url:downloads.example.test/private/{marker}",
    )
    combined = log.read_text() + b"".join(responses).decode()
    assert marker not in combined
    assert "other-private-value" not in combined
    entry = json.loads(log.read_text())
    assert "downloads.example.test/private/" in entry["url"]
    from benchflow.sandbox.egress_denylist import stop_egress_denylist

    class DownloadEnv(FakeSandbox):
        async def download_file(self, remote, local):
            local.write_bytes(log.read_bytes())  # local is a capability

    await stop_egress_denylist(DownloadEnv(), tmp_path / "trial")
    persisted = (tmp_path / "trial/trajectory/egress_denylist.jsonl").read_text()
    assert marker not in persisted
    assert "other-private-value" not in persisted


def test_dns_and_tls_log_boundaries_scrub_normalized_solution_values(tmp_path):
    """Guards the redaction's DNS/SNI siblings from 4adfb00f, after case folding."""
    from benchflow.sandbox import _egress_denylist_proxy as proxy

    marker = "Synthetic-Oracle-Credential"
    kwargs = (
        {"diagnostic_secrets": [marker]}
        if "diagnostic_secrets" in inspect.signature(proxy.Policy).parameters
        else {}
    )
    policy = proxy.Policy([], ["downloads.example.test"], **kwargs)
    log_path = tmp_path / "blocked.jsonl"
    log = proxy.Log(str(log_path))
    # The base accepts the ordinary Python attribute but ignores it, so the
    # fails-before check is a leaked value, not a missing argument or import.
    log.policy = policy
    for method in ("DNS", "CONNECT"):
        log.write(
            action="blocked",
            method=method,
            url=f"{marker.lower()}.downloads.example.test",
            rule="tls-sni-mismatch",
            server_name=f"{marker}.downloads.example.test",
        )
    persisted = log_path.read_text()
    assert marker.lower() not in persisted.lower()
    assert "downloads.example.test" in persisted


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "encoded",
    [
        "synthetic/demo%2Btoken",
        "%73ynthetic/demo+token",
        "synthetic/%64emo%252Btoken",
        "synthetic%25252Fdemo%2btoken",
        "synthetic/demo" + "%25" + "25" * 12 + "2Btoken",
    ],
)
async def test_mixed_encoded_secrets_never_reach_response_or_download(
    tmp_path, encoded
):
    """Guards FF-VERIFIER-CONFIRM-1 against ed290b1d's finite encoding list."""
    import json
    from urllib.parse import unquote

    from benchflow.sandbox import _egress_denylist_proxy as proxy
    from benchflow.sandbox.egress_denylist import stop_egress_denylist

    marker = "synthetic/demo+token"
    policy = proxy.Policy(
        ["https://downloads.example.test/private"], [], diagnostic_secrets=[marker]
    )
    handler = object.__new__(proxy.Handler)
    log = tmp_path / "blocked.jsonl"
    handler.server = SimpleNamespace(policy=policy, log=proxy.Log(str(log)))
    responses = []
    url = f"https://downloads.example.test/private/{encoded}"
    rule = policy.url_rule("downloads.example.test", f"/private/{encoded}")
    assert rule is not None
    handler._deny(SimpleNamespace(sendall=responses.append), "GET", url, rule)

    class DownloadEnv(FakeSandbox):
        async def download_file(self, remote, local):
            local.write_bytes(log.read_bytes())  # local is a capability

    await stop_egress_denylist(DownloadEnv(), tmp_path / "trial")
    persisted = (tmp_path / "trial/trajectory/egress_denylist.jsonl").read_text()
    json.loads(persisted)
    for diagnostic in (persisted, b"".join(responses).decode()):
        for _ in range(20):
            assert marker.lower() not in diagnostic.lower()
            diagnostic = unquote(diagnostic)
    # Sanitization must not change the policy's input or its block decision.
    assert policy.url_rule("downloads.example.test", f"/private/{encoded}") == rule


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["response", "download"])
@pytest.mark.parametrize(
    ("marker", "encoded"),
    [
        ("synthetic demo token", "synthetic+demo%20token"),
        ("synthetic demo token", "synthetic%20demo+token"),
        ("synthetic demo token", "%73ynthetic%2Bdemo%2520token"),
        ("synthetic demo token", "synthetic%2520demo%2btoken"),
        ("synthetic demo token", "SYNTHETIC+DEMO%20TOKEN"),
        ("synthetic  demo token", "synthetic+%20demo+token"),
        ("synthetic demo token end", "synthetic+demo%20token+end"),
        ("synthetic demo token end", "synthetic%20demo+token%20end"),
        ("synthetic+demo token end", "synthetic%2Bdemo+token%20end"),
        ("synthetic+demo token end", "synthetic+demo%20token+end"),
        ("synthetic.demo [token] end", "synthetic.demo+%5Btoken%5D%20end"),
    ],
)
async def test_mixed_form_spaces_never_reach_response_or_download(
    tmp_path, marker, encoded, boundary
):
    """Guards confirm2 FF-VERIFIER-CONFIRM-1: 8194b01a misses mixed spaces."""
    import json
    from urllib.parse import unquote, unquote_plus

    from benchflow.sandbox import _egress_denylist_proxy as proxy
    from benchflow.sandbox.egress_denylist import stop_egress_denylist

    policy = proxy.Policy(
        ["https://downloads.example.test/private"], [], diagnostic_secrets=[marker]
    )
    handler = object.__new__(proxy.Handler)
    log = tmp_path / "blocked.jsonl"
    handler.server = SimpleNamespace(policy=policy, log=proxy.Log(str(log), policy))
    responses = []
    url = f"https://downloads.example.test/private/{encoded}"
    rule = policy.url_rule("downloads.example.test", f"/private/{encoded}")
    assert rule is not None
    handler._deny(SimpleNamespace(sendall=responses.append), "GET", url, rule)
    # DNS/TLS diagnostics also enter Log directly, without Handler._deny.
    handler.server.log.write(method="CONNECT", evidence={"server_name": [encoded]})

    class DownloadEnv(FakeSandbox):
        async def download_file(self, remote, local):
            local.write_bytes(log.read_bytes())  # local is a capability

    await stop_egress_denylist(DownloadEnv(), tmp_path / "trial")
    persisted = (tmp_path / "trial/trajectory/egress_denylist.jsonl").read_text()
    records = [json.loads(line) for line in persisted.splitlines()]
    diagnostic = (
        b"".join(responses).decode() if boundary == "response" else str(records)
    )
    # Literal '+' credentials must remain covered too: percent decoding keeps
    # them intact, while form decoding checks recoverable space spellings.
    for decode in (unquote, unquote_plus):
        decoded = diagnostic
        for _ in range(20):
            assert marker.lower() not in decoded.lower()
            decoded = decode(decoded)
    assert "downloads.example.test/private/" in diagnostic
    assert policy.url_rule("downloads.example.test", f"/private/{encoded}") == rule


def test_mixed_form_redaction_preserves_literal_plus_and_unrelated_diagnostics():
    """Guards confirm2 FF-VERIFIER-CONFIRM-1 without treating literal '+' as space."""
    from benchflow.sandbox import _egress_denylist_proxy as proxy

    policy = proxy.Policy([], [], diagnostic_secrets=["synthetic+demo token end"])
    for value in ("synthetic demo token end", "unrelated+demo%20token"):
        from urllib.parse import unquote

        assert policy.diagnostic(value) == unquote(value)
