"""``bench hosted`` and :mod:`benchflow.hosted` against a local stub of the hosted service.

The stub speaks BenchHub's hosted-run API (benchhub PR #7, src/benchhub/api/app.py): ``POST /api/v1/runs`` (and
``/estimate``) with ``env_id`` a hub version id, answering ``{"run_id", "warnings", "runtime"}`` or a 400 with a
``detail``; ``GET /api/v1/runs/{id}`` with ``status`` queued, starting, running, then succeeded, failed or cancelled,
``n_traces``, ``n_expected`` and ``mean_reward``; ``/events``, ``/log``, ``/metrics``, ``GET
/api/v1/traces?run_id=`` (``{total, items}``), ``/export?format=atif`` (JSON lines) and ``POST /cancel``, all behind
``Authorization: Bearer``. It records every request, so the tests can say that a mistyped key never reaches the
network and that a 429 or 503 is not retried. No test calls the live service.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from typer.testing import CliRunner

from benchflow import hosted
from benchflow.cli.main import app
from benchflow.hosted import (
    ALT_KEY_ENV,
    HOSTED_URL_ENV,
    KEY_ENV,
    KEY_FILE_ENV,
    SKIP_CHECKSUM_ENV,
    HostedClient,
    HostedError,
    HostedTimeout,
    check_api_key,
    key_checksum,
)

BODY = "0123456789abcdefghijABCDEFGHIJ"
KEY = f"bfk_{BODY}{key_checksum(BODY)}"
BH_KEY = "bh_Ab3dE9xq_s3cr3t-Value_0123456789abcdef"
TRACES = [
    {"id": f"t{i}", "task_name": f"task-{i}", "reward": 1.0, "outcome": "passed"}
    for i in range(5)
]


class Stub:
    """The hosted service's routes, a request log, and errors to answer with."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.statuses = ["queued", "starting", "running", "succeeded"]
        self.fail: tuple[int, dict[str, str], bytes] | None = None
        self.key = KEY
        self.url = ""
        self.page_size = 2
        # Text every reply carries (a run's note, an event, a log line), e.g.
        # an echoed key.
        self.echo = ""
        self.secrets: list[dict[str, Any]] = []

    def answer(self, method: str, path: str, headers: dict[str, str], body: bytes):
        self.requests.append(
            {"method": method, "path": path, "headers": headers, "body": body}
        )
        if self.fail is not None:
            return self.fail
        if headers.get("Authorization") != f"Bearer {self.key}":
            return 401, {}, _json({"detail": "invalid API key"})
        parts = urlsplit(path)
        query = {k: v[0] for k, v in parse_qs(parts.query).items()}
        route = (method, parts.path)
        if route == ("POST", "/api/v1/runs"):
            if json.loads(body)["env_id"] == "PKG-UNKNOWN":
                return 400, {}, _json({"detail": "no hub version PKG-UNKNOWN"})
            return (
                200,
                {},
                _json(
                    {
                        "run_id": "r1",
                        "warnings": [
                            "concurrency lowered to 2: the cap is 4 sandboxes"
                        ],
                        "runtime": {"id": "bf010"},
                    }
                ),
            )
        if route == ("POST", "/api/v1/runs/estimate"):
            return 200, {}, _json({"warnings": [], "budget": {"fits": True}})
        if route == ("GET", "/api/v1/runs/r1"):
            status = (
                self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            )
            record: dict[str, Any] = {
                "id": "r1",
                "note": self.echo,
                "status": status,
                "error": None,
                "n_traces": 0,
                "n_expected": 2,
            }
            if status not in hosted.ACTIVE_STATUSES:
                record.update(n_traces=2, mean_reward=1.0)
            return 200, {}, _json(record)
        if route == ("GET", "/api/v1/runs/r1/events"):
            message = f"Queued {self.echo}".strip()
            return 200, {}, _json([{"id": 1, "kind": "queued", "message": message}])
        if route == ("GET", "/api/v1/runs/r1/log"):
            log = "line one\nline two\n" + (f"{self.echo}\n" if self.echo else "")
            return 200, {"Content-Type": "text/plain"}, log.encode()
        if route == ("GET", "/api/v1/runs/r1/metrics"):
            keys = [{"key": "reward", "n": 5, "mean": 1.0, "sd": 0.0}]
            return 200, {}, _json({"n_traces": 5, "keys": keys, "errors": []})
        if route == ("GET", "/api/v1/traces") and query.get("run_id") == "r1":
            start = int(query.get("offset", 0))
            end = start + min(int(query.get("limit", 50)), self.page_size)
            return 200, {}, _json({"total": len(TRACES), "items": TRACES[start:end]})
        if route == ("GET", "/api/v1/runs/r1/export"):
            fmt = query.get("format", "atif")
            lines = f'{{"format": "{fmt}", "trace": 1}}\n{{"trace": 2}}\n'
            return 200, {"Content-Type": "application/x-ndjson"}, lines.encode()
        if route == ("POST", "/api/v1/runs/r1/cancel"):
            return 200, {}, _json({"ok": True})
        if route == ("GET", "/api/v1/secrets"):
            return 200, {}, _json({"kinds": {}, "items": self.secrets})
        if route == ("POST", "/api/v1/secrets"):
            got = json.loads(body)
            if got["kind"] == "daytona":  # the service quotes what it refused
                return (
                    400,
                    {},
                    _json({"detail": f"an account key can't save {got['value']}"}),
                )
            self.secrets.append(
                {
                    "id": f"sec{len(self.secrets)}",
                    "name": got["name"],
                    "kind": got["kind"],
                    "hint": "…" + got["value"][-4:],
                }
            )
            return 200, {}, _json(self.secrets[-1])
        if route == ("POST", "/api/v1/secrets/endpoint"):
            got = json.loads(body)
            items = [
                {
                    "id": f"sec{len(self.secrets) + i}",
                    "name": f"{got['name']}_{x}",
                    "kind": k,
                    "hint": "…",
                }
                for i, (x, k) in enumerate(
                    (("URL", "provider_base_url"), ("KEY", "provider_api_key"))
                )
            ]
            self.secrets.extend(items)
            return 200, {}, _json({"items": items})
        if method == "DELETE" and parts.path.startswith("/api/v1/secrets/"):
            sid = parts.path.rsplit("/", 1)[1]
            before = len(self.secrets)
            self.secrets = [x for x in self.secrets if x["id"] != sid]
            return 200, {}, _json({"ok": len(self.secrets) < before})
        return 404, {}, _json({"detail": "not found"})


def _json(value: Any) -> bytes:
    return json.dumps(value).encode()


@pytest.fixture
def stub(monkeypatch, tmp_path) -> Iterator[Stub]:
    service = Stub()

    class Handler(BaseHTTPRequestHandler):
        def _serve(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            code, headers, payload = service.answer(
                self.command, self.path, dict(self.headers), body
            )
            self.send_response(code)
            for name, value in {"Content-Type": "application/json", **headers}.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_GET = do_POST = do_DELETE = _serve

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    service.url = f"http://127.0.0.1:{server.server_address[1]}"
    monkeypatch.setenv(HOSTED_URL_ENV, service.url)
    monkeypatch.setenv(KEY_ENV, KEY)
    for name in (ALT_KEY_ENV, KEY_FILE_ENV, SKIP_CHECKSUM_ENV):
        monkeypatch.delenv(name, raising=False)
    # No key file of the real user's; the CLI's wait polls without sleeping.
    monkeypatch.setattr(hosted, "DEFAULT_KEY_FILE", tmp_path / "no-such-key")
    monkeypatch.setattr(hosted, "_sleep", lambda seconds: None)
    try:
        yield service
    finally:
        server.shutdown()
        server.server_close()


def _bench(*args: str):
    return CliRunner().invoke(
        app, ["hosted", *args], prog_name="bench", env={"COLUMNS": "200"}
    )


def _text(result) -> str:
    return " ".join(result.output.split())


# --- the key, offline ---------------------------------------------------------------------------------


def test_the_checksum_is_crc32_in_base62():
    # zlib.crc32(b"0123456789abcdefghijABCDEFGHIJ") == 3469960357 == "3mpbCX" in 0-9A-Za-z.
    assert key_checksum(BODY) == "3mpbCX"
    # crc32 14871316 needs five digits: zero-padded to six.
    assert key_checksum("0" * 29 + "1") == "010Ohw"
    assert check_api_key(KEY) == KEY


@pytest.mark.parametrize("index", [4, 20, 33, 39])
def test_one_mistyped_character_fails_the_checksum(index, monkeypatch):
    monkeypatch.delenv(SKIP_CHECKSUM_ENV, raising=False)
    swapped = "Z" if KEY[index] != "Z" else "Y"
    typo = KEY[:index] + swapped + KEY[index + 1 :]
    with pytest.raises(HostedError, match="checksum does not match") as caught:
        check_api_key(typo)
    assert caught.value.exit_code == 2
    # Named by its first 8 characters only.
    # Named by its kind only: no character of its secret (review).
    assert "the key bfk_… is not valid" in str(caught.value)
    assert typo[4:8] not in str(caught.value) and BODY not in str(caught.value)


def test_the_checksum_can_be_left_to_the_service(monkeypatch):
    typo = KEY[:-1] + ("A" if KEY[-1] != "A" else "B")
    monkeypatch.setenv(SKIP_CHECKSUM_ENV, "1")
    assert check_api_key(typo) == typo


@pytest.mark.parametrize(
    ("key", "words"),
    [
        (None, "is not set"),
        ("  ", "is not set"),
        (KEY[:-1], "wrong length"),
        (KEY + "x", "wrong length"),
        ("sk-ant-" + "x" * 30, "not a BenchFlow Hub or BenchHub key"),
        ("bhw_abc_" + "x" * 30, "not a BenchFlow Hub or BenchHub key"),
    ],
)
def test_keys_that_cannot_be_keys_are_refused(key, words):
    with pytest.raises(HostedError, match=words):
        check_api_key(key)


def test_benchhub_keys_are_sent_unchanged(stub):
    stub.key = BH_KEY
    client = HostedClient.from_env(key=f" {BH_KEY}\n")
    assert client.key == BH_KEY
    client.status("r1")
    assert stub.requests[-1]["headers"]["Authorization"] == f"Bearer {BH_KEY}"


@pytest.mark.parametrize(
    "key", [None, KEY[:-1] + ("A" if KEY[-1] != "A" else "B"), "not-a-key"]
)
def test_a_bad_key_never_reaches_the_service(stub, monkeypatch, key):
    if key is None:
        monkeypatch.delenv(KEY_ENV)
    else:
        monkeypatch.setenv(KEY_ENV, key)
    result = _bench("status", "r1")
    assert result.exit_code == 2, result.output
    assert stub.requests == []
    if key:
        assert key not in result.output
    else:
        assert KEY_ENV in result.output


def test_the_key_comes_from_a_private_file(stub, monkeypatch, tmp_path):
    monkeypatch.delenv(KEY_ENV)
    key_file = tmp_path / "hub-key"
    key_file.write_text(KEY + "\n")
    key_file.chmod(0o600)
    result = _bench("status", "r1", "--key-file", str(key_file))
    assert result.exit_code == 0, result.output
    assert stub.requests[-1]["headers"]["Authorization"] == f"Bearer {KEY}"
    monkeypatch.setenv(KEY_FILE_ENV, str(key_file))
    assert HostedClient.from_env().key == KEY
    # The default file, when nothing else names one.
    monkeypatch.delenv(KEY_FILE_ENV)
    monkeypatch.setattr(hosted, "DEFAULT_KEY_FILE", key_file)
    assert HostedClient.from_env().key == KEY


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o604])
def test_a_key_file_others_can_read_is_refused(stub, monkeypatch, tmp_path, mode):
    monkeypatch.delenv(KEY_ENV)
    key_file = tmp_path / "hub-key"
    key_file.write_text(KEY)
    os.chmod(key_file, mode)
    result = _bench("status", "r1", "--key-file", str(key_file))
    assert result.exit_code == 2
    assert "can be read by others" in _text(result) and "chmod 600" in _text(result)
    assert stub.requests == [] and KEY not in result.output


def test_an_explicit_key_file_wins_over_the_environment(stub, monkeypatch, tmp_path):
    """review: BENCHFLOW_HUB_KEY beat an explicit --key-file."""
    monkeypatch.setenv(KEY_ENV, "bh_other_" + "q" * 30)
    key_file = tmp_path / "hub-key"
    key_file.write_text(KEY + "\n")
    key_file.chmod(0o600)
    result = _bench("status", "r1", "--key-file", str(key_file))
    assert result.exit_code == 0, result.output
    assert stub.requests[-1]["headers"]["Authorization"] == f"Bearer {KEY}"


def test_the_client_masks_what_the_service_sends_back(stub):
    stub.echo = f"seen key {KEY}"
    client = HostedClient.from_env()
    assert KEY not in json.dumps(client.status("r1"))
    assert KEY not in json.dumps(client.events("r1"))
    assert KEY not in client.logs("r1")


def test_the_older_key_variable_is_read_too(stub, monkeypatch):
    monkeypatch.delenv(KEY_ENV)
    monkeypatch.setenv(ALT_KEY_ENV, KEY)
    assert HostedClient.from_env().key == KEY


class _NoNetwork:
    """Stands in for the client's opener: records any request instead."""

    def __init__(self) -> None:
        self.opened: list[str] = []

    def open(self, request, timeout=None):
        self.opened.append(request.full_url)
        raise AssertionError(f"a request was sent to {request.full_url}")


@pytest.mark.parametrize(
    "command",
    [
        ("run", "PKG-GPQA-DIAMOND"),
        ("status", "r1"),
        ("results", "r1"),
        ("cancel", "r1"),
        ("estimate", "PKG-GPQA-DIAMOND"),
    ],
)
def test_no_address_means_no_request(stub, monkeypatch, command):
    """review: there is no default address; with neither --hub nor
    BENCHFLOW_HOSTED_URL every command stops, naming both, and sends nothing."""
    monkeypatch.delenv(HOSTED_URL_ENV)
    network = _NoNetwork()
    monkeypatch.setattr(hosted, "_OPENER", network)
    result = _bench(*command)
    assert result.exit_code == 2, result.output
    assert "--hub" in result.output and HOSTED_URL_ENV in result.output
    assert network.opened == [] and stub.requests == []
    with pytest.raises(HostedError, match="no hosted service named"):
        HostedClient.from_env(key=KEY)


@pytest.mark.parametrize(
    "url", ["http://hosted.example.org", "http://10.0.0.5:8000", "http://[::1]:80"]
)
def test_plain_http_is_refused_off_this_machine(stub, monkeypatch, url):
    monkeypatch.setenv(HOSTED_URL_ENV, url)
    result = _bench("status", "r1")
    assert result.exit_code == 2
    assert "refusing plain http" in " ".join(result.output.split())
    assert stub.requests == []
    for local in ("http://localhost:9", "http://127.0.0.1:9"):
        assert HostedClient.from_env(local, KEY).url == local


def test_the_address_comes_from_the_flag_or_the_environment(monkeypatch):
    monkeypatch.setenv(HOSTED_URL_ENV, "https://env.example.org/")
    assert HostedClient.from_env(key=KEY).url == "https://env.example.org"
    flagged = HostedClient.from_env("https://flag.example.org", KEY)
    assert flagged.url == "https://flag.example.org"
    with pytest.raises(HostedError, match="refusing plain http"):
        HostedClient.from_env("http://hosted.example.org", KEY)


# --- the key is never shown ---------------------------------------------------------------------------


def test_the_client_repr_hides_the_key():
    client = HostedClient.from_env("https://hosted.example.org", KEY)
    assert KEY not in repr(client) and BODY not in repr(client)


def test_an_error_that_echoes_the_key_is_masked(stub):
    stub.fail = (401, {}, _json({"detail": f"key {KEY} is revoked"}))
    with pytest.raises(HostedError) as caught:
        HostedClient.from_env().status("r1")
    text = str(caught.value)
    assert "refused the key bfk_… (401)" in text and "revoked" in text
    assert KEY[4:8] not in text
    assert KEY not in text and BODY not in text
    result = _bench("status", "r1")
    assert result.exit_code == 1
    assert KEY not in result.output and BODY not in result.output
    assert "revoked" in result.output


def test_an_echoed_benchhub_secret_is_masked(stub):
    """review: a bh_<prefix>_<secret> key's secret is masked even when
    the service echoes it alone. The server splits a key with split("_", 2),
    and a token_urlsafe secret often holds "_" itself (this one does)."""
    stub.key = BH_KEY
    secret = BH_KEY.split("_", 2)[2]
    assert "_" in secret
    stub.fail = (401, {}, _json({"detail": f"secret {secret} was revoked"}))
    with pytest.raises(HostedError) as caught:
        HostedClient.from_env(key=BH_KEY).status("r1")
    text = str(caught.value)
    assert "refused the key bh_… (401)" in text and "was revoked" in text
    # No part of the secret either side of its "_".
    for part in secret.split("_"):
        assert part not in text, part


def test_a_key_across_the_message_cut_leaks_nothing(stub):
    """review: the message was cut to 500 characters before masking, so
    a key across the cut printed its start. Another key-shaped token is masked
    too."""
    other = "bfk_" + "Z" * 36
    detail = f"another {other} " + "x" * (490 - len(other) - 9) + KEY
    stub.fail = (401, {}, _json({"detail": detail}))
    with pytest.raises(HostedError) as caught:
        HostedClient.from_env().status("r1")
    text = str(caught.value)
    assert "bfk_0123" not in text and BODY[:6] not in text
    assert "Z" * 8 not in text


def test_no_command_prints_the_key(stub, tmp_path):
    """The stub echoes the key in every run record, event and log: none of it
    reaches the output (review: this test could not fail before)."""
    stub.statuses = ["succeeded"]
    stub.echo = f"seen key {KEY} and secret {BODY}"
    outputs = [
        _bench("run", "PKG-GPQA-DIAMOND", "--wait").output,
        _bench("status", "r1", "--json").output,
        _bench("results", "r1").output,
        _bench("events", "r1").output,
        _bench("logs", "r1").output,
        _bench("trajectories", "r1", "-o", str(tmp_path / "r1.jsonl")).output,
        _bench("cancel", "r1").output,
    ]
    for output in outputs:
        assert KEY not in output and BODY not in output


# --- errors -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "words", "hint"),
    [
        (401, "refused the key bfk_… (401)", "revoked"),
        (403, "refused the key bfk_… (403)", "scope"),
        (404, "nothing at GET /api/v1/runs/r1 (404)", "check the id"),
        (429, "limiting this key's requests (429)", "try again after 30 s"),
        (503, "BenchFlow Hub keys can't be checked right now", "try again after 30 s"),
    ],
)
def test_each_refusal_is_named_once(stub, code, words, hint):
    detail = "BenchFlow Hub keys can't be checked right now" if code == 503 else "no"
    stub.fail = (code, {"Retry-After": "30"}, _json({"detail": detail}))
    with pytest.raises(HostedError) as caught:
        HostedClient.from_env().status("r1")
    assert words in str(caught.value)
    assert hint in (caught.value.hint or "")
    # Never retried by the client: one request, whatever the answer.
    assert len(stub.requests) == 1


def test_a_503_is_never_a_bad_key(stub):
    stub.fail = (503, {"Retry-After": "30"}, _json({"detail": "key service down"}))
    with pytest.raises(HostedError) as caught:
        HostedClient.from_env().submit("PKG-GPQA-DIAMOND")
    assert "refused" not in str(caught.value) and "invalid" not in str(caught.value)
    assert "says nothing about the key" in (caught.value.hint or "")


@pytest.mark.parametrize("code", [429, 503])
def test_a_refused_wait_stops_instead_of_retrying(stub, code):
    stub.fail = (code, {}, _json({"detail": "slow down"}))
    sleeps: list[float] = []
    with pytest.raises(HostedError, match=str(code)):
        HostedClient.from_env().wait("r1", sleep=sleeps.append)
    assert len(stub.requests) == 1 and sleeps == []


def test_the_cli_shows_the_services_messages(stub):
    stub.fail = (
        503,
        {"Retry-After": "30"},
        _json({"detail": "Daytona is unavailable"}),
    )
    result = _bench("run", "PKG-GPQA-DIAMOND")
    assert result.exit_code == 1
    assert "Daytona is unavailable" in _text(result)
    assert "try again after 30 s" in _text(result)
    stub.fail = None
    result = _bench("run", "PKG-UNKNOWN")
    assert result.exit_code == 2
    assert "(400): no hub version PKG-UNKNOWN" in _text(result)


# --- run, estimate and the run's endpoints ------------------------------------------------------------


def test_run_sends_the_contracts_body(stub):
    answer = HostedClient.from_env().submit(
        "PKG-GPQA-DIAMOND",
        max_tasks=2,
        concurrency=2,
        name="gpqa oracle",
        max_sandbox_seconds=1800,
        secrets=["OPENAI_KEY"],
    )
    assert answer["run_id"] == "r1" and answer["runtime"] == {"id": "bf010"}
    request = stub.requests[-1]
    assert (request["method"], request["path"]) == ("POST", "/api/v1/runs")
    assert request["headers"]["Authorization"] == f"Bearer {KEY}"
    assert request["headers"]["Content-Type"] == "application/json"
    assert json.loads(request["body"]) == {
        "env_id": "PKG-GPQA-DIAMOND",
        "agent": "oracle",
        "model": None,
        "max_tasks": 2,
        "tasks": [],
        "n_trials": 1,
        "concurrency": 2,
        "name": "gpqa oracle",
        "extra": {"max_sandbox_seconds": 1800, "secrets": ["OPENAI_KEY"]},
    }


def test_the_cli_run_maps_its_flags(stub):
    result = _bench(
        "run", "PKG-TERMINALBENCH-2-0", "-a", "claude-agent-acp",
        "-m", "claude-sonnet-4-5", "-n", "5", "--include", "a,b", "-i", "c",
        "--trials", "3", "-j", "2",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert json.loads(stub.requests[-1]["body"]) == {
        "env_id": "PKG-TERMINALBENCH-2-0",
        "agent": "claude-agent-acp",
        "model": "claude-sonnet-4-5",
        "max_tasks": 5,
        "tasks": ["a", "b", "c"],
        "n_trials": 3,
        "concurrency": 2,
    }
    text = _text(result)
    assert "queued run r1" in text
    assert f"page: {stub.url}/runs/r1 (sign in to read it)" in text
    assert "concurrency lowered to 2" in text
    assert "bench hosted wait r1" in text


def test_submit_takes_the_version_as_an_option(stub):
    result = _bench("submit", "--version", "PKG-GPQA-DIAMOND", "--n-tasks", "2")
    assert result.exit_code == 0, result.output
    body = json.loads(stub.requests[-1]["body"])
    assert body["env_id"] == "PKG-GPQA-DIAMOND" and body["max_tasks"] == 2


def test_run_waits_for_the_result(stub):
    result = _bench("run", "PKG-GPQA-DIAMOND", "--wait")
    assert result.exit_code == 0, result.output
    text = _text(result)
    assert "r1: status queued, trials 0/2" in text and "r1: status running" in text
    assert "status succeeded, trials 2/2, mean reward 1" in text
    gets = [r for r in stub.requests if r["method"] == "GET"]
    assert len(gets) == 4


def test_estimate_starts_nothing(stub):
    result = _bench("estimate", "PKG-GPQA-DIAMOND", "-n", "2")
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"warnings": [], "budget": {"fits": True}}
    assert [r["path"] for r in stub.requests] == ["/api/v1/runs/estimate"]


def test_a_failed_run_exits_one(stub):
    stub.statuses = ["running", "failed"]
    result = _bench("wait", "r1")
    assert result.exit_code == 1
    assert "run r1 ended failed" in _text(result)


def test_results_read_every_page_of_traces(stub):
    stub.statuses = ["succeeded"]
    results = HostedClient.from_env().results("r1")
    assert results["run"]["status"] == "succeeded"
    assert results["metrics"]["keys"][0]["mean"] == 1.0
    assert [t["task_name"] for t in results["traces"]] == [
        f"task-{i}" for i in range(5)
    ]
    pages = [r["path"] for r in stub.requests if r["path"].startswith("/api/v1/traces")]
    assert len(pages) == 3
    result = _bench("results", "r1")
    assert result.exit_code == 0, result.output
    text = _text(result)
    assert "reward: mean 1 ± 0 over 5" in text
    assert "task-4 reward 1.0 passed" in text


def test_results_wait_for_a_final_status(stub):
    with pytest.raises(HostedError, match="still queued"):
        HostedClient.from_env().results("r1")


def test_trajectories_events_logs_and_cancel(stub, tmp_path):
    client = HostedClient.from_env()
    out = client.trajectories("r1", tmp_path / "out" / "r1.jsonl")
    assert out.read_text().splitlines()[0] == '{"format": "atif", "trace": 1}'
    assert stub.requests[-1]["path"] == "/api/v1/runs/r1/export?format=atif"
    assert not (tmp_path / "out" / "r1.jsonl.part").exists()
    client.trajectories("r1", tmp_path / "m.jsonl", format="messages")
    assert stub.requests[-1]["path"] == "/api/v1/runs/r1/export?format=messages"
    with pytest.raises(HostedError, match="not one of atif, messages"):
        client.trajectories("r1", tmp_path / "x.jsonl", format="csv")
    assert client.events("r1") == [{"id": 1, "kind": "queued", "message": "Queued"}]
    assert client.logs("r1") == "line one\nline two\n"
    assert client.cancel("r1") == {"ok": True}
    assert stub.requests[-1]["method"] == "POST"

    target = tmp_path / "cli.jsonl"
    result = _bench("trajectories", "r1", "--output", str(target))
    assert result.exit_code == 0, result.output
    assert target.read_bytes().count(b"trace") == 2
    other = tmp_path / "d.jsonl"
    result = _bench("download", "r1", "-o", str(other), "--format", "messages")
    assert result.exit_code == 0, result.output
    assert '"format": "messages"' in other.read_text()
    assert "queued: Queued" in _text(_bench("events", "r1"))
    result = _bench("logs", "r1")
    assert result.exit_code == 0 and result.output == "line one\nline two\n"
    assert "asked the service to stop run r1" in _text(_bench("cancel", "r1"))


def test_trajectories_refuse_a_folder(stub, tmp_path):
    with pytest.raises(HostedError, match="is a folder"):
        HostedClient.from_env().trajectories("r1", tmp_path)
    assert stub.requests == []


def test_the_module_functions_read_the_environment(stub, tmp_path):
    assert hosted.submit("PKG-GPQA-DIAMOND")["run_id"] == "r1"
    assert hosted.status("r1")["status"] == "queued"
    assert hosted.download("r1", tmp_path / "t.jsonl").stat().st_size > 0
    assert hosted.cancel("r1") == {"ok": True}


# --- the poll loop ------------------------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def __call__(self) -> float:
        return self.now


def test_wait_polls_with_backoff_until_a_final_status(stub):
    stub.statuses = ["queued"] * 3 + ["starting"] + ["running"] * 6 + ["succeeded"]
    clock = Clock()
    seen: list[str] = []
    record = HostedClient.from_env().wait(
        "r1",
        poll=2,
        max_poll=10,
        on_status=lambda r: seen.append(r["status"]),
        sleep=clock.sleep,
        clock=clock,
    )
    assert record["status"] == "succeeded"
    assert seen == ["queued", "starting", "running", "succeeded"]
    assert clock.sleeps == [2, 3, 4.5, 6.75, 10, 10, 10, 10, 10, 10]


def test_wait_gives_up_at_its_timeout(stub):
    stub.statuses = ["running"]
    clock = Clock()
    with pytest.raises(HostedTimeout, match="still running after 7 s") as caught:
        HostedClient.from_env().wait(
            "r1", timeout=7, poll=2, sleep=clock.sleep, clock=clock
        )
    # The last sleep is cut to the time left; nothing past the timeout.
    assert clock.sleeps == [2, 3, 2]
    assert "bench hosted wait r1" in (caught.value.hint or "")


def test_an_unknown_status_is_final(stub):
    stub.statuses = ["queued", "archived"]
    clock = Clock()
    record = HostedClient.from_env().wait("r1", sleep=clock.sleep, clock=clock)
    assert record["status"] == "archived" and clock.sleeps == [2.0]


def test_the_run_page(stub):
    page = HostedClient.from_env().run_page("r 1/x")
    assert page == f"{stub.url}/runs/r%201%2Fx"


def test_the_group_lists_its_commands():
    root = CliRunner().invoke(app, ["--help"], env={"COLUMNS": "200"}).output
    assert "hosted" in root
    listed = _bench("--help").output
    for command in (
        "run", "submit", "estimate", "status", "wait", "results",
        "trajectories", "download", "events", "logs", "cancel",
    ):  # fmt: skip
        assert command in listed, command
    run_help = _bench("run", "--help").output
    for flag in ("--agent", "--model", "--n-tasks", "--include", "--trials", "--wait"):
        assert flag in run_help, flag
    # The key is never a flag.
    assert "--key " not in run_help and "--api-key" not in run_help


# --- the run's own model keys -------------------------------------------------------------------

DS = "sk-" + "d" * 32
CPA = "cpa-" + "c" * 40


def test_a_run_brings_its_own_keys_from_files_and_variables(
    stub, tmp_path, monkeypatch
):
    keyfile = tmp_path / "ds.key"
    keyfile.write_text(DS + "\n")
    monkeypatch.setenv("MY_CPA_KEY", CPA)
    result = _bench(
        "run", "PKG-GPQA-DIAMOND", "-a", "claude-agent-acp", "-m", "deepseek/deepseek-flash",
        "--model-key", f"deepseek=file:{keyfile}", "--max-cost-usd", "1.5",
        "--endpoint", "https://cpa.example.com", "--endpoint-key", "env:MY_CPA_KEY",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    body = json.loads(stub.requests[-1]["body"])
    assert body["extra"] == {"max_cost_usd": 1.5}
    assert body["secrets_once"] == [
        {"name": "DEEPSEEK", "kind": "deepseek", "value": DS},
        {
            "name": "ENDPOINT_URL",
            "kind": "provider_base_url",
            "value": "https://cpa.example.com",
        },
        {"name": "ENDPOINT_KEY", "kind": "provider_api_key", "value": CPA},
    ]
    assert DS not in result.output and CPA not in result.output
    # an estimate carries no key
    _bench("estimate", "PKG-GPQA-DIAMOND", "-m", "deepseek/deepseek-flash")
    assert "secrets_once" not in json.loads(stub.requests[-1]["body"])


@pytest.mark.parametrize(
    "args, words",
    [
        (["--model-key", "deepseek=" + DS], "file:PATH, env:VAR or -"),
        (["--model-key", "openai=file:/x"], "KIND=SOURCE"),
        (["--model-key", "deepseek=env:NOPE_NOT_SET"], "NOPE_NOT_SET is not set"),
        (["--endpoint", "https://cpa.example.com"], "go together"),
    ],
)
def test_a_bad_key_flag_stops_before_anything_is_sent(stub, args, words, monkeypatch):
    monkeypatch.delenv("NOPE_NOT_SET", raising=False)
    result = _bench("run", "PKG-GPQA-DIAMOND", "-a", "claude-agent-acp", *args)
    assert result.exit_code == 2 and words in _text(result), result.output
    assert DS not in result.output and stub.requests == []


def test_secrets_are_stored_listed_and_deleted_without_their_values_showing(
    stub, tmp_path
):
    result = CliRunner().invoke(
        app,
        ["hosted", "secret", "set", "DS", "--kind", "deepseek"],
        input=DS + "\n",
        prog_name="bench",
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 0, result.output
    assert json.loads(stub.requests[-1]["body"]) == {
        "name": "DS",
        "kind": "deepseek",
        "value": DS,
    }
    assert DS not in result.output and "--secret DS" in _text(result)
    keyfile = tmp_path / "cpa.key"
    keyfile.write_text(CPA)
    result = _bench(
        "secret",
        "endpoint",
        "CPA",
        "--url",
        "https://cpa.example.com",
        "--from",
        f"file:{keyfile}",
    )
    assert result.exit_code == 0, result.output
    assert json.loads(stub.requests[-1]["body"]) == {
        "name": "CPA",
        "url": "https://cpa.example.com",
        "key": CPA,
    }
    assert (
        "--secret CPA_URL --secret CPA_KEY" in _text(result)
        and CPA not in result.output
    )
    listed = _bench("secret", "list", "--json")
    assert [x["name"] for x in json.loads(listed.output)] == [
        "DS",
        "CPA_URL",
        "CPA_KEY",
    ]
    assert _bench("secret", "rm", "DS").exit_code == 0
    assert _bench("secret", "rm", "DS").exit_code == 1
    # a refusal that quotes the value is masked
    refused = CliRunner().invoke(
        app,
        ["hosted", "secret", "set", "D", "--kind", "daytona"],
        input=DS,
        prog_name="bench",
        env={"COLUMNS": "200"},
    )
    assert (
        refused.exit_code == 2 and DS not in refused.output and "***" in refused.output
    )
