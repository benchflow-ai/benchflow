"""``bench doctor`` and the remote Docker host (``--sandbox remote-docker``).

The remote host gets its own line, checked when it is the selected sandbox or
``BENCHFLOW_REMOTE_DOCKER_HOST`` is set: reachable, Docker version, CPUs and
memory, named without its ssh user (often an access token), and no free-disk
figure read from this machine. The ssh user of the user's own
``DOCKER_HOST`` stays out of the local Docker line too.
"""

from __future__ import annotations

import dataclasses
import json
import os

import pytest

from benchflow import doctor
from benchflow.sandbox.protocol import SandboxStartupError
from benchflow.sandbox.remote_docker import REMOTE_DOCKER_HOST_ENV, HostCapacity
from tests import _fake_docker_cli
from tests.test_doctor import CLAUDE_TOKEN, all_output, by_id, make_probes
from tests.test_remote_docker_local_daemon import (
    DOCKER_29_SAYS,
    SSH_URL,
    TOKEN,
    TOKEN_B,
    _ssh_refused,
)


@pytest.fixture(autouse=True)
def _clean_docker_env(monkeypatch):
    for name in (
        REMOTE_DOCKER_HOST_ENV,
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
        "DOCKER_TLS_VERIFY",
        "DOCKER_CERT_PATH",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    return _fake_docker_cli.install(tmp_path, monkeypatch)


def _remote_probes(tmp_path, *, env=None, info=None, **kwargs):
    probes = make_probes(
        tmp_path,
        env={
            "CLAUDE_CODE_OAUTH_TOKEN": CLAUDE_TOKEN,
            REMOTE_DOCKER_HOST_ENV: SSH_URL,
            **(env or {}),
        },
        **kwargs,
    )
    seen = []

    def remote_docker_info(host, environ, timeout):
        seen.append(host)
        if isinstance(info, Exception):
            raise info
        return info or HostCapacity(cpus=16, memory_mb=64298, server_version="29.1.3")

    return dataclasses.replace(probes, remote_docker_info=remote_docker_info), seen


def test_doctor_reports_the_remote_host_redacted(tmp_path):
    probes, seen = _remote_probes(tmp_path)
    report = doctor.run_doctor(sandbox="remote-docker", probes=probes, offline=True)
    row = by_id(report)["remote-docker"]
    assert row.status == "pass"
    assert row.summary == ("Docker 29.1.3 on remote-docker[ssh] (16 CPU, 62.8 GiB)")
    assert row.details == {
        "host": "remote-docker[ssh]",
        "server_version": "29.1.3",
        "cpus": 16,
        "memory_bytes": 64298 * 1024 * 1024,
    }
    assert [h.url for h in seen] == [SSH_URL]
    assert TOKEN not in all_output(report)
    # The model proxy is installed in the remote sandbox, not run here.
    proxy = by_id(report)["proxy.litellm"]
    assert proxy.status == "pass"
    assert "inside each sandbox" in proxy.summary


def test_doctor_names_an_unreachable_remote_host_without_its_user(
    tmp_path, monkeypatch, fake
):
    # The real probe, through the fake docker CLI.
    fake.set(fail={"info": _ssh_refused(TOKEN)})
    probes = make_probes(
        tmp_path,
        env={
            "CLAUDE_CODE_OAUTH_TOKEN": CLAUDE_TOKEN,
            REMOTE_DOCKER_HOST_ENV: SSH_URL,
            "PATH": str(fake.root),
        },
    )
    report = doctor.run_doctor(sandbox="remote-docker", probes=probes, offline=True)
    row = by_id(report)["remote-docker"]
    assert row.status == "fail"
    assert row.summary.startswith("unreachable: remote-docker[ssh]")
    assert "builder.example.test" not in all_output(report)
    assert "Permission denied (publickey)" in row.summary
    assert TOKEN not in all_output(report)
    assert not report.ok


def test_doctor_checks_the_remote_host_only_when_asked_or_set(tmp_path):
    plain = make_probes(tmp_path, env={"CLAUDE_CODE_OAUTH_TOKEN": CLAUDE_TOKEN})
    assert "remote-docker" not in by_id(doctor.run_doctor(probes=plain, offline=True))
    missing = doctor.run_doctor(sandbox="remote-docker", probes=plain, offline=True)
    row = by_id(missing)["remote-docker"]
    assert row.status == "fail"
    assert REMOTE_DOCKER_HOST_ENV in row.summary
    # Set but not the selected sandbox: a warning at most, never a failure.
    probes, _ = _remote_probes(tmp_path, info=SandboxStartupError("x: y"))
    report = doctor.run_doctor(probes=probes, offline=True)
    assert by_id(report)["remote-docker"].status == "warn"


def test_doctor_hides_the_ssh_user_of_the_users_own_docker_host(tmp_path):
    probes = make_probes(
        tmp_path,
        env={
            "CLAUDE_CODE_OAUTH_TOKEN": CLAUDE_TOKEN,
            "DOCKER_HOST": f"ssh://{TOKEN}@10.0.0.5",
        },
        commands={
            ("docker", "info", "--format", "{{json .}}"): doctor.CommandResult(
                1, "", f"{TOKEN}@10.0.0.5: Permission denied (publickey)."
            )
        },
    )
    report = doctor.run_doctor(probes=probes, offline=True)
    assert "Permission denied" in by_id(report)["docker"].summary
    assert TOKEN not in all_output(report)


@pytest.mark.parametrize(
    "url",
    [
        f"ssh://{TOKEN}@10.0.0.5:abc",
        f"ssh://{TOKEN}@10.0.0.5:99999",
        f"ssh://{TOKEN}@[::1",
    ],
)
def test_doctor_reports_a_mistyped_ssh_docker_host_as_the_docker_row(
    tmp_path, monkeypatch, url
):
    # RD-R1-B-1: with an ssh binary on PATH, redacting docker's error read the
    # address's port and brackets through urlsplit, which raised ValueError
    # out of check_docker, so plain `bench doctor` printed no report.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh = bin_dir / "ssh"
    ssh.write_text("#!/bin/sh\nexit 255\n")
    ssh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ.get('PATH', '')}")
    probes = make_probes(
        tmp_path,
        env={"CLAUDE_CODE_OAUTH_TOKEN": CLAUDE_TOKEN, "DOCKER_HOST": url},
        commands={
            ("docker", "info", "--format", "{{json .}}"): doctor.CommandResult(
                1, "", f'invalid docker host "{url}": {TOKEN}@ is not reachable'
            )
        },
    )
    report = doctor.run_doctor(probes=probes, offline=True)
    assert by_id(report)["docker"].status != "pass"
    assert TOKEN not in all_output(report)


@pytest.mark.parametrize("url", list(DOCKER_29_SAYS))
def test_doctor_hides_a_user_docker_splits_or_escapes_in_the_users_docker_host(
    tmp_path, url
):
    # docker's real messages: with an unencoded /, ? or # it shows one or both
    # halves of the user; with a tab it quotes the address escaped.
    probes = make_probes(
        tmp_path,
        env={"CLAUDE_CODE_OAUTH_TOKEN": CLAUDE_TOKEN, "DOCKER_HOST": url},
        commands={
            ("docker", "info", "--format", "{{json .}}"): doctor.CommandResult(
                1, "", DOCKER_29_SAYS[url]
            )
        },
    )
    report = doctor.run_doctor(probes=probes, offline=True)
    assert "percent-encode its ssh user" in by_id(report)["docker"].summary
    assert TOKEN not in all_output(report)
    assert TOKEN_B not in all_output(report)


@pytest.mark.parametrize(
    "value",
    [
        "ssh://user@[::1",
        f"ssh://{TOKEN}@[gpu-box:22",
        f"ssh://{TOKEN}@[gpu-box]:22",
        f"ssh://{TOKEN}\uff03{TOKEN_B}@box.example",
        f"ssh://{TOKEN}\t{TOKEN_B}@gpu-box:22",
    ],
)
def test_doctor_reports_an_address_docker_cannot_read_as_a_row(tmp_path, value):
    # urlsplit raised a plain ValueError for the first four (doctor crashed
    # with "This is a bug"); it dropped the tab in the last, so the address
    # passed and docker's error quoted it whole.
    probes, seen = _remote_probes(tmp_path, env={REMOTE_DOCKER_HOST_ENV: value})
    report = doctor.run_doctor(probes=probes, offline=True)
    assert by_id(report)["remote-docker"].status == "warn"
    selected = doctor.run_doctor(sandbox="remote-docker", probes=probes, offline=True)
    row = by_id(selected)["remote-docker"]
    assert row.status == "fail"
    assert seen == []
    for out in (all_output(report), all_output(selected)):
        assert TOKEN not in out and TOKEN_B not in out


def test_bench_doctor_json_reports_a_broken_ipv6_host(tmp_path, monkeypatch, fake):
    from typer.testing import CliRunner

    from benchflow.cli.main import app

    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, f"ssh://{TOKEN}@[::1")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app, ["doctor", "--sandbox", "remote-docker", "--offline", "--json"]
    )
    data = json.loads(result.stdout)
    (row,) = [c for c in data["checks"] if c["id"] == "remote-docker"]
    assert row["status"] == "fail"
    assert "IPv6" in row["summary"]
    assert TOKEN not in result.output


def test_bench_doctor_json_reports_the_remote_host(tmp_path, monkeypatch, fake):
    from typer.testing import CliRunner

    from benchflow.cli.main import app

    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    fake.set(info={"NCPU": 16, "MemTotal": 64 * 1024**3, "ServerVersion": "29.1.3"})
    result = CliRunner().invoke(
        app, ["doctor", "--sandbox", "remote-docker", "--offline", "--json"]
    )
    data = json.loads(result.stdout)
    (row,) = [c for c in data["checks"] if c["id"] == "remote-docker"]
    assert row["status"] == "pass"
    assert row["summary"] == ("Docker 29.1.3 on remote-docker[ssh] (16 CPU, 64.0 GiB)")
    assert TOKEN not in result.output
    assert "builder.example.test" not in result.output
