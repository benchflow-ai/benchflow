"""The BenchFlow Hub client: reproducible task archives, digest-checked pulls, controls, publish and push.

``-d org/name@version`` resolves from a Hub
(``--registry hub:<url>`` or ``$BENCHFLOW_HUB_URL``) with every archive and task digest checked, ``bench hub show``
prints the version's oracle and no-op controls, and ``bench hub publish``/``push`` upload. A fake Hub serves the
registry v1 wire format from a thread (design note: hub-registry-v1).
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import tarfile
import threading
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchflow._utils import dataset_registry
from benchflow._utils.dataset_registry import DatasetResolutionError, resolve_dataset
from benchflow._utils.task_authoring import task_digest
from benchflow.cli.main import app
from benchflow.hub.client import HubClient, HubError, normalize_hub_url
from benchflow.hub.datasets import (
    build_manifest,
    dataset_digest,
    pack_task,
    unpack_task,
)


def _task(root: Path, name: str, body: str = "Do it.") -> Path:
    d = root / name
    (d / "environment").mkdir(parents=True)
    (d / "task.md").write_text(
        f"---\nschema_version: '1.3'\ntask:\n  name: {name}\n---\n\n{body}\n"
    )
    (d / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    solve = d / "oracle" / "solve.sh"
    solve.parent.mkdir()
    solve.write_text("#!/bin/bash\necho solved\n")
    solve.chmod(0o755)
    return d


class FakeHub:
    """Registry v1 from memory: datasets published through POST /api/v1/registry, blobs, controls, run uploads."""

    def __init__(self) -> None:
        self.datasets: dict[tuple[str, str], dict] = {}
        self.blobs: dict[str, bytes] = {}
        self.controls: dict[tuple[str, str], dict] = {}
        self.uploads: list[dict] = []
        self.tamper: dict[str, bytes] = {}
        self.refuse: set[str] = (
            set()
        )  # blobs the Hub refuses to serve (its stored bytes changed)
        hub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def _send(
                self, code: int, payload, ctype: str = "application/json"
            ) -> None:
                body = (
                    payload
                    if isinstance(payload, bytes)
                    else json.dumps(payload).encode()
                )
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                m = re.fullmatch(
                    r"/api/v1/registry/([^/]+)/([^/]+)/([^/]+)(/controls)?", self.path
                )
                if m:
                    key = (f"{m[1]}/{m[2]}", m[3])
                    if key not in hub.datasets:
                        return self._send(404, {"detail": "no such dataset version"})
                    if m[4]:
                        return self._send(
                            200,
                            hub.controls.get(
                                key,
                                {
                                    "dataset": f"{key[0]}@{key[1]}",
                                    "controls": [],
                                    "validated": False,
                                },
                            ),
                        )
                    return self._send(200, hub.datasets[key])
                m = re.fullmatch(r"/api/v1/blobs/([0-9a-f]{64})", self.path)
                if m and m[1] in hub.refuse:
                    return self._send(
                        500,
                        {
                            "detail": "this archive's stored bytes no longer hash to its sha256"
                        },
                    )
                if m and m[1] in hub.blobs:
                    return self._send(
                        200, hub.tamper.get(m[1], hub.blobs[m[1]]), "application/gzip"
                    )
                self._send(404, {"detail": "not found"})

            def do_POST(self) -> None:
                length = int(self.headers["Content-Length"])
                raw = self.rfile.read(length)
                msg = BytesParser(policy=HTTP).parsebytes(
                    f"Content-Type: {self.headers['Content-Type']}\r\n\r\n".encode()
                    + raw
                )
                fields, files = {}, []
                for part in msg.iter_parts():
                    name = part.get_param("name", header="content-disposition")
                    if part.get_filename():
                        files.append(
                            (name, part.get_filename(), part.get_payload(decode=True))
                        )
                    else:
                        fields[name] = part.get_content()
                if self.path == "/api/v1/registry":
                    manifest = json.loads(fields["manifest"])
                    key = (manifest["name"], manifest["version"])
                    if key in hub.datasets:
                        same = hub.datasets[key]["digest"] == manifest["digest"]
                        return self._send(
                            200 if same else 409,
                            hub.datasets[key]
                            if same
                            else {"detail": "versions are immutable"},
                        )
                    blobs = {fn.removesuffix(".tar.gz"): data for _, fn, data in files}
                    for t in manifest["tasks"]:
                        data = blobs[t["name"]]
                        sha = hashlib.sha256(data).hexdigest()
                        if sha != t["archive"]["sha256"]:
                            return self._send(
                                400, {"detail": "archive sha256 mismatch"}
                            )
                        hub.blobs[sha] = data
                        t["archive"]["url"] = f"/api/v1/blobs/{sha}"
                    hub.datasets[key] = {
                        **manifest,
                        "published_at": "2026-11-04T12:00:00Z",
                        "published_by": "Test",
                    }
                    return self._send(
                        201, hub.datasets[key]
                    )  # the real Hub answers the document
                if self.path == "/api/v1/runs/upload":
                    hub.uploads.append({"fields": fields, "files": files})
                    run = {"run_id": "run_test1", "n_traces": 2}
                    if fields.get(
                        "dataset"
                    ):  # the real Hub says whether the run's records tie it to the version
                        run |= {
                            "dataset": fields["dataset"],
                            "dataset_content": "unchecked",
                        }
                    return self._send(200, {"imported": [run]})
                self._send(404, {"detail": "not found"})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def publish(self, tasks_dir: Path, name: str, version: str, scratch: Path) -> dict:
        manifest, archives = build_manifest(
            tasks_dir, name, version, scratch, kind="embodied"
        )
        HubClient(self.url).publish(manifest, archives)
        return self.datasets[(name, version)]


@pytest.fixture
def hub(tmp_path, monkeypatch):
    monkeypatch.chdir(
        tmp_path
    )  # the pull cache is .cache/datasets-hub under the working folder
    monkeypatch.delenv("BENCHFLOW_HUB_URL", raising=False)
    monkeypatch.delenv("BENCHFLOW_HUB_TOKEN", raising=False)
    fake = FakeHub()
    yield fake
    fake.server.shutdown()


def test_task_archives_are_reproducible_and_plain(tmp_path) -> None:
    task = _task(tmp_path / "src", "reach")
    a, b = tmp_path / "a.tar.gz", tmp_path / "b.tar.gz"
    sha_a, _ = pack_task(task, a)
    os.utime(task / "task.md", (1, 1))  # a new mtime packs to the same bytes
    sha_b, _ = pack_task(task, b)
    assert sha_a == sha_b and a.read_bytes() == b.read_bytes()
    with tarfile.open(a) as tar:
        modes = {m.name: m.mode for m in tar.getmembers()}
    assert modes["oracle/solve.sh"] == 0o755 and modes["task.md"] == 0o644
    (task / "link").symlink_to("task.md")
    with pytest.raises(HubError, match="is a link"):
        pack_task(task, tmp_path / "c.tar.gz")


def test_unpacking_refuses_paths_out_of_the_task(tmp_path) -> None:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 2
        tar.addfile(info, io.BytesIO(b"hi"))
    evil = tmp_path / "evil.tar.gz"
    evil.write_bytes(gzip.compress(raw.getvalue()))
    (tmp_path / "out").mkdir()
    with pytest.raises(HubError, match="refusing entry"):
        unpack_task(evil, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_dataset_digest_is_the_sorted_task_lines() -> None:
    lines = "a sha256:1\nb sha256:2\n"
    assert (
        dataset_digest({"b": "sha256:2", "a": "sha256:1"})
        == "sha256:" + hashlib.sha256(lines.encode()).hexdigest()
    )


def test_a_dataset_resolves_from_the_hub_with_every_digest_checked(
    tmp_path, hub
) -> None:
    src = tmp_path / "src"
    for name in ("reach", "push"):
        _task(src, name)
    hub.publish(src, "robouse/sample", "0.6.1", tmp_path / "scratch")
    resolved = resolve_dataset("robouse/sample@0.6.1", registry=f"hub:{hub.url}")
    assert resolved.task_names == {"reach", "push"}
    assert resolved.registry == f"hub:{hub.url}"
    for name in resolved.task_names:
        assert (
            task_digest(resolved.tasks_dir / name)
            == resolved.task_digests[name]
            == task_digest(src / name)
        )
    assert resolved.provenance["type"] == "benchflow_hub"
    assert (
        resolved.provenance["resolved_sha"]
        == hub.datasets[("robouse/sample", "0.6.1")]["digest"]
    )
    # a cached task edited on disk is pulled again, not run as the Hub's
    (resolved.tasks_dir / "reach" / "task.md").write_text("changed")
    again = resolve_dataset("robouse/sample@0.6.1", registry=f"hub:{hub.url}")
    assert task_digest(again.tasks_dir / "reach") == task_digest(src / "reach")


def test_a_tampered_archive_or_task_is_refused(tmp_path, hub) -> None:
    src = tmp_path / "src"
    _task(src, "reach")
    doc = hub.publish(src, "robouse/sample", "1", tmp_path / "scratch")
    sha = doc["tasks"][0]["archive"]["sha256"]
    blob = bytearray(hub.blobs[sha])
    blob[len(blob) // 2] ^= 0xFF  # one flipped byte, same size
    hub.tamper[sha] = bytes(blob)
    with pytest.raises(DatasetResolutionError, match="does not match the manifest"):
        resolve_dataset("robouse/sample@1", registry=f"hub:{hub.url}")
    hub.tamper.clear()
    doc["tasks"][0]["digest"] = "sha256:" + "0" * 64
    doc["digest"] = dataset_digest({"reach": doc["tasks"][0]["digest"]})
    with pytest.raises(DatasetResolutionError, match="the manifest pins"):
        resolve_dataset("robouse/sample@1", registry=f"hub:{hub.url}")
    doc["digest"] = "sha256:" + "1" * 64
    with pytest.raises(DatasetResolutionError, match="does not match its tasks"):
        resolve_dataset("robouse/sample@1", registry=f"hub:{hub.url}")
    # the Hub itself refuses a blob whose bytes changed on its disk; the next step is a command that exists
    doc["digest"] = dataset_digest({"reach": doc["tasks"][0]["digest"]})
    hub.refuse.add(sha)
    with pytest.raises(DatasetResolutionError, match="no longer hash") as refused:
        resolve_dataset("robouse/sample@1", registry=f"hub:{hub.url}")
    assert refused.value.hint == (
        "the Hub's stored copy is damaged, so tell its admin; "
        f"`bench hub show robouse/sample@1 --hub {hub.url}` shows what the Hub holds"
    )


def test_benchflow_hub_url_serves_what_benchflows_registry_lacks(
    tmp_path, hub, monkeypatch
) -> None:
    empty = tmp_path / "empty-registry.json"
    empty.write_text("[]")
    monkeypatch.setattr(dataset_registry, "DEFAULT_REGISTRY_SOURCES", (str(empty),))
    _task(tmp_path / "src", "reach")
    hub.publish(tmp_path / "src", "robouse/sample", "1", tmp_path / "scratch")
    with pytest.raises(
        DatasetResolutionError, match="not found in BenchFlow's registry"
    ):
        resolve_dataset("robouse/sample@1")
    monkeypatch.setenv("BENCHFLOW_HUB_URL", hub.url)
    assert resolve_dataset("robouse/sample@1").task_names == {"reach"}
    assert (
        dataset_registry.dataset_registry_entry("robouse/sample@1")["tasks"][0]["name"]
        == "reach"
    )


def test_plain_http_is_refused_for_a_hub_off_this_machine() -> None:
    assert normalize_hub_url("http://127.0.0.1:8870/") == "http://127.0.0.1:8870"
    with pytest.raises(HubError, match="plain http"):
        normalize_hub_url("http://hub.example.org")
    assert normalize_hub_url("https://hub.example.org") == "https://hub.example.org"


def test_show_prints_the_controls_and_each_tasks_result(tmp_path, hub) -> None:
    for name in ("reach", "push"):
        _task(tmp_path / "src", name)
    hub.publish(tmp_path / "src", "robouse/sample", "1", tmp_path / "scratch")
    hub.controls[("robouse/sample", "1")] = {
        "dataset": "robouse/sample@1",
        "validated": False,
        "controls": [
            {
                "agent": "oracle",
                "source": "uploaded",
                "content_checked": False,
                "run_id": "run_o",
                "bench_version": "0.10.0rc1",
                "sandbox": "docker",
                "finished_at": "2026-11-04T10:00:00Z",
                "tasks": 2,
                "as_expected": 1,
                "unexpected": ["push"],
                "errored_tasks": 0,
                "ok": False,
                "per_task": [
                    {"task": "reach", "reward": 1.0},
                    {"task": "push", "reward": 0.0},
                ],
            },
        ],
    }
    result = CliRunner().invoke(
        app, ["hub", "show", "robouse/sample@1", "--hub", hub.url]
    )
    assert result.exit_code == 0, result.output
    out = " ".join(result.output.split())
    assert "not validated" in out
    assert "oracle 1/2 scored 1" in out and "not as expected: push" in out
    assert "reported by the publisher" in out
    assert "no-op not run on this version" in out
    assert "whether a task passes with no work is untested" in out
    assert "content not checked" in out
    # a run on content not checked shows what it reported, never as a pass or a fail of this version
    assert "1.0*" in out and "0.0*" in out
    assert (
        "* content not checked: what that run reported; its records don't name this version's digest"
        in out
    )
    # one term for it: "unchecked" and "not tied" are gone
    assert "unchecked" not in out and "not tied" not in out
    # each command is printed on its own line, unwrapped, so it can be pasted as it is
    lines = result.output.splitlines()
    assert f"    bench run -d robouse/sample@1 --hub {hub.url} --agent nop" in lines
    assert f"    bench hub push <job> --hub {hub.url}" in lines
    # one flag names the Hub in every command it prints
    assert "--registry" not in out


def test_show_reads_benchflow_hub_url_only_for_org_name_datasets(
    hub, monkeypatch
) -> None:
    import benchflow.cli._hub_client as hub_client
    import benchflow.cli.hub as hub_cli

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        hub_client, "show", lambda spec, hub, **_: calls.append(("hub", spec))
    )
    monkeypatch.setattr(
        hub_cli,
        "hosted_env_show",
        lambda source_env, **_: calls.append(("registry", source_env)),
    )
    monkeypatch.setenv("BENCHFLOW_HUB_URL", hub.url)
    for spec in ("robouse/sample@1", "terminal-bench@2.0"):
        result = CliRunner().invoke(app, ["hub", "show", spec])
        assert result.exit_code == 0, result.output
    result = CliRunner().invoke(
        app, ["hub", "show", "terminal-bench@2.0", "--hub", hub.url]
    )
    assert result.exit_code == 0, result.output
    # as `bench run -d`: the env var serves org/name datasets; --hub reads that Hub for any name
    assert calls == [
        ("hub", "robouse/sample@1"),
        ("registry", "terminal-bench@2.0"),
        ("hub", "terminal-bench@2.0"),
    ]


def test_publish_then_push_a_job(tmp_path, hub) -> None:
    _task(tmp_path / "src", "reach")
    result = CliRunner().invoke(
        app,
        [
            "hub",
            "publish",
            str(tmp_path / "src"),
            "--name",
            "robouse/sample@2",
            "--hub",
            hub.url,
            "--kind",
            "embodied",
        ],
    )
    assert result.exit_code == 0, result.output
    assert ("robouse/sample", "2") in hub.datasets
    again = CliRunner().invoke(
        app,
        [
            "hub",
            "publish",
            str(tmp_path / "src"),
            "--name",
            "robouse/sample@2",
            "--hub",
            hub.url,
        ],
    )
    assert again.exit_code == 0 and "Already published, unchanged" in again.output
    (tmp_path / "src" / "reach" / "task.md").write_text(
        "---\ntask:\n  name: reach\n---\nchanged\n"
    )
    changed = CliRunner().invoke(
        app,
        [
            "hub",
            "publish",
            str(tmp_path / "src"),
            "--name",
            "robouse/sample@2",
            "--hub",
            hub.url,
        ],
    )
    assert changed.exit_code == 1, changed.output
    # the next step is the command, not the error again in other words
    assert (
        f"next: bench hub publish {tmp_path / 'src'} --name robouse/sample@<new version> --hub {hub.url}"
        in " ".join(changed.output.split())
    )

    job = tmp_path / "jobs" / "2026-11-04__12-00-00"
    trial = job / "reach__abc"
    (trial / "verifier").mkdir(parents=True)
    (job / "summary.json").write_text("{}")
    (trial / "config.json").write_text(
        json.dumps({"dataset_name": "robouse/sample", "dataset_version": "2"})
    )
    (trial / "verifier" / "reward.txt").write_text("1")
    (trial / ".env").write_text("SECRET=1")
    pushed = CliRunner().invoke(
        app, ["hub", "push", str(job), "--hub", hub.url, "--label", "first try"]
    )
    assert pushed.exit_code == 0, pushed.output
    out = " ".join(pushed.output.split())
    assert "Pushed first try to the Hub" in out and "run run_test1" in out
    assert "linked to robouse/sample@2, content not checked" in out
    assert hub.uploads[-1]["fields"]["name"] == "first try"
    upload = hub.uploads[-1]
    assert upload["fields"]["dataset"] == "robouse/sample@2"
    with tarfile.open(fileobj=io.BytesIO(upload["files"][0][2])) as tar:
        names = tar.getnames()
    assert f"{job.name}/reach__abc/verifier/reward.txt" in names
    assert not any(n.endswith("/.env") for n in names)

    # bench run writes jobs/<job>/<time>/<trial>: the job folder pushes its one run; with two it asks which
    (tmp_path / "jobs" / "summary.json").write_text("{}")
    pushed = CliRunner().invoke(
        app, ["hub", "push", str(tmp_path / "jobs"), "--hub", hub.url]
    )
    assert pushed.exit_code == 0, pushed.output
    with tarfile.open(fileobj=io.BytesIO(hub.uploads[-1]["files"][0][2])) as tar:
        assert f"{job.name}/reach__abc/verifier/reward.txt" in tar.getnames()
    second = tmp_path / "jobs" / "2026-11-04__13-00-00"
    second.mkdir()
    (second / "summary.json").write_text("{}")
    two = CliRunner().invoke(
        app, ["hub", "push", str(tmp_path / "jobs"), "--hub", hub.url]
    )
    assert two.exit_code == 2 and "holds 2 runs" in two.output
    assert "bench hub push" in two.output and second.name in two.output


def _control(agent: str, **kw) -> dict:
    return {
        "agent": agent,
        "source": "uploaded",
        "content_checked": True,
        "run_id": f"run_{agent}",
        "bench_version": "0.10.0rc1",
        "sandbox": "docker",
        "finished_at": "2026-11-04T10:00:00Z",
        "tasks": 2,
        "as_expected": 2,
        "unexpected": [],
        "errored_tasks": 0,
        "ok": True,
        "runs": 1,
        "disagreeing_runs": 0,
        "tries_per_task": {"min": 1, "max": 1},
        "per_task": [
            {"task": "reach", "reward": 1.0 if agent == "oracle" else 0.0},
            {"task": "push", "reward": 1.0 if agent == "oracle" else 0.0},
        ],
        **kw,
    }


def test_show_says_who_validated_and_what_to_do_next(tmp_path, hub) -> None:
    """The status names who ran the controls; a failed control asks for a new
    version, not a rerun; a hidden run and earlier runs that disagree are named."""
    for name in ("reach", "push"):
        _task(tmp_path / "src", name)
    hub.publish(tmp_path / "src", "robouse/sample", "1", tmp_path / "scratch")
    key = ("robouse/sample", "1")

    def show() -> tuple[str, list[str]]:
        result = CliRunner().invoke(
            app, ["hub", "show", "robouse/sample@1", "--hub", hub.url]
        )
        assert result.exit_code == 0, result.output
        return " ".join(result.output.split()), result.output.splitlines()

    hub.controls[key] = {
        "validated": True,
        "controls": [_control("oracle"), _control("nop")],
    }
    out, _ = show()
    assert (
        "validated: both controls behaved on every task (reported by the publisher, not re-run by the Hub)"
        in out
    )
    assert (
        "1 try per task" in out and "bench run -d" in out
    )  # validated: only the command to run it
    assert "robouse/sample@1: 2 embodied tasks on the Hub at" in out
    assert "bench 0.10.0rc1 on docker" in out and "·" not in out
    assert "finished 2026-11-04" in out  # the date says what it is

    hub.controls[key] = {
        "validated": False,
        "controls": [
            _control(
                "oracle",
                ok=False,
                as_expected=1,
                unexpected=["push"],
                runs=3,
                disagreeing_runs=1,
                run_id=None,
            ),
            _control("nop"),
        ],
    }
    out, lines = show()
    assert "not validated: a control did not behave" in out
    assert "not as expected: push" in out
    assert "1 other run of it on this version came out the other way" in out
    assert "the run is hidden from your workspace, latest of 3 runs" in out
    assert "fix push, then publish the fixed tasks as a new version" in out
    assert (
        f"    bench hub publish <tasks-dir> --name robouse/sample@<new version> --hub {hub.url}"
        in lines
    )
    assert not any(line.strip().startswith("bench run -d") for line in lines)

    # a control whose runs on the version came out both ways validates nothing, and the status says so
    hub.controls[key] = {
        "validated": False,
        "controls": [_control("oracle", runs=2, disagreeing_runs=1), _control("nop")],
    }
    out, _ = show()
    assert "not validated: a control's runs on this version came out both ways" in out

    # a control run on content the Hub couldn't match to this version says so, in the same words as its line
    hub.controls[key] = {
        "validated": False,
        "controls": [_control("oracle", content_checked=False), _control("nop")],
    }
    out, _ = show()
    assert "not validated: content not checked for the oracle's run" in out
    # source and license are what the publisher declared
    assert "(as the publisher declared)" in out or "source not given" in out


def test_a_control_run_is_told_to_push_itself_not_to_run_again(hub, capsys):
    """The notice before an oracle run told it to run the oracle."""
    from benchflow.cli._hub_client import print_controls

    print_controls(
        "robouse/sample",
        "1",
        {"validated": False, "controls": []},
        hub.url,
        running="oracle",
    )
    out = " ".join(capsys.readouterr().out.split())
    assert "this run is the oracle control; when it ends, push its job:" in out
    assert "--agent oracle" not in out


def test_bench_run_takes_the_hub_as_hub_commands_do(monkeypatch) -> None:
    """`bench run` named the Hub as `--registry hub:<url>`, every `bench hub`
    command as `--hub <url>`; the CLI's own hints mixed the two up."""
    import benchflow.cli.main as main

    calls: list[dict] = []
    monkeypatch.setattr(main, "eval_run", lambda **kw: calls.append(kw))
    url = "http://127.0.0.1:18871"
    result = CliRunner().invoke(
        app, ["run", "-d", "robouse/sample@1", "--hub", url, "--agent", "oracle"]
    )
    assert result.exit_code == 0, result.output
    assert calls[-1]["registry"] == f"hub:{url}"
    assert calls[-1]["dataset"] == "robouse/sample@1"
    both = CliRunner().invoke(
        app,
        ["run", "-d", "robouse/sample@1", "--hub", url, "--registry", f"hub:{url}"],
    )
    assert both.exit_code == 2 and "--hub <url> is --registry hub:<url>" in " ".join(
        both.output.split()
    )
    alone = CliRunner().invoke(app, ["run", "--hub", url])
    assert alone.exit_code == 2
    assert f"bench run -d <org/name@version> --hub {url}" in " ".join(
        alone.output.split()
    )
    assert len(calls) == 1
