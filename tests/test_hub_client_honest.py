"""Against a Hub that can't do what was asked, `bench` says what the Hub is, and `bench hub push` says a push
worked only when the Hub linked the run to its dataset and kept its trial folders.

A Hub from before the dataset registry knows no registry route: a GET answers 404 "Not Found" and a POST
405, from its page route, and its route list (``/api/openapi.json``) names no ``/api/v1/registry``. Its upload
route takes no dataset and drops the job folder after reading the traces. So "check the name and version" or
"check the Hub address and your key" sent people the wrong way, and a push to it printed a green tick for a run
no version will ever count. ``FakeHub(registry=False)`` answers that way; ``FakeHub()`` answers as registry v1
does, with switches for the answers a push must not take as success.

Guards the fix for the misleading next steps and false pushes of the first Hub client.
"""

from __future__ import annotations

import json
import re
import threading
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchflow._utils.dataset_registry import DatasetResolutionError, resolve_dataset
from benchflow.cli.main import app

NO_REGISTRY = "has no dataset registry"


def _fields(content_type: str, raw: bytes) -> dict[str, str]:
    msg = BytesParser(policy=HTTP).parsebytes(
        f"Content-Type: {content_type}\r\n\r\n".encode() + raw
    )
    return {
        part.get_param("name", header="content-disposition"): part.get_content()
        for part in msg.iter_parts()
        if not part.get_filename()
    }


class FakeHub:
    """A Hub on 127.0.0.1. ``registry=False`` answers as a Hub did before it had the dataset registry; the other
    switches make a registry Hub's answers go wrong one way each."""

    def __init__(
        self,
        *,
        registry: bool = True,
        describe: bool = True,
        link: bool = True,
        keep: bool = True,
        readable: bool = True,
        answer_run: bool = True,
    ) -> None:
        self.uploads: list[dict[str, str]] = []
        hub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def _send(self, code: int, payload) -> None:
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path == "/api/openapi.json":
                    if not describe:
                        return self._send(404, {"detail": "Not Found"})
                    paths = {
                        "/api/v1/runs/upload": {"post": {}},
                        "/api/v1/runs/{run_id}": {"get": {}},
                    }
                    if registry:
                        paths |= {
                            "/api/v1/registry": {"post": {}},
                            "/api/v1/registry/{org}/{name}/{version}": {"get": {}},
                        }
                    return self._send(200, {"openapi": "3.1.0", "paths": paths})
                if self.path.startswith("/api/v1/registry/"):
                    return self._send(
                        404,
                        {
                            "detail": "no such dataset version"
                            if registry
                            else "Not Found"
                        },
                    )
                m = re.fullmatch(r"/api/v1/runs/([\w-]+)", self.path)
                if m:
                    if not readable:
                        return self._send(403, {"detail": "needs the 'read' scope"})
                    kept = registry and keep
                    return self._send(
                        200,
                        {
                            "id": m[1],
                            "job_dir": f"/hub/uploads/{m[1]}/job" if kept else None,
                        },
                    )
                self._send(404, {"detail": "Not Found"})

            def do_POST(self) -> None:
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                if self.path == "/api/v1/registry" and not registry:
                    return self._send(405, {"detail": "Method Not Allowed"})
                if self.path == "/api/v1/runs/upload":
                    fields = _fields(self.headers["Content-Type"], raw)
                    hub.uploads.append(fields)
                    if not answer_run:
                        return self._send(200, {"imported": []})
                    run = {"run_id": "run_1", "n_traces": 1}
                    if registry and link and fields.get("dataset"):
                        run |= {
                            "dataset": fields["dataset"],
                            "dataset_content": "checked",
                        }
                    return self._send(200, {"imported": [run]})
                self._send(404, {"detail": "Not Found"})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture
def make_hub(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BENCHFLOW_HUB_URL", raising=False)
    monkeypatch.delenv("BENCHFLOW_HUB_TOKEN", raising=False)
    made: list[FakeHub] = []

    def make(**kw) -> FakeHub:
        made.append(FakeHub(**kw))
        return made[-1]

    yield make
    for hub in made:
        hub.server.shutdown()


def _job(root: Path, dataset: str | None = "acme/tasks@1") -> Path:
    """A finished run's folder, its trial recording the dataset it ran when ``dataset`` is set."""
    job = root / "jobs" / "2026-11-07__00-00-00"
    trial = job / "hello__abc"
    (trial / "verifier").mkdir(parents=True)
    (job / "summary.json").write_text("{}")
    name, _, version = (dataset or "").partition("@")
    record = {"dataset_name": name, "dataset_version": version} if dataset else {}
    (trial / "config.json").write_text(json.dumps(record))
    (trial / "verifier" / "reward.txt").write_text("1")
    return job


def _out(result) -> str:
    return " ".join(result.output.split())


def test_show_on_a_hub_without_the_registry_says_so(make_hub) -> None:
    hub = make_hub(registry=False)
    result = CliRunner().invoke(app, ["hub", "show", "acme/tasks@1", "--hub", hub.url])
    out = _out(result)
    assert result.exit_code == 2, out
    assert NO_REGISTRY in out
    assert "check the name and version" not in out
    assert "bench run <tasks-dir>" in out


def test_publish_to_a_hub_without_the_registry_says_so(tmp_path, make_hub) -> None:
    hub = make_hub(registry=False)
    task = tmp_path / "src" / "hello"
    task.mkdir(parents=True)
    (task / "task.md").write_text("---\ntask:\n  name: hello\n---\n\nSay hello.\n")
    result = CliRunner().invoke(
        app,
        [
            "hub",
            "publish",
            str(tmp_path / "src"),
            "--name",
            "acme/tasks@1",
            "--hub",
            hub.url,
        ],
    )
    out = _out(result)
    assert result.exit_code == 2, out
    assert NO_REGISTRY in out
    assert "check the Hub address and your key" not in out


def test_running_by_name_from_a_hub_without_the_registry_says_so(make_hub) -> None:
    hub = make_hub(registry=False)
    with pytest.raises(DatasetResolutionError) as failed:
        resolve_dataset("acme/tasks@1", registry=f"hub:{hub.url}")
    assert NO_REGISTRY in str(failed.value)
    assert "shows what the Hub holds" not in failed.value.hint
    assert "bench run <tasks-dir>" in failed.value.hint
    assert failed.value.exit_code == 2


def test_a_missing_version_on_a_registry_hub_still_asks_for_the_name(make_hub) -> None:
    hub = make_hub()
    result = CliRunner().invoke(app, ["hub", "show", "acme/tasks@9", "--hub", hub.url])
    out = _out(result)
    assert result.exit_code == 2, out
    assert "check the name and version" in out
    assert NO_REGISTRY not in out


def test_a_hub_without_a_route_list_gets_both_possibilities(make_hub) -> None:
    hub = make_hub(describe=False)
    result = CliRunner().invoke(app, ["hub", "show", "acme/tasks@1", "--hub", hub.url])
    out = _out(result)
    assert result.exit_code == 2, out
    assert "check the name and version" in out
    assert "without the dataset registry" in out


def test_push_refuses_a_dataset_the_hub_cannot_link(tmp_path, make_hub) -> None:
    hub = make_hub(registry=False)
    result = CliRunner().invoke(
        app, ["hub", "push", str(_job(tmp_path)), "--hub", hub.url]
    )
    out = _out(result)
    assert result.exit_code == 2, out
    assert NO_REGISTRY in out and "nothing was uploaded" in out
    assert hub.uploads == []
    assert "✓" not in out


def test_push_fails_when_the_hub_keeps_no_trial_folders(tmp_path, make_hub) -> None:
    hub = make_hub(registry=False)
    result = CliRunner().invoke(
        app, ["hub", "push", str(_job(tmp_path, dataset=None)), "--hub", hub.url]
    )
    out = _out(result)
    assert result.exit_code == 1, out
    assert "kept no trial folders" in out and "run_1" in out
    assert "✓" not in out


def test_push_fails_when_the_hub_drops_the_link(tmp_path, make_hub) -> None:
    hub = make_hub(link=False)
    result = CliRunner().invoke(
        app, ["hub", "push", str(_job(tmp_path)), "--hub", hub.url]
    )
    out = _out(result)
    assert result.exit_code == 1, out
    assert "isn't linked to acme/tasks@1" in out
    assert "✓" not in out


def test_push_fails_when_a_registry_hub_drops_the_folders(tmp_path, make_hub) -> None:
    hub = make_hub(keep=False)
    result = CliRunner().invoke(
        app, ["hub", "push", str(_job(tmp_path)), "--hub", hub.url]
    )
    out = _out(result)
    assert result.exit_code == 1, out
    assert "kept no trial folders" in out
    assert "✓" not in out


def test_push_fails_when_the_key_cannot_read_the_run_back(tmp_path, make_hub) -> None:
    hub = make_hub(readable=False)
    result = CliRunner().invoke(
        app, ["hub", "push", str(_job(tmp_path)), "--hub", hub.url]
    )
    out = _out(result)
    assert result.exit_code == 1, out
    assert "read scope" in out
    assert "✓" not in out


def test_push_fails_when_the_hub_answers_without_a_run(tmp_path, make_hub) -> None:
    hub = make_hub(answer_run=False)
    result = CliRunner().invoke(
        app, ["hub", "push", str(_job(tmp_path)), "--hub", hub.url]
    )
    out = _out(result)
    assert result.exit_code == 1, out
    assert "without a run" in out
    assert "✓" not in out


def test_push_succeeds_when_linked_and_kept(tmp_path, make_hub) -> None:
    hub = make_hub()
    result = CliRunner().invoke(
        app, ["hub", "push", str(_job(tmp_path)), "--hub", hub.url]
    )
    out = _out(result)
    assert result.exit_code == 0, out
    assert "✓ Pushed" in out and "linked to acme/tasks@1" in out
    assert hub.uploads[-1]["dataset"] == "acme/tasks@1"
