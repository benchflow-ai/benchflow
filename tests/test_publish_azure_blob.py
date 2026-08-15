"""Offline tests for trajectory staging, redaction, and Azure direct upload."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import ModuleType

import pytest

from benchflow.publish.azure_blob import upload_capture_direct
from benchflow.publish.traj_capture import stage_trajectory_capture


class ResourceExistsError(Exception):
    pass


class ResourceNotFoundError(Exception):
    pass


class ClientAuthenticationError(Exception):
    pass


class HttpResponseError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class FakeContainerClient:
    def __init__(self, failures: dict[str, Exception] | None = None) -> None:
        self.failures = failures or {}
        self.calls: list[dict] = []

    def upload_blob(self, **kwargs) -> None:
        self.calls.append(kwargs)
        failure = self.failures.get(kwargs["name"])
        if failure is not None:
            raise failure


def _trial(tmp_path: Path, files: dict[str, str] | None = None) -> Path:
    trial = tmp_path / "trial-01"
    trajectory = trial / "trajectory"
    trajectory.mkdir(parents=True)
    for name, content in (
        files
        or {
            "llm_trajectory.jsonl": '{"request":{"model":"demo"}}\n',
            "acp_trajectory.jsonl": '{"type":"message","text":"hello"}\n',
        }
    ).items():
        (trajectory / name).write_text(content, encoding="utf-8")
    return trial


def _install_fake_azure(
    monkeypatch: pytest.MonkeyPatch, client: FakeContainerClient
) -> None:
    azure = ModuleType("azure")
    core = ModuleType("azure.core")
    exceptions = ModuleType("azure.core.exceptions")
    identity = ModuleType("azure.identity")
    storage = ModuleType("azure.storage")
    blob = ModuleType("azure.storage.blob")

    class ContainerClient:
        @staticmethod
        def from_container_url(container_url: str, credential=None):
            client.container_url = container_url
            client.credential = credential
            return client

    class ContentSettings:
        def __init__(self, *, content_type: str) -> None:
            self.content_type = content_type

    exceptions.ResourceExistsError = ResourceExistsError
    exceptions.ResourceNotFoundError = ResourceNotFoundError
    exceptions.ClientAuthenticationError = ClientAuthenticationError
    exceptions.HttpResponseError = HttpResponseError
    identity.DefaultAzureCredential = lambda: "default-credential"
    blob.ContainerClient = ContainerClient
    blob.ContentSettings = ContentSettings
    for name, module in {
        "azure": azure,
        "azure.core": core,
        "azure.core.exceptions": exceptions,
        "azure.identity": identity,
        "azure.storage": storage,
        "azure.storage.blob": blob,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_trial_resolution_uses_only_jsonl_and_reports_ignored(tmp_path: Path) -> None:
    """A trial stages non-recursive trajectory JSONL and reports ignored siblings."""
    trial = _trial(tmp_path)
    (trial / "trajectory" / "notes.txt").write_text("ignored", encoding="utf-8")
    (trial / "result.json").write_text('{"agent":"demo"}', encoding="utf-8")

    with stage_trajectory_capture(trial, source_id="source/demo") as staged:
        assert [item.relname for item in staged.files] == [
            "trajectory/acp_trajectory.jsonl",
            "trajectory/llm_trajectory.jsonl",
            "manifest.json",
        ]
        assert staged.ignored == ("notes.txt",)
        assert staged.manifest["run"]["agent"] == "demo"


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_file_and_bare_directory_normalize_under_trajectory(
    tmp_path: Path, kind: str
) -> None:
    """Single-file and bare-directory inputs use the same object namespace."""
    source = tmp_path / "capture.jsonl"
    source.write_text('{"type":"demo"}\n', encoding="utf-8")
    path = source if kind == "file" else tmp_path

    with stage_trajectory_capture(path, source_id="demo") as staged:
        assert staged.files[0].relname == "trajectory/capture.jsonl"


def test_invalid_empty_and_oversize_inputs_fail_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Staging rejects missing, malformed, empty, and over-limit trajectories."""
    with pytest.raises(ValueError, match=r"no \.jsonl"):
        stage_trajectory_capture(tmp_path, source_id="demo").__enter__()

    malformed = tmp_path / "bad.jsonl"
    malformed.write_text("{bad\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl: line 1"):
        stage_trajectory_capture(malformed, source_id="demo").__enter__()

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        stage_trajectory_capture(empty, source_id="demo").__enter__()

    invalid_utf8 = tmp_path / "llm_trajectory.jsonl"
    invalid_utf8.write_bytes(b"\xff\n")
    with pytest.raises(ValueError, match="must be UTF-8"):
        stage_trajectory_capture(invalid_utf8, source_id="demo").__enter__()

    import benchflow.publish.traj_capture as capture_module

    bounded = tmp_path / "bounded.jsonl"
    bounded.write_text(json.dumps({"text": "x" * 64}) + "\n", encoding="utf-8")
    monkeypatch.setattr(capture_module, "MAX_JSONL_RECORD_BYTES", 32)
    with pytest.raises(ValueError, match="JSONL record exceeds"):
        stage_trajectory_capture(bounded, source_id="demo").__enter__()

    nested: object = "leaf"
    for _ in range(101):
        nested = {"child": nested}
    deeply_nested = tmp_path / "deeply-nested.jsonl"
    deeply_nested.write_text(json.dumps(nested) + "\n", encoding="utf-8")
    monkeypatch.setattr(capture_module, "MAX_JSONL_RECORD_BYTES", 8 * 1024**2)
    with pytest.raises(ValueError, match="JSON nesting exceeds"):
        stage_trajectory_capture(deeply_nested, source_id="demo").__enter__()

    monkeypatch.setattr(capture_module, "MAX_FILE_BYTES", 1)
    with pytest.raises(ValueError, match="exceeds"):
        stage_trajectory_capture(malformed, source_id="demo").__enter__()


def test_redaction_is_structural_counted_and_preserves_untouched_lines(
    tmp_path: Path,
) -> None:
    """Nested keys and token values redact while untouched lines remain byte-identical."""
    untouched = '{"type": "message", "text": "safe"}\n'
    secret = "sk-1234567890abcdefghijklmnop"
    trial = _trial(
        tmp_path,
        {
            "acp_trajectory.jsonl": (
                untouched
                + json.dumps(
                    {
                        "nested": {"api_key": "prefixless"},
                        "OPENAI_API_KEY": "another-prefixless-value",
                        "credentials": {"token": "opaque-object-secret"},
                        "secret": ["opaque-list-secret"],
                        "password": 123456,
                        "aws_session_key": "ASIAQWERTYUIOPASDFGH",
                        "text": f"token={secret}",
                    }
                )
                + "\n"
            )
        },
    )

    with stage_trajectory_capture(trial, source_id="demo") as staged:
        payload = staged.files[0].local_path.read_text(encoding="utf-8")
        assert payload.startswith(untouched)
        assert secret not in payload
        assert "opaque-object-secret" not in payload
        assert "opaque-list-secret" not in payload
        assert "123456" not in payload
        assert "ASIAQWERTYUIOPASDFGH" not in payload
        assert '"api_key":"[REDACTED]"' in payload
        assert "another-prefixless-value" not in payload
        assert staged.redaction_replacements == 7
        assert staged.manifest["redaction"] == {"applied": True, "replacements": 7}

    with stage_trajectory_capture(trial, source_id="demo", redact=False) as staged:
        assert secret in staged.files[0].local_path.read_text(encoding="utf-8")
        assert staged.manifest["redaction"] == {"applied": False, "replacements": 0}


@pytest.mark.parametrize("source_id", ["../private", "team/../private", "team/./run"])
def test_source_id_rejects_relative_path_segments(
    tmp_path: Path, source_id: str
) -> None:
    """Direct upload labels cannot introduce relative-looking blob segments."""
    trial = _trial(tmp_path)
    with pytest.raises(ValueError, match="invalid source id"):
        stage_trajectory_capture(trial, source_id=source_id).__enter__()


def test_digest_manifest_and_metadata_are_transport_independent(tmp_path: Path) -> None:
    """Digest order, manifest schema, metadata, and manifest-last order are stable."""
    trial = _trial(tmp_path)
    (trial / "result.json").write_text(
        json.dumps({"agent": "codex", "model": "gpt-demo", "rewards": {"reward": 1.0}}),
        encoding="utf-8",
    )
    (trial / "config.json").write_text(
        json.dumps({"skill_mode": "with-skill", "task_id": "demo-task"}),
        encoding="utf-8",
    )

    with stage_trajectory_capture(trial, source_id="demo") as first:
        first_digest = first.traj_digest
        manifest = first.manifest
        assert first.files[-1].relname == "manifest.json"
        assert manifest["schema_version"] == "1.0.0"
        assert manifest["kind"] == "bronze.trajectory"
        assert "mode" not in manifest["tool"]
        assert manifest["run"] == {
            "agent": "codex",
            "model": "gpt-demo",
            "harness": None,
            "skill_mode": "with-skill",
            "task_id": "demo-task",
            "reward": 1.0,
        }

    with stage_trajectory_capture(trial, source_id="demo") as second:
        assert second.traj_digest == first_digest

    path = trial / "trajectory" / "acp_trajectory.jsonl"
    path.write_text('{"type":"changed"}\n', encoding="utf-8")
    with stage_trajectory_capture(trial, source_id="demo") as changed:
        assert changed.traj_digest != first_digest


def test_direct_upload_preserves_canonical_order_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Azure direct upload is a create-only loop over staged files verbatim."""
    client = FakeContainerClient()
    _install_fake_azure(monkeypatch, client)
    trial = _trial(tmp_path)

    with stage_trajectory_capture(trial, source_id="demo/source") as staged:
        result = upload_capture_direct(
            staged,
            container_url="https://tasksminerdata.blob.core.windows.net/bronze",
        )
        expected = [f"{result.prefix}{item.relname}" for item in staged.files]

    assert [call["name"] for call in client.calls] == expected
    assert client.calls[-1]["name"].endswith("manifest.json")
    assert all(call["overwrite"] is False for call in client.calls)
    assert all("-" not in key for key in client.calls[0]["metadata"])
    assert client.calls[0]["content_settings"].content_type == "application/jsonl"
    assert result.url.startswith("https://tasksminerdata.blob.core.windows.net/bronze/")


def test_direct_upload_suppresses_azure_sdk_info_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Azure request diagnostics do not clutter or disclose the direct CLI path."""

    class LoggingClient(FakeContainerClient):
        def upload_blob(self, **kwargs) -> None:
            logging.getLogger("azure.core.pipeline").info(
                "signed request %s", kwargs["name"]
            )
            super().upload_blob(**kwargs)

    client = LoggingClient()
    _install_fake_azure(monkeypatch, client)
    caplog.set_level(logging.INFO)
    trial = _trial(tmp_path)

    with stage_trajectory_capture(trial, source_id="demo") as staged:
        upload_capture_direct(
            staged,
            container_url="https://tasksminerdata.blob.core.windows.net/bronze",
        )

    assert "signed request" not in caplog.text


def test_direct_upload_skips_existing_blobs_and_continues_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing Azure blobs are resumable no-ops, including the commit marker."""
    trial = _trial(tmp_path)
    with stage_trajectory_capture(trial, source_id="demo") as staged:
        prefix = f"sources/demo/{staged.traj_digest}/"
        failures = {
            prefix + staged.files[0].relname: ResourceExistsError(),
            prefix + "manifest.json": ResourceExistsError(),
        }
        client = FakeContainerClient(failures)
        _install_fake_azure(monkeypatch, client)
        result = upload_capture_direct(
            staged,
            container_url="https://tasksminerdata.blob.core.windows.net/bronze",
        )

    assert len(result.skipped) == 2
    assert result.skipped[-1].endswith("manifest.json")
    assert len(client.calls) == len(staged.files)


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (ResourceNotFoundError(), "container not found"),
        (ClientAuthenticationError(), "az login"),
        (HttpResponseError("forbidden", 403), "Blob Data Creator"),
    ],
)
def test_direct_upload_surfaces_azure_prerequisites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    message: str,
) -> None:
    """Azure SDK failures become actionable CLI-safe errors."""
    trial = _trial(tmp_path, {"acp_trajectory.jsonl": '{"type":"demo"}\n'})
    with stage_trajectory_capture(trial, source_id="demo") as staged:
        name = f"sources/demo/{staged.traj_digest}/{staged.files[0].relname}"
        _install_fake_azure(monkeypatch, FakeContainerClient({name: failure}))
        with pytest.raises(ValueError, match=message):
            upload_capture_direct(
                staged,
                container_url="https://tasksminerdata.blob.core.windows.net/bronze",
            )


def test_direct_upload_explains_missing_optional_sdk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stock broker-only install gets precise optional-extra guidance."""
    for name in tuple(sys.modules):
        if name == "azure" or name.startswith("azure."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "azure", None)
    trial = _trial(tmp_path, {"acp_trajectory.jsonl": '{"type":"demo"}\n'})
    with (
        stage_trajectory_capture(trial, source_id="demo") as staged,
        pytest.raises(ValueError, match=r"pip install 'benchflow\[azure\]'"),
    ):
        upload_capture_direct(
            staged,
            container_url="https://tasksminerdata.blob.core.windows.net/bronze",
        )
