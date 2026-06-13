from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

from benchflow.environment.browser_runtime import (
    browser_environment_runtime,
    browser_runtime_session,
    browser_task_instruction,
    expected_from_prompt,
)


def test_browser_environment_runtime_serves_local_fixture(tmp_path: Path) -> None:
    """Guards 0.7 browser environment setup from living inside agent shims."""
    app = tmp_path / "app"
    fixture = app / "browser_fixture"
    fixture.mkdir(parents=True)
    (fixture / "index.html").write_text("<main>ready</main>\n")

    with browser_environment_runtime(app) as handle:
        assert handle.url is not None
        assert handle.allowed_domains == ["127.0.0.1"]
        readiness = handle.check_readiness()
        body = urlopen(handle.url, timeout=5).read().decode()

    assert "ready" in body
    assert readiness.status == "ready"
    assert readiness.ok is True
    assert readiness.http_status == 200
    assert readiness.content_bytes == len("<main>ready</main>\n")
    assert isinstance(readiness.content_sha256, str)
    assert len(readiness.content_sha256) == 64
    assert handle.to_dict()["adapter"] == "browser"
    assert handle.to_dict()["fixture"] == "browser_fixture"
    assert handle.to_dict(readiness=readiness)["readiness"]["status"] == "ready"


def test_browser_environment_runtime_handles_freeform_prompt(tmp_path: Path) -> None:
    app = tmp_path / "app"
    app.mkdir()

    with browser_environment_runtime(app) as handle:
        assert handle.url is None
        assert handle.allowed_domains is None
        readiness = handle.check_readiness()
        instruction = browser_task_instruction(
            prompt="Browse example.com.",
            expected=None,
            handle=handle,
        )

    assert instruction == "Browse example.com."
    assert readiness.status == "not-applicable"
    assert readiness.ok is True
    assert "fixture" not in handle.to_dict()


def test_browser_environment_readiness_scrubs_html_title(tmp_path: Path) -> None:
    """Guards browser readiness artifacts from storing raw fixture HTML."""
    app = tmp_path / "app"
    fixture = app / "browser_fixture"
    fixture.mkdir(parents=True)
    (fixture / "index.html").write_text(
        "<title>Secret Fixture Title</title><main>ready</main>\n"
    )

    with browser_environment_runtime(app) as handle:
        readiness = handle.check_readiness()

    payload = readiness.to_dict()
    assert payload["status"] == "ready"
    assert "Secret Fixture Title" not in str(payload)
    assert isinstance(payload["title_sha256"], str)
    assert len(payload["title_sha256"]) == 64


def test_browser_runtime_session_writes_trace_artifact(tmp_path: Path) -> None:
    """Guards browser artifact writing from drifting back into agent shims."""
    app = tmp_path / "app"
    fixture = app / "browser_fixture"
    fixture.mkdir(parents=True)
    (fixture / "index.html").write_text("<main>ready</main>\n")
    artifact_path = tmp_path / "artifacts" / "browser-use-smoke-trace.json"

    with browser_runtime_session(app, require_ready=True) as session:
        artifact = session.write_trace_artifact(
            artifact_path,
            framework="benchflow-browser-test",
            steps=["open", {"action": "screenshot"}],
            screenshots_b64=["abc", ""],
            final_result="ready",
            duration_sec=0.25,
            extra={"custom": "value"},
        )

    assert artifact_path.is_file()
    assert artifact["schema"] == "benchflow.browser-runtime-trace.v1"
    assert artifact["framework"] == "benchflow-browser-test"
    assert artifact["screenshots_b64"] == ["abc"]
    assert artifact["environment"]["adapter"] == "browser"
    assert artifact["environment"]["readiness"]["status"] == "ready"
    assert artifact["custom"] == "value"
    assert "ready</main>" not in artifact_path.read_text()


def test_browser_task_instruction_uses_served_url(tmp_path: Path) -> None:
    app = tmp_path / "app"
    fixture = app / "browser_fixture"
    fixture.mkdir(parents=True)
    (fixture / "index.html").write_text("<main>browser-use-smoke: ready</main>\n")

    with browser_environment_runtime(app) as handle:
        instruction = browser_task_instruction(
            prompt="Final answer must be exactly: browser-use-smoke: ready",
            expected="browser-use-smoke: ready",
            handle=handle,
        )

    assert "http://127.0.0.1:" in instruction
    assert "file://" not in instruction
    assert instruction.endswith("browser-use-smoke: ready")


def test_expected_from_prompt_extracts_exact_final_answer() -> None:
    assert (
        expected_from_prompt('Final answer must be exactly: "browser-use-smoke: ready"')
        == "browser-use-smoke: ready"
    )
