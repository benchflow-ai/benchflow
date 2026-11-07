"""Guards FF-sw-docker-C14, the callback regression introduced by 79b9a530."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox._base import ExecResult
from tests.test_compose_raw_output import emit_process, transport
from tests.test_live_llm_trajectory import _sandbox_process


@pytest.mark.parametrize("backend", ["sdk", "docker", "remote-docker"])
@pytest.mark.parametrize("download", ["absent", "failed", "empty", "complete"])
@pytest.mark.parametrize(
    "content",
    [
        "https://example.com",
        "http://example.com",
        "https://example.com/path?query=value",
        "https://example.com/日本語",
    ],
)
async def test_callback_import_preserves_records_and_accounting(
    tmp_path, monkeypatch, backend, download, content
):
    """Ordinary URLs followed by email must survive every callback read path."""
    record = {
        "event": "success",
        "request": {"body": {"messages": [{"role": "user", "content": content}]}},
        "response": {
            "choices": [{"message": {"role": "assistant", "content": "u@example.org"}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        },
        "response_cost": 0.01,
    }
    text = json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
    if backend == "sdk":
        sandbox = SimpleNamespace(
            exec=AsyncMock(return_value=ExecResult(stdout=text, return_code=0))
        )
    else:
        sandbox = transport(tmp_path, backend, "fixture-login")
        sandbox._main_container = "fixture-container"
        emit_process(monkeypatch, lambda args: (text, "unrelated diagnostic", 0))
    # Backends with no download API must take the same raw cat path.
    if download == "absent":
        sandbox.download_file = None
    else:

        async def fetch(source, destination):
            if download == "failed":
                raise OSError("fixture download unavailable")
            destination.write_text(text if download == "complete" else "")

        sandbox.download_file = fetch
    process = _sandbox_process(sandbox)
    await process._load_callback_log()
    assert len(process.trajectory.exchanges) == 1
    assert process.trajectory.total_input_tokens == 4
    assert process.trajectory.total_output_tokens == 2
    assert process.trajectory.total_cost_usd == 0.01


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
@pytest.mark.parametrize(
    "consumer", ["json", "trajectory", "numeric", "path", "implicit-string"]
)
def test_diagnostic_text_cannot_be_consumed_as_data(stream, consumer):
    """C14: the presentation type cannot satisfy a parser's string interface."""
    from pathlib import Path

    from benchflow.providers.litellm_logging import trajectory_from_litellm_callback_log
    from benchflow.sandbox._exec_output import diagnostic_result

    raw = ExecResult(stdout='{"value": 1}', stderr='{"value": 2}', return_code=0)
    diagnostic = diagnostic_result(raw)
    text = getattr(diagnostic, stream)
    readers = {
        "json": json.loads,
        "trajectory": lambda value: trajectory_from_litellm_callback_log(
            value, session_id="fixture", agent_name="fixture"
        ),
        "numeric": int,
        "path": Path,
        "implicit-string": str,
    }
    with pytest.raises((TypeError, AttributeError)):
        readers[consumer](text)
    assert text.render() == getattr(raw, stream)
    assert raw.stdout == '{"value": 1}'
    assert raw.stderr == '{"value": 2}'
    assert diagnostic_result(diagnostic) is diagnostic
