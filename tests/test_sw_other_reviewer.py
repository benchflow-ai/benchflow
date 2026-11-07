"""Guards FF-sw-other reviewer containment at 363d815b without model calls."""

import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

from benchflow.experimental.mcp import reviewer_server


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "escape",
    ["../app-private/reference.txt", "absolute", "leaf", "ancestor", "swap", "fifo"],
)
async def test_reviewer_only_reads_regular_files_beneath_app(
    tmp_path, monkeypatch, escape
):
    app = tmp_path / "app"
    app.mkdir()
    private = tmp_path / "app-private"
    private.mkdir()
    marker = private / "reference.txt"
    marker.write_text("DUMMY_PRIVATE_MARKER")
    (app / "good.py").write_text("HONEST_CODE")
    if escape == "absolute":
        target = str(marker)
    elif escape == "leaf":
        (app / "link").symlink_to(marker)
        target = "link"
    elif escape == "ancestor":
        (app / "sub").symlink_to(private, target_is_directory=True)
        target = "sub/reference.txt"
    elif escape == "swap":
        (app / "sub").mkdir()
        (app / "sub/reference.txt").write_text("HONEST_NESTED")
        target = "sub/reference.txt"
    elif escape == "fifo":
        os.mkfifo(app / "pipe")
        target = "pipe"
    else:
        target = escape
    tools = {}

    class FakeMCP:
        def __init__(self, *args):
            pass

        def tool(self):
            def register(fn):
                tools[fn.__name__] = fn
                return fn

            return register

    fake = ModuleType("fastmcp")
    fake.FastMCP = FakeMCP
    monkeypatch.setitem(sys.modules, "fastmcp", fake)
    calls = []
    genai = ModuleType("google.generativeai")
    genai.configure = lambda **kw: None
    genai.GenerativeModel = lambda model: SimpleNamespace(
        generate_content=lambda prompt: (
            calls.append(prompt) or SimpleNamespace(text="ok")
        )
    )
    monkeypatch.setitem(sys.modules, "google.generativeai", genai)
    import google

    monkeypatch.setattr(google, "generativeai", genai, raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    original_path = reviewer_server.Path
    monkeypatch.setattr(
        reviewer_server,
        "Path",
        lambda value: app if str(value) == "/app" else original_path(value),
    )
    if escape == "swap":
        original_exists, original_open = original_path.exists, os.open
        swapped = False

        def swap():
            nonlocal swapped
            if not swapped:
                (app / "sub").rename(app / "held")
                (app / "sub").symlink_to(private, target_is_directory=True)
                swapped = True

        def exists(path):
            result = original_exists(path)
            if path == app / target:
                swap()  # The old check/use path must not open this new target.
            return result

        def opened(path, flags, *args, **kwargs):
            fd = original_open(path, flags, *args, **kwargs)
            if path == "sub":
                swap()  # The fixed reader must use its pinned directory fd.
            return fd

        monkeypatch.setattr(original_path, "exists", exists)
        monkeypatch.setattr(reviewer_server.os, "open", opened)
    reviewer_server.create_reviewer_server()
    await tools["review_code"](files=[target, "good.py"])
    assert calls and "HONEST_CODE" in calls[0]
    assert "DUMMY_PRIVATE_MARKER" not in calls[0]


@pytest.fixture(autouse=True)
def bounded_alias_repro():
    import signal

    def expired(*args):
        raise TimeoutError("unsafe file open blocked on agent FIFO")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(3)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
