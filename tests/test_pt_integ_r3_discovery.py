"""PT-integ round 3: plugin discovery's answer where the kit holds no Python.

``_discover_without_kit_python`` answers only when the answer is decided by
data: the kit's RECORD exists without a Python, and the trusted PYTHONPATH
and every pre-agent search path are empty, so the discovery script would
search nothing. Folders to search, no kit, or a kit Python are left to the
caller's rules (PT-INTEG-R3-1); an unreadable record fails closed. The real-Docker case is in
``test_pt_integ_r3_no_kit_python_docker.py``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from benchflow.sandbox import lockdown


def _answer(monkeypatch, return_code: int, stdout: str = "") -> list[str]:
    commands: list[str] = []

    async def read_exec(env, command, **kwargs):
        commands.append(command)
        return SimpleNamespace(return_code=return_code, stdout=stdout, stderr="")

    monkeypatch.setattr(lockdown, "_read_exec", read_exec)
    return commands


def _record(pythonpath: str, root: list[str], each: list[list[str]] = ()) -> str:
    lines = "".join(
        json.dumps({"exe": f"/x{i}", "path": p}) + "\n" for i, p in enumerate(each)
    )
    return f"{pythonpath}\0{json.dumps({'path': root, 'user_site': []})}\n\0{lines}"


@pytest.mark.asyncio
async def test_an_empty_record_answers_no_plugins_without_a_python(monkeypatch):
    commands = _answer(monkeypatch, 0, _record("", [], [[]]))
    assert await lockdown._discover_without_kit_python(object(), None) == {
        "plugins": [],
        "rejected": [],
    }
    # Kit tools by absolute path, and no Python started.
    (command,) = commands
    assert "/opt/benchflow/rt/bin/cat " in command
    assert "python" not in command.replace("IMAGE_PYTHON_PATHS", "").replace(
        "/opt/benchflow/rt/py/python3", ""
    )


@pytest.mark.asyncio
async def test_no_kit_or_a_kit_python_is_left_to_the_caller(monkeypatch):
    _answer(monkeypatch, 3)
    assert await lockdown._discover_without_kit_python(object(), None) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stdout, pythonpath",
    [
        (_record("", ["/usr/lib/python3.12/site-packages"]), None),
        (_record("", [], [["/opt/conda/lib/python3.12/site-packages"]]), None),
        (_record("/srv/lib", []), None),
        (_record("", []), "/srv/lib"),
    ],
    ids=[
        "root-record",
        "interpreter-record",
        "sandbox-pythonpath",
        "trusted-pythonpath",
    ],
)
async def test_folders_to_search_without_a_python_are_left_to_the_caller(
    monkeypatch, stdout, pythonpath
):
    """PT-INTEG-R3-1: only a Python could read them, so the caller's 127
    rules decide, as on both bases (requested flags only when root's route
    has no python3)."""
    _answer(monkeypatch, 0, stdout)
    assert await lockdown._discover_without_kit_python(object(), pythonpath) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "return_code, stdout",
    [(4, ""), (0, "\0{not json\0"), (0, "\0[]\0"), (0, "only-pythonpath")],
    ids=["unreadable", "not-json", "not-a-record", "truncated"],
)
async def test_an_unreadable_record_fails_closed(monkeypatch, return_code, stdout):
    _answer(monkeypatch, return_code, stdout)
    with pytest.raises(ValueError):
        await lockdown._discover_without_kit_python(object(), None)
