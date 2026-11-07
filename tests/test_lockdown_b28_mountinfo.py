"""Guard CONFIRM-lockdown-5 at bd23e190: use the kernel's mountinfo grammar."""

import sys

import pytest

from benchflow.sandbox import _pytest_plugin_guard as guard
from benchflow.sandbox._mount_aliases import blocked_mount_aliases
from tests.test_verifier_uv_state import install_plugin

# Linux escapes only these four bytes. Every other Unicode whitespace character
# stays literal, including the characters splitlines() treats as record endings.
WHITESPACE = tuple(chr(n) for n in range(sys.maxunicode + 1) if chr(n).isspace())


def encoded(value):
    for char, escape in (
        ("\\", r"\134"),
        (" ", r"\040"),
        ("\t", r"\011"),
        ("\n", r"\012"),
    ):
        value = value.replace(char, escape)
    return value


@pytest.mark.parametrize("char", WHITESPACE)
@pytest.mark.parametrize("backend", ["overlay", "ext4", "tmpfs"])
@pytest.mark.parametrize("position", ["root", "target"])
def test_every_whitespace_preserves_mount_coordinates(char, backend, position):
    root = "/volume" + (char + "part" if position == "root" else "")
    alias = "/opt/plugin" + (char + "view" if position == "target" else "")
    info = (
        "1 0 0:1 / / rw - overlay overlay rw\n"
        f"2 1 0:2 {encoded(root)} /app/plugins rw shared:5 - {backend} dummy rw\n"
        f"3 1 0:2 {encoded(root)} {encoded(alias)} rw - {backend} dummy rw\n"
    )
    blocked = blocked_mount_aliases(info, ("/app",))
    assert alias in blocked
    assert guard.untrusted_reason(alias + "/plugin.py", blocked)
    assert "/opt/unrelated" not in blocked


@pytest.mark.parametrize("char", WHITESPACE)
def test_discovery_and_runtime_refuse_whitespace_alias(tmp_path, monkeypatch, char):
    alias = tmp_path / ("plugin" + char + "view")
    alias.mkdir()
    install_plugin(alias, "raise AssertionError('agent plugin must never import')\n")
    info = (
        "1 0 0:1 / / rw - overlay overlay rw\n"
        "2 1 0:2 / /app/plugins rw - tmpfs dummy rw\n"
        f"3 1 0:2 / {encoded(str(alias))} rw - tmpfs dummy rw\n"
    )
    blocked = blocked_mount_aliases(info, ("/app",))
    monkeypatch.setattr(sys, "path", [str(alias), *sys.path])
    monkeypatch.setattr(guard, "_BENCHFLOW_OWNERSHIP", False)
    monkeypatch.setattr(guard, "_BENCHFLOW_BLOCKED", blocked)
    assert "ctrf" in guard.discover(blocked, ["ctrf"])["rejected"]
    with pytest.raises(guard.Rejected, match="ctrf"):
        guard._validate(["ctrf"])
    with pytest.raises(guard.Rejected, match="ctrf_model"):
        guard._validate(["ctrf_model"], entry_points=False)
    assert "ctrf_model" not in sys.modules


@pytest.mark.parametrize(
    "record",
    [
        "x 0 0:1 / / rw - overlay dummy rw",
        "1 x 0:1 / / rw - overlay dummy rw",
        "\u0661 0 0:1 / / rw - overlay dummy rw",
        "1 0 \u0660:1 / / rw - overlay dummy rw",
        "1 0 0:1 relative / rw - overlay dummy rw",
        "1 0 0:1 / relative rw - overlay dummy rw",
        "1 0 0:1 / / rw extra - overlay dummy",
        "1 0 0:1 / / rw - overlay dummy rw extra",
        "1 0 0:1 / / rw - overlay dummy rw\n\n",
        "1 0 0:1 / / rw  - overlay dummy rw",
        "1 0 0:1 / / rw\t- overlay dummy rw",
        "1 0 0:1 / / rw - overlay dummy rw\0",
        r"1 0 0:1 /bad\999path / rw - overlay dummy rw",
        r"1 0 0:1 / /bad\path rw - overlay dummy rw",
    ],
)
def test_malformed_mount_records_fail_closed(record):
    with pytest.raises(ValueError):
        blocked_mount_aliases(record, ("/app",))


def test_all_kernel_escapes_decode_once():
    path = "/opt/space tab\tline\nback\\040"
    info = (
        "1 0 0:1 / / rw - overlay dummy rw\n"
        "2 1 0:2 / /app rw - tmpfs dummy rw\n"
        f"3 1 0:2 / {encoded(path)} rw - tmpfs dummy rw\n"
    )
    assert path in blocked_mount_aliases(info, ("/app",))


@pytest.mark.parametrize("char", WHITESPACE)
@pytest.mark.parametrize("soft", [False, True], ids=["final", "soft"])
@pytest.mark.parametrize("backend", ["overlay", "ext4", "tmpfs"])
async def test_cleanup_preserves_every_whitespace_alias(
    tmp_path, monkeypatch, char, soft, backend
):
    from tests.test_lockdown_b22r1_workspace import (
        test_temporary_cleanup_preserves_only_workspace_mount_aliases,
    )

    await test_temporary_cleanup_preserves_only_workspace_mount_aliases(
        tmp_path, monkeypatch, soft, "project" + char + "view", backend
    )
