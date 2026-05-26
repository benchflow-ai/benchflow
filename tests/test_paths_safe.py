"""Tests for benchflow._paths — segment validation and containment checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow._paths import assert_within, safe_path_segment

# ---------------------------------------------------------------------------
# safe_path_segment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "",  # empty
        ".",  # current dir
        "..",  # parent dir
        "a/b",  # forward slash separator
        "a\\b",  # backslash separator
        "../escape",  # traversal prefix
        "-x",  # leading dash (CLI flag risk)
        " name",  # leading whitespace
        "name ",  # trailing whitespace
        "\tname",  # leading tab
        "na\x00me",  # NUL byte
        "\x00",  # bare NUL
    ],
)
def test_rejects_unsafe(bad: str) -> None:
    with pytest.raises(ValueError):
        safe_path_segment(bad, kind="case id")


@pytest.mark.parametrize(
    "good",
    [
        "case-001",
        "a",
        "skill_name",
        "MixedCase123",
        "Tëst-üñïcödé",  # arbitrary unicode is allowed
        "name.with.dots",
        "_underscore_lead",
        "a-b-c",
    ],
)
def test_accepts_safe(good: str) -> None:
    assert safe_path_segment(good) == good


def test_rejects_non_string() -> None:
    with pytest.raises(ValueError):
        safe_path_segment(123)  # type: ignore[arg-type]


def test_error_message_includes_kind() -> None:
    with pytest.raises(ValueError, match="skill name"):
        safe_path_segment("../escape", kind="skill name")


# ---------------------------------------------------------------------------
# assert_within
# ---------------------------------------------------------------------------


def test_within_accepts_nested(tmp_path: Path) -> None:
    child = tmp_path / "a" / "b"
    child.mkdir(parents=True)
    assert assert_within(child, tmp_path) == child.resolve()


def test_within_accepts_root_itself(tmp_path: Path) -> None:
    assert assert_within(tmp_path, tmp_path) == tmp_path.resolve()


def test_within_rejects_sibling(tmp_path: Path) -> None:
    root = tmp_path / "root"
    sibling = tmp_path / "sibling"
    root.mkdir()
    sibling.mkdir()
    with pytest.raises(ValueError, match="outside"):
        assert_within(sibling, root)


def test_within_rejects_parent_traversal(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    escape = root / ".." / "escape"
    with pytest.raises(ValueError):
        assert_within(escape, root)


def test_within_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    link = root / "link"
    link.symlink_to(outside)
    # Resolution should follow the symlink and detect the escape.
    with pytest.raises(ValueError):
        assert_within(link, root)
