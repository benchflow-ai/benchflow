"""A key file is checked and read through one open descriptor.

The hosted client checked a key file with ``lstat`` (a regular file, mode 600 or stricter) and then read
it by its path, following links: a file swapped in between the two passed the check as one file and was
read as another. Now the file is opened once, without following a link where the platform can refuse one,
and the check (``fstat``: a regular file, mode 600 or stricter, this user's) and the read use that
descriptor. Every key here is made up.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from benchflow import hosted
from benchflow.hosted import HostedError, read_key_file, read_secret

# Built at run time: no key-shaped literal in the source.
CHECKED = "bfk_" + "C" * 36
SWAPPED = "bfk_" + "S" * 36


def _reader(kind: str):
    return {
        "key file": read_key_file,
        "model key": lambda path: read_secret(f"file:{path}"),
    }[kind]


def _key_file(tmp_path: Path) -> Path:
    path = tmp_path / "key"
    path.write_text(CHECKED + "\n")
    path.chmod(0o600)
    return path


def _swap_after_the_check(monkeypatch, path: Path, swap) -> list[str]:
    """Run *swap* right after the first stat of the key file, whichever call makes it: the moment between
    the check and the read."""
    done: list[str] = []
    real = {name: getattr(os, name) for name in ("stat", "lstat", "fstat")}

    def wrap(name):
        def call(target, *args, **kwargs):
            info = real[name](target, *args, **kwargs)
            if not done and (
                name == "fstat"
                or isinstance(target, int)
                or Path(os.fspath(target)) == path
            ):
                done.append(name)
                swap()
            return info

        return call

    for name in real:
        monkeypatch.setattr(os, name, wrap(name))
    return done


@pytest.mark.parametrize("kind", ["key file", "model key"])
def test_a_file_swapped_in_after_the_check_is_never_read(monkeypatch, tmp_path, kind):
    path = _key_file(tmp_path)
    other = tmp_path / "other"
    other.write_text(SWAPPED + "\n")
    other.chmod(0o644)
    done = _swap_after_the_check(monkeypatch, path, lambda: os.replace(other, path))
    value = _reader(kind)(path)
    assert done, "the key file was never checked"
    assert value.strip() == CHECKED


@pytest.mark.parametrize("kind", ["key file", "model key"])
def test_a_link_swapped_in_after_the_check_is_never_followed(
    monkeypatch, tmp_path, kind
):
    path = _key_file(tmp_path)
    target = tmp_path / "elsewhere"
    target.write_text(SWAPPED + "\n")
    link = tmp_path / "link"
    link.symlink_to(target)
    done = _swap_after_the_check(monkeypatch, path, lambda: os.replace(link, path))
    value = _reader(kind)(path)
    assert done, "the key file was never checked"
    assert value.strip() == CHECKED


@pytest.mark.parametrize("kind", ["key file", "model key"])
def test_a_link_is_refused(tmp_path, kind):
    target = _key_file(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(HostedError) as refused:
        _reader(kind)(link)
    assert "not a regular file" in str(refused.value)


@pytest.mark.parametrize("kind", ["key file", "model key"])
def test_a_key_file_another_user_owns_is_refused(monkeypatch, tmp_path, kind):
    path = _key_file(tmp_path)
    owner = path.stat().st_uid
    monkeypatch.setattr(hosted.os, "getuid", lambda: owner + 1)
    with pytest.raises(HostedError) as refused:
        _reader(kind)(path)
    assert f"the key file {path} belongs to another user" in str(refused.value)


@pytest.mark.timeout(20)
@pytest.mark.parametrize("kind", ["key file", "model key"])
def test_a_fifo_is_refused_without_waiting_for_a_writer(tmp_path, kind):
    path = tmp_path / "fifo"
    os.mkfifo(path, 0o600)
    with pytest.raises(HostedError) as refused:
        _reader(kind)(path)
    assert "not a regular file" in str(refused.value)
