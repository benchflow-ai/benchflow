"""Run BenchFlow's root setup shell locally, the test's user standing in for root.

In a sandbox, BenchFlow's root-only checks (``benchflow._root_only``) and
the ``[ "$(id -u)" = 0 ]`` gates around them run as root, and they accept
only files uid 0 owns. Tests that run real install or proxy commands on the
machine run as an unprivileged user, so every file they make is that
user's. :func:`as_test_root` lets that user count as root in those checks;
every other rule (no group or other write, links walked hop by hop, sticky
folders) is unchanged, so a folder someone else may write, or a link out of
the tree, is still refused.

:func:`run_sh` runs a command from a script file, as install.py stages a
long install script: an install recipe can be longer than one argument may
be (128 KiB on Linux).
"""

from __future__ import annotations

import os
import subprocess
import tempfile


def as_test_root(command: str) -> str:
    uid = os.getuid()
    if uid == 0:
        return command
    return (
        command.replace("-user 0", f"\\( -user 0 -o -user {uid} \\)")
        .replace('[ "$(id -u)" = 0 ]', f'[ "$(id -u)" = {uid} ]')
        .replace('[ "$(id -u)" != 0 ]', f'[ "$(id -u)" != {uid} ]')
    )


def run_sh(
    command: str, *, shell: str = "/bin/sh", **kwargs
) -> subprocess.CompletedProcess:
    """``subprocess.run([shell, "-c", command], **kwargs)``, from a file, with
    :func:`as_test_root` applied."""
    fd, path = tempfile.mkstemp(prefix="bf-test-root-", suffix=".sh")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(as_test_root(command))
        return subprocess.run([shell, path], **kwargs)
    finally:
        os.unlink(path)
