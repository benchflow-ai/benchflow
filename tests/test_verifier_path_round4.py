"""Guard CONFIRM157-R2-1/2 against the regressions at 5ff423c0."""

import dataclasses
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.sandbox import lockdown
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox._exec_receipt import unwrap_receipt_command

# Reuse the provenance fixtures, not root privileges or a live Docker daemon.
from tests.test_verifier_path_identities import frozen_permissions as frozen_permissions
from tests.test_verifier_path_identities import inspector_tools as inspector_tools


class LocalSetup:
    """Run the setup's real file modes; model root ownership on non-root CI."""

    async def exec(self, command, **kwargs):
        if os.getuid() != 0:
            command = command.replace("chown 0:0", "true")
            command = command.replace(
                'case $s in "0 "*)', f'case $s in "0 "*|"{os.getuid()} "*)'
            )
            command = command.replace("m=${s#0 }", "m=${s#* }")
        done = subprocess.run(
            ["/bin/sh", "-c", command],
            capture_output=True,
            text=True,
            timeout=60,
            env={"PATH": "/usr/bin:/bin", **kwargs.get("env", {})},
        )
        return ExecResult(
            stdout=done.stdout, stderr=done.stderr, return_code=done.returncode
        )


@pytest.mark.parametrize("service", ["main", "target"])
async def test_receipt_utilities_are_public_but_provenance_is_private(
    tmp_path, monkeypatch, service
):
    """Guards 5ff423c0: all verifier UIDs can traverse only encoder routes."""
    # /tmp is a writable ancestor in unit tests. The root-only container
    # replay also checks the untouched production ancestor predicate.
    monkeypatch.setattr(lockdown, "_BASH_ENV_INSPECTOR_ROOT", str(tmp_path))
    env = LocalSetup()
    original = env.exec

    async def execute(command, **kwargs):
        command = command.replace("[ $((0$m & 022)) -eq 0 ]", "true")
        return await original(command, **kwargs)

    env.exec = execute
    await lockdown._prepare_bash_env_inspector(env, service=service)
    inspector = Path(env._benchflow_path_inspectors[service])
    assert inspector.stat().st_mode & 0o777 == 0o711
    for name in ("od", "tr", "cksum"):
        assert (inspector / name).stat().st_mode & 0o777 == 0o555
    for name in ("before-links", "stat", "readlink", "sh"):
        assert (inspector / name).stat().st_mode & 0o777 == 0o700
    assert (inspector / "before").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("separator", [chr(i) for i in range(1, 32)] + [chr(127), ":"])
@pytest.mark.parametrize(
    "route", ["absolute", "relative", "chain", "directory-hop", "basename"]
)
async def test_ambiguous_extra_tool_is_omitted_without_losing_its_siblings(
    tmp_path, frozen_permissions, separator, route
):
    """Guards 5ff423c0: separators in any permission-checked hop omit one tool."""
    from benchflow._utils.trusted_path import PATH_QUERY_CMD
    from benchflow.sandbox._exec_receipt import encode_receipt
    from benchflow.sandbox._mount_aliases import PathExclusions
    from tests.test_verifier_target_aliases import mount

    tools = tmp_path / "tools with ' quote"
    safe = tmp_path / "safe"
    tools.mkdir()
    safe.mkdir()
    # The exact reported newline suffix looks like a writable ls record.
    destination = safe / ("check" + separator + "-rwxrwxrwx 1 0 0")
    destination.write_text("#!/bin/sh\nexit 0\n")
    destination.chmod(0o755)
    bad = tools / "unused"
    if route == "basename":
        bad = tools / ("unused" + separator)
        bad.symlink_to(destination)
    elif route == "directory-hop":
        hop = safe / ("hop" + separator)
        hop.symlink_to(safe, target_is_directory=True)
        bad.symlink_to(hop / destination.name)
    elif route == "chain":
        hop = safe / ("hop" + separator)
        hop.symlink_to(destination.name)
        bad.symlink_to(hop)
    else:
        bad.symlink_to(
            str(destination) if route == "absolute" else "../safe/" + destination.name
        )
    honest = tools / "honest"
    honest.write_text('#!/bin/sh\nexec /bin/cat "${0%/*}/data"\n')
    (tools / "data").write_text("honest")
    honest.chmod(0o755)

    class Image:
        async def exec(self, command, **kwargs):
            wrapped = unwrap_receipt_command(command)
            if wrapped and wrapped[0] == PATH_QUERY_CMD:
                return ExecResult(
                    stdout=encode_receipt(str(tools).encode(), 0, wrapped[1]),
                    return_code=0,
                )
            result = subprocess.run(
                ["/bin/sh", "-c", command], capture_output=True, text=True, timeout=60
            )
            return ExecResult(
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
            )

    blocked = PathExclusions.capture(mount("/", "/", "0:1", 1), ("/app",))
    env = Image()
    blocked = dataclasses.replace(blocked, inspector=lockdown._path_inspector(env))
    report = await lockdown._judge_verifier_path(env, "nobody", "/app", blocked=blocked)
    await lockdown._check_required_path_tools(
        env, report, SimpleNamespace(required_tools=None)
    )
    await lockdown._check_required_path_tools(
        env, report, SimpleNamespace(required_tools=[bad.name, "honest"], user="root")
    )
    assert report.dropped == ()
    result = subprocess.run(
        ["/bin/sh", "-c", "honest"],
        env={"PATH": report.path},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout == "honest"
    assert report.path != str(tools), "unfiltered image directory remains on PATH"
    assert report.to_record()["omitted_tools"][0]["tool"] == str(bad)
    assert not any(
        (Path(folder) / bad.name).exists() for folder in report.path.split(":")
    )


async def test_tool_omission_is_published_without_changing_the_grade(tmp_path):
    """Guards 5ff423c0: a per-tool omission records even with no folder drop."""
    import json
    from unittest.mock import AsyncMock, MagicMock

    from benchflow._utils.trusted_path import TrustedPathReport, report_trusted_path
    from benchflow.rollout._setup import _verify_rollout
    from tests.test_verifier_trusted_path import _rollout_paths, _task

    paths = _rollout_paths(tmp_path)
    report = TrustedPathReport(
        path="/usr/bin:/bin", omitted_tools=(("/opt/tools/unused", "separator"),)
    )
    verifier = MagicMock(execution_receipt=None)
    verifier.verify = AsyncMock(return_value=SimpleNamespace(rewards={"reward": 1.0}))

    async def harden(*args, **kwargs):
        report_trusted_path(report)

    rewards, error, _ = await _verify_rollout(
        SimpleNamespace(),
        _task(),
        paths,
        {},
        SimpleNamespace(harden_before_verify=harden, verifier=lambda **kw: verifier),
        "agent",
        "/app",
    )
    assert (rewards, error) == ({"reward": 1.0}, None)
    record = json.loads((paths.verifier_dir / "trusted-path.json").read_text())
    assert record["omitted_tools"] == [
        {"tool": "/opt/tools/unused", "why": "separator"}
    ]
    assert record["dropped"] == []


async def test_filtered_launchers_are_private_even_before_final_chmod(
    tmp_path, frozen_permissions, inspector_tools
):
    """Guards eb36c76d: a permissive inherited umask must expose no writable window."""
    import shlex

    from benchflow.sandbox._mount_aliases import PathExclusions
    from tests.test_verifier_target_aliases import mount

    tools = tmp_path / "tools"
    tools.mkdir()
    honest = tools / "honest"
    honest.write_text("#!/bin/sh\nexit 0\n")
    honest.chmod(0o755)
    (tools / "unused").symlink_to("honest\n")
    modes = tmp_path / "creation-modes"
    # Observe actual permissions immediately before chmod changes them.
    copied_chmod = Path(inspector_tools) / "chmod"
    copied_chmod.write_text(
        '#!/bin/sh\n/usr/bin/stat -c %a -- "$3" >> '
        + shlex.quote(str(modes))
        + '\nexec /usr/bin/chmod "$@"\n'
    )
    captured = dataclasses.replace(
        PathExclusions.capture(mount("/", "/", "0:1", 1), ("/app",)),
        inspector=inspector_tools,
    )
    command = lockdown._trusted_path_extras_cmd(str(tools), captured, "nobody")
    done = subprocess.run(
        ["/bin/sh", "-c", "umask 000; " + command],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert modes.read_text().splitlines() == ["600"]
