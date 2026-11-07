"""Behavioral regressions for U5a-CONF-2/3/4 at 7b2c5334."""

import os
import re
import shlex
import shutil
import subprocess

import pytest

from benchflow.sandbox import lockdown
from tests.test_verifier_prep_hooks import _sandbox_user_site
from tests.test_verifier_prep_path import backend as backend


@pytest.mark.parametrize("baseline_present", [False, True])
async def test_posix_purge_keeps_baseline_caches(backend, tmp_path, baseline_present):
    """Guards CONF-3 from 7b2c5334, including newline/glob/quote-bearing paths."""
    box, _, _ = backend
    workspace = tmp_path / "work [x]*? ' \n"
    baseline = tmp_path / "baseline"
    workspace.mkdir()
    if baseline_present:
        baseline.mkdir()
    for parent in ("plain", "space name", "line\nend\n", "[abc]*?", "'quote", "-dash"):
        for name in ("old", "new"):
            rel = f"{parent}/{name}/__pycache__"
            folder = workspace / rel
            folder.mkdir(parents=True)
            (folder / "marker").write_text(name)
            if name == "old" and baseline_present:
                (baseline / rel).mkdir(parents=True)
                # The image's own cache file: V8's purge keeps only the
                # files a baseline cache holds the same.
                (baseline / rel / "marker").write_text(name)
    command = lockdown._purge_pycache_cmd(str(workspace)).replace(
        "/testbed_verify", str(baseline)
    )
    box.verifier_prep_script = True
    await lockdown._run_prep_steps(
        box, [lockdown._PrepStep("purge", "purge", command, 10)]
    )
    for folder in workspace.glob("*/old/__pycache__"):
        assert baseline_present
        assert (folder / "marker").read_text() == "old"
    assert len(list(workspace.glob("*/old/__pycache__"))) == (
        6 if baseline_present else 0
    )
    assert not list(workspace.glob("*/new/__pycache__"))


@pytest.mark.parametrize("kind", ["checked", "unchecked", "best_effort"])
async def test_no_space_env_allocation_still_runs_fallback(
    backend, tmp_path, monkeypatch, kind
):
    """Guards CONF-4 from 7b2c5334: no env allocation precedes the fallback."""
    box, _, calls = backend
    box.verifier_prep_script = True
    # /dev/full is the kernel's deterministic ENOSPC writer. The same env
    # materialization used by each backend must not be needed for fixed settings.
    # Docker and remote Docker carry the env on the exec's stdin (main-sync
    # b156) and allocate no env file, so only the file wrappers are filled.
    from benchflow.sandbox import _base, daytona, daytona_dind, daytona_pty

    original = _base.wrap_command_with_env_file

    def full(env, command, **kwargs):
        wrapped = original(env, command, env_path_prefix=str(tmp_path / "exec-env-"))
        return re.sub(r"> [^ ]+\) &&", "> /dev/full) &&", wrapped, count=1)

    monkeypatch.setattr(daytona_pty, "wrap_command_with_env_file", full)
    monkeypatch.setattr(daytona, "_wrap_daytona_command_with_env_file", full)
    monkeypatch.setattr(daytona_dind, "_wrap_daytona_command_with_env_file", full)
    # Batch allocation fails without starting a step. Its valid receipt must
    # still be available; the individual commands need no files to launch.
    monkeypatch.setattr(lockdown, "_PREP_DIR_PREFIX", "/dev/full/prep-")
    marker = tmp_path / "fallback-ran"
    step = lockdown._PrepStep(
        "probe",
        "probe",
        f"printf honest > {shlex.quote(str(marker))}",
        5,
        "probe failed" if kind == "checked" else None,
        best_effort=kind == "best_effort",
    )
    await lockdown._run_prep_steps(box, [step])
    assert marker.read_text() == "honest"
    assert len(calls) == 2


@pytest.mark.parametrize("helper", ["reclaim", "symlinks", "restore"])
@pytest.mark.parametrize("source", ["cwd", "pythonpath", "site"])
def test_stdlib_prep_helpers_do_not_import_agent_modules(tmp_path, helper, source):
    """Guards CONF-2 at 7b2c5334 for helpers whose isolation is retained.

    Hook cleanup's authorized CONF-5 rollback is covered separately: it must
    retain the original directories, and its startup-import gap stays open.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    planted = workspace if source == "cwd" else tmp_path / "modules"
    planted.mkdir(exist_ok=True)
    marker = tmp_path / "import-ran"
    modules = {
        "reclaim": ("glob",),
        "symlinks": ("pathlib",),
        "restore": ("shutil",),
    }[helper]
    for module in modules:
        (planted / f"{module}.py").write_text(
            f"open({str(marker)!r}, 'w').write('dummy')\nraise RuntimeError('dummy')\n"
        )
    env = {**os.environ, "PATH": lockdown._SAFE_VERIFIER_PATH}
    if source == "pythonpath":
        env["PYTHONPATH"] = str(planted)
    if source == "site":
        env["PYTHONUSERBASE"] = str(planted)
        # The user site of the python3 the helpers run (on the verifier's
        # PATH), not the test runner's: on a 3.13 runner with the distro's
        # 3.12 a runner-versioned folder is never searched, and the test
        # would pass without planting anything.
        site = _sandbox_user_site(env)
        site.mkdir(parents=True)
        for module in modules:
            (site / f"{module}.py").write_bytes((planted / f"{module}.py").read_bytes())
    if helper == "reclaim":
        command = lockdown.build_reclaim_caches_cmd(str(workspace))
        # Run the exact snippet against a hermetic fake filesystem.
        ws = shlex.quote(str(workspace))
        command = command.replace(
            f"{ws}); else", f"{ws} {shlex.quote(str(tmp_path / 'fs'))}); else"
        )
    elif helper == "symlinks":
        command = lockdown._purge_external_symlinks_cmd(str(workspace))
    else:
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        (baseline / "answer").write_text("honest")
        command = lockdown._restore_workspace_cmd(str(workspace)).replace(
            "/testbed_verify", str(baseline)
        )
        command = command.replace("rsync -a --delete", "false")
    # The interpreter the host judged, as the root route names it.
    judged = shutil.which("python3", path=env["PATH"])
    assert judged, "no python3 on the verifier PATH"
    command = f"__bf_root_python={shlex.quote(judged)}; " + command
    subprocess.run(
        ["/bin/sh", "-c", command],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert not marker.exists(), marker.read_text() if marker.exists() else ""
    if helper == "restore":
        assert (workspace / "answer").read_text() == "honest"


@pytest.mark.parametrize("fallback", [False, True])
async def test_fixed_settings_win_after_persistent_source(
    backend, tmp_path, monkeypatch, fallback
):
    """Guards CONF-4 at 7b2c5334 without reopening CONF-1 on either path."""
    from benchflow.sandbox._base import SYSTEM_SHELL_UNSET_VARS

    box, _, _ = backend
    box.verifier_prep_script = True
    box._persistent_env = {key: "/tmp/dummy-u5a" for key in SYSTEM_SHELL_UNSET_VARS}
    box._persistent_env["PATH"] = "/tmp/dummy-u5a"
    if fallback:
        monkeypatch.setattr(lockdown, "_PREP_DIR_PREFIX", "/dev/full/prep-")
    checks = [f'[ -z "${{{key}-}}" ]' for key in SYSTEM_SHELL_UNSET_VARS]
    # PATH stays the route's (C8); the persistent value never
    # becomes it.
    checks += ['case ":$PATH:" in *:/tmp/dummy-u5a:*) false;; esac']
    await lockdown._run_prep_steps(
        box,
        [
            lockdown._PrepStep(
                "env", "env", " && ".join(checks), 5, "fixed settings lost"
            )
        ],
    )
