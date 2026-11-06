"""The LiteLLM proxy installs its locked, hashed set, and extras stay under it.

The in-sandbox proxy used to resolve ``litellm[proxy]==X`` and
``boto3>=1.40`` afresh in every trial, never audited. It now installs
``src/benchflow/providers/proxy_requirements/`` (the ``litellm-proxy``
dependency groups exported from ``uv.lock``) with ``--require-hashes``
(and ``UV_COMPILE_BYTECODE=1`` where the venv is reused); a run's extras
install with that file as constraints, are refused before any sandbox when
they would move a locked package, and are recorded in ``config.json``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from benchflow.providers import litellm_runtime as runtime_mod
from benchflow.providers import proxy_requirements as reqs
from benchflow.providers.litellm_config import resolve_litellm_route
from benchflow.rollout import Rollout, RolloutConfig
from tests.test_litellm_hardening import _FakeSandbox
from tools import export_proxy_requirements as export_tool

UV = os.environ.get("UV") or shutil.which("uv")
EXTRAS_ENV = reqs.EXTRA_REQUIREMENTS_ENV


@pytest.fixture(autouse=True)
def _no_extras_and_fresh_checks(monkeypatch):
    monkeypatch.delenv(EXTRAS_ENV, raising=False)
    reqs._CHECKED.clear()
    yield
    reqs._CHECKED.clear()
    reqs.requirements_text.cache_clear()
    reqs.locked_pins.cache_clear()


# ----- the shipped files ----------------------------------------------------


@pytest.mark.parametrize("group", reqs.GROUPS)
def test_shipped_files_match_uv_lock(group):
    """The shipped file is exactly what uv.lock exports for its group, so the
    proxy installs the audited set; a lock or group change without
    regenerating fails here with the one command that fixes it."""
    assert UV, "this check runs uv export; run the suite with `uv run pytest`"
    expected = export_tool.render(group, uv=UV)
    shipped = reqs.requirements_file(group).read_text(encoding="utf-8")
    assert shipped == expected, (
        f"{reqs.requirements_file(group).name} has drifted from uv.lock; "
        f"regenerate it on a build VM with: {reqs.REGENERATE_COMMAND}"
    )
    assert export_tool.REGENERATE_COMMAND == reqs.REGENERATE_COMMAND


def test_drift_check_names_the_regenerate_command(tmp_path, monkeypatch):
    """A stale file makes --check fail and the header names the command."""
    monkeypatch.setattr(export_tool, "OUT_DIR", tmp_path)
    monkeypatch.setattr(export_tool, "ROOT", tmp_path.parent)
    monkeypatch.setattr(export_tool, "uv_executable", lambda: "uv")
    monkeypatch.setattr(
        export_tool,
        "render",
        lambda group, uv=None: export_tool.HEADER.format(group=group),
    )
    for group in export_tool.GROUPS:
        (tmp_path / f"{group}.txt").write_text("litellm==0.0.1\n")
    assert export_tool.main(["--check"]) == 1
    assert export_tool.main([]) == 0
    assert export_tool.main(["--check"]) == 0
    text = (tmp_path / "litellm-proxy.txt").read_text()
    assert reqs.REGENERATE_COMMAND in text


def test_files_pin_every_package_with_hashes_and_the_core_litellm():
    plain = reqs.locked_pins(reqs.PROXY_GROUP)
    vertex = reqs.locked_pins(reqs.VERTEX_GROUP)
    core = Requirement(runtime_mod.LITELLM_VERSION_SPEC)
    assert core.specifier.contains(plain["litellm"])
    assert "boto3" in plain
    assert "google-cloud-aiplatform" not in plain
    assert "google-cloud-aiplatform" in vertex
    # One universal lock: the Vertex set is the plain set plus the SDK's
    # closure, at the same versions.
    assert {name: vertex[name] for name in plain} == plain
    for group in reqs.GROUPS:
        blocks = reqs.requirements_text(group).split("\n")
        pinned = [line for line in blocks if reqs._PIN_LINE.match(line)]
        assert len(pinned) == len(reqs.locked_pins(group))
        for index, line in enumerate(blocks):
            if reqs._PIN_LINE.match(line):
                assert blocks[index + 1].strip().startswith("--hash=sha256:"), line


def test_files_are_package_data_next_to_the_module():
    """Hatch ships every file under src/benchflow; the files live there."""
    package_dir = Path(reqs.__file__).parent
    names = {path.name for path in package_dir.glob("*.txt")}
    assert names == {f"{group}.txt" for group in reqs.GROUPS}


# ----- the in-sandbox install -----------------------------------------------


def _minimax_route():
    return resolve_litellm_route(
        "minimax/MiniMax-M3",
        {"MINIMAX_API_KEY": "k", "MINIMAX_BASE_URL": "https://api.minimax.io/v1"},
    )


async def _start(sandbox, route=None):
    return await runtime_mod._start_sandbox_litellm(
        sandbox=sandbox,
        route=route or _minimax_route(),
        master_key="sk-master",
        agent_env={
            "MINIMAX_API_KEY": "k",
            "MINIMAX_BASE_URL": "https://api.minimax.io/v1",
        },
        session_id="s",
        agent_name="openhands",
    )


def _install_command(sandbox) -> str:
    return next(
        command
        for command in sandbox.exec_calls
        if "pip install" in command and "litellm" in command
    )


@pytest.mark.asyncio
async def test_sandbox_install_reads_the_hashed_file():
    sandbox = _FakeSandbox()
    await _start(sandbox)

    uploaded = next(p for p in sandbox.uploaded if p.endswith("/litellm-proxy.txt"))
    assert sandbox.uploaded[uploaded] == reqs.requirements_text(reqs.PROXY_GROUP)
    command = _install_command(sandbox)
    # A one-trial venv compiles nothing ahead; see install_commands.
    assert "UV_COMPILE_BYTECODE" not in command
    assert f"--require-hashes -r {uploaded}" in command
    # No fresh resolve: neither the old spec nor an open boto3 range.
    assert "litellm[proxy]==" not in command
    assert "boto3>=" not in command
    assert " -c " not in command


@pytest.mark.asyncio
async def test_extras_install_after_the_locked_set_with_it_as_constraints(
    monkeypatch,
):
    monkeypatch.setenv(EXTRAS_ENV, "humanize==4.11.0 humanize==4.11.0")
    monkeypatch.setattr(reqs, "_resolve", lambda extras, group, uv: None)
    monkeypatch.setattr(reqs, "_uv_executable", lambda: "uv")
    sandbox = _FakeSandbox()
    await _start(sandbox)

    uploaded = next(p for p in sandbox.uploaded if p.endswith("/litellm-proxy.txt"))
    command = _install_command(sandbox)
    locked = command.index(f"--require-hashes -r {uploaded}")
    extra = command.index(f"-c {uploaded} humanize==4.11.0\n")
    assert locked < extra
    assert reqs.extra_requirements_from_env() == ("humanize==4.11.0",)
    # The pinned uv always runs the install (benchflow._uv_pin), so the
    # sandbox command has no pip branch left.
    assert "-m pip install" not in command


def test_a_reused_venv_compiles_bytecode_and_a_one_trial_venv_does_not():
    args = {"uv": "uv", "python": "py", "requirements_path": "req.txt"}
    reused = reqs.install_commands(**args, extras=("humanize",)).splitlines()
    assert len(reused) == 2
    assert all(
        line.startswith("UV_COMPILE_BYTECODE=1 uv pip install") for line in reused
    )
    once = reqs.install_commands(**args, extras=("humanize",), compile_bytecode=False)
    assert "UV_COMPILE_BYTECODE" not in once
    assert "--require-hashes -r req.txt" in once
    assert "-c req.txt humanize" in once


def test_both_install_lines_take_wheels_only():
    """A hash-checked sdist still builds with a backend uv resolves apart
    from the file and checks against no hash."""
    args = {"uv": "uv", "python": "py", "requirements_path": "req.txt"}
    lines = reqs.install_commands(**args, extras=("humanize",)).splitlines()
    assert len(lines) == 2
    for line in lines:
        assert " pip install " in line and " --no-build " in line


# ----- refusing extras that would move a locked package ---------------------


def test_extra_pinning_another_litellm_is_refused_naming_both():
    with pytest.raises(reqs.ProxyExtraRequirementError) as raised:
        reqs.check_extra_requirements(("litellm==1.80.0",), reqs.PROXY_GROUP)
    message = str(raised.value)
    locked = reqs.locked_pins(reqs.PROXY_GROUP)["litellm"]
    assert "'litellm==1.80.0'" in message
    assert f"litellm=={locked}" in message


def test_extra_whose_dependency_needs_another_locked_version_is_refused(
    monkeypatch,
):
    botocore = reqs.locked_pins(reqs.PROXY_GROUP)["botocore"]
    uv_message = (
        "error: No solution found when resolving dependencies\n"
        "  cause: Because aiobotocore==2.4.0 depends on botocore>=1.27.59,"
        f"<1.27.60 and botocore=={botocore}, we can conclude that "
        "aiobotocore==2.4.0 cannot be used."
    )
    calls: list[tuple[str, ...]] = []

    def fake_resolve(extras, group, uv):
        calls.append(extras)
        return uv_message if "aiobotocore==2.4.0" in extras else None

    monkeypatch.setattr(reqs, "_resolve", fake_resolve)
    with pytest.raises(reqs.ProxyExtraRequirementError) as raised:
        reqs.check_extra_requirements(
            ("humanize==4.11.0", "aiobotocore==2.4.0"), reqs.PROXY_GROUP, uv="uv"
        )
    message = str(raised.value)
    assert "'aiobotocore==2.4.0'" in message
    assert "'humanize==4.11.0'" not in message
    assert f"the locked botocore=={botocore}" in message


@pytest.mark.parametrize(
    "extra", ["not a requirement!", "evil @ https://example.invalid/evil.whl"]
)
def test_extras_that_are_not_index_requirements_are_refused(extra):
    with pytest.raises(reqs.ProxyExtraRequirementError, match=EXTRAS_ENV):
        reqs.check_extra_requirements((extra,), reqs.PROXY_GROUP, uv="uv")


@pytest.mark.parametrize(
    "extra",
    [
        "mypkg@https://someone:fake-secret@pkgs.example.invalid/mypkg.whl",
        "https://someone:fake-secret@pkgs.example.invalid/mypkg.whl",
        "https://pkgs.example.invalid/mypkg.whl?token=fake-secret",
        "https://pkgs.example.invalid/simple/fake-secret/mypkg.whl",
    ],
    ids=["url-requirement", "bare-url", "query-token", "path-token"],
)
def test_a_refused_url_extra_never_repeats_its_credentials(extra):
    with pytest.raises(reqs.ProxyExtraRequirementError) as raised:
        reqs.check_extra_requirements((extra,), reqs.PROXY_GROUP, uv="uv")
    assert "fake-secret" not in str(raised.value)
    assert EXTRAS_ENV in str(raised.value)
    # Nor does a chained parse error shown in a traceback.
    error = raised.value
    assert error.__cause__ is None
    assert error.__context__ is None or error.__suppress_context__


def test_an_unreachable_index_is_not_reported_as_a_conflict(monkeypatch):
    calls: list[tuple[str, ...]] = []

    def unreachable(extras, group, uv):
        calls.append(extras)
        return (
            "error: Failed to fetch: `https://pypi.example.invalid/simple/humanize/`\n"
            "  Caused by: dns error: failed to lookup address information"
        )

    monkeypatch.setattr(reqs, "_resolve", unreachable)
    extras = ("humanize", "google-genai>=1.0")
    with pytest.raises(reqs.ProxyExtraRequirementError) as raised:
        reqs.check_extra_requirements(extras, reqs.PROXY_GROUP, uv="uv")
    message = str(raised.value)
    assert message.startswith(f"could not check {EXTRAS_ENV}")
    assert "dns error" in message
    assert "conflict" not in message
    # No per-extra re-resolve, and a later check tries the index again.
    assert calls == [extras]
    with pytest.raises(reqs.ProxyExtraRequirementError, match="could not check"):
        reqs.check_extra_requirements(extras, reqs.PROXY_GROUP, uv="uv")
    assert calls == [extras, extras]


def test_the_resolve_runs_once_per_job(monkeypatch):
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        reqs, "_resolve", lambda extras, group, uv: calls.append(extras)
    )
    for _ in range(3):
        reqs.check_extra_requirements(("humanize",), reqs.PROXY_GROUP, uv="uv")
    assert calls == [("humanize",)]


@pytest.mark.asyncio
async def test_conflicting_extra_never_reaches_the_sandbox(monkeypatch):
    monkeypatch.setenv(EXTRAS_ENV, "litellm==1.80.0")
    sandbox = _FakeSandbox()
    with pytest.raises(reqs.ProxyExtraRequirementError):
        await _start(sandbox)
    assert sandbox.exec_calls == []
    assert sandbox.uploaded == {}


def _task(tmp_path: Path) -> Path:
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM alpine:3.20\n")
    (task / "instruction.md").write_text("solve\n")
    (task / "task.toml").write_text('version = "1.0"\n')
    return task


def _rollout(tmp_path: Path, environment: str = "docker"):
    planes = MagicMock()
    planes.resolve_locked_paths.return_value = []
    planes.resolve_agent_env.return_value = {"ANTHROPIC_API_KEY": "fake-key"}
    planes.agent_launch.return_value = "claude-agent-acp"
    planes.create_environment.return_value = MagicMock()
    rollout = Rollout(
        RolloutConfig.from_legacy(
            task_path=_task(tmp_path),
            agent="claude-agent-acp",
            model="anthropic/claude-haiku-4-5",
            environment=environment,
            jobs_dir=tmp_path / "jobs",
            planes=planes,
        )
    )
    return rollout, planes


@pytest.mark.asyncio
async def test_config_json_records_the_locked_file_and_the_extras(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(EXTRAS_ENV, "humanize==4.11.0")
    monkeypatch.setattr(reqs, "_resolve", lambda extras, group, uv: None)
    monkeypatch.setattr(reqs, "_uv_executable", lambda: "uv")
    rollout, _ = _rollout(tmp_path, environment="daytona")
    await rollout.setup()

    config = json.loads((rollout._rollout_dir / "config.json").read_text())
    assert config["litellm_proxy"] == {
        "location": "sandbox",
        "requirements": "litellm-proxy.txt",
        "requirements_sha256": reqs.requirements_sha256(reqs.PROXY_GROUP),
        "extra_requirements": ["humanize==4.11.0"],
    }


@pytest.mark.asyncio
async def test_the_setup_check_resolves_off_the_event_loop(tmp_path, monkeypatch):
    """The resolve can take minutes; other trials on the loop keep running."""
    import threading

    threads: list[threading.Thread] = []

    def resolve(extras, group, uv):
        threads.append(threading.current_thread())

    monkeypatch.setenv(EXTRAS_ENV, "humanize==4.11.0")
    monkeypatch.setattr(reqs, "_resolve", resolve)
    monkeypatch.setattr(reqs, "_uv_executable", lambda: "uv")
    rollout, _ = _rollout(tmp_path, environment="daytona")
    await rollout.setup()
    assert threads
    assert threading.main_thread() not in threads


@pytest.mark.asyncio
async def test_config_json_says_a_plain_host_proxy_installs_nothing(tmp_path):
    rollout, _ = _rollout(tmp_path)
    await rollout.setup()
    config = json.loads((rollout._rollout_dir / "config.json").read_text())
    assert config["litellm_proxy"] == {
        "location": "host",
        "requirements": None,
        "requirements_sha256": None,
        "extra_requirements": [],
    }


@pytest.mark.asyncio
async def test_conflicting_extra_is_refused_before_any_sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv(EXTRAS_ENV, "litellm==1.80.0")
    rollout, planes = _rollout(tmp_path)
    with pytest.raises(reqs.ProxyExtraRequirementError, match=r"litellm==1\.80\.0"):
        await rollout.setup()
    planes.create_environment.assert_not_called()


@pytest.mark.asyncio
async def test_a_job_refuses_a_conflicting_extra_before_any_trial(
    tmp_path, monkeypatch
):
    from benchflow.errors import UserError
    from benchflow.evaluation import Evaluation, EvaluationConfig

    monkeypatch.setenv(EXTRAS_ENV, "litellm==1.80.0")

    async def no_trial(*args, **kwargs):
        raise AssertionError("a trial started")

    monkeypatch.setattr(Rollout, "setup", no_trial)
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    _task(tasks).rename(tasks / "only-task")
    job = Evaluation(
        tasks_dir=tasks,
        jobs_dir=tmp_path / "jobs",
        job_name="extras",
        preflight=False,
        config=EvaluationConfig(
            agent="claude-agent-acp",
            model="anthropic/claude-haiku-4-5",
            agent_env={"ANTHROPIC_API_KEY": "fake-key"},
        ),
    )
    with pytest.raises(reqs.ProxyExtraRequirementError) as raised:
        await job.run()
    assert isinstance(raised.value, UserError)
    assert "'litellm==1.80.0'" in str(raised.value)
    assert not list((tmp_path / "jobs").rglob("config.json"))


# ----- a real install: an extra cannot move a locked package ----------------


def _wheel(directory: Path, name: str, version: str, requires: tuple[str, ...] = ()):
    """A minimal pure-Python wheel; returns (path, sha256)."""
    dist = f"{name}-{version}.dist-info"
    files = {
        f"{name}/__init__.py": f"VERSION = {version!r}\n",
        f"{dist}/METADATA": "Metadata-Version: 2.1\n"
        f"Name: {name}\nVersion: {version}\n"
        + "".join(f"Requires-Dist: {r}\n" for r in requires),
        f"{dist}/WHEEL": "Wheel-Version: 1.0\nGenerator: test\n"
        "Root-Is-Purelib: true\nTag: py3-none-any\n",
    }
    record = []
    for path, text in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(text.encode()).digest())
        record.append(f"{path},sha256={digest.decode().rstrip('=')},{len(text)}")
    record.append(f"{dist}/RECORD,,")
    files[f"{dist}/RECORD"] = "\n".join(record) + "\n"
    wheel = directory / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for path, text in files.items():
            archive.writestr(path, text)
    return wheel, hashlib.sha256(wheel.read_bytes()).hexdigest()


def _freeze(python: Path) -> dict[str, str]:
    out = subprocess.run(
        [UV, "pip", "freeze", "--python", str(python)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return dict(line.split("==", 1) for line in out.splitlines() if "==" in line)


@pytest.mark.skipif(UV is None, reason="needs uv (the suite runs under uv run)")
def test_an_extra_installs_on_top_without_moving_a_locked_version(tmp_path):
    """The real install commands, offline against a local wheel index: the
    locked set installs hash-checked; an extra installs beside it; an extra
    that wants another version of a locked package fails, and nothing that
    was installed moves. Without the constraints file uv would upgrade it."""
    index = tmp_path / "index"
    index.mkdir()
    _, base_sha = _wheel(index, "lockedbase", "1.0")
    _wheel(index, "lockedbase", "2.0")
    _wheel(index, "goodextra", "1.0", ("lockedbase>=1",))
    _wheel(index, "badextra", "1.0", ("lockedbase>=2",))
    locked = tmp_path / "locked.txt"
    locked.write_text(f"lockedbase==1.0 \\\n    --hash=sha256:{base_sha}\n")
    venv = tmp_path / "venv"
    subprocess.run(
        [UV, "venv", "-q", "--python", sys.executable, str(venv)], check=True
    )
    env = {
        **os.environ,
        "UV": UV,
        "UV_OFFLINE": "1",
        "UV_NO_INDEX": "1",
        "UV_FIND_LINKS": str(index),
        "UV_CACHE_DIR": str(tmp_path / "cache"),
    }

    def install(extras: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        script = reqs.install_commands(
            uv='"$UV"',
            python=str(venv / "bin" / "python"),
            requirements_path=str(locked),
            extras=extras,
        )
        return subprocess.run(
            ["sh", "-c", "set -eu\n" + script],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    assert install(()).returncode == 0
    before = _freeze(venv / "bin" / "python")
    assert before == {"lockedbase": "1.0"}

    good = install(("goodextra",))
    assert good.returncode == 0, good.stderr
    after = _freeze(venv / "bin" / "python")
    assert after == {"lockedbase": "1.0", "goodextra": "1.0"}

    bad = install(("badextra",))
    assert bad.returncode != 0
    assert "lockedbase" in bad.stderr
    assert _freeze(venv / "bin" / "python") == after

    # The control: the same extra with no constraints moves the locked one.
    unconstrained = subprocess.run(
        [
            UV,
            "pip",
            "install",
            "-q",
            "--python",
            str(venv / "bin" / "python"),
            "badextra",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert unconstrained.returncode == 0, unconstrained.stderr
    assert _freeze(venv / "bin" / "python")["lockedbase"] == "2.0"

    # A tampered locked wheel fails the hash check.
    tampered = tmp_path / "tampered.txt"
    tampered.write_text(f"lockedbase==1.0 \\\n    --hash=sha256:{'0' * 64}\n")
    fresh = tmp_path / "fresh"
    subprocess.run(
        [UV, "venv", "-q", "--python", sys.executable, str(fresh)], check=True
    )
    result = subprocess.run(
        [
            "sh",
            "-c",
            reqs.install_commands(
                uv='"$UV"',
                python=str(fresh / "bin" / "python"),
                requirements_path=str(tampered),
                extras=(),
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "hash" in result.stderr.lower()


def test_pip_fallback_installs_an_extra_under_the_locked_set(tmp_path):
    """An image without uv installs with pip, the fallback's real lines,
    offline against a local wheel index. pip turns on --require-hashes for a
    whole command when any constraint has a hash, so with the hashed file as
    constraints every extra was refused; with its pins only, the locked set
    installs hash-checked, an extra installs beside it, and an extra that
    would move a locked version still fails."""
    index = tmp_path / "index"
    index.mkdir()
    _, base_sha = _wheel(index, "lockedbase", "1.0")
    _wheel(index, "lockedbase", "2.0")
    _wheel(index, "goodextra", "1.0", ("lockedbase>=1",))
    _wheel(index, "badextra", "1.0", ("lockedbase>=2",))
    locked = tmp_path / "locked.txt"
    locked.write_text(
        "# exported\n"
        f"lockedbase==1.0 ; python_version >= '3' \\\n"
        f"    --hash=sha256:{base_sha} \\\n"
        f"    --hash=sha256:{'f' * 64}\n"
        "    # via goodextra\n"
    )
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    python = venv / "bin" / "python"
    env = {
        **os.environ,
        "PIP_NO_INDEX": "1",
        "PIP_FIND_LINKS": str(index),
        "PIP_CACHE_DIR": str(tmp_path / "pip-cache"),
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    }

    def install(extras: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        script = reqs.pip_install_commands(
            python=str(python),
            requirements_path=str(locked),
            constraints_path=str(tmp_path / "locked.txt.pins"),
            extras=extras,
        )
        return subprocess.run(
            ["sh", "-c", "set -eu\n" + script],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def freeze() -> dict[str, str]:
        out = subprocess.run(
            [str(python), "-m", "pip", "freeze"],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return dict(line.split("==", 1) for line in out.splitlines() if "==" in line)

    first = install(())
    assert first.returncode == 0, first.stderr
    assert freeze() == {"lockedbase": "1.0"}

    good = install(("goodextra",))
    assert good.returncode == 0, good.stderr
    assert freeze() == {"lockedbase": "1.0", "goodextra": "1.0"}

    # pip -q names only the extra and the conflict; nothing moved.
    bad = install(("badextra",))
    assert bad.returncode != 0
    assert "badextra" in bad.stderr
    assert "conflicting dependencies" in bad.stderr
    assert freeze() == {"lockedbase": "1.0", "goodextra": "1.0"}


@pytest.mark.parametrize("group", reqs.GROUPS)
def test_pip_constraints_are_the_shipped_pins_without_hashes(tmp_path, group):
    """The pins-only copy the pip fallback writes from each shipped file holds
    every locked pin, with its marker, and no hash or continuation."""
    out = tmp_path / "pins.txt"
    script = reqs.pip_install_commands(
        python="true",
        requirements_path=str(reqs.requirements_file(group)),
        constraints_path=str(out),
        extras=("anything",),
    )
    sed_line = next(line for line in script.splitlines() if line.startswith("sed "))
    subprocess.run(["sh", "-c", "set -eu\n" + sed_line], check=True)
    text = out.read_text()
    assert "--hash" not in text
    assert "\\" not in text
    requirements = [
        Requirement(line)
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    pins = {canonicalize_name(req.name): str(req.specifier) for req in requirements}
    assert pins == {
        name: f"=={version}" for name, version in reqs.locked_pins(group).items()
    }
    assert any(req.marker for req in requirements)


@pytest.mark.skipif(UV is None, reason="needs uv (the suite runs under uv run)")
def test_host_venv_with_extras_is_built_once_and_never_half_built(
    tmp_path, monkeypatch
):
    """A host proxy's own venv is built with hash-checked commands and removed
    on failure. FF-install-C7 replaces reuse by key with fresh environments:
    the extra's unresolved closure can change without changing its string."""
    index = tmp_path / "index"
    index.mkdir()
    _, base_sha = _wheel(index, "lockedbase", "1.0")
    _wheel(index, "lockedbase", "2.0")
    _wheel(index, "goodextra", "1.0", ("lockedbase>=1",))
    _wheel(index, "badextra", "1.0", ("lockedbase>=2",))
    locked = tmp_path / "litellm-proxy.txt"
    locked.write_text(f"lockedbase==1.0 \\\n    --hash=sha256:{base_sha}\n")
    monkeypatch.setattr(reqs, "requirements_file", lambda group: locked)
    reqs.requirements_text.cache_clear()
    reqs.locked_pins.cache_clear()
    monkeypatch.setattr(reqs, "PROXY_PYTHON", sys.executable)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache-home"))
    for name, value in {
        "UV_OFFLINE": "1",
        "UV_NO_INDEX": "1",
        "UV_FIND_LINKS": str(index),
        "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
    }.items():
        monkeypatch.setenv(name, value)

    good = reqs.ProxyInstall(reqs.PROXY_GROUP, ("goodextra",))
    venv = reqs.ensure_host_venv(good, uv=UV)
    assert venv.parent == tmp_path / "cache-home" / "benchflow" / "litellm-proxy"
    assert _freeze(venv / "bin" / "python") == {
        "lockedbase": "1.0",
        "goodextra": "1.0",
    }
    marker = venv / ".benchflow-complete"
    built = marker.stat().st_mtime_ns
    second = reqs.ensure_host_venv(good, uv=UV)
    assert second != venv
    assert _freeze(second / "bin" / "python") == _freeze(venv / "bin" / "python")
    assert marker.stat().st_mtime_ns == built

    bad = reqs.ProxyInstall(reqs.PROXY_GROUP, ("badextra",))
    with pytest.raises(RuntimeError, match="host LiteLLM proxy venv"):
        reqs.ensure_host_venv(bad, uv=UV)
    leftovers = [p.name for p in venv.parent.iterdir() if p.is_dir()]
    assert set(leftovers) == {venv.name, second.name}


def test_an_extras_check_that_cannot_run_refuses(tmp_path):
    missing_uv = str(tmp_path / "no-such-uv")
    with pytest.raises(reqs.ProxyExtraRequirementError, match="could not check"):
        reqs.check_extra_requirements(("humanize",), reqs.PROXY_GROUP, uv=missing_uv)
