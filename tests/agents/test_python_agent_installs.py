"""The three agents BenchFlow installs from Python packages install from pins.

openhands, deepagents and harvey-lab-harness used to drift: openhands passed
``--refresh`` (every install revalidated every cached package), deepagents
installed whatever ``deepagents`` and ``langchain-openai`` were newest, and
harvey-lab cloned the head of ``harveyai/harvey-labs`` and ran
``python -m pip install`` on nine unpinned packages. A different agent version
can change a score, so these tests pin the generated install commands: exact
sources and versions, uv rather than pip, and no cache revalidation.

The tests that check the command text use literals and import only the
registry, so they fail on a tree without the pins by what the command says,
not by an import error.
"""

from __future__ import annotations

import base64
import os
import re
import shutil
import subprocess
import sys
import zipfile
import zlib

import pytest

from benchflow.agents.registry import AGENTS
from tests._root_stand_in import run_sh

_PYTHON_AGENTS = ("openhands", "deepagents", "harvey-lab-harness")
_HARVEY_COMMIT = "a2b429eb6c9683c4fdeced3bc6b3af36edf239a6"
# One requirement line: an exact version, optionally an environment marker.
_PIN_LINE = re.compile(r"^[a-z0-9][a-z0-9._-]*==[A-Za-z0-9.+!_-]+( ; .+)?$")


def _has_arg(cmd: str, requirement: str) -> bool:
    """Read the compressed requirement payload carried by the install."""
    match = re.search(r"printf '%s' ([A-Za-z0-9+/=]+) \|", cmd)
    assert match, "the install must carry its dependency lock"
    text = zlib.decompress(base64.b64decode(match[1])).decode()
    return any(line.split()[0] == requirement for line in text.splitlines())


@pytest.mark.parametrize("agent", _PYTHON_AGENTS)
def test_install_never_revalidates_the_uv_cache(agent):
    assert "--refresh" not in AGENTS[agent].install_cmd


@pytest.mark.parametrize("agent", _PYTHON_AGENTS)
def test_venv_python_is_uv_managed_not_the_images(agent):
    """The venv runs on a uv-managed CPython the sandbox user can read."""
    cmd = AGENTS[agent].install_cmd
    assert "export UV_PYTHON_PREFERENCE=only-managed" in cmd
    assert "UV_PYTHON_INSTALL_DIR=/opt/runtime/python" in cmd
    assert "UV_PYTHON_DOWNLOADS=automatic" in cmd
    assert "--python 3.12.15" in cmd
    assert "umask 022;" in cmd


@pytest.mark.parametrize("agent", ["deepagents", "harvey-lab-harness"])
def test_install_can_run_twice_in_one_sandbox(agent):
    """uv 0.13 refuses to create a venv over an existing one without a flag.

    On deepagents that stopped a re-run before the package install, while the
    install still exited 0, so the pins never reached a venv already there.
    """
    assert "uv venv -q --clear --python 3.12.15 " in AGENTS[agent].install_cmd


@pytest.mark.parametrize("agent", _PYTHON_AGENTS)
def test_install_uses_uv_not_pip(agent):
    cmd = AGENTS[agent].install_cmd
    assert "-m pip" not in cmd
    assert "pip3" not in cmd
    # Every pip-style install goes through uv.
    assert re.search(r"(?<!uv )\bpip install\b", cmd) is None


def test_openhands_keeps_its_pinned_source_and_sdk_overrides():
    cmd = AGENTS["openhands"].install_cmd
    assert (
        "uv tool install --force "
        "--overrides /opt/runtime/agents/oh-sdk-overrides.txt "
        "--from "
        "'git+https://github.com/OpenHands/OpenHands-CLI.git@"
        "283c6ac3610f4299f405f7389f65766dcb83e12a' "
        "openhands --python 3.12" in cmd
    )
    assert "openhands-sdk==1.28.1" in cmd
    assert "openhands-tools==1.28.1" in cmd
    # Rust's PyPI pin would pull SDK 1.21.0, a downgrade.
    assert "openhands==1.16.0" not in cmd


def test_deepagents_installs_exact_versions_from_a_requirements_file():
    cmd = AGENTS["deepagents"].install_cmd
    assert _has_arg(cmd, "deepagents==0.7.26")
    assert _has_arg(cmd, "langchain-openai==1.7.2")
    # Transitive packages are pinned too, not only the two named ones.
    assert _has_arg(cmd, "langchain-core==1.6.11")
    assert _has_arg(cmd, "langgraph==1.2.15")
    assert "> /opt/runtime/agents/deepagents-requirements.txt" in cmd
    assert (
        "uv pip install -q --require-hashes --python /opt/runtime/agents/deepagents-venv/bin/python "
        "-r /opt/runtime/agents/deepagents-requirements.txt" in cmd
    )
    # The old unpinned spelling is gone.
    assert "deepagents langchain-openai" not in cmd


def test_deepagents_install_checks_the_names_the_shim_imports():
    cmd = AGENTS["deepagents"].install_cmd
    assert "from deepagents import create_deep_agent" in cmd
    assert "from deepagents.backends import LocalShellBackend" in cmd
    assert "from langchain_openai import ChatOpenAI" in cmd


def test_harvey_lab_fetches_a_pinned_commit_not_the_head():
    cmd = AGENTS["harvey-lab-harness"].install_cmd
    assert "git clone" not in cmd
    assert (
        "fetch -q --depth 1 --filter=blob:none "
        f"https://github.com/harveyai/harvey-labs.git {_HARVEY_COMMIT}" in cmd
    )
    assert f"checkout -q --force {_HARVEY_COMMIT}" in cmd
    assert f'"{_HARVEY_COMMIT}" ]' in cmd
    assert "sparse-checkout set harness sandbox utils" in cmd
    assert "test -f /opt/runtime/agents/harvey-labs/harness/tools.py" in cmd


def test_harvey_lab_installs_exact_versions_with_uv_on_managed_python():
    cmd = AGENTS["harvey-lab-harness"].install_cmd
    assert "python3 -m venv" not in cmd
    assert (
        "uv venv -q --clear --python 3.12.15 /opt/runtime/agents/harvey-lab-venv" in cmd
    )
    for pin in (
        "anthropic==0.88.0",
        "openai==2.30.0",
        "google-genai==1.70.0",
        "python-docx==1.2.0",
        "pdfplumber==0.11.9",
        "openpyxl==3.1.5",
        "python-pptx==1.0.2",
        "markitdown==0.1.5",
        "pandas==3.0.2",
    ):
        assert _has_arg(cmd, pin)
    assert (
        "uv pip install -q --require-hashes --python /opt/runtime/agents/harvey-lab-venv/bin/python "
        "-r /opt/runtime/agents/harvey-lab-requirements.txt" in cmd
    )
    assert AGENTS["harvey-lab-harness"].launch_cmd.startswith(
        "HARVEY_LABS_ROOT=/opt/runtime/agents/harvey-labs /opt/runtime/agents/harvey-lab-venv/bin/python "
    )


@pytest.mark.parametrize(
    ("agent", "name"),
    [
        ("deepagents", "DEEPAGENTS_REQUIREMENTS"),
        ("harvey-lab-harness", "HARVEY_LAB_REQUIREMENTS"),
    ],
)
def test_every_requirement_is_an_exact_pin_written_verbatim(agent, name, tmp_path):
    from benchflow.agents import _python_agent_pins as pins
    from benchflow.agents.registry import _write_requirements

    requirements = getattr(pins, name)
    assert requirements
    for line in requirements:
        assert _PIN_LINE.match(line.split(" --hash=", 1)[0]), line
        hashes = re.findall(r"--hash=sha256:([0-9a-f]{64})(?: |$)", line)
        assert hashes, line
    names = [re.split(r"[=; ]", line, maxsplit=1)[0] for line in requirements]
    assert len(names) == len(set(names))

    # The install writes exactly these lines; run the same snippet on a temp
    # path and read the file back.
    target = tmp_path / "pins" / "requirements.txt"
    snippet = _write_requirements(str(target), requirements)
    assert (
        _write_requirements(
            f"/opt/runtime/agents/{'deepagents' if agent == 'deepagents' else 'harvey-lab'}"
            "-requirements.txt",
            requirements,
            python=f"/opt/runtime/agents/{'deepagents' if agent == 'deepagents' else 'harvey-lab'}-venv/bin/python",
        )
        in AGENTS[agent].install_cmd
    )
    run_sh(snippet, shell="sh", check=True)
    assert target.read_text().splitlines() == list(requirements)


def test_harvey_lab_pins_name_the_commit_the_install_fetches():
    from benchflow.agents import _python_agent_pins as pins

    assert pins.HARVEY_LAB_COMMIT == _HARVEY_COMMIT
    assert re.fullmatch(r"[0-9a-f]{40}", pins.HARVEY_LAB_COMMIT)


@pytest.mark.parametrize("agent", _PYTHON_AGENTS)
@pytest.mark.parametrize("failure", ["bootstrap", "install"])
def test_failed_required_step_cannot_use_an_old_agent(agent, failure, tmp_path):
    """Guards fix 7cdcfed0 (U4-C1): an old agent cannot hide setup failure."""
    from benchflow._uv_pin import (
        UV_BIN,
        managed_python_install_command,
        uv_install_command,
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # No downloads or user/permission changes: only the install's shell control
    # flow is exercised. All writes are redirected into this temporary tree.
    commands = {
        "curl": "exit 0",
        "git": "exit 0",
        "chmod": "exit 0",
        "openhands": "exit 0",
        "uv": (
            'case "$*" in "pip install"*|"tool install"*) exit 42;; '
            '"tool list") echo "openhands v1.0";; esac; exit 0'
        ),
    }
    for name, body in commands.items():
        path = bin_dir / name
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(0o755)
    old_python = tmp_path / "opt/runtime/agents/deepagents-venv/bin/python"
    old_python.parent.mkdir(parents=True)
    old_python.write_text("#!/bin/sh\nexit 0\n")
    old_python.chmod(0o755)
    harvey_python = tmp_path / "opt/runtime/agents/harvey-lab-venv/bin/python"
    harvey_python.parent.mkdir(parents=True)
    harvey_python.write_text("#!/bin/sh\nexit 0\n")
    harvey_python.chmod(0o755)
    harness = tmp_path / "opt/runtime/agents/harvey-labs/harness/tools.py"
    harness.parent.mkdir(parents=True)
    harness.touch()
    home = tmp_path / "home"
    home.mkdir()
    command = AGENTS[agent].install_cmd.replace(
        uv_install_command(), "false" if failure == "bootstrap" else "true"
    )
    # The provisioning command has separate tests; get as far as the failed
    # package install even with the protected interpreter path.
    import shlex

    command = command.replace(
        managed_python_install_command(shlex.quote(UV_BIN)), "true"
    )
    for root in ("/opt/", "/root/", "/usr/local/bin/"):
        command = command.replace(root, str(tmp_path) + root)
    done = run_sh(
        command,
        shell="sh",
        env={
            **os.environ,
            "HOME": str(home),
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
        },
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert done.returncode == (1 if failure == "bootstrap" else 42), (
        done.stdout + done.stderr
    )


@pytest.mark.parametrize(
    "agent,name",
    [
        ("deepagents", "DEEPAGENTS_REQUIREMENTS"),
        ("harvey-lab-harness", "HARVEY_LAB_REQUIREMENTS"),
    ],
)
def test_same_version_replacement_wheel_is_rejected(agent, name, tmp_path):
    """Guards fix 7cdcfed0 (U4-C3) with different bytes at the pinned version."""
    from benchflow.agents import _python_agent_pins as pins

    requirement = next(
        line for line in getattr(pins, name) if line.startswith("annotated-types==")
    )
    version = requirement.split("==", 1)[1].split()[0]
    dist = f"annotated_types-{version}.dist-info"
    wheel = tmp_path / f"annotated_types-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"{dist}/METADATA",
            f"Metadata-Version: 2.1\nName: annotated-types\nVersion: {version}\n",
        )
        archive.writestr(
            f"{dist}/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(f"{dist}/RECORD", "")
        archive.writestr("annotated_types/__init__.py", "replacement = True\n")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(requirement + "\n")
    uv = os.environ.get("UV") or shutil.which("uv")
    assert uv, "run this test with uv run pytest"
    hash_flag = (
        ["--require-hashes"] if "--require-hashes" in AGENTS[agent].install_cmd else []
    )
    done = subprocess.run(
        [
            uv,
            "--no-config",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--no-deps",
            "--no-index",
            "--find-links",
            str(tmp_path),
            "--target",
            str(tmp_path / "installed"),
            *hash_flag,
            "-r",
            str(requirements),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode != 0, "a same-version replacement wheel was installed"
    assert "Hash mismatch" in done.stderr, done.stderr


@pytest.mark.parametrize(
    "agent,name,package,module,smoke",
    [
        (
            "deepagents",
            "DEEPAGENTS_REQUIREMENTS",
            "langchain-openai",
            "langchain_openai",
            "ChatOpenAI",
        ),
        (
            "harvey-lab-harness",
            "HARVEY_LAB_REQUIREMENTS",
            "openpyxl",
            "openpyxl",
            "load_workbook",
        ),
    ],
)
def test_existing_same_version_replacement_cannot_bypass_hashes(
    agent, name, package, module, smoke, tmp_path
):
    """Guards fix 9becba0b (U4-CONF-2): matching metadata is not provenance."""
    from benchflow.agents import _python_agent_pins as pins

    requirement = next(
        line for line in getattr(pins, name) if line.startswith(package + "==")
    )
    version = requirement.split("==", 1)[1].split()[0]
    wheel_name = package.replace("-", "_")
    dist = f"{wheel_name}-{version}.dist-info"
    wheel = tmp_path / f"{wheel_name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"{dist}/METADATA",
            f"Metadata-Version: 2.1\nName: {package}\nVersion: {version}\n",
        )
        archive.writestr(
            f"{dist}/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(f"{dist}/RECORD", "")
        archive.writestr(
            f"{module}/__init__.py", f"{smoke} = object()\nreplacement_marker = True\n"
        )
    uv = os.environ.get("UV") or shutil.which("uv")
    assert uv, "run this test with uv run pytest"
    venv = tmp_path / "agent-venv"
    python = venv / "bin/python"
    subprocess.run(
        [uv, "--no-config", "venv", "--python", sys.executable, str(venv)], check=True
    )
    subprocess.run(
        [
            uv,
            "--no-config",
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            str(wheel),
        ],
        check=True,
    )
    # This is a usable same-version package: the shim's import smoke succeeds.
    subprocess.run(
        [
            str(python),
            "-c",
            f"from {module} import {smoke}, replacement_marker; assert replacement_marker",
        ],
        check=True,
    )
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(requirement + "\n")
    command = AGENTS[agent].install_cmd
    # Exercise the production venv/install steps with one dependency and a local
    # index. All its requirements are already satisfied before the hash check.
    create = re.search(r"uv venv .*? && ", command)[0].removesuffix(" && ")
    install = re.search(r"uv pip install .*? && ", command)[0].removesuffix(" && ")
    import shlex

    create = re.sub(
        r"--python 3\.12(?:\.15)? \S+",
        f"--python {shlex.quote(sys.executable)} {shlex.quote(str(venv))}",
        create,
    )
    install = re.sub(r"--python \S+", f"--python {shlex.quote(str(python))}", install)
    install = re.sub(
        r"-r \S+",
        f"--no-deps --no-index --find-links {shlex.quote(str(tmp_path))} -r {shlex.quote(str(requirements))}",
        install,
    )
    done = run_sh(
        create + " && " + install,
        shell="sh",
        env={**os.environ, "PATH": str(os.path.dirname(uv)) + ":" + os.environ["PATH"]},
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode != 0, (
        "the already-installed replacement bypassed artifact verification"
    )
    assert "Hash mismatch" in done.stderr, done.stderr
