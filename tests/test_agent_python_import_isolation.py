"""Guard FF-install-C11 at a0c81002: root helpers never import the workspace."""

import os
import re
import shlex
import subprocess
import sys

import pytest

from benchflow.agents.registry import AGENTS


@pytest.fixture(scope="module")
def isolated_agent_python(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("trusted-agent-python")
    venv = tmp_path / "venv"
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "venv",
            "--without-pip",
            "--system-site-packages",
            str(venv),
        ],
        check=True,
    )
    python = venv / "bin/python"
    site = next((venv / "lib").glob("python*/site-packages"))
    packages = {
        "deepagents/__init__.py": "create_deep_agent = object()",
        "deepagents/backends.py": "LocalShellBackend = object()",
        "langchain_openai.py": "ChatOpenAI = object()",
        "google/__init__.py": "",
        "google/genai.py": "",
        **{
            f"{name}.py": ""
            for name in (
                "anthropic",
                "openai",
                "docx",
                "pdfplumber",
                "openpyxl",
                "pptx",
                "markitdown",
                "pandas",
            )
        },
    }
    for name, source in packages.items():
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    return python


def _helper(agent, kind, python):
    command = AGENTS[agent].install_cmd
    quoted = shlex.quote(str(python))
    if kind == 0:
        # The decoder runs on root's protected Python (PT's root_python_sh):
        # the kit's once the sandbox has one, else this venv's, isolated.
        body = command.index("zlib.decompress(base64.b64decode(")
        start = command.rindex("( if [ -f ", 0, body)
        end = command.index("' )", body) + len("' )")
        assert command.count("zlib.decompress(base64.b64decode(") == 1
        decoder = command[start:end]
        return re.sub(
            r"else bf_py=\S+; fi", f"else bf_py={quoted}; fi", decoder, count=1
        )
    smokes = re.findall(r"\S+/bin/python (?:-I )?-c '[^']*'", command)
    assert len(smokes) == 1, "the import smoke must be exercised"
    return re.sub(r"^\S+", quoted, smokes[0], count=1)


@pytest.mark.parametrize("agent", ["deepagents", "harvey-lab-harness"])
@pytest.mark.parametrize("kind", [0, 1], ids=["decoder", "smoke"])
@pytest.mark.parametrize("source", ["cwd", "pythonpath", "user-site", "pythonhome"])
def test_privileged_python_ignores_ambient_imports(
    tmp_path, isolated_agent_python, agent, kind, source
):
    """Guards a0c81002: every root decoder/smoke isolates cwd, user site and env."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    injected = tmp_path / "injected"
    injected.mkdir()
    marker = tmp_path / "imported-agent-code"
    payload = f"open({str(marker)!r}, 'w').write('dummy')\nraise RuntimeError('untrusted import')\n"
    env = {**os.environ, "HOME": str(workspace)}
    if source == "cwd":
        name = (
            "base64"
            if kind == 0
            else ("deepagents" if agent == "deepagents" else "anthropic")
        )
        (workspace / f"{name}.py").write_text(payload)
    elif source == "pythonpath":
        (injected / "sitecustomize.py").write_text(payload)
        env["PYTHONPATH"] = str(injected)
    elif source == "user-site":
        site = (
            injected
            / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
        )
        site.mkdir(parents=True)
        for name in ("sitecustomize", "usercustomize"):
            (site / f"{name}.py").write_text(payload)
        env["PYTHONUSERBASE"] = str(injected)
    else:
        env["PYTHONHOME"] = str(injected)
    # The exact decoder's payload is immaterial to the import boundary.
    import base64
    import zlib

    data = base64.b64encode(zlib.compress(b"dummy==1\n")).decode()
    result = subprocess.run(
        ["/bin/sh", "-c", _helper(agent, kind, isolated_agent_python)],
        input=data,
        text=True,
        capture_output=True,
        cwd=workspace,
        env=env,
    )
    assert not marker.exists(), "privileged helper imported agent code"
    assert result.returncode == 0, result.stderr
    if kind == 0:
        assert result.stdout == "dummy==1\n"


@pytest.mark.parametrize(
    "agent,kind,module",
    [
        (agent, 0, module)
        for agent in ("deepagents", "harvey-lab-harness")
        for module in ("base64", "zlib")
    ]
    + [("deepagents", 1, module) for module in ("deepagents", "langchain_openai")]
    + [
        ("harvey-lab-harness", 1, module)
        for module in (
            "anthropic",
            "openai",
            "google",
            "docx",
            "pdfplumber",
            "openpyxl",
            "pptx",
            "markitdown",
            "pandas",
        )
    ],
)
def test_every_privileged_import_excludes_cwd(
    tmp_path, isolated_agent_python, agent, kind, module
):
    """Guards a0c81002: cover every named import, including extension modules."""
    marker = tmp_path / "shadow-import"
    (tmp_path / f"{module}.py").write_text(
        f"open({str(marker)!r}, 'w').write('dummy')\n"
    )
    import base64
    import zlib

    result = subprocess.run(
        ["/bin/sh", "-c", _helper(agent, kind, isolated_agent_python)],
        input=base64.b64encode(zlib.compress(b"dummy==1\n")).decode(),
        text=True,
        capture_output=True,
        cwd=tmp_path,
    )
    assert not marker.exists(), f"root helper imported workspace {module}"
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("source", ["json", "pathlib", "pythonpath"])
def test_privileged_settings_merge_ignores_workspace(tmp_path, source):
    """Guards a0c81002: root policy setup and its Node fallback share isolation."""
    from benchflow.agents.registry import _json_settings_merge_py

    marker = tmp_path / "settings-import"
    name = "sitecustomize" if source == "pythonpath" else source
    (tmp_path / f"{name}.py").write_text(f"open({str(marker)!r}, 'w').write('dummy')\n")
    settings = tmp_path / "settings.json"
    command = _json_settings_merge_py(str(settings), 'd["policy"] = False')
    result = subprocess.run(
        ["/bin/sh", "-c", command],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert not marker.exists(), "settings merge imported agent code"
    assert result.returncode == 0, result.stderr
    import json

    assert json.loads(settings.read_text()) == {"policy": False}
