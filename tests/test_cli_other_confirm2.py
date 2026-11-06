"""Guards CONF-2 at 5bb84a1b: prior-role Python startup files are not consumers."""

import json
import os
import subprocess
import sys

import pytest

from benchflow.agents import registry
from benchflow.sandbox.process._acp_lines import filter_command


@pytest.mark.parametrize("hook", ["usercustomize", "sitecustomize", "module-shadow"])
@pytest.mark.parametrize(
    "agent,source_name,binary",
    [
        ("muse-code", "_MUSE_SHIM", "muse-acp-shim"),
        ("antigravity", "_ANTIGRAVITY_SHIM", "antigravity-acp-shim"),
        ("pi-acp", "_PI_LAUNCHER", "pi-acp-launcher"),
        ("filter", "", ""),
        ("settings", "", ""),
        ("settings-local", "", ""),
    ],
)
def test_prior_role_startup_files_never_run(tmp_path, hook, agent, source_name, binary):
    """5bb84a1b executes a harmless marker hook before the next shim starts."""
    target = tmp_path / (binary or "dummy-filter")
    if agent not in {"filter", "settings", "settings-local"}:
        subprocess.run(
            [
                "sh",
                "-c",
                registry._install_python_script(
                    str(target), getattr(registry, source_name)
                ),
            ],
            check=True,
        )
        command = registry.AGENTS[agent].launch_cmd.replace(
            f"/opt/runtime/bin/{binary}", str(target)
        )
    elif agent.startswith("settings"):
        command = registry._json_settings_merge(
            str(tmp_path / "dummy-settings.json"), "d['dummy'] = True"
        )
        if agent == "settings-local":
            # The image layout with only /usr/local/bin/python3. This host
            # need not have one (a stock Ubuntu 24.04 has none): a
            # stand-in at a path of the test's own, the same interpreter as
            # /usr/bin/python3, takes its place.
            local = tmp_path / "usr-local-bin" / "python3"
            local.parent.mkdir()
            local.symlink_to("/usr/bin/python3")
            command = command.replace("/usr/local/bin/python3", str(local)).replace(
                "/usr/bin/python3", "/missing/dummy-python3"
            )
    else:
        command = filter_command("printf 'dummy-filter\\n'")
    marker = tmp_path / "dummy-startup-ran"
    # Marker only: this hook neither copies nor prints credential contents.
    hook_source = f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "PATH": "/usr/bin:/bin",
        "MUSE_AUTH_JSON": "dummy-login",
    }
    version = subprocess.check_output(
        [
            "/usr/bin/python3",
            "-I",
            "-S",
            "-c",
            "import sys; print('%d.%d' % sys.version_info[:2])",
        ],
        text=True,
    ).strip()
    if hook == "usercustomize":
        folder = tmp_path / ".local" / "lib" / f"python{version}" / "site-packages"
        folder.mkdir(parents=True)
        (folder / "usercustomize.py").write_text(hook_source)
        env.pop("PYTHONNOUSERSITE", None)
    elif hook == "sitecustomize":
        (tmp_path / "sitecustomize.py").write_text(hook_source)
        env["PYTHONPATH"] = str(tmp_path)
    else:
        (tmp_path / "json.py").write_text(hook_source)
        # The filter is a -c program; shims use the script directory.
    request = (
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        + "\n"
    )
    result = subprocess.run(
        ["bash", "-c", command],
        input=request,
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env=env,
        timeout=15,
    )
    assert not marker.exists()
    if agent in {"muse-code", "antigravity"}:
        assert json.loads(result.stdout.splitlines()[-1])["id"] == 1
    elif agent == "pi-acp":
        assert "pi-acp:" in result.stderr
        assert "not found" in result.stderr
    elif agent.startswith("settings"):
        assert json.loads((tmp_path / "dummy-settings.json").read_text()) == {
            "dummy": True
        }
    else:
        assert result.stdout == "dummy-filter\n"


def test_dependency_shims_keep_trusted_site_packages_but_ignore_user_paths():
    """5bb84a1b did not isolate the pinned venv shims from PYTHONPATH."""
    for name in ("harvey-lab-harness", "deepagents"):
        command = registry.AGENTS[name].launch_cmd
        assert "/opt/runtime/" in command
        assert "/bin/python -I " in command
        assert " -S " not in command


def test_openhands_package_helper_ignores_user_import_paths(tmp_path):
    """Guards CONF-2 at 83cfa63c with a real venv and a harmless cwd hook."""
    venv = tmp_path / "dummy-tool"
    subprocess.run(
        [sys.executable, "-I", "-m", "venv", "--without-pip", str(venv)], check=True
    )
    package = (
        venv
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "openhands_cli"
    )
    package.mkdir()
    (package / "__init__.py").write_text("")
    source = package / "utils.py"
    source.write_text('CONFIG = """\n        Tool(name=task_tool_name),\n"""\n')
    marker = tmp_path / "dummy-startup-ran"
    shadow_package = tmp_path / "openhands_cli"
    shadow_package.mkdir()
    (shadow_package / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
    )
    (shadow_package / "utils.py").write_text(source.read_text())
    # Execute the registry's actual package-editing helper with a dummy
    # trusted package; no upstream install, login or model call is needed.
    command = (
        '"$OH_PY" -'
        + registry.AGENTS["openhands"]
        .launch_cmd.split('"$OH_PY" -', 1)[1]
        .split("; fi &&", 1)[0]
    )
    result = subprocess.run(
        ["/bin/sh", "-c", command],
        cwd=tmp_path,
        env={
            **os.environ,
            "OH_PY": str(venv / "bin/python"),
            "LLM_API_KEY": "dummy-login",
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert not marker.exists()
    assert "delegation disabled for this run" in source.read_text()
