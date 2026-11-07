"""Guard C4/C5/C6 against the regressions reported at 693c34c2."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from benchflow.sandbox import daytona_layers as layers
from benchflow.sandbox._base import (
    SYSTEM_PATH,
    SYSTEM_SHELL_EMPTY_VARS,
    with_system_path,
)
from benchflow.sandbox.daytona_content_snapshots import check_recipe


@pytest.mark.parametrize("kind", ["proxy", "agent"])
@pytest.mark.parametrize("tool", ["stat", "find", "readlink", "sh", "cat"])
async def test_cache_probe_uses_only_system_tools(tmp_path, monkeypatch, kind, tool):
    """Guard C4: neither cache lookup may run tools planted by a previous role."""
    attack = tmp_path / "bin"
    attack.mkdir()
    marker = tmp_path / "executed"
    planted = attack / tool
    planted.write_text(f"#!/bin/sh\n/usr/bin/touch {marker}\nexit 1\n")
    planted.chmod(0o755)
    layer = layers.InstallLayer(
        kind=kind,
        name="litellm",
        script="dummy",
        venv=layers.PROXY_LAYER_VENV if kind == "proxy" else None,
    )
    monkeypatch.setattr(layers, "_proxy_custody_command", lambda: "stat / >/dev/null")

    class Sandbox:
        def cached_install_layer(self, *_):
            return layer, "key"

        async def exec(self, command, **kwargs):
            # Reproduce the provider's hardened-shell contract, including the
            # launch environment, without requiring a live provider or root.
            hardened = kwargs.get("system_shell", False)
            env = {**os.environ, "PATH": f"{attack}:{SYSTEM_PATH}"}
            env.update(kwargs.get("env") or {})
            if hardened:
                assert kwargs["user"] == "root"
                assert all(env.get(name) == "" for name in SYSTEM_SHELL_EMPTY_VARS)
                command = with_system_path(command)
            result = subprocess.run(
                ["/bin/sh" if hardened else "sh", "-c", command],
                env=env,
                capture_output=True,
                text=True,
            )
            assert not marker.exists()
            assert hardened
            return SimpleNamespace(return_code=result.returncode, stdout=result.stdout)

    # The broad cache-miss handler must not hide the regression assertion.
    calls = []
    original = Sandbox.exec

    async def checked(self, command, **kwargs):
        calls.append(kwargs)
        return await original(self, command, **kwargs)

    monkeypatch.setattr(Sandbox, "exec", checked)
    await layers.cached_layer_in_sandbox(Sandbox(), kind, "litellm", script="dummy")
    assert calls[0].get("system_shell") is True
    assert calls[0].get("user") == "root"
    assert not marker.exists()


@pytest.mark.parametrize("mode", [0o666, 0o664, 0o622, 0o777])
@pytest.mark.parametrize("target", ["../data", "../sub/../data", "../../runtime/data"])
def test_build_seals_writable_entries_and_internal_links(
    tmp_path, monkeypatch, mode, target
):
    """Guard C5: trusted install output is sealed before a reusable key exists."""
    from benchflow.providers import litellm_runtime

    root = tmp_path / "runtime"
    (root / "sub").mkdir(parents=True)
    (root / "data").write_text("trusted")
    (root / "sub" / "advisory.lock").write_text("")
    (root / "sub" / "advisory.lock").chmod(mode)
    (root / "sub" / "terminfo").symlink_to(target)
    # The seal runs the layer venv's bin/python. A stand-in venv whose python
    # is this test's interpreter: a host need not have /usr/local/bin/python
    # (a stock Ubuntu 24.04 has none).
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(sys.executable)
    monkeypatch.setattr(layers, "PROXY_LAYER_ROOT", str(root))
    monkeypatch.setattr(layers, "PROXY_LAYER_VENV", str(venv))
    monkeypatch.setattr(
        litellm_runtime, "sandbox_litellm_install_command", lambda *a, **k: "true\n"
    )
    result = subprocess.run(
        ["/bin/sh", "-c", layers.proxy_install_command(None)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (root / "sub" / "advisory.lock").stat().st_mode & 0o022 == 0
    assert os.readlink(root / "sub" / "terminfo") == str(root / "data")


@pytest.mark.parametrize(
    "keyword", ["ONBUILD", "onbuild", "OnBuIlD", "FROM", "COPY", "RUN"]
)
@pytest.mark.parametrize("split", [1, 2, 3])
@pytest.mark.parametrize("suffix", ["", "\n# comment", "\n"])
def test_continued_instruction_names_always_fall_back(keyword, split, suffix):
    """Guard C6: no parallel Docker grammar may approve a continued recipe."""
    prefix = "FROM image@sha256:" + "1" * 64 + " AS base\n"
    continued = keyword[:split] + "\\\n" + keyword[split:] + " RUN echo example\n"
    assert check_recipe(prefix + continued + suffix)[1] is not None


@pytest.mark.parametrize(
    "directive", ["# escape=`", "# escape=\\", "# EsCaPe = `", "#escape=`"]
)
def test_escape_directives_fall_back(directive):
    """Guard C6: changing Docker's escape grammar cannot bypass eligibility."""
    recipe = directive + "\nFROM image@sha256:" + "1" * 64 + "\nRUN true\n"
    assert check_recipe(recipe)[1] is not None


@pytest.mark.parametrize(
    "line",
    [
        "BOGUS ignored",
        "RUN\\",
        "ONBU\\\nILD RUN true",
        "RUN --mount=type=cache,\\\nfrom=alpine,target=/src true",
    ],
)
def test_unmodeled_recipe_forms_fall_back(line):
    """Guard C6: unknown and truncated forms fail closed, not silently ignored."""
    assert check_recipe("FROM scratch\n" + line)[1] is not None


def test_simple_supported_recipe_remains_eligible():
    """Guard C6's positive path with modeled stage, mount and metadata inputs."""
    assert (
        check_recipe(
            'FROM scratch AS base\nENV A=b\nRUN --mount=type=tmpfs,target=/tmp true\nFROM base\nCMD ["true"]\n'
        )[1]
        is None
    )


@pytest.mark.parametrize(
    "entry", ["absolute-escape", "relative-escape", "dangling", "loop", "fifo"]
)
def test_build_refuses_unsealable_runtime(tmp_path, monkeypatch, entry):
    """Guard C5: normalizing trusted installs cannot bless external runtime inputs."""
    from benchflow.providers import litellm_runtime

    root = tmp_path / "runtime"
    root.mkdir()
    external = tmp_path / "external"
    external.write_text("outside")
    path = root / "entry"
    if entry == "fifo":
        os.mkfifo(path)
    else:
        target = {
            "absolute-escape": str(external),
            "relative-escape": "../external",
            "dangling": "absent",
            "loop": "entry",
        }[entry]
        path.symlink_to(target)
    monkeypatch.setattr(layers, "PROXY_LAYER_ROOT", str(root))
    monkeypatch.setattr(layers, "PROXY_LAYER_VENV", "/usr/local")
    monkeypatch.setattr(
        litellm_runtime, "sandbox_litellm_install_command", lambda *a, **k: "true\n"
    )
    result = subprocess.run(
        ["/bin/sh", "-c", layers.proxy_install_command(None)], capture_output=True
    )
    assert result.returncode != 0
    assert external.read_text() == "outside"
