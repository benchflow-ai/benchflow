"""PT round 12: regressions introduced by fe895b04, retained by 0b3b07a9.

Exercise the generated shell with honest configuration data; no loader
payload or real credential is needed to distinguish the unsafe spelling.
"""

import shlex
import subprocess

import pytest

from benchflow._root_only import ROOT_CHAIN_FUNCS
from benchflow._runtime_tree import kit_prepare_command
from benchflow.sandbox._base import SYSTEM_SHELL_EMPTY_VARS


def shell(script, *args):
    return subprocess.run(
        ["/bin/sh", "-c", script, "sh", *args],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
        timeout=30,
    )


@pytest.mark.parametrize("suffix", ["", "=libc6", "=libc5", "=libc4"])
@pytest.mark.parametrize(
    "directory", ["/opt/vendor/lib", "/opt/shared libs", "/opt/a,b"]
)
def test_glibc_typed_paths_preserve_directory_bytes(tmp_path, suffix, directory):
    """Guards fe895b04's typed-path regression without invoking a loader.

    Both readings are checked (PT-R13-B2): the line as written (glibc 2.37
    and later), and cut at its '=' (2.36 and earlier); bytes, spaces and
    commas kept."""
    config = tmp_path / "ld.so.conf"
    config.write_text(f"  {directory}{suffix}  # trusted image entry\n")
    seal = kit_prepare_command("seal")
    parser = seal[
        seal.index("  while IFS= read -r l") : seal.index(
            "\ndone\n", seal.index("  while IFS= read -r l")
        )
    ]
    result = shell(
        "bf_nl='\n'; bf_ldirs=; f=$1\n" + parser + '\nprintf %s "$bf_ldirs"',
        str(config),
    )
    assert result.returncode == 0, result.stderr
    expected = "\n" + directory + suffix + ("\n" + directory if suffix else "")
    assert result.stdout == expected


@pytest.mark.parametrize("name", SYSTEM_SHELL_EMPTY_VARS)
@pytest.mark.parametrize("renamed", [False, True])
def test_image_environment_receipt_uses_original_names(tmp_path, name, renamed):
    """Guards fe895b04's AgentCore/provider provenance-name regression."""
    incoming = tmp_path / "environ"
    incoming.write_bytes(f"{'_BF_IMAGE_' if renamed else ''}{name}=/usr/lib\0".encode())
    seal = kit_prepare_command("seal")
    start = seal.index(': > "$KIT/IMAGE_ENV_OK"')
    end = seal.index("\n{ echo kit=1;", start)
    receipt = seal[start:end].replace("/proc/1/environ", shlex.quote(str(incoming)))
    result = shell(ROOT_CHAIN_FUNCS + "\nKIT=$1\n" + receipt, str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "IMAGE_ENV_OK").read_text() == f"{name}=/usr/lib\n"


def test_provider_receipt_does_not_admit_arbitrary_prefixed_names(tmp_path):
    """Only the provider's recognized startup names may gain provenance."""
    incoming = tmp_path / "environ"
    incoming.write_bytes(b"_BF_IMAGE_NOT_A_LOADER_SETTING=/usr/lib\0")
    seal = kit_prepare_command("seal")
    start = seal.index(': > "$KIT/IMAGE_ENV_OK"')
    end = seal.index("\n{ echo kit=1;", start)
    result = shell(
        ROOT_CHAIN_FUNCS
        + "\nKIT=$1\n"
        + seal[start:end].replace("/proc/1/environ", shlex.quote(str(incoming))),
        str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "IMAGE_ENV_OK").read_text() == ""
