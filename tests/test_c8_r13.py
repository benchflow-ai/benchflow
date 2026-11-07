"""PT round 13: unit checks next to ``test_c8_r13_container`` (real Docker).

- PT-C8-CONF-6: root's task route restores every start-up name the route
  and S1 clear, with one value rule per kind of name (``bf_value_ok``).
- The b179 gate's manifest failure: ``requirements_files`` is core-owned
  and travels with the core ``install_cmd`` it belongs to.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import replace

import pytest

from benchflow._root_only import ROOT_CHAIN_FUNCS
from benchflow.sandbox._base import (
    IMAGE_ENV_PREFIX,
    SYSTEM_SHELL_UNSET_VARS,
    restore_image_env,
)


def _sh(script: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["/bin/sh", "-c", script],
        env={"PATH": "/usr/bin:/bin", **(env or {})},
        capture_output=True,
        text=True,
    )


# --- PT-C8-CONF-6 ------------------------------------------------------------


def test_every_name_the_route_clears_is_one_root_task_commands_get_back():
    from benchflow.sandbox._base import IMAGE_RESTORED_VARS

    assert set(IMAGE_RESTORED_VARS) == set(SYSTEM_SHELL_UNSET_VARS)


@pytest.mark.parametrize(
    "name", ["PYTHONPATH", "TAR_OPTIONS", "PERL5LIB", "LD_PRELOAD"]
)
def test_a_kept_image_value_comes_back_under_its_name_and_the_copy_goes(name):
    """Fails on 7342d55c for every name outside the loader's thirteen: the
    value stayed under its _BF_IMAGE_ name and never reached the command."""
    done = _sh(
        restore_image_env()
        + f'printf "%s|%s" "${{{name}-unset}}" "$(env | grep -c "^{IMAGE_ENV_PREFIX}")"',
        {f"{IMAGE_ENV_PREFIX}{name}": "/opt/kept"},
    )
    assert done.stdout == "/opt/kept|0", (done.stdout, done.stderr)


def test_the_value_the_command_sourced_wins_over_the_kept_one():
    done = _sh(
        "PYTHONPATH=/task; export PYTHONPATH; "
        + restore_image_env(skip={"PYTHONPATH": "/task"})
        + 'printf %s "$PYTHONPATH"',
        {f"{IMAGE_ENV_PREFIX}PYTHONPATH": "/opt/kept"},
    )
    assert done.stdout == "/task"


def _value_ok(name: str, value: str) -> bool:
    from benchflow.sandbox._base import env_value_check_func

    done = subprocess.run(
        [
            "/bin/sh",
            "-c",
            ROOT_CHAIN_FUNCS + env_value_check_func() + 'bf_value_ok "$1" "$2"',
            "sh",
            name,
            value,
        ],
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    return done.returncode == 0


@pytest.mark.skipif(
    os.getuid() == 0 or os.stat("/usr/lib").st_uid != 0,
    reason="needs a root-owned /usr/lib and a non-root test user",
)
@pytest.mark.parametrize(
    ("name", "value", "ok"),
    [
        # Path lists: every entry absolute and root's alone.
        ("PYTHONPATH", "/usr/lib", True),
        ("PYTHONPATH", "/usr/lib:{own}", False),
        ("PYTHONPATH", "relative", False),
        ("LD_PRELOAD", "libsearched.so", False),
        ("NODE_PATH", "/usr/lib", True),
        # Inert options: a file a token names must be root's alone.
        ("TAR_OPTIONS", "--no-same-owner", True),
        ("TAR_OPTIONS", "--use-compress-program={own}/zip", False),
        ("TAR_OPTIONS", "--use-compress-program=/usr/bin/gzip", True),
        ("GLIBC_TUNABLES", "glibc.malloc.check=3", True),
        ("LD_DEBUG", "libs", True),
        # Code-loading options: plain flags only.
        ("NODE_OPTIONS", "--max-old-space-size=4096", True),
        ("NODE_OPTIONS", "--require mod", False),
        ("NODE_OPTIONS", "-r mod", False),
        ("NODE_OPTIONS", "--import=/usr/lib/x.mjs", False),
        ("PERL5OPT", "-Mstrict", False),
        ("PERL5OPT", "-w", True),
    ],
)
def test_one_value_rule_per_kind_of_name(tmp_path, name, value, ok):
    assert _value_ok(name, value.format(own=tmp_path)) is ok


# --- the b179 gate: requirements_files is core-owned ------------------------


def test_a_manifest_that_keeps_the_core_install_keeps_its_locks():
    """Fails on 7342d55c: the merge took the manifest's empty locks, so the
    core install_cmd's lock recipe ran in the sandbox (CONF-9's decoder)."""
    from benchflow.agents.manifest import _merge_core_shim_only
    from benchflow.agents.registry import AGENTS

    core = AGENTS["deepagents"]
    assert core.requirements_files
    manifest = replace(core, requirements_files=(), description="from a manifest")
    merged = _merge_core_shim_only(manifest, core)
    assert merged.requirements_files == core.requirements_files
    assert merged.description == "from a manifest"


def test_a_manifest_with_its_own_install_has_no_core_locks():
    from benchflow.agents.manifest import _merge_core_shim_only
    from benchflow.agents.registry import AGENTS

    core = AGENTS["deepagents"]
    manifest = replace(core, install_cmd="echo own install", requirements_files=())
    merged = _merge_core_shim_only(manifest, core)
    assert merged.requirements_files == ()
    assert merged.install_cmd == "echo own install"
