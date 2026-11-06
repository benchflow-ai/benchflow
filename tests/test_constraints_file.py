"""constraints.txt: the lock's versions for installs that never read uv.lock.

``uv tool install`` and ``pip install`` resolve afresh; each release tag carries
``constraints.txt`` so a user can install what CI tested. These tests keep it
equal to ``uv.lock`` and keep the packages a fresh resolve broke bounded
(REPORT B13: typer 0.27.3 failed 9 tests while the lock held 0.24.1; the
Daytona SDK from 0.197 turns the PTY heartbeat patch off).
"""

from __future__ import annotations

import os
import shutil
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from tools import export_constraints as export_tool

ROOT = Path(__file__).resolve().parents[1]
UV = os.environ.get("UV") or shutil.which("uv")


def _pins(text: str) -> dict[str, list[tuple[str, str]]]:
    """``name -> [(version, marker), ...]`` from requirement lines."""
    pins: dict[str, list[tuple[str, str]]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        req = Requirement(line)
        (spec,) = req.specifier
        assert spec.operator == "==", line
        pins.setdefault(canonicalize_name(req.name), []).append(
            (spec.version, str(req.marker or ""))
        )
    return pins


def _lock_versions() -> dict[str, set[str]]:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    versions: dict[str, set[str]] = {}
    for package in lock["package"]:
        versions.setdefault(canonicalize_name(package["name"]), set()).add(
            package["version"]
        )
    return versions


def test_constraints_file_matches_uv_lock():
    """The file is exactly what uv.lock exports, so a lock change without
    regenerating fails here with the one command that fixes it."""
    assert UV, "this check runs uv export; run the suite with `uv run pytest`"
    expected = export_tool.render(uv=UV)
    shipped = (ROOT / "constraints.txt").read_text(encoding="utf-8")
    assert shipped == expected, (
        "constraints.txt has drifted from uv.lock; regenerate it on a build VM "
        f"with: {export_tool.REGENERATE_COMMAND}"
    )


def test_every_pin_is_the_locked_version_and_benchflow_is_not_pinned():
    """Read without uv: each pin is a version the lock holds for that
    package, the core dependencies are all there, and BenchFlow itself is
    left to the install command."""
    pins = _pins((ROOT / "constraints.txt").read_text(encoding="utf-8"))
    locked = _lock_versions()
    assert "benchflow" not in pins
    for name, entries in pins.items():
        for version, _marker in entries:
            assert version in locked.get(name, set()), (name, version)
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    for dependency in project["project"]["dependencies"]:
        assert canonicalize_name(Requirement(dependency).name) in pins, dependency
    # The dev extra (pytest, ruff, ty) is for contributors, who use the lock.
    assert "ruff" not in pins
    assert "typer" in pins and "click" in pins


def test_check_mode_reports_a_stale_file_and_names_the_command(
    tmp_path, monkeypatch, capsys
):
    output = tmp_path / "constraints.txt"
    monkeypatch.setattr(export_tool, "OUTPUT", output)
    monkeypatch.setattr(
        export_tool, "render", lambda: export_tool.HEADER + "typer==0.24.1\n"
    )
    output.write_text("typer==0.27.3\n", encoding="utf-8")
    assert export_tool.main(["--check"]) == 1
    assert export_tool.REGENERATE_COMMAND in capsys.readouterr().out
    assert output.read_text(encoding="utf-8") == "typer==0.27.3\n"
    assert export_tool.main([]) == 0
    assert export_tool.main(["--check"]) == 0
    assert export_tool.REGENERATE_COMMAND in output.read_text(encoding="utf-8")


def _requirement(name: str, extra: str | None) -> Requirement:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    table = project["project"]
    dependencies = (
        table["dependencies"]
        if extra is None
        else table["optional-dependencies"][extra]
    )
    for dependency in dependencies:
        req = Requirement(dependency)
        if canonicalize_name(req.name) == name:
            return req
    raise AssertionError(f"{name} is not in {extra or 'the core dependencies'}")


@pytest.mark.parametrize(
    ("name", "extra"),
    [("typer", None), ("click", None), ("daytona", "sandbox-daytona")],
)
def test_bounded_packages_carry_an_upper_bound_the_lock_satisfies(name, extra):
    """typer, click and the Daytona SDK have upper bounds, set at the newest
    minors tools/ci/fresh_resolve.sh passes on: on a fresh resolve
    typer 0.26 failed 9 tests and daytona 0.197 turned the PTY heartbeat
    patch off. The locked version is inside each bound."""
    req = _requirement(name, extra)
    uppers = [s for s in req.specifier if s.operator in ("<", "<=", "~=", "==")]
    assert uppers, f"{req} has no upper bound"
    (locked,) = _lock_versions()[name]
    assert Version(locked) in req.specifier, (str(req), locked)


def _floor(req: Requirement) -> Version:
    floors = [Version(s.version) for s in req.specifier if s.operator == ">="]
    assert floors, f"{req} has no lower bound"
    return max(floors)


def test_typer_floor_builds_the_cli_with_the_click_floor():
    """pip keeps an installed typer the floor allows, so the floor has to
    be a typer that runs this CLI. typer before 0.19 cannot build a
    ``typing.Literal`` option (``bench tasks check --level``), so every
    command stops with ``RuntimeError: Type not yet supported``; typer
    before 0.16 also breaks on click 8.2's ``Parameter.make_metavar(ctx)``,
    and the click floor upgrades click under a kept typer (typer 0.15 with
    click 8.1 became click 8.5, and ``--help`` raised ``TypeError:
    TyperArgument.make_metavar() takes 1 positional argument but 2 were
    given``)."""
    click_floor = _floor(_requirement("click", None))
    typer_floor = _floor(_requirement("typer", None))
    assert typer_floor >= Version("0.19"), (
        f"typer>={typer_floor}: typer before 0.19 cannot build the CLI's "
        "Literal options"
    )
    if click_floor >= Version("8.2"):
        assert typer_floor >= Version("0.16"), (
            f"click>={click_floor} needs typer>=0.16, the floor is {typer_floor}"
        )
