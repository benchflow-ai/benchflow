"""`make bfx` installs only the pinned bytes.

A checkout has no bfx (git ignores src/benchflow/_bfx/bin), and a build that
installs whatever it made leaves a tree whose Docker trials run on another
binary than the release's. The window check's invalid start (rustup without
the musl targets) and a fresh VM's 101 refusals before the turn both came from
a missing or unchecked bfx. `make bfx-dev` stays for work on the crate."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _recipes() -> dict[str, list[str]]:
    recipes: dict[str, list[str]] = {}
    target = None
    for line in (ROOT / "Makefile").read_text().splitlines():
        if line.startswith("\t") and target:
            recipes[target].append(line.strip())
            continue
        m = re.match(r"^([A-Za-z0-9_-]+):", line)
        target = m.group(1) if m else None
        if target:
            recipes[target] = []
    return recipes


def test_make_bfx_requires_the_pin():
    recipe = " ".join(_recipes()["bfx"])
    assert "scripts/build_bfx.sh" in recipe
    assert "BFX_REQUIRE_PIN=1" in recipe


def test_an_unpinned_build_has_its_own_name():
    recipe = " ".join(_recipes()["bfx-dev"])
    assert "scripts/build_bfx.sh" in recipe and "BFX_REQUIRE_PIN" not in recipe


def test_the_check_still_compares_with_the_pins():
    assert "BFX_CHECK=1" in " ".join(_recipes()["bfx-check"])
