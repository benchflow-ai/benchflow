"""Guards the fix against 6af6742b3's empty ``rollout/_separate_verifier.py.new``,
which was tracked and so shipped in the 0.9.0 wheel: the package holds no
merge, patch or editor leftovers."""

from tests._package_files import PACKAGE_ROOT

LEFTOVER_SUFFIXES = (".new", ".orig", ".rej", ".bak", ".swp", "~")


def test_the_package_ships_no_leftover_files():
    leftovers = sorted(
        str(path.relative_to(PACKAGE_ROOT))
        for path in PACKAGE_ROOT.rglob("*")
        if path.is_file() and path.name.endswith(LEFTOVER_SUFFIXES)
    )
    assert leftovers == []
