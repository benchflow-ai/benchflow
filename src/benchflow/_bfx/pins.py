"""The SHA-256 of each bfx binary this package ships (one per architecture).

``scripts/build_bfx.sh`` rebuilds them byte for byte from ``bfx/`` (the crate,
its Cargo.lock and the pinned toolchain); ``make bfx-check`` compares a
rebuild with these values, and the release workflows build with
``BFX_REQUIRE_PIN=1`` (``tests/test_bfx_release.py`` checks that wiring). A
binary whose hash is not listed here is never delivered into a sandbox.
"""

from __future__ import annotations

#: The crate version these binaries were built from.
VERSION = "0.1.0"

SHA256 = {
    "x86_64": "3a196c9189c1114be4f0eaa4368717115ca08e1b78844df785eae606b88e4253",
    "aarch64": "4e2c0cea6409fce86909dea85bb62680dcc2af6bbf32dc60aa028a3589645a27",
}
