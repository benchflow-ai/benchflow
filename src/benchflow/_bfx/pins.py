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
    "x86_64": "ee3a76b5c183717049d4c4e34a8faa022f0f6994fd85d31c7ac33418f6360539",
    "aarch64": "79f6b036adafafcb58e8dd40db33ba4841e561eadf1ec87557fb79abacb0a47f",
}
