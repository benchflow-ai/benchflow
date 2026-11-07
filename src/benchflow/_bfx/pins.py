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
    "x86_64": "0cc62aa585473d1d5ffb07fb6b28faf5c67d0dae3a2d19d28f7f2cb68d3054d6",
    "aarch64": "931c67ce049297d209db493ae354e6f7d69537ae3237ed99834a395d42c09a32",
}
