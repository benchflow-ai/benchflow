# Build BenchFlow's static helper for both architectures (on a Linux build
# machine with rustup; see scripts/build_bfx.sh). It installs a binary only
# when its bytes are the pin in src/benchflow/_bfx/pins.py, so a checkout's
# Docker trials run the release's bfx or none.
.PHONY: bfx bfx-dev bfx-check
bfx:
	BFX_REQUIRE_PIN=1 scripts/build_bfx.sh

# An unpinned build, for work on the crate itself: it installs what it built.
bfx-dev:
	scripts/build_bfx.sh

# Rebuild and compare with the pins in src/benchflow/_bfx/pins.py.
bfx-check:
	BFX_CHECK=1 scripts/build_bfx.sh
