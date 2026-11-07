# Build BenchFlow's static helper for both architectures (on a Linux build
# machine with rustup; see scripts/build_bfx.sh).
.PHONY: bfx bfx-check
bfx:
	scripts/build_bfx.sh

# Rebuild and compare with the pins in src/benchflow/_bfx/pins.py.
bfx-check:
	BFX_CHECK=1 scripts/build_bfx.sh
