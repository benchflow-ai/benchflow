#!/usr/bin/env bash
# Run bfx's Rust tests as root in a throwaway container, with a tmpfs mount
# point inside the test root (/bfxt/root/m) so the no-mount-crossing cases
# run. Needs Docker and the toolchain scripts/build_bfx.sh uses.
#   BFX_TEST_IMAGE (default alpine:3.20)   BFX_TEST_NAME (container name)
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
crate="$repo/bfx"
toolchain=$(sed -n 's/^channel = "\(.*\)"/\1/p' "$crate/rust-toolchain.toml")
image=${BFX_TEST_IMAGE:-alpine:3.20}
name=${BFX_TEST_NAME:-bfx-test-$$}
cd "$crate"
bin=$(cargo "+$toolchain" test --locked --release --no-run --target x86_64-unknown-linux-musl --message-format=json 2>/dev/null \
    | sed -n 's/.*"executable":"\([^"]*\)".*/\1/p' | tail -1)
[ -x "$bin" ] || { echo "no test binary" >&2; exit 1; }
docker run --rm --name "$name" --network none \
    --tmpfs /bfxt/root/m:rw,size=16m \
    -v "$bin:/bfx-tests:ro" "$image" /bfx-tests --test-threads=1 "$@"
