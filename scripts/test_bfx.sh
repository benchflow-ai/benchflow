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
target=x86_64-unknown-linux-musl
cd "$crate"
# The unit tests (src/), and the hold rows (tests/hold.rs), which run the
# built bfx itself: it is mounted at the path the hold binary names. Each
# executable is picked by its target's name and kind from cargo's artifact
# list; anything but exactly one match stops the script.
artifact() { # NAME TEST(true|false) CARGO-ARGS...
    local name=$1 test=$2
    shift 2
    local found
    found=$(cargo "+$toolchain" test --locked --release --no-run --target "$target" "$@" --message-format=json 2>/dev/null \
        | grep '"reason":"compiler-artifact"' | grep "\"name\":\"$name\"" | grep "\"test\":$test[,}]" \
        | sed -n 's/.*"executable":"\([^"]*\)".*/\1/p' | sort -u)
    if [ "$(printf '%s\n' "$found" | grep -c .)" != 1 ] || [ ! -x "$found" ]; then
        echo "test_bfx.sh: expected one executable for $name (test=$test), found: ${found:-none}" >&2
        exit 1
    fi
    printf '%s\n' "$found"
}
unit=$(artifact bfx true --bins)
hold=$(artifact hold true --test hold)
bfx=$(artifact bfx false --test hold)
docker run --rm --name "$name" --network none \
    --tmpfs /bfxt/root/m:rw,size=16m \
    -v "$unit:/bfx-tests:ro" "$image" /bfx-tests --test-threads=1 "$@"
# The hold rows in their own container, with CAP_SYS_PTRACE: the held waiter
# is not dumpable, and the rows read its /proc/<pid>/environ.
docker run --rm --name "$name-hold" --network none --cap-add SYS_PTRACE \
    -v "$hold:/bfx-hold-tests:ro" -v "$bfx:$bfx:ro" "$image" /bfx-hold-tests "$@"
