#!/usr/bin/env bash
# Build bfx for x86_64 and aarch64 (static musl), reproducibly, into
# src/benchflow/_bfx/bin/bfx-<arch>, and print each SHA-256.
#
# Inputs that decide the bytes: the crate source, bfx/Cargo.lock (every
# dependency pinned by SHA-256), the toolchain in bfx/rust-toolchain.toml
# and the flags below. Paths are remapped, so the checkout's location and
# CARGO_HOME do not reach the binary. Dependencies are vendored from the
# lock first (`cargo vendor --locked`, the only step that needs the
# network), then the build runs --offline --locked against that vendor tree.
#
# Linking uses rust-lld and the toolchain's self-contained musl crt objects;
# only the host's build scripts of dependencies need the host linker.
#
#   BFX_OUT=<dir>      where the binaries go (default src/benchflow/_bfx/bin)
#   BFX_ARCHES="x86_64 aarch64"
#   BFX_CHECK=1        do not install; compare with src/benchflow/_bfx/pins.py
#   BFX_REQUIRE_PIN=1  install only a binary whose SHA-256 is the pin (releases)
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
crate="$repo/bfx"
out=${BFX_OUT:-$repo/src/benchflow/_bfx/bin}
arches=${BFX_ARCHES:-x86_64 aarch64}
toolchain=$(sed -n 's/^channel = "\(.*\)"/\1/p' "$crate/rust-toolchain.toml")
cargo_home=${CARGO_HOME:-$HOME/.cargo}
# The crate's last commit time, so nothing time-dependent varies.
SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH:-$(git -C "$repo" log -1 --format=%ct -- bfx 2>/dev/null || echo 0)}
export SOURCE_DATE_EPOCH

cd "$crate"
cargo "+$toolchain" vendor --locked --versioned-dirs --quiet vendor >/dev/null
mkdir -p "$out"
for arch in $arches; do
    target="$arch-unknown-linux-musl"
    flags=(
        -C target-feature=+crt-static
        -C relocation-model=static
        -C link-self-contained=yes
        -C linker=rust-lld
        -C linker-flavor=ld.lld
        -C strip=symbols
        -C debuginfo=0
        "--remap-path-prefix=$cargo_home=/cargo"
        "--remap-path-prefix=$crate=/bfx"
        "--remap-path-prefix=$crate/vendor=/vendor"
    )
    CARGO_ENCODED_RUSTFLAGS=$(IFS=$'\x1f'; echo "${flags[*]}") \
        cargo "+$toolchain" build --release --locked --offline --quiet \
        --target "$target" \
        --config 'source.crates-io.replace-with="vendored-sources"' \
        --config 'source.vendored-sources.directory="vendor"'
    built="target/$target/release/bfx"
    sum=$(sha256sum "$built" | cut -d' ' -f1)
    pin=$(sed -n "s/^    \"$arch\": \"\([0-9a-f]\{64\}\)\",\$/\1/p" "$repo/src/benchflow/_bfx/pins.py")
    if [ "${BFX_REQUIRE_PIN:-0}" = 1 ] && [ "$sum" != "$pin" ]; then
        echo "bfx-$arch: built $sum, pinned $pin; not installed" >&2
        exit 1
    fi
    if [ "${BFX_CHECK:-0}" = 1 ]; then
        if [ "$sum" != "$pin" ]; then
            echo "bfx-$arch: built $sum, pinned $pin" >&2
            exit 1
        fi
        echo "bfx-$arch $sum matches the pin"
    else
        install -m 0555 "$built" "$out/bfx-$arch"
        echo "bfx-$arch $sum"
    fi
done
