#!/usr/bin/env bash
# Install the release leak gate's secret scanners (tools/release_leak_gate.py)
# into DIR, each pinned by version and by the SHA-256 its maintainers publish
# in the release's checksums file. Linux x86_64, as the release runner is.
#
# usage: tools/install_leak_scanners.sh DIR
set -euo pipefail

GITLEAKS_VERSION=8.30.1
GITLEAKS_SHA256=551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb
TRUFFLEHOG_VERSION=3.99.8
TRUFFLEHOG_SHA256=7c3b02f160f28929a467e29e20fba98746e0729f6e37c3c25761e4f8789abe08

dest="${1:?usage: tools/install_leak_scanners.sh DIR}"
if [ "$(uname -s) $(uname -m)" != "Linux x86_64" ]; then
  echo "install_leak_scanners: the pins are for Linux x86_64" >&2
  exit 2
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

fetch() {  # name url sha256
  curl --fail --silent --show-error --location --retry 3 --output "$tmp/$1" "$2"
  echo "$3  $tmp/$1" | sha256sum --check --strict --quiet
}

fetch gitleaks.tar.gz \
  "https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_linux_x64.tar.gz" \
  "$GITLEAKS_SHA256"
fetch trufflehog.tar.gz \
  "https://github.com/trufflesecurity/trufflehog/releases/download/v${TRUFFLEHOG_VERSION}/trufflehog_${TRUFFLEHOG_VERSION}_linux_amd64.tar.gz" \
  "$TRUFFLEHOG_SHA256"

tar -xzf "$tmp/gitleaks.tar.gz" -C "$tmp" gitleaks
tar -xzf "$tmp/trufflehog.tar.gz" -C "$tmp" trufflehog
mkdir -p "$dest"
install -m 0755 "$tmp/gitleaks" "$tmp/trufflehog" "$dest/"
"$dest/gitleaks" version
"$dest/trufflehog" --version
