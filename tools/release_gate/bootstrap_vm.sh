#!/usr/bin/env bash
# Prepare a fresh Ubuntu 24.04 VM for tools/release_gate/run_gate.py: pinned
# Docker, Compose and buildx, sqlite3 and a C compiler (as GitHub's runners
# have them; 146 unit tests use them), this user in the docker group, and the
# uv BenchFlow pins (src/benchflow/_uv_pin.py), checked against its SHA-256
# before it is installed in ~/.local/bin. Run from the checkout, as the user
# who will run the gate; it uses sudo for apt. Log in again afterwards (the
# docker group applies to new sessions), then check: docker info && uv --version
#
#   tools/release_gate/bootstrap_vm.sh [--docker ubuntu|ce]
#
# --docker ubuntu (the default) installs Ubuntu's own packages: their Compose
# is the one the direct Docker path is proven on
# (PROVEN_COMPOSE_VERSIONS in src/benchflow/sandbox/_direct_docker.py), which
# this script checks. --docker ce installs Docker CE from Docker's apt
# repository, its signing key checked by fingerprint, for the gate's docker-ce
# leg (tools/release_gate/legs.json). Every package is pinned to an exact
# version: a version the archive no longer offers stops the script, and the
# pin is changed on purpose. Other distributions are refused.
set -euo pipefail

DOCKER=ubuntu
if [ "${1:-}" = --docker ]; then
    DOCKER=${2:?--docker needs ubuntu or ce}
fi
case "$DOCKER" in ubuntu | ce) ;; *) echo "bootstrap_vm: --docker ubuntu or ce" >&2; exit 2 ;; esac

UBUNTU_PACKAGES=(
    docker.io=29.1.3-0ubuntu3~24.04.2
    containerd=2.2.1-0ubuntu1~24.04.4
    runc=1.3.4-0ubuntu1~24.04.1
    docker-compose-v2=2.40.3+ds1-0ubuntu1~24.04.1
    docker-buildx=0.30.1-0ubuntu1~24.04.1
)
CE_PACKAGES=(
    docker-ce=5:29.9.3-1~ubuntu.24.04~noble
    docker-ce-cli=5:29.9.3-1~ubuntu.24.04~noble
    containerd.io=2.4.2-4~ubuntu.24.04~noble
    docker-compose-plugin=5.7.1-1~ubuntu.24.04~noble
    docker-buildx-plugin=0.39.0-1~ubuntu.24.04~noble
)
# Docker's release signing key (https://docs.docker.com/engine/install/ubuntu/).
CE_KEY_FINGERPRINT=9DC858229FC7DD38854AE2D88D81803C0EBFCD88
TOOLS=(sqlite3 build-essential)

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
. /etc/os-release
if [ "${ID:-}" != ubuntu ] || [ "${VERSION_ID:-}" != 24.04 ]; then
    echo "bootstrap_vm: Ubuntu 24.04 only (this is ${PRETTY_NAME:-unknown})" >&2
    exit 2
fi
case "$(uname -m)" in
    x86_64) TARGET=x86_64-unknown-linux-gnu ;;
    aarch64) TARGET=aarch64-unknown-linux-gnu ;;
    *) echo "bootstrap_vm: no uv pin for $(uname -m)" >&2; exit 2 ;;
esac
read -r UV_VERSION UV_SHA256 < <(python3 - "$ROOT/src/benchflow/_uv_pin.py" "$TARGET" <<'PY'
import ast, sys
tree = ast.parse(open(sys.argv[1]).read())
values = {}
for node in tree.body:
    target = getattr(node, "target", None) or (node.targets[0] if isinstance(node, ast.Assign) else None)
    if isinstance(target, ast.Name) and target.id in ("UV_VERSION", "UV_SHA256"):
        values[target.id] = ast.literal_eval(node.value)
print(values["UV_VERSION"], values["UV_SHA256"][sys.argv[2]])
PY
)

# A fresh VM's unattended upgrades hold apt's lock for a while: wait for it.
APT=(sudo DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=900)
apt_install() {
    "${APT[@]}" install -y -q --no-install-recommends "$@"
}

"${APT[@]}" update -q
if [ "$DOCKER" = ubuntu ]; then
    apt_install "${UBUNTU_PACKAGES[@]}" "${TOOLS[@]}"
else
    work=$(mktemp -d)
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o "$work/docker.asc"
    if ! gpg --show-keys --with-colons "$work/docker.asc" | grep -qx "fpr:::::::::$CE_KEY_FINGERPRINT:"; then
        echo "bootstrap_vm: Docker's apt key does not have fingerprint $CE_KEY_FINGERPRINT" >&2
        exit 1
    fi
    sudo install -m 0644 -D "$work/docker.asc" /etc/apt/keyrings/docker.asc
    rm -r -- "$work"
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu noble stable" |
        sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
    "${APT[@]}" update -q
    apt_install "${CE_PACKAGES[@]}" "${TOOLS[@]}"
fi
sudo systemctl enable --now docker >/dev/null

if [ "$DOCKER" = ubuntu ]; then
    compose=$(docker compose version --short)
    proven=$(python3 - "$ROOT/src/benchflow/sandbox/_direct_docker.py" <<'PY'
import ast, sys
for node in ast.parse(open(sys.argv[1]).read()).body:
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "PROVEN_COMPOSE_VERSIONS":
        print(" ".join(sorted(ast.literal_eval(node.value.args[0]))))
PY
)
    case " $proven " in
        *" $compose "*) ;;
        *) echo "bootstrap_vm: Compose $compose is not a proven version ($proven)" >&2; exit 1 ;;
    esac
fi
docker buildx version >/dev/null
command -v sqlite3 >/dev/null && command -v cc >/dev/null
if ! id -nG "$USER" | tr ' ' '\n' | grep -qx docker; then
    sudo usermod -aG docker "$USER"
fi

if [ "$("$HOME/.local/bin/uv" --version 2>/dev/null | awk '{print $2}')" != "$UV_VERSION" ]; then
    work=$(mktemp -d)
    archive="uv-$TARGET.tar.gz"
    curl -fsSL "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/$archive" -o "$work/$archive"
    echo "$UV_SHA256  $work/$archive" | sha256sum -c -
    tar -C "$work" -xzf "$work/$archive"
    mkdir -p "$HOME/.local/bin"
    install -m 0755 "$work/uv-$TARGET/uv" "$work/uv-$TARGET/uvx" "$HOME/.local/bin/"
    rm -r -- "$work"
fi
echo "bootstrap_vm: docker $(sudo docker version --format '{{.Server.Version}}') ($DOCKER), compose $(docker compose version --short), buildx $(docker buildx version | awk '{print $2}'), uv $("$HOME/.local/bin/uv" --version | awk '{print $2}') in ~/.local/bin"
