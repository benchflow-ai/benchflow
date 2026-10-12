#!/bin/sh
# Install bench, BenchFlow's command line, from a GitHub release.
#
#   curl -fsSL https://raw.githubusercontent.com/benchflow-ai/benchflow/bench-rs/install.sh | sh
#   sh install.sh --from bench-0.8.1-x86_64-unknown-linux-musl.tar.gz
#
# It downloads bench-<version>-<target>.tar.gz and its .sha256 from the
# release, checks the checksum, and puts one file, bench, in the install
# folder. No sudo, no Docker, and it never edits your shell profile.
#
# Environment:
#   BENCH_VERSION       version to install, like 0.8.1 or its release tag
#                       bench-v0.8.1 (default: latest)
#   BENCH_INSTALL_DIR   folder to install into (default: ~/.local/bin)
#   BENCH_TARGET        target to download instead of this machine's, like
#                       x86_64-unknown-linux-gnu
#   BENCH_RELEASES_URL  releases address, for a mirror
#                       (default: https://github.com/benchflow-ai/benchflow/releases)
#   BENCH_RELEASES_API  the releases list as JSON, where the latest bench
#                       release is found (default: GitHub's API; a mirror
#                       without it uses <releases>/latest)
#   NO_COLOR            set to turn off color

# Everything is in functions and runs from the last line, so a download cut
# short by the network runs nothing.

REPO=https://github.com/benchflow-ai/benchflow
# The repository's releases as JSON, newest first: the latest bench release
# is found here, since <releases>/latest is often the Python package's.
RELEASES_API="https://api.github.com/repos/benchflow-ai/benchflow/releases?per_page=100"
SCRIPT_URL=https://raw.githubusercontent.com/benchflow-ai/benchflow/bench-rs/install.sh
FROM_SOURCE="cargo install --locked --git $REPO --branch bench-rs benchflow-cli"

usage() {
    cat <<EOF
Install bench, BenchFlow's command line.

Usage: install.sh [--from <tarball>] [--force]

Options:
  --from <tarball>  Install a local bench-<version>-<target>.tar.gz instead of
                    downloading one; its .sha256 is checked when it sits next to it
  --force           Replace an existing bench that this installer did not put there
  -h, --help        Print this help

Environment:
  BENCH_VERSION       version to install, like 0.8.1 or bench-v0.8.1 (default: latest)
  BENCH_INSTALL_DIR   folder to install into (default: ~/.local/bin)
  BENCH_TARGET        target to download instead of this machine's
  BENCH_RELEASES_URL  releases address, for a mirror (default: $REPO/releases)
  BENCH_RELEASES_API  the releases list as JSON, to find the latest bench release
EOF
}

setup_colors() {
    if [ -t 1 ] && [ -z "${NO_COLOR:-}" ] && [ "${TERM:-}" != dumb ]; then
        esc=$(printf '\033')
        bold="${esc}[1m" dim="${esc}[2m" red="${esc}[31m" green="${esc}[32m" yellow="${esc}[33m" reset="${esc}[0m"
    else
        bold='' dim='' red='' green='' yellow='' reset=''
    fi
}

say() { printf '%s\n' "$*"; }
ok() { printf '  %s✓%s %s\n' "$green" "$reset" "$*"; }
warn() { printf '  %s!%s %s\n' "$yellow" "$reset" "$*"; }

# die <what failed> [<next step>...]: print one error block and stop.
die() {
    printf '%s✗ %s%s\n' "$red" "$1" "$reset" >&2
    shift
    first=1
    for line in "$@"; do
        if [ "$first" = 1 ]; then
            printf '  %snext:%s %s\n' "$dim" "$reset" "$line" >&2
            first=0
        else
            printf '        %s\n' "$line" >&2
        fi
    done
    exit 1
}

usage_error() {
    printf '%s✗ %s%s\n' "$red" "$1" "$reset" >&2
    given=''
    printf '  %snext:%s see the options: %s\n' "$dim" "$reset" "$(rerun '' --help)" >&2
    exit 2
}

have() { command -v "$1" >/dev/null 2>&1; }

# A path for people: ~ for the home folder.
# shellcheck disable=SC2088 # the ~ is printed, not expanded
pretty() {
    if [ -z "${HOME:-}" ]; then
        printf '%s' "$1"
        return
    fi
    case "$1" in
        "$HOME") printf '~' ;;
        "$HOME"/*) printf '~/%s' "${1#"$HOME"/}" ;;
        *) printf '%s' "$1" ;;
    esac
}

# rerun [<VAR=value>] [<flag>]: this run's command line plus a setting or a
# flag, written the way it was run (from a file, or piped from curl).
rerun() {
    flags="$given${2:+ $2}"
    flags=${flags# }
    if [ -f "$0" ] && [ "$(basename "$0")" = install.sh ]; then
        printf '%ssh %s%s' "${1:+$1 }" "$0" "${flags:+ $flags}"
    else
        printf 'curl -fsSL %s | %ssh%s' "$SCRIPT_URL" "${1:+$1 }" "${flags:+ -s -- $flags}"
    fi
}

cleanup() {
    if [ -n "${staged:-}" ]; then rm -f "$staged"; fi
    if [ -n "${tmp:-}" ]; then rm -rf "$tmp"; fi
}

# The release target for this machine.
detect_target() {
    os=$(uname -s 2>/dev/null)
    arch=$(uname -m 2>/dev/null)
    case "$os" in
        Darwin)
            # A shell under Rosetta reports x86_64 on Apple silicon, which runs arm64 binaries.
            if [ "$arch" = x86_64 ] && [ "$(sysctl -n sysctl.proc_translated 2>/dev/null)" = 1 ]; then
                arch=arm64
            fi
            case "$arch" in
                arm64 | aarch64) echo aarch64-apple-darwin; return 0 ;;
            esac
            ;;
        Linux)
            case "$arch" in
                # The static (musl) build runs on any x86_64 Linux, whatever its glibc.
                x86_64 | amd64) echo x86_64-unknown-linux-musl; return 0 ;;
            esac
            ;;
        MINGW* | MSYS* | CYGWIN* | Windows_NT)
            die "there is no prebuilt bench for Windows yet" \
                "use WSL and run this installer inside it"
            ;;
    esac
    die "there is no prebuilt bench for $os on $arch yet" \
        "build it from source (needs Rust, https://rustup.rs):" \
        "$FROM_SOURCE"
}

# fetch <url> <file>: download, and say why it failed.
fetch() {
    if have curl; then
        code=$(curl --proto '=https,http' --proto-redir '=https' -sSL --retry 3 --connect-timeout 20 \
            -o "$2" -w '%{http_code}' "$1") || {
            fetch_error="network error (curl exit $?)"
            return 1
        }
        if [ "$code" != 200 ]; then
            fetch_error="HTTP $code"
            return 1
        fi
    else
        wget -q -O "$2" "$1" || {
            fetch_error="wget exit $?"
            return 1
        }
    fi
}

# The newest bench release that is not a prerelease, like bench-v0.8.1, from
# the releases list (BENCH_RELEASES_API, for a mirror that has one). The
# repository's own latest release is often the Python package's (v<version>).
# Whether <url> is https on GitHub's API host: api.github.com, or the host
# of GITHUB_API_URL when that is https.
token_host() {
    # A token for a GitHub Enterprise server (GITHUB_API_URL elsewhere) is
    # not github.com's, so it goes only to its own server.
    case "$1" in
        https://api.github.com/*)
            case "${GITHUB_API_URL:-https://api.github.com}" in
                https://api.github.com | https://api.github.com/) return 0 ;;
                *) return 1 ;;
            esac
            ;;
    esac
    case "${GITHUB_API_URL:-}" in
        https://?*)
            ghe=${GITHUB_API_URL#https://}
            ghe=${ghe%%/*}
            case "$1" in
                "https://$ghe"/*) return 0 ;;
            esac
            ;;
    esac
    return 1
}

# Sets found, or returns 1 with fetch_error saying why.
latest_bench_tag() {
    list=$tmp/releases.json
    api=${BENCH_RELEASES_API:-$RELEASES_API}
    # GitHub allows 60 unauthenticated API calls an hour per address (shared
    # CI runners run out); a GITHUB_TOKEN raises that. It is sent only over
    # https to GitHub's API (api.github.com, or GITHUB_API_URL's host, as
    # GitHub Actions sets it for an Enterprise server), never to a mirror.
    # The header goes to curl as a config on stdin, never in its arguments,
    # which other users of this machine can read; a token with other
    # characters than a token's is not sent at all.
    case "${GITHUB_TOKEN:-}" in
        '' | *[!A-Za-z0-9_.-]*) send_token=0 ;;
        *) send_token=1 ;;
    esac
    if [ "$send_token" = 1 ] && have curl && token_host "$api"; then
        code=$(printf 'header = "Authorization: Bearer %s"\n' "$GITHUB_TOKEN" |
            curl -K - --proto '=https' --proto-redir '=https' -sSL --retry 3 --connect-timeout 20 \
                -o "$list" -w '%{http_code}' "$api") || {
            fetch_error="network error (curl exit $?)"
            return 1
        }
        case "$code" in
            200) ;;
            # A stale or foreign token is refused even on public data:
            # ask again without it.
            401 | 403) fetch "$api" "$list" || return 1 ;;
            *)
                fetch_error="HTTP $code"
                return 1
                ;;
        esac
    else
        fetch "$api" "$list" || return 1
    fi
    # One field per line in GitHub's JSON; "tag_name" comes before
    # "prerelease" in each release.
    found=$(tr ',' '\n' <"$list" | awk '
        /"tag_name"[ ]*:/ { t = $0; sub(/.*"tag_name"[ ]*:[ ]*"/, "", t); sub(/".*/, "", t); next }
        /"prerelease"[ ]*:[ ]*false/ { if (t ~ /^bench-v[A-Za-z0-9_.+-]+$/) { print t; exit } }
        /"prerelease"[ ]*:/ { t = "" }
    ')
    [ -n "$found" ] || {
        fetch_error='it lists no bench-v<version> release that is not a prerelease'
        return 1
    }
}

# The tag that <releases>/latest redirects to, like bench-v0.8.1.
latest_tag() {
    if have curl; then
        final=$(curl --proto '=https,http' --proto-redir '=https' -fsSLI --retry 3 --connect-timeout 20 \
            -o /dev/null -w '%{url_effective}' "$releases/latest" 2>/dev/null) || return 1
    elif have wget; then
        final=$(wget -S --spider "$releases/latest" 2>&1 |
            sed -n 's/^ *[Ll]ocation: *\([^ ]*\).*/\1/p' | tail -n 1 | tr -d '\r')
    else
        return 1
    fi
    case "$final" in
        */releases/tag/?*) printf '%s\n' "${final##*/releases/tag/}" ;;
        *) return 1 ;;
    esac
}

sha256_of() {
    if have sha256sum; then
        sha256sum "$1" | awk '{print $1}'
    elif have shasum; then
        shasum -a 256 "$1" | awk '{print $1}'
    elif have openssl; then
        openssl dgst -sha256 "$1" | awk '{print $NF}'
    else
        return 1
    fi
}

# verify <tarball> <sha256 file>
verify() {
    expected=$(awk 'NR == 1 {print $1}' "$2" | tr 'A-F' 'a-f')
    case "$expected" in
        *[!0-9a-f]*) expected='' ;;
    esac
    if [ "${#expected}" != 64 ]; then
        die "$(basename "$2") is not a sha256 checksum file" \
            "download it again, or report it: $REPO/issues"
    fi
    actual=$(sha256_of "$1") || die "cannot check the download: no sha256sum, shasum, or openssl here" \
        "install one of them (coreutils has sha256sum) and run this again"
    if [ "$actual" != "$expected" ]; then
        die "the checksum of $(basename "$1") does not match; nothing was installed
  expected $expected
  got      $actual" \
            "run this again; if it keeps failing, report it: $REPO/issues"
    fi
}

# path_hint <dir>: how to put <dir> on PATH in this user's shell.
# shellcheck disable=SC2088 # the ~ is printed, not expanded
path_hint() {
    shown=$1
    if [ -n "${HOME:-}" ]; then
        case "$1" in
            "$HOME"/*) shown="\$HOME/${1#"$HOME"/}" ;;
        esac
    fi
    case "$(basename "${SHELL:-sh}")" in
        zsh) rc='~/.zshrc' ;;
        bash) if [ "$(uname -s)" = Darwin ]; then rc='~/.bash_profile'; else rc='~/.bashrc'; fi ;;
        fish)
            warn "$(pretty "$1") is not on your PATH. Add it for every new terminal with:"
            say "      fish_add_path $(pretty "$1")"
            return
            ;;
        *) rc='~/.profile' ;;
    esac
    warn "$(pretty "$1") is not on your PATH. Add it for every new terminal with:"
    say "      echo 'export PATH=\"$shown:\$PATH\"' >> $rc"
    say "    and for this one: export PATH=\"$shown:\$PATH\""
}

main() {
    setup_colors
    given=$*
    from=''
    force=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --from)
                [ $# -ge 2 ] || usage_error "--from needs a tarball path"
                from=$2
                shift 2
                ;;
            --from=*)
                from=${1#--from=}
                shift
                ;;
            --force)
                force=1
                shift
                ;;
            -h | --help)
                usage
                exit 0
                ;;
            *) usage_error "unknown argument: $1" ;;
        esac
    done

    if [ -z "${HOME:-}" ] && [ -z "${BENCH_INSTALL_DIR:-}" ]; then
        die "HOME is not set, so there is no default install folder" \
            "set BENCH_INSTALL_DIR to the folder to install into"
    fi
    dir=${BENCH_INSTALL_DIR:-$HOME/.local/bin}
    case "$dir" in
        "~") dir=$HOME ;;
        "~"/*) dir=$HOME/${dir#"~"/} ;;
    esac
    case "$dir" in
        /*) ;;
        *) dir=$(pwd)/$dir ;;
    esac
    dir=${dir%/}
    dest=$dir/bench

    tmp=$(mktemp -d 2>/dev/null || mktemp -d -t bench-install) ||
        die "cannot create a temporary folder" "check that TMPDIR (or /tmp) exists and has space"
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    if [ -n "$from" ]; then
        [ -f "$from" ] || die "no such file: $from" "pass the path of a bench-<version>-<target>.tar.gz"
        say "Installing bench from $(pretty "$from")"
        tarball=$from
        if [ -f "$from.sha256" ]; then
            verify "$tarball" "$from.sha256"
            ok "checksum   matches $(basename "$from").sha256"
        else
            warn "no $(basename "$from").sha256 next to it, so the checksum was not checked"
        fi
    else
        if ! have curl && ! have wget; then
            die "neither curl nor wget is installed, so nothing can be downloaded" \
                "install curl and run this again, or download the tarball yourself and use --from"
        fi
        releases=${BENCH_RELEASES_URL:-$REPO/releases}
        releases=${releases%/}
        target=${BENCH_TARGET:-}
        [ -n "$target" ] || target=$(detect_target) || exit 1
        case "$target" in
            *[!A-Za-z0-9_.-]*) usage_error "BENCH_TARGET is not a target name: $target" ;;
        esac
        tag=${BENCH_VERSION:-latest}
        case "$tag" in
            *[!A-Za-z0-9_.+-]*) usage_error "BENCH_VERSION is not a release tag: $tag" ;;
        esac
        which=''
        if [ "$tag" = latest ] && { [ -z "${BENCH_RELEASES_URL:-}" ] || [ -n "${BENCH_RELEASES_API:-}" ]; }; then
            # From the releases list; a mirror without one uses <releases>/latest.
            # When the list cannot be read (GitHub's API limit), the
            # repository's latest release, if it is bench's.
            if ! latest_bench_tag; then
                listed_error=$fetch_error
                found=$(latest_tag) || found=''
                case "$found" in
                    bench-v*) ;;
                    *) found='' ;;
                esac
            fi
            [ -n "$found" ] ||
                die "found no bench release in ${BENCH_RELEASES_API:-$RELEASES_API} ($listed_error)" \
                    "pick a release from $releases and run: $(rerun 'BENCH_VERSION=<version>')" \
                    "or build from source: $FROM_SOURCE"
            tag=$found
            which=' (the latest)'
        fi
        if [ "$tag" = latest ]; then
            tag=$(latest_tag) || die "could not find the latest release at $releases/latest" \
                "pick a release from $releases and run: $(rerun 'BENCH_VERSION=<version>')"
            # bench's releases are tagged bench-v<version>; v<version> tags
            # are the Python package's, and have no bench in them.
            case "$tag" in
                bench-v*) ;;
                *) die "the latest release at $releases/latest is $tag, the Python package's, not bench's" \
                    "bench's releases are tagged bench-v<version>: pick one from $releases" \
                    "and run: $(rerun 'BENCH_VERSION=<version>')" \
                    "or build from source: $FROM_SOURCE" ;;
            esac
            which=' (the latest)'
        fi
        case "$tag" in
            bench-v*) ;;
            v*) tag=bench-$tag ;;
            *) tag=bench-v$tag ;;
        esac
        version=${tag#bench-v}
        asset=bench-$version-$target.tar.gz
        say "Installing bench from release $tag$which for $target"
        tarball=$tmp/$asset
        if ! fetch "$releases/download/$tag/$asset" "$tarball"; then
            die "could not download $asset from release $tag ($fetch_error)" \
                "check the release has it: $releases/tag/$tag" \
                "bench's releases are tagged bench-v<version>; v<version> releases ship only the Python package (pip install benchflow);" \
                "pick one that has bench: $(rerun 'BENCH_VERSION=<version>')" \
                "or build from source: $FROM_SOURCE"
        fi
        fetch "$releases/download/$tag/$asset.sha256" "$tmp/$asset.sha256" ||
            die "could not download $asset.sha256 from release $tag ($fetch_error)" \
                "run this again; a release without checksums is not installed"
        ok "downloaded $asset ($(wc -c <"$tarball" | awk '{printf "%.1f MB", $1 / 1000000}'))"
        verify "$tarball" "$tmp/$asset.sha256"
        ok "checksum   sha256 matches"
    fi

    mkdir "$tmp/x"
    tar -xzf "$tarball" -C "$tmp/x" 2>"$tmp/tar.err" ||
        die "could not unpack $(basename "$tarball"): $(head -n 1 "$tmp/tar.err")" \
            "check it is a bench-<version>-<target>.tar.gz; download it again if it is truncated"
    bin=''
    for candidate in "$tmp/x/bench" "$tmp/x"/*/bench; do
        if [ -f "$candidate" ]; then
            bin=$candidate
            break
        fi
    done
    [ -n "$bin" ] || die "$(basename "$tarball") has no bench in it" \
        "check it is a bench-<version>-<target>.tar.gz"
    chmod 755 "$bin"

    # Run it before replacing anything: a wrong architecture or C library fails here.
    if ! got=$("$bin" --version 2>&1); then
        case "$(uname -s)" in
            Darwin) also="if macOS blocked it, the tarball came from a browser: xattr -d com.apple.quarantine <tarball>" ;;
            Linux) also="the x86_64-unknown-linux-musl build needs no glibc and runs on any x86_64 Linux" ;;
            *) also='' ;;
        esac
        die "the bench in $(basename "$tarball") does not run here: $(printf '%s' "$got" | head -n 1)" \
            "check the tarball was built for this machine ($(uname -s) $(uname -m))" ${also:+"$also"}
    fi
    case "$got" in
        "bench "*) ;;
        *) die "$(basename "$tarball") holds a bench that does not report a version" \
            "check it is a bench-<version>-<target>.tar.gz" ;;
    esac
    if [ -n "${version:-}" ]; then
        case "$got" in
            "bench $version" | "bench $version "*) ;;
            *) warn "release $tag holds $got" ;;
        esac
    fi

    if [ -d "$dest" ] && [ ! -L "$dest" ]; then
        die "$(pretty "$dest") is a folder" "remove it, or set BENCH_INSTALL_DIR to another folder"
    fi
    # Another tool's bench (BenchFlow 0.8's Python CLI is a script, or a link
    # into a uv or pipx environment) is only replaced on request.
    if [ "$force" = 0 ] && { [ -L "$dest" ] || [ "$(dd if="$dest" bs=2 count=1 2>/dev/null)" = '#!' ]; }; then
        if [ -L "$dest" ]; then
            what="a link to $(pretty "$(readlink "$dest")")"
        else
            what='a script'
        fi
        back=''
        if [ "$(dd if="$dest" bs=2 count=1 2>/dev/null)" = '#!' ] && grep -q benchflow "$dest" 2>/dev/null; then
            what="$what: BenchFlow 0.8's Python CLI"
            back=" (uv tool install --force benchflow brings it back)"
        fi
        die "$(pretty "$dest") already exists and is $what; nothing was changed" \
            "install next to it: $(rerun 'BENCH_INSTALL_DIR=~/.bench/bin')" \
            "or replace it: $(rerun '' --force)$back"
    fi
    mkdir -p "$dir" 2>/dev/null || die "cannot create $(pretty "$dir")" \
        "set BENCH_INSTALL_DIR to a folder you can write to; no sudo needed"
    [ -w "$dir" ] || die "cannot write to $(pretty "$dir")" \
        "set BENCH_INSTALL_DIR to a folder you can write to; no sudo needed"
    replacing=0
    if [ -e "$dest" ] || [ -L "$dest" ]; then replacing=1; fi
    staged=$dir/.bench.install.$$
    if ! { cp "$bin" "$staged" && chmod 755 "$staged" && mv -f "$staged" "$dest"; }; then
        die "could not write $(pretty "$dest")" "check the folder has space and you own it"
    fi
    staged=''
    if [ "$replacing" = 1 ]; then
        ok "installed  $(pretty "$dest") (replaced the previous one): $got"
    else
        ok "installed  $(pretty "$dest"): $got"
    fi

    run=bench
    case ":${PATH:-}:" in
        *":$dir:"* | *":$dir/:"*)
            found=$(command -v bench 2>/dev/null || true)
            if [ -n "$found" ] && [ "$found" != "$dest" ]; then
                warn "$(pretty "$found") comes first on your PATH, so \`bench\` runs that one"
                say "    remove it, or put $(pretty "$dir") earlier in PATH"
                run=$(pretty "$dest")
            else
                ok "on PATH    $(pretty "$dir")"
            fi
            ;;
        *)
            path_hint "$dir"
            run=$(pretty "$dest")
            ;;
    esac

    width=$((${#run} + 12))
    say ""
    say "${bold}Next:${reset}"
    printf "  %-${width}s  %s\n" "$run doctor" "${dim}check this machine: sandboxes, agent logins, disk${reset}"
    printf "  %-${width}s  %s\n" "$run new my-task" "${dim}create a task that already passes its own test${reset}"
}

main "$@"
