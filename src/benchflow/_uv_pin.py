"""The one uv BenchFlow installs in a sandbox, and the Python its venvs use.

Every place BenchFlow fetches uv inside a sandbox (the in-sandbox LiteLLM
proxy, the deepagents and openhands agent installs, the Toolathlon task
images) uses :func:`uv_install_command`, so the version and its checksums
live here and nowhere else. CI and the build VMs use the same version.

Why pin: uv 0.13.0 made Python 3.15 the default when no suitable Python is
installed, and fastuuid (which LiteLLM 1.91 needs) has no cp315 wheel, so an
unpinned ``uv venv`` on an image without Python broke the proxy install
before the agent started (TB2's 34 images without Python). Why a checksum:
the install runs as the sandbox's root before the agent, and the proxy it
builds holds the provider key, so the tarball is checked against the sum the
release published before uv ever runs.

The proxy's venv is built on a uv-managed CPython ``PYTHON_VERSION`` in
``PYTHON_INSTALL_DIR`` (``UV_PYTHON_PREFERENCE=only-managed``): the image's
own Python varies and is the task's, not BenchFlow's. The version is the
exact patch release (``3.12.15``, selected with the original uv 0.13.1
pin): ``uv python install 3.12`` takes any 3.12 already in
the folder, so a minor version alone would let an image (or an older
BenchFlow's layer) decide which patch the proxy runs on.

Every uv call that builds a BenchFlow venv runs as root with uv's settings
taken from :data:`MANAGED_PYTHON_ENV` only (:func:`managed_python_exports`):
no ``uv.toml`` or ``pyproject.toml`` from the working folder (the agent's
workspace, which it can write before a later role's proxy installs), and no
``UV_*`` variable from the image. Either could otherwise swap uv's
download table for one with no checksums (``python-downloads-json-url``),
or point uv at a cache or index of the task's choosing.
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath

from benchflow._root_only import root_only_tree_check
from benchflow._runtime_tree import RUNTIME_ROOT

UV_VERSION = "0.13.4"

# From the release's published ``.sha256`` files, fetched 2026-10-24:
# https://github.com/astral-sh/uv/releases/download/0.13.4/uv-<target>.tar.gz.sha256
UV_SHA256: Mapping[str, str] = {
    "x86_64-unknown-linux-gnu": (
        "2b428396f12572d4429cf210ca234dd599a33b3af0958a4d5ed569d63e3f344f"
    ),
    "aarch64-unknown-linux-gnu": (
        "1ae72b4718729e2ccc4f1ae2c6d1e531ca8467800e0546e9130b3886023a60ab"
    ),
    "x86_64-unknown-linux-musl": (
        "28d4385f9f9d1d1c7654d85b3855e0e2615b2f6cc13136a2edff858bbdc16ff1"
    ),
    "aarch64-unknown-linux-musl": (
        "253a17cadcb659ae77a0fe3b9a2f1c14297528d480c421899e61e64d64f24e8f"
    ),
}

# Tried in order. Astral's mirror serves the same files as the GitHub release
# and is also the host uv downloads managed Pythons from, so a sandbox that
# reaches one host gets both; the checksum is what makes either one safe.
UV_DOWNLOAD_BASES: tuple[str, ...] = (
    f"https://releases.astral.sh/github/uv/releases/download/{UV_VERSION}",
    f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}",
)

# Where BenchFlow keeps what it installs in a sandbox and later runs as root
# (uv, the managed Python, the model proxy's cached venv, agent tools): the
# protected runtime tree. No workspace grant covers it
# (``_runtime_tree.workspace_grant_problem``), and each root use first checks
# it root's alone with ``_root_only.root_only_tree_check`` (C8).
BENCHFLOW_ROOT = RUNTIME_ROOT

# A versioned folder, so a new pin never reuses an older binary.
UV_DIR = f"{BENCHFLOW_ROOT}/uv/{UV_VERSION}"
UV_BIN = f"{UV_DIR}/uv"

PYTHON_VERSION = "3.12.15"
PYTHON_INSTALL_DIR = f"{BENCHFLOW_ROOT}/python"
# uv's cache for BenchFlow's root installs, in the protected runtime tree
# (C8): a cache under a home the agent user was granted (``/root`` is the
# workspace of an image without a WORKDIR) would hand root its wheels.
# uv 0.13.1 makes every lock file it creates writable by everyone (mode 666
# whatever the umask; one per cached wheel, the cache's own ``.lock``), so
# the cache's parent is root's private folder (700) while uv runs, and
# :func:`uv_lock_seal_command` takes the write bits off afterwards: nothing
# under the runtime tree is ever writable by another user (DET-R1).

UV_CACHE_DIR = f"{BENCHFLOW_ROOT}/cache/uv"
# Where uv 0.13.1 fetches managed CPython (python-build-standalone builds):
# Astral's mirror first, the GitHub releases as the fallback.
PYTHON_DOWNLOAD_HOSTS: tuple[str, ...] = ("releases.astral.sh", "github.com")

# Set for every uv call that builds a BenchFlow venv, after every other
# ``UV_*`` variable is unset. ``UV_PYTHON_DOWNLOADS`` overrides an image that
# sets ``never``, which would refuse even an explicit ``uv python install``.
# ``UV_NO_CONFIG`` keeps uv from reading a ``uv.toml`` or ``pyproject.toml`` in
# the working folder (or the user's or system's). ``UV_CACHE_DIR`` is in the
# protected tree: uv trusts the hashes its cache records, and root's default
# cache sits under ``/root``, which is the agent's workspace when the image's
# working folder is ``/``.
MANAGED_PYTHON_ENV: Mapping[str, str] = {
    "UV_PYTHON_PREFERENCE": "only-managed",
    "UV_PYTHON_INSTALL_DIR": PYTHON_INSTALL_DIR,
    "UV_PYTHON_DOWNLOADS": "automatic",
    "UV_NO_CONFIG": "1",
    "UV_CACHE_DIR": UV_CACHE_DIR,
}


def _hosts(bases: Sequence[str]) -> str:
    hosts: list[str] = []
    for base in bases:
        host = base.split("://", 1)[-1].split("/", 1)[0]
        if host not in hosts:
            hosts.append(host)
    return " or ".join(hosts)


def uv_install_command(
    *,
    uv_dir: str = UV_DIR,
    sums: Mapping[str, str] | None = None,
    bases: Sequence[str] = UV_DOWNLOAD_BASES,
    install_packages: bool = True,
) -> str:
    """One line of POSIX sh that leaves the pinned uv at ``<uv_dir>/uv``.

    It is a subshell, so it composes with ``&&`` and exits non-zero on any
    failure (it never relies on ``set -e``, which a caller's ``&&`` turns
    off). If ``<uv_dir>/uv`` already reports the pinned version it does
    nothing. Otherwise it picks the release for the machine (x86_64 or
    aarch64) and libc (musl when ``/lib/ld-musl-*`` exists, else glibc),
    downloads it with curl or wget (installing curl with the image's package
    manager when it has neither), and checks its sha256 with the first of
    ``sha256sum``, ``shasum -a 256``, ``openssl dgst -sha256`` or
    ``busybox sha256sum`` the image has; with none of them it stops rather
    than skip the check. Only a tarball whose sum matches is unpacked; a
    mismatch stops with the file name and the expected and actual sums.

    ``install_packages=False`` never changes system packages. On minimal
    Debian/Ubuntu it downloads with the existing apt-helper; authenticity is
    established by the pinned SHA-256, including when no CA store exists.
    ``sums`` and ``bases`` exist for tests; real installs use the pins.
    """
    sums = UV_SHA256 if sums is None else sums
    version = UV_VERSION
    cases = " ".join(
        f"{shlex.quote(target)}) want={shlex.quote(digest)} ;;"
        for target, digest in sorted(sums.items())
    )
    urls = " ".join(shlex.quote(base) for base in bases)
    hosts = _hosts(bases)
    steps = [
        # What it creates is writable by root alone, whatever the image's
        # umask: the check below refuses a group-writable folder next time.
        "umask 022",
        f"d={shlex.quote(uv_dir)}",
        # Root runs nothing from the folder, its version probe included,
        # before it is shown root's alone along its whole route (C8,
        # PT-C8-R3-5).
        'if [ "$(id -u)" = 0 ]; then mkdir -p "$d" || exit 1; '
        + root_only_tree_check(uv_dir, "BenchFlow's uv folder")
        + " || exit 1; fi",
        f'case "$("$d/uv" --version 2>/dev/null)" in '
        f'"uv {version}"|"uv {version} "*) exit 0 ;; esac',
        'case "$(uname -m)" in x86_64|amd64) arch=x86_64 ;; '
        "aarch64|arm64) arch=aarch64 ;; "
        f'*) echo "benchflow: no pinned uv {version} for machine $(uname -m)" >&2; '
        "exit 1 ;; esac",
        "libc=gnu",
        'for f in /lib/ld-musl-*; do if [ -e "$f" ]; then libc=musl; fi; done',
        'target="$arch-unknown-linux-$libc"',
        'want=""',
        f'case "$target" in {cases} *) '
        f'echo "benchflow: no pinned uv {version} for $target" >&2; exit 1 ;; esac',
        'file="uv-$target.tar.gz"',
        # Download beside the uv folder, never under /tmp: /tmp's owner (the
        # agent, when /tmp is its workspace) could swap the tarball between
        # the sum check and the unpack. The folders above passed the check.
        f"mkdir -p {shlex.quote(str(PurePosixPath(uv_dir).parent))} || exit 1",
        'tmp="$(mktemp -d '
        + shlex.quote(str(PurePosixPath(uv_dir).parent))
        + '/.download.XXXXXXXXXX)" || exit 1',
        "trap 'rm -rf \"$tmp\"' EXIT",
        # Slim images (ubuntu:24.04 itself) ship neither curl nor wget.
        (
            "if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; "
            "then if command -v apt-get >/dev/null 2>&1; then "
            "{ DEBIAN_FRONTEND=noninteractive apt-get update -qq && "
            "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
            "--no-install-recommends curl ca-certificates; } >/dev/null 2>&1; "
            "elif command -v apk >/dev/null 2>&1; then "
            "apk add --no-cache curl ca-certificates >/dev/null 2>&1; "
            "elif command -v dnf >/dev/null 2>&1; then "
            "dnf -y install curl ca-certificates >/dev/null 2>&1; "
            "elif command -v yum >/dev/null 2>&1; then "
            "yum -y install curl ca-certificates >/dev/null 2>&1; fi; fi"
            if install_packages
            else ":"
        ),
        "if command -v curl >/dev/null 2>&1; then "
        'fetch() { curl -fsSL --retry 3 --connect-timeout 20 -o "$2" "$1"; }; '
        "elif command -v wget >/dev/null 2>&1; then "
        'fetch() { wget -q -T 60 -O "$2" "$1"; }; '
        # apt-helper is already in minimal Debian/Ubuntu. No apt update or
        # package install: only the pinned archive is written, in the private
        # download folder. The mandatory digest below authenticates the bytes
        # even on images without ca-certificates. No downloaded code runs first.
        + (
            "elif [ -x /usr/lib/apt/apt-helper ]; then "
            "fetch() { /usr/lib/apt/apt-helper -o Acquire::https::Verify-Peer=false "
            '-o Acquire::Retries=3 download-file "$1" "$2" "SHA256:$want"; }; '
            if not install_packages
            else ""
        )
        + f'else echo "benchflow: cannot download uv {version}: the image has '
        + (
            "neither curl nor wget, and installing curl failed"
            if install_packages
            else "no supported downloader; requires curl, wget, or Debian's apt-helper"
        )
        + '" >&2; exit 1; fi',
        f'ok=0; for base in {urls}; do if fetch "$base/$file" "$tmp/$file"; '
        "then ok=1; break; fi; done",
        f'if [ "$ok" != 1 ]; then echo "benchflow: could not download $file '
        f'(uv {version}) from {hosts}" >&2; exit 1; fi',
        'got=""',
        "if command -v sha256sum >/dev/null 2>&1; then "
        'got="$(sha256sum "$tmp/$file")"; '
        "elif command -v shasum >/dev/null 2>&1; then "
        'got="$(shasum -a 256 "$tmp/$file")"; '
        "elif command -v openssl >/dev/null 2>&1; then "
        'got="$(openssl dgst -sha256 -r "$tmp/$file")"; '
        "elif command -v busybox >/dev/null 2>&1; then "
        'got="$(busybox sha256sum "$tmp/$file")"; '
        'else echo "benchflow: cannot check the sha256 of $file: the image has '
        'no sha256sum, shasum, openssl or busybox; uv not installed" >&2; '
        "exit 1; fi",
        'got="${got%% *}"',
        'if [ "$got" != "$want" ]; then echo "benchflow: sha256 mismatch for '
        '$file: expected $want, got ${got:-nothing}; uv not installed" >&2; '
        "exit 1; fi",
        'tar -xzf "$tmp/$file" -C "$tmp" || exit 1',
        'mkdir -p "$d" || { echo "benchflow: cannot create $d for uv" >&2; exit 1; }',
        'cp "$tmp/uv-$target/uv" "$tmp/uv-$target/uvx" "$d/" || exit 1',
        'chmod 755 "$d/uv" "$d/uvx"',
    ]
    return "( " + "; ".join(steps) + " )"


def private_tmpdir_exports(folder: str) -> str:
    """sh that makes ``folder`` root's private temp folder and exports it as
    ``TMPDIR`` (``TMP``, ``TEMP``) for the root uv steps that follow.

    uv keeps its Python downloads and build files in ``TMPDIR``, unpacked
    after their hashes were checked. Under ``/tmp`` the agent can own the
    parent and swap those files before root installs them (FF-install-C12).
    ``folder`` belongs in the protected runtime tree; as root it is checked
    root's alone (``_root_only.root_only_tree_check``) once made. It exits
    the shell on failure: ``set -e`` does not stop at a failed check inside
    a list, and uv must never fall back to ``/tmp``.
    """
    q = shlex.quote(folder)
    check = root_only_tree_check(folder, "the folder for uv temp files")
    return (
        f"{{ ( umask 077; mkdir -p {q} ) && "
        f'{{ [ "$(id -u)" != 0 ] || {check}; }} && '
        f"export TMPDIR={q} TMP={q} TEMP={q} || exit 1; }}"
    )


def uv_lock_file_command(folder: str) -> str:
    """sh that makes ``<folder>/.lock`` root's, mode 644, unless it exists.

    uv 0.13.1 opens a lock file that exists as it is, and creates a missing
    one writable by everyone (mode 666, past the umask). Run it on a folder
    uv will lock (the managed Python's, a venv's) before the uv step that
    would make it, so that file is never another user's to write (DET-R1).
    """
    lock = shlex.quote(f"{folder.rstrip('/')}/.lock")
    return f"{{ [ -e {lock} ] || ( umask 022; : > {lock} ); }}"


def uv_lock_seal_command(*folders: str) -> str:
    """sh that takes group and other write off every root-owned uv lock file
    in uv's cache, the managed Python and ``folders``: ``*.lock``, and the
    files of a git cache's ``locks`` folder (``git-v1/locks/<hash>``), which
    have no suffix.

    Run it as root after a recipe's last uv step and before anything checks
    the runtime tree: uv makes the lock files it creates (one per cached
    wheel among them) writable by everyone, which the root-only check of
    ``/opt/runtime`` refuses. A lock file holds no data; only its mode
    changes. Folders that do not exist are skipped (DET-R1). An agent uv
    installs from git (openhands) left a git lock writable, and the
    in-sandbox model proxy then refused the cache.
    """
    names = " ".join(
        shlex.quote(f) for f in (UV_CACHE_DIR, PYTHON_INSTALL_DIR, *folders)
    )
    return (
        f'{{ for bf_d in {names}; do [ ! -d "$bf_d" ] || '
        "find \"$bf_d\" -type f \\( -name '*.lock' -o -path '*/git-v[0-9]*/locks/*' \\) "
        "-user 0 \\( -perm -0020 -o -perm -0002 \\) -exec chmod go-w {} + || exit 1; done; }"
    )


def managed_python_exports() -> str:
    """sh that unsets every inherited ``UV_*`` variable, then exports
    :data:`MANAGED_PYTHON_ENV`: uv's settings are BenchFlow's alone.

    ``UV_PYTHON_DOWNLOADS_JSON_URL`` and ``UV_CONFIG_FILE`` are named as well,
    so they go even on an image whose ``env`` or ``sed`` cannot list them.
    """
    clear = (
        # uv's folders (the managed Python, the venv) writable by root alone,
        # whatever the image's umask (the root-only check refuses group-writable).
        "umask 022; "
        "unset UV_PYTHON_DOWNLOADS_JSON_URL UV_CONFIG_FILE; "
        "for v in $(env | sed -n 's/^\\(UV_[A-Za-z0-9_]*\\)=.*/\\1/p'); "
        'do unset "$v"; done'
    )
    return f"{clear}; export " + " ".join(
        f"{key}={shlex.quote(value)}" for key, value in MANAGED_PYTHON_ENV.items()
    )


def managed_python_install_command(uv: str = '"$UV"') -> str:
    """Install the pinned managed CPython, or stop naming the download hosts.

    ``uv`` is the shell word that runs uv. Run it after
    :func:`managed_python_exports`. ``--no-bin`` keeps uv from adding
    ``python3.12`` links to the sandbox's PATH: the task's own Python stays
    the one the agent sees. Reinstall from the pinned uv catalog: matching
    only the patch version accepts another python-build-standalone build
    already in the image (FF-install-C4). Cached layers avoid this download;
    their recipe identity includes this command and their image ID is trusted.
    """
    hosts = " (fallback: ".join(PYTHON_DOWNLOAD_HOSTS) + ")"
    # uv runs the managed Pythons it finds there (C8): their folder, and
    # its cache, are checked root's alone first. Then the cache's parent is
    # made root's private folder again (an agent install's ``chmod -R a+rX``
    # opens it) and the managed Python's lock file is made before uv would
    # make it writable by everyone (DET-R1).
    guard = " && ".join(
        [
            f"mkdir -p {shlex.quote(PYTHON_INSTALL_DIR)}",
            f"( umask 077; mkdir -p {shlex.quote(UV_CACHE_DIR)} )",
            root_only_tree_check(PYTHON_INSTALL_DIR, "BenchFlow's managed Python"),
            root_only_tree_check(UV_CACHE_DIR, "BenchFlow's uv cache"),
            f"chmod 700 {shlex.quote(str(PurePosixPath(UV_CACHE_DIR).parent))}",
            uv_lock_file_command(PYTHON_INSTALL_DIR),
        ]
    )
    return (
        f'if [ "$(id -u)" = 0 ]; then {{ {guard}; }} || exit 1; fi; '
        f"{uv} python install -q --no-bin --reinstall {PYTHON_VERSION} || "
        f'{{ echo "benchflow: uv {UV_VERSION} could not install managed CPython '
        f"{PYTHON_VERSION} into {PYTHON_INSTALL_DIR}; the sandbox must reach "
        f'{hosts} before the agent starts" >&2; exit 1; }}'
    )
