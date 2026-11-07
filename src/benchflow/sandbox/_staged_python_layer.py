"""A cached delegate Python, authenticated before its first instruction.

The expected tree hashes are derived from the checksum-verified archives by
tools/staged_python_digest.py. GNU tar encodes every name, type, owner, mode,
link target and file byte; only timestamps and owner names are normalized.
The pinned links stay within the tree. Matching that stream proves content
and custody together, after checking the route to its private root. Neither
an image's marker nor its Python participates in this proof.

Images without the required tar format or hash tool stage afresh as before.
"""

from __future__ import annotations

import shlex

from benchflow._root_only import root_only_path_check

# Key by archive digest: a new pin without a derived tree hash cannot reuse.
TREE_SHA256 = {
    "178d3d83b6e8e99285a65d3fce9c16b87952cc5efdb4d1a0e3ff3be18ff0fc9c": "81ed91eb97ae23572309c0b4defbe1b8523cd95ac3ec3dbc47b167eb347f8175",
    "9467a99bacc7d90b64b02de7fa9ecfc99b72905e64e74f7ff6b460ad95cfed5a": "5ed46aaf1c6ca20927484231d206b3461fdb3ff6e961d976ab509a1a685e564a",
    "99e53f010e58b62dd5bc849cc4822bde9fff7171c6d020fbe4afed2d861ccc77": "adc7aa290a608aefa807dd92fdea40d8908b53ce48f7f87e236b553784ff50b9",
    "efd6e0900431214a9b559803071939421a8a8489375b9be063221a8cfe991c60": "3bac2e344a8eb63fc7b1a8f40d41c087feb2168da4ec54805d4fae361a84bd64",
}
TAR_OPTIONS = (
    "--sort=name --format=gnu --mtime=@0 --numeric-owner "
    "--hard-dereference --blocking-factor=20"
)
TREE_SUFFIX = "\nbenchflow-staged-python-v1\n"
FOOTPRINT_MB = 192
IMAGE_PYTHONS = ("/usr/local/bin/python3", "/usr/bin/python3", "/bin/python3")
BUNDLED_ARCHIVE = "/tmp/runtime-delegate-python.tar.gz"


def archive_add_line() -> str:
    """Supply Daytona's usual platform even in images without a downloader."""
    from benchflow.sandbox import _delegate_disarm as stage

    name = stage.staged_python_file("x86_64", "gnu").replace("+", "%2B")
    return f"ADD {stage.STAGED_PYTHON_BASES[0]}/{name} {BUNDLED_ARCHIVE}"


def tree_check(arch: str, libc: str) -> str:
    """Check with custody-checked image tools; never execute the cached tree."""
    from benchflow.sandbox import _delegate_disarm as stage

    want = TREE_SHA256.get(stage.STAGED_PYTHON_SHA256[(arch, libc)])
    if want is None:
        return "false"
    path = stage.staged_python_dir(arch, libc)
    return (
        "( unset TAR_OPTIONS POSIXLY_CORRECT; LC_ALL=C; export LC_ALL; "
        + root_only_path_check(path, "staged Python")
        + "; [ $? = 0 ] || exit 1; "
        "bf_tar=$(tool tar) && bf_sum=$(tool sha256sum) || exit 1; "
        # Include diagnostics in the digest: tar can ignore sockets with
        # only a warning. The suffix is emitted only on successful traversal.
        f'bf_digest=$( ("$bf_tar" {TAR_OPTIONS} -C {shlex.quote(path)} '
        f"-cf - . 2>&1 && printf %s {shlex.quote(TREE_SUFFIX)}) | "
        '"$bf_sum") || exit 1; '
        f'[ "$bf_digest" = "{want}  -" ] )'
    )


def install_command() -> str:
    """Stage when supported; a failed attempt must discard the layered build."""
    from benchflow.sandbox import _delegate_disarm as stage
    from benchflow.sandbox.lockdown import _cloud_metadata_root_prelude

    cases = []
    for (arch, libc), pin in stage.STAGED_PYTHON_SHA256.items():
        if pin not in TREE_SHA256:
            continue
        urls = " ".join(
            shlex.quote(f"{base}/{stage.staged_python_file(arch, libc)}")
            for base in stage.STAGED_PYTHON_BASES
        )
        cases.append(
            f"{arch}:{libc}) d={shlex.quote(stage.staged_python_dir(arch, libc))}; "
            f"want={pin}; urls={shlex.quote(urls)} ;;"
        )
    # An image with an executable Python may supply PT's kit. Conservatively
    # skip it here; an unusable Python still takes the ordinary live fallback.
    return (
        '( [ "$(id -u)" = 0 ] || exit 1; '
        + _cloud_metadata_root_prelude()
        + "unset TAR_OPTIONS POSIXLY_CORRECT; "
        "for p in "
        + " ".join(shlex.quote(p) for p in IMAGE_PYTHONS)
        + " "
        + shlex.quote(stage.PYTHON_INSTALL_DIR)
        + '/cpython-*/bin/python3.12; do [ ! -x "$p" ] || exit 0; done; '
        "a=$(uname -m); case $a in amd64) a=x86_64;; arm64) a=aarch64;; esac; "
        'c=none; [ -e "/lib/ld-musl-$a.so.1" ] && c=musl; '
        'if [ "$c" = none ]; then for l in /lib64/ld-linux-x86-64.so.2 '
        "/lib/ld-linux-x86-64.so.2 /lib/ld-linux-aarch64.so.1 "
        '/lib64/ld-linux-aarch64.so.1; do [ -e "$l" ] && c=gnu; done; fi; '
        'case "$a:$c" in ' + " ".join(cases) + " *) exit 0;; esac; "
        "bf_tar=$(tool tar) && bf_sum=$(tool sha256sum) || exit 0; "
        'case $("$bf_tar" --version 2>/dev/null) in *"GNU tar"*) ;; *) exit 0;; esac; '
        f"archive={shlex.quote(BUNDLED_ARCHIVE)}; ok=; "
        'if [ -f "$archive" ]; then got=$("$bf_sum" "$archive") && '
        '[ "${got%% *}" = "$want" ] && ok=1; fi; '
        'if [ "$ok" != 1 ]; then '
        "bf_curl=$(tool curl) || bf_curl=; bf_wget=$(tool wget) || bf_wget=; "
        '[ -n "$bf_curl$bf_wget" ] || exit 0; fi; '
        f"mkdir -p {shlex.quote(stage.RUNTIME_ROOT)} && "
        f"chmod go-w {shlex.quote(stage.RUNTIME_ROOT)} && "
        f"mkdir -p {shlex.quote(stage.STAGED_ROOT)} && "
        f"chown 0:0 {shlex.quote(stage.STAGED_ROOT)} && "
        f"chmod 700 {shlex.quote(stage.STAGED_ROOT)} && "
        'rm -rf "$d" && mkdir -m 700 "$d" && chown 0:0 "$d" && '
        'walk "$d" && [ -d "$W" ] || exit 1; '
        'if [ "$ok" != 1 ]; then archive="$d/python.tar.gz"; '
        "for url in $urls; do "
        'if [ -n "$bf_curl" ]; then "$bf_curl" -fL --max-time 120 '
        '-o "$d/python.tar.gz" "$url"; '
        'elif [ -n "$bf_wget" ]; then "$bf_wget" -T 120 '
        '-O "$d/python.tar.gz" "$url"; else break; fi; '
        'got=$("$bf_sum" "$d/python.tar.gz") || continue; '
        '[ "${got%% *}" != "$want" ] || { ok=1; break; }; done; fi; '
        '[ "$ok" = 1 ] || exit 1; '
        'umask 022; "$bf_tar" --no-same-owner --no-same-permissions '
        '-xzf "$archive" -C "$d" && rm -f "$d/python.tar.gz" )'
    )
