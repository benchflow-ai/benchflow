"""C8's checks that a runtime path is root's alone, as shell text.

A leaf module (no BenchFlow imports) so the agent install recipes in
``benchflow.agents.registry``, the protected runtime tree
(:mod:`benchflow._runtime_tree`) and the sandbox backends share one check.

Every check walks the path as written, one name at a time, and follows each
link hop by hop (PT-C8-R3-9 at e8a016a9): a folder on the way, or the folder
a link hop sits in, that someone other than root may write could hand root
another file, whatever the path resolves to today. Only after the walk does
a check canonicalize anything.
"""

from __future__ import annotations

import shlex

__all__ = [
    "ROOT_CHAIN_FUNCS",
    "root_only_path_check",
    "root_only_tree_check",
]

# POSIX sh functions (dash, bash, busybox ash), defined at the start of every
# command that uses them. ``bfc_chain`` runs in a subshell per call, so its
# recursion over link hops keeps no state between calls.
#
# bfc_dir <folder> [s]: root's, and nobody else may write it (with ``s``, a
#   sticky folder others may write, such as /tmp, also passes).
# bfc_file <file>: root's, not a folder, nobody else may write it.
# bfc_link <link>: the link itself is root's. Its folder passed bfc_dir, so
#   nobody else may replace it, a sticky folder's included (PY8-PT-R8-1).
# bfc_chain <absolute path> <hops so far, one character each> (no
#   arithmetic expansion: install recipes stay plain POSIX sh): 0 when every folder on the way to
#   the path is root's (sticky allowed above the last), every link on the
#   way is root's and followed hop by hop under the same rule, and the path itself is a
#   root-only folder or file; 2 when it, or a folder on the way to it, does
#   not exist and only root could create it (the last folder that exists is
#   root's alone, not sticky; PT-C8-R9-2: a HOME with no .ssh); 1 otherwise.
ROOT_CHAIN_FUNCS = (
    'bfc_dir() { if [ "${2-}" = s ]; then '
    '[ -n "$(find "$1" -maxdepth 0 -type d -user 0 '
    '\\( -perm -1000 -o \\( ! -perm -0020 ! -perm -0002 \\) \\) -print 2>/dev/null)" ]; '
    'else [ -n "$(find "$1" -maxdepth 0 -type d -user 0 ! -perm -0020 ! -perm -0002 '
    '-print 2>/dev/null)" ]; fi; }; '
    'bfc_file() { [ -n "$(find "$1" -maxdepth 0 ! -type d ! -type l -user 0 '
    '! -perm -0020 ! -perm -0002 -print 2>/dev/null)" ]; }; '
    'bfc_link() { [ -n "$(find "$1" -maxdepth 0 -type l -user 0 -print 2>/dev/null)" ]; }; '
    "bfc_chain() ( p=$1; n=${2-}; [ ${#n} -le 40 ] || exit 1; "
    "case $p in /*) ;; *) exit 1;; esac; "
    "bfc_dir / s || exit 1; cur=; rest=${p#/}; "
    'while [ -n "$rest" ]; do '
    "case $rest in */*) c=${rest%%/*}; rest=${rest#*/};; *) c=$rest; rest=;; esac; "
    "case $c in ''|.) continue;; ..) cur=${cur%/*}; continue;; esac; "
    "nx=$cur/$c; "
    # The sentinel keeps command substitution from eating target bytes.
    # Remove it and exactly the one newline readlink adds.
    'if [ -L "$nx" ]; then bfc_link "$nx" || exit 1; '
    't=$(readlink "$nx" && printf .) || exit 1; t=${t%.}; t=${t%?}; '
    "case $t in /*) ;; *) t=$cur/$t;; esac; "
    'if [ -z "$rest" ]; then bfc_chain "$t" "${n}x"; exit $?; fi; '
    # A root-owned link on the way, to a missing target only root could
    # create: what follows is only root's to create too (status 2;
    # PT-C8-R10-2 sol: a HOME whose .ssh is such a link).
    'bfc_chain "$t" "${n}x"; s=$?; [ "$s" = 0 ] || exit "$s"; '
    'cur=$(cd -P -- "$t" 2>/dev/null && pwd -P && printf .) || exit 1; '
    'cur=${cur%.}; cur=${cur%?}; [ "$cur" != / ] || cur=; '
    'elif [ -n "$rest" ]; then '
    'if [ ! -e "$nx" ]; then bfc_dir "${cur:-/}" && exit 2; exit 1; fi; '
    '[ -d "$nx" ] && bfc_dir "$nx" s || exit 1; cur=$nx; '
    'elif [ -d "$nx" ]; then bfc_dir "$nx"; exit $?; '
    'elif [ -e "$nx" ]; then bfc_file "$nx"; exit $?; '
    'else bfc_dir "${cur:-/}" && exit 2; exit 1; fi; '
    "done; exit 0 ); "
)


def root_only_path_check(path: str, what: str) -> str:
    """sh: status 0 only when *path* is root's alone along its whole route
    (C8): every folder on the way root's (sticky allowed above the last),
    every link hop under the same rule, the path itself a root-only file or
    folder. Its contents are not walked (see :func:`root_only_tree_check`).
    Use it before root executes a single program from a runtime tree.
    """
    q = shlex.quote(path)
    fail = shlex.quote(f"benchflow: {what} ({path}) is not root's alone; refusing it")
    return f"( {ROOT_CHAIN_FUNCS}bfc_chain {q} 0 ) || {{ echo {fail} >&2; false; }}"


def root_only_tree_check(path: str, what: str) -> str:
    """sh: status 0 only when *path* is root's alone (C8), else a message on
    stderr and status 1. Runs on root's plumbing route, whose PATH names
    root-only folders, so ``find`` and ``readlink`` are protected ones.

    Root's alone: the route to *path* passes :func:`root_only_path_check`;
    everything in it is owned by root and nobody else may write it; a link
    in it passes the same hop-by-hop walk, or names a missing file only root
    could create (uv's shared ``.lock`` files excepted, below). BenchFlow's
    runtimes under /opt/benchflow and /tmp are reused across roles in one
    sandbox, and root runs what they hold.
    """
    q = shlex.quote(path)
    fail = shlex.quote(f"benchflow: {what} ({path}) is not root's alone; refusing it")
    # uv makes its lock files (``.lock`` in a venv or a managed Python's
    # folder) writable by everyone so users can share them: a root-owned
    # ``.lock`` with no execute bit holds no code, and replacing it needs its
    # folder, which must still be root's alone.
    uv_lock = (
        "\\( -type f -name .lock -user 0 ! -perm -0100 ! -perm -0010 ! -perm -0001 \\)"
    )
    tree_bad = (
        f'find "$bf_r" \\( -type l ! -user 0 \\) -o \\( ! -type l ! {uv_lock} '
        "\\( ! -user 0 -o -perm -0020 -o -perm -0002 \\) \\) -print 2>&1 | head -n 1"
    )
    # Each link is walked as written, from the folder it sits in: a hop
    # through a folder the agent can write fails even when the link resolves
    # to a root-only file today. A missing target passes only when nobody but
    # root could create it.
    # A link whose first hop (made absolute from its folder, which find
    # reached without following a link) and whose final file both stay in
    # the tree needs no walk: every other link in the tree is walked here,
    # so any hop that leaves the tree is checked where it leaves.
    link_ok = (
        'bf_norm() { r=; s=${1#/}; while [ -n "$s" ]; do '
        "case $s in */*) c=${s%%/*}; s=${s#*/};; *) c=$s; s=;; esac; "
        "case $c in ''|.) ;; ..) r=${r%/*};; *) r=$r/$c;; esac; done; "
        'echo "${r:-/}"; }; '
        'find "$bf_r" -type l -print | while IFS= read -r bf_l; do '
        'bf_t="$(readlink -- "$bf_l")" || exit 1; '
        'case "$bf_t" in /*) ;; *) bf_t="${bf_l%/*}/$bf_t";; esac; '
        'case "$(bf_norm "$bf_t")" in "$bf_r"/*) '
        'case "$(readlink -f -- "$bf_l" 2>/dev/null)" in "$bf_r"|"$bf_r"/*) '
        "continue;; esac;; esac; "
        'bfc_chain "$bf_t" 1; bf_s=$?; [ "$bf_s" = 0 ] || [ "$bf_s" = 2 ] || '
        '{ echo "benchflow: a link leaves it: $bf_l -> $bf_t" >&2; exit 1; }; done'
    )
    route = shlex.quote(
        f"benchflow: not root-only: {path} or a folder or link on the way to it"
    )
    return (
        f"( {ROOT_CHAIN_FUNCS}"
        f"bfc_chain {q} 0 || {{ echo {route} >&2; exit 1; }}; "
        f'bf_r="$(cd -P -- {q} 2>/dev/null && pwd -P)" || exit 1; '
        f'bf_b="$({tree_bad})"; [ -z "$bf_b" ] || '
        '{ echo "benchflow: not root-only: $bf_b" >&2; exit 1; }; '
        f"{link_ok} || exit 1 ) || {{ echo {fail} >&2; false; }}"
    )
