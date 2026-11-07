"""An authenticated, channel-only Python in the protected kit."""

from __future__ import annotations

import posixpath
import shlex
from collections import defaultdict

from benchflow._runtime_tree import KIT, KIT_CHANNEL_PYTHON, KIT_PYTHON, KIT_SH
from benchflow.sandbox._channel_python_files import DIRECTORIES_BY_TREE, FILES_BY_TREE

CHANNEL_TREE = f"{KIT}/channel"
CHANNEL_BINARY = f"{CHANNEL_TREE}/python/bin/python3.12"


def copy_command(arch: str, libc: str) -> str:
    """Copy the derived closure only for its exact, authenticated source tree."""
    from benchflow.sandbox import _delegate_disarm as stage
    from benchflow.sandbox._staged_python_layer import TREE_SHA256

    source = shlex.quote(stage.staged_python_dir(arch, libc))
    target = shlex.quote(CHANNEL_TREE)
    digest = TREE_SHA256.get(stage.STAGED_PYTHON_SHA256[arch, libc], "")
    files = FILES_BY_TREE.get(digest)
    if files is None:
        return f"cp -a {source} {target}"
    common = posixpath.commonpath(files)
    groups: dict[str, list[str]] = defaultdict(list)
    for file in files:
        groups[posixpath.relpath(file, common).split("/", 1)[0]].append(file)
    directories = set(DIRECTORIES_BY_TREE[digest])
    copies = []
    for paths in groups.values():
        parent = posixpath.commonpath([posixpath.dirname(path) for path in paths])
        directories.add(parent)
        names = shlex.join(posixpath.relpath(path, parent) for path in paths)
        copies.append(
            f"( cd {shlex.quote(parent)} && cp -a --parents {names} "
            f'"$bf_to"/{shlex.quote(parent)} )'
        )
    folders = " ".join(f'"$bf_to"/{shlex.quote(p)}' for p in sorted(directories))
    return (
        f"( cd {source} && bf_to={target} && mkdir -p {folders} && "
        + " && ".join(copies)
        + " )"
    )


def prepare_command() -> str:
    """Copy a pinned layer before the agent runs; never change the kit Python.

    A failed proof or copy leaves no channel wrapper. The ordinary kit seal
    includes the copy in its custody check and manifest before it is used.
    No interpreter from the layer executes during preparation.
    """
    from benchflow.sandbox import _delegate_disarm as stage
    from benchflow.sandbox._staged_python_layer import tree_check
    from benchflow.sandbox.lockdown import _cloud_metadata_root_prelude

    choices = []
    for arch, libc in stage.STAGED_PYTHON_SHA256:
        binary = shlex.quote(stage.staged_python_bin(arch, libc))
        choices.append(
            f"{'if' if not choices else 'elif'} [ -x {binary} ] "
            f"&& {{ {tree_check(arch, libc)}; }} "
            "2>/dev/null; then "
            f"{copy_command(arch, libc)}; "
        )
    # This is part of the fresh build, before the workspace grant. Existing
    # kits still take their original custody/manifest validation and exit.
    return (
        f"if [ ! -x {shlex.quote(KIT_PYTHON)} ]; then\n"
        "( "
        + _cloud_metadata_root_prelude()
        + "".join(choices)
        + "else exit 1; fi || exit 1; "
        f"printf %s {shlex.quote(wrapper())} > {shlex.quote(KIT_CHANNEL_PYTHON)} "
        f"&& chmod 755 {shlex.quote(KIT_CHANNEL_PYTHON)}; exit $?; ) || {{ "
        f"rm -rf {shlex.quote(CHANNEL_TREE)}; "
        f"rm -f {shlex.quote(KIT_CHANNEL_PYTHON)}; }}\nfi"
    )


def wrapper() -> str:
    """The only channel-only wrapper accepted by the Daytona launcher."""
    return f'#!{KIT_SH}\nexec {CHANNEL_BINARY} -I -S -B "$@"\n'
