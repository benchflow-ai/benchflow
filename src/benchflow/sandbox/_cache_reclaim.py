"""Pre-verifier cache reclaim command construction.

The sandbox cannot import benchflow internals, so the reclaim implementation is
an embedded Python snippet passed to ``python3 -c``. Keep it here instead of in
``lockdown.py`` so the hardening orchestrator stays readable.
"""

import shlex

# Symlink- and workspace-safe cache reclaim for #601.
#
# The previous shell form (`rm -rf "$u/.cache/uv" ...; rm -rf /tmp/uv-*`) could
# delete agent-visible state before scoring: `rm -rf` traverses intermediate
# symlinks (an agent-planted `~/.cache -> /app` turned the cache delete into
# `rm -rf /app/uv`), and the `/tmp/uv-*` glob matched a legitimate
# `/tmp/uv-workspace`. The textual `$WS`/`$u` guard saw none of that. Since
# ``restore_workspace`` defaults to False, nothing undid the damage.
#
# This snippet hardens every deletion candidate:
#  - rejects any candidate with a symlink path component, including the final
#    one, so agent-controlled parents can no longer redirect the delete;
#  - resolves protected roots (workspace + /logs, covering /logs/artifacts and
#    /logs/verifier) via realpath and skips candidates overlapping in either
#    direction, applied uniformly to per-user caches, /tmp globs, and apt;
#  - deletes with shutil.rmtree, which removes child symlinks without following
#    them.
#
# argv[1] = active workspace (or /nonexistent). argv[2] = optional filesystem
# prefix, "" in production; tests pass a temp root so the exact production code
# runs hermetically against a fake filesystem.
#
# Fail-safe by construction: if python3 is unavailable the reclaim simply does
# not run (the trailing `true` swallows it). Losing best-effort ENOSPC mitigation
# is strictly better than deleting the wrong state.
RECLAIM_CACHES_PY = (
    "import glob, os, shutil, sys;"
    "ws = sys.argv[1];"
    "px = sys.argv[2] if len(sys.argv) > 2 else '';"
    "\n"
    "def linked(path):\n"
    "    p = px or ''\n"
    "    for part in path[len(px):].strip('/').split('/'):\n"
    "        p = p + '/' + part\n"
    "        if os.path.islink(p):\n"
    "            return True\n"
    "    return False\n"
    "\n"
    "protected = []\n"
    "for root in (ws, px + '/logs'):\n"
    "    try:\n"
    "        if root and os.path.exists(root):\n"
    "            protected.append(os.path.realpath(root))\n"
    "    except OSError:\n"
    "        pass\n"
    "\n"
    "def overlaps(path):\n"
    "    return any(\n"
    "        path == r or path.startswith(r + '/') or r.startswith(path + '/')\n"
    "        for r in protected\n"
    "    )\n"
    "\n"
    "cands = [\n"
    "    u + '/.cache/' + name\n"
    "    for u in [px + '/root'] + sorted(glob.glob(px + '/home/*'))\n"
    "    for name in ('uv', 'pip', 'uv_build')\n"
    "]\n"
    "cands += sorted(glob.glob(px + '/tmp/uv-*'))\n"
    "cands += sorted(glob.glob(px + '/tmp/.uv-*'))\n"
    "cands += sorted(glob.glob(px + '/var/cache/apt/archives/*.deb'))\n"
    "for c in cands:\n"
    "    try:\n"
    "        if not os.path.lexists(c) or linked(c):\n"
    "            continue\n"
    "        # linked() proved no symlink components below the prefix, so\n"
    "        # realpath only normalizes the prefix itself - consistent with\n"
    "        # the realpath'd protected roots.\n"
    "        if overlaps(os.path.realpath(c)):\n"
    "            continue\n"
    "        if os.path.isdir(c):\n"
    "            shutil.rmtree(c, ignore_errors=True)\n"
    "        else:\n"
    "            os.remove(c)\n"
    "    except OSError:\n"
    "        pass\n"
)


def build_reclaim_caches_cmd(workspace: str | None) -> str:
    """Build the best-effort cache reclaim shell command."""
    workspace_arg = shlex.quote(workspace) if workspace else "/nonexistent"
    return (
        f"python3 -c {shlex.quote(RECLAIM_CACHES_PY)} {workspace_arg} 2>/dev/null; true"
    )
