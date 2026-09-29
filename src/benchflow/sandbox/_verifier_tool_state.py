"""Give the verifier's uv and pip state a fresh home the plugin guard trusts by path.

Runs in the sandbox as root after the agent has stopped (``python3 -c``). It
moves the verifier's uv cache, uv tool environments, uv-managed Pythons and
pip cache into one new root-owned directory that did not exist while the agent
ran, whether or not the agent could have written their usual places. The
pytest plugin guard trusts code in that directory by its path, so a plugin a
test.sh installs with ``uvx`` is trusted whatever modes the runtime's
file-mode mask gave it. Before, the guard judged the verifier's own install by
ownership and mode bits: under ``WORKDIR /root`` uv's cache was in the
workspace and refused, and on a runtime whose ``docker exec`` mask is 0000
(Docker-in-Docker) ``/root/.cache/uv`` came out world-writable and refused.

Because the guard trusts whatever lands in the directory, uv must not read
configuration the agent could have written: ``UV_CONFIG_FILE`` names a safe
user or system file, or an empty one, so the workspace's ``uv.toml`` or
``[tool.uv]`` cannot send the verifier's installs to the agent's own index.
pip's configuration is replaced only when its user files are unsafe, since
pip reads no workspace configuration and installs outside this directory.

Like the plugin guard, this must run on any Python 3 a task image ships, so it
uses no annotations and no f-strings.

argv[1]: JSON object with the variables BenchFlow sets for the verifier; they
override the image environment this process inherits.
argv[2]: JSON list of agent-writable path prefixes.
argv[3]: the directory to create.

Prints a JSON object of variables to set for the verifier.
"""

import json
import os
import stat
import sys

# Variables this module reads; BenchFlow passes the verifier's values for them.
INPUT_KEYS = (
    "HOME",
    "XDG_CACHE_HOME",
    "XDG_DATA_HOME",
    "XDG_CONFIG_HOME",
    "XDG_CONFIG_DIRS",
    "UV_CACHE_DIR",
    "UV_TOOL_DIR",
    "UV_PYTHON_INSTALL_DIR",
    "UV_CONFIG_FILE",
    "UV_NO_CONFIG",
    "PIP_CACHE_DIR",
    "PIP_CONFIG_FILE",
)
# Variables this module may set, and the entry each one gets in the new directory.
MOVED = (
    ("UV_CACHE_DIR", "uv-cache"),
    ("UV_TOOL_DIR", "uv-tools"),
    ("UV_PYTHON_INSTALL_DIR", "uv-python"),
    ("PIP_CACHE_DIR", "pip-cache"),
)
OUTPUT_KEYS = (*(key for key, _ in MOVED), "UV_CONFIG_FILE", "PIP_CONFIG_FILE")
# Empty configuration files written into the new directory.
UV_CONFIG = "uv.toml"
PIP_CONFIG = "pip.conf"
# Where uv looks for system configuration: $XDG_CONFIG_DIRS (default below), then this.
SYSTEM_CONFIG_DIRS = "/etc/xdg"
SYSTEM_UV_CONFIG = "/etc/uv/uv.toml"


def under(path, prefix):
    prefix = prefix.rstrip("/")
    return path == prefix or path.startswith(prefix + "/")


def owned_safely(st):
    """Root-owned and not group- or world-writable: only root can change it."""
    return st.st_uid == 0 and not st.st_mode & (stat.S_IWGRP | stat.S_IWOTH)


def safe(path, blocked, lexical=False):
    """Whether only root can have decided what the verifier finds at *path*.

    *path* need not exist. Written and resolved, it must lie outside every
    agent-writable prefix, and it and every existing ancestor must be owned
    safely. *lexical* checks the prefixes of the path as written only, for
    when this runs outside the sandbox. ``/dev/null`` (an empty
    configuration) is safe.
    """
    if path == os.devnull:
        return True
    if not path or not path.startswith("/"):
        return False
    candidates = [os.path.normpath(path)]
    if not lexical:
        candidates.append(os.path.realpath(path))
    for candidate in candidates:
        if any(under(candidate, prefix) for prefix in blocked):
            return False
        if lexical:
            continue
        while True:
            try:
                st = os.stat(candidate)
            except FileNotFoundError:
                st = None
            except OSError:
                return False
            if st is not None and not owned_safely(st):
                return False
            parent = os.path.dirname(candidate)
            if parent == candidate:
                break
            candidate = parent
    return True


def truthy(value):
    return (value or "").strip().lower() in ("1", "true", "yes", "on", "y", "t")


def _base(env, name, default):
    """An XDG base directory as uv and pip resolve it: absolute value or $HOME default."""
    value = env.get(name) or ""
    if value.startswith("/"):
        return value
    return os.path.join(env.get("HOME") or "/root", default)


def locations(env):
    """Where the verifier's uv and pip keep caches, tools and managed Pythons."""
    cache = _base(env, "XDG_CACHE_HOME", ".cache")
    data = _base(env, "XDG_DATA_HOME", ".local/share")
    defaults = {
        "UV_CACHE_DIR": os.path.join(cache, "uv"),
        "UV_TOOL_DIR": os.path.join(data, "uv", "tools"),
        "UV_PYTHON_INSTALL_DIR": os.path.join(data, "uv", "python"),
        "PIP_CACHE_DIR": os.path.join(cache, "pip"),
    }
    # A relative value resolves against the working directory, the workspace;
    # safe() refuses it for not being absolute.
    return dict((key, env.get(key) or default) for key, default in defaults.items())


def uv_system_configs(env):
    """uv's system configuration candidates, in the order uv looks for them."""
    dirs = env.get("XDG_CONFIG_DIRS") or SYSTEM_CONFIG_DIRS
    candidates = [os.path.join(d, "uv", "uv.toml") for d in dirs.split(":") if d]
    return [*candidates, SYSTEM_UV_CONFIG]


def overrides(env, blocked, directory, lexical=False):
    """Return the variables that move the verifier's uv and pip state into *directory*.

    Every uv and pip location moves. uv reads configuration from
    ``UV_CONFIG_FILE`` alone when it is set; else, unless ``UV_NO_CONFIG``, from
    the working directory's project (the workspace), the user file under
    ``$XDG_CONFIG_HOME`` or ``$HOME`` and the system files. Whatever uv installs
    lands in the trusted directory, so configuration from the workspace must not
    decide what that is: uv reads a safe ``UV_CONFIG_FILE`` the verifier already
    names, else the safe user file, else the first safe system file, else an
    empty one. pip reads the user files unless ``PIP_CONFIG_FILE`` names an
    existing file; an empty file in the new directory replaces them when they
    are unsafe and keeps the image's global configuration.
    """
    isfile = (lambda p: False) if lexical else os.path.isfile
    result = dict((key, os.path.join(directory, entry)) for key, entry in MOVED)

    config_home = _base(env, "XDG_CONFIG_HOME", ".config")
    explicit = env.get("UV_CONFIG_FILE")
    if explicit:
        keep = safe(explicit, blocked, lexical)
    else:
        keep = truthy(env.get("UV_NO_CONFIG"))
    if not keep:
        candidates = [os.path.join(config_home, "uv", "uv.toml")]
        candidates += uv_system_configs(env)
        chosen = [p for p in candidates if isfile(p) and safe(p, blocked, lexical)]
        result["UV_CONFIG_FILE"] = (
            chosen[0] if chosen else os.path.join(directory, UV_CONFIG)
        )

    pip_file = env.get("PIP_CONFIG_FILE") or ""
    pip_unsafe = bool(pip_file) and not safe(pip_file, blocked, lexical)
    if not pip_file or lexical or not os.path.exists(pip_file):
        # pip skips the user files only when PIP_CONFIG_FILE names a file that exists.
        home = env.get("HOME") or "/root"
        pip_unsafe = (
            pip_unsafe
            or not safe(os.path.join(home, ".pip", "pip.conf"), blocked, lexical)
            or not safe(os.path.join(config_home, "pip", "pip.conf"), blocked, lexical)
        )
    if pip_unsafe:
        result["PIP_CONFIG_FILE"] = os.path.join(directory, PIP_CONFIG)
    return result


def create(directory):
    """Create the root-owned directory and its empty configuration files."""
    os.mkdir(directory, 0o755)
    os.chmod(directory, 0o755)
    for name in (UV_CONFIG, PIP_CONFIG):
        path = os.path.join(directory, name)
        with open(path, "x"):
            pass
        os.chmod(path, 0o644)


if __name__ == "__main__":
    environment = dict(os.environ)
    environment.update(json.loads(sys.argv[1]))
    prefixes = json.loads(sys.argv[2])
    blocked = tuple(
        candidate
        for prefix in prefixes
        for candidate in (os.path.abspath(prefix), os.path.realpath(prefix))
    )
    found = overrides(environment, blocked, sys.argv[3])
    create(sys.argv[3])
    print(json.dumps(found))
