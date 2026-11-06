"""Give the verifier's uv and pip state a fresh home the plugin guard trusts by path.

Runs in the sandbox as root after the agent has stopped (``python3 -I -c``). It
moves the verifier's uv cache, uv tool environments, unsafe uv-managed Pythons
and pip cache into one new root-owned directory that did not exist while the agent
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
Protected installed Pythons remain available for offline verification. pip's
file configuration is disabled for every interpreter: the verifier may choose
or create another environment after this probe. Trusted pip index settings
belong in verifier environment variables or the verifier script's own pip
options. The same ``/dev/null`` is pinned on every verifier path, hardened or
not (``benchflow._verifier_env``).

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
import shutil
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
)
# Variables this module may set, and the entry each one gets in the new directory.
MOVED = (
    ("UV_CACHE_DIR", "uv-cache"),
    ("UV_TOOL_DIR", "uv-tools"),
    ("UV_PYTHON_INSTALL_DIR", "uv-python"),
    ("PIP_CACHE_DIR", "pip-cache"),
)
# When the verifier's HOME is not root's alone (WORKDIR /root is the agent's
# workspace), its tools get a new home in the directory: an env script the
# verifier sources from ~/.local/bin, a ~/.curlrc and every other per-user
# file the agent could have left would run or apply as root while it grades
# (one planted file scored 1 on 33 SkillsBench tasks).
HOME_ENTRY = "home"
# Empty files that stop a tool from falling back to the passwd entry's home
# (curl reads /root/.curlrc when $HOME has none).
HOME_FILES = (".curlrc", ".wgetrc")
XDG_HOMES = (
    ("XDG_CONFIG_HOME", ".config"),
    ("XDG_CACHE_HOME", ".cache"),
    ("XDG_DATA_HOME", ".local/share"),
    ("XDG_STATE_HOME", ".local/state"),
)
OUTPUT_KEYS = (
    *(key for key, _ in MOVED),
    "UV_CONFIG_FILE",
    "PIP_CONFIG_FILE",
    "HOME",
    *(key for key, _ in XDG_HOMES),
)
# Empty configuration files written into the new directory.
UV_CONFIG = "uv.toml"
PIP_CONFIG = "pip.conf"
# Where uv looks for system configuration: $XDG_CONFIG_DIRS (default below), then this.
SYSTEM_CONFIG_DIRS = "/etc/xdg"
SYSTEM_UV_CONFIG = "/etc/uv/uv.toml"
SYSTEM_PIP_CONFIG = "/etc/pip.conf"


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


def home_is_safe(home, blocked, lexical=False):
    """Whether only root can have put what the verifier's tools read in
    *home*: safe() for the folder and, in the sandbox, for every entry in it
    (``~/.local``, ``~/.config``, ``~/.curlrc``, ...)."""
    if not safe(home, blocked, lexical):
        return False
    if lexical:
        return True
    try:
        names = os.listdir(home)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    for name in names:
        try:
            st = os.lstat(os.path.join(home, name))
        except OSError:
            return False
        if not owned_safely(st):
            return False
    return True


def home_overrides(env, blocked, directory, lexical=False):
    """HOME and the XDG bases in the new directory, or nothing when the
    verifier's HOME is root's alone (an image's own files there still count)."""
    home = env.get("HOME") or "/root"
    if home_is_safe(home, blocked, lexical):
        return {}
    fresh = os.path.join(directory, HOME_ENTRY)
    result = {"HOME": fresh}
    for key, entry in XDG_HOMES:
        value = env.get(key) or ""
        if not value.startswith("/") or not safe(value, blocked, lexical):
            result[key] = os.path.join(fresh, entry)
    return result


def truthy(value):
    return (value or "").strip().lower() in ("1", "true", "yes", "on", "y", "t")


def _base(env, name, default):
    """An XDG base directory as uv resolves it: absolute value or $HOME default."""
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


def protected_python_installation(path, blocked):
    """Keep an installed runtime only when the agent could alter none of its tree.

    Caches and tool environments move regardless; installed Python is neither
    a download cache nor new verifier code and may be required offline.
    """
    if not os.path.isdir(path) or not safe(path, blocked):
        return False
    try:

        def failed(error):
            raise error

        seen = set()
        for root, dirs, files in os.walk(path, onerror=failed, followlinks=True):
            identity = os.stat(root)
            identity = (identity.st_dev, identity.st_ino)
            if identity in seen:
                dirs[:] = []
                continue
            seen.add(identity)
            for name in dirs + files:
                if not safe(os.path.join(root, name), blocked):
                    return False
    except OSError:
        return False
    return True


def overrides(env, blocked, directory, lexical=False):
    """Return the variables that move the verifier's uv and pip state into *directory*.

    Caches, tools and unsafe managed installations move. uv reads configuration from
    ``UV_CONFIG_FILE`` alone when it is set; else, unless ``UV_NO_CONFIG``, from
    the working directory's project (the workspace), the user file under
    ``$XDG_CONFIG_HOME`` or ``$HOME`` and the system files. Whatever uv installs
    lands in the trusted directory, so configuration from the workspace must not
    decide what that is: uv reads a safe ``UV_CONFIG_FILE`` the verifier already
    names, else the safe user file, else the first safe system file, else an
    empty one. pip receives the fixed /dev/null file policy: its interpreter-site
    configuration cannot be enumerated before a verifier chooses its Python,
    and its relative/empty XDG paths can resolve against any verifier cwd.
    """
    isfile = (lambda p: False) if lexical else os.path.isfile
    result = dict((key, os.path.join(directory, entry)) for key, entry in MOVED)
    python_installation = locations(env)["UV_PYTHON_INSTALL_DIR"]
    if not lexical and protected_python_installation(python_installation, blocked):
        # Omit the override so both an explicit path and uv's default still
        # discover the trusted image's installed runtimes.
        del result["UV_PYTHON_INSTALL_DIR"]

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

    # An empty file skips only user config. /dev/null suppresses every file
    # across pip interpreters and cwd/XDG spellings, without trusting a finite
    # inventory of Pythons that test.sh can expand after hardening.
    result["PIP_CONFIG_FILE"] = os.devnull
    result.update(home_overrides(env, blocked, directory, lexical))
    return result


# Not a variable: which pip configuration files the image has, for the warning
# a task gets when its verifier's pip reads none (lockdown pops it).
PIP_FILES_KEY = "pip_config_files"


def image_pip_config_files(environ):
    """The image's pip configuration files that exist, in pip's order."""
    dirs = environ.get("XDG_CONFIG_DIRS") or "/etc/xdg"
    found = [os.path.join(d, "pip", "pip.conf") for d in dirs.split(":") if d]
    found += ["/etc/pip.conf", os.path.join(sys.prefix, "pip.conf")]
    named = environ.get("PIP_CONFIG_FILE") or ""
    if named and named != os.devnull:
        found.append(named)
    seen = []
    for path in found:
        if path not in seen and os.path.isfile(path):
            seen.append(path)
    return seen


def create(directory, home=None):
    """Create the root-owned directory and its empty configuration files, and
    *home* (the verifier's new HOME) with its empty rc files."""
    os.mkdir(directory, 0o755)
    os.chmod(directory, 0o755)
    for name in (UV_CONFIG, PIP_CONFIG):
        path = os.path.join(directory, name)
        with open(path, "x"):
            pass
        os.chmod(path, 0o644)
    if home:
        os.mkdir(home, 0o700)
        os.chmod(home, 0o700)
        for name in HOME_FILES:
            path = os.path.join(home, name)
            with open(path, "x"):
                pass
            os.chmod(path, 0o600)


def seed_managed_python(env, blocked, destination):
    """Copy only protected installed trees, never shared locks or writable state.

    uv creates a world-writable .lock in an otherwise protected installation
    directory. Reusing that lock can let a kept process stall verification;
    dropping every installed interpreter would instead break offline tasks.
    The fresh directory gets its own lock and only immutable image runtimes.
    """
    source = locations(env)["UV_PYTHON_INSTALL_DIR"]
    if not safe(source, blocked) or not os.path.isdir(source):
        return
    for name in os.listdir(source):
        candidate = os.path.join(source, name)
        if protected_python_installation(candidate, blocked):
            if not os.path.exists(destination):
                os.mkdir(destination, 0o755)
            shutil.copytree(candidate, os.path.join(destination, name), symlinks=True)


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
    create(sys.argv[3], found.get("HOME"))
    found[PIP_FILES_KEY] = image_pip_config_files(os.environ)
    if "UV_PYTHON_INSTALL_DIR" in found:
        seed_managed_python(environment, blocked, found["UV_PYTHON_INSTALL_DIR"])
    print(json.dumps(found))
