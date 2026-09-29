"""Pytest plugin provenance checks, also installed as a protected verifier guard.

Selective PR #1117 adaptation (tulerfeng).
Checks entry-point registrations and module paths without importing candidates.
The public pytest hooks run inside the final verifier/uvx Python environment.

That environment can be any Python 3 and any pytest a task image ships, and a
guard that fails to import or register aborts pytest, which test.sh records as
reward 0. So this module carries no annotations (``tuple[str, ...]`` raises at
import before Python 3.9) and hooks take only arguments pytest 7 passes.

Because test.sh may discard pytest's output, the armed guard also reports to
the verifier through marker files named ``<prefix>.<token>.<kind>``: a
``loading`` marker written on import and removed once pytest has registered
the guard or the guard itself stopped pytest. One left behind means pytest
imported the guard but never ran it.

The guard stops pytest in two ways. Refusing a plugin raises ``Rejected``: the
run is scored as test.sh reports it, since that is what agent tampering should
earn. Any other exception from a guard hook is the guard failing (a bug, a
pytest API it did not expect); the hook leaves a ``crashed`` marker holding the
traceback, still stops pytest, and the verifier reports an error instead of a
score. Inspecting plugin files and metadata is the exception: an agent can
plant those, and a malformed ``entry_points.txt`` line alone makes CPython
3.12's importlib.metadata raise TypeError, so any error while inspecting them
is a ``Rejected`` and no planted file can make a failing run unscored. A guard
bug inside that inspection is therefore still scored.

A refusal the verifier caused itself is not tampering either: test.sh installed
the plugin after hardening, into a place the guard cannot trust (a venv in the
workspace, say). Hardening wrote this module after the agent was stopped and
the workspace frozen, and only the kernel sets a file's change time, so a
plugin file changed after this module was written came from the verifier. When
every refused plugin has at least one untrusted file behind it and none of
those is older than this module, the guard leaves an ``installed`` marker
listing them, each with why it is untrusted, and still stops pytest; the
verifier reports an error instead of a score. A refusal no untrusted file
explains (a name two trusted distributions register, a plugin installed
nowhere) stays scored. Every refusal message names each refused plugin and
what about it the guard could not trust.
"""

import json
import os
import shlex
import stat
import sys
import traceback
from contextlib import suppress
from importlib.machinery import PathFinder

# Reassigned by lines appended to this source for the protected runtime plugin.
_BENCHFLOW_BLOCKED = ()  # type: tuple[str, ...]
_BENCHFLOW_REQUESTED = []  # type: list[str]
# Directories hardening created after the agent stopped, for the verifier's uv
# and pip state. Code below them is trusted by path, whatever modes the
# runtime's file-mode mask gave it; the directories and their parents are not.
_BENCHFLOW_TRUSTED = ()  # type: tuple[str, ...]
# In the agent's own sandbox code must also be root's alone (root-owned, not
# group- or world-writable, parents included). A separate verifier sandbox
# never ran the agent: only the blocked paths, which the transfer wrote, are
# the agent's, and everything else is the verifier image's own.
_BENCHFLOW_OWNERSHIP = True
# Marker path prefix for this verification; empty writes no markers.
_BENCHFLOW_MARKERS = ""
# Keeps this process's markers apart from other pytest runs of one verifier.
_TOKEN = str(os.getpid()) + "-" + os.urandom(4).hex()


def _marker(kind):
    return ".".join((_BENCHFLOW_MARKERS, _TOKEN, kind))


def _mark(kind, detail=""):
    """Leave a ``kind`` marker for the verifier; never fail pytest doing so."""
    if not _BENCHFLOW_MARKERS:
        return
    try:
        with open(_marker(kind), "x") as handle:
            handle.write(detail)
    except Exception:
        pass


def _unmark(kind):
    if _BENCHFLOW_MARKERS:
        with suppress(OSError):
            os.remove(_marker(kind))


class Rejected(RuntimeError):
    """The guard refuses a plugin; pytest stops and the run stays scored."""


def _stopping(exc):
    """Record why this guard, not a failure to load it, is stopping pytest."""
    _unmark("loading")
    if not isinstance(exc, Rejected):
        _mark(
            "crashed",
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )


def under(path, prefix):
    prefix = prefix.rstrip("/")
    return path == prefix or path.startswith(prefix + "/")


def _trusted_root(path):
    """The directory in ``_BENCHFLOW_TRUSTED`` that holds *path*, if any."""
    for root in _BENCHFLOW_TRUSTED:
        for form in (os.path.abspath(root), os.path.realpath(root)):
            if under(path, form):
                return form
    return None


def _blocked_why():
    if _BENCHFLOW_OWNERSHIP:
        return "which the agent could write"
    return "which holds files copied from the agent's sandbox"


def ownership_problem(path, st):
    """Why the inode at *path* (stat *st*) is not root's alone, or None."""
    if st.st_uid != 0:
        return path + " is owned by uid " + str(st.st_uid) + ", not root"
    if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        mode = format(stat.S_IMODE(st.st_mode), "04o")
        return path + " is group- or world-writable (mode " + mode + ")"
    return None


def untrusted_reason(path, blocked):
    """Why pytest must not load code from *path*, or None when it may."""
    if not path or not path.startswith("/"):
        return repr(path) + " is not an absolute path"
    for candidate in (os.path.abspath(path), os.path.realpath(path)):
        for prefix in blocked:
            if under(candidate, prefix):
                return candidate + " is under " + prefix + ", " + _blocked_why()
        try:
            os.stat(candidate)
        except OSError as exc:
            return candidate + " cannot be read (" + type(exc).__name__ + ")"
        if not _BENCHFLOW_OWNERSHIP:
            continue
        # Below a trusted directory only the directory and its parents are
        # checked: hardening made it root-owned 0755 after the agent stopped,
        # so only the verifier wrote what is inside, in whatever modes.
        candidate = _trusted_root(candidate) or candidate
        # A protected file is replaceable when any containing directory is
        # writable. Check both lexical and resolved paths, including symlink
        # parents, rather than trusting the final inode alone.
        while True:
            try:
                st = os.stat(candidate)
            except OSError as exc:
                return candidate + " cannot be read (" + type(exc).__name__ + ")"
            problem = ownership_problem(candidate, st)
            if problem:
                return problem
            parent = os.path.dirname(candidate)
            if parent == candidate:
                break
            candidate = parent
    return None


def trusted(path, blocked):
    return untrusted_reason(path, blocked) is None


def trusted_module(module, blocked):
    parts = module.split(".")
    if not parts or not all(part.isidentifier() for part in parts):
        return False
    search = None
    for i in range(len(parts)):
        try:
            spec = PathFinder.find_spec(".".join(parts[: i + 1]), path=search)
        except Exception:
            return False
        if spec is None:
            return False
        # A namespace package has no origin, only search locations; a
        # builtin has a non-path origin. Requiring at least one real path
        # keeps both from passing by vacuous truth.
        locations = [spec.origin] if (spec.origin or "").startswith("/") else []
        locations += [str(p) for p in (spec.submodule_search_locations or [])]
        if not locations or not all(trusted(p, blocked) for p in locations):
            return False
        search = [str(p) for p in (spec.submodule_search_locations or [])]
        if i < len(parts) - 1 and not search:
            return False
    return True


def registrations():
    """Return ``(entry point, dist-info path or None)`` for every pytest11 entry."""
    try:
        from importlib.metadata import entry_points
    except ImportError:  # Python 3.7: pytest itself requires the backport there
        from importlib_metadata import entry_points

    try:
        eps = list(entry_points(group="pytest11"))
    except TypeError:
        eps = None
    # CPython's filesystem-backed distributions expose the metadata path.
    # Unknown/custom distribution providers cannot establish provenance.
    if eps is not None and all(hasattr(ep, "dist") for ep in eps):
        return [(ep, getattr(ep.dist, "_path", None)) for ep in eps]
    # Before Python 3.10 an entry point does not name its distribution, so take
    # each one from the distribution that declares it, as pluggy loads them.
    try:
        from importlib.metadata import distributions
    except ImportError:
        from importlib_metadata import distributions
    return [
        (ep, getattr(dist, "_path", None))
        for dist in distributions()
        for ep in dist.entry_points
        if ep.group == "pytest11"
    ]


def discover(blocked, requested, builtin_names=(), ignore_blocked_registrations=False):
    found = registrations()
    if ignore_blocked_registrations:
        # Pre-verifier discovery only proposes ``-p`` names. A registration in
        # an agent-writable tree can never be one, and must not veto or drop a
        # protected plugin of the same name either; the runtime guard still
        # counts it wherever the verifier's pytest can actually see it.
        found = [
            (ep, registration)
            for ep, registration in found
            if registration is None
            or not any(
                under(candidate, prefix)
                for candidate in (
                    os.path.abspath(registration),
                    os.path.realpath(registration),
                )
                for prefix in blocked
            )
        ]
    eps = [ep for ep, _ in found]
    counts = {}
    for ep in eps:
        counts[ep.name] = counts.get(ep.name, 0) + 1
    names = []
    for ep, registration in found:
        # pytest loads entry points by name. A second registration with that
        # name can select different code from the one just inspected.
        if counts[ep.name] != 1:
            continue
        if registration is None or not trusted(os.path.abspath(registration), blocked):
            continue
        registration_file = os.path.join(str(registration), "entry_points.txt")
        if not trusted(os.path.abspath(registration_file), blocked):
            continue
        if trusted_module(ep.value.split(":")[0].strip(), blocked):
            names.append(ep.name)
    for name in requested:
        module = "_pytest." + name if name in builtin_names else name
        if name not in counts and trusted_module(module, blocked):
            names.append(name)
    trusted_names = sorted(set(names))
    return {
        "plugins": trusted_names,
        "rejected": sorted(
            (set(ep.name for ep in eps) | set(requested)) - set(trusted_names)
        ),
    }


def _module_paths(module):
    """The files and package directories *module* resolves to, without importing it."""
    found = []
    search = None
    parts = module.split(".")
    for i in range(len(parts)):
        spec = PathFinder.find_spec(".".join(parts[: i + 1]), path=search)
        if spec is None:
            break
        if (spec.origin or "").startswith("/"):
            found.append(spec.origin)
        search = [str(p) for p in (spec.submodule_search_locations or [])]
        found.extend(search)
        if not search:
            break
    return found


def _plugin_paths(name, entry_points, builtin_names):
    """The registrations and code that decide what pytest loads for *name*."""
    if not entry_points:
        return _module_paths(name)
    paths = []
    modules = []
    for ep, registration in registrations():
        if ep.name != name:
            continue
        if registration is None:
            return []
        paths.append(str(registration))
        paths.append(os.path.join(str(registration), "entry_points.txt"))
        modules.append(ep.value.split(":")[0].strip())
    if not modules:
        modules.append("_pytest." + name if name in builtin_names else name)
    for module in modules:
        paths.extend(_module_paths(module))
    return paths


def _armed_ns():
    """When hardening wrote this guard: after the agent stopped and the workspace froze."""
    return os.stat(__file__).st_ctime_ns


def _installed_during_verification(names, entry_points, blocked, builtin_names):
    """Return the untrusted files showing the verifier itself installed the refused *names*.

    Every refused name needs an untrusted file behind it that changed after
    the guard was written, and no untrusted file behind any of them may be
    older: an older one existed while the agent ran. A refusal that no
    untrusted file explains (a name two trusted distributions register, a
    plugin installed nowhere) is not the verifier installing where the guard
    cannot trust. Returns an empty list, a scored refusal, when that does not
    hold or anything cannot be read.
    """
    try:
        armed = _armed_ns()
        evidence = []
        for name in names:
            untrusted = [
                path
                for path in _plugin_paths(name, entry_points, builtin_names)
                if not trusted(path, blocked)
            ]
            if not untrusted or any(
                os.stat(path).st_ctime_ns < armed for path in untrusted
            ):
                return []
            evidence.extend(untrusted)
        return evidence
    except Exception:
        return []


def _why_refused(name, entry_points, blocked, builtin_names):
    """Say what made the guard refuse *name*; never raise."""
    try:
        if entry_points:
            found = [reg for ep, reg in registrations() if ep.name == name]
            if len(found) > 1:
                return (
                    "registered " + str(len(found)) + " times, so the name does "
                    "not say which code pytest loads"
                )
            if found and found[0] is None:
                return "registered by a distribution whose files cannot be checked"
        paths = _plugin_paths(name, entry_points, builtin_names)
        if not paths:
            return "not installed where this pytest looks"
        for path in paths:
            if not trusted(path, blocked):
                return untrusted_reason(path, blocked) or path + " is not trusted"
        return "not trusted"
    except Exception as exc:
        return "cannot inspect: " + type(exc).__name__


def _validate(names, *, entry_points=True):
    from _pytest.config import builtin_plugins

    names = set(names) - {__name__}
    if not names:
        return
    # pytest maps its builtin short names to _pytest.<name> before importing.
    modules = ["_pytest." + name if name in builtin_plugins else name for name in names]
    blocked = tuple(
        candidate
        for path in _BENCHFLOW_BLOCKED
        for candidate in (os.path.abspath(path), os.path.realpath(path))
    )
    try:
        if entry_points:
            # CLI -p checks entry-point aliases BEFORE falling back to builtin
            # module names, including optional builtins such as pytester.
            receipt = discover(blocked, names, builtin_plugins)
            refused = names - set(receipt["plugins"])
        else:
            # pytest_plugins / PYTEST_PLUGINS import module names directly; a
            # same-named trusted entry point does not authorize that module.
            refused = {name for name in modules if not trusted_module(name, blocked)}
    except Exception as exc:
        # Files and metadata on sys.path may be planted; what cannot be
        # inspected cannot be trusted.
        raise Rejected(
            "Verifier plugin trust rejected: "
            + ", ".join(sorted(names))
            + " (cannot inspect: "
            + type(exc).__name__
            + ": "
            + str(exc)
            + ")"
        ) from exc
    if refused:
        refused = sorted(refused)
        reasons = [
            name + ": " + _why_refused(name, entry_points, blocked, builtin_plugins)
            for name in refused
        ]
        message = (
            "Verifier plugin trust rejected: "
            + ", ".join(refused)
            + " ("
            + "; ".join(reasons)
            + ")"
        )
        installed = _installed_during_verification(
            refused, entry_points, blocked, builtin_plugins
        )
        if installed:
            _mark(
                "installed",
                "".join(
                    path
                    + "\t"
                    + (untrusted_reason(path, blocked) or "untrusted")
                    + "\n"
                    for path in installed
                ),
            )
            message += (
                "; the verifier installed it after the agent stopped, "
                "so the run is not scored"
            )
        raise Rejected(message)


def pytest_addhooks(pluginmanager):
    """Validate actual invocation -p imports before pytest visits later aliases."""
    try:
        config = pluginmanager.get_plugin("pytestconfig")
        # pytest < 5.1 (4.6.x aside) has no invocation_params; its parse() keeps
        # the command line in _origargs before importing any -p plugin.
        params = getattr(config, "invocation_params", None)
        args = [
            *shlex.split(os.environ.get("PYTEST_ADDOPTS", "")),
            *(params.args if params is not None else config._origargs),
        ]
        names = list(_BENCHFLOW_REQUESTED)
        index = 0
        while index < len(args):
            arg = args[index]
            index += 1
            if arg == "-p":
                if index == len(args):
                    raise Rejected("Verifier plugin trust: missing -p argument")
                name = args[index]
                index += 1
            elif arg.startswith("-p"):
                name = arg[2:]
            else:
                continue
            name = name.strip()
            if not name.startswith("no:"):
                names.append(name)
        _validate(names)
        _validate(
            [
                name.strip()
                for name in os.environ.get("PYTEST_PLUGINS", "").split(",")
                if name.strip()
            ],
            entry_points=False,
        )
    except Exception as exc:
        _stopping(exc)
        raise


def pytest_plugin_registered(plugin, manager):
    """Check declared dependencies before pytest's consider_module imports them."""
    try:
        if plugin is sys.modules.get(__name__):
            # pytest calls this for the guard itself only after pluggy
            # accepted every one of its hooks.
            _unmark("loading")
        dependencies = getattr(plugin, "pytest_plugins", ())
        if isinstance(dependencies, str):
            dependencies = dependencies.split(",") if dependencies else ()
        if isinstance(dependencies, (tuple, list)):
            _validate(dependencies, entry_points=False)
    except Exception as exc:
        _stopping(exc)
        raise


if __name__ == "__main__":
    prefixes = json.loads(sys.argv[1])
    blocked = tuple(
        candidate
        for path in prefixes
        for candidate in (os.path.abspath(path), os.path.realpath(path))
    )
    try:
        requested = json.loads(sys.argv[2]) if len(sys.argv) > 2 else []
        # argv[3]: the policy the armed guard gets from hardening.
        policy = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
        _BENCHFLOW_TRUSTED = tuple(policy.get("trusted", ()))
        _BENCHFLOW_OWNERSHIP = bool(policy.get("ownership", True))
        print(
            json.dumps(discover(blocked, requested, ignore_blocked_registrations=True))
        )
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        sys.exit(1)
