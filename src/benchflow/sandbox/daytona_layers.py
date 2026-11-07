"""Cached install layers in Daytona's image recipe.

Every Daytona sandbox BenchFlow creates from a task image goes through
Daytona's declarative builder: the SDK sends a Dockerfile text (``FROM <task
image>`` or the task's own Dockerfile) and Daytona builds it on a runner.
Daytona keeps a built recipe for 24 hours and reuses it when the same recipe
is sent again on that runner (https://www.daytona.io/docs/en/declarative-builder).
Without BenchFlow's installs in the recipe, every trial installs the in-sandbox
LiteLLM proxy (about 104 packages) and the agent (Node plus npm) cold.

This module appends install layers to the recipe, each a ``RUN`` of what the
per-trial install runs (``litellm_runtime.sandbox_litellm_install_command``
for the proxy: the same pinned uv, managed CPython and locked, hashed
requirement set, with bytecode compiled; the registry's ``install_cmd`` for
the agent), so an unchanged
recipe is the same Dockerfile text and Daytona reuses its build. Each layer
writes its key beside the private proxy or under ``/opt/runtime/layers``; the trial
compares that file with the key it expects before it skips an install, and
installs fresh when they differ. No snapshot is registered: the cache is
Daytona's own, so nothing is left to clean up.

What stays per trial: the proxy's config, master key, callbacks and logs (in
``/opt/runtime/proxy/<token>``), the agent user, credentials, skills and
lockdown. The layers hold only what the install scripts write, which take no
secrets.

The daemon guard's packages are prepared even for a single trial. A failed
install discards the layered build; the runtime guard still checks the tools
and enforces every rule on the plain image.
The delegate Python layer also discards failed attempts. Its live probe
authenticates the complete pinned tree; its key never authorizes execution.

``--fresh-install`` (``BENCHFLOW_FRESH_INSTALL=1``) leaves the layers out, so
the trial installs everything in the live sandbox as before.

#1169: a task's ``storage_mb`` is what the task needs, not what BenchFlow's own
installs need. ``reasoning-gym-hard`` asks for 1024 MB and its image takes most
of that, so the proxy install ran out of space before the agent started. The
sandbox's disk is now requested as ``storage_mb`` plus the installs' footprint
(:func:`disk_gib`), whether they come from a layer or are installed fresh:
Daytona's API documents ``disk`` only as "the disk quota for the sandbox" and
the issue shows the image counting against it, so a layer alone is not
assumed to be free.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from benchflow._root_only import root_only_tree_check
from benchflow.sandbox._exec_output import exec_raw

#: Environment switch for a fresh install, read when no flag is passed.
FRESH_INSTALL_ENV = "BENCHFLOW_FRESH_INSTALL"
#: Set when the run starts more than one trial of a task image (``--trials``
#: N > 1 sets it): only then does building the layers pay for itself.
LAYER_REUSE_ENV = "BENCHFLOW_DAYTONA_LAYER_REUSE"
#: Where each layer writes its key inside the image.
LAYER_KEY_DIR = "/opt/runtime/layers"
#: The complete cached proxy runtime; the per-trial install keeps its own path.
PROXY_LAYER_ROOT = "/opt/runtime/daytona-proxy"
PROXY_LAYER_VENV = f"{PROXY_LAYER_ROOT}/venv"
#: Disk the proxy install takes, rounded up from a measurement: on
#: python:3.13-slim the venv is 618 MB and the fresh install leaves 785 MB on
#: the disk, with pip's cache (results/U3b/footprint.txt in the round notes).
PROXY_FOOTPRINT_MB = 1024
#: Disk an agent install takes: on ubuntu:24.04 claude-agent-acp leaves 1030 MB
#: (Node and the agent in /opt/runtime, npm's cache, apt lists), codex-acp
#: 891 MB and opencode 934 MB.
AGENT_FOOTPRINT_MB = 1280
#: Allow for firewall packages and the package lists an install may refresh.
GUARD_FOOTPRINT_MB = 128
#: Scratch caches the layer scripts use, removed in the same ``RUN`` so the
#: image keeps only what the install put in place.
_LAYER_CACHE_DIR = "/tmp/runtime-layer-cache"
_LAYER_SCRIPT = "/tmp/runtime-layer.sh"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_USER_LINE = re.compile(r"^\s*USER\s+(\S.*?)\s*$", re.IGNORECASE)
_FROM_LINE = re.compile(r"^\s*FROM\s", re.IGNORECASE)
# A build record older than the sandbox by more than this existed before it.
_BUILD_RECORD_MARGIN = timedelta(seconds=5)


def fresh_install_requested(flag: bool | None = None) -> bool:
    """Whether this trial installs fresh: the flag, else the environment."""
    if flag:
        return True
    value = os.environ.get(FRESH_INSTALL_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


# Layer keys of the recipes this process built, and of those being built.
_BUILT: set[tuple[tuple[str, str], ...]] = set()
_BUILDING: set[tuple[tuple[str, str], ...]] = set()
_FAILED: set[tuple[tuple[str, str], ...]] = set()


def layer_build_failed(keys: Mapping[str, str]) -> bool:
    """A confirmed failure stays plain for this process; a new process retries."""
    return tuple(sorted(keys.items())) in _FAILED


def claim_layer_build(keys: Mapping[str, str]) -> str | None:
    """Whether a trial's recipe carries its layers (DFX-SCALE M4).

    ``"built"``: this process built the same recipe. ``"build"``: the run
    reuses the image (:data:`LAYER_REUSE_ENV`) and no other trial is building
    it; this trial does, and reports with :func:`layer_build_finished`. None:
    the trial installs in its sandbox, since a cold build costs the one trial
    more time than the installs it saves, and no trial waits on another's.
    """
    ident = tuple(sorted(keys.items()))
    if ident in _FAILED:
        return None
    if ident in _BUILT:
        return "built"
    reuse = os.environ.get(LAYER_REUSE_ENV, "").strip().lower()
    if ident in _BUILDING or reuse not in {"1", "true", "yes", "on"}:
        return None
    _BUILDING.add(ident)
    return "build"


def layer_build_finished(
    keys: Mapping[str, str], *, built: bool, failed: bool = False
) -> None:
    ident = tuple(sorted(keys.items()))
    _BUILDING.discard(ident)
    if failed:
        _FAILED.add(ident)
    if built:
        _BUILT.add(ident)
    else:
        _BUILT.discard(ident)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class InstallLayer:
    """One cached install: the script the trial would run, and where it goes."""

    kind: str  # "guard", "proxy", "agent" or "python"
    name: str  # "daemon", "litellm", "delegate" or the agent's name
    script: str
    venv: str | None = None
    extra_requirements: tuple[str, ...] = ()
    timeout_sec: int = 900

    @property
    def marker(self) -> str:
        if self.kind == "proxy":
            return f"{PROXY_LAYER_ROOT}/proxy.key"
        return f"{LAYER_KEY_DIR}/{self.kind}-{self.name}.key"

    def key(self, base_recipe: str) -> str:
        """The layer's key: what it installs and the image it is built on.

        The script holds the pinned versions and the install recipe; the base
        recipe is the task image's Dockerfile text (its ``FROM`` line names the
        image), so two task images never share a key. The libc and
        architecture are the runner's: the layer is built inside that image.
        """
        payload = json.dumps(
            {
                "schema": 2,  # self-contained uv installs and post-cleanup validation
                "kind": self.kind,
                "name": self.name,
                "recipe_sha256": _sha256(self.script),
                "base_sha256": _sha256(base_recipe),
            },
            sort_keys=True,
        )
        return _sha256(payload)[:32]


@dataclass(frozen=True)
class InstallLayerPlan:
    """What the trial installs, and whether the recipe carries it."""

    fresh: bool
    proxy: InstallLayer | None = None
    agent: InstallLayer | None = None
    footprint_mb: int = 0
    #: Carry proxy/agent layers only as :func:`claim_layer_build` decides.
    only_if_reused: bool = False
    guard: InstallLayer | None = None
    python: InstallLayer | None = None

    @property
    def layers(self) -> tuple[InstallLayer, ...]:
        if self.fresh:
            return ()
        return tuple(
            layer
            for layer in (self.guard, self.proxy, self.agent, self.python)
            if layer
        )

    @property
    def build_allowance_sec(self) -> int:
        """Keep the budgets the same installs have when run in the trial."""
        return sum(layer.timeout_sec for layer in self.layers)


def plan_install_layers(
    *,
    agent: str | None,
    model: str | None,
    agent_env: Mapping[str, str] | None = None,
    fresh: bool = False,
    install_agent: bool = True,
    sandbox_setup_timeout: int = 120,
    only_if_reused: bool = False,
    guard_daemon: bool = False,
    stage_python: bool = False,
) -> InstallLayerPlan:
    """The installs a Daytona trial of ``agent`` on ``model`` runs, as layers.

    The proxy layer is planned when the agent's model traffic goes through
    BenchFlow's in-sandbox proxy (always on Daytona); a route that cannot be
    resolved gets no layer, so the trial fails where it always did, with the
    route's own error. The agent layer is planned when the agent has an
    install script. Guard packages and delegate Python are planned only for
    their callers; these small layers do not require reuse.
    """
    from benchflow.agents.env import uses_native_subscription_auth
    from benchflow.agents.install import effective_install_timeout
    from benchflow.agents.registry import AGENT_INSTALLERS, cacheable_install
    from benchflow.providers.litellm_config import resolve_litellm_route
    from benchflow.providers.litellm_runtime import needs_litellm_runtime
    from benchflow.providers.proxy_requirements import proxy_install

    env = dict(agent_env or {})
    footprint = 0
    proxy: InstallLayer | None = None
    agent_layer: InstallLayer | None = None
    guard: InstallLayer | None = None
    python: InstallLayer | None = None
    if guard_daemon:
        from benchflow.sandbox.lockdown import DAEMON_GUARD_TIMEOUT_SEC

        footprint += GUARD_FOOTPRINT_MB
        guard = InstallLayer(
            kind="guard",
            name="daemon",
            script=daemon_guard_install_command(),
            timeout_sec=DAEMON_GUARD_TIMEOUT_SEC,
        )
    if stage_python:
        from benchflow.sandbox import _staged_python_layer

        footprint += _staged_python_layer.FOOTPRINT_MB
        python = InstallLayer(
            kind="python",
            name="delegate",
            script=_staged_python_layer.install_command(),
            timeout_sec=300,
        )
    agent_base = (agent or "").split()[0] if (agent or "").split() else ""
    if (
        agent_base
        and needs_litellm_runtime(agent_base, model)
        and not uses_native_subscription_auth(agent_base, model, env)
    ):
        footprint += PROXY_FOOTPRINT_MB
        try:
            route = resolve_litellm_route(model or "", env)
        except Exception:  # the runtime reports the route's own error
            route = None
        if route is not None:
            # The locked, hashed set of the route (litellm-proxy-vertex for
            # Vertex) and the run's extras, as in the trial's install.
            install = proxy_install(route.provider_name)
            proxy = InstallLayer(
                kind="proxy",
                name="litellm",
                script=proxy_install_command(install),
                venv=PROXY_LAYER_VENV,
                extra_requirements=install.extra_requirements,
                timeout_sec=max(600, sandbox_setup_timeout),
            )
    if install_agent and agent_base in AGENT_INSTALLERS:
        footprint += AGENT_FOOTPRINT_MB
        # Only the audited, pinned built-in recipe is credential-free. Custom
        # and manifest commands (including overrides of built-ins) stay live.
        cached = cacheable_install(agent_base)
        if cached is not None and _SAFE_NAME.match(agent_base):
            agent_layer = InstallLayer(
                kind="agent",
                name=agent_base,
                script=cached[1],
                timeout_sec=effective_install_timeout(agent_base, sandbox_setup_timeout)
                or 0,
            )
    return InstallLayerPlan(
        fresh=fresh,
        proxy=proxy,
        agent=agent_layer,
        footprint_mb=footprint,
        only_if_reused=only_if_reused,
        guard=guard,
        python=python,
    )


def daemon_guard_install_command() -> str:
    """Prepare both families: the build kernel cannot predict runtime IPv6."""
    return (
        '[ "$(id -u)" = 0 ] || exit 1; '
        "if ! command -v iptables >/dev/null 2>&1 || "
        "! command -v ip6tables >/dev/null 2>&1; then "
        "if command -v apt-get >/dev/null 2>&1; then "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y -q iptables || "
        "{ apt-get update -q && "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y -q iptables; }; "
        "elif command -v dnf >/dev/null 2>&1; then dnf -y install iptables; "
        "elif command -v apk >/dev/null 2>&1; then "
        "apk add --no-cache iptables ip6tables; "
        "else exit 0; fi || exit $?; fi; "
        "command -v iptables >/dev/null 2>&1 && "
        "command -v ip6tables >/dev/null 2>&1"
    )


def proxy_install_command(install: Any) -> str:
    """Build the venv and its complete managed Python under one private root.

    The image build is trusted. No live role repairs or takes ownership of
    this tree: a changed owner/mode or an escaping link makes reuse fail.
    uv's advisory locks and CPython's internal relative links are normalized
    here, while the installer output is still trusted, before a key is written.
    """
    from benchflow.providers.litellm_runtime import sandbox_litellm_install_command

    root = shlex.quote(PROXY_LAYER_ROOT)
    return (
        f"set -eu\nmkdir -p {root}\nchmod 700 {root}\n"
        + sandbox_litellm_install_command(
            PROXY_LAYER_VENV,
            install,
            python_install_dir=f"{PROXY_LAYER_ROOT}/python",
        )
        + "\n"
        + _seal_proxy_command()
    )


def _seal_proxy_command() -> str:
    """Make trusted install output expressible by the strict custody guard.

    Resolve every link before changing any, refuse external/broken links and
    special files, then remove write access for group/other on every inode.
    No live sandbox invokes this repair: only the trusted image build does.
    """
    code = """import os, stat, sys
from pathlib import Path
root = Path(sys.argv[1])
links = []
entries = [root]
for directory, dirs, files in os.walk(root, followlinks=False):
    entries.extend(Path(directory) / name for name in dirs + files)
for path in entries:
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        target = path.resolve(strict=True)
        if not target.is_relative_to(root):
            raise RuntimeError('proxy install link escapes private root')
        links.append((path, target))
    elif not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
        raise RuntimeError('proxy install contains a special file')
for path, target in links:
    path.unlink()
    path.symlink_to(target)
for path in entries:
    mode = path.lstat().st_mode
    if not stat.S_ISLNK(mode):
        path.chmod(stat.S_IMODE(mode) & ~0o022)
"""
    python = shlex.quote(f"{PROXY_LAYER_VENV}/bin/python")
    return f"{python} -I -c {shlex.quote(code)} {shlex.quote(PROXY_LAYER_ROOT)}\n"


def _proxy_custody_command() -> str:
    """Check custody with image tools, without executing the cached Python.

    Ancestors cannot be aliases or writable; every runtime file and resolved
    link stays in the private root. Relative links containing '..' are refused
    rather than traversing an unprotected intermediate directory. This also
    refuses workspaces overlapping the tree after workspace ownership setup.
    """
    root = shlex.quote(PROXY_LAYER_ROOT)
    return f"""(
set -eu
root={root}
p="$root"
while :; do
    test ! -L "$p" && test -d "$p" || exit 1
    test "$(stat -c %u "$p")" = 0 || exit 1
    mode=$(stat -c %a "$p") || exit 1
    test "$((0$mode & 022))" = 0 || exit 1
    test "$p" != / || break
    p=${{p%/*}}
    test -n "$p" || p=/
done
test "$(stat -c %a "$root")" = 700 || exit 1
bad=$(find -L "$root" \\( ! -user root -o -perm /022 -o \\( ! -type f -a ! -type d \\) \\) -print) || exit 1
test -z "$bad" || exit 1
bad=$(find "$root" -type l -exec sh -c '
    root=$1; shift
    for link do
        target=$(readlink "$link") || exit 1
        case "$target" in
            "$root"/*) ;;
            /*) echo escape; continue ;;
        esac
        case "/$target/" in */../*) echo escape; continue ;; esac
        resolved=$(readlink -f "$link") || exit 1
        case "$resolved" in "$root"/*) ;; *) echo escape ;; esac
    done
' sh "$root" {{}} +) || exit 1
test -z "$bad"
)"""


def disk_gib(storage_mb: int, footprint_mb: int, max_storage_mb: int) -> int:
    """The disk to request, in GiB: the task's storage plus BenchFlow's installs.

    Rounded up (Daytona takes whole GiB); never above the account's maximum,
    and never below what the task asked for before (``storage_mb // 1024``).
    """
    task_gib = storage_mb // 1024
    if footprint_mb <= 0:
        return task_gib
    wanted = math.ceil((storage_mb + max(footprint_mb, 0)) / 1024)
    ceiling = max(max_storage_mb // 1024, task_gib)
    return max(min(wanted, ceiling), task_gib)


def _last_user(dockerfile: str) -> str | None:
    """The ``USER`` the final build stage runs as, or None for the image's own.

    A ``USER`` does not carry across ``FROM``: in a multi-stage Dockerfile only
    the last stage's ``USER`` lines decide the image's user.
    """
    user = None
    for line in dockerfile.splitlines():
        if _FROM_LINE.match(line):
            user = None
            continue
        match = _USER_LINE.match(line)
        if match:
            user = match.group(1)
    return user


def layer_run_line(layer: InstallLayer, key: str) -> str:
    """The ``RUN`` instruction that builds ``layer`` and records its key.

    The script travels base64-encoded so its newlines and quotes survive the
    Dockerfile; it runs under ``bash`` as the per-trial install does. Package
    caches go to a scratch folder removed in the same instruction, so they
    never reach the image or the sandbox's disk. The guard and Python scripts
    run directly; failed or interrupted installs discard the build.
    """
    if layer.kind in {"guard", "python"}:
        script = shlex.quote(layer.script)
        before = after = ""
        ready = (
            "command -v iptables >/dev/null 2>&1 && "
            "command -v ip6tables >/dev/null 2>&1"
        )
        if layer.kind == "python":
            from benchflow.sandbox import _delegate_disarm, _staged_python_layer

            before = _staged_python_layer.archive_add_line() + "\n"
            ready = (
                " || ".join(
                    f"[ -x {shlex.quote(_delegate_disarm.staged_python_bin(a, c))} ]"
                    for a, c in _delegate_disarm.STAGED_PYTHON_SHA256
                )
                or "false"
            )
            after = f" && rm -f {shlex.quote(_staged_python_layer.BUNDLED_ARCHIVE)}"
        # Only an unattemptable install may succeed without the tools. Failed
        # attempts must use the plain image, never a partial package transaction.
        return (
            before + "RUN ( if command -v timeout >/dev/null 2>&1; then "
            f"timeout {layer.timeout_sec} sh -c {script}; "
            f"else sh -c {script}; fi && "
            f"if {ready}; then "
            f"mkdir -p {LAYER_KEY_DIR} && "
            f"printf '%s\\n' {shlex.quote(key)} > {shlex.quote(layer.marker)} "
            "; fi" + after + " )"
        )
    body = (
        f"export UV_CACHE_DIR={_LAYER_CACHE_DIR}/uv "
        "UV_LINK_MODE=copy "
        f"PIP_CACHE_DIR={_LAYER_CACHE_DIR}/pip "
        f"npm_config_cache={_LAYER_CACHE_DIR}/npm\n" + layer.script + "\n"
    )
    encoded = base64.b64encode(body.encode("utf-8")).decode("ascii")
    # Bound both Dockerfile lines and the shell's argv. Continuations alone
    # still make one shell argument larger than Linux's MAX_ARG_STRLEN.
    staged = f"{_LAYER_SCRIPT}.b64"
    chunks = [encoded[i : i + 32768] for i in range(0, len(encoded), 32768)]
    writes = "\n".join(
        f"RUN printf '%s' {chunk} {'>' if i == 0 else '>>'} {staged}"
        for i, chunk in enumerate(chunks)
    )
    marker = shlex.quote(layer.marker)
    validate = (
        f"{shlex.quote(layer.venv)}/bin/python -c 'import litellm' && "
        if layer.venv
        else ""
    )
    return (
        writes + "\n"
        f"RUN base64 -d {staged} > {_LAYER_SCRIPT} && "
        f"bash {_LAYER_SCRIPT} && "
        f"rm -rf {_LAYER_SCRIPT} {_LAYER_CACHE_DIR} {staged} && "
        f"{validate}"
        f"mkdir -p {LAYER_KEY_DIR} && "
        f"printf '%s\\n' {key} > {marker} && "
        f"chmod -R a+rX /opt/runtime && "
        # Both layers expose the agent tools; keep the proxy's private root
        # private after either layer, including the later agent install.
        f"if [ -d {PROXY_LAYER_ROOT} ]; then chmod 700 {PROXY_LAYER_ROOT}; fi"
    )


@dataclass
class LayeredRecipe:
    """The recipe's added lines and the keys they carry."""

    lines: list[str] = field(default_factory=list)
    keys: dict[str, str] = field(default_factory=dict)


def layered_recipe(plan: InstallLayerPlan, base_recipe: str) -> LayeredRecipe:
    """The Dockerfile lines that add ``plan``'s layers on top of ``base_recipe``.

    A task Dockerfile that switches to another user gets ``USER root`` for the
    layers and its own user back after them. ``FROM <image>`` recipes run as
    the image's user; a build that fails there falls back to the plain recipe
    (``_DaytonaDirect.start``).
    """
    recipe = LayeredRecipe()
    layers = plan.layers
    if not layers:
        return recipe
    user = _last_user(base_recipe)
    if user is not None:
        recipe.lines.append("USER root")
    for layer in layers:
        key = layer.key(base_recipe)
        recipe.keys[f"{layer.kind}-{layer.name}"] = key
        recipe.lines.append(layer_run_line(layer, key))
    if user is not None:
        recipe.lines.append(f"USER {user}")
    return recipe


def _daytona_time(value: Any) -> datetime | None:
    """A Daytona timestamp (a datetime, or the API's ISO text) in UTC."""
    if isinstance(value, str) and value:
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def build_record_outcome(sandbox: Any) -> dict[str, Any]:
    """What Daytona's build record says about this sandbox's recipe, as facts.

    ``build_record`` is "reused" when Daytona's build record for the recipe
    was created before this sandbox (by more than a few seconds), "new" when
    it was created with it, and "unknown" when either time is missing. Both
    times are Daytona's, so this machine's clock plays no part.

    It is not a per-runner cache hit: Daytona keeps one build record per
    recipe for every runner, and a trial that lands on a runner that has not
    built the recipe can build the layers again under a "reused" record. Whether
    a trial started warm is read across trials: an earlier trial with the
    same ``build_ref`` on the same ``runner_id``.
    """
    info = getattr(sandbox, "build_info", None)
    built = _daytona_time(getattr(info, "created_at", None))
    created = _daytona_time(getattr(sandbox, "created_at", None))
    outcome: dict[str, Any] = {"build_record": "unknown"}
    if built is not None:
        outcome["build_created_at"] = built.isoformat()
    if created is not None:
        outcome["sandbox_created_at"] = created.isoformat()
    if built is not None and created is not None:
        outcome["build_record"] = (
            "reused" if built < created - _BUILD_RECORD_MARGIN else "new"
        )
    ref = getattr(info, "snapshot_ref", None)
    if isinstance(ref, str) and ref:
        outcome["build_ref"] = ref
    runner = getattr(sandbox, "runner_id", None)
    if isinstance(runner, str) and runner:
        outcome["runner_id"] = runner
    return outcome


async def cached_layer_in_sandbox(
    sandbox: Any,
    kind: str,
    name: str,
    *,
    script: str | None = None,
) -> InstallLayer | None:
    """The cached ``kind``/``name`` layer when the live sandbox holds it, else None.

    The sandbox says which layers its image was built with
    (``DaytonaSandbox.cached_install_layer``); the layer counts only when the
    key file in the sandbox matches the key the recipe carried. A proxy also
    requires the complete current install script at its fixed venv path,
    including the locked dependency set, extras and toolchain. Otherwise the
    caller installs as it always did. Looked up on the type, so a mocked sandbox
    never fakes a layer.
    """
    lookup = getattr(type(sandbox), "cached_install_layer", None)
    if not callable(lookup):
        return None
    found = lookup(sandbox, kind, name)
    if not found:
        return None
    layer, key = found
    note = getattr(type(sandbox), "note_install_layer", None)
    if kind == "proxy" and (
        script is None or layer.script != script or layer.venv != PROXY_LAYER_VENV
    ):
        if callable(note):
            note(sandbox, kind, name, "not used: other install recipe, installed fresh")
        return None
    checks = [f"cat {shlex.quote(layer.marker)}"]
    if layer.venv:
        # Root runs this venv's Python next: only while it is root's alone
        # (C8), not merely while its key matches. Else the trial installs
        # its own in its runtime folder.
        venv = shlex.quote(layer.venv)
        checks[:0] = [
            f"test -x {venv}/bin/python",
            f"test -x {venv}/bin/litellm",
            root_only_tree_check(layer.venv, "the cached proxy venv"),
        ]
    if kind == "proxy":
        checks.insert(0, _proxy_custody_command())
    try:
        # Lookup can follow an agent role. The inspection tools and marker
        # read must use the same trusted launch path as other root plumbing;
        # an image PATH may include a workspace the earlier role could write.
        from benchflow.sandbox._base import SYSTEM_SHELL_EMPTY_VARS

        result = await exec_raw(
            sandbox.exec,
            " && ".join(checks),
            timeout_sec=30,
            user="root",
            system_shell=True,
            env=dict.fromkeys(SYSTEM_SHELL_EMPTY_VARS, ""),
        )
        found_key = (getattr(result, "stdout", "") or "").strip()
        ok = getattr(result, "return_code", 1) == 0 and found_key == key
    except Exception:
        ok = False
    if callable(note):
        note(sandbox, kind, name, "used" if ok else "not found, installed fresh")
    return layer if ok else None
