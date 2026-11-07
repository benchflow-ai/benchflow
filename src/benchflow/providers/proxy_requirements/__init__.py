"""What BenchFlow's LiteLLM model proxy installs: the locked set, with hashes.

The proxy used to install by resolving ``litellm[proxy]==X`` and
``boto3>=1.40`` afresh in every sandbox, so what ran there was whatever PyPI
served that day and no audit of ``uv.lock`` covered it (the Oct 6 audit
listed PYSEC-2026-4066 against litellm 1.91.0). The two files beside this
module are the ``litellm-proxy`` and ``litellm-proxy-vertex`` dependency
groups of ``pyproject.toml``, exported from ``uv.lock`` with every
distribution's sha256 (``tools/export_proxy_requirements.py``). They ship in
the wheel because users have no lock, and every proxy install reads them with
``--require-hashes``, so a tampered mirror or a moved release fails the
install instead of running.

A run may add packages to the proxy (``BENCHFLOW_LITELLM_EXTRA_REQUIREMENTS``,
PEP 508 requirements separated by whitespace, for example a provider SDK a
LiteLLM route imports). Extras install after the locked set with the same
file as constraints (``-c``; under pip, its pins without the hashes, see
``pip_install_commands``), so they can never move a locked package: an
extra that would (``litellm==1.80.0``, or a package whose own dependency
needs another botocore) is refused before any sandbox starts, naming the
extra and the locked package. Extras carry no hashes; the trial's
``config.json`` records them under ``litellm_proxy``.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from benchflow.errors import UserError

PROXY_GROUP = "litellm-proxy"
VERTEX_GROUP = "litellm-proxy-vertex"
GROUPS = (PROXY_GROUP, VERTEX_GROUP)
# Vertex routes import the Vertex SDK inside the proxy.
VERTEX_PROVIDERS = frozenset({"google-vertex", "anthropic-vertex"})
EXTRA_REQUIREMENTS_ENV = "BENCHFLOW_LITELLM_EXTRA_REQUIREMENTS"
REGENERATE_COMMAND = "uv run python tools/export_proxy_requirements.py"
# The Python the proxy venv is made with, and the one extras are checked
# against (the sandbox install pins it too).
PROXY_PYTHON = "3.12"

# ``name==version`` at the start of an exported requirement line.
_PIN_LINE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)")
# ``name==version`` anywhere in a resolver message.
_PIN_MENTION = re.compile(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([0-9][^\s,;)]*)")
_CHECK_TIMEOUT_SEC = 180
# How uv's resolver begins an answer that the requirements cannot be met;
# any other failure (the index unreachable) says nothing about the extras.
_NO_SOLUTION = "No solution found"


class ProxyExtraRequirementError(ValueError, UserError):
    """An extra requirement the proxy cannot take on top of its locked set.

    A user error: the CLI prints the message (which names the extra and the
    locked package) without a traceback.
    """


def group_for_provider(provider_name: str | None) -> str:
    """The dependency group a route's proxy installs."""
    return VERTEX_GROUP if provider_name in VERTEX_PROVIDERS else PROXY_GROUP


def requirements_file(group: str) -> Path:
    """The shipped, hashed requirement file of one group."""
    if group not in GROUPS:
        raise ValueError(f"unknown proxy dependency group {group!r}")
    return Path(__file__).with_name(f"{group}.txt")


@cache
def requirements_text(group: str) -> str:
    return requirements_file(group).read_text(encoding="utf-8")


def requirements_sha256(group: str) -> str:
    return hashlib.sha256(requirements_text(group).encode("utf-8")).hexdigest()


@cache
def locked_pins(group: str) -> dict[str, str]:
    """Canonical package name to locked version, for every line of the file."""
    pins: dict[str, str] = {}
    for line in requirements_text(group).splitlines():
        match = _PIN_LINE.match(line)
        if match:
            pins[canonicalize_name(match.group(1))] = match.group(2)
    return pins


def extra_requirements_from_env(
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """The run's extra proxy requirements, in order, without repeats."""
    raw = (os.environ if environ is None else environ).get(EXTRA_REQUIREMENTS_ENV, "")
    extras: list[str] = []
    for item in raw.split():
        if item not in extras:
            extras.append(item)
    return tuple(extras)


def _locked(group: str, name: str) -> str:
    return f"{name}=={locked_pins(group)[name]}"


def _static_conflicts(extras: Iterable[str], group: str) -> None:
    """Refuse what is wrong without a resolve: syntax, URLs, a locked name moved."""
    pins = locked_pins(group)
    file_name = requirements_file(group).name
    for number, extra in enumerate(extras, 1):
        try:
            requirement = Requirement(extra)
        except InvalidRequirement:
            # Malformed URLs need not contain :// or @. Neither the input
            # nor the parser diagnostic is safe to repeat (FF-install-C6).
            what = f"item {number} (not repeated here: it may hold credentials)"
            raise ProxyExtraRequirementError(
                f"{EXTRA_REQUIREMENTS_ENV}: {what} is not a PEP 508 requirement; "
                "give package names with optional versions, separated by spaces, "
                "for example 'google-genai>=1.0'"
            ) from None
        if requirement.url:
            raise ProxyExtraRequirementError(
                f"{EXTRA_REQUIREMENTS_ENV}: {requirement.name!r} installs from a "
                "URL (not repeated here, it may hold credentials); proxy "
                "extras come from the package index only, so the run records what "
                "it installed by name and version"
            )
        name = canonicalize_name(requirement.name)
        locked = pins.get(name)
        if locked is not None and not requirement.specifier.contains(
            locked, prereleases=True
        ):
            raise ProxyExtraRequirementError(
                f"proxy extra requirement {extra!r} conflicts with the locked "
                f"{_locked(group, name)} in BenchFlow's LiteLLM proxy set "
                f"({file_name}); extras cannot change a locked package. Drop the "
                f"extra or ask for {requirement.name} in the proxy's lock."
            )


def _uv_executable() -> str | None:
    return os.environ.get("UV") or shutil.which("uv")


def _resolver_env() -> dict[str, str]:
    """The preflight reads neither a project's uv settings nor its cache.

    CLI policy flags win over config, but inherited UV_* settings can also
    redirect interpreter downloads and indexes before resolution begins.
    """
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("UV_", "PIP_", "PYTHON"))
    }


def _resolve(extras: tuple[str, ...], group: str, uv: str) -> str | None:
    """Resolve the extras under the locked set; the resolver's message on failure."""
    try:
        result = subprocess.run(
            [
                uv,
                "pip",
                "compile",
                "--no-build",
                "--no-config",
                "--no-cache",
                "--quiet",
                "--universal",
                "--python-version",
                PROXY_PYTHON,
                "--no-header",
                "--no-annotate",
                "-c",
                str(requirements_file(group)),
                "-",
            ],
            input="\n".join(extras) + "\n",
            env=_resolver_env(),
            capture_output=True,
            text=True,
            timeout=_CHECK_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        # Unchecked extras never install: say why the check could not run.
        raise ProxyExtraRequirementError(
            f"could not check {EXTRA_REQUIREMENTS_ENV} against the proxy's "
            f"locked set with {uv}: {exc}"
        ) from exc
    if result.returncode == 0:
        return None
    return (result.stderr or result.stdout or "").strip() or (
        f"uv pip compile exited {result.returncode}"
    )


def _named_locked(message: str, group: str) -> list[str]:
    """Locked ``name==version`` pins the resolver's message mentions."""
    pins = locked_pins(group)
    named: list[str] = []
    for raw_name, version in _PIN_MENTION.findall(message):
        name = canonicalize_name(raw_name)
        pin = f"{name}=={version}"
        if pins.get(name) == version and pin not in named:
            named.append(pin)
    return named


_CHECKED: dict[tuple[str, str, tuple[str, ...]], str | None] = {}


def check_extra_requirements(
    extras: tuple[str, ...], group: str, *, uv: str | None = None
) -> None:
    """Refuse extras that would change the proxy's locked set, before any sandbox.

    The static check catches a locked package named with another version;
    a resolve with the hashed file as constraints (``uv pip compile -c``, on
    this machine, for the proxy's Python) catches an extra whose own
    dependency needs another version of a locked package. The resolve's
    answer is kept per process, so a job's trials check once.
    """
    if not extras:
        return
    _static_conflicts(extras, group)
    key = (group, requirements_sha256(group), extras)
    if key in _CHECKED and _CHECKED[key] is None:
        return
    uv = uv or _uv_executable()
    if uv is None:
        raise ProxyExtraRequirementError(
            f"{EXTRA_REQUIREMENTS_ENV} is set, and checking extras against the "
            "proxy's locked set needs uv on this machine: install uv "
            "(https://docs.astral.sh/uv/) or unset the variable"
        )
    if key not in _CHECKED:
        _CHECKED[key] = _resolve(extras, group, uv)
    message = _CHECKED[key]
    if message is None:
        return
    if _NO_SOLUTION not in message:
        # Not an answer about the extras (the index unreachable, a mirror
        # down): refuse, say so, and let a later check try again.
        del _CHECKED[key]
        raise ProxyExtraRequirementError(
            f"could not check {EXTRA_REQUIREMENTS_ENV} against the proxy's "
            f"locked set with {uv}: {message}"
        )
    culprits = list(extras)
    if len(extras) > 1:
        # Name the extra that fails on its own; when none does alone, the
        # extras conflict with each other and all of them are named.
        alone = [extra for extra in extras if _resolve((extra,), group, uv)]
        culprits = alone or culprits
    named = _named_locked(message, group)
    with_what = (
        f"the locked {', '.join(named)}"
        if named
        else "the proxy's locked set or each other"
    )
    raise ProxyExtraRequirementError(
        f"proxy extra requirement{'s' if len(culprits) > 1 else ''} "
        f"{', '.join(repr(c) for c in culprits)} conflict{'' if len(culprits) > 1 else 's'} "
        f"with {with_what} in BenchFlow's LiteLLM proxy set "
        f"({requirements_file(group).name}); extras cannot change a locked "
        f"package. Resolver: {message}"
    )


@dataclass(frozen=True)
class ProxyInstall:
    """One proxy install: the locked group and the run's extras."""

    group: str
    extra_requirements: tuple[str, ...] = ()

    @property
    def file_name(self) -> str:
        return requirements_file(self.group).name

    @property
    def text(self) -> str:
        return requirements_text(self.group)

    @property
    def sha256(self) -> str:
        return requirements_sha256(self.group)

    def check(self) -> None:
        check_extra_requirements(self.extra_requirements, self.group)

    def record(self, *, location: str) -> dict[str, object]:
        """The ``litellm_proxy`` block of a trial's ``config.json``.

        A host proxy with no extras runs the LiteLLM installed with BenchFlow
        and installs nothing, so it names no requirement file.
        """
        installs = location == "sandbox" or bool(self.extra_requirements)
        return {
            "location": location,
            "requirements": self.file_name if installs else None,
            "requirements_sha256": self.sha256 if installs else None,
            "extra_requirements": list(self.extra_requirements),
        }


def proxy_install(
    provider_name: str | None, environ: Mapping[str, str] | None = None
) -> ProxyInstall:
    return ProxyInstall(
        group=group_for_provider(provider_name),
        extra_requirements=extra_requirements_from_env(environ),
    )


def host_venv_root(environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    cache_home = (env.get("XDG_CACHE_HOME") or "").strip()
    base = Path(cache_home) if cache_home else Path.home() / ".cache"
    return base / "benchflow" / "litellm-proxy"


def ensure_host_venv(
    install: ProxyInstall, *, uv: str | None = None, root: Path | None = None
) -> Path:
    """Build a fresh host venv when the proxy has unresolved extras.

    A host proxy normally runs the LiteLLM installed with BenchFlow, but
    extras must not go into BenchFlow's own environment, so a run with
    extras gets its own venv, installed exactly as in a sandbox. Requirement
    strings do not identify the resolved dependency closure, so every start
    gets a separate directory (FF-install-C7). The host process places it
    inside its runtime directory and removes it when the proxy stops.
    """
    import fcntl
    import sys

    uv = uv or _uv_executable()
    if uv is None:
        raise ProxyExtraRequirementError(
            f"{EXTRA_REQUIREMENTS_ENV} is set, and the host proxy's venv is "
            "built with uv: install uv (https://docs.astral.sh/uv/)"
        )
    digest = hashlib.sha256(
        "\n".join(
            [install.sha256, PROXY_PYTHON, sys.platform, *install.extra_requirements]
        ).encode("utf-8")
    ).hexdigest()[:16]
    root = root if root is not None else host_venv_root()
    root.mkdir(parents=True, exist_ok=True)
    venv = (
        Path(tempfile.mkdtemp(prefix=f"{install.group}-{digest}-", dir=root))
        if install.extra_requirements
        else root / f"{install.group}-{digest}"
    )
    done = venv / ".benchflow-complete"
    with (root / f"{venv.name}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if done.is_file():
            return venv
        shutil.rmtree(venv, ignore_errors=True)
        try:
            subprocess.run(
                [uv, "venv", "-q", "--python", PROXY_PYTHON, str(venv)],
                check=True,
                capture_output=True,
                text=True,
            )
            requirements = venv / install.file_name
            requirements.write_text(install.text, encoding="utf-8")
            subprocess.run(
                [
                    "sh",
                    "-c",
                    "set -eu\n"
                    + install_commands(
                        uv='"$UV"',
                        python='"$VENV/bin/python"',
                        requirements_path='"$VENV"/' + install.file_name,
                        extras=install.extra_requirements,
                    ),
                ],
                env={**os.environ, "UV": uv, "VENV": str(venv)},
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            shutil.rmtree(venv, ignore_errors=True)
            raise RuntimeError(
                f"installing the host LiteLLM proxy venv failed ({exc.returncode}): "
                f"{(exc.stderr or exc.stdout or '').strip()[-2000:]}"
            ) from exc
        done.write_text(digest, encoding="utf-8")
    return venv


def install_commands(
    *,
    uv: str,
    python: str,
    requirements_path: str,
    extras: tuple[str, ...],
    compile_bytecode: bool = True,
) -> str:
    """Shell lines that install the locked set, then the extras under it.

    ``uv``, ``python`` and ``requirements_path`` are already shell words
    (quoted, or a ``"$UV"`` expansion). The locked set installs with
    ``--require-hashes``: a distribution whose sha256 is not in the file
    fails the install.

    Both lines take wheels only (``--no-build``). A source distribution
    builds with a backend (``flit_core``, ``setuptools``) that uv resolves
    apart from the file and checks against no hash, and the backend runs, so
    a hash-checked sdist could still bring in unchecked code.
    Every locked package has a wheel for CPython 3.12 on glibc Linux; an extra
    published only as an sdist is refused, with uv naming it.

    ``compile_bytecode`` (``UV_COMPILE_BYTECODE=1``) is for a venv that is
    built once and reused (the host venv here, a cached layer): every
    proxy start then imports from .pyc. A venv used by one trial only should
    not compile: compiling all 104 packages took 8.7 s on one CPU (11.3 s
    against 2.6 s, cold cache, October 2026) and saved 0.7 s of
    the proxy's health wait, because Python caches what the proxy imports on
    its first import anyway.
    """
    from shlex import quote

    env = "UV_COMPILE_BYTECODE=1 " if compile_bytecode else ""
    lines = [
        f"{env}{uv} pip install --python {python} -q --no-build "
        f"--require-hashes -r {requirements_path}"
    ]
    if extras:
        lines.append(
            f"{env}{uv} pip install --python {python} -q --no-build "
            f"-c {requirements_path} " + " ".join(quote(extra) for extra in extras)
        )
    return "\n".join(lines)


def pip_install_commands(
    *,
    python: str,
    requirements_path: str,
    constraints_path: str,
    extras: tuple[str, ...],
) -> str:
    """Shell lines for an image without uv: pip installs the locked set, then
    the extras under it.

    ``python``, ``requirements_path`` and ``constraints_path`` are already
    shell words. pip turns on ``--require-hashes`` for a whole command as soon
    as any requirement or constraint carries a hash, so the extras cannot take
    the hashed file as constraints: every unhashed extra would be refused.
    They take ``constraints_path`` instead, written here from the same file
    with its ``--hash`` lines and line continuations removed: the same pins,
    so an extra still cannot move a locked package. The locked set itself
    installed hash-checked on the line before.
    """
    from shlex import quote

    lines = [f"{python} -m pip install -q --require-hashes -r {requirements_path}"]
    if extras:
        lines.append(
            "sed -e '/^[[:space:]]*--hash=/d' -e 's/[[:space:]]*\\\\$//' "
            f"{requirements_path} > {constraints_path}"
        )
        lines.append(
            f"{python} -m pip install -q -c {constraints_path} "
            + " ".join(quote(extra) for extra in extras)
        )
    return "\n".join(lines)
