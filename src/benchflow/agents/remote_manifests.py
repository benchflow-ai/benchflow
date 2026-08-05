"""Miss-driven auto-load of DECLARATIVE agent manifests from a remote source.

Design: #876 (Phase 2a). When ``--agent <name>`` does not resolve locally,
benchflow fetches the pinned agents source (default: the first-party
``benchflow-ai/agents`` repo, cloned+cached through the same
``benchmark_repos`` machinery task sources use) and registers every
``manifest.toml`` agent found there that does not collide with anything already
registered — then resolution is retried.

Why this is safe to do automatically, unlike installing agent packages: a
manifest is **pure data**. Its ``install_cmd``/``launch_cmd`` strings execute
inside the task sandbox — exactly the trust level of a task fetched with
``--source-repo`` (and of harbor's ``acp:<id>`` registry auto-fetch, which also
fetches data and executes it sandboxed). No remote code ever runs in the host
process. Host-side *python* agent adapters (e.g. omnigent's session-factory)
are deliberately NOT auto-loaded — those remain explicit installs.

Semantics:

* **Gap-fill only** — an agent name or alias that already exists locally
  always wins; the remote manifest for it is skipped (never overwritten).
* **One-shot per process** — the first resolution miss triggers at most one
  fetch; later misses fail fast as before.
* **Guarded** — a broken manifest (or an unreachable source) logs a warning
  and never breaks agent resolution.
* **Opt-out / re-point** — ``BENCHFLOW_AGENTS_SOURCE=off`` disables;
  ``BENCHFLOW_AGENTS_SOURCE=owner/repo[@ref]`` re-pins; a local directory path
  is also accepted (dev/tests).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from benchflow.agents.manifest import (
    LoadedManifest,
    discover_manifests,
    load_agent_manifest,
    register_manifest_agents,
)

logger = logging.getLogger(__name__)

AGENTS_SOURCE_ENV = "BENCHFLOW_AGENTS_SOURCE"
DEFAULT_AGENTS_SOURCE = "benchflow-ai/agents@main"
_OFF_VALUES = frozenset({"off", "0", "none", "disabled", "false"})

# One-shot latch: the first resolution miss triggers at most one fetch per
# process. A human-readable description of what was consulted is kept for the
# unknown-agent error path.
_attempted = False
last_source_description: str = ""


def _source_root(spec: str) -> Path:
    """Resolve the source spec to a local directory of manifests.

    A local directory path is used verbatim (dev/tests); otherwise the spec is
    ``owner/repo[@ref]`` and is cloned+cached via the task-source machinery
    (data-only shallow clone, same cache as ``--source-repo``).
    """
    local = Path(spec).expanduser()
    if local.is_dir():
        return local
    repo, _, ref = spec.partition("@")
    from benchflow._utils import benchmark_repos

    try:
        return benchmark_repos.resolve_source(repo, ref=ref or None)
    except Exception:
        # Offline (or the refresh fetch failed): the cached clone is the
        # catalog of record — a user who saw the full catalog online must
        # not silently lose it, so fall back before giving up.
        org, _, name = repo.partition("/")
        cached = benchmark_repos._cache_dir() / org / name
        if cached.is_dir():
            logger.warning(
                "could not refresh agents source %s; using the cached catalog at %s",
                spec,
                cached,
            )
            return cached
        raise


def _gap_fill(
    manifests: list[LoadedManifest],
    *,
    agents: dict,
    aliases: dict,
) -> dict[str, LoadedManifest]:
    """Keep only manifests (and aliases) that collide with nothing local.

    Local always wins: an existing agent name, an existing alias, or a name
    shadowed by an alias disqualifies the remote manifest; colliding aliases on
    an otherwise-fresh manifest are stripped rather than fatal.
    """
    kept: dict[str, LoadedManifest] = {}
    for lm in manifests:
        name = lm.config.name
        if name in agents or name in aliases or name in kept:
            continue
        fresh_aliases = tuple(
            a
            for a in lm.aliases
            if a != name and a not in agents and a not in aliases and a not in kept
        )
        kept[name] = LoadedManifest(config=lm.config, aliases=fresh_aliases)
    return kept


def autoload_remote_manifest_agents() -> int:
    """Fetch + register remote manifest agents once; return how many were added.

    Called from ``resolve_agent``'s miss path. Never raises: any failure logs a
    warning and returns 0 so the normal unknown-agent error still surfaces.
    """
    global _attempted, last_source_description
    if _attempted:
        return 0
    _attempted = True

    spec = os.environ.get(AGENTS_SOURCE_ENV, DEFAULT_AGENTS_SOURCE).strip()
    if not spec or spec.lower() in _OFF_VALUES:
        last_source_description = "agents source disabled"
        return 0
    last_source_description = f"agents source {spec!r}"

    try:
        root = _source_root(spec)
    except Exception as exc:
        logger.warning(
            "Agent manifest auto-load: could not fetch %s (%s); remote agents "
            "unavailable this run.",
            spec,
            exc,
        )
        return 0

    manifests: list[LoadedManifest] = []
    for path in discover_manifests(root):
        try:
            manifests.append(load_agent_manifest(path))
        except Exception as exc:
            logger.warning(
                "Agent manifest auto-load: skipping unreadable manifest %s: %s",
                path,
                exc,
            )

    from benchflow.agents.registry import (
        AGENT_ALIASES,
        AGENT_INSTALLERS,
        AGENT_LAUNCH,
        AGENTS,
    )

    fresh = _gap_fill(manifests, agents=AGENTS, aliases=AGENT_ALIASES)
    if not fresh:
        logger.info(
            "Agent manifest auto-load: %s had no agents beyond the local registry.",
            spec,
        )
        return 0
    register_manifest_agents(
        fresh,
        agents=AGENTS,
        aliases=AGENT_ALIASES,
        installers=AGENT_INSTALLERS,
        launch=AGENT_LAUNCH,
    )
    logger.info(
        "Agent manifest auto-load: registered %d agent(s) from %s: %s",
        len(fresh),
        spec,
        ", ".join(sorted(fresh)),
    )
    return len(fresh)


def _reset_for_tests() -> None:
    global _attempted, last_source_description
    _attempted = False
    last_source_description = ""


def fetch_one(name: str) -> bool:
    """Fetch and register ONE catalog agent's manifest — never the full repo.

    Local-dir sources read ``acp/<name>/manifest.toml`` directly; remote
    ``owner/repo[@ref]`` sources fetch that single file over HTTPS (raw
    GitHub), so browsing/selecting an agent costs one small request instead
    of a multi-hundred-MB clone. Local-wins semantics: a name already
    registered is left untouched (returns True — the agent is available).
    Returns False when the manifest cannot be fetched or is invalid.
    """
    import os

    from benchflow.agents import registry
    from benchflow.agents.manifest import load_agent_manifest

    if name in registry.AGENTS:
        return True
    spec = os.environ.get(AGENTS_SOURCE_ENV) or DEFAULT_AGENTS_SOURCE
    if spec.strip().lower() in _OFF_VALUES:
        return False
    local = Path(spec).expanduser()
    try:
        if local.is_dir():
            text = (local / "acp" / name / "manifest.toml").read_text()
        else:
            import httpx

            repo, _, ref = spec.partition("@")
            url = (
                f"https://raw.githubusercontent.com/{repo}/{ref or 'main'}"
                f"/acp/{name}/manifest.toml"
            )
            resp = httpx.get(url, timeout=30, follow_redirects=True)
            if resp.status_code != 200:
                logger.warning(
                    "catalog agent %r: manifest fetch returned HTTP %s (%s)",
                    name,
                    resp.status_code,
                    url,
                )
                return False
            text = resp.text
    except Exception as exc:
        logger.warning("catalog agent %r: manifest fetch failed: %s", name, exc)
        return False

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        mpath = Path(td) / "manifest.toml"
        mpath.write_text(text)
        try:
            loaded = load_agent_manifest(mpath)
        except Exception as exc:
            logger.warning("catalog agent %r: invalid manifest: %s", name, exc)
            return False
    from benchflow.agents.manifest import register_manifest_agents

    kept = _gap_fill([loaded], agents=registry.AGENTS, aliases=registry.AGENT_ALIASES)
    if kept:
        register_manifest_agents(
            kept,
            agents=registry.AGENTS,
            aliases=registry.AGENT_ALIASES,
            installers=registry.AGENT_INSTALLERS,
            launch=registry.AGENT_LAUNCH,
        )
        logger.info("catalog agent %r: registered from single-manifest fetch", name)
    return name in registry.AGENTS
