"""Codex model catalogs that providers publish, shipped with BenchFlow.

Codex takes a model's tools, base instructions and reasoning levels from its
model catalog: the one its binary bundles, or the file ``model_catalog_json``
names. A model the catalog lacks runs on fallback metadata (plain function
tools, no ``apply_patch``), which BenchFlow refuses as an
``agent_model_catalog`` integration failure. A provider that serves the
Responses API may publish a Codex catalog for its own models; BenchFlow ships
those bytes unchanged (``sources.json``: source, fetch date, sha256) and, when
Codex runs one of those models on the provider's own route:

- :func:`benchflow.agents.codex_config.apply_codex_model_catalog` names the
  file in the run's CODEX_CONFIG (``model_catalog_json``), which codex-acp's
  app-server launcher and native ``codex exec`` both pass as ``-c``;
- :func:`install_codex_model_catalog` writes it into the sandbox before Codex
  starts, root-owned under :data:`INSTALL_DIR`. Not ``$CODEX_HOME``: the agent
  owns that folder, and native Codex reads the catalog again every turn.

``README.md`` has the sources and how to refresh them.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import tempfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any
from uuid import uuid4

from benchflow.sandbox._exec_output import exec_raw
from benchflow.trust.state import TRUSTED_TRANSFER, admit

_HERE = Path(__file__).parent
#: Where the sandbox gets each catalog: root's, outside every agent tree.
INSTALL_DIR = "/etc/codex/model-catalogs"
#: The install's receipt in the rollout directory.
RECEIPT = "codex_model_catalog.json"


class CodexModelCatalogError(RuntimeError):
    """A shipped catalog's bytes, or its install in the sandbox, did not check out."""


@dataclass(frozen=True)
class ProviderCatalog:
    """One provider's published catalog, as ``sources.json`` records it."""

    provider: str
    file: str
    url: str
    fetched: str
    sha256: str
    minimal_client_version: str

    def data(self) -> bytes:
        """The file's bytes; refused unless their sha256 is the recorded one."""
        data = (_HERE / self.file).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != self.sha256:
            raise CodexModelCatalogError(
                f"{self.file} has sha256 {digest}, not the recorded {self.sha256}"
            )
        return data

    def entries(self) -> dict[str, dict[str, Any]]:
        """The catalog's models by slug."""
        return _entries(self)

    @property
    def install_path(self) -> str:
        return f"{INSTALL_DIR}/{self.provider}-{self.sha256[:12]}.json"


@cache
def _entries(catalog: ProviderCatalog) -> dict[str, dict[str, Any]]:
    return {model["slug"]: model for model in json.loads(catalog.data())["models"]}


def _load() -> dict[str, ProviderCatalog]:
    sources = json.loads((_HERE / "sources.json").read_text(encoding="utf-8"))
    return {
        provider: ProviderCatalog(
            provider=provider,
            file=source["file"],
            url=source["url"],
            fetched=source["fetched"],
            sha256=source["sha256"],
            minimal_client_version=source["minimal_client_version"],
        )
        for provider, source in sources.items()
    }


#: The shipped catalogs, keyed by BenchFlow provider name.
CATALOGS: dict[str, ProviderCatalog] = _load()


def _lookup(model: str | None) -> tuple[ProviderCatalog, dict[str, Any]] | None:
    if not isinstance(model, str) or not model.strip():
        return None
    from benchflow.agents.providers import (
        find_provider,
        find_provider_for_bare_model,
        strip_provider_prefix,
    )

    base = model.split("[", 1)[0].strip()
    provider = find_provider(base) or find_provider_for_bare_model(base)
    catalog = CATALOGS.get(provider[0]) if provider else None
    if catalog is None:
        return None
    entry = catalog.entries().get(strip_provider_prefix(base).lower())
    return (catalog, entry) if entry else None


def codex_provider_catalog(model: str | None) -> ProviderCatalog | None:
    """The shipped catalog that lists *model* on its provider's own route
    (``deepseek/deepseek-flash`` or a bare ``deepseek-flash``), else None.
    Another route serving the same name (``openai/deepseek-flash``) gets none:
    a catalog describes its provider's endpoint."""
    found = _lookup(model)
    return found[0] if found else None


def catalog_efforts(model: str | None) -> tuple[str, ...] | None:
    """The reasoning levels *model*'s shipped catalog lists, in its order."""
    found = _lookup(model)
    if found is None:
        return None
    levels = found[1].get("supported_reasoning_levels") or []
    return tuple(level["effort"] for level in levels if isinstance(level, dict))


def catalog_for_install_path(path: object) -> ProviderCatalog | None:
    """The shipped catalog installed at *path*, None for any other path."""
    return next((c for c in CATALOGS.values() if c.install_path == path), None)


def trusted_folder_function(uid: str = "0") -> str:
    """A shell function ``t DIR``: true when DIR is a folder (not a symlink)
    owned by *uid* with no group or other write bit."""
    return (
        't() { [ ! -L "$1" ] && [ -d "$1" ] && '
        f'[ "$(stat -c %u "$1")" = {uid} ] && '
        'm=$(stat -c %a "$1") && [ $(( 0$m & 022 )) -eq 0 ]; }'
    )


def _install_script(staged: str, catalog: ProviderCatalog) -> str:
    target = shlex.quote(catalog.install_path)
    temporary = shlex.quote(f"{catalog.install_path}.tmp")
    steps = [trusted_folder_function(), "t / || exit 3", "t /etc || exit 3"]
    for folder in ("/etc/codex", INSTALL_DIR):
        steps += [
            f"[ -e {folder} ] || [ -L {folder} ] || mkdir -m 755 {folder} || exit 3",
            f"t {folder} || exit 3",
        ]
    steps += [
        f"rm -f {temporary} && cp {shlex.quote(staged)} {temporary} && "
        f"chown 0:0 {temporary} && chmod 644 {temporary} && "
        f"mv -f {temporary} {target} || exit 4",
        f"sha256sum {target}",
    ]
    return "; ".join(steps)


async def install_codex_model_catalog(
    env: Any,
    agent_env: dict[str, str],
    *,
    rollout_dir: Path | None = None,
    timeout_sec: int = 60,
) -> dict[str, Any] | None:
    """Install the shipped catalog the run's CODEX_CONFIG names, before Codex
    starts; None (nothing done) when it names none of them.

    The bytes go up into a fresh root-only folder below ``/`` (no root
    transfer lands in a tree the agent controls), and root copies them into
    :data:`INSTALL_DIR` once ``/``, ``/etc``, ``/etc/codex`` and the folder
    itself are shown root's, not group- or other-writable and not symlinks.
    The sha256 read back must be the recorded one. Any failure raises
    :class:`CodexModelCatalogError`, so Codex never starts on another
    catalog. The receipt (:data:`RECEIPT`) names the file and its source.
    """
    if rollout_dir is not None:
        (rollout_dir / RECEIPT).unlink(missing_ok=True)
    try:
        config = json.loads(agent_env.get("CODEX_CONFIG") or "{}")
    except ValueError:
        return None
    path = config.get("model_catalog_json") if isinstance(config, dict) else None
    catalog = catalog_for_install_path(path)
    if catalog is None:
        return None
    data = catalog.data()
    stage = f"/.runtime-codex-catalog-{uuid4().hex}"
    staged = f"{stage}/catalog.json"
    prepared = await exec_raw(
        env.exec,
        "case $(stat -c '%u:%a' /) in 0:755|0:555|0:711|0:700) ;; *) exit 1;; esac; "
        f"umask 077; mkdir -m 700 {stage}",
        user="root",
        timeout_sec=timeout_sec,
    )
    if prepared.return_code != 0:
        raise CodexModelCatalogError(
            "Codex model catalog staging failed: / is not root's alone"
        )
    try:
        with tempfile.TemporaryDirectory(prefix="benchflow-codex-catalog-") as local:
            source = Path(local) / "catalog.json"
            source.write_bytes(data)
            await env.upload_file(
                source,
                staged,
                mode="600",
                admission=admit(
                    env,
                    "main",
                    TRUSTED_TRANSFER,
                    reason="the Codex model catalog BenchFlow ships",
                ),
            )
        installed = await exec_raw(
            env.exec,
            _install_script(staged, catalog),
            user="root",
            timeout_sec=timeout_sec,
        )
    finally:
        await exec_raw(
            env.exec,
            f"rm -f {staged}; rmdir {stage}",
            user="root",
            timeout_sec=timeout_sec,
        )
    if installed.return_code != 0:
        where = (
            f"a folder on {INSTALL_DIR} is not root's alone"
            if installed.return_code == 3
            else f"writing {catalog.install_path} failed"
        )
        raise CodexModelCatalogError(f"Codex model catalog install refused: {where}")
    found = (installed.stdout or "").split()
    if not found or found[0] != catalog.sha256:
        raise CodexModelCatalogError(
            f"{catalog.install_path} read back with sha256 "
            f"{found[0] if found else 'none'}, not {catalog.sha256}"
        )
    receipt = {
        "provider": catalog.provider,
        "path": catalog.install_path,
        "sha256": catalog.sha256,
        "source": catalog.url,
        "fetched": catalog.fetched,
        "minimal_client_version": catalog.minimal_client_version,
        "models": sorted(catalog.entries()),
    }
    if rollout_dir is not None:
        rollout_dir.mkdir(parents=True, exist_ok=True)
        (rollout_dir / RECEIPT).write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
