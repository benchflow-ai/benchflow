"""The top-level ``benchflow`` namespace a Python user sees.

Using an Environment-plane manifest from Python needed a private-looking deep
import (``benchflow.environment.manifest``), and the deprecated shims gave no
signal: ``benchflow.sdk.SDK`` and the pre-#384 ``bf.snapshot`` /
``bf.restore`` / ``bf.list_snapshots`` aliases kept working silently.
"""

from __future__ import annotations

import warnings

import pytest

import benchflow as bf


def test_manifest_types_are_top_level() -> None:
    from benchflow.environment.manifest import EnvironmentManifest, load_manifest

    assert bf.EnvironmentManifest is EnvironmentManifest
    assert bf.load_manifest is load_manifest
    assert {"EnvironmentManifest", "load_manifest"} <= set(bf.__all__)


def test_every_public_name_resolves_without_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name in bf.__all__:
            getattr(bf, name)


@pytest.mark.parametrize(
    "alias,target",
    [
        ("snapshot", "workspace_snapshot"),
        ("restore", "workspace_restore"),
        ("list_snapshots", "list_workspace_snapshots"),
    ],
)
def test_workspace_snapshot_aliases_warn(alias: str, target: str) -> None:
    with pytest.warns(DeprecationWarning, match=target):
        value = getattr(bf, alias)
    assert value is getattr(bf, target)
    assert alias not in bf.__all__


def test_sdk_shim_warns_and_names_the_replacement() -> None:
    from benchflow.sdk import SDK

    with pytest.warns(DeprecationWarning, match=r"bf\.run"):
        SDK()


def test_unknown_attribute_still_raises() -> None:
    with pytest.raises(AttributeError, match="no_such_name"):
        bf.no_such_name  # noqa: B018


def test_resolve_source_is_public() -> None:
    """The docs' Python examples fetch remote tasks with resolve_source, which
    was importable only from the private ``benchflow._utils`` package."""
    from benchflow._utils.benchmark_repos import resolve_source

    assert bf.resolve_source is resolve_source
    assert "resolve_source" in bf.__all__
