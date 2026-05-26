"""Regression tests for Dockerfile COPY path traversal (issue #363, finding 3).

``stage_dockerfile_deps`` copies COPY sources from ``context_root`` into
``environment/_deps/`` and rewrites the Dockerfile to reference the staged
path. A malicious Dockerfile with ``COPY ../outside-secret.txt /loot`` could
previously stage arbitrary files from outside ``context_root`` into the build
context — which is one of the few host-state leak vectors the sandbox
otherwise forbids.

The fix resolves both paths and refuses to stage sources that escape
``context_root``. These tests exercise the boundary without exposing real
secrets — the "outside" file just contains a sentinel string.
"""

from __future__ import annotations

from pathlib import Path

from benchflow.sandbox.setup import _stage_copy_source, stage_dockerfile_deps


def _setup_context(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Build ``tmp_path/{outside.txt, ctx/{environment/Dockerfile}}``.

    Returns ``(context_root, env_dir, outside_path)``.
    """
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("HOST_ONLY_SECRET")

    context_root = tmp_path / "ctx"
    env_dir = context_root / "environment"
    env_dir.mkdir(parents=True)
    return context_root, env_dir, outside


def test_stage_copy_source_rejects_dotdot_traversal(tmp_path: Path) -> None:
    """`../outside-secret.txt` must not be staged into ``_deps/``."""
    context_root, env_dir, outside = _setup_context(tmp_path)

    rewritten = _stage_copy_source("../outside-secret.txt", env_dir, context_root)

    # The path is returned unchanged (not rewritten to ``_deps/``) and no
    # ``_deps/`` entry is created from outside the context.
    assert rewritten == "../outside-secret.txt"
    assert not (env_dir / "_deps").exists() or not any((env_dir / "_deps").iterdir())
    # The outside file is untouched (we did not move/copy it).
    assert outside.read_text() == "HOST_ONLY_SECRET"


def test_stage_copy_source_rejects_deep_traversal(tmp_path: Path) -> None:
    """``foo/../../outside`` resolves outside context_root and must be rejected."""
    context_root, env_dir, _ = _setup_context(tmp_path)
    # Create a real subdir so the literal prefix exists.
    (context_root / "foo").mkdir()

    rewritten = _stage_copy_source(
        "foo/../../outside-secret.txt", env_dir, context_root
    )

    assert rewritten == "foo/../../outside-secret.txt"


def test_stage_copy_source_allows_in_context_sources(tmp_path: Path) -> None:
    """Legitimate in-context COPY sources still stage normally."""
    context_root, env_dir, _ = _setup_context(tmp_path)
    src = context_root / "app.py"
    src.write_text("print('hi')")

    rewritten = _stage_copy_source("app.py", env_dir, context_root)

    assert rewritten == "_deps/app.py"
    assert (env_dir / "_deps" / "app.py").read_text() == "print('hi')"


def test_stage_dockerfile_deps_does_not_exfiltrate_outside(tmp_path: Path) -> None:
    """End-to-end: a malicious Dockerfile cannot stage host secrets.

    The Dockerfile's traversal COPY line is *left as-is* (so Docker itself
    will fail the build, which is the safe outcome) and the secret is never
    copied into ``environment/_deps/``.
    """
    context_root, env_dir, outside = _setup_context(tmp_path)
    dockerfile = env_dir / "Dockerfile"
    dockerfile.write_text("FROM scratch\nCOPY ../outside-secret.txt /loot\n")

    # ``stage_dockerfile_deps`` expects ``task_path/environment/Dockerfile``;
    # here task_path is ``env_dir.parent`` (i.e. ``context_root``).
    stage_dockerfile_deps(env_dir.parent, context_root)

    rewritten = dockerfile.read_text()
    # The line is left untouched — the staged path was not substituted, so
    # there is no ``_deps/outside-secret.txt`` written.
    assert "../outside-secret.txt" in rewritten
    deps_dir = env_dir / "_deps"
    if deps_dir.exists():
        # If a _deps dir exists, it must not contain the outside file.
        for entry in deps_dir.rglob("*"):
            assert "HOST_ONLY_SECRET" not in (
                entry.read_text() if entry.is_file() else ""
            )
    # The outside file itself remains in place, unread/unwritten.
    assert outside.read_text() == "HOST_ONLY_SECRET"
