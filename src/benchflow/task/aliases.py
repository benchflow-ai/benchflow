"""Native/legacy task directory alias validation."""

from __future__ import annotations

from pathlib import Path

from benchflow.task.paths import TaskPaths


def normalized_tree_map(root: Path) -> dict[str, bytes]:
    """Map relative POSIX paths to file bytes for every file under *root*."""
    if not root.is_dir():
        return {}
    files: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    return files


def alias_dir_collision_issues(paths: TaskPaths) -> list[str]:
    """Return fail-closed diagnostics when native and legacy alias trees disagree."""
    issues: list[str] = []
    pairs = (
        (
            paths.oracle_dir,
            paths.legacy_solution_dir,
            TaskPaths.NATIVE_ORACLE_DIRNAME,
            TaskPaths.LEGACY_SOLUTION_DIRNAME,
        ),
        (
            paths.verifier_source_dir,
            paths.legacy_tests_dir,
            TaskPaths.NATIVE_VERIFIER_DIRNAME,
            TaskPaths.LEGACY_TESTS_DIRNAME,
        ),
    )
    for native_dir, legacy_dir, native_name, legacy_name in pairs:
        if not native_dir.is_dir() or not legacy_dir.is_dir():
            continue
        if normalized_tree_map(native_dir) == normalized_tree_map(legacy_dir):
            continue
        issues.append(
            f"Alias collision: {native_name}/ and {legacy_name}/ both exist "
            "but are not byte-identical"
        )
    return issues
