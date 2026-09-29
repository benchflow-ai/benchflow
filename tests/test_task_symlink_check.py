"""A task that relies on symlinks is refused by bench tasks check.

Regression test: sandbox uploads skip symlinks on purpose
(#411), so a task whose ``verifier/`` is a symlink to another task's reached
the sandbox with an empty ``/tests``; the verifier failed as
``Verifier setup failed: chmod exited with rc=1`` (``verifier_infra``, which
is retried), and the agent ran again on every retry for nothing. The
bundled ``citation-check-network`` example was built that way.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from benchflow._utils.task_authoring import check_task

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs" / "examples" / "task-md" / "real-skillsbench" / "citation-check"


def test_symlinked_task_directory_is_an_issue(tmp_path: Path) -> None:
    base = tmp_path / "base"
    shutil.copytree(BASE, base)
    variant = tmp_path / "variant"
    variant.mkdir()
    shutil.copy(base / "task.md", variant / "task.md")
    for name in ("environment", "oracle", "verifier"):
        (variant / name).symlink_to(Path("..") / "base" / name)
    issues = check_task(variant)
    assert any("symlink" in issue and "verifier" in issue for issue in issues), issues
    assert check_task(base) == [] or not any("symlink" in i for i in check_task(base))


def test_bundled_examples_have_no_symlinks() -> None:
    examples = [ROOT / "docs" / "examples", ROOT / "tests" / "examples"]
    links = [p for root in examples for p in root.rglob("*") if p.is_symlink()]
    assert links == []


def _linked_batch(tmp_path: Path, *, legacy: bool) -> Path:
    root = tmp_path / "tasks"
    plain = root / "plain"
    (plain / "environment").mkdir(parents=True)
    (plain / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    tests_dir = "tests" if legacy else "verifier"
    solve_dir = "solution" if legacy else "oracle"
    (plain / tests_dir).mkdir()
    (plain / tests_dir / "test.sh").write_text(
        "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"
    )
    (plain / solve_dir).mkdir()
    (plain / solve_dir / "solve.sh").write_text("#!/bin/bash\ntrue\n")
    if legacy:
        (plain / "task.toml").write_text(
            'version = "1.0"\n[agent]\ntimeout_sec = 60.0\n'
        )
        (plain / "instruction.md").write_text("Do nothing.\n")
    else:
        (plain / "task.md").write_text(
            '---\nschema_version: "1.3"\nagent:\n  timeout_sec: 60\n---\n\n## prompt\n\nDo nothing.\n'
        )
    linked = root / "linked"
    linked.mkdir()
    for name in ("task.toml", "instruction.md", "task.md"):
        if (plain / name).exists():
            shutil.copy(plain / name, linked / name)
    for name in ("environment", tests_dir, solve_dir):
        (linked / name).symlink_to(Path("..") / "plain" / name)
    return root


import pytest  # noqa: E402


@pytest.mark.parametrize("legacy", [False, True], ids=["task-md", "legacy-toml"])
def test_batch_skips_a_symlinked_task_with_a_warning(
    tmp_path: Path, caplog, legacy
) -> None:
    import benchflow as bf

    root = _linked_batch(tmp_path, legacy=legacy)
    ev = bf.Evaluation(
        tasks_dir=root,
        jobs_dir=tmp_path / "jobs",
        config=bf.EvaluationConfig(agent="oracle", environment="daytona"),
        preflight=False,
    )
    with caplog.at_level("WARNING"):
        selected = ev._get_task_dirs()
    assert [p.name for p in selected] == ["plain"]
    assert any(
        "linked" in r.getMessage() and "symlink" in r.getMessage()
        for r in caplog.records
    )
