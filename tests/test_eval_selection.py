"""--include/--exclude globs, --n-tasks sampling and the --timeout-multiplier / --extra-instruction overrides."""
from pathlib import Path
from types import SimpleNamespace

from benchflow.evaluation import _budget_overrides, sample_task_dirs, task_name_selected


def test_globs_and_exact_names():
    assert task_name_selected("libero-10-3", {"libero-10-*"}, set())
    assert not task_name_selected("libero-90-3", {"libero-10-*"}, set())
    assert not task_name_selected("libero-10-3", {"libero-*"}, {"*-3"})
    assert task_name_selected("a[1]", {"a[1]"}, set())  # exact names still match verbatim
    assert task_name_selected("x", set(), set())


def test_sampling_is_seeded_and_sorted():
    dirs = [Path(f"t{i:02d}") for i in range(20)]
    a = sample_task_dirs(dirs, 5, 7)
    assert a == sample_task_dirs(dirs, 5, 7) and a == sorted(a) and len(a) == 5
    assert sample_task_dirs(dirs, 3, None) == dirs[:3]
    assert sample_task_dirs(dirs, None, 1) == dirs


def test_budget_overrides(tmp_path):
    (tmp_path / "task.md").write_text("---\nschema_version: '1.0'\nagent:\n  timeout_sec: 300\n---\n\nhi\n")
    cfg = SimpleNamespace(timeout_multiplier=2.0, extra_instruction="Be brief.")
    out = _budget_overrides(cfg, tmp_path)
    assert out["prompt_suffix"] == "Be brief."
    assert out["timeout"] == 600
