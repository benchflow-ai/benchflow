"""Guards non-destructive HF output replacement at PY8 base 363d815b."""

import pytest

from benchflow._utils import hf_datasets as hf


def copy_output(source, output, selected):
    if selected:
        hf._copy_selected_tasks(source, output, ["task-a"], overwrite=True)
    else:
        hf._copy_tree(source, output, overwrite=True)


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("relation", ["inside", "outside", "same", "symlink"])
def test_overlapping_output_is_rejected_without_changing_either_tree(
    tmp_path, selected, relation
):
    """Guards 363d815b: canonical ancestors never become replacement targets."""
    source = tmp_path / "source"
    (source / "task-a").mkdir(parents=True)
    (source / "task-a" / "marker").write_text("downloaded")
    if relation == "inside":
        output = source / "output"
    elif relation == "outside":
        output = tmp_path
    elif relation == "symlink":
        output = tmp_path / "alias"
        output.symlink_to(source, target_is_directory=True)
    else:
        output = source
    with pytest.raises(ValueError, match="overlap"):
        copy_output(source, output, selected)
    assert (source / "task-a" / "marker").read_text() == "downloaded"


@pytest.mark.parametrize("selected", [False, True])
def test_copy_failure_keeps_previous_output(tmp_path, monkeypatch, selected):
    """Guards 363d815b: copying finishes before an existing output is replaced."""
    source, output = tmp_path / "source", tmp_path / "output"
    (source / "task-a").mkdir(parents=True)
    output.mkdir()
    (output / "previous").write_text("previous snapshot")

    def fail(*args, **kwargs):
        raise OSError("synthetic copy failure")

    monkeypatch.setattr(hf.shutil, "copytree", fail)
    with pytest.raises(OSError, match="synthetic"):
        copy_output(source, output, selected)
    assert (output / "previous").read_text() == "previous snapshot"


@pytest.mark.parametrize("selected", [False, True])
def test_replacement_publishes_only_complete_output(tmp_path, selected):
    """Control for the 363d815b fix: successful replacement drops old files."""
    source, output = tmp_path / "source", tmp_path / "output"
    (source / "task-a").mkdir(parents=True)
    (source / "task-a" / "marker").write_text("new snapshot")
    output.mkdir()
    (output / "previous").write_text("old snapshot")
    copy_output(source, output, selected)
    assert (output / "task-a" / "marker").read_text() == "new snapshot"
    assert not (output / "previous").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["output", "source"]


@pytest.mark.parametrize("selected", [False, True])
def test_metadata_failure_keeps_previous_snapshot(tmp_path, monkeypatch, selected):
    """Guards 363d815b: metadata also completes before publishing new bytes."""
    source, output = tmp_path / "source", tmp_path / "output"
    (source / "task-a").mkdir(parents=True)
    output.mkdir()
    (output / "previous").write_text("previous snapshot")
    monkeypatch.setattr(hf, "_read_hf_revision", lambda *a: "a" * 40)
    monkeypatch.setattr(hf, "_download_snapshot", lambda *a, **kw: source)

    def fail(*args, **kwargs):
        raise OSError("synthetic metadata failure")

    monkeypatch.setattr(hf, "write_source_sidecar", fail)
    with pytest.raises(OSError, match="synthetic"):
        hf.snapshot_hf_dataset(
            "org/tasks",
            output_dir=output,
            overwrite=True,
            include_tasks=["task-a"] if selected else (),
        )
    assert (output / "previous").read_text() == "previous snapshot"


@pytest.mark.parametrize("error", [OSError, KeyboardInterrupt])
def test_publish_failure_restores_previous_snapshot(tmp_path, monkeypatch, error):
    """Control for the 363d815b staging fix's failed rename rollback."""
    source, output = tmp_path / "source", tmp_path / "output"
    (source / "task-a").mkdir(parents=True)
    output.mkdir()
    (output / "previous").write_text("previous snapshot")
    rename = type(output).rename

    def fail_publish(path, target):
        if path.name == "snapshot":
            raise error("synthetic publish failure")
        return rename(path, target)

    monkeypatch.setattr(type(output), "rename", fail_publish)
    with pytest.raises(error, match="synthetic publish failure"):
        copy_output(source, output, False)
    assert (output / "previous").read_text() == "previous snapshot"


@pytest.mark.parametrize("selected", [False, True])
def test_explicit_overwrite_can_replace_an_existing_file(tmp_path, selected):
    """Control for 363d815b: the staged publisher also replaces plain files."""
    source, output = tmp_path / "source", tmp_path / "output"
    (source / "task-a").mkdir(parents=True)
    (source / "task-a" / "marker").write_text("downloaded")
    output.write_text("previous file")
    copy_output(source, output, selected)
    assert (output / "task-a" / "marker").read_text() == "downloaded"
