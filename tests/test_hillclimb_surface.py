"""Surfaces: parsing, versions, deployment, diffs, the pasted-text check, history."""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from benchflow.hillclimbing.surface import (
    SurfaceError,
    SurfaceHistory,
    SurfaceStore,
    added_text,
    deploy_settings,
    diff_versions,
    parse_surface,
    pasted_spans,
    validate_version,
)


def _skills(root, text="Use pandas."):
    skill = root / "skills" / "csv"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(f"---\nname: csv\ndescription: d\n---\n{text}\n")
    return root / "skills"


def test_a_directory_is_a_skills_surface_and_a_file_a_prompt(tmp_path):
    skills = _skills(tmp_path)
    prompt = tmp_path / "prompt.md"
    prompt.write_text("Be careful.\n")
    assert parse_surface(skills).kind == "skills"
    assert parse_surface(prompt).kind == "prompt"
    assert parse_surface(f"prompt={prompt}").name == "prompt.md"
    with pytest.raises(SurfaceError, match="must be a directory"):
        parse_surface(f"skills={prompt}")
    with pytest.raises(SurfaceError, match="does not exist"):
        parse_surface(tmp_path / "missing")


def test_versions_deploy_through_the_existing_mechanisms(tmp_path):
    skills = _skills(tmp_path)
    prompt = tmp_path / "prompt.md"
    prompt.write_text("Check units.\n")
    store = SurfaceStore(
        tmp_path / "surfaces", [parse_surface(skills), parse_surface(prompt)]
    )
    version, skipped = store.baseline()
    assert version == "v000" and skipped == []
    deploy = deploy_settings(
        store.path("v000"), store.specs, {"agent": {"timeout_sec": 60}}
    )
    assert deploy["skill_mode"] == "with-skill"
    assert deploy["skills_dir"] == str(tmp_path / "surfaces/v000/skills")
    # The prompt joins the caller's own overlay (bench eval run --config-override).
    assert deploy["config_override"] == {
        "agent": {"timeout_sec": 60, "prompt_prefix": "Check units."}
    }


def test_an_edited_surface_drops_symlinks_and_is_validated(tmp_path):
    skills = _skills(tmp_path)
    store = SurfaceStore(tmp_path / "surfaces", [parse_surface(skills)])
    store.baseline()
    edited = tmp_path / "edited"
    shutil.copytree(tmp_path / "surfaces/v000", edited)
    os.symlink("/etc/passwd", edited / "skills/csv/leak")
    (edited / "skills/broken").mkdir()
    problems, skipped = store.add("v001", edited)
    assert skipped == ["csv/leak"]
    assert not (tmp_path / "surfaces/v001/skills/csv/leak").exists()
    assert problems == ["skills/broken has no SKILL.md"]


def test_frontmatter_is_required(tmp_path):
    skills = _skills(tmp_path)
    (skills / "csv" / "SKILL.md").write_text("no frontmatter\n")
    store = SurfaceStore(tmp_path / "surfaces", [parse_surface(skills)])
    with pytest.raises(SurfaceError, match="no YAML frontmatter"):
        store.baseline()
    assert validate_version(tmp_path / "surfaces/v000", store.specs)


def test_the_diff_shows_what_a_patch_added(tmp_path):
    old = _skills(tmp_path / "a", "Use pandas.").parent
    new = _skills(tmp_path / "b", "Use pandas.\nValidate the output file.").parent
    (new / "skills" / "units").mkdir()
    (new / "skills" / "units" / "SKILL.md").write_text("---\nname: units\n---\nSI.\n")
    diff, stats = diff_versions(old, new)
    assert "+Validate the output file." in diff
    assert "+++ b/skills/units/SKILL.md" in diff and "--- /dev/null" in diff
    assert stats.files_changed == 2 and stats.removed == 0
    assert "Validate the output file." in added_text(diff)
    assert diff_versions(old, old)[0] == ""


def test_pasted_text_is_found_by_long_word_runs():
    instruction = (
        "Compute the monthly flood exceedance for every USGS gauge in the input "
        "folder and write one JSON line per gauge."
    )
    pasted = "Tip: compute the monthly flood exceedance for every USGS gauge in the input folder."
    general = "Tip: always validate JSON output before finishing the task."
    matches = pasted_spans(pasted, {"train task instruction: t1": instruction})
    assert matches and matches[0]["source"] == "train task instruction: t1"
    assert pasted_spans(general, {"train task instruction: t1": instruction}) == []


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_history_commits_each_kept_version(tmp_path, monkeypatch):
    # A user's global git config (signing, hooks) must not leak in.
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "hostile-gitconfig"))
    (tmp_path / "hostile-gitconfig").write_text("[commit]\n\tgpgsign = true\n")
    v0 = _skills(tmp_path / "v0", "one").parent
    v1 = _skills(tmp_path / "v1", "two").parent
    history = SurfaceHistory(tmp_path / "history")
    first = history.commit(v0, ["skills"], "baseline surface")
    second = history.commit(v1, ["skills"], "r01-c1: say two")
    assert first and second and first != second
    log = subprocess.run(
        ["git", "-C", str(tmp_path / "history"), "log", "--format=%s"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\n")
    assert log[:2] == ["r01-c1: say two", "baseline surface"]
    assert "two" in (tmp_path / "history/skills/csv/SKILL.md").read_text()
