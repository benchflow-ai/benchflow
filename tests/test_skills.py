"""Tests for benchflow.skills — skill discovery and parsing."""

from pathlib import Path

from benchflow.skills import SkillInfo, discover_skills, parse_skill


def _write_skill_md(skill_dir: Path, frontmatter: str, body: str = "") -> Path:
    """Create a SKILL.md with the given YAML frontmatter."""
    skill_dir.mkdir(parents=True, exist_ok=True)
    md = skill_dir / "SKILL.md"
    md.write_text(f"---\n{frontmatter}---\n{body}")
    return md


class TestParseSkill:
    """parse_skill(skill_md) -> SkillInfo | None"""

    def test_basic_frontmatter(self, tmp_path):
        md = _write_skill_md(tmp_path / "my-skill", "name: my-skill\ndescription: A test skill\n")
        info = parse_skill(md)
        assert info is not None
        assert info.name == "my-skill"
        assert info.description == "A test skill"
        assert info.path == tmp_path / "my-skill"

    def test_all_fields(self, tmp_path):
        frontmatter = (
            "name: full-skill\n"
            "description: Full featured\n"
            "version: 1.2.3\n"
            "compatibility: claude-code\n"
            "metadata:\n"
            "  key: value\n"
        )
        md = _write_skill_md(tmp_path / "full-skill", frontmatter)
        info = parse_skill(md)
        assert info.name == "full-skill"
        assert info.description == "Full featured"
        assert info.version == "1.2.3"
        assert info.compatibility == "claude-code"
        assert info.metadata == {"key": "value"}

    def test_name_defaults_to_dir_name(self, tmp_path):
        md = _write_skill_md(tmp_path / "fallback-name", "description: no name field\n")
        info = parse_skill(md)
        assert info is not None
        assert info.name == "fallback-name"

    def test_no_frontmatter_returns_none(self, tmp_path):
        skill_dir = tmp_path / "no-front"
        skill_dir.mkdir()
        md = skill_dir / "SKILL.md"
        md.write_text("# Just a markdown file\nNo frontmatter here.")
        assert parse_skill(md) is None

    def test_missing_closing_delimiter_returns_none(self, tmp_path):
        skill_dir = tmp_path / "no-close"
        skill_dir.mkdir()
        md = skill_dir / "SKILL.md"
        md.write_text("---\nname: broken\nno closing delimiter")
        assert parse_skill(md) is None

    def test_invalid_yaml_returns_none(self, tmp_path):
        skill_dir = tmp_path / "bad-yaml"
        skill_dir.mkdir()
        md = skill_dir / "SKILL.md"
        md.write_text("---\n: : : invalid\n---\n")
        assert parse_skill(md) is None

    def test_frontmatter_not_dict_returns_none(self, tmp_path):
        md = _write_skill_md(tmp_path / "list-front", "- item1\n- item2\n")
        assert parse_skill(md) is None

    def test_missing_file_returns_none(self, tmp_path):
        assert parse_skill(tmp_path / "nonexistent" / "SKILL.md") is None

    def test_body_after_frontmatter(self, tmp_path):
        md = _write_skill_md(
            tmp_path / "with-body",
            "name: bodied\n",
            "\n# Instructions\nDo things.\n",
        )
        info = parse_skill(md)
        assert info is not None
        assert info.name == "bodied"

    def test_empty_frontmatter_returns_none(self, tmp_path):
        """Empty YAML frontmatter (safe_load returns None) should return None."""
        skill_dir = tmp_path / "empty-front"
        skill_dir.mkdir()
        md = skill_dir / "SKILL.md"
        md.write_text("---\n---\n")
        assert parse_skill(md) is None


class TestDiscoverSkills:
    """discover_skills(*search_dirs) -> list[SkillInfo]"""

    def test_finds_skills_in_directory(self, tmp_path):
        _write_skill_md(tmp_path / "skill-a", "name: alpha\n")
        _write_skill_md(tmp_path / "skill-b", "name: beta\n")
        skills = discover_skills(tmp_path)
        assert len(skills) == 2
        names = {s.name for s in skills}
        assert names == {"alpha", "beta"}

    def test_skips_non_skill_dirs(self, tmp_path):
        _write_skill_md(tmp_path / "real-skill", "name: real\n")
        # Directory without SKILL.md
        (tmp_path / "not-a-skill").mkdir()
        (tmp_path / "not-a-skill" / "README.md").write_text("hi")
        # File (not directory)
        (tmp_path / "just-a-file.txt").write_text("hi")
        skills = discover_skills(tmp_path)
        assert len(skills) == 1
        assert skills[0].name == "real"
        assert skills[0].path == tmp_path / "real-skill"

    def test_nonexistent_dir_is_skipped(self, tmp_path):
        skills = discover_skills(tmp_path / "nope")
        assert skills == []

    def test_multiple_search_dirs(self, tmp_path):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        _write_skill_md(dir_a / "s1", "name: one\n")
        _write_skill_md(dir_b / "s2", "name: two\n")
        skills = discover_skills(dir_a, dir_b)
        assert len(skills) == 2

    def test_empty_directory(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert discover_skills(empty) == []

    def test_accepts_string_paths(self, tmp_path):
        _write_skill_md(tmp_path / "s", "name: str-path\n")
        skills = discover_skills(str(tmp_path))
        assert len(skills) == 1
        assert skills[0].name == "str-path"

    def test_sorted_by_directory_name(self, tmp_path):
        _write_skill_md(tmp_path / "zzz", "name: last\n")
        _write_skill_md(tmp_path / "aaa", "name: first\n")
        skills = discover_skills(tmp_path)
        assert skills[0].path.name == "aaa"
        assert skills[1].path.name == "zzz"


class TestSkillInfo:
    """SkillInfo dataclass basics."""

    def test_repr(self):
        s = SkillInfo(name="test", path=Path("/a/b"))
        assert "test" in repr(s)
        assert "/a/b" in repr(s)
