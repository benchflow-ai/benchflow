"""Skill management — discover, install, load, and unload agent skills.

Skills follow the agentskills.io spec (SKILL.md with YAML frontmatter).
"""

import logging
import subprocess
import yaml
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Default skills directory (user-level)
DEFAULT_SKILLS_DIR = Path.home() / ".claude" / "skills"


@dataclass
class SkillInfo:
    """Parsed skill metadata from SKILL.md frontmatter."""

    name: str
    description: str = ""
    version: str = ""
    path: Path = field(default_factory=Path)
    compatibility: str = ""
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        return f"Skill({self.name}, {self.path})"


def discover_skills(*search_dirs: Path | str) -> list[SkillInfo]:
    """Discover skills from one or more directories.

    Scans each directory for subdirectories containing SKILL.md.
    Returns parsed SkillInfo for each discovered skill.
    """
    skills = []
    for d in search_dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for skill_dir in sorted(d.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            info = parse_skill(skill_md)
            if info:
                skills.append(info)
    return skills


def parse_skill(skill_md: Path) -> SkillInfo | None:
    """Parse a SKILL.md file and return SkillInfo."""
    try:
        text = skill_md.read_text()
        # Extract YAML frontmatter
        if not text.startswith("---"):
            return None
        end = text.index("---", 3)
        frontmatter = yaml.safe_load(text[3:end])
        if not isinstance(frontmatter, dict):
            return None
        return SkillInfo(
            name=frontmatter.get("name", skill_md.parent.name),
            description=frontmatter.get("description", ""),
            version=frontmatter.get("version", ""),
            path=skill_md.parent,
            compatibility=frontmatter.get("compatibility", ""),
            metadata=frontmatter.get("metadata", {}),
        )
    except Exception as e:
        logger.debug(f"Failed to parse {skill_md}: {e}")
        return None


def install_skill(spec: str, target_dir: Path | None = None) -> Path | None:
    """Install a skill from skills.sh.

    Args:
        spec: Skill specifier in the form "owner/repo@skill-name"
            (e.g. "anthropics/skills@find-skills"). The part after "@"
            is used to locate the installed directory. If no "@" is
            present, the last path segment is used as a guess. In either
            case, falls back to scanning target_dir for a matching
            SKILL.md name if the guessed directory doesn't exist.
        target_dir: Where to install. Default: ~/.claude/skills/

    Returns:
        Path to installed skill directory, or None on failure.
    """
    target = target_dir or DEFAULT_SKILLS_DIR
    try:
        result = subprocess.run(
            ["npx", "skills", "add", spec],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(target.parent),
        )
        if result.returncode != 0:
            logger.error(f"Failed to install skill {spec}: {result.stderr}")
            return None
        # Find the installed skill
        skill_name = spec.split("@")[-1] if "@" in spec else spec.split("/")[-1]
        installed = target / skill_name
        if installed.is_dir():
            logger.info(f"Installed skill: {skill_name} → {installed}")
            return installed
        # Search for it
        for d in target.iterdir():
            if d.is_dir() and (d / "SKILL.md").exists():
                info = parse_skill(d / "SKILL.md")
                if info and info.name == skill_name:
                    return d
        return None
    except FileNotFoundError:
        logger.error("npx not found — install Node.js to use skills.sh")
        return None
    except Exception as e:
        logger.error(f"Skill install failed: {e}")
        return None
