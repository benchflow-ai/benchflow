import re
import subprocess
from pathlib import Path

CODE_EXTENSIONS = {".css", ".html", ".js", ".py", ".sh", ".toml", ".yaml", ".yml"}
DECORATIVE_COMMENT_RE = re.compile(r"^\s*#\s*(?:[─━═]{2,}|-{3,}|={3,}|--\s+[A-Za-z].*)")
CSS_DECORATIVE_COMMENT_RE = re.compile(r"^\s*/\*\s*(?:[─━═]{2,}|-{3,}|={3,}).*\*/\s*$")
SECTION_HEADING_COMMENT_RE = re.compile(
    r"^\s*#\s*(?:"
    r"ACP(?:\s+(?:notification helpers|stdio I/O|trajectory))?|"
    r"Adapter setup|"
    r"Agent registry(?:\s+[—-]\s+all supported agents)?|"
    r"CLI|"
    r"Configuration \(\$C\$\)|"
    r"Core types|"
    r"Data classes|"
    r"Data models|"
    r"Database|"
    r"Declarative types|"
    r"Dataset loading|"
    r"Ephemeral task generation|"
    r"Evaluation orchestration|"
    r"External adapters|"
    r"GEPA export|"
    r"Helpers?(?:\s*\([^)]*\))?|"
    r"Main(?:\s+(?:ACP loop|environment class|entry point|loop))?|"
    r"Metrics|"
    r"Models / errors|"
    r"Outcome|"
    r"Parser|"
    r"Rewards plane|"
    r"Rollout|"
    r"Rollout paths|"
    r"Rollout-level run\(\)/cleanup\(\) ordering \(PR #[0-9]+ deeper review\)|"
    r"Sandbox paths|"
    r"Sandbox protocol|"
    r"Sandbox services|"
    r"Scene authoring desugaring|"
    r"SDK|"
    r"SDK\._verify\(\) integration|"
    r"Single entry point|"
    r"SkillEvalResult|"
    r"SkillEvaluator result collection|"
    r"Skills|"
    r"Task \(\$T\$\)|"
    r"Trajectories|"
    r"User abstraction \(progressive disclosure\)|"
    r"Utilities|"
    r"Verifier \(\$V\$\)|"
    r"generate_tasks"
    r")\s*$"
)


def test_section_heading_comment_detector_examples() -> None:
    """Make the section-heading anti-pattern explicit."""
    rejected_section_headings = [
        "# Database",
        "# Parser",
        "# Adapter setup",
        "# Data models",
        "    # GEPA export",
    ]
    rejected_decorative = [
        "# -- helpers",
        "/* ---- Header ---- */",
    ]
    allowed = [
        "# Parser falls back to strict mode for malformed traces.",
        "# Source .env from repo root if it exists",
        "# --rootdir is injected dynamically",
        "    # CLI --environment-manifest wins over YAML's value",
    ]

    for comment in rejected_section_headings:
        assert SECTION_HEADING_COMMENT_RE.match(comment), comment
    for comment in rejected_decorative:
        assert DECORATIVE_COMMENT_RE.match(comment) or CSS_DECORATIVE_COMMENT_RE.match(
            comment
        ), comment
    for comment in allowed:
        assert not SECTION_HEADING_COMMENT_RE.match(comment), comment
        assert not DECORATIVE_COMMENT_RE.match(comment), comment
        assert not CSS_DECORATIVE_COMMENT_RE.match(comment), comment


def test_tracked_code_avoids_section_heading_comments() -> None:
    """Guards commit 84eb49bf's repo-wide comment-style cleanup."""
    tracked_files = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    violations: list[str] = []

    for rel_path in tracked_files:
        path = Path(rel_path)
        if path.suffix not in CODE_EXTENSIONS:
            continue
        try:
            lines = path.read_text().splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            if (
                DECORATIVE_COMMENT_RE.match(line)
                or CSS_DECORATIVE_COMMENT_RE.match(line)
                or SECTION_HEADING_COMMENT_RE.match(line)
            ):
                violations.append(f"{rel_path}:{line_number}: {line.strip()}")

    assert violations == []
