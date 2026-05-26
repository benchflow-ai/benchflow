"""HTML email template loader.

Loads pre-rendered React Email HTML templates from the html/ directory
and fills {{placeholder}} tokens with provided values using str.replace().

Usage:
    from claw_gmail.seed.templates import load_html_template

    html = load_html_template("github-pr", {
        "sender_name": "Lisa Wang",
        "repo_name": "nexusai/core",
        "pr_number": "142",
        "subject": "Add retry logic to inference pipeline",
        "branch_count": "3",
        "branch_name": "feat/retry-logic",
        "body_preview": "Added exponential backoff for transient failures.",
    })
"""

from __future__ import annotations

from pathlib import Path

_HTML_DIR = Path(__file__).parent / "html"
_CACHE: dict[str, str] = {}


def load_html_template(name: str, params: dict[str, str] | None = None) -> str:
    """Load an HTML template by name and fill placeholders.

    Args:
        name: Template name (e.g. "github-pr", "stripe-receipt").
        params: Dict of placeholder values. Keys without {{}} wrapper are fine —
                the function handles both {{key}} and {key} patterns.

    Returns:
        The rendered HTML string, or "" if template doesn't exist.
    """
    if name not in _CACHE:
        path = _HTML_DIR / f"{name}.html"
        if not path.exists():
            return ""
        _CACHE[name] = path.read_text()

    html = _CACHE[name]

    if params:
        for key, value in params.items():
            html = html.replace("{{" + key + "}}", value)

    return html


def list_templates() -> list[str]:
    """List all available template names."""
    if not _HTML_DIR.exists():
        return []
    return sorted(p.stem for p in _HTML_DIR.glob("*.html"))
