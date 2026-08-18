"""Interactive page assembly: payload + packaged assets → one HTML page."""

import html
from importlib import resources
from pathlib import Path
from typing import Any

from .payload import _build_acp_payload, _safe_json

_PAYLOAD_PLACEHOLDER = "__BENCHFLOW_PAYLOAD__"
_TITLE_PLACEHOLDER = "__BENCHFLOW_TITLE__"


def _theme_css() -> str:
    """The shared design tokens (assets/theme.css) — one theme for the
    interactive template and the inline legacy renderers alike."""
    assets = resources.files("benchflow.trajectories.viewer") / "assets"
    return (assets / "theme.css").read_text(encoding="utf-8")


def _load_template() -> str:
    """Assemble the self-contained page from the packaged assets."""
    assets = resources.files("benchflow.trajectories.viewer") / "assets"
    page = (assets / "template.html").read_text(encoding="utf-8")
    page = page.replace(
        "/*__BENCHFLOW_THEME_CSS__*/",
        (assets / "theme.css").read_text(encoding="utf-8"),
        1,
    )
    page = page.replace(
        "/*__BENCHFLOW_VIEWER_CSS__*/",
        (assets / "viewer.css").read_text(encoding="utf-8"),
        1,
    )
    page = page.replace(
        "//__BENCHFLOW_VIEWER_JS__",
        (assets / "viewer.js").read_text(encoding="utf-8"),
        1,
    )
    return page


def _render_shell(title: str, boot: dict[str, Any]) -> str:
    """Inject a boot document into the template (script-breakout escaped)."""
    page = _load_template()
    page = page.replace(_TITLE_PLACEHOLDER, html.escape(title), 1)
    return page.replace(_PAYLOAD_PLACEHOLDER, _safe_json(boot).replace("</", "<\\/"), 1)


def _render_acp_trajectory(
    rollout_dir: Path, acp_path: Path, prompts: list[str] | None
) -> str:
    """Render a canonical ACP rollout as a self-contained interactive page.

    Trajectory content is untrusted input: it travels as JSON data (``</``
    escaped so it cannot break out of the script tag) and the template
    renders it exclusively via ``textContent``.
    """
    payload = _build_acp_payload(rollout_dir, prompts)
    return _render_shell(
        rollout_dir.name, {"mode": "single", "payload": payload.to_payload()}
    )
