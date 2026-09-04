"""Dark theme and syntax highlighting assets of the interactive viewer page."""

from __future__ import annotations

from importlib import resources

from benchflow.trajectories.viewer.render import (
    _load_template,
    _render_shell,
    _theme_css,
)

_ASSETS = resources.files("benchflow.trajectories.viewer") / "assets"


def test_theme_defines_light_first_and_dark_as_an_override() -> None:
    """The light tokens stay the unconditional default; dark only applies via data-theme."""
    css = _theme_css()
    assert css.index(":root {") < css.index('html[data-theme="dark"]')
    dark = css[css.index('html[data-theme="dark"]') :]
    assert "color-scheme: dark" in dark
    for token in (
        "--background",
        "--card",
        "--code-bg",
        "--kind-execute-bg",
        "--hl-kw",
    ):
        assert f"{token}:" in dark, token


def test_page_applies_stored_theme_before_paint_and_ships_a_toggle() -> None:
    """The head script reads bf-theme (falling back to the system preference)
    before the stylesheet, and the wordmark carries the toggle button."""
    page = _load_template()
    head_script = page.index('localStorage.getItem("bf-theme")')
    assert head_script < page.index("--background:")
    assert "prefers-color-scheme: dark" in page
    assert 'id="theme-toggle"' in page
    assert "BF.theme = " in page


def test_page_inlines_the_vendored_highlighter_before_the_app_scripts() -> None:
    """highlight.js is packaged (no network fetch) and loads before detail.js uses it."""
    page = _load_template()
    vendor = page.index("Highlight.js v11.9.0")
    assert vendor < page.index("BF.highlight = ")
    assert (_ASSETS / "vendor" / "LICENSE.highlight.js").is_file()
    assert "cdnjs" not in page


def test_render_shell_keeps_both_themes_in_one_document() -> None:
    """A rendered page carries the light tokens, the dark override, and the
    highlight token rules, so switching needs no reload."""
    page = _render_shell("t", {"mode": "browse", "capped": False, "rollouts": []})
    assert 'html[data-theme="dark"]' in page
    assert ".hljs-keyword" in page
    assert ".tout.term {" in page
