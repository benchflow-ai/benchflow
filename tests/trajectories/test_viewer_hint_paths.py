"""Guard D2b-CONF8-2 against e9bcbd89's HTML whitespace collapsing.

Read rendered and selected text: DOM textContent alone misses this defect.
"""

import shlex
from pathlib import Path

import pytest

from benchflow.trajectories.viewer.legacy import render_rollout
from benchflow.trajectories.viewer.sources import LocalPathSource, parse_source


@pytest.fixture(scope="module")
def browser():
    pw = pytest.importorskip("playwright.sync_api")
    with pw.sync_playwright() as playwright:
        try:
            instance = playwright.chromium.launch()
        except pw.Error:
            pytest.skip("chromium not installed — run `playwright install chromium`")
        yield instance
        instance.close()


@pytest.mark.parametrize("width", [240, 1280])
@pytest.mark.parametrize("kind", ["stream", "acp"])
@pytest.mark.parametrize(
    "folder",
    [
        "run  a",
        "   leading and trailing   ",
        "a  '  b  \"  c",
        "[bold]  :smile:  [/bold]",
        "<b>  &  literal</b>",
        "$(printf D2B_HINT)  ;#",
        "hf:local  job",
        "a " * 80,
    ],
)
def test_legacy_hint_preserves_rendered_and_copied_path(
    browser, tmp_path, monkeypatch, folder, kind, width
):
    """Guards CONF8-1/2: both legacy formats keep literal, local commands."""
    monkeypatch.chdir(tmp_path)
    job = Path(folder)
    rollout = job / "trial  one"
    rollout.mkdir(parents=True)
    if kind == "stream":
        (rollout / "turn1.txt").touch()
    else:
        (rollout / "trajectory").mkdir()
        (rollout / "trajectory/acp_trajectory.jsonl").touch()
    page = browser.new_page(viewport={"width": width, "height": 600})
    try:
        page.set_content(render_rollout(job))
        command = page.locator("code").filter(has_text="bench view ")
        assert command.count() == 1
        raw = command.text_content()
        rendered = command.inner_text()
        copied = command.evaluate(
            """element => {
                const range = document.createRange();
                range.selectNodeContents(element);
                const selection = window.getSelection();
                selection.removeAllRanges();
                selection.addRange(range);
                return selection.toString();
            }"""
        )
        assert rendered == raw
        assert copied == raw
        argv = shlex.split(copied)
        assert argv[:2] == ["bench", "view"] and len(argv) == 3
        assert parse_source(argv[2]) == LocalPathSource(rollout)
        # One physical line even in a narrow viewport: copying must not add
        # layout newlines to a long, quoted path.
        assert command.evaluate("element => element.getClientRects().length") == 1
    finally:
        page.close()
