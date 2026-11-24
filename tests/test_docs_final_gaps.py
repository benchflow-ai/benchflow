"""What the 0.10.0 final candidate's dogfood found in the docs, read as a new user would.

- install.md's Linux lines installed Docker before any `apt-get update`: on a fresh Ubuntu 24.04
  image `docker.io` has no installation candidate.
- agents.md's provider table was 0.9.2's: it said Codex and Claude Code can't use `deepseek/`,
  and both run `deepseek/deepseek-flash`.
- README, the quick start and the export page said to take the job folder from the `Artifacts:`
  line; `bench run`'s footer ends with `Results: <job>/`.
- The branching page's second command picked `--checkpoint prompt:1` from a branch that kept only
  its fork (`--retain-snapshots`), and was refused: `no such checkpoint`.
- The quick start's Python snippet ran `tasks/hello`, which nothing on the page makes.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

from benchflow.agents.env import unroutable_agent_model
from benchflow.agents.providers import PROVIDERS
from benchflow.cli._output.run_summary import summary_lines

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text()


def _section(text: str, heading: str) -> str:
    m = re.search(rf"^#+ {re.escape(heading)}\n(.*?)(?=^#+ |\Z)", text, re.M | re.S)
    assert m, heading
    return m.group(1)


def _blocks(text: str, lang: str) -> list[str]:
    return re.findall(rf"^```{lang}\n(.*?)^```", text, re.M | re.S)


def test_install_updates_the_package_lists_before_installing_docker():
    (block,) = _blocks(
        _section(_read("docs/start/install.md"), "Docker on Linux"), "bash"
    )
    lines = [line.split("#")[0].strip() for line in block.splitlines()]
    install = next(
        i for i, line in enumerate(lines) if line.startswith("sudo apt-get install")
    )
    assert "sudo apt-get update" in lines[:install]


_PROTOCOL = {
    "anthropic-messages": "Anthropic Messages",
    "openai-completions": "Chat Completions",
    "openai-responses": "Responses",
}


def test_the_provider_table_says_what_the_code_routes():
    section = _section(_read("docs/evaluate/agents.md"), "The provider prefixes")
    header, _, *rows = [line for line in section.splitlines() if line.startswith("|")]
    columns = [c.strip() for c in header.strip("|").split("|")]
    for row in rows:
        cells = dict(
            zip(columns, (c.strip() for c in row.strip("|").split("|")), strict=True)
        )
        for prefix in re.findall(r"`([^`]+)/`", cells["Prefix"]):
            provider = PROVIDERS[prefix]
            model = f"{prefix}/some-model"
            assert unroutable_agent_model("claude", model) is None, prefix
            codex = "no" if unroutable_agent_model("codex", model) else "yes"
            assert cells["Codex"] == codex, (prefix, cells["Codex"])
            if cells.get("Claude Code") is not None:
                assert cells["Claude Code"] == "yes", prefix
            if provider.base_url:
                serves = sorted(
                    _PROTOCOL[p] for p, url in provider.all_endpoints.items() if url
                )
                assert cells["Serves"] == ", ".join(serves), (prefix, cells["Serves"])
    assert "Claude Code runs a model of every prefix below" in _read(
        "docs/evaluate/agents.md"
    )


def _footer_line() -> str:
    """The last line of `bench run`'s footer, from the code that prints it."""
    result = SimpleNamespace(
        results={}, total=1, passed=1, config=SimpleNamespace(agent="oracle")
    )
    return summary_lines(result, Path("jobs/2026-11-18__14-01-29"))[-1]


def test_the_pages_name_the_line_bench_run_prints():
    line = _footer_line()
    label = line.split(":")[0]
    assert line == f"{label}: jobs/2026-11-18__14-01-29/"
    for rel in ("README.md", "docs/start/quick-start.md", "docs/train/export.md"):
        text = _read(rel)
        assert "Artifacts:" not in text, rel
        assert f"`{label}:`" in text, rel
    for block in _blocks(_read("docs/start/quick-start.md"), "text"):
        if "Wall time" in block:
            assert re.search(rf"^{label}: jobs/\S+/$", block, re.M), block


def test_the_second_branch_command_starts_from_a_snapshot_the_first_keeps():
    section = _section(
        _read("docs/train/environment.md"), "Branch a rollout at a checkpoint"
    )
    first, second = _blocks(section, "bash")[:2]
    assert "--from-checkpoint" in second
    assert "--retain-snapshots" in first or "--checkpoints" in first
    picked = re.search(r"--checkpoint (\S+)", second)
    if picked and picked.group(1).startswith("prompt:"):
        assert "--checkpoints" in first


def test_the_python_snippet_fetches_the_task_the_page_runs():
    page = _read("docs/start/quick-start.md")
    (snippet,) = _blocks(_section(page, "From Python"), "python")
    compile(snippet, "quick-start.md", "exec")
    task = re.search(r"^TASK=(\S+)@([0-9a-f]{40})$", page, re.M)
    assert task, "the page's TASK line"
    owner, repo, path = task.group(1).split("/", 2)
    assert "bf.resolve_source(" in snippet
    for value in (f"{owner}/{repo}", path, task.group(2)):
        assert f'"{value}"' in snippet, value
    assert '"tasks/hello"' not in snippet
