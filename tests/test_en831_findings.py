"""Three findings from the Harbor comparison.

- Bad key: an upstream 401 surfaced as "provider unavailable (HTTP 500)",
  because LiteLLM's callback log records the refused call as a 500. The proxy's
  own request record has the 401 the agent got; the trial's error now leads
  with the refused key and names the provider and its key variable.
- Discovery: ``bench run gpqa-diamond -n 2`` found nothing without an explicit
  registry. BenchFlow's packaged registry now pins GPQA Diamond 1.0 (198 tasks,
  a digest each), and a bare dataset name runs it.
- Output volume: the live panel is a header, one progress row and one plain
  row per running trial (no bar, no boxed table).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow._utils.scoring import PROVIDER_AUTH, classify_error


def _events(path: Path, statuses: list[tuple[str, int]]) -> Path:
    path.write_text(
        "".join(json.dumps({"event": e, "status": s}) + "\n" for e, s in statuses)
    )
    return path


def test_the_proxy_record_s_401_is_a_refused_key(tmp_path):
    from benchflow.rollout._usage import _provider_failure_from_attempts

    log = _events(
        tmp_path / "provider_requests.jsonl",
        [
            ("upstream_end", 200),
            ("response_start", 200),
            ("request_end", 200),
            ("upstream_end", 401),
            ("response_start", 401),
            ("request_end", 401),
        ],
    )
    failure = _provider_failure_from_attempts(log)
    assert failure is not None and failure.status == 401
    assert failure.marker == "provider auth failed"


@pytest.mark.parametrize("last", [200, 500, 429])
def test_only_a_refused_credential_overrides(tmp_path, last):
    from benchflow.rollout._usage import _provider_failure_from_attempts

    log = _events(tmp_path / "p.jsonl", [("request_end", 401), ("request_end", last)])
    assert _provider_failure_from_attempts(log) is None
    assert _provider_failure_from_attempts(tmp_path / "missing.jsonl") is None


def test_the_refusal_names_the_provider_and_its_key_and_leads_the_category():
    from benchflow.rollout._usage import (
        ProviderFailure,
        provider_refused_key_text,
    )

    runtime = SimpleNamespace(
        server=SimpleNamespace(
            route=SimpleNamespace(
                provider_name="deepseek",
                required_env=("BENCHFLOW_PROVIDER_API_KEY",),
            )
        )
    )
    head = provider_refused_key_text(
        ProviderFailure(401, "provider auth failed"), runtime
    )
    assert head.startswith("The model provider refused the key (HTTP 401): deepseek")
    # The user's variable, not the proxy's own copy (the first real run
    # named BENCHFLOW_PROVIDER_API_KEY).
    assert "for DEEPSEEK_API_KEY;" in head and "BENCHFLOW_" not in head
    wrapped = (
        f"{head}\noriginal error: agent integration failure [no_activity]: 0 tool calls"
    )
    assert classify_error(wrapped) == PROVIDER_AUTH


def test_gpqa_diamond_is_in_the_packaged_registry():
    from benchflow._utils.dataset_registry import PACKAGED_REGISTRY_SOURCE

    entries = json.loads(Path(PACKAGED_REGISTRY_SOURCE).read_text())
    (gpqa,) = [e for e in entries if e["name"] == "gpqa-diamond"]
    assert gpqa["version"] == "1.0"
    assert len(gpqa["tasks"]) == 198
    assert [t["name"] for t in gpqa["tasks"][:2]] == ["0", "1"]
    assert {t["git_commit_id"] for t in gpqa["tasks"]} == {
        "1983ac5c4d43f43cb7a9af9f89c54d09025589ec"
    }
    assert all(t["digest"].startswith("sha256:") for t in gpqa["tasks"])


@pytest.fixture
def registry(monkeypatch):
    import benchflow._utils.dataset_registry as reg

    entries = [
        {"name": "gpqa-diamond", "version": "1.0"},
        {"name": "terminal-bench", "version": "2.0"},
        {"name": "terminal-bench", "version": "4.1"},
    ]
    monkeypatch.setattr(reg, "load_benchflow_registry", lambda: (entries, []))


def test_a_bare_dataset_name_runs_its_one_version(registry, tmp_path, monkeypatch):
    from benchflow.cli.new_verbs import catalog_dataset

    monkeypatch.chdir(tmp_path)
    assert catalog_dataset("gpqa-diamond") == "gpqa-diamond@1.0"
    assert catalog_dataset("gpqa-diamond@1.0") == "gpqa-diamond@1.0"
    assert catalog_dataset("not-a-dataset") is None
    assert catalog_dataset("org/repo") is None
    (tmp_path / "gpqa-diamond").mkdir()
    assert catalog_dataset("gpqa-diamond") is None  # a folder here wins


def test_a_name_with_several_versions_asks_for_one(registry, tmp_path, monkeypatch):
    from click.exceptions import UsageError

    from benchflow.cli.new_verbs import catalog_dataset

    monkeypatch.chdir(tmp_path)
    with pytest.raises(UsageError, match="2 versions"):
        catalog_dataset("terminal-bench")
    with pytest.raises(UsageError, match="has no terminal-bench@9"):
        catalog_dataset("terminal-bench@9")


def test_the_live_panel_is_rows_not_a_bar_and_box():
    import io

    from rich.console import Console

    from benchflow.cli._live_progress import LiveEvalProgress

    console = Console(file=io.StringIO(), width=100, force_terminal=False)
    panel = LiveEvalProgress(
        console, label="state", agent="claude-agent-acp", model="m", sandbox="docker"
    )
    panel.on_plan(total=2, done=0, remaining=2)
    panel.on_task_start("0")
    panel.on_task_start("1")
    console.print(panel)
    text = console.file.getvalue()
    assert "0/2 done" in text and "2 running" in text
    assert "━" not in text and "┏" not in text and "│" not in text
    assert "queued" not in text
    assert len(text.encode()) < 600
