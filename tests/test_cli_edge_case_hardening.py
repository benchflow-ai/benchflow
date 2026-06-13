"""Regression tests for the v0.6 edge-case + UX sweep (round-2 hardening).

Each test pins one confirmed bug from the sweep: a raw traceback / wrong exit
code / silent corruption on a malformed-but-plausible input. They are grouped
by the command surface they protect.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from rich.console import Console
from typer.testing import CliRunner

from benchflow.cli.main import app

runner = CliRunner()


# ── continue: exit code (H1/H2) + concurrency floor (M4) ──────────────────────


def test_continue_exits_1_on_agent_error(tmp_path, monkeypatch):
    # H1/H2: a failed continuation must report failure to $? (it printed a green
    # ✓ then a yellow warning and exited 0, so scripts read it as success).
    import benchflow.continue_run.orchestrator as orch

    async def fake_continue_run(*args, **kwargs):
        return SimpleNamespace(
            rollout_dir=tmp_path / "r",
            n_recorded=1,
            n_live=0,
            divergences=0,
            rewards=None,
            error="agent boom",
        )

    monkeypatch.setattr(orch, "continue_run", fake_continue_run)
    res = runner.invoke(app, ["continue", str(tmp_path)])
    assert res.exit_code == 1
    assert "agent error" in res.output


def test_continue_exits_0_when_no_error(tmp_path, monkeypatch):
    # The success path must still exit 0 (guard against over-correcting H1).
    import benchflow.continue_run.orchestrator as orch

    async def fake_continue_run(*args, **kwargs):
        return SimpleNamespace(
            rollout_dir=tmp_path / "r",
            n_recorded=1,
            n_live=2,
            divergences=0,
            rewards={"reward": 1.0},
            error=None,
        )

    monkeypatch.setattr(orch, "continue_run", fake_continue_run)
    res = runner.invoke(app, ["continue", str(tmp_path)])
    assert res.exit_code == 0


def test_continue_batch_rejects_zero_concurrency(tmp_path):
    # M4: --concurrency 0 reached an unguarded asyncio.run -> raw ValueError.
    res = runner.invoke(app, ["continue-batch", str(tmp_path), "--concurrency", "0"])
    assert res.exit_code != 0
    assert not isinstance(res.exception, ValueError)  # typer rejects it, not a crash


# ── agent run: missing codex binary (H3) ─────────────────────────────────────


def test_subprocess_exec_missing_binary_raises_launch_error():
    # H3: a missing codex binary raised a raw FileNotFoundError that the CLI's
    # except (InvalidBenchmarkName, CodexLaunchError) did not catch.
    from benchflow.agent_router import CodexLaunchError, _subprocess_exec

    with pytest.raises(CodexLaunchError, match="codex binary not found"):
        _subprocess_exec(["/nonexistent/codex-xyz", "--help"], cwd=".", env={})


# ── tasks init: bad --dir (H4) ────────────────────────────────────────────────


def test_tasks_init_into_file_dir_exits_clean(tmp_path):
    # H4: `--dir <a file>` -> mkdir NotADirectoryError; init was the only
    # task subcommand that did not catch it.
    afile = tmp_path / "afile"
    afile.write_text("x")
    res = runner.invoke(app, ["tasks", "init", "mytask", "--dir", str(afile)])
    assert res.exit_code == 1
    assert isinstance(res.exception, SystemExit)  # clean Exit, not a raw traceback


# ── tasks generate --from-file: malformed trace files (H5) ────────────────────


def test_parse_claude_code_file_empty_returns_empty(tmp_path):
    from benchflow.traces.parsers import parse_claude_code_file

    p = tmp_path / "empty.jsonl"
    p.write_text("")
    assert parse_claude_code_file(p) == []


def test_parse_claude_code_file_non_object_lines_return_empty(tmp_path):
    # H5: list / scalar JSONL lines hit `entry.get(...)` -> AttributeError.
    from benchflow.traces.parsers import parse_claude_code_file

    p = tmp_path / "garbage.jsonl"
    p.write_text("[1, 2, 3]\n42\n")
    assert parse_claude_code_file(p) == []


def test_detect_format_non_dict_first_line(tmp_path):
    from benchflow.cli.trace_import import _detect_format

    p = tmp_path / "x.jsonl"
    p.write_text("[1, 2, 3]\n")
    assert _detect_format(p) == "claude-code"


# ── skills list: numeric / null frontmatter (H6) ─────────────────────────────


@pytest.mark.parametrize(
    "frontmatter, field, expected",
    [
        ("name: 123", "name", "123"),
        ("version: 1.0", "version", "1.0"),
        ("description: 42", "description", "42"),
        ("description: null", "description", ""),
    ],
)
def test_parse_skill_coerces_scalar_frontmatter(tmp_path, frontmatter, field, expected):
    # H6: unquoted numeric / explicit-null frontmatter rendered as non-strings
    # and crashed the whole `skills list` table (NotRenderableError / slicing).
    from benchflow.skills import parse_skill

    sk = tmp_path / "SKILL.md"
    sk.write_text(f"---\nname: ok\n{frontmatter}\n---\nbody")
    info = parse_skill(sk)
    assert info is not None
    assert getattr(info, field) == expected
    # The crash site was `description[:60]` — all fields must be sliceable str.
    assert info.description[:60] == info.description[:60]


# ── eval create --config: malformed YAML (H7) + prompts string-split (H8) ─────


def test_from_yaml_rejects_non_mapping(tmp_path):
    from benchflow.evaluation import Evaluation

    p = tmp_path / "c.yaml"
    p.write_text("- a\n- b\n")
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        Evaluation.from_yaml(p)


def test_from_yaml_rejects_empty_file(tmp_path):
    from benchflow.evaluation import Evaluation

    p = tmp_path / "e.yaml"
    p.write_text("")
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        Evaluation.from_yaml(p)


def test_from_yaml_rejects_non_string_repo(tmp_path):
    from benchflow.evaluation import Evaluation

    p = tmp_path / "r.yaml"
    p.write_text("source:\n  repo: 12345\n")
    with pytest.raises(ValueError, match=r"source\.repo"):
        Evaluation.from_yaml(p)


def test_yaml_string_prompts_are_wrapped_in_a_list(tmp_path):
    # H8: a bare `prompts: do the thing` string was iterated char-by-char into
    # one turn per character. It must be wrapped to a single-element list.
    from benchflow.evaluation import Evaluation

    tdir = tmp_path / "tasks"
    tdir.mkdir()
    p = tmp_path / "c.yaml"
    p.write_text(f"tasks_dir: {tdir}\nagent: gemini\nprompts: do the thing\n")
    ev = Evaluation.from_yaml(p)
    assert ev._config.prompts == ["do the thing"]


# ── markup in user input crashes error handlers (H10/M13) ─────────────────────


def test_hub_check_markup_registry_does_not_crash_handler(monkeypatch):
    import benchflow.hub.harbor_registry as hr

    def boom(*args, **kwargs):
        raise RuntimeError("bad registry /tmp/[/red]x.json")

    monkeypatch.setattr(hr, "check_harbor_registry", boom)
    res = runner.invoke(app, ["hub", "check", "--registry", "/tmp/[/red]x.json"])
    assert res.exit_code == 1
    # If the [red] markup were not escaped, the handler's own console.print
    # would raise MarkupError instead of exiting cleanly.
    assert isinstance(res.exception, SystemExit)
    assert "Harbor compatibility check failed" in res.output


# ── environment create: existing non-task dir (H11) ──────────────────────────


def test_environment_create_non_task_dir_exits_clean(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    res = runner.invoke(
        app, ["environment", "create", str(empty), "--sandbox", "docker"]
    )
    assert res.exit_code == 1
    assert "Not a valid task directory" in res.output


# ── live dashboard: markup in task names (H12/M11) ───────────────────────────


def test_dashboard_render_survives_markup_task_name():
    from benchflow.cli._live_progress import LiveEvalProgress

    d = LiveEvalProgress(Console(), label="x", agent="a", model="m", sandbox="docker")
    d.on_plan(total=1, done=0, remaining=1)
    d.on_task_start("task-[red]danger[/red]")
    d.__rich__()  # must not raise MarkupError


# ── eval list: corrupt summary.json (M1) + non-dir arg (M2) ──────────────────


def test_eval_list_corrupt_summary_does_not_crash(tmp_path):
    (tmp_path / "summary.json").write_text("{ not valid json")
    res = runner.invoke(app, ["eval", "list", str(tmp_path)])
    assert res.exit_code == 0
    assert "corrupt summary" in res.output


def test_eval_list_non_directory_exits_clean(tmp_path):
    afile = tmp_path / "afile"
    afile.write_text("x")
    res = runner.invoke(app, ["eval", "list", str(afile)])
    assert res.exit_code == 1
    assert "Not a directory" in res.output


# ── error handlers must escape user input echoed in their own message ─────────
# (review follow-up: the handlers this PR added/touched were themselves
# vulnerable to the MarkupError crash the PR fixes elsewhere)


def test_eval_config_handler_escapes_markup_in_path(tmp_path):
    # A --config path containing Rich markup + a non-mapping body: the
    # "Invalid eval config" handler must escape the path, not crash on it.
    bad = tmp_path / "[bad].yaml"
    bad.write_text("- a\n- b\n")
    res = runner.invoke(app, ["eval", "create", "--config", str(bad)])
    assert res.exit_code == 1
    assert isinstance(res.exception, SystemExit)  # not a MarkupError
    assert "Invalid eval config" in res.output


def test_tasks_init_handler_escapes_markup_in_path(tmp_path):
    # --dir is a file whose name contains markup → mkdir OSError whose message
    # echoes the path; the handler must escape it.
    afile = tmp_path / "[x].txt"
    afile.write_text("x")
    res = runner.invoke(app, ["tasks", "init", "mytask", "--dir", str(afile)])
    assert res.exit_code == 1
    assert isinstance(res.exception, SystemExit)  # not a MarkupError
