"""Regression tests for the v0.6 edge-case + UX sweep (round-2 hardening).

Each test pins one confirmed bug from the sweep: a raw traceback / wrong exit
code / silent corruption on a malformed-but-plausible input. They are grouped
by the command surface they protect.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
from rich.console import Console
from typer.testing import CliRunner

from benchflow.cli.main import app

runner = CliRunner()


# ── continue: exit code (H1/H2) + concurrency floor (M4) ──────────────────────


# Run the exit-code guards through BOTH spellings: the canonical
# `bench eval continue` and the deprecated top-level `bench continue` alias.
# Without the canonical case, a broken `eval_app` registration (PR #800) would
# slip through because the alias path would still pass.
_CONTINUE_SPELLINGS = pytest.mark.parametrize(
    "invoke", [["eval", "continue"], ["continue"]], ids=["eval", "alias"]
)


@_CONTINUE_SPELLINGS
def test_continue_exits_1_on_agent_error(invoke, tmp_path, monkeypatch):
    """Guards the continue exit-code contract (H1/H2: a failed continuation must
    report failure to $? — it printed a green ✓ then exited 0). Parametrized over
    both spellings after PR #800 moved the command under `bench eval continue`."""
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
    res = runner.invoke(app, [*invoke, str(tmp_path)])
    assert res.exit_code == 1
    assert "agent error" in res.output


@_CONTINUE_SPELLINGS
def test_continue_exits_0_when_no_error(invoke, tmp_path, monkeypatch):
    """Guards the continue success path (must still exit 0, against over-correcting
    H1), on both the canonical `bench eval continue` and the deprecated alias
    after PR #800."""
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
    res = runner.invoke(app, [*invoke, str(tmp_path)])
    assert res.exit_code == 0


@pytest.mark.parametrize(
    "invoke", [["eval", "continue-batch"], ["continue-batch"]], ids=["eval", "alias"]
)
def test_continue_batch_rejects_zero_concurrency(invoke, tmp_path):
    """Guards the M4 concurrency floor (--concurrency 0 reached an unguarded
    asyncio.run -> raw ValueError), on both the canonical `bench eval
    continue-batch` and the deprecated alias after PR #800."""
    res = runner.invoke(app, [*invoke, str(tmp_path), "--concurrency", "0"])
    assert res.exit_code != 0
    assert not isinstance(res.exception, ValueError)  # typer rejects it, not a crash


def test_continue_is_canonical_under_eval_group(tmp_path):
    """Guards PR #800: `continue` is canonical under the `eval` group.

    `bench eval continue` is the canonical spelling (visible in `bench eval
    --help`); the original top-level `bench continue` stays as a hidden,
    deprecated alias so existing scripts keep working.
    """
    eval_help = runner.invoke(app, ["eval", "--help"])
    assert eval_help.exit_code == 0
    assert "continue" in eval_help.output  # visible under the eval group

    # Both spellings resolve to the same command and fail the same way on an
    # empty (config-less) run folder.
    canonical = runner.invoke(app, ["eval", "continue", str(tmp_path)])
    alias = runner.invoke(app, ["continue", str(tmp_path)])
    assert canonical.exit_code == 1
    assert alias.exit_code == 1


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


# ── eval run --config: malformed YAML (H7) + prompts string-split (H8) ─────


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


def test_from_yaml_rejects_missing_tasks_dir_with_programbench_hint(tmp_path):
    """Guards the fix from PR #918 / Linear ENG-162: missing ProgramBench tasks fail cleanly."""
    from benchflow.evaluation import Evaluation

    p = tmp_path / "programbench.yaml"
    p.write_text("tasks_dir: .cache/programbench-benchflow\nagent: oracle\n")

    with pytest.raises(ValueError, match=r"run_programbench\.py"):
        Evaluation.from_yaml(p)


def test_eval_create_config_missing_tasks_dir_no_raw_traceback(tmp_path):
    """Guards the fix from PR #918 / Linear ENG-162 at the CLI config boundary."""
    p = tmp_path / "programbench.yaml"
    p.write_text("tasks_dir: .cache/programbench-benchflow\nagent: oracle\n")

    res = runner.invoke(app, ["eval", "create", "--config", str(p)])

    assert res.exit_code == 1
    assert "Traceback (most recent call last)" not in res.output
    normalized = " ".join(res.output.split())
    assert "YAML tasks_dir not found" in normalized
    assert "run_programbench.py" in res.output


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


# ── round-9 stress sweep: error stream, raw-traceback, markup, deprecation ────


def test_cli_errors_go_to_stderr_not_stdout(tmp_path):
    # print_error is the single CLI error sink; it must write to stderr so a
    # `bench … --json | jq` pipeline never gets a non-JSON line on stdout.
    res = runner.invoke(app, ["eval", "metrics", str(tmp_path / "nope"), "--json"])
    assert res.exit_code == 1
    assert res.stdout == ""  # JSON channel stays clean
    assert "Not a directory" in res.stderr


def test_eval_create_bad_source_repo_no_raw_traceback(monkeypatch):
    # A clone failure used to escape as a raw CalledProcessError traceback.
    import subprocess

    import benchflow._utils.benchmark_repos as br

    def boom(*a, **k):
        raise subprocess.CalledProcessError(128, ["git", "clone"])

    monkeypatch.setattr(br, "resolve_source_with_metadata", boom)
    res = runner.invoke(app, ["eval", "create", "--source-repo", "x/y"])
    assert res.exit_code == 1
    assert "Traceback (most recent call last)" not in res.output
    assert "Could not resolve --source-repo" in res.stderr


def test_eval_create_tasks_dir_is_a_file_clean_error(tmp_path):
    afile = tmp_path / "notadir.txt"
    afile.write_text("x")
    res = runner.invoke(
        app, ["eval", "create", "--tasks-dir", str(afile), "--agent", "oracle"]
    )
    assert res.exit_code == 1
    assert "Traceback (most recent call last)" not in res.output
    assert "Not a directory" in res.stderr


def test_hub_env_list_json_is_valid_json_even_when_narrow(monkeypatch):
    # The raw payload must be emitted verbatim, not through Rich's console (which
    # soft-wraps long strings and injects newlines mid-value → unparseable JSON).
    import benchflow.hosted_env as hosted

    long_desc = "calibration-as-action. " + "a cheap base model " * 30
    payload = json.dumps({"environments": [{"name": "a/b", "description": long_desc}]})
    monkeypatch.setattr(hosted, "prime_env_list", lambda **kw: payload)
    monkeypatch.setenv("COLUMNS", "40")
    res = runner.invoke(app, ["hub", "list", "--json"])
    assert res.exit_code == 0
    assert json.loads(res.stdout) == json.loads(payload)  # round-trips cleanly


def test_eval_metrics_markup_in_path_does_not_crash(tmp_path):
    # The full path string spans a closing-tag-shaped substring ("[/red]");
    # interpolated unescaped into the Rich Table title it raised MarkupError.
    d = tmp_path / "done[" / "red]"
    d.mkdir(parents=True)
    res = runner.invoke(app, ["eval", "metrics", str(d)])
    assert res.exit_code == 0
    assert "MarkupError" not in res.output
    assert "Traceback (most recent call last)" not in res.output


def test_skills_list_markup_in_metadata_does_not_crash(tmp_path):
    skill = tmp_path / "llm"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: prompt-helper\n"
        'description: "Wraps user text in [INST] ... [/INST] blocks"\n'
        'version: "1.0"\n---\nbody\n'
    )
    res = runner.invoke(app, ["skills", "list", "--dir", str(tmp_path)])
    assert res.exit_code == 0
    assert "MarkupError" not in res.output
    assert "Traceback (most recent call last)" not in res.output


def test_environment_alias_emits_exactly_one_deprecation_line():
    import benchflow.cli._shared as shared

    shared._DEPRECATION_WARNED.clear()
    res = runner.invoke(app, ["environment", "list"])
    # Exactly one project notice; NOT also Typer's generic per-call line.
    assert res.stderr.count("deprecation:") == 1
    assert "DeprecationWarning: The command" not in res.stderr


def test_environment_help_hides_aliased_verbs():
    res = runner.invoke(app, ["environment", "--help"])
    assert res.exit_code == 0
    assert "(deprecated)" not in res.output  # verbs hidden, like agent/eval adopt


def test_skills_eval_schema_error_hides_pydantic_internals(tmp_path):
    evals = tmp_path / "evals"
    evals.mkdir()
    (evals / "evals.json").write_text('{"cases": []}')
    res = runner.invoke(app, ["skills", "eval", str(tmp_path)])
    assert res.exit_code == 1
    assert "_EvalsJsonModel" not in res.output  # no private model class name
    assert "pydantic.dev" not in res.output  # no docs URL
    assert "cases" in res.output  # actionable per-field text kept


def test_viewer_renders_corrupt_artifacts_without_raising(tmp_path):
    # eval view's render must degrade, not dump a raw traceback, on partial data.
    from benchflow.trajectories.viewer import render_rollout

    # null session_id + a corrupt prompts.json (auxiliary → degrade to defaults)
    (tmp_path / "turn1.txt").write_text(
        '{"type":"system","session_id":null,"model":"m"}\n'
    )
    (tmp_path / "prompts.json").write_text("not json{{{")
    html = render_rollout(tmp_path)
    assert "<!DOCTYPE html>" in html

    # a truncated ACP trajectory line must be skipped, not crash
    acp = tmp_path / "acp"
    (acp / "trajectory").mkdir(parents=True)
    (acp / "trajectory" / "acp_trajectory.jsonl").write_text(
        '{"type":"user_message","text":"hi"}\nTRUNCATED-NOT-JSON'
    )
    (acp / "result.json").write_text("garbage{")  # corrupt → degrade to no metadata
    html2 = render_rollout(acp)
    assert isinstance(html2, str) and html2


@pytest.mark.parametrize(
    "argv, needle",
    [
        (["tasks", "generate"], "Specify a source"),
        (
            ["eval", "create", "--source-ref", "abc", "--tasks-dir", "."],
            "require --source-repo",
        ),
    ],
)
def test_arg_validation_errors_route_to_stderr(argv, needle):
    # The print_error sink is uniform: sibling arg-validation guards (not just the
    # ones the sweep happened to hit) put their error on stderr, leaving stdout clean.
    res = runner.invoke(app, argv)
    assert res.exit_code == 1
    assert needle in res.stderr
    assert needle not in res.stdout


def test_skills_eval_exits_nonzero_when_cases_error(tmp_path, monkeypatch):
    # A run where cases errored (e.g. missing credentials) must exit non-zero,
    # matching `eval run` — a 100%-error run printed `0/1` but exited 0, so CI
    # read a total failure as success.
    from benchflow.skill_eval._core import CaseResult, SkillEvalResult, SkillEvaluator

    evals = tmp_path / "evals"
    evals.mkdir()
    (evals / "evals.json").write_text(
        '{"skill_name":"s","cases":[{"id":"c","question":"q",'
        '"ground_truth":"g","expected_behavior":["x"]}]}'
    )

    async def fake_run(self, **kwargs):
        return SkillEvalResult(
            skill_name="s",
            n_cases=1,
            agents=["oracle"],
            case_results=[
                CaseResult(
                    case_id="c",
                    agent="oracle",
                    model="",
                    with_skill=True,
                    reward=None,
                    error="boom",
                )
            ],
        )

    monkeypatch.setattr(SkillEvaluator, "run", fake_run)
    res = runner.invoke(app, ["skills", "eval", str(tmp_path), "--agent", "oracle"])
    assert res.exit_code == 1
    assert "errored" in res.stderr


def test_skills_eval_rejects_model_agent_length_mismatch_before_header(tmp_path):
    """Guards PR #650 against #550's raw model/agent cardinality failure."""
    evals = tmp_path / "evals"
    evals.mkdir()
    (evals / "evals.json").write_text(
        '{"skill_name":"citation-management",'
        '"cases":[{"id":"case-1","question":"Q?","ground_truth":"A"}]}'
    )

    res = runner.invoke(
        app,
        [
            "skills",
            "eval",
            str(tmp_path),
            "--agent",
            "gemini",
            "--model",
            "gemini-2.5-flash",
            "--model",
            "extra-model",
            "--no-baseline",
            "--jobs-dir",
            str(tmp_path / "jobs"),
        ],
    )

    assert res.exit_code == 1
    assert (
        "--model may be provided once for all agents or once per --agent" in res.stderr
    )
    assert "got 2 models" in res.stderr
    assert "for 1 agents" in res.stderr
    assert "Skill eval:" not in res.stdout
    assert "Traceback" not in res.output


def test_viewer_job_dir_indexes_rollout_subdirs(tmp_path):
    # `eval view <job_dir>` used to render a blank "No trajectory files found"
    # (the natural value from create's "Artifacts:" line); now it indexes the
    # rollout subdirectories so the user knows to drill in.
    from benchflow.trajectories.viewer import render_rollout

    rollout = tmp_path / "task-a__trial-1"
    rollout.mkdir()
    (rollout / "turn1.txt").write_text(
        '{"type":"system","session_id":"s","model":"m"}\n'
    )
    html_out = render_rollout(tmp_path)
    assert "job directory" in html_out
    assert "task-a__trial-1" in html_out
    assert "No trajectory files found" not in html_out


# ── round-3 sweep: error-handling defects found by the CLI audit workflow ─────


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="chmod 000 does not restrict root, so the unreadable-dir probe is moot",
)
def test_tasks_digest_skips_unreadable_subdir_without_traceback(tmp_path):
    """Guards PR #789 (CLI error-handling hardening)."""
    # P0: the directory scan stat'd task.toml inside an unreadable subdir and
    # dumped a raw PermissionError traceback. It must skip what it cannot stat.
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "task.toml").write_text("")
    locked.chmod(0o000)
    try:
        res = runner.invoke(app, ["tasks", "digest", str(tmp_path)])
    finally:
        locked.chmod(0o755)  # restore so pytest tmp cleanup can recurse in
    assert res.exit_code == 1
    assert not isinstance(res.exception, OSError), res.exception
    assert "No tasks under" in res.output


def test_tasks_generate_from_file_directory_is_clean_error(tmp_path):
    """Guards PR #789 (CLI error-handling hardening)."""
    # P0: a directory passed exists() then raised IsADirectoryError in the format
    # sniff's path.open(). It must be rejected with a clean message.
    res = runner.invoke(
        app, ["tasks", "generate", "--from-file", str(tmp_path), "--dry-run"]
    )
    assert res.exit_code == 1
    assert not isinstance(res.exception, OSError), res.exception
    assert "Not a file" in res.output


def test_tasks_generate_rejects_invalid_outcome(tmp_path):
    """Guards PR #789 (CLI error-handling hardening)."""
    # P3: a bogus --outcome silently matched nothing and exited 0; it must be
    # rejected against the advertised choice set.
    trace = tmp_path / "t.jsonl"
    trace.write_text('{"foo":"bar"}\n')
    res = runner.invoke(
        app,
        [
            "tasks",
            "generate",
            "--from-file",
            str(trace),
            "--outcome",
            "zzz",
            "--dry-run",
        ],
    )
    assert res.exit_code == 1
    assert "Invalid --outcome" in res.output


def test_continue_batch_rejects_nonexistent_root(tmp_path):
    """Guards PR #789 (CLI error-handling hardening)."""
    # P1: a typo'd ROOT exited 0 "No timeout run folders found." (silent success).
    res = runner.invoke(app, ["continue-batch", str(tmp_path / "nope-12345")])
    assert res.exit_code == 1
    assert "does not exist" in res.output


def test_continue_batch_rejects_file_root(tmp_path):
    """Guards PR #789 (CLI error-handling hardening)."""
    # P1: a regular-file ROOT also exited 0; it must fail like `bench continue`.
    f = tmp_path / "a-file.txt"
    f.write_text("x")
    res = runner.invoke(app, ["continue-batch", str(f)])
    assert res.exit_code == 1
    assert "not a directory" in res.output


def test_environment_create_empty_dir_names_task_md(tmp_path):
    """Guards PR #789 (CLI error-handling hardening)."""
    # P2: the error named the legacy instruction.md; it must name the formats the
    # author can actually create (task.md / task.toml).
    empty = tmp_path / "empty"
    empty.mkdir()
    res = runner.invoke(
        app, ["environment", "create", str(empty), "--sandbox", "docker"]
    )
    assert res.exit_code == 1
    assert "task.md" in res.output


def test_eval_view_empty_dir_fails_fast_without_writing(tmp_path):
    """Guards PR #789 (CLI error-handling hardening)."""
    # P3: `eval view` wrote a blank trajectory.html into an unrelated dir and
    # started a server; an empty dir must fail fast and write nothing.
    res = runner.invoke(app, ["eval", "view", str(tmp_path), "--port", "0"])
    assert res.exit_code == 1
    assert "No trajectories found" in res.output
    assert not (tmp_path / "trajectory.html").exists()


def test_print_error_echoes_colon_tokens_verbatim_no_emoji(monkeypatch):
    """Guards PR #789 (CLI error-handling hardening)."""
    # P3: print_error rendered user input through Rich with emoji=True, so a
    # hosted-env ref like primeintellect:a:b had :a: swapped for an emoji. The
    # echoed value must be literal.
    import io

    import benchflow.cli._shared as shared

    rec = Console(file=io.StringIO(), width=200)
    monkeypatch.setattr(shared, "err_console", rec)
    shared.print_error("Invalid hosted environment reference: primeintellect:a:b")
    out = rec.file.getvalue()
    assert "primeintellect:a:b" in out


# ── round-4 sweep: deep fix-hunt regressions across the merged CLI ─────────────

_MIN_TRACE = (
    '{"type":"user","sessionId":"s1","message":{"role":"user","content":"x"}}\n'
    '{"type":"assistant","sessionId":"s1","message":'
    '{"role":"assistant","content":[{"type":"text","text":"y"}]}}\n'
)


def test_tasks_normalize_output_is_dir_clean_error(tmp_path):
    """Guards PR #789 (CLI error-handling hardening)."""
    # P0: --output an existing directory raised a raw IsADirectoryError.
    assert (
        runner.invoke(
            app, ["tasks", "init", "demotask", "--dir", str(tmp_path)]
        ).exit_code
        == 0
    )
    outdir = tmp_path / "outdir"
    outdir.mkdir()
    res = runner.invoke(
        app, ["tasks", "normalize", str(tmp_path / "demotask"), "--output", str(outdir)]
    )
    assert res.exit_code == 1
    assert not isinstance(res.exception, OSError), res.exception
    assert "directory" in res.output.lower()


def test_skills_eval_evals_json_as_dir_clean_error(tmp_path):
    """Guards PR #789 (CLI error-handling hardening)."""
    # P0: evals/evals.json being a directory raised a raw IsADirectoryError.
    (tmp_path / "evals" / "evals.json").mkdir(parents=True)
    res = runner.invoke(app, ["skills", "eval", str(tmp_path)])
    assert res.exit_code == 1
    assert not isinstance(res.exception, OSError), res.exception
    assert "No evals/evals.json found" in res.output


def test_sandbox_create_task_file_as_dir_clean_error(tmp_path):
    """Guards PR #789 (CLI error-handling hardening)."""
    # P0: task.md itself being a directory raised a raw IsADirectoryError.
    (tmp_path / "task.md").mkdir()
    res = runner.invoke(
        app, ["sandbox", "create", str(tmp_path), "--sandbox", "docker"]
    )
    assert res.exit_code == 1
    assert not isinstance(res.exception, OSError), res.exception
    assert "Not a valid task directory" in res.output


def test_tasks_generate_output_is_file_clean_error(tmp_path):
    """Guards PR #789 (CLI error-handling hardening)."""
    # P0: --output an existing file raised a raw NotADirectoryError.
    trace = tmp_path / "t.jsonl"
    trace.write_text(_MIN_TRACE)
    outfile = tmp_path / "outfile"
    outfile.write_text("x")
    res = runner.invoke(
        app, ["tasks", "generate", "--from-file", str(trace), "--output", str(outfile)]
    )
    assert res.exit_code == 1
    assert not isinstance(res.exception, OSError), res.exception
    assert "is not a directory" in res.output


def test_tasks_generate_zero_results_exits_nonzero(tmp_path):
    """Guards PR #789 (CLI error-handling hardening)."""
    # P2: generating 0 tasks (everything filtered) used to print a green
    # "Generated 0 tasks" and exit 0 — a silent no-op success.
    trace = tmp_path / "t.jsonl"
    trace.write_text(_MIN_TRACE)
    res = runner.invoke(
        app,
        [
            "tasks",
            "generate",
            "--from-file",
            str(trace),
            "--min-steps",
            "999",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    assert res.exit_code == 1
    assert "No tasks generated" in res.output


def test_eval_list_explicit_nonexistent_dir_exits_nonzero(tmp_path):
    """Guards PR #789 (CLI error-handling hardening)."""
    # P3: a typo'd explicit jobs-dir exited 0 (silent) while `eval metrics` exits 1.
    res = runner.invoke(app, ["eval", "list", str(tmp_path / "nope-xyz")])
    assert res.exit_code == 1
    assert "No such jobs directory" in res.output
