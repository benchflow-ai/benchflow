"""Guard FF-TEST-ENV-C7/C8 at 727eab15 at the actual child output boundary."""

from __future__ import annotations

import ast
import json
import shlex
import sys
from pathlib import Path
from urllib.parse import quote, quote_plus

import pytest

from benchflow import doctor, doctor_smoke
from tests.test_cli_credential_diagnostics import (
    isolated_credentials as isolated_credentials,
)
from tests.test_doctor_smoke import _job


@pytest.mark.parametrize("sandbox", ["docker", "daytona"])
@pytest.mark.parametrize(
    "route",
    [
        "rollout",
        "verifier",
        "retry",
        "retry-verifier",
        "resume-solver",
        "resume-retry",
        "resume-reuse",
    ],
)
@pytest.mark.parametrize(
    "spelling", ["raw", "shell", "json", "repr", "url", "url-plus"]
)
def test_actual_evaluation_logs(route, spelling, sandbox, monkeypatch, tmp_path):
    """Guard 727eab15 C7 logs and cfca0513's resume-loader keyword contract."""
    secret = "synthetic-first " + "X" * 256 + "'\"☃\\synthetic-last"
    values = {
        "raw": secret,
        "shell": shlex.quote(secret),
        "json": json.dumps(secret)[1:-1],
        "repr": repr(secret)[1:-1],
        "url": quote(secret, safe=""),
        "url-plus": quote_plus(secret, safe=""),
    }
    monkeypatch.setenv("DUMMY_TOKEN", secret)
    script = (
        """
import asyncio, logging
from pathlib import Path
from types import SimpleNamespace as NS
import benchflow.evaluation as module
from benchflow.evaluation import Evaluation
logging.basicConfig(level=logging.INFO, format='%(message)s')
route, error, job_dir = ROUTE, ERROR, Path(JOB_DIR)
job_dir.mkdir()
retry = NS(max_retries=1, should_retry=lambda *a, **kw: route == 'retry', should_retry_verifier_error=lambda *a: route != 'resume-reuse', reruns_verifier_error_on_resume=lambda *a: route != 'resume-reuse', reruns_unjudged_solver=lambda *a, **kw: True, backoff_delay=lambda *a: 0)
e = Evaluation.__new__(Evaluation)
e._config = NS(retry=retry, job_mode='independent')
e._budget_guard = None
e._on_result = None
r = NS(error=error, verifier_error=None, scoring=None, error_category='acp_error', rewards=None, score_outcome='errored', n_tool_calls=0)
if route in ('verifier', 'retry-verifier'):
    r.error, r.verifier_error = None, error
if route in ('rollout', 'verifier'):
    e._log_and_report(Path('hello'), r)
elif route.startswith('retry'):
    async def run(*args): return r
    async def sweep(): pass
    e._run_single_task, e._sweep_docker = run, sweep
    asyncio.run(e._run_task(Path('hello')))
else:
    e._jobs_dir, e._job_name = job_dir.parent, job_dir.name
    record = {'error': error, 'scoring': {}} if route == 'resume-solver' else {'verifier_error': error, 'rewards': None}
    def load_task_results(root, *, include_terminal_errors=False):
        assert root == job_dir
        assert include_terminal_errors is False
        return {'hello': record}
    module.load_task_results = load_task_results
    module._scoring_block = lambda *args: None
    e._get_completed_tasks()
""".replace("ROUTE", repr(route))
        .replace("ERROR", repr("ACP error -32603: " + values[spelling]))
        .replace("JOB_DIR", repr(str(tmp_path / "eval-job")))
    )
    monkeypatch.setenv("PYTHONPATH", str(Path(doctor.__file__).parents[1]))
    monkeypatch.setattr(
        doctor_smoke, "_bench_argv", lambda args: [sys.executable, "-c", script]
    )
    original = _job(tmp_path)
    job = doctor_smoke.SmokeJob(
        original.target,
        original.task_dir,
        original.jobs_dir,
        original.log_path,
        sandbox,
        original.timeout_sec,
    )
    outcome = doctor_smoke.subprocess_runner(job)
    shown = job.log_path.read_text()
    assert outcome.exit_code == 0, shown
    assert "synthetic-first" not in shown
    assert "***" in shown
    assert "ACP error -32603: ***" in shown


def _unsafe_previews(source, *, diagnostic_module=True):
    """Small source guard: raw truncators cannot appear outside the renderer."""
    tree = ast.parse(source)

    def redacted_input(value):
        if isinstance(value, ast.Call):
            return ast.unparse(value.func) in {
                "redact",
                "render_text",
                "render_diagnostic",
            }
        if not isinstance(value, ast.Name):
            return False
        writes = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Name)
            and isinstance(n.ctx, ast.Store)
            and n.id == value.id
        ]
        if len(writes) != 1:
            return False
        return any(
            isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and n.targets[0] is writes[0]
            and n.lineno < value.lineno
            and isinstance(n.value, ast.Call)
            and ast.unparse(n.value.func)
            in {"redact", "render_text", "render_diagnostic"}
            for n in ast.walk(tree)
        )

    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func).split(".")[-1]
            if (
                name == "truncate_end" or (diagnostic_module and "preview" in name)
            ) and (not node.args or not redacted_input(node.args[0])):
                bad.append(node.lineno)
        if diagnostic_module and isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                isinstance(t, ast.Name) and t.id.endswith("_preview") for t in targets
            ) and (
                not isinstance(node.value, ast.Call)
                or ast.unparse(node.value.func) != "render_text"
            ):
                bad.append(node.lineno)
    return bad


def test_previews_are_constructed_by_the_renderer():
    """727eab15 C7: guard every production truncate call and preview assignment."""
    root = Path(doctor.__file__).parent
    problems = {}
    for path in root.rglob("*.py"):
        source = path.read_text()
        bad = _unsafe_previews(
            source,
            diagnostic_module=path.relative_to(root).as_posix()
            in {
                "_diagnostics.py",
                "evaluation.py",
                "eval_artifacts.py",
                "doctor.py",
                "doctor_smoke.py",
                "failures.py",
                "cli/_errors.py",
                "cli/_shared.py",
                "cli/_failure_evidence.py",
            },
        )
        # Only the shared boundary may import or call the raw text truncator.
        if path.name != "_diagnostics.py" and any(
            (
                isinstance(n, ast.ImportFrom)
                and any(a.name == "truncate_end" for a in n.names)
            )
            or (
                isinstance(n, ast.Call)
                and ast.unparse(n.func).split(".")[-1] == "truncate_end"
            )
            for n in ast.walk(ast.parse(source))
        ):
            bad.append(0)
        if bad:
            problems[str(path.relative_to(root))] = bad
    assert not problems
    boundary = (root / "_diagnostics.py").read_text()
    assert "redacted = redact(text, _secrets())" in boundary
    assert _unsafe_previews("truncate_end(raw, 50)")
    assert _unsafe_previews("preview(raw)")
    assert _unsafe_previews("error_preview(raw, 50)")
    assert not _unsafe_previews("preview(redact(raw, secrets))")
    assert _unsafe_previews("redacted = raw\ntruncate_end(redacted, 50)")
    assert _unsafe_previews(
        "redacted = redact(raw, secrets)\nredacted = raw\ntruncate_end(redacted, 50)"
    )
    assert not _unsafe_previews(
        "redacted = redact(raw, secrets)\ntruncate_end(redacted, 50)"
    )
    assert _unsafe_previews("err_preview = raw[:50]")
    assert _unsafe_previews("err_preview = truncate_end(raw, 50)")
    assert not _unsafe_previews("err_preview = render_text(raw, 50)")


@pytest.mark.parametrize("width", [0, 1, 2, 3, 50, 60, 80, 120, 200, 500])
@pytest.mark.parametrize(
    "spelling", ["raw", "shell", "json", "repr", "url", "url-plus"]
)
def test_render_before_every_preview_width(width, spelling, monkeypatch):
    """727eab15 C7: credentials exceed every diagnostic preview width."""
    from benchflow._diagnostics import render_text
    from benchflow._utils.text import truncate_end

    secret = "synthetic-first " + "X" * 1000 + "'\"☃\\last"
    monkeypatch.setenv("DUMMY_TOKEN", secret)
    values = {
        "raw": secret,
        "shell": shlex.quote(secret),
        "json": json.dumps(secret)[1:-1],
        "repr": repr(secret)[1:-1],
        "url": quote(secret, safe=""),
        "url-plus": quote_plus(secret, safe=""),
    }
    assert render_text("error " + values[spelling] + " safe", width) == truncate_end(
        "error *** safe", width
    )


@pytest.mark.parametrize("source", ["setting", "markdown-error", "markdown-verifier"])
def test_sibling_summary_previews(source, monkeypatch, tmp_path):
    """727eab15 C7: settings and Markdown must redact before repr/escaping/cuts."""
    from benchflow import eval_artifacts
    from benchflow.evaluation import _shown_setting

    secret = "synthetic-first |\n" + "X" * 500 + "synthetic-last"
    monkeypatch.setenv("DUMMY_TOKEN", secret)
    if source == "setting":
        shown = _shown_setting(secret)
    else:
        row = {
            "task_id": "hello",
            "reward": None,
            "scored": False,
            "tool_calls": 0,
            "error" if source == "markdown-error" else "verifier_error": secret,
        }
        monkeypatch.setattr(eval_artifacts, "canonical_task_rows", lambda job: [row])
        shown = eval_artifacts.render_run_summary_markdown(tmp_path)
    assert "synthetic-first" not in shown
    assert "synthetic-last" not in shown
    assert "***" in shown
