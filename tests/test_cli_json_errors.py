"""With ``--json``, an error is one JSON object on stdout and nothing else is.

Rust's ``bench`` prints ``{error, code, owner, next, detail, exit_code}`` on
stdout for every error under ``--json`` (docs/rust/cli.md, "JSON output");
before round PY8's D2, Python's printed the error on stderr only, so a
``bench ... --json | jq`` pipeline read an empty stdout. D1's unknown
command and flag errors take their text and next step from cli/verbs.py, and
so does their JSON form.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import typer

from benchflow.cli import verbs
from benchflow.cli._errors import run_cli
from benchflow.cli._shared import print_error
from benchflow.errors import MissingCredentialError

KEYS = ["error", "code", "owner", "next", "detail", "exit_code"]
SECRET = "sk-test-0123456789abcdefghijklmnop"

# The real console script, with `bench` as its program name.
_LAUNCH = (
    "import sys; sys.argv[0] = 'bench'; from benchflow.cli.main import main; main()"
)


def _bench(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "NO_COLOR": "1",
        "COLUMNS": "100",
        "BENCHFLOW_LOG_DIR": str(tmp_path / "logs"),
    }
    return subprocess.run(
        [sys.executable, "-c", _LAUNCH, *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )


def _one_object(stdout: str) -> dict[str, Any]:
    """stdout is exactly one JSON object with the six keys, in order."""
    lines = stdout.splitlines()
    assert len(lines) == 1, stdout
    obj = json.loads(lines[0])
    assert list(obj) == KEYS, obj
    return obj


# --- D1's usage errors, through the real console script -----------------------


def test_an_unknown_flag_is_one_object_with_exit_2(tmp_path):
    done = _bench(tmp_path, "eval", "view", "--bogus", "--json")
    assert done.returncode == 2, done.stderr
    obj = _one_object(done.stdout)
    assert "--bogus" in obj["error"]
    assert obj["next"] == verbs.unknown_flag_next("bench eval view", "--bogus")
    assert obj["owner"] == "you" and obj["exit_code"] == 2 and obj["code"] is None
    assert "--bogus" not in done.stderr  # the error is printed once, as JSON


def test_an_unknown_run_flag_points_at_every_run_flag(tmp_path):
    done = _bench(tmp_path, "run", "--frobnicate", "x", "--json")
    assert done.returncode == 2, done.stderr
    obj = _one_object(done.stdout)
    assert obj["next"] == verbs.unknown_flag_next("bench run", "--frobnicate")
    assert "--help-all" in obj["next"]


def test_an_unknown_command_is_one_object_with_exit_2(tmp_path):
    done = _bench(tmp_path, "frobnicate", "--json")
    assert done.returncode == 2, done.stderr
    obj = _one_object(done.stdout)
    assert "frobnicate" in obj["error"]
    assert obj["next"] == verbs.unknown_command_next("bench")
    assert obj["exit_code"] == 2


def test_a_mistyped_word_gets_d1s_message_and_next_step(tmp_path):
    done = _bench(tmp_path, "compare", "--json")
    assert done.returncode == 2, done.stderr
    obj = _one_object(done.stdout)
    known = verbs.known_word("compare")
    assert known is not None
    assert (obj["error"], obj["next"]) == known


def test_an_unknown_command_in_a_group_names_the_groups_help(tmp_path):
    done = _bench(tmp_path, "eval", "frobnicate", "--json")
    assert done.returncode == 2, done.stderr
    obj = _one_object(done.stdout)
    assert obj["next"] == verbs.unknown_command_next("bench eval")


def test_a_08_group_typed_alone_puts_its_map_in_detail(tmp_path):
    done = _bench(tmp_path, "eval", "--json")
    assert done.returncode == 2, done.stderr
    obj = _one_object(done.stdout)
    assert obj["error"].startswith("`bench eval` was BenchFlow 0.8's")
    assert obj["detail"] == verbs.group_map("eval").rstrip("\n")
    assert obj["exit_code"] == 2
    assert "Where its commands went" not in done.stderr


def test_a_missing_argument_names_the_commands_help(tmp_path):
    done = _bench(tmp_path, "eval", "inspect", "--json")
    assert done.returncode == 2, done.stderr
    obj = _one_object(done.stdout)
    assert obj["next"] == "`bench eval inspect --help` lists what it takes"


def test_without_json_the_usage_error_stays_on_stderr(tmp_path):
    done = _bench(tmp_path, "eval", "view", "--bogus")
    assert done.returncode == 2
    assert done.stdout == ""
    assert "next: `bench eval view --help` lists its flags" in " ".join(
        done.stderr.split()
    )


@pytest.mark.parametrize("verb", [["regrade"], ["eval", "regrade"]])
def test_a_regrade_of_a_missing_folder_is_one_object(tmp_path, verb):
    # Regrade printed its failure on stdout, which read as the command's own
    # JSON: stdout held one plain-text line and no object (D2-R1-A-1).
    missing = tmp_path / "no-such-job"
    done = _bench(tmp_path, *verb, str(missing), "--json")
    assert done.returncode == 2, done.stderr
    obj = _one_object(done.stdout)
    assert obj["error"] == f"No such file or directory: {missing}"
    assert obj["exit_code"] == 2
    assert "No such file or directory" in done.stderr


def test_json_after_a_double_dash_is_an_argument():
    from benchflow.cli._errors import json_requested

    assert json_requested(["run", "--json"])
    assert not json_requested(["run", "--", "--json"])
    assert not json_requested(["run", "--jsonx"])


# --- errors a command raises or prints ----------------------------------------


def _app(action) -> typer.Typer:
    app = typer.Typer()

    @app.command()
    def go(
        output_json: bool = typer.Option(False, "--json"),
    ) -> None:
        action()

    @app.command()
    def other() -> None:
        pass

    return app


def _run(app: typer.Typer, argv: list[str]) -> int:
    with pytest.raises(SystemExit) as caught:
        run_cli(app, argv)
    return int(caught.value.code or 0)


def test_an_expected_error_is_its_message_and_next_step(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_LOG_DIR", str(tmp_path / "logs"))

    def fail() -> None:
        raise MissingCredentialError(
            "ANTHROPIC_API_KEY required for model 'm' but not set.",
            hint="run `claude setup-token`",
        )

    assert _run(_app(fail), ["go", "--json"]) == 1
    out, err = capsys.readouterr()
    obj = _one_object(out)
    assert obj == {
        "error": "ANTHROPIC_API_KEY required for model 'm' but not set.",
        "code": None,
        "owner": "you",
        "next": "run `claude setup-token`",
        "detail": None,
        "exit_code": 1,
    }
    assert "ANTHROPIC_API_KEY" not in err
    assert not (tmp_path / "logs").exists()


def test_a_missing_file_is_the_users_problem(capsys):
    def fail() -> None:
        raise FileNotFoundError(2, "No such file or directory", "/tmp/victim-a/x.toml")

    assert _run(_app(fail), ["go", "--json"]) == 1
    obj = _one_object(capsys.readouterr().out)
    assert obj["error"] == "No such file or directory: /tmp/victim-a/x.toml"
    assert obj["owner"] == "you" and obj["exit_code"] == 1


def test_a_bug_is_benchflows_with_the_log_and_the_traceback_on_stderr(
    capsys, tmp_path, monkeypatch
):
    monkeypatch.setenv("BENCHFLOW_LOG_DIR", str(tmp_path / "logs"))

    def fail() -> None:
        raise RuntimeError("boom")

    assert _run(_app(fail), ["go", "--json"]) == 1
    out, err = capsys.readouterr()
    obj = _one_object(out)
    assert obj["error"] == "unexpected error (RuntimeError): boom"
    assert obj["detail"] is None
    assert obj["owner"] == "benchflow" and obj["exit_code"] == 1
    logs = list((tmp_path / "logs").iterdir())
    assert len(logs) == 1 and str(logs[0]) in obj["next"]
    assert "Traceback" in err and "RuntimeError: boom" in err


def test_a_printed_error_then_an_exit_is_the_object(capsys):
    def fail() -> None:
        print_error("Not a directory: /tmp/victim-a", next_step="check the path")
        raise typer.Exit(3)

    assert _run(_app(fail), ["go", "--json"]) == 3
    out, err = capsys.readouterr()
    obj = _one_object(out)
    assert obj["error"] == "Not a directory: /tmp/victim-a"
    assert obj["next"] == "check the path"
    assert obj["exit_code"] == 3
    assert "Not a directory" in err  # print_error still writes stderr


def test_a_printed_usage_error_is_the_users_problem(capsys):
    def fail() -> None:
        print_error("no trial under /tmp/victim-a")
        raise typer.Exit(2)

    assert _run(_app(fail), ["go", "--json"]) == 2
    obj = _one_object(capsys.readouterr().out)
    assert obj["error"] == "no trial under /tmp/victim-a"
    assert obj["owner"] == "you" and obj["next"] is None


def test_a_command_that_printed_its_own_json_keeps_it_alone(capsys):
    def report_and_fail() -> None:
        typer.echo(json.dumps({"ok": False}))
        raise typer.Exit(1)

    assert _run(_app(report_and_fail), ["go", "--json"]) == 1
    assert json.loads(capsys.readouterr().out) == {"ok": False}


def test_coloured_json_of_its_own_is_still_its_own(capsys):
    def report_and_fail() -> None:
        # typer.echo drops colour codes off a terminal; rich forced to colour
        # does not.
        sys.stdout.write('\x1b[1m{\x1b[0m"ok": false}\n')
        raise typer.Exit(1)

    assert _run(_app(report_and_fail), ["go", "--json"]) == 1
    assert capsys.readouterr().out == '\x1b[1m{\x1b[0m"ok": false}\n'


def test_plain_text_on_stdout_does_not_hide_the_error(capsys):
    # Any write to stdout used to read as the command's own JSON, so a
    # command that printed its failure on stdout gave no object (D2-R1-A-1).
    def fail() -> None:
        typer.echo("Something failed: /tmp/victim-a")
        raise typer.Exit(1)

    assert _run(_app(fail), ["go", "--json"]) == 1
    text, _, last = capsys.readouterr().out.rstrip("\n").rpartition("\n")
    assert text == "Something failed: /tmp/victim-a"
    obj = json.loads(last)
    assert list(obj) == KEYS
    assert obj["error"] == "the command failed with exit code 1"
    assert obj["exit_code"] == 1


def test_a_silent_failure_is_still_one_object(capsys):
    def fail() -> None:
        raise typer.Exit(4)

    assert _run(_app(fail), ["go", "--json"]) == 4
    obj = _one_object(capsys.readouterr().out)
    assert obj["error"] == "the command failed with exit code 4"
    assert obj["exit_code"] == 4


def test_success_prints_only_the_commands_output(capsys):
    def ok() -> None:
        typer.echo('{"fine": true}')

    assert _run(_app(ok), ["go", "--json"]) == 0
    assert capsys.readouterr().out == '{"fine": true}\n'


def test_a_secret_in_an_error_is_masked(capsys, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)

    def fail() -> None:
        raise MissingCredentialError(f"the key {SECRET} was rejected")

    assert _run(_app(fail), ["go", "--json"]) == 1
    out = capsys.readouterr().out
    assert SECRET not in out
    assert "***" in _one_object(out)["error"]


def test_a_usage_error_of_a_plain_app_is_one_object(capsys):
    assert _run(_app(lambda: None), ["go", "--json", "--nope"]) == 2
    obj = _one_object(capsys.readouterr().out)
    assert "--nope" in obj["error"]
    assert obj["owner"] == "you" and obj["exit_code"] == 2
    assert obj["next"].endswith("--help` lists what it takes")


# --- the click classes of the installed typer (D2b-R2-B-2) ---------------------


def _typer_027_exceptions() -> Any:
    """``typer._click.exceptions`` as typer 0.27 ships it: no ``Abort``.

    typer 0.27.3 (what ``pip install`` resolves under ``typer>=0.9``) moved
    ``Abort`` to ``typer.exceptions``; uv.lock's typer 0.24.1 has no
    ``typer._click`` at all, so CI never saw this layout.
    """
    import types

    import click

    fake = types.ModuleType("typer._click.exceptions")
    fake.ClickException = click.ClickException  # type: ignore[attr-defined]
    fake.UsageError = click.UsageError  # type: ignore[attr-defined]
    return fake


def test_the_click_classes_of_the_installed_typer_are_exception_classes():
    from benchflow.cli._errors import _click_classes

    for cls in _click_classes():
        assert isinstance(cls, type) and issubclass(cls, BaseException), cls
    # What `typer.Abort()` raises is what the --json run catches as an abort.
    assert issubclass(typer.Abort, _click_classes()[2])


def test_a_typer_without_click_abort_still_runs_json_commands(capsys, monkeypatch):
    """D2b-R2-B-2: with typer 0.27, every --json command crashed.

    ``_click_classes`` read ``Abort`` off ``typer._click.exceptions``, which
    typer 0.27 does not have: ``bench agents list --json`` exited 1 with an
    AttributeError before the command ran.
    """
    from benchflow.cli._errors import _click_classes

    monkeypatch.setitem(sys.modules, "typer._click.exceptions", _typer_027_exceptions())
    assert _click_classes()[2] is typer.Abort

    def ok() -> None:
        typer.echo('{"fine": true}')

    assert _run(_app(ok), ["go", "--json"]) == 0
    assert capsys.readouterr().out == '{"fine": true}\n'

    def aborted() -> None:
        raise typer.Abort()

    assert _run(_app(aborted), ["go", "--json"]) == 1
    assert _one_object(capsys.readouterr().out)["error"] == "aborted"


# A credential with a newline in it (synthetic): split into error and detail
# before masking, neither half matched the whole value.
MULTILINE_SECRET = "synthetic-D2-first\nsynthetic-D2-second"


# The same with whitespace around it: an error's text is stripped before it is
# masked (error_parts), so a newline-terminated value no longer matched as a
# whole and both lines leaked (FF-CLI-C1).
@pytest.mark.parametrize(
    "secret",
    [
        MULTILINE_SECRET,
        MULTILINE_SECRET + "\n",
        "\n  " + MULTILINE_SECRET + " \n",
    ],
    ids=["bare", "newline-terminated", "padded"],
)
@pytest.mark.parametrize("how", ["printed", "raised", "bug"])
def test_a_secret_with_a_newline_is_masked_before_the_split(
    capsys, monkeypatch, tmp_path, how, secret
):
    monkeypatch.setenv("D2_PASSWORD", secret)
    monkeypatch.setenv("BENCHFLOW_LOG_DIR", str(tmp_path / "logs"))

    def fail() -> None:
        if how == "printed":
            print_error(f"No such path: {secret}")
            raise typer.Exit(1)
        if how == "raised":
            raise MissingCredentialError(f"No such path: {secret}")
        raise RuntimeError(f"No such path: {secret}")

    assert _run(_app(fail), ["go", "--json"]) == 1
    out, err = capsys.readouterr()
    obj = _one_object(out)
    for half in MULTILINE_SECRET.splitlines():
        assert half not in out
        assert half not in err
    # Batch 7 at f79b2260: retain the message around the exact value.
    assert "***" in obj["error"]
    assert "No such path: ***" in obj["error"]
    # The crash log strips the traceback before it masks it, too.
    for log in (tmp_path / "logs").glob("*.log"):
        assert MULTILINE_SECRET.splitlines()[0] not in log.read_text()


@pytest.mark.parametrize("how", ["raised", "stopped"])
def test_a_newline_terminated_secret_is_masked_on_the_printed_path(
    capsys, monkeypatch, how
):
    # Without --json: report() printed user_message(), whose text error_parts
    # had already stripped, so the trailing newline kept the value from
    # matching (FF-CLI-C1); an error with no hint shows it. stopped_run_result
    # records the same stripped text.
    from benchflow.cli._shared import stopped_run_result
    from benchflow.errors import UserError

    secret = MULTILINE_SECRET + "\n"
    monkeypatch.setenv("D2_PASSWORD", secret)

    def fail() -> None:
        exc = UserError(f"No such path: {secret}")
        if how == "stopped":
            stopped_run_result(exc)
        raise exc

    argv = ["go", "--json"] if how == "stopped" else ["go"]
    assert _run(_app(fail), argv) == 1
    out, err = capsys.readouterr()
    shown = out if how == "stopped" else err
    for half in MULTILINE_SECRET.splitlines():
        assert half not in shown


@pytest.mark.parametrize("how", ["printed", "raised", "usage"])
def test_a_next_line_inside_an_error_is_not_its_next_step(capsys, how):
    # A path with a newline (or a tool's output) quoted in an error supplied
    # BenchFlow's "next" field; the step comes only from the
    # error's hint or the caller's next_step.
    forged = "/tmp/d2-missing\nnext: echo D2-FORGED"

    def fail() -> None:
        if how == "printed":
            print_error(f"Regrade failed: Not a job or trial folder: {forged}")
            raise typer.Exit(1)
        if how == "raised":
            raise MissingCredentialError(f"No such path: {forged}", hint="the hint")
        raise typer.BadParameter(f"no such path: {forged}")

    code = _run(_app(fail), ["go", "--json"])
    obj = _one_object(capsys.readouterr().out)
    assert obj["next"] != "echo D2-FORGED"
    assert "next: echo D2-FORGED" in obj["detail"]
    if how == "raised":
        assert code == 1 and obj["next"] == "the hint"
    if how == "usage":
        assert code == 2 and obj["next"].endswith("--help` lists what it takes")


def test_a_forged_next_line_in_a_regrade_path_stays_in_the_detail(tmp_path):
    missing = tmp_path / "d2-missing\nnext: echo D2-FORGED"
    done = _bench(tmp_path, "regrade", str(missing), "--json")
    assert done.returncode == 2, done.stderr
    obj = _one_object(done.stdout)
    assert obj["next"] == (
        "pass a job or trial folder of a run with --freeze-workspace"
    )
    assert obj["detail"] == "| next: echo D2-FORGED"


def test_the_error_object_reaches_an_ascii_stdout(tmp_path):
    # ensure_ascii=False wrote "café" as is, and an ASCII stdout raised
    # UnicodeEncodeError: no object, a traceback and exit 1.
    home = tmp_path / "home"
    home.mkdir()
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "NO_COLOR": "1",
        "BENCHFLOW_LOG_DIR": str(tmp_path / "logs"),
        "PYTHONIOENCODING": "ascii",
    }
    done = subprocess.run(
        [sys.executable, "-c", _LAUNCH, "explain", "café", "--json"],
        capture_output=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert done.returncode == 2, done.stderr
    obj = _one_object(done.stdout.decode("ascii"))
    assert "café" in obj["error"]
    assert obj["exit_code"] == 2
    assert b"Traceback" not in done.stderr


def test_a_double_dash_option_value_leaves_a_later_json_real(tmp_path):
    # `--reason --` gives --reason the value "--"; the --json after it is a
    # real flag, but the scan cut the line at that "--" and printed text.
    from benchflow.cli._errors import json_requested
    from benchflow.cli.main import app

    missing = tmp_path / "d2-missing"
    line = ["regrade", str(missing), "--reason", "--", "--json"]
    done = _bench(tmp_path, *line)
    assert done.returncode == 2, done.stderr
    obj = _one_object(done.stdout)
    assert obj["error"] == f"No such file or directory: {missing}"
    assert json_requested(line, app)
    # After a real delimiter --json is still an argument.
    assert not json_requested(["regrade", str(missing), "--", "--json"], app)
    assert not json_requested(["run", "--", "--json"], app)
