"""How ``bench`` ends on an error: one message when expected, a log when not.

* An **expected** error, one the user can fix, prints one message, its next
  step, and exits 1: a :class:`benchflow.errors.UserError` (a task file that
  does not parse, a bad flag value, a missing login, a spent usage limit) or
  a missing or unreadable file.
* An **unexpected** error is a bug. It prints the traceback and the path of
  a log file with the command, the versions, the traceback and the run's
  last log lines, and exits 1.

With ``--json`` on the command line, every error is instead one JSON object
on stdout, ``{error, code, owner, next, detail, exit_code}``, as Rust's
``bench`` prints it (:func:`json_error`), and the exit code is
``exit_code``: a usage error (an unknown command or flag, a bad value, exit
2), an expected error (exit 1), a bug (exit 1; the traceback still goes to
stderr and the log), and a command that printed an error and stopped
(``print_error`` then a non-zero exit). Every key is always there, null when
the error has nothing for it. ``code`` is a code ``bench explain`` explains;
``owner`` is whose problem it is, in Rust's words: ``task``, ``agent``,
``model``, ``sandbox``, ``you`` or ``benchflow``. A command that exits
non-zero after printing its own JSON (``bench doctor --json`` with a failed
check) keeps its document as the only one on stdout; stdout that does not
start with ``{`` or ``[`` is not such a document, and the object follows it.

Diagnostics replace exact known credential values before formatting or splitting
the text. Configuring a credential never hides an unrelated message or hint.
Credential sources are named
environment/``.env`` variables, credential-named CLI options and assignments,
and Claude/Codex login files. Crash logs never include argument values, and
tracebacks never include note payloads, even without known credentials.
An unrecognised credential source (for example ``--agent-env FOO2=<token>``)
cannot be masked: read a log before attaching it to a public issue.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import logging
import os
import platform
import re
import sys
import tempfile
from collections import deque
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO, cast

from benchflow._diagnostics import redact, render_diagnostic
from benchflow.errors import UserError

ISSUES_URL = "https://github.com/benchflow-ai/benchflow/issues"
LOG_DIR_ENV = "BENCHFLOW_LOG_DIR"
# Missing or unreadable files the user named: expected, whatever raised them.
_FILE_ERRORS = (
    FileNotFoundError,
    NotADirectoryError,
    IsADirectoryError,
    PermissionError,
)


class RecentLog(logging.Handler):
    """Keeps the last ``capacity`` formatted log records for a crash log.

    Attached to the ``benchflow`` logger, not the root, so it keeps recording
    while the live dashboard swaps the root's console handlers out: the log
    holds what the dashboard hid.
    """

    def __init__(self, capacity: int = 2000) -> None:
        super().__init__(level=logging.DEBUG)
        self.lines: deque[str] = deque(maxlen=capacity)
        self.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )

    def emit(self, record: logging.LogRecord) -> None:
        # A log line must never break the run.
        with contextlib.suppress(Exception):
            self.lines.append(self.format(record))


RECENT_LOG = RecentLog()

# print_error can be called below a command, including one run with explicit
# argv in an embedding. Keep its credential arguments until that run ends.
_CLI_ARGV: ContextVar[Sequence[str] | None] = ContextVar("cli_error_argv", default=None)


def _redact(text: str, secrets: Sequence[str]) -> str:
    """Compatibility entry point; output sinks share the same redactor."""
    return redact(text, secrets)


def install_recent_log() -> None:
    logger = logging.getLogger("benchflow")
    if RECENT_LOG not in logger.handlers:
        logger.addHandler(RECENT_LOG)


# Login files whose contents never appear in the environment; their strings
# are redacted like an inline ``CODEX_AUTH_JSON`` (``_JSON`` keys contribute
# every long string inside them).
_LOGIN_FILES = ("~/.claude/.credentials.json", "~/.codex/auth.json")
_LOGIN_FILE_MAX_BYTES = 64 * 1024


def _login_file_secrets() -> dict[str, str]:
    found: dict[str, str] = {}
    from benchflow.agents.login_dirs import login_path

    # The home copies and, when CLAUDE_CONFIG_DIR or CODEX_HOME moves them,
    # the files the agents read.
    paths = [Path(name).expanduser() for name in _LOGIN_FILES]
    for name in _LOGIN_FILES:
        moved = login_path(name)
        if moved not in paths:
            paths.append(moved)
    configured = os.environ.get("CODEX_AUTH_JSON", "").strip()
    if configured and not configured.startswith("{"):
        paths.append(Path(configured).expanduser())
    for index, path in enumerate(paths):
        with contextlib.suppress(Exception):
            if path.stat().st_size <= _LOGIN_FILE_MAX_BYTES:
                found[f"LOGIN_FILE_{index}_AUTH_JSON"] = path.read_text()
    return found


def _secrets(argv: Sequence[str] | None = None) -> list[str]:
    from benchflow._dotenv import load_dotenv_env
    from benchflow.doctor import _SECRET_NAME_RE, secret_values

    environ: dict[str, str] = {}
    with contextlib.suppress(Exception):
        environ.update(load_dotenv_env())
    environ.update(os.environ)
    environ.update(_login_file_secrets())
    if argv is None:
        argv = _CLI_ARGV.get()
        if argv is None:
            argv = sys.argv[1:]
    words = list(argv)
    for index, arg in enumerate(words):
        pairs = [arg]
        if arg.startswith("-") and "=" in arg:
            # --agent-env=KEY=VALUE: the option's value is itself a pair.
            pairs.append(arg.split("=", 1)[1])
        for pair in pairs:
            key, sep, value = pair.partition("=")
            if sep and value and _SECRET_NAME_RE.search(key.upper()):
                # Named so secret_values() keeps it (it matches key names).
                environ[f"ARGV_{index}_{key.upper()}"] = value
        # --hf-token VALUE: an option named like a credential, then its value.
        following = words[index + 1] if index + 1 < len(words) else ""
        if (
            arg.startswith("-")
            and "=" not in arg
            and _SECRET_NAME_RE.search(arg.upper().replace("-", "_"))
            and following
        ):
            environ[f"ARGV_{index + 1}_TOKEN"] = following
    return secret_values(environ)


def expected_error_parts(exc: BaseException) -> tuple[str, str | None]:
    """Keep intact expected-error fields until the output choke point."""
    hint = getattr(exc, "hint", None)
    return str(exc) or type(exc).__name__, hint if isinstance(hint, str) else None


def _log_dir() -> Path:
    configured = os.environ.get(LOG_DIR_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    cache = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(cache).expanduser() if cache else Path.home() / ".cache"
    return base / "benchflow" / "logs"


def write_crash_log(
    exc: BaseException,
    argv: Sequence[str],
    *,
    secrets: Sequence[str],
) -> Path | None:
    """Write the crash log; its path, or None when no directory was writable."""
    from benchflow import __version__
    from benchflow.doctor import redacted_traceback

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    prefix = (
        "\n".join(
            [
                f"bench crash log, {datetime.now(UTC).isoformat()}",
                f"command: bench ({len(argv)} arguments omitted)",
                f"benchflow {__version__}, Python {platform.python_version()}, "
                f"{platform.system()} {platform.machine()}",
                "",
                "traceback:",
            ]
        )
        + "\n"
    )
    suffix = "\n".join(
        [
            "",
            "",
            f"last {len(RECENT_LOG.lines)} log lines:",
            "*** (recent log payloads omitted)",
            "",
        ]
    )
    for directory in (_log_dir(), Path(tempfile.gettempdir()) / "benchflow-logs"):
        try:
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"bench-{stamp}-{os.getpid()}.log"
            path.write_text(
                redacted_traceback(exc, secrets, prefix=prefix, suffix=suffix),
                encoding="utf-8",
            )
            return path
        except OSError:
            continue
    return None


def is_expected(exc: BaseException) -> bool:
    from benchflow._utils.disk_space import is_disk_full

    return isinstance(exc, (UserError, *_FILE_ERRORS)) or is_disk_full(exc)


# --- --json errors ------------------------------------------------------------

# Whose problem, in Rust's words, for each fault an expected error names
# (benchflow.errors.Fault; the same map as cli/explain.py's pages).
_OWNER_OF_FAULT = {
    "setup": "you",
    "task": "task",
    "agent": "agent",
    "infrastructure": "sandbox",
}


def json_requested(argv: Sequence[str], app: Any = None) -> bool:
    """Whether the command line asks for JSON: ``--json`` before any ``--``.

     A ``--`` can also be an option's value (``--reason -- --json``), with the
     ``--json`` after it still a real flag. When a ``--json`` comes only after
     a ``--`` and ``app`` is given, click's own parse of the line decides
    ; without it, or when that parse fails, the ``--`` ends the
     options.
    """
    words = list(argv)
    if "--json" not in words:
        return False
    if "--" not in words or words.index("--json") < words.index("--"):
        return True
    return app is not None and _json_flag_parsed(app, words) is True


def _json_flag_parsed(app: Any, words: list[str]) -> bool | None:
    """The value click parses for the ``--json`` of the command ``words`` run.

    The line is parsed down to its command with ``resilient_parsing`` (no
    command runs, no prompt), as click parses it to run it, with anything an
    option's callback prints (``--version``) thrown away: the real run prints
    it. None when it does not parse.
    """
    import io

    try:
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            return _parse_json_flag(app, words)
    except Exception:
        return None


def _parse_json_flag(app: Any, words: list[str]) -> bool | None:
    import typer

    command = typer.main.get_command(app) if isinstance(app, typer.Typer) else app
    ctx = command.make_context("bench", list(words), resilient_parsing=True)
    while True:
        for param in command.params:
            if "--json" in getattr(param, "opts", ()):
                return bool(ctx.params.get(param.name or ""))
        resolve = getattr(command, "resolve_command", None)
        # A group keeps the words after its own options here (click 8.2 and
        # later name it _protected_args).
        protected = getattr(ctx, "_protected_args", None)
        if protected is None:
            protected = getattr(ctx, "protected_args", [])
        rest = [*protected, *ctx.args]
        if resolve is None or not rest:
            # A command without a --json option: the word is an argument.
            return False
        name, sub, rest = resolve(ctx, rest)
        if sub is None:
            return None
        ctx = sub.make_context(name, rest, parent=ctx, resilient_parsing=True)
        command = sub


def json_error(
    error: str,
    *,
    exit_code: int,
    code: str | None = None,
    owner: str | None = None,
    next_step: str | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    """The one object an error is under ``--json``: every key, null when empty."""
    return {
        "error": error,
        "code": code or None,
        "owner": owner or None,
        "next": next_step or None,
        "detail": detail or None,
        "exit_code": exit_code,
    }


def split_detail(text: str) -> tuple[str, str | None]:
    """An error's text as its message (the first line) and its detail (the rest).

    The next step is never read out of the text: it comes from the error's
    ``hint``, the caller's ``next_step`` or the usage error's own, kept apart
    from the text, because the text can quote a downloaded task or a tool's
    output (D2b-R2-B-1). A ``next:`` line in the detail is marked as quoted
    (``| next:``, cli/_shared.py ``quote_next_lines``).
    """
    from benchflow.cli._shared import quote_next_lines

    lines = [
        line for line in quote_next_lines(text.strip()).splitlines() if line.strip()
    ]
    if not lines:
        return "", None
    return lines[0].strip(), "\n".join(lines[1:]) or None


class _JsonRun:
    """What a ``--json`` command printed: its error messages, and the first
    character it printed on stdout."""

    def __init__(self) -> None:
        # Each error's text and its next step, as print_error got them.
        self.errors: list[tuple[str, str | None]] = []
        # The first non-blank character on stdout (colour codes skipped).
        self.stdout_start = ""

    @property
    def stdout_is_json(self) -> bool:
        """The command began its own JSON document on stdout."""
        return self.stdout_start in ("{", "[")


# A terminal colour or cursor code, which is not the text a command printed.
_ANSI_CODE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


# Set while run_cli runs a --json command; print_error records into it.
_JSON_RUN: _JsonRun | None = None


def record_error(message: str, next_step: str | None = None) -> None:
    """Keep an error ``print_error`` printed, and its step, for the JSON error."""
    if _JSON_RUN is not None:
        _JSON_RUN.errors.append((str(message), next_step))


class _WatchedStdout:
    """sys.stdout, noting the first character the command wrote to it."""

    def __init__(self, inner: TextIO, run: _JsonRun) -> None:
        self._inner = inner
        self._run = run

    def write(self, text: Any) -> int:
        if text and not self._run.stdout_start:
            self._run.stdout_start = _ANSI_CODE.sub("", str(text)).lstrip()[:1]
        return self._inner.write(text)

    def writelines(self, lines: Any) -> None:
        for line in lines:
            self.write(line)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _click_classes() -> tuple[type[Exception], type[Exception], type[Exception]]:
    """ClickException, UsageError and Abort of the click typer runs on.

    typer 0.26 and later carry their own click (``typer._click``); their
    commands raise its classes, which click's ``except`` would miss. Their
    ``Abort`` is not in it (typer 0.27's is ``typer.exceptions.Abort``), so it
    is typer's own ``typer.Abort``, which is click's on the typers before.
    """
    import typer

    try:
        exceptions = importlib.import_module("typer._click.exceptions")
    except ImportError:
        exceptions = importlib.import_module("click.exceptions")
    abort = getattr(exceptions, "Abort", None) or typer.Abort
    return exceptions.ClickException, exceptions.UsageError, abort


def _usage_json(exc: Exception, secrets: Sequence[str]) -> dict[str, Any]:
    """A usage error as its JSON object: the message, and its next step."""
    message = getattr(exc, "bench_message", None)
    next_step = getattr(exc, "bench_next", None)
    detail = getattr(exc, "bench_detail", None)
    if message is None:
        format_message = getattr(exc, "format_message", None)
        text = format_message() if callable(format_message) else str(exc)
        # click's own text, with the words the user typed: no step of ours.
        message = text
    if next_step is None:
        ctx = getattr(exc, "ctx", None)
        path = getattr(ctx, "command_path", None) or "bench"
        next_step = f"`{path} --help` lists what it takes"
    exit_code = getattr(exc, "exit_code", 2)
    return json_error(
        message,
        exit_code=exit_code if isinstance(exit_code, int) else 2,
        owner="you",
        next_step=next_step,
        detail=detail,
    )


def _explain_code(exc: BaseException) -> str | None:
    """The ``bench explain`` id an error carries as ``code``, if it has a page."""
    code = getattr(exc, "code", None)
    if not isinstance(code, str):
        return None
    from benchflow.cli.explain import find

    page = find(code)
    return page.code if page is not None else None


def _emit_json(
    obj: dict[str, Any],
    secrets: Sequence[str],
    extra: Sequence[tuple[str, str | None]] = (),
) -> None:
    from benchflow.cli._shared import one_line, with_next_step

    keys = [key for key, value in obj.items() if isinstance(value, str)]
    fields = [obj[key] for key in keys]
    for message, step in extra:
        fields.extend((message, step or ""))

    def render(protected: Sequence[str]) -> dict[str, Any]:
        result = dict(obj)
        result.update(zip(keys, protected, strict=False))
        message, detail = split_detail(result["error"])
        result["error"] = message
        result["next"] = one_line(result["next"])
        details = [value for value in (detail, result["detail"]) if value]
        details.extend(
            with_next_step(protected[i], protected[i + 1] or None).strip()
            for i in range(len(keys), len(protected), 2)
        )
        result["detail"] = "\n".join(details) or None
        return result

    # ASCII escapes: the object must reach stdout whatever its encoding (an
    # ASCII locale, a surrogate-escaped file name) or it would end in a
    # UnicodeEncodeError and no object.
    sys.stdout.write(
        render_diagnostic(
            fields, secrets, render, serialize=lambda result: json.dumps(result) + "\n"
        )
    )
    sys.stdout.flush()


def report(exc: BaseException, argv: Sequence[str], *, as_json: bool = False) -> int:
    """Print what the user needs about ``exc``; the exit code.

    With ``as_json``, an expected error is only its JSON object on stdout; a
    bug is its object on stdout, and its traceback on stderr and in the log.
    """
    secrets = _secrets(argv)
    if is_expected(exc):
        # A UserError's next step is its hint; with none, its message is the
        # whole story. The step is never read out of the message's text.
        step: str | None
        from benchflow._utils.disk_space import (
            DISK_FULL_STEP,
            DISK_STEP,
            disk_full_text,
            is_disk_full,
        )

        if is_disk_full(exc) and not isinstance(exc, UserError):
            # The host's disk, not a bug (DF-p5-cli-craft-5): which file, how
            # much is free, and what to do. The file names the disk (the
            # cache, the temporary folder, a job: review L-A2).
            assert isinstance(exc, OSError)
            where = Path(exc.filename) if exc.filename else None
            text = disk_full_text(where, job_file=False)
            fault, step = "setup", DISK_STEP if where is not None else DISK_FULL_STEP
        elif isinstance(exc, _FILE_ERRORS) and not isinstance(exc, UserError):
            name = exc.filename or ""
            reason = exc.strerror or type(exc).__name__
            text = f"{reason}: {name}" if name else str(exc)
            fault = "setup"
            # The path is the user's; what to do is the same for each of these.
            step = "check the path, and that you can read it"
        else:
            text, step = expected_error_parts(exc)
            fault = getattr(exc, "fault", "setup")
        if as_json:
            _emit_json(
                json_error(
                    text or type(exc).__name__,
                    exit_code=1,
                    code=_explain_code(exc),
                    owner=_OWNER_OF_FAULT.get(str(fault)),
                    next_step=step,
                ),
                secrets,
            )
            return 1
        from benchflow.cli._shared import print_error

        print_error(
            text,
            next_step=step,
        )
        return 1
    from benchflow.doctor import redacted_traceback

    kind = type(exc).__name__
    log = write_crash_log(exc, argv, secrets=secrets)
    where = (
        f"The diagnostic log is in {log}."
        if log is not None
        else "No log file could be written."
    )
    sys.stderr.write(
        redacted_traceback(
            exc,
            secrets,
            suffix="\n\n"
            + (
                f"bench: unexpected error ({kind}). This is a bug in "
                f"BenchFlow, not in your task or setup. {where} Please attach it to "
                f"an issue at {ISSUES_URL}."
            )
            + "\n",
        )
    )
    if as_json:
        attach = f" with {log}" if log is not None else ""
        text = str(exc)
        _emit_json(
            json_error(
                f"unexpected error ({type(exc).__name__}): {text}",
                exit_code=1,
                owner="benchflow",
                next_step=f"this is a BenchFlow bug; please report it at "
                f"{ISSUES_URL}{attach}",
            ),
            secrets,
        )
    return 1


def _exit_json(code: int, run: _JsonRun, argv: Sequence[str]) -> None:
    """The JSON error of a command that ended with a non-zero exit code."""
    if code == 0:
        return
    if run.stdout_is_json:
        # The command printed its own JSON (a report with a failed check): it
        # stays the one document on stdout. Plain text a command printed on
        # stdout is not one, so its error still comes out as the object.
        return
    secrets = _secrets(argv)
    if run.errors:
        first, next_step = run.errors[0]
        message, detail = first, None
    elif code == 130:
        message, next_step, detail = "interrupted", None, None
    else:
        message = f"the command failed with exit code {code}"
        next_step, detail = None, "its messages, if any, are on stderr"
    _emit_json(
        json_error(
            message or f"the command failed with exit code {code}",
            exit_code=code,
            # Exit 2 is a usage error in every bench command.
            owner="you" if code == 2 else None,
            next_step=next_step,
            detail=detail,
        ),
        secrets,
        extra=run.errors[1:],
    )


def _run_json(app: Callable[..., Any], args: list[str], argv_given: bool) -> int:
    """Run ``app`` for a --json command line; every error is one JSON object."""
    global _JSON_RUN
    click_exception, usage_error, abort = _click_classes()
    run = _JsonRun()
    stdout = sys.stdout
    _JSON_RUN = run
    sys.stdout = cast("TextIO", _WatchedStdout(stdout, run))
    error: dict[str, Any] | None = None
    code = 0
    try:
        # Not standalone: click raises its errors instead of printing them, and
        # returns an exit code instead of exiting.
        if argv_given:
            returned = app(args=args, standalone_mode=False)
        else:
            returned = app(standalone_mode=False)
        # A command's own return value is not an exit code (bench exits 0 after
        # one, as click's standalone mode does); click returns an int only for
        # an exit (typer.Exit, ctx.exit) it caught.
        if isinstance(returned, int) and not isinstance(returned, bool):
            code = returned
    except SystemExit as exc:
        if isinstance(exc.code, int) or exc.code is None:
            code = exc.code or 0
        else:
            run.errors.append((str(exc.code), None))
            code = 1
    except usage_error as exc:
        error = _usage_json(exc, _secrets(args))
    except click_exception as exc:
        error = _usage_json(exc, _secrets(args))
    except abort:
        error = json_error("aborted", exit_code=1)
    except KeyboardInterrupt:
        code = 130
    except (Exception, BaseExceptionGroup) as exc:
        # The finally below runs after this return's report(): restore first.
        sys.stdout = stdout
        _JSON_RUN = None
        return report(exc, args, as_json=True)
    finally:
        sys.stdout = stdout
        _JSON_RUN = None
    if error is not None:
        _emit_json(error, _secrets(args))
        return int(error["exit_code"])
    _exit_json(code, run, args)
    return code


def run_cli(app: Callable[..., Any], argv: Sequence[str] | None = None) -> None:
    """Run the Typer ``app`` as the ``bench`` console script."""
    args = list(sys.argv[1:] if argv is None else argv)
    token = _CLI_ARGV.set(args)
    try:
        _run_cli(app, args, argv is not None)
    finally:
        _CLI_ARGV.reset(token)


def _run_cli(app: Callable[..., Any], args: list[str], argv_given: bool) -> None:
    install_recent_log()
    if json_requested(args, app):
        sys.exit(_run_json(app, args, argv_given))
    click_exception, _, abort = _click_classes()
    try:
        if argv_given:
            returned = app(args=args, standalone_mode=False)
        else:
            returned = app(standalone_mode=False)
        if isinstance(returned, int) and not isinstance(returned, bool):
            sys.exit(returned)
        sys.exit(0)
    except click_exception as exc:
        # Click/Typer's standalone mode prints usage errors itself, before
        # the shared redactor sees them. Render to a buffer, then mask it.
        # Exception constructors may require more than args (for example
        # UsageErrorWithNext). Copy their state without calling __init__.
        click_error = cast(Any, BaseException.__new__(type(exc)))
        click_error.__dict__.update(exc.__dict__)
        click_error.args = exc.args
        secrets = _secrets(args)

        def render(fields: Sequence[str]) -> str:
            click_error.message = fields[0]
            rendered = io.StringIO()
            click_error.show(file=rendered)
            return rendered.getvalue()

        sys.stderr.write(render_diagnostic((click_error.message,), secrets, render))
        sys.exit(click_error.exit_code)
    except abort:
        sys.stderr.write("Aborted!\n")
        sys.exit(1)
    except KeyboardInterrupt:
        sys.stderr.write("Interrupted.\n")
        sys.exit(130)
    except (Exception, BaseExceptionGroup) as exc:
        sys.exit(report(exc, args))
    finally:
        if not argv_given:
            _skip_exit_collection()


def _skip_exit_collection() -> None:
    """Keep the interpreter's exit from walking every object a command made.

    The console script is about to exit: its collections at shutdown walk
    every object of the run (the modules of an ``eval run`` included), about
    25 ms after ``bench eval run`` has printed its result. Frozen objects are
    left to the process exit; files are closed and logs flushed as before
    (``atexit`` and the standard streams do not depend on the collector).
    Only the console script does this (``argv is None``), never a caller that
    runs the app in its own process.
    """
    import gc

    gc.freeze()
