"""How ``bench`` ends on an error: one message when expected, a log when not.

* An **expected** error, one the user can fix, prints one message, its next
  step, and exits 1: a :class:`benchflow.errors.UserError` (a task file that
  does not parse, a bad flag value, a missing login, a spent usage limit) or
  a missing or unreadable file.
* An **unexpected** error is a bug. It prints the traceback and the path of
  a log file with the command, the versions, the traceback and the run's
  last log lines, and exits 1.

Credential values are replaced with ``***`` before anything is written:
every credential-looking variable of the environment and of ``.env``, every
``KEY=VALUE`` argument whose key looks like a credential (``--agent-env K=V``
and ``--agent-env=K=V``), the value of an option named like one
(``--hf-token V``, ``--hf-token=V``), and the strings inside the Claude and
Codex login files. Redaction matches names and values, so it cannot catch a
secret under a name that looks like nothing (``--bearer <token>``,
``--agent-env FOO2=<token>``) or one shorter than eight characters: read a
log before attaching it to a public issue.
"""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import re
import sys
import tempfile
import traceback
from collections import deque
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchflow.errors import UserError, user_message

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


def install_recent_log() -> None:
    logger = logging.getLogger("benchflow")
    if RECENT_LOG not in logger.handlers:
        logger.addHandler(RECENT_LOG)


# An option's value that reads as another option, so the value of a flag with
# no value is not taken for a secret and redacted out of the log.
# A long flag, or a short one of at most three characters ("-e", "-rf"): a
# longer "-..." word is far likelier to be a secret that begins with a dash.
_OPTION_LIKE = re.compile(r"^(--[A-Za-z][\w-]*|-[A-Za-z]\w{0,2})$")
# Login files whose contents never appear in the environment; their strings
# are redacted like an inline ``CODEX_AUTH_JSON`` (``_JSON`` keys contribute
# every long string inside them).
_LOGIN_FILES = ("~/.claude/.credentials.json", "~/.codex/auth.json")
_LOGIN_FILE_MAX_BYTES = 64 * 1024


def _login_file_secrets() -> dict[str, str]:
    found: dict[str, str] = {}
    paths = [Path(name).expanduser() for name in _LOGIN_FILES]
    configured = os.environ.get("CODEX_AUTH_JSON", "").strip()
    if configured and not configured.startswith("{"):
        paths.append(Path(configured).expanduser())
    for index, path in enumerate(paths):
        with contextlib.suppress(Exception):
            if path.stat().st_size <= _LOGIN_FILE_MAX_BYTES:
                found[f"LOGIN_FILE_{index}_AUTH_JSON"] = path.read_text()
    return found


def _secrets(argv: Sequence[str]) -> list[str]:
    from benchflow._dotenv import load_dotenv_env
    from benchflow.doctor import _SECRET_NAME_RE, secret_values

    environ: dict[str, str] = {}
    with contextlib.suppress(Exception):
        environ.update(load_dotenv_env())
    environ.update(os.environ)
    environ.update(_login_file_secrets())
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
            and not _OPTION_LIKE.match(following)
        ):
            environ[f"ARGV_{index + 1}_TOKEN"] = following
    return secret_values(environ)


def _redact(text: str, secrets: Sequence[str]) -> str:
    from benchflow.doctor import redact

    return redact(text, secrets)


def _log_dir() -> Path:
    configured = os.environ.get(LOG_DIR_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    cache = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(cache).expanduser() if cache else Path.home() / ".cache"
    return base / "benchflow" / "logs"


def write_crash_log(
    exc: BaseException, argv: Sequence[str], *, secrets: Sequence[str]
) -> Path | None:
    """Write the crash log; its path, or None when no directory was writable."""
    from benchflow import __version__

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    body = "\n".join(
        [
            f"bench crash log, {datetime.now(UTC).isoformat()}",
            "command: bench " + " ".join(argv),
            f"benchflow {__version__}, Python {platform.python_version()}, "
            f"{platform.system()} {platform.machine()}",
            "",
            "traceback:",
            "".join(traceback.format_exception(exc)).rstrip(),
            "",
            f"last {len(RECENT_LOG.lines)} log lines:",
            *RECENT_LOG.lines,
            "",
        ]
    )
    body = _redact(body, secrets)
    for directory in (_log_dir(), Path(tempfile.gettempdir()) / "benchflow-logs"):
        try:
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"bench-{stamp}-{os.getpid()}.log"
            path.write_text(body, encoding="utf-8")
            return path
        except OSError:
            continue
    return None


def is_expected(exc: BaseException) -> bool:
    return isinstance(exc, (UserError, *_FILE_ERRORS))


def report(exc: BaseException, argv: Sequence[str]) -> int:
    """Print what the user needs about ``exc``; the exit code."""
    secrets = _secrets(argv)
    if is_expected(exc):
        if isinstance(exc, _FILE_ERRORS) and not isinstance(exc, UserError):
            name = exc.filename or ""
            reason = exc.strerror or type(exc).__name__
            text = f"{reason}: {name}" if name else str(exc)
        else:
            text = user_message(exc)
        from benchflow.cli._shared import print_error

        print_error(_redact(text, secrets).rstrip())
        return 1
    log = write_crash_log(exc, argv, secrets=secrets)
    trace = "".join(traceback.format_exception(exc)).rstrip()
    where = (
        f"The traceback and this run's last log lines are in {log}."
        if log is not None
        else "No log file could be written."
    )
    sys.stderr.write(
        _redact(trace, secrets)
        + "\n\n"
        + _redact(
            f"bench: unexpected error ({type(exc).__name__}). This is a bug in "
            f"BenchFlow, not in your task or setup. {where} Please attach it to "
            f"an issue at {ISSUES_URL}.",
            secrets,
        )
        + "\n"
    )
    return 1


def run_cli(app: Callable[..., Any], argv: Sequence[str] | None = None) -> None:
    """Run the Typer ``app`` as the ``bench`` console script."""
    args = list(sys.argv[1:] if argv is None else argv)
    install_recent_log()
    try:
        if argv is None:
            app()
        else:
            app(args=args)
    except KeyboardInterrupt:
        sys.stderr.write("Interrupted.\n")
        sys.exit(130)
    except Exception as exc:
        sys.exit(report(exc, args))
