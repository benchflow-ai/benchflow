"""Where a verifier's CTRF report lands: the rule ``bench check`` and the run share.

A run reads a task's CTRF report from one path, ``/logs/verifier/ctrf.json``:
the verifier downloads it with the reward files (``verifier_core``), and
task.md grading and the job record read that copy. A test.sh that passes
``--ctrf`` should leave a report there, by naming that path or by copying,
moving or writing a file there itself.

Where a task.md criterion is graded from the report (a test-judged criterion,
taskmd ``verify``), a report that never arrives fails every such criterion, a
0 the run never earned: the task.md plan refuses it (``required``), so
``bench check``, the dry run and the run refuse it alike, before any sandbox.
Everywhere else the task grades itself and nothing is graded from the report:
``bench check`` (``check_task_warnings``) and the run warn, in the same words,
and the task runs.

The reading is conservative. A test.sh is found to lose its report only when
its ``--ctrf`` paths are plain literals, none of them is the report path, and
nothing in test.sh puts a file there. Whatever the reader can't follow leaves
the report possibly there: a path, copy target or redirection built at run
time, ``eval`` or ``source``, a shell run from a string or a heredoc, a
relative ``--ctrf`` path in a script that changes directory, or the report
path inside code it doesn't parse.

Without ``--ctrf`` (``required`` only) the reader finds the report lost only
in a script it reads whole: every command one that writes no file but
through a redirection (``printf``, ``cmp``, ``echo`` ...), none of them
writing the report path. A script that runs anything else (pytest, python, a
program of its own) may write the report there.
"""

import posixpath
import re
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Literal

from benchflow.task.paths import SandboxPaths

CTRF_REPORT_PATH = str(SandboxPaths.verifier_dir / "ctrf.json")
_REPORT_DIR = str(SandboxPaths.verifier_dir)
_REPORT_NAME = "ctrf.json"

_WORD = r"""(?:"[^"\n]*"|'[^'\n]*'|\\.|[^\s;&|()<>"'\\])"""
# One token of the shell text: a redirection with its target (``2>/dev/null``,
# ``<<'PY'``), a word with its quotes, an operator that ends a command, or a
# character none of those take (a quote left open).
_TOKEN = re.compile(
    rf"(?P<redirect>\d*(?:&>>?|>>|>&|>\||>|<<<|<<-|<<|<&|<)[ \t]*(?P<target>{_WORD}*))"
    rf"|(?P<word>{_WORD}+)"
    r"|(?P<end>[;&|()\n])"
    r"|(?P<stray>\S)"
)
_QUOTED = re.compile(r""""([^"\n]*)"|'([^'\n]*)'|\\(.)""")
_HEREDOC = re.compile(r"""(?<!<)<<(-?)[ \t]*(["']?)([A-Za-z_][\w.-]*)\2""")
_ASSIGNMENT = re.compile(r"[A-Za-z_]\w*=.*")
_PREFIXES = frozenset({"sudo", "command", "exec", "env", "time", "nohup"})
_COPIES = frozenset({"cp", "mv", "install"})
_SHELLS = frozenset({"sh", "bash", "dash", "zsh", "ksh"})
_OPAQUE = frozenset({"eval", "source", "."})
_CHDIR = frozenset({"cd", "pushd", "popd"})
# Shell words that start a command without being its program.
_KEYWORDS = frozenset(
    {"if", "then", "else", "elif", "do", "while", "until", "!", "{", "time"}
)
# Commands that write no file except through a redirection.
_INERT = frozenset(
    {
        "printf",
        "echo",
        "cat",
        "cmp",
        "diff",
        "test",
        "[",
        "[[",
        "]]",
        "grep",
        "mkdir",
        "rm",
        "rmdir",
        "chmod",
        "chown",
        "ls",
        "true",
        "false",
        ":",
        "exit",
        "set",
        "export",
        "unset",
        "local",
        "return",
        "read",
        "sleep",
        "wc",
        "head",
        "tail",
        "sort",
        "uniq",
        "cut",
        "tr",
        "basename",
        "dirname",
        "date",
        "id",
        "pwd",
        "stat",
        "fi",
        "done",
        "esac",
        "}",
        "then",
        "else",
        *_CHDIR,
    }
)
# Commands that read, test or remove a path and never write it.
_READERS = frozenset(
    {
        "rm",
        "test",
        "[",
        "[[",
        "ls",
        "stat",
        "cat",
        "head",
        "tail",
        "wc",
        "grep",
        "chmod",
        "chown",
        "echo",
        "printf",
    }
)

Verdict = Literal["no-ctrf", "reaches", "undecided", "missing"]


@dataclass(frozen=True)
class CtrfVerdict:
    """Whether test.sh's CTRF report reaches the path a run reads it from.

    ``verdict`` is ``no-ctrf`` (no ``--ctrf`` at all), ``reaches``,
    ``undecided`` (the reader can't tell, and the task is accepted) or
    ``missing`` (refused); ``reason`` says why.
    """

    verdict: Verdict
    reason: str


@dataclass(frozen=True)
class _Script:
    commands: list[list[str]]
    redirects: list[str]
    heredocs: list[str]
    shell_heredoc: bool
    stray: bool


def _unquote(word: str) -> str:
    return _QUOTED.sub(lambda m: next(g for g in m.groups() if g is not None), word)


def _built(path: str) -> bool:
    """Whether the shell builds *path* at run time (a variable, a substitution)."""
    return bool(set(path) & set("$`")) or path.startswith("~")


def _plain(path: str) -> bool:
    """Whether *path* reads literally: built at run time nowhere, and no glob."""
    return not (_built(path) or set(path) & set("*?["))


def _norm(path: str) -> str:
    return posixpath.normpath(path) if path.startswith("/") and _plain(path) else path


def _uncommented(line: str) -> str:
    """*line* without its comment, as the shell reads one: a `#` that starts a
    word (at the line's start, after a space or a tab, or after an operator),
    outside quotes. A `#` inside a word is the word's (`$#`, `${#var}`,
    `${var##*/}`, `http://h/x#frag`), and so is one in quotes (`'step #2'`).
    """
    quote = ""
    index = 0
    while index < len(line):
        char = line[index]
        if quote:
            if char == "\\" and quote == '"':
                index += 1
            elif char == quote:
                quote = ""
        elif char == "\\":
            index += 1
        elif char in "\"'":
            quote = char
        elif char == "#" and (index == 0 or line[index - 1] in " \t;&|()<>"):
            return line[:index]
        index += 1
    return line


def _parse(script: str) -> _Script:
    """test.sh's simple commands (unquoted words), redirections and heredocs."""
    lines = script.splitlines()
    shell: list[str] = []
    heredocs: list[str] = []
    shell_heredoc = False
    index = 0
    while index < len(lines):
        line = _uncommented(lines[index])
        index += 1
        shell.append(line)
        for dash, _, end in _HEREDOC.findall(line):
            body: list[str] = []
            while index < len(lines):
                text = lines[index]
                index += 1
                if (text.lstrip("\t") if dash else text) == end:
                    break
                body.append(text)
            heredocs.append("\n".join(body))
            words = re.findall(rf"{_WORD}+", line)
            shell_heredoc |= any(
                _unquote(w).rsplit("/", 1)[-1] in _SHELLS for w in words
            )
    commands: list[list[str]] = [[]]
    redirects: list[str] = []
    stray = False
    for token in _TOKEN.finditer("\n".join(shell).replace("\\\n", " ")):
        if token.group("word") is not None:
            commands[-1].append(_unquote(token.group("word")))
        elif token.group("redirect") is not None:
            redirects.append(_unquote(token.group("target")))
        elif token.group("end") is not None:
            if commands[-1]:
                commands.append([])
        else:
            stray = True
    return _Script(
        [command for command in commands if command],
        redirects,
        heredocs,
        shell_heredoc,
        stray,
    )


def _name(command: list[str]) -> tuple[int, str]:
    """Where the command word is in *command*, and its base name: past a
    prefix (``sudo``, ``env``), an assignment and a shell keyword (``then cp``)."""
    start = 0
    while start < len(command) and (
        command[start] in _PREFIXES
        or command[start] in _KEYWORDS
        or _ASSIGNMENT.fullmatch(command[start])
    ):
        start += 1
    if start == len(command):
        return start, ""
    return start, command[start].rsplit("/", 1)[-1]


def _ctrf_paths(command: list[str]) -> list[str]:
    """The report paths a command passes to ``--ctrf``."""
    paths = []
    for index, word in enumerate(command):
        if word == "--ctrf" and index + 1 < len(command):
            paths.append(command[index + 1])
        elif word.startswith("--ctrf="):
            paths.append(word.removeprefix("--ctrf="))
    return [path for path in paths if path]


def _writes(command: list[str]) -> tuple[list[str], list[str]] | None:
    """The sources and targets of a ``cp``, ``mv``, ``install`` or ``tee``."""
    start, name = _name(command)
    if name == "tee":
        return ["-"], [w for w in command[start + 1 :] if not w.startswith("-")]
    if name not in _COPIES:
        return None
    operands: list[str] = []
    directory: str | None = None
    words = iter(command[start + 1 :])
    for word in words:
        if word == "-t":
            directory = next(words, "")
        elif word.startswith("--target-directory="):
            directory = word.split("=", 1)[1]
        elif not word.startswith("-"):
            operands.append(word)
    if directory is not None:
        return operands, [directory]
    if len(operands) < 2:
        return None
    return operands[:-1], [operands[-1]]


def _carries_report(source: str, reports: list[str]) -> bool:
    """Whether a copy of *source* into the report folder lands as ``ctrf.json``."""
    if _plain(source):
        return posixpath.basename(source) == _REPORT_NAME
    return any(
        posixpath.basename(path) == _REPORT_NAME and fnmatchcase(path, source)
        for path in reports
        if _plain(path)
    )


def _program(command: list[str]) -> str:
    """The base name of the program *command* runs ("" for an assignment)."""
    start, name = _name(command)
    while name in _KEYWORDS and start + 1 < len(command):
        offset, name = _name(command[start + 1 :])
        start += 1 + offset
    return "" if name in _KEYWORDS else name


def ctrf_report_verdict(script: str, *, required: bool = False) -> CtrfVerdict:
    """Whether the CTRF report *script* (a test.sh) asks for reaches the run.

    *required*: criteria are graded from the report (task.md test-judged
    criteria), so the script must leave one there even with no ``--ctrf``.
    """
    parsed = _parse(script)
    reports = [_norm(p) for c in parsed.commands for p in _ctrf_paths(c)]
    if not reports and not required:
        return CtrfVerdict("no-ctrf", "test.sh passes no --ctrf")
    if CTRF_REPORT_PATH in reports:
        return CtrfVerdict("reaches", "a --ctrf path is the report path")
    writes = [w for w in map(_writes, parsed.commands) if w is not None]
    for sources, targets in writes:
        for target in map(_norm, targets):
            if target == CTRF_REPORT_PATH or (
                target == _REPORT_DIR
                and any(_carries_report(s, reports) for s in sources)
            ):
                return CtrfVerdict("reaches", "test.sh copies or moves a file there")
    if CTRF_REPORT_PATH in map(_norm, parsed.redirects):
        return CtrfVerdict("reaches", "test.sh writes the report path")

    unread: list[str] = []
    if not all(map(_plain, reports)):
        unread.append("a --ctrf path is built at run time")
    names = [_name(c) for c in parsed.commands]
    if any(not p.startswith("/") for p in reports) and any(
        n in _CHDIR for _, n in names
    ):
        unread.append("a relative --ctrf path in a script that changes directory")
    if (
        any(n in _OPAQUE for _, n in names)
        or parsed.shell_heredoc
        or any(
            n in _SHELLS and "-c" in c[i + 1 :]
            for c, (i, n) in zip(parsed.commands, names, strict=True)
        )
    ):
        unread.append("test.sh runs code the reader can't follow")
    if any(
        not _plain(t) or (_norm(t) == _REPORT_DIR and any(map(_built, sources)))
        for sources, targets in writes
        for t in targets
    ) or not all(map(_plain, parsed.redirects)):
        unread.append("a copy or redirection target is built at run time")
    mentioned = any(
        CTRF_REPORT_PATH in word
        for command, (i, name) in zip(parsed.commands, names, strict=True)
        if name not in _READERS and name not in _COPIES and name != "tee"
        for word in command[i + 1 :]
    ) or any(CTRF_REPORT_PATH in body or "--ctrf" in body for body in parsed.heredocs)
    if mentioned:
        unread.append("the report path is used in code the reader doesn't parse")
    if parsed.stray:
        unread.append("test.sh has a quote the reader can't pair")
    if not reports:
        programs = sorted({_program(c) for c in parsed.commands} - _INERT - {""})
        if programs:
            unread.append(f"test.sh runs {programs[0]}, which may write the report")
    if unread:
        return CtrfVerdict("undecided", "; ".join(unread))
    if not reports:
        return CtrfVerdict(
            "missing",
            f"test.sh never writes '{CTRF_REPORT_PATH}', the report they are "
            "graded from",
        )
    return CtrfVerdict(
        "missing",
        f"test.sh uses non-standard CTRF path '{reports[0]}' (expected "
        f"'{CTRF_REPORT_PATH}') and never copies it there; a run reads the "
        "report only from that path",
    )


def ctrf_report_issue(script: str, *, required: bool = False) -> str | None:
    """The refusal ``bench check`` and the run give *script*, or None."""
    verdict = ctrf_report_verdict(script, required=required)
    return verdict.reason if verdict.verdict == "missing" else None


def verifier_script_ctrf_issue(test_sh: Path, *, required: bool = False) -> str | None:
    """:func:`ctrf_report_issue` for the test.sh at *test_sh*; None when unreadable."""
    try:
        script = test_sh.read_text()
    except (OSError, UnicodeDecodeError):
        return None
    return ctrf_report_issue(script, required=required)
