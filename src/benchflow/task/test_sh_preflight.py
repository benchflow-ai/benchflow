"""What a task's test.sh needs, checked before the agent's turn.

The verifier runs test.sh in the task's image as a plain, non-login shell,
with hardening's PATH, as the verifier user. A command test.sh needs that is
not on that PATH can never run. Before this, such a trial ran the agent and
then ended "verifier wrote no usable reward": autocodebench's 200 tasks call
``uv``, which their image puts on PATH only through ``/root/.profile``.

The check reads test.sh's own text from the task folder on the host, never
anything the agent can write. Just before the agent's turn, after
BenchFlow's own setup (the agent's install, the sandbox user, the PATH lock),
it resolves what that text names on the PATH hardening would give the
verifier, as the verifier user:

- **The interpreter** BenchFlow starts test.sh with: its shebang, else bash,
  else sh.
- **The fatal prefix.** With errexit on, these are the plain top-level
  simple commands before the first line that could install a tool, change
  PATH, define a function, branch, or run code the check does not read. If
  one of them, or the interpreter, is missing, test.sh ends before any
  reward every time. The trial is then refused before the agent's turn,
  unscored and never retried (:class:`TestShPreflightRefused`).
- **Every command test.sh names** at top level, for a warning only. Such a
  command may be installed by test.sh itself or tolerated (``|| true``), so
  it changes no score. When a warned command is the one the shell last
  reported "not found" in ``verifier/test-stdout.txt`` and the trial scored
  0, the run summary lists it as a wrong-0 candidate.

A command that resolved here and is missing at grading is the agent's doing.
It stays what it always was: no usable reward, or a refusal stub's 0. The
reading is deliberately narrow, because a refusal must never come from a
guess. Anything the check cannot read for certain ends the fatal prefix, and
a script that is not sh or bash is not read at all.
"""

from __future__ import annotations

import logging
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: The head of the refusal; scoring reads it at the start (never retried).
REFUSAL_HEAD = "sandbox startup refused: test.sh needs "
#: How a warning reads in result.json's warnings.
WARNING_TEMPLATE = "test.sh names `{}`, not on the verifier PATH at preflight"

_KEYWORDS_OPEN = frozenset(
    ("if", "while", "until", "for", "case", "select", "{", "(", "[[")
)
_KEYWORDS_CLOSE = frozenset(("fi", "done", "esac", "}", ")", "]]"))
_KEYWORDS_PASS = frozenset(("then", "do", "else", "elif", "time", "in", "coproc"))
_BUILTINS = frozenset(
    [
        ".",
        ":",
        "[",
        "alias",
        "bg",
        "bind",
        "break",
        "builtin",
        "caller",
        "cd",
        "command",
        "compgen",
        "complete",
        "compopt",
        "continue",
        "declare",
        "dirs",
        "disown",
        "echo",
        "enable",
        "eval",
        "exec",
        "exit",
        "export",
        "false",
        "fc",
        "fg",
        "getopts",
        "hash",
        "help",
        "history",
        "jobs",
        "kill",
        "let",
        "local",
        "logout",
        "mapfile",
        "popd",
        "printf",
        "pushd",
        "pwd",
        "read",
        "readarray",
        "readonly",
        "return",
        "set",
        "shift",
        "shopt",
        "source",
        "suspend",
        "test",
        "times",
        "trap",
        "true",
        "type",
        "typeset",
        "ulimit",
        "umask",
        "unalias",
        "unset",
        "wait",
    ]
)
# Builtins after which the check reads no further: they can change what runs.
_BUILTINS_ENDING = frozenset(
    [
        ".",
        "source",
        "eval",
        "alias",
        "hash",
        "export",
        "unalias",
        "exec",
        "exit",
        "trap",
    ]
)
# Commands that can install a tool or run code the check does not read.
_INSTALLERS = frozenset(
    [
        "apt",
        "apt-get",
        "aptitude",
        "apk",
        "yum",
        "dnf",
        "microdnf",
        "zypper",
        "pacman",
        "brew",
        "port",
        "pip",
        "pip3",
        "pipx",
        "conda",
        "mamba",
        "micromamba",
        "npm",
        "npx",
        "pnpm",
        "yarn",
        "corepack",
        "gem",
        "bundle",
        "cargo",
        "rustup",
        "go",
        "curl",
        "wget",
        "sudo",
        "su",
        "env",
        "nix",
        "nix-env",
        "spack",
        "sdk",
        "asdf",
        "mise",
        "pyenv",
        "nvm",
        "make",
    ]
)
# An argument that marks a command as an install (`uv add`, `pip install`).
_INSTALL_ARGS = frozenset(("install", "add", "sync", "tool", "get", "-m"))
_SEPARATORS = frozenset((";", "&", "&&", "||", "|", "|&", "(", ")", ";;", ";&", ";;&"))
_REDIRECT = re.compile(r"^(?:\d*|&)(?:>>?|<<?|>&|<&|<>|>\|)$")
_REDIRECT_GLUED = re.compile(r"^(?:\d*|&)(?:>>?|<<?|>&|<&|<>|>\|).+")
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\[[^]]*\])?\+?=")
_HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z0-9_.-]+)\1")
_FUNCTION = re.compile(r"^\s*(?:function\s+[^\s(]+|[A-Za-z_][\w.:-]*\s*\(\s*\))")
_SHELLS = frozenset(("sh", "bash", "dash", "ash"))
# `$((...))` arithmetic, which the tokenizer would split at its parentheses.
_ARITHMETIC = re.compile(r"\$\(\([^()]*\)\)")
# A word that can name a program: it starts with a letter, digit, `_`, `.`
# or `/` and has a letter or digit in it.
_NAME = re.compile(r"^(?=.*[A-Za-z0-9])[A-Za-z0-9_./][\w.+@%/:-]*$")
_ERREXIT_ON = re.compile(r"^-[A-Za-z]*e[A-Za-z]*$")
_ERREXIT_OFF = re.compile(r"^\+[A-Za-z]*e[A-Za-z]*$")
# The shell's own message for a command it could not find (bash, then dash).
_NOT_FOUND = re.compile(
    r"(?:^|: )(?:line \d+: )?([^\s:/][^\s:]*|/[^\s:]+): (?:command not found|not found)\s*$"
)


@dataclass(frozen=True)
class TestShNeeds:
    """What test.sh's text says it needs."""

    # ("path", "/bin/bash"), ("env", "bash") or ("default", None): bash, else sh.
    interpreter: tuple[str, str | None]
    # The fatal prefix's command words (bare names or absolute paths).
    fatal: tuple[str, ...] = ()
    # Every top-level command word test.sh names (for the warning).
    named: tuple[str, ...] = ()
    # What ended the fatal prefix ("" when it reached the end of test.sh).
    prefix_end: str = ""


def _interpreter(first: str) -> tuple[str, str | None] | None:
    if not first.startswith("#!"):
        return ("default", None)
    parts = first[2:].split()
    if not parts:
        return None
    program = parts[0]
    if program.rsplit("/", 1)[-1] == "env":
        args = [a for a in parts[1:] if not a.startswith("-")]
        if not args or args[0] not in _SHELLS:
            return None
        return ("env", args[0])
    if program.rsplit("/", 1)[-1] not in _SHELLS:
        return None
    return ("path", program)


def _logical_lines(lines: list[str]) -> list[str]:
    """Lines with continuations joined; each heredoc body becomes one marker."""
    out: list[str] = []
    pending = ""
    i = 0
    while i < len(lines):
        line = pending + lines[i]
        i += 1
        if line.endswith("\\") and not line.endswith("\\\\"):
            pending = line[:-1]
            continue
        pending = ""
        out.append(line)
        for delimiter, tabs in _heredocs(line):
            while i < len(lines):
                body = lines[i].lstrip("\t") if tabs else lines[i]
                i += 1
                if body == delimiter:
                    break
            out.append("\x00heredoc")
    return out


# Redirections that use `&` (`2>&1`, `>&2`, `&>file`), which the tokenizer
# would otherwise split at the `&` of a background job.
_AMP_REDIRECT = re.compile(r"\d*[<>]&(?:\d+|-)|&>>?")


def _heredocs(line: str) -> list[tuple[str, bool]]:
    """Each heredoc a line opens, outside quotes: (delimiter, <<- strips tabs)."""
    found: list[tuple[str, bool]] = []
    quote = ""
    i = 0
    while i < len(line):
        c = line[i]
        if quote:
            if c == "\\" and quote == '"':
                i += 2
                continue
            if c == quote:
                quote = ""
        elif c in ("'", '"'):
            quote = c
        elif c == "#" and (i == 0 or line[i - 1].isspace()):
            break
        elif line.startswith("<<", i) and not line.startswith("<<<", i):
            match = _HEREDOC.match(line, i)
            if match:
                found.append((match.group(2), match.group(0).startswith("<<-")))
                i = match.end()
                continue
        i += 1
    return found


def _tokens(line: str) -> list[str] | None:
    line = _AMP_REDIRECT.sub(
        lambda m: " > " if m.group(0).startswith("&") else " ", line
    )
    lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|()")
    lexer.whitespace_split = True
    lexer.commenters = "#"
    try:
        return list(lexer)
    except ValueError:
        return None


def analyse(text: str) -> TestShNeeds | None:
    """Read test.sh *text*; None when it is not an sh or bash script."""
    lines = text.split("\n")
    interpreter = _interpreter(lines[0] if lines else "")
    if interpreter is None:
        return None
    first = lines[0] if lines and lines[0].startswith("#!") else ""
    errexit = any(_ERREXIT_ON.match(a) for a in first[2:].split()[1:])
    body = lines[1:] if first else lines
    fatal: list[str] = []
    named: list[str] = []
    prefix_end = ""
    depth = 0
    function_braces = 0

    def end(why: str) -> None:
        nonlocal prefix_end
        if not prefix_end:
            prefix_end = why

    lines_ = _logical_lines(body)
    functions = {
        m.group(1) or m.group(2)
        for m in (
            re.match(r"^\s*(?:function\s+([^\s(]+)|([A-Za-z_][\w.:-]*)\s*\(\s*\))", ln)
            for ln in lines_
        )
        if m
    }
    for line in lines_:
        if line == "\x00heredoc":
            end("a heredoc")
            continue
        if function_braces > 0:
            function_braces += line.count("{") - line.count("}")
            continue
        if _FUNCTION.match(line):
            end("a function definition")
            function_braces = max(0, line.count("{") - line.count("}"))
            if function_braces == 0 and "{" not in line and "(" in line:
                function_braces = 1  # `f()` with its `{` on the next line
            continue
        if line.lstrip().startswith("(("):
            end("an arithmetic command")
            continue
        tokens = _tokens(_ARITHMETIC.sub("0", line))
        if tokens is None:
            end("a line the check cannot split")
            continue
        if not tokens:
            continue
        if any(s in line for s in ("$(", "`", "<(", ">(")):
            end("a command substitution")
        separators = [t for t in tokens if t in _SEPARATORS]
        plain = all(t == ";" for t in separators)
        if not plain:
            end("a list, pipeline, subshell or background job")
        segment: list[str] = []
        for tok in [*tokens, ";"]:
            if tok not in _SEPARATORS:
                segment.append(tok)
                continue
            words, segment = segment, []
            negated = False
            k = 0
            while k < len(words):
                w = words[k]
                if _ASSIGNMENT.match(w):
                    if w.startswith(("PATH=", "PATH+=")):
                        end("an assignment to PATH")
                    k += 1
                elif _REDIRECT.match(w):
                    k += 2
                elif _REDIRECT_GLUED.match(w):
                    k += 1
                elif w in _KEYWORDS_OPEN:
                    depth += 1
                    end(f"a compound command (`{w}`)")
                    if w in ("for", "case", "select", "[["):
                        k = len(words)
                    else:
                        k += 1
                elif w in _KEYWORDS_CLOSE:
                    depth = max(0, depth - 1)
                    k += 1
                elif w in _KEYWORDS_PASS:
                    k += 1
                elif w == "!":
                    # A negated command's failure never ends the script.
                    negated = True
                    end("a negated command")
                    k += 1
                elif w in ("command", "builtin", "nohup") and k + 1 < len(words):
                    if words[k + 1].startswith("-"):
                        k = len(words)  # `command -v x` looks x up
                    else:
                        k += 1
                else:
                    break
            if k >= len(words):
                continue
            word, args = words[k], words[k + 1 :]
            if word == "set":
                if any(_ERREXIT_ON.match(a) for a in args) or (
                    "-o" in args and "errexit" in args
                ):
                    errexit = True
                if any(_ERREXIT_OFF.match(a) for a in args) or (
                    "+o" in args and "errexit" in args
                ):
                    errexit = False
                continue
            if word in _BUILTINS:
                if word in _BUILTINS_ENDING:
                    end(f"`{word}`, which can change what runs")
                continue
            if any(c in word for c in "$*?[]{}~") or (
                "/" in word and not word.startswith("/")
            ):
                end("a command the check cannot name")
                continue
            if word in functions or not _NAME.match(word):
                continue
            named.append(word)
            if not prefix_end and depth == 0 and errexit and not negated:
                fatal.append(word)
            if word.rsplit("/", 1)[-1] in _INSTALLERS or any(
                a in _INSTALL_ARGS for a in args
            ):
                end(f"`{word}`, which can install a tool or change what runs")
    return TestShNeeds(
        interpreter=interpreter,
        fatal=tuple(dict.fromkeys(fatal)),
        named=tuple(dict.fromkeys(named)),
        prefix_end=prefix_end,
    )


def last_not_found(stdout: str) -> str | None:
    """The command the shell last reported not found in test.sh's output."""
    name = None
    for line in stdout.splitlines():
        match = _NOT_FOUND.search(line)
        if match:
            name = match.group(1)
    return name


#: The head of a branch child's ending when its source recorded no verdict
#: and none could be decided on a fresh image (setup, never the task's).
UNAVAILABLE_HEAD = (
    "sandbox startup refused: no test.sh preflight verdict for this branch child"
)


class TestShPreflightUnavailable(RuntimeError):
    """A branch child whose source trial recorded no preflight verdict (a
    checkpoint from before the check), when none could be decided on a fresh
    sandbox of the task's image either: the child is not run. Unscored, never
    retried, BenchFlow's setup, never the task's fault."""


class TestShPreflightRefused(RuntimeError):
    """test.sh can never write a reward in this image: refused before the
    agent's turn. The message starts with "sandbox startup refused", so the
    trial is the task's setup failure, unscored and never retried.
    ``record`` is the trial's ``test_sh_preflight``."""

    def __init__(self, message: str, record: dict[str, Any]) -> None:
        super().__init__(message)
        self.record = record


# $1 the PATH, then each name: prints each that does not resolve, NUL-framed.
# File tests, as the required_tools contract's (no builtin or function counts).
_PROBE = r"""
trusted=$1
shift
for name do
  case $name in
    /*) if [ -f "$name" ] && [ -x "$name" ]; then continue; fi ;;
    *)
      rest=$trusted found=no
      while :; do
        folder=${rest%%:*}
        if [ -n "$folder" ] && [ -f "$folder/$name" ] && [ -x "$folder/$name" ]; then found=yes; break; fi
        [ "$rest" = "$folder" ] && break
        rest=${rest#*:}
      done
      [ "$found" = yes ] && continue ;;
  esac
  bf_encode "$name"; printf '\000'
done
"""


def _test_sh(task: Any) -> Path | None:
    path = getattr(getattr(task, "paths", None), "test_path", None)
    return Path(path) if path is not None and Path(path).is_file() else None


def applies(task: Any) -> bool:
    """The check runs where BenchFlow itself runs ``tests/test.sh`` in the
    agent's sandbox: a test-script verifier in ``main``, not a separate
    sandbox, from a legacy ``tests/`` folder (a native ``verifier/`` package
    picks its own strategy and is left as it is)."""
    from benchflow.task.config import VerifierSandboxMode

    verifier = getattr(getattr(task, "config", None), "verifier", None)
    paths = getattr(task, "paths", None)
    if verifier is None or paths is None:
        return False
    if getattr(verifier, "type", "test-script") != "test-script":
        return False
    if getattr(verifier, "service", "main") != "main" or getattr(
        verifier, "sandbox", None
    ):
        return False
    if getattr(verifier, "sandbox_mode", None) == VerifierSandboxMode.SEPARATE:
        return False
    if getattr(paths, "uses_native_verifier_dir", False):
        return False
    return _test_sh(task) is not None


def needs_verdict(task: Any) -> bool:
    """Whether a run of *task* has a preflight verdict to decide: the check
    applies and test.sh is an sh or bash script it reads (a ``#!/usr/bin/env
    python3`` test.sh is not read, so it has none). Reads the task folder on
    the host only."""
    if not applies(task):
        return False
    script = _test_sh(task)
    try:
        return (
            script is not None
            and analyse(script.read_text(errors="replace")) is not None
        )
    except Exception:
        return False


async def check(
    env: Any, task: Any, *, sandbox_user: str | None, workspace: str | None
) -> dict[str, Any] | None:
    """Resolve what test.sh needs, as the verifier will run it; the record,
    or None when the check does not apply. Raises
    :class:`TestShPreflightRefused` when test.sh can never write a reward."""
    from benchflow._utils.trusted_path import PATH_FIELD_ENCODE_SH, decode_path_field
    from benchflow.sandbox import lockdown

    if not applies(task):
        return None
    script = _test_sh(task)
    assert script is not None
    try:
        needs = analyse(script.read_text(errors="replace"))
    except Exception as exc:
        # The check's own reading failed: no evidence, never a refusal.
        logger.warning("The test.sh preflight could not read test.sh: %s", exc)
        return {
            "checked": False,
            "why": f"test.sh could not be read: {type(exc).__name__}",
        }
    if needs is None:
        return None
    user = getattr(task.config.verifier, "user", None)
    kind, program = needs.interpreter
    interpreter: list[str] = [program] if program else ["bash", "sh"]
    names = list(dict.fromkeys([*interpreter, *needs.fatal, *needs.named]))
    try:
        report = await lockdown._judge_verifier_path(env, sandbox_user, workspace)
        result = await lockdown._read_exec(
            env,
            lockdown._system_probe(
                PATH_FIELD_ENCODE_SH + "\n" + _PROBE,
                "bf-test-sh-preflight",
                report.path,
                *names,
                user=user,
            ),
            user=user,
            timeout_sec=lockdown.verifier_setup_timeout_sec(),
        )
    except Exception as exc:
        # BenchFlow's own judge or probe failed: no evidence either way, and
        # never a refusal.
        logger.warning("The test.sh preflight could not run: %s", exc)
        return {
            "checked": False,
            "why": f"the preflight could not run: {type(exc).__name__}",
        }
    if lockdown._exec_return_code(result) != 0:
        # BenchFlow's own probe failed: no evidence either way.
        return {"checked": False, "why": "the preflight probe could not run"}
    missing = [
        decode_path_field(n)
        for n in (getattr(result, "stdout", "") or "").split("\x00")
        if n
    ]
    as_user = user if user is not None else "the image's USER"
    record: dict[str, Any] = {
        "checked": True,
        "user": as_user,
        "path": report.path,
        "interpreter": program or "bash, else sh",
        "fatal": list(needs.fatal),
        "prefix_end": needs.prefix_end or "the end of test.sh",
        "missing": [n for n in needs.named if n in missing],
        "warnings": [WARNING_TEMPLATE.format(n) for n in needs.named if n in missing],
    }
    no_shell = kind == "default" and "bash" in missing and "sh" in missing
    needed = [n for n in interpreter if n in missing] if kind != "default" else []
    if no_shell:
        needed = ["bash or sh"]
    needed += [n for n in needs.fatal if n in missing]
    if needed:
        first = needed[0]
        record["refused"] = first
        raise TestShPreflightRefused(
            f"{REFUSAL_HEAD}`{first}`, which is not on the verifier's PATH "
            f"(checked before the agent's turn, as the verifier runs test.sh: "
            f"a plain shell, user {as_user}, PATH={report.path}); put it on "
            "the image's PATH (ENV PATH) or call it by its full path",
            record,
        )
    return record


def inherited(record: dict[str, Any]) -> dict[str, Any]:
    """A branch child's ``test_sh_preflight``: its source's verdict, carried
    and marked, never re-decided on the restored post-agent sandbox. The
    source's own wrong-0 reading is its own and is dropped (the child's
    comes from its own test-stdout.txt)."""
    carried = {k: v for k, v in record.items() if k != "wrong_zero_candidate"}
    carried["inherited"] = True
    return carried


def carried_refusal(record: dict[str, Any]) -> TestShPreflightRefused | None:
    """The refusal a carried verdict holds: the child is refused as its
    source was, for the same command."""
    name = record.get("refused")
    if not name:
        return None
    return TestShPreflightRefused(
        f"{REFUSAL_HEAD}`{name}`, which is not on the verifier's PATH (decided "
        "before the source trial's turn, as the verifier runs test.sh: a plain "
        f"shell, user {record.get('user')}, PATH={record.get('path')}); put it "
        "on the image's PATH (ENV PATH) or call it by its full path",
        record,
    )


async def decide_on_fresh_image(
    env: Any, task: Any, planes: Any, *, sandbox_user: str | None
) -> dict[str, Any] | None:
    """The verdict for a checkpoint that recorded none, decided on *env*, a
    sandbox just started from the task's image (nothing of anyone's run in
    it, and about to be replaced by the checkpoint): BenchFlow's PATH lock
    first, as a trial runs it before its agent, then the check. None when it
    cannot be decided."""
    from benchflow._utils.path_lock_record import recording_into
    from benchflow.rollout import _resolve_agent_cwd

    if not needs_verdict(task):
        return None
    try:
        workspace = await _resolve_agent_cwd(env, task)
        lock = getattr(planes, "lock_system_path", None)
        if lock is not None and getattr(env, "supports_system_shell", False) is True:
            with recording_into({}):
                await lock(env, workspace=workspace)
        try:
            record = await check(
                env, task, sandbox_user=sandbox_user, workspace=workspace
            )
        except TestShPreflightRefused as refused:
            record = refused.record
    except Exception as exc:
        logger.warning("The test.sh preflight could not run on a fresh image: %s", exc)
        return None
    if not record or not record.get("checked"):
        return None
    return {**record, "decided_on": "a fresh sandbox of the task's image"}


def source_record(trial_dir: Any) -> dict[str, Any] | None:
    """The ``test_sh_preflight`` an earlier trial recorded in its result.json,
    or None (no file, or none recorded)."""
    import json

    try:
        data = json.loads((Path(trial_dir) / "result.json").read_text())
    except (OSError, ValueError, TypeError):
        return None
    record = data.get("test_sh_preflight") if isinstance(data, dict) else None
    return record if isinstance(record, dict) else None


def wrong_zero_candidate(
    record: dict[str, Any] | None, reward: float | None, test_stdout: str | None
) -> str | None:
    """The warned command a 0 most likely came from: the trial scored 0 and
    the last command the shell reported not found in test-stdout.txt is one
    the preflight warned about. Output only names the candidate; it never
    changes a score."""
    if not record or not record.get("missing") or reward != 0 or not test_stdout:
        return None
    name = last_not_found(test_stdout)
    return name if name in record["missing"] else None


#: Where the wrong-0 candidates' "last command run" comes from.
WRONG_ZERO_SOURCE = (
    "the command the shell last reported not found in verifier/test-stdout.txt"
)


def _record_of(row: dict[str, Any], job_dir: Path | None) -> Any:
    """A row's ``test_sh_preflight``, else its trial's result.json's: a row
    built from the typed result does not carry it."""
    if "test_sh_preflight" in row or job_dir is None or not row.get("rollout_name"):
        return row.get("test_sh_preflight")
    try:
        import json

        data = json.loads(
            (Path(job_dir) / row["rollout_name"] / "result.json").read_text()
        )
    except (OSError, ValueError):
        return None
    return data.get("test_sh_preflight") if isinstance(data, dict) else None


def summary(results: Any, job_dir: Path | None = None) -> dict[str, Any]:
    """summary.json's ``test_sh_preflight``: trials checked, refused and
    warned, and the wrong-0 candidates (task, trial, command)."""
    checked = refused = warned = 0
    candidates: list[dict[str, Any]] = []
    for row in results:
        record = _record_of(row or {}, job_dir) or {}
        if not isinstance(record, dict):
            continue
        checked += bool(record.get("checked"))
        refused += bool(record.get("refused"))
        # A refused trial is counted as refused, not also as warned.
        warned += bool(record.get("missing")) and not record.get("refused")
        if record.get("wrong_zero_candidate"):
            candidates.append(
                {
                    "task": row.get("task_name"),
                    "trial": row.get("rollout_name"),
                    "command": record["wrong_zero_candidate"],
                }
            )
    return {
        "checked": checked,
        "refused": refused,
        "warned": warned,
        "wrong_zero_candidates": candidates,
        "wrong_zero_source": WRONG_ZERO_SOURCE,
    }
