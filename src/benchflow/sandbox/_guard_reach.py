"""Whether test.sh starts a pytest the plugin guard reaches, read from the
host's copy of the task's verifier scripts ((g2), the lead's
ruling of 2026-11-16 13:57Z).

In ``--integrity strict`` the guard rides in BenchFlow's ``PYTEST_ADDOPTS``
and leaves a signed marker in every pytest it reaches. A run whose pytest
left none is never a pass: its reward is not test.sh's. But "no marker" is
also every honest run that never starts pytest (a shell test, a Node or
Go suite) and every run whose pytest the guard does not reach on the task's
own terms (``tox`` builds its own environment; ``sudo``, ``su`` and
``env -i`` drop BenchFlow's variables; ``ssh`` and ``docker exec`` leave
the sandbox), and those score exactly as before.

So the evidence that a pytest the guard reaches ran is the task's own text,
on the host, where neither the agent nor test.sh's user can change it: a
pytest at a command position of test.sh (or of a script of the task's
verifier folder it runs), started without anything that drops or replaces
``PYTEST_ADDOPTS``. Not the sandbox's outputs: code test.sh runs can stop
pytest before it prints a word, writes a cache or a junit file (CIR's
``si-exit``: an ini's ``-p`` module that calls ``os._exit(0)``), and it can
delete or forge any of them.

The reading is conservative in one direction only. Any construct it does
not follow (a variable in command position, a script outside the verifier
folder, ``make``) finds no pytest, and the run scores as before: it can miss
a pytest, never invent one where test.sh starts none. A pytest inside a
branch test.sh does not take counts (``if ...; then pytest; else ...``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

#: Scripts followed from test.sh, how deep and how many bytes in all.
_MAX_DEPTH = 4
_MAX_SCRIPTS = 32
_MAX_BYTES = 1024 * 1024
#: How much of a command is kept as written, and how many of each kind.
_SHOWN_CHARS = 160
_SHOWN = 3

_PYTEST = {"pytest", "py.test", "pytest-3", "pytest3"}
# pytest runners that build their own environment from their own pass-list.
_OWN_ENV = {"tox", "nox", "detox"}
# Commands that run their command with BenchFlow's variables dropped, or
# outside this sandbox.
_RESET = {
    "sudo",
    "su",
    "runuser",
    "doas",
    "ssh",
    "chroot",
    "nsenter",
    "docker",
    "podman",
    "kubectl",
    "machinectl",
    "systemd-run",
    "schroot",
    "lxc-attach",
}
_SHELLS = {"bash", "sh", "dash", "zsh", "ksh"}
# Environment managers whose ``run`` keeps the caller's variables.
_RUNNERS = {
    "poetry": ("run",),
    "pipenv": ("run",),
    "pdm": ("run",),
    "hatch": ("run",),
    "rye": ("run",),
    "pixi": ("run",),
    "conda": ("run",),
    "mamba": ("run",),
    "micromamba": ("run",),
}
# Options of those runners (and of ``uv run``/``uvx``) that take a value.
_RUNNER_VALUE_OPTIONS = {
    "-n",
    "--name",
    "-p",
    "--prefix",
    "--python",
    "--with",
    "-w",
    "--from",
    "--with-requirements",
    "--project",
    "--directory",
    "--package",
    "--extra",
    "--group",
    "--env-file",
    "--index",
    "--index-url",
    "--extra-index-url",
    "-i",
    "--cwd",
    "-e",
    "--environment",
    "--manifest-path",
}
# Wrappers that run their command with the same variables: how many operands
# they take before it, beside options, and their options that take a value.
_WRAPPERS: dict[str, tuple[int, frozenset[str]]] = {
    "exec": (0, frozenset({"-a"})),
    "builtin": (0, frozenset()),
    "nohup": (0, frozenset()),
    "setsid": (0, frozenset()),
    "unbuffer": (0, frozenset()),
    "time": (0, frozenset({"-f", "-o", "--format", "--output"})),
    "catchsegv": (0, frozenset()),
    "xvfb-run": (0, frozenset({"-n", "-s", "-e", "-f", "-w", "--server-args"})),
    "stdbuf": (0, frozenset({"-i", "-o", "-e"})),
    "nice": (0, frozenset({"-n"})),
    "ionice": (0, frozenset({"-c", "-n", "-p"})),
    "timeout": (1, frozenset({"-s", "-k", "--signal", "--kill-after"})),
    "flock": (1, frozenset({"-w", "-E", "--timeout", "--conflict-exit-code"})),
    "setpriv": (0, frozenset()),
    "gosu": (1, frozenset()),
    "su-exec": (1, frozenset()),
    "xargs": (0, frozenset({"-n", "-L", "-I", "-P", "-d", "-E", "-s", "-a"})),
}
_KEYWORDS = {
    "if",
    "then",
    "else",
    "elif",
    "fi",
    "do",
    "done",
    "while",
    "until",
    "!",
    "{",
    "}",
    "esac",
}
_SKIP_COMMANDS = {"for", "case", "select", "function"}
_DECLARE = {"export", "declare", "typeset", "local", "readonly"}
_ASSIGNMENT = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)(\+?)=(.*)", re.DOTALL)
_PYTHON = re.compile(r"(?:python|pypy)[0-9.]*(?:-dbg)?")
# A Python script that starts pytest: pytest.main(), or a subprocess of it.
_PYTHON_PYTEST = re.compile(
    r"\bpytest\s*\.\s*(?:main|console_main)\s*\("
    r"|[\"']-m[\"']\s*,\s*[\"']pytest[\"']"
    r"|\[\s*[\"'](?:pytest|py\.test)[\"']\s*[,\]]"
)
_KEEPS_ADDOPTS = re.compile(r"\$\{?PYTEST_ADDOPTS\b")


@dataclass(frozen=True)
class PytestReach:
    """What the host's copy of the verifier's scripts says about pytest.

    ``direct``: commands that start pytest where BenchFlow's variables reach
    it. ``outside``: commands that start pytest where they do not (its own
    environment, a reset one, another machine). ``overrides``: what sets or
    clears ``PYTEST_ADDOPTS``. Each holds at most a few, as written.
    ``plugins`` and ``disabled``: the ``-p NAME`` and ``-p no:NAME`` of the
    pytest commands in ``direct``, every one. ``paths``: the paths those
    commands give pytest, absolute (a relative one only where test.sh's
    folder is known from a literal ``cd``), and ``bare``: the folder each
    that gives none runs in, ``""`` where it is not known."""

    direct: tuple[str, ...] = ()
    outside: tuple[str, ...] = ()
    overrides: tuple[str, ...] = ()
    plugins: tuple[str, ...] = ()
    disabled: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()
    bare: tuple[str, ...] = ()

    @property
    def expects_record(self) -> bool:
        """Every pytest test.sh starts is one the guard reaches, and it
        starts one: a run with no marker of the guard is not test.sh's
        pass."""
        return bool(self.direct) and not self.outside and not self.overrides


def script_reach(
    text: str, tests_dir: Any = None, mount: str = "/tests", cwd: str | None = None
) -> PytestReach:
    """Read *text*, test.sh or a script strategy's command, as the verifier
    runs it from *mount* (in the folder *cwd*, where the verifier names one);
    scripts it starts are read from *tests_dir* (the host's copy of the
    verifier folder)."""
    scan = _Scan(_folder(tests_dir), mount.rstrip("/") or "/")
    if isinstance(cwd, str) and cwd.startswith("/"):
        scan.cwd = _normpath(cwd)
    first = text.split("\n", 1)[0]
    if first.startswith("#!") and "python" in first:
        # A test.sh in Python runs under its shebang.
        if _PYTHON_PYTEST.search(text):
            scan.note("direct", _shown([first[2:].strip(), "(pytest.main)"]))
    else:
        scan.shell(text, 0)
    return PytestReach(
        tuple(scan.found["direct"][:_SHOWN]),
        tuple(scan.found["outside"][:_SHOWN]),
        tuple(scan.found["overrides"][:_SHOWN]),
        tuple(scan.found["plugins"]),
        tuple(scan.found["disabled"]),
        tuple(scan.found["paths"]),
        tuple(scan.found["bare"]),
    )


def task_reach(
    task: Any, strategy_command: str | None = None, cwd: str | None = None
) -> PytestReach:
    """:func:`script_reach` of a task's own test.sh (or the script
    strategy's command), on the host; an unreadable one finds nothing."""
    paths = getattr(task, "paths", None)
    tests_dir = getattr(paths, "tests_dir", None)
    native = getattr(paths, "uses_native_verifier_dir", False) is True
    mount = "/verifier" if native else "/tests"
    if strategy_command is not None:
        return script_reach(strategy_command, tests_dir, mount, cwd)
    test_path = getattr(paths, "test_path", None)
    if test_path is None:
        return PytestReach()
    try:
        path = Path(test_path)
        if path.stat().st_size > _MAX_BYTES:
            return PytestReach()
        text = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, TypeError, ValueError):
        return PytestReach()
    return script_reach(text, tests_dir, mount, cwd)


def _folder(tests_dir: Any) -> Path | None:
    try:
        folder = Path(tests_dir).resolve()
    except (OSError, TypeError, ValueError, RuntimeError):
        return None
    return folder if folder.is_dir() else None


def _shown(words: list[str]) -> str:
    text = " ".join(words)
    return text if len(text) <= _SHOWN_CHARS else text[: _SHOWN_CHARS - 1] + "…"


class _Scan:
    def __init__(self, folder: Path | None, mount: str) -> None:
        self.folder = folder
        self.mount = mount
        self.found: dict[str, list[str]] = {
            "direct": [],
            "outside": [],
            "overrides": [],
            "plugins": [],
            "disabled": [],
            "paths": [],
            "bare": [],
        }
        self.read: set[Path] = set()
        self.budget = _MAX_BYTES
        # test.sh's folder where a literal ``cd`` set it, else None.
        self.cwd: str | None = None

    def note(self, kind: str, what: str) -> None:
        if what not in self.found[kind]:
            self.found[kind].append(what)

    # -- shell ---------------------------------------------------------------

    def shell(self, text: str, depth: int) -> None:
        if depth > _MAX_DEPTH:
            return
        for words, bodies in _commands(text):
            self.command(words, bodies, depth, False)

    def command(
        self, words: list[str], bodies: list[str], depth: int, reset: bool
    ) -> None:
        all_words = words
        w = list(words)
        while w:
            head = w[0]
            base = head.rsplit("/", 1)[-1]
            assignment = _ASSIGNMENT.fullmatch(head)
            if assignment:
                self.assign(assignment)
                w = w[1:]
                continue
            if head in _SKIP_COMMANDS:
                return
            if head in _KEYWORDS:
                w = w[1:]
                continue
            if base in _DECLARE:
                for word in w[1:]:
                    found = _ASSIGNMENT.fullmatch(word)
                    if found:
                        self.assign(found)
                    elif word == "PYTEST_ADDOPTS" and base != "export":
                        self.note("overrides", _shown(w))
                return
            if base == "unset":
                if "PYTEST_ADDOPTS" in w[1:]:
                    self.note("overrides", _shown(w))
                return
            if base in ("cd", "pushd", "popd"):
                self.cwd = (
                    _folder_after_cd(w[1:], self.cwd, self.mount)
                    if base != "popd"
                    else None
                )
                return
            if base == "env":
                w, reset = self.env(w, reset)
                continue
            if base == "command":
                if len(w) > 1 and w[1] in ("-v", "-V"):
                    return
                w = [word for word in w[1:2] if word != "-p"] + w[2:]
                continue
            if base in _WRAPPERS:
                w = _unwrapped(w, *_WRAPPERS[base])
                continue
            if base in _RESET:
                # Its own options and target first: whatever it runs runs
                # without BenchFlow's variables, or on another machine.
                if _mentions_pytest(w[1:]):
                    self.note("outside", _shown(all_words))
                return
            break
        if not w:
            return
        self.program(w, bodies, depth, reset, all_words)

    def env(self, w: list[str], reset: bool) -> tuple[list[str], bool]:
        i = 1
        while i < len(w):
            word = w[i]
            if word in ("-i", "-", "--ignore-environment"):
                reset = True
            elif word in ("-u", "--unset"):
                if i + 1 < len(w) and w[i + 1] == "PYTEST_ADDOPTS":
                    self.note("overrides", _shown(w))
                i += 1
            elif word.startswith(("--unset=", "-u")) and word.endswith(
                "PYTEST_ADDOPTS"
            ):
                self.note("overrides", _shown(w))
            elif word in ("-C", "--chdir", "-S", "--split-string"):
                i += 1
            elif word.startswith("-"):
                pass
            else:
                found = _ASSIGNMENT.fullmatch(word)
                if not found:
                    break
                self.assign(found)
            i += 1
        return w[i:], reset

    def assign(self, found: re.Match[str]) -> None:
        name, append, value = found.groups()
        if name == "PYTEST_ADDOPTS" and not append and not _KEEPS_ADDOPTS.search(value):
            self.note("overrides", _shown([found.group(0)]))

    def program(
        self,
        w: list[str],
        bodies: list[str],
        depth: int,
        reset: bool,
        all_words: list[str],
    ) -> None:
        head = w[0]
        base = head.rsplit("/", 1)[-1]
        if head.startswith("$") and _python_runs_pytest(w[1:]):
            # ``$PYTHON -m pytest``: whatever Python it names
            self.pytest(all_words, reset)
        elif base in _PYTEST:
            self.pytest(all_words, reset)
        elif base in _OWN_ENV:
            self.note("outside", _shown(all_words))
        elif _PYTHON.fullmatch(base):
            self.python(w[1:], bodies, depth, reset, all_words)
        elif base == "uv":
            rest = w[1:]
            if rest[:1] == ["run"]:
                self.runner(rest[1:], bodies, depth, reset, all_words)
            elif rest[:2] == ["tool", "run"]:
                self.tool(rest[2:], reset, all_words)
        elif base == "uvx":
            self.tool(w[1:], reset, all_words)
        elif base in _RUNNERS and w[1:2] and w[1] in _RUNNERS[base]:
            self.runner(w[2:], bodies, depth, reset, all_words)
        elif base == "coverage":
            if w[1:2] == ["run"]:
                self.python(w[2:], bodies, depth, reset, all_words)
        elif base in _SHELLS:
            self.sub_shell(w[1:], bodies, depth, reset)
        elif base in ("source", "."):
            if len(w) > 1:
                self.script(w[1], depth, reset, shell=True)
        elif base == "eval":
            self.shell_in(" ".join(w[1:]), depth, reset)
        elif head.endswith((".sh", ".bash", ".py")) or head.startswith(
            ("./", self.mount + "/", "$TEST_DIR/", "${TEST_DIR}/")
        ):
            self.script(head, depth, reset, shell=None, all_words=all_words)

    def pytest(self, words: list[str], reset: bool) -> None:
        self.note("outside" if reset else "direct", _shown(words))
        if not reset:
            for name in _plugin_args(words):
                if name.startswith("no:"):
                    self.note("disabled", name[3:].strip())
                elif name:
                    self.note("plugins", name)
            given = _path_args(words)
            for path in given:
                found = _absolute(path, self.cwd, self.mount)
                if found:
                    self.note("paths", found)
            if not given:
                self.found["bare"].append(self.cwd or "")

    def python(
        self,
        args: list[str],
        bodies: list[str],
        depth: int,
        reset: bool,
        all_words: list[str],
    ) -> None:
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "-m" or (arg.startswith("-m") and len(arg) > 2):
                module = (
                    arg[2:]
                    if arg != "-m"
                    else (args[i + 1] if i + 1 < len(args) else "")
                )
                rest = args[i + 2 :] if arg == "-m" else args[i + 1 :]
                if module in ("pytest", "py.test"):
                    self.pytest(all_words, reset)
                elif module in _OWN_ENV:
                    self.note("outside", _shown(all_words))
                elif module == "coverage" and rest[:1] == ["run"]:
                    self.python(rest[1:], bodies, depth, reset, all_words)
                elif module == "uv" and rest[:1] == ["run"]:
                    self.runner(rest[1:], bodies, depth, reset, all_words)
                return
            if arg == "-c":
                if i + 1 < len(args) and _PYTHON_PYTEST.search(args[i + 1]):
                    self.pytest(all_words, reset)
                return
            if arg in ("-W", "-X", "-Q", "--rcfile", "--source", "--include", "--omit"):
                i += 2
                continue
            if arg == "-":
                break
            if arg.startswith("-"):
                i += 1
                continue
            self.script(arg, depth, reset, shell=False, all_words=all_words)
            return
        if any(_PYTHON_PYTEST.search(body) for body in bodies):
            self.pytest(all_words, reset)

    def runner(
        self,
        args: list[str],
        bodies: list[str],
        depth: int,
        reset: bool,
        all_words: list[str],
    ) -> None:
        rest = _after_options(args)
        if not rest:
            return
        if rest[0] == "--":
            rest = rest[1:]
        if rest:
            self.command(rest, bodies, depth, reset)

    def tool(self, args: list[str], reset: bool, all_words: list[str]) -> None:
        rest = _after_options(args)
        if not rest:
            return
        name = re.split(r"[=<>!~@\[ ]", rest[0], maxsplit=1)[0]
        if name in _PYTEST:
            self.pytest(all_words, reset)
        elif name in _OWN_ENV:
            self.note("outside", _shown(all_words))

    def sub_shell(
        self, args: list[str], bodies: list[str], depth: int, reset: bool
    ) -> None:
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "-c" or (
                arg.startswith("-") and not arg.startswith("--") and "c" in arg[1:]
            ):
                if i + 1 < len(args):
                    self.shell_in(args[i + 1], depth, reset)
                return
            if arg in ("-o", "+o", "-O", "+O"):
                i += 2
                continue
            if arg.startswith(("-", "+")):
                i += 1
                continue
            self.script(arg, depth, reset, shell=True)
            return
        for body in bodies:
            self.shell_in(body, depth, reset)

    def shell_in(self, text: str, depth: int, reset: bool) -> None:
        if depth + 1 > _MAX_DEPTH:
            return
        for words, bodies in _commands(text):
            self.command(words, bodies, depth + 1, reset)

    def script(
        self,
        spelled: str,
        depth: int,
        reset: bool,
        *,
        shell: bool | None,
        all_words: list[str] | None = None,
    ) -> None:
        path = self.lookup(spelled)
        if path is None or depth + 1 > _MAX_DEPTH or len(self.read) >= _MAX_SCRIPTS:
            return
        try:
            size = path.stat().st_size
            if size > self.budget:
                return
            self.budget -= size
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        self.read.add(path)
        first = text.split("\n", 1)[0]
        python = not shell and (
            shell is False
            or path.suffix == ".py"
            or (first.startswith("#!") and "python" in first)
        )
        if python:
            if _PYTHON_PYTEST.search(text):
                self.pytest(all_words or [spelled], reset)
            return
        for words, bodies in _commands(text):
            self.command(words, bodies, depth + 1, reset)

    def lookup(self, spelled: str) -> Path | None:
        """The host's copy of a script the verifier names *spelled*: in the
        verifier folder, by its mount path, ``$TEST_DIR`` or a relative
        path; anything else is not followed."""
        if self.folder is None or not spelled:
            return None
        for prefix in ("$TEST_DIR/", "${TEST_DIR}/", self.mount + "/"):
            if spelled.startswith(prefix):
                relative = spelled[len(prefix) :]
                break
        else:
            if spelled.startswith(("/", "$", "~")):
                return None
            relative = spelled
        parts = PurePosixPath(relative).parts
        if not parts or any(part in ("..", "") for part in parts):
            return None
        try:
            path = self.folder.joinpath(*parts).resolve()
        except (OSError, RuntimeError, ValueError):
            return None
        if path in self.read or not path.is_file():
            return None
        if self.folder != path and self.folder not in path.parents:
            return None
        return path


def _unwrapped(w: list[str], operands: int, takes_value: frozenset[str]) -> list[str]:
    i = 1
    while i < len(w) and w[i].startswith("-") and w[i] != "--":
        i += 2 if w[i] in takes_value else 1
    if i < len(w) and w[i] == "--":
        i += 1
    return w[i + operands :]


_PYTEST_WORD = re.compile(
    r"(?:^|[\s;&|(`])(?:\S*/)?(?:py\.?test(?:-3|3)?|(?:python|pypy)[0-9.]*\s+(?:-\S+\s+)*-m\s*py\.?test)(?=$|[\s;&|)`=])"
)


def _mentions_pytest(words: list[str]) -> bool:
    """Whether *words* (a command another program runs, or a shell string)
    start pytest anywhere."""
    return bool(_PYTEST_WORD.search(" ".join(words)))


def _python_runs_pytest(args: list[str]) -> bool:
    for i, arg in enumerate(args):
        if arg == "-m":
            return args[i + 1 : i + 2] in (["pytest"], ["py.test"])
        if arg.startswith("-m"):
            return arg[2:] in ("pytest", "py.test")
        if not arg.startswith("-"):
            return False
    return False


def _plugin_args(words: list[str]) -> list[str]:
    """The ``-p`` values of a pytest command: the words after its pytest."""
    start = None
    for i, word in enumerate(words):
        name = re.split(r"[=<>!~@\[ ]", word.rsplit("/", 1)[-1], maxsplit=1)[0]
        if name in _PYTEST or (
            word in ("pytest", "py.test") and i and words[i - 1] in ("-m",)
        ):
            start = i + 1
            break
        if word in ("-mpytest", "-mpy.test"):
            start = i + 1
            break
    if start is None:
        return []
    found = []
    args = words[start:]
    for i, arg in enumerate(args):
        if arg == "-p" and i + 1 < len(args):
            found.append(args[i + 1].strip())
        elif arg.startswith("-p") and len(arg) > 2 and not arg.startswith("--"):
            found.append(arg[2:].strip())
    return found


# pytest options that take the next word as their value (``--opt=value``
# and ``-xVALUE`` are one word).
_PYTEST_VALUE_OPTIONS = frozenset(
    {
        "-k",
        "-m",
        "-p",
        "-c",
        "-o",
        "-W",
        "-n",
        "-r",
        "--rootdir",
        "--confcutdir",
        "--basetemp",
        "--junitxml",
        "--junit-xml",
        "--junit-prefix",
        "--tb",
        "--maxfail",
        "--durations",
        "--durations-min",
        "--deselect",
        "--ignore",
        "--ignore-glob",
        "--log-file",
        "--log-level",
        "--log-cli-level",
        "--log-format",
        "--log-date-format",
        "--log-file-level",
        "--cov",
        "--cov-report",
        "--cov-config",
        "--cov-fail-under",
        "--ctrf",
        "--json-report-file",
        "--html",
        "--timeout",
        "--timeout-method",
        "--dist",
        "--maxprocesses",
        "--capture",
        "--import-mode",
        "--color",
        "--code-highlight",
        "--override-ini",
        "--report-log",
        "--doctest-glob",
        "--keyword",
        "--reruns",
        "--reruns-delay",
        "--randomly-seed",
        "--tx",
        "--rsyncdir",
        "--ds",
        "--dc",
        "--hypothesis-seed",
        "--hypothesis-profile",
        "--pyargs-root",
        "--inifile",
        "--config-file",
        "--continue-on-collection-errors-max",
        "--benchmark-json",
        "--alluredir",
        "--result-log",
        "--resultlog",
    }
)


def _program_start(words: list[str]) -> int | None:
    """Where a pytest command's own arguments start: after its ``pytest``
    word (a path ending in one, or ``-m pytest``), not a ``pytest==8`` pin
    that a runner installs."""
    for i, word in enumerate(words):
        if word.rsplit("/", 1)[-1] in _PYTEST:
            return i + 1
        if word in ("pytest", "py.test") and i and words[i - 1] == "-m":
            return i + 1
        if word in ("-mpytest", "-mpy.test"):
            return i + 1
    return None


def _path_args(words: list[str]) -> list[str]:
    """The paths a pytest command gives pytest (node ids cut at ``::``)."""
    start = _program_start(words)
    if start is None:
        return []
    found: list[str] = []
    args = words[start:]
    i = 0
    while i < len(args):
        arg = args[i]
        i += 1
        if arg == "--":
            found += [a.split("::", 1)[0] for a in args[i:] if a]
            break
        if arg.startswith("-"):
            if arg in _PYTEST_VALUE_OPTIONS:
                i += 1
            continue
        found.append(arg.split("::", 1)[0])
    return [path for path in found if path]


def _absolute(path: str, cwd: str | None, mount: str) -> str:
    """*path* made absolute, ``""`` when it cannot be known."""
    for name in ("${TEST_DIR}", "$TEST_DIR"):
        if path.startswith(name):
            path = mount + path[len(name) :]
    if "$" in path or "`" in path or path.startswith("~"):
        return ""
    if not path.startswith("/"):
        if cwd is None:
            return ""
        path = cwd.rstrip("/") + "/" + path
    return str(PurePosixPath(_normpath(path)))


def _normpath(path: str) -> str:
    parts: list[str] = []
    for part in path.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/" + "/".join(parts)


def _folder_after_cd(args: list[str], cwd: str | None, mount: str) -> str | None:
    """The folder a ``cd`` (or ``pushd``) with *args* moves to, None when it
    is not a literal path."""
    operands = [a for a in args if a not in ("-L", "-P", "-e", "-@", "--")]
    if len(operands) != 1 or operands[0] == "-":
        return None
    return _absolute(operands[0], cwd, mount) or None


def _after_options(args: list[str]) -> list[str]:
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--":
            return args[i:]
        if not arg.startswith("-"):
            return args[i:]
        i += 2 if arg in _RUNNER_VALUE_OPTIONS else 1
    return []


# -- a shell lexer: words after quote removal, split at operators -------------

_OPERATORS = ("&&", "||", ";;", "|&", ";", "&", "|", "(", ")")


def _commands(text: str) -> list[tuple[list[str], list[str]]]:
    """The simple commands of *text*: their words (quotes removed,
    redirections dropped) and the here-documents fed to each."""
    commands: list[tuple[list[str], list[str]]] = []
    words: list[str] = []
    waiting: list[list[str]] = []  # here-document bodies, in order
    current_bodies: list[str] = []

    def end() -> None:
        nonlocal words, current_bodies
        if words:
            commands.append((words, current_bodies))
        words = []
        current_bodies = []

    for kind, value in _lex(text):
        if kind == "word":
            if value in ("{", "}"):
                end()
            else:
                words.append(value)
        elif kind == "heredoc-start":
            waiting.append(current_bodies)
        elif kind == "heredoc":
            if waiting:
                waiting.pop(0).append(value)
        else:
            end()
    end()
    return commands


def _lex(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    buf: list[str] = []
    in_word = False
    skip_target = False
    heredocs: list[tuple[str, bool]] = []
    i, n = 0, len(text)

    def flush() -> None:
        nonlocal buf, in_word, skip_target
        if in_word:
            if skip_target:
                skip_target = False
            else:
                out.append(("word", "".join(buf)))
        buf = []
        in_word = False

    while i < n:
        c = text[i]
        if c == "\\" and i + 1 < n:
            if text[i + 1] == "\n":
                i += 2
                continue
            buf.append(text[i + 1])
            in_word = True
            i += 2
            continue
        if c == "'":
            j = text.find("'", i + 1)
            j = n if j < 0 else j
            buf.append(text[i + 1 : j])
            in_word = True
            i = j + 1
            continue
        if c == '"':
            j = i + 1
            part: list[str] = []
            while j < n and text[j] != '"':
                if text[j] == "\\" and j + 1 < n and text[j + 1] in '"\\$`\n':
                    part.append(text[j + 1])
                    j += 2
                    continue
                part.append(text[j])
                j += 1
            buf.append("".join(part))
            in_word = True
            i = j + 1
            continue
        if c == "#" and not in_word:
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if c in " \t\r":
            flush()
            i += 1
            continue
        if c == "\n":
            flush()
            out.append(("op", "\n"))
            i += 1
            for delimiter, tabs in heredocs:
                body: list[str] = []
                while i < n:
                    j = text.find("\n", i)
                    line = text[i : n if j < 0 else j]
                    i = n if j < 0 else j + 1
                    if (line.lstrip("\t") if tabs else line) == delimiter:
                        break
                    body.append(line)
                out.append(("heredoc", "\n".join(body)))
            heredocs = []
            continue
        if text.startswith("$((", i):
            depth, j = 0, i + 1
            while j < n:
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            buf.append(text[i : j + 1])
            in_word = True
            i = j + 1
            continue
        if text.startswith("$(", i):
            flush()
            out.append(("op", "$("))
            i += 2
            continue
        if c == "`":
            flush()
            out.append(("op", "`"))
            i += 1
            continue
        if text.startswith("<<", i) and not text.startswith("<<<", i):
            if in_word and "".join(buf).isdigit():
                buf, in_word = [], False
            flush()
            j = i + 2
            tabs = j < n and text[j] == "-"
            j += 1 if tabs else 0
            while j < n and text[j] in " \t":
                j += 1
            found = re.match(r"(['\"]?)([^\s'\"<>;&|()]+)\1", text[j:])
            if found:
                heredocs.append((found.group(2), tabs))
                out.append(("heredoc-start", found.group(2)))
                i = j + found.end()
            else:
                i = j
            continue
        if c in "<>" or (c == "&" and text.startswith(("&>", "&>>"), i)):
            if in_word and "".join(buf).isdigit():
                buf, in_word = [], False
            flush()
            j = i
            while j < n and text[j] in "<>&|":
                j += 1
            operator = text[i:j]
            i = j
            if (
                operator.endswith("&")
                and i < n
                and (text[i].isdigit() or text[i] == "-")
            ):
                while i < n and (text[i].isdigit() or text[i] == "-"):
                    i += 1
            elif operator.endswith(("(",)) or (i < n and text[i] == "("):
                # process substitution <(...) / >(...): its commands still run
                pass
            else:
                skip_target = True
            continue
        if c in ";&|()":
            flush()
            for operator in _OPERATORS:
                if text.startswith(operator, i):
                    out.append(("op", operator))
                    i += len(operator)
                    break
            continue
        buf.append(c)
        in_word = True
        i += 1
    flush()
    return out
