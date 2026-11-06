"""bench's verb set, the BenchFlow 0.8 spellings it translates, and next steps.

Rust's ``bench`` 0.8.2 (``docs/rust/cli.md`` at bench-rs 518009b0) and this
``bench`` share one verb set: ``bench new``, ``check``, ``run``, ``results``,
``view``, ``regrade``, ``branch``, ``import``, ``export``, ``doctor``, ``agents``
and ``explain``. Each runs the Python command that already did the
job (``bench new`` is ``bench tasks init``), except ``run``, ``results`` and
``agents``, which take Rust's arguments (``cli/new_verbs.py``), and
``explain``, which prints the pages of ``cli/explain.py``.

Every 0.8 spelling keeps working. One with a new spelling prints one note on
stderr naming it, with the flags renamed as Rust's ``coming-from-0.8.md``
table renames them where this ``bench`` has the flag. ``bench eval``,
``bench tasks`` or ``bench agent`` typed alone prints where their commands
went. A mistyped word (``bench compare``), an unknown command and an unknown
flag exit 2 with a next step.

This module is the one map all of that reads: ``cli/main.py`` (the help
groups, the notes, the maps, the next steps), ``tests/test_cli_verbs.py``
(the parity list against Rust 0.8.2) and the JSON errors, which take their
text and next step from :func:`known_word`, :func:`unknown_command_next`,
:func:`unknown_flag_next` and :func:`group_map`. Standard library only, and
imported only when a command runs or a usage error is printed:
``bench --version`` and ``bench --help`` never load it.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass

from benchflow._utils.command_hints import command_arg

# ``bench --help``'s groups, in Rust's order, then the Python-only commands.
TASKS = "Tasks"
RUNS = "Runs"
THIS_MACHINE = "This machine"
MORE = "More"
HELP_PANELS = (TASKS, RUNS, THIS_MACHINE, MORE)


@dataclass(frozen=True)
class Verb:
    """A verb Rust 0.8.2 and this ``bench`` share: its help group, what it runs."""

    name: str
    panel: str
    runs: str


VERBS: tuple[Verb, ...] = (
    Verb("new", TASKS, "bench tasks init"),
    Verb("check", TASKS, "bench tasks check"),
    Verb("import", TASKS, "bench tasks migrate"),
    Verb("run", RUNS, "bench eval run with Rust's flags; without -a, the oracle"),
    Verb("branch", RUNS, "bench eval branch"),
    Verb(
        "results",
        RUNS,
        "bench eval list (no job, or a folder of jobs), bench eval inspect "
        "(one job) or bench eval compare (two jobs)",
    ),
    Verb("view", RUNS, "bench eval view"),
    Verb("regrade", RUNS, "bench eval regrade"),
    Verb("export", RUNS, "bench tasks export"),
    Verb("doctor", THIS_MACHINE, "bench doctor"),
    Verb("agents", THIS_MACHINE, "bench agent list, or bench agent show <name>"),
    Verb("explain", THIS_MACHINE, "bench explain (a page per failures.py cause)"),
    Verb(
        "cache",
        THIS_MACHINE,
        "bench cache list or clean: cached layers, their build records, content "
        "snapshots",
    ),
)

# Rust 0.8.2's commands this bench does not have, and why (tests/test_cli_verbs.py
# lists Rust's commands, from docs/rust/cli.md at bench-rs 518009b0). A Rust-only verb is
# added only when a Python command already does the same thing.
NOT_PORTED: dict[str, str] = {
    "test": (
        "no Python command runs the oracle, an empty run and exploit probes side "
        "by side; run `bench run <task>` (the oracle) and `bench run <task> -a nop`"
    ),
    "shell": "no Python command opens a shell in a task's sandbox and grades it",
    "handoff": "no Python command hands a trial's agent session to its CLI",
    "agents check": (
        "`bench doctor --agent-start <agent>` installs an agent in a fresh sandbox "
        "and opens its ACP connection without a prompt"
    ),
    "schema": "Python has no job files to describe; `bench run` takes flags",
    "completions": (
        "`bench --show-completion` prints the completion script and "
        "`bench --install-completion` installs it"
    ),
}

# The default agent of `bench eval run` (benchflow.evaluation.DEFAULT_AGENT;
# a test keeps them equal). `bench run` runs the oracle instead.
EVAL_RUN_DEFAULT_AGENT = "claude-agent-acp"

# What a 0.8 command line becomes. ``same`` spellings take the same
# arguments; the others are translated flag by flag below.
_SAME_ARGS: dict[tuple[str, str], str] = {
    ("eval", "view"): "view",
    ("eval", "regrade"): "regrade",
    ("eval", "branch"): "branch",
    ("eval", "list"): "results",
    ("tasks", "init"): "new",
    ("tasks", "check"): "check",
    ("tasks", "migrate"): "import",
    ("tasks", "export"): "export",
    ("agent", "list"): "agents",
    ("agent", "show"): "agents",
}

# `bench eval inspect` and `compare` flags `bench results` has, by whether they
# take a value. Their other flags (--trajectories, --verifier, --out, --csv,
# --labels, --vary, --on-mismatch, --k, --solve-threshold) have no spelling.
_RESULTS_FLAGS: dict[str, bool] = {
    "--json": False,
    "--include-controls": False,
    "--attempts": True,
    "--by": True,
}

# `bench eval run` flags `bench run` has: old name -> (new name, takes a value),
# as Rust's coming-from-0.8.md table renames them.
_RUN_FLAGS: dict[str, tuple[str, bool]] = {
    "--agent": ("--agent", True),
    "--model": ("--model", True),
    "--sandbox": ("--sandbox", True),
    "--concurrency": ("--concurrency", True),
    "--jobs-dir": ("--jobs-dir", True),
    "--include": ("--include", True),
    "--exclude": ("--exclude", True),
    "--timeout-multiplier": ("--timeout-multiplier", True),
    "--extra-instruction": ("--extra-instruction", True),
    "--fail-under": ("--fail-under", True),
    "--dataset": ("--dataset", True),
    "-d": ("--dataset", True),
    "--registry": ("--registry", True),
    "--fresh": ("--fresh", False),
    "--dry-run": ("--dry-run", False),
    "--quiet": ("--quiet", False),
    "--job-name": ("--job", True),
    "--n-tasks": ("--limit", True),
    "--agent-env": ("--env", True),
    "--reasoning-effort": ("--effort", True),
    "--freeze-workspace": ("--save-outputs", False),
    "--verifier-sandbox": ("--verifier-sandbox", True),
    # The short flags both verbs take (dogfood p5-11).
    "-a": ("--agent", True),
    "-m": ("--model", True),
    "-s": ("--sandbox", True),
    "-j": ("--concurrency", True),
    "-i": ("--include", True),
    "-x": ("--exclude", True),
    "-n": ("--limit", True),
    "--limit": ("--limit", True),
}
# Every 0.8 flag of `bench eval run` is a `bench run` flag too, the renamed
# ones as aliases (new_verbs.RunCommand), so no flag needs a pointer here.

# The visible commands of 0.8's groups, in their help order (a test keeps
# them equal to the real groups), for the maps below.
GROUP_COMMANDS_08: dict[str, tuple[str, ...]] = {
    "eval": (
        "smoke",
        "run",
        "adopt",
        "compare-lift",
        "score",
        "regrade",
        "list",
        "metrics",
        "view",
        "continue",
        "branch",
        "inspect",
        "compare",
        "resume",
        "branches",
    ),
    "tasks": (
        "generate",
        "list-sources",
        "init",
        "check",
        "overlap",
        "migrate",
        "normalize",
        "export",
        "snapshot-hf",
        "digest",
    ),
    "agent": ("list", "show"),
}


def new_spelling(group: str, command: str) -> str | None:
    """The verb a 0.8 ``bench <group> <command>`` is now, or None if it has none."""
    if (group, command) in _SAME_ARGS:
        return _SAME_ARGS[(group, command)]
    if group == "eval" and command in ("run", "create"):
        return "run"
    if group == "eval" and command in ("inspect", "compare"):
        return "results"
    return None


def group_map(group: str) -> str:
    """What ``bench <group>`` alone prints: each of its commands and its spelling now."""
    moved: dict[str, list[str]] = {}
    stay: list[str] = []
    names = list(GROUP_COMMANDS_08[group])
    if group == "eval":
        # The hidden alias, listed with the command it is.
        names.insert(names.index("run") + 1, "create")
    for name in names:
        verb = new_spelling(group, name)
        if verb is None:
            stay.append(name)
        else:
            moved.setdefault(verb, []).append(name)
    rows = [
        (f"{group} {', '.join(old)}", f"bench {verb}") for verb, old in moved.items()
    ]
    if group == "eval":
        rows = [
            (
                old,
                new + " (without -a it runs the oracle; 0.8 ran claude-agent-acp)"
                if new == "bench run"
                else new,
            )
            for old, new in rows
        ]
    width = max(len(old) for old, _ in rows)
    lines = [f"`bench {group}` was BenchFlow 0.8's. Where its commands went:"]
    lines += [f"  {old:<{width}}  {new}" for old, new in rows]
    if stay:
        lines.append(
            f"  {group} {', '.join(stay)}: only here, as bench {group} <command>"
        )
    lines.append(
        "Every one still works as before; the ones with a new spelling print it."
    )
    lines.append(f"  next: `bench {group} --help` lists its commands")
    return "\n".join(lines) + "\n"


def group_typed_alone(words: Sequence[str]) -> str | None:
    """The 0.8 group typed with no command (``bench eval``, ``bench tasks --x``)."""
    if not words or words[0] not in GROUP_COMMANDS_08:
        return None
    rest = words[1:]
    if not rest or (rest[0].startswith("-") and rest[0] != "--help"):
        return words[0]
    return None


# Words people type for a command bench has under another name, or for
# something bench is not (Rust's main.rs `known_word`).
_KNOWN_WORDS: dict[str, str] = {
    "compare": "`bench results <job> <job>` compares two jobs",
    "clean": "`bench sandbox cleanup` removes the sandboxes BenchFlow left",
    "list": "`bench results` lists jobs and their scores",
    "ls": "`bench results` lists jobs and their scores",
    "jobs": "`bench results` lists jobs and their scores",
    "show": "`bench results` lists jobs and their scores",
    "status": "`bench results` lists jobs and their scores",
    "open": "`bench view <job>` opens a job in your browser",
    "init": "`bench new <name>` creates a task",
    "version": "`bench --version`",
    "update": "install it again: uv tool install --python 3.12 --upgrade benchflow",
    "upgrade": "install it again: uv tool install --python 3.12 --upgrade benchflow",
    "self": "install it again: uv tool install --python 3.12 --upgrade benchflow",
    "login": "`bench agents <name>` says how that agent logs in; `bench doctor` lists the logins it finds",
    "uninstall": "remove the program: uv tool uninstall benchflow",
    "remove": "remove the program: uv tool uninstall benchflow",
}


def known_word(word: str) -> tuple[str, str] | None:
    """A mistyped command word: the error and its next step, or None."""
    next_step = _KNOWN_WORDS.get(word)
    if next_step is None:
        return None
    return f"bench has no `{word}` command", next_step


def unknown_command_next(command_path: str) -> str:
    """The next step after ``No such command`` under ``command_path`` (``bench eval``)."""
    if command_path.strip() == "bench":
        return "`bench --help` lists the commands"
    return f"`{command_path} --help` lists its commands"


def unknown_flag_next(command_path: str, flag: str) -> str:
    """The next step after an unknown flag of ``command_path`` (``bench run``)."""
    del flag  # every bench eval run flag is a bench run flag (RunCommand)
    if command_path.split() == ["bench", "run"]:
        # --help shows the short page; --help-all every run setting.
        return "`bench run --help-all` lists its flags"
    return f"`{command_path} --help` lists its flags"


def bad_value_next(command_path: str) -> str:
    """The next step after a flag value Click refused (``bench run -j abc``)."""
    if command_path.split() == ["bench", "run"]:
        return "`bench run --help-all` says what each flag takes"
    return f"`{command_path} --help` says what each flag takes"


# --- the note a 0.8 spelling prints -----------------------------------------


def glob_bracket_end(value: str, index: int) -> int | None:
    """Where the fnmatch bracket expression opening at ``value[index]`` ends.

    fnmatch allows ! for negation and a literal ] first in a bracket
    expression; an unmatched [ is just a literal [ (None).
    """
    closing = index + 1
    if closing < len(value) and value[closing] == "!":
        closing += 1
    if closing < len(value) and value[closing] == "]":
        closing += 1
    closing = value.find("]", closing)
    return None if closing == -1 else closing


def _one_task_pattern(value: str) -> str:
    """A ``bench eval run`` filter value as one ``bench run`` pattern.

    ``bench eval run`` takes each value whole; ``bench run`` splits it on
    commas outside brackets and strips each piece. So such a comma becomes
    ``[,]``, an unmatched ``[`` becomes ``[[]`` (a later ``]`` would close it)
    and edge whitespace is bracketed: fnmatch then matches the same names
    as the old glob did. Not the old exact match, which also takes the value
    as a literal name (``a,b``, never ``a[,]b``): a value this changes keeps
    its ``bench eval run`` line.
    """
    if "," not in value and value == value.strip():
        return value
    out: list[str] = []
    index = 0
    while index < len(value):
        char = value[index]
        if char == "[":
            end = glob_bracket_end(value, index)
            if end is not None:
                out.append(value[index : end + 1])
                index = end + 1
                continue
            out.append("[[]")
        elif char == "," or (
            char.isspace() and not (value[:index].strip() and value[index:].strip())
        ):
            out.append(f"[{char}]")
        else:
            out.append(char)
        index += 1
    return "".join(out)


_ENV_PAIR = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=")
# The flag that takes a variable name, not KEY=VALUE, and the name as people
# write one: a lower-case or mixed word may be a token (ghp_..., hex), and a
# bare word under --agent-env is a value typed without its name.
_NAME_FLAGS = frozenset({"--allow-task-env"})
_ENV_NAME = re.compile(r"[A-Z_][A-Z0-9_]*")


def _is_variable_name(value: str) -> bool:
    """An upper-case variable name that is not shaped like a credential.

    A provider key with a known prefix (``AKIA...``, ``sk-...``, ``ghp_...``,
    as the trajectory redactor knows them), a long unbroken token (16 or
    more characters with no ``_``) or a word that is half digits is hidden even
    when it is all upper case (review L-R2-3).
    """
    if not _ENV_NAME.fullmatch(value):
        return False
    if len(value) >= 16 and "_" not in value:
        return False
    if sum(c.isdigit() for c in value) * 2 >= len(value):
        return False  # half or more digits: a key, not a name
    from benchflow.trajectories.types import redact_trajectory_text_with_count

    return redact_trajectory_text_with_count(value)[1] == 0


_SECRET_FLAG = re.compile(
    r"(token|key|secret|password|credentials?|apikey)$", re.IGNORECASE
)


def _shown(flag: str | None, value: str) -> str:
    """A typed value as a note shows it: never a value that may be a secret.

    An environment value becomes a reference to its variable (``K="$K"``),
    any other ``KEY=VALUE`` shows its key and hides its value
    (``KEY='<hidden>'``), and a bare variable name (``--allow-task-env
    HF_TOKEN``) shows as typed (review L1); a
    flag named like a credential, or an address with a login, a query or a
    fragment in it (a private raw link's ``?token=``, a presigned URL's
    signature), shows ``'<hidden>'``; anything else is quoted for the shell.
    """
    pair = _ENV_PAIR.match(value)
    env_flag = flag in ("--agent-env", "--env") or (
        flag is not None and flag.endswith("-env") and flag != "--source-env"
    )
    if pair and env_flag:
        return f'{pair.group(1)}="${pair.group(1)}"'
    if pair:
        # A KEY=VALUE given anywhere (--allow-task-env NAME=secret,
        # --source-env-arg n=5): the key, never the value.
        return f"{pair.group(1)}='<hidden>'"
    if env_flag and not (flag in _NAME_FLAGS and _is_variable_name(value)):
        # Not a variable name: maybe a value typed where a name belongs.
        return "'<hidden>'"
    if flag is not None and _SECRET_FLAG.search(flag.lstrip("-").replace("-", "")):
        return "'<hidden>'"
    if "://" in value:
        rest = value.split("://", 1)[1]
        if "@" in rest.split("/", 1)[0] or "?" in rest or "#" in rest:
            return "'<hidden>'"
    if flag in ("--jobs-dir", "--registry"):
        return command_arg(value, "<path>")
    return shlex.quote(value)


def _split(word: str) -> tuple[str, str | None]:
    if word.startswith("--") and "=" in word:
        flag, value = word.split("=", 1)
        return flag, value
    return word, None


@dataclass
class _Translation:
    words: list[str]
    remarks: list[str]
    unspelled: list[str]


def _typed(words: Sequence[str], *flags: str) -> bool:
    return any(word.split("=", 1)[0] in flags for word in words)


def _translate_run(rest: Sequence[str]) -> _Translation:
    """``bench eval run``'s arguments in ``bench run``'s spelling."""
    out: list[str] = []
    remarks: list[str] = []
    unspelled: list[str] = []
    repo = path = ref = None
    target: str | None = None
    agent = False
    dataset = False
    env_shown = False
    words = list(rest)
    i = 0
    while i < len(words):
        word = words[i]
        i += 1
        if word == "--":
            unspelled.extend(w for w in words[i:])
            break
        flag, inline = _split(word)
        if not flag.startswith("-"):
            unspelled.append(word)
            continue

        def value(flag: str = flag, inline: str | None = inline) -> str:
            nonlocal i
            if inline is not None:
                return inline
            if i < len(words):
                i += 1
                return words[i - 1]
            unspelled.append(flag)
            return ""

        if flag in _RUN_FLAGS:
            new, takes = _RUN_FLAGS[flag]
            if new == "--agent":
                agent = True
            dataset = dataset or new == "--dataset"
            out.append(new)
            if new in ("--include", "--exclude"):
                pattern = value()
                # '' selects no task in 0.8 and bench run would drop it. A
                # value bench run would split or strip has no spelling either:
                # 0.8 also matches it as an exact name, which escaping changes.
                if (not pattern and flag not in unspelled) or (
                    _one_task_pattern(pattern) != pattern
                ):
                    unspelled.append(f"{flag} {shlex.quote(pattern)}")
                out.append(_shown(new, pattern))
            elif takes:
                shown = _shown(new, value())
                env_shown = env_shown or (new == "--env" and '="$' in shown)
                out.append(shown)
        elif flag == "--tasks-dir":
            target = command_arg(value(), "<task folder>")
        elif flag == "--source-repo":
            repo = value()
        elif flag == "--source-path":
            path = value()
        elif flag == "--source-ref":
            ref = value()
        elif flag in ("--agent-idle-timeout", "--idle-timeout"):
            seconds = value()
            out += [
                "--idle-timeout",
                shlex.quote(f"{seconds}s" if seconds.isdecimal() else seconds),
            ]
        elif flag == "--harness":
            # One set of values on both verbs: acp or native (dogfood p5-9).
            out += ["--harness", shlex.quote(value())]
        elif flag == "--skill-mode":
            mode = value()
            if mode == "with-skill":
                out.append("--task-skills")
            elif mode != "no-skill":
                out += ["--skill-mode", shlex.quote(mode)]
        else:
            # bench run takes every other bench eval run flag as it is
            # (new_verbs.RunCommand): the flag and its value, if it takes one
            # (bench eval run has no positionals).
            out.append(flag if inline is None else f"{flag}={_shown(flag, inline)}")
            if inline is None and i < len(words) and not words[i].startswith("-"):
                out.append(_shown(flag, words[i]))
                i += 1
    if repo is not None:
        spec = repo.strip("/")
        if path:
            spec += "/" + path.strip("/")
        if ref:
            spec += "@" + ref
        target = _shown(None, spec)
    if (path or ref) and repo is None:
        unspelled.append("--source-path" if path else "--source-ref")
    # A run config or a hosted environment names its own tasks; a matrix
    # still needs a task folder (review L-R2-5). All three name their agents.
    own_tasks = _typed(words, "--config", "--run-config", "--source-env")
    whole_run = own_tasks or _typed(words, "--matrix")
    if target is not None:
        out.insert(0, target)
    elif not dataset and not own_tasks:
        # No task was typed (DF-p1-first-run-4): the line names the slot, so
        # it is not a command that fails with "bench run needs a target".
        out.insert(0, "<task>")
    if not agent and not whole_run:
        out += ["--agent", EVAL_RUN_DEFAULT_AGENT]
        remarks.append(
            f"--agent {EVAL_RUN_DEFAULT_AGENT} is added: without --agent, bench eval "
            "run runs it and bench run runs the oracle"
        )
    if env_shown:
        remarks.append("set the variables it names first")
    return _Translation(out, remarks, unspelled)


def _translate_results(rest: Sequence[str]) -> _Translation:
    """``bench eval inspect`` and ``compare`` arguments in ``bench results``'s."""
    out: list[str] = []
    unspelled: list[str] = []
    words = list(rest)
    i = 0
    while i < len(words):
        word = words[i]
        i += 1
        flag, inline = _split(word)
        if word == "--":
            out.append("--")
            out.extend(command_arg(item, "<job dir>") for item in words[i:])
            break
        if not flag.startswith("-"):
            out.append(command_arg(word, "<job dir>"))
            continue
        if flag in _RESULTS_FLAGS:
            out.append(flag)
            if _RESULTS_FLAGS[flag]:
                if inline is not None:
                    out.append(shlex.quote(inline))
                elif i < len(words):
                    out.append(shlex.quote(words[i]))
                    i += 1
        else:
            unspelled.append(flag)
    return _Translation(out, [], unspelled)


def note_for(words: Sequence[str]) -> str | None:
    """The note a 0.8 spelling prints (``words`` as typed after ``bench``), or None.

    None for anything that is not a 0.8 spelling with a new one, and for its
    ``--help``, which prints the command's own help.
    """
    if len(words) < 2 or words[0] not in GROUP_COMMANDS_08 or "--help" in words:
        return None
    group, command, rest = words[0], words[1], list(words[2:])
    verb = new_spelling(group, command)
    if verb is None:
        return None
    old = f"bench {group} {command}"
    head = f"`{old}` is now `bench {verb}`"
    if (group, command) in _SAME_ARGS:
        return f"{head}, with the same arguments"
    if command in ("run", "create"):
        translation = _translate_run(rest)
    else:
        translation = _translate_results(rest)
    if translation.unspelled:
        # One short note (dogfood p5-4): the verb, never a paragraph about
        # the flags; the line runs as typed.
        flags = ", ".join(dict.fromkeys(translation.unspelled))
        return f"{head} ({flags} only as `{old}`)"
    line = " ".join(["bench", verb, *translation.words])
    note = f"{head}: {line}"
    if translation.remarks:
        note += " (" + "; ".join(translation.remarks) + ")"
    return note
