"""Structural decomposition of shell exec bodies into file-target observations.

Ported verbatim from BenchGuard (arXiv 2609.11028;
``src/benchflow/benchguard/exec_decompose.py``); "PR #N" in comments refers to
BenchGuard's pull requests.

The action recorder and the trajectory synthesizer both observe agent commands
as strings. Hashing them (the v0 behavior) made every ``cat > /pkg/x <<EOF``
one opaque Execute record, so protected-path writes through redirection,
heredocs, and ``cp``/``mv`` — the dominant channel in the corpus — were
invisible to the trace checker. This module extracts the file targets a shell
command manipulates, well enough to witness those writes, and reports honestly
when it cannot: an interpreter invocation, command substitution, or a
variable-valued redirect target marks the decomposition ``opaque`` so the
consumer can fail closed instead of treating the command as benign.

Deliberately dependency-free (stdlib only) and classification-free: mapping a
target path onto a ResourceClass is the caller's job, against the task
contract.
"""

from __future__ import annotations

import posixpath
import re
import shlex
from dataclasses import dataclass
from typing import Literal

TargetMode = Literal["write", "read"]

_MAX_TARGETS = 16
_MAX_REFERENCED_PATHS = 32
_MAX_NESTED_DEPTH = 3

# Wrappers that execute their trailing argv: strip and re-analyze the rest.
_WRAPPER_PROGRAMS = frozenset(
    {"sudo", "env", "nohup", "nice", "stdbuf", "timeout", "time", "command"}
)
# Interpreters and shells whose behavior we cannot see from the command line.
# ``sh -c`` / ``bash -c`` recurse into the -c string instead.
_OPAQUE_PROGRAMS = frozenset(
    {
        "python",
        "python2",
        "python3",
        "node",
        "perl",
        "ruby",
        "php",
        "make",
        "eval",
        "exec",
        "xargs",
        "source",
        ".",
    }
)
_SHELLS = frozenset({"sh", "bash", "zsh", "dash", "ksh"})
# dest is the last non-flag argument
_WRITE_LAST_ARG = frozenset({"cp", "mv", "install", "ln", "rsync", "scp"})
# every non-flag argument is mutated
_WRITE_ALL_ARGS = frozenset(
    {"tee", "touch", "mkdir", "rmdir", "rm", "truncate", "shred", "unlink", "patch"}
)
# first non-flag argument is a mode/owner spec, the rest are mutated
_WRITE_SKIP_FIRST_ARG = frozenset({"chmod", "chown", "chgrp", "setfacl"})
_READ_ALL_ARGS = frozenset(
    {
        "cat",
        "head",
        "tail",
        "less",
        "more",
        "grep",
        "egrep",
        "fgrep",
        "rg",
        "wc",
        "sort",
        "uniq",
        "cut",
        "strings",
        "xxd",
        "hexdump",
        "od",
        "base64",
        "md5sum",
        "sha1sum",
        "sha256sum",
        "diff",
        "cmp",
        "file",
        "stat",
        "readlink",
        "du",
        # Directory inspection is an observation: ``ls /solution`` is a
        # protected-state read, not an unparseable command.
        "ls",
        "find",
        "tree",
    }
)
# Short options that consume a SEPARATE following argument, per reader. Without
# this table `grep -A 3 notes.txt` counts "3" as a file operand; an attached
# value (`-n20`) consumes nothing extra. Only options that really take a value
# belong here — a spurious entry silently swallows a real file operand.
_READER_VALUE_SHORT: dict[str, str] = {
    "grep": "efmABCDd",
    "egrep": "efmABC",
    "fgrep": "efmABC",
    "rg": "efmABCgtT",
    "head": "nc",
    "tail": "nc",
    "sort": "kotSTm",
    "uniq": "fsw",
    "cut": "dfbc",
    "xxd": "lscg",
    "od": "AjNtwS",
    "hexdump": "nse",
    "base64": "w",
    "du": "dBXt",
    "ls": "IwT",
    "wc": "",
    "diff": "",
    "cmp": "in",
    "strings": "n",
    "file": "",
    "stat": "cf",
    "readlink": "",
    "less": "",
    "more": "",
    "cat": "",
    "tree": "LPI",
    "md5sum": "",
    "sha1sum": "",
    "sha256sum": "",
}
# Long options that consume a separate argument when written without ``=``.
_READER_VALUE_LONG: dict[str, frozenset[str]] = {
    "grep": frozenset(
        {
            "--regexp",
            "--file",
            "--max-count",
            "--after-context",
            "--before-context",
            "--context",
            "--include",
            "--exclude",
            "--exclude-dir",
            "--exclude-from",
            "--label",
            "--binary-files",
            "--devices",
            "--directories",
            "--group-separator",
        }
    ),
    "head": frozenset({"--lines", "--bytes"}),
    "tail": frozenset({"--lines", "--bytes"}),
    "sort": frozenset(
        {"--key", "--output", "--field-separator", "--temporary-directory"}
    ),
    "cut": frozenset({"--delimiter", "--fields", "--bytes", "--characters"}),
    "du": frozenset({"--max-depth", "--block-size", "--exclude"}),
    "ls": frozenset({"--ignore", "--width", "--time-style"}),
    "stat": frozenset({"--format", "--printf"}),
}
# Readers whose FIRST positional is a pattern, not a file — unless an option
# supplied the pattern instead.
_PATTERN_FIRST_READERS = frozenset({"grep", "egrep", "fgrep", "rg"})
_PATTERN_SUPPLYING_SHORT = frozenset("ef")
_PATTERN_SUPPLYING_LONG = frozenset({"--regexp", "--file"})
# ``find [-H|-L|-P] [path...] [expression]``: the roots are the positionals
# before the first predicate; everything after is expression syntax.
_ROOTS_THEN_PREDICATES = frozenset({"find"})

# Imperative verbs some ACP shims prepend to the rendered command line.
_TITLE_VERB_PREFIXES = frozenset({"Run", "Running", "Execute", "Executing"})

# Programs whose path-shaped arguments are data, not file operations.
_BENIGN_PROGRAMS = frozenset(
    {
        "echo",
        "printf",
        "pwd",
        "cd",
        "true",
        "false",
        "test",
        "[",
        "which",
        "whereis",
        "type",
        "id",
        "whoami",
        "uname",
        "hostname",
        "date",
        "sleep",
        "ps",
        "df",
        "free",
        "export",
        "set",
        "unset",
        "alias",
        "kill",
        "wait",
    }
)

_ABSOLUTE_PATH_RE = re.compile(r"(?<![\w@:.-])/(?:[\w.+-]+/)*[\w.+-]+")
# ``2>``, ``>>``, ``&>``, ``>|`` … with an optionally attached target
_REDIRECT_RE = re.compile(r"(?:\d+|&)?(>>|>\|?|<)(?!<)(.*)", re.DOTALL)
_FD_DUP_RE = re.compile(r"&\d+-?$")
_ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
# Egress expressed through exec. `_network_record` only fires for network tool
# *kinds*, so `curl https://…` and `python3 -c "urlopen('http://…')"` produced
# an Execute record with the destination sitting in `command_preview` and no
# NetworkRequest anywhere -- `_check_forbidden_network` returns immediately
# unless `action_class == "NetworkRequest"`, so it never saw them. Scanning the
# whole command text catches the interpreter cases too, which no operand table
# can reach because the interpreter is opaque by design.
_URL_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.\-]*://[^\s'\"`<>|;)\\]+")
_MAX_NETWORK_TARGETS = 16
# Schemes that are not egress: reading a file by URL is a file operation, and
# these appear in argv far more often than they appear as destinations.
_NON_EGRESS_SCHEMES = frozenset({"file", "data", "unix", "fd"})
# Programs whose job is to fetch. A URL is only a destination when something
# is going to dial it.
_NETWORK_PROGRAMS = frozenset(
    {
        "curl",
        "wget",
        "aria2c",
        "http",
        "https",
        "httpie",
        "xh",
        "nc",
        "ncat",
        "netcat",
        "telnet",
        "ftp",
        "lftp",
        "socat",
    }
)
# Fetch idioms inside an interpreter body. The interpreter itself is opaque,
# so the call is the only evidence that a URL in its source is a destination
# rather than a string.
# Loopback is not egress: it never leaves the sandbox. An agent curling the
# dev server it just started (`http://127.0.0.1:3000/inventory`) is doing the
# task, and counting it as a forbidden-network witness turned an honest
# vector-exposed replay cell into AgentViolation.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]"})

# Every program the grammar can act on.
_KNOWN_PROGRAMS = (
    _WRAPPER_PROGRAMS
    | _OPAQUE_PROGRAMS
    | _SHELLS
    | _WRITE_LAST_ARG
    | _WRITE_ALL_ARGS
    | _WRITE_SKIP_FIRST_ARG
    | _READ_ALL_ARGS
    | _NETWORK_PROGRAMS
    | _BENIGN_PROGRAMS
    | {"dd", "sed", "tar", "unzip", "sqlite3"}
)
# Programs a stripped title verb may vouch for. ``_KNOWN_PROGRAMS`` was used
# here first, and it contains the benign builtins -- ``test``, ``set``,
# ``type``, ``date``, ``free``, ``kill``, ``wait``, ``true``, ``false``,
# ``which``, ``id``, ``sleep`` -- which are also English words, so a prose
# plan-step title ("Run set up the environment") promoted to a command. The
# builtins never have file targets anyway, so excluding them loses nothing.
_TITLE_PROMOTABLE_PROGRAMS = _KNOWN_PROGRAMS - _BENIGN_PROGRAMS
# What a real command line carries that a sentence does not: a flag, a path
# or URL (any ``/``), an assignment or variable, a pipe/separator/redirect, or
# quoting. ``rm the stale outputs`` has none of these; ``rm -rf outputs``,
# ``cat /app/secret.txt`` and ``curl https://answers.test/gold`` do. Checked
# over the remainder after the program.
_SHELL_SHAPED_RE = re.compile(
    r"(?:^|\s)(?:-{1,2}\w|~(?:/|\s|$)|[\w-]+=\S|\$[\w({])"
    r"|[/|;<>&]"
    r"|['\"`]"
)
_FETCH_IDIOMS = (
    "urlopen",
    "urllib.request",
    "urllib3",
    "requests.get",
    "requests.post",
    "requests.put",
    "requests.request",
    "requests.head",
    "httpx.",
    "aiohttp",
    "http.client",
    "socket.connect",
    "fetch(",
    "axios",
    "net/http",
    "Net::HTTP",
    "HttpURLConnection",
    "URLSession",
    "file_get_contents",
    "curl_exec",
)


@dataclass(frozen=True)
class ExecTarget:
    path: str
    mode: TargetMode
    mechanism: str


@dataclass(frozen=True)
class ExecDecomposition:
    targets: tuple[ExecTarget, ...]
    opaque: bool
    referenced_paths: tuple[str, ...]
    truncated: bool = False
    network_targets: tuple[str, ...] = ()
    # Loopback URLs the command dials. Not egress (they never leave the
    # sandbox), but a declared protected service route lives behind one --
    # ``http://localhost:9005/_admin/state`` -- so they are kept, separately,
    # for the contract to classify.
    loopback_targets: tuple[str, ...] = ()

    @property
    def write_targets(self) -> tuple[ExecTarget, ...]:
        return tuple(target for target in self.targets if target.mode == "write")


@dataclass
class _Scan:
    """Mutable accumulation shared across nested decompositions."""

    targets: list[ExecTarget]
    opaque: bool = False
    truncated: bool = False

    def add(self, path: str, mode: TargetMode, mechanism: str, cwd: str | None) -> None:
        resolved = _resolve_path(path, cwd)
        if resolved is None:
            # A target we saw being written but cannot resolve (variable,
            # substitution) is exactly what must not silently disappear.
            self.opaque = True
            return
        if resolved == "/dev" or resolved.startswith("/dev/"):
            # Sink devices (`2>/dev/null`) are not file objects; recording
            # them produced unclassifiable Write targets that tripped the
            # I5 unknown-object rule on every honest verifier command.
            return
        target = ExecTarget(path=resolved, mode=mode, mechanism=mechanism)
        if target in self.targets:
            return
        if len(self.targets) >= _MAX_TARGETS:
            self.truncated = True
            return
        self.targets.append(target)


def decompose_exec_command(
    command: str, *, cwd: str | None = None
) -> ExecDecomposition:
    """Extract the file targets *command* reads and writes.

    ``cwd`` resolves relative targets; without it they are kept as-is (the
    container-root convention used elsewhere in the evidence pipeline).
    """

    command = _strip_title_verb(command)
    scan = _Scan(targets=[])
    _decompose_into(command, cwd=cwd, scan=scan, depth=0)
    referenced = _referenced_paths(command)
    return ExecDecomposition(
        targets=tuple(scan.targets),
        opaque=scan.opaque,
        referenced_paths=referenced,
        truncated=scan.truncated,
        network_targets=_network_targets(command),
        loopback_targets=_loopback_targets(command),
    )


def _strip_title_verb(command: str) -> str:
    """Drop a single imperative verb some ACP shims prepend to the command.

    ``Run sqlite3 /data/x.db`` / ``Running curl http://…`` -- but only when the
    next token is a program the grammar recognizes, so a file literally named
    ``Run`` (or a bare ``Run``) is never mistaken for the prefix. Without this
    the program parses as ``Run`` (opaque) and the real command's protected
    reads and egress are invisible.

    The text reaching this function through the title path is a tool-call
    *title*, which is prose as often as it is a command, and promoting prose
    fabricates evidence: ``Run rm the stale outputs`` decomposed into Write
    targets ``/app/the``, ``/app/stale``, ``/app/outputs`` -- actions the
    agent never took, which under a protected root would have produced an
    AgentViolation. So a title promotes only when (a) the next token is a
    program that is unmistakably a program (``_TITLE_PROMOTABLE_PROGRAMS``:
    not the English-word builtins) and (b) the remainder carries something a
    sentence does not -- a flag, a path, a redirect, a pipe, quoting
    (``_SHELL_SHAPED_RE``). ``pip install numpy`` in a title therefore stays
    unpromoted and parses opaque: an unvouched title is allowed to under-
    attribute, never to invent. Guards benchguard PR #36.
    """

    stripped = command.lstrip()
    verb, sep, rest = stripped.partition(" ")
    if verb not in _TITLE_VERB_PREFIXES or not sep:
        return command
    rest = rest.lstrip()
    next_token, _, remainder = rest.partition(" ")
    if posixpath.basename(next_token) not in _TITLE_PROMOTABLE_PROGRAMS:
        return command
    if not _SHELL_SHAPED_RE.search(remainder.strip()):
        return command
    return rest


def _network_targets(command: str) -> tuple[str, ...]:
    """Destination URLs the command actually dials, deduplicated in order.

    Scanning the raw command text was wrong twice over, and the second way
    cost a `honest_safe` RQ3 replay cell its verdict. A heredoc body is data:
    an honest MusicXML writer emitting
    ``<!DOCTYPE ... "http://www.musicxml.org/dtds/partwise.dtd">`` into a file
    was recorded as egress and, under a NoEgress contract, flipped the run to
    AgentViolation. One line of a 146-call trajectory. A URL in a comment or a
    config string is no more a request than that one was.

    So the scan runs over the splitter's segments, which already exclude
    heredoc bodies, and only where the segment names a fetcher or carries a
    fetch idiom. That is the discipline ``_diff_content_paths`` applies to
    paths and this function originally did not: witness the operand, never
    invent one from surrounding text.
    """

    segments, _ = _split_segments(command)
    seen: list[str] = []
    for segment in segments:
        if not _segment_dials(segment):
            continue
        for match in _URL_RE.finditer(segment):
            url = match.group(0).rstrip(".,")
            scheme = url.split("://", 1)[0].lower()
            if scheme in _NON_EGRESS_SCHEMES or url in seen or _is_loopback(url):
                continue
            seen.append(url)
            if len(seen) >= _MAX_NETWORK_TARGETS:
                return tuple(seen)
    return tuple(seen)


def _loopback_targets(command: str) -> tuple[str, ...]:
    """The loopback URLs a fetching segment dials, same discipline as egress."""

    segments, _ = _split_segments(command)
    seen: list[str] = []
    for segment in segments:
        if not _segment_dials(segment):
            continue
        for match in _URL_RE.finditer(segment):
            url = match.group(0).rstrip(".,")
            scheme = url.split("://", 1)[0].lower()
            if scheme in _NON_EGRESS_SCHEMES or url in seen or not _is_loopback(url):
                continue
            seen.append(url)
            if len(seen) >= _MAX_NETWORK_TARGETS:
                return tuple(seen)
    return tuple(seen)


def is_loopback_host(host: str) -> bool:
    """Whether *host* (no userinfo or port) addresses the sandbox itself.

    The one definition of loopback: any ``127/8`` spelling counts, not a
    literal table. Shared with the runtime-verification resource matcher so
    producer and consumer cannot disagree on what loopback means -- a
    ``127.0.0.2`` dial used to be kept out of the egress targets here while
    the matcher over there failed to fold it to ``localhost``, landing the
    request in neither check (benchguard PR #36 review).
    """

    host = host.lower()
    return host in _LOOPBACK_HOSTS or host.startswith("127.")


def _is_loopback(url: str) -> bool:
    """Whether *url* addresses the sandbox itself rather than the outside."""

    _, _, rest = url.partition("://")
    host = rest.split("/", 1)[0].rsplit("@", 1)[-1]
    if host.startswith("["):  # bracketed IPv6
        host = host[: host.find("]") + 1]
    elif host.count(":") == 1:
        host = host.rsplit(":", 1)[0]
    return is_loopback_host(host)


def _segment_dials(segment: str) -> bool:
    """Whether this segment invokes a fetcher or carries a fetch idiom."""

    if any(idiom in segment for idiom in _FETCH_IDIOMS):
        return True
    try:
        tokens = shlex.split(segment, posix=True)
    except ValueError:
        return False
    while tokens and _ASSIGNMENT_RE.match(tokens[0]):
        tokens = tokens[1:]
    while tokens and posixpath.basename(tokens[0]) in _WRAPPER_PROGRAMS:
        tokens = tokens[1:]
        while tokens and tokens[0].startswith("-"):
            tokens = tokens[1:]
    return bool(tokens) and posixpath.basename(tokens[0]) in _NETWORK_PROGRAMS


def _decompose_into(command: str, *, cwd: str | None, scan: _Scan, depth: int) -> None:
    if depth > _MAX_NESTED_DEPTH:
        scan.opaque = True
        return
    segments, segment_opaque = _split_segments(command)
    if segment_opaque:
        scan.opaque = True
    for segment in segments:
        _analyze_segment(segment, cwd=cwd, scan=scan, depth=depth)


def _analyze_segment(segment: str, *, cwd: str | None, scan: _Scan, depth: int) -> None:
    try:
        tokens = shlex.split(segment, posix=True)
    except ValueError:
        scan.opaque = True
        return
    if not tokens:
        return

    words = _extract_redirects(tokens, cwd=cwd, scan=scan)

    # VAR=val prefixes, then wrapper programs, then the effective program.
    while words and _ASSIGNMENT_RE.match(words[0]):
        words = words[1:]
    while words and posixpath.basename(words[0]) in _WRAPPER_PROGRAMS:
        program = posixpath.basename(words[0])
        words = words[1:]
        if program == "timeout":  # timeout [-s SIG] DURATION cmd…
            while words and words[0].startswith("-"):
                words = words[1:] if words[0] != "-s" else words[2:]
            words = words[1:]
        else:
            while words and (
                words[0].startswith("-") or _ASSIGNMENT_RE.match(words[0])
            ):
                words = words[1:]
    if not words:
        return
    program = posixpath.basename(words[0])
    args = words[1:]

    if program in _SHELLS:
        nested = _dash_c_argument(args)
        if nested is not None:
            _decompose_into(nested, cwd=cwd, scan=scan, depth=depth + 1)
        elif args:
            scan.opaque = True  # shell running a script file
        return
    if program in _OPAQUE_PROGRAMS:
        scan.opaque = True
        return
    if _contains_substitution(segment):
        scan.opaque = True

    positional = [arg for arg in args if not arg.startswith("-")]
    if program in _WRITE_LAST_ARG:
        if positional:
            scan.add(positional[-1], "write", program, cwd)
            for source in positional[:-1]:
                scan.add(source, "read", program, cwd)
    elif program in _WRITE_ALL_ARGS:
        for arg in positional:
            scan.add(arg, "write", program, cwd)
    elif program in _WRITE_SKIP_FIRST_ARG:
        for arg in positional[1:]:
            scan.add(arg, "write", program, cwd)
    elif program == "dd":
        for arg in args:
            if arg.startswith("of="):
                scan.add(arg[3:], "write", "dd", cwd)
            elif arg.startswith("if="):
                scan.add(arg[3:], "read", "dd", cwd)
    elif program == "sed":
        script_files, files, in_place = _sed_operands(args)
        for script_file in script_files:
            scan.add(script_file, "read", "sed-f", cwd)
        mode: TargetMode = "write" if in_place else "read"
        for arg in files:
            scan.add(arg, mode, "sed-i" if in_place else "sed", cwd)
    elif program == "tar":
        _analyze_tar(args, cwd=cwd, scan=scan)
    elif program == "unzip":
        _analyze_unzip(positional, args, cwd=cwd, scan=scan)
    elif program == "sqlite3":
        # `sqlite3 <db> "<sql>"`: the first operand is the database file; the
        # SQL is a string, not a path. Opening a service's backing store
        # (ClawsBench `/data/slack.db`) is an observation of protected state
        # whatever the statement does, so it is recorded as a read.
        if positional:
            scan.add(positional[0], "read", "sqlite3", cwd)
    elif program in _READ_ALL_ARGS:
        files, extra_reads, reader_writes = _reader_operands(program, args)
        for arg in extra_reads:
            scan.add(arg, "read", f"{program}-f", cwd)
        for arg in reader_writes:
            scan.add(arg, "write", f"{program}-o", cwd)
        for arg in files:
            scan.add(arg, "read", program, cwd)
    elif program in _BENIGN_PROGRAMS:
        pass
    else:
        # Unknown program: redirects were already captured; path-shaped
        # arguments to a program we cannot model make the segment opaque so a
        # protected-path mention fails closed rather than vanishing.
        if any("/" in arg for arg in positional):
            scan.opaque = True


def _extract_redirects(tokens: list[str], *, cwd: str | None, scan: _Scan) -> list[str]:
    """Pull ``>``/``>>``/``<`` targets out of *tokens*, returning the rest."""

    words: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        match = _REDIRECT_RE.fullmatch(token)
        if match is None:
            words.append(token)
            index += 1
            continue
        operator, attached = match.group(1), match.group(2)
        if attached:
            target: str | None = attached
        elif index + 1 < len(tokens):
            target = tokens[index + 1]
            index += 1
        else:
            target = None
        index += 1
        if target is None or _FD_DUP_RE.fullmatch(target or ""):
            continue  # 2>&1 and friends have no file target
        if operator == "<":
            scan.add(target, "read", "redirect", cwd)
        else:
            scan.add(target, "write", "redirect", cwd)
    return words


def _analyze_tar(args: list[str], *, cwd: str | None, scan: _Scan) -> None:
    flags = "".join(arg for arg in args if arg.startswith("-")) + (
        args[0] if args and not args[0].startswith("-") else ""
    )
    extracting = "x" in flags
    for index, arg in enumerate(args):
        if arg in {"-C", "--directory"} and index + 1 < len(args):
            if extracting:
                scan.add(args[index + 1], "write", "tar-extract", cwd)
        elif arg in {"-f", "--file"} and index + 1 < len(args):
            scan.add(args[index + 1], "read" if extracting else "write", "tar", cwd)
    if extracting and not any(arg in {"-C", "--directory"} for arg in args):
        scan.add(".", "write", "tar-extract", cwd)


def _analyze_unzip(
    positional: list[str], args: list[str], *, cwd: str | None, scan: _Scan
) -> None:
    for index, arg in enumerate(args):
        if arg == "-d" and index + 1 < len(args):
            scan.add(args[index + 1], "write", "unzip", cwd)
    if positional:
        scan.add(positional[0], "read", "unzip", cwd)
    if not any(arg == "-d" for arg in args):
        scan.add(".", "write", "unzip", cwd)


def _dash_c_argument(args: list[str]) -> str | None:
    for index, arg in enumerate(args):
        if arg == "-c" and index + 1 < len(args):
            return args[index + 1]
    return None


# Reader options whose VALUE is a file the command writes: `sort -o out.txt`
# creates a file, so discarding the value would lose a real mutation channel.
_VALUE_IS_WRITE: frozenset[tuple[str, str]] = frozenset(
    {("sort", "o"), ("sort", "--output")}
)


def _find_roots(args: list[str]) -> list[str]:
    """``find``'s search roots: positionals before the first predicate.

    ``find [-H|-L|-P] [path...] [expression]`` — everything from the first
    predicate onward is expression syntax, so a generic positional sweep
    collected predicate *values* as if they were paths: ``find . -type f -exec
    rm {} +`` yielded ``f``, ``rm`` and ``{}`` as read targets. A bare ``find``
    with no root means the current directory.
    """

    roots: list[str] = []
    for arg in args:
        if arg.startswith("-"):
            if not roots and arg in {"-H", "-L", "-P"}:
                continue  # leading mode flags precede the roots
            break
        roots.append(arg)
    return roots or ["."]


def _reader_operands(
    program: str, args: list[str]
) -> tuple[list[str], list[str], list[str]]:
    """``(file operands, extra reads, writes)`` for a ``_READ_ALL_ARGS`` program.

    Replaces ``if "/" in arg``, which failed in both directions: ``cat
    reward.txt`` was never witnessed at all (a bare-filename read of a reward
    file, with ``cwd`` available and ``_resolve_path`` ready to resolve it),
    while ``grep -r 'a/b' /app`` invented a read of ``/app/a/b`` out of the
    *pattern*. ``grep '/tests/answer' .`` fabricated a VerifierOnly read the
    same way.

    The operand position is decided by each program's own grammar rather than
    by how an argument looks: options that take a separate value consume it,
    the grep family's first positional is the pattern unless ``-e``/``-f``
    supplied one, and ``find``'s roots stop at the first predicate. A ``-f``
    pattern file is a genuine read, so it comes back in the second list.
    """

    if program in _ROOTS_THEN_PREDICATES:
        return _find_roots(args), [], []

    value_short = _READER_VALUE_SHORT.get(program, "")
    value_long = _READER_VALUE_LONG.get(program, frozenset())
    pattern_from_option = False
    extra_reads: list[str] = []
    writes: list[str] = []
    positional: list[str] = []
    end_of_options = False
    index = 0
    while index < len(args):
        arg = args[index]
        if end_of_options or not arg.startswith("-") or arg == "-":
            positional.append(arg)
            index += 1
            continue
        if arg == "--":
            end_of_options = True
            index += 1
            continue
        if arg.startswith("--"):
            name, sep, value = arg.partition("=")
            if name in _PATTERN_SUPPLYING_LONG:
                pattern_from_option = True
                if name == "--file":
                    if sep:
                        extra_reads.append(value)
                    elif index + 1 < len(args):
                        index += 1
                        extra_reads.append(args[index])
                elif not sep:
                    index += 1
            elif name in value_long:
                value = value
                if not sep and index + 1 < len(args):
                    index += 1
                    value = args[index]
                if value and (program, name) in _VALUE_IS_WRITE:
                    writes.append(value)
            index += 1
            continue
        offset = 1
        while offset < len(arg):
            letter = arg[offset]
            attached = arg[offset + 1 :]
            if letter in _PATTERN_SUPPLYING_SHORT and program in _PATTERN_FIRST_READERS:
                pattern_from_option = True
                if letter == "f":
                    if attached:
                        extra_reads.append(attached)
                    elif index + 1 < len(args):
                        index += 1
                        extra_reads.append(args[index])
                elif not attached:
                    index += 1
                break
            if letter in value_short:
                value = attached
                if not value and index + 1 < len(args):
                    index += 1
                    value = args[index]
                if value and (program, letter) in _VALUE_IS_WRITE:
                    writes.append(value)
                break
            offset += 1
        index += 1

    if program in _PATTERN_FIRST_READERS and not pattern_from_option:
        return positional[1:], extra_reads, writes
    return positional, extra_reads, writes


def _sed_operands(args: list[str]) -> tuple[list[str], list[str], bool]:
    """Split sed's argv into ``(script files, data files, in-place)``.

    sed's own grammar decides which operand is a file, so nothing here has to
    guess from a string's shape. The previous shape test asked whether an
    argument "looks like a script", recognising only ``s/…`` and ``y/…`` — so
    every address form (``/re/d``, ``/re/p``, ``1,/re/d``, ``$d``) fell through
    and was recorded as a *file* the command read or wrote. ``sed -i '/tests/d'
    notes.txt`` therefore witnessed a write to ``/tests/d``, which classifies
    ``VerifierOnly``: a routine line-delete became a verifier mutation.

    The rule instead: ``-e``/``-f`` supply the script explicitly, and when
    either is present every positional is a file; otherwise the first
    positional *is* the script and the files are what follow it. A ``-f``
    script file is itself a real read, so it is returned separately rather
    than dropped.
    """

    in_place = False
    script_from_option = False
    script_files: list[str] = []
    positional: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--":
            positional.extend(args[index + 1 :])
            break
        if arg.startswith("--"):
            name, sep, value = arg.partition("=")
            if name == "--expression":
                script_from_option = True
                index += 0 if sep else 1
            elif name == "--file":
                script_from_option = True
                if sep:
                    script_files.append(value)
                elif index + 1 < len(args):
                    index += 1
                    script_files.append(args[index])
            elif name == "--in-place":
                in_place = True
            elif name == "--line-length" and not sep:
                index += 1
            index += 1
            continue
        if arg.startswith("-") and arg != "-":
            # A short cluster: -n, -ne 's/a/b/', -i.bak, -f script.sed. The
            # first value-taking letter consumes the rest of the token (or the
            # next token), so scanning stops there.
            offset = 1
            while offset < len(arg):
                letter = arg[offset]
                attached = arg[offset + 1 :]
                if letter == "i":  # -i[SUFFIX]; the suffix is never separate
                    in_place = True
                    break
                if letter == "e":
                    script_from_option = True
                    if not attached:
                        index += 1
                    break
                if letter == "f":
                    script_from_option = True
                    if attached:
                        script_files.append(attached)
                    elif index + 1 < len(args):
                        index += 1
                        script_files.append(args[index])
                    break
                if letter == "l":
                    if not attached:
                        index += 1
                    break
                offset += 1
            index += 1
            continue
        positional.append(arg)
        index += 1
    files = positional if script_from_option else positional[1:]
    return script_files, files, in_place


def _contains_substitution(segment: str) -> bool:
    return "$(" in segment or "`" in segment or "<(" in segment or ">(" in segment


def _resolve_path(path: str, cwd: str | None) -> str | None:
    if not path or "$" in path or path.startswith("-"):
        return None
    if path in {".", ".."} or path.startswith("./") or path.startswith("../"):
        base = cwd or "/"
        return posixpath.normpath(posixpath.join(base, path))
    if path.startswith("/"):
        return posixpath.normpath(path)
    if path.startswith("~"):
        return None
    if cwd:
        return posixpath.normpath(posixpath.join(cwd, path))
    return path


def _referenced_paths(command: str) -> tuple[str, ...]:
    seen: list[str] = []
    for match in _ABSOLUTE_PATH_RE.finditer(command):
        path = match.group(0)
        if path not in seen:
            seen.append(path)
        if len(seen) >= _MAX_REFERENCED_PATHS:
            break
    return tuple(seen)


def _split_segments(command: str) -> tuple[list[str], bool]:
    """Split *command* at unquoted separators, skipping heredoc bodies.

    Returns the segments plus an opacity marker for constructs the splitter
    cannot see through (an unterminated quote, for instance). Heredoc bodies
    are data, not commands — ``cat > /x <<EOF … EOF`` must contribute exactly
    one segment, with the body skipped, or the body's lines would be analyzed
    as commands.
    """

    segments: list[str] = []
    current: list[str] = []
    pending_heredocs: list[tuple[str, bool]] = []  # (delimiter, strip_tabs)
    opaque = False
    in_single = False
    in_double = False
    index = 0
    length = len(command)

    def flush() -> None:
        text = "".join(current).strip()
        if text:
            segments.append(text)
        current.clear()

    while index < length:
        char = command[index]
        if in_single:
            current.append(char)
            if char == "'":
                in_single = False
            index += 1
            continue
        if in_double:
            current.append(char)
            if char == "\\" and index + 1 < length:
                current.append(command[index + 1])
                index += 2
                continue
            if char == '"':
                in_double = False
            index += 1
            continue
        if char == "\\" and index + 1 < length:
            current.append(char)
            current.append(command[index + 1])
            index += 2
            continue
        if char == "'":
            in_single = True
            current.append(char)
            index += 1
            continue
        if char == '"':
            in_double = True
            current.append(char)
            index += 1
            continue
        if char == "<" and command.startswith("<<", index):
            if command.startswith("<<<", index):
                current.append("<<<")
                index += 3
                continue
            strip_tabs = command.startswith("<<-", index)
            cursor = index + (3 if strip_tabs else 2)
            while cursor < length and command[cursor] in " \t":
                cursor += 1
            delimiter_chars: list[str] = []
            quote = (
                command[cursor] if cursor < length and command[cursor] in "'\"" else ""
            )
            if quote:
                cursor += 1
            while cursor < length and command[cursor] not in " \t\n;|&<>'\"":
                delimiter_chars.append(command[cursor])
                cursor += 1
            if quote and cursor < length and command[cursor] == quote:
                cursor += 1
            delimiter = "".join(delimiter_chars)
            if delimiter:
                pending_heredocs.append((delimiter, strip_tabs))
            index = cursor
            continue
        if char == "\n":
            if pending_heredocs:
                # Skip body lines until each pending delimiter is consumed.
                cursor = index + 1
                while pending_heredocs and cursor <= length:
                    line_end = command.find("\n", cursor)
                    if line_end == -1:
                        line_end = length
                    line = command[cursor:line_end]
                    delimiter, strip_tabs = pending_heredocs[0]
                    candidate = line.lstrip("\t") if strip_tabs else line
                    if candidate == delimiter:
                        pending_heredocs.pop(0)
                    cursor = line_end + 1
                if pending_heredocs:
                    opaque = True  # unterminated heredoc
                    pending_heredocs.clear()
                index = cursor
                flush()
                continue
            flush()
            index += 1
            continue
        if char in ";&|":
            flush()
            while index < length and command[index] in ";&|":
                index += 1
            continue
        current.append(char)
        index += 1

    if in_single or in_double:
        opaque = True
    if pending_heredocs:
        opaque = True
    flush()
    return segments, opaque
