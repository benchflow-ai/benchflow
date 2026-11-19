"""Price a Claude Code rollout from its own session log when the gateway could not.

A rollout's ``cost_usd`` comes from BenchFlow's model gateway, which prices
the calls it proxies. A Claude subscription (``CLAUDE_CODE_OAUTH_TOKEN``), or
a run with usage tracking off, sends Claude Code's calls around the gateway:
its tokens still arrive over ACP, but they have no price. At cleanup, before
the sandbox stops, such a rollout's Claude Code session log (the sandbox
user's ``~/.claude/projects``) is copied into the trial's
``agent/claude-sessions/`` (credentials redacted) and priced by
:func:`benchflow._utils.session_cost.estimate_claude_code_cost`. The result
records it as an estimate: ``cost_usd`` holds the figure, ``price_source`` is
``agent_session_log`` and ``usage_details.cost_estimate`` says how it was
computed.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import logging
import os
import re
import shlex
import shutil
import stat
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote, quote_plus

from benchflow._utils.reachable_write import (
    ReachablePath,
    ReachableWriteError,
    private_folder,
    snapshot_tree,
)
from benchflow.sandbox._exec_output import exec_raw
from benchflow.usage_tracking import USAGE_SOURCE_AGENT_NATIVE_ACP

logger = logging.getLogger(__name__)

PRICE_SOURCE = "agent_session_log"
SESSION_DIR = "claude-sessions"  # under the trial's agent/ folder
# Where Claude Code keeps its session logs, under the agent's home.
CLAUDE_SESSION_LOGS = ".claude/projects"
CODEX_SESSION_DIR = "codex-sessions"  # under the trial's agent/ folder
# Where Codex keeps its session rollouts, under its home (CODEX_HOME).
CODEX_SESSION_LOGS = "sessions"
#: ``usage_details.token_count.source`` of a codex-acp trial counted from
#: Codex's own log (``count_codex_tokens``).
CODEX_TOKENS_SOURCE = "codex-session-log"
_TOKEN_FIELDS = (
    "n_input_tokens",
    "n_output_tokens",
    "n_cache_read_tokens",
    "n_cache_creation_tokens",
    "total_tokens",
)
#: Keep the agent's session log even when the proxy saw its calls (a check
#: of the proxy's records against the agent's own).
KEEP_SESSION_LOGS_ENV = "BENCHFLOW_KEEP_AGENT_SESSION_LOGS"
_MAX_BYTES = 64 * 1024 * 1024
_MAX_ENTRIES = 5000
_TIMEOUT_SEC = 120


def _home(cfg: Any) -> str:
    """The home of the user the agent runs as in the sandbox."""
    return f"/home/{cfg.sandbox_user}" if cfg.sandbox_user else "/root"


def _runs_claude_code(agent: str) -> bool:
    name = agent.rsplit("/", 1)[-1]
    return name in ("claude-agent-acp", "claude")


def _runs_codex(agent: str) -> bool:
    name = agent.rsplit("/", 1)[-1]
    return name in ("codex-acp", "codex")


def _contributors(rollout: Any) -> set[str]:
    """The primary agent and every native role that made the trial's calls."""
    return {rollout._config.primary_agent or ""} | getattr(
        rollout, "_native_usage_agents", set()
    )


def _secrets(rollout: Any) -> list[str]:
    """Credential values the agent was given, to scrub from its log.

    Only the credential-bearing entries, by the same rule that keeps them out
    of the rollout's ``config.json``. Scrubbing every value instead would take
    the public ones with it -- a model name passed as ``ANTHROPIC_MODEL``, a
    base URL -- and the estimate reads the model of every logged response, so
    a scrubbed model name leaves it unpriceable.
    """
    from benchflow._utils.config_redaction import _should_record_env_entry

    env = getattr(rollout, "_agent_env", None) or {}
    values: list[str] = [
        value
        for name, value in env.items()
        if isinstance(value, str)
        and len(value) >= 8
        and not _should_record_env_entry(str(name), value)
    ]
    values.sort(key=len, reverse=True)  # longest first: no partial replacement
    return values


#: The characters JavaScript's ``encodeURIComponent`` leaves as they are,
#: and ``encodeURI`` (ECMA-262, 19.2.6); Python's ``quote`` keeps letters,
#: digits and ``_.-~`` whatever ``safe`` says.
_URI_COMPONENT_KEEPS = "!*'()"
_URI_KEEPS = _URI_COMPONENT_KEEPS + ";/?:@&=+$,#"


def _url_forms(value: str) -> set[str]:
    """*value* as each URL encoder writes it, once and twice: Python's
    ``quote`` (``/`` kept, and nothing kept) and ``quote_plus`` (nothing and
    ``/`` kept), JavaScript's ``encodeURIComponent`` and ``encodeURI``, and a
    form body or ``URLSearchParams`` (WHATWG urlencoded: ``*`` kept, ``~``
    escaped, a space as ``+``)."""
    encoders = (
        lambda v: quote(v, safe=""),
        quote,
        quote_plus,
        lambda v: quote_plus(v, safe="/"),
        lambda v: quote(v, safe=_URI_COMPONENT_KEEPS),
        lambda v: quote(v, safe=_URI_KEEPS),
        lambda v: quote_plus(v, safe="*").replace("~", "%7E"),
    )
    forms = set()
    for encode in encoders:
        once = encode(value)
        forms.update({once, encode(once)})
    return forms


def _secret_forms(value: str) -> list[str]:
    """Every form a credential value takes in a copied log, as the encoders
    write it, never a list of escapes of our own: as given; JSON-escaped
    (``json.dumps``, ASCII-only and not, up to three times: a JSON line held
    in a string field of a JSON line); each of those %-encoded (``quote``,
    once and again: ``%25XX``); the value as each URL encoder writes it
    (:func:`_url_forms`)."""
    forms: set[str] = {value}
    for ascii_only in (True, False):
        escaped = value
        for _ in range(3):
            escaped = json.dumps(escaped, ensure_ascii=ascii_only)[1:-1]
            forms.add(escaped)
    for form in list(forms):
        encoded = form
        for _ in range(2):
            encoded = quote(encoded, safe="")
            forms.add(encoded)
    forms.update(_url_forms(value))
    return sorted(forms, key=lambda form: -len(form))


#: One unit of a form: an escape whose hex digits come in either case
#: (``%XX``, ``\\uXXXX``), or one character.
_UNIT = re.compile(r"%[0-9A-Fa-f]{2}|\\u[0-9A-Fa-f]{4}|.", re.DOTALL)


def _unit_key(unit: str) -> str:
    if len(unit) == 3 and unit[0] == "%":
        return unit.upper()
    if len(unit) == 6 and unit.startswith("\\u"):
        return unit.lower()
    return unit


def _unit_pattern(unit: str) -> str:
    if len(unit) == 3 and unit[0] == "%":
        return f"%(?i:{unit[1:]})"
    if len(unit) == 6 and unit.startswith("\\u"):
        return f"\\\\u(?i:{unit[2:]})"
    return re.escape(unit)


class _Node:
    """A node of the forms' trie: its edges by their first unit, each the
    run of units no other form shares there and the node it leads to, and
    whether a form ends here."""

    __slots__ = ("edges", "end")

    def __init__(self) -> None:
        self.edges: dict[str, list[Any]] = {}
        self.end = False


def _insert(root: _Node, units: list[str]) -> None:
    """Add one form's units to the trie, splitting an edge where it parts
    from another form's. A loop, not recursion: a form is as deep as it is
    long, and a credential can be kilobytes."""
    node, i = root, 0
    while i < len(units):
        edge = node.edges.get(units[i])
        if edge is None:
            leaf = _Node()
            leaf.end = True
            node.edges[units[i]] = [units[i:], leaf]
            return
        label, child = edge
        shared = 1
        while (
            shared < len(label)
            and i + shared < len(units)
            and label[shared] == units[i + shared]
        ):
            shared += 1
        if shared < len(label):
            middle = _Node()
            middle.edges[label[shared]] = [label[shared:], child]
            edge[0], edge[1] = label[:shared], middle
            child = middle
        node, i = child, i + shared
    node.end = True


def _trie_pattern(root: _Node) -> str:
    """The forms in the trie as one regex: a branch per edge, and a form
    that ends at a node only when no longer one goes on (greedy ``?``).
    Built bottom up with a stack, so no form is too long for it."""
    built: dict[int, str] = {}
    stack: list[tuple[_Node, bool]] = [(root, False)]
    while stack:
        node, ready = stack.pop()
        if not ready:
            stack.append((node, True))
            stack.extend((child, False) for _, child in node.edges.values())
            continue
        branches = [
            "".join(map(_unit_pattern, label)) + built.pop(id(child))
            for label, child in node.edges.values()
        ]
        if not branches:
            built[id(node)] = ""
            continue
        body = branches[0] if len(branches) == 1 else "(?:" + "|".join(branches) + ")"
        built[id(node)] = f"(?:{body})?" if node.end else body
    return built[id(root)]


@functools.lru_cache(maxsize=64)
def _secrets_pattern(secrets: tuple[str, ...]) -> re.Pattern[str]:
    """One pattern for every form of every value: a trie of their units, so
    a match costs the form's length, not the number of forms, and the
    longest form wins; an escape's hex digits in either case."""
    root = _Node()
    for value in secrets:
        for form in _secret_forms(value):
            _insert(root, [_unit_key(unit) for unit in _UNIT.findall(form)])
    return re.compile(_trie_pattern(root))


def scrub_secrets(text: str, secrets: list[str]) -> str:
    """*text* with each credential value the agent was given (``_secrets``)
    replaced, wherever it appears and in whichever form (:func:`_secret_forms`):
    a value with a quote, a backslash or a non-ASCII character is escaped in
    a JSON copy, and a URL carries it %-encoded. The teardown copy and the
    copy made while the turns run both use it."""
    if not secrets:
        return text
    return _secrets_pattern(tuple(secrets)).sub("[redacted]", text)


def scrub_secrets_obj(obj: Any, secrets: list[str]) -> Any:
    """A parsed log line with :func:`scrub_secrets` applied to every string
    in it, keys included: a value is found in its decoded form, before the
    line is written as JSON again."""
    if not secrets:
        return obj
    if isinstance(obj, str):
        return scrub_secrets(obj, secrets)
    if isinstance(obj, list):
        return [scrub_secrets_obj(item, secrets) for item in obj]
    if isinstance(obj, dict):
        return {
            scrub_secrets_obj(key, secrets): scrub_secrets_obj(item, secrets)
            for key, item in obj.items()
        }
    return obj


def redact_line(line: str, secrets: list[str]) -> str:
    """One line of a copied log, redacted. The exact values the agent was
    given go first (:func:`scrub_secrets`), so a value is replaced whole
    before a trajectory pattern can take a token-shaped part of it and leave
    the rest; then the trajectory patterns; then the exact values again over
    the line as written. Both copies of a session log (the teardown copy and
    the one made while the turns run) redact each line with it."""
    from benchflow.trajectories.types import (
        redact_trajectory_obj,
        redact_trajectory_text,
    )

    try:
        text = json.dumps(
            redact_trajectory_obj(scrub_secrets_obj(json.loads(line), secrets))
        )
    except ValueError:
        text = redact_trajectory_text(scrub_secrets(line, secrets))
    return scrub_secrets(text, secrets)


async def off_loop[T](
    func: Callable[..., T], *args: Any, then: Callable[[T], None]
) -> T:
    """``func(*args)`` in a worker thread, then ``then(result)`` on the event
    loop, so redacting a large log never holds up the loop, and with it the
    agent's turn and the run's other trials (a 27 MB log takes half a
    minute). A cancel waits for both and then takes effect: what was read
    is redacted and kept whole, as when the redaction ran on the loop and a
    cancel could not cut it. ``then`` is skipped when ``func`` raised."""
    work = asyncio.ensure_future(asyncio.to_thread(func, *args))
    cancelled = False
    while not work.done():
        try:
            await asyncio.wait({work})
        except asyncio.CancelledError:
            cancelled = True
    try:
        if work.exception() is None:
            then(work.result())
    finally:
        if cancelled:
            raise asyncio.CancelledError
    return work.result()


def _redact(path: Path, secrets: list[str]) -> None:
    """Redact one copied log file: a .jsonl line by line (:func:`redact_line`),
    other files whole, the exact values before and after the trajectory
    patterns."""
    from benchflow._utils.reachable_write import destination
    from benchflow.trajectories.types import redact_trajectory_text

    held = destination(path)
    text = held.read_text(errors="replace")
    if path.suffix == ".jsonl":
        lines = []
        for line in text.splitlines():
            lines.append(redact_line(line, secrets))
        clean = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
        if any(len(value.splitlines()) > 1 for value in secrets):
            clean = scrub_secrets(clean, secrets)  # a value that spans lines
    else:
        clean = scrub_secrets(
            redact_trajectory_text(scrub_secrets(text, secrets)), secrets
        )
    if clean != text:
        held.write_text(clean)


class RedactionFailed(Exception):
    """A session-log copy could not be redacted, so it is not kept. Names
    the exception's class only: its message may hold the text it was
    redacting."""

    def __init__(self, error: str) -> None:
        super().__init__(error)
        self.error = error


def note_redaction_failure(rollout: Any, error: str, *, copy: str) -> None:
    """A copy of the agent's session log (``teardown``, or the ``mirror``
    made while its turns ran) failed its redaction and was not kept: say so
    once per copy, loudly, by the exception's class, and leave the mark on
    the trial's count to :func:`~benchflow.rollout._cost.mark_incomplete_usage`."""
    attr = f"_{copy}_redaction_failed"
    if getattr(rollout, attr, None):
        return
    setattr(rollout, attr, error)
    logger.warning(
        "The agent's session log could not be redacted (%s), so the copy %s "
        "was not kept; the trial's token count is marked incomplete",
        error,
        "made at teardown" if copy == "teardown" else "made while its turns ran",
    )


def _redact_files(files: list[Path], secrets: list[str]) -> None:
    try:
        for path in files:
            _redact(path, secrets)
    except Exception as exc:
        raise RedactionFailed(type(exc).__name__) from None


async def _capture(
    env: Any, source: str, bundle: Path, secrets: list[str]
) -> Path | None:
    """Copy ``source`` from the sandbox into ``bundle`` (a private folder),
    redacted; the copied tree, or None when ``source`` is absent."""
    from benchflow.review.evidence import capture_workspace

    probe = await exec_raw(
        env.exec, f"test -d {shlex.quote(source)}", user="root", timeout_sec=30
    )
    if probe.return_code != 0:
        return None
    await capture_workspace(
        env,
        source,
        bundle,
        max_bytes=_MAX_BYTES,
        max_entries=_MAX_ENTRIES,
        timeout_sec=_TIMEOUT_SEC,
    )
    tree = bundle / "workspace"
    files = [
        path
        for path in sorted(tree.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]
    await off_loop(_redact_files, files, secrets, then=lambda _: None)
    return tree


async def _copy_out(
    env: Any, source: str, target: ReachablePath, secrets: list[str]
) -> bool:
    """Copy ``source`` from the sandbox to ``target``, redacted; False when absent."""
    # target.parent is agent/, /logs/agent in a Docker sandbox: the copy is
    # captured and redacted in private custody, then placed there by one
    # rename through the held folder (reachable_write, R4-1).
    target.parent.mkdir(parents=True, exist_ok=True)
    with private_folder(target.parent) as temp:
        tree = await _capture(env, source, Path(temp) / "bundle", secrets)
        if tree is None:
            return False
        target.publish(tree)
    return True


@dataclass
class SessionLogSnapshot:
    """The agent's logs teardown counts and prices from, as they were at the
    end of the agent's turn (:func:`snapshot_session_logs`).

    They are held in a private folder outside every mount until cleanup
    ends, laid out as the trial folder lays them out: ``view`` holds
    ``agent/claude-sessions`` or ``agent/codex-sessions`` (``tree``, the
    redacted copy of ``source``; None when it was absent or could not be
    copied), and teardown's counts read them there
    (:func:`session_log_root`), never the trial's agent/, which the sandbox
    shares. ``stream`` is the native harness's stream log; ``published``
    the folders placed back into agent/ for the record.
    """

    holder: contextlib.ExitStack
    view: Path
    source: str | None = None
    tree: Path | None = None
    stream: Path | None = None
    published: set[str] = field(default_factory=set)


def _session_log_source(rollout: Any) -> tuple[str, str] | None:
    """The sandbox folder of the session log teardown reads for this trial
    (:func:`price_from_session_log`, :func:`keep_session_logs`,
    :func:`count_compactions`, the Codex count and the model calls), and its
    folder name under agent/: Claude Code's when it made every call, Codex's
    when it did; None otherwise, or when the sandbox holds another rollout's
    turns."""
    agents = _contributors(rollout)
    if (
        not agents
        or getattr(rollout, "_branch_child_active", False)
        or getattr(rollout, "_branch_forks", None)
        or getattr(rollout, "_from_branch_snapshot", False)
    ):
        return None
    cfg = rollout._config
    if all(_runs_claude_code(agent) for agent in agents):
        return f"{_home(cfg)}/{CLAUDE_SESSION_LOGS}", SESSION_DIR
    if all(_runs_codex(agent) for agent in agents):
        env = getattr(rollout, "_agent_env", None) or {}
        home = env.get("CODEX_HOME") or f"{_home(cfg)}/.codex"
        return f"{home}/{CODEX_SESSION_LOGS}", CODEX_SESSION_DIR
    return None


def _refuse_stream_log(rollout: Any, name: str, error: BaseException) -> None:
    """agent/``name``, the native harness's stream log, is not a regular file
    the snapshot can read without following it: say so in the trial's
    warnings."""
    logger.info("The native stream log was not copied: %s", error)
    warning = (
        f"agent/{name}, the native harness's stream log, is not a regular file "
        "(a link, a folder or a special file): BenchFlow did not read it, so a "
        "native Claude Code turn cut off before its report is not counted from it"
    )
    logger.warning("%s", warning)
    kept = vars(rollout).setdefault("_warnings", [])
    if warning not in kept:
        kept.append(warning)


def session_log_root(rollout: Any) -> Path:
    """The folder teardown reads the agent's session logs under
    (``agent/claude-sessions``, ``agent/codex-sessions``): the private copy
    made at the end of the agent's turn when one was made, else the trial
    folder (no verifier ran in the sandbox)."""
    snapshot = getattr(rollout, "_session_log_snapshot", None)
    if snapshot is not None:
        return snapshot.view
    return Path(rollout._rollout_paths.rollout_dir)


def close_session_log_snapshot(rollout: Any) -> None:
    snapshot = getattr(rollout, "_session_log_snapshot", None)
    rollout._session_log_snapshot = None
    if snapshot is not None:
        with contextlib.suppress(Exception):
            snapshot.holder.close()


async def snapshot_session_logs(rollout: Any) -> None:
    """Copy the agent's session log and the native harness's stream log at
    the end of the agent's turn, before hardening and the verifier.

    test.sh runs as root, with code the agent wrote: it could rewrite the
    session log in the sandbox, or anything in agent/ (which the sandbox
    shares), before teardown read it. From here on teardown reads only these
    copies, also when one was absent or could not be made: a log test.sh
    rewrites or plants, in the sandbox or in agent/, is never read. Never
    fails the rollout.
    """
    close_session_log_snapshot(rollout)
    paths = getattr(rollout, "_rollout_paths", None)
    if paths is None:
        return
    holder = contextlib.ExitStack()
    try:
        folder = Path(holder.enter_context(private_folder(paths.agent_dir)))
    except Exception as exc:
        logger.info("No private folder for the session log copies: %s", exc)
        holder.close()
        return
    snapshot = SessionLogSnapshot(holder=holder, view=folder / "view")
    (snapshot.view / "agent").mkdir(parents=True)
    rollout._session_log_snapshot = snapshot
    from benchflow.rollout._salvage import STREAM_LOG

    # Through the reachable helper, as agent/ is the sandbox's: a regular
    # file only, its copy read through a no-follow descriptor, so nothing
    # outside the run folder is read and a special file cannot hold the copy.
    source = paths.agent_dir / STREAM_LOG
    copy = folder / STREAM_LOG
    try:
        if not stat.S_ISREG(source.lstat().st_mode):
            raise ReachableWriteError("not a regular file")
        snapshot_tree(source, copy, refuse_links=True)
        if copy.is_symlink() or not copy.is_file():
            raise ReachableWriteError("not a regular file")
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as exc:
        if copy.is_dir() and not copy.is_symlink():
            shutil.rmtree(copy, ignore_errors=True)
        else:
            copy.unlink(missing_ok=True)
        _refuse_stream_log(rollout, STREAM_LOG, exc)
    else:
        snapshot.stream = copy
    found = _session_log_source(rollout)
    env = getattr(rollout, "_env", None)
    if found is None or env is None:
        return
    source, name = found
    snapshot.source = source
    if getattr(rollout, "_teardown_redaction_failed", None):
        return
    try:
        tree = await asyncio.wait_for(
            _capture(env, source, folder / "bundle", _secrets(rollout)),
            timeout=_TIMEOUT_SEC + 60,
        )
    except RedactionFailed as exc:
        note_redaction_failure(rollout, exc.error, copy="teardown")
        return
    except Exception as exc:  # the sandbox may be gone; the log stays unread
        logger.info("Could not copy the agent's session log: %s", exc)
        return
    if tree is not None:
        snapshot.tree = tree.rename(snapshot.view / "agent" / name)


def publish_session_log_copy(rollout: Any, name: str, tree: Path) -> None:
    """Place a copy of ``tree`` at agent/``name`` for the trial's record,
    replacing what is there (a copy test.sh may have planted)."""
    target = rollout._rollout_paths.agent_dir / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with private_folder(target.parent) as temp:
        staged = Path(temp) / "tree"
        shutil.copytree(tree, staged, symlinks=True)
        # A folder or link already at the name is retired, never followed
        # (reachable_write), so a planted one is replaced.
        target.publish(staged)
    snapshot = getattr(rollout, "_session_log_snapshot", None)
    if snapshot is not None:
        snapshot.published.add(name)


async def clear_session_log_folders_in_sandbox(rollout: Any) -> None:
    """Remove /logs/agent/claude-sessions and /logs/agent/codex-sessions in
    the sandbox, as root, once the copy point passed.

    BenchFlow places agent/claude-sessions and agent/codex-sessions on the
    host, at teardown: one in the sandbox's /logs/agent was written there by
    test.sh or the agent. On Docker /logs/agent is the trial's agent/ (a bind
    mount), and what root wrote there stays root's until the sandbox stops,
    so the host can neither replace nor empty it; elsewhere it would be
    copied out with the rest of /logs/agent. Runs before teardown places its
    copies. Never fails the rollout.
    """
    snapshot = getattr(rollout, "_session_log_snapshot", None)
    env = getattr(rollout, "_env", None)
    paths = getattr(rollout, "_rollout_paths", None)
    if snapshot is None or env is None or paths is None:
        return
    # Never a folder teardown placed: on a mount, it is the host's copy.
    names = tuple(
        name
        for name in (SESSION_DIR, CODEX_SESSION_DIR)
        if name not in snapshot.published
    )
    # A mounted /logs/agent shows on the host what the sandbox holds.
    if not names or (
        getattr(env, "is_mounted", False) is True
        and not any(
            os.path.lexists(Path(paths.rollout_dir) / "agent" / name) for name in names
        )
    ):
        return
    from benchflow.task.paths import SandboxPaths

    targets = " ".join(
        shlex.quote(f"{SandboxPaths.agent_dir}/{name}") for name in names
    )
    try:
        result = await asyncio.wait_for(
            exec_raw(
                env.exec,
                f"rm -rf -- {targets}",
                user="root",
                system_shell=True,
                timeout_sec=30,
            ),
            timeout=40,
        )
    except Exception as exc:
        logger.info("Could not clear the session log folders in the sandbox: %s", exc)
        return
    if result.return_code != 0:
        logger.info(
            "Could not clear the session log folders in the sandbox (exit %s)",
            result.return_code,
        )


def settle_session_log_folders(rollout: Any) -> None:
    """After teardown placed its copies: an agent/claude-sessions or
    agent/codex-sessions teardown did not place there is not the trial's
    (BenchFlow writes them only at teardown): emptied, so no later reader
    (a cost backfill) takes it for the agent's log."""
    snapshot = getattr(rollout, "_session_log_snapshot", None)
    paths = getattr(rollout, "_rollout_paths", None)
    if snapshot is None or paths is None:
        return
    for name in (SESSION_DIR, CODEX_SESSION_DIR):
        target = paths.agent_dir / name
        if name in snapshot.published or not target.exists():
            continue
        try:
            with private_folder(target.parent) as temp:
                empty = Path(temp) / "empty"
                empty.mkdir()
                target.publish(empty)
        except Exception as exc:
            logger.warning("Could not empty agent/%s: %s", name, exc)


async def _copy_session_log(rollout: Any, source: str, target: ReachablePath) -> bool:
    """Place the agent's session log at ``target``: a copy of the one made at
    the end of the agent's turn when that point was reached
    (:func:`snapshot_session_logs`), replacing what is there; else from the
    sandbox (no verifier ran in it)."""
    snapshot = getattr(rollout, "_session_log_snapshot", None)
    if snapshot is None:
        return await asyncio.wait_for(
            _copy_out(rollout._env, source, target, _secrets(rollout)),
            timeout=_TIMEOUT_SEC + 60,
        )
    if snapshot.source != source or snapshot.tree is None:
        return False
    publish_session_log_copy(rollout, target.name, snapshot.tree)
    return True


async def price_from_session_log(rollout: Any) -> None:
    """Estimate an unpriced Claude Code rollout's USD from its session log.

    Does nothing when the gateway supplied usage (even unpriced or mixed
    with native usage), any contributing agent is not Claude Code, the
    rollout's sandbox holds another rollout's turns, or no log is found.
    Never fails the rollout.

    A rollout is left out when its turns cannot be told apart from turns
    someone else is already charged for: a fork's in-place child writes into
    the parent's own log, and a rollout whose sandbox came from a snapshot --
    an isolated branch child, or a run continued from a kept checkpoint --
    finds the source rollout's session file already in it, so it would be
    charged for turns the source's own record already carries.
    """
    metrics = getattr(rollout, "_usage_metrics", None) or {}
    cfg = rollout._config
    # Null provider cost means unknown coverage, not permission to replace
    # it with an agent-owned log. In particular, a Claude log cannot price
    # other roles or deduplicate their provider subtotal in mixed usage.
    if metrics.get("usage_source") == "provider_response":
        return
    contributors = _contributors(rollout)
    if not contributors or any(not _runs_claude_code(agent) for agent in contributors):
        return
    if metrics.get("cost_usd") is not None or not _runs_claude_code(
        cfg.primary_agent or ""
    ):
        return
    paths = getattr(rollout, "_rollout_paths", None)
    if paths is None or (
        rollout._env is None and getattr(rollout, "_session_log_snapshot", None) is None
    ):
        return
    if (
        getattr(rollout, "_branch_child_active", False)
        or getattr(rollout, "_branch_forks", None)
        # Started from a snapshot (an isolated branch child, or a continued
        # checkpoint): the sandbox carries the source rollout's session log.
        or getattr(rollout, "_from_branch_snapshot", False)
    ):
        return
    target = paths.agent_dir / SESSION_DIR
    source = f"{_home(cfg)}/{CLAUDE_SESSION_LOGS}"
    if getattr(rollout, "_teardown_redaction_failed", None):
        return
    try:
        copied = await _copy_session_log(rollout, source, target)
    except RedactionFailed as exc:
        note_redaction_failure(rollout, exc.error, copy="teardown")
        return
    except Exception as exc:  # the sandbox may be gone; the cost stays unknown
        logger.info("Could not copy Claude Code's session log: %s", exc)
        return
    if not copied:
        return
    rollout._session_log_kept = True
    from benchflow._utils.reported_models import session_log_models
    from benchflow._utils.session_cost import estimate_claude_code_cost

    # Read from the copy teardown holds, not agent/, which the sandbox
    # shares.
    held = session_log_root(rollout) / "agent" / SESSION_DIR
    # The models Claude Code says answered (P6, G-28): the run asked for an
    # alias, and only the session log names the model behind it.
    rollout._session_reported_models = session_log_models(held)

    estimate = estimate_claude_code_cost(held)
    if estimate is None:
        return
    details = dict(metrics.get("usage_details") or {})
    details["cost_estimate"] = {
        "source": estimate["source"],
        "method": estimate["method"],
        "path": f"agent/{SESSION_DIR}",
        "sessions": estimate["sessions"],
        "responses": estimate["responses"],
        "models": estimate["models"],
        "context_1m": estimate["context_1m"],
    }
    rollout._usage_metrics = {
        **metrics,
        "cost_usd": estimate["usd"],
        "price_source": PRICE_SOURCE,
        "usage_details": details,
    }
    logger.info(
        "Cost estimated from Claude Code's session log: $%.4f (%s)",
        estimate["usd"],
        estimate["method"],
    )


async def keep_session_logs(rollout: Any) -> None:
    """Copy the agent's own session log when the proxy saw none of its calls
    and that agent made every call in the trial.

    Claude Code's (``~/.claude/projects``) goes to ``agent/claude-sessions/``
    and Codex's (``$CODEX_HOME/sessions``, ``~/.codex/sessions`` by default)
    to ``agent/codex-sessions/``, redacted like the price's copy, so a run
    around the proxy, with no OpenTelemetry from the agent, still has a
    record of each model call (``trajectories/model_calls.py``). With
    ``BENCHFLOW_KEEP_AGENT_SESSION_LOGS=1`` the log is kept for proxied runs
    too, and for a trial with a token or turn budget, whose subagents' share
    of it only the agent's own log tells (``trial_budget.subagents``). Never
    fails the rollout.
    """
    import os

    metrics = getattr(rollout, "_usage_metrics", None) or {}
    proxied = metrics.get("usage_source") == "provider_response"
    if (
        proxied
        and os.environ.get(KEEP_SESSION_LOGS_ENV) != "1"
        and getattr(rollout, "_trial_budget", None) is None
    ):
        return
    cfg = rollout._config
    paths = getattr(rollout, "_rollout_paths", None)
    if paths is None or (
        rollout._env is None and getattr(rollout, "_session_log_snapshot", None) is None
    ):
        return
    if (
        getattr(rollout, "_branch_child_active", False)
        or getattr(rollout, "_branch_forks", None)
        or getattr(rollout, "_from_branch_snapshot", False)
    ):
        # The sandbox holds another rollout's turns (see price_from_session_log).
        return
    # A log is the trial's record only when its agent made every call: with
    # another agent's turns in the trial, Claude's log would give its calls,
    # and its cost, as the trial's (the price above refuses the same).
    agents = _contributors(rollout)
    if all(_runs_claude_code(agent) for agent in agents):
        source, target = (
            f"{_home(cfg)}/{CLAUDE_SESSION_LOGS}",
            paths.agent_dir / SESSION_DIR,
        )
    elif all(_runs_codex(agent) for agent in agents):
        env = getattr(rollout, "_agent_env", None) or {}
        home = env.get("CODEX_HOME") or f"{_home(cfg)}/.codex"
        source, target = (
            f"{home}/{CODEX_SESSION_LOGS}",
            paths.agent_dir / CODEX_SESSION_DIR,
        )
    else:
        return
    snapshot = getattr(rollout, "_session_log_snapshot", None)
    if snapshot is not None and target.name in snapshot.published:
        return
    # A folder already there is the trial's own only when no copy point was
    # reached; after it, one test.sh planted is replaced.
    if (snapshot is None and target.exists()) or getattr(
        rollout, "_teardown_redaction_failed", None
    ):
        return
    try:
        copied = await _copy_session_log(rollout, source, target)
    except RedactionFailed as exc:
        note_redaction_failure(rollout, exc.error, copy="teardown")
        return
    except Exception as exc:  # the sandbox may be gone; the calls stay unrecorded
        logger.info("Could not copy the agent's session log: %s", exc)
        return
    if copied:
        rollout._session_log_kept = True


async def count_codex_tokens(rollout: Any) -> None:
    """Count a Codex trial's tokens from Codex's own session log.

    codex-acp 2.0.1 answers each prompt with the usage of the turn's last
    model call (``PromptResponse.usage`` is Codex's ``tokenUsage.last``), and
    its ``usage_update`` notifications carry only how full the context is
    (``used``, ``size``). The running total Codex keeps (``tokenUsage.total``)
    never crosses ACP, so a trial of 75 tool calls recorded one call's
    tokens. Codex logs every response's usage in its session log, which
    :func:`keep_session_logs` has copied into ``agent/codex-sessions/``. Their
    sum replaces the ACP figures, ``usage_source`` is ``session_log``, and ``usage_details.token_count`` names the log and keeps what
    ACP reported.

    The native harness's turns report Codex's running total of the main
    thread; its trial is counted from the log too, so a turn that
    ended without that report (a timeout, a cancel) and a sub-agent's calls
    count, and both harnesses count alike. The log's sum replaces the
    reported figure, never adds to it.

    Does nothing when the proxy metered the trial, another agent made calls,
    the sandbox holds another rollout's turns, or there is no log to read.
    The reported figure then stays, without ``token_count``. Never fails the
    rollout.
    """
    from benchflow.native_harness.spec import HARNESS_NATIVE
    from benchflow.rollout._cost import INCOMPLETE_CODEX_ACP, INCOMPLETE_MIRROR
    from benchflow.rollout._salvage import codex_session_slices
    from benchflow.trajectories.model_calls import codex_session_tokens
    from benchflow.usage_tracking import USAGE_SESSION_LOG, is_token_usage_available

    metrics = getattr(rollout, "_usage_metrics", None) or {}
    if metrics.get("usage_source") == "provider_response" or not all(
        _runs_codex(agent) for agent in _contributors(rollout)
    ):
        return
    reported = is_token_usage_available(metrics)
    paths = getattr(rollout, "_rollout_paths", None)
    if (
        paths is None
        or getattr(rollout, "_branch_child_active", False)
        or getattr(rollout, "_branch_forks", None)
        or getattr(rollout, "_from_branch_snapshot", False)
    ):
        # The sandbox holds another rollout's turns (see price_from_session_log).
        return
    try:
        # Read as the model-call record reads it: from the trial's folder,
        # never through a link (agent/ is a folder the sandbox shares).
        root = session_log_root(rollout)
        totals = await asyncio.to_thread(codex_session_tokens, root)
        slices = await asyncio.to_thread(codex_session_slices, root)
    except Exception as exc:  # the trial keeps what ACP reported
        logger.info("Could not read Codex's session log: %s", exc)
        return
    if totals is None:
        logger.info(
            "No Codex session log to count: the trial's tokens are the ones "
            "Codex reported at the end of its turns"
        )
        if reported and getattr(rollout._config, "harness", None) != HARNESS_NATIVE:
            # codex-acp's report is each turn's last call: the rest are missing.
            rollout._usage_incomplete_reason = INCOMPLETE_CODEX_ACP
        return
    details = dict(metrics.get("usage_details") or {})
    acp_reported: dict[str, Any] | None = None
    if reported:
        acp_reported = {field: metrics.get(field) for field in _TOKEN_FIELDS}
        acp_reported["thought_tokens"] = details.get("thought_tokens")
    details["thought_tokens"] = totals["thought_tokens"]
    details["token_count"] = {
        "source": CODEX_TOKENS_SOURCE,
        "path": f"agent/{CODEX_SESSION_DIR}",
        "calls": totals["calls"],
        "acp_reported": acp_reported,
    }
    counted = {
        **metrics,
        **{field: totals[field] for field in _TOKEN_FIELDS},
        # Codex logs no 5-minute/1-hour split of cache writes.
        "n_cache_write_5m_tokens": None,
        "n_cache_write_1h_tokens": None,
        "usage_source": USAGE_SESSION_LOG,
        "usage_source_legacy": USAGE_SOURCE_AGENT_NATIVE_ACP,
        "usage_details": details,
    }
    if metrics.get("price_source") == "list_price":
        counted.update(cost_usd=None, price_source=None, cost_source=None)
    rollout._usage_metrics = counted
    rollout._native_usage_metrics = dict(counted)
    rollout._usage_slices = slices
    rollout._usage_salvaged = True
    if (getattr(rollout, "_mirror_placed", None) or {}).get(CODEX_SESSION_DIR) is False:
        rollout._usage_incomplete_reason = INCOMPLETE_MIRROR
    logger.info(
        "Tokens counted from Codex's session log: %d calls, %d tokens "
        "(Codex reported %s at the end of its turns)",
        totals["calls"],
        totals["total_tokens"],
        acp_reported["total_tokens"] if acp_reported else "nothing",
    )


def compactions_in(lines: list[str]) -> dict[str, Any]:
    """The compactions Claude Code's session log records: each line it
    writes at a compaction (``"subtype": "compact_boundary"``), with what
    triggered it and the context's tokens before it (``compactMetadata``)."""
    pre_tokens: list[int] = []
    count = auto = manual = subagent = 0
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict) or row.get("subtype") != "compact_boundary":
            continue
        if row.get("isSidechain") is True:
            subagent += 1
            continue
        count += 1
        meta = row.get("compactMetadata")
        meta = meta if isinstance(meta, dict) else {}
        trigger = meta.get("trigger")
        auto += trigger == "auto"
        manual += trigger == "manual"
        tokens = meta.get("preTokens")
        if isinstance(tokens, int) and not isinstance(tokens, bool):
            pre_tokens.append(tokens)
    return {
        "count": count,
        "auto": auto,
        "manual": manual,
        "subagent": subagent,
        "pre_tokens": pre_tokens,
        "source": "claude-code-session-log",
    }


async def count_compactions(rollout: Any) -> None:
    """Count Claude Code's compactions in this trial (``_compactions``, for
    result.json): from the log copied into ``agent/claude-sessions/`` when
    there is one, else the lines that mark them, read in the sandbox while
    it is up. Left None when another agent made calls, the sandbox holds
    another rollout's turns, or the log cannot be read. Never fails the
    rollout."""
    agents = _contributors(rollout)
    paths = getattr(rollout, "_rollout_paths", None)
    if (
        not agents
        or not all(_runs_claude_code(agent) for agent in agents)
        or paths is None
        or getattr(rollout, "_branch_child_active", False)
        or getattr(rollout, "_branch_forks", None)
        or getattr(rollout, "_from_branch_snapshot", False)
    ):
        return
    snapshot = getattr(rollout, "_session_log_snapshot", None)
    if rollout._env is None and snapshot is None:
        return
    copied = paths.agent_dir / SESSION_DIR
    lines: list[str] = []
    try:
        # The copy made at the end of the agent's turn when that point was
        # reached, else the published copy, else the sandbox.
        if snapshot is not None:
            held: Path | None = snapshot.view / "agent" / SESSION_DIR
            if not held.is_dir():
                return
        else:
            held = copied if copied.is_dir() else None
        if held is not None:
            for path in sorted(held.rglob("*.jsonl")):
                if path.is_file() and not path.is_symlink():
                    lines += [
                        line
                        for line in path.read_text(errors="replace").splitlines()
                        if "compact_boundary" in line
                    ]
        else:
            source = f"{_home(rollout._config)}/{CLAUDE_SESSION_LOGS}"
            found = await exec_raw(
                rollout._env.exec,
                f"test -d {shlex.quote(source)} || exit 3; "
                f"grep -rhs --include='*.jsonl' compact_boundary {shlex.quote(source)}"
                " || true",
                user="root",
                timeout_sec=30,
            )
            if found.return_code == 3:
                return
            lines = str(found.stdout or "").splitlines()
    except Exception as exc:  # the sandbox may be gone; the count stays unknown
        logger.info("Could not count Claude Code's compactions: %s", exc)
        return
    rollout._compactions = compactions_in(lines)
