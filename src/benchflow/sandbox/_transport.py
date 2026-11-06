"""The exec transport contract every backend keeps (DAYTONA-EXEC).

What one exec hands its provider, after every wrapper a backend adds (the
session command text on Daytona, the argv on Docker, Apple container and
Modal, the JSON command on AgentCore), is at most :data:`COMMAND_BOUND`
bytes. Linux refuses one argument over 128 KiB (``MAX_ARG_STRLEN``), and the
backends wrap a command in up to four quoting layers (Daytona: the route's
``sh -c``, ``su -c``, the outer ``sh -c``), each of which turns every ``'``
into five bytes and so compounds: a 58 KB verifier preparation script became
a 156 KB argument there and never ran. Half of that limit for the whole
command keeps every argument in it below the kernel's limit however a
provider splits it.

Each backend builds what it sends as a pure function of the text the
innermost shell runs (its env wrapper included) and calls :func:`fit`. When
that text fits, it is sent as before, byte for byte. When it does not, the
text travels as data on the exec's standard input (the backend's own pipe,
or a file only root can reach that root's shell opens before anything runs)
and :data:`READER`'s constant line, built through the same wrappers, runs
it: the same shell, user, route, ``timeout`` and directory, no quoting of
the script at all. :func:`sent` is called with exactly what the backend is
about to hand over, at the one place it hands it over, and refuses anything
over the bound (:class:`CommandTooLarge`): nothing else checks a size.

Output the harness parses crosses as a receipt
(:mod:`benchflow.sandbox._exec_receipt`), never as raw bytes: Daytona's
session logs drop NUL, move ``\\x01\\x01\\x01``/``\\x02\\x02\\x02`` runs to the
other stream and turn bytes that are not UTF-8 into U+FFFD.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

#: The most bytes one exec may hand its provider.
COMMAND_BOUND = 64 * 1024

#: The folder root's staged payloads go in, one fresh folder each
#: (``benchflow._runtime_tree.RUN_DIR``: root's, mode 700, made before the
#: agent's first turn).
STAGE_ROOT = "/opt/benchflow/run"

#: Exit status of a staged command whose payload could not be read.
READ_FAILED = 126

# Observers of every sent command (the unit suite's tripwire): called with
# the backend's name and the size, before the bound is applied.
_observers: list[Callable[[str, int], None]] = []


class CommandTooLarge(RuntimeError):
    """A backend was about to send more than :data:`COMMAND_BOUND` bytes."""


def size(sent: str | Sequence[str]) -> int:
    """Bytes of *sent*: a command text, or an argv (each element and its NUL)."""
    if isinstance(sent, str):
        return len(sent.encode("utf-8", "surrogateescape"))
    return sum(len(str(part).encode("utf-8", "surrogateescape")) + 1 for part in sent)


def sent(
    backend: str,
    what: str | Sequence[str],
    bound: int = COMMAND_BOUND,
    *,
    measure: Callable[[Any], int] = size,
) -> None:
    """Called by *backend* with exactly what it hands its provider, right
    before it does. Raises :class:`CommandTooLarge` over *bound* (never more
    than :data:`COMMAND_BOUND`). *measure*: how the provider counts what it
    is handed (AgentCore: the JSON-escaped body), :func:`size` by default."""
    n = measure(what)
    for observe in list(_observers):
        observe(backend, n)
    limit = min(bound, COMMAND_BOUND)
    if n > limit:
        raise CommandTooLarge(
            f"{backend}: a {n}-byte command is over the {limit}-byte bound of "
            "one exec; BenchFlow sends a larger script as data, so this is a "
            "BenchFlow bug: please report it"
        )


#: What every staged payload ends with: this text and the reader's nonce,
#: with nothing after them. The reader runs nothing unless the payload it
#: read ends so, so a payload cut on the way (a client killed mid-write, an
#: upload that stopped) is refused whole, never run as a prefix.
END_MARK = "bf_payload_end_"


def new_end() -> str:
    """A fresh end line for one staged payload (:data:`END_MARK` and 128
    random bits, so no prefix of another payload can end with it)."""
    return f"{END_MARK}{secrets.token_hex(16)}"


def framed(data: bytes, end: str) -> bytes:
    """*data* as a staged payload that ends with *end*."""
    mark = end.encode()
    if mark in data:
        raise ValueError("a staged payload holds its own end line")
    return data + mark


def reader(cat: str, end: str, *, fd: int = 0) -> str:
    """The constant line that runs a staged payload: read it all from
    descriptor *fd* (with *cat*, named by absolute path), close that
    descriptor (stdin becomes /dev/null), refuse it with status
    :data:`READ_FAILED` unless it ends with *end* (:func:`framed`), and
    evaluate the rest in this shell, as ``-c`` would have run it.

    *fd* 0 is the exec's stdin. Another descriptor leaves stdin to the
    command (the DinD VM's ``docker compose exec`` passes it on)."""
    read = f"$({cat})" if fd == 0 else f"$({cat} <&{fd})"
    close = "exec </dev/null" if fd == 0 else f"exec {fd}<&-"
    return (
        f"bf_payload={read} || {{ echo 'benchflow: cannot read the staged "
        f"command' >&2; exit {READ_FAILED}; }}; {close}; "
        f"case $bf_payload in *{end}) ;; *) echo 'benchflow: the staged "
        f"command arrived incomplete; nothing of it ran' >&2; "
        f"exit {READ_FAILED};; esac; "
        f'eval "${{bf_payload%{end}}}"'
    )


def reader_cat(route: Any) -> str:
    """The ``cat`` :func:`reader` names on *route*: the protected kit's once
    the sandbox has one (C8), else the system's."""
    from benchflow._runtime_tree import KIT_BIN

    return f"{KIT_BIN}/cat" if getattr(route, "kit", False) else "/bin/cat"


def stage_folder() -> str:
    """A fresh folder for one staged payload, under :data:`STAGE_ROOT`."""
    return f"{STAGE_ROOT}/stage-{secrets.token_hex(16)}"


@dataclass(frozen=True)
class Fit:
    """What :func:`fit` decided: *sent* goes to the provider; *payload*, when
    not None, is the exec's standard input, already :func:`framed`. *end*:
    the end line its reader checks; a backend that builds its own stdin
    (an env ahead of the command) frames it with :meth:`frame`. *line*: the
    reader *sent* was built around."""

    sent: Any
    payload: bytes | None = None
    end: str = ""
    line: str = ""

    def frame(self, data: bytes) -> bytes:
        return framed(data, self.end)


def fit(
    build: Callable[[str], Any],
    inner: str,
    *,
    cat: str,
    bound: int = COMMAND_BOUND,
    agent: Any = None,
    measure: Callable[[Any], int] = size,
    fd: int = 0,
) -> Fit:
    """``build(inner)`` when it fits *bound*; else ``build`` of the reader,
    with *inner* as the payload (on descriptor *fd*, see :func:`reader`).

    *agent*: the exec's ``AgentExecCommand`` when *inner* is built from one
    (K1). Its payload is the agent's own command and *build* is then given
    the command rebuilt around the reader (``AgentExecCommand.rebuilt``),
    so the reader runs as the selected account inside the barrier.
    """
    whole = build(inner)
    if measure(whole) <= min(bound, COMMAND_BOUND):
        return Fit(whole)
    end = new_end()
    line = reader(cat, end, fd=fd)
    if agent is not None:
        return Fit(
            build(agent.rebuilt(line)),
            framed(agent.agent_command.encode(), end),
            end,
            line,
        )
    return Fit(build(line), framed(inner.encode(), end), end, line)


def _stage_root_setup(root: str) -> str:
    """Prepare the staging root only after its parents cannot be renamed."""
    import shlex

    run = shlex.quote(root)
    parents = ""
    if root.startswith("/opt/benchflow/"):
        parents = (
            "[ ! -L /opt ] && mkdir -p /opt && "
            "{ bfc_dir /opt s || { chown -h 0:0 /opt && [ ! -L /opt ] && "
            "chmod 1777 /opt; }; } && "
            "[ ! -L /opt/benchflow ] && mkdir -p /opt/benchflow && "
            "chown -h 0:0 /opt/benchflow && [ ! -L /opt/benchflow ] && "
            "chmod go-w /opt/benchflow && "
        )
    return (
        f"{parents}[ ! -L {run} ] && mkdir -p {run} && chown -h 0:0 {run} && "
        f"[ ! -L {run} ] && chmod 700 {run}"
    )


def prepare_stage_root(*, shell: str, path: str, root: str = STAGE_ROOT) -> str:
    """Prepare and prove root's staging folder before creating any payload.

    Reuse the same parent custody as :func:`stage_proof`; checking payloads
    after upload cannot undo an earlier chmod through a substituted link.
    """
    import shlex

    from benchflow._root_only import ROOT_CHAIN_FUNCS
    from benchflow.sandbox._base import system_shell_prefix

    body = (
        f"{system_shell_prefix(path)}; {ROOT_CHAIN_FUNCS}"
        f"{_stage_root_setup(root)} && bfc_chain {shlex.quote(root)} 0"
    )
    return f"( {system_shell_prefix(path)}; {shell} -c {shlex.quote(body)} )"


def stage_proof(
    files: Sequence[str], *, shell: str, path: str, root: str = STAGE_ROOT
) -> str:
    """sh, run by root's shell before it opens any of *files* (staged
    payloads in one fresh folder under *root*, :data:`STAGE_ROOT` or
    AgentCore's sealed folder before the kit): status 0 only when *root* is
    no link, is made root's with mode 700 (as the private staging folder
    always is, PT-C8-R3-8), and the folder and every file pass the root-only
    walk (``bfc_chain``: every folder and link on the way root's alone). A
    subshell, on *shell* by absolute path with PATH *path* (the route's
    tools), so nothing it sets reaches the command.

    Under ``/opt/benchflow`` the folders above *root* are first made safe
    the way the kit's preparation makes them (``/opt`` and ``/opt/benchflow``
    no link; ``/opt`` root's or sticky, else root's and 1777;
    ``/opt/benchflow`` root's, nobody else may write): the kit's own build
    is a staged command, so an honest image whose ``/opt`` another user owns
    is made safe here first, as the kit would make it, and is not refused.

    Nobody but root can change a route this proves, so what root opens next
    is what root staged (DX-1 review A M1)."""
    import shlex

    from benchflow._root_only import ROOT_CHAIN_FUNCS
    from benchflow.sandbox._base import system_shell_prefix

    folder = files[0].rsplit("/", 1)[0]
    chain = " && ".join(f"bfc_chain {shlex.quote(p)} 0" for p in (root, folder, *files))
    # A folder is made root's without following a link (``chown -h``) and
    # is tested again before its mode changes: once root's, in a parent
    # only root may rename in (root's, sticky or not writable by others),
    # nobody else can swap it for a link, so ``chmod`` changes that folder
    # and no link's target (DAYTONA-GAPS review A L2: a link swapped in
    # after the first test).
    body = (
        f"{system_shell_prefix(path)}; {ROOT_CHAIN_FUNCS}"
        f"{_stage_root_setup(root)} && {chain}"
    )
    # As every pinned root step starts (``_session_command``): the loader
    # and start-up names unset and PATH pinned before the shell starts.
    return f"( {system_shell_prefix(path)}; {shell} -c {shlex.quote(body)} )"


def open_staged(
    command: str, files: dict[int, str], proof: str, *, first: str = ""
) -> str:
    """Root's shell text that runs *command* with each staged file of
    *files* (descriptor to path) open on its descriptor, only after
    *proof* (:func:`stage_proof`) passed; otherwise a ``benchflow:`` line
    and status :data:`READ_FAILED`, and nothing of *command* runs. *first*:
    text run inside the opened group before *command* (AgentCore removes
    the folder there)."""
    import shlex

    redirects = " ".join(
        f"{'' if fd == 0 else fd}<{shlex.quote(path)}"
        for fd, path in sorted(files.items())
    )
    folder = next(iter(files.values())).rsplit("/", 1)[0]
    refuse = shlex.quote(
        f"benchflow: the staged command's folder {folder} is not root's "
        "alone; nothing of it ran"
    )
    return (
        f"if {proof}; then {{ {first}{command}\n}} {redirects}; "
        f"else echo {refuse} >&2; (exit {READ_FAILED}); fi"
    )
