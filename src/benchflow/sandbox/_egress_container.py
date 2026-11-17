"""The verifier phase's egress proxy in a container of its own.

test.sh runs as root in ``main``. In main the proxy is a root process of
main's pid namespace, so test.sh could signal it (``kill``, ``kill -STOP``,
``kill -0``) even without NET_ADMIN or SYS_PTRACE, and could rewrite its
refusal log. On Docker the verifier phase's proxy runs instead in a container
of its own that joins main's network namespace (``--network container:main``)
and nothing else: its own pid namespace, where no process of main can name
it, and its own filesystem, where its log of refusals is out of test.sh's
reach. It holds NET_ADMIN only, to mark its upstream sockets (SO_MARK), which
the verifier's rules let out; nothing else leaves. It is BenchFlow's
metadata-helper image (``_metadata_helper``), whose recipe brings python3 and
openssl, run read-only with the proxy's files on a tmpfs.

Killed or gone, it fails closed as the in-sandbox proxy does: nothing
unmarked leaves main, and no process there can mark.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import secrets
import shlex
import tarfile
from collections.abc import Mapping
from typing import Any

from benchflow.sandbox import _metadata_helper as helper_mod
from benchflow.sandbox._exec_receipt import (
    new_receipt_nonce,
    read_receipt,
    receipt_command,
)

#: Where the proxy's files are, in its own container.
EGRESS_DIR = "/run/egress"
BLOCKED_LOG = f"{EGRESS_DIR}/blocked.jsonl"
#: The container stops itself after this long past the verifier's own
#: timeout, whatever happens to BenchFlow.
LIFETIME_MARGIN_SEC = 900
#: How long its start waits for BenchFlow to place the files.
_START_WAIT_SEC = 120


class EgressContainerError(RuntimeError):
    """The proxy's container could not be started or read."""


def _tar(files: Mapping[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        folders = sorted({name.rsplit("/", 1)[0] for name in files if "/" in name})
        for folder in folders:
            info = tarfile.TarInfo(folder)
            info.type, info.mode = tarfile.DIRTYPE, 0o700
            tar.addfile(info)
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o600
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def proxy_argv(*, port: int, dns_port: int | None, mark: int) -> list[str]:
    """The proxy's command line in its container."""
    return [
        "python3",
        "-I",
        "-S",
        "-B",
        f"{EGRESS_DIR}/proxy.py",
        "--port",
        str(port),
        "--policy",
        f"{EGRESS_DIR}/policy.json",
        "--cert-dir",
        f"{EGRESS_DIR}/certs",
        "--ca-cert",
        f"{EGRESS_DIR}/ca.crt",
        "--ca-key",
        f"{EGRESS_DIR}/ca.key",
        "--openssl",
        "/usr/bin/openssl",
        "--log",
        BLOCKED_LOG,
        "--allowed-log",
        f"{EGRESS_DIR}/allowed.jsonl",
        *(["--dns-port", str(dns_port)] if dns_port is not None else []),
        "--mark",
        str(mark),
    ]


class EgressContainer:
    """One sandbox's verifier-phase proxy container, in ``main``'s network
    namespace. ``docker`` runs a docker CLI command on the sandbox's daemon."""

    def __init__(
        self,
        docker: helper_mod.DockerRun,
        *,
        daemon: str,
        name: str,
        labels: Mapping[str, str],
    ) -> None:
        self._docker = docker
        self._daemon = daemon
        self.name = f"{name}-{secrets.token_hex(6)}"
        self._labels = dict(labels)
        self.container: str | None = None

    async def start(
        self,
        main: str,
        files: Mapping[str, bytes],
        argv: list[str],
        *,
        lifetime_sec: int,
    ) -> None:
        """Run the container in *main*'s network namespace, place *files* in
        it and start the proxy (*argv*). Raises :class:`EgressContainerError`."""
        self.container = await self._run_container(main, argv, lifetime_sec)
        placed = await self._docker(
            ["exec", "-i", self.container, "tar", "-x", "-C", EGRESS_DIR, "-f", "-"],
            timeout=helper_mod._RUN_SEC,
            stdin=_tar(files),
        )
        if placed.return_code != 0:
            raise EgressContainerError(
                "the egress proxy's files could not be placed: "
                + (helper_mod._last_line(placed.stderr or "") or "no output")
            )
        out = await self.run(f"touch {EGRESS_DIR}/.go && echo placed")
        if "placed" not in out:
            raise EgressContainerError("the egress proxy could not be started")

    async def _run_container(
        self, main: str, argv: list[str], lifetime_sec: int
    ) -> str:
        """``docker run`` of the container (no exec): its id, as the run
        printed it."""
        image = await helper_mod.ensure_image(self._docker, self._daemon)
        go = f"{EGRESS_DIR}/.go"
        # The container's only process waits for its files, then becomes the
        # proxy under a deadline: when the proxy ends, the container does.
        boot = (
            f"i=0; while [ ! -e {go} ]; do i=$((i + 1)); "
            f"[ $i -gt {_START_WAIT_SEC * 5} ] && exit 1; sleep 0.2; done; "
            f"exec timeout {int(lifetime_sec)} "
            + " ".join(shlex.quote(a) for a in argv)
        )
        labels = [f"--label={k}={v}" for k, v in sorted(self._labels.items())]
        started = await self._docker(
            [
                "run",
                "--detach",
                "--rm",
                "--name",
                self.name,
                "--network",
                f"container:{main}",
                "--user",
                "0:0",
                "--cap-drop",
                "ALL",
                # SO_MARK on its upstream sockets, and nothing more.
                "--cap-add",
                "NET_ADMIN",
                "--security-opt",
                "no-new-privileges",
                "--read-only",
                "--tmpfs",
                f"{EGRESS_DIR}:rw,noexec,nosuid,nodev,size=64m,mode=0700",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,nodev,size=16m",
                "--ipc",
                "none",
                "--pids-limit",
                "512",
                "--memory",
                "512m",
                f"--label={helper_mod.HELPER_LABEL}=metadata",
                *labels,
                "--entrypoint",
                "/bin/sh",
                image,
                "-c",
                boot,
            ],
            timeout=helper_mod._RUN_SEC,
        )
        container = helper_mod._last_line(started.stdout or "")
        if started.return_code != 0 or not helper_mod._CONTAINER_ID.fullmatch(
            container
        ):
            raise EgressContainerError(
                "the egress proxy's container did not start: "
                + (
                    helper_mod._last_line(started.stderr or started.stdout or "")
                    or "no output"
                )
            )
        if container == main:
            raise EgressContainerError("the egress proxy's run reported main's id")
        return container

    async def run(self, command: str, *, timeout_sec: int = 60) -> str:
        """Root's *command* in the container: its output ("" when the exec
        failed), read as a checked receipt like the helper's."""
        if self.container is None:
            return ""
        nonce = new_receipt_nonce()
        try:
            result = await self._docker(
                [
                    "exec",
                    self.container,
                    "/bin/sh",
                    "-c",
                    receipt_command(command, nonce),
                ],
                timeout=timeout_sec,
            )
        except Exception:
            return ""
        read = read_receipt(result, nonce)
        out = getattr(read, "stdout", "") or ""
        return out if isinstance(out, str) else ""

    async def wait_healthy(self, url: str, timeout_sec: int) -> None:
        """Until the proxy answers *url* (its ``/healthz``) from inside its
        own container, which has the Python main's image may lack. Raises
        :class:`EgressContainerError` after *timeout_sec*."""
        probe = (
            "import urllib.request,sys;"
            "o=urllib.request.build_opener(urllib.request.ProxyHandler({}));"
            f"sys.exit(0 if o.open({url!r},timeout=2).status==200 else 1)"
        )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_sec
        while True:
            out = await self.run(
                f"python3 -I -S -c {shlex.quote(probe)} 2>/dev/null && echo BF egress-healthy",
                timeout_sec=30,
            )
            if "BF egress-healthy" in out:
                return
            if loop.time() >= deadline:
                break
            await asyncio.sleep(0.5)
        logs = ""
        with contextlib.suppress(Exception):
            got = await self._docker(
                ["logs", "--tail", "20", self.container or self.name],
                timeout=helper_mod._RUN_SEC,
            )
            logs = (got.stderr or got.stdout or "").strip()[-500:]
        raise EgressContainerError(
            f"the egress proxy's container did not answer within {timeout_sec}s"
            + (f": {logs}" if logs else "")
        )

    async def refusals(self) -> list[dict[str, Any]] | None:
        """The proxy's refusals so far (its blocked log, one JSON object a
        line), or None when the container cannot be read: a proxy that is
        gone refused nothing more."""
        out = await self.run(
            f"if [ -e {BLOCKED_LOG} ]; then cat {BLOCKED_LOG}; fi; echo BF egress-log-end"
        )
        if "BF egress-log-end" not in out:
            return None
        found: list[dict[str, Any]] = []
        for line in out.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            with contextlib.suppress(ValueError):
                entry = json.loads(line)
                if isinstance(entry, dict):
                    found.append(entry)
        return found

    async def aclose(self) -> None:
        target, self.container = self.container or self.name, None
        with contextlib.suppress(Exception):
            await self._docker(["rm", "--force", target], timeout=helper_mod._RUN_SEC)
