"""Mock cloud metadata endpoints for real-Docker tests.

A test never connects to a real metadata server: on a cloud VM the endpoints
are the VM's own credentials service. Before anything in a sandbox connects
to an endpoint, a container joined to main's network namespace puts each IPv4
endpoint on main's loopback and listens on port 80 there, so a connect to an
endpoint never leaves the namespace (connects leave from 127.0.0.1: from the
endpoint's own address, the block's TCP reset would go to that address and
the rule would refuse its own reset). The IPv6 endpoints are not mocked
(Docker gives these containers no IPv6), so no test connects to them.

:func:`mock_endpoints` starts the mock for one main and returns once main's
namespace holds every IPv4 endpoint as a local address; :class:`MockWatcher`
does it for every main a ``bench`` run starts that a test names.
"""

from __future__ import annotations

import contextlib
import subprocess
import threading
import time
from collections.abc import Callable, Iterator

from benchflow.sandbox import _metadata_helper as helper_mod
from benchflow.sandbox.cloud_metadata import CLOUD_METADATA_V4

MOCK_SH = (
    " ".join(
        f"ip addr add {a}/32 dev lo; "
        f"ip route replace local {a} dev lo table local src 127.0.0.1;"
        for a in CLOUD_METADATA_V4
    )
    # BusyBox's persistent server keeps the listening socket open between
    # connects. Restarting nc after each accept races the next endpoint.
    + " exec nc -lk -p 80 -e /bin/true"
)

# Prints the endpoints that are local addresses of the namespace it runs in.
LOCAL_ENDPOINTS_SH = (
    "for a in "
    + " ".join(CLOUD_METADATA_V4)
    + '; do awk -v a="$a" \'$1 == "|--" && $2 == a {f = 1; next} '
    "f && /host LOCAL/ {print a; exit} {f = 0}' /proc/net/fib_trie; done"
)


def _docker(*args: str, timeout: float = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout
    )


def start_mock(main: str, label: str) -> str:
    """Start the mock in ``main``'s namespace; its container ID once every
    IPv4 endpoint is a local address there. Raises if that never happens."""
    started = _docker(
        "run",
        "--detach",
        "--rm",
        "--label",
        label,
        "--network",
        f"container:{main}",
        "--cap-drop",
        "ALL",
        "--cap-add",
        "NET_ADMIN",
        "--cap-add",
        "NET_BIND_SERVICE",
        helper_mod.HELPER_BASE,
        "sh",
        "-c",
        MOCK_SH,
        timeout=300,
    )
    mock = started.stdout.strip()
    if started.returncode != 0 or not mock:
        raise RuntimeError(f"the metadata mock did not start: {started.stderr}")
    for _ in range(100):
        local = _docker("exec", mock, "sh", "-c", LOCAL_ENDPOINTS_SH).stdout.split()
        if set(local) == set(CLOUD_METADATA_V4):
            return mock
        time.sleep(0.1)
    _docker("rm", "--force", mock)
    raise RuntimeError("the metadata mock's endpoints never became local")


@contextlib.contextmanager
def mock_endpoints(main: str, label: str) -> Iterator[str]:
    """The mock for one main, removed afterwards."""
    mock = start_mock(main, label)
    try:
        yield mock
    finally:
        _docker("rm", "--force", mock)


class MockWatcher:
    """Mocks the endpoints for every running container ``wanted`` accepts
    (its ID and labels), as soon as it runs, until stopped; then removes
    every mock it started (by ID) and anything else carrying ``label``."""

    def __init__(self, wanted: Callable[[str, dict[str, str]], bool], label: str):
        self.wanted = wanted
        self.label = label
        self.mains: list[str] = []
        self.mocks: list[str] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)

    def __enter__(self) -> MockWatcher:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=60)
        for mock in self.mocks:
            _docker("rm", "--force", mock)
        left = _docker("ps", "-aq", "--filter", f"label={self.label}").stdout.split()
        if left:
            _docker("rm", "--force", *left)

    def _watch(self) -> None:
        while not self._stop.is_set():
            running = _docker(
                "ps",
                "--no-trunc",
                "--filter",
                "status=running",
                "--format",
                "{{.ID}}\t{{.Labels}}",
            ).stdout.splitlines()
            for line in running:
                cid, _, raw = line.partition("\t")
                if cid in self.mains:
                    continue
                labels = dict(kv.split("=", 1) for kv in raw.split(",") if "=" in kv)
                if not self.wanted(cid, labels):
                    continue
                self.mains.append(cid)
                try:
                    self.mocks.append(start_mock(cid, self.label))
                except Exception as exc:  # the test sees it in .errors
                    self.errors.append(f"{cid[:12]}: {exc}")
            time.sleep(0.2)


def untag_recorded_helpers(state_dirs: list[str]) -> list[str]:
    """Remove the kept tag of every helper image recorded in ``state_dirs``
    (the per-test state folders conftest gives), so a real-Docker suite
    leaves no helper image of its own behind. Only those tags: the shared
    tag, and anything another process built, stay. Returns the tags removed."""
    import json
    from pathlib import Path

    removed = []
    for folder in state_dirs:
        for record in Path(folder).glob("*.json"):
            try:
                image = json.loads(record.read_text()).get("image")
            except (OSError, ValueError):
                continue
            if not isinstance(image, str):
                continue
            tag = helper_mod.kept_tag(image)
            if _docker("rmi", tag).returncode == 0:
                removed.append(tag)
            for daemon, ready in list(helper_mod._READY.items()):
                if ready == image:
                    helper_mod._READY.pop(daemon, None)
    return removed
