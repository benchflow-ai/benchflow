"""The real access probe script on generated trees, as root in a container.

Each tree from ``tests/integrity_fs_model.py`` is built for real (folders,
files, links, owners, groups, modes, sticky bits; names with newlines, tabs,
spaces, backslashes, leading dashes and non-UTF-8 bytes) in its own directory
of a throwaway container, with the tools the probe uses copied in, and the
probe runs under ``chroot`` there as root. Trees and the probe command travel
as bytes (hex in JSON) to a builder inside the container, so no name is
re-spelt by a shell on the way. What the probe prints must equal what the
model says it prints, so the model test's verdicts hold for the real script;
its assessment must also never certify what the reference shows open.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from benchflow.integrity.access import parse_probe, path_state, probe_command
from tests.integrity_fs_model import Node, Tree, random_case

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
]

IMAGE = "python:3.12-slim"
SEED = 20261019
TREES = 2000

# A minimal root for every tree: the probe's tools, their libraries,
# /dev/null and the agent user. Real folders, so a tree's own entries never
# meet a link here.
_BASE = r"""
set -eu
mkdir -p /base/bin /base/usr/bin /base/etc /base/dev
mknod -m 666 /base/dev/null c 1 3
cp /bin/dash /base/bin/sh
for t in env stat ls readlink od tr; do cp "$(readlink -f /usr/bin/$t)" /base/usr/bin/$t; done
for f in /base/bin/sh /base/usr/bin/*; do
  ldd "$f" | awk '/=> \//{print $3} /^\t\//{print $1}' | while read -r lib; do
    mkdir -p "/base$(dirname "$lib")"; cp -n "$(readlink -f "$lib")" "/base$lib"
  done
done
printf 'root:x:0:0::/:/bin/sh\nagent:x:1000:1000::/:/bin/sh\n' > /base/etc/passwd
printf 'root:x:0:\nagent:x:1000:\nother:x:2000:\n' > /base/etc/group
"""

# Runs inside the container as root: build each tree from bytes, probe it.
_BUILDER = r"""
import json, os, subprocess, sys
spec = json.load(open(sys.argv[1]))
out = sys.stdout.buffer
for case in spec:
    root = b"/t/%d" % case["index"]
    os.makedirs(b"/t", exist_ok=True)
    subprocess.run([b"cp", b"-a", b"/base/.", root + b"/"] if os.path.isdir(root)
                   else [b"cp", b"-a", b"/base", root], check=True)
    nodes = [(bytes.fromhex(p), kind, uid, gid, mode, bytes.fromhex(t))
             for p, kind, uid, gid, mode, t in case["nodes"]]
    for path, kind, uid, gid, mode, target in sorted(nodes, key=lambda n: n[0].count(b"/")):
        if path == b"/":
            continue
        real = root + path
        if kind == "dir":
            os.mkdir(real)
        elif kind == "file":
            open(real, "wb").close()
        else:
            os.symlink(target, real)
    for path, kind, uid, gid, mode, target in nodes:
        real = root if path == b"/" else root + path
        os.lchown(real, uid, gid)
        if kind != "link":
            os.chmod(real, mode)
    run = subprocess.run([b"chroot", root, b"/bin/sh", b"-c", bytes.fromhex(case["command"])],
                         capture_output=True)
    out.write(b"=== %d\n" % case["index"])
    out.write(run.stdout)
    subprocess.run([b"rm", b"-rf", root], check=True)
out.flush()
"""


def _hex(text: str) -> str:
    return text.encode("utf-8", "surrogateescape").hex()


def _docker_ready() -> bool:
    try:
        return (
            subprocess.run(
                ["docker", "info"], capture_output=True, timeout=10
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


def _real_probe(tmp_path: Path, cases: list[tuple[Tree, str | list[str]]]) -> list[str]:
    """The real probe's output for each ``(tree, protected path or paths)``."""
    if not _docker_ready():
        pytest.skip("docker daemon not running")
    spec = [
        {
            "index": index,
            "command": _hex(
                probe_command("agent", path if isinstance(path, list) else [path])
            ),
            "nodes": [
                [_hex(p), n.kind, n.uid, n.gid, n.mode, _hex(n.target)]
                for p, n in tree.nodes.items()
            ],
        }
        for index, (tree, path) in enumerate(cases)
    ]
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    (tmp_path / "builder.py").write_text(_BUILDER)
    (tmp_path / "base.sh").write_text(_BASE)
    name = f"bf-access-probe-{uuid.uuid4().hex[:8]}"
    try:
        start = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "--network",
                "none",
                "--label",
                "target=FF-sw-integrity",
                IMAGE,
                "sleep",
                "1800",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert start.returncode == 0, start.stderr[-500:]
        for file in ("spec.json", "builder.py", "base.sh"):
            subprocess.run(
                ["docker", "cp", str(tmp_path / file), f"{name}:/{file}"],
                check=True,
                capture_output=True,
            )
        subprocess.run(
            ["docker", "exec", name, "sh", "/base.sh"],
            check=True,
            capture_output=True,
            timeout=300,
        )
        run = subprocess.run(
            ["docker", "exec", name, "python3", "/builder.py", "/spec.json"],
            capture_output=True,
            timeout=1800,
        )
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    assert run.returncode == 0, run.stderr[-2000:]
    chunks = run.stdout.decode("ascii").split("=== ")[1:]
    assert len(chunks) == len(cases)
    return [chunk.partition("\n")[2] for chunk in chunks]


def test_r9_1_the_real_probe_follows_a_target_ending_in_a_newline(tmp_path):
    """A link whose target ends in a newline: the real probe must report the
    real target, and the host must see it readable and writable."""
    tree = Tree({"/": Node("dir", 0, 0, 0o755, ident=0)})
    tree.nodes["/shared"] = Node("dir", 0, 0, 0o755, ident=1)
    tree.nodes["/shared/current"] = Node("link", 0, 0, 0o777, "data\n", ident=2)
    tree.nodes["/shared/data\n"] = Node("file", 0, 0, 0o666, ident=3)
    (real,) = _real_probe(tmp_path, [(tree, "/shared/current")])
    probe = parse_probe(real)
    assert path_state(probe, "/shared/current") == "open"
    assert path_state(probe, "/shared/current", write=True) == "open"


def test_the_real_probe_prints_what_the_model_resolves(tmp_path):
    rng = random.Random(SEED)
    cases = [random_case(rng) for _ in range(TREES)]
    outputs = _real_probe(tmp_path, cases)
    for index, (real, (tree, path)) in enumerate(zip(outputs, cases, strict=True)):
        expected = tree.probe_output([path])
        context = f"tree {index}: {path!r} in {sorted(tree.nodes.items())}"
        assert real == expected, f"{context}\nreal:\n{real}\nmodel:\n{expected}"
        probe = parse_probe(real)
        if tree.reference_write_open(path):
            assert path_state(probe, path, write=True) == "open", context
        if tree.reference_read_open(path):
            assert path_state(probe, path) == "open", context


E2E_TREES = 400


def test_the_whole_flow_on_the_real_probe(tmp_path):
    """derive_contract -> protection_checks -> the real probe -> assess_access
    on generated selectors (backslashes, newlines, tabs, spaces, dashes, ``..``,
    non-UTF-8 bytes), against the brute-force reference."""
    from benchflow.integrity.access import assess_access, protection_checks
    from tests.test_integrity_access_e2e import _CLASSES, _contract, _selector

    rng = random.Random(SEED + 1)
    flows = []
    while len(flows) < E2E_TREES:
        tree, _path = random_case(rng)
        selectors = [(_selector(rng, tree), rng.choice(_CLASSES)) for _ in range(2)]
        contract = _contract("/app", selectors)
        if contract.violations:
            continue
        checks, _ = protection_checks(contract)
        flows.append((tree, contract, checks))
    outputs = _real_probe(tmp_path, [(tree, list(checks)) for tree, _, checks in flows])
    for real, (tree, contract, checks) in zip(outputs, flows, strict=True):
        assert real == tree.probe_output(list(checks))
        stdout = {"status": "ok", "stdout": real}
        assessment = assess_access(contract, {"pre": stdout, "post": stdout})
        for path, ops in checks.items():
            if "read" in ops and tree.reference_read_open(path):
                assert path in assessment.read_open, path
            if "write" in ops and tree.reference_write_open(path):
                assert path in assessment.write_open, path
