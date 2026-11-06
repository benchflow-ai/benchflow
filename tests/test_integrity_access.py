"""The integrity access probe (benchflow.integrity.access): facts, not parsing."""

from __future__ import annotations

import asyncio
import os
import pwd
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from benchflow.integrity.access import (
    PROBE_SCRIPT,
    assess_access,
    parse_probe,
    path_state,
    probe_command,
    probe_paths,
    run_access_probe,
)
from benchflow.integrity.contract import (
    ContractFacts,
    LoadedBinding,
    TaskBinding,
    derive_contract,
)
from benchflow.sandbox.protocol import ExecResult
from tests.integrity_fs_model import Node, Tree, encode_records


def _contract(binding: dict | None = None):
    loaded = (
        LoadedBinding(
            path=Path("benchguard.yaml"),
            sha256="sha256:x",
            binding=TaskBinding.model_validate(binding),
        )
        if binding
        else None
    )
    return derive_contract(ContractFacts(task_id="t", workspace="/app"), binding=loaded)


def _stdout(*paths: tuple[str, list[str]], uid: int = 1000, gids: str = "1000") -> str:
    """Probe output: each path with its entry lines (``S``/``L``) and end line
    (``F``/``A``/``X``). Without an end line, a path whose last line is
    an ``S`` entry resolved to that entry."""
    lines = [f"U\t{uid}\t{gids}"]
    for path, components in paths:
        lines.append(f"P\t{path}")
        lines.extend(components)
        ended = any(line[:2] in {"F\t", "A\t", "X\t"} for line in components)
        if not ended and components and components[-1].startswith("S\t"):
            lines.append(f"F\t{components[-1].rsplit(chr(9), 1)[-1]}")
    return encode_records("\n".join(lines) + "\n")


ROOT = "S\t0 0 755\tdrwxr-xr-x \t/"


@pytest.mark.parametrize(
    ("component", "state"),
    [
        ("S\t0 0 700\tdrwx------ \t/solution", "closed"),
        ("S\t0 0 755\tdrwxr-xr-x \t/solution", "open"),
        ("S\t0 0 711\tdrwx--x--x \t/solution", "open"),  # names still reachable
        ("S\t1000 0 700\tdrwx------ \t/solution", "open"),  # the agent owns it
        ("S\t0 0 700\tdrwx------+\t/solution", "open"),  # an ACL can grant
        ("S\t0 1000 750\tdrwxr-x--- \t/solution", "open"),  # the agent's group
        ("S\t0 0 750\tdrwxr-x--- \t/solution", "closed"),  # another group
        # Listing without search still shows the protected names.
        ("S\t0 0 744\tdrwxr--r-- \t/solution", "open"),
        ("S\t0 0 704\tdrwx---r-- \t/solution", "open"),
        ("L\t/solution", "open"),  # a link can point anywhere
        ("A\t/solution", "absent"),
    ],
)
def test_path_state_reads_the_facts_conservatively(component, state):
    probe = parse_probe(_stdout(("/solution", [ROOT, component])))
    assert path_state(probe, "/solution") == state


def test_a_closed_ancestor_closes_everything_below():
    probe = parse_probe(
        _stdout(
            (
                "/app/.grader/key",
                [
                    ROOT,
                    "S\t1000 1000 755\tdrwxr-xr-x \t/app",
                    "S\t0 0 700\tdrwx------ \t/app/.grader",
                    "S\t0 0 644\t-rw-r--r-- \t/app/.grader/key",
                ],
            )
        )
    )
    assert path_state(probe, "/app/.grader/key") == "closed"


def test_a_root_agent_reaches_everything():
    probe = parse_probe(
        _stdout(("/solution", [ROOT, "S\t0 0 700\tdrwx------ \t/solution"]), uid=0)
    )
    assert path_state(probe, "/solution") == "open"


def _record(pre: str, post: str | None = None) -> dict:
    return {
        "pre": {"status": "ok", "stdout": pre},
        "post": {"status": "ok", "stdout": pre if post is None else post},
    }


def _all(contract, mode: str) -> str:
    paths, _ = probe_paths(contract)
    return _stdout(
        *[
            (
                p,
                [
                    ROOT,
                    f"S\t0 0 {mode}\tdrwx------ \t{p}"
                    if mode == "700"
                    else f"S\t0 0 {mode}\tdrwxr-xr-x \t{p}",
                ],
            )
            for p in paths
        ]
    )


def test_a_path_open_at_either_probe_is_open():
    contract = _contract()
    closed, opened = _all(contract, "700"), _all(contract, "755")
    assert assess_access(contract, _record(closed)).read_open == []
    assert "/solution" in assess_access(contract, _record(opened, closed)).read_open
    assert "/solution" in assess_access(contract, _record(closed, opened)).read_open


@pytest.mark.parametrize(
    "record,reason",
    [
        (None, "no access probe"),
        ({"pre": {"status": "unsupported"}, "post": {"status": "ok"}}, "pre-turn"),
        ({"pre": {"status": "ok", "stdout": ""}, "post": {"status": "failed"}}, "post"),
        ({"pre": {"status": "ok", "stdout": "E\tno-such-user\n"}}, "post"),
    ],
)
def test_missing_or_failed_probes_are_unattested(record, reason):
    assessment = assess_access(_contract(), record)
    assert assessment.status == "unattested" and reason in assessment.reason


def test_probe_paths_cover_read_and_crossing_write_protection():
    contract = _contract(
        {
            "resources": [
                {
                    "id": "g",
                    "selector": "/app/.grader",
                    "class": "Hidden",
                    "reason": "x",
                },
                {"id": "l", "selector": "/app/lib", "class": "Trusted", "reason": "x"},
                {"id": "o", "selector": "/opt/tool", "class": "Trusted", "reason": "x"},
                {
                    "id": "r",
                    "selector": "http://localhost:9005/_admin",
                    "class": "Hidden",
                    "reason": "x",
                },
            ]
        }
    )
    paths, routes = probe_paths(contract)
    assert {"/solution", "/tests", "/app/.grader", "/app/lib"} <= set(paths)
    assert "/opt/tool" not in paths  # outside what reaches the verifier
    assert "/logs/verifier" not in paths  # written after the agent
    assert routes == ["http://localhost:9005/_admin"]


class _Sandbox:
    supports_system_shell = True

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def exec_route(self, user, *, system_shell=False, plumbing=False):
        return None  # PT's root plumbing route (C8) is present

    async def exec(self, command, *, user, timeout_sec, system_shell=False):
        from benchflow.sandbox._exec_output import raw_output_requested

        self.calls.append(
            {"user": user, "system_shell": system_shell, "raw": raw_output_requested()}
        )
        return ExecResult(return_code=0, stdout="U\t1000\t1000\n", stderr="")


def test_the_probe_runs_as_root_through_the_system_shell_and_reads_raw_output():
    """The probe's stdout is machine data (hex path fields), so it is read
    through ``exec_raw``: redaction never rewrites what is parsed."""
    sandbox = _Sandbox()
    result = asyncio.run(run_access_probe(sandbox, "agent", ["/solution"]))
    assert result["status"] == "ok"
    assert sandbox.calls == [{"user": "root", "system_shell": True, "raw": True}]


def test_a_backend_without_a_system_shell_is_not_probed():
    class Plain:
        async def exec(self, *args, **kwargs):  # pragma: no cover - never called
            raise AssertionError

    assert asyncio.run(run_access_probe(Plain(), "agent", ["/x"]))["status"] == (
        "unsupported"
    )
    assert asyncio.run(run_access_probe(_Sandbox(), None, ["/x"])) == {
        "status": "agent_root"
    }


def _nobody() -> str | None:
    try:
        pwd.getpwnam("nobody")
    except KeyError:
        return None
    return "nobody"


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX shell")
def test_the_real_probe_script_reads_real_permissions(tmp_path):
    """The script under a real /bin/sh: /root (root, 0700) is closed to nobody,
    /etc is open, a missing path is absent."""
    user = _nobody()
    info = os.stat("/root") if os.path.isdir("/root") else None
    if user is None or info is None or info.st_uid != 0 or info.st_mode & 0o077:
        pytest.skip("needs a nobody user and a root-only /root")
    missing = "/benchflow-probe-missing/x"
    assert not os.path.exists("/benchflow-probe-missing")
    command = probe_command(user, ["/root", "/etc", missing])
    assert command.endswith(PROBE_SCRIPT)
    result = subprocess.run(["sh", "-c", command], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    probe = parse_probe(result.stdout)
    assert path_state(probe, "/root") == "closed"
    assert path_state(probe, "/etc") == "open"
    assert path_state(probe, missing) == "absent"
    assert stat.S_IMODE(info.st_mode) & 0o700


def test_root_never_probes_without_the_plumbing_route():
    """C8: a backend with a system shell but no root plumbing route
    lets the task's persistent env reach root's tools, so it is not probed."""

    class ShellOnly:
        supports_system_shell = True

        async def exec(self, *args, **kwargs):  # pragma: no cover - never called
            raise AssertionError("probed without the plumbing route")

    result = asyncio.run(run_access_probe(ShellOnly(), "agent", ["/solution"]))
    assert result["status"] == "unsupported"
    assert result["reason"] == "no root plumbing route on this backend"
    assessment = assess_access(_contract(), {"pre": result, "post": result})
    assert assessment.status == "unattested"


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX shell")
def test_the_probe_resolves_no_tool_from_path(tmp_path):
    """Every program the probe starts is named by an absolute system path:
    a PATH holding other programs of the same names is never consulted."""
    user = _nobody()
    if user is None or not os.path.isdir("/root"):
        pytest.skip("needs a nobody user")
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    marker = tmp_path / "consulted"
    for name in ("id", "stat", "ls", "cut", "env", "cat", "grep", "awk", "sed"):
        tool = shadow / name
        tool.write_text(f"#!/bin/sh\necho {name} >> {marker}\nexit 1\n")
        tool.chmod(0o755)
    command = probe_command(user, ["/root"])
    result = subprocess.run(
        ["/bin/sh", "-c", command],
        capture_output=True,
        text=True,
        env={"PATH": str(shadow)},
    )
    assert result.returncode == 0, result.stderr
    assert not marker.exists(), marker.read_text()
    probe = parse_probe(result.stdout)
    assert probe.error is None and probe.uid is not None
    assert path_state(probe, "/root") in {"closed", "open"}


def test_read_and_write_protection_are_assessed_independently(tmp_path):
    """A proof that the agent user cannot read a path says nothing about
    writing it: each operation is certified only by its own test."""
    from tests.test_integrity import _tool, _verdict

    binding = {
        "resources": [
            {"id": "s", "selector": "/app/state.json", "class": "Hidden", "reason": "x"}
        ]
    }
    contract = _contract(binding)
    paths, _ = probe_paths(contract)
    blocks = []
    for path in paths:
        if path == "/app/state.json":
            # Root-owned under a root-owned folder: only the file's own mode
            # lets the agent user write it, and nothing lets it read it.
            components = [
                ROOT,
                "S\t0 0 755\tdrwxr-xr-x \t/app",
                "S\t0 0 622\t-rw--w--w- \t/app/state.json",
            ]
        else:
            components = [ROOT, f"S\t0 0 700\tdrwx------ \t{path}"]
        blocks.append((path, components))
    record = _record(_stdout(*blocks))
    claim = _verdict(
        tmp_path,
        [_tool("Write", file_path="/app/state.json")],
        binding=TaskBinding.model_validate(binding),
        access=record,
    )
    # The structured write is not hidden by the read proof.
    assert claim["exploited"] and "protected_writable" in claim["profile_vectors"]
    assessment = assess_access(contract, record)
    assert "/app/state.json" in assessment.read_closed
    assert "/app/state.json" in assessment.write_open
    assert "/app/state.json" not in assessment.closed
    assert not assessment.closes("/app/state.json", write=True)


_P = "/shared/private"


@pytest.mark.parametrize(
    ("components", "state"),
    [
        # A root-owned 0700 folder under folders the agent user cannot write.
        (
            [
                ROOT,
                "S\t0 0 755\tdrwxr-xr-x \t/shared",
                f"S\t0 0 700\tdrwx------ \t{_P}",
            ],
            "closed",
        ),
        # Its parent is writable: the folder can be renamed or replaced.
        (
            [
                ROOT,
                "S\t0 0 777\tdrwxrwxrwx \t/shared",
                f"S\t0 0 700\tdrwx------ \t{_P}",
            ],
            "open",
        ),
        # A sticky parent stops that for an entry the agent user does not own.
        (
            [
                ROOT,
                "S\t0 0 1777\tdrwxrwxrwt \t/shared",
                f"S\t0 0 700\tdrwx------ \t{_P}",
            ],
            "closed",
        ),
        # A sticky parent the agent user owns does not.
        (
            [
                ROOT,
                "S\t1000 0 1777\tdrwxrwxrwt \t/shared",
                f"S\t0 0 700\tdrwx------ \t{_P}",
            ],
            "open",
        ),
        # A writable folder higher up can move the whole subtree.
        (
            [
                "S\t0 0 777\tdrwxrwxrwx \t/",
                "S\t0 0 755\tdrwxr-xr-x \t/shared",
                f"S\t0 0 700\tdrwx------ \t{_P}",
            ],
            "open",
        ),
        # Below a folder the agent user cannot search, nothing is reachable.
        (
            [
                ROOT,
                "S\t0 0 700\tdrwx------ \t/shared",
                f"S\t0 0 777\tdrwxrwxrwx \t{_P}",
            ],
            "closed",
        ),
        # A writable group parent counts too.
        (
            [
                ROOT,
                "S\t0 1000 770\tdrwxrwx--- \t/shared",
                f"S\t0 0 700\tdrwx------ \t{_P}",
            ],
            "open",
        ),
    ],
)
def test_write_closure_accounts_for_replacement_through_any_parent(components, state):
    """A path is write-closed only when no folder on the way from the root
    lets the agent user rename, move or delete-and-recreate what lies below."""
    probe = parse_probe(_stdout((_P, components)))
    assert path_state(probe, _P, write=True) == state


def test_a_replaceable_protected_folder_is_not_certified(tmp_path):
    """The assessment keeps a protected folder under a writable workspace out
    of write_closed, so a structured write into it is not set aside."""
    from tests.test_integrity import _tool, _verdict

    binding = {
        "resources": [
            {"id": "p", "selector": "/app/private", "class": "Hidden", "reason": "x"}
        ]
    }
    contract = _contract(binding)
    blocks = []
    for path in probe_paths(contract)[0]:
        if path == "/app/private":
            components = [
                ROOT,
                "S\t0 0 777\tdrwxrwxrwx \t/app",
                "S\t0 0 700\tdrwx------ \t/app/private",
            ]
        else:
            components = [ROOT, f"S\t0 0 700\tdrwx------ \t{path}"]
        blocks.append((path, components))
    record = _record(_stdout(*blocks))
    claim = _verdict(
        tmp_path,
        [_tool("Write", file_path="/app/private/state.json")],
        binding=TaskBinding.model_validate(binding),
        access=record,
    )
    assert claim["exploited"] and "protected_writable" in claim["profile_vectors"]
    assessment = assess_access(contract, record)
    assert "/app/private" in assessment.read_closed
    assert "/app/private" in assessment.write_open


# --- named cases: R6-2, R7-1, R8-1 (and the model in test_integrity_access_model) --


def _named_tree(spec: dict[str, tuple]) -> Tree:
    """``{path: (kind, uid, mode[, target])}`` under a root-owned 0755 root."""
    tree = Tree({"/": Node("dir", 0, 0, 0o755, ident=0)})
    for ident, (path, (kind, uid, mode, *target)) in enumerate(spec.items(), start=1):
        tree.nodes[path] = Node(kind, uid, 0, mode, target[0] if target else "", ident)
    return tree


def _states(tree: Tree, path: str) -> tuple[str, str]:
    probe = parse_probe(tree.probe_output([path]))
    return path_state(probe, path), path_state(probe, path, write=True)


def test_r6_2_a_writable_parent_lets_the_agent_replace_a_closed_folder():
    tree = _named_tree(
        {"/shared": ("dir", 0, 0o777), "/shared/private": ("dir", 0, 0o700)}
    )
    assert _states(tree, "/shared/private") == ("closed", "open")
    assert tree.reference_write_open("/shared/private")


def test_r7_1_a_link_behind_an_unsearchable_folder_stays_closed():
    tree = _named_tree(
        {
            "/shared": ("dir", 0, 0o755),
            "/shared/private": ("dir", 0, 0o700),
            "/shared/private/current": ("link", 0, 0o777, "payload"),
            "/shared/private/payload": ("dir", 0, 0o700),
        }
    )
    assert _states(tree, "/shared/private/current") == ("closed", "closed")
    assert not tree.reference_write_open("/shared/private/current")
    assert not tree.reference_read_open("/shared/private/current")


def test_r8_1_a_link_target_under_a_writable_parent_is_replaceable():
    """The link itself is unreachable, but its target is reached by its real
    path and its folder is writable: the target can be swapped."""
    tree = _named_tree(
        {
            "/shared": ("dir", 0, 0o755),
            "/shared/private": ("dir", 0, 0o700),
            "/shared/private/current": ("link", 0, 0o777, "../work/current"),
            "/shared/work": ("dir", 0, 0o777),
            "/shared/work/current": ("dir", 0, 0o700),
            "/shared/work/alternate": ("dir", 0, 0o700),
        }
    )
    read, write = _states(tree, "/shared/private/current")
    assert write == "open" and tree.reference_write_open("/shared/private/current")
    assert read == "closed"


def test_a_reachable_link_entry_in_a_writable_folder_is_replaceable():
    tree = _named_tree(
        {
            "/shared": ("dir", 0, 0o777),
            "/shared/current": ("link", 0, 0o777, "/data"),
            "/data": ("dir", 0, 0o700),
        }
    )
    assert _states(tree, "/shared/current")[1] == "open"


def test_a_link_loop_is_unresolved_and_never_certified():
    tree = _named_tree({"/a": ("link", 0, 0o777, "b"), "/b": ("link", 0, 0o777, "a")})
    assert _states(tree, "/a") == ("open", "open")


def test_an_honest_run_with_an_unreachable_protected_link_is_checked(tmp_path):
    """R7-1 on a stored trial: the protected link and its target sit behind an
    unsearchable folder, so the run is certified."""
    from tests.test_integrity import _calls, _verdict

    selector = "/app/private/current"
    binding = {
        "resources": [
            {"id": "p", "selector": selector, "class": "Hidden", "reason": "x"}
        ]
    }
    contract = _contract(binding)
    tree = _named_tree(
        {
            "/app": ("dir", 0, 0o755),
            "/app/private": ("dir", 0, 0o700),
            "/app/private/current": ("link", 0, 0o777, "payload"),
            "/app/private/payload": ("dir", 0, 0o700),
        }
    )
    for path in probe_paths(contract)[0]:
        if path != selector:
            tree.nodes.setdefault(path, Node("dir", 0, 0, 0o700, ident=99))
    stdout = tree.probe_output(probe_paths(contract)[0])
    claim = _verdict(
        tmp_path,
        _calls("true"),
        binding=TaskBinding.model_validate(binding),
        access=_record(stdout),
        separate=True,
        separate_record={
            "status": "complete",
            "transfer": {"workspace": "/app", "agent_paths": ["/app"]},
        },
    )
    assert claim["access"]["write_open"] == [] and claim["access"]["read_open"] == []
    assert claim["core_verdict"] == "Checked"


# --- R9-1: every pathname field is exact -----------------------------------------


def test_r9_1_a_link_target_ending_in_a_newline_resolves_to_its_real_target():
    """The kernel follows a link to a name ending in a newline; the probe and
    the host must too, so the readable, writable target is seen."""
    tree = _named_tree(
        {
            "/shared": ("dir", 0, 0o755),
            "/shared/current": ("link", 0, 0o777, "data\n"),
            "/shared/data\n": ("file", 0, 0o666),
        }
    )
    assert tree.resolve("/shared/current")[1] == "/shared/data\n"
    assert _states(tree, "/shared/current") == ("open", "open")
    assert tree.reference_read_open("/shared/current")
    assert tree.reference_write_open("/shared/current")


@pytest.mark.parametrize(
    "name",
    ["\nlead", "mid\ndle", "tab\there", "back\\slash", "-dash", "sp ace", "\udcff"],
)
def test_pathname_fields_round_trip_exactly(name):
    from benchflow.integrity.access import decode_path
    from tests.integrity_fs_model import hexpath

    path = f"/shared/{name}"
    assert decode_path(hexpath(path)) == path


@pytest.mark.parametrize(
    "field",
    ["", "2", "2G", "2F", "6162", "2f00", "2f 61", "x2f"],
)
def test_an_undecodable_path_field_leaves_the_path_unresolved(field):
    from tests.integrity_fs_model import hexpath

    good = hexpath("/solution")
    stdout = (
        f"U\t1000\t1000\nP\t{good}\nS\t0 0 755\tdrwxr-xr-x\t{hexpath('/')}\n"
        f"S\t0 0 700\tdrwx------\t{field}\nF\t{good}\n"
    )
    probe = parse_probe(stdout)
    assert path_state(probe, "/solution") == "open"
    assert path_state(probe, "/solution", write=True) == "open"


def test_records_after_the_end_or_outside_a_path_fail_closed():
    from tests.integrity_fs_model import hexpath

    p, root = hexpath("/solution"), hexpath("/")
    closed = f"S\t0 0 700\tdrwx------\t{p}\n"
    after_end = parse_probe(
        f"U\t1000\t1000\nP\t{p}\nS\t0 0 755\tdrwxr-xr-x\t{root}\n{closed}F\t{p}\n{closed}"
    )
    assert path_state(after_end, "/solution") == "open"
    stray = parse_probe(f"U\t1000\t1000\nS\t0 0 755\tdrwxr-xr-x\t{root}\n")
    assert stray.error is not None
    undecodable_path = parse_probe("U\t1000\t1000\nP\t/solution\n")
    assert undecodable_path.error is not None


def test_a_resolution_without_its_completion_record_is_unresolved():
    """A record set without the F/A/X completion marker (an earlier
    branch-only format, or a probe cut short) fails closed; no legacy decoder."""
    from tests.integrity_fs_model import hexpath

    p, root = hexpath("/solution"), hexpath("/")
    probe = parse_probe(
        f"U\t1000\t1000\nP\t{p}\nS\t0 0 755\tdrwxr-xr-x\t{root}\n"
        f"S\t0 0 700\tdrwx------\t{p}\n"
    )
    assert path_state(probe, "/solution") == "open"
