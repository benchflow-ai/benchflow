"""The whole access flow on generated selectors and trees, against the model.

``derive_contract`` (with a binding naming generated selectors) ->
``protection_checks``/``probe_paths`` -> the probe's records for that
selection (the model prints what the probe prints; the real script is held to
the same output in ``test_integrity_access_real_probe``) -> ``assess_access``.
Selectors and names carry backslashes, newlines, tabs, spaces, leading
dashes, ``..``, link names and non-UTF-8 bytes. A POSIX pathname is bytes from
end to end: each selector must be probed as written, and the verdict for it
must never certify what the brute-force reference shows open.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from benchflow.integrity.access import assess_access, protection_checks
from benchflow.integrity.contract import (
    ContractFacts,
    LoadedBinding,
    TaskBinding,
    derive_contract,
)
from tests.integrity_fs_model import _NAMES, Node, Tree, random_tree

SEED = 20261020
CASES = 3000
_CLASSES = ("Hidden", "VerifierOnly", "ResetState", "Trusted", "Public")


def as_written(selector: str) -> str:
    """A selector spelt as the sandbox reads it: the same bytes, without
    trailing slashes (which name the same entry)."""
    return selector.rstrip("/") or "/"


def _contract(workspace: str, selectors: list[tuple[str, str]]):
    binding = TaskBinding.model_validate(
        {
            "resources": [
                {"id": f"s{i}", "selector": sel, "class": cls, "reason": "generated"}
                for i, (sel, cls) in enumerate(selectors)
            ]
        }
    )
    loaded = LoadedBinding(
        path=Path("benchguard.yaml"), sha256="sha256:x", binding=binding
    )
    return derive_contract(
        ContractFacts(task_id="t", workspace=workspace), binding=loaded
    )


def _record(stdout: str) -> dict:
    return {
        "pre": {"status": "ok", "stdout": stdout},
        "post": {"status": "ok", "stdout": stdout},
    }


def _selector(rng: random.Random, tree: Tree) -> str:
    entries = [p for p in tree.nodes if p != "/"]
    base = rng.choice(entries) if entries and rng.random() < 0.6 else ""
    extra = [rng.choice((*_NAMES, "..")) for _ in range(rng.randint(0, 2))]
    path = "/".join([base, *extra]) if base else "/" + "/".join(extra or ["a"])
    return path if path.startswith("/") else "/" + path


def _flow(tree: Tree, workspace: str, selectors: list[tuple[str, str]]):
    contract = _contract(workspace, selectors)
    checks, _ = protection_checks(contract)
    assessment = assess_access(contract, _record(tree.probe_output(list(checks))))
    return contract, checks, assessment


def test_r10_1_a_backslash_selector_is_probed_as_written(tmp_path):
    """A literal backslash is a name byte in a POSIX sandbox: the Hidden file
    is probed at its own path, found readable, and the run is not Checked."""
    from tests.test_integrity import _calls, _verdict

    tree = Tree({"/": Node("dir", 0, 0, 0o755, ident=0)})
    tree.nodes["/shared"] = Node("dir", 0, 0, 0o755, ident=1)
    tree.nodes["/shared/g\\h"] = Node("file", 0, 0, 0o644, ident=2)
    selectors = [("/shared/g\\h", "Hidden")]
    _, checks, assessment = _flow(tree, "/app", selectors)
    assert "/shared/g\\h" in checks and "/shared/g/h" not in checks
    assert "/shared/g\\h" in assessment.read_open
    binding = TaskBinding.model_validate(
        {
            "resources": [
                {
                    "id": "s0",
                    "selector": "/shared/g\\h",
                    "class": "Hidden",
                    "reason": "x",
                }
            ]
        }
    )
    claim = _verdict(
        tmp_path,
        _calls("cat '/shared/g\\h'"),
        binding=binding,
        access=_record(tree.probe_output(list(checks))),
        separate=True,
        separate_record={
            "status": "complete",
            "transfer": {"workspace": "/app", "agent_paths": ["/app"]},
        },
    )
    assert claim["core_verdict"] != "Checked"
    assert "protected_readable" in claim["profile_vectors"]


@pytest.mark.parametrize(
    "selector",
    [
        "/a\\b",
        "/a/\\",
        "/a\nb",
        "/a\tb",
        "/a b",
        "/-a",
        "/a\udcffb",
        "/A",
    ],
)
def test_contract_selectors_keep_their_bytes(selector):
    contract = _contract("/app", [(selector, "Hidden")])
    checks, _ = protection_checks(contract)
    assert as_written(selector) in checks


_WRITE_CLASSES = {"Trusted", "Public", "VerifierOnly", "Hidden", "ResetState"}


def _is_canonical(selector: str) -> bool:
    """The independent rule: absolute, no empty, '.' or '..' component."""
    if "\0" in selector or not selector.startswith("/"):
        return False
    return selector == "/" or all(
        part not in {"", ".", ".."} for part in selector.split("/")[1:]
    )


@pytest.mark.parametrize(
    "selector", ["/./app/rules", "/app//rules", "/app/rules/", "/app/x/../rules"]
)
def test_r11_1_a_non_canonical_selector_is_refused_and_never_certified(selector):
    """A selector that is not canonical is an authoring error: refused at
    load with its canonical spelling, and the run is unattested."""
    tree = Tree({"/": Node("dir", 0, 0, 0o755, ident=0)})
    tree.nodes["/app"] = Node("dir", 0, 0, 0o755, ident=1)
    tree.nodes["/app/rules"] = Node("file", 0, 0, 0o666, ident=2)
    contract, _, assessment = _flow(tree, "/app", [(selector, "Trusted")])
    assert any("'/app/rules'" in v for v in contract.violations), contract.violations
    assert assessment.status == "unattested"


def test_r11_1_the_canonical_selector_is_write_checked_and_open():
    tree = Tree({"/": Node("dir", 0, 0, 0o755, ident=0)})
    tree.nodes["/app"] = Node("dir", 0, 0, 0o755, ident=1)
    tree.nodes["/app/rules"] = Node("file", 0, 0, 0o666, ident=2)
    _, checks, assessment = _flow(tree, "/app", [("/app/rules", "Trusted")])
    assert "write" in checks["/app/rules"]
    assert "/app/rules" in assessment.write_open


@pytest.mark.parametrize(
    "selector",
    [
        "http:\\\\localhost:9005\\_admin",
        "HTTP://LocalHost:9005/_admin",
        "http:/\\localhost:9005/_admin/",
    ],
)
def test_r11_2_a_url_selector_is_a_route_never_a_file(selector):
    """A WHATWG http URL in any spelling is the route it names: unattestable,
    never probed (or certified) as an absent file."""
    tree = Tree({"/": Node("dir", 0, 0, 0o755, ident=0)})
    contract = _contract("/app", [(selector, "Hidden")])
    checks, routes = protection_checks(contract)
    assert routes == ["http://localhost:9005/_admin"]
    assert not any("localhost" in path for path in checks)
    assessment = assess_access(contract, _record(tree.probe_output(list(checks))))
    assert assessment.unattested_routes == ["http://localhost:9005/_admin"]


def test_the_whole_flow_never_certifies_what_the_agent_can_reach():
    rng = random.Random(SEED)
    exact = probed = accepted = 0
    for case in range(CASES):
        tree = random_tree(rng, size=rng.randint(3, 11))
        folders = [p for p, n in tree.nodes.items() if n.kind == "dir" and p != "/"]
        workspace = rng.choice(folders) if folders and rng.random() < 0.7 else "/app"
        selectors = [
            (_selector(rng, tree), rng.choice(_CLASSES))
            for _ in range(rng.randint(1, 3))
        ]
        contract, checks, assessment = _flow(tree, workspace, selectors)
        context = f"case {case}: ws={workspace!r} sel={selectors!r} tree={sorted(tree.nodes.items())}"
        refused = [v for v in contract.violations if v.startswith("selector_refused")]
        non_canonical = [sel for sel, _ in selectors if not _is_canonical(sel)]
        assert bool(refused) == bool(non_canonical), context
        if refused:
            # Refused at load: the contract is invalid and nothing is certified.
            assert assessment.status == "unattested", context
            continue
        if contract.violations:
            continue  # a conflicting or protected-root binding: refused upstream
        accepted += 1
        # Write coverage, from the binding alone (not from the checks list):
        # every write-protected selector inside the workspace or the declared
        # artifacts folder must be write-checked.
        crossing = (workspace, "/logs/artifacts")
        expected_writes = {
            sel
            for sel, cls in selectors
            if cls in _WRITE_CLASSES
            and any(sel == root or sel.startswith(root + "/") for root in crossing)
        }
        write_checked = {path for path, ops in checks.items() if "write" in ops}
        assert expected_writes <= write_checked, f"write check missing, {context}"
        for path, ops in checks.items():
            probed += 1
            exact_kind = tree.final_kind(path) in {"file", "absent", "unresolved"}
            for op in ops:
                write = op == "write"
                reference = (
                    tree.reference_write_open(path)
                    if write
                    else tree.reference_read_open(path)
                )
                opened = path in (
                    assessment.write_open if write else assessment.read_open
                )
                if reference:
                    assert opened, f"certified {op} of {path!r}, {context}"
                if exact_kind:
                    assert opened == reference, f"{op} of {path!r} not exact, {context}"
                    exact += 1
        for selector, cls in selectors:
            if cls in {"Hidden", "VerifierOnly", "ResetState"}:
                assert as_written(selector) in checks, (
                    f"not probed as written, {context}"
                )
    assert accepted > CASES // 3 and probed > CASES and exact > CASES // 3
