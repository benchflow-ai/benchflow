"""Every ingress goes through ``trust.admit`` (TRUSTED-RUNTIME v2 §5).

The rules are in :mod:`benchflow.trust.ingress`. Every backend's transfer
primitive requires the Admission only ``admit()`` makes, checked when it is
called; this test checks that check on every backend and on synthetic call
forms, enumerates the transfer references in the source tree, and fails on
any registry entry that no longer matches the source.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import benchflow
from benchflow.trust import ingress
from benchflow.trust.state import IngressKind

SRC = Path(benchflow.__file__).resolve().parent
ADMIT_NAMES = frozenset({"admit"})
INHERIT_NAMES = frozenset({"restore_from", "begin_restored"})


@dataclass
class Function:
    module: str
    qualname: str
    node: ast.FunctionDef | ast.AsyncFunctionDef
    calls: list[tuple[int, str, ast.Call]] = field(default_factory=list)
    #: Every ``x.<primitive>`` attribute in the function, with its parent.
    references: list[tuple[ast.Attribute, ast.AST | None]] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.module}:{self.qualname}"

    def first_line(self, names: frozenset[str]) -> int | None:
        lines = [line for line, name, _ in self.calls if name in names]
        return min(lines) if lines else None

    def admitted_kinds(self) -> set[str]:
        kinds = set()
        for _, name, call in self.calls:
            if name not in ADMIT_NAMES:
                continue
            for arg in (*call.args, *(k.value for k in call.keywords)):
                if (
                    isinstance(arg, ast.Attribute)
                    and isinstance(arg.value, ast.Attribute | ast.Name)
                    and ast.unparse(arg.value).endswith("IngressKind")
                ):
                    kinds.add(arg.attr)
        return kinds


def _call_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _functions() -> Iterator[Function]:
    for path in sorted(SRC.rglob("*.py")):
        module = _module_name(path)
        if module.startswith("benchflow.trust"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        yield from _walk(module, tree, [])


def _walk(module: str, node: ast.AST, stack: list[str]) -> Iterator[Function]:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            yield from _walk(module, child, [*stack, child.name])
        elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            qualname = ".".join([*stack, child.name])
            function = Function(module, qualname, child)
            for inner, parent in _own_nodes_with_parents(child):
                if isinstance(inner, ast.Call):
                    name = _call_name(inner)
                    if name is not None:
                        function.calls.append((inner.lineno, name, inner))
                if (
                    isinstance(inner, ast.Attribute)
                    and inner.attr in ingress.TRANSFER_PRIMITIVES
                ):
                    function.references.append((inner, parent))
            yield function
            yield from _walk(module, child, [*stack, child.name])
        else:
            yield from _walk(module, child, stack)


def _own_nodes(function: ast.AST) -> Iterator[ast.AST]:
    """Nodes of ``function`` outside nested functions and classes."""
    for node, _ in _own_nodes_with_parents(function):
        yield node


def _own_nodes_with_parents(
    function: ast.AST,
) -> Iterator[tuple[ast.AST, ast.AST | None]]:
    todo: list[tuple[ast.AST, ast.AST | None]] = [
        (child, function) for child in ast.iter_child_nodes(function)
    ]
    while todo:
        node, parent = todo.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        yield node, parent
        todo.extend((child, node) for child in ast.iter_child_nodes(node))


FUNCTIONS = list(_functions())
BY_KEY = {f.key: f for f in FUNCTIONS}


def _is_stub(function: Function) -> bool:
    body = [
        stmt
        for stmt in function.node.body
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))
    ]
    return all(
        isinstance(stmt, ast.Pass)
        or (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))
        for stmt in body
    )


def _passes_admission(call: ast.Call) -> bool:
    return any(
        keyword.arg == "admission"
        and not (
            isinstance(keyword.value, ast.Constant) and keyword.value.value is None
        )
        for keyword in call.keywords
    )


def _is_partial(call: ast.Call) -> bool:
    return isinstance(call.func, ast.Name | ast.Attribute) and (
        ast.unparse(call.func) in ("partial", "functools.partial")
    )


def _reference_problems(function: Function) -> list[str]:
    """Each transfer reference of ``function`` that is not a call passing an
    admission, nor the callable of a partial that passes one."""
    if function.key in ingress.NOT_SANDBOX:
        return []
    if function.node.name in ingress.TRANSFER_PRIMITIVES:
        # A backend's own primitive: the guard admitted the call it is in.
        return []
    problems = []
    for attribute, parent in function.references:
        text = ast.unparse(attribute)
        if isinstance(parent, ast.Call) and parent.func is attribute:
            if _passes_admission(parent):
                continue
            problems.append(f"{function.key} line {attribute.lineno}: {text}(...)")
        elif (
            isinstance(parent, ast.Call)
            and _is_partial(parent)
            and parent.args
            and parent.args[0] is attribute
        ):
            if _passes_admission(parent):
                continue
            problems.append(
                f"{function.key} line {attribute.lineno}: partial({text}, ...)"
            )
        else:
            problems.append(
                f"{function.key} line {attribute.lineno}: {text} used as a value"
            )
    return problems


def test_every_transfer_reference_passes_an_admission():
    problems = [p for function in FUNCTIONS for p in _reference_problems(function)]
    assert not problems, (
        "transfer primitives used without admission=; call trust.admit() (or "
        "admit(..., TRUSTED_TRANSFER, reason=...) for the task's or BenchFlow's "
        "own bytes) and pass what it returns:\n" + "\n".join(problems)
    )


def test_trusted_transfers_state_their_reason():
    missing = []
    for function in FUNCTIONS:
        for line, name, call in function.calls:
            if name not in ADMIT_NAMES:
                continue
            args = [*call.args, *(k.value for k in call.keywords)]
            if not any(ast.unparse(arg).endswith("TRUSTED_TRANSFER") for arg in args):
                continue
            reason = next((k.value for k in call.keywords if k.arg == "reason"), None)
            if not (
                isinstance(reason, ast.Constant)
                and isinstance(reason.value, str)
                and reason.value.strip()
            ):
                missing.append(f"{function.key} line {line}")
    assert not missing, "trusted transfers without a literal reason:\n" + "\n".join(
        missing
    )


def test_self_admitting_primitives_admit_in_their_definitions():
    missing = []
    for function in FUNCTIONS:
        if function.node.name not in ingress.SELF_ADMITTING:
            continue
        if function.key in ingress.SELF_ADMITTING_EXEMPT or _is_stub(function):
            continue
        if function.first_line(ADMIT_NAMES | INHERIT_NAMES) is None:
            missing.append(function.key)
    assert not missing, (
        "self-admitting primitives that neither admit nor inherit lineage:\n"
        + "\n".join(missing)
    )


@pytest.mark.parametrize("key", sorted(ingress.ADMITTING))
def test_admitting_site_admits_its_kind(key):
    function = BY_KEY.get(key)
    assert function is not None, f"registry names a missing function {key}"
    kind = ingress.ADMITTING[key]
    assert kind.name in function.admitted_kinds(), (
        f"{key} must call admit(..., IngressKind.{kind.name})"
    )


@pytest.mark.parametrize("key", sorted(ingress.INHERITING))
def test_inheriting_site_inherits_before_any_other_await(key):
    function = BY_KEY.get(key)
    assert function is not None, f"registry names a missing function {key}"
    inherit_line = function.first_line(INHERIT_NAMES)
    assert inherit_line is not None, f"{key} must call trust restore_from"
    awaits = [
        node.lineno for node in _own_nodes(function.node) if isinstance(node, ast.Await)
    ]
    assert all(line > inherit_line for line in awaits), (
        f"{key} awaits sandbox work before it inherits the snapshot's lineage"
    )


def test_every_ingress_kind_is_admitted_somewhere():
    admitted = {kind for kind in ingress.ADMITTING.values()}
    admitted |= set(ingress.KINDS_ADMITTED_ELSEWHERE)
    # Restores admit SNAPSHOT_RESTORE inside the state module (restore_from).
    admitted.add(IngressKind.SNAPSHOT_RESTORE)
    assert admitted == set(IngressKind)


def test_registry_entries_name_existing_code():
    stale = []
    for key in ingress.NOT_SANDBOX:
        function = BY_KEY.get(key)
        if function is None:
            stale.append(f"NOT_SANDBOX {key}: no such function")
        elif not function.references:
            stale.append(f"NOT_SANDBOX {key}: no transfer reference")
    for key in ingress.SELF_ADMITTING_EXEMPT:
        if key not in BY_KEY:
            stale.append(f"SELF_ADMITTING_EXEMPT {key}: no such function")
    assert not stale, "\n".join(stale)


def _synthetic(source: str) -> Function:
    tree = ast.parse(source)
    return next(iter(_walk("benchflow.example", tree, [])))


@pytest.mark.parametrize(
    "source",
    [
        "async def f(env, src):\n    await env.upload_file(src, '/x')\n",
        "async def f(env, src, flag):\n"
        "    if flag:\n"
        "        admit(env, 'main', IngressKind.SDK_UPLOAD)\n"
        "    await env.upload_file(src, '/x')\n",
        "async def f(env, src):\n"
        "    admit(env, 'main', IngressKind.SDK_UPLOAD)\n"
        "    await functools.partial(env.upload_file, src, '/x')()\n",
        "async def f(env):\n"
        "    admit(env, 'main', IngressKind.AGENT_LAUNCH)\n"
        "    launch = env.live_process\n"
        "    await launch()\n",
        "async def f(env, src):\n    await env.upload_dir(src, '/x', admission=None)\n",
        "async def f(env, src):\n    await run(callback=env.upload_file)\n",
    ],
    ids=["direct", "conditional", "partial", "alias", "none", "callback"],
)
def test_registry_refuses_transfer_forms_without_an_admission(source):
    assert _reference_problems(_synthetic(source))


@pytest.mark.parametrize(
    "source",
    [
        "async def f(env, src):\n"
        "    a = admit(env, 'main', IngressKind.SDK_UPLOAD)\n"
        "    await env.upload_file(src, '/x', admission=a)\n",
        "async def f(env, src):\n"
        "    a = admit(env, 'main', IngressKind.SDK_UPLOAD)\n"
        "    await functools.partial(env.upload_file, src, '/x', admission=a)()\n",
    ],
    ids=["call", "partial"],
)
def test_registry_accepts_transfers_that_pass_an_admission(source):
    assert not _reference_problems(_synthetic(source))
