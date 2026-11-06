"""Every controller read, copy and write at a reachable folder uses the helper.

A rollout's ``agent/``, ``verifier/`` and ``artifacts/`` folders are what the
sandbox can reach (bind-mounted on Docker, filled from the sandbox elsewhere).
This scan follows paths built from those folders through assignments, ``self``
attributes and calls inside ``benchflow`` and fails on any write, rename, copy,
download, read open or temporary file touching one of them without going through
:mod:`benchflow._utils.reachable_write` (CONFIRM-taskmd-rewards-R4-1).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import benchflow

SRC = Path(benchflow.__file__).parent
HELPER = SRC / "_utils" / "reachable_write.py"

# Attributes (RolloutPaths and friends) and parameter names that name a
# reachable folder or a file in one.
REACHABLE_NAMES = {
    "agent_dir",
    "verifier_dir",
    "artifacts_dir",
    "reward_text_path",
    "reward_json_path",
    "reward_details_json_path",
    "criteria_runner_manifest_path",
    "test_stdout_path",
    "test_stderr_path",
}
# ``<rollout> / "agent"`` and the like.
REACHABLE_LITERALS = {"agent", "verifier", "artifacts"}
ROLLOUT_ROOT_HINTS = ("rollout_dir", "trial_dir", "attempt")
# In-sandbox path constants: strings for exec, never host paths.
SANDBOX_SIDE = {"SandboxPaths", "sandbox_paths", "EnvironmentPaths", "remote"}
# Attributes whose result is no longer under the reachable folder.
LEAVES = {"parent", "parents", "name", "stem", "suffix", "anchor"}
PATH_METHODS = {
    "with_name",
    "with_suffix",
    "with_stem",
    "joinpath",
    "absolute",
    "expanduser",
    "resolve",
}

# Writers that are not the helper but keep the invariant another way, each
# with its reason. A stale entry fails the test.
ALLOWED: set[tuple[str, str, str]] = set()


@dataclass
class Func:
    module: str
    qualname: str
    node: ast.FunctionDef | ast.AsyncFunctionDef
    cls: str | None
    params: list[str] = field(default_factory=list)


@dataclass
class Module:
    rel: str
    tree: ast.Module
    funcs: dict[str, Func] = field(default_factory=dict)
    imports: dict[str, tuple[str, str]] = field(default_factory=dict)
    self_attrs: set[str] = field(default_factory=set)


def _base_name(node: ast.AST) -> str | None:
    while isinstance(node, ast.Attribute | ast.Subscript | ast.Call):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def _mentions_rollout_root(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in ROLLOUT_ROOT_HINTS:
            return True
        if isinstance(sub, ast.Attribute) and sub.attr in ROLLOUT_ROOT_HINTS:
            return True
    return False


class Taint:
    def __init__(self, module: Module, names: set[str], params: set[str] = frozenset()):
        self.module = module
        self.names = names
        # A parameter called agent_dir (and the like) is the rollout's folder;
        # a local of that name is only tainted by what it was built from.
        self.params = params

    def __call__(self, node: ast.AST | None) -> bool:
        if node is None:
            return False
        if isinstance(node, ast.Name):
            return node.id in self.names or (
                node.id in REACHABLE_NAMES and node.id in self.params
            )
        if isinstance(node, ast.Attribute):
            if node.attr in LEAVES:
                return False
            # Task export's immutable authoring view is task-owned source,
            # never a RolloutPaths mount (docs/sandboxes/threat-model.md).
            if (self.module.rel, _dotted(node)) == (
                "task/export.py",
                "view.verifier_dir",
            ):
                return False
            if node.attr in REACHABLE_NAMES:
                return _base_name(node.value) not in SANDBOX_SIDE
            return (
                isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and node.attr in self.module.self_attrs
            )
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if self(node.left):
                return True
            right = node.right
            return (
                isinstance(right, ast.Constant)
                and isinstance(right.value, str)
                and right.value.split("/")[0] in REACHABLE_LITERALS
                and _mentions_rollout_root(node.left)
            )
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "_mounted_host_path":
                return True
            if isinstance(func, ast.Attribute) and func.attr in PATH_METHODS:
                return self(func.value)
            if isinstance(func, ast.Name) and func.id in {"Path", "PurePath", "str"}:
                return bool(node.args) and self(node.args[0])
            return False
        if isinstance(node, ast.IfExp):
            return self(node.body) or self(node.orelse)
        if isinstance(node, ast.BoolOp):
            return any(self(v) for v in node.values)
        if isinstance(node, ast.NamedExpr):
            return self(node.value)
        return False

    def anywhere(self, node: ast.AST) -> bool:
        return any(self(sub) for sub in ast.walk(node))


def _targets(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.Tuple | ast.List):
        out: list[ast.AST] = []
        for elt in node.elts:
            out.extend(_targets(elt))
        return out
    if isinstance(node, ast.Starred):
        return _targets(node.value)
    return [node]


def _kw(call: ast.Call, *names: str) -> ast.AST | None:
    for kw in call.keywords:
        if kw.arg in names:
            return kw.value
    return None


def _arg(call: ast.Call, index: int, *names: str) -> ast.AST | None:
    if len(call.args) > index:
        return call.args[index]
    return _kw(call, *names)


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    return ""


def _sinks(call: ast.Call, taint: Taint) -> list[str]:
    """I/O that bypasses the helper on a tainted source or destination."""
    func = call.func
    name = _dotted(func)
    first, _dot, rest = name.partition(".")
    if first in taint.module.imports:
        base, imported = taint.module.imports[first]
        name = ".".join(p for p in (base, imported, rest) if p)
    found: list[str] = []
    if isinstance(func, ast.Attribute):
        receiver = func.value
        attr = func.attr
        if attr in {"write_text", "write_bytes", "touch", "symlink_to", "hardlink_to"}:
            if taint(receiver):
                found.append(f".{attr}")
        elif attr == "open" and name != "os.open":
            if taint(receiver):
                found.append(".open")
        elif attr in {"rename", "replace"} and name not in {"os.rename", "os.replace"}:
            if taint(_arg(call, 0, "file", "path")):
                found.append(f".{attr}")
        elif attr in {"download_file", "download_dir"}:
            target = _arg(call, 1, "target_path", "target_dir", "target")
            if taint(target):
                found.append(f".{attr}")
        elif attr in {"extractall", "extract"} and taint(_arg(call, 0, "path")):
            found.append(f".{attr}")
    if name in {"os.replace", "os.rename"}:
        if taint(_arg(call, 1, "dst")) and _kw(call, "dst_dir_fd") is None:
            found.append(name)
    elif name in {"os.symlink", "os.link"}:
        if taint(_arg(call, 1, "dst")):
            found.append(name)
    elif name == "os.open":
        if taint(_arg(call, 0, "file", "path")):
            found.append(name)
    elif name in {"open", "io.open", "builtins.open"}:
        if taint(_arg(call, 0, "file", "path")):
            found.append("open")
    elif name in {
        "shutil.copy",
        "shutil.copy2",
        "shutil.copyfile",
        "shutil.copytree",
        "shutil.copyfileobj",
        "shutil.move",
    }:
        if taint(_arg(call, 0, "src", "fsrc")) or taint(_arg(call, 1, "dst", "fdst")):
            found.append(name)
    elif name == "os.walk" and taint(_arg(call, 0, "top")):
        follow = _arg(call, 3, "followlinks")
        if follow is not None and not (
            isinstance(follow, ast.Constant) and follow.value is False
        ):
            found.append(name)
    elif name in {
        "tempfile.mkstemp",
        "tempfile.mkdtemp",
        "tempfile.NamedTemporaryFile",
        "tempfile.TemporaryDirectory",
    } and taint(_kw(call, "dir")):
        found.append(name)
    return found


def _load() -> dict[str, Module]:
    modules: dict[str, Module] = {}
    for path in sorted(SRC.rglob("*.py")):
        if path == HELPER or "_vendor" in path.parts:
            continue
        rel = path.relative_to(SRC).as_posix()
        module = Module(rel, ast.parse(path.read_text(), filename=str(path)))
        dotted = "benchflow." + rel.removesuffix(".py").replace("/", ".")
        dotted = dotted.removesuffix(".__init__")
        for node in ast.walk(module.tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                base = node.module
                if node.level:
                    parent = dotted.split(".")
                    if not rel.endswith("__init__.py"):
                        parent = parent[:-1]
                    parent = parent[: len(parent) - node.level + 1]
                    base = ".".join([*parent, node.module])
                for alias in node.names:
                    module.imports[alias.asname or alias.name] = (base, alias.name)

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    module.imports[alias.asname or alias.name] = (alias.name, "")

        def visit(body, prefix: str, cls: str | None, module=module, rel=rel) -> None:
            for node in body:
                if isinstance(node, ast.ClassDef):
                    visit(node.body, f"{prefix}{node.name}.", node.name, module, rel)
                elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    params = [a.arg for a in node.args.posonlyargs + node.args.args]
                    params += [a.arg for a in node.args.kwonlyargs]
                    qual = f"{prefix}{node.name}"
                    module.funcs[qual] = Func(rel, qual, node, cls, params)
                    visit(node.body, f"{qual}.", cls, module, rel)

        visit(module.tree.body, "", None)
        modules[dotted] = module
    return modules


def _resolve(
    call: ast.Call, module: Module, func: Func, modules: dict[str, Module]
) -> tuple[Module, Func] | None:
    target = call.func
    if isinstance(target, ast.Name):
        if target.id in module.funcs:
            return module, module.funcs[target.id]
        constructor = f"{target.id}.__init__"
        if constructor in module.funcs:
            return module, module.funcs[constructor]
        if target.id in module.imports:
            base, name = module.imports[target.id]
            other = modules.get(base)
            if other is not None and name in other.funcs:
                return other, other.funcs[name]
            if other is not None and f"{name}.__init__" in other.funcs:
                return other, other.funcs[f"{name}.__init__"]
        return None
    if (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
        and func.cls is not None
    ):
        qual = f"{func.cls}.{target.attr}"
        for key, candidate in module.funcs.items():
            if key == qual or key.endswith(f".{qual}"):
                return module, candidate
    return None


def _scan() -> tuple[list[tuple[str, str, str, int]], set[tuple[str, str, str]]]:
    modules = _load()
    # Seeds: every function, with its parameters tainted by name only.
    tainted_params: dict[tuple[str, str], set[str]] = {}
    findings: dict[tuple[str, str, str, int], None] = {}
    changed = True
    rounds = 0
    while changed and rounds < 20:
        changed = False
        rounds += 1
        for module in modules.values():
            for func in module.funcs.values():
                key = (module.rel, func.qualname)
                names = set(tainted_params.get(key, set()))
                if (module.rel, func.qualname) == (
                    "learner_skills.py",
                    "capture_skills",
                ):
                    names.add("export_dir")
                taint = Taint(module, names, set(func.params))
                # Local propagation to a fixpoint.
                for _ in range(10):
                    before = (len(names), len(module.self_attrs))
                    for node in ast.walk(func.node):
                        value = None
                        targets: list[ast.AST] = []
                        if isinstance(node, ast.Assign):
                            value, targets = node.value, node.targets
                        elif isinstance(node, ast.AnnAssign) and node.value:
                            value, targets = node.value, [node.target]
                        elif isinstance(node, ast.For | ast.AsyncFor):
                            if taint.anywhere(node.iter):
                                for t in _targets(node.target):
                                    if isinstance(t, ast.Name):
                                        names.add(t.id)
                            continue
                        elif isinstance(node, ast.withitem) and node.optional_vars:
                            value, targets = node.context_expr, [node.optional_vars]
                        elif isinstance(node, ast.NamedExpr):
                            value, targets = node.value, [node.target]
                        if value is None:
                            continue
                        hit = taint(value) or (
                            isinstance(value, ast.Tuple | ast.List | ast.Dict)
                            and taint.anywhere(value)
                        )
                        if not hit:
                            continue
                        for target in targets:
                            for t in _targets(target):
                                if isinstance(t, ast.Name):
                                    names.add(t.id)
                                elif (
                                    isinstance(t, ast.Attribute)
                                    and isinstance(t.value, ast.Name)
                                    and t.value.id == "self"
                                    and t.attr not in module.self_attrs
                                ):
                                    module.self_attrs.add(t.attr)
                                    changed = True
                    if (len(names), len(module.self_attrs)) == before:
                        break
                for node in ast.walk(func.node):
                    if not isinstance(node, ast.Call):
                        continue
                    for kind in _sinks(node, taint):
                        findings.setdefault(
                            (module.rel, func.qualname, kind, node.lineno), None
                        )
                    resolved = _resolve(node, module, func, modules)
                    if resolved is None:
                        continue
                    callee_module, callee = resolved
                    if callee_module.rel == HELPER.relative_to(SRC).as_posix():
                        continue
                    params = callee.params
                    offset = 1 if params[:1] in (["self"], ["cls"]) else 0
                    hits: set[str] = set()
                    for index, arg in enumerate(node.args):
                        if index + offset < len(params) and taint(arg):
                            hits.add(params[index + offset])
                    for kw in node.keywords:
                        if kw.arg in params and taint(kw.value):
                            hits.add(kw.arg)
                    callee_key = (callee_module.rel, callee.qualname)
                    known = tainted_params.setdefault(callee_key, set())
                    if not hits <= known:
                        known |= hits
                        changed = True
    sites = sorted(findings)
    return sites, {(rel, qual, kind) for rel, qual, kind, _ in sites}


def test_every_reachable_write_goes_through_the_helper():
    sites, _ = _scan()
    bypass = [site for site in sites if site[:3] not in ALLOWED]
    assert not bypass, "I/O at a sandbox-reachable rollout folder bypasses " + (
        "benchflow._utils.reachable_write:\n"
        + "\n".join(f"  {rel}:{line} {qual} {kind}" for rel, qual, kind, line in bypass)
    )


def test_every_allowed_writer_still_exists():
    _, kinds = _scan()
    assert kinds >= ALLOWED, f"stale ALLOWED entries: {sorted(ALLOWED - kinds)}"


def test_the_scan_sees_the_writers_it_guards(tmp_path):
    """The scan flags the shapes the native judge used before the fix."""
    source = """
import json, os, shutil
class V:
    def judge(self):
        self._rollout_paths.reward_json_path.write_text("{}")
        path = self._rollout_paths.reward_details_json_path
        path.write_text("{}")
        tmp = path.with_name("x.tmp")
        os.replace(tmp, path)
        helper(self._rollout_paths.verifier_dir)
        with open(self._rollout_paths.verifier_dir / "m", "w") as f:
            f.write("x")
def helper(folder):
    (folder / "marker").write_text("x")
def outside(rollout_dir):
    (rollout_dir / "agent" / "log.txt").write_text("x")
    (rollout_dir / "result.json").write_text("x")
"""
    module = Module("probe.py", ast.parse(source))
    for node in ast.walk(module.tree):
        if isinstance(node, ast.FunctionDef):
            module.funcs[node.name] = Func("probe.py", node.name, node, None)
    tree_funcs = {f.node.name: f for f in module.funcs.values()}
    judge = tree_funcs["judge"]
    taint = Taint(module, set())
    names: set[str] = set()
    taint = Taint(module, names)
    for _ in range(3):
        for node in ast.walk(judge.node):
            if isinstance(node, ast.Assign) and taint(node.value):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        names.add(t.id)
    kinds = [
        kind
        for node in ast.walk(judge.node)
        if isinstance(node, ast.Call)
        for kind in _sinks(node, taint)
    ]
    assert kinds.count(".write_text") == 2
    assert "os.replace" in kinds
    assert "open" in kinds
    outside = tree_funcs["outside"]
    taint = Taint(module, set())
    kinds = [
        kind
        for node in ast.walk(outside.node)
        if isinstance(node, ast.Call)
        for kind in _sinks(node, taint)
    ]
    assert kinds == [".write_text"]


def test_archive_extractors_share_component_preflight():
    """Guards R7-1 at a2052fa6: all three extraction paths use one inventory validator."""
    expected = {
        ("review/evidence.py", "_admit_members"),
        ("taskmd/verify.py", "unpack_kept"),
        ("sandbox/_bounded_tar.py", "extract_download"),
        ("hub/datasets.py", "unpack_task"),
    }
    found = set()
    for module in _load().values():
        for function in module.funcs.values():
            calls = [
                node for node in ast.walk(function.node) if isinstance(node, ast.Call)
            ]
            if any(_dotted(call.func) == "ArchivePaths" for call in calls):
                assert any(_dotted(call.func) == "inventory.add" for call in calls)
                found.add((module.rel, function.qualname))
            for call in calls:
                if isinstance(call.func, ast.Attribute) and call.func.attr in {
                    "extract",
                    "extractall",
                }:
                    assert (module.rel, function.qualname) in expected
    assert expected <= found


def test_backend_downloads_and_exports_use_shared_boundary():
    """Guards R7-2 at a2052fa6: new raw backend download or export paths fail this scan."""
    backend_methods = {
        "sandbox/docker.py": {"download_file", "download_dir"},
        "sandbox/apple_container.py": {"download_file", "download_dir"},
        "sandbox/modal_impl.py": {"download_file", "download_dir"},
        "sandbox/agentcore.py": {"download_file", "download_dir"},
        "sandbox/daytona.py": {
            "_sdk_download_file",
            "_sdk_download_dir",
            "_sdk_download_files",
        },
    }
    for rel, methods in backend_methods.items():
        tree = ast.parse((SRC / rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name in methods:
                assert any(
                    isinstance(decorator, ast.Call)
                    and _dotted(decorator.func) == "private_download"
                    for decorator in node.decorator_list
                ), (rel, node.name)
    # Remote Docker inherits the protected endpoints; both Daytona strategies
    # forward to its protected SDK download endpoints (including fallback).
    for rel in (
        "sandbox/remote_docker.py",
        "sandbox/daytona_dind.py",
        "sandbox/daytona_strategies.py",
        "rollout/_user_loop.py",
        "learner_skills.py",
    ):
        module = next(m for m in _load().values() if m.rel == rel)
        for function in module.funcs.values():
            if (
                "download" not in function.qualname
                and "export" not in function.qualname
                and "capture_skills" not in function.qualname
            ):
                continue
            for node in ast.walk(function.node):
                if isinstance(node, ast.Call):
                    assert _dotted(node.func) not in {
                        "shutil.copytree",
                        "shutil.copy",
                        "shutil.copy2",
                        "shutil.copyfile",
                    }
