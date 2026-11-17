"""Every timing test calls tests/_timing.py (the lead's ruling,
2026-11-14): no test file asserts on a delta of a timer of its own
(time.time, time.perf_counter, time.monotonic, time.process_time and their
_ns forms, by any name, the module's too: `import time as t`; and
timeit.default_timer). A datetime's delta and an event loop's time() are
ages and deadlines in these tests, not timings: decides them.

A delta is a subtraction with a timer's reading on either side, or a name
that holds one. The check follows it through assignments, `.append` and
`.extend`, and into the functions that return one, in the same file or
imported from another test file, and fails on an `assert` (or a unittest
`assert*` call) that reads one. tests/_timing.py is the only file that may.
A timer's reading compared without a subtraction (a timestamp) is no
delta.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from tests import _timing

TESTS = Path(__file__).resolve().parent
HELPER = TESTS / "_timing.py"
TIMERS = frozenset(
    {
        "time",
        "time_ns",
        "perf_counter",
        "perf_counter_ns",
        "monotonic",
        "monotonic_ns",
        "process_time",
        "process_time_ns",
        "thread_time",
        "thread_time_ns",
    }
)


def _bare_timers(tree: ast.Module) -> set[str]:
    """The names a module calls a timer by: after `from time import ...` or
    `from timeit import default_timer`, as they are; and each name it gives
    the time module (`import time as t`), as `time.<name>`."""
    names = {"time.time"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names |= {
                alias.asname or alias.name
                for alias in node.names
                if (node.module == "time" and alias.name in TIMERS)
                or (node.module == "timeit" and alias.name == "default_timer")
            }
        elif isinstance(node, ast.Import):
            names |= {
                "time." + (alias.asname or alias.name)
                for alias in node.names
                if alias.name == "time"
            }
    return names


def _owner_name(node: ast.AST) -> str:
    """The last name of what an attribute is read from (`loop` in
    `loop.time`, `datetime` in `datetime.datetime.now`, `get_running_loop`
    in `asyncio.get_running_loop().time`)."""
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    return node.id if isinstance(node, ast.Name) else ""


def _is_timer(node: ast.AST, bare: set[str]) -> bool:
    """A timer's reading: a timer of the time module by any of its names
    (`import time as t`), timeit.default_timer(), or a bare imported timer.
    A datetime's delta and an event loop's time() are not read: in the tests
    they are ages and deadlines, not timings (decides them)."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute):
        owner = _owner_name(func.value)
        return (
            func.attr in TIMERS
            and isinstance(func.value, ast.Name)
            and "time." + func.value.id in bare
        ) or (func.attr == "default_timer" and owner == "timeit")
    return isinstance(func, ast.Name) and func.id in bare


def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


class _Scope:
    """One function's body (or a module's top level), read flow-insensitively."""

    def __init__(self, nodes: list[ast.AST], bare: set[str], timed: set[str]):
        self.nodes, self.bare, self.timed = nodes, bare, timed
        self.clocks: set[str] = set()
        self.held: set[str] = set()
        statements = [n for node in nodes for n in ast.walk(node)]
        for node in statements:
            if (
                isinstance(node, ast.Assign | ast.AnnAssign)
                and node.value is not None
                and any(_is_timer(n, bare) for n in ast.walk(node.value))
            ):
                for target in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                ):
                    self.clocks |= _names(target)
        changed = True
        while changed:
            changed = False
            for node in statements:
                targets: list[ast.AST] = []
                value: ast.AST | None = None
                if isinstance(node, ast.Assign):
                    targets, value = node.targets, node.value
                elif (
                    isinstance(node, ast.AnnAssign | ast.AugAssign)
                    and node.value is not None
                ):
                    targets, value = [node.target], node.value
                elif isinstance(node, ast.For):
                    targets, value = [node.target], node.iter
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("append", "extend", "add", "insert")
                    and isinstance(node.func.value, ast.Name)
                ):
                    if node.func.value.id not in self.held and any(
                        self.delta(arg) for arg in node.args
                    ):
                        self.held.add(node.func.value.id)
                        changed = True
                    continue
                if value is None or not self.delta(value):
                    continue
                for target in targets:
                    new = _names(target) - self.held
                    if new:
                        self.held |= new
                        changed = True

    def delta(self, node: ast.AST) -> bool:
        """Whether *node* reads a timer's delta."""
        for n in ast.walk(node):
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Sub):
                for side in (n.left, n.right):
                    if any(_is_timer(m, self.bare) for m in ast.walk(side)) or (
                        _names(side) & self.clocks
                    ):
                        return True
            if isinstance(n, ast.Name) and n.id in self.held:
                return True
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id in self.timed
            ):
                return True
        return False

    def returns_delta(self) -> bool:
        return any(
            isinstance(n, ast.Return | ast.Yield)
            and n.value is not None
            and self.delta(n.value)
            for node in self.nodes
            for n in ast.walk(node)
        )

    def offenses(self) -> list[int]:
        found = []
        for node in self.nodes:
            for n in ast.walk(node):
                if (isinstance(n, ast.Assert) and self.delta(n.test)) or (
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr.startswith("assert")
                    and any(self.delta(arg) for arg in n.args)
                ):
                    found.append(n.lineno)
        return found


def _scopes(tree: ast.Module) -> list[tuple[str | None, list[ast.AST]]]:
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    top = [
        node
        for node in tree.body
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    ]
    return [(f.name, list(f.body)) for f in functions] + [(None, top)]


def offenses(sources: dict[str, str]) -> list[tuple[str, int]]:
    """The `module:line` of each assert on a timer's delta in *sources*
    (module name, as `tests.x`, to its text); the helper is not read."""
    trees = {name: ast.parse(text) for name, text in sources.items()}
    timed: dict[str, set[str]] = {name: set() for name in trees}
    changed = True
    while changed:
        changed = False
        for name, tree in trees.items():
            local = set(timed[name])
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module in timed:
                    local |= {
                        alias.asname or alias.name
                        for alias in node.names
                        if alias.name in timed[node.module]
                    }
            bare = _bare_timers(tree)
            for function, body in _scopes(tree):
                if (
                    function is not None
                    and function not in timed[name]
                    and _Scope(body, bare, local).returns_delta()
                ):
                    timed[name].add(function)
                    changed = True
    found = []
    for name, tree in trees.items():
        local = set(timed[name])
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in timed:
                local |= {
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name in timed[node.module]
                }
        bare = _bare_timers(tree)
        for _function, body in _scopes(tree):
            found += [(name, line) for line in _Scope(body, bare, local).offenses()]
    return sorted(set(found))


# A file that names no timer and imports nothing from a test file that
# does can hold no delta: only the others are parsed.
_TIMER_TEXT = re.compile(
    r"\btime\.(?:" + "|".join(sorted(TIMERS)) + r")\b|\bfrom time import\b"
    r"|\bimport time as\b|\btimeit\b"
)


def _test_sources() -> dict[str, str]:
    texts = {}
    for path in sorted(TESTS.rglob("*.py")):
        if path == HELPER or "__pycache__" in path.parts:
            continue
        module = ".".join(("tests", *path.relative_to(TESTS).with_suffix("").parts))
        texts[module] = path.read_text(encoding="utf-8")
    timed = {module for module, text in texts.items() if _TIMER_TEXT.search(text)}
    importing = {
        module
        for module, text in texts.items()
        if any(f"from {other} import" in text for other in timed)
    }
    return {module: texts[module] for module in sorted(timed | importing)}


# Files outside the ruling that already time with a timer of their own
# (timeouts, waits, a ReDoS guard), at their count on 2026-11-14: none may
# gain one, and no other file may have any.
ELSEWHERE = {
    "tests.e2e.harness": 1,
    "tests.environment.test_sweep_readiness_b22r1": 1,
    "tests.test_agent_muse": 2,
    # PR #198 (36b45b3ef, 2026-11-14): two deadline bounds on the wait for the
    # agent's last OTel export. converts or justifies the counts here.
    "tests.test_agent_otel": 2,
    "tests.test_cli_termination": 2,
    "tests.test_credential_names": 1,
    "tests.test_daytona_context_lifecycle": 2,
    "tests.test_daytona_context_scheduling": 2,
    "tests.test_daytona_start_r2": 2,
    "tests.test_daytona_start_r3": 2,
    "tests.test_daytona_start_r4": 2,
    "tests.test_dfx_scale_r3": 12,
    "tests.test_dx_a_container": 34,
    "tests.test_egress_relay_idle": 4,
    # PR #308 (2026-11-16): two deadline bounds: a readiness command
    # past its timeout ends with its group, one in time returns at once.
    "tests.test_en698_policy_first_instruction": 2,
    "tests.test_exec_channel": 4,
    "tests.test_exec_diet_units": 1,
    "tests.test_exec_transport_contract": 1,
    "tests.test_freeze_bfx_chown": 1,
    "tests.test_hub_push_redaction": 1,
    "tests.test_job_budget": 1,
    # PR #147 (5ec1c7d04, 2026-11-14): two bounds on how long a forked child may
    # hold the job lock, a deadline. converts or justifies the counts here.
    "tests.test_job_lock_forked_child": 2,
    "tests.test_job_matrix": 1,
    "tests.test_layer_lock": 1,
    "tests.test_metadata_helper": 1,
    "tests.test_python_sdk_load_job": 2,
    "tests.test_sighup_container": 9,
    "tests.test_source_cache_lock": 2,
    "tests.test_streaming_diagnostics": 1,
    "tests.test_teardown_queue": 2,
    # 2 since PR #193 (d85f2f088, 2026-11-14): its retry test bounds a wait,
    # a timeout bound. converts or justifies the counts in this table.
    "tests.test_verifier_start_receipts": 2,
    "tests.trajectories.test_redaction": 1,
}


def test_no_test_file_asserts_on_a_timer_of_its_own():
    """The lead's ruling: every timing test calls tests/_timing.py; the
    files listed above keep no more than they had."""
    counts: dict[str, int] = {}
    for module, _line in offenses(_test_sources()):
        counts[module] = counts.get(module, 0) + 1
    over = {
        module: count
        for module, count in counts.items()
        if count > ELSEWHERE.get(module, 0)
    }
    assert over == {}, (
        f"raw timer assertions over the allowed count: {over}. To time how a"
        " call grows with its input, use tests/_timing.py (linear_seconds,"
        " assert_linear). For a timeout or a deadline bound, raise that"
        " file's count in ELSEWHERE with a comment giving the reason and the"
        " PR."
    )


PLANTED = {
    "direct": (
        "import time\n"
        "def test_x():\n"
        "    start = time.perf_counter()\n"
        "    work()\n"
        "    assert time.perf_counter() - start < 1.0\n"
    ),
    "through a name": (
        "import time\n"
        "def test_x():\n"
        "    t0 = time.time()\n"
        "    work()\n"
        "    took = time.time() - t0\n"
        "    assert took < 2.0, took\n"
    ),
    "through a helper and a list": (
        "import time\n"
        "def _best(call, text):\n"
        "    start = time.monotonic()\n"
        "    out = call(text)\n"
        "    return time.monotonic() - start, out\n"
        "def test_x():\n"
        "    seconds = []\n"
        "    for size in (1, 4):\n"
        "        took, _ = _best(work, 'a' * size)\n"
        "        seconds.append(took)\n"
        "    assert seconds[1] < 2.0, seconds\n"
    ),
    "the module by another name": (
        "import time as t\n"
        "def test_x():\n"
        "    begin = t.perf_counter()\n"
        "    work()\n"
        "    assert t.perf_counter() - begin < 1.0\n"
    ),
    "timeit's timer": (
        "import timeit\n"
        "def test_x():\n"
        "    begin = timeit.default_timer()\n"
        "    work()\n"
        "    assert timeit.default_timer() - begin < 1.0\n"
    ),
    "a bare name": (
        "from time import process_time as clock\n"
        "def test_x():\n"
        "    begin = clock()\n"
        "    work()\n"
        "    elapsed = clock() - begin\n"
        "    self.assertLess(elapsed, 1.0)\n"
    ),
}


def test_a_planted_raw_timing_assertion_is_found():
    """Each planted shape is found, in its own file and through a helper
    imported from another test file; the helper's own calls and a timestamp
    compared without a subtraction are not."""
    for shape, text in PLANTED.items():
        assert offenses({"tests.planted": text}) != [], shape
    imported = {
        "tests.timer": PLANTED["through a helper and a list"].split("def test_x")[0],
        "tests.user": (
            "from tests.timer import _best\n"
            "def test_y():\n"
            "    took, out = _best(work, 'a')\n"
            "    assert took < 2.0\n"
        ),
    }
    assert offenses(imported) == [("tests.user", 4)]
    clean = (
        "import time\n"
        "from tests._timing import assert_linear, linear_seconds\n"
        "def test_z():\n"
        "    seconds, outs = linear_seconds(work, lambda n: 'a' * n)\n"
        "    assert all(outs)\n"
        "    assert_linear(seconds)\n"
        "    assert record['ts'] <= time.time()\n"
    )
    assert offenses({"tests.clean": clean}) == []


def test_the_helper_is_where_the_timer_is():
    """The helper itself times with CPU time and is the one file not read."""
    text = HELPER.read_text(encoding="utf-8")
    assert "time.process_time()" in text
    assert offenses({"tests._timing": text}) != []
    assert _timing.RATIO == 5 and _timing.ROUNDS == 7 and _timing.CEILING == 10.0
