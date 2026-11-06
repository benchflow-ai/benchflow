"""ACC-RUN2 units: when the protected runner is built, and the hook clean-up's
root-made rule on the bfx route. The real Docker trials are in
``test_acc_run2_docker.py``."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from benchflow.sandbox import _hook_cleanup
from benchflow.sandbox._verifier_runtime import (
    Interpreter,
    Resolution,
    build_tree,
    runner_requested,
)
from tests.test_hook_cleanup_bfx import INPUTS, _FakeFs

TB2_TEST = """\
#!/bin/bash
uvx -p 3.13 -w pytest==8.4.1 -w pytest-json-ctrf==0.3.5 \\
  pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA
"""


def _task(tmp_path, files, env=None, native=False):
    folder = tmp_path / ("verifier" if native else "tests")
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (folder / name).write_text(text)
    return SimpleNamespace(
        task_dir=tmp_path,
        paths=SimpleNamespace(uses_native_verifier_dir=native),
        config=SimpleNamespace(verifier=SimpleNamespace(env=env or {})),
    )


def test_a_tb2_test_script_does_not_ask_for_the_runner(tmp_path):
    assert not runner_requested(_task(tmp_path, {"test.sh": TB2_TEST}))


@pytest.mark.parametrize(
    "line",
    [
        '"$BENCHFLOW_PYTEST" --ctrf /logs/verifier/ctrf.json /tests',
        "${BENCHFLOW_PYTEST:-pytest} /tests",
        "PYTEST=$BENCHFLOW_PYTEST",
    ],
)
def test_naming_the_runner_asks_for_it(tmp_path, line):
    assert runner_requested(_task(tmp_path, {"test.sh": "#!/bin/bash\n" + line + "\n"}))


def test_a_helper_beside_test_sh_or_the_verifier_env_can_ask(tmp_path):
    assert runner_requested(
        _task(
            tmp_path, {"test.sh": TB2_TEST, "run.sh": 'exec "$BENCHFLOW_PYTEST" "$@"\n'}
        )
    )
    assert runner_requested(
        _task(
            tmp_path / "e", {"test.sh": TB2_TEST}, env={"PYTEST": "$BENCHFLOW_PYTEST"}
        )
    )
    assert runner_requested(
        _task(tmp_path / "n", {"test.sh": '"$BENCHFLOW_PYTEST"\n'}, native=True)
    )


def test_other_benchflow_pytest_names_do_not_ask(tmp_path):
    text = (
        'PYTEST_BIN="${BENCHFLOW_PYTEST_BIN:-pytest}"\necho $BENCHFLOW_PYTEST_MARKERS\n'
    )
    assert not runner_requested(_task(tmp_path, {"test.sh": text}))


def test_what_cannot_be_read_keeps_the_runner(tmp_path):
    assert runner_requested(SimpleNamespace(task_dir=None, config=None))
    assert runner_requested(SimpleNamespace(task_dir=tmp_path / "missing", config=None))


def _interpreter():
    return Interpreter(
        exe="/usr/local/bin/python3.12",
        version=(3, 12),
        machine="x86_64",
        glibc=(2, 36),
        musl=None,
        markers={},
        stdlib=["/opt/benchflow/rt/lib/python3.12"],
        image_sites=["/usr/local/lib/python3.12/site-packages"],
    )


@pytest.mark.parametrize("wanted", [True, False])
def test_without_the_runner_the_tree_holds_no_image_sites(tmp_path, wanted):
    resolution = Resolution(wheels=[], plugins={}, plugin_top=[], runner=True)
    manifest = build_tree(
        resolution,
        interpreter=_interpreter(),
        tests_dir="/tests",
        into=tmp_path / "tree",
        runner_wanted=wanted,
    )
    assert (manifest["runner"] is not None) is wanted
    assert bool(manifest["image_sites"]) is wanted
    assert (tmp_path / "tree" / "run.py").exists() is wanted
    assert (tmp_path / "tree" / "site").is_dir()


# -- the bfx clean-up's root-made rule -----------------------------------------

SITE = b"/root/.local/lib/python3.12/site-packages"


class _OwnedFs(_FakeFs):
    """_FakeFs with owners, modes and link counts: root's, closed and one
    name unless overridden (a link count of None: bfx gave none)."""

    def __init__(self, tree, owners=None, modes=None, links=None):
        super().__init__(tree)
        self.owners = owners or {}
        self.modes = modes or {}
        self.links = links or {}

    def facts(self, path, shown=None):
        row = super().facts(path, shown)
        row["uid"] = self.owners.get(path, 0)
        if path in self.modes:
            row["mode"] = self.modes[path]
        if self.links.get(path, 1) is not None:
            row["nlink"] = self.links.get(path, 1)
        return row


def _clean(fs, *, agent_user=True, workspace="/app"):
    loop = asyncio.get_event_loop()
    return _hook_cleanup.run(
        fs.call,
        _hook_cleanup.parse_inputs(INPUTS.replace("a:b", "")),
        cleanup_conftests=False,
        workspace=workspace,
        workspace_aliases=(),
        verifier_dir="/opt/benchflow/verifier",
        deadline=loop.time() + 30,
        agent_user=agent_user,
    )


@pytest.mark.asyncio
async def test_a_new_hook_only_root_could_write_stays_with_a_sandbox_user():
    fs = _OwnedFs({b"/app": "d", SITE + b"/cmeel.pth": b"import cmeel_pth\n"})
    report = await _clean(fs)
    assert fs.tree[SITE + b"/cmeel.pth"] == b"import cmeel_pth\n"
    assert report.root_made == [(SITE + b"/cmeel.pth").decode()]
    # Written again, fresh: a new inode at the name.
    assert any(c == "write" for c, _ in fs.calls)


@pytest.mark.asyncio
async def test_without_a_sandbox_user_a_new_hook_goes():
    fs = _OwnedFs({b"/app": "d", SITE + b"/cmeel.pth": b"import cmeel_pth\n"})
    report = await _clean(fs, agent_user=False)
    assert SITE + b"/cmeel.pth" not in fs.tree
    assert report.root_made == []


@pytest.mark.parametrize(
    "owners,modes,links",
    [
        ({SITE + b"/cmeel.pth": 1000}, {}, {}),  # the agent user's file
        ({}, {SITE + b"/cmeel.pth": 0o666}, {}),  # others may write it
        ({}, {SITE: 0o777}, {}),  # others may write the folder
        ({}, {b"/root/.local": 0o775}, {}),  # or a folder above it
        ({b"/root": 1000}, {}, {}),  # a folder above it is another user's
        # A second name, maybe one in the workspace the agent user could reach
        # (review A, L1); a count bfx did not give counts as more.
        ({}, {}, {SITE + b"/cmeel.pth": 2}),
        ({}, {}, {SITE + b"/cmeel.pth": None}),
    ],
)
@pytest.mark.asyncio
async def test_a_new_hook_the_agent_user_could_have_written_goes(owners, modes, links):
    fs = _OwnedFs(
        {b"/app": "d", SITE + b"/cmeel.pth": b"import cmeel_pth\n"},
        owners,
        modes,
        links,
    )
    await _clean(fs)
    assert SITE + b"/cmeel.pth" not in fs.tree


@pytest.mark.asyncio
async def test_a_hook_package_with_an_entry_the_agent_owns_goes():
    pkg = SITE + b"/sitecustomize"
    fs = _OwnedFs(
        {b"/app": "d", pkg + b"/__init__.py": b"x = 1\n", pkg + b"/m.py": b"y\n"},
        owners={pkg + b"/m.py": 1000},
    )
    await _clean(fs)
    assert not any(p.startswith(pkg) for p in fs.tree)


@pytest.mark.asyncio
async def test_a_root_owned_hook_in_the_workspace_goes():
    """The freeze chowns the workspace to root before the clean-up: owner
    says nothing there."""
    site = b"/app/.venv/lib/python3.12/site-packages"
    fs = _OwnedFs({site + b"/evil.pth": b"import os\n"})
    report = await _hook_cleanup.root_made(
        _hook_cleanup._Facts(fs.call, asyncio.get_event_loop().time() + 30),
        site.decode(),
        "evil.pth",
        ["/app", *_hook_cleanup.ROOT_MADE_EXCLUDED],
    )
    assert report is False


# -- who the two rules are on for ----------------------------------------------


@pytest.mark.parametrize(
    "user,agent_paths,expect",
    [
        ("agent", None, (True, "/app")),
        ("root", None, (False, "/app")),
        (None, None, (False, "/app")),
        # A separate verifier sandbox: the agent's files came by BenchFlow's
        # transfer, as root, so root's files prove nothing there.
        ("agent", ("/app",), (False, None)),
    ],
)
@pytest.mark.asyncio
async def test_the_rules_hold_only_in_the_agents_own_sandbox(
    monkeypatch, user, agent_paths, expect
):
    from benchflow.sandbox import lockdown

    seen = []

    async def steps(*args, **kwargs):
        seen.append((lockdown._hook_agent_user.get(), lockdown._path_workspace.get()))

    monkeypatch.setattr(lockdown, "_harden_steps", steps)
    await lockdown.harden_before_verify(
        SimpleNamespace(), SimpleNamespace(), user, "/app", agent_paths=agent_paths
    )
    assert seen == [expect]
    assert (lockdown._hook_agent_user.get(), lockdown._path_workspace.get()) == (
        False,
        None,
    )
