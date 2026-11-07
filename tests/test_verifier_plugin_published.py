"""The plugin guard trusts verifier-installed plugins by their bytes.

Seven review rounds (V2-R6 to V2-R13) each found test.sh spellings whose
installer read a file the agent left while the guard's parser said it did
not. The guard no longer asks which files an installer read: a plugin test.sh
installs into the agent's trees, or with uvx into the verifier's uv state,
loads when its files are a published release BenchFlow vouches for, byte for
byte, or when a proof frozen before test.sh says the agent changed no file in
its trees. These tests fail on f05ae800 (the reviewed tip 233b7333 with
integ/py-0.8 7b4907c8 merged in) where that tip admitted a hostile plugin or
refused an honest one, and show that the outcome no longer depends on how
test.sh spells its install.
"""

from pathlib import Path

import pytest

from tests import test_verifier_uv_state as uv_state
from tests.test_verifier_uv_state import (
    CTRF_PLUGIN,
    FAILING,
    PASSING,
    hostile_ctrf,
    install_plugin,
    markers,
    plugin_never_ran,
    refused_scored_zero,
    score,
)

# ---------------------------------------------------------------------------
# Through hardening, the real guard and a real pytest (installs modelled).
# ---------------------------------------------------------------------------

# V2-R13-CONF-1: inline script metadata naming a workspace wheel.
INLINE_METADATA = (
    "# /// script\n"
    "# dependencies = [\n"
    '#   "pytest",\n'
    '#   "pytest-json-ctrf @ file:///app/pytest_json_ctrf-0.3.5-py3-none-any.whl",\n'
    "# ]\n"
    "# ///\n"
)
# Inline metadata that really is empty: an honest supported input.
EMPTY_METADATA = "# /// script\n# dependencies = []\n# ///\n"

# How test.sh installs, in every spelling a review used to get a plugin chosen
# by the agent's files past the parser, plus some no review tried. {venv} is
# the fresh environment, {ws} the workspace.
SPELLINGS = {
    "r13-inline-metadata": "uv pip install --python {venv}/bin/python -r /tests/deps.py",
    "r13-printf-v": (
        "printf -v UV_CONFIG_FILE %s {ws}/install.toml\n"
        "uv pip install --python {venv}/bin/python --no-deps pytest-json-ctrf"
    ),
    "r13-quoted-wildcard": (
        "uv pip install --python {venv}/bin/python 'pytest==8.*' pytest-json-ctrf"
    ),
    "r13-quoted-extras": (
        "uv pip install --python {venv}/bin/python 'pytest[testing]' pytest-json-ctrf"
    ),
    "r12-bare-archive": (
        "pip --python {venv}/bin/python install pytest_json_ctrf-0.3.5-py3-none-any.whl"
    ),
    "r12-pip-sync": "{venv}/bin/pip-sync",
    "r11-timed-group": "time {{ uv pip install -r {ws}/deps.txt; }}",
    "r10-brace-group": "{{ uv pip install -r {ws}/deps.txt; }}",
    "r10-format": (
        "python3 -c 'import subprocess; "
        'subprocess.run(["uv", "pip", "install", "-r{{}}".format("{ws}/deps.txt")])\''
    ),
    "r9-attached": "uv pip install -r{ws}/deps.txt",
    "r9-module": "python3 -mpip install -r {ws}/deps.txt",
    "r8-locale-heredoc": "bash <<$'SH'\nuv pip install -r {ws}/deps.txt\nSH",
    "r8-descriptor": "bash -s 3<<'SH' < /dev/fd/3\nuv pip install -r {ws}/deps.txt\nSH",
    "literal-protected": "uv pip install -r /tests/deps.txt",
    "eval": 'eval "$(cat {ws}/install-command)"',
    "xargs": "cat {ws}/deps.txt | xargs uv pip install",
    "make": "make -C {ws} install-test-deps",
    "env-requirement": "PIP_REQUIREMENT={ws}/deps.txt pip install",
    "future-installer": "frobnicate-install --from-anywhere {ws}",
}


def runner(spelling: str, layout) -> str:
    venv = f"{layout.workspace}/.venv"
    return (
        f"uv venv {venv}\n"
        + SPELLINGS[spelling].format(venv=venv, ws=layout.workspace)
        + f"\n{venv}/bin/python -m pytest --ctrf /logs/verifier/ctrf.json "
        "/tests/test_outputs.py\n"
    )


def agent_inputs(layout):
    """What a hostile agent leaves for any of the spellings to pick up."""
    for name, text in {
        "deps.txt": "pytest-json-ctrf @ file:///app/wheels/plugin.whl\n",
        "install.toml": "[pip]\nno-index = true\nfind-links = ['/app/wheels']\n",
        "install-command": "uv pip install -r /app/deps.txt\n",
        "requirements.txt": "pytest-json-ctrf @ file:///app/wheels/plugin.whl\n",
        "pytest_json_ctrf-0.3.5-py3-none-any.whl": "DUMMY agent wheel",
        "wheels/plugin.whl": "DUMMY agent wheel",
    }.items():
        path = layout.workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def honest_answer(layout):
    (layout.workspace / "answer.txt").write_text("42\n")


def use_runner(monkeypatch, spelling, deps="pytest-json-ctrf==0.3.5\n"):
    original = uv_state.make_task

    def make_task(layout, env=None):
        task = original(layout, test_sh=runner(spelling, layout), env=env)
        (task.paths.tests_dir / "deps.txt").write_text(deps)
        (task.paths.tests_dir / "deps.py").write_text(
            INLINE_METADATA if spelling == "r13-inline-metadata" else EMPTY_METADATA
        )
        # An empty task overlay: the reviews' precise subset needed one.
        task.config.verifier.env = {}
        return task

    monkeypatch.setattr(uv_state, "make_task", make_task)


def venv_install(source):
    def install(env, layout):
        site = layout.workspace / ".venv/lib/python3.12/site-packages"
        install_plugin(site, source)
        (layout.workspace / ".venv/pyvenv.cfg").write_text("home = /usr/bin\n")
        return [site]

    return install


@pytest.mark.parametrize("workspace", ["app", "testbed", "root"])
@pytest.mark.parametrize("spelling", ["r13-inline-metadata", "r13-printf-v"])
async def test_r13_hostile_inputs_never_choose_a_trusted_plugin(
    tmp_path, monkeypatch, spelling, workspace
):
    """V2-R13-CONF-1/2 at 233b7333: the parser certified these runners, so the
    proof ignored the agent's wheel and configuration, and the plugin they
    chose passed every test of a failing solution."""
    proof = tmp_path / "hostile-imported"
    use_runner(monkeypatch, spelling)

    layout, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        FAILING,
        venv_install(hostile_ctrf(proof)),
        before=agent_inputs,
        workspace_name=workspace,
    )

    assert not proof.exists()
    assert plugin_never_ran(rewards, error), (rewards, error)
    assert markers(layout, "installed") == []


@pytest.mark.parametrize("workspace", ["app", "testbed", "root"])
@pytest.mark.parametrize("plane", ["same", "separate"])
@pytest.mark.parametrize(
    "spelling", ["r13-quoted-wildcard", "r13-quoted-extras", "literal-protected"]
)
@pytest.mark.parametrize("case", ["oracle", "nop"])
async def test_honest_supported_installs_are_scored(
    tmp_path, monkeypatch, case, spelling, plane, workspace
):
    """V2-R13-CONF-3 at 233b7333: quoted extras and version wildcards made
    every agent file an input, and an honest answer then refused the
    verifier's own ctrf. A published release loads however it was spelled."""
    use_runner(monkeypatch, spelling)

    _, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        PASSING if case == "oracle" else FAILING,
        venv_install(CTRF_PLUGIN),
        before=honest_answer,
        workspace_name=workspace,
        agent_paths=("{workspace}",) if plane == "separate" else None,
    )

    assert rewards == {"reward": float(case == "oracle")} and error is None, (
        rewards,
        error,
    )


@pytest.mark.parametrize("case", ["oracle", "nop"])
async def test_inline_metadata_that_is_really_empty_is_scored(
    tmp_path, monkeypatch, case
):
    """An honest ``-r /tests/deps.py`` whose metadata lists nothing, beside an
    honest answer: scored, as before, now without any parser's say."""
    use_runner(monkeypatch, "r13-inline-metadata")
    original = uv_state.make_task

    def make_task(layout, env=None):
        task = original(layout, env=env)
        (task.paths.tests_dir / "deps.py").write_text(EMPTY_METADATA)
        return task

    monkeypatch.setattr(uv_state, "make_task", make_task)

    _, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        PASSING if case == "oracle" else FAILING,
        venv_install(CTRF_PLUGIN),
        before=honest_answer,
        workspace_name="app",
    )

    assert rewards == {"reward": float(case == "oracle")} and error is None, (
        rewards,
        error,
    )


@pytest.mark.parametrize("spelling", sorted(SPELLINGS))
@pytest.mark.parametrize("case", ["oracle", "nop", "hostile"])
async def test_the_outcome_does_not_depend_on_how_test_sh_installs(
    tmp_path, monkeypatch, spelling, case
):
    """The whole class: for every spelling (each review's, and ones no review
    tried), the agent's files present, a published ctrf is scored and a
    hostile one is refused before it is imported."""
    proof = tmp_path / "hostile-imported"
    use_runner(monkeypatch, spelling)

    def leave(layout):
        agent_inputs(layout)
        honest_answer(layout)

    layout, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        PASSING if case == "oracle" else FAILING,
        venv_install(hostile_ctrf(proof) if case == "hostile" else CTRF_PLUGIN),
        before=leave,
        workspace_name="app",
    )

    assert not proof.exists()
    if case == "hostile":
        assert plugin_never_ran(rewards, error), (rewards, error)
        assert markers(layout, "installed") == []
    else:
        assert rewards == {"reward": float(case == "oracle")} and error is None, (
            rewards,
            error,
        )


def uvx_from_agent_wheels(source):
    """``uvx --find-links /app/wheels --with pytest-json-ctrf pytest``: what
    the wheels hold lands in the verifier's uv cache, a root-only directory."""

    def place(env, layout):
        site = Path(env["UV_CACHE_DIR"]) / "archive-v0/ctrf-env/site-packages"
        install_plugin(site, source)
        return [site]

    return place


@pytest.mark.parametrize("case", ["oracle", "nop", "hostile"])
async def test_uvx_inputs_cannot_choose_trusted_code_in_the_verifier_uv_state(
    tmp_path, monkeypatch, case
):
    """The same class below the verifier's uv state: at 233b7333 code there was
    trusted by its path alone, whichever agent files uvx read to choose it."""
    proof = tmp_path / "hostile-imported"
    uvx_test_sh = (
        "uvx --find-links /app/wheels --with pytest-json-ctrf "
        "pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py\n"
    )
    original = uv_state.make_task
    monkeypatch.setattr(
        uv_state,
        "make_task",
        lambda layout, env=None: original(layout, test_sh=uvx_test_sh, env=env),
    )

    def leave(layout):
        agent_inputs(layout)
        honest_answer(layout)

    _, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        PASSING if case == "oracle" else FAILING,
        uvx_from_agent_wheels(
            hostile_ctrf(proof) if case == "hostile" else CTRF_PLUGIN
        ),
        before=leave,
    )

    assert not proof.exists()
    if case == "hostile":
        assert plugin_never_ran(rewards, error), (rewards, error)
    else:
        assert rewards == {"reward": float(case == "oracle")} and error is None, (
            rewards,
            error,
        )


async def test_an_unpinned_plugin_uvx_installs_is_refused_even_when_the_agent_left_nothing(
    tmp_path, monkeypatch, caplog
):
    """Nothing test.sh installs after the agent's turn loads, even
    below the verifier's uv state, even where the proof says the agent left
    its files alone: the plugin must come from the protected runtime (a pin)
    or the image. This used to load by the uv state's path."""
    _, _, rewards, error = await score(
        tmp_path, monkeypatch, FAILING, uv_state.uvx, published=False
    )
    assert refused_scored_zero(rewards, error), (rewards, error)
    assert "pytest plugin guard not confirmed" in caplog.text
    assert error is None
