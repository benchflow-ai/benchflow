"""ACC-RUN2: the PY8-only classes of acceptance run 2, as real Docker trials.

Each task is the shape of a Terminal-Bench task that PY8 left unscored or
scored wrong while the reference harnesses scored it (results/ACC-RUN2.md):

- a root-installed start-up hook (largest-eigenval: eigenpy is a cmeel
  package whose ``cmeel.pth`` is ``import cmeel_pth``);
- a root link on ``PATH`` to a binary the solution built in the workspace
  (sqlite-with-gcov);
- a protected runtime kit whose Python copy failed once (usaco/987);
- the protected runner's copy of the image's site-packages, written for every
  ``--ctrf`` task (the disk class: mteb-retrieve, pytorch-model-recovery,
  mteb-leaderboard).

Each runs under the default sandbox user ``agent`` with the oracle running
solve.sh as root, as on Daytona. The controls keep the threats closed: a
hook the agent user owns or could have written still goes, and a link into a
temporary folder still refuses grading.
"""

from __future__ import annotations

import pytest

from tests.exec_diet_fakes import PYTHON, docker_available

# The real protected-verifier-runtime check, as on Daytona (conftest stubs it
# out otherwise).
pytestmark = [pytest.mark.deterministic, pytest.mark.verifier_runtime]

SITE = (
    "SITE=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()[\"purelib\"])')"
)

BASE = f"""\
FROM {PYTHON}
WORKDIR /app
RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts
"""

WITH_PYTEST = (
    BASE + "RUN pip install --no-cache-dir -q pytest==8.4.1 pytest-json-ctrf==0.3.5\n"
)

CTRF_TEST = """\
#!/bin/bash
cd /app
python3 -m pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA
if [ $? -eq 0 ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
"""


def _task(tmp_path, name, *, dockerfile, solve, test, tests_py=None):
    task = tmp_path / name
    (task / "environment").mkdir(parents=True)
    (task / "solution").mkdir()
    (task / "tests").mkdir()
    (task / "task.toml").write_text(
        'version = "1.0"\n\n[verifier]\ntimeout_sec = 300.0\n\n'
        "[agent]\ntimeout_sec = 300.0\n\n[environment]\nbuild_timeout_sec = 900.0\n"
    )
    (task / "instruction.md").write_text("Do the task.\n")
    (task / "environment" / "Dockerfile").write_text(dockerfile)
    (task / "solution" / "solve.sh").write_text("#!/bin/bash\nset -e\n" + solve)
    (task / "tests" / "test.sh").write_text(test)
    if tests_py is not None:
        (task / "tests" / "test_outputs.py").write_text(tests_py)
    return task


async def _run(task, agent):
    from benchflow.rollout import Rollout, RolloutConfig

    return await Rollout(
        RolloutConfig(
            task_path=task,
            agent=agent,
            environment="docker",
            sandbox_user="agent",
            jobs_dir=task.parent / "jobs",
            job_name=f"{task.name}-{agent}",
            rollout_name=f"{task.name}__{agent}",
        )
    ).run()


def _scored(result, reward):
    assert result.error is None, result.error
    assert result.verifier_error is None, result.verifier_error
    assert result.rewards == {"reward": reward}


@pytest.fixture(autouse=True)
def _needs_docker():
    if not docker_available():
        pytest.skip("needs a Docker daemon")


# -- start-up hooks: largest-eigenval ------------------------------------------

# What pip leaves for a cmeel package: a .pth whose line imports a module that
# puts the package's own prefix on sys.path.
HOOK_SOLVE = (
    SITE
    + """
mkdir -p /usr/local/lib/accrun2-prefix
echo 'VALUE = "from-the-prefix"' > /usr/local/lib/accrun2-prefix/accmod.py
printf 'import sys\\nsys.path.append("/usr/local/lib/accrun2-prefix")\\n' > "$SITE/accrun2_pth.py"
echo 'import accrun2_pth' > "$SITE/accrun2.pth"
"""
)

HOOK_TEST = """\
#!/bin/bash
cd /app
v=$(python3 -c 'import accmod; print(accmod.VALUE)' 2>&1)
echo "$v"
if [ "$v" = from-the-prefix ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
"""


async def test_a_code_pth_root_installed_during_the_turn_stays(tmp_path):
    """largest-eigenval: the oracle (root) pip-installs a package whose .pth
    imports a module; the agent user could not have written the root-only
    site folder, so the hook stays and the import works, as in the
    references. At 57d28d656 the hook is removed and this scores 0."""
    task = _task(
        tmp_path, "accrun2-hook", dockerfile=BASE, solve=HOOK_SOLVE, test=HOOK_TEST
    )
    _scored(await _run(task, "oracle"), 1.0)


async def test_a_code_pth_the_agent_user_owns_still_goes(tmp_path):
    """Control: the same hook owned by the agent user is removed."""
    task = _task(
        tmp_path,
        "accrun2-hook-agent",
        dockerfile=BASE,
        solve=HOOK_SOLVE + 'chown agent "$SITE/accrun2.pth"\n',
        test=HOOK_TEST,
    )
    _scored(await _run(task, "oracle"), 0.0)


async def test_a_code_pth_in_a_site_folder_others_may_write_still_goes(tmp_path):
    """Control: root wrote it, but into a site folder the agent user may
    write, where the agent could have put it; it is removed."""
    task = _task(
        tmp_path,
        "accrun2-hook-open",
        dockerfile=BASE + f'RUN {SITE} && chmod o+w "$SITE"\n',
        solve=HOOK_SOLVE,
        test=HOOK_TEST,
    )
    _scored(await _run(task, "oracle"), 0.0)


async def test_a_code_pth_the_agent_user_owns_and_linked_into_the_workspace_goes(
    tmp_path,
):
    """Review A, L1: an agent-owned hook with a second name in the workspace
    keeps its owner through the freeze, and its link count still says the
    agent user could have written it, so it is removed. At round 1
    (71c123fe0) it is kept and this scores 1."""
    task = _task(
        tmp_path,
        "accrun2-hook-hardlink",
        dockerfile=BASE,
        solve=HOOK_SOLVE
        + 'ln "$SITE/accrun2.pth" /app/accrun2-link.pth\n'
        + 'chown agent "$SITE/accrun2.pth"\n',
        test=HOOK_TEST,
    )
    _scored(await _run(task, "oracle"), 0.0)


@pytest.mark.parametrize("delivery", ["off", "auto"])
async def test_a_hook_whose_second_name_the_conftest_sweep_removes_goes(
    tmp_path, monkeypatch, delivery
):
    """ACC-RUN2 r2, L1: the second name is /app/conftest.py, which the
    clean-up's conftest sweep removes before the hook rule counts names. The
    freeze leaves a file with a second name as it is, with bfx or without
    (``off``: the plain freeze, as on Daytona), so the hook is still the
    agent user's and is removed. At dc509087a the plain freeze made it
    root's, the sweep left it one name, and it was kept (1)."""
    monkeypatch.setenv("BENCHFLOW_BFX_DELIVERY", delivery)
    task = _task(
        tmp_path,
        f"accl1-hook-conftest-{delivery}",
        dockerfile=BASE,
        solve=HOOK_SOLVE
        + 'ln "$SITE/accrun2.pth" /app/conftest.py\n'
        + 'chown agent "$SITE/accrun2.pth"\n',
        test=HOOK_TEST,
    )
    _scored(await _run(task, "oracle"), 0.0)


async def test_without_bfx_a_hook_with_a_second_name_elsewhere_goes(
    tmp_path, monkeypatch
):
    """Control for the case above on the plain freeze: the second name is
    one no clean-up removes."""
    monkeypatch.setenv("BENCHFLOW_BFX_DELIVERY", "off")
    task = _task(
        tmp_path,
        "accl1-hook-plain-off",
        dockerfile=BASE,
        solve=HOOK_SOLVE
        + 'ln "$SITE/accrun2.pth" /app/accrun2-link.pth\n'
        + 'chown agent "$SITE/accrun2.pth"\n',
        test=HOOK_TEST,
    )
    _scored(await _run(task, "oracle"), 0.0)


async def test_without_bfx_a_root_installed_hook_stays(tmp_path, monkeypatch):
    """The plain freeze keeps the largest-eigenval case: one name, root's."""
    monkeypatch.setenv("BENCHFLOW_BFX_DELIVERY", "off")
    task = _task(
        tmp_path,
        "accl1-hook-root-off",
        dockerfile=BASE,
        solve=HOOK_SOLVE,
        test=HOOK_TEST,
    )
    _scored(await _run(task, "oracle"), 1.0)


# -- PATH: sqlite-with-gcov ----------------------------------------------------

TOOL_TEST = """\
#!/bin/bash
cd /app
if [ "$(acctool 2>&1)" = acc-tool-ok ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
"""


def _tool_solve(folder):
    return f"""\
mkdir -p {folder}
printf '#!/bin/sh\\necho acc-tool-ok\\n' > {folder}/acctool
chmod 755 {folder}/acctool
ln -s {folder}/acctool /usr/local/bin/acctool
"""


async def test_a_root_link_to_a_binary_built_in_the_workspace_runs(tmp_path):
    """sqlite-with-gcov: solve.sh builds the tool in /app and links it into
    /usr/local/bin; the tests run it from PATH. At 57d28d656 hardening stops
    ("captured PATH exclusions could not be enforced")."""
    task = _task(
        tmp_path,
        "accrun2-tool",
        dockerfile=BASE,
        solve=_tool_solve("/app/tool"),
        test=TOOL_TEST,
    )
    _scored(await _run(task, "oracle"), 1.0)


async def test_without_the_link_the_tool_is_missing(tmp_path):
    task = _task(
        tmp_path,
        "accrun2-tool-nop",
        dockerfile=BASE,
        solve=_tool_solve("/app/tool"),
        test=TOOL_TEST,
    )
    _scored(await _run(task, "nop"), 0.0)


async def test_a_root_link_into_a_temporary_folder_still_refuses(tmp_path):
    """Control: /tmp is not the submission; the route there still refuses."""
    task = _task(
        tmp_path,
        "accrun2-tool-tmp",
        dockerfile=BASE,
        solve=_tool_solve("/tmp/acc"),
        test=TOOL_TEST,
    )
    result = await _run(task, "oracle")
    assert result.verifier_error is not None
    assert "/usr/local/bin" in result.verifier_error
    assert result.rewards in (None, {})


async def test_a_link_into_the_workspace_the_image_shipped_still_refuses(tmp_path):
    """Review B, M1: the image, not the turn, links a PATH name into the
    workspace; whatever the agent leaves there would run under that name, so
    the route still refuses grading, as at 57d28d656. At round 1 (71c123fe0)
    it is kept and this scores 1."""
    task = _task(
        tmp_path,
        "accrun2-tool-image",
        dockerfile=BASE
        + "RUN mkdir -p /app/tool && ln -s /app/tool/acctool /usr/local/bin/acctool\n",
        solve=_tool_solve("/app/tool").replace(
            "ln -s /app/tool/acctool /usr/local/bin/acctool\n", ""
        ),
        test=TOOL_TEST,
    )
    result = await _run(task, "oracle")
    assert result.verifier_error is not None
    assert "/usr/local/bin" in result.verifier_error
    assert result.rewards in (None, {})


# Review B's configuration check (M1): the judge's answer for an image PATH
# folder whose links into the workspace the image shipped, and for links root
# added during the turn. No trial: the record is made, the turn's links are
# added, and the judge runs, each as root in one container.
_PATHCFG_IMAGE = (
    "useradd -m agent && mkdir -p /app/bin /opt/tools /opt/added /opt/plain"
    " && printf '#!/bin/sh\\necho hi\\n' > /app/bin/x && chmod 755 /app/bin/x"
    " && chown -R agent /app"
    " && for n in sqlite3 python3 pytest; do ln -s /app/bin/x /opt/tools/$n; done"
    " && cp /app/bin/x /opt/plain/ok && cp /app/bin/x /opt/added/ok"
)
_PATHCFG_TURN = (
    "mkdir -p /opt/late && ln -s /app/bin/x /opt/late/sqlite3"
    " && ln -s /app/bin/x /opt/added/sqlite3"
)
_PATHCFG_FOLDERS = ("/opt/tools", "/opt/late", "/opt/added", "/opt/plain")


def _pathcfg(with_record=True):
    """{folder: 'K' or 'D'} for each of _PATHCFG_FOLDERS, by the judge of
    whatever commit is checked out (older ones take fewer arguments)."""
    import inspect
    import subprocess
    import uuid

    from benchflow._utils.trusted_path import decode_path_field
    from benchflow.sandbox import lockdown

    record_script = getattr(lockdown, "_PATH_NAMES_RECORD", None)
    params = inspect.signature(lockdown._trusted_path_extras_cmd).parameters
    name = f"accr2b2-pathcfg-{uuid.uuid4().hex[:8]}"
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--label",
            "accr2b2=1",
            "--network",
            "none",
            "--name",
            name,
            PYTHON,
            "sleep",
            "600",
        ],
        check=True,
        capture_output=True,
    )

    def sh(script, *args):
        return subprocess.run(
            ["docker", "exec", "-u", "0", name, "sh", "-c", script, "sh", *args],
            capture_output=True,
            text=True,
        )

    try:
        assert sh(_PATHCFG_IMAGE).returncode == 0
        record = ""
        if record_script is not None and with_record:
            # Before the turn, as _prepare_bash_env_inspector does.
            assert sh("mkdir -m 700 /etc/accr2b2-insp").returncode == 0
            made = sh(
                record_script,
                "/etc/accr2b2-insp",
                ":".join(_PATHCFG_FOLDERS),
                *lockdown._SAFE_VERIFIER_PATH_PARTS,
            )
            assert made.returncode == 0, made.stderr
            record = "/etc/accr2b2-insp/" + lockdown._PATH_NAMES_DIR
        assert sh(_PATHCFG_TURN).returncode == 0
        kw = {}
        if "workspace" in params:
            kw["workspace"] = "/app"
        if "record" in params:
            kw["record"] = record
        answer = sh(
            lockdown._trusted_path_extras_cmd(
                ":".join(_PATHCFG_FOLDERS), ("/app",), agent="agent", **kw
            )
        )
        assert answer.returncode == 0, answer.stderr
        return {
            decode_path_field(line[2:]): line[0]
            for line in answer.stdout.splitlines()
            if line[:2] in ("K ", "D ")
        }
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def test_path_config_image_links_left_out_turn_links_kept():
    """57d28d656: every folder with a link into /app is D. Round 1
    (71c123fe0): all K, the image's python3 and pytest links included. Now:
    the image's links D, links root added during the turn K."""
    assert _pathcfg() == {
        "/opt/tools": "D",
        "/opt/late": "K",
        "/opt/added": "K",
        "/opt/plain": "K",
    }


def test_path_config_without_a_pre_agent_record_keeps_nothing_in_the_workspace():
    assert _pathcfg(with_record=False) == {
        "/opt/tools": "D",
        "/opt/late": "D",
        "/opt/added": "D",
        "/opt/plain": "K",
    }


# -- kit Python: usaco/987 -----------------------------------------------------

# The image's cp fails once, the first time its destination is the kit's copy
# of the standard library: a step that fails once on a busy sandbox.
FLAKY_CP = r"""
RUN mv /usr/bin/cp /usr/bin/cp.real \
 && printf '%s\n' '#!/bin/sh' \
    'for a in "$@"; do case "$a" in /opt/benchflow/rt/lib/python*) if [ ! -e /var/lib/accrun2-cp-failed ]; then : > /var/lib/accrun2-cp-failed; echo "accrun2: cp failed once" >&2; exit 1; fi;; esac; done' \
    'exec /usr/bin/cp.real "$@"' > /usr/bin/cp \
 && chmod 755 /usr/bin/cp
"""

ANSWER_TEST_PY = """\
def test_answer():
    assert open("/app/answer.txt").read().strip() == "ok"
"""


async def test_a_kit_python_copy_that_failed_once_is_made_again(tmp_path):
    """usaco/987: one failed step of the kit's Python copy recorded
    python=no on a Python image, and the --ctrf verifier's trust discovery
    then refused. At 57d28d656 this trial is unscored."""
    task = _task(
        tmp_path,
        "accrun2-kit",
        dockerfile=WITH_PYTEST + FLAKY_CP,
        solve="echo ok > /app/answer.txt\n",
        test=CTRF_TEST,
        tests_py=ANSWER_TEST_PY,
    )
    _scored(await _run(task, "oracle"), 1.0)


# -- disk: the runner's copy of the image's site-packages ----------------------

# 48 MB in the image's site-packages, standing in for a CUDA torch.
BIG_SITE = (
    WITH_PYTEST
    + f'RUN {SITE} && mkdir "$SITE/accbig" && python3 -c '
    + "\"import os, sys; open(sys.argv[1], 'wb').write(os.urandom(48 << 20))\" "
    + '"$SITE/accbig/blob.bin"\n'
)

DISK_TEST_PY = (
    ANSWER_TEST_PY
    + """

def test_benchflow_wrote_no_copy_of_the_image_site_packages():
    import subprocess

    out = subprocess.run(
        ["du", "-sk", "/opt/benchflow/verifier"], capture_output=True, text=True
    ).stdout
    assert int(out.split()[0]) < 16 * 1024, out
"""
)


async def test_a_ctrf_task_gets_no_copy_of_the_image_site_packages(tmp_path):
    """The disk class: a TB2-shaped --ctrf task never starts the protected
    runner, so its copy of the image's site-packages (here 48 MB, for a torch
    image 5 to 8 GB) is not written. At 57d28d656 it is, and this scores 0."""
    task = _task(
        tmp_path,
        "accrun2-disk",
        dockerfile=BIG_SITE,
        solve="echo ok > /app/answer.txt\n",
        test=CTRF_TEST,
        tests_py=DISK_TEST_PY,
    )
    _scored(await _run(task, "oracle"), 1.0)


async def test_a_task_that_names_the_runner_still_gets_it(tmp_path):
    """Control: a test.sh that runs $BENCHFLOW_PYTEST gets the runner, which
    still serves the image's packages from its pre-agent copy."""
    task = _task(
        tmp_path,
        "accrun2-runner",
        dockerfile=BIG_SITE,
        solve="echo ok > /app/answer.txt\n",
        test=CTRF_TEST.replace("python3 -m pytest", '"$BENCHFLOW_PYTEST"'),
        tests_py=ANSWER_TEST_PY
        + """

def test_the_image_package_is_served():
    import accbig  # noqa: F401
""",
    )
    _scored(await _run(task, "oracle"), 1.0)
