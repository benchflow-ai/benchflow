"""test.sh runs under the shell its task was written for, on real Docker.

Deterministic tier (``tests/integration/README.md``): real sandboxes, no model
(the ``oracle`` agent runs each task's solution). Runs where Docker answers
and skips with a reason otherwise.

Guards the fix for verifier scripts whose shebang is not on line 1: Docker
starts the verifier through ``sh -c``, dash on Ubuntu, so SWE-bench
Verified's test.sh (``#!/bin/bash`` on line 2) died on ``set -o pipefail``
and the oracle's trial was an unscored verifier crash. The local models of
every sandbox kind are in ``tests/test_verifier_script_shell.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
from benchflow.sandbox.docker import DockerSandbox
from benchflow.task import Task, Verifier
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths
from tests.integration.deterministic import harness as h

pytestmark = pytest.mark.deterministic
# Whole trials on real Docker: the exec channel and the side-by-side steps
# before the verifier stay on, as in a run (tests/conftest.py).
DOCKER_RUN_DEFAULTS = True

SANDBOX, SKIP_REASON = h.select_sandbox()

TASK_TOML = """version = "1.0"

[metadata]
author_name = "benchflow"
difficulty = "easy"
category = "sanity"

[verifier]
timeout_sec = 60.0

[agent]
timeout_sec = 120.0

[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 512
allow_internet = true
"""
SOLVE_SH = "#!/bin/bash\nprintf 'done\\n' > /app/done.txt\n"

# The SWE-bench Verified shape: a comment, then ``#!/bin/bash`` on line 2,
# then lines dash cannot run.
LATE_SHEBANG_TEST_SH = """# Run the instance's tests
#!/bin/bash
set -uo pipefail
files=(/app/done.txt)
if [[ -f ${files[0]} && -n "${BASH_VERSION:-}" ]]; then r=1; else r=0; fi
echo "$r" > /logs/verifier/reward.txt
"""

# Perl on line 1: the kernel runs it, never a shell (ubuntu's perl-base).
PERL_TEST_SH = """#!/usr/bin/env perl
open(my $f, ">", "/logs/verifier/reward.txt") or die;
print $f (-f "/app/done.txt" ? "1\\n" : "0\\n");
"""

# ubuntu:24.04 is the deterministic tier's base image, so the tier's hosts
# already hold it.
UBUNTU = "FROM ubuntu:24.04\nWORKDIR /app\n"
# Alpine has no bash; pinned by digest (3.20).
ALPINE = (
    "FROM alpine@sha256:"
    "d9e853e87e55526f6b2917df91a2115c36dd7c696a35be12163d44e6e2a4b6bc\n"
    "WORKDIR /app\n"
)

# POSIX, no shebang: 1 only when a shell other than bash ran it in an image
# with no bash.
POSIX_TEST_SH = """set -eu
if [ -z "${BASH_VERSION:-}" ] && ! command -v bash >/dev/null 2>&1; then
  r=1
else
  r=0
fi
echo "$r" > /logs/verifier/reward.txt
"""


@pytest.fixture
def require_docker() -> None:
    if SANDBOX != "docker":
        reason = SKIP_REASON if SANDBOX is None else f"Docker-only scenario ({SANDBOX})"
        pytest.skip(reason)


# Each test names its tasks apart from every other test's: Docker tags a
# task's image ``bf__<name>``, so two same-named tasks with different
# Dockerfiles, run at once under ``-n``, would build over each other's image.
def _task(
    root: Path, name: str, test_sh: str, dockerfile: str, solve_sh: str = SOLVE_SH
) -> Path:
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "solution").mkdir()
    (task / "tests").mkdir()
    (task / "task.toml").write_text(TASK_TOML)
    (task / "instruction.md").write_text("Write /app/done.txt.\n")
    (task / "environment" / "Dockerfile").write_text(dockerfile)
    for path, text in (
        (task / "solution" / "solve.sh", solve_sh),
        (task / "tests" / "test.sh", test_sh),
    ):
        path.write_text(text)
        path.chmod(0o755)
    return task


@pytest.mark.usefixtures("require_docker")
async def test_oracle_scores_late_shebang_and_interpreter_scripts(tmp_path):
    """The oracle scores 1 on a line-2 ``#!/bin/bash`` test.sh and on a Perl one."""
    tasks = tmp_path / "tasks"
    _task(tasks, "late-shebang", LATE_SHEBANG_TEST_SH, UBUNTU)
    _task(tasks, "perl-shebang", PERL_TEST_SH, UBUNTU)
    evaluation = Evaluation(
        tasks_dir=tasks,
        jobs_dir=tmp_path / "jobs",
        job_name="test-sh-shell",
        config=EvaluationConfig(
            agent="oracle",
            environment="docker",
            concurrency=2,
            retry=RetryConfig(max_retries=0),
        ),
    )

    result = await evaluation.run()

    assert (result.total, result.passed, result.errored) == (2, 2, 0), result


@pytest.mark.usefixtures("require_docker")
async def test_image_without_bash_runs_a_posix_test_sh_under_sh(tmp_path):
    """Alpine has no bash: a POSIX test.sh without a shebang runs under sh."""
    task_dir = _task(tmp_path / "tasks", "no-bash", POSIX_TEST_SH, ALPINE)
    paths = RolloutPaths(rollout_dir=tmp_path / "run")
    paths.mkdir()
    sandbox = DockerSandbox(
        environment_dir=task_dir / "environment",
        environment_name="test-sh-no-bash",
        session_id="no-bash",
        rollout_paths=paths,
        task_env_config=SandboxConfig(cpus=1, memory_mb=512),
    )
    try:
        await sandbox.start(force_build=False)
        result = await Verifier(
            task=Task(task_dir), rollout_paths=paths, sandbox=sandbox
        ).verify()
    finally:
        await sandbox.stop(delete=True)

    assert result.rewards == {"reward": 1.0}, (
        paths.test_stdout_path.read_text() if paths.test_stdout_path.exists() else ""
    )


# The image names a bash startup file in the agent's workspace; the "agent"
# (the oracle's solve.sh) writes one that scores 1 and exits. Every
# non-interactive bash runs that file before its script, so before the fix a
# ``#!/bin/bash`` test.sh, and a test.sh without a shebang once it ran under
# bash, scored 1 without running.
BASH_ENV_IMAGE = UBUNTU + "ENV BASH_ENV=/app/verifier-startup.sh\n"
STARTUP_SOLVE_SH = """#!/bin/bash
cat > /app/verifier-startup.sh <<'EOF'
printf '1\\n' > /logs/verifier/reward.txt
exit 0
EOF
"""
ZERO_TEST_SH = "printf '0\\n' > /logs/verifier/reward.txt\n"


@pytest.mark.usefixtures("require_docker")
async def test_agent_written_bash_env_file_never_runs_in_the_verifier(tmp_path):
    """Both test.sh shapes run and score 0; the agent's file does not run."""
    tasks = tmp_path / "tasks"
    for name, test_sh in (
        ("no-shebang", ZERO_TEST_SH),
        ("bash-shebang", "#!/bin/bash\n" + ZERO_TEST_SH),
    ):
        _task(
            tasks, f"startup-{name}", test_sh, BASH_ENV_IMAGE, solve_sh=STARTUP_SOLVE_SH
        )
    evaluation = Evaluation(
        tasks_dir=tasks,
        jobs_dir=tmp_path / "jobs",
        job_name="test-sh-bash-env",
        config=EvaluationConfig(
            agent="oracle",
            environment="docker",
            concurrency=2,
            retry=RetryConfig(max_retries=0),
        ),
    )

    result = await evaluation.run()

    assert (result.total, result.passed, result.errored) == (2, 0, 0), result


# The image activates its test env through a root-owned BASH_ENV in root-only
# folders (a conda or testbed activation), and test.sh needs that env. The
# verifier keeps a trusted image BASH_ENV as it keeps trusted PATH entries,
# so both test.sh shapes score 1, as they did before BASH_ENV was pinned.
# The trust check needs only the image's shell, ``readlink`` and ``stat``, so
# a shell-only image (bash, no python3) keeps it too.
ACTIVATION = (
    "RUN printf 'export TESTBED_ACTIVE=1\\n' > /etc/profile.d/activate-testbed.sh"
    " && chmod 644 /etc/profile.d/activate-testbed.sh\n"
    "ENV BASH_ENV=/etc/profile.d/activate-testbed.sh\n"
)
ACTIVATING_IMAGES = {
    "python": "FROM python:3.12-slim\nWORKDIR /app\n" + ACTIVATION,
    "shell-only": UBUNTU
    + "RUN command -v bash && ! command -v python3 && ! command -v python\n"
    + ACTIVATION,
}
ACTIVATED_TEST_SH = (
    'if [ "${TESTBED_ACTIVE:-}" = 1 ]; then r=1; else r=0; fi\n'
    'echo "$r" > /logs/verifier/reward.txt\n'
)


@pytest.mark.usefixtures("require_docker")
@pytest.mark.parametrize("image", sorted(ACTIVATING_IMAGES))
async def test_root_owned_image_bash_env_still_activates_the_verifier(tmp_path, image):
    """The image's root-owned BASH_ENV runs for test.sh: the oracle scores 1."""
    tasks = tmp_path / "tasks"
    for name, test_sh in (
        ("no-shebang", ACTIVATED_TEST_SH),
        ("bash-shebang", "#!/bin/bash\n" + ACTIVATED_TEST_SH),
    ):
        _task(tasks, f"activate-{image}-{name}", test_sh, ACTIVATING_IMAGES[image])
    evaluation = Evaluation(
        tasks_dir=tasks,
        jobs_dir=tmp_path / "jobs",
        job_name="test-sh-image-bash-env",
        config=EvaluationConfig(
            agent="oracle",
            environment="docker",
            concurrency=2,
            retry=RetryConfig(max_retries=0),
        ),
    )

    result = await evaluation.run()

    assert (result.total, result.passed, result.errored) == (2, 2, 0), result


# The task's BASH_ENV names a link in /dev/shm, outside every agent-writable
# prefix the verifier knows. The "agent" (the oracle's solve.sh) points it at
# ``/proc/self/cwd/etc/profile.d/bf-safe.sh``: from ``/``, where the check
# ran, that is the image's root-owned file; from ``/app``, where test.sh
# runs, it is the agent's copy, which scores 1 and exits. Before the fix the
# check kept the link and bash read the agent's file. The
# image has python3 so that the check before the fix ran at all.
LINK_IMAGE = (
    "FROM python:3.12-slim\nWORKDIR /app\n"
    "RUN printf 'export SAFE=1\\n' > /etc/profile.d/bf-safe.sh"
    " && chmod 644 /etc/profile.d/bf-safe.sh\n"
)
LINK_TASK_TOML = TASK_TOML + '\n[verifier.env]\nBASH_ENV = "/dev/shm/bf-env/start.sh"\n'
LINK_SOLVE_SH = """#!/bin/bash
mkdir -p /dev/shm/bf-env /app/etc/profile.d
chmod 755 /dev/shm/bf-env
ln -sfn /proc/self/cwd/etc/profile.d/bf-safe.sh /dev/shm/bf-env/start.sh
cat > /app/etc/profile.d/bf-safe.sh <<'EOF'
printf '1\\n' > /logs/verifier/reward.txt
exit 0
EOF
"""


@pytest.mark.usefixtures("require_docker")
async def test_a_bash_env_link_never_reaches_the_agent_file(tmp_path):
    """Both test.sh shapes run with the checked file and score 0."""
    tasks = tmp_path / "tasks"
    for name, test_sh in (
        ("no-shebang", ZERO_TEST_SH),
        ("bash-shebang", "#!/bin/bash\n" + ZERO_TEST_SH),
    ):
        task = _task(tasks, f"link-{name}", test_sh, LINK_IMAGE, solve_sh=LINK_SOLVE_SH)
        (task / "task.toml").write_text(LINK_TASK_TOML)
    evaluation = Evaluation(
        tasks_dir=tasks,
        jobs_dir=tmp_path / "jobs",
        job_name="test-sh-bash-env-link",
        config=EvaluationConfig(
            agent="oracle",
            environment="docker",
            concurrency=2,
            retry=RetryConfig(max_retries=0),
        ),
    )

    result = await evaluation.run()

    assert (result.total, result.passed, result.errored) == (2, 0, 0), result


# The image's PATH starts with a folder that is root's, so the verifier's
# PATH keeps it, but its ``stat`` is world-writable; ``/usr/local/bin/stat``
# is too. The task's BASH_ENV names a file in a folder the agent can write.
# The "agent" (the oracle's solve.sh) rewrites both ``stat``s to call every
# file root's and writes a startup file that scores 1 and exits. Before the
# fix the check believed the planted ``stat`` and bash ran the agent's file
# as root; now the check takes its tools from the system
# folders only and refuses a tool anyone can write.
PLANTED_STAT_IMAGE = (
    "FROM python:3.12-slim\nWORKDIR /app\n"
    "RUN mkdir -p /opt/check-tools /opt/agent-state"
    " && cp /usr/bin/stat /opt/check-tools/stat"
    " && cp /usr/bin/stat /usr/local/bin/stat"
    " && chmod 0777 /opt/check-tools/stat /usr/local/bin/stat /opt/agent-state\n"
    "ENV PATH=/opt/check-tools:$PATH\n"
)
PLANTED_STAT_TASK_TOML = (
    TASK_TOML + '\n[verifier.env]\nBASH_ENV = "/opt/agent-state/start.sh"\n'
)
PLANTED_STAT_SOLVE_SH = r"""#!/bin/sh
for f in /opt/check-tools/stat /usr/local/bin/stat; do
  printf '%s\n' '#!/bin/sh' \
    'case "$*" in *"%u %a"*) echo "0 755"; exit 0 ;; esac' \
    'exec /usr/bin/stat "$@"' > "$f"
done
printf '%s\n' "printf '1\\n' > /logs/verifier/reward.txt" 'exit 0' \
  > /opt/agent-state/start.sh
# The agent owns what it writes (the oracle may run as root).
chown 65534:65534 /opt/agent-state/start.sh 2>/dev/null || true
"""


@pytest.mark.usefixtures("require_docker")
async def test_a_planted_stat_never_lets_the_agent_file_run(tmp_path):
    """Both test.sh shapes run without the agent's file and score 0."""
    tasks = tmp_path / "tasks"
    for name, test_sh in (
        ("no-shebang", ZERO_TEST_SH),
        ("bash-shebang", "#!/bin/bash\n" + ZERO_TEST_SH),
    ):
        task = _task(
            tasks,
            f"planted-stat-{name}",
            test_sh,
            PLANTED_STAT_IMAGE,
            solve_sh=PLANTED_STAT_SOLVE_SH,
        )
        (task / "task.toml").write_text(PLANTED_STAT_TASK_TOML)
    evaluation = Evaluation(
        tasks_dir=tasks,
        jobs_dir=tmp_path / "jobs",
        job_name="test-sh-bash-env-planted-stat",
        config=EvaluationConfig(
            agent="oracle",
            environment="docker",
            concurrency=2,
            retry=RetryConfig(max_retries=0),
        ),
    )

    result = await evaluation.run()

    assert (result.total, result.passed, result.errored) == (2, 0, 0), result
