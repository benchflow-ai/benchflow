"""EN-724: an image whose setpriv cannot drop "all" caps still runs test.sh.

SWE-bench Pro's ansible and qutebrowser images ship a util-linux setpriv
built on an old libcap-ng: its --help lists --bounding-set, but
``--inh-caps=-all`` makes it print ``setpriv: libcap-ng is too old for "all"
caps`` and exit. The verifier's bounded launch trusted --help alone, so
test.sh never started and every such trial ended "verifier launch refused:
test.sh's launch record is unreadable" (unscored) where 0.9.2 scored it.
The launch now tries the exact drop on a no-op first and, where it fails,
makes the same drop from BenchFlow's kit Python: test.sh runs held, and the
trial is scored.

The fixture's setpriv behaves as that old one does: --help lists
--bounding-set, any "all" capability set fails with that message, and
anything else runs its command.

Deterministic tier (``tests/integration/README.md``): real Docker, the
oracle, no model.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tests.integration.deterministic import harness as h

pytestmark = pytest.mark.deterministic

SANDBOX, SKIP_REASON = h.select_sandbox()
OLD_SETPRIV = r"""#!/bin/sh
case " $* " in *" --help "*) echo " --bounding-set <caps>  set capability bounding set"; exit 0;; esac
case " $* " in *=-all*|*",-all"*) echo 'setpriv: libcap-ng is too old for "all" caps' >&2; exit 1;; esac
while [ $# -gt 0 ] && [ "$1" != "--" ]; do shift; done
shift
exec "$@"
"""
DOCKERFILE = (
    "FROM python:3.12-slim@sha256:"
    "a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1\n"
    "RUN mkdir -p /app && chmod 1777 /app\n"
    "COPY old-setpriv /usr/bin/setpriv\n"
    "RUN chmod 755 /usr/bin/setpriv\n"
    "WORKDIR /app\n"
)
TEST_SH = (
    "#!/bin/bash\n"
    'if [ "$(cat /app/answer.txt 2>/dev/null)" = ok ]; then r=1; else r=0; fi\n'
    'echo "$r" > /logs/verifier/reward.txt\n'
)
TASK_TOML = """version = "1.0"
[metadata]
author_name = "benchflow"
difficulty = "easy"
category = "fixture"
[verifier]
timeout_sec = 120.0
[agent]
timeout_sec = 120.0
[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 1024
storage_mb = 4096
"""


@pytest.fixture(scope="module")
def trial(tmp_path_factory):
    if SANDBOX != "docker":
        pytest.skip(SKIP_REASON if SANDBOX is None else f"Docker-only ({SANDBOX})")
    root = tmp_path_factory.mktemp("old-setpriv")
    task = root / "tasks" / "old-setpriv"
    for sub in ("environment", "solution", "tests"):
        (task / sub).mkdir(parents=True)
    (task / "task.toml").write_text(TASK_TOML)
    (task / "instruction.md").write_text("Write ok to /app/answer.txt.\n")
    (task / "environment" / "Dockerfile").write_text(DOCKERFILE)
    (task / "environment" / "old-setpriv").write_text(OLD_SETPRIV)
    for path, text in (
        (task / "solution" / "solve.sh", "#!/bin/bash\necho ok > /app/answer.txt\n"),
        (task / "tests" / "test.sh", TEST_SH),
    ):
        path.write_text(text)
        path.chmod(0o755)
    jobs = root / "jobs"
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchflow.cli.main",
            "eval",
            "run",
            "--tasks-dir",
            str(root / "tasks"),
            "--agent",
            "oracle",
            "--sandbox",
            "docker",
            "--jobs-dir",
            str(jobs),
            "--retry-attempts",
            "0",
        ],
        env={**os.environ, "HARBOR_TELEMETRY": "off", "NO_COLOR": "1"},
        capture_output=True,
        text=True,
        timeout=1800,
    )
    (job_dir,) = [p for p in jobs.iterdir() if p.is_dir()]
    (trial_dir,) = [p for p in job_dir.glob("old-setpriv__*") if p.is_dir()]
    return run, trial_dir


def test_an_old_libcap_ng_setpriv_still_runs_test_sh_held(trial):
    run, trial_dir = trial
    result = json.loads((trial_dir / "result.json").read_text())
    assert result["verifier_error"] is None, (
        result["verifier_error"],
        run.stdout[-2000:],
    )
    assert result["rewards"] == {"reward": 1.0}


def test_the_fallback_launch_is_held(trial):
    """The kit Python made the same drop: the launch's own check passed and
    its record shows no NET_ADMIN, NET_RAW or SYS_PTRACE held."""
    _, trial_dir = trial
    stdout = (trial_dir / "verifier" / "test-stdout.txt").read_text()
    assert "libcap-ng is too old" not in stdout
    assert "BF verifier-launch refused" not in stdout
    record = json.loads((trial_dir / "verifier" / "verifier-network.json").read_text())
    fields = dict(f.split("=", 1) for f in record["launch"].split()[2:])
    assert fields["uid"] == "0" and fields["NoNewPrivs"] == "1", record
    held = sum(1 << c for c in (12, 13, 19))  # NET_ADMIN, NET_RAW, SYS_PTRACE
    assert int(fields["CapBnd"], 16) & held == 0, record
    assert int(fields["CapEff"], 16) & held == 0, record
