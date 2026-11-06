"""FIX-49 r4 audit: agent code run by test.sh cannot make a trial unscored.

All attack cases fail at c278f22a6. Each oracle writes the program that the
task's test.sh executes as the verifier user; no host result is forged.
"""

import json

import pytest

from tests.exec_diet_fakes import docker_available
from tests.test_acc_run2_docker import BASE, _run, _scored, _task

pytestmark = [pytest.mark.deterministic, pytest.mark.verifier_runtime]


@pytest.fixture(autouse=True)
def needs_docker():
    if not docker_available():
        pytest.skip("needs a Docker daemon")


PRELUDE = """import hashlib, hmac, json, os, pathlib, shlex, time
args = shlex.split(os.environ['PYTEST_ADDOPTS'])
guard = next(n for f, n in zip(args, args[1:]) if f == '-p' and n.startswith('_runtime_guard_'))
def mark(kind):
    root = pathlib.Path('/') / guard
    name = guard + '.1-deadbeef.' + kind
    body = json.dumps('written by code test.sh ran')
    mac = hmac.new((root / '.key').read_bytes().strip(), (name + '\\n' + body).encode(), hashlib.sha256).hexdigest()
    (root / '.markers' / name).write_text(mac + '\\t' + body + '\\n')
"""


@pytest.mark.parametrize(
    "attack,reason",
    [
        (
            "print(f'Error importing plugin \"{guard}\"', flush=True)",
            "guard not confirmed",
        ),
        (
            'mark("registered")\nprint(f\'Error importing plugin "{guard}"\', flush=True)',
            "guard not confirmed",
        ),
        ('mark("checking")', "guard not confirmed"),
        ('mark("crashed")', "guard not confirmed"),
        ("time.sleep(40)", "timed out"),
        (
            '[p.unlink() for p in pathlib.Path("/run/runtime").glob("verifier-started-*")]\ntime.sleep(40)',
            "Verifier scored 0",
        ),
    ],
    ids=[
        "guard-text",
        "forged-registered",
        "checking",
        "crashed",
        "timeout",
        "deleted-receipt",
    ],
)
async def test_verifier_user_cannot_drop_the_trial(tmp_path, attack, reason):
    task = _task(
        tmp_path,
        "fix49-r4-evidence",
        dockerfile=BASE,
        solve="cat > /app/check.py <<'PY'\n"
        + PRELUDE
        + attack
        + "\nraise SystemExit(1)\nPY\n",
        test=(
            "#!/bin/bash\ncd /app\n"
            "if python3 check.py; then echo 1; else echo 0; fi > /logs/verifier/reward.txt\n"
        ),
    )
    if "sleep" in attack:
        file = task / "task.toml"
        file.write_text(
            file.read_text().replace("timeout_sec = 300.0", "timeout_sec = 10.0", 1)
        )
    result = await _run(task, "oracle")
    _scored(result, 0.0)
    warnings = json.loads((result.rollout_dir / "result.json").read_text()).get(
        "warnings", []
    )
    assert any(reason in warning for warning in warnings), warnings


@pytest.mark.parametrize(
    "attack",
    [
        'pathlib.Path("/logs/verifier/reward.json").write_text(\'{"reward": "x"}\')',
        'pathlib.Path("/logs/verifier/reward.json").mkdir()',
    ],
    ids=["malformed-json", "directory-json"],
)
async def test_a_reward_file_test_sh_breaks_is_an_error_that_still_fails(
    tmp_path, attack
):
    """A reward.json the code test.sh ran broke is no usable reward: the
    verifier's error, never retried, and still a failure in every
    score, so the trial is not dropped."""
    from benchflow._utils.scoring import (
        NO_USABLE_REWARD,
        VERIFIER_FAILED,
        classify_verifier_error,
        score_summary_fields,
    )

    task = _task(
        tmp_path,
        "fix49-r4-evidence",
        dockerfile=BASE,
        solve="cat > /app/check.py <<'PY'\n"
        + PRELUDE
        + attack
        + "\nraise SystemExit(1)\nPY\n",
        test=(
            "#!/bin/bash\ncd /app\n"
            "if python3 check.py; then echo 1; else echo 0; fi > /logs/verifier/reward.txt\n"
        ),
    )
    result = await _run(task, "oracle")
    assert result.error is None, result.error
    assert result.rewards is None
    assert result.verifier_error.startswith(NO_USABLE_REWARD), result.verifier_error
    assert "reward.json" in result.verifier_error
    assert classify_verifier_error(result.verifier_error) == VERIFIER_FAILED
    row = json.loads((result.rollout_dir / "result.json").read_text())
    summary = score_summary_fields([row])
    assert (summary["unusable_rewards"], summary["score_excl_errors_ratio"]) == (1, 0.0)
