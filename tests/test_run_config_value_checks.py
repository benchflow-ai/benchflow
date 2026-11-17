"""A run config's values get the checks the same flags get.

Guards the audit fix on branch ``audit-run-config-checks``
(finding CLI-4). The flags are checked when the command line is planned; a
config file's values reached the run unchecked: ``timeout_multiplier: -1``
gave every agent a one-second limit, ``include: t-plain`` selected tasks by
the letters of the name, ``concurrency: '8'`` and ``max_retries: lots`` died
as "a bug in BenchFlow", and a misspelt key under ``retry:`` was ignored.
"""

from __future__ import annotations

import re

import pytest
import yaml

from benchflow.errors import UserError
from benchflow.evaluation import Evaluation, RunConfigValueError
from tests.test_job_matrix import _bench, _native, _plan
from tests.test_job_matrix import project as project
from tests.test_job_matrix import ran as ran


def _load(tmp_path, **settings):
    tasks = tmp_path / "tasks"
    tasks.mkdir(exist_ok=True)
    path = tmp_path / "run.yaml"
    path.write_text(
        yaml.safe_dump({"tasks_dir": str(tasks), "agent": "oracle", **settings})
    )
    return Evaluation.from_yaml(path, preflight=False, warn_default_agent=False)


REFUSED = [
    ({"include": "t-plain"}, "include"),
    ({"exclude": "t-plain"}, "exclude"),
    ({"include": [1, 2]}, "include"),
    ({"concurrency": 0}, "concurrency"),
    ({"concurrency": "8"}, "concurrency"),
    ({"concurrency": True}, "concurrency"),
    ({"build_concurrency": 0}, "build_concurrency"),
    ({"n_tasks": 0}, "n_tasks"),
    ({"max_retries": "lots"}, "max_retries"),
    ({"max_retries": -1}, "max_retries"),
    ({"warm_pool": 99}, "warm_pool"),
    ({"warm_pool": 3, "concurrency": 2}, "warm_pool"),
    ({"timeout_multiplier": -1}, "timeout_multiplier"),
    ({"timeout_multiplier": 0}, "timeout_multiplier"),
    ({"timeout_multiplier": "2"}, "timeout_multiplier"),
    ({"environment": "dockr"}, "environment"),
    ({"retry": {"max_retry": 0}}, "retry"),
    ({"retry": {"max_retries": "lots"}}, "max_retries"),
]


@pytest.mark.parametrize("settings,key", REFUSED)
def test_a_value_the_flag_refuses_is_refused_in_a_run_config(tmp_path, settings, key):
    with pytest.raises(RunConfigValueError) as refused:
        _load(tmp_path, **settings)

    assert isinstance(refused.value, UserError)
    path = re.escape(str(tmp_path / "run.yaml"))
    assert re.match(rf"{path}:\d+: {key}:", str(refused.value)), str(refused.value)
    assert refused.value.hint


def test_values_the_flags_accept_still_load(tmp_path):
    evaluation = _load(
        tmp_path,
        include=["t-plain", "hard-*"],
        exclude=["slow"],
        concurrency=8,
        build_concurrency=2,
        warm_pool=0,
        n_tasks=3,
        timeout_multiplier=1.5,
        environment="docker",
        retry={"max_retries": 0, "retry_on_infra": False},
    )
    config = evaluation._config

    assert config.include_tasks == {"t-plain", "hard-*"}
    assert config.exclude_tasks == {"slow"}
    assert (config.concurrency, config.warm_pool, config.n_tasks) == (8, 0, 3)
    assert config.timeout_multiplier == 1.5
    assert config.retry.max_retries == 0


def test_a_config_with_none_of_these_keys_loads(tmp_path):
    assert _load(tmp_path)._config.concurrency == 4


def _said(result) -> str:
    return " ".join(result.output.split())


def test_a_flag_concurrency_above_the_files_warm_pool_runs(project, ran):
    """Review of PR #136, 2026-11-13: ``warm_pool: 6`` with ``--concurrency 8``
    exited 1, the file's pool checked against the default concurrency of 4
    before the flag was on."""
    path = _native(project, "warm_pool: 6\n")

    dry = _bench("run", "--config", str(path), "--concurrency", "8", "--dry-run")
    assert dry.exit_code == 0, dry.output
    assert _plan(dry)["concurrency"] == 8
    result = _bench("run", "--config", str(path), "--concurrency", "8")
    assert result.exit_code == 0, result.output
    assert len(ran) == 1


@pytest.mark.parametrize(
    "extra,flags,limit",
    [
        ("warm_pool: 6\n", [], "concurrency (4)"),
        ("warm_pool: 6\nconcurrency: 4\n", [], "concurrency (4)"),
        ("warm_pool: 6\nconcurrency: 8\n", ["--concurrency", "4"], "--concurrency (4)"),
    ],
)
def test_a_files_warm_pool_above_the_final_concurrency_is_refused(
    project, extra, flags, limit
):
    """Review of PR #136, 2026-11-13: a warm pool above the concurrency the
    run ends up with (the file's, the default, or ``--concurrency``) is
    refused by file and line, with a next step and exit 2. ``--concurrency 4``
    under the file's ``concurrency: 8`` let ``warm_pool: 6`` through."""
    result = _bench(
        "run", "--config", str(_native(project, extra)), *flags, "--dry-run"
    )

    assert result.exit_code == 2, result.output
    said = _said(result)
    assert f"native.yaml:3: warm_pool: 6 is more than {limit}" in said
    assert "next: give warm_pool a number from 0 to 4" in said


def test_a_flag_warm_pool_is_checked_against_the_files_concurrency(project):
    """Review of PR #136, 2026-11-13, the same check from the flag's side:
    ``--warm-pool 6`` was refused against the default concurrency of 4 when
    the file sets ``concurrency: 8``. Against the file's 4 it is refused."""
    ok = _native(project, "concurrency: 8\n")
    result = _bench("run", "--config", str(ok), "--warm-pool", "6", "--dry-run")
    assert result.exit_code == 0, result.output

    low = _native(project, "concurrency: 4\n")
    result = _bench("run", "--config", str(low), "--warm-pool", "6", "--dry-run")
    assert result.exit_code == 2, result.output
    assert "native.yaml:3: --warm-pool 6 is more than concurrency (4)" in _said(result)


@pytest.mark.parametrize(
    "settings,key",
    [
        ({"include": 5}, "include"),
        ({"include": True}, "include"),
        ({"exclude": 3.5}, "exclude"),
        ({"include": {"a": 1}}, "include"),
        ({"exclude": {"slow": None}}, "exclude"),
    ],
)
def test_an_include_that_is_not_a_list_of_strings_is_refused(tmp_path, settings, key):
    """Review of PR #136, 2026-11-13: ``include: 5``, ``include: true`` and
    ``exclude: 3.5`` raised TypeError; ``include: {a: 1}`` selected task a."""
    with pytest.raises(RunConfigValueError, match=rf": {key}: .* is not a list"):
        _load(tmp_path, **settings)


@pytest.mark.parametrize("value", [3, "3", [3]])
def test_a_bare_retry_count_is_refused_with_what_to_write(tmp_path, value):
    """Review of PR #136, 2026-11-13: ``retry: 3`` loaded with the default 2
    retries and the number unread. ``retry:`` takes a table; the count
    ``--retry-attempts N`` sets is ``max_retries: N``."""
    with pytest.raises(
        RunConfigValueError, match=r": retry: .* is not a table"
    ) as refused:
        _load(tmp_path, retry=value)

    assert "max_retries: N" in (refused.value.hint or "")


@pytest.mark.parametrize(
    "extra,said",
    [
        (
            "timeout_multiplier: -1\n",
            "native.yaml:3: timeout_multiplier: -1 is not a positive number "
            "next: write timeout_multiplier: 2.0",
        ),
        (
            "include: 5\n",
            "native.yaml:3: include: 5 is not a list of task names or patterns "
            "next: give include a list of strings",
        ),
        (
            "concurrency: '8'\n",
            "native.yaml:3: concurrency: '8' is not a whole number of at least 1 "
            "next: write concurrency: 8 (a number, without quotes)",
        ),
        (
            "retry:\n  max_retry: 0\n",
            "native.yaml:4: retry: BenchFlow reads no key 'max_retry' there",
        ),
        (
            "retry: 3\n",
            "native.yaml:3: retry: 3 is not a table of retry settings "
            "next: for N retries per task write max_retries: N",
        ),
    ],
)
def test_bench_run_gives_the_line_and_the_hint_and_exits_2(project, extra, said):
    """Review of PR #136, 2026-11-13: these printed "Invalid eval config ...:
    RunConfigValueError: ..." with neither the line nor the check's own hint,
    and exited 1. A key the run config does not read is reported by file and
    line with exit 2 (``check_run_config_keys``); so are these."""
    result = _bench("run", "--config", str(_native(project, extra)), "--dry-run")

    assert result.exit_code == 2, result.output
    assert said in _said(result)


@pytest.mark.parametrize(
    "settings,hint",
    [
        ({"concurrency": "8"}, "write concurrency: 8 "),
        ({"build_concurrency": " 2 "}, "write build_concurrency: 2 "),
        ({"n_tasks": "3"}, "write n_tasks: 3 "),
        ({"max_retries": "0"}, "write max_retries: 0 "),
        ({"warm_pool": "2", "concurrency": 4}, "write warm_pool: 2 "),
        ({"retry": {"max_retries": "3"}}, "write max_retries: 3 "),
        # Not a number the key takes: the hint gives the least one.
        ({"concurrency": "0"}, "write concurrency: 1 "),
        ({"concurrency": "-8"}, "write concurrency: 1 "),
        ({"concurrency": "eight"}, "write concurrency: 1 "),
        ({"concurrency": 0}, "write concurrency: 1 "),
    ],
)
def test_a_quoted_whole_number_is_hinted_unquoted(tmp_path, settings, hint):
    """Review of PR #136, 2026-11-13: ``concurrency: '8'`` was refused with
    the hint "write concurrency: 1". A quoted whole number the key takes is
    hinted as itself without quotes (``write concurrency: 8``)."""
    with pytest.raises(RunConfigValueError) as refused:
        _load(tmp_path, **settings)

    assert (refused.value.hint or "").startswith(hint), refused.value.hint
