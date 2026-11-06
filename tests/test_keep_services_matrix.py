"""Guards DT2-R9-1/2 at 4d12b9f1: every row of DT2's retention matrix through
``bench tasks check``, a run's plan gate and setup's account resolution.

D3's rule: check refuses what a run refuses. The refusals that need the
image's accounts (an alias of root, an inherited default user) are refused at
setup only, and the rows below say so. A root oracle is exempt whatever
``--sandbox-user`` says: it runs as its own account (DT2-R9-2).
Guards DT2-R10-1 at 8d53f3f3: a named oracle account needs the image's
accounts before its UID can decide retention, including a root alias.
Guards DT2-R11-1 at b2b50251: su reads an oracle's ``[agent] user = "1000"``
as an account name, so check, the plan gate and setup must not read it as
UID 1000; setup asks the oracle's own su, and check defers every oracle.
"""

from __future__ import annotations

from typing import NamedTuple
from unittest.mock import AsyncMock, MagicMock

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.task import Task, keep_services
from tests.test_check_run_plan import _isolate, _run_refusal, _task  # noqa: F401
from tests.test_service_identity_scope import sandbox

KEEP = "[metadata]\nx-keep-services = [{port = 8080}]\n"
SIDECAR = "service 'grader'"
SPELLING = "another spelling"


class Row(NamedTuple):
    """One matrix row: the task.toml after the version line, the run's agent
    and --sandbox-user, the main image's default user, the refusal check and
    a run's plan gate give, the one setup gives, and whether setup keeps."""

    name: str
    toml: str
    agent: str = "nop"
    user: str = "agent"
    default: str = "root"
    check: str | None = None
    setup: str | None = None
    kept: bool = False
    alias_uid: int = 0


def verifier(user: str | None = None, service: str | None = None) -> str:
    lines = [] if user is None else [f"user = {user}"]
    lines += [] if service is None else [f'service = "{service}"']
    return KEEP + "[verifier]\n" + "".join(line + "\n" for line in lines)


#: Accounts whose names are digits (BusyBox and util-linux su select them by
#: name): "1000" is UID 0 beside the real UID-1000 account "agent", "0" and
#: "00" are not root. No account is named "2000".
NUMERIC_ACCOUNTS = (
    "1000:x:0:0::/root:/bin/bash\n0:x:1001:1001::/:/bin/sh\n00:x:1002:1002::/:/bin/sh\n"
)
ROOT_ORACLE = '[agent]\nuser = "root"\n'
AGENT_ORACLE = '[agent]\nuser = "agent"\n'
ROWS = [
    Row("main-unset", KEEP, kept=True),
    Row("main-root", verifier('"root"'), kept=True),
    Row("main-zero-int", verifier("0"), kept=True),
    Row("main-zero-str", verifier('"0"'), check=SPELLING, setup=SPELLING),
    Row("main-zero-pair", verifier('"0:0"'), check=SPELLING, setup=SPELLING),
    # A name, or an empty name, needs the image even on Docker: setup only.
    Row("main-root-group", verifier('"root:root"'), setup=SPELLING),
    Row("main-empty-name", verifier('":1000"'), setup=SPELLING),
    # An alias needs the image's passwd: setup refuses it, check cannot tell.
    Row("main-alias", verifier('"toor"'), setup=SPELLING),
    Row("main-nonroot", verifier('"1000"')),
    Row("grader-root", verifier('"root"', "grader"), setup=SIDECAR),
    Row("grader-zero", verifier('"0"', "grader"), check=SIDECAR, setup=SIDECAR),
    # The grader's own default user (root here) needs its image.
    Row("grader-inherited-root", verifier(None, "grader"), setup=SIDECAR),
    Row("grader-nonroot", verifier('"checker"', "grader")),
    Row("root-agent-none", verifier('"0"'), user="none"),
    Row("root-agent-named", verifier('"0"', "grader"), user="root"),
    Row(
        "oracle-root-declared", ROOT_ORACLE + verifier('"0"'), "oracle", default="agent"
    ),
    Row(
        "oracle-root-declared-grader",
        ROOT_ORACLE + verifier('"root"', "grader"),
        "oracle",
        default="agent",
    ),
    Row("oracle-image-root", verifier('"0"'), "oracle"),
    # An oracle with no account of its own runs as root whatever the image's
    # USER (setup gives the workspace and /logs/agent to the sandbox
    # user, so an image user could use neither), exempt like a root image's.
    Row("oracle-image-nonroot", verifier('"0"'), "oracle", default="agent"),
    Row(
        "oracle-agent-declared",
        AGENT_ORACLE + verifier('"0"'),
        "oracle",
        "none",
        setup=SPELLING,
    ),
    Row("oracle-agent-declared-root", AGENT_ORACLE + KEEP, "oracle", "none", kept=True),
    Row(
        "oracle-missing-account",
        '[agent]\nuser = "missing"\n' + verifier('"0"'),
        "oracle",
        "none",
        setup="cannot resolve",
    ),
    Row("no-keep", '[verifier]\nuser = "0"\nservice = "grader"\n'),
    # DT2-R10-1's own input: a root alias as the oracle's account, run with
    # --sandbox-user none. Check defers; setup finds UID 0, nothing is kept.
    Row(
        "r10-1-oracle-toor-none",
        '[agent]\nuser = "toor"\n' + verifier('"0"'),
        "oracle",
        "none",
    ),
    # DT2-R11-1's own input: "1000" names a UID-0 account under su.
    Row(
        "r11-1-oracle-numeric-root-none",
        '[agent]\nuser = "1000"\n' + verifier('"0"'),
        "oracle",
        "none",
    ),
    Row(
        "oracle-numeric-absent",
        '[agent]\nuser = "2000"\n' + verifier('"0"'),
        "oracle",
        "none",
        setup="cannot resolve",
    ),
    # A --sandbox-user that is UID 0 in the image: check counts it non-root,
    # so setup refuses it before any agent runs, whatever the verifier says.
    Row("sandbox-user-root-alias", KEEP, user="toor", setup="is UID 0"),
    Row(
        "sandbox-user-root-alias-zero",
        verifier('"0"'),
        user="toor",
        check=SPELLING,
        setup="is UID 0",
    ),
]
VERIFIER_SHAPES = (
    ("unset", None, None, None),
    ("root", '"root"', None, None),
    ("int", "0", None, None),
    ("zero", '"0"', None, SPELLING),
    ("zeros", '"000"', None, SPELLING),
    ("plus", '"+0"', None, SPELLING),
    ("minus", '"-00"', None, SPELLING),
    ("pair", '"0:0"', None, SPELLING),
    ("group", '"root:root"', None, SPELLING),
    ("empty", '":1000"', None, SPELLING),
    ("alias", '"root-verifier"', None, SPELLING),
    ("nonroot", '"1000"', None, None),
    ("grader-root", '"root"', "grader", SIDECAR),
    ("grader-zero", '"0"', "grader", SIDECAR),
    ("grader-unset", None, "grader", SIDECAR),
    ("grader-nonroot", '"checker"', "grader", None),
)
# Names that look like UIDs: whatever their digits, su gives the account's
# own UID. Check defers each one; setup decides from that UID alone.
for account, uid in (("1000", 0), ("0", 1001), ("00", 1002)):
    for user in ("none", "agent"):
        for name, selector, service, refusal in VERIFIER_SHAPES:
            ROWS.append(
                Row(
                    f"oracle-numeric-{account}-{user}-{name}",
                    f'[agent]\nuser = "{account}"\n' + verifier(selector, service),
                    "oracle",
                    user,
                    setup=refusal if uid else None,
                    kept=bool(
                        uid
                        and refusal is None
                        and name not in ("nonroot", "grader-nonroot")
                    ),
                )
            )
# Account names carry no UID information. Exercise aliases of both root and
# the sandbox UID, both CLI sandbox-user choices, and every refusal shape;
# check defers them all, but setup must still refuse the non-root cases.
for alias in ("toor", "oracle-alias"):
    for uid in (0, 1000):
        for user in ("none", "agent"):
            for name, selector, service, refusal in VERIFIER_SHAPES:
                ROWS.append(
                    Row(
                        f"oracle-{alias}-{uid}-{user}-{name}",
                        f'[agent]\nuser = "{alias}"\n' + verifier(selector, service),
                        "oracle",
                        user,
                        setup=refusal if uid else None,
                        kept=bool(
                            uid
                            and refusal is None
                            and name not in ("nonroot", "grader-nonroot")
                        ),
                        alias_uid=uid,
                    )
                )
IDS = [row.name for row in ROWS]


def _sandbox_user(value: str) -> str | None:
    return None if value == "none" else value


@pytest.mark.parametrize("command", [["tasks", "check"], ["check"]])
@pytest.mark.parametrize("row", ROWS, ids=IDS)
def test_check_and_run_plan_agree(tmp_path, row, command):
    """Guards DT2-R9-1: check prints the plan gate's own refusal, or passes."""
    task = _task(tmp_path / row.name, row.toml)
    reason = _run_refusal(
        task, tmp_path, agent=row.agent, sandbox_user=_sandbox_user(row.user)
    )
    result = CliRunner().invoke(
        app, [*command, str(task), "--agent", row.agent, "--sandbox-user", row.user]
    )
    if row.check is None:
        assert reason is None, reason
        assert result.exit_code == 0, result.output
        return
    assert reason is not None and row.check in reason
    assert keep_services.KEY in reason
    assert result.exit_code == 1, result.output
    assert reason in " ".join(result.stderr.split()) or reason in result.stderr


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["docker", "remote", "dind"])
@pytest.mark.parametrize("row", ROWS, ids=IDS)
async def test_setup_resolves_each_row(tmp_path, backend, row):
    """Guards DT2-R9-2: setup uses the account that actually runs."""
    env, accounts, configs, _ = sandbox(tmp_path, backend, grader_uid=0)
    for service in accounts:
        accounts[service] += "toor:x:0:0::/:/bin/sh\nroot-verifier:x:0:0::/:/bin/sh\n"
    # Only main's aliases change identity: the verifier's toor stays UID 0.
    accounts["main"] = accounts["main"].replace(
        "toor:x:0:0:", f"toor:x:{row.alias_uid}:0:"
    )
    accounts["main"] += f"oracle-alias:x:{row.alias_uid}:0::/:/bin/sh\n"
    accounts["main"] += NUMERIC_ACCOUNTS
    accounts["grader"] = accounts["grader"].replace("checker:x:0:", "checker:x:2000:")
    configs["a" * 64]["User"] = row.default
    configs["b" * 64]["User"] = "root"
    trial = Task(_task(tmp_path / row.name, row.toml))
    runs_as = keep_services.run_identity(trial, row.agent, _sandbox_user(row.user))
    prepare = keep_services.prepare_identities(
        env, trial, runs_as, inherit=row.agent == "oracle"
    )
    if row.setup is not None:
        with pytest.raises(RuntimeError, match=row.setup):
            await prepare
        return
    await prepare
    if runs_as is None:
        # No agent user: root's processes are never killed, so none is kept.
        assert keep_services.trial_keep(trial, runs_as) is None and not row.kept
        return
    assert (keep_services.kept_by_identity(trial) is not None) == row.kept
    if runs_as:
        assert (keep_services.trial_keep(trial, runs_as) is not None) == row.kept


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "row",
    [row for row in ROWS if row.agent == "oracle"],
    ids=[row.name for row in ROWS if row.agent == "oracle"],
)
async def test_oracle_install_resolves_the_oracles_account(tmp_path, row, monkeypatch):
    """Guards DT2-R9-2 through the real install step of an oracle run with
    the default --sandbox-user, as the reviewer's input runs it."""
    from benchflow import rollout as rollout_module
    from benchflow.rollout import Rollout, RolloutConfig

    env, accounts, configs, _ = sandbox(tmp_path, "docker", grader_uid=0)
    accounts["main"] += (
        f"toor:x:{row.alias_uid}:0::/:/bin/sh\n"
        f"oracle-alias:x:{row.alias_uid}:0::/:/bin/sh\n"
        "root-verifier:x:0:0::/:/bin/sh\n" + NUMERIC_ACCOUNTS
    )
    accounts["grader"] += "toor:x:0:0::/:/bin/sh\n"
    accounts["grader"] = accounts["grader"].replace("checker:x:0:", "checker:x:2000:")
    configs["a" * 64]["User"] = row.default
    configs["b" * 64]["User"] = "root"
    task_dir = _task(tmp_path / row.name, row.toml)
    monkeypatch.setattr(
        rollout_module, "_resolve_agent_cwd", AsyncMock(return_value="/app")
    )
    monkeypatch.setattr(rollout_module, "prepare_capture_runtime", AsyncMock())
    rollout = Rollout(
        RolloutConfig(
            task_path=task_dir,
            agent="oracle",
            sandbox_user=_sandbox_user(row.user),
            jobs_dir=tmp_path / "jobs",
            planes=MagicMock(),
        )
    )
    rollout._task = Task(task_dir)
    rollout._env = env
    rollout._rollout_dir = tmp_path / "rollout"
    rollout._rollout_dir.mkdir()
    planes = MagicMock()
    for step in (
        "setup_sandbox_user",
        "prepare_log_dirs",
        "snapshot_build_config",
        "seed_verifier_workspace",
        "prepare_verifier_runtime",
        "deploy_skills",
        "lockdown_paths",
        # K1: a trial that keeps services proves the freezer before solve.sh.
        "probe_kept_service_freezer",
    ):
        setattr(planes, step, AsyncMock(return_value="/app"))
    rollout._planes = planes
    if row.setup is not None:
        with pytest.raises(RuntimeError, match=row.setup):
            await rollout.install_agent()
        return
    await rollout.install_agent()
    # The turn-end kill asks with --sandbox-user's name (root when none).
    kept_now = keep_services.trial_keep(
        rollout._task, _sandbox_user(row.user) or "root"
    )
    assert (kept_now is not None) == row.kept
