"""Guards DT2-CONF-7 at b891c49a: policy follows effective identities."""

import pytest

from tests.test_script_agent_cleanup import BACKENDS, PYTHON_IMAGE, service_rollout
from tests.test_script_agent_docker import started_box


@pytest.mark.parametrize("user", ["root", "toor", "0", "000", "+00", "-00"])
@pytest.mark.parametrize("group", ["", ":root", ":wheel", ":0", ":000"])
def test_account_aliases_resolve_to_numeric_pair(user, group):
    """Guards CONF-7: every root name/number/group spelling shares numbers."""
    from benchflow.sandbox._identities import Identities

    table = Identities(
        "root:x:0:0::/:/bin/sh\ntoor:x:0:0::/:/bin/sh\n",
        "root:x:0:\nwheel:x:0:\n",
    )
    identity = table.resolve(user + group)
    assert (identity.uid, identity.gid) == (0, 0)


@pytest.mark.parametrize("user", [None, "", "grader", "2000", "002000"])
@pytest.mark.parametrize("default", ["grader", "2000:3000", "002000:003000"])
def test_default_user_and_nonmatching_gid_are_preserved(user, default):
    """Guards CONF-7: the primary GID is independent of the numeric UID."""
    from benchflow.sandbox._identities import Identities

    table = Identities("grader:x:2000:3000::/:/bin/sh\n", "grader:x:3000:\n", default)
    identity = table.resolve(user)
    assert (identity.uid, identity.gid) == (2000, 3000)


@pytest.mark.parametrize(
    "user",
    [
        "root ",
        "root\n",
        "0\n",
        "000\t",
        " 0",
        "toor\r",
        "2000 ",
        "0:root\n",
        "0:missing",
        "missing",
        "--0",
        "+-0",
        "0:0:0",
        False,
    ],
)
def test_unresolved_identity_fails_with_next_step(user):
    """Guards CONF-7: unresolved/whitespace identities never become root."""
    from benchflow.sandbox._identities import Identities, resolve

    table = Identities("root:x:0:0::/:/bin/sh\n", "root:x:0:\n")
    with pytest.raises(RuntimeError, match=r"use an existing.*without whitespace"):
        resolve(table, user)


@pytest.mark.asyncio
async def test_snapshot_is_shared_by_cleanup_and_retention():
    """Guards CONF-7: later account changes cannot change service ownership."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from benchflow.sandbox._identities import kill_prelude, snapshot
    from benchflow.sandbox.lockdown import _KILL_PRELUDE
    from tests.receipt_fakes import receipt_answer

    accounts = SimpleNamespace(
        return_code=0,
        stdout="agent:x:1000:3000::/:/bin/sh\n\0agent:x:3000:\n",
    )
    # The account read crosses as a receipt (its NUL separator).
    root = AsyncMock(
        side_effect=lambda command, **kwargs: receipt_answer(
            command, lambda read: accounts
        )
    )
    env = SimpleNamespace()
    table = await snapshot(env, root_exec=root, config={"User": "agent"})
    first = table.resolve("agent")
    assert await snapshot(env, root_exec=root, config={"User": "0"}) is table
    assert table.resolve(None) == first
    script = kill_prelude(env, _KILL_PRELUDE, "agent")
    assert "uid=1000\ngid=3000\n" in script
    assert "uid=$(id -u" not in script and "gid=$(id -g" not in script
    root.assert_awaited_once()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("verifier_user", ["toor", "toor:root", "toor:000"])
async def test_root_account_alias_is_refused(tmp_path, backend, verifier_user):
    """Guards CONF-7: a trusted passwd alias of UID 0 is root, and retention
    beside it is refused (write "root") rather than matched by spelling."""
    with started_box(
        image=PYTHON_IMAGE,
        setup="useradd -m -s /bin/bash agent; printf 'toor:x:0:0::/root:/bin/bash\\n' >> /etc/passwd",
    ) as initial:
        box = backend(initial.name)
        result, observed = await service_rollout(
            box, tmp_path, keep=True, verifier_user=verifier_user
        )
        assert "another spelling" in (result.error or "")
        assert result.rewards is None and not observed


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
async def test_empty_verifier_inherits_nonroot_container_user(tmp_path, backend):
    """Guards CONF-7: Docker's empty selector inherits Config.User."""
    with started_box(
        image=PYTHON_IMAGE,
        setup="useradd -m -s /bin/bash agent; useradd -u 2000 -m grader",
        run_flags=("--user", "2000:2000"),
    ) as initial:
        box = backend(initial.name)
        assert (
            await box.exec("mkdir -p /logs/verifier; chmod 777 /logs/verifier")
        ).return_code == 0
        result, observed = await service_rollout(
            box, tmp_path, keep=True, verifier_user=""
        )
        assert result.error is None
        assert not observed["waited"] and observed.get("released", 0) == 0
        assert (await box.exec("test -x /logs", user="2000:2000")).return_code == 0
        assert (
            await box.exec("printf 0 > /logs/verifier/reward.txt", user="2000:2000")
        ).return_code == 0
