"""Guards DT2-CONF-5/6 at d70d1336: exact paths and root UID spellings."""

import posixpath
import shlex
from types import SimpleNamespace

import pytest

from benchflow.task import keep_services
from tests.test_script_agent_cleanup import (
    BACKENDS,
    PYTHON_IMAGE,
    SYSTEM_DIRS,
    service_rollout,
)
from tests.test_script_agent_docker import (
    BUSYBOX_IMAGE,
    BUSYBOX_SETUP,
    IMAGE,
    UBUNTU_SETUP,
    run_scored,
    started_box,
)

ODD_NAMES = [
    "tool\n",
    "tool\n\n",
    "tool \n",
    "tool\npart",
    "tool\t",
    "tool\r",
    "tool *?[]'$()`;",
]
ROOT_USERS = [
    None,
    0,
    "root",
    "0",
    "00",
    "000",
    "root:root",
    "root:0",
    "root:agent",
    "0:0",
    "0:root",
    "00:agent",
    "+0",
    "-0",
    "+00",
    "-00",
    "+0:0",
    "-0:root",
    "0:",
    "",
    ":",
    ":0",
    ":root",
    ":1000",
]
NONROOT_USERS = [
    "agent",
    "1000",
    "1000:0",
    "agent:root",
    "10",
    "01",
    " 0",
    "+1",
    "-1",
    "+-0",
    "--0",
    "0x0",
    "0.0",
    "0e0",
    "\u0660",
    False,
]


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
async def test_readlink_failure_refuses_runner(tmp_path, backend):
    """Guards DT2-CONF-5: sentinel capture must preserve readlink failure."""
    with started_box() as initial:
        box = backend(initial.name)
        assert (
            await box.exec(
                "printf '#!/bin/sh\\nexit 1\\n' > /usr/bin/readlink; chmod 755 /usr/bin/readlink"
            )
        ).return_code == 0
        result = await run_scored(box, tmp_path, "printf ran > /app/ran\n")
        assert "before the script started" in (result.error or "")
        assert result.rewards is None
        assert (await box.exec("test -e /app/ran")).return_code == 1


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("directory", SYSTEM_DIRS)
@pytest.mark.parametrize(
    "route", ["absolute", "relative", "ancestor", "dot", "parent", "chain"]
)
@pytest.mark.parametrize("name", ODD_NAMES)
async def test_cleanup_checks_exact_link_bytes(
    tmp_path, backend, directory, route, name
):
    """Guards DT2-CONF-5: all legal filename bytes reach the permission check."""
    with started_box() as initial:
        box = backend(initial.name)
        folder = "/opt/helpers"
        target = f"{folder}/{name}"
        setup = f"mkdir -p {shlex.quote(folder)}; "
        if route == "ancestor":
            folder = target
            target += "/tool"
            setup += f"mkdir {shlex.quote(folder)}; chown agent:agent {shlex.quote(folder)}; "
        setup += f"cp /usr/bin/true {shlex.quote(target)}; "
        if route != "ancestor":
            setup += f"chown agent:agent {shlex.quote(target)}; "
        link = f"{directory}/dt2-exact-helper"
        physical = await box.exec(f"readlink -f -- {shlex.quote(directory)}")
        assert physical.return_code == 0
        destination = (
            target
            if route != "relative"
            else posixpath.relpath(target, physical.stdout.strip())
        )
        if route == "dot":
            destination = "/opt//helpers/./" + name
        elif route == "parent":
            destination = "/opt/helpers/../helpers/" + name
        elif route == "chain":
            setup += f"ln -s {shlex.quote(target)} /opt/dt2-chain; "
            destination = "/opt/dt2-chain"
        setup += f"ln -s {shlex.quote(destination)} {shlex.quote(link)}; test {shlex.quote(link)} -ef {shlex.quote(target)}"
        assert (await box.exec(setup)).return_code == 0
        result = await run_scored(box, tmp_path, "printf ran > /app/ran\n")
        assert "before the script started" in (result.error or "")
        assert result.rewards is None
        assert (await box.exec("test -e /app/ran")).return_code == 1


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "image,setup", [(IMAGE, UBUNTU_SETUP), (BUSYBOX_IMAGE, BUSYBOX_SETUP)]
)
@pytest.mark.parametrize("name", ODD_NAMES)
async def test_protected_odd_link_targets_still_run(
    tmp_path, backend, image, setup, name
):
    """Guards DT2-CONF-5: GNU and BusyBox preserve safe path bytes too."""
    with started_box(image=image, setup=setup) as initial:
        box = backend(initial.name)
        target = "/opt/" + name
        assert (
            await box.exec(
                f"cp /bin/true {shlex.quote(target)}; ln -s {shlex.quote(target)} /usr/local/bin/dt2-exact-helper"
            )
        ).return_code == 0
        result = await run_scored(box, tmp_path, "printf ran > /app/ran\n")
        assert result.error is None and result.rewards == {"reward": 0.0}
        assert (await box.exec("test -f /app/ran")).return_code == 0


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("helper", ["setpriv", "grep"])
@pytest.mark.parametrize("suffix", ["\n", "\n\n"])
async def test_newline_cleanup_helper_cannot_disclose_marker(
    tmp_path, backend, helper, suffix
):
    """Guards DT2-CONF-5: genuine helpers become hostile only after agent start."""
    with started_box() as initial:
        box = backend(initial.name)
        target = f"/opt/helpers/{helper}{suffix}"
        setup = (
            f"mkdir -p /opt/helpers; cp /usr/bin/{helper} {shlex.quote(target)}; "
            f"chown agent:agent {shlex.quote(target)}; "
            f"ln -s {shlex.quote(target)} /usr/local/bin/{helper}; "
            "printf dummy-marker > /root/dt2-marker; chmod 600 /root/dt2-marker"
        )
        assert (await box.exec(setup)).return_code == 0
        replacement = (
            "#!/bin/sh\ncat /root/dt2-marker > /app/leaked\n"
            + f'exec /usr/bin/{helper} "$@"\n'
        )
        body = (
            f"printf %s {shlex.quote(replacement)} > {shlex.quote(target)}; sleep 60\n"
        )
        result = await run_scored(box, tmp_path, body, timeout=1)
        assert (await box.exec("test -s /app/leaked")).return_code == 1, (
            "root cleanup disclosed dummy marker"
        )
        assert "before the script started" in (result.error or "")
        assert result.rewards is None


@pytest.mark.parametrize("verifier_user", ROOT_USERS + NONROOT_USERS)
@pytest.mark.parametrize("sandbox_user", ["agent", "1000", "1000:0", *ROOT_USERS])
@pytest.mark.parametrize("separate", [False, True])
def test_shared_retention_policy_uses_uid(verifier_user, sandbox_user, separate):
    """Guards DT2-CONF-6 at d70d1336 for every shared retention consumer."""
    task = SimpleNamespace(
        config=SimpleNamespace(
            verifier=SimpleNamespace(
                user=verifier_user,
                sandbox_mode="separate" if separate else None,
                sandbox=None,
            )
        )
    )
    expected = keep_services.KeptServices(
        (keep_services.KeepService(8080),), "the task"
    )
    # A cached policy isolates identity handling from metadata/digest parsing.
    task._benchflow_kept_services = expected
    from benchflow.sandbox._identities import Identities, resolve
    from tests.test_script_agent import ACCOUNT_FILES

    passwd, group = ACCOUNT_FILES.split("\0")
    table = Identities(passwd, group)
    if sandbox_user and not separate:
        try:
            task._benchflow_service_identities = (
                resolve(table, sandbox_user),
                resolve(table, verifier_user),
            )
        except RuntimeError:
            with pytest.raises(RuntimeError, match="resolve"):
                keep_services.trial_keep(task, sandbox_user)
            return
    root_verifier = any(
        type(verifier_user) is type(root) and verifier_user == root
        for root in ROOT_USERS
    )
    keeps = sandbox_user not in ROOT_USERS and root_verifier and not separate
    # Root is kept beside only as root, 0 or unset; other spellings of UID 0
    # are refused rather than matched (the narrowed matrix in DT2.md).
    canonical = verifier_user in (None, "", "root") or (
        type(verifier_user) is int and verifier_user == 0
    )
    if keeps and not canonical:
        with pytest.raises(RuntimeError, match="another spelling"):
            keep_services.trial_keep(task, sandbox_user)
        return
    assert keep_services.trial_keep(task, sandbox_user) is (expected if keeps else None)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("verifier_user", ["root", None])
@pytest.mark.parametrize("failure", ["watchdog", "transport", "timeout_error"])
async def test_root_uid_timeout_keeps_listener(
    tmp_path, backend, verifier_user, failure
):
    """Guards DT2-CONF-6: production turn-end freeze, release, scoring and end."""
    with started_box(image=PYTHON_IMAGE) as initial:
        box = backend(initial.name)
        result, observed = await service_rollout(
            box, tmp_path, keep=True, failure=failure, verifier_user=verifier_user
        )
        assert result.error is None and result.rewards == {"reward": 1.0}
        assert result.trajectory[0]["timed_out"] is True
        assert observed["waited"] and observed["frozen"]
        assert observed["released"] == 1
        assert observed["record"]["kept"][0]["port"] == 8080
        assert observed["stray_ended"] and observed["ended"]


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "verifier_user", ["0", "00", "root:agent", "0:0", "+0", "-00:0", ":1000"]
)
async def test_other_root_spellings_refused_before_script(
    tmp_path, backend, verifier_user
):
    """Root by another spelling is refused before the script, never scored."""
    with started_box(image=PYTHON_IMAGE) as initial:
        box = backend(initial.name)
        identity = await box.exec("id -u", user=verifier_user)
        assert identity.return_code == 0 and identity.stdout.strip() == "0"
        result, observed = await service_rollout(
            box, tmp_path, keep=True, verifier_user=verifier_user
        )
        assert "another spelling" in (result.error or "")
        assert result.rewards is None
        assert not observed.get("waited") and not observed.get("released")
