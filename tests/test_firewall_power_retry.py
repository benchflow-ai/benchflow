"""Guard S3-CONF-1, the unrestricted retry introduced by e4108744.

The transport choice fixes the launch policy for the entire audit, including
missing reports, failed drops, retained state and reconnects.
"""

from types import SimpleNamespace

import pytest

from benchflow.sandbox.firewall_power import check_firewall_power, net_admin_dropped
from benchflow.sandbox.lockdown import build_priv_drop_cmd
from tests._receipt_fakes import receipt_aware

ZERO = "0000000000000000"
SAFE = f"bnd={ZERO} inh={ZERO} amb={ZERO} prm={ZERO} eff={ZERO} nnp=1 power="
ROOT = "BF fwpower-root net_admin=1 setpriv=ok nnp=1\n"
SCAN = "BF fwpower-scan scope=path caps=getcap\n"
OUTCOMES = {
    "safe": "BF fwpower-agent " + SAFE,
    "empty": "",
    "partial": "BF fwpower-agent nnp=1",
    "missing-nnp": "BF fwpower-agent " + SAFE.replace("nnp=1", ""),
    "nnp-zero": "BF fwpower-agent " + SAFE.replace("nnp=1", "nnp=0"),
    "permitted": "BF fwpower-agent " + SAFE.replace(f"prm={ZERO}", "prm=1000"),
    "effective": "BF fwpower-agent " + SAFE.replace(f"eff={ZERO}", "eff=1000"),
    "ambient": "BF fwpower-agent " + SAFE.replace(f"amb={ZERO}", "amb=1000"),
    "power": "BF fwpower-agent " + SAFE + "nft",
    "exception": RuntimeError("probe transport failed"),
    "timeout": TimeoutError("probe timed out"),
}


class AuditEnv:
    supports_system_shell = True

    def __init__(self, outcome, *, root=ROOT, keep=True):
        self.outcome = outcome
        self.root = root
        self.keep = keep
        self.calls = []

    def __setattr__(self, name, value):
        if name == "_benchflow_net_admin_dropped_for" and not self.keep:
            raise AttributeError("transport cannot retain launch state")
        super().__setattr__(name, value)

    @receipt_aware
    async def exec(self, command, *, user, **kwargs):
        self.calls.append((command, user))
        if "BF fwpower-root" in command:
            out = self.root + SCAN if self.root else ""
        elif "BF fwpower-agent" in command:
            if isinstance(self.outcome, Exception):
                raise self.outcome
            out = self.outcome
        else:
            out = SCAN.replace("scope=path", "scope=all")
        return SimpleNamespace(stdout=out, stderr="", return_code=0)

    def probes(self):
        return [
            (command, user)
            for command, user in self.calls
            if "BF fwpower-agent AUDIT" in command
        ]


@pytest.mark.parametrize("outcome", OUTCOMES)
@pytest.mark.parametrize("saved_drop", [False, True])
@pytest.mark.parametrize("user", ["agent", "1000", "worker-name"])
async def test_shared_audit_never_retries_without_the_barrier(
    monkeypatch, outcome, saved_drop, user
):
    """Guards S3-CONF-1 at c7b94d56 on initial connection and repeated checks."""
    # Replace only the probe body, never the actual production launcher.
    # Equality checks both branches inside build_priv_drop_cmd, not a flag
    # mentioned in a diagnostic or in an unreachable shell branch.
    from benchflow.sandbox import firewall_power

    body = "printf 'BF fwpower-agent AUDIT\\n'"
    monkeypatch.setattr(firewall_power, "_agent_check_cmd", lambda *args: body)
    env = AuditEnv(OUTCOMES[outcome])
    if saved_drop:
        env._benchflow_net_admin_dropped_for = user
    for _ in range(2):
        env.calls.clear()
        record, refusal = await check_firewall_power(env, user, holds=("denylist",))
        expected = firewall_power._root_route(
            build_priv_drop_cmd(body, user, drop_net_admin=True, root_tools=True),
            required=firewall_power._ROOT_LAUNCH_TOOLS,
            optional=("su",),
        )
        assert env.probes() == [(expected, "root")]
        if outcome == "safe":
            assert refusal is None and record["state"] == "disarmed"
            assert net_admin_dropped(env, user)
        else:
            assert refusal and "task's image" in refusal
            assert record["state"] == "refused"
            assert not net_admin_dropped(env, user)


@pytest.mark.parametrize(
    "root",
    [
        "",
        ROOT.replace("setpriv=ok nnp=1", "setpriv=none nnp=0"),
        ROOT.replace("setpriv=ok nnp=1", "setpriv=no-bounding-set nnp=0"),
    ],
)
async def test_missing_shared_launch_prerequisites_never_run_an_identity_probe(root):
    """Guards S3-CONF-1's sibling path before the restricted audit starts."""
    env = AuditEnv(OUTCOMES["safe"], root=root)
    record, refusal = await check_firewall_power(env, "agent", holds=("denylist",))
    assert not any("BF fwpower-agent" in command for command, _ in env.calls)
    assert record["state"] == "refused" and refusal
    assert "util-linux setpriv" in refusal and "task's image" in refusal


async def test_failure_to_retain_drop_state_never_retries_unwrapped():
    """Guards S3-CONF-1 when the restricted probe works but its state cannot stick."""
    env = AuditEnv(OUTCOMES["safe"], keep=False)
    record, refusal = await check_firewall_power(env, "agent", holds=("denylist",))
    probes = [(c, u) for c, u in env.calls if "BF fwpower-agent" in c]
    assert len(probes) == 1 and probes[0][1] == "root"
    assert "--no-new-privs --bounding-set=-net_admin,-net_raw" in probes[0][0]
    assert record["state"] == "refused" and refusal


@pytest.mark.parametrize("outcome", OUTCOMES)
async def test_only_session_factory_audits_use_the_unwrapped_identity(outcome):
    """Guards e4108744's necessary identity audit while fixing S3-CONF-1."""
    env = AuditEnv(OUTCOMES[outcome])
    await check_firewall_power(env, "agent", holds=("denylist",), can_drop=False)
    probes = [(c, u) for c, u in env.calls if "BF fwpower-agent" in c]
    assert len(probes) == 1 and probes[0][1] == "agent"
    assert "--no-new-privs" not in probes[0][0]


@pytest.mark.parametrize("mode", ["denylist", "allowlist", "no-network", "open"])
async def test_honest_tasks_still_connect_and_reconnect(tmp_path, mode):
    """Guards S3-CONF-1 without refusing honest shared-launch transports."""
    from tests.test_firewall_power import FixtureEnv, _connect, _rollout

    env = FixtureEnv("cap", mode=mode)
    trial = _rollout(tmp_path, env, mode)
    assert await _connect(trial) == ["agent drop=agent"]
    trial._phase = "idle"
    assert await _connect(trial) == ["agent drop=agent"]
    probes = env.commands("BF fwpower-agent")
    assert len(probes) == 2
    assert all(user == "root" for _, user in probes)
    assert all(
        "--no-new-privs --bounding-set=-net_admin,-net_raw" in c for c, _ in probes
    )
