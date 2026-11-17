"""Root is refused the cloud metadata endpoints from the end of the agent's
turn through the verifier (EN-609).

The agent's block refused them to non-root users only, so test.sh, which runs
as root and runs the agent's code, reached a GCE VM's metadata server under
every network policy. Hardening now adds the rules again with root covered,
from BenchFlow's metadata helper (else as root in main), and proves them with
root's own refused connect counted by the rule; only the task's own opt-out
leaves the verifier the endpoints. The deterministic tier runs it in a real
sandbox (test_deterministic_integration's metadata scenario).
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from benchflow._utils import timing_spans
from benchflow.sandbox import lockdown
from benchflow.sandbox.cloud_metadata import CLOUD_METADATA_V4, CLOUD_METADATA_V6
from tests._receipt_fakes import inner_command, receipt_answer

ROOT_OK = (
    "BF metadata status=0 fw4=iptables fw6=ip6tables v6=local install=none "
    "install_t=- hits=0\n"
)


class HelperEnv:
    """A sandbox with BenchFlow's metadata helper: root's script and the counter
    run in the helper, root's test connect in main."""

    def __init__(
        self,
        *,
        install: str | None = ROOT_OK,
        install_failed: str = "",
        connect: str = "refused",
        hits: tuple[int, int] = (3, 4),
        main_script: str = "",
    ) -> None:
        self.install_out = install
        self.install_failed = install_failed
        self.connect = connect
        self.hits = list(hits)
        self.main_script = main_script
        self.installed: list[str] = []
        self.main_commands: list[tuple[str, str | None]] = []
        self.closed = 0

    async def metadata_rules_install(self, command: str):
        self.installed.append(command)
        if self.install_out is None:
            return None
        return SimpleNamespace(
            stdout=self.install_out, return_code=0
        ), self.install_failed

    async def metadata_rules_run(self, command: str, *, timeout_sec: int, user=None):
        assert "BF metadata-hits" in command
        return SimpleNamespace(
            stdout=f"BF metadata-hits {self.hits.pop(0)}\n", return_code=0
        ), ""

    def metadata_rules_close(self) -> None:
        self.closed += 1

    async def exec(self, command: str, user: str | None = None, **_: object):
        return receipt_answer(command, self.answer(inner_command(command), user))

    def answer(self, command: str, user: str | None):
        if "BF verifier-network routes" in command:
            # main has a route out (not network: none).
            return SimpleNamespace(
                stdout="BF verifier-network routes 1\n", stderr="", return_code=0
            )
        self.main_commands.append((command, user))
        if "BF metadata-connect" in command:
            assert user == "root"
            return SimpleNamespace(
                stdout=f"BF metadata-connect {self.connect}\n", stderr="", return_code=0
            )
        if "BF metadata-hits" in command:
            return SimpleNamespace(
                stdout=f"BF metadata-hits {self.hits.pop(0)}\n",
                stderr="",
                return_code=0,
            )
        return SimpleNamespace(stdout=self.main_script, stderr="", return_code=0)


def _task(opt_out: str = "", environment_dir=None):
    return SimpleNamespace(
        config=SimpleNamespace(cloud_metadata_opt_out=opt_out),
        paths=SimpleNamespace(environment_dir=environment_dir),
    )


def test_the_rules_cover_root():
    script = lockdown._cloud_metadata_block_cmd(cover_root=True)
    assert "--uid-owner" not in script
    for address in (*CLOUD_METADATA_V4, *CLOUD_METADATA_V6):
        # Every user, root included, on every port but DNS.
        assert f"-d {address} -p tcp ! --dport 53 -j REJECT" in script
    assert "-m owner ! --uid-owner 0" in lockdown._cloud_metadata_rule(
        "169.254.169.254"
    )


@pytest.mark.asyncio
async def test_root_is_refused_from_the_helper_and_the_rule_counts_it():
    env = HelperEnv()
    record = await lockdown.cover_cloud_metadata_for_verifier(env, _task())
    assert record == {
        "state": "on",
        "why": "root's test connect was refused and the rule counted it",
        "rules_from": "helper",
    }
    assert env.installed == [lockdown._cloud_metadata_block_cmd(cover_root=True)]
    # Nothing is installed in main; root's own connect is made there.
    assert [u for c, u in env.main_commands] == ["root"]
    assert "BF metadata-connect" in env.main_commands[0][0]
    assert env.closed == 1


@pytest.mark.asyncio
async def test_the_task_opt_out_alone_leaves_the_endpoints_open():
    env = HelperEnv()
    record = await lockdown.cover_cloud_metadata_for_verifier(
        env, _task("allow_cloud_metadata = true")
    )
    assert record == {
        "state": "off",
        "why": "the task opts out (allow_cloud_metadata = true)",
    }
    assert env.installed == [] and env.main_commands == [] and env.closed == 0


@pytest.mark.asyncio
async def test_a_main_with_no_route_out_needs_no_rule():
    """network: none (an oracle's or a script's offline run): nothing to
    refuse, and test.sh runs (the verifier network's proof shows the kernel
    has no route)."""

    class Offline(HelperEnv):
        def answer(self, command: str, user: str | None):
            if "BF verifier-network routes" in command:
                return SimpleNamespace(
                    stdout="BF verifier-network routes 0\n", stderr="", return_code=0
                )
            return super().answer(command, user)

    env = Offline()
    record = await lockdown.cover_cloud_metadata_for_verifier(env, _task())
    assert record == {"state": "not_reachable", "why": "main has no route out"}
    assert env.installed == [] and env.main_commands == []


@pytest.mark.asyncio
async def test_a_sandbox_without_the_helper_is_left_to_its_provider():
    class Provider:
        async def exec(self, *a, **k):  # pragma: no cover - must not run
            raise AssertionError("nothing runs on a provider's sandbox")

    assert await lockdown.cover_cloud_metadata_for_verifier(Provider(), _task()) is None


@pytest.mark.asyncio
async def test_without_the_helper_root_installs_in_main():
    env = HelperEnv(install=None, main_script=ROOT_OK)
    record = await lockdown.cover_cloud_metadata_for_verifier(env, _task())
    assert record["rules_from"] == "sandbox"
    script = lockdown._cloud_metadata_block_cmd(cover_root=True)
    assert env.main_commands[0][0] == script
    assert all(u == "root" for _, u in env.main_commands)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "why"),
    [
        # The helper could not add the rules, and main's root has no NET_ADMIN.
        (
            {
                "install": "BF metadata status=1 fw4=iptables fw6=ip6tables\n",
                "main_script": "BF metadata status=4 fw4=- fw6=-\n",
            },
            "gives root no NET_ADMIN",
        ),
        # Root's connect got through.
        ({"connect": "reached"}, "was reached and the rule counted 1"),
        # Something else refused it (a host's own block): the rule never saw it.
        ({"hits": (5, 5)}, "was refused and the rule counted 0"),
    ],
)
async def test_test_sh_does_not_run_unless_root_is_proven_refused(kwargs, why):
    env = HelperEnv(**kwargs)
    with pytest.raises(
        RuntimeError,
        match=r"Verifier hardening failed: the cloud metadata endpoints could not be refused to root before test\.sh",
    ) as raised:
        await lockdown.cover_cloud_metadata_for_verifier(env, _task())
    assert why in str(raised.value), str(raised.value)
    assert "allow_cloud_metadata = true" in str(raised.value)
    assert env.closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["container:other", "service:gone"])
async def test_a_main_on_an_outside_network_gets_no_rule_and_no_test_sh(tmp_path, mode):
    """The helper would add the rules to that namespace: the host's own
    firewall for ``host``. Nothing is installed, and test.sh does not run."""
    (tmp_path / "docker-compose.yaml").write_text(
        f'services:\n  main:\n    network_mode: "{mode}"\n  sidecar:\n    image: x\n'
    )
    env = HelperEnv()
    with pytest.raises(
        RuntimeError,
        match=r"could not be refused to root before test\.sh \(main shares the host's network or another container's \(network_mode: (container:other|service:gone) in the task's docker-compose\.yaml\)",
    ):
        await lockdown.cover_cloud_metadata_for_verifier(
            env, _task(environment_dir=tmp_path)
        )
    assert env.installed == [] and env.main_commands == []


@pytest.mark.asyncio
async def test_a_main_that_could_restart_gets_no_rule_and_no_test_sh():
    class Restarts(HelperEnv):
        async def metadata_main_restarts(self) -> str:
            return "main's restart policy is always"

    env = Restarts()
    with pytest.raises(RuntimeError, match="restart policy is always, so no rule"):
        await lockdown.cover_cloud_metadata_for_verifier(env, _task())
    assert env.installed == []


def test_hardening_refuses_root_before_anything_else_runs_as_root():
    """The step follows the kill of the agent's processes and comes before
    every later step (0.9.x has no delegate-disarm restore between them)."""
    steps = inspect.getsource(lockdown._harden_steps)
    cover = steps.index("cover_cloud_metadata_for_verifier(env, task)")
    assert steps.index("_kill_sandbox_user_procs(env, sandbox_user)") < cover
    assert cover < steps.index(
        "in_order(env, guard, bash_env, aliases, early_pythonpath)"
    )
    order = timing_spans.HARDEN_STEPS
    assert (
        order.index("cloud_metadata_verifier")
        == order.index("kill_sandbox_user_procs") + 1
    )


@pytest.mark.asyncio
async def test_a_main_on_the_hosts_network_gets_no_rule_and_says_so(tmp_path):
    """EN-609 (RCI 03:59Z): no rule of the trial's goes into the host's own
    namespace. A restricted policy was refused before start, so the network
    is open: the record says "no rules; main is the host's network" and
    test.sh runs as on the rc, the host's tables untouched."""
    (tmp_path / "docker-compose.yaml").write_text(
        'services:\n  main:\n    network_mode: "host"\n'
    )
    env = HelperEnv()
    record = await lockdown.cover_cloud_metadata_for_verifier(
        env, _task(environment_dir=tmp_path)
    )
    assert record == {
        "state": "host network",
        "why": "no rules; main is the host's network",
    }
    assert env.installed == [] and env.main_commands == []


@pytest.mark.asyncio
async def test_a_compose_file_that_includes_others_is_refused_for_the_verifier(
    tmp_path,
):
    """EN-609 (F2b's ruling, fail closed): a task compose file with an
    ``include`` list may define main elsewhere, so main's network cannot be
    read from it; no rule goes in and test.sh does not run, and the message
    says why. The task's own opt-out runs it (test_local_daemon_docker's
    include build-cache case)."""
    (tmp_path / "side.yaml").write_text("services:\n  side:\n    image: x\n")
    (tmp_path / "docker-compose.yaml").write_text("include:\n  - side.yaml\n")
    env = HelperEnv()
    with pytest.raises(
        RuntimeError,
        match=r"main may share the host's network \(the task's docker-compose\.yaml includes other compose files \(include\)",
    ) as raised:
        await lockdown.cover_cloud_metadata_for_verifier(
            env, _task(environment_dir=tmp_path)
        )
    assert "includes other compose files (include)" in str(raised.value)
    assert env.installed == [] and env.main_commands == []
