"""ENG-263: runtime network-control behavior.

TDD for the three runtime fixes: daytona fail-closed enforcement, forcing the
host-side usage proxy under a restrictive policy, and the docker relock plan.
Tests exercise pure decision functions through public interfaces.
"""

import pytest

from benchflow.task.config import SandboxConfig

# ---- Fix #2: daytona fail-closed enforcement -------------------------------


def test_network_blocks_all_for_daytona():
    from benchflow.sandbox.network_policy import network_blocks_all

    assert (
        network_blocks_all(SandboxConfig(network_mode="no-network"), "daytona") is True
    )
    assert network_blocks_all(SandboxConfig(), "daytona") is False  # public default
    # allowlist is now ENFORCEABLE on daytona (native IPv4-CIDR allow list), so it
    # resolves to ALLOWLIST — NOT block-all. Faithfulness (wildcards / >10 IPs)
    # is decided by plan_daytona_allowlist at lockdown, not here.
    assert (
        network_blocks_all(
            SandboxConfig(network_mode="allowlist", allowed_hosts=["x.com"]), "daytona"
        )
        is False
    )


def test_blockall_enforcement_violation():
    from benchflow.sandbox.network_policy import blockall_enforcement_violation

    # a block-all policy is VIOLATED when the sandbox can still reach the canary
    assert blockall_enforcement_violation(block_all=True, canary_reachable=True)
    # block-all and correctly unreachable → no violation
    assert not blockall_enforcement_violation(block_all=True, canary_reachable=False)
    # not block-all → never a violation
    assert not blockall_enforcement_violation(block_all=False, canary_reachable=True)


# ---- Fix #3: force host-side usage proxy under a restrictive policy ----------


def test_network_is_restrictive():
    from benchflow.sandbox.network_policy import network_is_restrictive

    assert (
        network_is_restrictive(SandboxConfig(network_mode="no-network"), "docker")
        is True
    )
    assert (
        network_is_restrictive(
            SandboxConfig(network_mode="allowlist", allowed_hosts=["x.com"]), "docker"
        )
        is True
    )
    assert network_is_restrictive(SandboxConfig(), "docker") is False  # public


def test_proxy_unavailable_is_fatal():
    from benchflow.sandbox.network_policy import proxy_unavailable_is_fatal

    # 'required' is always fatal (existing contract)
    assert proxy_unavailable_is_fatal(usage_mode="required", network_restrictive=False)
    # a restrictive policy can't silently fall back to the (blocked) direct provider
    assert proxy_unavailable_is_fatal(usage_mode="auto", network_restrictive=True)
    # public 'auto' may skip the proxy and use the provider directly
    assert not proxy_unavailable_is_fatal(usage_mode="auto", network_restrictive=False)
    # 'off' is an explicit opt-out — the caller owns provider reachability
    assert not proxy_unavailable_is_fatal(usage_mode="off", network_restrictive=True)


@pytest.mark.asyncio
async def test_rollout_lockdown_uses_effective_sandbox_task_env_config():
    """Guards PR #785: role-level restrictions must drive relock at launch."""
    from types import SimpleNamespace

    from benchflow.rollout import Rollout
    from benchflow.task.config import TaskConfig

    calls = []

    class _Env:
        task_env_config = SandboxConfig(
            network_mode="no-network", allow_model_endpoint=False
        )

        def relock_network(self, *, extra_allowed_hosts=()):
            calls.append(extra_allowed_hosts)
            return {}

    rollout = Rollout.__new__(Rollout)
    rollout._env = _Env()
    rollout._task = SimpleNamespace(
        config=TaskConfig.model_validate({"environment": {"network_mode": "public"}})
    )
    rollout._config = SimpleNamespace(
        environment="docker", primary_model="openai/gpt-4o"
    )
    rollout._agent_env = {}

    await Rollout._lock_down_network(rollout)

    assert calls == [()]


# ---- Fix #1b: relock fail-closed verification (greptile P1) ------------------


def test_lockdown_complete_docker_relock():
    from benchflow.sandbox.network_policy import lockdown_complete

    default = "proj_default"
    internal = "proj_bf_egress_internal"
    # allowlist / model-lane: detached from the public bridge, on the internal net
    assert lockdown_complete({internal}, default, internal) is True
    # disconnect silently failed -> still on default + internal -> egress BYPASS
    assert lockdown_complete({default, internal}, default, internal) is False
    # connect silently failed -> sidecar expected but not on internal -> stranded
    assert lockdown_complete(set(), default, internal) is False
    # hermetic (no sidecar): detached, attached to nothing -> fully dark, ok
    assert lockdown_complete(set(), default, None) is True
    # hermetic but disconnect failed -> still on default -> not locked down
    assert lockdown_complete({default}, default, None) is False


# ---- Fix #1c: plain-HTTP origin-form rewrite (greptile P2 #1; lane blocker) ---


def test_to_origin_form_rewrites_absolute_uri():
    from benchflow.sandbox._egress_proxy import _to_origin_form

    # proxy absolute-URI request line -> origin-form; headers + body preserved
    h = b"POST http://172.17.0.1:8080/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n{}"
    assert (
        _to_origin_form(h)
        == b"POST /chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n{}"
    )


def test_to_origin_form_query_and_root_default():
    from benchflow.sandbox._egress_proxy import _to_origin_form

    assert (
        _to_origin_form(b"GET http://h:80/p?a=1&b=2 HTTP/1.1\r\n\r\n")
        == b"GET /p?a=1&b=2 HTTP/1.1\r\n\r\n"
    )
    assert (
        _to_origin_form(b"GET http://h HTTP/1.1\r\n\r\n") == b"GET / HTTP/1.1\r\n\r\n"
    )


def test_to_origin_form_passthrough_already_origin():
    from benchflow.sandbox._egress_proxy import _to_origin_form

    h = b"GET /already HTTP/1.1\r\nHost: h\r\n\r\n"
    assert _to_origin_form(h) == h


# ---- Fix #3 pivot: resolve provider host to allowlist under restrictive policy ---


def test_provider_host_for_model():
    from benchflow.agents.providers import provider_host_for_model

    assert (
        provider_host_for_model(
            "deepseek/deepseek-v4-flash",
            {"DEEPSEEK_BASE_URL": "https://api.deepseek.com"},
        )
        == "api.deepseek.com"
    )
    assert provider_host_for_model("openai/gpt-4o", {}) == "api.openai.com"
    # bare model ids still resolve through the provider catalog
    assert provider_host_for_model("deepseek-v4-flash", {}) == "api.deepseek.com"
    # provider prefix resolves through the provider catalog default
    assert provider_host_for_model("deepseek/x", {}) == "api.deepseek.com"


# ---- Daytona allowlist parity: enforce-when-faithful plan (ENG-219 follow-up) ----


def _fake_resolver(table):
    def resolve(host):
        return tuple(table.get(host, ()))

    return resolve


def test_daytona_plan_resolves_hosts_and_model_to_cidrs():
    from benchflow.sandbox.network_policy import plan_daytona_allowlist

    plan = plan_daytona_allowlist(
        ("api.crossref.org", "doi.org"),
        model_host="api.deepseek.com",
        resolve=_fake_resolver(
            {
                "api.crossref.org": ("1.1.1.1",),
                "doi.org": ("2.2.2.2",),
                "api.deepseek.com": ("3.3.3.3",),
            }
        ),
    )
    assert plan.enforceable
    assert plan.reject_reason is None
    assert plan.cidrs == ("1.1.1.1/32", "2.2.2.2/32", "3.3.3.3/32")


def test_daytona_plan_dedups_shared_ips():
    from benchflow.sandbox.network_policy import plan_daytona_allowlist

    plan = plan_daytona_allowlist(
        ("a.example.com", "b.example.com"),
        model_host=None,
        resolve=_fake_resolver(
            {"a.example.com": ("9.9.9.9", "8.8.8.8"), "b.example.com": ("9.9.9.9",)}
        ),
    )
    assert plan.enforceable
    assert plan.cidrs == ("9.9.9.9/32", "8.8.8.8/32")


def test_daytona_plan_rejects_wildcard():
    from benchflow.sandbox.network_policy import plan_daytona_allowlist

    plan = plan_daytona_allowlist(
        ("*.crossref.org", "doi.org"),
        model_host="api.deepseek.com",
        resolve=_fake_resolver(
            {"doi.org": ("2.2.2.2",), "api.deepseek.com": ("3.3.3.3",)}
        ),
    )
    assert not plan.enforceable
    assert plan.cidrs == ()
    assert "wildcard" in plan.reject_reason.lower()
    assert "*.crossref.org" in plan.reject_reason


def test_daytona_plan_rejects_unresolvable_host():
    from benchflow.sandbox.network_policy import plan_daytona_allowlist

    plan = plan_daytona_allowlist(
        ("api.crossref.org", "nope.invalid"),
        model_host=None,
        resolve=_fake_resolver({"api.crossref.org": ("1.1.1.1",)}),
    )
    assert not plan.enforceable
    assert "nope.invalid" in plan.reject_reason
    assert "resolve" in plan.reject_reason.lower()


def test_daytona_plan_rejects_over_ten_ips():
    from benchflow.sandbox.network_policy import plan_daytona_allowlist

    table = {f"h{i}.example.com": (f"10.0.0.{i}",) for i in range(11)}
    plan = plan_daytona_allowlist(
        tuple(table), model_host=None, resolve=_fake_resolver(table)
    )
    assert not plan.enforceable
    assert "10" in plan.reject_reason


def test_daytona_plan_rejects_empty_resolution():
    from benchflow.sandbox.network_policy import plan_daytona_allowlist

    plan = plan_daytona_allowlist((), model_host=None, resolve=_fake_resolver({}))
    assert not plan.enforceable


# ---- Daytona relock_network: enforce-or-fail-closed orchestration ----

import logging  # noqa: E402


def _make_daytona_stub(task_env_config):
    from benchflow.sandbox.daytona import DaytonaSandbox

    sb = DaytonaSandbox.__new__(DaytonaSandbox)
    sb.task_env_config = task_env_config
    sb.logger = logging.getLogger("test-daytona")
    sb._compose_mode = False
    applied = {}

    class _Inner:
        async def update_network_settings(
            self, *, network_allow_list=None, network_block_all=None
        ):
            applied["allow_list"] = network_allow_list

    sb._sandbox = _Inner()
    return sb, applied


@pytest.mark.asyncio
async def test_daytona_relock_applies_faithful_allowlist(monkeypatch):
    from benchflow.sandbox import network_policy
    from benchflow.task.config import SandboxConfig

    sb, applied = _make_daytona_stub(
        SandboxConfig(network_mode="allowlist", allowed_hosts=["a.com"])
    )
    monkeypatch.setattr(
        network_policy,
        "plan_daytona_allowlist",
        lambda hosts, *, model_host, resolve=None: network_policy.DaytonaAllowlistPlan(
            cidrs=("9.9.9.9/32", "3.3.3.3/32")
        ),
    )

    async def _unreachable(canary=None):
        return False  # non-allowlisted canary correctly blocked

    sb._egress_reachable = _unreachable
    out = await sb.relock_network(extra_allowed_hosts=("api.model.test",))
    assert out == {}
    assert applied["allow_list"] == "9.9.9.9/32,3.3.3.3/32"


@pytest.mark.asyncio
async def test_daytona_relock_fails_closed_on_unfaithful_plan(monkeypatch):
    from benchflow.sandbox import network_policy
    from benchflow.sandbox.protocol import SandboxStartupError
    from benchflow.task.config import SandboxConfig

    sb, _ = _make_daytona_stub(
        SandboxConfig(network_mode="allowlist", allowed_hosts=["a.com"])
    )
    monkeypatch.setattr(
        network_policy,
        "plan_daytona_allowlist",
        lambda hosts, *, model_host, resolve=None: network_policy.DaytonaAllowlistPlan(
            reject_reason="resolves to 13 IPv4 addresses, exceeding daytona's 10-CIDR limit"
        ),
    )
    with pytest.raises(SandboxStartupError, match="cannot enforce"):
        await sb.relock_network(extra_allowed_hosts=())


@pytest.mark.asyncio
async def test_daytona_relock_fails_closed_if_canary_still_reachable(monkeypatch):
    from benchflow.sandbox import network_policy
    from benchflow.sandbox.protocol import SandboxStartupError
    from benchflow.task.config import SandboxConfig

    sb, _ = _make_daytona_stub(
        SandboxConfig(network_mode="allowlist", allowed_hosts=["a.com"])
    )
    monkeypatch.setattr(
        network_policy,
        "plan_daytona_allowlist",
        lambda hosts, *, model_host, resolve=None: network_policy.DaytonaAllowlistPlan(
            cidrs=("9.9.9.9/32",)
        ),
    )

    async def _reachable(canary=None):
        return True  # platform did NOT enforce -> leaked egress

    sb._egress_reachable = _reachable
    with pytest.raises(SandboxStartupError, match="did not enforce"):
        await sb.relock_network(extra_allowed_hosts=())


@pytest.mark.asyncio
async def test_daytona_relock_noop_for_public_policy():
    from benchflow.task.config import SandboxConfig

    sb, applied = _make_daytona_stub(SandboxConfig(network_mode="public"))
    out = await sb.relock_network()
    assert out == {}
    assert applied == {}  # update_network_settings never called


@pytest.mark.asyncio
async def test_daytona_relock_no_network_model_lane_allowlists_provider(monkeypatch):
    """Guards PR #785: Daytona no-network needs a post-install model lane."""
    from benchflow.sandbox import network_policy
    from benchflow.task.config import SandboxConfig

    sb, applied = _make_daytona_stub(SandboxConfig(network_mode="no-network"))
    seen = {}

    def _plan(hosts, *, model_host, resolve=None):
        seen["hosts"] = hosts
        seen["model_host"] = model_host
        return network_policy.DaytonaAllowlistPlan(cidrs=("3.3.3.3/32",))

    monkeypatch.setattr(network_policy, "plan_daytona_allowlist", _plan)

    async def _unreachable(canary=None):
        return False

    sb._egress_reachable = _unreachable
    await sb.relock_network(extra_allowed_hosts=("api.model.test",))

    assert seen == {"hosts": (), "model_host": "api.model.test"}
    assert applied["allow_list"] == "3.3.3.3/32"


@pytest.mark.asyncio
async def test_daytona_relock_no_network_hermetic_uses_platform_block_all():
    """Guards PR #785: fully hermetic Daytona no-network stays block-all."""
    from benchflow.task.config import SandboxConfig

    sb, applied = _make_daytona_stub(
        SandboxConfig(network_mode="no-network", allow_model_endpoint=False)
    )
    out = await sb.relock_network()
    assert out == {}
    assert applied == {}


@pytest.mark.asyncio
async def test_ensure_litellm_skips_under_restrictive_daytona():
    """Under a restrictive daytona policy the in-sandbox usage proxy can't be
    installed (pypi blocked post-lockdown), so ensure_litellm_runtime skips it and
    lets the agent reach the (allowlisted) provider directly — same as docker."""
    from benchflow.providers import litellm_runtime
    from benchflow.task.config import SandboxConfig

    class _FakeSandbox:
        task_env_config = SandboxConfig(
            network_mode="allowlist", allowed_hosts=["a.com"]
        )

    agent_env = {
        "LLM_BASE_URL": "https://api.deepseek.com",
        "LLM_MODEL": "openai/deepseek-v4-flash",
        "DEEPSEEK_API_KEY": "sk-test",
        "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
    }
    out_env, runtime = await litellm_runtime.ensure_litellm_runtime(
        agent="openhands",
        agent_env=agent_env,
        model="deepseek/deepseek-v4-flash",
        runtime=None,
        environment="daytona",
        usage_tracking="auto",
        sandbox=_FakeSandbox(),
    )
    assert runtime is None
    assert out_env is agent_env  # skipped: env returned unchanged


@pytest.mark.asyncio
async def test_daytona_relock_pins_etc_hosts(monkeypatch):
    from benchflow.sandbox import network_policy
    from benchflow.task.config import SandboxConfig

    sb, applied = _make_daytona_stub(
        SandboxConfig(network_mode="allowlist", allowed_hosts=["a.com"])
    )
    execs = []

    async def _fake_exec(cmd, *a, **k):
        execs.append(cmd)

        class _R:
            stdout = ""
            result = ""

        return _R()

    sb.exec = _fake_exec
    monkeypatch.setattr(
        network_policy,
        "plan_daytona_allowlist",
        lambda hosts, *, model_host, resolve=None: network_policy.DaytonaAllowlistPlan(
            cidrs=("9.9.9.9/32", "3.3.3.3/32"),
            host_ips=(("a.com", "9.9.9.9"), ("api.model.test", "3.3.3.3")),
        ),
    )

    async def _unreachable(canary=None):
        return False

    sb._egress_reachable = _unreachable
    await sb.relock_network(extra_allowed_hosts=("api.model.test",))
    # /etc/hosts pinned for both hosts before the allow list was applied
    hosts_writes = [c for c in execs if "/etc/hosts" in c]
    assert hosts_writes, "expected an /etc/hosts pin write"
    assert "9.9.9.9" in hosts_writes[0] and "a.com" in hosts_writes[0]
    assert applied["allow_list"] == "9.9.9.9/32,3.3.3.3/32"


# ---- origin-form rewrite: query handling + scheme-prefix detection (audit P1) ----


def test_to_origin_form_preserves_query():
    from benchflow.sandbox._egress_proxy import _to_origin_form

    assert (
        _to_origin_form(b"GET http://h?x=1 HTTP/1.1\r\n\r\n")
        == b"GET /?x=1 HTTP/1.1\r\n\r\n"
    )
    # a '/' inside the query value must not be taken as the path boundary
    assert (
        _to_origin_form(b"GET http://h?a=b/c HTTP/1.1\r\n\r\n")
        == b"GET /?a=b/c HTTP/1.1\r\n\r\n"
    )
    # authority with a port, query-only
    assert (
        _to_origin_form(b"GET http://h:80?a=1 HTTP/1.1\r\n\r\n")
        == b"GET /?a=1 HTTP/1.1\r\n\r\n"
    )
    # normal path+query is unchanged in shape
    assert (
        _to_origin_form(b"GET http://h/p?q=1 HTTP/1.1\r\n\r\n")
        == b"GET /p?q=1 HTTP/1.1\r\n\r\n"
    )


def test_to_origin_form_leaves_origin_form_with_url_query_untouched():
    from benchflow.sandbox._egress_proxy import _to_origin_form

    # already origin-form, but the query contains '://' — must be returned verbatim
    h = b"GET /cb?u=http://e.com HTTP/1.1\r\nHost: api.example.com\r\n\r\n"
    assert _to_origin_form(h) == h


def test_absolute_uri_host_ignores_scheme_in_query():
    from benchflow.sandbox._egress_proxy import _absolute_uri_host

    # absolute-form: authority is the real host, stopping at '/', '?' or '#'
    assert _absolute_uri_host("http://api.example.com/p?x=1") == "api.example.com"
    assert (
        _absolute_uri_host("https://api.example.com?u=http://e.com")
        == "api.example.com"
    )
    assert _absolute_uri_host("http://h:8080") == "h:8080"
    # origin-form with '://' in the query is NOT absolute -> None (use Host header)
    assert _absolute_uri_host("/cb?u=http://evil.com") is None
    assert _absolute_uri_host("/plain") is None


# ---- bare model-id provider host resolution (audit P1: -32603 for bare ids) ----


def test_provider_host_for_model_resolves_bare_id():
    from benchflow.agents.providers import provider_host_for_model

    # the documented dev model in BARE form (prefix already stripped) must still
    # resolve its host so a restrictive run allowlists it
    assert (
        provider_host_for_model(
            "deepseek-v4-flash", {"DEEPSEEK_BASE_URL": "https://api.deepseek.com"}
        )
        == "api.deepseek.com"
    )
    # prefixed form keeps working
    assert (
        provider_host_for_model(
            "deepseek/deepseek-v4-flash",
            {"DEEPSEEK_BASE_URL": "https://api.deepseek.com"},
        )
        == "api.deepseek.com"
    )
    # genuinely unknown model -> None (caller fails closed)
    assert provider_host_for_model("totally-unknown-model-xyz", {}) is None


@pytest.mark.asyncio
async def test_ensure_litellm_fails_closed_when_provider_host_unresolvable():
    """Under a restrictive policy, if the provider host can't be resolved (so it
    was never allowlisted), skipping the proxy would launch an agent that 403s on
    its model CONNECT. ensure_litellm_runtime must fail closed instead."""
    import pytest as _pytest

    from benchflow.providers import litellm_runtime
    from benchflow.task.config import SandboxConfig

    class _FakeSandbox:
        task_env_config = SandboxConfig(
            network_mode="allowlist", allowed_hosts=["a.com"]
        )

    with _pytest.raises(RuntimeError, match="cannot resolve a provider host"):
        await litellm_runtime.ensure_litellm_runtime(
            agent="openhands",
            agent_env={"LLM_BASE_URL": "https://x"},
            model="totally-unknown-model-xyz",
            runtime=None,
            environment="docker",
            usage_tracking="auto",
            sandbox=_FakeSandbox(),
        )


@pytest.mark.asyncio
async def test_ensure_litellm_fails_closed_when_model_lane_disabled():
    """Guards PR #785: restrictive tasks that opt out of the model lane must not
    skip the proxy and launch an agent whose model endpoint is unreachable."""
    from benchflow.providers import litellm_runtime
    from benchflow.task.config import SandboxConfig

    class _FakeSandbox:
        task_env_config = SandboxConfig(
            network_mode="allowlist",
            allowed_hosts=["api.deepseek.com"],
            allow_model_endpoint=False,
        )

    with pytest.raises(RuntimeError, match="allow_model_endpoint=false"):
        await litellm_runtime.ensure_litellm_runtime(
            agent="openhands",
            agent_env={"DEEPSEEK_API_KEY": "sk"},
            model="deepseek/deepseek-v4-flash",
            runtime=None,
            environment="docker",
            usage_tracking="auto",
            sandbox=_FakeSandbox(),
        )


# ---- docker relock fail-closed gate: deny extras + inspect-rc (audit P1/P2) ----


def test_lockdown_complete_denies_extra_network():
    from benchflow.sandbox.network_policy import lockdown_complete

    internal = "proj_bf_egress_internal"
    default = "proj_default"
    assert lockdown_complete({internal}, default, internal) is True
    # a stray non-internal net survives the swap -> NOT complete (bypass risk)
    assert lockdown_complete({internal, "proj_mynet"}, default, internal) is False
    assert lockdown_complete(set(), default, None) is True
    assert lockdown_complete({"proj_mynet"}, default, None) is False
    assert lockdown_complete({internal, default}, default, internal) is False
    # explicitly-permitted benign net is allowed
    assert (
        lockdown_complete(
            {internal, "proj_ok"}, default, internal, frozenset({"proj_ok"})
        )
        is True
    )


def _relock_project():
    from benchflow.sandbox.docker import _sanitize_docker_compose_project_name

    return _sanitize_docker_compose_project_name("relocktest")


def _make_docker_relock_stub(inspect_stdout, inspect_rc):
    import logging

    from benchflow.sandbox._base import ExecResult
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.task.config import SandboxConfig

    sb = DockerSandbox.__new__(DockerSandbox)
    sb.task_env_config = SandboxConfig(
        network_mode="allowlist", allowed_hosts=["a.com"]
    )
    sb.session_id = "relocktest"
    sb._network_locked = False
    sb._extra_allowed_hosts = ()
    sb.logger = logging.getLogger("relock-test")

    async def _cid():
        return "cid123"

    sb._main_container_id = _cid
    sb._network_policy_compose_paths = lambda: ["/x/egress.json"]

    async def _compose(args, check=True):
        if args == ["ps", "--quiet", "bf-egress"]:
            return ExecResult(stdout="egresscid\n", stderr="", return_code=0)
        return ExecResult(stdout="", stderr="", return_code=0)

    sb._run_docker_compose_command = _compose

    async def _cli(args, check=True):
        if args[:2] == ["inspect", "cid123"]:
            return ExecResult(stdout=inspect_stdout, stderr="", return_code=inspect_rc)
        if args[:2] == ["inspect", "egresscid"]:
            return ExecResult(stdout="healthy\n", stderr="", return_code=0)
        return ExecResult(stdout="", stderr="", return_code=0)

    sb._docker_cli = _cli
    return sb


def test_docker_allowlist_omits_extra_hosts_when_model_lane_disabled(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    from benchflow.sandbox import docker_network_lockdown
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.task.config import SandboxConfig

    captured = {}

    def fake_build(hosts, *, out_dir, model_lane):
        captured["hosts"] = hosts
        captured["model_lane"] = model_lane
        return Path("/tmp/egress.json")

    monkeypatch.setattr(docker_network_lockdown, "build_egress_override", fake_build)
    sb = DockerSandbox.__new__(DockerSandbox)
    sb.task_env_config = SandboxConfig(
        network_mode="allowlist",
        allowed_hosts=["task.example"],
        allow_model_endpoint=False,
    )
    sb._network_locked = True
    sb._extra_allowed_hosts = ("api.model.test",)
    sb.rollout_paths = SimpleNamespace(rollout_dir=Path("/tmp"))

    assert sb._network_policy_compose_paths() == [Path("/tmp/egress.json")]
    assert captured["hosts"] == ("task.example",)
    assert captured["model_lane"] is None


@pytest.mark.asyncio
async def test_docker_relock_raises_on_stray_network():
    from benchflow.sandbox.protocol import SandboxStartupError

    proj = _relock_project()
    sb = _make_docker_relock_stub(f"{proj}_bf_egress_internal {proj}_mynet", 0)
    with pytest.raises(SandboxStartupError, match="did not take effect"):
        await sb.relock_network()


@pytest.mark.asyncio
async def test_docker_relock_raises_on_inspect_error():
    from benchflow.sandbox.protocol import SandboxStartupError

    sb = _make_docker_relock_stub("", 1)
    with pytest.raises(SandboxStartupError, match="could not inspect"):
        await sb.relock_network()


@pytest.mark.asyncio
async def test_docker_relock_happy_sidecar_returns_proxy_env():
    proj = _relock_project()
    sb = _make_docker_relock_stub(f"{proj}_bf_egress_internal", 0)
    out = await sb.relock_network()
    assert out.get("HTTPS_PROXY", "").startswith("http://bf-egress:")


@pytest.mark.asyncio
async def test_docker_relock_waits_for_egress_health_before_proxy_env(monkeypatch):
    """Guards PR #785 against returning proxy env before bf-egress is ready."""
    from benchflow.sandbox import docker_network_lockdown
    from benchflow.sandbox._base import ExecResult

    monkeypatch.setattr(docker_network_lockdown, "_EGRESS_HEALTH_INTERVAL_SEC", 0)
    proj = _relock_project()
    sb = _make_docker_relock_stub(f"{proj}_bf_egress_internal", 0)
    events: list[str] = []
    health_statuses = iter(["starting\n", "healthy\n"])

    async def _compose(args, check=True):
        events.append(f"compose:{' '.join(args)}")
        if args == ["ps", "--quiet", "bf-egress"]:
            return ExecResult(stdout="egresscid\n", stderr="", return_code=0)
        return ExecResult(stdout="", stderr="", return_code=0)

    async def _cli(args, check=True):
        if args[:2] == ["inspect", "cid123"]:
            events.append("main-networks-inspected")
            return ExecResult(
                stdout=f"{proj}_bf_egress_internal", stderr="", return_code=0
            )
        if args[:2] == ["inspect", "egresscid"]:
            status = next(health_statuses)
            events.append(f"egress-health:{status.strip()}")
            return ExecResult(stdout=status, stderr="", return_code=0)
        return ExecResult(stdout="", stderr="", return_code=0)

    sb._run_docker_compose_command = _compose
    sb._docker_cli = _cli

    out = await sb.relock_network()

    assert out["HTTP_PROXY"].startswith("http://bf-egress:")
    assert events.index("main-networks-inspected") < events.index(
        "compose:ps --quiet bf-egress"
    )
    assert events.index("egress-health:starting") < events.index(
        "egress-health:healthy"
    )


@pytest.mark.asyncio
async def test_docker_restore_rejects_restrictive_network_policy():
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.sandbox.protocol import SandboxImage, SandboxSnapshotNotSupported
    from benchflow.task.config import SandboxConfig

    sb = DockerSandbox.__new__(DockerSandbox)
    sb.task_env_config = SandboxConfig(
        network_mode="allowlist", allowed_hosts=["a.com"]
    )

    with pytest.raises(SandboxSnapshotNotSupported, match="restrictive"):
        await sb.restore(SandboxImage(provider="docker", ref="snapshot:latest"))


# ---- daytona allowlist hardening (audit P2: canary / compose / probe) ----


def test_blockall_violation_treats_unverifiable_probe_as_violation():
    from benchflow.sandbox.network_policy import blockall_enforcement_violation

    assert blockall_enforcement_violation(block_all=True, canary_reachable=True) is True
    assert (
        blockall_enforcement_violation(block_all=True, canary_reachable=False) is False
    )
    # probe could not run -> cannot confirm blocked -> fail closed
    assert blockall_enforcement_violation(block_all=True, canary_reachable=None) is True
    assert (
        blockall_enforcement_violation(block_all=False, canary_reachable=None) is False
    )


def test_pick_canary_avoids_allowlisted_ip():
    from benchflow.sandbox.daytona import _pick_canary

    assert _pick_canary(()) == "1.1.1.1"
    # a host resolved to 1.1.1.1 is allowlisted -> canary must move off it
    assert _pick_canary(("1.1.1.1/32",)) == "8.8.8.8"
    assert _pick_canary(("1.1.1.1/32", "8.8.8.8/32")) == "9.9.9.9"
    first_ten = (
        "1.1.1.1/32",
        "8.8.8.8/32",
        "9.9.9.9/32",
        "1.0.0.1/32",
        "8.8.4.4/32",
        "9.9.9.10/32",
        "208.67.222.222/32",
        "208.67.220.220/32",
        "64.6.64.6/32",
        "64.6.65.6/32",
    )
    assert _pick_canary(first_ten) == "76.76.2.0"


def test_daytona_start_block_all_allows_model_lane_relock():
    """Guards PR #785: no-network+model lane must not pre-block install."""
    from benchflow.sandbox.daytona import _start_with_platform_block_all
    from benchflow.task.config import SandboxConfig

    assert (
        _start_with_platform_block_all(SandboxConfig(network_mode="no-network"))
        is False
    )
    assert (
        _start_with_platform_block_all(
            SandboxConfig(network_mode="no-network", allow_model_endpoint=False)
        )
        is True
    )


@pytest.mark.asyncio
async def test_daytona_relock_fails_closed_for_compose_mode():
    import logging

    from benchflow.sandbox.daytona import DaytonaSandbox
    from benchflow.sandbox.protocol import SandboxStartupError
    from benchflow.task.config import SandboxConfig

    sb = DaytonaSandbox.__new__(DaytonaSandbox)
    sb.task_env_config = SandboxConfig(
        network_mode="allowlist", allowed_hosts=["a.com"]
    )
    sb._compose_mode = True
    sb.logger = logging.getLogger("daytona-compose-test")
    with pytest.raises(SandboxStartupError, match="compose/DinD"):
        await sb.relock_network(extra_allowed_hosts=())


@pytest.mark.asyncio
async def test_daytona_relock_omits_model_host_when_lane_disabled(monkeypatch):
    from benchflow.sandbox import network_policy
    from benchflow.task.config import SandboxConfig

    sb, _ = _make_daytona_stub(
        SandboxConfig(
            network_mode="allowlist",
            allowed_hosts=["a.com"],
            allow_model_endpoint=False,
        )
    )
    seen = {}

    def _plan(hosts, *, model_host, resolve=None):
        seen["hosts"] = hosts
        seen["model_host"] = model_host
        return network_policy.DaytonaAllowlistPlan(cidrs=("9.9.9.9/32",))

    monkeypatch.setattr(network_policy, "plan_daytona_allowlist", _plan)

    async def _unreachable(canary=None):
        return False

    sb._egress_reachable = _unreachable
    await sb.relock_network(extra_allowed_hosts=("api.model.test",))
    assert seen == {"hosts": ("a.com",), "model_host": None}
