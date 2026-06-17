"""ENG-263: runtime network-control behavior.

TDD for the three runtime fixes: daytona fail-closed enforcement, forcing the
host-side usage proxy under a restrictive policy, and the docker relock plan.
Tests exercise pure decision functions through public interfaces.
"""

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
    # no provider prefix -> unknown -> None (caller leaves allowlist unchanged)
    assert provider_host_for_model("deepseek-v4-flash", {}) is None
    # provider prefix but its base_url env is missing -> None (don't guess)
    assert provider_host_for_model("deepseek/x", {}) is None


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

import pytest  # noqa: E402


def _make_daytona_stub(task_env_config):
    from benchflow.sandbox.daytona import DaytonaSandbox

    sb = DaytonaSandbox.__new__(DaytonaSandbox)
    sb.task_env_config = task_env_config
    sb.logger = logging.getLogger("test-daytona")
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

    async def _unreachable():
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

    async def _reachable():
        return True  # platform did NOT enforce -> leaked egress

    sb._egress_reachable = _reachable
    with pytest.raises(SandboxStartupError, match="did not enforce"):
        await sb.relock_network(extra_allowed_hosts=())


@pytest.mark.asyncio
async def test_daytona_relock_noop_for_non_allowlist():
    from benchflow.task.config import SandboxConfig

    sb, applied = _make_daytona_stub(SandboxConfig(network_mode="no-network"))
    out = await sb.relock_network()
    assert out == {}
    assert applied == {}  # update_network_settings never called


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

    async def _unreachable():
        return False

    sb._egress_reachable = _unreachable
    await sb.relock_network(extra_allowed_hosts=("api.model.test",))
    # /etc/hosts pinned for both hosts before the allow list was applied
    hosts_writes = [c for c in execs if "/etc/hosts" in c]
    assert hosts_writes, "expected an /etc/hosts pin write"
    assert "9.9.9.9" in hosts_writes[0] and "a.com" in hosts_writes[0]
    assert applied["allow_list"] == "9.9.9.9/32,3.3.3.3/32"
