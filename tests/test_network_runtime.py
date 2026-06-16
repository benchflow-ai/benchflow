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
    # allowlist is unenforceable on daytona → resolve fails closed to block-all
    assert (
        network_blocks_all(
            SandboxConfig(network_mode="allowlist", allowed_hosts=["x.com"]), "daytona"
        )
        is True
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
