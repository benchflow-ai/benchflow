"""Denylist egress mode: task config, capability gate, and stdlib mirrors.

Guards the denylist egress mode, benchflow-ai/FrontierPhysics#365.
"""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import pytest

from benchflow.sandbox.providers import (
    DENYLIST_UNSUPPORTED_PROVIDERS,
    NO_NETWORK_UNSUPPORTED_PROVIDERS,
    PROVIDERS_BY_NAME,
)
from benchflow.task import NetworkMode, TaskConfig, validate_task_runtime_support
from benchflow.task.config import (
    SandboxConfig,
    VerifierConfig,
    _validate_network_policy_fields,
)
from benchflow.task.document import TaskDocument

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "tests" / "integration"))
sys.path.insert(0, str(_REPO_ROOT / ".github" / "scripts"))

import build_integration_review_pack as pack_mod  # noqa: E402
import rubric_checks  # noqa: E402


def test_denylist_is_a_network_mode() -> None:
    """The enum accepts the new literal alongside the existing modes."""
    assert NetworkMode("denylist") is NetworkMode.DENYLIST
    assert NetworkMode.DENYLIST.value == "denylist"
    assert NetworkMode.DENYLIST != NetworkMode.ALLOWLIST


def test_denylist_fields_parse_from_task_md_frontmatter() -> None:
    """task.md frontmatter carries blocked_urls/blocked_hosts on sandbox."""
    document = TaskDocument.from_text(
        dedent(
            """\
            ---
            schema_version: "1.3"
            sandbox:
              network_mode: denylist
              blocked_urls: [arxiv.org/abs/2401.00001]
              blocked_hosts: [github.com]
            ---
            ## prompt

            Reproduce the paper.
            """
        )
    )

    cfg = document.config
    assert cfg.agent.network_mode is None
    assert cfg.sandbox.network_mode == NetworkMode.DENYLIST
    assert cfg.sandbox.blocked_urls == ["https://arxiv.org/abs/2401.00001"]
    assert cfg.sandbox.blocked_hosts == ["github.com"]
    assert cfg.sandbox.allow_internet is True


def test_denylist_fields_parse_from_task_toml() -> None:
    """Legacy task.toml spells the same fields under [environment]."""
    cfg = TaskConfig.model_validate_toml(
        dedent(
            """\
            schema_version = "1.3"

            [environment]
            network_mode = "denylist"
            blocked_urls = ["https://github.com/org/repo"]
            blocked_hosts = ["arxiv.org", "semanticscholar.org"]
            """
        )
    )

    assert cfg.sandbox.network_mode == NetworkMode.DENYLIST
    assert cfg.sandbox.blocked_urls == ["https://github.com/org/repo"]
    assert cfg.sandbox.blocked_hosts == ["arxiv.org", "semanticscholar.org"]


def test_blocked_urls_are_normalized() -> None:
    """Scheme is added, host lowercased, query and fragment dropped, duplicates collapsed."""
    cfg = SandboxConfig(
        network_mode="denylist",
        blocked_urls=[
            "ArXiv.org/abs/2401.00001?context=physics",
            "https://GitHub.com/Org/Repo#readme",
            "http://example.com",
            " example.com ",
            "example.com",
            "https://arxiv.org/abs/2401.00001",
        ],
    )

    assert cfg.blocked_urls == [
        "https://arxiv.org/abs/2401.00001",
        "https://github.com/Org/Repo",
        "http://example.com",
        "https://example.com",
    ]


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("https://1.2.3.4/paper", "not an IP"),
        ("https://*.example.com/x", "valid hostnames"),
        ("https://user:secret@example.com/x", "userinfo"),
        ("https://example.com:8443/x", "port"),
        ("", "non-empty"),
        ("ftp://example.com/x", "http or https"),
        ("https:///x", "non-empty hostnames"),
    ],
)
def test_blocked_urls_rejections(url: str, message: str) -> None:
    """IP literals, wildcards, userinfo, ports, empties, and odd schemes are refused."""
    with pytest.raises(ValueError, match=message):
        SandboxConfig(network_mode="denylist", blocked_urls=[url])


def test_blocked_urls_drop_trailing_slash_and_blocked_hosts_refuse_addresses() -> None:
    """Guards two review findings: a trailing slash must not exempt the page itself, and hosts cannot be IPs."""
    cfg = SandboxConfig(network_mode="denylist", blocked_urls=["github.com/org/repo/"])
    assert cfg.blocked_urls == ["https://github.com/org/repo"]
    with pytest.raises(ValueError, match="not an IP"):
        SandboxConfig(network_mode="denylist", blocked_hosts=["1.2.3.4"])


def test_blocked_hosts_reuse_the_hostname_rules() -> None:
    """blocked_hosts normalizes like allowed_hosts and refuses URL-shaped entries."""
    cfg = SandboxConfig(
        network_mode="denylist", blocked_hosts=["ArXiv.org.", "github.com"]
    )
    assert cfg.blocked_hosts == ["arxiv.org", "github.com"]

    with pytest.raises(ValueError, match="blocked_hosts entries must be hostnames"):
        SandboxConfig(network_mode="denylist", blocked_hosts=["https://arxiv.org"])
    with pytest.raises(ValueError, match="blocked_hosts entries must be non-empty"):
        SandboxConfig(network_mode="denylist", blocked_hosts=["  "])


def test_allowed_hosts_error_strings_are_unchanged() -> None:
    """Factoring the hostname rules must keep the allowed_hosts messages verbatim."""
    with pytest.raises(ValueError, match="allowed_hosts entries must be hostnames"):
        SandboxConfig(network_mode="allowlist", allowed_hosts=["https://x.com"])
    with pytest.raises(ValueError, match="allowed_hosts entries must be non-empty"):
        SandboxConfig(network_mode="allowlist", allowed_hosts=[""])
    with pytest.raises(ValueError, match="allowed_hosts entries must be valid"):
        SandboxConfig(network_mode="allowlist", allowed_hosts=["bad_host"])


@pytest.mark.parametrize("fields", [{}, {"blocked_urls": []}, {"blocked_hosts": []}])
def test_denylist_without_lists_is_rejected(fields: dict[str, list[str]]) -> None:
    """A denylist that blocks nothing is a misconfiguration, not a public sandbox."""
    with pytest.raises(ValueError, match="requires blocked_urls or blocked_hosts"):
        SandboxConfig(network_mode="denylist", **fields)


@pytest.mark.parametrize("mode", ["public", "no-network", "allowlist"])
def test_blocked_lists_outside_denylist_are_rejected(mode: str) -> None:
    """blocked_urls/blocked_hosts carry no meaning under any other mode."""
    extra = {"allowed_hosts": ["x.com"]} if mode == "allowlist" else {}
    with pytest.raises(ValueError, match="only valid for network_mode='denylist'"):
        SandboxConfig(network_mode=mode, blocked_hosts=["arxiv.org"], **extra)
    with pytest.raises(ValueError, match="only valid for network_mode='denylist'"):
        SandboxConfig(
            network_mode=mode, blocked_urls=["https://arxiv.org/abs/1"], **extra
        )


def test_allowlist_with_blocked_lists_is_rejected_before_allowlist_check() -> None:
    """The allowlist messages stay verbatim and fire before the denylist ones."""
    with pytest.raises(ValueError, match="allowed_hosts must be non-empty"):
        _validate_network_policy_fields(NetworkMode.ALLOWLIST, None, ["x"], None)
    with pytest.raises(ValueError, match="allowed_hosts is only valid"):
        _validate_network_policy_fields(NetworkMode.DENYLIST, ["x.com"], ["x"], None)


def test_verifier_rejects_denylist() -> None:
    """The verifier has no blocked lists, so a denylist verifier cannot exist."""
    with pytest.raises(ValueError, match=r"verifier\.network_mode='denylist' is not"):
        VerifierConfig(network_mode="denylist")
    with pytest.raises(ValueError, match="blocked_hosts"):
        VerifierConfig(network_mode="public", blocked_hosts=["arxiv.org"])


def test_denylist_keeps_allow_internet_true() -> None:
    """A denylist sandbox still has internet; only the listed targets are cut."""
    cfg = SandboxConfig(network_mode="denylist", blocked_hosts=["arxiv.org"])
    assert cfg.allow_internet is True
    assert cfg.network_mode == NetworkMode.DENYLIST


def test_denylist_with_explicit_allow_internet_false_is_a_contradiction() -> None:
    """Explicit allow_internet=False against denylist is the existing hard error."""
    with pytest.raises(ValueError, match="allow_internet=False contradicts"):
        SandboxConfig(
            network_mode="denylist",
            blocked_hosts=["arxiv.org"],
            allow_internet=False,
        )


def test_registry_declares_denylist_enforcement() -> None:
    """Only docker and daytona run the loopback proxy behind the uid firewall."""
    assert PROVIDERS_BY_NAME["docker"].enforces_denylist is True
    assert PROVIDERS_BY_NAME["daytona"].enforces_denylist is True
    assert (
        frozenset({"modal", "apple-container", "agentcore"})
        == DENYLIST_UNSUPPORTED_PROVIDERS
    )
    assert NO_NETWORK_UNSUPPORTED_PROVIDERS <= DENYLIST_UNSUPPORTED_PROVIDERS


@pytest.mark.parametrize("sandbox", ["docker", "daytona"])
def test_capability_gate_accepts_denylist_on_enforcing_backends(sandbox: str) -> None:
    """Docker and daytona launch a denylist task without a capability issue."""
    config = TaskConfig.model_validate(
        {
            "sandbox": {
                "network_mode": "denylist",
                "blocked_urls": ["https://arxiv.org/abs/2401.00001"],
            },
        }
    )

    assert validate_task_runtime_support(config, sandbox=sandbox) == []


@pytest.mark.parametrize("role", ["agent", "verifier"])
def test_role_level_denylist_is_rejected(role: str) -> None:
    """denylist is a sandbox policy; role overrides would be silently unenforced."""
    with pytest.raises(
        ValueError, match=rf"{role}\.network_mode='denylist' is not supported"
    ):
        TaskConfig.model_validate({role: {"network_mode": "denylist"}})


@pytest.mark.parametrize("sandbox", ["modal", "apple-container", "agentcore"])
def test_capability_gate_refuses_denylist_elsewhere(sandbox: str) -> None:
    """Backends without the proxy fail closed on the sandbox policy."""
    config = TaskConfig.model_validate(
        {
            "sandbox": {"network_mode": "denylist", "blocked_hosts": ["arxiv.org"]},
        }
    )

    issues = validate_task_runtime_support(config, sandbox=sandbox)

    assert [(issue.path, issue.reason) for issue in issues] == [
        (
            "sandbox.network_mode",
            f"network_mode='denylist' is not enforced by {sandbox}",
        ),
    ]


def test_rubric_checks_v_network_denylist_outcomes() -> None:
    """The stdlib mirror passes a populated denylist and fails an empty one."""
    assert "denylist" in rubric_checks._VALID_NETWORK_MODES
    by_urls = rubric_checks.network_hardening(
        {"network_mode": "denylist", "blocked_urls": ["https://arxiv.org/abs/1"]}
    )
    by_hosts = rubric_checks.network_hardening(
        {"network_mode": "denylist", "blocked_hosts": ["arxiv.org"]},
        verifier_or_sandbox_pr=True,
    )
    empty = rubric_checks.network_hardening({"network_mode": "denylist"})
    stray = rubric_checks.network_hardening(
        {
            "network_mode": "denylist",
            "blocked_hosts": ["arxiv.org"],
            "allowed_hosts": ["x.com"],
        }
    )

    assert by_urls[0] == "V-NETWORK"
    assert by_urls[1] == "pass"
    assert by_hosts[1] == "pass"
    assert empty[1:] == ("fail", "denylist without blocked_urls/blocked_hosts")
    assert stray[1] == "fail"


def test_review_pack_normalizes_denylist_cells() -> None:
    """The review pack maps a denylist cell onto the V-NETWORK gate with its lists."""
    base = {"task": "citation-check", "agent": "openhands", "network_mode": "denylist"}
    hardened = pack_mod.normalize_cell(
        {**base, "id": "cit-deny", "blocked_hosts": ["arxiv.org"]}
    )
    bare = pack_mod.normalize_cell({**base, "id": "cit-bare"})

    assert pack_mod._cell_network_config(hardened) == "denylist"
    md = pack_mod.hardening_summary_md(
        slots=[], cells=[hardened, bare], verifier_or_sandbox_pr=True
    )
    assert "cit-deny: V-NETWORK=pass" in md
    assert "cit-bare: V-NETWORK=fail" in md
