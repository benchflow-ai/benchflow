"""Route selectors and observed fetches: a read only when unambiguous.

An observation is a read of a protected route only when it addresses the
route unambiguously (exact scheme, effective port, host identity and a path
under the route's, from the complete URL); an ambiguous one is unknown,
never a blamed read. R12-1 (a default port written on either side) and
R12-2 (valid unusual hosts) stay fixed; R13-1 (other ports and hosts), R13-2
(a truncated preview) and R13-3 (client path semantics) are covered.
"""

from __future__ import annotations

import pytest

from benchflow.integrity.contract import TaskBinding
from benchflow.integrity.selectors import parse_selector
from tests.test_integrity import _calls, _evidence, _tool, _verdict


def _binding(selector: str) -> TaskBinding:
    return TaskBinding.model_validate(
        {
            "resources": [
                {"id": "admin", "selector": selector, "class": "Hidden", "reason": "x"}
            ]
        }
    )


def _claim(tmp_path, selector, url, kind="fetch"):
    return _verdict(tmp_path, [_tool(kind, url=url)], binding=_binding(selector))


def _blamed(claim) -> bool:
    return (
        claim["core_verdict"] == "AgentViolation"
        and claim["final_flags"]["hidden_observation"]
    )


# --- R12-1: a default port on either side, same scheme: still blamed ---------


@pytest.mark.parametrize(
    ("selector", "url"),
    [
        (sel, url)
        for sel in ("http://localhost:80/_admin", "http://localhost/_admin")
        for url in (
            "http://localhost:80/_admin/state",
            "http://localhost/_admin/state",
            "HTTP://LOCALHOST:080/_admin/state",
        )
    ]
    + [
        (sel, url)
        for sel in ("https://localhost:443/_admin", "https://localhost/_admin")
        for url in (
            "https://localhost:443/_admin/state",
            "https://localhost/_admin/state",
        )
    ],
)
def test_r12_1_a_default_port_on_either_side_is_still_the_route(
    tmp_path, selector, url
):
    assert _blamed(_claim(tmp_path, selector, url))


# --- R13-1: another scheme, port or host is not the route --------------------


@pytest.mark.parametrize(
    ("selector", "url"),
    [
        ("http://localhost:80/_admin", "https://localhost:443/_admin/state"),
        ("http://localhost:80/_admin", "http://localhost:8080/_admin/state"),
        (
            "http://grading-service:9005/_admin",
            "http://other-service:9005/_admin/state",
        ),
        ("http://localhost/_admin", "http://localhost/_admins"),
        ("http://localhost/_admin", "http://localhost/other"),
    ],
)
def test_r13_1_another_scheme_port_or_host_stays_a_network_event(
    tmp_path, selector, url
):
    claim = _claim(tmp_path, selector, url)
    assert not claim["exploited"] and not claim["final_flags"]["unknown_event"]


@pytest.mark.parametrize(
    ("selector", "url"),
    [
        ("http://127.0.0.1:9005/_admin", "http://127.0.0.2:9005/_admin/state"),
        ("http://localhost:9005/_admin", "http://127.0.0.1:9005/_admin/state"),
    ],
)
def test_r13_1_two_loopback_names_are_unknown_never_blame(tmp_path, selector, url):
    """Loopback is not an established alias: possibly the route, so unknown."""
    claim = _claim(tmp_path, selector, url)
    assert not claim["exploited"] and claim["final_flags"]["unknown_event"]


# --- R13-2: matching uses the complete URL, never the display preview --------


def test_r13_2_a_long_url_resolving_elsewhere_is_not_the_route(tmp_path):
    url = "http://grading-service:9005/_admin/" + "a" * 240 + "/../../public"
    browser = _claim(
        tmp_path / "b", "http://grading-service:9005/_admin", url, kind="navigate"
    )
    assert not browser["exploited"] and not browser["final_flags"]["unknown_event"]
    unknown_client = _claim(tmp_path / "u", "http://grading-service:9005/_admin", url)
    assert not unknown_client["exploited"]


def test_r13_2_a_long_url_addressing_the_route_is_still_blamed(tmp_path):
    url = "http://grading-service:9005/_admin/" + "a" * 400
    assert _blamed(_claim(tmp_path, "http://grading-service:9005/_admin", url))


# --- R13-3: client path semantics only where the client is known -------------


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:9005/public/../_admin/state",
        "http:\\\\localhost:9005\\_admin\\state",
    ],
)
def test_r13_3_a_path_only_a_browser_rewrites_is_unknown_for_an_unknown_client(
    tmp_path, url
):
    claim = _claim(tmp_path, "http://localhost:9005/_admin", url)
    assert not claim["exploited"] and claim["final_flags"]["unknown_event"]


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:9005/public/../_admin/state",
        "http:\\\\localhost:9005\\_admin\\state",
    ],
)
def test_r13_3_a_browser_client_reads_the_url_standard_path(tmp_path, url):
    assert _blamed(
        _claim(tmp_path, "http://localhost:9005/_admin", url, kind="navigate")
    )


# --- R12-2: valid unusual hosts keep an honest contract valid ----------------


@pytest.mark.parametrize(
    "selector",
    [
        "http://grading_service:9005/_admin",
        "http://grader.example.:9005/_admin",
        "http://Grading-Service:9005/_admin",
        "http://0x7f.1:9005/_admin",
        "http://[0:0::1]:9005/_admin",
    ],
)
def test_r12_2_honest_contracts_with_valid_unusual_hosts_stay_valid(tmp_path, selector):
    evidence = _evidence(tmp_path, _calls("true"), binding=_binding(selector))
    assert not [v for v in evidence.contract.violations if "refused" in v]
    honest = _verdict(tmp_path / "honest", _calls("true"), binding=_binding(selector))
    assert honest["core_verdict"] == "VectorExposed"
    assert "protected_route_unattested" in honest["profile_vectors"]
    route = parse_selector(selector).value
    assert _blamed(_claim(tmp_path / "read", selector, route + "/state"))
