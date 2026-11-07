"""Guards the required-tool and marker protocols from V3-CONFIRM7 at f656f4a0."""

from types import SimpleNamespace

import pytest

from benchflow._utils.trusted_path import TrustedPathReport, refused_programs
from benchflow.sandbox.lockdown import _check_required_path_tools, _InService
from benchflow.task.config import TaskConfig
from tests._receipt_fakes import receipt_mock


@pytest.mark.parametrize(
    "names", [None, [], ["cmp", "cmp"], ["-tool", "tool name", "\r", "x" * 255]]
)
def test_required_tools_round_trip(names):
    """Guards CONFIRM7-5: omitted and explicitly empty contracts differ."""
    config = TaskConfig.model_validate({"verifier": {"required_tools": names}})
    expected = None if names is None else list(dict.fromkeys(names))
    assert config.verifier.required_tools == expected
    assert (
        TaskConfig.model_validate(config.model_dump()).verifier.required_tools
        == expected
    )


@pytest.mark.parametrize("name", ["", "bin/tool", "tool\x00"])
def test_required_tools_reject_ambiguous_names(name):
    """Guards CONFIRM7-5: dependencies are literal program basenames."""
    with pytest.raises(ValueError, match="program basenames"):
        TaskConfig.model_validate({"verifier": {"required_tools": [name]}})


@pytest.mark.asyncio
@pytest.mark.parametrize("service", ["main", "target"])
@pytest.mark.parametrize("user", ["root", "nobody"])
async def test_contract_probe_uses_the_verifiers_service_and_user(service, user):
    """Guards CONFIRM7-3/5 on every backend through the common exec contract."""
    env = SimpleNamespace(exec=receipt_mock(SimpleNamespace(return_code=0, stdout="")))
    target = _InService(env, service)
    report = TrustedPathReport(
        path="/usr/bin:/bin", dropped=(("bin", "relative"),), service=service
    )
    verifier = TaskConfig.model_validate(
        {"verifier": {"required_tools": ["cmp"], "user": user}}
    ).verifier
    await _check_required_path_tools(target, report, verifier)
    assert env.exec.call_args.kwargs["service"] == service
    assert env.exec.call_args.kwargs["user"] == user


@pytest.mark.parametrize(
    "stdout",
    ["", "1 no", "1 maybe\x00", "1 no\x001 yes\x00", "2 no\x00", "1 no\x00garbage"],
)
def test_invalid_marker_protocol_is_never_no_hits(stdout):
    """Guards CONFIRM7-4: incomplete, duplicate and malformed evidence fails closed."""
    report = TrustedPathReport(path="/usr/bin:/bin", stubs=(("x" * 255, "/opt/open"),))
    with pytest.raises(ValueError, match="refusal marker"):
        refused_programs(report, stdout)


def test_markers_map_all_folder_aliases_without_filename_bytes():
    """Guards CONFIRM7-4 across duplicates, CR, whitespace and NAME_MAX names."""
    report = TrustedPathReport(
        path="/usr/bin:/bin",
        stubs=(("\r", "/opt/a"), ("x" * 255, "/opt/a"), ("\r", "/opt/b ")),
    )
    assert refused_programs(report, "1 yes\x002 no\x00") == [
        ("\r", "/opt/a"),
        ("\r", "/opt/b "),
    ]
