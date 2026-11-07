"""Guard FF-install-C6 against malformed credentials echoed by cd4d492d."""

import pytest

from benchflow.providers import proxy_requirements as reqs


@pytest.mark.parametrize(
    "item",
    [
        "https:/pkgs.example.invalid/sdk.whl?token=FAKE_TOKEN",
        "https:pkgs.example.invalid/FAKE_TOKEN/sdk.whl",
        "sdk==FAKE_TOKEN?signature=dummy",
        "FAKE_TOKEN!",
    ],
)
def test_invalid_extras_never_echo_unparseable_input(item):
    with pytest.raises(reqs.ProxyExtraRequirementError) as raised:
        reqs.check_extra_requirements(("humanize", item), reqs.PROXY_GROUP)
    message = str(raised.value)
    assert "item 2" in message
    assert "FAKE_TOKEN" not in message
    assert item not in message
    assert "package names with optional versions" in message
