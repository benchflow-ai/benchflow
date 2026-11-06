"""Guards FIX-RUNTIME-SH diagnostics for the setup failure at 1f02fbc4c."""

from types import SimpleNamespace

import pytest

from benchflow.sandbox import lockdown
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox._exec_receipt import read_receipt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "diagnostic",
    [
        "/bin/sh: eval: line 930: syntax error near unexpected token `}'",
        "sh: 930: Syntax error: end of file unexpected",
        "bash: -c: line 930: unexpected EOF while looking for matching `)'",
    ],
)
async def test_setup_syntax_error_names_shell_and_does_not_blame_modes(
    monkeypatch, diagnostic
):
    calls = []

    async def read(env, command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            # A parser failure has no checked receipt. Preserve that evidence
            # rather than supplying an indistinguishable child exit status.
            return read_receipt(ExecResult(return_code=2, stderr=diagnostic), "a" * 24)
        return ExecResult(
            return_code=0, stdout="/bin/sh -> /usr/bin/bash\nbash 5.1.8(1)-release\n"
        )

    monkeypatch.setattr(lockdown, "_read_exec", read)
    with pytest.raises(RuntimeError) as caught:
        await lockdown.prepare_runtime_tree(
            SimpleNamespace(supports_runtime_kit=True), phase="build"
        )
    message = str(caught.value)
    assert "syntax error in BenchFlow's setup script" in message
    assert "bash 5.1.8(1)-release" in message
    assert "image needs" not in message
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_ownership_failure_retains_the_image_requirement(monkeypatch):
    async def read(*args, **kwargs):
        return ExecResult(
            return_code=1, stderr="benchflow: image has no root-owned /bin/sh"
        )

    monkeypatch.setattr(lockdown, "_read_exec", read)
    with pytest.raises(RuntimeError, match="image needs a root-owned /bin/sh"):
        await lockdown.prepare_runtime_tree(
            SimpleNamespace(supports_runtime_kit=True), phase="build"
        )


@pytest.mark.asyncio
async def test_child_tool_diagnostic_does_not_claim_setup_parse_failure(monkeypatch):
    """A child tool can print a shell error while our setup parses normally."""

    async def read(*args, **kwargs):
        return ExecResult(
            return_code=1, stderr="/bin/sh: syntax error in an image tool"
        )

    monkeypatch.setattr(lockdown, "_read_exec", read)
    with pytest.raises(RuntimeError, match="image needs a root-owned /bin/sh"):
        await lockdown.prepare_runtime_tree(
            SimpleNamespace(supports_runtime_kit=True), phase="build"
        )
