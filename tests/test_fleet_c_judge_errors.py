"""Guards PR #58 against credential-bearing SDK errors."""

import io

import pytest
from rich.console import Console
from rich.traceback import Traceback

from benchflow.rewards import llm


@pytest.mark.parametrize("model", ["claude-test", "unknown-model"])
@pytest.mark.parametrize("message_has_key", [True, False])
async def test_judge_failure_hides_sdk_locals_and_message(
    monkeypatch, caplog, model, message_has_key
):
    planted = "synthetic-judge-credential-298"

    async def fail(*args):
        api_key = args[-1]["ANTHROPIC_API_KEY"]
        raise RuntimeError(api_key if message_has_key else "provider failed")

    for provider in ("anthropic", "openai", "google"):
        monkeypatch.setattr(llm, f"_call_{provider}", fail)
    with pytest.raises(RuntimeError) as caught:
        await llm.call_judge(
            model, "judge prompt", retries=1, env={"ANTHROPIC_API_KEY": planted}
        )
    # Render the library boundary onward; a caller's own credential variables
    # are outside the boundary the judge can sanitize.
    tb = caught.value.__traceback__
    assert tb is not None and tb.tb_next is not None
    output = io.StringIO()
    Console(file=output, width=160).print(
        Traceback.from_exception(
            type(caught.value), caught.value, tb.tb_next, show_locals=True
        )
    )
    assert planted not in output.getvalue()
    assert planted not in caplog.text
