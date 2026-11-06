"""Whether each task input the agent read reached the model.

Guards what a published run showed: on some routes the model got only
Claude Code's ``PDF file read: /root/proposal.pdf (47KB)`` line for the task's
own input (the chat-completions bridge dropped the document), and nothing in
the trial said so. The proxy's requests now say, per file the agent read,
what its result carried (a document or image block, or text with its length
and hash), and a stub is a warning on the trial and a count in the job.
"""

from __future__ import annotations

import base64
import importlib
import importlib.util
import json
from datetime import datetime

import pytest

from benchflow._utils.evaluation_results import model_call_summary
from benchflow.providers.litellm_logging import callback_module_source
from benchflow.rollout._results import _build_rollout_result
from benchflow.trajectories.model_calls import STUB_TEXT_CHARS, write_model_calls
from tests.test_litellm_bridge_documents import _pdf

PDF = "/root/proposal.pdf"
PLACEHOLDER = f"PDF file read: {PDF} (47KB)"
DOCUMENT = {
    "type": "document",
    "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0x"},
}


def _row(messages, route="openai/glm-5.3"):
    return {
        "request": {"body": {"model": "m", "messages": messages}},
        "response": {
            "status_code": 200,
            "body": {"choices": [{"finish_reason": "stop"}]},
        },
        "metadata": {"call_purpose": "agent", "provider_model": route},
    }


def _anthropic_read(
    call_id, path, result, *, is_error=False, name="Read", arg="file_path"
):
    """The turn Claude Code sends after a Read: its tool_use, then the result."""
    return [
        {"role": "user", "content": "Read /root/proposal.pdf first."},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": call_id, "name": name, "input": {arg: path}}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": result,
                    **({"is_error": True} if is_error else {}),
                }
            ],
        },
    ]


def _bridged_read(call_id, path, text, name="Read"):
    """The same turn as the chat-completions bridge sends it on."""
    return [
        {"role": "user", "content": "Read /root/proposal.pdf first."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps({"file_path": path}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": text},
    ]


def _trial(tmp_path, rows, name="task__1"):
    trial = tmp_path / name
    (trial / "trajectory").mkdir(parents=True)
    (trial / "trajectory/llm_trajectory.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    return trial


def _files(tmp_path, rows):
    summary = write_model_calls(_trial(tmp_path, rows))
    assert summary is not None
    return summary["input_files"]


def test_a_document_block_reached_the_model(tmp_path):
    inputs = _files(
        tmp_path,
        [
            _row(
                _anthropic_read(
                    "t1", PDF, [{"type": "text", "text": PLACEHOLDER}, DOCUMENT]
                )
            )
        ],
    )
    assert (inputs["reads"], inputs["stubs"]) == (1, 0)
    (entry,) = inputs["files"]
    assert (entry["path"], entry["kind"], entry["stub"]) == (PDF, "document", False)


def test_the_placeholder_alone_is_a_stub(tmp_path):
    """The bridge kept the read's text line and dropped its document."""
    inputs = _files(tmp_path, [_row(_bridged_read("c1", PDF, PLACEHOLDER))])
    (entry,) = inputs["files"]
    assert entry["stub"] is True and entry["kind"] == "text"
    assert entry["text_chars"] == len(PLACEHOLDER)
    assert entry["route"] == "openai/glm-5.3" and entry["recovered"] is False
    assert inputs["stubs"] == 1


def test_a_few_characters_for_a_pdf_are_a_stub_its_text_is_not(tmp_path):
    extracted = "Proposal. " * 500
    inputs = _files(
        tmp_path,
        [
            _row(_bridged_read("c1", PDF, "[document omitted]")),
            _row(_bridged_read("c2", "/root/notes.pdf", extracted)),
            _row(_bridged_read("c3", "/root/readme.txt", "ok")),
        ],
    )
    by_path = {entry["path"]: entry for entry in inputs["files"]}
    assert by_path[PDF]["stub"] is True
    assert by_path["/root/notes.pdf"]["stub"] is False
    assert by_path["/root/notes.pdf"]["text_chars"] == len(extracted)
    assert len(by_path["/root/notes.pdf"]["text_sha256"]) == 64
    # A short text file is short, not a stub.
    assert by_path["/root/readme.txt"]["stub"] is False


def test_the_agents_own_files_and_failed_reads_are_not_inputs(tmp_path):
    write = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "w1",
                    "name": "Write",
                    "input": {"file_path": "/app/out.pdf"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "w1", "content": "ok"}],
        },
    ]
    inputs = _files(
        tmp_path,
        [
            _row(write),
            _row(_anthropic_read("r1", "/app/out.pdf", "x")),
            _row(
                _anthropic_read(
                    "r2", "/root/missing.pdf", "File does not exist.", is_error=True
                )
            ),
            _row(
                _bridged_read(
                    "r3",
                    "/root/gone.pdf",
                    "<tool_use_error>File does not exist.</tool_use_error>",
                )
            ),
        ],
    )
    assert inputs is None


def test_a_retried_request_counts_its_read_once(tmp_path):
    turn = _bridged_read("c1", PDF, PLACEHOLDER)
    inputs = _files(tmp_path, [_row(turn), _row(turn)])
    assert inputs["reads"] == 1 and inputs["files"][0]["reads"] == 1


def test_an_agent_that_extracts_the_text_itself_is_marked(tmp_path):
    """After the stub the agent ran pdftotext on the file and got its text
    (2 of 8 sampled trials of one row did)."""
    extract = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "b1",
                    "name": "Bash",
                    "input": {"command": f"pdftotext {PDF} -"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "b1", "content": "Aims. " * 400}
            ],
        },
    ]
    inputs = _files(
        tmp_path, [_row(_bridged_read("c1", PDF, PLACEHOLDER)), _row(extract)]
    )
    (entry,) = inputs["files"]
    assert entry["stub"] is True and entry["recovered"] is True


def test_the_trial_warns_and_the_job_counts_stubs(tmp_path):
    results = {}
    for index, text in enumerate([PLACEHOLDER, "Proposal. " * 500]):
        trial = _trial(
            tmp_path, [_row(_bridged_read("c1", PDF, text))], name=f"t{index}__1"
        )
        write_model_calls(trial)
        _build_rollout_result(
            trial,
            task_name="task",
            rollout_name=trial.name,
            agent="claude-agent-acp",
            agent_name="claude-agent-acp",
            model="m",
            n_tool_calls=1,
            prompts=["task"],
            error=None,
            verifier_error=None,
            trajectory=[],
            partial_trajectory=False,
            rewards={"reward": 0.0},
            started_at=datetime.now(),
            timing={},
        )
        results[trial.name] = json.loads((trial / "result.json").read_text())

    stubbed = results["t0__1"]
    assert stubbed["input_delivery"]["stubs"] == 1
    assert stubbed["warnings"] == [
        f"input file {PDF} may not have reached the model: the agent's read of it "
        f"reached the model as {len(PLACEHOLDER)} characters of text and no "
        "document or image (route openai/glm-5.3)"
    ]
    # Never the score.
    assert stubbed["rewards"] == {"reward": 0.0}
    assert "warnings" not in results["t1__1"]

    block = model_call_summary(results)["model_calls"]
    assert (block["input_reads"], block["input_stubs"], block["input_stub_trials"]) == (
        2,
        1,
        1,
    )


# ----- the bridge to chat completions ---------------------------------------
#
# The proxy records the agent's request, from before LiteLLM's bridge
# translates it: there the read still holds its document. The callback says
# what the bridge made of it (metadata.bridge_tool_results).


@pytest.fixture
def callback(tmp_path, monkeypatch):
    """The proxy's callback module, loaded as the proxy loads it; the bridge
    to chat completions as BenchFlow's proxy config sets it."""
    import litellm

    monkeypatch.setattr(
        litellm, "use_chat_completions_url_for_anthropic_messages", True
    )
    path = tmp_path / "benchflow_litellm_callback.py"
    path.write_text(callback_module_source(), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("benchflow_litellm_callback", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bridged_call(raw: bytes, provider="openai") -> dict:
    return {
        "call_type": "anthropic_messages",
        "custom_llm_provider": provider,
        "model": "glm-5.3",
        "messages": _anthropic_read(
            "toolu_1",
            PDF,
            [{"type": "text", "text": PLACEHOLDER}, _pdf_document(raw)],
        ),
    }


def _pdf_document(raw: bytes) -> dict:
    return {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64.b64encode(raw).decode(),
        },
    }


def _stock_bridge(monkeypatch):
    """LiteLLM's own translation, without BenchFlow's patches: from a fresh
    copy of its module."""
    name = "litellm.llms.anthropic.experimental_pass_through.adapters.transformation"
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.loader is not None
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    adapter = importlib.import_module(name).LiteLLMAnthropicMessagesAdapter
    monkeypatch.setattr(
        adapter,
        "translate_anthropic_messages_to_openai",
        fresh.LiteLLMAnthropicMessagesAdapter.translate_anthropic_messages_to_openai,
    )


def _scan(tmp_path, bridged):
    row = _row(
        _anthropic_read(
            "toolu_1", PDF, [{"type": "text", "text": PLACEHOLDER}, DOCUMENT]
        )
    )
    row["metadata"]["bridge_tool_results"] = bridged
    return _files(tmp_path, [row])


def test_the_bridge_record_shows_the_pdf_reached_the_model_as_text(callback, tmp_path):
    raw = _pdf([["Proposal: ground state energy per site"]])
    bridged = callback._bridge_delivery(_bridged_call(raw))
    entry = bridged["toolu_1"]
    # The read's line, then the PDF's text.
    assert entry["parts"] == ["text", "text"] and entry["chars"] > STUB_TEXT_CHARS
    assert entry["head"].startswith(f"{PLACEHOLDER}\n[PDF converted to text")
    (scanned,) = _scan(tmp_path, bridged)["files"]
    assert (scanned["kind"], scanned["stub"]) == ("text", False)
    assert scanned["text_chars"] == entry["chars"]


def test_without_the_bridge_patch_the_record_shows_the_stub(
    callback, tmp_path, monkeypatch
):
    """LiteLLM 1.91.5's own bridge kept the read's text line and dropped the
    PDF. 1.103.3's keeps the PDF as an image part holding its data URL
    (BerriAI/litellm#38251), which a chat-completions model does not read as
    the document: the record names that part image_url:application/pdf, and
    the read's line is still all the model can read, a stub."""
    _stock_bridge(monkeypatch)
    bridged = callback._bridge_delivery(_bridged_call(_pdf([["Proposal"]])))
    assert bridged["toolu_1"]["parts"] == ["text", "image_url:application/pdf"]
    assert bridged["toolu_1"]["head"] == PLACEHOLDER
    inputs = _scan(tmp_path, bridged)
    (scanned,) = inputs["files"]
    assert (scanned["kind"], scanned["stub"]) == ("text", True)
    assert inputs["stubs"] == 1


def test_a_provider_that_takes_messages_needs_no_bridge_record(callback):
    assert (
        callback._bridge_delivery(_bridged_call(b"%PDF", provider="anthropic")) is None
    )
    plain = _bridged_call(b"%PDF")
    plain["messages"] = _anthropic_read("toolu_1", "/root/a.txt", "text")
    assert callback._bridge_delivery(plain) is None


def _reminded_read() -> list[dict]:
    """Claude Code 2.1.316's read of a PDF in a turn's first tool result: the
    read's line, the document, then the system reminders it appends there
    (the environment update, the tokens left)."""
    result = [
        {"type": "text", "text": f"PDF file read: {PDF} (47KB)"},
        _pdf_document(_pdf([["Proposal: ground state energy per site"]])),
        {
            "type": "text",
            "text": "<system-reminder>\n# Environment update\n - Additional "
            "working directories added:\n  - /root\n</system-reminder>",
        },
        {
            "type": "text",
            "text": "<system-reminder>\n<total_tokens>14998950 tokens left"
            "</total_tokens>\n</system-reminder>",
        },
    ]
    return _anthropic_read("toolu_1", PDF, result)


def _reminded_delivery(callback, tmp_path, messages) -> tuple[dict, dict]:
    bridged = callback._bridge_delivery(
        {
            "call_type": "anthropic_messages",
            "custom_llm_provider": "openai",
            "model": "glm-5.3",
            "messages": messages,
        }
    )
    row = _row(messages)
    row["metadata"]["bridge_tool_results"] = bridged
    (scanned,) = _files(tmp_path, [row])["files"]
    return bridged["toolu_1"], scanned


def test_a_bridged_pdf_read_with_claude_codes_reminders_is_still_a_stub(
    callback, tmp_path, monkeypatch
):
    """LiteLLM's own bridge keeps the text and sends the PDF as an image part
    holding its data URL (BerriAI/litellm#38251; LiteLLM 1.91.5 dropped it),
    so the model reads the read's line and the reminders: a stub, though the
    text runs past 200 characters."""
    _stock_bridge(monkeypatch)
    entry, scanned = _reminded_delivery(callback, tmp_path, _reminded_read())
    assert entry["parts"] == ["text", "image_url:application/pdf", "text", "text"]
    assert entry["chars"] > 200
    assert scanned["stub"] is True


def test_with_the_bridge_patch_the_reminded_read_reaches_the_model_as_text(
    callback, tmp_path
):
    """BenchFlow's bridge patch turns the document between the read's line
    and the reminders into its text, as it does a document at the end."""
    entry, scanned = _reminded_delivery(callback, tmp_path, _reminded_read())
    assert entry["parts"] == ["text", "text", "text", "text"]
    assert entry["head"].startswith(f"{PLACEHOLDER}\n[PDF converted to text")
    assert (scanned["kind"], scanned["stub"]) == ("text", False)
