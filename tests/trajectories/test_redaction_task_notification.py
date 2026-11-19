"""Claude Code's ``<task-notification>`` tags are not masked; keys still are.

The generic ``sk-`` key rule matched inside the tag's word:
``<task-notification>`` came out as ``<ta***REDACTED***>`` ("ta" +
"sk-notification", 12 letters after ``sk-``), in every layer and in released
0.9.2 too. 0.9.3 leaves exactly that word (``task-notification`` or
``task-notifications``, as a whole word, in its escaped forms) and masks
everything else as 0.9.2 does, a key glued to what precedes it included.
Cerebras keys (``csk-``, where the letter before ``sk-`` is the key's own)
are masked whole by a rule of their own.

Every value here is made up. The keys are built from split literals so no
whole key shape sits in the source.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from collections import Counter

import pytest

from benchflow.publish.redact import REDACTED as PUBLISH_MASK
from benchflow.publish.redact import redact_value, redact_value_to_stability
from benchflow.trajectories.export import export_trajectories_to_jsonl
from benchflow.trajectories.types import (
    LLMExchange,
    LLMRequest,
    LLMResponse,
    Trajectory,
    redact_acp_trajectory_jsonl,
    redact_trajectory_obj,
    redact_trajectory_text,
)

# The trajectory writers' mark; the publish step writes its own (PUBLISH_MASK),
# read back here as this one.
MASK = "***REDACTED***"

# A background task's notification as Claude Code writes it into the user turn
# (made-up id, command and path).
TAG = (
    "<task-notification>\n"
    "<task-id>bq7xk2m9</task-id>\n"
    "<output-file>/tmp/claude/tasks/bq7xk2m9.output</output-file>\n"
    "<status>completed</status>\n"
    '<summary>Background command "uv run pytest -q tests/test_export.py" '
    "completed (exit code 0)</summary>\n"
    "</task-notification>"
)
NOTIFICATION = (
    f"{TAG}\nRead the output file to retrieve the result: "
    "/tmp/claude/tasks/bq7xk2m9.output"
)

GENERIC = "sk-" + "FXACPXmadeup0123456789abcdefghijKLMNOP"
HEX = "sk-" + "0f8e1a2b3c4d5e6f7a8b9c0d1e2f3a4b"
SHORT = "sk-" + "a1B2c3D4e5F6"
LETTERS = "sk-" + "FXACPXmadeupAbcdefGhijklMnopqrSt"
LOWER_LETTERS = "sk-" + "fxacpxmadeupabcdefghijklmnopqrst"
CEREBRAS = "c" + "sk-" + "fxacpxmadeup0123456789abcdefghijklmnop0123456789"
ANTHROPIC = "sk-" + "ant-api03-" + "FXACPXmadeup_0123456789abcdefghijKLMNOPqrst"
PROJECT = "sk-" + "proj-" + "FXACPXmadeup_0123456789abcdefghijKLMNOPqrst"


def _event(text: str) -> dict:
    """An ACP user-message event carrying *text*, as the ACP writer stores it."""
    return {
        "type": "user_message",
        "session_id": "sess-fxacpx",
        "content": [{"type": "text", "text": text}],
    }


def _record(text: str) -> dict:
    """A Verifiers export record carrying *text* in the prompt and the completion."""
    return {
        "prompt": [{"role": "user", "content": text}],
        "completion": [{"role": "assistant", "content": f"Noted: {text}"}],
        "info": {"task_id": "fxacpx-task"},
    }


def _layers(text: str, tmp_path) -> dict[str, str]:
    """*text* after each redaction layer 0.9.x has, read back as text."""
    event = _event(text)
    out = tmp_path / "export.jsonl"
    export_trajectories_to_jsonl([_record(text)], out)
    exported = json.loads(out.read_text())
    published, _ = redact_value(_record(text))
    stable, _ = redact_value_to_stability(_record(text))
    exchange = Trajectory(
        session_id="t",
        exchanges=[
            LLMExchange(
                request=LLMRequest(body={"messages": [{"content": text}]}),
                response=LLMResponse(),
            )
        ],
    )
    proxied = json.loads(exchange.to_jsonl(redact_keys=True))
    return {
        "text": redact_trajectory_text(text),
        "json string": json.loads(redact_trajectory_text(json.dumps({"c": text})))["c"],
        "acp jsonl": json.loads(redact_acp_trajectory_jsonl([event]))["content"][0][
            "text"
        ],
        "object": redact_trajectory_obj(event)["content"][0]["text"],
        "object field": redact_trajectory_obj({"content": text})["content"],
        "export": exported["prompt"][0]["content"],
        "export completion": exported["completion"][0]["content"].removeprefix(
            "Noted: "
        ),
        "publish": published["prompt"][0]["content"].replace(PUBLISH_MASK, MASK),
        "publish stable": stable["prompt"][0]["content"].replace(PUBLISH_MASK, MASK),
        "proxy exchange": proxied["request"]["body"]["messages"][0]["content"],
    }


def test_a_task_notification_is_not_masked_in_any_layer(tmp_path):
    for layer, value in _layers(NOTIFICATION, tmp_path).items():
        assert value == NOTIFICATION, layer


def test_the_notification_counts_no_replacement(tmp_path):
    categories: Counter[str] = Counter()
    _, count = redact_value(_record(NOTIFICATION), categories=categories)
    assert count == 0
    assert not categories


@pytest.mark.parametrize(
    "glue",
    [
        "",  # right after the closing tag's `>`
        " ",
        "=",
        ":",
        "_",
        "-",
        "/",
        '"',
        "'",
        "(",
        "\n",
        "\t",
        "\x1b[0m",  # a colour code ends in a letter
        "\x1b[38;5;208m",
        "%0A",  # a URL-encoded line break ends in a letter
    ],
    ids=repr,
)
@pytest.mark.parametrize(
    "key", [GENERIC, ANTHROPIC, PROJECT], ids=["sk", "sk-ant", "sk-proj"]
)
def test_a_key_next_to_the_tag_is_still_masked(tmp_path, glue, key):
    text = f"{TAG}{glue}{key} trailing"
    for layer, value in _layers(text, tmp_path).items():
        assert value == f"{TAG}{glue}{MASK} trailing", layer


def test_a_key_inside_the_tag_is_still_masked(tmp_path):
    text = NOTIFICATION.replace("bq7xk2m9</task-id>", f"{GENERIC}</task-id>", 1)
    expected = NOTIFICATION.replace("bq7xk2m9</task-id>", f"{MASK}</task-id>", 1)
    for layer, value in _layers(text, tmp_path).items():
        assert value == expected, layer


@pytest.mark.parametrize(
    "escape", ["\\n", "\\t", "\\r", "\\u001b[32m", "%0a", "\\x1b[1m"]
)
def test_a_generic_key_after_an_escape_written_as_text_is_masked(escape):
    """Text that arrives already escaped: the letter before `sk-` ends the
    escape, so the key starts a word."""
    assert redact_trajectory_text(f"<task-notification>{escape}{GENERIC}") == (
        f"<task-notification>{escape}{MASK}"
    )


@pytest.mark.parametrize(
    "before",
    [
        "xyz",
        "task",
        "risk",
        "Desk",
        "a1",
        "7",
        "python3agent.py",  # a /proc/<pid>/cmdline dump with its NULs dropped
        "mysql -uroot -p",  # a short option with its value attached
        "n",  # `read` without -r turned a written `\\n` into `n`
    ],
)
@pytest.mark.parametrize(
    "key",
    [GENERIC, HEX, SHORT, LETTERS, LOWER_LETTERS, ANTHROPIC, PROJECT, CEREBRAS],
    ids=["sk", "sk-hex", "sk-12", "sk-letters", "sk-lower", "sk-ant", "sk-proj", "csk"],
)
def test_a_key_glued_after_a_letter_or_a_digit_is_masked_as_in_0_9_2(
    tmp_path, before, key
):
    text = f"{before}{key} trailing"
    for layer, value in _layers(text, tmp_path).items():
        assert value == f"{before}{MASK} trailing", layer


@pytest.mark.parametrize(
    "form",
    [
        "<task-notification>",
        "</task-notification>",
        "<task-notifications>",
        '<task-notification id="x">',  # attributes after a space
        "task-notification",
        "task-notifications",
        "\\u003ctask-notification\\u003e",  # JSON with < and > escaped
        "%3Ctask-notification%3E",  # URL-encoded
        '\\"task-notification\\"',  # an escaped quote
        "&lt;task-notification&gt;",  # HTML entities
        "\\ntask-notification",  # an escaped line break right before the word
        "\x1b[2m<task-notification>\x1b[0m",  # in a coloured terminal log
    ],
)
def test_the_tags_word_is_not_masked_in_any_form(tmp_path, form):
    text = f"see {form} here"
    for layer, value in _layers(text, tmp_path).items():
        assert value == text, layer


@pytest.mark.parametrize(
    "word",
    [
        "/app/task-implementation",
        "risk-assessmentreport",
        "desk-reservations",
        "flask-sqlalchemyextension",
        "task-notificationsfoo",
        "mytask-notification",
        "task-notification_id",
        "task-notification-foo",  # a hyphen after the word makes another name
        "x-task-notification",  # and so does one before it
        "_task-notification",
        "Task-Notification",
        "task-Notification",
        "task-1a2b3c4d5e6f7a8b9c0d",
        "desk-123456789012",
    ],
)
def test_every_other_word_ending_in_sk_is_masked_as_in_0_9_2(tmp_path, word):
    """Only the tag's exact word is left; 0.9.2's other over-masks stay."""
    text = f"see {word} here"
    expected = re.sub(r"sk-[a-zA-Z0-9]{12,}", MASK, text)
    assert expected != text
    for layer, value in _layers(text, tmp_path).items():
        assert value == expected, layer


CEREBRAS_ROWS = {
    "bare": "{k}",
    "echo": "$ echo $CEREBRAS_API_KEY\n{k}",
    "provider error": (
        "openai.AuthenticationError: Error code: 401 - "
        "Incorrect API key provided: {k}. You can find your API key in the console."
    ),
    "--api-key": "llm --api-key {k} -m cerebras/llama-4-scout",
    "positional": 'client = make_client("{k}")',
    "escaped positional": 'client = make_client(\\"{k}\\")',
    "NAME=": "CEREBRAS_API_KEY={k}",
    "JSON": '{{"api_key": "{k}"}}',
    "Bearer": "Authorization: Bearer {k}",
}


@pytest.mark.parametrize("row", list(CEREBRAS_ROWS))
def test_a_cerebras_key_is_masked_whole_in_every_layer(tmp_path, row):
    text = CEREBRAS_ROWS[row].format(k=CEREBRAS)
    secret = CEREBRAS[len("csk-") :]
    for layer, value in _layers(text, tmp_path).items():
        assert secret[:12] not in value and secret[-12:] not in value, layer
        assert f"c{MASK}" not in value, layer
    assert redact_trajectory_text(text) == text.replace(CEREBRAS, MASK)


def _as_0_9_2_masks(text: str) -> str:
    return re.sub(r"sk-[a-zA-Z0-9]{12,}", MASK, text)


ENCODINGS = {
    "raw": lambda text: text,
    "json-escaped": lambda text: json.dumps(text)[1:-1],
    "json-twice": lambda text: json.dumps(json.dumps(text))[1:-1],
    "url-quoted": lambda text: urllib.parse.quote(text, safe=""),
    "py-repr": lambda text: repr(text)[1:-1],
}


def _every_pd_character():
    import unicodedata

    return [
        chr(code) for code in range(0x110000) if unicodedata.category(chr(code)) == "Pd"
    ]


@pytest.mark.parametrize("side", ["before", "after"])
def test_every_dash_on_either_side_makes_another_name(tmp_path, side):
    """Every character of Unicode category Pd, as unicodedata lists them."""
    dashes = _every_pd_character()
    assert "-" in dashes and "\u2010" in dashes and len(dashes) >= 26
    for dash in dashes:
        word = (
            f"{dash}task-notification"
            if side == "before"
            else f"task-notification{dash}"
        )
        text = f"see {word} here"
        for layer, value in _layers(text, tmp_path).items():
            assert value == _as_0_9_2_masks(text), (hex(ord(dash)), layer)


# Any non-ASCII character next to the word, raw or escaped, makes another
# name, masked as 0.9.2 masks it.
NEIGHBOURS = {
    "U+2010 hyphen": "\u2010",
    "U+2011 non-breaking hyphen": "\u2011",
    "U+2012 figure dash": "\u2012",
    "U+2013 en dash": "\u2013",
    "U+2014 em dash": "\u2014",
    "U+2212 minus sign (Sm, not Pd)": "\u2212",
    "U+00A0 no-break space": "\u00a0",
    "U+3002 ideographic full stop": "\u3002",
    "U+00E9 e acute": "\u00e9",
    "U+0663 Arabic-Indic three": "\u0663",
    "U+4E2D CJK": "\u4e2d",
}
FORMS = {
    "bare, after": "see task-notification{c} here",
    "bare, before": "see {c}task-notification here",
    "tag, after": "see <task-notification{c}> here",
    "tag, before": "see <{c}task-notification> here",
    "plural, after": "see task-notifications{c} here",
    "plural, before": "see {c}task-notifications here",
}


@pytest.mark.parametrize("encoding", ["raw", "json-escaped", "url-quoted"])
@pytest.mark.parametrize("form", list(FORMS))
@pytest.mark.parametrize("neighbour", list(NEIGHBOURS))
def test_a_non_ascii_neighbour_makes_another_name(neighbour, form, encoding):
    text = ENCODINGS[encoding](FORMS[form].format(c=NEIGHBOURS[neighbour]))
    expected = _as_0_9_2_masks(text)
    assert expected != text
    assert redact_trajectory_text(text) == expected


@pytest.mark.parametrize("form", list(FORMS))
@pytest.mark.parametrize("neighbour", list(NEIGHBOURS))
def test_a_non_ascii_neighbour_makes_another_name_in_every_layer(
    tmp_path, neighbour, form
):
    text = FORMS[form].format(c=NEIGHBOURS[neighbour])
    for layer, value in _layers(text, tmp_path).items():
        assert value == _as_0_9_2_masks(text), layer


@pytest.mark.parametrize("encoding", list(ENCODINGS))
@pytest.mark.parametrize(
    "text",
    [
        "<task-notification>",
        "</task-notification>",
        '<task-notification id="x">',
        "see task-notifications here",
        "a task-notification arrived",
        TAG,
    ],
)
def test_the_tag_stays_in_every_escaped_form(text, encoding):
    encoded = ENCODINGS[encoding](text)
    assert redact_trajectory_text(encoded) == encoded


# An escape of a character that continues a word, written as text and the
# whole text URL-encoded again: its backslash becomes %5C.
WRITTEN_ESCAPES = [
    "\\u2010",
    "\\u00e9",
    "\\u00a0",
    "\\ud83d\\ude00",
    "\\u4e2d",
    "\\xad",
    "\\xe9",
    "\\xa0",
    "\\u002d",
    "\\u005f",
    "\\u0041",
    "\\u0030",
    "\\x2d",
    "\\x41",
]


@pytest.mark.parametrize(
    "template", ["see {L}task-notification{R} here", "<{L}task-notifications{R}>"]
)
@pytest.mark.parametrize("side", ["L", "R"])
@pytest.mark.parametrize("escape", WRITTEN_ESCAPES)
def test_a_written_escape_url_encoded_again_makes_another_name(escape, side, template):
    raw = template.format(
        L=escape if side == "L" else "", R=escape if side == "R" else ""
    )
    for text in (
        urllib.parse.quote(raw, safe=""),
        urllib.parse.quote(urllib.parse.quote(raw, safe=""), safe=""),
    ):
        assert redact_trajectory_text(text) == _as_0_9_2_masks(text), text
