"""benchflow.integrity.urls against the URL Standard's own conformance data.

``tests/fixtures/integrity/wpt_urltestdata_special.json`` is a vendored
subset of WPT's ``url/resources/urltestdata.json``; the file records its
source, commit, the full file's sha256 and the selection rule (absolute
inputs with the http, https, ws or wss scheme; successes with non-ASCII hosts
left out, since this parser refuses IDNA hosts and the callers fail closed).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchflow.integrity.urls import URLFailure, URLUnsupported, parse_special_url

FIXTURE = (
    Path(__file__).parent / "fixtures" / "integrity" / "wpt_urltestdata_special.json"
)
DATA = json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_the_fixture_records_where_it_comes_from():
    assert DATA["source"].startswith("https://github.com/web-platform-tests/wpt/blob/")
    assert DATA["commit"] in DATA["source"] and len(DATA["commit"]) == 40
    assert len(DATA["file_sha256"]) == 64
    assert len(DATA["cases"]) >= 250


@pytest.mark.parametrize(
    "case", DATA["cases"], ids=lambda case: repr(case["input"])[:60]
)
def test_wpt_case(case):
    if case.get("failure"):
        with pytest.raises((URLFailure, URLUnsupported)):
            parse_special_url(case["input"])
        return
    url = parse_special_url(case["input"])
    assert url.scheme + ":" == case["protocol"]
    assert url.host == case["hostname"]
    assert ("" if url.port is None else str(url.port)) == case["port"]
    assert url.path == case["pathname"]
