import json
from pathlib import Path

SUMMARY = Path("/app/summary.json")


def summary():
    return json.loads(SUMMARY.read_text())


def test_summary_is_an_object():
    assert isinstance(summary(), dict)


def test_count():
    assert summary()["count"] == 5


def test_mean():
    assert abs(summary()["mean"] - 81.4) < 1e-6


def test_max():
    assert summary()["max"] == {"name": "Grace", "score": 97}
