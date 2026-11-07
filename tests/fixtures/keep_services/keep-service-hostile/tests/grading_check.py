"""Collected by test.sh's pytest while the kept service attacks: a conftest.py
or plugin it planted, had pytest loaded it, would have written its marker."""

import os


def test_no_planted_conftest_or_plugin_was_loaded():
    for kind in ("conftest", "plugin"):
        assert not os.path.exists(f"/logs/verifier/planted-{kind}-loaded")
