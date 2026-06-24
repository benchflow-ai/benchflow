"""Tests for the extended OSWorld metric ports + file getters."""

from __future__ import annotations

import json

from benchflow.adapters.osworld_eval import _get_state, evaluate
from benchflow.adapters.osworld_metrics import (
    check_direct_json_object,
    check_list,
    compare_text_file,
    resolve_metric,
)


def _subst(v):  # identity substitution for tests
    return v


class TestCheckDirectJsonObject:
    def test_exact_match_object(self):
        rules = {"expected": {"a": 1, "b": "x"}}
        assert check_direct_json_object({"a": 1, "b": "x", "c": 9}, rules) == 1.0

    def test_mismatch(self):
        assert check_direct_json_object({"a": 2}, {"expected": {"a": 1}}) == 0.0

    def test_parses_json_string_and_single_quotes(self):
        assert check_direct_json_object("{'a': 1}", {"expected": {"a": 1}}) == 1.0

    def test_expect_in_result_substring(self):
        rules = {"expect_in_result": True, "expected": {"name": "Kili"}}
        assert check_direct_json_object({"name": "Mount Kili"}, rules) == 1.0

    def test_evaluation_failed_sentinel(self):
        rules = {"expected": {"k": "__EVALUATION_FAILED__"}}
        assert check_direct_json_object({"k": "v"}, rules) == 0.0

    def test_none_result(self):
        assert check_direct_json_object(None, {"expected": {"a": 1}}) == 0.0


class TestCheckList:
    def test_expect_and_unexpect(self, tmp_path):
        f = tmp_path / "out.txt"
        f.write_text("alpha\nbeta\ngamma\n")
        assert check_list(str(f), {"expect": ["alpha", "gamma"]}) == 1.0
        assert check_list(str(f), {"expect": ["alpha", "delta"]}) == 0.0
        assert check_list(str(f), {"expect": ["alpha"], "unexpect": ["beta"]}) == 0.0

    def test_none(self):
        assert check_list(None, {"expect": ["x"]}) == 0.0


class TestCompareTextFile:
    def test_equal(self, tmp_path):
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("hello world\n")
        b.write_text("hello world\n")
        assert compare_text_file(str(a), str(b)) == 1.0

    def test_ignore_case_and_blanks(self, tmp_path):
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("Hello   World")
        b.write_text("hello world")
        assert compare_text_file(str(a), str(b)) == 0.0
        assert (
            compare_text_file(str(a), str(b), ignore_case=True, ignore_blanks=True)
            == 1.0
        )

    def test_missing_actual(self):
        assert compare_text_file(None, "/nonexistent") == 0.0


class TestGetters:
    def test_vm_file_local_path(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("hi")
        cfg = {"type": "vm_file", "path": str(f), "dest": "x.txt"}
        assert _get_state(cfg, lambda c, s: "", _subst, str(tmp_path)) == str(f)
        missing = {"type": "vm_file", "path": str(tmp_path / "nope"), "dest": "n"}
        assert _get_state(missing, lambda c, s: "", _subst, str(tmp_path)) is None

    def test_cache_file_path(self, tmp_path):
        cfg = {"type": "cache_file", "path": "diff.out"}
        assert _get_state(cfg, lambda c, s: "", _subst, str(tmp_path)) == str(
            tmp_path / "diff.out"
        )

    def test_rule_returns_rules(self, tmp_path):
        cfg = {"type": "rule", "rules": {"expected": "y"}}
        assert _get_state(cfg, lambda c, s: "", _subst, str(tmp_path)) == {
            "expected": "y"
        }


class TestEvaluateWithOptionsAndFileGetters:
    def test_compare_text_file_end_to_end(self, tmp_path):
        # result = a vm_file the "agent" produced; expected = a local reference file
        res = tmp_path / "res.txt"
        res.write_text("LINE one\n")
        exp = tmp_path / "exp.txt"
        exp.write_text("line  one")
        task = {
            "evaluator": {
                "func": "compare_text_file",
                "result": {"type": "vm_file", "path": str(res), "dest": "res.txt"},
                "expected": {"type": "local", "path": str(exp)},
                "options": {"ignore_case": True, "ignore_blanks": True},
            }
        }
        assert (
            evaluate(task, run_command=lambda c, s: "", cache_dir=str(tmp_path)) == 1.0
        )

    def test_check_direct_json_object_via_command(self, tmp_path):
        task = {
            "evaluator": {
                "func": "check_direct_json_object",
                "result": {"type": "vm_command_line", "command": "echo", "shell": True},
                "expected": {"type": "rule", "rules": {"expected": {"ok": True}}},
            }
        }
        rc = lambda c, s: json.dumps({"ok": True, "extra": 1})  # noqa: E731
        assert evaluate(task, run_command=rc, cache_dir=str(tmp_path)) == 1.0


def test_registry_has_new_metrics():
    for name in ("check_list", "check_direct_json_object", "compare_text_file"):
        assert callable(resolve_metric(name))
