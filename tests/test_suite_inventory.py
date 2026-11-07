"""tools/suite_inventory.py: no test removed or newly skipped without a decision.

Each case writes a tiny base project and a changed copy to ``tmp_path``
(each with its own ``pytest.ini``), runs pytest on them as a subprocess
through the tool, and checks that ``compare`` names exactly the predicted
findings. The JUnit cases use the xunit1 shapes pytest 9.0.3 and
pytest-xdist 3.8.0 write (checked on a build VM, round PY8 GATE-noskip).
"""

from __future__ import annotations

import json
import shlex
import socket
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from tools import suite_inventory as si

_BASE = {
    "pytest.ini": """\
        [pytest]
        addopts = -m "not live"
        markers =
            live: needs the outside world
        """,
    "conftest.py": "collect_ignore = []\n",
    "test_math.py": """\
        import pytest


        def test_add():
            assert 1 + 1 == 2


        def test_sub():
            assert 2 - 1 == 1


        @pytest.mark.parametrize("n", ["a", "b", "c"])
        def test_param(n):
            assert n


        class TestGroup:
            def test_one(self):
                assert True
        """,
    "test_other.py": """\
        def test_other():
            assert True
        """,
}
BASE = {rel: textwrap.dedent(text) for rel, text in _BASE.items()}

DECISIONS = """\
# Decisions

| # | Question | Decision | Reason |
|---|---|---|---|
| PY8-D100 | GATE-noskip example: rename | P2 renamed `test_math.py::test_sub` -> `test_math.py::test_subtract` | names it |

- **PY8-D101 (P2, 2026-10-16):** removed `test_math.py::test_sub` on purpose.
- **PY8-D102 (coordinator, 2026-10-16):** deselected `test_math.py::test_add`: it needs the network.
  - continued: also new-skip `test_math.py::test_sub`.
- **PY8-D102 (DT2, 2026-10-16):** an unrelated entry that reuses the number.
- **PY8-D103 (V4, 2026-10-16):** strict mode now keeps the done key (no ids named).
"""


@pytest.fixture(autouse=True)
def _nested_pytest_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # The nested pytest runs see only their own project and pytest's own plugins.
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    monkeypatch.delenv("PYTEST_PLUGINS", raising=False)


def changed(files: dict[str, str], **edits: str | None) -> dict[str, str]:
    """A copy of ``files``; ``name__py`` keys stand for ``name.py``; None deletes."""
    out = dict(files)
    for key, text in edits.items():
        rel = key.replace("__", ".").replace("_SLASH_", "/")
        if text is None:
            out.pop(rel, None)
        else:
            out[rel] = text
    return out


def make(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def pytest_cmd(*args: str) -> list[str]:
    return [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args]


def run_leg(
    manifest: Path,
    root: Path,
    name: str,
    *,
    role: str = "primary",
    select: tuple[str, ...] = (),
    extra: tuple[str, ...] = (),
    prefix: tuple[str, ...] = (),
) -> int:
    return si.main(
        [
            "run-leg",
            "--manifest",
            str(manifest),
            "--name",
            name,
            "--role",
            role,
            "--junit",
            f"{name}.xml",
            "--root",
            str(root),
            *[f"--select={s}" for s in select],
            "--",
            *prefix,
            *pytest_cmd(*select, *extra),
        ]
    )


def arm(
    tmp_path: Path,
    name: str,
    files: dict[str, str],
    *,
    run: bool = True,
    legs: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
        ("primary", (), ()),
    ),
    retry: tuple[str, ...] = (),
    expect: int = 0,
) -> Path:
    """Write a project, run its legs, and collect its inventory."""
    root = make(tmp_path / name, files)
    out = tmp_path / f"{name}.json"
    args = ["collect", "--root", str(root), "--out", str(out), "--tree-label", name]
    if run:
        manifest = tmp_path / f"{name}-run" / "manifest.json"
        manifest.parent.mkdir()
        for leg_name, select, extra in legs:
            run_leg(manifest, root, leg_name, select=select, extra=extra)
        if retry:
            run_leg(manifest, root, "retry", role="retry", select=retry)
        args += ["--run-manifest", str(manifest)]
    assert si.main(args) == expect
    return out


def compare(
    tmp_path: Path,
    base: Path,
    tip: Path,
    *,
    mode: str = "runtime",
    allow: str = "",
    decisions: str = DECISIONS,
    strict: bool = False,
    only: tuple[str, ...] = (),
    name: str = "cmp",
) -> tuple[int, dict[str, Any]]:
    allow_path = tmp_path / f"{name}-allow.tsv"
    allow_path.write_text(allow)
    decisions_path = tmp_path / f"{name}-DECISIONS.md"
    decisions_path.write_text(decisions)
    out = tmp_path / name
    rc = si.main(
        [
            "compare",
            "--mode",
            mode,
            "--base",
            str(base),
            "--tip",
            str(tip),
            "--allow",
            str(allow_path),
            "--decisions",
            str(decisions_path),
            "--out",
            str(out),
            *(["--strict-allow"] if strict else []),
            *[f"--only={o}" for o in only],
        ]
    )
    return rc, json.loads((tmp_path / f"{name}.json").read_text())


def open_findings(result: dict[str, Any]) -> list[tuple[str, str]]:
    return sorted(
        (f["kind"], f["id"]) for f in result["findings"] if not f["allowed_by"]
    )


def allow_line(kind: str, old: str, new: str, ref: str) -> str:
    return f"{kind}\t{old}\t{new}\t{ref}\towner\treason\n"


MATH_WITHOUT_SUB = BASE["test_math.py"].replace(
    "def test_sub():\n    assert 2 - 1 == 1\n", ""
)
MATH_RENAMED = BASE["test_math.py"].replace("def test_sub()", "def test_subtract()")


# --------------------------------------------------------------------------
# the judge on hidden tests (fails-before: the plain gate passes them)


HIDING_MUTANTS = {
    "deleted test": (
        {"test_math.py": MATH_WITHOUT_SUB},
        [("removed", "test_math.py::test_sub")],
    ),
    "dropped parametrize case": (
        {"test_math.py": BASE["test_math.py"].replace('["a", "b", "c"]', '["a", "b"]')},
        [("removed", "test_math.py::test_param[c]")],
    ),
    "deleted file": (
        {"test_other.py": None},
        [("removed", "test_other.py::test_other")],
    ),
    "module importorskip fires": (
        {
            "test_other.py": 'import pytest\npytest.importorskip("no_such_module_gn")\n\n\n'
            "def test_other():\n    assert True\n"
        },
        [("module-skipped", "test_other.py")],
    ),
    "new collect_ignore entry": (
        {"conftest.py": 'collect_ignore = ["test_other.py"]\n'},
        [("removed", "test_other.py::test_other")],
    ),
    "modifyitems drops an item": (
        {
            "conftest.py": "collect_ignore = []\n\n\n"
            "def pytest_collection_modifyitems(items):\n"
            '    items[:] = [i for i in items if i.name != "test_add"]\n'
        },
        [("removed", "test_math.py::test_add")],
    ),
    "existing test newly marked live": (
        {
            "test_math.py": BASE["test_math.py"].replace(
                "def test_add():", "@pytest.mark.live\ndef test_add():"
            )
        },
        [("deselected", "test_math.py::test_add")],
    ),
    "body pytest.skip": (
        {
            "test_math.py": BASE["test_math.py"].replace(
                "def test_sub():\n", 'def test_sub():\n    pytest.skip("not today")\n'
            )
        },
        [("new-skip", "test_math.py::test_sub")],
    ),
    "new xfail marker on a passing test": (
        {
            "test_math.py": BASE["test_math.py"].replace(
                "def test_add():", '@pytest.mark.xfail(reason="flaky")\ndef test_add():'
            )
        },
        [
            ("new-xfail", "test_math.py::test_add"),
            (
                "xfail-site",
                'test_math.py::test_add pytest-collected {"args": {"tuple": []}, "kind": "pytest.mark.xfail", "kwargs": {"dict": {"reason": "flaky"}}}',
            ),
            (
                "xfail-site",
                "test_math.py::test_add pytest.mark.xfail(reason='flaky')",
            ),
        ],
    ),
}


def _edits(spec: dict[str, str | None]) -> dict[str, str]:
    out = dict(BASE)
    for rel, text in spec.items():
        if text is None:
            out.pop(rel)
        else:
            out[rel] = text
    return out


@pytest.mark.parametrize("mutant", sorted(HIDING_MUTANTS))
def test_the_plain_gate_passes_hidden_tests_that_the_inventory_flags(
    tmp_path: Path, mutant: str
) -> None:
    edits, predicted = HIDING_MUTANTS[mutant]
    base = arm(tmp_path, "base", BASE)
    tip_files = _edits(edits)
    tip = arm(tmp_path, "tip", tip_files)
    plain = subprocess.run(
        pytest_cmd(), cwd=tmp_path / "tip", capture_output=True, text=True, check=False
    )
    rc, result = compare(tmp_path, base, tip)
    print(
        f"{mutant}: plain pytest exit {plain.returncode} "
        f"({plain.stdout.strip().splitlines()[-1]}); inventory compare exit {rc}: "
        f"{open_findings(result)}"
    )
    assert plain.returncode == 0
    assert rc == 1
    assert open_findings(result) == sorted(predicted)


def test_an_unchanged_pair_exits_0_with_an_empty_report(tmp_path: Path) -> None:
    base = arm(tmp_path, "base", BASE)
    tip = arm(tmp_path, "tip", BASE)
    rc, result = compare(tmp_path, base, tip)
    assert rc == 0
    assert result["findings"] == []
    assert result["allow_errors"] == []
    assert result["invalid"] == []
    assert all(not v for v in result["report_only"].values())
    assert result["review"] == []
    markdown = (tmp_path / "cmp.md").read_text()
    assert markdown.rstrip().endswith("No findings.")
    rc, result = compare(tmp_path, base, tip, mode="collection", name="cmp2")
    assert rc == 0
    assert result["findings"] == []


def test_a_firing_importorskip_is_one_module_finding_not_a_test_id(
    tmp_path: Path,
) -> None:
    edits, _ = HIDING_MUTANTS["module importorskip fires"]
    base = arm(tmp_path, "base", BASE)
    tip = arm(tmp_path, "tip", _edits(edits))
    rc, result = compare(tmp_path, base, tip)
    assert rc == 1
    (finding,) = result["findings"]
    assert finding["kind"] == "module-skipped"
    assert finding["hidden_ids"] == ["test_other.py::test_other"]
    assert "no_such_module_gn" in finding["reason"]


def test_a_new_test_that_skips_is_a_new_skip(tmp_path: Path) -> None:
    tip_files = changed(
        BASE,
        test_math__py=BASE["test_math.py"]
        + '\n\n@pytest.mark.skip(reason="later")\ndef test_new():\n    assert True\n',
    )
    base = arm(tmp_path, "base", BASE)
    tip = arm(tmp_path, "tip", tip_files)
    rc, result = compare(tmp_path, base, tip)
    assert rc == 1
    assert open_findings(result) == [("new-skip", "test_math.py::test_new")]
    (finding,) = result["findings"]
    assert finding["new_test"] is True
    assert finding["reason"] == "later"
    assert len(result["report_only"]["new_skip_sites"]) == 2
    assert any(
        s["site"] == "test_math.py::test_new pytest.mark.skip(reason='later')"
        for s in result["report_only"]["new_skip_sites"]
    )


def test_a_new_xfail_call_is_a_site_and_a_runtime_xfail(tmp_path: Path) -> None:
    tip_files = changed(
        BASE,
        test_math__py=BASE["test_math.py"].replace(
            "def test_sub():\n", 'def test_sub():\n    pytest.xfail("known")\n'
        ),
    )
    base = arm(tmp_path, "base", BASE)
    tip = arm(tmp_path, "tip", tip_files)
    rc, result = compare(tmp_path, base, tip)
    assert rc == 1
    assert open_findings(result) == [
        ("new-xfail", "test_math.py::test_sub"),
        ("xfail-site", "test_math.py::test_sub pytest.xfail('known')"),
    ]
    # Collection mode sees the site; the run outcome is reported as unchecked.
    rc, result = compare(tmp_path, base, tip, mode="collection", name="cmp2")
    assert rc == 1
    assert open_findings(result) == [
        ("xfail-site", "test_math.py::test_sub pytest.xfail('known')")
    ]
    assert any("not checked" in p for p in result["report_only"]["pending"])


def test_a_passing_xfail_reads_as_passed_in_junit_so_only_the_site_catches_it(
    tmp_path: Path,
) -> None:
    edits, _ = HIDING_MUTANTS["new xfail marker on a passing test"]
    tip = arm(tmp_path, "tip", _edits(edits))
    inventory = json.loads(tip.read_text())
    assert (
        si.primary_outcomes(inventory)["test_math.py::test_add"]["outcome"] == "passed"
    )


def test_an_xfail_site_swap_in_one_file_and_skip_to_xfail_are_flagged(
    tmp_path: Path,
) -> None:
    base_files = changed(
        BASE,
        test_math__py=BASE["test_math.py"]
        .replace("def test_add():", '@pytest.mark.xfail(reason="a")\ndef test_add():')
        .replace("def test_sub():\n", 'def test_sub():\n    pytest.skip("env")\n'),
    )
    tip_files = changed(
        BASE,
        test_math__py=BASE["test_math.py"]
        .replace(
            "    def test_one(self):",
            '    @pytest.mark.xfail(reason="a")\n    def test_one(self):',
        )
        .replace("def test_sub():\n", 'def test_sub():\n    pytest.xfail("env")\n'),
    )
    base = arm(tmp_path, "base", base_files)
    tip = arm(tmp_path, "tip", tip_files)
    rc, result = compare(tmp_path, base, tip)
    assert rc == 1
    assert open_findings(result) == [
        ("new-xfail", "test_math.py::TestGroup::test_one"),
        ("new-xfail", "test_math.py::test_sub"),
        (
            "xfail-site",
            "test_math.py::TestGroup.test_one pytest.mark.xfail(reason='a')",
        ),
        (
            "xfail-site",
            'test_math.py::TestGroup::test_one pytest-collected {"args": {"tuple": []}, "kind": "pytest.mark.xfail", "kwargs": {"dict": {"reason": "a"}}}',
        ),
        ("xfail-site", "test_math.py::test_sub pytest.xfail('env')"),
    ]
    assert result["report_only"]["removed_xfail_sites"] == [
        'test_math.py::test_add pytest-collected {"args": {"tuple": []}, "kind": "pytest.mark.xfail", "kwargs": {"dict": {"reason": "a"}}}',
        "test_math.py::test_add pytest.mark.xfail(reason='a')",
    ]
    (new_xfail,) = [
        f
        for f in result["findings"]
        if f["kind"] == "new-xfail" and f["id"] == "test_math.py::test_sub"
    ]
    assert new_xfail["base_exemptions"] == {"skip": "env"}


# --------------------------------------------------------------------------
# renames and allow lines


def test_a_rename_needs_a_renamed_line_whose_new_id_exists_and_passes(
    tmp_path: Path,
) -> None:
    base = arm(tmp_path, "base", BASE)
    tip = arm(tmp_path, "tip", changed(BASE, test_math__py=MATH_RENAMED))
    rc, result = compare(tmp_path, base, tip)
    assert rc == 1
    assert open_findings(result) == [("removed", "test_math.py::test_sub")]
    assert result["report_only"]["new_ids"] == ["test_math.py::test_subtract"]

    allow = allow_line(
        "renamed", "test_math.py::test_sub", "test_math.py::test_subtract", "PY8-D100"
    )
    rc, result = compare(tmp_path, base, tip, allow=allow, name="ok")
    assert rc == 0, result
    assert result["findings"][0]["allowed_by"].startswith("allow line 1")

    # Collection mode: the new id exists; its pass is pending.
    rc, result = compare(
        tmp_path, base, tip, allow=allow, mode="collection", name="col"
    )
    assert rc == 0
    assert any(
        "pass is checked in runtime mode" in p for p in result["report_only"]["pending"]
    )

    # A rename whose new id does not exist at the tip is not authorized.
    wrong = allow_line(
        "renamed", "test_math.py::test_sub", "test_math.py::test_nope", "PY8-D100"
    )
    rc, result = compare(tmp_path, base, tip, allow=wrong, name="wrong")
    assert rc == 1
    assert result["allow_errors"][0]["kind"] == "unauthorized-allow"


def test_a_renamed_test_that_fails_at_the_tip_is_not_authorized(tmp_path: Path) -> None:
    failing = MATH_RENAMED.replace("assert 2 - 1 == 1", "assert 2 - 1 == 0")
    base = arm(tmp_path, "base", BASE)
    tip = arm(tmp_path, "tip", changed(BASE, test_math__py=failing))
    allow = allow_line(
        "renamed", "test_math.py::test_sub", "test_math.py::test_subtract", "PY8-D100"
    )
    rc, result = compare(tmp_path, base, tip, allow=allow)
    assert rc == 1
    assert "did not pass" in result["allow_errors"][0]["why"]
    assert open_findings(result) == [("removed", "test_math.py::test_sub")]


def test_a_rename_into_a_deselected_id_is_not_authorized_in_either_mode(
    tmp_path: Path,
) -> None:
    # The base ran test_sub; the tip renames it and marks it live, so the
    # round's selection drops it. Collection mode can see that (U8's job).
    live = BASE["test_math.py"].replace(
        "def test_sub()", "@pytest.mark.live\ndef test_subtract()"
    )
    base = arm(tmp_path, "base", BASE)
    tip = arm(tmp_path, "tip", changed(BASE, test_math__py=live))
    allow = allow_line(
        "renamed", "test_math.py::test_sub", "test_math.py::test_subtract", "PY8-D100"
    )
    for mode in ("collection", "runtime"):
        rc, result = compare(tmp_path, base, tip, allow=allow, mode=mode, name=mode)
        assert rc == 1, (mode, result)
        assert [e["why"] for e in result["allow_errors"]] == [
            "renamed: new id test_math.py::test_subtract is not selected at the tip"
        ]
        assert open_findings(result) == [("removed", "test_math.py::test_sub")]


def test_a_failed_new_id_that_passes_its_serial_retry_counts_as_passing(
    tmp_path: Path,
) -> None:
    flaky = MATH_RENAMED.replace(
        "def test_subtract():\n",
        'def test_subtract():\n    import os\n    assert not os.environ.get("GN_PRIMARY")\n',
    )
    base = arm(tmp_path, "base", BASE)
    root = make(tmp_path / "tip", changed(BASE, test_math__py=flaky))
    manifest = tmp_path / "tip-run" / "manifest.json"
    manifest.parent.mkdir()
    run_leg(manifest, root, "primary", prefix=("env", "GN_PRIMARY=1"))
    run_leg(
        manifest, root, "retry", role="retry", select=("test_math.py::test_subtract",)
    )
    tip = tmp_path / "tip.json"
    assert (
        si.main(
            [
                "collect",
                "--root",
                str(root),
                "--out",
                str(tip),
                "--run-manifest",
                str(manifest),
            ]
        )
        == 0
    )
    allow = allow_line(
        "renamed", "test_math.py::test_sub", "test_math.py::test_subtract", "PY8-D100"
    )
    rc, result = compare(tmp_path, base, tip, allow=allow)
    assert rc == 0, result
    assert result["report_only"]["outcome_changes"] == []


@pytest.mark.parametrize(
    ("ref", "why"),
    [
        ("PY8-D999", "matches no decision entry"),
        ("PY8-D103", "does not name"),
        ("PY8-D102", "matches 2 decision entries"),
        ("D101", "is not PY8-D<number> or PR-<number>"),
    ],
)
def test_an_allow_line_needs_a_unique_decision_naming_its_kind_and_id(
    tmp_path: Path, ref: str, why: str
) -> None:
    base = arm(tmp_path, "base", BASE, run=False)
    tip = arm(tmp_path, "tip", changed(BASE, test_math__py=MATH_WITHOUT_SUB), run=False)
    allow = allow_line("removed", "test_math.py::test_sub", "-", ref)
    rc, result = compare(tmp_path, base, tip, allow=allow, mode="collection")
    assert rc == 1
    (error,) = result["allow_errors"]
    assert error["kind"] == "unauthorized-allow"
    assert why in error["why"]
    assert open_findings(result) == [("removed", "test_math.py::test_sub")]


def test_a_decision_naming_an_unrelated_id_or_kind_does_not_authorize(
    tmp_path: Path,
) -> None:
    base = arm(tmp_path, "base", BASE, run=False)
    tip = arm(tmp_path, "tip", changed(BASE, test_math__py=MATH_WITHOUT_SUB), run=False)
    # PY8-D102@coordinator names test_math.py::test_add (deselected), not this removal.
    allow = allow_line("removed", "test_math.py::test_sub", "-", "PY8-D102@coordinator")
    rc, result = compare(tmp_path, base, tip, allow=allow, mode="collection")
    assert rc == 1
    assert "the kind 'removed'" in result["allow_errors"][0]["why"]
    # The right entry authorizes it.
    allow = allow_line("removed", "test_math.py::test_sub", "-", "PY8-D101")
    rc, result = compare(tmp_path, base, tip, allow=allow, mode="collection", name="ok")
    assert rc == 0, result


def test_a_reused_decision_number_resolves_with_a_discriminator(tmp_path: Path) -> None:
    tip_files = changed(
        BASE,
        test_math__py=BASE["test_math.py"].replace(
            "def test_add():", "@pytest.mark.live\ndef test_add():"
        ),
    )
    base = arm(tmp_path, "base", BASE, run=False)
    tip = arm(tmp_path, "tip", tip_files, run=False)
    allow = allow_line(
        "deselected", "test_math.py::test_add", "-", "PY8-D102@coordinator"
    )
    rc, result = compare(tmp_path, base, tip, allow=allow, mode="collection")
    assert rc == 0, result
    allow = allow_line("deselected", "test_math.py::test_add", "-", "PY8-D102@DT2")
    rc, result = compare(
        tmp_path, base, tip, allow=allow, mode="collection", name="dt2"
    )
    assert rc == 1


def test_public_ci_cites_the_pull_request_description(tmp_path: Path) -> None:
    base = arm(tmp_path, "base", BASE, run=False)
    tip = arm(tmp_path, "tip", changed(BASE, test_math__py=MATH_RENAMED), run=False)
    allow = allow_line(
        "renamed", "test_math.py::test_sub", "test_math.py::test_subtract", "PR-42"
    )
    body = (
        "Renames a test.\n\n- **PR-42:** renamed `test_math.py::test_sub` -> "
        "`test_math.py::test_subtract` (the subtraction helper was renamed).\n"
    )
    rc, result = compare(
        tmp_path, base, tip, allow=allow, decisions=body, mode="collection"
    )
    assert rc == 0, result
    rc, result = compare(
        tmp_path,
        base,
        tip,
        allow=allow,
        decisions="Renames a test.\n",
        mode="collection",
        name="no",
    )
    assert rc == 1


def test_a_stale_allow_line_is_red_only_under_strict_allow(tmp_path: Path) -> None:
    base = arm(tmp_path, "base", BASE, run=False)
    tip = arm(tmp_path, "tip", BASE, run=False)
    allow = allow_line("removed", "test_math.py::test_sub", "-", "PY8-D101")
    rc, result = compare(tmp_path, base, tip, allow=allow, mode="collection")
    assert rc == 0
    assert result["report_only"]["stale_allow"][0]["authorization"] == "ok"
    rc, result = compare(
        tmp_path, base, tip, allow=allow, mode="collection", strict=True, name="s"
    )
    assert rc == 1
    assert result["allow_errors"][0]["kind"] == "stale-allow"


def test_a_malformed_allow_file_is_invalid_input(tmp_path: Path) -> None:
    base = arm(tmp_path, "base", BASE, run=False)
    allow_path = tmp_path / "allow.tsv"
    allow_path.write_text("removed\ttest_math.py::test_sub\tPY8-D101\n")
    rc = si.main(
        [
            "compare",
            "--mode",
            "collection",
            "--base",
            str(base),
            "--tip",
            str(base),
            "--allow",
            str(allow_path),
        ]
    )
    assert rc == 2
    allow_path.write_text("renamed\ttest_math.py::test_sub\t-\tPY8-D100\towner\twhy\n")
    rc = si.main(
        [
            "compare",
            "--mode",
            "collection",
            "--base",
            str(base),
            "--tip",
            str(base),
            "--allow",
            str(allow_path),
        ]
    )
    assert rc == 2


def test_decision_entries_are_table_rows_and_bullets_with_continuations() -> None:
    entries = si.parse_decisions(DECISIONS)
    assert [e["number"] for e in entries] == [
        "PY8-D100",
        "PY8-D101",
        "PY8-D102",
        "PY8-D102",
        "PY8-D103",
    ]
    assert "new-skip `test_math.py::test_sub`" in entries[2]["text"]
    assert "coordinator" in entries[2]["header"]
    assert "GATE-noskip example" in entries[0]["header"]


def test_an_exact_id_is_not_named_by_a_longer_id() -> None:
    text = (
        "removed `tests/a.py::test_xy`, tests/a.py::test_x[1] and tests/a.py::T::test_x"
    )
    assert not si._names_exactly(text, "tests/a.py::test_x")
    assert not si._names_exactly(text, "a.py::test_xy")
    assert si._names_exactly(text, "tests/a.py::test_xy")
    assert si._names_exactly("renamed: `tests/a.py::test_x`.", "tests/a.py::test_x")
    assert si._names_exactly("renamed: x", "renamed")


# --------------------------------------------------------------------------
# run legs (a parallel primary leg, the separate module, retries)


SPLIT_LEGS = (
    ("primary-parallel", ("--ignore=test_other.py",), ()),
    ("primary-serial", ("test_other.py",), ()),
)


def test_split_primary_legs_union_to_one_selection_each_id_once(tmp_path: Path) -> None:
    base = arm(tmp_path, "base", BASE, legs=SPLIT_LEGS)
    tip = arm(tmp_path, "tip", BASE, legs=SPLIT_LEGS)
    inventory = json.loads(base.read_text())
    assert inventory["selected_ids"] == inventory["all_ids"]
    assert len(inventory["selected_ids"]) == 7
    covered = inventory["runtime"]["primary_cover"]
    assert covered["test_other.py::test_other"] == ["primary-serial"]
    assert all(len(legs) == 1 for legs in covered.values())
    rc, result = compare(tmp_path, base, tip)
    assert rc == 0, result


def test_a_selected_id_missing_from_the_primary_junit_is_not_run_even_after_a_retry(
    tmp_path: Path,
) -> None:
    base = arm(tmp_path, "base", BASE)
    tip = arm(
        tmp_path,
        "tip",
        BASE,
        legs=(("primary", (), ("--deselect", "test_math.py::test_add")),),
        retry=("test_math.py::test_add",),
    )
    rc, result = compare(tmp_path, base, tip)
    assert rc == 1
    assert open_findings(result) == [("not-run", "test_math.py::test_add")]


def test_a_retry_cannot_erase_a_primary_skip(tmp_path: Path) -> None:
    skipping = BASE["test_math.py"].replace(
        "def test_sub():\n",
        "def test_sub():\n    import os\n"
        '    if os.environ.get("GN_PRIMARY"):\n        pytest.skip("primary only")\n',
    )
    base = arm(tmp_path, "base", BASE)
    root = make(tmp_path / "tip", changed(BASE, test_math__py=skipping))
    manifest = tmp_path / "tip-run" / "manifest.json"
    manifest.parent.mkdir()
    run_leg(manifest, root, "primary", prefix=("env", "GN_PRIMARY=1"))
    run_leg(manifest, root, "retry", role="retry", select=("test_math.py::test_sub",))
    tip = tmp_path / "tip.json"
    assert (
        si.main(
            [
                "collect",
                "--root",
                str(root),
                "--out",
                str(tip),
                "--run-manifest",
                str(manifest),
            ]
        )
        == 0
    )
    rc, result = compare(tmp_path, base, tip)
    assert rc == 1
    assert open_findings(result) == [("new-skip", "test_math.py::test_sub")]


def test_a_skip_seen_only_in_a_retry_leg_still_counts(tmp_path: Path) -> None:
    skipping = BASE["test_math.py"].replace(
        "def test_sub():\n",
        "def test_sub():\n    import os\n"
        '    assert not os.environ.get("GN_PRIMARY")\n'
        '    pytest.skip("retry skip")\n',
    )
    base = arm(tmp_path, "base", BASE)
    root = make(tmp_path / "tip", changed(BASE, test_math__py=skipping))
    manifest = tmp_path / "tip-run" / "manifest.json"
    manifest.parent.mkdir()
    run_leg(manifest, root, "primary", prefix=("env", "GN_PRIMARY=1"))
    run_leg(manifest, root, "retry", role="retry", select=("test_math.py::test_sub",))
    tip = tmp_path / "tip.json"
    assert (
        si.main(
            [
                "collect",
                "--root",
                str(root),
                "--out",
                str(tip),
                "--run-manifest",
                str(manifest),
            ]
        )
        == 0
    )
    rc, result = compare(tmp_path, base, tip)
    assert rc == 1
    assert open_findings(result) == [("new-skip", "test_math.py::test_sub")]


def test_a_deleted_primary_junit_is_invalid_evidence(tmp_path: Path) -> None:
    root = make(tmp_path / "tip", BASE)
    manifest = tmp_path / "run" / "manifest.json"
    manifest.parent.mkdir()
    for leg_name, select, extra in SPLIT_LEGS:
        run_leg(manifest, root, leg_name, select=select, extra=extra)
    (manifest.parent / "primary-serial.xml").unlink()
    out = tmp_path / "tip.json"
    assert (
        si.main(
            [
                "collect",
                "--root",
                str(root),
                "--out",
                str(out),
                "--run-manifest",
                str(manifest),
            ]
        )
        == 2
    )
    errors = json.loads(out.read_text())["evidence_errors"]
    assert any("JUnit file missing" in e for e in errors)


def test_a_changed_junit_and_an_interrupted_leg_are_invalid_evidence(
    tmp_path: Path,
) -> None:
    interrupting = BASE["test_math.py"].replace(
        "def test_sub():\n", "def test_sub():\n    raise KeyboardInterrupt\n"
    )
    root = make(tmp_path / "tip", changed(BASE, test_math__py=interrupting))
    manifest = tmp_path / "run" / "manifest.json"
    manifest.parent.mkdir()
    assert run_leg(manifest, root, "primary") == 2
    out = tmp_path / "tip.json"
    assert (
        si.main(
            [
                "collect",
                "--root",
                str(root),
                "--out",
                str(out),
                "--run-manifest",
                str(manifest),
            ]
        )
        == 2
    )
    errors = json.loads(out.read_text())["evidence_errors"]
    assert any("exited 2" in e for e in errors)

    root2 = make(tmp_path / "tip2", BASE)
    manifest2 = tmp_path / "run2" / "manifest.json"
    manifest2.parent.mkdir()
    run_leg(manifest2, root2, "primary")
    junit = manifest2.parent / "primary.xml"
    junit.write_text(junit.read_text().replace('name="test_add"', 'name="test_add2"'))
    out2 = tmp_path / "tip2.json"
    assert (
        si.main(
            [
                "collect",
                "--root",
                str(root2),
                "--out",
                str(out2),
                "--run-manifest",
                str(manifest2),
            ]
        )
        == 2
    )
    errors = json.loads(out2.read_text())["evidence_errors"]
    assert any("changed since the leg ran" in e for e in errors)
    assert any("unmatched JUnit testcase" in e for e in errors)


def test_a_retry_leg_takes_ids_with_spaces_from_a_file(tmp_path: Path) -> None:
    files = changed(
        BASE,
        test_other__py='import pytest\n\n\n@pytest.mark.parametrize("v", ["x y"])\n'
        "def test_other(v):\n    assert v\n",
    )
    root = make(tmp_path / "p", files)
    manifest = tmp_path / "run" / "manifest.json"
    manifest.parent.mkdir()
    run_leg(manifest, root, "primary")
    ids = tmp_path / "failed.txt"
    ids.write_text("test_other.py::test_other[x y]\ntest_math.py::test_add\n")
    rc = si.main(
        ["run-leg", "--manifest", str(manifest), "--name", "retry", "--role", "retry",
         "--junit", "retry.xml", "--root", str(root), "--ids-from", str(ids), "--", *pytest_cmd()]
    )  # fmt: skip
    assert rc == 0
    out = tmp_path / "p.json"
    assert (
        si.main(
            [
                "collect",
                "--root",
                str(root),
                "--out",
                str(out),
                "--run-manifest",
                str(manifest),
            ]
        )
        == 0
    )
    retry = json.loads(out.read_text())["runtime"]["legs"][1]
    assert retry["select_args"] == [
        "test_other.py::test_other[x y]",
        "test_math.py::test_add",
    ]
    assert sorted(retry["outcomes"]) == [
        "test_math.py::test_add",
        "test_other.py::test_other[x y]",
    ]


def test_a_leg_naming_a_collect_ignored_file_is_invalid_evidence(
    tmp_path: Path,
) -> None:
    # pytest collects a file named on the command line despite collect_ignore.
    files = changed(BASE, conftest__py='collect_ignore = ["test_other.py"]\n')
    out = arm(
        tmp_path,
        "p",
        files,
        legs=(("primary", ("test_math.py", "test_other.py"), ()),),
        expect=2,
    )
    errors = json.loads(out.read_text())["evidence_errors"]
    assert any("whole-tree collection hides" in e for e in errors)


def test_run_leg_refuses_a_reused_junit_and_a_select_arg_not_in_the_command(
    tmp_path: Path,
) -> None:
    root = make(tmp_path / "p", BASE)
    manifest = tmp_path / "run" / "manifest.json"
    manifest.parent.mkdir()
    run_leg(manifest, root, "primary")
    rc = si.main(
        ["run-leg", "--manifest", str(manifest), "--name", "again", "--role", "primary",
         "--junit", "primary.xml", "--root", str(root), "--", *pytest_cmd()]
    )  # fmt: skip
    assert rc == 2
    rc = si.main(
        ["run-leg", "--manifest", str(manifest), "--name", "sel", "--role", "primary",
         "--junit", "sel.xml", "--root", str(root), "--select=test_other.py", "--", *pytest_cmd()]
    )  # fmt: skip
    assert rc == 2


# --------------------------------------------------------------------------
# collection-only and runtime modes


def test_runtime_mode_refuses_an_arm_without_run_evidence(tmp_path: Path) -> None:
    base = arm(tmp_path, "base", BASE, run=False)
    tip = arm(tmp_path, "tip", BASE)
    rc, result = compare(tmp_path, base, tip)
    assert rc == 2
    assert any("no run evidence" in e for e in result["invalid"])
    rc, result = compare(tmp_path, base, tip, mode="collection", name="col")
    assert rc == 0


def test_runtime_mode_refuses_incompatible_provenance(tmp_path: Path) -> None:
    base = arm(tmp_path, "base", BASE)
    tip = arm(tmp_path, "tip", BASE)
    data = json.loads(base.read_text())
    data["provenance"]["host"] = "another-vm"
    data["provenance"]["env"]["compared"]["BENCHFLOW_DETERMINISTIC_SANDBOX"] = "daytona"
    base.write_text(json.dumps(data))
    rc, result = compare(tmp_path, base, tip)
    assert rc == 2
    assert any("host" in e for e in result["invalid"])
    assert any("BENCHFLOW_DETERMINISTIC_SANDBOX" in e for e in result["invalid"])
    rc, result = compare(tmp_path, base, tip, mode="collection", name="col")
    assert rc == 0
    assert any("host" in r for r in result["review"])


def collect_with(root: Path, out: Path, *extra: str) -> int:
    return si.main(["collect", "--root", str(root), "--out", str(out), *extra])


BODY_SKIP = BASE["test_math.py"].replace(
    "def test_sub():\n", 'def test_sub():\n    pytest.skip("later")\n'
)


def test_the_base_manifest_cannot_stand_in_for_the_tips_runs(tmp_path: Path) -> None:
    # A copy-paste slip: the tip (which only adds a body skip) is collected
    # with the base's manifest, so the base's outcomes would read as the tip's.
    base = arm(tmp_path, "base", BASE)
    tip_root = make(tmp_path / "tip", changed(BASE, test_math__py=BODY_SKIP))
    base_manifest = tmp_path / "base-run" / "manifest.json"
    tip = tmp_path / "tip.json"
    assert collect_with(tip_root, tip, "--run-manifest", str(base_manifest)) == 2
    errors = json.loads(tip.read_text())["evidence_errors"]
    assert any("ran with cwd" in e for e in errors), errors
    assert any("ran with tree" in e for e in errors), errors
    rc, result = compare(tmp_path, base, tip)
    assert any("tip: leg primary: ran with cwd" in e for e in result["invalid"])
    assert rc == 2


def test_a_leg_from_another_host_or_other_tree_contents_is_invalid(
    tmp_path: Path,
) -> None:
    # Another VM's run: the leg names another host.
    root = make(tmp_path / "p", BASE)
    manifest = tmp_path / "run" / "manifest.json"
    manifest.parent.mkdir()
    run_leg(manifest, root, "primary")
    data = json.loads(manifest.read_text())
    data["legs"][0]["host"] = "another-vm"
    manifest.write_text(json.dumps(data))
    out = tmp_path / "p.json"
    assert collect_with(root, out, "--run-manifest", str(manifest)) == 2
    errors = json.loads(out.read_text())["evidence_errors"]
    assert errors == [
        "leg primary: ran with host 'another-vm', not this arm's "
        f"{socket.gethostname()!r} (another arm's or another machine's run)"
    ]

    # Another tree at the same path (a gate folder re-extracted for a new tip).
    root2 = make(tmp_path / "q", BASE)
    manifest2 = tmp_path / "run2" / "manifest.json"
    manifest2.parent.mkdir()
    run_leg(manifest2, root2, "primary")
    first = tmp_path / "q-first.json"
    assert collect_with(root2, first) == 0
    (root2 / "test_math.py").write_text(BODY_SKIP)
    out2 = tmp_path / "q.json"
    assert collect_with(root2, out2, "--run-manifest", str(manifest2)) == 2
    errors = json.loads(out2.read_text())["evidence_errors"]
    assert len(errors) == 1 and "ran with tree" in errors[0], errors
    # --collection-from reuse refuses the changed contents as well.
    reused = tmp_path / "q-reused.json"
    assert collect_with(root2, reused, "--collection-from", str(first)) == 2


def test_job_folders_a_leg_writes_to_the_root_do_not_break_the_tree_check(
    tmp_path: Path,
) -> None:
    # The deterministic tier's `bench` runs write ./jobs/... into the tree
    # during the first leg; the second leg and collect still see the same tree.
    writer = (
        "from pathlib import Path\n\n\ndef test_writes_a_job():\n"
        '    Path("jobs/run-1").mkdir(parents=True, exist_ok=True)\n'
        '    Path("jobs/run-1/config.json").write_text("{}")\n'
    )
    root = make(tmp_path / "p", changed(BASE, test_writer__py=writer))
    manifest = tmp_path / "run" / "manifest.json"
    manifest.parent.mkdir()
    for leg_name, select, extra in SPLIT_LEGS:
        run_leg(manifest, root, leg_name, select=select, extra=extra)
    assert (root / "jobs" / "run-1" / "config.json").exists()
    out = tmp_path / "p.json"
    assert collect_with(root, out, "--run-manifest", str(manifest)) == 0
    # A jobs folder below the root is source, not run output.
    (root / "sub" / "jobs").mkdir(parents=True)
    (root / "sub" / "jobs" / "x.txt").write_text("x")
    out2 = tmp_path / "p2.json"
    assert collect_with(root, out2, "--run-manifest", str(manifest)) == 2


def test_runtime_mode_compares_the_legs_environments_not_only_collects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The base's legs ran under another sandbox setting than the tip's; both
    # collects ran in one shell, so their own provenance matches.
    invs = {}
    for name, value in (("base", "docker"), ("tip", None)):
        root = make(tmp_path / name, BASE)
        manifest = tmp_path / f"{name}-run" / "manifest.json"
        manifest.parent.mkdir()
        if value:
            monkeypatch.setenv("BENCHFLOW_E2E_SANDBOX", value)
        run_leg(manifest, root, "primary")
        monkeypatch.delenv("BENCHFLOW_E2E_SANDBOX", raising=False)
        invs[name] = tmp_path / f"{name}.json"
        assert collect_with(root, invs[name], "--run-manifest", str(manifest)) == 0
    rc, result = compare(tmp_path, invs["base"], invs["tip"])
    assert rc == 2
    assert result["invalid"] == [
        "incompatible arms: leg environment BENCHFLOW_E2E_SANDBOX: "
        "base 'docker', tip None"
    ]
    rc, result = compare(tmp_path, invs["base"], invs["tip"], mode="collection")
    assert rc == 0


def test_environment_values_are_hashed_never_written_or_printed_in_clear(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Credential-bearing values under recorded prefixes whose names do not
    # look secret (all fake). Each arm gets its own, so compare reports them.
    fakes = {
        "ANTHROPIC_CUSTOM_HEADERS": "Authorization: Bearer FAKE-SECRET-{n}",
        "OPENAI_BASE_URL": "https://user:FAKE-SECRET-{n}@proxy.example/v1",
        "LITELLM_BASE_URL": "https://proxy.example/v1?api-key=FAKE-SECRET-{n}",
        "UV_INDEX_URL": "https://__tok__:FAKE-SECRET-{n}@index.example/simple",
    }
    invs = {}
    for n, name in enumerate(("base", "tip"), 1):
        for var, value in fakes.items():
            monkeypatch.setenv(var, value.format(n=n))
        monkeypatch.setenv("BENCHFLOW_DETERMINISTIC_SANDBOX", f"plain-{n}")
        invs[name] = arm(tmp_path, name, BASE)
    rc, result = compare(tmp_path, invs["base"], invs["tip"])
    assert rc == 2
    named = " ".join(result["invalid"])
    for var in fakes:
        assert f"environment {var}: base 'sha256:" in named
    # An allowlisted value stays readable.
    assert "base 'plain-1', tip 'plain-2'" in named
    printed = capsys.readouterr()
    assert "FAKE-SECRET" not in printed.out + printed.err
    written = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert any(p.name == "manifest.json" for p in written)
    leaked = [p.name for p in written if b"FAKE-SECRET" in p.read_bytes()]
    assert leaked == []


def test_runtime_mode_refuses_a_base_missing_outcomes(tmp_path: Path) -> None:
    base = arm(
        tmp_path,
        "base",
        BASE,
        legs=(("primary", (), ("--deselect", "test_math.py::test_add")),),
    )
    tip = arm(tmp_path, "tip", BASE)
    rc, result = compare(tmp_path, base, tip)
    assert rc == 2
    assert any("test_math.py::test_add" in e for e in result["invalid"])


def test_a_collection_error_in_both_arms_is_an_inherited_blind_spot(
    tmp_path: Path,
) -> None:
    files = changed(BASE, test_broken__py='raise ImportError("boom")\n')
    base = arm(tmp_path, "base", files, run=False)
    tip = arm(tmp_path, "tip", files, run=False)
    rc, result = compare(tmp_path, base, tip, mode="collection")
    assert rc == 0
    assert result["report_only"]["inherited_blind_spots"][0]["id"] == "test_broken.py"
    markdown = (tmp_path / "cmp.md").read_text()
    assert "inherited blind spots; their tests are not verified" in markdown


def test_a_module_that_starts_erroring_is_a_module_error(tmp_path: Path) -> None:
    base = arm(tmp_path, "base", BASE, run=False)
    tip = arm(
        tmp_path,
        "tip",
        changed(
            BASE,
            test_other__py='raise ImportError("boom")\n\n\ndef test_other():\n    pass\n',
        ),
        run=False,
    )
    rc, result = compare(tmp_path, base, tip, mode="collection")
    assert rc == 1
    assert open_findings(result) == [("module-error", "test_other.py")]
    assert result["findings"][0]["hidden_ids"] == ["test_other.py::test_other"]


def test_only_limits_findings_to_the_declared_paths_and_says_so(tmp_path: Path) -> None:
    base = arm(tmp_path, "base", BASE, run=False)
    tip = arm(
        tmp_path,
        "tip",
        changed(BASE, test_math__py=MATH_WITHOUT_SUB, test_other__py=None),
        run=False,
    )
    rc, result = compare(
        tmp_path, base, tip, mode="collection", only=("test_other.py",)
    )
    assert rc == 1
    assert open_findings(result) == [("removed", "test_other.py::test_other")]
    assert result["scope"] == ["test_other.py"]
    assert "Scope limited to: test_other.py" in (tmp_path / "cmp.md").read_text()


def test_two_collects_of_one_tree_give_identical_id_lists(tmp_path: Path) -> None:
    root = make(tmp_path / "p", BASE)
    for n in (1, 2):
        assert (
            si.main(
                ["collect", "--root", str(root), "--out", str(tmp_path / f"{n}.json")]
            )
            == 0
        )
        si.main(
            [
                "ids",
                "--inventory",
                str(tmp_path / f"{n}.json"),
                "--out",
                str(tmp_path / f"{n}.txt"),
            ]
        )
    assert (tmp_path / "1.txt").read_bytes() == (tmp_path / "2.txt").read_bytes()
    assert len((tmp_path / "1.txt").read_text().splitlines()) == 7


def test_skips_lists_every_skip_xfail_and_module_skip(tmp_path: Path) -> None:
    files = changed(
        BASE,
        test_math__py=BASE["test_math.py"].replace(
            "def test_sub():\n", 'def test_sub():\n    pytest.skip("no tool")\n'
        ),
        test_mod__py='import pytest\npytest.importorskip("no_such_module_gn")\n',
    )
    inventory = arm(tmp_path, "p", files)
    si.main(
        ["skips", "--inventory", str(inventory), "--out", str(tmp_path / "skips.txt")]
    )
    lines = (tmp_path / "skips.txt").read_text().splitlines()
    assert lines[0] == "test_math.py::test_sub\tskip\tno tool"
    assert lines[1].startswith("test_mod.py\tmodule-skip\t")


def test_a_bare_junit_gives_outcomes_but_no_runtime_verdict(tmp_path: Path) -> None:
    root = make(tmp_path / "p", BASE)
    junit = tmp_path / "plain.xml"
    plain = subprocess.run(
        pytest_cmd(f"--junitxml={junit}", "-o", "junit_family=xunit1"),
        cwd=root,
        check=False,
    )
    assert plain.returncode == 0
    out = tmp_path / "p.json"
    assert (
        si.main(
            ["collect", "--root", str(root), "--out", str(out), "--junit", str(junit)]
        )
        == 0
    )
    inventory = json.loads(out.read_text())
    assert len(si.primary_outcomes(inventory)) == 7
    rc, result = compare(tmp_path, out, out)
    assert rc == 2
    assert any("exit codes unknown" in e for e in result["invalid"])


def test_the_tool_copy_is_pinned_by_hash(tmp_path: Path) -> None:
    root = make(tmp_path / "p", BASE)
    out = tmp_path / "x.json"
    rc = si.main(
        [
            "collect",
            "--root",
            str(root),
            "--out",
            str(out),
            "--expect-tool-sha256",
            "0" * 64,
        ]
    )
    assert rc == 2
    real = si.sha256_file(Path(si.__file__))
    assert (
        si.main(
            [
                "collect",
                "--root",
                str(root),
                "--out",
                str(out),
                "--expect-tool-sha256",
                real,
            ]
        )
        == 0
    )
    assert json.loads(out.read_text())["tool"]["sha256"] == real


# --------------------------------------------------------------------------
# JUnit identity contracts (pytest 9.0.3 xunit1, pytest-xdist 3.8.0)


def _junit(cases: str) -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?><testsuites name="pytest tests">'
        f'<testsuite name="pytest" tests="1">{cases}</testsuite></testsuites>'
    )


IDS = [
    "test_a.py::TestK::test_m",
    "test_a.py::TestK::test_p[a::b]",
    "test_a.py::TestK::test_p[x y]",
    "test_a.py::TestK::test_p[\\x07bell]",
    "test_a.py::test_teardown",
    "test_a.py::test_setup",
    "sub/test_c.py::TestA::test_b",
]


def _read(
    tmp_path: Path, cases: str, ids: tuple[str, ...] = tuple(IDS)
) -> tuple[dict, list, list]:
    path = tmp_path / "leg.xml"
    path.write_text(_junit(cases))
    return si.read_leg_junit(path, si.JunitIndex(ids))


def test_class_param_and_escaped_ids_map_back_to_collected_ids(tmp_path: Path) -> None:
    reports, modules, errors = _read(
        tmp_path,
        '<testcase classname="test_a.TestK" name="test_m" file="test_a.py" line="4"/>'
        '<testcase classname="test_a.TestK" name="test_p[a::b]" file="test_a.py"/>'
        '<testcase classname="test_a.TestK" name="test_p[x y]" file="test_a.py"/>'
        '<testcase classname="test_a.TestK" name="test_p[\\x07bell]" file="test_a.py"/>'
        '<testcase classname="sub.test_c.TestA" name="test_b" file="sub/test_c.py"/>',
    )
    assert errors == []
    assert modules == []
    assert sorted(reports) == sorted([*IDS[:4], "sub/test_c.py::TestA::test_b"])
    assert si.junit_key("t.py::test_p[\x07]") == ("t", "test_p[#x07]")


def test_setup_and_teardown_errors_and_two_reports_for_one_node(tmp_path: Path) -> None:
    reports, _, errors = _read(
        tmp_path,
        '<testcase classname="test_a" name="test_teardown" file="test_a.py">'
        '<failure message="assert 0">x</failure></testcase>'
        '<testcase classname="test_a" name="test_teardown" file="test_a.py">'
        '<error message="failed on teardown with &quot;RuntimeError&quot;">x</error></testcase>'
        '<testcase classname="test_a" name="test_setup" file="test_a.py">'
        '<error message="failed on setup with &quot;RuntimeError: setup boom&quot;">x</error>'
        "</testcase>",
    )
    assert errors == []
    assert si.outcome_of(reports["test_a.py::test_teardown"]) == "failed"
    assert sum(c["tag"] == "testcase" for c in reports["test_a.py::test_teardown"]) == 2
    assert si.outcome_of(reports["test_a.py::test_setup"]) == "error"


def test_unmatched_and_ambiguous_addresses_are_evidence_errors(tmp_path: Path) -> None:
    _, _, errors = _read(
        tmp_path, '<testcase classname="test_a" name="test_gone" file="test_a.py"/>'
    )
    assert any("unmatched" in e for e in errors)
    # tests/a/b.py and tests/a.b.py both mangle to classname "tests.a.b".
    ids = ("tests/a/b.py::test_x", "tests/a.b.py::test_x")
    _, _, errors = _read(
        tmp_path,
        '<testcase classname="tests.a.b" name="test_x" file="tests/a/b.py"/>',
        ids,
    )
    assert errors == []
    _, _, errors = _read(
        tmp_path,
        '<testcase classname="tests.a.b" name="test_x" file="tests/base.py"/>',
        ids,
    )
    assert any("ambiguous" in e for e in errors)


def test_an_inherited_test_maps_back_though_junit_names_its_defining_file(
    tmp_path: Path,
) -> None:
    files = changed(
        BASE,
        shared_checks__py="class Checks:\n    def test_shared(self):\n        assert True\n",
        test_child__py="from shared_checks import Checks\n\n\nclass TestChild(Checks):\n    pass\n",
    )
    inventory = json.loads(arm(tmp_path, "p", files).read_text())
    assert inventory["evidence_errors"] == []
    junit = (tmp_path / "p-run" / "primary.xml").read_text()
    assert 'name="test_shared" file="shared_checks.py"' in junit
    outcome = si.primary_outcomes(inventory)["test_child.py::TestChild::test_shared"]
    assert outcome["outcome"] == "passed"


def test_module_collection_reports_are_modules_not_test_ids(tmp_path: Path) -> None:
    reports, modules, errors = _read(
        tmp_path,
        '<testcase classname="" name="test_modskip" file="test_modskip.py">'
        "<skipped message=\"collection skipped\">('/x/test_modskip.py', 2, "
        "\"Skipped: could not import 'nope'\")</skipped></testcase>"
        '<testcase classname="" name="sub.test_err" file="sub/test_err.py">'
        '<error message="collection failure">E ImportError: boom</error></testcase>',
    )
    assert errors == []
    assert reports == {}
    assert [(m["id"], m["outcome"]) for m in modules] == [
        ("test_modskip.py", "skipped"),
        ("sub/test_err.py", "error"),
    ]


@pytest.mark.parametrize(
    ("content", "why"),
    [
        (
            '<?xml version="1.0"?><testsuites><testsuite><testcase classname="test_a"',
            "invalid or truncated",
        ),
        ("", "invalid or truncated"),
        ('<?xml version="1.0"?><html/>', "root <html>"),
    ],
)
def test_invalid_or_truncated_xml_is_an_error(
    tmp_path: Path, content: str, why: str
) -> None:
    path = tmp_path / "leg.xml"
    path.write_text(content)
    with pytest.raises(si.InventoryError, match=why):
        si.read_leg_junit(path, si.JunitIndex(IDS))


def test_an_xdist_worker_crash_and_an_internal_error_are_evidence_errors(
    tmp_path: Path,
) -> None:
    _, _, errors = _read(
        tmp_path,
        '<testcase classname="test_a" name="test_setup" file="test_a.py">'
        "<error message=\"failed on setup with &quot;worker 'gw1' crashed while running "
        "'test_a.py::test_setup'&quot;\">x</error></testcase>"
        '<testcase classname="pytest" name="internal"><error message="internal error">x</error>'
        "</testcase>",
    )
    assert any("xdist worker crashed" in e for e in errors)
    assert any("internal error" in e for e in errors)


def test_skip_and_xfail_reports_keep_their_kind_and_reason(tmp_path: Path) -> None:
    reports, _, _ = _read(
        tmp_path,
        '<testcase classname="test_a.TestK" name="test_m" file="test_a.py">'
        '<skipped type="pytest.skip" message="body reason">x</skipped></testcase>'
        '<testcase classname="test_a" name="test_setup" file="test_a.py">'
        '<skipped type="pytest.xfail" message="known"/></testcase>',
    )
    assert si.exemptions_of(reports["test_a.py::TestK::test_m"]) == {
        "skip": "body reason"
    }
    assert si.outcome_of(reports["test_a.py::test_setup"]) == "xfailed"


# --------------------------------------------------------------------------
# the static site inventory


def _sites(tmp_path: Path, source: str) -> dict[str, Any]:
    path = tmp_path / "test_s.py"
    path.write_text(textwrap.dedent(source))
    return si.scan_file(path, "test_s.py")


def test_sweep_command_credentials_never_reach_records(tmp_path: Path) -> None:
    """Guards against 9b7edb1c argv leaks."""
    root = make(tmp_path / "tree", BASE)
    manifest = tmp_path / "records" / "manifest.json"
    manifest.parent.mkdir()
    markers = ("DUMMY-ARGV-KEY", "DUMMY-URL-PASSWORD", "DUMMY-QUERY-TOKEN")
    assert (
        run_leg(
            manifest,
            root,
            "primary",
            prefix=("env", f"OPENAI_API_KEY={markers[0]}"),
            extra=(
                "-o",
                f"index_url=https://user:{markers[1]}@example.invalid/?token={markers[2]}",
            ),
        )
        == 0
    )
    out = tmp_path / "inventory.json"
    assert collect_with(root, out, "--run-manifest", str(manifest)) == 0
    for path in (manifest, out):
        assert all(marker not in path.read_text() for marker in markers)


def test_sweep_verbose_collection_cannot_look_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards the fix against 9b7edb1c accepting an unparsed verbose listing."""
    monkeypatch.setenv("PYTEST_ADDOPTS", "-v")
    root = make(tmp_path / "tree", BASE)
    out = tmp_path / "inventory.json"
    rc = collect_with(root, out)
    inv = json.loads(out.read_text())
    assert rc == 2 or len(inv["all_ids"]) == 7
    assert rc != 0 or len(inv["selected_ids"]) == 7


def test_sweep_all_collection_clears_environment_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards the fix against 9b7edb1c hiding deleted live tests."""
    monkeypatch.setenv("PYTEST_ADDOPTS", "-m 'not live'")
    files = changed(
        BASE, test_live__py="import pytest\n@pytest.mark.live\ndef test_live(): pass\n"
    )
    base = arm(tmp_path, "base", files, run=False)
    tip = arm(tmp_path, "tip", BASE, run=False)
    rc, result = compare(tmp_path, base, tip, mode="collection", strict=True)
    assert rc == 1
    assert ("removed", "test_live.py::test_live") in open_findings(result)
    inv = json.loads(base.read_text())
    assert "test_live.py::test_live" not in inv["selected_ids"]


def test_sweep_local_aliases_cannot_hide_module_xfail(tmp_path: Path) -> None:
    """Guards the fix against 9b7edb1c merging unrelated lexical scopes."""
    source = """\
        import pytest as p
        @p.mark.xfail(strict=False)
        def test_a(): assert True
        def test_b():
            import unittest as p
            assert p.TestCase is not None
        def test_c():
            import pytest as q
            @q.mark.xfail(strict=False)
            def inner(): pass
        class TestD:
            import unittest as p
            @p.expectedFailure
            def test_d(self): p.xfail("module alias in a method body")
        """
    sites = _sites(tmp_path, source)["sites"]
    assert {(s["owner"], s["kind"]) for s in sites} == {
        ("test_a", "pytest.mark.xfail"),
        ("test_c.inner", "pytest.mark.xfail"),
        ("TestD.test_d", "unittest.expectedFailure"),
        ("TestD.test_d", "pytest.xfail"),
    }


@pytest.mark.parametrize(
    "change", ["environment", "pytest_args", "tests_roots", "python"]
)
def test_sweep_collection_reuse_checks_current_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    """Guards the fix against 9b7edb1c reusing incompatible collection."""
    root = make(tmp_path / "tree", BASE)
    first = tmp_path / "first.json"
    assert collect_with(root, first) == 0
    extra: tuple[str, ...] = ()
    if change == "environment":
        monkeypatch.setenv("BENCHFLOW_FAST_TESTS", "1")
    elif change == "pytest_args":
        extra = ("--pytest-arg=-k", "--pytest-arg=test_add")
    elif change == "tests_roots":
        extra = ("--tests-root=test_math.py",)
    else:
        monkeypatch.setattr(si.platform, "python_version", lambda: "0.0.0")
    assert (
        collect_with(
            root, tmp_path / "reuse.json", "--collection-from", str(first), *extra
        )
        == 2
    )


def test_sweep_relative_junit_uses_manifest_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards the fix against 9b7edb1c changing relative output destinations."""
    monkeypatch.chdir(tmp_path)
    make(Path("tree"), BASE)
    Path("results").mkdir()
    manifest = Path("results/manifest.json")
    assert run_leg(manifest, Path("tree"), "primary") == 0
    assert Path("results/primary.xml").is_file()
    assert json.loads(manifest.read_text())["legs"][0]["junit_sha256"]
    assert (
        collect_with(
            Path("tree"), Path("inventory.json"), "--run-manifest", str(manifest)
        )
        == 0
    )


def test_sweep_runtime_refuses_a_foreign_interpreter(tmp_path: Path) -> None:
    """Guards the fix against 9b7edb1c recording the wrapper's interpreter."""
    root = make(tmp_path / "tree", BASE)
    foreign = tmp_path / "foreign-python"
    foreign.symlink_to(sys.executable)
    assert (
        si.main(
            [
                "run-leg",
                "--root",
                str(root),
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--name",
                "primary",
                "--role",
                "primary",
                "--junit",
                "primary.xml",
                "--",
                str(foreign),
                "-m",
                "pytest",
                "-q",
            ]
        )
        == 2
    )
    assert not (tmp_path / "manifest.json").exists()


def test_sweep_directory_symlink_invalidates_cached_collection(tmp_path: Path) -> None:
    """Guards the fix against 9b7edb1c omitting directory links from its digest."""
    root = make(
        tmp_path / "tree",
        {
            "pytest.ini": "[pytest]\ntestpaths = tests\n",
            "variants/a/test_case.py": "def test_a(): pass\n",
            "variants/b/test_case.py": "import pytest\n@pytest.mark.xfail(strict=False)\ndef test_a(): pass\n",
        },
    )
    link = root / "tests"
    link.symlink_to("variants/a", target_is_directory=True)
    first = tmp_path / "first.json"
    assert collect_with(root, first) == 0
    link.unlink()
    link.symlink_to("variants/b", target_is_directory=True)
    assert (
        collect_with(root, tmp_path / "reuse.json", "--collection-from", str(first))
        == 2
    )


def test_sites_resolve_aliases_module_marks_and_param_marks(tmp_path: Path) -> None:
    result = _sites(
        tmp_path,
        """\
        import pytest as pt
        from pytest import mark
        from pytest import skip as sk

        xf = pt.mark.xfail(reason="alias")
        pytestmark = mark.skipif(True, reason="module")


        @xf
        def test_a():
            sk("now")


        @pt.mark.parametrize("v", [1, pt.param(2, marks=mark.xfail(reason="p"))])
        def test_b(v):
            assert v


        def test_c():
            code = "pytest.xfail('in a string')"
            assert code
        """,
    )
    assert result["parse_error"] is None
    keys = sorted((s["owner"], s["kind"], s["expr"]) for s in result["sites"])
    assert keys == [
        ("<module>", "pytest.mark.skipif", "pytest.mark.skipif(True, reason='module')"),
        ("<module>", "pytest.mark.xfail", "pytest.mark.xfail(reason='alias')"),
        ("test_a", "pytest.mark.xfail", "pytest.mark.xfail(reason='alias')"),
        ("test_a", "pytest.skip", "pytest.skip('now')"),
        ("test_b", "pytest.mark.xfail", "pytest.mark.xfail(reason='p')"),
    ]


def test_moving_an_xfail_to_another_test_in_the_file_is_a_new_site(
    tmp_path: Path,
) -> None:
    before = _sites(
        tmp_path,
        "import pytest\n\n@pytest.mark.xfail\ndef test_a():\n    pass\n\ndef test_b():\n    pass\n",
    )
    after = _sites(
        tmp_path,
        "import pytest\n\ndef test_a():\n    pass\n\n@pytest.mark.xfail\ndef test_b():\n    pass\n",
    )
    assert [si.site_key(s) for s in before["sites"]] == [
        "test_s.py::test_a pytest.mark.xfail"
    ]
    assert [si.site_key(s) for s in after["sites"]] == [
        "test_s.py::test_b pytest.mark.xfail"
    ]


def test_unittest_spellings_and_raised_skips_are_sites(tmp_path: Path) -> None:
    result = _sites(
        tmp_path,
        """\
        import unittest
        import pytest


        class TestU(unittest.TestCase):
            @unittest.expectedFailure
            def test_x(self):
                self.skipTest("later")

            @unittest.skipIf(True, "never")
            def test_y(self):
                raise pytest.skip.Exception("raised")
        """,
    )
    kinds = sorted((s["owner"], s["kind"], s["family"]) for s in result["sites"])
    assert kinds == [
        ("TestU.test_x", "unittest.TestCase.skipTest", "skip"),
        ("TestU.test_x", "unittest.expectedFailure", "xfail"),
        ("TestU.test_y", "raise pytest.skip.Exception", "skip"),
        ("TestU.test_y", "unittest.skipIf", "skip"),
    ]


def test_dynamic_marks_are_reported_for_review(tmp_path: Path) -> None:
    result = _sites(
        tmp_path,
        """\
        import pytest

        NAME = "xfail"


        def test_d(request):
            request.node.add_marker(getattr(pytest.mark, NAME))
            assert True
        """,
    )
    whats = sorted(d["what"] for d in result["dynamic"])
    assert whats == [
        "add_marker(<not a literal mark>)",
        "escaped exemption namespace",
        "getattr(pytest.mark, <dynamic>)",
        "unmodeled reflective lookup",
    ]


def test_fewer_asserts_in_a_surviving_test_are_reported_not_gated(
    tmp_path: Path,
) -> None:
    weaker = BASE["test_math.py"].replace("assert 2 - 1 == 1", "pass")
    base = arm(tmp_path, "base", BASE, run=False)
    tip = arm(tmp_path, "tip", changed(BASE, test_math__py=weaker), run=False)
    rc, result = compare(tmp_path, base, tip, mode="collection")
    assert rc == 0
    assert result["report_only"]["assertion_drops"] == [
        {"test": "test_math.py::test_sub", "base": 1, "tip": 0}
    ]


def test_the_compare_report_is_written_as_markdown_and_json(tmp_path: Path) -> None:
    base = arm(tmp_path, "base", BASE, run=False)
    tip = arm(tmp_path, "tip", changed(BASE, test_math__py=MATH_WITHOUT_SUB), run=False)
    compare(tmp_path, base, tip, mode="collection")
    markdown = (tmp_path / "cmp.md").read_text()
    assert "## Findings without an allow line" in markdown
    assert "- removed: test_math.py::test_sub" in markdown
    data = json.loads((tmp_path / "cmp.json").read_text())
    assert data["exit_code"] == 1


@pytest.mark.parametrize("location", ["leg", "pytest_arg", "old_manifest"])
def test_confirm2_selection_records_redact_credentials(
    tmp_path: Path, location: str
) -> None:
    """Guards against raw selections retained at 4f7a30a3."""
    root = make(tmp_path / "tree", BASE)
    marker = "DUMMY-SELECTION-PASSWORD"
    override = f"index_url=https://user:{marker}@example.invalid/"
    out = tmp_path / "inventory.json"
    if location == "old_manifest":
        manifest = tmp_path / "manifest.json"
        assert run_leg(manifest, root, "primary") == 0
        data = json.loads(manifest.read_text())
        data["legs"][0]["select_args"] = ["-o", override]
        manifest.write_text(json.dumps(data))
        extra = ("--run-manifest", str(manifest))
    elif location == "leg":
        extra = (f"--leg=-o {override}",)
    else:
        extra = ("--pytest-arg=-o", f"--pytest-arg={override}")
    assert collect_with(root, out, *extra) == 0
    assert marker not in out.read_text()


def test_confirm2_case_verbosity_cannot_hide_deselection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards against the empty fallback at 4f7a30a3."""
    monkeypatch.setenv("PYTEST_ADDOPTS", "-v -o verbosity_test_cases=0")
    files = {
        "pytest.ini": BASE["pytest.ini"],
        "test_a.py": "def test_a(): pass\ndef test_b(): pass\n",
    }
    base = arm(tmp_path, "base", files, run=False)
    tip = arm(
        tmp_path,
        "tip",
        {
            **files,
            "test_a.py": "import pytest\n@pytest.mark.live\ndef test_a(): pass\ndef test_b(): pass\n",
        },
        run=False,
    )
    rc, result = compare(tmp_path, base, tip, mode="collection")
    assert rc == 1
    assert ("deselected", "test_a.py::test_a") in open_findings(result)


def test_confirm2_unknown_fallback_never_establishes_empty_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards against unrecognized output accepted at 4f7a30a3."""
    root = make(tmp_path / "tree", BASE)

    def unparsed(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        argv = args[0]
        junit = next(a.split("=", 1)[1] for a in argv if a.startswith("--junitxml="))
        Path(junit).write_text("<testsuites><testsuite /></testsuites>")
        return subprocess.CompletedProcess(
            argv,
            0,
            "================ session starts ================\n<Module test_math.py>\n",
            "",
        )

    monkeypatch.setattr(si.subprocess, "run", unparsed)
    assert si.run_collect_only(root, [], [])["errors"]


@pytest.mark.parametrize("enclosing", ["module", "function"])
def test_confirm2_nested_classes_use_lexical_free_aliases(
    tmp_path: Path, enclosing: str
) -> None:
    """Guards against enclosing-class aliases at 4f7a30a3."""
    source = """\
import pytest as p
class TestOuter:
    import unittest as p
    @p.expectedFailure
    class TestInner:
        @p.mark.xfail(strict=False)
        def test_a(self): assert True
"""
    if enclosing == "function":
        source = "def factory():\n" + textwrap.indent(source, "    ")
    sites = _sites(tmp_path, source)["sites"]
    prefix = "factory." if enclosing == "function" else ""
    assert {(s["owner"], s["kind"]) for s in sites} == {
        (prefix + "TestOuter.TestInner", "unittest.expectedFailure"),
        (prefix + "TestOuter.TestInner.test_a", "pytest.mark.xfail"),
    }


@pytest.mark.parametrize("location", ["pytest_arg", "leg"])
def test_confirm2_redacted_urls_do_not_collide_on_reuse(
    tmp_path: Path, location: str
) -> None:
    """Guards against lossy request identity at 4f7a30a3."""
    root = make(
        tmp_path / "tree",
        {
            "pytest.ini": "[pytest]\n",
            "conftest.py": "def pytest_addoption(parser): parser.addoption('--target-url')\ndef pytest_collection_modifyitems(config, items):\n    if config.getoption('--target-url') == 'https://new.example':\n        removed = [i for i in items if i.name == 'test_extra']\n        items[:] = [i for i in items if i not in removed]\n        config.hook.pytest_deselected(items=removed)\n",
            "test_a.py": "def test_a(): pass\ndef test_extra(): pass\n",
        },
    )

    def options(url: str) -> tuple[str, ...]:
        if location == "leg":
            return (f"--leg=--target-url {url}",)
        return ("--pytest-arg=--target-url", f"--pytest-arg={url}")

    first = tmp_path / "first.json"
    assert collect_with(root, first, *options("https://old.example")) == 0
    assert (
        collect_with(
            root,
            tmp_path / "same.json",
            "--collection-from",
            str(first),
            *options("https://old.example"),
        )
        == 0
    )
    assert (
        collect_with(
            root,
            tmp_path / "reuse.json",
            "--collection-from",
            str(first),
            *options("https://new.example"),
        )
        == 2
    )


@pytest.mark.parametrize(
    "location", ["command", "environment", "ini", "override", "proxy", "ids_file"]
)
def test_confirm2_explicit_xdist_workers_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, location: str
) -> None:
    """Guards against unverified workers accepted at 4f7a30a3."""
    root = make(
        tmp_path / "tree",
        {"pytest.ini": "[pytest]\n", "test_a.py": "def test_a(): pass\n"},
    )
    monkeypatch.setenv("PYTEST_PLUGINS", "xdist.plugin")
    spec = f"popen//python={sys.executable}"
    extra: tuple[str, ...] = ()
    worker_args = ("-d", "--tx", spec)
    if location == "ids_file":
        ids = tmp_path / "ids.txt"
        ids.write_text("\n".join(worker_args) + "\n")
        manifest = tmp_path / "manifest.json"
        assert (
            si.main(
                [
                    "run-leg",
                    "--manifest",
                    str(manifest),
                    "--root",
                    str(root),
                    "--name",
                    "retry",
                    "--role",
                    "retry",
                    "--junit",
                    "retry.xml",
                    "--ids-from",
                    str(ids),
                    "--",
                    *pytest_cmd(),
                ]
            )
            == 2
        )
        assert not manifest.exists()
        return
    if location == "command":
        extra = worker_args
    elif location == "environment":
        monkeypatch.setenv("PYTEST_ADDOPTS", " ".join(worker_args))
    elif location == "ini":
        (root / "pytest.ini").write_text(
            "[pytest]\naddopts = " + " ".join(worker_args) + "\n"
        )
    elif location == "override":
        extra = ("-o", "addopts=" + " ".join(worker_args))
    else:
        extra = ("--px", "id=proxy//popen", "--tx", spec)
    manifest = tmp_path / "manifest.json"
    assert run_leg(manifest, root, "primary", extra=extra) == 2
    assert not manifest.exists()


@pytest.mark.parametrize("location", ["pytest_arg", "leg"])
@pytest.mark.parametrize("kind", ["filename", "plugin", "mixed_addopts"])
def test_confirm2_complete_collection_preserves_discovery(
    tmp_path: Path, location: str, kind: str
) -> None:
    """Guards CONFIRM-1 against dropping discovery arguments at 4f7a30a3."""
    files = {
        "pytest.ini": "[pytest]\nmarkers = live: selected out\n",
        "check_alpha.py": "def test_a(): pass\n",
    }
    if kind == "plugin":
        files["conftest.py"] = (
            "def pytest_addoption(parser): parser.addoption('--discovery-name')\ndef pytest_configure(config): config.addinivalue_line('python_files', config.getoption('--discovery-name'))\n"
        )
        args = ["--discovery-name", "check_*.py"]
    elif kind == "mixed_addopts":
        args = ["-o", "python_files=check_*.py", "-o", "addopts=-m 'not live'"]
        files["check_alpha.py"] += (
            "import pytest\n@pytest.mark.live\ndef test_live(): pass\n"
        )
    else:
        args = ["-o", "python_files=check_*.py"]
    root = make(tmp_path / "tree", files)
    out = tmp_path / "inventory.json"
    extra = (
        tuple(f"--pytest-arg={arg}" for arg in args)
        if location == "pytest_arg"
        else ("--leg=" + shlex.join(args),)
    )
    assert collect_with(root, out, *extra) == 0
    inv = json.loads(out.read_text())
    assert "check_alpha.py::test_a" in inv["all_ids"]
    if kind == "mixed_addopts":
        assert "check_alpha.py::test_live" in inv["all_ids"]
        assert "check_alpha.py::test_live" not in inv["selected_ids"]


def test_confirm2_split_discovery_and_boolean_options(tmp_path: Path) -> None:
    """Guards CONFIRM-1 against 4f7a30a3 dropping split-leg discovery."""
    root = make(
        tmp_path / "tree",
        {
            "pytest.ini": "[pytest]\n",
            "alpha/check_a.py": "def test_a(): pass\n",
            "beta/spec_b.py": "def test_b(): pass\n",
        },
    )
    out = tmp_path / "inventory.json"
    assert (
        collect_with(
            root,
            out,
            "--leg=--disable-warnings -o python_files=check_*.py alpha",
            "--leg=-x -o python_files=spec_*.py beta",
        )
        == 0
    )
    assert json.loads(out.read_text())["all_ids"] == [
        "alpha/check_a.py::test_a",
        "beta/spec_b.py::test_b",
    ]


@pytest.mark.parametrize("mode", ["collection", "runtime"])
def test_confirm2_nested_xpass_is_a_finding(tmp_path: Path, mode: str) -> None:
    """Guards against the green XPASS compare at 4f7a30a3."""
    source = "import pytest as p\nclass TestOuter:\n    import unittest as p\n    class TestInner:\n        def test_a(self): assert True\n"
    base = arm(
        tmp_path,
        "base",
        {"pytest.ini": "[pytest]\n", "test_a.py": source},
        run=mode == "runtime",
    )
    source = source.replace(
        "        def test_a", "        @p.mark.xfail(strict=False)\n        def test_a"
    )
    tip = arm(
        tmp_path,
        "tip",
        {"pytest.ini": "[pytest]\n", "test_a.py": source},
        run=mode == "runtime",
    )
    rc, result = compare(tmp_path, base, tip, mode=mode)
    assert rc == 1
    assert any(kind == "xfail-site" for kind, _ in open_findings(result))


@pytest.mark.parametrize(
    "expression",
    [
        "lambda: p.mark.xfail(strict=False)",
        "[p.mark.xfail(strict=False) for _ in [1]]",
        "{p.mark.xfail(strict=False) for _ in [1]}",
        "{i: p.mark.xfail(strict=False) for i in [1]}",
        "(p.mark.xfail(strict=False) for _ in [1])",
    ],
)
def test_confirm2_class_implicit_scopes_use_lexical_aliases(
    tmp_path: Path, expression: str
) -> None:
    """Class-scope siblings missed by 4f7a30a3 and 91d659a0."""
    source = f"import pytest as p\nclass TestOuter:\n    import unittest as p\n    marker = {expression}\n"
    sites = _sites(tmp_path, source)["sites"]
    assert [(s["owner"], s["kind"]) for s in sites] == [
        ("TestOuter", "pytest.mark.xfail")
    ]


@pytest.mark.parametrize(
    "expression",
    [
        "(lambda: p.mark.xfail(strict=False))()",
        "[p.mark.xfail(strict=False) for _ in [1]][0]",
        "{i: p.mark.xfail(strict=False) for i in [1]}[1]",
        "next(p.mark.xfail(strict=False) for _ in [1])",
    ],
)
def test_confirm2_class_implicit_xpass_is_a_finding(
    tmp_path: Path, expression: str
) -> None:
    """Guards the fix against the green class-factory XPASS at 91d659a0."""
    source = "import pytest as p\nclass TestOuter:\n    import unittest as p\n    def test_a(self): assert True\n"
    base = arm(tmp_path, "base", {"pytest.ini": "[pytest]\n", "test_a.py": source})
    source = source.replace("    def test_a", f"    @{expression}\n    def test_a")
    tip = arm(tmp_path, "tip", {"pytest.ini": "[pytest]\n", "test_a.py": source})
    rc, result = compare(tmp_path, base, tip, mode="runtime")
    assert rc == 1
    assert any(kind == "xfail-site" for kind, _ in open_findings(result))


def test_confirm2_missing_primary_manifest_is_invalid(tmp_path: Path) -> None:
    """Guards split discovery from an empty-primary crash introduced at 91d659a0."""
    root = make(tmp_path / "tree", BASE)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": si.MANIFEST_SCHEMA, "legs": []}))
    assert (
        collect_with(root, tmp_path / "inventory.json", "--run-manifest", str(manifest))
        == 2
    )


@pytest.mark.parametrize("location", ["leg", "pytest_arg"])
def test_confirm5_failed_collection_does_not_retain_credentials(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], location: str
) -> None:
    """Guards against failed-selection leaks at 58d4f17a."""
    root = make(tmp_path / "tree", BASE)
    marker = "DUMMY-COLLECTION-PASSWORD"
    args = ["-o", f"index_url=https://user:{marker}@example.invalid/", "missing.py"]
    extra = (
        ("--leg=" + shlex.join(args),)
        if location == "leg"
        else tuple(f"--pytest-arg={a}" for a in args)
    )
    out = tmp_path / "inventory.json"
    assert collect_with(root, out, *extra) == 2
    inventory = json.loads(out.read_text())
    assert inventory["all_ids"]
    assert inventory["evidence_errors"]
    printed = capsys.readouterr()
    assert marker not in out.read_text() + printed.out + printed.err


def test_confirm5_collection_diagnostics_do_not_retain_subprocess_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards the leak's siblings in stdout/stderr fields at 58d4f17a."""
    root = make(tmp_path / "tree", BASE)
    marker = "DUMMY-UNSTRUCTURED-OUTPUT"
    monkeypatch.setattr(
        si.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 4, f"{marker}\n\nSKIPPED [1] {marker}\n", marker
        ),
    )
    result = si.run_collect_only(root, [], [])
    assert result["exit_code"] == 4
    assert result["errors"]
    assert marker not in json.dumps(result)


def test_confirm5_collection_error_does_not_retain_argument_diagnostics(
    tmp_path: Path,
) -> None:
    """Guards the leak's JUnit error-text sibling at 58d4f17a."""
    root = make(
        tmp_path / "tree",
        {
            "pytest.ini": "[pytest]\n",
            "test_a.py": "import sys\nraise RuntimeError(next(a for a in sys.argv if a.startswith('index_url=')))\n",
        },
    )
    marker = "DUMMY-JUNIT-PASSWORD"
    result = si.run_collect_only(
        root, ["-o", f"index_url=https://user:{marker}@example.invalid/"], []
    )
    assert result["module_reports"][0]["outcome"] == "error"
    assert marker not in json.dumps(result)


@pytest.mark.parametrize("selection", ["-xktest_keep", "-vvktest_keep", "-xmkeep"])
@pytest.mark.parametrize("location", ["leg", "environment", "ini"])
def test_confirm5_grouped_short_selection_cannot_hide_removed_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selection: str, location: str
) -> None:
    """Guards CONFIRM3-1 against grouped selection retained at 58d4f17a."""
    files = {
        "pytest.ini": "[pytest]\nmarkers = keep: retained test\n",
        "test_a.py": "import pytest\n@pytest.mark.keep\ndef test_keep(): pass\ndef test_drop(): pass\n",
    }
    extra = (f"--leg={selection}",) if location == "leg" else ()
    if location == "environment":
        monkeypatch.setenv("PYTEST_ADDOPTS", selection)
    elif location == "ini":
        files["pytest.ini"] += f"addopts = {selection}\n"
    paths = []
    for name in ("base", "tip"):
        source = files["test_a.py"]
        if name == "tip":
            source = source.replace("def test_drop(): pass\n", "")
        root = make(tmp_path / name, {**files, "test_a.py": source})
        out = tmp_path / f"{name}.json"
        assert collect_with(root, out, *extra) == 0
        assert json.loads(out.read_text())["selected_ids"] == ["test_a.py::test_keep"]
        paths.append(out)
    rc, result = compare(tmp_path, *paths, mode="collection")
    assert rc == 1
    assert open_findings(result) == [("removed", "test_a.py::test_drop")]


@pytest.mark.parametrize("mode", ["collection", "runtime"])
@pytest.mark.parametrize(
    "expression",
    [
        "(lambda p=p: p.mark.xfail(strict=False))()",
        "(lambda *, q=p: q.mark.xfail(strict=False))()",
        "(lambda p=p, /: (lambda: p.mark.xfail(strict=False))())()",
        "(lambda xf=p.mark.xfail: xf(strict=False))()",
    ],
)
def test_confirm5_lambda_captures_cannot_hide_xpass(
    tmp_path: Path, mode: str, expression: str
) -> None:
    """Guards CONFIRM3-2 against lambda captures missed at 58d4f17a."""
    source = "import unittest as p\nclass TestA:\n    import pytest as p\n    def test_a(self): assert True\n"
    base = arm(
        tmp_path,
        "base",
        {"pytest.ini": "[pytest]\n", "test_a.py": source},
        run=mode == "runtime",
    )
    source = source.replace("    def test_a", f"    @{expression}\n    def test_a")
    tip = arm(
        tmp_path,
        "tip",
        {"pytest.ini": "[pytest]\n", "test_a.py": source},
        run=mode == "runtime",
    )
    rc, result = compare(tmp_path, base, tip, mode=mode)
    assert rc == 1
    assert any(kind == "xfail-site" for kind, _ in open_findings(result))


def test_confirm5_parameter_binding_shadows_free_aliases(tmp_path: Path) -> None:
    """Guards CONFIRM3-2 siblings against inherited parameter aliases at 58d4f17a."""
    result = _sites(
        tmp_path,
        """\
        import pytest as p
        import unittest as u
        f = lambda p=u: p.skip("expected")
        g = lambda p: p.mark.xfail(strict=False)
        def helper(p=u):
            p.skip("expected")
        """,
    )
    assert [(s["owner"], s["kind"]) for s in result["sites"]] == [
        ("<module>", "unittest.skip"),
        ("helper", "unittest.skip"),
    ]
    assert any("unresolved" in d["what"] for d in result["dynamic"])


@pytest.mark.parametrize(
    "expression",
    [
        "[p.mark.xfail(strict=False) for p in [p]][0]",
        "next(iter({lambda: p.mark.xfail(strict=False) for p in [p]}))()",
        "{0: p.mark.xfail(strict=False) for p in [p]}[0]",
        "next(p.mark.xfail(strict=False) for p in [p])",
    ],
)
def test_confirm5_comprehension_captures_cannot_hide_xpass(
    tmp_path: Path, expression: str
) -> None:
    """Guards CONFIRM3-2's comprehension-binding siblings at 58d4f17a."""
    source = "import unittest as p\nclass TestA:\n    import pytest as p\n    def test_a(self): assert True\n"
    base = arm(tmp_path, "base", {"pytest.ini": "[pytest]\n", "test_a.py": source})
    source = source.replace("    def test_a", f"    @{expression}\n    def test_a")
    tip = arm(tmp_path, "tip", {"pytest.ini": "[pytest]\n", "test_a.py": source})
    rc, result = compare(tmp_path, base, tip)
    assert rc == 1
    assert any(kind == "xfail-site" for kind, _ in open_findings(result))


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("runtime", [False, True])
def test_confirm5_options_precede_selection_terminator(
    tmp_path: Path, fallback: bool, runtime: bool
) -> None:
    """Guards CONFIRM3-3 and recorder/fallback siblings at 58d4f17a."""
    files = {
        "pytest.ini": "[pytest]\n",
        "test_a.py": "def test_keep(): pass\ndef test_drop(): pass\n",
    }
    if fallback:
        files["conftest.py"] = (
            "def pytest_collection_modifyitems(items):\n    items[:] = [i for i in items if i.name == 'test_keep']\n"
        )
    root = make(tmp_path / "tree", files)
    out = tmp_path / "inventory.json"
    if runtime:
        manifest = tmp_path / "manifest.json"
        assert run_leg(manifest, root, "primary", select=("--", "test_a.py")) == 0
        extra = ("--run-manifest", str(manifest))
    else:
        extra = ("--leg=-- test_a.py",)
    assert collect_with(root, out, *extra) == 0
    inv = json.loads(out.read_text())
    expected = ["test_a.py::test_keep"]
    if not fallback:
        expected.append("test_a.py::test_drop")
    assert inv["selected_ids"] == sorted(expected)
    assert not inv["evidence_errors"]


_INI_SPELLINGS = [
    "-o {}",
    "-o{}",
    "-o={}",
    "--override-ini {}",
    "--override-ini={}",
    "-qvo={}",
    "-xvo {}",
    "-xvo{}",
    "-Zo {}",
]


@pytest.mark.parametrize("spelling", _INI_SPELLINGS)
@pytest.mark.parametrize(
    "key",
    [
        "service.password",
        "service/value",
        "service:value",
        "service value",
        "CI",
        "--ignore",
        "test_a.py::setting",
    ],
)
def test_confirm7_ini_assignment_values_are_never_retained(
    spelling: str, key: str
) -> None:
    """Guards the fix against ini-name grammar and env allowlists at b83e2cb5."""
    marker = "DUMMY-INI-VALUE"
    args = shlex.split(spelling.format(shlex.quote(f"{key}={marker}")))
    assert marker not in json.dumps(si.sanitized_argv(args))


@pytest.mark.parametrize("spelling", _INI_SPELLINGS)
@pytest.mark.parametrize("location", ["leg", "pytest_arg", "old_manifest", "run_leg"])
def test_confirm7_dotted_ini_redacted_at_every_boundary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], spelling: str, location: str
) -> None:
    """Guards the fix against retained collection/manifest diagnostics at b83e2cb5."""
    root = make(
        tmp_path / "tree",
        {
            "pytest.ini": "[pytest]\n",
            "conftest.py": 'def pytest_addoption(parser):\n    parser.addini("service.password", "password")\n    parser.addoption("-Z", action="store_true")\n',
            "test_a.py": "def test_a(): assert True\n",
        },
    )
    marker = "DUMMY-DOTTED-PASSWORD"
    args = shlex.split(spelling.format(f"service.password={marker}"))
    out = tmp_path / "inventory.json"
    manifest = tmp_path / "manifest.json"
    if location == "run_leg":
        assert run_leg(manifest, root, "primary", extra=tuple(args)) == 0
        extra = ("--run-manifest", str(manifest))
        expected = 0
    elif location == "old_manifest":
        assert run_leg(manifest, root, "primary") == 0
        data = json.loads(manifest.read_text())
        data["legs"][0]["select_args"] = [*args, "missing.py"]
        data["legs"][0]["argv"] += args
        manifest.write_text(json.dumps(data))
        extra = ("--run-manifest", str(manifest))
        expected = 2
    else:
        args += ["missing.py"]
        extra = (
            ("--leg=" + shlex.join(args),)
            if location == "leg"
            else tuple(f"--pytest-arg={a}" for a in args)
        )
        expected = 2
    assert collect_with(root, out, *extra) == expected
    inventory = json.loads(out.read_text())
    assert inventory["all_ids"] == ["test_a.py::test_a"]
    assert bool(inventory["evidence_errors"]) == bool(expected)
    printed = capsys.readouterr()
    retained = out.read_text() + printed.out + printed.err
    if location == "run_leg":
        retained += manifest.read_text()
    assert marker not in retained


_LAMBDA_BINDINGS = [
    "(lambda p=u: p.mark.xfail(strict=False))(p)",
    "(lambda p=u: p.mark.xfail(strict=False))(p=p)",
    "(lambda p=u, /: p.mark.xfail(strict=False))(p)",
    "(lambda *, p=u: p.mark.xfail(strict=False))(p=p)",
    "(lambda unused=None, p=u: p.mark.xfail(strict=False))(None, p)",
    "(lambda p=u: (lambda: p.mark.xfail(strict=False))())(p)",
    "(lambda p=u: p.mark.xfail(strict=False))(*[p])",
    "(lambda p=u: p.mark.xfail(strict=False))(**{'p': p})",
    "(lambda p=u, *rest, **kw: p.mark.xfail(strict=False))(p, 1, extra=2)",
]


@pytest.mark.parametrize("mode", ["collection", "runtime"])
@pytest.mark.parametrize("expression", _LAMBDA_BINDINGS)
def test_confirm7_explicit_lambda_binding_cannot_hide_xpass(
    tmp_path: Path, mode: str, expression: str
) -> None:
    """Guards CONFIRM6-1 against defaults overriding actual arguments at b83e2cb5."""
    source = "import pytest as p\nimport unittest as u\ndef test_a(): assert True\n"
    base = arm(
        tmp_path,
        "base",
        {"pytest.ini": "[pytest]\n", "test_a.py": source},
        run=mode == "runtime",
    )
    source = source.replace("def test_a", f"@{expression}\ndef test_a")
    tip = arm(
        tmp_path,
        "tip",
        {"pytest.ini": "[pytest]\n", "test_a.py": source},
        run=mode == "runtime",
    )
    rc, result = compare(tmp_path, base, tip, mode=mode)
    assert rc == 1
    assert any(kind == "xfail-site" for kind, _ in open_findings(result))


@pytest.mark.parametrize("expression", _LAMBDA_BINDINGS)
def test_confirm7_lambda_arguments_use_the_calling_class_scope(
    tmp_path: Path, expression: str
) -> None:
    """Guards CONFIRM6-1's class/default and nested-scope siblings at b83e2cb5."""
    source = (
        "import unittest as p\nimport unittest as u\nclass TestA:\n    import pytest as p\n    marker = "
        + expression
        + "\n"
    )
    result = _sites(tmp_path, source)
    assert [(s["owner"], s["kind"]) for s in result["sites"]] == [
        ("TestA", "pytest.mark.xfail")
    ]


@pytest.mark.parametrize(
    "actual", ["unknown", "*unknown", "**unknown", "p=unknown", "**{'p': unknown}"]
)
def test_confirm7_unknown_lambda_arguments_never_reuse_defaults(
    tmp_path: Path, actual: str
) -> None:
    """Guards CONFIRM6-1 against optimistic defaults for unknown calls at b83e2cb5."""
    result = _sites(
        tmp_path,
        f"import unittest as u\nmarker = (lambda p=u: p.mark.xfail(strict=False))({actual})\n",
    )
    assert not result["sites"]
    assert any("unresolved" in d["what"] for d in result["dynamic"])


@pytest.mark.parametrize("actual", ["u", "p=u", "*[u]", "**{'p': u}"])
def test_confirm7_explicit_lambda_binding_overrides_pytest_default(
    tmp_path: Path, actual: str
) -> None:
    """Guards CONFIRM6-1 against falsely retaining a pytest default at b83e2cb5."""
    result = _sites(
        tmp_path,
        f'import pytest as p\nimport unittest as u\nmarker = (lambda p=p: p.skip("expected"))({actual})\n',
    )
    assert [s["kind"] for s in result["sites"]] == ["unittest.skip"]


@pytest.mark.parametrize(
    "argv",
    [
        ["--", "test_a.py"],
        ["-kfoo", "--ignore=test_a.py"],
        ["test_a.py::test_value[a=b]"],
        ["--deselect=test_a.py::test_value[a=b]"],
    ],
)
def test_confirm7_redaction_preserves_selection_addresses(argv: list[str]) -> None:
    """Guards the fix from altering path terminators or parameter IDs."""
    assert si.sanitized_argv(argv) == argv


@pytest.mark.parametrize(
    "argument",
    [
        "https://user:DUMMY-URL-VALUE@example.invalid/?setting=value",
        "https://user:DUMMY-URL-VALUE@example.invalid/test_a.py::test_a",
        "https://user:DUMMY-URL-VALUE@example.invalid/test_a.py::test_a[setting=value]",
    ],
)
def test_confirm7_url_masking_precedes_assignment_and_address_parsing(
    argument: str,
) -> None:
    """Guards the fix from URL precedence regressions introduced at 6b6c0207."""
    assert "DUMMY-URL-VALUE" not in json.dumps(si.sanitized_argv([argument]))


@pytest.mark.parametrize(
    "actual", ["p, p=u", "p=p, **{'p': u}", "*[p, u]", "missing=p"]
)
def test_confirm7_invalid_lambda_binding_is_unresolved(
    tmp_path: Path, actual: str
) -> None:
    """Guards CONFIRM6-1 against default fallback after an invalid call at b83e2cb5."""
    result = _sites(
        tmp_path,
        f"import pytest as p\nimport unittest as u\nmarker = (lambda p=u: p.mark.xfail(strict=False))({actual})\n",
    )
    assert not result["sites"]
    assert any(
        "unresolved lambda argument binding" in d["what"] for d in result["dynamic"]
    )


def test_confirm7_plugin_short_group_cannot_turn_ini_value_into_test_address(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Guards the key-shape/short-group sibling left by 69d4ec22."""
    root = make(
        tmp_path / "tree",
        {
            "pytest.ini": "[pytest]\n",
            "conftest.py": 'def pytest_addoption(parser):\n    parser.addini("test_a.py::setting", "value")\n    parser.addoption("-Z", action="store_true")\n',
            "test_a.py": "def test_a(): assert True\n",
        },
    )
    out = tmp_path / "inventory.json"
    assert (
        collect_with(
            root, out, "--leg=-Zo test_a.py::setting=DUMMY-ADDRESS-VALUE missing.py"
        )
        == 2
    )
    inventory = json.loads(out.read_text())
    assert inventory["all_ids"] == ["test_a.py::test_a"]
    printed = capsys.readouterr()
    assert "DUMMY-ADDRESS-VALUE" not in out.read_text() + printed.out + printed.err


@pytest.mark.parametrize("option", ["-o", "--override-ini", "-xvo", "-Zo"])
@pytest.mark.parametrize("location", ["leg", "old_manifest"])
@pytest.mark.parametrize("missing", [False, True])
def test_b8_split_ini_arguments_keep_consumption_context(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    option: str,
    location: str,
    missing: bool,
) -> None:
    """Guards the split option/selection context at 4660ed24."""
    root = make(
        tmp_path / "tree",
        {
            "pytest.ini": "[pytest]\n",
            "conftest.py": 'def pytest_addoption(parser):\n    parser.addini("test_a.py::setting", "value")\n    parser.addoption("-Z", action="store_true")\n',
            "test_a.py": "def test_a(): assert True\n",
        },
    )
    marker = "DUMMY-SPLIT-INI-VALUE"
    selection = [f"test_a.py::setting={marker}"]
    if missing:
        selection.append("missing.py")
    out = tmp_path / "inventory.json"
    if location == "old_manifest":
        manifest = tmp_path / "manifest.json"
        assert run_leg(manifest, root, "primary") == 0
        data = json.loads(manifest.read_text())
        data["legs"][0]["select_args"] = selection
        data["legs"][0]["argv"] += [option, *selection]
        manifest.write_text(json.dumps(data))
        extra = ("--run-manifest", str(manifest))
    else:
        extra = ("--leg=" + shlex.join(selection),)
    assert collect_with(root, out, f"--pytest-arg={option}", *extra) == (
        2 if missing else 0
    )
    inventory = json.loads(out.read_text())
    assert inventory["all_ids"] == ["test_a.py::test_a"]
    printed = capsys.readouterr()
    assert marker not in out.read_text() + printed.out + printed.err
    selected = inventory["collections"]["selected"][0]
    assert "sha256:" in selected["select_args"][0]


@pytest.mark.parametrize("option", ["-o", "--override-ini", "-xvo", "-Zo"])
def test_b8_run_leg_refuses_consumed_selection_values(
    tmp_path: Path, option: str
) -> None:
    """Guards the command/selection recording sibling at 4660ed24."""
    root = make(
        tmp_path / "tree",
        {
            "pytest.ini": "[pytest]\n",
            "conftest.py": 'def pytest_addoption(parser):\n    parser.addini("test_a.py::setting", "value")\n    parser.addoption("-Z", action="store_true")\n',
            "test_a.py": "def test_a(): assert True\n",
        },
    )
    manifest = tmp_path / "manifest.json"
    value = "test_a.py::setting=DUMMY-RECORDED-INI-VALUE"
    assert (
        si.main(
            [
                "run-leg",
                "--root",
                str(root),
                "--manifest",
                str(manifest),
                "--name",
                "primary",
                "--role",
                "primary",
                "--junit",
                "primary.xml",
                f"--select={value}",
                "--",
                *pytest_cmd(option, value),
            ]
        )
        == 2
    )
    assert not manifest.exists()


_VARIADIC_LAMBDAS = [
    "(lambda *mods: mods[0].mark.xfail)(p)(strict=False)",
    "(lambda **mods: mods['p'].mark.xfail)(p=p)(strict=False)",
    "(lambda *mods: mods[-1].mark.xfail)(u, p)(strict=False)",
    "(lambda *mods: mods[+1].mark.xfail)(u, p)(strict=False)",
    "(lambda *mods: mods[0].mark.xfail)(*[p])(strict=False)",
    "(lambda **mods: mods['p'].mark.xfail)(**{'p': p})(strict=False)",
    "(lambda first, *mods: mods[0].mark.xfail)(u, p)(strict=False)",
    "(lambda p=u, **mods: mods['module'].mark.xfail)(module=p)(strict=False)",
    "(lambda *mods: (lambda: mods[0].mark.xfail)())(p)(strict=False)",
    "(lambda **mods: (lambda p=mods['p']: p.mark.xfail)())(p=p)(strict=False)",
    "(lambda *mods: mods[0][0].mark.xfail)([p])(strict=False)",
    "(lambda **mods: mods['outer']['p'].mark.xfail)(outer={'p': p})(strict=False)",
    "(lambda mods: mods[0].mark.xfail)([p])(strict=False)",
    "(lambda mods: mods['p'].mark.xfail)({'p': p})(strict=False)",
]


@pytest.mark.parametrize("mode", ["collection", "runtime"])
@pytest.mark.parametrize("expression", _VARIADIC_LAMBDAS)
def test_b8_variadic_lambda_bindings_cannot_hide_xpass(
    tmp_path: Path, mode: str, expression: str
) -> None:
    """Guards CONFIRM6-1's dropped variadic bindings at 4660ed24."""
    source = "import pytest as p\nimport unittest as u\ndef test_a(): assert True\n"
    files = {"pytest.ini": "[pytest]\n", "test_a.py": source}
    base = arm(tmp_path, "base", files, run=mode == "runtime")
    files["test_a.py"] = source.replace("def test_a", f"@{expression}\ndef test_a")
    tip = arm(tmp_path, "tip", files, run=mode == "runtime")
    rc, result = compare(tmp_path, base, tip, mode=mode)
    assert rc == 1
    assert any(kind == "xfail-site" for kind, _ in open_findings(result))


@pytest.mark.parametrize("expression", _VARIADIC_LAMBDAS)
def test_b8_variadic_bindings_capture_calling_class_scope(
    tmp_path: Path, expression: str
) -> None:
    """Guards CONFIRM6-1's class and nested-container siblings at 4660ed24."""
    result = _sites(
        tmp_path,
        "import unittest as p\nimport unittest as u\nclass TestA:\n    import pytest as p\n    marker = "
        + expression
        + "\n",
    )
    assert [s["kind"] for s in result["sites"]] == ["pytest.mark.xfail"]
    assert not result["dynamic"]


@pytest.mark.parametrize(
    "expression",
    [
        "(lambda *mods: mods[index].mark.xfail)(p)",
        "(lambda **mods: mods[key].mark.xfail)(p=p)",
        "(lambda *mods: mods[9].mark.xfail)(p)",
        "(lambda **mods: mods['missing'].mark.xfail)(p=p)",
        "(lambda *mods: mods[0].mark.xfail)(unknown)",
        "(lambda **mods: mods['p'].mark.xfail)(p=unknown)",
        "(lambda *mods: mods[0].mark.xfail)(*unknown)",
        "(lambda **mods: mods['p'].mark.xfail)(**unknown)",
        "(lambda *mods: mods[0].mark.xfail)()",
        "(lambda **mods: mods['p'].mark.xfail)()",
        "(lambda *mods: mods[0].mark.xfail)(*[p for p in source])",
        "(lambda **mods: mods['p'].mark.xfail)(**{k: p for k in source})",
    ],
)
def test_b8_unresolved_variadic_access_is_reported(
    tmp_path: Path, expression: str
) -> None:
    """Guards CONFIRM6-1's silent unknown bindings at 4660ed24."""
    result = _sites(tmp_path, "import pytest as p\nmarker = " + expression + "\n")
    assert not result["sites"]
    assert any("unresolved" in d["what"] for d in result["dynamic"])


@pytest.mark.parametrize(
    "expression",
    [
        "(lambda *mods: mods[1].skip('expected'))(p, u)",
        "(lambda **mods: mods['u'].skip('expected'))(p=p, u=u)",
        "(lambda *mods: (lambda *mods: mods[0].skip('expected'))(u))(p)",
        "(lambda **mods: (lambda **mods: mods['p'].skip('expected'))(p=u))(p=p)",
    ],
)
def test_b8_variadic_bindings_keep_selected_object_and_shadowing(
    tmp_path: Path, expression: str
) -> None:
    """Guards CONFIRM6-1's selection and shadowing siblings at 4660ed24."""
    result = _sites(
        tmp_path,
        "import pytest as p\nimport unittest as u\nmarker = " + expression + "\n",
    )
    assert [s["kind"] for s in result["sites"]] == ["unittest.skip"]
    assert not result["dynamic"]


@pytest.mark.parametrize(
    "attribute, kind",
    [
        ("mark.xfail", "pytest.mark.xfail"),
        ("mark.skip", "pytest.mark.skip"),
        ("mark.skipif", "pytest.mark.skipif"),
        ("skip", "pytest.skip"),
        ("xfail", "pytest.xfail"),
        ("importorskip", "pytest.importorskip"),
    ],
)
@pytest.mark.parametrize("binding", ["*mods: mods[0]", "**mods: mods['p']"])
def test_b8_variadic_bindings_cover_skip_and_xfail_families(
    tmp_path: Path, attribute: str, kind: str, binding: str
) -> None:
    """Guards CONFIRM6-1's skip/xfail-family siblings at 4660ed24."""
    argument = "p=p" if binding.startswith("**") else "p"
    result = _sites(
        tmp_path,
        f"import pytest as p\nmarker = (lambda {binding}.{attribute}('dummy'))({argument})\n",
    )
    assert [s["kind"] for s in result["sites"]] == [kind]
    assert not result["dynamic"]


@pytest.mark.parametrize(
    "access", ["mods[index]", "mods[0:1][0]", "mods[0 + 0]", "mods.get('p')"]
)
@pytest.mark.parametrize(
    "attribute",
    [
        "xfail",
        "skip",
        "skipif",
        "importorskip",
        "skipIf",
        "skipUnless",
        "expectedFailure",
        "skipTest",
    ],
)
def test_b8_unsupported_container_access_warns_on_bare_mark(
    tmp_path: Path, access: str, attribute: str
) -> None:
    """Guards CONFIRM6-1's unresolved bare-mark access at 4660ed24."""
    result = _sites(
        tmp_path,
        f"import pytest as p\nmarker = (lambda *mods: {access}.{attribute})(p)\n",
    )
    assert not result["sites"]
    assert any("unresolved" in d["what"] for d in result["dynamic"])


def test_b8_local_import_overrides_captured_parameter(tmp_path: Path) -> None:
    """Guards the binding fix at 06f8e883 against hiding a local import."""
    result = _sites(
        tmp_path,
        "import unittest as p\ndef helper(p=p):\n    import pytest as p\n    p.xfail('dummy')\n",
    )
    assert [s["kind"] for s in result["sites"]] == ["pytest.xfail"]
    assert not result["dynamic"]


@pytest.mark.parametrize("option", ["-o", "--override-ini", "-xvo", "-Zo"])
@pytest.mark.parametrize("command_state", ["raw", "sanitized", "missing"])
@pytest.mark.parametrize("role", ["primary", "retry"])
def test_recheck1_imported_selections_keep_their_own_command_context(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    option: str,
    command_state: str,
    role: str,
) -> None:
    """Guards GATE-noskip-RECHECK-1 against legacy disclosure at 817dd530."""
    root = make(
        tmp_path / "tree",
        {
            "pytest.ini": "[pytest]\n",
            "conftest.py": 'def pytest_addoption(parser):\n    parser.addini("test_a.py::setting", "value")\n    parser.addoption("-Z", action="store_true")\n',
            "test_a.py": "def test_a(): assert True\n",
        },
    )
    marker = "DUMMY-LEG-INI-VALUE"
    value = f"test_a.py::setting={marker}"
    manifest = tmp_path / "manifest.json"
    assert run_leg(manifest, root, "primary") == 0
    if role == "retry":
        assert run_leg(manifest, root, "retry", role=role, extra=(option, value)) == 0
    data = json.loads(manifest.read_text())
    leg = data["legs"][-1]
    leg["select_args"] = [value]
    # Recreate both historical formats: raw commands and commands whose ini
    # values were sanitized before selections gained the same protection.
    leg["argv"] = [*pytest_cmd(option, value)]
    if command_state == "sanitized":
        leg["argv"] = si.sanitized_argv(leg["argv"])
    elif command_state == "missing":
        leg.pop("argv")
    manifest.write_text(json.dumps(data))
    out = tmp_path / "inventory.json"
    # A primary with this incomplete replay context is invalid evidence;
    # a retry must not change the otherwise valid primary collection.
    assert collect_with(root, out, "--run-manifest", str(manifest)) == (
        2 if role == "primary" else 0
    )
    printed = capsys.readouterr()
    assert marker not in out.read_text() + printed.out + printed.err
    retained = json.loads(out.read_text())["runtime"]["legs"][-1]
    assert "sha256:" in retained["select_args"][0]


@pytest.mark.parametrize("mode", ["collection", "runtime"])
@pytest.mark.parametrize(
    "expression",
    [
        '(lambda *mods: getattr(mods[0].mark, "xfail", None))(p)(strict=False)',
        '(lambda **mods: getattr(mods["p"].mark, "xfail", None))(p=p)(strict=False)',
        '(lambda *mods: getattr(getattr(mods[0], "mark", None), "xfail", None))(p)(strict=False)',
    ],
)
def test_recheck2_literal_getattr_cannot_hide_xpass(
    tmp_path: Path, mode: str, expression: str
) -> None:
    """Guards GATE-noskip-RECHECK-2 against invisible XPASS at 817dd530."""
    source = "import pytest as p\ndef test_a(): assert True\n"
    files = {"pytest.ini": "[pytest]\n", "test_a.py": source}
    base = arm(tmp_path, "base", files, run=mode == "runtime")
    files["test_a.py"] = source.replace("def test_a", f"@{expression}\ndef test_a")
    tip = arm(tmp_path, "tip", files, run=mode == "runtime")
    rc, result = compare(tmp_path, base, tip, mode=mode)
    assert rc == 1
    assert any(kind == "xfail-site" for kind, _ in open_findings(result))


@pytest.mark.parametrize("default", ["", ", None"])
@pytest.mark.parametrize(
    "access", ["mods[0 + 0]", "mods[index]", "mods[0:1][0]", "mods.get('p')"]
)
@pytest.mark.parametrize(
    "attribute",
    [
        "xfail",
        "skip",
        "skipif",
        "importorskip",
        "skipIf",
        "skipUnless",
        "expectedFailure",
        "skipTest",
    ],
)
def test_recheck2_unresolved_literal_getattr_is_reported(
    tmp_path: Path, default: str, access: str, attribute: str
) -> None:
    """Guards GATE-noskip-RECHECK-2's unresolved family at 817dd530."""
    result = _sites(
        tmp_path,
        f'import pytest as p\nmarker = (lambda *mods: getattr({access}.mark, "{attribute}"{default}))(p)\n',
    )
    assert not result["sites"]
    assert any("unresolved" in d["what"] for d in result["dynamic"])


@pytest.mark.parametrize("default", ["", ", None"])
@pytest.mark.parametrize(
    "target, attribute, kind, suffix",
    [
        ("p.mark", "xfail", "pytest.mark.xfail", ""),
        ("p.mark", "skip", "pytest.mark.skip", ""),
        ("p.mark", "skipif", "pytest.mark.skipif", ""),
        ("p", "skip", "pytest.skip", "('dummy')"),
        ("p", "xfail", "pytest.xfail", "('dummy')"),
        ("p", "importorskip", "pytest.importorskip", "('dummy')"),
        ("u", "skip", "unittest.skip", "('dummy')"),
        ("u", "skipIf", "unittest.skipIf", "(True, 'dummy')"),
        ("u", "skipUnless", "unittest.skipUnless", "(False, 'dummy')"),
        ("u", "expectedFailure", "unittest.expectedFailure", ""),
        ("u.TestCase", "skipTest", "unittest.TestCase.skipTest", "('dummy')"),
    ],
)
def test_recheck2_known_literal_getattr_resolves_all_families(
    tmp_path: Path, default: str, target: str, attribute: str, kind: str, suffix: str
) -> None:
    """Guards GATE-noskip-RECHECK-2's resolved family at 817dd530."""
    result = _sites(
        tmp_path,
        f'import pytest as p\nimport unittest as u\nmarker = getattr({target}, "{attribute}"{default}){suffix}\n',
    )
    assert [s["kind"] for s in result["sites"]] == [kind]
    assert not result["dynamic"]


@pytest.mark.parametrize("mode", ["collection", "runtime"])
@pytest.mark.parametrize(
    "scope", ["function", "class", "module", "parameter", "hook", "finish-hook"]
)
@pytest.mark.parametrize(
    "expression",
    [
        'getattr(*(p.mark, "xfail"))(strict=False)',
        'getattr(*(p.mark, "xfail", None))(strict=False)',
        'getattr(*[p.mark, "xfail"])(strict=False)',
        'getattr(*[p.mark, "xfail", None])(strict=False)',
        'getattr(*(p.mark, *["xfail", None]))(strict=False)',
        'lookup(*(p.mark, "x" + "fail", None))(strict=False)',
        "factory()(strict=False)",
        "make_mark()",
    ],
)
def test_b10_pytest_collected_xfail_cannot_hide_behind_construction(
    tmp_path: Path, mode: str, scope: str, expression: str
) -> None:
    """Guards B8R3-1 at 811de76c using pytest's marks, including XPASS."""
    source = (
        "import pytest as p\nfrom builtins import getattr as lookup\n"
        "from helper import factory, make_mark\n"
    )
    bodies = {
        "function": "def test_a(): assert True\n",
        "class": "class TestA:\n    def test_a(self): assert True\n",
        "module": "def test_a(): assert True\n",
        "parameter": '@p.mark.parametrize("value", [p.param(1, id="one")])\ndef test_a(value): assert True\n',
        "hook": "def test_a(): assert True\n",
        "finish-hook": "def test_a(): assert True\n",
    }
    files = {
        "pytest.ini": "[pytest]\n",
        "helper.py": 'import pytest\ndef factory(): return getattr(pytest.mark, "x" + "fail")\ndef make_mark(): return factory()(strict=False)\n',
        "test_a.py": source + bodies[scope],
    }
    base = arm(tmp_path, "base", files, run=mode == "runtime")
    if scope in ("function", "class"):
        files["test_a.py"] = source + f"@{expression}\n" + bodies[scope]
    elif scope == "module":
        files["test_a.py"] = source + f"pytestmark = {expression}\n" + bodies[scope]
    elif scope == "parameter":
        files["test_a.py"] = files["test_a.py"].replace(
            'id="one"', f'id="one", marks=[{expression}]'
        )
    elif scope == "hook":
        files["conftest.py"] = source + (
            "def pytest_collection_modifyitems(items):\n"
            f"    for item in items: item.add_marker({expression})\n"
        )
    else:
        files["conftest.py"] = source + (
            "@p.hookimpl(trylast=True)\ndef pytest_collection_finish(session):\n"
            f"    for item in session.items: item.add_marker({expression})\n"
        )
    tip = arm(tmp_path, "tip", files, run=mode == "runtime")
    rc, result = compare(tmp_path, base, tip, mode=mode)
    assert rc == 1
    # This assertion requires actual collected evidence even for spellings
    # the supplementary AST inventory already understands.
    assert any(
        f["kind"] == "xfail-site" and f.get("source") == "pytest-collection"
        for f in result["findings"]
    )
    if mode == "runtime":
        assert all(
            o["outcome"] == "passed"
            for o in si.primary_outcomes(json.loads(tip.read_text())).values()
        )


@pytest.mark.parametrize("name", ["skip", "skipif", "xfail"])
@pytest.mark.parametrize(
    "scope", ["function", "class", "module", "parameter", "hook", "finish-hook"]
)
def test_b10_collected_exemptions_keep_arguments_and_inherited_marks(
    tmp_path: Path, name: str, scope: str
) -> None:
    """Guards B8R3-1 at 811de76c for the entire pytest mark family."""
    source = 'import pytest\nmark = getattr(pytest.mark, "' + name + '")\n'
    mark = 'mark(reason="dummy")' if name == "skip" else 'mark(False, reason="dummy")'
    body = "def test_a(): assert True\n"
    if scope == "class":
        body = "class TestA:\n    def test_a(self): assert True\n"
    if scope == "parameter":
        body = '@pytest.mark.parametrize("v", [pytest.param(1, id="one")])\ndef test_a(v): assert True\n'
    files = {"pytest.ini": "[pytest]\n", "test_a.py": source + body}
    base = arm(tmp_path, "base", files, run=False)
    if scope in ("function", "class"):
        files["test_a.py"] = source + f"@{mark}\n" + body
    elif scope == "module":
        files["test_a.py"] = source + f"pytestmark = [{mark}]\n" + body
    elif scope == "parameter":
        files["test_a.py"] = files["test_a.py"].replace(
            'id="one"', f'id="one", marks={mark}'
        )
    elif scope == "hook":
        files["conftest.py"] = (
            source
            + f"def pytest_collection_modifyitems(items):\n    for item in items: item.add_marker({mark})\n"
        )
    else:
        files["conftest.py"] = source + (
            "@pytest.hookimpl(trylast=True)\ndef pytest_collection_finish(session):\n"
            f"    for item in session.items: item.add_marker({mark})\n"
        )
    tip = arm(tmp_path, "tip", files, run=False)
    rc, result = compare(tmp_path, base, tip, mode="collection")
    assert rc == (1 if name == "xfail" else 0)
    inv = json.loads(tip.read_text())
    sites = [
        s for s in inv["static"]["sites"] if s.get("source") == "pytest-collection"
    ]
    assert len(sites) == 1
    assert sites[0]["kind"] == f"pytest.mark.{name}"
    assert '"dummy"' in sites[0]["expr"]
    assert (
        result["findings"]
        if name == "xfail"
        else result["report_only"]["new_skip_sites"]
    )


@pytest.mark.parametrize(
    "arguments",
    [
        'reason="changed"',
        'False, reason="base"',
        'run=False, reason="base"',
        'strict=True, reason="base"',
        'raises=(ValueError, TypeError), reason="base"',
    ],
)
def test_b10_changed_collected_mark_arguments_are_not_erased(
    tmp_path: Path, arguments: str
) -> None:
    """Guards B8R3-1 at 811de76c even when a helper hides the mark site."""
    files = {
        "pytest.ini": "[pytest]\n",
        "helper.py": 'import pytest\ndef make_mark(): return pytest.mark.xfail(reason="base")\n',
        "test_a.py": "from helper import make_mark\n@make_mark()\ndef test_a(): assert True\n",
    }
    base = arm(tmp_path, "base", files, run=False)
    files["helper.py"] = (
        f"import pytest\ndef make_mark(): return pytest.mark.xfail({arguments})\n"
    )
    tip = arm(tmp_path, "tip", files, run=False)
    rc, result = compare(tmp_path, base, tip, mode="collection")
    assert rc == 1
    assert any(f.get("source") == "pytest-collection" for f in result["findings"])


@pytest.mark.parametrize("exemption", ["skip", "expectedFailure"])
def test_b10_collected_unittest_exemptions(tmp_path: Path, exemption: str) -> None:
    """Guards B8R3-1 at 811de76c for pytest's unittest collection backend."""
    source = "import unittest as u\nclass TestA(u.TestCase):\n    def test_a(self): self.assertTrue(True)\n"
    files = {"pytest.ini": "[pytest]\n", "test_a.py": source}
    base = arm(tmp_path, "base", files, run=False)
    mark = f'getattr(*(u, "{exemption}"))' + (
        '("dummy")' if exemption == "skip" else ""
    )
    files["test_a.py"] = source.replace(
        "    def test_a", f"    @{mark}\n    def test_a"
    )
    tip = arm(tmp_path, "tip", files, run=False)
    rc, _result = compare(tmp_path, base, tip, mode="collection")
    assert rc == (1 if exemption == "expectedFailure" else 0)
    sites = json.loads(tip.read_text())["static"]["sites"]
    assert any(
        s.get("source") == "pytest-collection" and s["kind"] == f"unittest.{exemption}"
        for s in sites
    )


@pytest.mark.parametrize(
    "mark", ["xfail(reason=object())", 'skipif(object(), reason="dummy")']
)
def test_b10_unrepresentable_collected_marks_are_invalid(
    tmp_path: Path, mark: str
) -> None:
    """Guards B8R3-1 at 811de76c: unknown values cannot erase evidence."""
    root = make(
        tmp_path / "tree",
        {
            "pytest.ini": "[pytest]\n",
            "test_a.py": f"import pytest\n@pytest.mark.{mark}\ndef test_a(): assert True\n",
        },
    )
    assert si.run_collect_only(root, [], [])["errors"]


@pytest.mark.parametrize("malformed", [False, True])
def test_b10_missing_collector_evidence_is_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, malformed: bool
) -> None:
    """Guards B8R3-1 at 811de76c against missing or truncated evidence."""
    root = make(
        tmp_path / "tree",
        {"pytest.ini": "[pytest]\n", "test_a.py": "def test_a(): assert True\n"},
    )

    def incomplete(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        junit = next(a.split("=", 1)[1] for a in argv if a.startswith("--junitxml="))
        Path(junit).write_text("<testsuites><testsuite /></testsuites>")
        if malformed:
            (Path(junit).parent / "_suite_inventory_collector.json").write_text(
                '{"items":'
            )
        return subprocess.CompletedProcess(
            argv, 0, "test_a.py::test_a\n\n1 test collected\n", ""
        )

    monkeypatch.setattr(si.subprocess, "run", incomplete)
    assert si.run_collect_only(root, [], [])["errors"]


@pytest.mark.parametrize(
    "container",
    [
        "({target}, {attribute})",
        "({target}, {attribute}, None)",
        "[{target}, {attribute}]",
        "[{target}, {attribute}, None]",
    ],
)
@pytest.mark.parametrize(
    "target, attribute, kind",
    [
        ("p.mark", "xfail", "pytest.mark.xfail"),
        ("p.mark", "skip", "pytest.mark.skip"),
        ("p.mark", "skipif", "pytest.mark.skipif"),
        ("p", "skip", "pytest.skip"),
        ("p", "xfail", "pytest.xfail"),
        ("p", "importorskip", "pytest.importorskip"),
        ("u", "skip", "unittest.skip"),
        ("u", "skipIf", "unittest.skipIf"),
        ("u", "skipUnless", "unittest.skipUnless"),
        ("u", "expectedFailure", "unittest.expectedFailure"),
        ("u.TestCase", "skipTest", "unittest.TestCase.skipTest"),
    ],
)
def test_b10_literal_unpacking_retains_static_call_inventory(
    tmp_path: Path, container: str, target: str, attribute: str, kind: str
) -> None:
    """Guards B8R3-1 at 811de76c for calls outside collection-time marks."""
    args = container.format(target=target, attribute=repr(attribute))
    result = _sites(
        tmp_path,
        f"import pytest as p\nimport unittest as u\ngetattr(*{args})('dummy')\n",
    )
    assert [s["kind"] for s in result["sites"]] == [kind]
    assert not result["dynamic"]


def test_b10_discovery_leg_variants_preserve_mark_multiplicity(tmp_path: Path) -> None:
    """Guards B8R3-1 at 811de76c across split primary discovery settings."""
    files = {
        "pytest.ini": "[pytest]\n",
        "conftest.py": 'import pytest\ndef pytest_addoption(parser): parser.addoption("--exempt", action="store_true")\ndef pytest_collection_modifyitems(config, items):\n    if config.getoption("--exempt"):\n        for item in items:\n            for _ in range(2): item.add_marker(getattr(pytest.mark, "x" + "fail")(False, reason="dummy"))\n',
        "test_a.py": "def test_a(): assert True\n",
    }
    base = arm(tmp_path, "base", files, run=False)
    tip = tmp_path / "tip.json"
    root = make(tmp_path / "tip", files)
    assert collect_with(root, tip, "--leg=", "--leg=--exempt") == 0
    rc, result = compare(tmp_path, base, tip, mode="collection")
    assert rc == 1
    findings = [f for f in result["findings"] if f.get("source") == "pytest-collection"]
    assert len(findings) == 1
    assert findings[0]["added"] == 2
    # An identical pair retains both copies once across selected/complete runs.
    # D790 also requires authorization for inherited unresolved constructions.
    rc, result = compare(tmp_path, tip, tip, mode="collection")
    assert rc == 1
    assert all(f["kind"] == "unknown-exemption" for f in result["findings"])
    assert not [f for f in result["findings"] if f.get("source") == "pytest-collection"]


@pytest.mark.parametrize("mode", ["collection", "runtime"])
@pytest.mark.parametrize("api", ["wrapper", "hookwrapper"])
@pytest.mark.parametrize("priority", ["tryfirst=True", "", "trylast=True"])
@pytest.mark.parametrize(
    "hook",
    [
        "pytest_collection_modifyitems",
        "pytest_collection_finish",
        "pytest_collection",
        "pytest_sessionfinish",
        "pytest_unconfigure",
    ],
)
@pytest.mark.parametrize(
    "exemption",
    ["skip", "skipif", "xfail", "unittest.skip", "unittest.expectedFailure"],
)
def test_b10r1_exemptions_are_captured_after_hook_stacks_return(
    tmp_path: Path, mode: str, api: str, priority: str, hook: str, exemption: str
) -> None:
    """Guards B8R3-CONFIRM-1 at a01c5932 beyond every hook wrapper's unwind."""
    unittest = exemption.startswith("unittest.")
    body = (
        "import unittest\nclass TestA(unittest.TestCase):\n    def test_a(self): assert True\n"
        if unittest
        else "def test_a(): assert True\n"
    )
    files = {"pytest.ini": "[pytest]\n", "test_a.py": body}
    base = arm(tmp_path, "base", files, run=mode == "runtime")
    if unittest:
        attribute = (
            "__unittest_skip__"
            if exemption == "unittest.skip"
            else "__unittest_expecting_failure__"
        )
        mutation = f"setattr(item.cls, {attribute!r}, True)"
        kind = exemption
    else:
        # A computed name has no supplementary static xfail site: pytest's
        # actual items must supply the evidence, regardless of spelling.
        arguments = (
            'strict=False, reason="dummy"'
            if exemption == "xfail"
            else (
                'False, reason="dummy"' if exemption == "skipif" else 'reason="dummy"'
            )
        )
        mutation = (
            f"item.add_marker(getattr(pytest.mark, ''.join({list(exemption)!r}))"
            f"({arguments}))"
        )
        kind = f"pytest.mark.{exemption}"
    parameter = "config" if hook == "pytest_unconfigure" else "session"
    files["conftest.py"] = (
        "import pytest\n"
        "_session = None\n"
        "def pytest_sessionstart(session):\n    global _session\n    _session = session\n"
        f"@pytest.hookimpl({api}=True{', ' + priority if priority else ''})\n"
        f"def {hook}({parameter}):\n"
        "    result = yield\n"
        "    for item in _session.items:\n"
        f"        {mutation}\n"
        "    return result\n"
    )
    tip = arm(tmp_path, "tip", files, run=mode == "runtime")
    inv = json.loads(tip.read_text())
    rc, result = compare(tmp_path, base, tip, mode=mode)
    if exemption in ("xfail", "unittest.expectedFailure"):
        assert rc == 1
        assert any(
            f["kind"] == "xfail-site" and f.get("source") == "pytest-collection"
            for f in result["findings"]
        )
    else:
        assert result["report_only"]["new_skip_sites"]
    if mode == "runtime" and exemption == "xfail":
        # Non-strict XPASS and marks added during shutdown both look passed
        # in JUnit; neither may erase the collected exemption.
        assert all(o["outcome"] == "passed" for o in si.primary_outcomes(inv).values())
    # Each selected/complete discovery leg must independently see the mark.
    for run in [*inv["collections"]["selected"], inv["collections"]["all"]]:
        assert not run["errors"]
        assert [m["kind"] for i in run["marked_items"] for m in i["marks"]] == [kind]


@pytest.mark.parametrize("mode", ["collection", "runtime"])
@pytest.mark.parametrize(
    "cleanup,api,priority",
    [("config", "", "")]
    + [
        (cleanup, api, priority)
        for cleanup in ("sessionfinish", "unconfigure")
        for api in ("wrapper", "hookwrapper")
        for priority in ("tryfirst=True", "", "trylast=True")
    ],
)
@pytest.mark.parametrize("inherited", [False, True])
@pytest.mark.parametrize(
    "exemption",
    ["skip", "skipif", "xfail", "unittest.skip", "unittest.expectedFailure"],
)
def test_b11_cleanup_cannot_erase_collected_exemptions(
    tmp_path: Path,
    mode: str,
    cleanup: str,
    api: str,
    priority: str,
    inherited: bool,
    exemption: str,
) -> None:
    """Guards B10R1-CONFIRM-1 at 24a5440b across teardown and both backends."""
    unittest = exemption.startswith("unittest.")
    body = (
        "import unittest\nclass TestA(unittest.TestCase):\n    def test_a(self): assert True\n"
        if unittest
        else "class TestA:\n    def test_a(self): assert True\n"
    )
    files = {"pytest.ini": "[pytest]\n", "test_a.py": body}
    base = arm(tmp_path, "base", files, run=mode == "runtime")
    if unittest:
        target = "item.cls" if inherited else "item.obj.__func__"
        attribute = (
            "__unittest_skip__"
            if exemption == "unittest.skip"
            else "__unittest_expecting_failure__"
        )
        mutation = f"patch.setattr({target}, {attribute!r}, True, raising=False)"
        kind = exemption
    else:
        target = "item.parent" if inherited else "item"
        arguments = (
            'strict=False, reason="dummy"'
            if exemption == "xfail"
            else 'reason="dummy"'
            if exemption == "skip"
            else 'False, reason="dummy"'
        )
        mutation = (
            f"mark = getattr(pytest.mark, ''.join({list(exemption)!r}))({arguments})\n"
            f"        patch.setattr({target}, 'own_markers', [*{target}.own_markers, mark.mark])"
        )
        kind = f"pytest.mark.{exemption}"
    files["conftest.py"] = (
        "import pytest\npatch = pytest.MonkeyPatch()\n"
        "def pytest_collection_modifyitems(config, items):\n"
        + ("    config.add_cleanup(patch.undo)\n" if cleanup == "config" else "")
        + "    for item in items:\n        "
        + mutation
        + "\n"
    )
    if cleanup != "config":
        files["conftest.py"] += (
            f"@pytest.hookimpl({api}=True{', ' + priority if priority else ''})\n"
            f"def pytest_{cleanup}():\n    result = yield\n    patch.undo()\n    return result\n"
        )
    tip = arm(tmp_path, "tip", files, run=mode == "runtime")
    inv = json.loads(tip.read_text())
    if mode == "runtime" and exemption == "xfail":
        assert all(o["outcome"] == "passed" for o in si.primary_outcomes(inv).values())
    for run in [*inv["collections"]["selected"], inv["collections"]["all"]]:
        assert not run["errors"]
        assert [m["kind"] for i in run["marked_items"] for m in i["marks"]] == [kind]
    rc, result = compare(tmp_path, base, tip, mode=mode)
    if exemption in ("xfail", "unittest.expectedFailure"):
        assert rc == 1
        assert any(f.get("source") == "pytest-collection" for f in result["findings"])
    else:
        assert result["report_only"]["new_skip_sites"]


@pytest.mark.parametrize("mutable", [False, True])
def test_b11_snapshot_keeps_argument_variants_and_duplicate_counts(
    tmp_path: Path, mutable: bool
) -> None:
    """Guards B10R1-CONFIRM-1 at 24a5440b against cleanup of values and duplicates."""
    files = {"pytest.ini": "[pytest]\n", "test_a.py": "def test_a(): assert True\n"}
    base = arm(tmp_path, "base", files, run=False)
    files["conftest.py"] = (
        "import pytest\npatch = pytest.MonkeyPatch()\n"
        "def pytest_collection_modifyitems(config, items):\n"
        "    mark = getattr(pytest.mark, 'x' + 'fail')(reason=['before'])\n"
        "    for item in items: patch.setattr(item, 'own_markers', [mark.mark, mark.mark])\n"
        + (
            "    config.add_cleanup(lambda: mark.kwargs['reason'].clear())\n"
            if mutable
            else "    config.add_cleanup(patch.undo)\n"
        )
    )
    tip = arm(tmp_path, "tip", files, run=False)
    rc, result = compare(tmp_path, base, tip, mode="collection")
    assert rc == 1
    findings = [f for f in result["findings"] if f.get("source") == "pytest-collection"]
    assert any(f["added"] == 2 and "before" in f["id"] for f in findings)
    if mutable:
        assert len(findings) == 2  # both immutable argument variants, twice each
    else:
        assert len(findings) == 1


@pytest.mark.parametrize("inherited", [False, True])
@pytest.mark.parametrize("mode", ["collection", "runtime"])
def test_b11_unresolved_exemptions_require_exact_authorization(
    tmp_path: Path, inherited: bool, mode: str
) -> None:
    """Guards B10R1-CONFIRM-1 at 24a5440b: unknown never means no exemption."""
    files = {"pytest.ini": "[pytest]\n", "test_a.py": "def test_a(): assert True\n"}
    uncertain = "import pytest\ndef pytest_collection_modifyitems(items):\n    for item in items: item.add_marker(getattr(pytest.mark, 'x' + 'fail')(False))\n"
    if inherited:
        files["conftest.py"] = uncertain
    base = arm(tmp_path, "base", files, run=mode == "runtime")
    files["conftest.py"] = uncertain
    tip = arm(tmp_path, "tip", files, run=mode == "runtime")
    rc, result = compare(tmp_path, base, tip, mode=mode)
    assert rc == 1
    unknowns = [f for f in result["findings"] if f["kind"] == "unknown-exemption"]
    assert unknowns
    assert all("supported literal form" in f["reason"] for f in unknowns)
    # An unrelated xfail-site authorization cannot authorize the unknown.
    assert all(f["allowed_by"] is None for f in unknowns)
    allow = ""
    decisions = ""
    for index, finding in enumerate(result["findings"], start=200):
        ref = f"PY8-D{index}"
        allow += allow_line(finding["kind"], finding["id"], "-", ref)
        decisions += f"- **{ref} (GATE-noskip, fixture):** {finding['kind']} `{finding['id']}` reviewed.\n"
    rc, listed = compare(
        tmp_path,
        base,
        tip,
        mode=mode,
        allow=allow,
        decisions=decisions,
        strict=True,
        name="listed",
    )
    assert rc == 0
    assert not open_findings(listed)


@pytest.mark.parametrize(
    "expression",
    [
        "getattr(pytest.mark, name)",
        "getattr(*args)",
        "getattr(**kwargs)",
        "getattr(unknown, 'xfail')",
        "unknown.skip()",
        "unknown.skipif()",
        "unknown.xfail()",
        "unknown.importorskip()",
        "unknown.skipIf()",
        "unknown.skipUnless()",
        "unknown.expectedFailure()",
        "unknown.skipTest()",
        "__import__('pytest')",
        "importlib.import_module('unittest')",
        "item.add_marker(factory())",
        "item.applymarker(factory())",
        "(lambda p: p.mark.xfail)(unknown)",
    ],
)
def test_b11_uncertain_exemption_forms_cannot_compare_clean(
    tmp_path: Path, expression: str
) -> None:
    """Guards B10R1-CONFIRM-1 at 24a5440b for the entire uncertainty registry."""
    files = {"pytest.ini": "[pytest]\n", "test_a.py": "def test_a(): assert True\n"}
    base = arm(tmp_path, "base", files, run=False)
    # An unexecuted helper has no collection mark or JUnit exemption to use
    # as attribution. Conservative refusal must give an actionable next step.
    files["conftest.py"] = (
        "import pytest\nimport importlib\ndef helper():\n    " + expression + "\n"
    )
    tip = arm(tmp_path, "tip", files, run=False)
    rc, result = compare(tmp_path, base, tip, mode="collection")
    assert rc == 1
    assert any(f["kind"] == "unknown-exemption" for f in result["findings"])


def test_b11_unencodable_pre_cleanup_evidence_stays_invalid(tmp_path: Path) -> None:
    """Guards B10R1-CONFIRM-1 at 24a5440b against unsupported values erased by cleanup."""
    root = make(
        tmp_path / "tree",
        {
            "pytest.ini": "[pytest]\n",
            "test_a.py": "def test_a(): assert True\n",
            "conftest.py": "import pytest\ndef pytest_collection_modifyitems(config, items):\n    patch = pytest.MonkeyPatch()\n    config.add_cleanup(patch.undo)\n    for item in items: patch.setattr(item, 'own_markers', [pytest.mark.xfail(reason=object()).mark])\n",
        },
    )
    run = si.run_collect_only(root, [], [])
    assert run["errors"]
    assert any("supported pytest/unittest" in error for error in run["errors"])


@pytest.mark.parametrize(
    "identity",
    [
        None,
        ("_pytest.python", "Function"),
        ("_pytest.unittest", "TestCaseFunction"),
        *[
            ("pytest_asyncio.plugin", name)
            for name in (
                "Coroutine",
                "AsyncGenerator",
                "AsyncStaticMethod",
                "AsyncHypothesisTest",
            )
        ],
    ],
)
def test_b11_unknown_collection_backend_is_invalid(
    tmp_path: Path, identity: tuple[str, str] | None
) -> None:
    """Guards B10R1-CONFIRM-1 at 24a5440b: custom backends need an explicit model."""
    root = make(
        tmp_path / "tree",
        {
            "pytest.ini": "[pytest]\n",
            "test_a.py": "def test_a(): assert True\n",
            "conftest.py": "import pytest\nclass Custom(pytest.Function): pass\n"
            + (
                f"Custom.__module__ = {identity[0]!r}\nCustom.__name__ = {identity[1]!r}\n"
                if identity
                else ""
            )
            + "def pytest_pycollect_makeitem(collector, name, obj):\n    if name == 'test_a': return Custom.from_parent(collector, name=name)\n",
        },
    )
    run = si.run_collect_only(root, [], [])
    assert run["ids"] == ["test_a.py::test_a"]
    assert run["errors"]


def test_b11_count_reconciliation_keeps_pre_cleanup_marks(tmp_path: Path) -> None:
    """Guards B10R1-CONFIRM-1 at 24a5440b on the second collection entry point."""
    root = make(
        tmp_path / "tree",
        {
            "pytest.ini": "[pytest]\n",
            "test_a.py": "def test_a(): assert True\ndef test_b(): assert True\n",
            "conftest.py": "import pytest\ndef pytest_collection_modifyitems(config, items):\n    items.pop()\n    patch = pytest.MonkeyPatch()\n    config.add_cleanup(patch.undo)\n    for item in items: patch.setattr(item, 'own_markers', [pytest.mark.xfail(strict=False).mark])\n",
        },
    )
    run = si.run_collect_only(root, [], [])
    assert run["ids"] == ["test_a.py::test_a"]
    assert not run["errors"]
    assert run["marked_items"][0]["marks"][0]["kind"] == "pytest.mark.xfail"


@pytest.mark.parametrize("name", ["skip", "skipif", "xfail"])
@pytest.mark.parametrize(
    "lookup",
    [
        "getattr(pytest.mark, {name!r})",
        "getattr(pytest.mark, {name!r}, None)",
        "getattr(*(pytest.mark, {name!r}))",
        "getattr(*[pytest.mark, {name!r}, None])",
        "pytest.mark.{name}",
    ],
)
def test_b11_literal_mark_alias_stays_supported(
    tmp_path: Path, name: str, lookup: str
) -> None:
    """Guards B10R1-CONFIRM-1 at 24a5440b: supported aliases need no unknown allow."""
    source = (
        "import pytest\nmark = " + lookup.format(name=name) + "\n"
        "def pytest_collection_modifyitems(items):\n"
        "    for item in items: item.add_marker(mark(False, reason='dummy'))\n"
    )
    assert not _sites(tmp_path, source)["dynamic"]
    files = {"pytest.ini": "[pytest]\n", "test_a.py": "def test_a(): assert True\n"}
    base = arm(tmp_path, "base", files, run=False)
    files["conftest.py"] = source
    tip = arm(tmp_path, "tip", files, run=False)
    rc, result = compare(tmp_path, base, tip, mode="collection")
    assert rc == (1 if name == "xfail" else 0)
    assert not [f for f in result["findings"] if f["kind"] == "unknown-exemption"]


@pytest.mark.parametrize("mode", ["collection", "runtime"])
def test_b11_reviewed_cleanup_repro(tmp_path: Path, mode: str) -> None:
    """Guards B10R1-CONFIRM-1 at 24a5440b with the exact reported active XPASS."""
    files = {"pytest.ini": "[pytest]\n", "test_a.py": "def test_a(): assert True\n"}
    base = arm(tmp_path, "base", files, run=mode == "runtime")
    files["conftest.py"] = (
        "import pytest\ndef pytest_collection_modifyitems(config, items):\n"
        "    patch = pytest.MonkeyPatch()\n    config.add_cleanup(patch.undo)\n"
        "    mark = getattr(pytest.mark, 'x' + 'fail')(strict=False)\n"
        "    for item in items: patch.setattr(item, 'own_markers', [*item.own_markers, mark.mark])\n"
    )
    tip = arm(tmp_path, "tip", files, run=mode == "runtime")
    rc, result = compare(tmp_path, base, tip, mode=mode)
    assert rc == 1
    assert any(f.get("source") == "pytest-collection" for f in result["findings"])
    if mode == "runtime":
        assert (
            si.primary_outcomes(json.loads(tip.read_text()))["test_a.py::test_a"][
                "outcome"
            ]
            == "passed"
        )


@pytest.mark.parametrize(
    "backend",
    ["Coroutine", "AsyncGenerator", "AsyncStaticMethod", "AsyncHypothesisTest"],
)
@pytest.mark.parametrize("name", ["skip", "skipif", "xfail"])
def test_b11_trusted_asyncio_function_backends_are_modeled(
    tmp_path: Path, backend: str, name: str
) -> None:
    """Guards B10R1-CONFIRM-1's supported backend regression at 21b9b45a."""
    root = make(
        tmp_path / "tree",
        {
            "pytest.ini": "[pytest]\n",
            "test_a.py": f"import pytest\n@pytest.mark.{name}(False, reason='dummy')\ndef test_a(): assert True\n",
            "conftest.py": f"from pytest_asyncio.plugin import {backend}\ndef pytest_pycollect_makeitem(collector, name, obj):\n    if name == 'test_a': return {backend}.from_parent(collector, name=name)\n",
        },
    )
    run = si.run_collect_only(root, [], [])
    assert not run["errors"]
    assert [m["kind"] for item in run["marked_items"] for m in item["marks"]] == [
        f"pytest.mark.{name}"
    ]


@pytest.mark.parametrize("mode", ["collection", "runtime"])
def test_b11_real_asyncio_cleanup_preserves_active_xfail(
    tmp_path: Path, mode: str
) -> None:
    """Guards B10R1-CONFIRM-1 at 24a5440b and honest async support at 21b9b45a."""
    files = {
        "pytest.ini": "[pytest]\n",
        "conftest.py": "pytest_plugins = ['pytest_asyncio.plugin']\n",
        "test_a.py": "import pytest\n@pytest.mark.asyncio\nasync def test_a(): assert True\n",
    }
    base = arm(tmp_path, "base", files, run=mode == "runtime")
    files["conftest.py"] += (
        "import pytest\ndef pytest_collection_modifyitems(config, items):\n"
        "    patch = pytest.MonkeyPatch()\n    config.add_cleanup(patch.undo)\n"
        "    for item in items: patch.setattr(item, 'own_markers', [*item.own_markers, pytest.mark.xfail(strict=False).mark])\n"
    )
    tip = arm(tmp_path, "tip", files, run=mode == "runtime")
    for run in [
        *json.loads(tip.read_text())["collections"]["selected"],
        json.loads(tip.read_text())["collections"]["all"],
    ]:
        assert any(
            m["kind"] == "pytest.mark.xfail"
            for i in run["marked_items"]
            for m in i["marks"]
        )
    rc, result = compare(tmp_path, base, tip, mode=mode)
    assert rc == 1
    assert any(f.get("source") == "pytest-collection" for f in result["findings"])
    if mode == "runtime":
        assert (
            si.primary_outcomes(json.loads(tip.read_text()))["test_a.py::test_a"][
                "outcome"
            ]
            == "passed"
        )


_B11R1_APPLICATIONS = [
    'request.node.add_marker(marker="xfail")',
    'request.node.add_marker(**{"marker": "xfail", "append": False})',
    'request.node.add_marker(*["xfail"])',
    'request.node.own_markers.append(operator.attrgetter("xfail")(pytest.mark)(strict=False).mark)',
    'request.node.own_markers.extend([operator.attrgetter("xfail")(pytest.mark)(strict=False).mark])',
    'request.node.own_markers[:] = [operator.attrgetter("xfail")(pytest.mark)(strict=False).mark]',
    'request.node.add_marker(operator.attrgetter("xfail")(pytest.mark)(strict=False))',
    'operator.methodcaller("add_marker", "xfail")(request.node)',
    'apply = request.node.add_marker; apply(marker="xfail")',
    'getattr(request.node, "add_marker")(marker="xfail")',
    'request.applymarker(marker="xfail")',
    'operator.methodcaller("add_" + "marker", "x" + "fail")(request.node)',
    'vars(request.node)["own_markers"].append(operator.attrgetter("xfail")(pytest.mark)(strict=False).mark)',
]


@pytest.mark.parametrize("mode", ["collection", "runtime"])
@pytest.mark.parametrize("backend", ["pytest", "unittest", "asyncio"])
@pytest.mark.parametrize("application", _B11R1_APPLICATIONS)
def test_b11r1_runtime_construction_class_is_conservative(
    tmp_path: Path, mode: str, backend: str, application: str
) -> None:
    """B11-CONFIRM-1: source-only uncertainty and real XPASS must both gate."""
    source = "def test_a(): assert True\n"
    if backend == "unittest":
        source = "import unittest\nclass TestA(unittest.TestCase):\n    def test_a(self): assert True\n"
    elif backend == "asyncio":
        source = (
            "import pytest\n@pytest.mark.asyncio\nasync def test_a(): assert True\n"
        )
    files = {"pytest.ini": "[pytest]\nxfail_strict=false\n", "test_a.py": source}
    if backend == "asyncio":
        files["conftest.py"] = "pytest_plugins = ['pytest_asyncio.plugin']\n"
    base = arm(tmp_path, "base", files, run=mode == "runtime")
    files["conftest.py"] = files.get("conftest.py", "") + (
        "import pytest\nimport operator\n@pytest.fixture(autouse=True)\n"
        "def exempt(request):\n    " + application + "\n"
    )
    tip = arm(tmp_path, "tip", files, run=mode == "runtime")
    rc, result = compare(tmp_path, base, tip, mode=mode)
    assert rc == 1
    assert open_findings(result)
    if mode == "runtime":
        outcomes = si.primary_outcomes(json.loads(tip.read_text()))
        assert outcomes and all("xfail" in o["exemptions"] for o in outcomes.values())
        assert any(f["kind"] == "new-xfail" for f in result["findings"])


@pytest.mark.parametrize(
    "expression",
    [
        'operator.attrgetter("xfail")(pytest.mark)',
        "operator.attrgetter(name)(pytest.mark)",
        "factory(pytest.mark)",
        "vars(pytest.mark)",
        "pytest.mark.__dict__",
        "marks = pytest.mark",
        "p = pytest",
        "request.node.own_markers",
        "request.node.keywords",
        "obj.__unittest_expecting_failure__ = True",
        "request.node.add_marker(marker=factory())",
        "request.node.add_marker(**kwargs)",
    ],
)
@pytest.mark.parametrize("local_import", [False, True])
def test_b11r1_namespace_and_metadata_escape_requires_listing(
    tmp_path: Path, expression: str, local_import: bool
) -> None:
    """B11-CONFIRM-1: unexecuted unsupported forms cannot disappear as safe."""
    source = (
        "import pytest\nimport operator\ndef helper(request):\n    " + expression + "\n"
    )
    if local_import:
        source = (
            "def helper(request):\n    import pytest\n    import operator\n    "
            + expression
            + "\n"
        )
    assert _sites(tmp_path, source)["dynamic"]


@pytest.mark.parametrize("mode", ["collection", "runtime"])
@pytest.mark.parametrize(
    "addition",
    [
        'getattr(pytest.mark, "x" + "fail")(strict=False).mark',
        'getattr(pytest.mark, "sm" + "oke")().mark',
        'getattr(pytest.mark, "smo" + "ke")().mark',
    ],
)
@pytest.mark.parametrize("same_line", [False, True])
def test_b11r1_unknown_authorization_is_per_construction_and_occurrence(
    tmp_path: Path, mode: str, addition: str, same_line: bool
) -> None:
    """B11-CONFIRM-2: one authorized unknown cannot absorb a distinct/extra site."""
    first = 'request.node.own_markers.append(getattr(pytest.mark, "sm" + "oke")().mark)'
    prefix = "import pytest\n@pytest.fixture(autouse=True)\ndef tag(request):\n    "
    files = {
        "pytest.ini": "[pytest]\nxfail_strict=false\nmarkers=smoke: harmless\n",
        "test_a.py": "def test_a(): assert True\n",
        "conftest.py": prefix + first + "\n",
    }
    base = arm(tmp_path, "base", files, run=mode == "runtime")
    _, baseline = compare(tmp_path, base, base, mode=mode, name="baseline")
    allow = ""
    decisions = ""
    for index, finding in enumerate(baseline["findings"], start=500):
        ref = f"PY8-D{index}"
        allow += allow_line(finding["kind"], finding["id"], "-", ref)
        decisions += f"- **{ref} (GATE-noskip, fixture):** {finding['kind']} `{finding['id']}` reviewed.\n"
    assert (
        compare(
            tmp_path,
            base,
            base,
            mode=mode,
            allow=allow,
            decisions=decisions,
            strict=True,
            name="authorized",
        )[0]
        == 0
    )
    separator = "; " if same_line else "\n    "
    files["conftest.py"] = (
        prefix + first + separator + f"request.node.own_markers.append({addition})\n"
    )
    tip = arm(tmp_path, "tip", files, run=mode == "runtime")
    rc, result = compare(
        tmp_path,
        base,
        tip,
        mode=mode,
        allow=allow,
        decisions=decisions,
        strict=True,
        name="extra",
    )
    assert rc == 1
    assert any(kind == "unknown-exemption" for kind, _ in open_findings(result))
    ids = [f["id"] for f in result["findings"] if f["kind"] == "unknown-exemption"]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("workers", [False, True])
@pytest.mark.parametrize("cleanup", [False, True])
@pytest.mark.parametrize("kind", ["xfail", "skip"])
def test_b11r1_runtime_evidence_keeps_exemptions_and_xpass(
    tmp_path: Path, workers: bool, cleanup: bool, kind: str
) -> None:
    """B11-CONFIRM-1: JUnit passed is insufficient, including local xdist workers."""
    files = {
        "pytest.ini": "[pytest]\nxfail_strict=false\n",
        "test_a.py": "def test_a(): assert True\n",
        "conftest.py": "import pytest\n@pytest.fixture(autouse=True)\ndef exempt(request):\n"
        f"    request.node.add_marker(marker='{kind}')\n"
        + ("    pytest.skip('dummy')\n" if kind == "skip" else "")
        + "    yield\n"
        + ("    request.node.own_markers.clear()\n" if cleanup else ""),
    }
    legs = (("primary", (), ("-p", "xdist.plugin", "-n", "2") if workers else ()),)
    tip = arm(tmp_path, "tip", files, legs=legs)
    outcome = si.primary_outcomes(json.loads(tip.read_text()))["test_a.py::test_a"]
    assert kind in outcome["exemptions"]
    if kind == "xfail":
        assert outcome["outcome"] == "passed"  # non-strict XPASS's lossy JUnit
    manifest = json.loads((tmp_path / "tip-run" / "manifest.json").read_text())
    assert manifest["legs"][0].get("runtime_sha256")


@pytest.mark.parametrize("mutation", ["missing", "changed", "empty", "unencodable"])
def test_b11r1_runtime_evidence_fails_closed(tmp_path: Path, mutation: str) -> None:
    """Missing, altered and unrepresentable evidence is never a clean pass."""
    root = make(
        tmp_path / "tree",
        {
            "pytest.ini": "[pytest]\n",
            "test_a.py": "def test_a(): assert True\n",
            "conftest.py": (
                "import pytest\n@pytest.fixture(autouse=True)\ndef exempt(request):\n"
                "    request.node.add_marker(pytest.mark.xfail(reason=object()))\n"
                if mutation == "unencodable"
                else ""
            ),
        },
    )
    manifest = tmp_path / "runs" / "manifest.json"
    manifest.parent.mkdir()
    run_leg(manifest, root, "primary")
    data = json.loads(manifest.read_text())
    path = data["legs"][0].get("runtime_evidence")
    assert path, "run-leg must retain runtime exemptions independent of JUnit"
    if mutation == "missing":
        Path(path).unlink()
    elif mutation in ("changed", "empty"):
        Path(path).write_text('{"captures": []}')
        if mutation == "empty":
            data["legs"][0]["runtime_sha256"] = si.sha256_file(Path(path))
            manifest.write_text(json.dumps(data))
    out = tmp_path / "inventory.json"
    assert (
        si.main(
            [
                "collect",
                "--root",
                str(root),
                "--out",
                str(out),
                "--run-manifest",
                str(manifest),
            ]
        )
        == 2
    )
    assert json.loads(out.read_text())["evidence_errors"]


@pytest.mark.parametrize(
    "source",
    [
        'import operator\napply = operator.methodcaller("add_marker", "xfail")\ndef helper(node): apply(node)\n',
        'from operator import methodcaller as factory\napply = factory("add_marker", "xfail")\ndef helper(node): apply(node)\n',
        'lookup = getattr\ndef helper(node): lookup(node, "own_markers")\n',
        "run = eval\ndef helper(node): run(code)\n",
        "def helper(node): node.__dict__[key] = value\n",
        'def helper(request):\n    apply = request.node.add_marker\n    apply(marker="xfail")\n',
    ],
)
def test_b11r1_opaque_reflection_aliases_require_authorization(
    tmp_path: Path, source: str
) -> None:
    """B11-CONFIRM-1: storing a reflective factory cannot turn unknown into safe."""
    assert _sites(tmp_path, source)["dynamic"]


def test_b11r1_runtime_capture_keeps_items_removed_at_shutdown(tmp_path: Path) -> None:
    """B11-CONFIRM-1: retain runtime snapshots independently of session.items."""
    files = {
        "pytest.ini": "[pytest]\nxfail_strict=false\n",
        "test_a.py": "def test_a(): assert True\n",
        "conftest.py": "import pytest\n@pytest.fixture(autouse=True)\ndef exempt(request):\n"
        "    request.node.add_marker(marker='xfail')\n"
        "def pytest_sessionfinish(session):\n"
        "    if not session.config.option.collectonly: session.items.clear()\n",
    }
    tip = arm(tmp_path, "tip", files)
    inv = json.loads(tip.read_text())
    assert "xfail" in si.primary_outcomes(inv)["test_a.py::test_a"]["exemptions"]
    assert any(s["family"] == "xfail" for s in inv["runtime"].get("sites", []))


@pytest.mark.parametrize(
    "marker",
    [
        'pytest.Mark("xfail", (), {"strict": False})',
        "pytest.MarkDecorator(factory())",
        "pytest.mark",
        "pytest.mark.smoke.with_args()",
        "pytest.fixture(factory())",
    ],
)
@pytest.mark.parametrize(
    "binding", ["{marker}", "marker={marker}", '**{{"marker": {marker}}}']
)
def test_b11r1_marker_application_admits_only_modeled_values(
    tmp_path: Path, marker: str, binding: str
) -> None:
    """B11-CONFIRM-1: a resolved dotted name is not proof of a modeled marker."""
    source = (
        "import pytest\ndef helper(request):\n    request.node.add_marker("
        + binding.format(marker=marker)
        + ")\n"
    )
    assert any(
        d["what"] == "add_marker(<not a literal mark>)"
        for d in _sites(tmp_path, source)["dynamic"]
    )


def test_b11r1_supported_inactive_skip_is_still_a_passing_test(tmp_path: Path) -> None:
    """B11-CONFIRM-1 control: marker retention must not invent an active skip."""
    files = {"pytest.ini": "[pytest]\n", "test_a.py": "def test_a(): assert True\n"}
    base = arm(tmp_path, "base", files)
    files["test_a.py"] = (
        "import pytest\n@pytest.mark.skipif(False, reason='dummy')\n"
        + files["test_a.py"]
    )
    tip = arm(tmp_path, "tip", files)
    assert compare(tmp_path, base, tip)[0] == 0
    outcome = si.primary_outcomes(json.loads(tip.read_text()))["test_a.py::test_a"]
    assert outcome["outcome"] == "passed" and not outcome["exemptions"]
