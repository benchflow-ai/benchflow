"""``bench explain``: a page for every cause the run summary names.

Round PY8's D2: each cause :func:`benchflow.failures.cause_of`
returns has a page under its id, the code Rust's ``bench`` 0.8.2 uses where
the meaning is the same, else the cause's key; every Rust code Python uses
has one; and each page names a next step that exists.
"""

from __future__ import annotations

import json
import re

import click
import pytest
import typer.main
from typer.core import TyperGroup
from typer.testing import CliRunner

from benchflow import failures
from benchflow.cli.main import app
from tests._package_files import PACKAGE_ROOT

SRC = PACKAGE_ROOT

# Rust bench 0.8.2's run failure codes (crates/benchflow-run/src/outcome.rs
# EXPLANATIONS at bench-rs 518009b0).
RUST_082_RUN_CODES = frozenset(
    [
        "R000",
        "R001",
        "R101",
        "R102",
        "R103",
        "R104",
        "R105",
        "R106",
        "R107",
        "R108",
        "R109",
        "R110",
        "R201",
        "R202",
        "R203",
        "R204",
        "R205",
        "R206",
        "R211",
        "R212",
        "R213",
        "R214",
        "R215",
        "R301",
        "R302",
        "R303",
        "R304",
        "R305",
        "R306",
        "R307",
        "R309",
        "R310",
        "R311",
        "R312",
        "R313",
        "R314",
        "R315",
        "R316",
        "R401",
        "R901",
    ]
)
OWNERS = {"task", "agent", "model", "sandbox", "you", "benchflow"}


def _bench(*args: str):
    return CliRunner().invoke(app, list(args), prog_name="bench")


def _listed() -> list[dict[str, str]]:
    result = _bench("explain", "--json")
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def _page(code: str) -> dict[str, str]:
    result = _bench("explain", code, "--json")
    assert result.exit_code == 0, (code, result.output)
    return json.loads(result.stdout)


def test_every_cause_the_summary_names_has_a_page_with_a_next_step():
    listed = {entry["code"] for entry in _listed()}
    for key in failures.cause_keys():
        code = failures.RUST_CODES.get(key, key)
        assert code in listed, key
        page = _page(code)
        assert page["code"] == code
        assert page["owner"] in OWNERS, page
        assert page["title"].strip() and page["body"].strip(), page
        assert page["next"].strip(), f"{code} names no next step"
        text = _bench("explain", code)
        assert text.exit_code == 0, text.output
        assert f"next: {page['next']}" in text.stdout


def test_every_rust_code_python_uses_has_a_page():
    used = set(failures.RUST_CODES.values())
    # Any Rust run code written in the source (an error's code) counts too.
    pattern = re.compile(r"[\"'`](R\d{3})[\"'`]")
    for path in SRC.rglob("*.py"):
        if "_vendor" in path.parts:
            continue
        used.update(pattern.findall(path.read_text(encoding="utf-8")))
    assert used, "no Rust code found"
    listed = {entry["code"] for entry in _listed()}
    for code in sorted(used):
        assert code in RUST_082_RUN_CODES, f"{code} is not one of Rust 0.8.2's codes"
        assert code in listed, f"{code} has no page"
        assert _page(code)["next"].strip()


def test_each_rust_code_stands_for_one_cause():
    assert len(set(failures.RUST_CODES.values())) == len(failures.RUST_CODES)
    assert set(failures.RUST_CODES) <= set(failures.cause_keys())


def test_a_page_is_found_by_any_case_and_by_its_category():
    assert _page("r212")["code"] == "R212"
    assert _page("provider_auth")["code"] == "R212"
    assert _page("VERIFIER_FAILURE")["code"] == "verifier_failure"


def test_the_list_names_every_id_once():
    codes = [entry["code"] for entry in _listed()]
    assert len(codes) == len(set(codes)) == len(failures.cause_keys())
    text = _bench("explain")
    assert text.exit_code == 0, text.output
    for code in codes:
        assert re.search(rf"^  {re.escape(code)} ", text.stdout, re.M), code


def test_an_id_with_no_page_exits_2_with_a_next_step():
    result = _bench("explain", "R999")
    assert result.exit_code == 2
    flat = " ".join(re.sub(r"[│╭╮╰╯─]", " ", result.output).split())
    assert "no explain page for 'R999'" in flat
    assert "next: `bench explain` lists every code" in flat


def test_a_category_with_a_page_per_cause_names_its_pages():
    result = _bench("explain", "agent_integration")
    assert result.exit_code == 2
    flat = " ".join(re.sub(r"[│╭╮╰╯─]", " ", result.output).split())
    assert "agent_integration:agent_auth" in flat
    assert "agent_integration:unknown" in flat


def _flags(*path: str) -> set[str]:
    command: click.Command = typer.main.get_command(app)
    for word in path:
        assert isinstance(command, TyperGroup)
        command = command.commands[word]
    return {
        opt
        for param in command.params
        for opt in getattr(param, "opts", [])
        if opt.startswith("--")
    }


def test_every_flag_a_page_names_is_a_real_flag():
    """A next step that names a flag bench does not have leads nowhere.

    The timeout cause told users to raise ``--timeout``, which no run command
    has; it is ``--timeout-multiplier``.
    """
    real = (
        _flags("eval", "run")
        | _flags("run")
        | _flags("doctor")
        | _flags("eval", "resume")
        | _flags("eval", "score")
    )
    for entry in _listed():
        page = _page(entry["code"])
        for flag in re.findall(r"(?<![\w-])--[a-z][a-z-]*", page["next"]):
            assert flag in real, (entry["code"], flag)


@pytest.mark.parametrize(
    ("record", "key"),
    [
        # A verifier category from another version: the general row and key.
        (
            {"verifier_error": "verifier exploded", "verifier_error_category": "x"},
            "verifier_other",
        ),
        # A crashed rubric review (review/persistence.py): the infrastructure's.
        (
            {
                "verifier_error": "verifier crashed: oops",
                "verifier_error_category": "infra_failure",
            },
            "infra_failure",
        ),
        # An integration cause no row names, or one that is not a word.
        (
            {
                "error": "agent integration failure",
                "error_category": "agent_integration",
                "integration_failure_info": {"cause": "agent_quota"},
            },
            "agent_integration:unknown",
        ),
        (
            {
                "error": "agent integration failure",
                "error_category": "agent_integration",
                "integration_failure_info": {"cause": ["agent_auth"]},
            },
            "agent_integration:unknown",
        ),
        ({"error": "something odd", "error_category": "brand_new"}, "other"),
    ],
)
def test_every_cause_key_has_a_page(record, key):
    """A key cause_of returns is always an id with a page, whatever the record holds."""
    cause = failures.cause_of(record)
    assert cause is not None
    assert cause.key == key
    assert cause.key in failures.cause_keys()
    assert _page(cause.code)["code"] == cause.code
