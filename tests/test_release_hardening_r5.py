"""Round-5 hardening: the markup-escape class + dataset-registry guards.

The stress sweep found that #732's markup-escaping was per-handler, not
systemic — ~6 more `console.print(f"[red]{user_input}[/red]")` sites (and the
#730 dashboard warning buffer, fed by #733's malformed-task warning) still
raised MarkupError on a `[`-bearing value. The fix is one `print_error()` sink
that escapes, routed everywhere. These tests pin the class closed and pin the
dataset-registry structural guards (raw KeyError/TypeError → DatasetResolutionError).
"""

from __future__ import annotations

import json
import logging

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app

runner = CliRunner()

# Rich markup that raises MarkupError if interpolated unescaped.
_MARKUP = "[/x]"


def test_print_error_escapes_markup() -> None:
    from benchflow.cli._shared import print_error

    # If print_error did not escape, console.print would raise MarkupError here.
    print_error(f"Unknown agent: {_MARKUP} and [bold]unclosed")


def test_warning_buffer_replay_escapes_markup() -> None:
    # #730 B5 buffer replays WARNING+ after the Live; #733's malformed-task
    # warning carries a user dir name that can contain markup.
    from benchflow.cli._live_progress import _WarningBuffer

    buf = _WarningBuffer()
    rec = logging.LogRecord(
        name="benchflow",
        level=logging.WARNING,
        pathname="f",
        lineno=1,
        msg="Skipping malformed task %r: boom",
        args=("a-[/red]-b",),
        exc_info=None,
    )
    buf.emit(rec)
    buf.replay()  # must not raise MarkupError


@pytest.mark.parametrize(
    "argv",
    [
        ["agent", "show", _MARKUP],
        ["eval", "metrics", f"/tmp/nope-{_MARKUP}-dir"],
    ],
)
def test_markup_in_cli_args_exits_clean_not_markup_error(argv) -> None:
    res = runner.invoke(app, argv)
    assert res.exit_code != 0
    # A clean typer.Exit becomes SystemExit; a MarkupError would surface as the
    # uncaught exception instead.
    assert isinstance(res.exception, SystemExit), (
        f"{argv} raised {type(res.exception).__name__}: {res.exception}"
    )


def test_eval_create_invalid_agent_env_markup_does_not_crash() -> None:
    res = runner.invoke(
        app, ["eval", "create", "--tasks-dir", ".", "--agent-env", _MARKUP]
    )
    assert res.exit_code != 0
    assert isinstance(res.exception, SystemExit)


# ── dataset registry: malformed entries → DatasetResolutionError ──────────────


@pytest.mark.parametrize(
    "registry",
    [
        # task object missing git_url/git_commit_id
        [{"name": "evil", "version": "1.0", "tasks": [{"path": "tasks/foo"}]}],
        # tasks is a list of strings (string indices error)
        [{"name": "evil", "version": "1.0", "tasks": ["foo"]}],
        # top-level list of strings (e.get on a str)
        ["evil"],
    ],
)
def test_resolve_dataset_malformed_registry_raises_clean(tmp_path, registry) -> None:
    from benchflow._utils.dataset_registry import (
        DatasetResolutionError,
        resolve_dataset,
    )

    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps(registry))
    # Must be a clean DatasetResolutionError, never a raw KeyError / TypeError /
    # AttributeError leaking from per-entry structural access.
    with pytest.raises(DatasetResolutionError):
        resolve_dataset("evil@1.0", str(reg))
