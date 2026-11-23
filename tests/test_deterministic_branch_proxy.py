"""Deterministic tier: an inline fork of an API-key (proxied) run.

Guards 0.10.0rc6's fix. The branch scenarios in test_deterministic_integration
take the native route, which starts no model proxy; branching refused a
rollout whose calls went through BenchFlow's LiteLLM proxy. These run the
same fork on the ``proxy`` route (``ANTHROPIC_API_KEY`` and
``BENCHFLOW_PROVIDER_BASE_URL``, agent -> proxy -> fake): ``bench eval
branch --prompt <draft> --prompt @instruction --checkpoint-after-prompt 1``
with a passing and a wrong child, in place and isolated. The parent's
result.json must price its own calls from the proxy, before the fork and
after it, and none of the children's; an isolated child's result.json its
own calls only (``_check_trial``: every proxied call matches one agent
message of that trial's ACP trajectory, and the cost follows).
"""

from __future__ import annotations

import pytest

from tests.integration.deterministic import harness as h
from tests.test_deterministic_integration import (
    DRAFT_PROMPT,
    PASS_CHILD,
    WRONG_CHILD,
    _check_trial,
    _run,
    det_root,  # noqa: F401 (the session fixtures these runs share)
    fake_url,  # noqa: F401
    needs_sandbox,
    require_sandbox,  # noqa: F401
)

pytestmark = pytest.mark.deterministic


def _proxied_branch(root, url, name: str, extra: list[str]) -> h.CliRun:
    return _run(
        root,
        name,
        ["eval", "branch"],
        [h.TaskVariant(name, "hello-pass")],
        url,
        route="proxy",
        extra=[
            "--prompt",
            DRAFT_PROMPT,
            "--prompt",
            "@instruction",
            "--checkpoint-after-prompt",
            "1",
            "--child",
            PASS_CHILD,
            "--child",
            WRONG_CHILD,
            *extra,
        ],
    )


@pytest.fixture(scope="session")
def proxied_branch_in_place(det_root, fake_url) -> h.CliRun:  # noqa: F811
    return _proxied_branch(
        det_root, fake_url, "proxied-branch-in-place", ["--in-place-children"]
    )


@pytest.fixture(scope="session")
def proxied_branch_parallel(det_root, fake_url) -> h.CliRun:  # noqa: F811
    return _proxied_branch(
        det_root, fake_url, "proxied-branch-parallel", ["--concurrency", "2"]
    )


def _check_proxied_branch(run: h.CliRun, *, isolated: bool) -> None:
    assert run.returncode == 0, run.output[-4000:]
    (trial,) = run.trial_dirs()
    (fork,) = h.read_json(trial / "tree.json")["forks"]
    children = fork["children"]
    assert [c["reward"] for c in children] == [1.0, 0.0]
    assert [c["status"] for c in children] == ["scored", "scored"]
    assert fork["status"] == "completed"
    assert fork["parent_restore"] == "restored"
    assert fork["children_mode"]["isolated"] is isolated
    result = h.read_json(trial / "result.json")
    assert result["rewards"] == {"reward": 1.0}
    assert result["n_prompts"] == 2
    _check_trial(trial, None, route="proxy")
    if isolated:
        child_root = trial / "branches" / fork["id"] / "children"
        for child in children:
            _check_trial(child_root / child["node_id"], None, route="proxy")


@needs_sandbox
def test_proxied_branch_in_place(proxied_branch_in_place):
    _check_proxied_branch(proxied_branch_in_place, isolated=False)


@needs_sandbox
def test_proxied_branch_parallel(proxied_branch_parallel):
    _check_proxied_branch(proxied_branch_parallel, isolated=True)
