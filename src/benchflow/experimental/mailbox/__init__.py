"""EXPERIMENTAL SURFACE — may change or be removed in any minor version.

Mailbox subpackage: a 2-role outbox-file scheduler for multi-agent runs.
The graduated ``benchflow.trial.Trial._run_scene`` path is single-role only;
multi-role execution lives here, opt-in.

Constellation
-------------
Library code (this subpackage)::

    _runner.py       MailboxRunner — exactly-2-role outbox-driven scheduler.
    _transport.py    MessageTransport ABC, in-memory MailboxTransport, Message.
    _role_runner.py  default_role_runner(env, role, prompt) helper —
                     install + connect ACP + execute + close. Use as the
                     ``role_runner`` argument to MailboxRunner.run when you
                     don't need custom per-role lifecycle.

Tests + demos elsewhere in the repo::

    tests/test_mailbox_runner.py
        Unit tests (12) — pytest-collected.
    tests/conformance/proof_multi_agent.py
        Live coder→reviewer proof — hand-run, not pytest-collected.
    benchmarks/followup-bench/runner.py
        Empirical multi-agent benchmark (coder + reviewer in shared sandbox).
        Standalone script: ``python benchmarks/followup-bench/runner.py --task-dir <path>``.
    sandbox/v2/spec/scene.quint
        UNION model — TurnLoop (graduated) + Mailbox (this subpackage) modes.
    sandbox/v2/tests/test_scene_against_main.py
        Bridge test pinning both modes against their implementations.

Design notes
------------
* ``MailboxRunner`` does NOT consume ``benchflow.contracts.trial_config.Scene``.
  It owns its own ``Role`` shape (with ``instruction`` and ``tools`` fields).
  The ``Role`` re-exported here is runner-internal; do not confuse with
  ``contracts.Role``.
* No graduated module may import this subpackage. Enforced by the
  ``experimental/ stays opt-in`` import-linter contract.
* Output is a ``list[Message]`` trajectory, not a ``RunResult``. There is no
  verifier integration today — followup-bench shows the runtime template, but
  produces no scored result. A ``trajectory_to_run_result`` adapter is a
  candidate follow-up if a real product-plane caller materializes.

Graduation criteria (per sandbox/PLAN_V2_impl.md §12.2, §12.5)
--------------------------------------------------------------
ALL must hold before promotion out of experimental/:

1. ≥1 month soak without churn that breaks the Quint spec.
2. ≥3 external callers OR an explicit "we're keeping this" decision.
3. Quint spec + bridge test — already in ``sandbox/v2/{spec,tests}/``.
4. Exposed via ``benchflow``/``bench`` CLI with a documented flag.
5. Passes import-linter in its promoted location.

Removal criteria (ANY one triggers)
-----------------------------------
* No caller growth after 1 month.
* Design fork unresolved after 2 spec attempts.
* Quint spec surfaces an invariant nobody wants to commit to.
* CLI exposure never earned past the soak window.
"""

from benchflow.experimental.mailbox._role_runner import default_role_runner
from benchflow.experimental.mailbox._runner import MailboxRole, MailboxRunner
from benchflow.experimental.mailbox._transport import (
    MailboxTransport,
    Message,
    MessageTransport,
)

# v0.4 deprecation alias — the mailbox-internal Role collided with
# benchflow.multi_agent.Role (Scene's Role). Removed in v0.5; new code should
# import MailboxRole directly.
Role = MailboxRole

__all__ = [
    "MailboxRole",
    "MailboxRunner",
    "MailboxTransport",
    "Message",
    "MessageTransport",
    "Role",  # deprecated alias for MailboxRole
    "default_role_runner",
]
