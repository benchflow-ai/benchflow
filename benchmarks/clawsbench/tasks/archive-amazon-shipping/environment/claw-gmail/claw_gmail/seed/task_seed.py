"""Per-task seed scenario: composable seeder that inserts task-specific needles
then fills from the shared content library.

Usage:
    seed_task_scenario(db, fake, user, personas, "vendor-report-organize")

Each task must have a ``data/needles.py`` under ``tasks/<task>/`` with:
    NEEDLES        — list of standalone email dicts
    NEEDLE_THREADS — list of multi-message thread dicts
    FILL_CONFIG    — dict with target_count, distribution ratios, flags

Optionally:
    PARAMS         — dict of parameterizable slot definitions for the manifest
    parameterize(rng) — function returning filled NEEDLES with params resolved
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

from faker import Faker
from sqlalchemy.orm import Session

from claw_gmail.models import Draft, Message, MessageLabel, Thread, User
from claw_gmail.seed.content_library.ambiguous import AMBIGUOUS_EMAILS
from claw_gmail.seed.content_library.newsletters import NEWSLETTERS
from claw_gmail.seed.content_library.notifications import NOTIFICATIONS
from claw_gmail.seed.content_library.personal import PERSONAL_EMAILS
from claw_gmail.seed.content_library.sent import SENT_EMAILS
from claw_gmail.seed.content_library.spam import SPAM_EMAILS
from claw_gmail.seed.content_library.work import WORK_THREADS
from claw_gmail.seed.long_context import (
    BUDGET_THREAD as _LC_BUDGET_THREAD,
    NEEDLES as _LC_NEEDLES,
    _WORK_POOL,
    _fill_from_pool,
    _get_html_for_email,
    _insert_single,
    _insert_thread,
    _make_id,
    _parameterize,
)

_HARBOR_DIR = Path(os.environ["TASKS_DIR"]) if "TASKS_DIR" in os.environ else Path(__file__).resolve().parents[5] / "tasks"


def _load_needles_module(task_dir_name: str):
    """Dynamically load data/needles.py for a harbor task."""
    needles_path = _HARBOR_DIR / task_dir_name / "data" / "needles.py"
    if not needles_path.exists():
        raise FileNotFoundError(f"Task needles not found: {needles_path}")

    module_name = f"harbor_needles_{task_dir_name.replace('-', '_')}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, needles_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _insert_needle_emails(db, user, needles, now, rng):
    """Insert standalone needle emails and return manifest entries.

    Returns:
        list of dicts: [{message_id, thread_id, role, params, ...}, ...]
    """
    inserted = []
    for needle in needles:
        thread_id = _make_id()
        msg_id = _make_id()
        if "received_at" in needle:
            msg_time = datetime.fromisoformat(needle["received_at"])
        else:
            days = needle.get("days_ago", rng.randint(1, 30))
            msg_time = now - timedelta(days=days, hours=rng.randint(6, 18))

        sender = f"{needle['sender_name']} <{needle['sender_email']}>"
        body_html = needle.get("body_html", "")
        if not body_html:
            body_html = _get_html_for_email(needle, rng)

        db.add(Thread(id=thread_id, user_id=user.id, snippet=needle["body_plain"][:200]))
        msg = Message(
            id=msg_id,
            thread_id=thread_id,
            user_id=user.id,
            sender=sender,
            to=needle.get("to", user.email_address),
            subject=needle["subject"],
            snippet=needle["body_plain"][:200],
            body_plain=needle["body_plain"],
            body_html=body_html,
            internal_date=msg_time,
            is_read=needle.get("is_read", False),
            is_starred=needle.get("is_starred", False),
        )
        db.add(msg)
        for lid in needle.get("labels", ["INBOX"]):
            db.add(MessageLabel(message_id=msg_id, label_id=lid))

        inserted.append({
            "message_id": msg_id,
            "thread_id": thread_id,
            "role": needle.get("role", "needle"),
            "subject": needle["subject"],
            "sender_email": needle["sender_email"],
            "params": needle.get("params", {}),
        })
    return inserted


def _insert_needle_threads(db, user, needle_threads, now, rng):
    """Insert multi-message needle threads."""
    count = 0
    for thread_tmpl in needle_threads:
        count += _insert_thread(db, user, thread_tmpl, now, rng)
    return count


def _write_manifest(task_dir_name: str, seed_val: int, needles: list[dict], db_path: str | None = None):
    """Write needle manifest JSON alongside the database.

    The manifest records every inserted needle's message ID, thread ID,
    and filled parameter values so evaluators can check without hardcoding.
    """
    if db_path:
        manifest_dir = Path(db_path).parent
    else:
        # Fallback: .data/ in project root
        manifest_dir = Path(__file__).resolve().parents[5] / ".data"

    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"_needle_manifest_{task_dir_name}.json"

    manifest = {
        "task": task_dir_name,
        "seed": seed_val,
        "seeded_at": datetime.utcnow().isoformat() + "Z",
        "needles": needles,
    }

    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def get_task_data_summary(task_dir_name: str) -> dict:
    """Return a summary of a task's seed data config for the admin API."""
    needles_path = _HARBOR_DIR / task_dir_name / "data" / "needles.py"
    if not needles_path.exists():
        return {"has_per_task_data": False}

    mod = _load_needles_module(task_dir_name)
    needles = getattr(mod, "NEEDLES", [])
    needle_threads = getattr(mod, "NEEDLE_THREADS", [])
    fill_config = getattr(mod, "GMAIL_FILL_CONFIG", {})

    needle_summary = []
    for n in needles:
        needle_summary.append({
            "sender_name": n.get("sender_name", ""),
            "sender_email": n.get("sender_email", ""),
            "subject": n.get("subject", ""),
            "labels": n.get("labels", ["INBOX"]),
            "days_ago": n.get("days_ago"),
        })

    thread_summary = []
    for t in needle_threads:
        thread_summary.append({
            "subject": t.get("subject", ""),
            "labels": t.get("labels", ["INBOX"]),
            "message_count": len(t.get("messages", [])),
        })

    return {
        "has_per_task_data": True,
        "needles": needle_summary,
        "needle_threads": thread_summary,
        "fill_config": fill_config,
    }


def seed_task_scenario(
    db: Session,
    fake: Faker,
    user: User,
    personas: list[dict],
    task_dir_name: str,
    *,
    db_path: str | None = None,
    seed_val: int = 42,
):
    """Seed a per-task scenario: insert needles, then fill from shared library.

    This mirrors long_context.py's structure but uses task-specific needle
    definitions and lets each task control its own fill distribution.

    If the needles module defines a ``parameterize(rng)`` function, it is
    called to produce parameterized needles.  The filled parameter values
    are recorded in a manifest JSON file alongside the database.
    """
    mod = _load_needles_module(task_dir_name)
    fill_config = getattr(mod, "GMAIL_FILL_CONFIG", {})

    now = datetime.utcnow()
    rng = random

    # Resolve needles — use parameterize() if available, else static NEEDLES
    parameterize_fn = getattr(mod, "parameterize", None)
    if parameterize_fn is not None:
        needles = parameterize_fn(rng)
    else:
        needles = getattr(mod, "NEEDLES", [])
    needle_threads = getattr(mod, "NEEDLE_THREADS", [])

    # Phase 1: Fixed-position emails

    fixed_count = 0

    # Standalone needle emails
    manifest_entries = _insert_needle_emails(db, user, needles, now, rng)
    fixed_count += len(manifest_entries)

    # Multi-message needle threads
    fixed_count += _insert_needle_threads(db, user, needle_threads, now, rng)

    # Ambiguous edge cases (if configured)
    if fill_config.get("include_ambiguous", True):
        for template in AMBIGUOUS_EMAILS:
            _insert_single(db, user, template, now, rng)
        fixed_count += len(AMBIGUOUS_EMAILS)

    # Phase 2: Fill from content library

    target_count = fill_config.get("target_count", 3000)
    remaining = target_count - fixed_count

    dist = fill_config.get("distribution", {})
    n_notif = int(remaining * dist.get("notifications", 0.35))
    n_newsletter = int(remaining * dist.get("newsletters", 0.20))
    n_work = int(remaining * dist.get("work", 0.20))
    n_personal = int(remaining * dist.get("personal", 0.10))
    n_sent = int(remaining * dist.get("sent", 0.05))
    n_spam = int(remaining * dist.get("spam", 0.05))
    # Remainder to notifications
    n_notif += remaining - n_notif - n_newsletter - n_work - n_personal - n_sent - n_spam

    # Work: insert threads first, then fill singles
    work_inserted = 0
    for thread_tmpl in WORK_THREADS:
        if work_inserted >= n_work:
            break
        count = _insert_thread(db, user, thread_tmpl, now, rng)
        work_inserted += count

    if work_inserted < n_work:
        _fill_from_pool(db, user, _WORK_POOL, n_work - work_inserted, now, rng)

    # Notifications
    old_notif_ratio = fill_config.get("old_notification_ratio", 0.5)
    _fill_from_pool(
        db, user, NOTIFICATIONS, n_notif, now, rng,
        parameterize=True,
        old_notification_ratio=old_notif_ratio,
    )

    # Newsletters and promos
    _fill_from_pool(db, user, NEWSLETTERS, n_newsletter, now, rng)

    # Personal
    _fill_from_pool(db, user, PERSONAL_EMAILS, n_personal, now, rng)

    # Sent
    _fill_from_pool(db, user, SENT_EMAILS, n_sent, now, rng, force_sent=True)

    # Spam
    _fill_from_pool(db, user, SPAM_EMAILS, n_spam, now, rng, force_spam=True)

    # Draft (if configured)
    if fill_config.get("include_draft", True):
        draft_id = _make_id()
        draft_msg_id = _make_id()
        draft_thread_id = _make_id()
        db.add(Thread(id=draft_thread_id, user_id=user.id, snippet=""))
        db.add(Message(
            id=draft_msg_id,
            thread_id=draft_thread_id,
            user_id=user.id,
            sender=user.email_address,
            to="team@nexusai.com",
            subject="Q1 Infrastructure Optimization Summary",
            snippet="Draft summary of Q1 infrastructure changes...",
            body_plain="Draft summary of Q1 infrastructure changes and cost savings...",
            internal_date=now,
            is_draft=True,
        ))
        db.add(MessageLabel(message_id=draft_msg_id, label_id="DRAFT"))
        db.add(Draft(id=draft_id, user_id=user.id, message_id=draft_msg_id))

    # Write needle manifest
    if manifest_entries:
        return _write_manifest(task_dir_name, seed_val, manifest_entries, db_path)
    return None
