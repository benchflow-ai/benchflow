"""Long-context seed scenario: ~3000 emails for stress-testing agents.

Draws from the curated content library, cycling templates with parameterized
variation (dates, read/unread state) to fill a realistic startup founder inbox.

Persona: Alex Chen, founder/CEO of NexusAI (alex@nexusai.com)

Distribution:
    - Notifications (tool services):  ~35%
    - Newsletters / promos:           ~20%
    - Work (@nexusai.com):            ~20%
    - Personal (@gmail.com):          ~10%
    - Sent (from user):               ~5%
    - Spam:                           ~5%
    - Ambiguous edge cases:           20 (fixed)
    - Needle emails:                  ~17 (fixed)
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta

from faker import Faker
from sqlalchemy.orm import Session

from claw_gmail.models import (
    User,
    Thread,
    Message,
    Label,
    LabelType,
    MessageLabel,
    Draft,
)
from claw_gmail.seed.content_library.notifications import NOTIFICATIONS
from claw_gmail.seed.content_library.newsletters import NEWSLETTERS
from claw_gmail.seed.content_library.work import WORK_THREADS, WORK_SINGLES
from claw_gmail.seed.content_library.personal import PERSONAL_EMAILS
from claw_gmail.seed.content_library.spam import SPAM_EMAILS
from claw_gmail.seed.content_library.sent import SENT_EMAILS
from claw_gmail.seed.content_library.ambiguous import AMBIGUOUS_EMAILS
from claw_gmail.seed.templates import load_html_template


def _make_id() -> str:
    return uuid.uuid4().hex[:16]


# ---------------------------------------------------------------------------
# Needle emails — specific emails that tasks reference for evaluation
# ---------------------------------------------------------------------------

NEEDLES = {
    "flight_confirmation": {
        "sender_name": "United Airlines",
        "sender_email": "no-reply@united.com",
        "subject": "Flight Confirmation - SFO to JFK, Mar 15",
        "body_plain": (
            "Booking Confirmation\n\n"
            "Confirmation #: UA7829K\n"
            "Flight: UA 246\n"
            "Date: March 15, 2026\n"
            "Departure: San Francisco (SFO) at 8:15 AM PST\n"
            "Arrival: New York (JFK) at 4:45 PM EST\n"
            "Passenger: Alex Chen\n"
            "Seat: 14A (Economy Plus)\n\n"
            "Check-in opens 24 hours before departure.\n"
            "Manage your reservation at united.com/manage\n"
        ),
        "labels": ["INBOX"],
        "is_read": False,
        "days_ago": 5,
    },
    "vendor_report_1": {
        "sender_name": "DataFlow Reports",
        "sender_email": "vendor-reports@dataflow.io",
        "subject": "Weekly Analytics Report - Jan 6",
        "body_plain": "Weekly analytics summary for the period Jan 1-6. Total API calls: 1.2M. Error rate: 0.3%.",
        "labels": ["INBOX"],
        "is_read": True,
        "days_ago": 55,
    },
    "vendor_report_2": {
        "sender_name": "DataFlow Reports",
        "sender_email": "vendor-reports@dataflow.io",
        "subject": "Weekly Analytics Report - Jan 13",
        "body_plain": "Weekly analytics summary for the period Jan 7-13. Total API calls: 1.4M. Error rate: 0.2%.",
        "labels": ["INBOX"],
        "is_read": True,
        "days_ago": 48,
    },
    "vendor_report_3": {
        "sender_name": "DataFlow Reports",
        "sender_email": "vendor-reports@dataflow.io",
        "subject": "Weekly Analytics Report - Jan 20",
        "body_plain": "Weekly analytics summary for the period Jan 14-20. Total API calls: 1.3M. Error rate: 0.4%.",
        "labels": ["INBOX"],
        "is_read": True,
        "days_ago": 41,
    },
    "vendor_report_4": {
        "sender_name": "DataFlow Reports",
        "sender_email": "vendor-reports@dataflow.io",
        "subject": "Weekly Analytics Report - Jan 27",
        "body_plain": "Weekly analytics summary for the period Jan 21-27. Total API calls: 1.5M. Error rate: 0.1%.",
        "labels": ["INBOX"],
        "is_read": True,
        "days_ago": 34,
    },
    "vendor_report_5": {
        "sender_name": "DataFlow Reports",
        "sender_email": "vendor-reports@dataflow.io",
        "subject": "Weekly Analytics Report - Feb 3",
        "body_plain": "Weekly analytics summary for the period Jan 28-Feb 3. Total API calls: 1.6M. Error rate: 0.2%.",
        "labels": ["INBOX"],
        "is_read": False,
        "days_ago": 27,
    },
    "vendor_report_6": {
        "sender_name": "DataFlow Reports",
        "sender_email": "vendor-reports@dataflow.io",
        "subject": "Weekly Analytics Report - Feb 10",
        "body_plain": "Weekly analytics summary for the period Feb 4-10. Total API calls: 1.8M. Error rate: 0.15%.",
        "labels": ["INBOX"],
        "is_read": False,
        "days_ago": 20,
    },
    "vendor_report_7": {
        "sender_name": "DataFlow Reports",
        "sender_email": "vendor-reports@dataflow.io",
        "subject": "Weekly Analytics Report - Feb 17",
        "body_plain": "Weekly analytics summary for the period Feb 11-17. Total API calls: 2.0M. Error rate: 0.1%. New milestone!",
        "labels": ["INBOX"],
        "is_read": False,
        "days_ago": 13,
    },
    "contract_email": {
        "sender_name": "Sarah Kim",
        "sender_email": "sarah@nexusai.com",
        "subject": "ACTION REQUIRED: Vendor Agreement - TechServ Inc.",
        "body_plain": (
            "Hi Alex,\n\n"
            "Please review and sign the attached vendor agreement for TechServ Inc.\n"
            "Key terms:\n"
            "- Contract value: $240,000/year\n"
            "- Duration: 24 months\n"
            "- SLA: 99.9% uptime\n"
            "- Auto-renewal clause: 60-day notice required\n\n"
            "Deadline: March 10, 2026\n\n"
            "Best,\nSarah Kim\nCTO, NexusAI"
        ),
        "labels": ["INBOX", "IMPORTANT"],
        "is_read": False,
        "is_starred": True,
        "days_ago": 3,
    },
}

# Budget thread needle — 8-message thread with critical detail in message #5
BUDGET_THREAD = {
    "subject": "RE: Infrastructure Cost Review - Q1 Budget",
    "labels": ["INBOX"],
    "messages": [
        {
            "sender_name": "Sarah Kim",
            "sender_email": "sarah@nexusai.com",
            "body_plain": "Team, let's review our Q1 infrastructure costs. I've noticed some unexpected spikes. Can we schedule a review?",
            "days_ago": 25,
            "hours_offset": 0,
            "is_read": True,
        },
        {
            "sender_name": "Marcus Rivera",
            "sender_email": "marcus@nexusai.com",
            "body_plain": "Good idea Sarah. I've been watching the AWS bills climb. Let me pull together the numbers.",
            "days_ago": 25,
            "hours_offset": 2,
            "is_read": True,
        },
        {
            "sender_name": "Alex Chen",
            "sender_email": "alex@nexusai.com",
            "body_plain": "I can join. I suspect the new Kubernetes cluster is the main cost driver. Let me check the usage metrics.",
            "days_ago": 24,
            "hours_offset": 0,
            "is_read": True,
            "is_sent": True,
        },
        {
            "sender_name": "Marcus Rivera",
            "sender_email": "marcus@nexusai.com",
            "body_plain": "Here are the raw numbers for Jan:\n- AWS EC2: $45,200\n- RDS: $12,800\n- S3: $3,400\n- Other: $8,600\nTotal: $70,000\n\nThat's 40% over budget.",
            "days_ago": 23,
            "hours_offset": 0,
            "is_read": True,
        },
        {
            "sender_name": "Sarah Kim",
            "sender_email": "sarah@nexusai.com",
            "body_plain": (
                "After the review meeting, here's the action plan:\n\n"
                "APPROVED BUDGET ADJUSTMENT: Q1 infrastructure budget increased to $185,000 "
                "(from original $150,000). Finance approved on Feb 5.\n\n"
                "Key actions:\n"
                "1. Right-size the K8s nodes (Marcus - by Feb 15)\n"
                "2. Enable S3 lifecycle policies (Alex - by Feb 10)\n"
                "3. Reserved instances for stable workloads (Sarah - by Feb 20)\n\n"
                "Expected monthly savings after optimization: $12,000-15,000.\n"
                "Next review: March 1."
            ),
            "days_ago": 20,
            "hours_offset": 0,
            "is_read": True,
        },
        {
            "sender_name": "Alex Chen",
            "sender_email": "alex@nexusai.com",
            "body_plain": "S3 lifecycle policies are done. Should save about $800/month on storage alone.",
            "days_ago": 18,
            "hours_offset": 0,
            "is_read": True,
            "is_sent": True,
        },
        {
            "sender_name": "Marcus Rivera",
            "sender_email": "marcus@nexusai.com",
            "body_plain": "K8s node right-sizing complete. Moved from m5.4xlarge to m5.2xlarge for non-prod. February bill should reflect the change.",
            "days_ago": 15,
            "hours_offset": 0,
            "is_read": True,
        },
        {
            "sender_name": "Sarah Kim",
            "sender_email": "sarah@nexusai.com",
            "body_plain": "Great progress team. Reserved instances purchased for the database and core API servers. We should see the full impact in the March bill.",
            "days_ago": 10,
            "hours_offset": 0,
            "is_read": False,
        },
    ],
}


# ---------------------------------------------------------------------------
# Parameterization pools for notification template placeholders
# ---------------------------------------------------------------------------

_PERSON_NAMES = [
    "Sarah Kim", "Marcus Rivera", "Priya Sharma", "James Liu",
    "Emily Ortiz", "David Park", "Lisa Wang", "Derek Nguyen",
    "Amara Osei", "Jun Sato", "Ravi Krishnan", "Tina Wu",
    "Michael Chen", "Rachel Green", "Olivia Davis", "Noah Wilson",
    "Emma Thompson", "Liam Johnson", "Sophia Martinez", "Benjamin Lee",
    "Mia Anderson", "Daniel Kim", "Isabella Patel", "Alexander Wright",
    "Charlotte Brown", "William Garcia", "Ava Rodriguez", "Henry Nguyen",
]

_COMPANIES = [
    "Stripe", "Notion", "Figma", "Vercel", "Supabase",
    "Linear", "Retool", "Snowflake", "Databricks", "Anthropic",
    "OpenAI", "Hugging Face", "Weights & Biases", "Pinecone", "Weaviate",
    "Neon", "PlanetScale", "Railway", "Render", "Fly.io",
    "Temporal", "Prefect", "dbt Labs", "Airbyte", "Fivetran",
]

_EVENT_NAMES = [
    "NexusAI Demo Day", "AI Founders Dinner", "YC AI Demo Night",
    "SF ML Meetup", "Agent Infrastructure Workshop",
    "Startup Pitch Night", "AI Safety Forum", "DevTools Meetup",
    "Founder Coffee Chat", "Series A Masterclass",
    "GPU Compute Summit", "RAG in Production", "LLM Ops Workshop",
    "Bay Area Tech Mixer", "AI Product Launch Party",
]


# ---------------------------------------------------------------------------
# HTML template mapping: sender email domain → (template_name, static_params)
# When inserting an email whose sender domain matches, the HTML template is
# loaded and filled with both these static params and dynamic ones from the
# email's subject/body.
# ---------------------------------------------------------------------------

_HTML_TEMPLATE_MAP: dict[str, tuple[str, dict[str, str]]] = {
    # Notifications
    "github.com": ("github-pr", {
        "repo_name": "nexusai/core",
        "pr_number": "142",
        "branch_count": "3",
        "branch_name": "feat/retry-logic",
        "body_preview": "This PR adds exponential backoff for transient failures in the inference pipeline.",
    }),
    "luma.com": ("luma-event", {
        "event_date": "Thursday, March 12 at 6:00 PM PST",
        "event_location": "Runway, 1355 Market St, San Francisco",
        "host_name": "Alex Chen",
        "event_description": "Join us for an evening of demos, networking, and discussion about the future of AI infrastructure.",
    }),
    "slack.com": ("slack-digest", {
        "workspace_name": "NexusAI",
        "channel_name": "engineering",
        "message_preview": "Marcus: the deploy pipeline fix is in staging now, can someone verify?",
    }),
    "pagerduty.com": ("pagerduty-incident", {
        "severity": "HIGH",
        "incident_title": "API latency > 2s on inference-prod",
        "service_name": "inference-api-prod",
        "escalation_policy": "Engineering On-Call",
        "triggered_at": "March 2, 2026 at 3:47 AM PST",
    }),
    "linear.app": ("linear-login", {
        "login_code": "847293",
    }),
    "dtdg.co": ("datadog-alert", {
        "alert_status": "WARN",
        "monitor_name": "High p99 latency on inference-api",
        "alert_message": "The p99 response time for inference-api-prod has exceeded the 2000ms threshold.",
        "host": "inference-api-prod-7b4c",
        "metric": "trace.express.request.duration.p99",
        "value": "2847ms",
        "threshold": "2000ms",
    }),
    "google.com": ("google-security-alert", {
        "alert_message": "We noticed a new sign-in to your Google Account.",
        "device_name": "MacBook Pro (Chrome)",
        "location": "San Francisco, CA, United States",
        "time": "March 2, 2026 at 10:23 AM PST",
        "user_email": "alex@nexusai.com",
    }),
    # Newsletters/promos
    "e.stripe.com": ("stripe-receipt", {
        "merchant_name": "Stripe Press",
        "receipt_id": "pi_3PkR2Y4e7xG9",
        "item_description": "Stripe Sessions 2026 — Early Bird",
        "amount": "$299.00",
        "date": "February 28, 2026",
        "card_last4": "4242",
    }),
    "stripe.com": ("stripe-receipt", {
        "merchant_name": "Stripe Billing",
        "receipt_id": "in_1Q8x3K2eZv",
        "item_description": "Stripe API Usage — February 2026",
        "amount": "$847.20",
        "date": "March 1, 2026",
        "card_last4": "4242",
    }),
    # Mercury financial digest
    "mercury.com": ("mercury-digest", {
        "company_name": "NexusAI Inc.",
        "date_range": "Feb 24 – Mar 2, 2026",
        "balance": "$284,720",
        "income": "$48,200",
        "expenses": "$31,450",
        "transaction_1": "AWS — $12,400.00",
        "transaction_2": "Payroll — $15,200.00",
        "transaction_3": "Anthropic API — $3,850.00",
    }),
    # Spam
    "invoice-notify.net": ("spam-urgent", {
        "headline": "PAYMENT CONFIRMATION",
        "body_text": "Your payment of $499.99 has been processed. If you did not authorize this transaction, click below immediately.",
        "cta_text": "DISPUTE CHARGE",
    }),
    "cryptoalpha-signals.io": ("spam-urgent", {
        "headline": "EXCLUSIVE ALERT",
        "body_text": "Our AI algorithm identified 3 altcoins set to 50x. Limited VIP spots available.",
        "cta_text": "JOIN VIP FREE",
    }),
    "aws-notifications.cloud": ("spam-urgent", {
        "headline": "BILLING ALERT",
        "body_text": "Unusual charges of $2,847.33 detected on your AWS account. Verify immediately.",
        "cta_text": "VERIFY ACCOUNT",
    }),
    "google-account-verify.net": ("spam-urgent", {
        "headline": "SECURITY ALERT",
        "body_text": "Someone used your password to sign in from Moscow, Russia. Change your password immediately.",
        "cta_text": "SECURE ACCOUNT NOW",
    }),
    "norton-subscription-alert.com": ("spam-urgent", {
        "headline": "AUTO-RENEWAL NOTICE",
        "body_text": "Your Norton 360 subscription ($399.99) has been renewed. Call 1-888-555-0147 to cancel.",
        "cta_text": "REQUEST REFUND",
    }),
}

# Domains to match by suffix (for multi-subdomain services)
_HTML_TEMPLATE_SUFFIX_MAP: dict[str, tuple[str, dict[str, str]]] = {
    "beehiiv.net": ("newsletter-tech", {
        "newsletter_name": "Tech Newsletter",
        "brand_color": "#6366f1",
        "headline": "This Week in Tech",
        "intro_paragraph": "Here's what happened in the tech world this week.",
        "body_content": "Read on for the latest updates on AI, cloud infrastructure, and developer tools.",
    }),
}


def _generic_html_wrap(template: dict) -> str:
    """Wrap body_plain in a clean HTML email layout for emails without a dedicated template."""
    body_plain = template.get("body_plain", "")
    if not body_plain:
        return ""
    sender_name = template.get("sender_name", "")
    subject = template.get("subject", "")
    sender_email = template.get("sender_email", "")
    # Convert plain text paragraphs to HTML
    paragraphs = body_plain.strip().split("\n\n")
    body_html_parts = []
    for p in paragraphs:
        lines = p.strip().split("\n")
        body_html_parts.append("<br/>".join(_html_escape(line) for line in lines))
    body_content = "".join(
        f'<p style="font-size:14px;line-height:22px;margin:0 0 12px;color:#333">{part}</p>'
        for part in body_html_parts
        if part.strip()
    )
    return (
        '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" '
        '"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">\n'
        '<html dir="ltr" lang="en"><head>'
        '<meta content="text/html; charset=UTF-8" http-equiv="Content-Type" />'
        "</head><body "
        'style=\'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;'
        "background-color:#f6f6f6'>"
        '<table align="center" width="100%" border="0" cellpadding="0" cellspacing="0" '
        'role="presentation" style="max-width:600px;margin:0 auto;background-color:#ffffff">'
        "<tbody><tr><td>"
        # Header
        '<table align="center" width="100%" border="0" cellpadding="0" cellspacing="0" '
        'role="presentation" style="background-color:#f8f9fa;padding:20px 32px;'
        'border-bottom:1px solid #e8e8e8"><tbody><tr><td>'
        f'<p style="font-size:16px;line-height:20px;margin:0;color:#1a1a1a;font-weight:600">'
        f"{_html_escape(sender_name)}</p>"
        f'<p style="font-size:12px;line-height:16px;margin:4px 0 0;color:#888">'
        f"{_html_escape(sender_email)}</p>"
        "</td></tr></tbody></table>"
        # Body
        '<table align="center" width="100%" border="0" cellpadding="0" cellspacing="0" '
        'role="presentation" style="padding:28px 32px"><tbody><tr><td>'
        f'<h2 style="font-size:20px;color:#1a1a1a;margin:0 0 16px;font-weight:600">'
        f"{_html_escape(subject)}</h2>"
        f"{body_content}"
        "</td></tr></tbody></table>"
        # Footer
        '<table align="center" width="100%" border="0" cellpadding="0" cellspacing="0" '
        'role="presentation" style="padding:16px 32px;border-top:1px solid #e8e8e8">'
        "<tbody><tr><td>"
        f'<p style="font-size:12px;line-height:18px;margin:0;color:#999">'
        f"Sent by {_html_escape(sender_name)}</p>"
        "</td></tr></tbody></table>"
        "</td></tr></tbody></table></body></html>"
    )


def _html_escape(text: str) -> str:
    """Minimal HTML escape for user-provided text."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _get_html_for_email(template: dict, rng: random.Random) -> str:
    """Try to generate HTML body for an email template based on sender domain.

    Returns rendered HTML or empty string if no matching template.
    """
    sender_email = template.get("sender_email", "")
    if "@" not in sender_email:
        return ""

    domain = sender_email.split("@", 1)[1].lower()

    # Direct domain match
    entry = _HTML_TEMPLATE_MAP.get(domain)
    if not entry:
        # Suffix match
        for suffix, e in _HTML_TEMPLATE_SUFFIX_MAP.items():
            if domain.endswith(suffix):
                entry = e
                break

    if not entry:
        # Generic fallback: wrap body_plain in a clean HTML email layout
        return _generic_html_wrap(template)

    tmpl_name, static_params = entry

    # Build params: start with static, add dynamic from email fields
    params = dict(static_params)
    params.setdefault("subject", template.get("subject", ""))
    params.setdefault("sender_name", template.get("sender_name", ""))
    # For notification count variation
    params.setdefault("unread_count", str(rng.randint(3, 28)))
    # For event name
    params.setdefault("event_name", template.get("subject", ""))

    return load_html_template(tmpl_name, params)


def _parameterize(template: dict, rng: random.Random) -> dict:
    """Fill {placeholder} tokens in subject and body_plain with random values."""
    params = {
        "attendee_name": rng.choice(_PERSON_NAMES),
        "event_name": rng.choice(_EVENT_NAMES),
        "company": rng.choice(_COMPANIES),
        "total_guests": str(rng.randint(15, 80)),
    }
    result = dict(template)
    for field in ("subject", "body_plain"):
        val = result.get(field, "")
        if "{" not in val:
            continue
        # Replace {{key}} (double-braced) then {key} (single-braced)
        for k, v in params.items():
            val = val.replace("{{" + k + "}}", v)
            val = val.replace("{" + k + "}", v)
        result[field] = val
    return result


# ---------------------------------------------------------------------------
# Flattened work pool: thread messages extracted as standalone templates
# ---------------------------------------------------------------------------

def _build_work_pool() -> list[dict]:
    """Build a combined pool of standalone work email templates.

    Includes WORK_SINGLES directly plus individual messages extracted
    from WORK_THREADS (using the thread subject as the email subject).
    """
    pool = list(WORK_SINGLES)
    for t in WORK_THREADS:
        for m in t["messages"]:
            pool.append({
                "sender_name": m["sender_name"],
                "sender_email": m["sender_email"],
                "subject": t["subject"],
                "body_plain": m["body_plain"],
                "body_html": m.get("body_html", ""),
                "is_sent": m.get("is_sent", False),
                "category": "",  # no category label — will get INBOX
                "is_read_probability": 0.7,
                "age_range": t.get("age_range", (0, 30)),
            })
    return pool


_WORK_POOL = _build_work_pool()  # 15 singles + 54 thread msgs = 69


# ---------------------------------------------------------------------------
# Insertion helpers
# ---------------------------------------------------------------------------

def _insert_single(
    db: Session,
    user: User,
    template: dict,
    now: datetime,
    rng: random.Random,
    *,
    force_days_ago: int | None = None,
    force_spam: bool = False,
    force_sent: bool = False,
) -> str:
    """Insert one email from a template dict. Returns the message ID."""
    thread_id = _make_id()
    msg_id = _make_id()

    # Age
    if force_days_ago is not None:
        days_ago = force_days_ago
    else:
        lo, hi = template.get("age_range", (0, 30))
        days_ago = rng.randint(lo, hi)
    hours = rng.randint(0, 23)
    msg_time = now - timedelta(days=days_ago, hours=hours)

    # Read state
    is_read = rng.random() < template.get("is_read_probability", 0.5)

    # Determine if this is a sent email
    category = template.get("category", "")
    is_sent = force_sent or template.get("is_sent", False) or category == "SENT"
    is_spam = force_spam or category == "SPAM"

    # Sender / recipient
    if is_sent:
        sender = f"{user.display_name} <{user.email_address}>"
        to_email = template.get("to_email", template.get("sender_email", ""))
        to_name = template.get("to_name", template.get("sender_name", ""))
        to_str = f"{to_name} <{to_email}>" if to_name and to_email else to_email
        is_read = True
    else:
        sender_name = template.get("sender_name", "Unknown")
        sender_email = template.get("sender_email", "unknown@example.com")
        sender = f"{sender_name} <{sender_email}>"
        to_str = user.email_address

    subject = template.get("subject", "(no subject)")
    body_plain = template.get("body_plain", "")
    body_html = template.get("body_html", "")

    # Auto-fill HTML from pre-rendered templates if not already set
    if not body_html:
        body_html = _get_html_for_email(template, rng)

    # Labels
    if is_sent:
        labels = ["SENT"]
    elif is_spam:
        labels = ["SPAM"]
    elif category.startswith("CATEGORY_"):
        labels = ["INBOX", category]
    else:
        labels = ["INBOX"]

    db.add(Thread(id=thread_id, user_id=user.id, snippet=body_plain[:200]))
    msg = Message(
        id=msg_id,
        thread_id=thread_id,
        user_id=user.id,
        sender=sender,
        to=to_str,
        subject=subject,
        snippet=body_plain[:200],
        body_plain=body_plain,
        body_html=body_html,
        internal_date=msg_time,
        is_read=is_read,
        is_sent=is_sent,
        is_spam=is_spam,
        is_starred=template.get("is_starred", False),
    )
    db.add(msg)
    for lid in labels:
        db.add(MessageLabel(message_id=msg_id, label_id=lid))

    return msg_id


def _insert_thread(
    db: Session,
    user: User,
    thread_template: dict,
    now: datetime,
    rng: random.Random,
) -> int:
    """Insert a multi-message thread. Returns the number of messages inserted."""
    thread_id = _make_id()
    messages = thread_template["messages"]

    lo, hi = thread_template.get("age_range", (0, 30))
    base_days_ago = rng.randint(lo, hi)
    base_time = now - timedelta(days=base_days_ago)

    last_body = messages[-1]["body_plain"] if messages else ""
    db.add(Thread(id=thread_id, user_id=user.id, snippet=last_body[:200]))

    labels = thread_template.get("labels", ["INBOX"])

    for i, m in enumerate(messages):
        msg_id = _make_id()
        minutes_offset = m.get("minutes_offset", i * rng.randint(30, 480))
        msg_time = base_time + timedelta(minutes=minutes_offset)

        is_sent = m.get("is_sent", False)
        if is_sent or m["sender_email"] == user.email_address:
            sender = f"{user.display_name} <{user.email_address}>"
            is_sent = True
        else:
            sender = f"{m['sender_name']} <{m['sender_email']}>"

        is_read_prob = m.get("is_read_probability", 0.8 if i < len(messages) - 1 else 0.5)
        is_read = rng.random() < is_read_prob

        msg = Message(
            id=msg_id,
            thread_id=thread_id,
            user_id=user.id,
            sender=sender,
            to=user.email_address if not is_sent else m.get("to_email", m["sender_email"]),
            subject=thread_template["subject"],
            snippet=m["body_plain"][:200],
            body_plain=m["body_plain"],
            body_html=m.get("body_html", ""),
            internal_date=msg_time,
            is_read=is_read,
            is_sent=is_sent,
        )
        db.add(msg)

        if is_sent:
            db.add(MessageLabel(message_id=msg_id, label_id="SENT"))
        else:
            for lid in labels:
                db.add(MessageLabel(message_id=msg_id, label_id=lid))

    return len(messages)


def _fill_from_pool(
    db: Session,
    user: User,
    pool: list[dict],
    count: int,
    now: datetime,
    rng: random.Random,
    *,
    parameterize: bool = False,
    force_spam: bool = False,
    force_sent: bool = False,
    old_notification_ratio: float = 0.0,
) -> int:
    """Cycle through a template pool to insert `count` emails.

    Args:
        old_notification_ratio: Fraction of emails forced to >14 days old.
            Used for notifications to ensure enough "old" ones exist for the
            cleanup task to score positively.
    """
    inserted = 0
    idx = 0
    while inserted < count:
        template = pool[idx % len(pool)]
        if parameterize:
            template = _parameterize(template, rng)

        # Force some notifications to be old (>14 days)
        force_days = None
        if old_notification_ratio > 0 and rng.random() < old_notification_ratio:
            force_days = rng.randint(15, 60)

        _insert_single(
            db, user, template, now, rng,
            force_days_ago=force_days,
            force_spam=force_spam,
            force_sent=force_sent,
        )
        inserted += 1
        idx += 1
    return inserted


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def seed_long_context_scenario(
    db: Session,
    fake: Faker,
    user: User,
    personas: list[dict],
    target_count: int = 3000,
):
    """Seed a long-context scenario with ~3000 realistic emails.

    Inserts needle emails first, then fills remaining slots by cycling
    through the curated content library with date/read-state variation.
    """
    user_email = user.email_address
    now = datetime.utcnow()

    # --- Phase 1: Insert fixed-position emails ---

    # Standalone needle emails
    for needle_key, needle in NEEDLES.items():
        thread_id = _make_id()
        msg_id = _make_id()
        days = needle.get("days_ago", random.randint(1, 30))
        msg_time = now - timedelta(days=days, hours=random.randint(6, 18))

        sender = f"{needle['sender_name']} <{needle['sender_email']}>"
        db.add(Thread(id=thread_id, user_id=user.id, snippet=needle["body_plain"][:200]))
        msg = Message(
            id=msg_id,
            thread_id=thread_id,
            user_id=user.id,
            sender=sender,
            to=user_email,
            subject=needle["subject"],
            snippet=needle["body_plain"][:200],
            body_plain=needle["body_plain"],
            internal_date=msg_time,
            is_read=needle.get("is_read", False),
            is_starred=needle.get("is_starred", False),
        )
        db.add(msg)
        for lid in needle.get("labels", ["INBOX"]):
            db.add(MessageLabel(message_id=msg_id, label_id=lid))

    # Budget thread needle (multi-message)
    budget_thread_id = _make_id()
    budget_msgs = BUDGET_THREAD["messages"]
    db.add(Thread(
        id=budget_thread_id,
        user_id=user.id,
        snippet=budget_msgs[-1]["body_plain"][:200],
    ))
    for i, bm in enumerate(budget_msgs):
        msg_id = _make_id()
        msg_time = now - timedelta(days=bm["days_ago"], hours=bm.get("hours_offset", 0))
        is_sent = bm.get("is_sent", False)

        if is_sent:
            sender = f"{user.display_name} <{user_email}>"
        else:
            sender = f"{bm['sender_name']} <{bm['sender_email']}>"

        msg = Message(
            id=msg_id,
            thread_id=budget_thread_id,
            user_id=user.id,
            sender=sender,
            to=user_email if not is_sent else "sarah@nexusai.com",
            subject=BUDGET_THREAD["subject"],
            snippet=bm["body_plain"][:200],
            body_plain=bm["body_plain"],
            internal_date=msg_time,
            is_read=bm.get("is_read", False),
            is_sent=is_sent,
        )
        db.add(msg)
        if is_sent:
            db.add(MessageLabel(message_id=msg_id, label_id="SENT"))
        else:
            for lid in BUDGET_THREAD.get("labels", ["INBOX"]):
                db.add(MessageLabel(message_id=msg_id, label_id=lid))

    # Ambiguous edge cases (exactly 20, inserted once each)
    for template in AMBIGUOUS_EMAILS:
        _insert_single(db, user, template, now, random)

    fixed_count = len(NEEDLES) + len(budget_msgs) + len(AMBIGUOUS_EMAILS)
    remaining = target_count - fixed_count

    # --- Phase 2: Fill from content library by category ---

    n_notif = int(remaining * 0.35)
    n_newsletter = int(remaining * 0.20)
    n_work = int(remaining * 0.20)
    n_personal = int(remaining * 0.10)
    n_sent = int(remaining * 0.05)
    n_spam = int(remaining * 0.05)
    # Remainder goes to notifications (the most naturally repetitive type)
    n_notif += remaining - n_notif - n_newsletter - n_work - n_personal - n_sent - n_spam

    # Work: insert all threads as proper multi-message threads first,
    # then fill remaining with singles from the flattened work pool.
    work_inserted = 0
    for thread_tmpl in WORK_THREADS:
        if work_inserted >= n_work:
            break
        count = _insert_thread(db, user, thread_tmpl, now, random)
        work_inserted += count

    if work_inserted < n_work:
        _fill_from_pool(
            db, user, _WORK_POOL, n_work - work_inserted, now, random,
        )

    # Notifications: parameterized, ~50% forced old (>14 days) for cleanup task
    _fill_from_pool(
        db, user, NOTIFICATIONS, n_notif, now, random,
        parameterize=True,
        old_notification_ratio=0.5,
    )

    # Newsletters and promos
    _fill_from_pool(db, user, NEWSLETTERS, n_newsletter, now, random)

    # Personal
    _fill_from_pool(db, user, PERSONAL_EMAILS, n_personal, now, random)

    # Sent
    _fill_from_pool(db, user, SENT_EMAILS, n_sent, now, random, force_sent=True)

    # Spam
    _fill_from_pool(db, user, SPAM_EMAILS, n_spam, now, random, force_spam=True)

    # Draft
    draft_id = _make_id()
    draft_msg_id = _make_id()
    draft_thread_id = _make_id()
    db.add(Thread(id=draft_thread_id, user_id=user.id, snippet=""))
    db.add(Message(
        id=draft_msg_id,
        thread_id=draft_thread_id,
        user_id=user.id,
        sender=user_email,
        to="team@nexusai.com",
        subject="Q1 Infrastructure Optimization Summary",
        snippet="Draft summary of Q1 infrastructure changes...",
        body_plain="Draft summary of Q1 infrastructure changes and cost savings...",
        internal_date=now,
        is_draft=True,
    ))
    db.add(MessageLabel(message_id=draft_msg_id, label_id="DRAFT"))
    db.add(Draft(id=draft_id, user_id=user.id, message_id=draft_msg_id))
