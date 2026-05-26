"""Notification email templates for the BenchFlow founder inbox scenario.

~60 realistic notification emails modeled after what a startup founder/developer
actually receives. Sender addresses, formatting, and tone match the real services.

Persona: Founder at BenchFlow (developer tools startup).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# NOTIFICATIONS — list of template dicts
# ---------------------------------------------------------------------------

NOTIFICATIONS: list[dict] = [
    # -----------------------------------------------------------------------
    # LUMA  (~15 templates)
    # -----------------------------------------------------------------------
    {
        "sender_name": "Luma",
        "sender_email": "support@luma.com",
        "subject": "{attendee_name} registered for {event_name}",
        "body_plain": (
            "New registration\n\n"
            "{attendee_name} ({company}) just registered for {event_name}.\n\n"
            "You now have {{total_guests}} guests registered.\n\n"
            "View guest list: https://lu.ma/manage/evt-benchflow\n\n"
            "-- Luma"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.3,
        "age_range": (0, 7),
    },
    {
        "sender_name": "Luma",
        "sender_email": "support@luma.com",
        "subject": "{attendee_name} registered for {event_name}",
        "body_plain": (
            "New registration\n\n"
            "{attendee_name} ({company}) just registered for {event_name}.\n\n"
            "You now have {{total_guests}} guests registered.\n\n"
            "View guest list: https://lu.ma/manage/evt-benchflow\n\n"
            "-- Luma"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.2,
        "age_range": (0, 3),
    },
    {
        "sender_name": "Luma",
        "sender_email": "support@luma.com",
        "subject": "{attendee_name} registered for {event_name}",
        "body_plain": (
            "New registration\n\n"
            "{attendee_name} ({company}) just registered for {event_name}.\n\n"
            "You now have {{total_guests}} guests registered.\n\n"
            "View guest list: https://lu.ma/manage/evt-benchflow\n\n"
            "-- Luma"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.15,
        "age_range": (0, 3),
    },
    {
        "sender_name": "Luma",
        "sender_email": "support@luma.com",
        "subject": "{attendee_name} registered for {event_name}",
        "body_plain": (
            "New registration\n\n"
            "{attendee_name} ({company}) just registered for {event_name}.\n\n"
            "You now have {{total_guests}} guests registered.\n\n"
            "View guest list: https://lu.ma/manage/evt-benchflow\n\n"
            "-- Luma"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.15,
        "age_range": (0, 5),
    },
    {
        "sender_name": "Luma",
        "sender_email": "support@luma.com",
        "subject": "{attendee_name} registered for {event_name}",
        "body_plain": (
            "New registration\n\n"
            "{attendee_name} ({company}) just registered for {event_name}.\n\n"
            "You now have {{total_guests}} guests registered.\n\n"
            "View guest list: https://lu.ma/manage/evt-benchflow\n\n"
            "-- Luma"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.1,
        "age_range": (0, 5),
    },
    {
        "sender_name": "Luma",
        "sender_email": "support@luma.com",
        "subject": "{attendee_name} registered for {event_name}",
        "body_plain": (
            "New registration\n\n"
            "{attendee_name} ({company}) just registered for {event_name}.\n\n"
            "You now have {{total_guests}} guests registered.\n\n"
            "View guest list: https://lu.ma/manage/evt-benchflow\n\n"
            "-- Luma"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.1,
        "age_range": (0, 2),
    },
    {
        "sender_name": "Luma",
        "sender_email": "support@luma.com",
        "subject": "{attendee_name} registered for {event_name}",
        "body_plain": (
            "New registration\n\n"
            "{attendee_name} ({company}) just registered for {event_name}.\n\n"
            "You now have {{total_guests}} guests registered.\n\n"
            "View guest list: https://lu.ma/manage/evt-benchflow\n\n"
            "-- Luma"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.1,
        "age_range": (0, 2),
    },
    {
        "sender_name": "Luma",
        "sender_email": "noreply@luma-mail.com",
        "subject": "You're registered for Skillathon SF Hackathon",
        "body_plain": (
            "You're in!\n\n"
            "Skillathon SF Hackathon\n"
            "Saturday, March 8, 2026 at 9:00 AM PST\n"
            "The Commons, 425 Mission St, San Francisco, CA\n\n"
            "Add to calendar: https://lu.ma/evt-skillathon-sf\n\n"
            "What to bring:\n"
            "- Laptop + charger\n"
            "- Government-issued ID for building entry\n\n"
            "Questions? Reply to this email or message us on the event page.\n\n"
            "See you there!\n"
            "-- Luma"
        ),
        "body_html": (
            "<div style='font-family: Inter, sans-serif; max-width: 480px; margin: 0 auto;'>"
            "<h2 style='margin-bottom: 4px;'>You're in!</h2>"
            "<p><strong>Skillathon SF Hackathon</strong><br>"
            "Saturday, March 8, 2026 at 9:00 AM PST<br>"
            "The Commons, 425 Mission St, San Francisco, CA</p>"
            "<a href='https://lu.ma/evt-skillathon-sf' "
            "style='display:inline-block;padding:10px 24px;background:#111;color:#fff;"
            "border-radius:6px;text-decoration:none;'>Add to Calendar</a>"
            "<p style='color:#666;font-size:13px;margin-top:20px;'>"
            "What to bring: Laptop + charger, Government-issued ID for building entry.</p>"
            "<p style='color:#999;font-size:12px;'>-- Luma</p></div>"
        ),
        "category": "CATEGORY_PERSONAL",
        "is_read_probability": 0.95,
        "age_range": (1, 10),
    },
    {
        "sender_name": "Luma",
        "sender_email": "noreply@luma-mail.com",
        "subject": "You're registered for AI Founders Dinner",
        "body_plain": (
            "You're in!\n\n"
            "AI Founders Dinner\n"
            "Thursday, March 13, 2026 at 7:00 PM PST\n"
            "Beretta, 1199 Valencia St, San Francisco, CA\n\n"
            "Add to calendar: https://lu.ma/evt-ai-founders-dinner\n\n"
            "This is a private dinner for 20 founders. Dress code: smart casual.\n\n"
            "See you there!\n"
            "-- Luma"
        ),
        "body_html": "",
        "category": "CATEGORY_PERSONAL",
        "is_read_probability": 0.9,
        "age_range": (2, 14),
    },
    {
        "sender_name": "Luma",
        "sender_email": "noreply@luma-mail.com",
        "subject": "Reminder: Skillathon SF Hackathon is tomorrow",
        "body_plain": (
            "Reminder: You're attending Skillathon SF Hackathon tomorrow.\n\n"
            "Saturday, March 8, 2026 at 9:00 AM PST\n"
            "The Commons, 425 Mission St, San Francisco, CA\n\n"
            "Doors open at 8:30 AM. Don't forget your laptop and ID.\n\n"
            "View event: https://lu.ma/evt-skillathon-sf\n\n"
            "-- Luma"
        ),
        "body_html": "",
        "category": "CATEGORY_PERSONAL",
        "is_read_probability": 0.85,
        "age_range": (0, 3),
    },
    {
        "sender_name": "Luma",
        "sender_email": "support@luma.com",
        "subject": "Your event submission is under review",
        "body_plain": (
            "Hi there,\n\n"
            "We received your event submission:\n\n"
            "  BenchFlow Demo Day - March 20\n\n"
            "Our team will review it and get back to you within 24 hours. "
            "Once approved, your event page will go live and you can start sharing it.\n\n"
            "If you need to make changes, you can edit the draft at:\n"
            "https://lu.ma/manage/evt-benchflow-demo-day\n\n"
            "Thanks,\n"
            "The Luma Team"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.95,
        "age_range": (3, 14),
    },
    {
        "sender_name": "Luma",
        "sender_email": "support@luma.com",
        "subject": "Your event has been approved",
        "body_plain": (
            "Great news! Your event has been approved and is now live.\n\n"
            "  BenchFlow Demo Day - March 20\n"
            "  https://lu.ma/benchflow-demo-day\n\n"
            "Share this link with your audience to start collecting registrations.\n\n"
            "Tips for a great event page:\n"
            "- Add a cover image (recommended 1200x630)\n"
            "- Fill out the description with an agenda\n"
            "- Enable the waitlist if you expect high demand\n\n"
            "Good luck!\n"
            "The Luma Team"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.95,
        "age_range": (2, 12),
    },
    {
        "sender_name": "Luma",
        "sender_email": "support@luma.com",
        "subject": "3 guests cancelled for Skillathon SF Hackathon",
        "body_plain": (
            "Guest update for Skillathon SF Hackathon\n\n"
            "3 guests have cancelled their registration:\n\n"
            "- Raj Mehta (Anthropic)\n"
            "- Sophia Lin (Stripe)\n"
            "- Tom Bradford (Independent)\n\n"
            "You now have 47 guests registered (50 spots remaining).\n\n"
            "View guest list: https://lu.ma/manage/evt-skillathon-sf\n\n"
            "-- Luma"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.6,
        "age_range": (0, 5),
    },
    {
        "sender_name": "Luma",
        "sender_email": "support@luma.com",
        "subject": "Weekly host digest: 12 new registrations",
        "body_plain": (
            "Your weekly Luma digest\n\n"
            "Here's what happened across your events this week:\n\n"
            "Skillathon SF Hackathon\n"
            "  +12 new registrations (47 total)\n"
            "  3 cancellations\n\n"
            "BenchFlow Demo Day\n"
            "  +5 new registrations (5 total)\n"
            "  0 cancellations\n\n"
            "Total page views: 342\n\n"
            "Manage your events: https://lu.ma/dashboard\n\n"
            "-- Luma"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.7,
        "age_range": (0, 7),
    },
    {
        "sender_name": "Luma",
        "sender_email": "noreply@luma-mail.com",
        "subject": "You're registered for YC W26 Demo Day Watch Party",
        "body_plain": (
            "You're in!\n\n"
            "YC W26 Demo Day Watch Party\n"
            "Wednesday, March 19, 2026 at 5:00 PM PST\n"
            "Shack15, 1 Ferry Building, San Francisco, CA\n\n"
            "Add to calendar: https://lu.ma/evt-yc-watch-party\n\n"
            "Drinks and light bites provided. Capacity is limited to 80.\n\n"
            "See you there!\n"
            "-- Luma"
        ),
        "body_html": "",
        "category": "CATEGORY_PERSONAL",
        "is_read_probability": 0.85,
        "age_range": (1, 10),
    },
    # -----------------------------------------------------------------------
    # GITHUB  (~8 templates)
    # -----------------------------------------------------------------------
    {
        "sender_name": "GitHub",
        "sender_email": "notifications@github.com",
        "subject": "[benchflow/benchflow-core] CI run failed: main (abc4f21)",
        "body_plain": (
            "Run failed: Build and Test\n\n"
            "benchflow/benchflow-core  main  abc4f21\n\n"
            "Jobs:\n"
            "  x  test-unit (ubuntu-latest)  FAILED\n"
            "     Error: FAILED tests/test_runner.py::test_parallel_execution -\n"
            "     AssertionError: Expected 4 results, got 3\n"
            "  v  lint (ubuntu-latest)  PASSED\n"
            "  v  typecheck (ubuntu-latest)  PASSED\n\n"
            "Commit: Fix race condition in task scheduler\n"
            "Author: alex-benchflow\n\n"
            "View run: https://github.com/benchflow/benchflow-core/actions/runs/12847392\n\n"
            "-- \n"
            "You are receiving this because you are subscribed to this repository."
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.85,
        "age_range": (0, 3),
    },
    {
        "sender_name": "GitHub",
        "sender_email": "notifications@github.com",
        "subject": "[benchflow/benchflow-core] CI run failed: feat/streaming (e91d0c3)",
        "body_plain": (
            "Run failed: Build and Test\n\n"
            "benchflow/benchflow-core  feat/streaming  e91d0c3\n\n"
            "Jobs:\n"
            "  x  test-integration (ubuntu-latest)  FAILED\n"
            "     Error: ConnectionRefusedError: [Errno 111] Connection refused\n"
            "     tests/integration/test_streaming.py::test_sse_reconnect\n"
            "  v  test-unit (ubuntu-latest)  PASSED\n"
            "  v  lint (ubuntu-latest)  PASSED\n\n"
            "Commit: Add SSE reconnection logic\n"
            "Author: alex-benchflow\n\n"
            "View run: https://github.com/benchflow/benchflow-core/actions/runs/12851203\n\n"
            "-- \n"
            "You are receiving this because you are subscribed to this repository."
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.8,
        "age_range": (0, 5),
    },
    {
        "sender_name": "GitHub",
        "sender_email": "notifications@github.com",
        "subject": "[benchflow/sdk-python] PR #87: Add retry middleware (review requested)",
        "body_plain": (
            "danielle-m requested your review on PR #87.\n\n"
            "Add retry middleware with exponential backoff\n\n"
            "+142 -23 across 4 files\n\n"
            "This PR adds configurable retry middleware for transient failures.\n"
            "Default: 3 retries with exponential backoff (1s, 2s, 4s).\n\n"
            "Key changes:\n"
            "- New RetryMiddleware class in sdk/middleware/retry.py\n"
            "- Configuration via BenchFlowClient(retries=3, backoff_factor=2)\n"
            "- Tests covering timeout, 429, and 503 scenarios\n\n"
            "Review: https://github.com/benchflow/sdk-python/pull/87\n\n"
            "-- \n"
            "You are receiving this because your review was requested."
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.9,
        "age_range": (0, 5),
    },
    {
        "sender_name": "GitHub",
        "sender_email": "notifications@github.com",
        "subject": "[benchflow/benchflow-core] PR #312 merged: Upgrade to Python 3.12",
        "body_plain": (
            "Pull request #312 merged by alex-benchflow.\n\n"
            "Upgrade to Python 3.12\n\n"
            "Base: main <- feat/py312\n"
            "+87 -42 across 12 files\n\n"
            "Summary:\n"
            "- Update pyproject.toml to require Python >= 3.12\n"
            "- Replace deprecated datetime.utcnow() calls\n"
            "- Update CI matrix to test 3.12 only\n"
            "- Remove 3.10/3.11 compatibility shims\n\n"
            "View: https://github.com/benchflow/benchflow-core/pull/312\n\n"
            "-- \n"
            "You are receiving this because you are subscribed to this repository."
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.75,
        "age_range": (1, 10),
    },
    {
        "sender_name": "GitHub",
        "sender_email": "noreply@github.com",
        "subject": "[GitHub] A new OAuth application was authorized on your account",
        "body_plain": (
            "Hey alex-benchflow,\n\n"
            "A new OAuth application was authorized on your account:\n\n"
            "  Application: Linear\n"
            "  Permissions: read:user, repo\n"
            "  Authorized at: 2026-02-28 14:32 UTC\n\n"
            "If you did not authorize this application, please review your "
            "authorized applications immediately:\n"
            "https://github.com/settings/applications\n\n"
            "Thanks,\n"
            "The GitHub Team"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.7,
        "age_range": (1, 14),
    },
    {
        "sender_name": "GitHub",
        "sender_email": "noreply@github.com",
        "subject": "[GitHub] You've been granted admin access to benchflow/infra",
        "body_plain": (
            "Hey alex-benchflow,\n\n"
            "You've been granted admin access to the repository:\n\n"
            "  benchflow/infra\n\n"
            "Granted by: @marco-benchflow\n\n"
            "You can now manage settings, collaborators, and protected branches "
            "for this repository.\n\n"
            "View repository: https://github.com/benchflow/infra\n\n"
            "Thanks,\n"
            "The GitHub Team"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.85,
        "age_range": (3, 21),
    },
    {
        "sender_name": "GitHub",
        "sender_email": "noreply@github.com",
        "subject": "[GitHub] GitHub Copilot: Your usage report for February 2026",
        "body_plain": (
            "Hi alex-benchflow,\n\n"
            "Here's your GitHub Copilot usage summary for February 2026:\n\n"
            "  Suggestions accepted: 1,247\n"
            "  Acceptance rate: 34%\n"
            "  Lines of code generated: 3,891\n"
            "  Most active language: Python (62%)\n"
            "  Chat conversations: 89\n\n"
            "Your team's usage across BenchFlow org:\n"
            "  Active seats: 6 / 8\n"
            "  Org acceptance rate: 29%\n\n"
            "View detailed report: https://github.com/orgs/benchflow/copilot/usage\n\n"
            "Thanks,\n"
            "The GitHub Team"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.6,
        "age_range": (1, 7),
    },
    {
        "sender_name": "GitHub",
        "sender_email": "notifications@github.com",
        "subject": "[benchflow/sdk-python] Issue #91: Memory leak in connection pool",
        "body_plain": (
            "kayla-user opened issue #91:\n\n"
            "Memory leak in connection pool\n\n"
            "Description:\n"
            "When using BenchFlowClient in a long-running process, memory usage grows\n"
            "unbounded. After ~12 hours the process is using 4GB+ RSS.\n\n"
            "Reproduction:\n"
            "```python\n"
            "client = BenchFlowClient()\n"
            "while True:\n"
            "    client.run(task_id='benchmark-001')\n"
            "    time.sleep(60)\n"
            "```\n\n"
            "Environment: Python 3.12, sdk-python 0.4.2, Ubuntu 22.04\n\n"
            "Labels: bug, priority:high\n\n"
            "View: https://github.com/benchflow/sdk-python/issues/91\n\n"
            "-- \n"
            "You are receiving this because you are subscribed to this repository."
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.95,
        "age_range": (0, 7),
    },
    # -----------------------------------------------------------------------
    # CAL.COM / OTTER.AI  (~6 templates)
    # -----------------------------------------------------------------------
    {
        "sender_name": "Cal.com",
        "sender_email": "hello@cal.com",
        "subject": "New booking: Product Demo with Sarah Kim",
        "body_plain": (
            "New Booking\n\n"
            "Product Demo\n"
            "30 min\n\n"
            "When: Monday, March 3, 2026 at 2:00 PM PST\n"
            "Who: Sarah Kim (sarah.kim@techcorp.io)\n"
            "Where: Google Meet (link below)\n\n"
            "Join: https://meet.google.com/abc-defg-hij\n\n"
            "Additional notes from guest:\n"
            "\"Interested in the CI integration for our monorepo setup. "
            "We have ~200 engineers.\"\n\n"
            "Manage booking: https://app.cal.com/bookings\n\n"
            "-- Cal.com"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.95,
        "age_range": (0, 3),
    },
    {
        "sender_name": "Cal.com",
        "sender_email": "hello@cal.com",
        "subject": "New booking: Investor Chat with David Park",
        "body_plain": (
            "New Booking\n\n"
            "Investor Chat\n"
            "45 min\n\n"
            "When: Wednesday, March 5, 2026 at 11:00 AM PST\n"
            "Who: David Park (dpark@gradient.vc)\n"
            "Where: Google Meet (link below)\n\n"
            "Join: https://meet.google.com/klm-nopq-rst\n\n"
            "No additional notes.\n\n"
            "Manage booking: https://app.cal.com/bookings\n\n"
            "-- Cal.com"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.95,
        "age_range": (0, 5),
    },
    {
        "sender_name": "Cal.com",
        "sender_email": "hello@cal.com",
        "subject": "Booking cancelled: Intro Call with James Oduya",
        "body_plain": (
            "Booking Cancelled\n\n"
            "Intro Call\n"
            "15 min\n\n"
            "Was: Tuesday, March 4, 2026 at 3:30 PM PST\n"
            "Who: James Oduya (james@foundermode.co)\n\n"
            "Reason: \"Something came up, will rebook next week.\"\n\n"
            "The calendar hold has been removed.\n\n"
            "Manage booking: https://app.cal.com/bookings\n\n"
            "-- Cal.com"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.9,
        "age_range": (0, 7),
    },
    {
        "sender_name": "Otter.ai",
        "sender_email": "no-reply@otter.ai",
        "subject": "Meeting summary: BenchFlow <> TechCorp Product Demo",
        "body_plain": (
            "Your meeting notes are ready\n\n"
            "BenchFlow <> TechCorp Product Demo\n"
            "March 3, 2026 | 2:00 PM - 2:34 PM PST | 2 participants\n\n"
            "Summary:\n"
            "Sarah Kim from TechCorp is evaluating CI tools for a 200-engineer org. "
            "Key requirements: monorepo support, GitHub Actions integration, and "
            "custom benchmark suites. Alex walked through the dashboard and streaming "
            "results API. Sarah wants to schedule a follow-up technical deep dive "
            "with her platform team.\n\n"
            "Action items:\n"
            "- Alex: Send pricing sheet for Team plan (200 seats)\n"
            "- Alex: Prepare monorepo demo environment\n"
            "- Sarah: Loop in platform lead (Wei Chen) for next call\n\n"
            "View full transcript: https://otter.ai/u/meeting-abc123\n\n"
            "-- Otter.ai"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.8,
        "age_range": (0, 5),
    },
    {
        "sender_name": "Otter.ai",
        "sender_email": "no-reply@otter.ai",
        "subject": "Meeting summary: Weekly Team Standup",
        "body_plain": (
            "Your meeting notes are ready\n\n"
            "Weekly Team Standup\n"
            "February 28, 2026 | 10:00 AM - 10:22 AM PST | 5 participants\n\n"
            "Summary:\n"
            "Sprint progress review. Streaming feature at 80% completion, blocked on "
            "WebSocket auth issue. SDK Python v0.5 release pushed to next week. "
            "Marco finishing infra migration to new k8s cluster. Hiring update: "
            "two final-round candidates for senior backend role.\n\n"
            "Action items:\n"
            "- Danielle: Fix WebSocket auth before EOD Friday\n"
            "- Marco: Complete k8s migration runbook\n"
            "- Alex: Final interviews Monday/Tuesday\n\n"
            "View full transcript: https://otter.ai/u/meeting-def456\n\n"
            "-- Otter.ai"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.75,
        "age_range": (1, 7),
    },
    {
        "sender_name": "Otter.ai",
        "sender_email": "no-reply@otter.ai",
        "subject": "Unable to record: Investor Chat with David Park",
        "body_plain": (
            "We were unable to record your meeting.\n\n"
            "Meeting: Investor Chat with David Park\n"
            "Scheduled: March 5, 2026 at 11:00 AM PST\n\n"
            "Reason: Otter was removed from the meeting by another participant "
            "before recording could begin.\n\n"
            "Tip: Ask participants to accept the Otter bot when it joins, or "
            "add no-reply@otter.ai to your calendar invite so it's expected.\n\n"
            "-- Otter.ai"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.65,
        "age_range": (0, 5),
    },
    # -----------------------------------------------------------------------
    # FINANCIAL — BREX, MERCURY  (~5 templates)
    # -----------------------------------------------------------------------
    {
        "sender_name": "Brex",
        "sender_email": "noreply@brex.com",
        "subject": "Your February 2026 statement is ready",
        "body_plain": (
            "Hi Alex,\n\n"
            "Your Brex statement for February 2026 is now available.\n\n"
            "Account: BenchFlow, Inc. - Operating (*4821)\n\n"
            "Statement period: Feb 1 - Feb 28, 2026\n"
            "Total spend: $14,832.47\n"
            "Cash back earned: $148.32\n\n"
            "Top categories:\n"
            "  Software & SaaS: $6,241.00\n"
            "  Cloud infrastructure: $5,892.33\n"
            "  Travel & meals: $1,847.14\n"
            "  Other: $852.00\n\n"
            "View statement: https://dashboard.brex.com/statements\n\n"
            "Thanks,\n"
            "Brex"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.7,
        "age_range": (1, 5),
    },
    {
        "sender_name": "Brex",
        "sender_email": "noreply@brex.com",
        "subject": "Transaction alert: $2,499.00 at AWS",
        "body_plain": (
            "Transaction Alert\n\n"
            "A charge was made on your Brex card:\n\n"
            "Amount: $2,499.00\n"
            "Merchant: Amazon Web Services\n"
            "Card: BenchFlow Operating (*4821)\n"
            "Date: March 1, 2026\n\n"
            "This matches your recurring charge pattern for this merchant.\n\n"
            "Not you? Freeze your card immediately at https://dashboard.brex.com\n\n"
            "-- Brex"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.5,
        "age_range": (0, 5),
    },
    {
        "sender_name": "Mercury",
        "sender_email": "hello@mercury.com",
        "subject": "Weekly cash flow digest - BenchFlow, Inc.",
        "body_plain": (
            "Weekly Cash Flow Digest\n"
            "BenchFlow, Inc.\n"
            "Week of Feb 24 - Mar 2, 2026\n\n"
            "Operating Account (*7392)\n"
            "  Starting balance: $284,102.55\n"
            "  Money in: +$12,500.00\n"
            "    - Stripe payout: $8,200.00\n"
            "    - Wire from Gradient Ventures: $4,300.00\n"
            "  Money out: -$18,341.22\n"
            "    - Payroll (Gusto): $11,200.00\n"
            "    - AWS: $2,499.00\n"
            "    - Brex card payment: $3,142.22\n"
            "    - Other: $1,500.00\n"
            "  Ending balance: $278,261.33\n\n"
            "Treasury Account (*7393)\n"
            "  Balance: $520,000.00 (no change)\n"
            "  APY: 4.85%\n\n"
            "View details: https://app.mercury.com/dashboard\n\n"
            "-- Mercury"
        ),
        "body_html": (
            "<div style='font-family: -apple-system, sans-serif; max-width: 560px;'>"
            "<h2 style='color: #1a1a2e;'>Weekly Cash Flow Digest</h2>"
            "<p style='color: #666;'>BenchFlow, Inc. | Week of Feb 24 - Mar 2, 2026</p>"
            "<table style='width:100%; border-collapse:collapse; margin:16px 0;'>"
            "<tr style='border-bottom:1px solid #eee;'>"
            "<td style='padding:8px 0;'><strong>Operating (*7392)</strong></td>"
            "<td style='text-align:right;'>$278,261.33</td></tr>"
            "<tr style='border-bottom:1px solid #eee;'>"
            "<td style='padding:8px 0;'>Money in</td>"
            "<td style='text-align:right; color:#22c55e;'>+$12,500.00</td></tr>"
            "<tr style='border-bottom:1px solid #eee;'>"
            "<td style='padding:8px 0;'>Money out</td>"
            "<td style='text-align:right; color:#ef4444;'>-$18,341.22</td></tr>"
            "<tr><td style='padding:8px 0;'><strong>Treasury (*7393)</strong></td>"
            "<td style='text-align:right;'>$520,000.00</td></tr>"
            "</table>"
            "<a href='https://app.mercury.com/dashboard' "
            "style='display:inline-block;padding:10px 20px;background:#5046e5;"
            "color:#fff;border-radius:6px;text-decoration:none;'>View Dashboard</a>"
            "<p style='color:#999;font-size:12px;margin-top:24px;'>-- Mercury</p></div>"
        ),
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.8,
        "age_range": (0, 7),
    },
    {
        "sender_name": "Mercury",
        "sender_email": "hello@mercury.com",
        "subject": "Wire received: $4,300.00 from Gradient Ventures",
        "body_plain": (
            "Wire Transfer Received\n\n"
            "Amount: $4,300.00\n"
            "From: Gradient Ventures LLC\n"
            "To: BenchFlow, Inc. - Operating (*7392)\n"
            "Date: February 27, 2026\n"
            "Reference: SAFE Note - Tranche 2\n\n"
            "Your new balance: $296,602.55\n\n"
            "View transaction: https://app.mercury.com/transactions\n\n"
            "-- Mercury"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.9,
        "age_range": (2, 10),
    },
    {
        "sender_name": "Brex",
        "sender_email": "noreply@brex.com",
        "subject": "New card request: Danielle Martinez",
        "body_plain": (
            "Card Request\n\n"
            "Danielle Martinez has requested a new Brex card.\n\n"
            "Requested card: Virtual card\n"
            "Spend limit: $2,000/month\n"
            "Category: Software & SaaS\n\n"
            "As an admin, you can approve or decline this request.\n\n"
            "Review: https://dashboard.brex.com/cards/requests\n\n"
            "-- Brex"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.85,
        "age_range": (1, 10),
    },
    # -----------------------------------------------------------------------
    # SAAS LIFECYCLE  (~5 templates)
    # -----------------------------------------------------------------------
    {
        "sender_name": "HubSpot",
        "sender_email": "noreply@notifications.hubspot.com",
        "subject": "Action required: Your HubSpot profile will be deleted in 30 days",
        "body_plain": (
            "Hi Alex,\n\n"
            "Your HubSpot account (alex@benchflow.dev) has been inactive for "
            "12 months. Per our data retention policy, your profile and associated "
            "data will be permanently deleted on April 1, 2026.\n\n"
            "Data that will be removed:\n"
            "- Contact records (127 contacts)\n"
            "- Deal pipeline (3 deals)\n"
            "- Email templates (8 templates)\n"
            "- Reporting dashboards\n\n"
            "To keep your account, simply log in before the deletion date:\n"
            "https://app.hubspot.com/login\n\n"
            "To export your data before deletion:\n"
            "https://app.hubspot.com/settings/export\n\n"
            "If you have questions, contact support@hubspot.com.\n\n"
            "-- The HubSpot Team"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.6,
        "age_range": (5, 21),
    },
    {
        "sender_name": "Zed",
        "sender_email": "hi@zed.dev",
        "subject": "Updated Terms of Service - effective March 15, 2026",
        "body_plain": (
            "Hi there,\n\n"
            "We're writing to let you know that we've updated our Terms of Service. "
            "The updated terms will take effect on March 15, 2026.\n\n"
            "Key changes:\n"
            "- Clarified data handling for Zed AI features (Section 4.2)\n"
            "- Updated dispute resolution process (Section 9)\n"
            "- Added terms for team/enterprise plans (new Section 12)\n\n"
            "You can review the full updated terms at:\n"
            "https://zed.dev/terms\n\n"
            "By continuing to use Zed after March 15, you agree to the updated terms.\n\n"
            "If you have questions, email legal@zed.dev.\n\n"
            "Best,\n"
            "The Zed Team"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.3,
        "age_range": (3, 14),
    },
    {
        "sender_name": "Arcade",
        "sender_email": "support@arcade.dev",
        "subject": "We've updated our Privacy Policy",
        "body_plain": (
            "Hi Alex,\n\n"
            "We've made updates to our Privacy Policy to provide more transparency "
            "about how we handle your data.\n\n"
            "What's changed:\n"
            "- More detail on how we use product analytics (Section 3)\n"
            "- Clarified third-party integrations data sharing (Section 5)\n"
            "- Updated cookie policy to reflect new consent requirements (Section 7)\n\n"
            "The updated policy is effective March 10, 2026.\n\n"
            "Read the full policy: https://arcade.dev/privacy\n\n"
            "Questions? Reach us at privacy@arcade.dev.\n\n"
            "-- Arcade Team"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.2,
        "age_range": (3, 14),
    },
    {
        "sender_name": "Mint Mobile",
        "sender_email": "no-reply@account.mintmobile.com",
        "subject": "Your Mint Mobile plan expires in 7 days",
        "body_plain": (
            "Hey Alex,\n\n"
            "Heads up - your 3-month Unlimited plan is expiring on March 10, 2026.\n\n"
            "Current plan: Unlimited (5G)\n"
            "Phone number: (415) 555-0142\n"
            "Expires: March 10, 2026\n\n"
            "Renew now to keep your number and avoid service interruption:\n"
            "https://account.mintmobile.com/renew\n\n"
            "Renewal options:\n"
            "  3 months: $30/mo\n"
            "  6 months: $25/mo\n"
            "  12 months: $15/mo (best value)\n\n"
            "If you don't renew, your number will be held for 60 days.\n\n"
            "Thanks for being a Mint customer!\n"
            "-- Mint Mobile"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.75,
        "age_range": (1, 10),
    },
    {
        "sender_name": "Notion",
        "sender_email": "notify@notifications.notion.so",
        "subject": "Your Notion AI usage summary for February",
        "body_plain": (
            "Hi Alex,\n\n"
            "Here's your Notion AI usage summary for February 2026.\n\n"
            "Your workspace: BenchFlow\n"
            "  AI requests used: 847 / unlimited\n"
            "  Most used features:\n"
            "    - Summarize: 312 requests\n"
            "    - Draft content: 245 requests\n"
            "    - Translate: 156 requests\n"
            "    - Other: 134 requests\n\n"
            "  Active AI users: 4 of 6 members\n\n"
            "Explore what's new with Notion AI: https://notion.so/ai\n\n"
            "-- The Notion Team"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.35,
        "age_range": (1, 7),
    },
    # -----------------------------------------------------------------------
    # LINKEDIN  (~5 templates)
    # -----------------------------------------------------------------------
    {
        "sender_name": "LinkedIn",
        "sender_email": "notifications-noreply@linkedin.com",
        "subject": "Alex, your profile was viewed by 42 people last week",
        "body_plain": (
            "Your weekly search stats\n\n"
            "42 people viewed your profile last week\n"
            "You appeared in 118 search results\n\n"
            "Top viewers:\n"
            "- Senior Recruiter at Anthropic\n"
            "- Engineering Manager at Stripe\n"
            "- VP of Platform at Notion\n"
            "- 2 people from your network\n\n"
            "See all viewers: https://www.linkedin.com/me/profile-views/\n\n"
            "Your profile is getting more attention than 89% of your connections.\n\n"
            "-- LinkedIn"
        ),
        "body_html": "",
        "category": "CATEGORY_SOCIAL",
        "is_read_probability": 0.5,
        "age_range": (0, 7),
    },
    {
        "sender_name": "LinkedIn",
        "sender_email": "notifications-noreply@linkedin.com",
        "subject": "Jessica Torres sent you a connection request",
        "body_plain": (
            "Jessica Torres wants to connect\n\n"
            "Jessica Torres\n"
            "Head of Partnerships at Replit\n"
            "San Francisco Bay Area\n\n"
            "Note: \"Hi Alex! Loved your post about CI/CD benchmarking. "
            "Would love to explore a potential integration with Replit. "
            "Let's connect!\"\n\n"
            "Accept: https://www.linkedin.com/comm/mynetwork/invite-accept/12345\n"
            "Ignore: https://www.linkedin.com/comm/mynetwork/invite-ignore/12345\n\n"
            "-- LinkedIn"
        ),
        "body_html": "",
        "category": "CATEGORY_SOCIAL",
        "is_read_probability": 0.65,
        "age_range": (0, 10),
    },
    {
        "sender_name": "LinkedIn",
        "sender_email": "notifications-noreply@linkedin.com",
        "subject": "Your post is getting attention - 1,247 impressions",
        "body_plain": (
            "Your recent post is performing well!\n\n"
            "\"We just open-sourced our benchmark runner. After 6 months of "
            "building BenchFlow internally...\"\n\n"
            "1,247 impressions\n"
            "89 reactions\n"
            "23 comments\n"
            "14 reposts\n\n"
            "Top comment:\n"
            "Sarah Drasner: \"This is exactly what the ecosystem needs. "
            "We've been doing this manually for years.\"\n\n"
            "View post: https://www.linkedin.com/feed/update/urn:li:activity:12345\n\n"
            "-- LinkedIn"
        ),
        "body_html": "",
        "category": "CATEGORY_SOCIAL",
        "is_read_probability": 0.7,
        "age_range": (1, 10),
    },
    {
        "sender_name": "LinkedIn",
        "sender_email": "notifications-noreply@linkedin.com",
        "subject": "Alex, 3 people endorsed you for Python this week",
        "body_plain": (
            "New endorsements\n\n"
            "3 people endorsed you for Python this week:\n"
            "- Marco Rossi (BenchFlow)\n"
            "- Priya Kapoor (ex-Google)\n"
            "- David Kim (Stripe)\n\n"
            "You now have 47 endorsements for Python.\n\n"
            "Your top skills:\n"
            "1. Python (47 endorsements)\n"
            "2. Distributed Systems (31)\n"
            "3. Technical Leadership (28)\n\n"
            "View your profile: https://www.linkedin.com/in/alex-benchflow\n\n"
            "-- LinkedIn"
        ),
        "body_html": "",
        "category": "CATEGORY_SOCIAL",
        "is_read_probability": 0.3,
        "age_range": (0, 14),
    },
    {
        "sender_name": "LinkedIn",
        "sender_email": "notifications-noreply@linkedin.com",
        "subject": "Trending in your network: AI infrastructure in 2026",
        "body_plain": (
            "Trending in your network\n\n"
            "Top stories your connections are engaging with:\n\n"
            "1. \"Why every startup needs a benchmarking strategy\" - 12K views\n"
            "   By: Martin Casado, a16z\n\n"
            "2. \"The real cost of running LLMs in production\" - 8.4K views\n"
            "   By: Chip Huyen\n\n"
            "3. \"Open source is eating CI/CD\" - 6.2K views\n"
            "   By: Mitchell Hashimoto\n\n"
            "Read more: https://www.linkedin.com/feed/\n\n"
            "-- LinkedIn"
        ),
        "body_html": "",
        "category": "CATEGORY_SOCIAL",
        "is_read_probability": 0.25,
        "age_range": (0, 7),
    },
    # -----------------------------------------------------------------------
    # GOOGLE DOCS / CALENDAR  (~5 templates)
    # -----------------------------------------------------------------------
    {
        "sender_name": "Marco Rossi (via Google Docs)",
        "sender_email": "comments-noreply@docs.google.com",
        "subject": "Marco Rossi suggested edits in \"BenchFlow Series A Deck\"",
        "body_plain": (
            "Marco Rossi suggested edits in BenchFlow Series A Deck\n\n"
            "Slide 7 - Market Size:\n"
            "Suggestion: Replace \"$4.2B TAM\" with \"$6.1B TAM\" - the Gartner "
            "report from Jan 2026 has updated numbers.\n\n"
            "Slide 12 - Team:\n"
            "Suggestion: Add Danielle's ML background from her Meta tenure.\n\n"
            "Open document: https://docs.google.com/presentation/d/1abc123/edit\n\n"
            "-- \n"
            "Google Docs: Create and edit documents online.\n"
            "Google LLC, 1600 Amphitheatre Parkway, Mountain View, CA 94043"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.85,
        "age_range": (0, 7),
    },
    {
        "sender_name": "Danielle Martinez (via Google Docs)",
        "sender_email": "comments-noreply@docs.google.com",
        "subject": "Danielle Martinez commented in \"Q1 2026 OKRs\"",
        "body_plain": (
            "Danielle Martinez commented in Q1 2026 OKRs\n\n"
            "On the text: \"Launch SDK v0.5 by Feb 28\"\n"
            "Comment: \"This is going to slip to March 7. The retry middleware "
            "PR is still in review and we found a connection pool leak. Want me to "
            "update the timeline?\"\n\n"
            "Reply or resolve this comment in the document:\n"
            "https://docs.google.com/document/d/1def456/edit?disco=comment123\n\n"
            "-- \n"
            "Google Docs: Create and edit documents online.\n"
            "Google LLC, 1600 Amphitheatre Parkway, Mountain View, CA 94043"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.9,
        "age_range": (0, 5),
    },
    {
        "sender_name": "Google Calendar",
        "sender_email": "calendar-notification@google.com",
        "subject": "Reminder: Board Meeting in 1 hour",
        "body_plain": (
            "Reminder\n\n"
            "Board Meeting\n"
            "When: Thursday, March 6, 2026 at 3:00 PM - 4:30 PM PST\n"
            "Where: Google Meet - https://meet.google.com/xyz-abcd-efg\n"
            "Calendar: alex@benchflow.dev\n\n"
            "Attendees:\n"
            "- Alex Thompson (organizer)\n"
            "- David Park (Gradient Ventures)\n"
            "- Maria Santos (Board Observer)\n"
            "- Marco Rossi (BenchFlow)\n\n"
            "Agenda doc: https://docs.google.com/document/d/board-q1-2026\n\n"
            "-- Google Calendar"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.95,
        "age_range": (0, 3),
    },
    {
        "sender_name": "Google Calendar",
        "sender_email": "calendar-notification@google.com",
        "subject": "Invitation: Technical Deep Dive - BenchFlow x TechCorp @ Mon Mar 10",
        "body_plain": (
            "You've been invited to:\n\n"
            "Technical Deep Dive - BenchFlow x TechCorp\n"
            "When: Monday, March 10, 2026 at 2:00 PM - 3:00 PM PST\n"
            "Where: Google Meet\n"
            "Who:\n"
            "  - Sarah Kim (sarah.kim@techcorp.io) - organizer\n"
            "  - Wei Chen (wei.chen@techcorp.io)\n"
            "  - Alex Thompson (alex@benchflow.dev)\n"
            "  - Danielle Martinez (danielle@benchflow.dev)\n\n"
            "Note from organizer: Follow-up from last week's product demo. "
            "Wei will have questions about monorepo support and custom runners.\n\n"
            "Going? Yes - Maybe - No\n"
            "https://calendar.google.com/calendar/event?action=RESPOND\n\n"
            "-- Google Calendar"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.9,
        "age_range": (0, 7),
    },
    {
        "sender_name": "Amy Liu (via Google Docs)",
        "sender_email": "comments-noreply@docs.google.com",
        "subject": "Amy Liu commented in \"BenchFlow - Customer Interview Notes\"",
        "body_plain": (
            "Amy Liu commented in BenchFlow - Customer Interview Notes\n\n"
            "On the text: \"Customer wants Datadog integration\"\n"
            "Comment: \"This is the third customer this month asking for Datadog. "
            "Should we prioritize this over the Grafana integration? I can put "
            "together a quick effort estimate.\"\n\n"
            "Reply or resolve this comment in the document:\n"
            "https://docs.google.com/document/d/1ghi789/edit?disco=comment456\n\n"
            "-- \n"
            "Google Docs: Create and edit documents online.\n"
            "Google LLC, 1600 Amphitheatre Parkway, Mountain View, CA 94043"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.8,
        "age_range": (0, 10),
    },
    # -----------------------------------------------------------------------
    # DEVOPS — VERCEL, PAGERDUTY, SENTRY, AWS  (~5 templates)
    # -----------------------------------------------------------------------
    {
        "sender_name": "Vercel",
        "sender_email": "notifications@vercel.com",
        "subject": "Deployment failed: benchflow-docs (main)",
        "body_plain": (
            "Deployment Failed\n\n"
            "Project: benchflow-docs\n"
            "Branch: main\n"
            "Commit: Update API reference for v0.5\n"
            "Author: alex-benchflow\n\n"
            "Error:\n"
            "  Build error: Module not found: Can't resolve '@/components/APIRef'\n"
            "  at pages/docs/api/streaming.mdx:3:1\n\n"
            "Build duration: 34s\n"
            "Framework: Next.js\n\n"
            "View build logs: https://vercel.com/benchflow/benchflow-docs/"
            "deployments/dpl_abc123\n\n"
            "-- Vercel"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.85,
        "age_range": (0, 5),
    },
    {
        "sender_name": "Vercel",
        "sender_email": "notifications@vercel.com",
        "subject": "Deployment successful: benchflow-app (main)",
        "body_plain": (
            "Deployment Successful\n\n"
            "Project: benchflow-app\n"
            "Branch: main\n"
            "Commit: Fix dashboard loading state\n"
            "Author: danielle-benchflow\n\n"
            "URL: https://benchflow-app.vercel.app\n"
            "Production: https://app.benchflow.dev\n\n"
            "Build duration: 1m 12s\n"
            "Framework: Next.js\n"
            "Region: sfo1\n\n"
            "View deployment: https://vercel.com/benchflow/benchflow-app/"
            "deployments/dpl_def456\n\n"
            "-- Vercel"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.4,
        "age_range": (0, 7),
    },
    {
        "sender_name": "PagerDuty",
        "sender_email": "no-reply@pagerduty.com",
        "subject": "[TRIGGERED] API latency > 2s on benchflow-prod",
        "body_plain": (
            "INCIDENT TRIGGERED\n\n"
            "Service: benchflow-prod-api\n"
            "Alert: API latency exceeds threshold\n"
            "Severity: HIGH\n"
            "Triggered at: 2026-03-02 03:14 UTC\n\n"
            "Details:\n"
            "  p99 latency: 3,412ms (threshold: 2,000ms)\n"
            "  Affected endpoints: /api/v1/runs, /api/v1/results\n"
            "  Error rate: 2.1% (normal: < 0.5%)\n\n"
            "On-call: Alex Thompson\n\n"
            "Acknowledge: https://app.pagerduty.com/incidents/P1234ABC\n"
            "Resolve: https://app.pagerduty.com/incidents/P1234ABC/resolve\n\n"
            "Escalation policy: BenchFlow Engineering (15 min to ack)\n\n"
            "-- PagerDuty"
        ),
        "body_html": (
            "<div style='font-family: -apple-system, sans-serif; max-width: 560px;'>"
            "<div style='background:#dc2626;color:#fff;padding:12px 16px;border-radius:6px 6px 0 0;'>"
            "<strong>INCIDENT TRIGGERED</strong></div>"
            "<div style='border:1px solid #e5e7eb;border-top:none;padding:16px;border-radius:0 0 6px 6px;'>"
            "<p><strong>Service:</strong> benchflow-prod-api</p>"
            "<p><strong>Alert:</strong> API latency exceeds threshold</p>"
            "<p><strong>Severity:</strong> <span style='color:#dc2626;'>HIGH</span></p>"
            "<p><strong>p99 latency:</strong> 3,412ms (threshold: 2,000ms)</p>"
            "<p><strong>On-call:</strong> Alex Thompson</p>"
            "<a href='https://app.pagerduty.com/incidents/P1234ABC' "
            "style='display:inline-block;padding:10px 20px;background:#dc2626;"
            "color:#fff;border-radius:6px;text-decoration:none;margin-top:8px;'>Acknowledge</a>"
            "</div></div>"
        ),
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.95,
        "age_range": (0, 3),
    },
    {
        "sender_name": "Sentry",
        "sender_email": "noreply@sentry.io",
        "subject": "New issue: KeyError in /api/v1/runs (benchflow-api)",
        "body_plain": (
            "New Issue in benchflow-api\n\n"
            "KeyError: 'metadata'\n"
            "  File: app/api/routes/runs.py, line 142\n"
            "  Function: create_run\n\n"
            "    result = payload['metadata']['runner_config']\n"
            "    KeyError: 'metadata'\n\n"
            "First seen: 1 hour ago\n"
            "Events: 23\n"
            "Users affected: 4\n\n"
            "Tags:\n"
            "  environment: production\n"
            "  release: 0.4.3\n"
            "  server_name: api-prod-2\n\n"
            "View issue: https://benchflow.sentry.io/issues/BENCHFLOW-1847/\n\n"
            "-- Sentry"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.9,
        "age_range": (0, 5),
    },
    {
        "sender_name": "Amazon Web Services",
        "sender_email": "no-reply@amazonaws.com",
        "subject": "AWS Billing Alert: BenchFlow account estimated charges exceed $2,000",
        "body_plain": (
            "AWS Billing Alert\n\n"
            "Account: BenchFlow, Inc. (1234-5678-9012)\n\n"
            "Your estimated charges for the current billing period (March 2026) "
            "have exceeded your alert threshold.\n\n"
            "Estimated charges: $2,147.33\n"
            "Alert threshold: $2,000.00\n\n"
            "Top services by cost:\n"
            "  Amazon EC2: $892.41\n"
            "  Amazon RDS: $534.22\n"
            "  Amazon S3: $312.88\n"
            "  Amazon CloudFront: $198.42\n"
            "  Other: $209.40\n\n"
            "Review your usage: https://console.aws.amazon.com/billing/\n\n"
            "This alert was configured in AWS Budgets. "
            "Manage alerts: https://console.aws.amazon.com/billing/home#/budgets\n\n"
            "-- Amazon Web Services"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.75,
        "age_range": (0, 5),
    },
    # -----------------------------------------------------------------------
    # SLACK  (~3 templates)
    # -----------------------------------------------------------------------
    {
        "sender_name": "Slack",
        "sender_email": "notification@slack.com",
        "subject": "Digest: 14 unread messages in #engineering",
        "body_plain": (
            "You have 14 unread messages in BenchFlow Slack\n\n"
            "#engineering (8 messages)\n"
            "  Marco: The k8s migration is done. New cluster is live on "
            "us-west-2. Runbook in Notion.\n"
            "  Danielle: Retry middleware PR is ready for final review.\n"
            "  Amy: Anyone else seeing flaky tests in test_streaming?\n"
            "  + 5 more messages\n\n"
            "#general (4 messages)\n"
            "  Alex: Team lunch tomorrow at Souvla, 12:30 PM. Who's in?\n"
            "  + 3 more messages\n\n"
            "#sales (2 messages)\n"
            "  Amy: TechCorp wants to move forward with a pilot. Need pricing.\n"
            "  + 1 more message\n\n"
            "Catch up: https://benchflow.slack.com\n\n"
            "-- Slack"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.45,
        "age_range": (0, 3),
    },
    {
        "sender_name": "Slack",
        "sender_email": "notification@slack.com",
        "subject": "New DM from Marco Rossi in BenchFlow",
        "body_plain": (
            "Marco Rossi sent you a direct message:\n\n"
            "\"Hey, can you review the infra PR when you get a chance? It's the "
            "Terraform changes for the new staging environment. Want to get it "
            "merged before the TechCorp deep dive on Monday.\"\n\n"
            "Reply in Slack: https://benchflow.slack.com/archives/D04ABC123\n\n"
            "-- Slack\n\n"
            "You're receiving this email because you have email notifications "
            "enabled for direct messages."
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.7,
        "age_range": (0, 5),
    },
    {
        "sender_name": "Slack",
        "sender_email": "notification@slack.com",
        "subject": "Digest: 6 unread messages in #product",
        "body_plain": (
            "You have 6 unread messages in BenchFlow Slack\n\n"
            "#product (6 messages)\n"
            "  Amy: Customer feedback roundup from this week - "
            "3 customers asked for Datadog integration, 2 want better "
            "GitHub Actions support.\n"
            "  Danielle: I can prototype the Datadog integration in a day.\n"
            "  Marco: Let's discuss in the Friday product sync.\n"
            "  + 3 more messages\n\n"
            "Catch up: https://benchflow.slack.com\n\n"
            "-- Slack"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.4,
        "age_range": (0, 5),
    },
    # -----------------------------------------------------------------------
    # MISC  (~3 templates)
    # -----------------------------------------------------------------------
    {
        "sender_name": "Stripe",
        "sender_email": "notifications@stripe.com",
        "subject": "Your weekly Stripe summary",
        "body_plain": (
            "Weekly Summary for BenchFlow, Inc.\n"
            "Feb 24 - Mar 2, 2026\n\n"
            "Gross volume: $8,200.00\n"
            "  Successful payments: 12\n"
            "  Failed payments: 1\n"
            "  Refunds: $0.00\n\n"
            "New customers: 3\n"
            "Active subscriptions: 8\n"
            "  Pro plan ($499/mo): 5\n"
            "  Team plan ($199/mo): 3\n\n"
            "MRR: $3,092.00 (+$499.00 from last week)\n\n"
            "Next payout: $7,954.00 on March 4\n\n"
            "View dashboard: https://dashboard.stripe.com\n\n"
            "-- Stripe"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.75,
        "age_range": (0, 7),
    },
    {
        "sender_name": "Cloudflare",
        "sender_email": "noreply@notify.cloudflare.com",
        "subject": "Security alert: Unusual traffic detected on benchflow.dev",
        "body_plain": (
            "Security Alert for benchflow.dev\n\n"
            "We detected unusual traffic patterns on your domain.\n\n"
            "Zone: benchflow.dev\n"
            "Time: March 1, 2026, 18:42 UTC\n"
            "Duration: ~45 minutes\n\n"
            "Details:\n"
            "  Requests blocked by WAF: 12,847\n"
            "  Attack type: HTTP flood (Layer 7)\n"
            "  Top source: AS13335 (multiple IPs)\n"
            "  Target: /api/v1/runs (POST)\n\n"
            "All malicious requests were blocked. No action is required, but "
            "you may want to review your rate limiting rules.\n\n"
            "Review: https://dash.cloudflare.com/benchflow/security\n\n"
            "-- Cloudflare"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.85,
        "age_range": (0, 7),
    },
    {
        "sender_name": "Product Hunt",
        "sender_email": "noreply@producthunt.com",
        "subject": "You've been invited to launch BenchFlow on Product Hunt",
        "body_plain": (
            "Hi Alex,\n\n"
            "Chris Messina has invited you to launch BenchFlow on Product Hunt!\n\n"
            "Chris said: \"BenchFlow is a great fit for the developer tools "
            "audience on PH. I'd be happy to hunt it for you if you're interested. "
            "Tuesday or Thursday launches tend to perform best.\"\n\n"
            "Getting started:\n"
            "1. Claim your product page: https://producthunt.com/posts/benchflow/edit\n"
            "2. Add a tagline, description, and screenshots\n"
            "3. Pick a launch date\n\n"
            "Tips for a successful launch:\n"
            "- Prepare a 1-min demo video\n"
            "- Have your team ready to answer comments\n"
            "- Share with your community early in the day (12:01 AM PT)\n\n"
            "Good luck!\n"
            "-- Product Hunt Team"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.8,
        "age_range": (3, 21),
    },
]
