"""Handwritten realistic email content for the default seed scenario.

Persona: Alex Thompson, Senior Software Engineer at Acme Corp.
All links point to real, publicly accessible URLs.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Named personas
# ---------------------------------------------------------------------------
PERSONAS = {
    "sarah": {"name": "Sarah Chen", "email": "sarah.chen@acmecorp.com", "role": "manager"},
    "marcus": {"name": "Marcus Johnson", "email": "marcus.johnson@acmecorp.com", "role": "coworker"},
    "priya": {"name": "Priya Patel", "email": "priya.patel@acmecorp.com", "role": "coworker"},
    "james": {"name": "James Wilson", "email": "james.wilson@acmecorp.com", "role": "coworker"},
    "emily": {"name": "Emily Rodriguez", "email": "emily.rodriguez@acmecorp.com", "role": "coworker"},
    "david": {"name": "David Kim", "email": "david.kim@acmecorp.com", "role": "coworker"},
    "lisa": {"name": "Lisa Wang", "email": "lisa.wang@acmecorp.com", "role": "hr"},
    "rachel": {"name": "Rachel Foster", "email": "rachel.foster@acmecorp.com", "role": "coworker"},
    "mike": {"name": "Mike Chen", "email": "mike.chen@gmail.com", "role": "friend"},
    "jenny": {"name": "Jenny Park", "email": "jenny.park@gmail.com", "role": "friend"},
    "tom": {"name": "Tom Harris", "email": "tom.harris@vendorco.com", "role": "vendor"},
    "nina": {"name": "Nina Sharma", "email": "nina.sharma@acmecorp.com", "role": "coworker"},
}

USER_EMAIL = "me@example.com"
USER_NAME = "Alex Thompson"

# ---------------------------------------------------------------------------
# Multi-message work threads
# ---------------------------------------------------------------------------
THREADS: list[dict] = [
    # --- Thread 1: Q1 Sprint Planning ---
    {
        "subject": "Q1 Sprint Planning - Backend Team",
        "labels": ["INBOX"],
        "messages": [
            {
                "sender": "sarah",
                "to": "marcus.johnson@acmecorp.com, {user}, priya.patel@acmecorp.com",
                "body_plain": (
                    "Hi team,\n\n"
                    "I've put together the agenda for our Q1 sprint planning on Thursday at 2pm.\n\n"
                    "1. Review Q4 velocity and burndown\n"
                    "2. Discuss carry-over stories from last sprint\n"
                    "3. Estimate new backlog items\n"
                    "4. Assign sprint goals\n\n"
                    "Please come prepared with your capacity estimates for the quarter. If you have any "
                    "items you'd like to add to the agenda, reply to this thread by EOD Wednesday.\n\n"
                    "I've shared the board here: https://github.com/orgs/community/discussions\n\n"
                    "Thanks,\nSarah"
                ),
                "is_read": True,
                "minutes_offset": 0,
            },
            {
                "sender": "{user}",
                "to": "sarah.chen@acmecorp.com, marcus.johnson@acmecorp.com, priya.patel@acmecorp.com",
                "body_plain": (
                    "Thanks Sarah,\n\n"
                    "My capacity estimate for Q1:\n"
                    "- I'll be out the week of Feb 17 for vacation\n"
                    "- Otherwise at full capacity (~8 story points/sprint)\n"
                    "- I can take on the auth service migration if we prioritize it\n\n"
                    "One agenda item to add: we should discuss the API rate limiting strategy. "
                    "We've been hitting throttling issues in staging. I've been reading the "
                    "NGINX rate limiting docs for reference:\n"
                    "https://www.nginx.com/blog/rate-limiting-nginx/\n\n"
                    "Alex"
                ),
                "is_read": True,
                "is_sent": True,
                "minutes_offset": 95,
            },
            {
                "sender": "marcus",
                "to": "sarah.chen@acmecorp.com, {user}, priya.patel@acmecorp.com",
                "body_plain": (
                    "Hey all,\n\n"
                    "Quick heads up — I want to bring up tech debt during the planning session. "
                    "We've been accumulating TODOs in the payment processing module and I'm worried "
                    "about reliability if we don't address them soon.\n\n"
                    "Specifically:\n"
                    "- The retry logic needs refactoring (currently no exponential backoff)\n"
                    "- We have 3 deprecated Stripe API calls that need updating per their migration guide:\n"
                    "  https://docs.stripe.com/upgrades\n"
                    "- Test coverage for edge cases is at 45%\n\n"
                    "Can we allocate at least 20% of sprint capacity for tech debt?\n\n"
                    "Marcus"
                ),
                "is_read": True,
                "minutes_offset": 180,
            },
            {
                "sender": "sarah",
                "to": "marcus.johnson@acmecorp.com, {user}, priya.patel@acmecorp.com",
                "body_plain": (
                    "Good points from both of you. I'll add API rate limiting and tech debt to the agenda.\n\n"
                    "Marcus — 20% for tech debt sounds reasonable. Let's discuss the exact allocation "
                    "on Thursday.\n\n"
                    "Alex — noted on your vacation week. I'll adjust the sprint velocity calculation.\n\n"
                    "See everyone Thursday!\nSarah"
                ),
                "is_read": False,
                "minutes_offset": 240,
            },
        ],
    },
    # --- Thread 2: PR Review ---
    {
        "subject": "RE: Code review needed - FastAPI auth middleware refactor PR #342",
        "labels": ["INBOX"],
        "messages": [
            {
                "sender": "priya",
                "to": "{user}",
                "body_plain": (
                    "Hey Alex,\n\n"
                    "I've opened PR #342 for the auth service refactor. It's a big one (~800 lines) "
                    "but most of it is moving existing code into the new module structure.\n\n"
                    "Key changes:\n"
                    "- Split monolithic auth.py into auth/tokens.py, auth/sessions.py, auth/middleware.py\n"
                    "- Added refresh token rotation based on the OAuth 2.0 spec:\n"
                    "  https://datatracker.ietf.org/doc/html/rfc6749\n"
                    "- Updated all tests to use the new imports\n\n"
                    "I used the FastAPI security patterns from:\n"
                    "https://fastapi.tiangolo.com/tutorial/security/\n\n"
                    "Would appreciate your review since you wrote a lot of the original auth code. "
                    "No rush — by end of week is fine.\n\n"
                    "Thanks!\nPriya"
                ),
                "is_read": True,
                "is_starred": True,
                "minutes_offset": 0,
            },
            {
                "sender": "{user}",
                "to": "priya.patel@acmecorp.com",
                "body_plain": (
                    "Priya,\n\n"
                    "Took a first pass — looks great overall! Left a few comments:\n\n"
                    "1. The token rotation logic in tokens.py L145-160 — I think we need to handle "
                    "the race condition where two requests try to refresh simultaneously. We hit this "
                    "in production last quarter. The python-jose docs cover thread safety here:\n"
                    "https://github.com/mpdavis/python-jose\n\n"
                    "2. Minor: the import in middleware.py L23 should use relative imports to stay "
                    "consistent with the rest of the codebase.\n\n"
                    "3. Love the separation of concerns. The sessions module is much cleaner now.\n\n"
                    "Happy to approve once the race condition fix is in.\n\n"
                    "Alex"
                ),
                "is_read": True,
                "is_sent": True,
                "minutes_offset": 300,
            },
            {
                "sender": "priya",
                "to": "{user}",
                "body_plain": (
                    "Great catch on the race condition! I added a distributed lock using Redis "
                    "based on the Redlock algorithm:\n"
                    "https://redis.io/docs/latest/develop/use/patterns/distributed-locks/\n\n"
                    "Also fixed the import style.\n\n"
                    "Can you take another look when you get a chance?\n\n"
                    "Priya"
                ),
                "is_read": False,
                "minutes_offset": 480,
            },
        ],
    },
    # --- Thread 3: Production incident ---
    {
        "subject": "[INCIDENT] Elevated error rates on checkout API",
        "labels": ["INBOX", "IMPORTANT"],
        "messages": [
            {
                "sender": "james",
                "to": "backend-team@acmecorp.com",
                "body_plain": (
                    "INCIDENT REPORT\n"
                    "Severity: P2\n"
                    "Time: Today 14:32 UTC\n\n"
                    "We're seeing elevated 500 errors on the /api/v2/checkout endpoint. "
                    "Error rate jumped from 0.1% to 4.7% in the last 15 minutes.\n\n"
                    "Preliminary investigation points to the database connection pool being exhausted. "
                    "The recent deployment (v2.14.3) may have introduced a connection leak.\n\n"
                    "SQLAlchemy connection pool docs for reference:\n"
                    "https://docs.sqlalchemy.org/en/20/core/pooling.html\n\n"
                    "I'm rolling back to v2.14.2 now. Will update once the rollback is complete.\n\n"
                    "James"
                ),
                "is_read": True,
                "is_starred": True,
                "minutes_offset": 0,
            },
            {
                "sender": "{user}",
                "to": "backend-team@acmecorp.com",
                "body_plain": (
                    "I think I found the issue. In the v2.14.3 diff, there's a new database query in "
                    "the inventory check that opens a connection but doesn't release it if the product "
                    "is out of stock (the early return path skips the finally block).\n\n"
                    "This is the exact pattern described in the SQLAlchemy FAQ:\n"
                    "https://docs.sqlalchemy.org/en/20/faq/connections.html\n\n"
                    "File: services/inventory.py, line 89\n\n"
                    "I'll have a fix ready as soon as the rollback stabilizes things.\n\n"
                    "Alex"
                ),
                "is_read": True,
                "is_sent": True,
                "minutes_offset": 12,
            },
            {
                "sender": "james",
                "to": "backend-team@acmecorp.com",
                "body_plain": (
                    "Rollback complete. Error rates are back to normal (0.08%).\n\n"
                    "Alex — that's exactly the issue. Nice find. Let's get the fix into a hotfix "
                    "branch and deploy after we add a regression test.\n\n"
                    "I'll schedule a post-mortem for tomorrow at 11am.\n\n"
                    "James"
                ),
                "is_read": True,
                "minutes_offset": 25,
            },
        ],
    },
    # --- Thread 4: 1:1 follow-up ---
    {
        "subject": "1:1 Follow-up - Career growth discussion",
        "labels": ["INBOX"],
        "messages": [
            {
                "sender": "sarah",
                "to": "{user}",
                "body_plain": (
                    "Hi Alex,\n\n"
                    "Great chat today in our 1:1. As discussed, here's a summary of the action items:\n\n"
                    "1. I'll nominate you for the Staff Engineer track — the committee meets in March\n"
                    "2. You'll put together a tech talk proposal for the engineering all-hands\n"
                    "3. We'll revisit the team lead opportunity for the new payments team in Q2\n\n"
                    "I think you're in a really good position for the Staff promotion. The auth service "
                    "migration and the observability improvements you drove last quarter are exactly "
                    "the kind of impact they look for.\n\n"
                    "This StaffEng guide is a great resource for framing your narrative:\n"
                    "https://staffeng.com/guides/getting-the-title-where-you-are/\n\n"
                    "Let me know if you want to do a practice run of the tech talk.\n\n"
                    "Sarah"
                ),
                "is_read": True,
                "is_starred": True,
                "minutes_offset": 0,
            },
            {
                "sender": "{user}",
                "to": "sarah.chen@acmecorp.com",
                "body_plain": (
                    "Thanks Sarah! Really appreciate the support.\n\n"
                    "I'll start drafting the tech talk proposal this week — thinking about doing it "
                    "on \"Building Resilient Microservices: Lessons from our Auth Migration.\"\n\n"
                    "Found some good inspiration from these talks:\n"
                    "- https://www.youtube.com/watch?v=lduRMow1GY0 (Netflix microservices)\n"
                    "- https://martinfowler.com/articles/microservices.html\n\n"
                    "Would love to do a dry run with you before the all-hands. How about the week of "
                    "March 10?\n\n"
                    "Alex"
                ),
                "is_read": True,
                "is_sent": True,
                "minutes_offset": 60,
            },
        ],
    },
    # --- Thread 5: Vendor integration ---
    {
        "subject": "RE: Acme Corp - Twilio SendGrid API Integration Timeline",
        "labels": ["INBOX"],
        "messages": [
            {
                "sender": "tom",
                "to": "{user}",
                "cc": "sarah.chen@acmecorp.com",
                "body_plain": (
                    "Hi Alex,\n\n"
                    "Following up on our call last week. Here's the updated integration timeline "
                    "for the SendGrid email API:\n\n"
                    "Phase 1 (Feb): Sandbox environment setup + credential exchange\n"
                    "Phase 2 (Mar): Core transactional email implementation\n"
                    "Phase 3 (Apr): Error handling, retries, monitoring\n"
                    "Phase 4 (May): Production rollout\n\n"
                    "API documentation: https://docs.sendgrid.com/api-reference\n"
                    "Python SDK: https://github.com/sendgrid/sendgrid-python\n\n"
                    "Let me know if the timeline works for your team.\n\n"
                    "Best,\nTom Harris\nSolutions Engineer, VendorCo"
                ),
                "is_read": True,
                "minutes_offset": 0,
            },
            {
                "sender": "{user}",
                "to": "tom.harris@vendorco.com",
                "cc": "sarah.chen@acmecorp.com",
                "body_plain": (
                    "Hi Tom,\n\n"
                    "Timeline looks good. A few questions:\n\n"
                    "1. What's the rate limit on the sandbox environment? I see the docs mention\n"
                    "   limits here: https://docs.sendgrid.com/api-reference/how-to-use-the-sendgrid-v3-api/rate-limits\n"
                    "2. Do you support webhook notifications for delivery events?\n"
                    "   https://docs.sendgrid.com/for-developers/tracking-events/event\n"
                    "3. Is there a staging environment between sandbox and production?\n\n"
                    "I'll have our DevOps team start setting up the sandbox environment this week.\n\n"
                    "Thanks,\nAlex"
                ),
                "is_read": True,
                "is_sent": True,
                "minutes_offset": 120,
            },
        ],
    },
    # --- Thread 6: Team lunch planning ---
    {
        "subject": "Team lunch Friday - suggestions?",
        "labels": ["INBOX"],
        "messages": [
            {
                "sender": "emily",
                "to": "backend-team@acmecorp.com",
                "body_plain": (
                    "Hey everyone!\n\n"
                    "It's my turn to organize team lunch this Friday. Any preferences?\n\n"
                    "Options so far:\n"
                    "1. Sushi Palace (we went last month though)\n"
                    "2. That new Thai place on Market St\n"
                    "3. Mediterranean Kitchen\n\n"
                    "I found some good options on Yelp:\n"
                    "https://www.yelp.com/search?find_desc=lunch&find_loc=San+Francisco%2C+CA\n\n"
                    "Reply with your vote or suggest something else!\n\n"
                    "Emily"
                ),
                "is_read": True,
                "minutes_offset": 0,
            },
            {
                "sender": "marcus",
                "to": "backend-team@acmecorp.com",
                "body_plain": "Thai place gets my vote! I've heard great things about their pad see ew.\n\nMarcus",
                "is_read": True,
                "minutes_offset": 15,
            },
            {
                "sender": "{user}",
                "to": "backend-team@acmecorp.com",
                "body_plain": "+1 for Thai! I'm in.\n\nAlex",
                "is_read": True,
                "is_sent": True,
                "minutes_offset": 22,
            },
            {
                "sender": "emily",
                "to": "backend-team@acmecorp.com",
                "body_plain": (
                    "Thai it is! I'll make a reservation for 12:30pm.\n\n"
                    "See you all Friday!\nEmily"
                ),
                "is_read": False,
                "minutes_offset": 45,
            },
        ],
    },
    # --- Thread 7: Database migration discussion ---
    {
        "subject": "PostgreSQL 16 upgrade plan",
        "labels": ["INBOX"],
        "messages": [
            {
                "sender": "david",
                "to": "{user}, james.wilson@acmecorp.com",
                "body_plain": (
                    "Hey Alex, James,\n\n"
                    "I've been looking into upgrading our PostgreSQL cluster from 14 to 16. "
                    "Key benefits from the release notes:\n"
                    "https://www.postgresql.org/docs/16/release-16.html\n\n"
                    "- Logical replication improvements (we need this for the read replica setup)\n"
                    "- Better query parallelism\n"
                    "- The new MERGE command would simplify our upsert logic:\n"
                    "  https://www.postgresql.org/docs/16/sql-merge.html\n\n"
                    "Proposed approach:\n"
                    "1. Set up PG16 replica alongside existing PG14\n"
                    "2. Run both in parallel for 2 weeks with shadow traffic\n"
                    "3. Cut over during the maintenance window\n\n"
                    "Upgrade guide: https://www.postgresql.org/docs/16/pgupgrade.html\n\n"
                    "Thoughts? I can put together a more detailed RFC if you think this is worth pursuing.\n\n"
                    "David"
                ),
                "is_read": True,
                "minutes_offset": 0,
            },
            {
                "sender": "{user}",
                "to": "david.kim@acmecorp.com, james.wilson@acmecorp.com",
                "body_plain": (
                    "David,\n\n"
                    "Definitely worth pursuing. The MERGE command alone would let us clean up "
                    "about 200 lines of awkward INSERT ... ON CONFLICT logic.\n\n"
                    "One thing to watch out for: we have some custom extensions (pg_trgm, "
                    "pg_stat_statements) that might need compatibility testing.\n"
                    "Extension compatibility notes: https://www.postgresql.org/docs/16/contrib.html\n\n"
                    "Let's add this to the Q2 roadmap. An RFC would be great — can you have "
                    "a draft ready for the architecture review on March 15?\n\n"
                    "Alex"
                ),
                "is_read": True,
                "is_sent": True,
                "minutes_offset": 90,
            },
        ],
    },
    # --- Thread 8: Onboarding new hire ---
    {
        "subject": "Onboarding buddy for new hire - Rachel Foster",
        "labels": ["INBOX"],
        "messages": [
            {
                "sender": "lisa",
                "to": "{user}",
                "body_plain": (
                    "Hi Alex,\n\n"
                    "We have a new backend engineer joining the team on Monday — Rachel Foster. "
                    "She's coming from Stripe and will be working on the payments platform.\n\n"
                    "Would you be willing to be her onboarding buddy for the first two weeks? "
                    "This would involve:\n"
                    "- Helping her set up her dev environment\n"
                    "- Walking through the codebase architecture\n"
                    "- Being available for questions\n"
                    "- Introducing her to the team\n\n"
                    "Let me know if you're up for it!\n\n"
                    "Thanks,\nLisa Wang\nPeople Operations"
                ),
                "is_read": True,
                "minutes_offset": 0,
            },
            {
                "sender": "{user}",
                "to": "lisa.wang@acmecorp.com",
                "body_plain": (
                    "Hi Lisa,\n\n"
                    "Happy to be Rachel's onboarding buddy! Coming from Stripe, she'll probably "
                    "pick things up quickly.\n\n"
                    "I'll block out some time on Monday morning to help with setup. Can you "
                    "send me her GitHub username so I can add her to our repos beforehand?\n\n"
                    "Alex"
                ),
                "is_read": True,
                "is_sent": True,
                "minutes_offset": 45,
            },
            {
                "sender": "lisa",
                "to": "{user}",
                "body_plain": (
                    "Awesome, thank you Alex! Her GitHub is @rachelfoster.\n\n"
                    "I'll send you both a calendar invite for Monday 9:30am.\n\n"
                    "Lisa"
                ),
                "is_read": True,
                "minutes_offset": 60,
            },
        ],
    },
    # --- Thread 9: Architecture decision ---
    {
        "subject": "RFC: Event-driven architecture for notifications",
        "labels": ["INBOX"],
        "messages": [
            {
                "sender": "{user}",
                "to": "backend-team@acmecorp.com",
                "body_plain": (
                    "Team,\n\n"
                    "I've drafted an RFC for migrating our notification system from synchronous "
                    "HTTP calls to an event-driven architecture using Apache Kafka.\n\n"
                    "TL;DR:\n"
                    "- Current system: direct HTTP calls to notification service (200ms p99 latency added)\n"
                    "- Proposed: publish events to Kafka, notification service consumes asynchronously\n"
                    "- Expected improvement: reduce checkout latency by ~180ms\n\n"
                    "Key references:\n"
                    "- Kafka docs: https://kafka.apache.org/documentation/\n"
                    "- Confluent Python client: https://github.com/confluentinc/confluent-kafka-python\n"
                    "- Event-driven patterns: https://martinfowler.com/articles/201701-event-driven.html\n\n"
                    "Please review and leave comments by Friday. I'd like to present this at the "
                    "architecture review next week.\n\n"
                    "Alex"
                ),
                "is_read": True,
                "is_sent": True,
                "minutes_offset": 0,
            },
            {
                "sender": "nina",
                "to": "backend-team@acmecorp.com",
                "body_plain": (
                    "Alex, this looks solid. A few questions:\n\n"
                    "1. Have we considered RabbitMQ instead of Kafka? Our volume might not "
                    "justify Kafka's operational complexity.\n"
                    "   Comparison: https://www.rabbitmq.com/tutorials\n\n"
                    "2. How do we handle the case where a notification must be sent before "
                    "confirming the order (e.g., payment confirmation email)?\n\n"
                    "3. What's the fallback if Kafka is down? Do we queue locally?\n"
                    "   Relevant: https://engineering.linkedin.com/kafka/running-kafka-scale\n\n"
                    "Left more detailed comments in the doc.\n\n"
                    "Nina"
                ),
                "is_read": False,
                "minutes_offset": 240,
            },
        ],
    },
    # --- Thread 10: CI/CD pipeline issue ---
    {
        "subject": "CI build times have doubled - investigation needed",
        "labels": ["INBOX"],
        "messages": [
            {
                "sender": "rachel",
                "to": "{user}, david.kim@acmecorp.com",
                "body_plain": (
                    "Hi Alex, David,\n\n"
                    "I noticed our CI pipeline is taking 18-22 minutes now, up from ~10 minutes "
                    "a month ago. This is really hurting our iteration speed.\n\n"
                    "I did some initial investigation:\n"
                    "- The Docker layer caching seems broken after the runner update\n"
                    "- Integration tests are running sequentially instead of in parallel\n"
                    "- We're installing ALL dependencies including dev tools in the test image\n\n"
                    "Docker multi-stage build best practices:\n"
                    "https://docs.docker.com/build/building/multi-stage/\n\n"
                    "Would either of you have time to look into this? I can help with the Docker "
                    "optimization but the test parallelization needs someone who knows the test "
                    "infrastructure better.\n\n"
                    "Rachel"
                ),
                "is_read": True,
                "minutes_offset": 0,
            },
            {
                "sender": "{user}",
                "to": "rachel.foster@acmecorp.com, david.kim@acmecorp.com",
                "body_plain": (
                    "Rachel,\n\n"
                    "I'll look into the test parallelization issue. I think the problem is that "
                    "we switched to a new test runner in v2.14 that doesn't support the --parallel "
                    "flag the same way.\n\n"
                    "Quick wins I can try:\n"
                    "1. Split the test suite into unit and integration stages\n"
                    "2. Use pytest-xdist for parallel execution:\n"
                    "   https://github.com/pytest-dev/pytest-xdist\n"
                    "3. Only run integration tests on the main branch\n\n"
                    "David — can you take the Docker caching issue? This guide might help:\n"
                    "https://docs.docker.com/build/cache/\n\n"
                    "Alex"
                ),
                "is_read": True,
                "is_sent": True,
                "minutes_offset": 35,
            },
        ],
    },
    # --- Thread 11: Standup async ---
    {
        "subject": "Async standup - Feb 26",
        "labels": ["INBOX"],
        "messages": [
            {
                "sender": "sarah",
                "to": "backend-team@acmecorp.com",
                "body_plain": (
                    "Good morning team! Since half the team is in different time zones today, "
                    "let's do async standup. Please reply with your update.\n\n"
                    "Mine:\n"
                    "- Yesterday: Finalized Q1 OKRs, met with product team about roadmap\n"
                    "- Today: Sprint planning prep, 1:1s with Alex and Marcus\n"
                    "- Blockers: None\n\n"
                    "Sarah"
                ),
                "is_read": True,
                "minutes_offset": 0,
            },
            {
                "sender": "{user}",
                "to": "backend-team@acmecorp.com",
                "body_plain": (
                    "Update:\n"
                    "- Yesterday: Fixed connection leak bug (PR #358), reviewed Priya's auth refactor\n"
                    "- Today: Writing regression tests for the connection leak fix, starting RFC for event-driven notifications\n"
                    "- Blockers: Waiting for Tom (VendorCo) to send sandbox API credentials\n\n"
                    "Alex"
                ),
                "is_read": True,
                "is_sent": True,
                "minutes_offset": 30,
            },
            {
                "sender": "marcus",
                "to": "backend-team@acmecorp.com",
                "body_plain": (
                    "Morning:\n"
                    "- Yesterday: Started Stripe API migration (deprecated calls), got 2 of 3 done\n"
                    "  Stripe migration guide: https://docs.stripe.com/upgrades\n"
                    "- Today: Finishing the last Stripe migration, adding retry tests for payment module\n"
                    "- Blockers: Need staging environment access for Stripe v2023-10 testing\n\n"
                    "Marcus"
                ),
                "is_read": True,
                "minutes_offset": 45,
            },
        ],
    },
    # --- Thread 12: Security review ---
    {
        "subject": "Security review findings - Q4 audit",
        "labels": ["INBOX", "IMPORTANT"],
        "messages": [
            {
                "sender": "james",
                "to": "{user}, sarah.chen@acmecorp.com",
                "body_plain": (
                    "Alex, Sarah,\n\n"
                    "The security team completed their Q4 audit of the backend services. "
                    "Here's the summary:\n\n"
                    "Critical (0):\n"
                    "None - great work!\n\n"
                    "High (2):\n"
                    "- SEC-2025-014: JWT tokens don't have a max lifetime cap\n"
                    "  Ref: https://cheatsheetseries.owasp.org/cheatsheets/JSON_Web_Token_for_Java_Cheat_Sheet.html\n"
                    "- SEC-2025-015: SQL injection possible in legacy search endpoint\n"
                    "  Ref: https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html\n\n"
                    "Medium (5):\n"
                    "- Various header security improvements\n"
                    "- Rate limiting gaps on auth endpoints\n"
                    "- Logging PII in debug mode\n\n"
                    "The two high-severity items need to be resolved before the end of Q1.\n"
                    "OWASP Top 10 reference: https://owasp.org/www-project-top-ten/\n\n"
                    "Alex, since you're working on the auth migration, can you pick up SEC-2025-014?\n\n"
                    "James"
                ),
                "is_read": True,
                "is_starred": True,
                "minutes_offset": 0,
            },
            {
                "sender": "{user}",
                "to": "james.wilson@acmecorp.com, sarah.chen@acmecorp.com",
                "body_plain": (
                    "James,\n\n"
                    "I'll take SEC-2025-014. The JWT max lifetime fix is actually something I was "
                    "already planning as part of the auth refactor.\n\n"
                    "For the SQL injection issue (SEC-2025-015), that's in the legacy search code. "
                    "We should prioritize migrating that to the new ORM-based queries. Marcus, "
                    "would that fit into the tech debt allocation we discussed?\n\n"
                    "I'll follow the JWT best practices from:\n"
                    "https://auth0.com/docs/secure/tokens/json-web-tokens\n\n"
                    "I'll have SEC-2025-014 fixed by end of next week.\n\n"
                    "Alex"
                ),
                "is_read": True,
                "is_sent": True,
                "minutes_offset": 60,
            },
        ],
    },
]

# ---------------------------------------------------------------------------
# Standalone notification emails (with HTML bodies)
# ---------------------------------------------------------------------------
NOTIFICATIONS: list[dict] = [
    # --- GitHub: real open-source repos ---
    {
        "sender_name": "GitHub",
        "sender_email": "notifications@github.com",
        "subject": "[fastapi/fastapi] Pull request #12501: Add WebSocket middleware support",
        "labels": ["INBOX", "CATEGORY_UPDATES"],
        "is_read": False,
        "body_plain": (
            "@tiangolo requested your review on:\n"
            "fastapi/fastapi#12501 Add WebSocket middleware support\n\n"
            "This PR adds native WebSocket middleware support to FastAPI, enabling\n"
            "authentication and rate limiting on WebSocket connections.\n"
            "+487 -203 across 12 files\n\n"
            "View it on GitHub:\n"
            "https://github.com/fastapi/fastapi/pulls"
        ),
        "body_html": (
            '<div style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif;max-width:600px;margin:0 auto;color:#24292f">'
            '<div style="border-bottom:1px solid #d0d7de;padding:16px 0">'
            '<a href="https://github.com/fastapi/fastapi" style="font-size:14px;color:#0969da;text-decoration:none">fastapi/fastapi</a></div>'
            '<div style="padding:16px 0">'
            '<p style="margin:0 0 8px"><strong>@tiangolo</strong> requested your review on: '
            '<a href="https://github.com/fastapi/fastapi/pulls" style="color:#0969da;text-decoration:none">'
            '#12501 Add WebSocket middleware support</a></p>'
            '<p style="color:#57606a;margin:0 0 16px;font-size:14px">'
            'This PR adds native WebSocket middleware support to FastAPI, enabling authentication and rate limiting on WebSocket connections.</p>'
            '<div style="background:#f6f8fa;border:1px solid #d0d7de;border-radius:6px;padding:12px;font-size:13px">'
            '<span style="color:#1a7f37;font-weight:600">+487</span> '
            '<span style="color:#cf222e;font-weight:600">-203</span> '
            '<span style="color:#57606a"> across 12 files</span></div>'
            '<div style="margin-top:16px">'
            '<a href="https://github.com/fastapi/fastapi/pulls" '
            'style="display:inline-block;padding:6px 16px;background:#2da44e;color:#fff;border-radius:6px;text-decoration:none;font-size:14px;font-weight:500">'
            'Review changes</a></div></div></div>'
        ),
    },
    {
        "sender_name": "GitHub",
        "sender_email": "notifications@github.com",
        "subject": "[psycopg/psycopg] Issue #892: Connection pool memory leak in async mode",
        "labels": ["INBOX", "CATEGORY_UPDATES"],
        "is_read": True,
        "body_plain": (
            "@dvarrazzo opened an issue:\n"
            "psycopg/psycopg#892 Connection pool memory leak in async mode\n\n"
            "The async connection pool gradually increases memory usage over 24h, "
            "reaching ~4GB before OOM killer terminates it.\n\n"
            "Steps to reproduce:\n"
            "1. Create async pool with min_size=5, max_size=20\n"
            "2. Run sustained load for 24 hours\n"
            "3. Observe linear growth of ~150MB/hour\n\n"
            "View on GitHub: https://github.com/psycopg/psycopg/issues"
        ),
        "body_html": (
            '<div style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif;max-width:600px;margin:0 auto;color:#24292f">'
            '<div style="border-bottom:1px solid #d0d7de;padding:16px 0">'
            '<a href="https://github.com/psycopg/psycopg" style="font-size:14px;color:#0969da;text-decoration:none">psycopg/psycopg</a></div>'
            '<div style="padding:16px 0">'
            '<p style="margin:0 0 8px"><strong>@dvarrazzo</strong> opened an issue: '
            '<a href="https://github.com/psycopg/psycopg/issues" style="color:#0969da;text-decoration:none">'
            '#892 Connection pool memory leak in async mode</a></p>'
            '<p style="color:#57606a;font-size:14px;margin:0 0 12px">'
            'The async connection pool gradually increases memory usage over 24h, '
            'reaching ~4GB before OOM killer terminates it.</p>'
            '<div style="background:#f6f8fa;border:1px solid #d0d7de;border-radius:6px;padding:12px;font-size:13px">'
            '<strong>Steps to reproduce:</strong><br>'
            '1. Create async pool with <code style="background:#eff1f3;padding:2px 6px;border-radius:3px">min_size=5, max_size=20</code><br>'
            '2. Run sustained load for 24 hours<br>'
            '3. Observe linear growth of ~150MB/hour</div></div></div>'
        ),
    },
    # --- Amazon: real products ---
    {
        "sender_name": "Amazon",
        "sender_email": "shipment-tracking@amazon.com",
        "subject": "Your Amazon order has shipped! (#112-4839201-7724563)",
        "labels": ["INBOX", "CATEGORY_UPDATES"],
        "is_read": True,
        "body_plain": (
            "Your package is on its way!\n\n"
            "Order #112-4839201-7724563\n"
            "Logitech MX Master 3S Wireless Mouse - Graphite\n"
            "Qty: 1\n\n"
            "Estimated delivery: Thursday, February 27\n"
            "Carrier: UPS\n\n"
            "Product page: https://www.amazon.com/dp/B09HM94VDS\n"
            "Track your package: https://www.amazon.com/gp/your-account/order-history"
        ),
        "body_html": (
            '<div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;color:#0F1111">'
            '<div style="background:#232F3E;padding:12px 20px;text-align:center">'
            '<a href="https://www.amazon.com" style="color:#FF9900;font-size:24px;font-weight:bold;text-decoration:none">amazon</a></div>'
            '<div style="padding:20px;border:1px solid #D5D9D9">'
            '<h2 style="color:#0F1111;font-size:18px;margin:0 0 16px">Your package is on its way!</h2>'
            '<div style="background:#F0F2F2;border-radius:8px;padding:16px;margin-bottom:16px">'
            '<p style="margin:0 0 4px;font-size:13px;color:#565959">Order #112-4839201-7724563</p>'
            '<p style="margin:0 0 8px;font-size:15px;font-weight:bold">'
            '<a href="https://www.amazon.com/dp/B09HM94VDS" style="color:#0F1111;text-decoration:none">Logitech MX Master 3S Wireless Mouse - Graphite</a></p>'
            '<p style="margin:0;font-size:14px;color:#565959">Qty: 1</p></div>'
            '<div style="margin-bottom:16px">'
            '<p style="margin:0 0 4px"><strong>Estimated delivery:</strong> Thursday, February 27</p>'
            '<p style="margin:0"><strong>Carrier:</strong> UPS</p></div>'
            '<a href="https://www.amazon.com/gp/your-account/order-history" '
            'style="display:inline-block;padding:8px 20px;background:#FFD814;color:#0F1111;border-radius:20px;text-decoration:none;font-size:14px;font-weight:500">'
            'Track your package</a></div></div>'
        ),
    },
    {
        "sender_name": "Amazon",
        "sender_email": "auto-confirm@amazon.com",
        "subject": "Your Amazon order of 'Designing Data-Intensive App...' and 1 more item",
        "labels": ["INBOX", "CATEGORY_UPDATES"],
        "is_read": True,
        "body_plain": (
            "Order Confirmation\n"
            "Order #112-9938475-6612340\n\n"
            "Items ordered:\n"
            "1. Designing Data-Intensive Applications by Martin Kleppmann - $35.99\n"
            "   https://www.amazon.com/dp/1449373321\n"
            "2. Anker USB-C Hub, 7-in-1 Adapter - $29.99\n"
            "   https://www.amazon.com/dp/B07ZVKTP53\n\n"
            "Order total: $71.87 (includes tax)\n"
            "Estimated delivery: March 1-3\n\n"
            "View or manage your order: https://www.amazon.com/gp/your-account/order-history"
        ),
        "body_html": (
            '<div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;color:#0F1111">'
            '<div style="background:#232F3E;padding:12px 20px;text-align:center">'
            '<a href="https://www.amazon.com" style="color:#FF9900;font-size:24px;font-weight:bold;text-decoration:none">amazon</a></div>'
            '<div style="padding:20px;border:1px solid #D5D9D9">'
            '<h2 style="color:#0F1111;font-size:18px;margin:0 0 4px">Order Confirmation</h2>'
            '<p style="color:#565959;font-size:13px;margin:0 0 16px">Order #112-9938475-6612340</p>'
            '<div style="border-top:1px solid #D5D9D9;border-bottom:1px solid #D5D9D9;padding:12px 0;margin-bottom:16px">'
            '<p style="margin:0 0 8px;font-size:14px"><strong>1.</strong> <a href="https://www.amazon.com/dp/1449373321" style="color:#0066c0;text-decoration:none">Designing Data-Intensive Applications</a> <span style="color:#565959">by Martin Kleppmann</span> — <strong>$35.99</strong></p>'
            '<p style="margin:0;font-size:14px"><strong>2.</strong> <a href="https://www.amazon.com/dp/B07ZVKTP53" style="color:#0066c0;text-decoration:none">Anker USB-C Hub, 7-in-1 Adapter</a> — <strong>$29.99</strong></p></div>'
            '<p style="font-size:16px;font-weight:bold;margin:0 0 4px">Order total: $71.87</p>'
            '<p style="font-size:13px;color:#565959;margin:0 0 16px">Estimated delivery: March 1-3</p>'
            '<a href="https://www.amazon.com/gp/your-account/order-history" '
            'style="display:inline-block;padding:8px 20px;background:#FFD814;color:#0F1111;border-radius:20px;text-decoration:none;font-size:14px">'
            'View or manage order</a></div></div>'
        ),
    },
    # --- Slack ---
    {
        "sender_name": "Slack",
        "sender_email": "notification@slack.com",
        "subject": "New messages in #backend-eng",
        "labels": ["INBOX", "CATEGORY_UPDATES"],
        "is_read": True,
        "body_plain": (
            "You have 3 unread messages in #backend-eng\n\n"
            "Marcus Johnson: Has anyone seen issues with the Redis cluster after this morning's update?\n\n"
            "David Kim: @marcus Yeah, I'm seeing elevated latency on cache reads. Looking into it.\n"
            "Redis docs on latency: https://redis.io/docs/latest/operate/oss_and_stack/management/optimization/latency/\n\n"
            "Marcus Johnson: Thanks David. Let me know if you need help.\n\n"
            "Reply in Slack: https://slack.com"
        ),
        "body_html": (
            '<div style="font-family:Lato,Helvetica Neue,Helvetica,Arial,sans-serif;max-width:600px;margin:0 auto;color:#1D1C1D">'
            '<div style="padding:20px 20px 0;border-bottom:1px solid #DDDDDD">'
            '<a href="https://slack.com" style="text-decoration:none"><span style="font-size:22px;font-weight:bold;color:#4A154B">slack</span></a></div>'
            '<div style="padding:20px">'
            '<h3 style="margin:0 0 4px;font-size:15px">3 new messages in <strong>#backend-eng</strong></h3>'
            '<p style="color:#616061;font-size:13px;margin:0 0 16px">acmecorp.slack.com</p>'
            '<div style="border-left:4px solid #36C5F0;padding:8px 12px;margin-bottom:12px;background:#F8F8F8">'
            '<p style="margin:0 0 2px;font-size:13px;font-weight:bold">Marcus Johnson</p>'
            '<p style="margin:0;font-size:14px">Has anyone seen issues with the Redis cluster after this morning\'s update?</p></div>'
            '<div style="border-left:4px solid #2EB67D;padding:8px 12px;margin-bottom:12px;background:#F8F8F8">'
            '<p style="margin:0 0 2px;font-size:13px;font-weight:bold">David Kim</p>'
            '<p style="margin:0;font-size:14px">@marcus Yeah, I\'m seeing elevated latency. '
            '<a href="https://redis.io/docs/latest/operate/oss_and_stack/management/optimization/latency/" style="color:#1264A3">Redis latency docs</a></p></div>'
            '<div style="border-left:4px solid #36C5F0;padding:8px 12px;margin-bottom:16px;background:#F8F8F8">'
            '<p style="margin:0 0 2px;font-size:13px;font-weight:bold">Marcus Johnson</p>'
            '<p style="margin:0;font-size:14px">Thanks David. Let me know if you need help.</p></div>'
            '<a href="https://slack.com" '
            'style="display:inline-block;padding:8px 16px;background:#4A154B;color:#fff;border-radius:4px;text-decoration:none;font-size:14px">'
            'Reply in Slack</a></div></div>'
        ),
    },
    # --- Google Calendar ---
    {
        "sender_name": "Google Calendar",
        "sender_email": "calendar-notification@google.com",
        "subject": "Reminder: Architecture Review tomorrow at 2:00pm",
        "labels": ["INBOX", "CATEGORY_UPDATES"],
        "is_read": False,
        "body_plain": (
            "Reminder: Architecture Review\n\n"
            "When: Thursday, February 27, 2025 2:00pm - 3:30pm (PST)\n"
            "Where: Conference Room 4B / Google Meet\n"
            "Calendar: Alex Thompson\n\n"
            "Agenda:\n"
            "- RFC review: Event-driven notifications (Alex)\n"
            "- PG16 upgrade proposal (David)\n"
            "- Q1 architecture goals\n\n"
            "Open Google Calendar: https://calendar.google.com"
        ),
        "body_html": (
            '<div style="font-family:Google Sans,Roboto,Arial,sans-serif;max-width:600px;margin:0 auto;color:#3C4043">'
            '<div style="background:#1a73e8;padding:16px 20px;border-radius:8px 8px 0 0">'
            '<h2 style="color:#fff;margin:0;font-size:18px;font-weight:400">Reminder</h2></div>'
            '<div style="padding:20px;border:1px solid #DADCE0;border-top:none;border-radius:0 0 8px 8px">'
            '<h3 style="margin:0 0 16px;font-size:20px;color:#202124">Architecture Review</h3>'
            '<div style="margin-bottom:16px;font-size:14px;line-height:1.6">'
            '<p style="margin:0 0 4px"><strong>When:</strong> Thursday, February 27, 2025 2:00pm - 3:30pm (PST)</p>'
            '<p style="margin:0 0 4px"><strong>Where:</strong> Conference Room 4B / Google Meet</p>'
            '<p style="margin:0"><strong>Calendar:</strong> Alex Thompson</p></div>'
            '<div style="background:#F1F3F4;border-radius:8px;padding:12px;margin-bottom:16px">'
            '<p style="margin:0 0 8px;font-weight:500">Agenda:</p>'
            '<ul style="margin:0;padding-left:20px;font-size:14px">'
            '<li>RFC review: Event-driven notifications (Alex)</li>'
            '<li>PG16 upgrade proposal (David)</li>'
            '<li>Q1 architecture goals</li></ul></div>'
            '<div style="text-align:center">'
            '<a href="https://calendar.google.com" style="display:inline-block;padding:8px 24px;margin:0 4px;background:#1a73e8;color:#fff;border-radius:4px;text-decoration:none;font-size:14px">Open Calendar</a>'
            '</div></div></div>'
        ),
    },
    # --- LinkedIn ---
    {
        "sender_name": "LinkedIn",
        "sender_email": "notifications-noreply@linkedin.com",
        "subject": "Sarah Chen endorsed you for Distributed Systems",
        "labels": ["INBOX", "CATEGORY_SOCIAL"],
        "is_read": True,
        "body_plain": (
            "Alex, Sarah Chen has endorsed you!\n\n"
            "Sarah Chen endorsed you for Distributed Systems.\n"
            "You now have 24 endorsements for this skill.\n\n"
            "View your profile: https://www.linkedin.com/in/"
        ),
        "body_html": (
            '<div style="font-family:-apple-system,system-ui,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;max-width:600px;margin:0 auto;color:#000000E6">'
            '<div style="background:#0A66C2;padding:12px 20px">'
            '<a href="https://www.linkedin.com" style="color:#fff;font-size:20px;font-weight:bold;text-decoration:none">LinkedIn</a></div>'
            '<div style="padding:20px;border:1px solid #E0E0E0">'
            '<p style="font-size:16px;margin:0 0 16px">Alex, <strong>Sarah Chen</strong> has endorsed you!</p>'
            '<div style="background:#F3F6F8;border-radius:8px;padding:16px;text-align:center;margin-bottom:16px">'
            '<p style="font-size:18px;font-weight:600;margin:0 0 4px">Distributed Systems</p>'
            '<p style="color:#666;font-size:14px;margin:0">You now have <strong>24</strong> endorsements for this skill</p></div>'
            '<a href="https://www.linkedin.com/in/" '
            'style="display:inline-block;padding:8px 24px;background:#0A66C2;color:#fff;border-radius:20px;text-decoration:none;font-size:14px;font-weight:600">'
            'View profile</a></div></div>'
        ),
    },
    {
        "sender_name": "LinkedIn",
        "sender_email": "notifications-noreply@linkedin.com",
        "subject": "5 people viewed your profile this week",
        "labels": ["INBOX", "CATEGORY_SOCIAL"],
        "is_read": True,
        "body_plain": (
            "Your profile is getting noticed!\n\n"
            "5 people viewed your profile this week.\n\n"
            "See who viewed your profile: https://www.linkedin.com/me/profile-views/"
        ),
        "body_html": (
            '<div style="font-family:-apple-system,system-ui,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;max-width:600px;margin:0 auto;color:#000000E6">'
            '<div style="background:#0A66C2;padding:12px 20px">'
            '<a href="https://www.linkedin.com" style="color:#fff;font-size:20px;font-weight:bold;text-decoration:none">LinkedIn</a></div>'
            '<div style="padding:20px;border:1px solid #E0E0E0">'
            '<h3 style="margin:0 0 16px;font-size:18px">Your profile is getting noticed!</h3>'
            '<p style="font-size:28px;font-weight:700;text-align:center;color:#0A66C2;margin:0 0 8px">5</p>'
            '<p style="text-align:center;font-size:14px;color:#666;margin:0 0 16px">people viewed your profile this week</p>'
            '<div style="text-align:center">'
            '<a href="https://www.linkedin.com/me/profile-views/" '
            'style="display:inline-block;padding:8px 24px;background:#0A66C2;color:#fff;border-radius:20px;text-decoration:none;font-size:14px;font-weight:600">'
            'See all viewers</a></div></div></div>'
        ),
    },
    # --- Uber ---
    {
        "sender_name": "Uber Receipts",
        "sender_email": "uber.us@uber.com",
        "subject": "Your Wednesday evening trip with Uber",
        "labels": ["INBOX", "CATEGORY_UPDATES"],
        "is_read": True,
        "body_plain": (
            "Trip receipt\n\n"
            "Thanks for riding, Alex!\n\n"
            "Trip on Wednesday, Feb 26\n"
            "UberX\n\n"
            "Pickup: 455 Market St, San Francisco\n"
            "Dropoff: 742 Evergreen Terrace, Daly City\n\n"
            "Trip fare: $18.43\n"
            "Booking fee: $2.95\n"
            "Total: $21.38\n\n"
            "View trip details: https://riders.uber.com/trips\n"
            "Payment: Visa ending in 4242"
        ),
        "body_html": (
            '<div style="font-family:UberMove,Helvetica Neue,Helvetica,Arial,sans-serif;max-width:600px;margin:0 auto;color:#000">'
            '<div style="padding:20px;background:#000;text-align:center">'
            '<a href="https://www.uber.com" style="color:#fff;font-size:24px;font-weight:bold;text-decoration:none">Uber</a></div>'
            '<div style="padding:24px">'
            '<h2 style="font-size:24px;font-weight:700;margin:0 0 4px">$21.38</h2>'
            '<p style="color:#6B6B6B;font-size:14px;margin:0 0 20px">Wednesday, Feb 26 &middot; UberX</p>'
            '<div style="margin-bottom:20px;font-size:14px">'
            '<p style="margin:0 0 16px"><strong>455 Market St</strong><br><span style="color:#6B6B6B">San Francisco, CA</span></p>'
            '<p style="margin:0"><strong>742 Evergreen Terrace</strong><br><span style="color:#6B6B6B">Daly City, CA</span></p></div>'
            '<div style="border-top:1px solid #E0E0E0;padding-top:12px;font-size:14px">'
            '<div style="display:flex;justify-content:space-between;margin-bottom:4px"><span>Trip fare</span><span>$18.43</span></div>'
            '<div style="display:flex;justify-content:space-between;margin-bottom:8px"><span>Booking fee</span><span>$2.95</span></div>'
            '<div style="display:flex;justify-content:space-between;font-weight:700;border-top:1px solid #E0E0E0;padding-top:8px"><span>Total</span><span>$21.38</span></div>'
            '<p style="color:#6B6B6B;font-size:12px;margin:12px 0 0">Visa ending in 4242</p></div>'
            '<div style="margin-top:16px;text-align:center">'
            '<a href="https://riders.uber.com/trips" style="display:inline-block;padding:8px 24px;background:#000;color:#fff;border-radius:8px;text-decoration:none;font-size:14px">View trip</a></div></div></div>'
        ),
    },
    # --- Google Security ---
    {
        "sender_name": "Google",
        "sender_email": "no-reply@accounts.google.com",
        "subject": "Security alert: New sign-in from MacBook Pro",
        "labels": ["INBOX", "CATEGORY_UPDATES"],
        "is_read": True,
        "body_plain": (
            "New sign-in to your Google Account\n\n"
            "Alex Thompson\nme@example.com\n\n"
            "Your Google Account was just signed in to from a new device.\n\n"
            "Device: MacBook Pro\n"
            "Location: San Francisco, CA, USA\n"
            "Time: Feb 26, 2025 9:14 AM PST\n\n"
            "If this was you, no further action is needed.\n"
            "If not, visit https://myaccount.google.com/security to secure your account."
        ),
        "body_html": (
            '<div style="font-family:Google Sans,Roboto,Arial,sans-serif;max-width:500px;margin:0 auto;color:#3C4043">'
            '<div style="text-align:center;padding:24px 0">'
            '<a href="https://myaccount.google.com" style="text-decoration:none">'
            '<div style="width:48px;height:48px;background:#4285F4;border-radius:50%;margin:0 auto 16px;display:flex;align-items:center;justify-content:center">'
            '<span style="color:#fff;font-size:20px">G</span></div></a>'
            '<h2 style="margin:0 0 8px;font-size:18px">New sign-in to your Google Account</h2>'
            '<p style="color:#5F6368;margin:0;font-size:14px">me@example.com</p></div>'
            '<div style="background:#F8F9FA;border-radius:8px;padding:16px;margin:0 0 16px">'
            '<p style="margin:0 0 8px;font-size:14px"><strong>Device:</strong> MacBook Pro</p>'
            '<p style="margin:0 0 8px;font-size:14px"><strong>Location:</strong> San Francisco, CA, USA</p>'
            '<p style="margin:0;font-size:14px"><strong>Time:</strong> Feb 26, 2025 9:14 AM PST</p></div>'
            '<p style="font-size:14px;color:#5F6368;margin:0 0 16px">If this was you, no further action is needed.</p>'
            '<a href="https://myaccount.google.com/security" '
            'style="display:inline-block;padding:8px 24px;background:#1a73e8;color:#fff;border-radius:4px;text-decoration:none;font-size:14px">'
            'Review activity</a></div>'
        ),
    },
    # --- Jira ---
    {
        "sender_name": "Jira",
        "sender_email": "jira@acmecorp.atlassian.net",
        "subject": "[BACKEND-1247] Assigned to you: Implement API rate limiting",
        "labels": ["INBOX", "CATEGORY_UPDATES"],
        "is_read": False,
        "body_plain": (
            "Sarah Chen assigned BACKEND-1247 to you.\n\n"
            "Title: Implement API rate limiting\n"
            "Priority: High\n"
            "Sprint: Q1 Sprint 3\n"
            "Story Points: 5\n\n"
            "Description:\n"
            "Implement token bucket rate limiting for all public API endpoints.\n"
            "Requirements:\n"
            "- 100 requests/min for free tier\n"
            "- 1000 requests/min for pro tier\n"
            "- Return 429 with Retry-After header\n\n"
            "Reference: https://www.ietf.org/rfc/rfc6585.txt (HTTP 429 spec)\n"
            "Algorithm: https://en.wikipedia.org/wiki/Token_bucket\n\n"
            "View issue: https://www.atlassian.com/software/jira"
        ),
        "body_html": (
            '<div style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;max-width:600px;margin:0 auto;color:#172B4D">'
            '<div style="background:#0052CC;padding:12px 20px">'
            '<a href="https://www.atlassian.com/software/jira" style="color:#fff;font-size:16px;font-weight:bold;text-decoration:none">Jira</a></div>'
            '<div style="padding:20px;border:1px solid #DFE1E6">'
            '<p style="font-size:13px;color:#6B778C;margin:0 0 4px">Sarah Chen assigned an issue to you</p>'
            '<h3 style="margin:0 0 12px"><a href="https://www.atlassian.com/software/jira" style="color:#0052CC;text-decoration:none">'
            'BACKEND-1247</a>: Implement API rate limiting</h3>'
            '<div style="display:flex;gap:16px;margin-bottom:16px;font-size:13px">'
            '<span style="background:#FF5630;color:#fff;padding:2px 8px;border-radius:3px;font-weight:500">High</span>'
            '<span style="color:#6B778C">Sprint: Q1 Sprint 3</span>'
            '<span style="color:#6B778C">Story Points: 5</span></div>'
            '<div style="background:#F4F5F7;border-radius:3px;padding:12px;font-size:14px">'
            '<p style="margin:0 0 8px">Implement <a href="https://en.wikipedia.org/wiki/Token_bucket" style="color:#0052CC">token bucket</a> rate limiting for all public API endpoints.</p>'
            '<ul style="margin:0;padding-left:20px;color:#42526E">'
            '<li>100 requests/min for free tier</li>'
            '<li>1000 requests/min for pro tier</li>'
            '<li>Return <a href="https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/429" style="color:#0052CC">429</a> with Retry-After header</li></ul></div></div></div>'
        ),
    },
]

# ---------------------------------------------------------------------------
# Personal emails
# ---------------------------------------------------------------------------
PERSONAL_EMAILS: list[dict] = [
    {
        "sender": "mike",
        "subject": "Dinner Saturday? That new ramen place",
        "labels": ["INBOX"],
        "body_plain": (
            "Hey Alex!\n\n"
            "Have you tried that new ramen spot on Valencia St yet? Tonkotsu King or something. "
            "Jenny and I were thinking of going Saturday around 7pm.\n\n"
            "Looks pretty legit:\n"
            "https://www.yelp.com/search?find_desc=ramen&find_loc=San+Francisco%2C+CA\n\n"
            "Want to join? I think they take reservations on Yelp.\n\n"
            "Mike"
        ),
        "is_read": True,
    },
    {
        "sender": "jenny",
        "subject": "Photos from the hiking trip",
        "labels": ["INBOX"],
        "body_plain": (
            "Hi Alex!\n\n"
            "Finally got around to uploading the photos from our Mt. Tam hike last weekend. "
            "Some of them turned out really great, especially the ones from the sunset viewpoint.\n\n"
            "Google Photos album: https://photos.google.com\n\n"
            "The trail we did: https://www.alltrails.com/trail/us/california/mount-tamalpais-east-peak-loop\n\n"
            "Let me know which ones you want me to send you the full-res versions of!\n\n"
            "Jenny"
        ),
        "is_read": True,
    },
    {
        "sender": "mike",
        "subject": "Board game night next Friday?",
        "labels": ["INBOX"],
        "body_plain": (
            "Alex,\n\n"
            "Thinking about hosting a board game night next Friday (Mar 7). "
            "I just got Wingspan and Terraforming Mars.\n\n"
            "Wingspan: https://boardgamegeek.com/boardgame/266192/wingspan\n"
            "Terraforming Mars: https://boardgamegeek.com/boardgame/167791/terraforming-mars\n\n"
            "So far it's me, Jenny, and maybe Chris and Sarah from the climbing gym. "
            "You in?\n\n"
            "Starts around 7, BYOB. My place.\n\n"
            "Mike"
        ),
        "is_read": False,
    },
]

# ---------------------------------------------------------------------------
# Promotional emails
# ---------------------------------------------------------------------------
PROMO_EMAILS: list[dict] = [
    {
        "sender_name": "O'Reilly Media",
        "sender_email": "deals@oreilly.com",
        "subject": "New release: 'Building Event-Driven Microservices' - 30% off this week",
        "labels": ["CATEGORY_PROMOTIONS"],
        "body_plain": (
            "New from O'Reilly\n\n"
            "Building Event-Driven Microservices\n"
            "by Adam Bellemare\n\n"
            "Master the principles of event-driven architecture. Learn to build "
            "resilient, scalable systems using Apache Kafka, event sourcing, and CQRS.\n\n"
            "30% off this week with code EVENTS30\n\n"
            "Shop now: https://www.oreilly.com/library/view/building-event-driven-microservices/9781492057888/"
        ),
        "is_read": False,
    },
    {
        "sender_name": "Uniqlo",
        "sender_email": "store@uniqlo.com",
        "subject": "New arrivals: Spring collection + free shipping over $75",
        "labels": ["CATEGORY_PROMOTIONS"],
        "body_plain": (
            "UNIQLO\n\n"
            "Spring Collection 2025\n\n"
            "Lightweight layers for the season ahead.\n\n"
            "Plus, enjoy FREE SHIPPING on orders over $75.\n\n"
            "Shop now: https://www.uniqlo.com/us/en/men\n\n"
            "Unsubscribe: https://www.uniqlo.com"
        ),
        "is_read": False,
    },
    {
        "sender_name": "The Pragmatic Bookshelf",
        "sender_email": "news@pragprog.com",
        "subject": "This week in tech books: Elixir, Rust, and System Design",
        "labels": ["CATEGORY_PROMOTIONS"],
        "body_plain": (
            "This Week at The Pragmatic Bookshelf\n\n"
            "New and Updated Titles:\n\n"
            "1. Programming Elixir 1.16 - Dave Thomas\n"
            "   https://pragprog.com/titles/elixir16/programming-elixir-1-6/\n\n"
            "2. Programming Rust, 2nd Edition - Jim Blandy\n"
            "   https://www.oreilly.com/library/view/programming-rust-2nd/9781492052586/\n\n"
            "3. Designing Data-Intensive Applications - Martin Kleppmann\n"
            "   https://dataintensive.net/\n\n"
            "Browse all titles: https://pragprog.com"
        ),
        "is_read": False,
    },
    {
        "sender_name": "Coursera",
        "sender_email": "no-reply@coursera.org",
        "subject": "Continue your learning: 'Machine Learning Specialization' is waiting",
        "labels": ["CATEGORY_PROMOTIONS"],
        "body_plain": (
            "Hi Alex,\n\n"
            "You're 40% through the Machine Learning Specialization by Andrew Ng. "
            "Don't lose your momentum!\n\n"
            "Your next module: Neural Networks and Deep Learning\n"
            "Estimated time: 3 hours\n\n"
            "Continue learning: https://www.coursera.org/specializations/machine-learning-introduction\n\n"
            "Coursera Team"
        ),
        "is_read": True,
    },
]

# ---------------------------------------------------------------------------
# Sent emails (not part of any thread above)
# ---------------------------------------------------------------------------
SENT_EMAILS: list[dict] = [
    {
        "to": "mike.chen@gmail.com",
        "subject": "RE: Dinner Saturday? That new ramen place",
        "body_plain": (
            "Sounds great! I'm in. 7pm works for me.\n\n"
            "Should I bring anything? I have a bottle of sake that would go perfectly.\n\n"
            "Alex"
        ),
    },
    {
        "to": "rachel.foster@acmecorp.com",
        "subject": "Dev environment setup guide",
        "body_plain": (
            "Hi Rachel,\n\n"
            "Welcome to the team! Here's the setup guide I promised:\n\n"
            "1. Install Docker Desktop: https://docs.docker.com/desktop/\n"
            "2. Install Python 3.12: https://www.python.org/downloads/\n"
            "3. Set up the project with Poetry: https://python-poetry.org/docs/\n"
            "4. Run the test suite: make test\n\n"
            "Useful references:\n"
            "- FastAPI docs: https://fastapi.tiangolo.com/\n"
            "- SQLAlchemy tutorial: https://docs.sqlalchemy.org/en/20/tutorial/\n"
            "- Our API follows the Google API design guide: https://cloud.google.com/apis/design\n\n"
            "Let me know if you hit any snags!\n\n"
            "Alex"
        ),
    },
    {
        "to": "sarah.chen@acmecorp.com",
        "subject": "Tech talk proposal: Building Resilient Microservices",
        "body_plain": (
            "Hi Sarah,\n\n"
            "Here's the draft proposal for my tech talk:\n\n"
            "Title: Building Resilient Microservices: Lessons from our Auth Migration\n\n"
            "Abstract: This talk covers the practical challenges and solutions we "
            "encountered while migrating our monolithic auth system to a distributed "
            "microservice architecture. Topics include:\n"
            "- Circuit breaker patterns: https://martinfowler.com/bliki/CircuitBreaker.html\n"
            "- Distributed session management\n"
            "- Zero-downtime migration strategies\n"
            "- Monitoring with OpenTelemetry: https://opentelemetry.io/docs/\n\n"
            "Duration: 30 minutes + 10 min Q&A\n\n"
            "Let me know what you think!\n\n"
            "Alex"
        ),
    },
]

# ---------------------------------------------------------------------------
# Draft
# ---------------------------------------------------------------------------
DRAFT_EMAIL: dict = {
    "to": "backend-team@acmecorp.com",
    "subject": "RFC: API Rate Limiting Implementation",
    "body_plain": (
        "Team,\n\n"
        "Following up on the discussion in sprint planning, here's my proposal for "
        "implementing rate limiting on our public API.\n\n"
        "Approach: Token bucket algorithm with Redis as the backing store.\n"
        "Reference: https://en.wikipedia.org/wiki/Token_bucket\n\n"
        "Key decisions:\n"
        "- Per-API-key limits (not per-IP) to support server-to-server use cases\n"
        "- Sliding window counter for burst handling\n"
        "- Response headers per RFC 6585: https://www.ietf.org/rfc/rfc6585.txt\n"
        "  X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset\n\n"
        "Libraries to evaluate:\n"
        "- https://github.com/long2ice/fastapi-limiter\n"
        "- https://github.com/laurentS/slowapi\n\n"
        "Open questions:\n"
        "- Should we use a shared Redis instance or deploy a dedicated one?\n"
        "- Do we need different limits per endpoint, or is a global limit sufficient?\n\n"
        "[DRAFT - still working on the implementation details section]"
    ),
}
