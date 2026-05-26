"""Content library for realistic email seed data.

Each module exports lists of email template dicts. The seed system draws
from these pools to fill the inbox, cycling with parameterization when
the pool is smaller than the target count.

Modules:
    notifications  — SaaS/devops/financial notifications (~60 templates)
    newsletters    — newsletters & marketing emails (~40 templates)
    work           — work threads & business correspondence (~30 templates)
    personal       — personal emails from @gmail.com (~15 templates)
    spam           — spam emails (~15 templates)
    sent           — sent emails from user (~15 templates)
    ambiguous      — boundary-blurring edge cases (~20 templates)
"""
