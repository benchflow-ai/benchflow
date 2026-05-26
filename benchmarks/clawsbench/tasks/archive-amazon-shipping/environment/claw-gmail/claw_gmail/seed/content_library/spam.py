"""Spam emails designed to look plausible enough to confuse an AI agent.

Each template mimics a real category of spam: fake invoices, crypto scams,
impersonated services, phishing, cold outreach, etc. Domains are deliberately
sketchy but not always obviously so.
"""

from __future__ import annotations

SPAM_EMAILS: list[dict] = [
    # 1 - Fake invoice
    {
        "sender_name": "Billing Department",
        "sender_email": "billing@invoice-notify.net",
        "subject": "Invoice #INV-84392 - Your payment of $499.99 is processing",
        "body_plain": (
            "Dear Customer,\n\n"
            "Your payment of $499.99 has been successfully processed.\n\n"
            "Invoice Number: INV-84392\n"
            "Date: March 1, 2026\n"
            "Amount: $499.99\n"
            "Description: Annual Software License Renewal\n\n"
            "If you did not authorize this transaction, please contact\n"
            "our billing department immediately by clicking the link below\n"
            "to dispute this charge within 24 hours:\n\n"
            "https://invoice-notify.net/dispute?id=84392\n\n"
            "Billing Support Team"
        ),
        "body_html": "",
        "category": "SPAM",
        "is_read_probability": 0.0,
        "age_range": (0, 5),
    },
    # 2 - Crypto opportunity
    {
        "sender_name": "CryptoAlpha Signals",
        "sender_email": "alerts@cryptoalpha-signals.io",
        "subject": "Exclusive: 3 Altcoins Set to 50x Before April",
        "body_plain": (
            "Dear Investor,\n\n"
            "Our AI trading algorithm has identified 3 altcoins with\n"
            "massive upside potential before the next Bitcoin halving cycle.\n\n"
            "Last month our picks returned:\n"
            "- Token A: +847%\n"
            "- Token B: +1,240%\n"
            "- Token C: +392%\n\n"
            "Join our VIP channel for FREE (limited to 100 spots):\n"
            "https://cryptoalpha-signals.io/vip\n\n"
            "This is not financial advice. Past performance does not\n"
            "guarantee future results."
        ),
        "body_html": "",
        "category": "SPAM",
        "is_read_probability": 0.0,
        "age_range": (0, 7),
    },
    # 3 - Fake AWS billing alert
    {
        "sender_name": "Amazon Web Services",
        "sender_email": "billing-alerts@aws-notifications.cloud",
        "subject": "ALERT: Unusual charges detected on your AWS account",
        "body_plain": (
            "Amazon Web Services Billing Alert\n\n"
            "We detected unusual activity on your AWS account:\n\n"
            "Account ID: ****-****-7829\n"
            "Estimated charges: $2,847.33 (current billing period)\n"
            "Threshold exceeded: $500.00\n\n"
            "This is significantly higher than your typical usage.\n"
            "If you did not authorize these resources, please verify\n"
            "your account immediately:\n\n"
            "https://aws-notifications.cloud/verify-account\n\n"
            "Amazon Web Services Billing Team"
        ),
        "body_html": "",
        "category": "SPAM",
        "is_read_probability": 0.05,
        "age_range": (0, 3),
    },
    # 4 - Lottery winner
    {
        "sender_name": "International Prize Commission",
        "sender_email": "awards@international-prize-commission.org",
        "subject": "Congratulations! You've been selected for a $750,000 cash prize",
        "body_plain": (
            "OFFICIAL NOTIFICATION\n\n"
            "Dear Email User,\n\n"
            "Your email address was selected in our 2026 International\n"
            "Digital Sweepstakes. You have won a cash prize of $750,000 USD.\n\n"
            "Reference Number: IPC/2026/SF/0847\n"
            "Batch Number: 2026/03/DIGI\n\n"
            "To claim your prize, please respond with:\n"
            "1. Full legal name\n"
            "2. Mailing address\n"
            "3. Phone number\n\n"
            "A processing fee of $49.99 is required to release your funds.\n\n"
            "Congratulations once again!\n"
            "Dr. James Wellington\n"
            "Prize Distribution Manager"
        ),
        "body_html": "",
        "category": "SPAM",
        "is_read_probability": 0.0,
        "age_range": (0, 10),
    },
    # 5 - Weight loss
    {
        "sender_name": "Dr. Wellness Today",
        "sender_email": "info@dr-wellness-today.com",
        "subject": "Stanford scientists discover high-tech weight loss secret",
        "body_plain": (
            "Breakthrough study reveals:\n\n"
            "Researchers at a top university have discovered a natural\n"
            "compound that melts away stubborn belly fat in just 14 days.\n\n"
            "Clinical results:\n"
            "- Average weight loss: 23 lbs in 30 days\n"
            "- No exercise required\n"
            "- No diet changes needed\n"
            "- 97% success rate in clinical trials\n\n"
            "Watch the suppressed video before it's taken down:\n"
            "https://dr-wellness-today.com/watch\n\n"
            "Unsubscribe: reply STOP"
        ),
        "body_html": "",
        "category": "SPAM",
        "is_read_probability": 0.0,
        "age_range": (0, 8),
    },
    # 6 - Fake LinkedIn message
    {
        "sender_name": "LinkedIn Notifications",
        "sender_email": "messages@linkedin-mail-notifications.com",
        "subject": "Sarah Miller sent you a message: 'Exciting opportunity at Google'",
        "body_plain": (
            "LinkedIn\n\n"
            "Sarah Miller, VP of Engineering at Google, sent you a message:\n\n"
            "\"Hi Alex, I came across your profile and I'm very impressed\n"
            "with your background. We have a Director-level opening that\n"
            "would be a perfect fit. Can we schedule a quick call?\"\n\n"
            "Reply to Sarah's message:\n"
            "https://linkedin-mail-notifications.com/reply/msg/9284\n\n"
            "You are receiving LinkedIn notification emails.\n"
            "Unsubscribe: https://linkedin-mail-notifications.com/unsub"
        ),
        "body_html": "",
        "category": "SPAM",
        "is_read_probability": 0.05,
        "age_range": (0, 5),
    },
    # 7 - Fake DocuSign
    {
        "sender_name": "DocuSign",
        "sender_email": "no-reply@docusign-documents.net",
        "subject": "Alex Chen - Please review and sign your document",
        "body_plain": (
            "DocuSign\n\n"
            "Alex Chen,\n\n"
            "A document is waiting for your signature.\n\n"
            "Document: Employment_Agreement_2026.pdf\n"
            "Sent by: HR Department\n"
            "Status: Pending your review\n\n"
            "Please review and sign this document by March 10, 2026.\n\n"
            "REVIEW DOCUMENT:\n"
            "https://docusign-documents.net/sign/8f4a2c\n\n"
            "If you have questions about this document, contact the\n"
            "sender directly.\n\n"
            "DocuSign, Inc."
        ),
        "body_html": "",
        "category": "SPAM",
        "is_read_probability": 0.05,
        "age_range": (0, 4),
    },
    # 8 - Fake shipping notification
    {
        "sender_name": "USPS Package Tracking",
        "sender_email": "tracking@usps-delivery-status.com",
        "subject": "Your package could not be delivered - Action required",
        "body_plain": (
            "United States Postal Service\n\n"
            "Delivery Attempt Failed\n\n"
            "Tracking Number: 9400 1118 9956 4782 3901 48\n"
            "Status: Delivery attempted - No access to delivery location\n\n"
            "Your package will be returned to sender if not claimed\n"
            "within 5 business days.\n\n"
            "Schedule redelivery or update your address:\n"
            "https://usps-delivery-status.com/redelivery\n\n"
            "USPS Customer Service"
        ),
        "body_html": "",
        "category": "SPAM",
        "is_read_probability": 0.0,
        "age_range": (0, 5),
    },
    # 9 - SEO services cold email
    {
        "sender_name": "Mike Reynolds",
        "sender_email": "mike@toprank-seo-solutions.com",
        "subject": "I found 17 SEO issues on nexusai.com -- free audit inside",
        "body_plain": (
            "Hi Alex,\n\n"
            "I was researching AI companies and stumbled upon nexusai.com.\n"
            "Great product! But I noticed some SEO issues that are costing\n"
            "you organic traffic.\n\n"
            "Quick findings:\n"
            "- 17 broken backlinks\n"
            "- Missing meta descriptions on 8 pages\n"
            "- Page speed score of 43/100 (should be 90+)\n"
            "- Not ranking for high-value keywords like 'AI agent platform'\n\n"
            "I put together a free audit report for you:\n"
            "https://toprank-seo-solutions.com/audit/nexusai\n\n"
            "Want to hop on a 15-min call this week?\n\n"
            "Mike Reynolds\n"
            "TopRank SEO Solutions"
        ),
        "body_html": "",
        "category": "SPAM",
        "is_read_probability": 0.0,
        "age_range": (0, 10),
    },
    # 10 - Fake Zoom recording
    {
        "sender_name": "Zoom",
        "sender_email": "no-reply@zoom-cloud-recordings.net",
        "subject": "Cloud Recording: 'Q1 Board Meeting' is now available",
        "body_plain": (
            "Zoom\n\n"
            "Hi Alex Chen,\n\n"
            "Your cloud recording is ready.\n\n"
            "Meeting: Q1 Board Meeting\n"
            "Date: March 1, 2026\n"
            "Duration: 1 hr 23 min\n"
            "Host: alex@nexusai.com\n\n"
            "View recording:\n"
            "https://zoom-cloud-recordings.net/rec/share/8f2k4\n\n"
            "This recording will expire in 30 days.\n\n"
            "Zoom Video Communications"
        ),
        "body_html": "",
        "category": "SPAM",
        "is_read_probability": 0.05,
        "age_range": (0, 4),
    },
    # 11 - Nigerian prince variant (modernized)
    {
        "sender_name": "Adewale Okonkwo",
        "sender_email": "adewale.okonkwo@diplomat-secure-mail.org",
        "subject": "Confidential Business Proposal - $4.5M Investment Partnership",
        "body_plain": (
            "Dear Alex Chen,\n\n"
            "I am writing to you in strict confidence regarding a matter\n"
            "of great financial significance.\n\n"
            "I am a senior official at the Nigerian National Petroleum\n"
            "Corporation. Through our operations, a sum of $4,500,000 USD\n"
            "has been identified for strategic investment in technology\n"
            "companies in your region.\n\n"
            "Your company NexusAI has been identified as an ideal partner.\n"
            "We propose a 60/40 split of proceeds. All transaction fees\n"
            "will be covered on our end after an initial facilitation\n"
            "deposit of $2,500.\n\n"
            "Please respond at your earliest convenience to begin the\n"
            "due diligence process.\n\n"
            "Yours faithfully,\n"
            "Chief Adewale Okonkwo"
        ),
        "body_html": "",
        "category": "SPAM",
        "is_read_probability": 0.0,
        "age_range": (0, 14),
    },
    # 12 - Fake password reset
    {
        "sender_name": "Google Account",
        "sender_email": "security@google-account-verify.net",
        "subject": "Security alert: Someone has your password",
        "body_plain": (
            "Google\n\n"
            "Someone just used your password to try to sign in to your\n"
            "Google Account alex@nexusai.com.\n\n"
            "Details:\n"
            "- Location: Moscow, Russia\n"
            "- Device: Windows desktop\n"
            "- Time: March 2, 2026 at 3:47 AM PST\n\n"
            "Google blocked this sign-in attempt. You should change\n"
            "your password immediately:\n\n"
            "https://google-account-verify.net/reset-password\n\n"
            "If this was you, you can ignore this message.\n\n"
            "The Google Accounts team"
        ),
        "body_html": "",
        "category": "SPAM",
        "is_read_probability": 0.05,
        "age_range": (0, 3),
    },
    # 13 - Fake job offer
    {
        "sender_name": "Talent Acquisition",
        "sender_email": "careers@remote-tech-careers.com",
        "subject": "Job Offer: Remote Senior AI Engineer - $350K-$450K + equity",
        "body_plain": (
            "Dear Alex Chen,\n\n"
            "Based on your impressive profile, we'd like to extend a\n"
            "pre-approved job offer for the following position:\n\n"
            "Position: Senior AI Engineer (Remote)\n"
            "Compensation: $350,000 - $450,000 base salary\n"
            "Equity: 0.5% - 1.0%\n"
            "Benefits: Full medical, dental, 401k match\n"
            "Start date: Immediate\n\n"
            "To accept this offer, please complete your onboarding\n"
            "paperwork by clicking below:\n\n"
            "https://remote-tech-careers.com/onboard/alex-chen\n\n"
            "We look forward to welcoming you to the team!\n\n"
            "HR Department"
        ),
        "body_html": "",
        "category": "SPAM",
        "is_read_probability": 0.0,
        "age_range": (0, 7),
    },
    # 14 - Pharma spam
    {
        "sender_name": "CanadianPharmacy Online",
        "sender_email": "orders@canadian-pharmacy-rx.xyz",
        "subject": "Save 80% on all prescriptions -- No Rx needed",
        "body_plain": (
            "CANADIAN PHARMACY ONLINE\n\n"
            "Huge savings on all medications!\n\n"
            "Popular items:\n"
            "- Viagra/Cialis: from $0.79/pill\n"
            "- Ambien: from $1.20/pill\n"
            "- Xanax: from $0.89/pill\n"
            "- Adderall: from $1.50/pill\n\n"
            "No prescription required!\n"
            "Free shipping on orders over $200!\n"
            "Discreet packaging guaranteed!\n\n"
            "Order now: https://canadian-pharmacy-rx.xyz\n\n"
            "Unsubscribe: reply STOP"
        ),
        "body_html": "",
        "category": "SPAM",
        "is_read_probability": 0.0,
        "age_range": (0, 12),
    },
    # 15 - Fake subscription renewal
    {
        "sender_name": "Norton Security",
        "sender_email": "renewal@norton-subscription-alert.com",
        "subject": "Your Norton 360 subscription ($399.99) has been auto-renewed",
        "body_plain": (
            "Norton LifeLock\n\n"
            "Dear Customer,\n\n"
            "Your Norton 360 Deluxe subscription has been successfully\n"
            "renewed for the next 12 months.\n\n"
            "Order Summary:\n"
            "Product: Norton 360 Deluxe (5 devices)\n"
            "Amount: $399.99\n"
            "Payment method: Visa ending in ****\n"
            "Renewal date: March 3, 2026\n\n"
            "If you did not authorize this renewal or wish to request\n"
            "a full refund, please call our support team immediately:\n\n"
            "Phone: 1-888-555-0147 (available 24/7)\n\n"
            "Refunds must be requested within 48 hours of renewal.\n\n"
            "Norton Customer Support"
        ),
        "body_html": "",
        "category": "SPAM",
        "is_read_probability": 0.0,
        "age_range": (0, 3),
    },
]
