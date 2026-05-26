"""Sent emails from Alex Chen (alex@nexusai.com), founder of NexusAI.

These represent outbound emails Alex has sent: replies to partners, follow-ups,
intros, scheduling, investor updates, thank-you notes, and team questions.
"""

from __future__ import annotations

SENT_EMAILS: list[dict] = [
    # 1 - Reply to partner email about integration
    {
        "to_name": "Jordan Lee",
        "to_email": "jordan@partnerstack.io",
        "subject": "RE: NexusAI + PartnerStack integration timeline",
        "body_plain": (
            "Hi Jordan,\n\n"
            "Thanks for the detailed spec. The API integration looks straightforward\n"
            "on our end -- I think we can have a working prototype by end of next week.\n\n"
            "Two quick questions:\n"
            "1. Does your webhook payload include the partner_id field? We'll need that\n"
            "   for attribution tracking.\n"
            "2. What's the rate limit on the /referrals endpoint?\n\n"
            "I'll have our lead engineer Ravi reach out to your technical team to\n"
            "coordinate the sandbox environment.\n\n"
            "Best,\n"
            "Alex"
        ),
        "body_html": "",
        "category": "SENT",
        "age_range": (1, 7),
    },
    # 2 - Follow-up after meeting
    {
        "to_name": "Christina Vasquez",
        "to_email": "christina.v@sequoiacap.com",
        "subject": "Great meeting today - next steps",
        "body_plain": (
            "Hi Christina,\n\n"
            "Really enjoyed our conversation today. A few follow-ups:\n\n"
            "1. I'll send over the updated financial model by EOD Friday.\n"
            "   We're projecting $2.4M ARR by Q3 (up from our previous estimate).\n"
            "2. Happy to set up a technical deep dive with our CTO for your\n"
            "   team whenever works.\n"
            "3. Reference calls -- I'll connect you with two of our design\n"
            "   partners next week.\n\n"
            "Let me know if you need anything else before the partner meeting.\n\n"
            "Best,\n"
            "Alex Chen\n"
            "CEO, NexusAI"
        ),
        "body_html": "",
        "category": "SENT",
        "age_range": (0, 5),
    },
    # 3 - Intro connecting two people
    {
        "to_name": "Maya Singh",
        "to_email": "maya.singh@databricks.com",
        "subject": "Intro: Maya (Databricks) <> Ravi (NexusAI CTO)",
        "body_plain": (
            "Hi Maya,\n\n"
            "As promised, I'd love to connect you with Ravi Krishnan, our CTO.\n"
            "He's been exploring Databricks for our ML pipeline and had some\n"
            "specific questions about the Model Serving layer.\n\n"
            "Ravi, Maya runs the startup partnerships program at Databricks\n"
            "and has been incredibly helpful with our onboarding.\n\n"
            "I'll let you two take it from here!\n\n"
            "Alex\n\n"
            "CC: ravi@nexusai.com"
        ),
        "body_html": "",
        "category": "SENT",
        "age_range": (2, 10),
    },
    # 4 - Scheduling response
    {
        "to_name": "Marcus Tan",
        "to_email": "marcus@a16z.com",
        "subject": "RE: Follow-up call - NexusAI Series A",
        "body_plain": (
            "Hi Marcus,\n\n"
            "Either time works for me! Let's go with Thursday at 2pm PT.\n\n"
            "I'll send a Zoom link. Should I plan to cover the go-to-market\n"
            "strategy as well, or keep it focused on the technical differentiation?\n\n"
            "Looking forward to it.\n\n"
            "Alex"
        ),
        "body_html": "",
        "category": "SENT",
        "age_range": (0, 5),
    },
    # 5 - Investor update
    {
        "to_name": "Investors",
        "to_email": "investors@nexusai.com",
        "subject": "NexusAI Monthly Update - February 2026",
        "body_plain": (
            "Hi everyone,\n\n"
            "Here's the February update for NexusAI.\n\n"
            "HIGHLIGHTS:\n"
            "- MRR: $187K (up 23% MoM)\n"
            "- New customers: 14 (including 2 enterprise)\n"
            "- Churn: 1.8% (down from 2.4%)\n"
            "- Runway: 16 months at current burn\n\n"
            "PRODUCT:\n"
            "- Launched multi-agent orchestration feature\n"
            "- Shipped Slack integration (most requested feature)\n"
            "- Started SOC2 Type II audit\n\n"
            "TEAM:\n"
            "- Hired senior ML engineer from Google DeepMind\n"
            "- Opening head of sales role (referrals welcome!)\n\n"
            "ASKS:\n"
            "- Intros to heads of engineering at mid-market SaaS companies\n"
            "- Feedback on our new pricing page (link in deck)\n\n"
            "Full deck attached. Happy to jump on a call if anyone wants\n"
            "to go deeper on anything.\n\n"
            "Best,\n"
            "Alex"
        ),
        "body_html": "",
        "category": "SENT",
        "age_range": (1, 8),
    },
    # 6 - Thank you note after demo
    {
        "to_name": "Brian Cho",
        "to_email": "bcho@snowflake.com",
        "subject": "Thanks for the demo today!",
        "body_plain": (
            "Hi Brian,\n\n"
            "Thanks for taking the time to walk us through the Snowflake\n"
            "Cortex platform today. The team was really impressed with the\n"
            "native LLM function support.\n\n"
            "A few things we'd love to explore further:\n"
            "- Can we get access to the private preview for Cortex Fine-tuning?\n"
            "- What does pricing look like for the Analyst tier?\n\n"
            "I think there's a genuine integration opportunity between our\n"
            "agent framework and your data layer. Would be great to discuss.\n\n"
            "Thanks again,\n"
            "Alex"
        ),
        "body_html": "",
        "category": "SENT",
        "age_range": (0, 7),
    },
    # 7 - Quick question to team member
    {
        "to_name": "Ravi Krishnan",
        "to_email": "ravi@nexusai.com",
        "subject": "Quick question on the embedding model switch",
        "body_plain": (
            "Hey Ravi,\n\n"
            "Quick one -- did we end up going with text-embedding-3-large\n"
            "or the Cohere embed-v3? I need to update the pricing model\n"
            "for the board deck.\n\n"
            "Also, what's the latency delta between the two? I remember\n"
            "it was marginal but want exact numbers.\n\n"
            "Thanks!\n"
            "Alex"
        ),
        "body_html": "",
        "category": "SENT",
        "age_range": (0, 4),
    },
    # 8 - Reply to customer support escalation
    {
        "to_name": "Sarah Okafor",
        "to_email": "sarah.okafor@acmelogistics.com",
        "subject": "RE: Agent timeout issues on large document batches",
        "body_plain": (
            "Hi Sarah,\n\n"
            "I'm sorry about the timeout issues your team has been hitting.\n"
            "This is our top priority right now.\n\n"
            "Here's what we've done so far:\n"
            "- Identified a memory leak in the document chunking step\n"
            "- Deployed a hotfix to production this morning\n"
            "- Increased the batch processing timeout from 5min to 15min\n\n"
            "Could you try rerunning the failed batches and let us know\n"
            "if the issue persists? I've also added your account to our\n"
            "priority support tier -- no extra charge.\n\n"
            "I'll personally monitor this through the week.\n\n"
            "Best,\n"
            "Alex Chen\n"
            "CEO, NexusAI"
        ),
        "body_html": "",
        "category": "SENT",
        "age_range": (0, 3),
    },
    # 9 - Cold outreach to potential hire
    {
        "to_name": "Priya Mehta",
        "to_email": "priya.mehta.ml@gmail.com",
        "subject": "Loved your RLHF paper - are you open to chatting?",
        "body_plain": (
            "Hi Priya,\n\n"
            "I just read your NeurIPS paper on reward model calibration\n"
            "for RLHF -- really impressive work, especially the approach\n"
            "to handling distribution shift in the preference data.\n\n"
            "I'm the founder of NexusAI. We're building an agent platform\n"
            "and the problems you're working on are directly relevant to\n"
            "our core architecture.\n\n"
            "Would you be open to a 20-minute chat? No strings attached --\n"
            "even if you're happy where you are, I'd love to geek out\n"
            "about reward modeling.\n\n"
            "Alex Chen\n"
            "CEO, NexusAI"
        ),
        "body_html": "",
        "category": "SENT",
        "age_range": (3, 14),
    },
    # 10 - Vendor negotiation reply
    {
        "to_name": "Account Team",
        "to_email": "enterprise-sales@openai.com",
        "subject": "RE: NexusAI enterprise agreement renewal - pricing discussion",
        "body_plain": (
            "Hi team,\n\n"
            "Thanks for sending over the renewal proposal. A few thoughts:\n\n"
            "- The volume commitment of 50M tokens/month works for us.\n"
            "- The per-token rate of $0.008 is higher than what we discussed.\n"
            "  Can we revisit the $0.006 rate from the original term sheet?\n"
            "- We'd like to add GPT-4o access to the agreement as well.\n\n"
            "Our usage has grown 3x since the original contract, so I think\n"
            "there's a strong case for better pricing. Happy to jump on a\n"
            "call to discuss.\n\n"
            "Alex"
        ),
        "body_html": "",
        "category": "SENT",
        "age_range": (2, 10),
    },
    # 11 - Team announcement
    {
        "to_name": "All Hands",
        "to_email": "team@nexusai.com",
        "subject": "Welcome Priya to the team!",
        "body_plain": (
            "Hey team!\n\n"
            "Excited to announce that Priya Mehta is joining us as\n"
            "Senior ML Engineer starting March 17th!\n\n"
            "Priya comes from Google DeepMind where she worked on RLHF\n"
            "and reward modeling. Her NeurIPS paper on reward calibration\n"
            "is directly relevant to our agent optimization work.\n\n"
            "She'll be sitting with the ML team but will be working\n"
            "closely with everyone during onboarding.\n\n"
            "Please give her a warm welcome on her first day!\n\n"
            "Alex"
        ),
        "body_html": "",
        "category": "SENT",
        "age_range": (0, 5),
    },
    # 12 - Response to conference invite
    {
        "to_name": "Events Team",
        "to_email": "speakers@ai-summit.com",
        "subject": "RE: Speaking invitation - AI Summit 2026",
        "body_plain": (
            "Hi there,\n\n"
            "Thanks for the invitation! I'd love to speak at the AI Summit.\n\n"
            "For topic, I'd propose:\n"
            "\"From Chatbots to Agents: Building Reliable AI Systems That Actually Work\"\n\n"
            "It would cover practical lessons from building NexusAI's agent\n"
            "platform -- failure modes, evaluation strategies, and why most\n"
            "agent demos don't translate to production.\n\n"
            "Happy with either a 30-min talk or a panel format.\n"
            "Please let me know the next steps.\n\n"
            "Best,\n"
            "Alex Chen"
        ),
        "body_html": "",
        "category": "SENT",
        "age_range": (3, 15),
    },
    # 13 - Quick reply to cofounder
    {
        "to_name": "Ravi Krishnan",
        "to_email": "ravi@nexusai.com",
        "subject": "RE: Production incident - agent loop detected",
        "body_plain": (
            "On it. I'll jump into the war room now.\n\n"
            "Can you page Tina from infra? We might need to scale\n"
            "the circuit breaker timeout.\n\n"
            "Alex"
        ),
        "body_html": "",
        "category": "SENT",
        "age_range": (0, 2),
    },
    # 14 - Feedback on a design doc
    {
        "to_name": "Tina Wu",
        "to_email": "tina@nexusai.com",
        "subject": "RE: Design doc -- Agent Memory Architecture v2",
        "body_plain": (
            "Tina, this is excellent work. A few comments:\n\n"
            "1. Love the tiered memory approach (working / episodic / semantic).\n"
            "   This maps well to how we see agents evolving.\n\n"
            "2. On the eviction policy (Section 4.2) -- have you considered\n"
            "   using a relevance score decay instead of strict LRU? Our\n"
            "   agents sometimes need to recall old but important context.\n\n"
            "3. The cost estimate in Section 6 looks right. Let's budget\n"
            "   for 2x headroom though -- usage patterns are unpredictable.\n\n"
            "Let's discuss at the architecture review on Thursday.\n\n"
            "Alex"
        ),
        "body_html": "",
        "category": "SENT",
        "age_range": (1, 7),
    },
    # 15 - Personal thank you to mentor
    {
        "to_name": "David Huang",
        "to_email": "dhuang@stanford.edu",
        "subject": "Thank you for the advice last week",
        "body_plain": (
            "Hi David,\n\n"
            "Just wanted to say thanks again for the coffee last week.\n"
            "Your perspective on navigating the Series A process was\n"
            "incredibly helpful -- especially the advice about not\n"
            "optimizing for valuation at the seed stage.\n\n"
            "We ended up restructuring our pitch deck based on your\n"
            "feedback and the conversations have been much stronger.\n\n"
            "I'll keep you posted on how things develop. Lunch is on\n"
            "me next time!\n\n"
            "Gratefully,\n"
            "Alex"
        ),
        "body_html": "",
        "category": "SENT",
        "age_range": (3, 14),
    },
]
