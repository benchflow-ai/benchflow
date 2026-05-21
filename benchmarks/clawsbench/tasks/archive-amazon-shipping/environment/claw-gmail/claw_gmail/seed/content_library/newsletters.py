"""Realistic newsletter and marketing email templates.

Based on what a startup founder / developer actually receives in Gmail.
The key design point: some newsletters land in CATEGORY_PROMOTIONS,
others in CATEGORY_UPDATES, and a few in CATEGORY_SOCIAL. This ambiguity
is intentional — it mirrors real Gmail classification and tests whether
agents can handle the nuance.

Distribution:
  ~20 CATEGORY_PROMOTIONS  (vendor marketing, SaaS product updates, event invites, ads)
  ~15 CATEGORY_UPDATES     (beehiiv / substack newsletters, community updates)
  ~5  CATEGORY_SOCIAL      (recruiter outreach, cold emails, intros)
"""

from __future__ import annotations

NEWSLETTERS: list[dict] = [
    # -----------------------------------------------------------------------
    # CATEGORY_PROMOTIONS  (~20)
    # -----------------------------------------------------------------------
    {
        "sender_name": "Stripe",
        "sender_email": "updates@e.stripe.com",
        "subject": "Pricing your AI product: lessons from 50 startups",
        "body_plain": (
            "Hi there,\n\n"
            "Pricing an AI product is fundamentally different from pricing traditional SaaS.\n\n"
            "We analyzed 50 startups on Stripe Billing and found three dominant patterns:\n\n"
            "1. Usage-based (tokens, API calls, compute minutes) - 42% of AI startups\n"
            "2. Outcome-based (per successful extraction, per resolved ticket) - 28%\n"
            "3. Hybrid seat + usage - 30%\n\n"
            "The median AI startup on Stripe processes $48K MRR within 9 months of launch.\n"
            "Companies using usage-based pricing see 23% higher net revenue retention.\n\n"
            "Read the full analysis on the Stripe blog:\n"
            "https://stripe.com/blog/pricing-ai-products\n\n"
            "-- The Stripe Team"
        ),
        "body_html": "",
        "category": "CATEGORY_PROMOTIONS",
        "is_read_probability": 0.6,
        "age_range": (1, 7),
    },
    {
        "sender_name": "Stripe",
        "sender_email": "updates@e.stripe.com",
        "subject": "Learn how Manus AI scaled to $100M ARR on Stripe",
        "body_plain": (
            "Case Study: Manus AI\n\n"
            "When Manus AI launched their autonomous agent platform in 2025,\n"
            "they needed billing infrastructure that could handle unpredictable\n"
            "workloads — some customers running 10 agents, others running 10,000.\n\n"
            "Key results after migrating to Stripe Billing:\n"
            "- 99.97% billing accuracy across 2.3M monthly invoices\n"
            "- 60% reduction in billing disputes\n"
            "- $100M ARR milestone reached in 14 months\n\n"
            "Their secret: metered billing with real-time usage streaming via\n"
            "the Stripe Meter Events API.\n\n"
            "Read the full case study: https://stripe.com/customers/manus-ai\n\n"
            "-- The Stripe Team"
        ),
        "body_html": "",
        "category": "CATEGORY_PROMOTIONS",
        "is_read_probability": 0.5,
        "age_range": (3, 14),
    },
    {
        "sender_name": "Replit",
        "sender_email": "contact@mail.replit.com",
        "subject": "Tomorrow: Build Lead Magnets with Replit Agent",
        "body_plain": (
            "Join us tomorrow for a live workshop!\n\n"
            "BUILD LEAD MAGNETS WITH REPLIT AGENT\n"
            "Wednesday, Feb 26 at 11:00 AM PST\n\n"
            "Learn how to use Replit Agent to build interactive calculators,\n"
            "ROI tools, and assessment quizzes that capture leads.\n\n"
            "What you'll build:\n"
            "- A custom pricing calculator deployed to a .replit.app URL\n"
            "- An email capture form with Resend integration\n"
            "- A Postgres-backed dashboard to track conversions\n\n"
            "No coding experience needed. Replit Agent handles the implementation\n"
            "while you focus on the business logic.\n\n"
            "Register now: https://replit.com/events/lead-magnets-workshop\n\n"
            "See you there,\n"
            "The Replit Team"
        ),
        "body_html": "",
        "category": "CATEGORY_PROMOTIONS",
        "is_read_probability": 0.3,
        "age_range": (1, 5),
    },
    {
        "sender_name": "Replit",
        "sender_email": "contact@mail.replit.com",
        "subject": "New Pro Plan & Core Agent Updates",
        "body_plain": (
            "Big updates to Replit this month:\n\n"
            "NEW PRO PLAN - $25/mo\n"
            "- Unlimited Replit Agent usage (was 10 sessions/day)\n"
            "- 50 GB storage per Repl\n"
            "- Always-on deployments included\n"
            "- Priority access to new models\n\n"
            "CORE AGENT UPDATES\n"
            "- Agent can now scaffold full-stack Next.js + Supabase apps\n"
            "- New 'explain' mode: ask Agent to walk through any codebase\n"
            "- 3x faster deployment pipeline\n"
            "- Git integration: Agent creates proper commits with messages\n\n"
            "Upgrade to Pro: https://replit.com/pricing\n\n"
            "Happy building,\n"
            "The Replit Team"
        ),
        "body_html": "",
        "category": "CATEGORY_PROMOTIONS",
        "is_read_probability": 0.5,
        "age_range": (5, 21),
    },
    {
        "sender_name": "Google AI Studio",
        "sender_email": "googleaistudio-noreply@google.com",
        "subject": "Gemini 3.1 Pro: A smarter model for complex tasks",
        "body_plain": (
            "Introducing Gemini 3.1 Pro\n\n"
            "Today we're releasing Gemini 3.1 Pro in Google AI Studio and\n"
            "the Gemini API. Key improvements:\n\n"
            "- 2x longer context window: now supports 4M tokens\n"
            "- 40% improvement on MMLU-Pro benchmark (92.1%)\n"
            "- Native tool use with structured output guarantees\n"
            "- Grounding with Google Search built-in\n"
            "- New pricing: $1.25 / 1M input tokens, $5.00 / 1M output tokens\n\n"
            "Available today in 230+ countries. Existing Gemini 3.0 Pro\n"
            "users can switch with a single model parameter change.\n\n"
            "Try it now: https://aistudio.google.com\n"
            "Read the technical report: https://ai.google/research/gemini-3-1\n\n"
            "-- Google AI Team"
        ),
        "body_html": "",
        "category": "CATEGORY_PROMOTIONS",
        "is_read_probability": 0.7,
        "age_range": (2, 10),
    },
    {
        "sender_name": "Lambda",
        "sender_email": "marketing@lambda.ai",
        "subject": "Lambda Cloud newsletter: February 2026",
        "body_plain": (
            "LAMBDA CLOUD - FEBRUARY 2026\n\n"
            "New GPU availability:\n"
            "- NVIDIA B200 instances now GA in us-east-1 and eu-west-1\n"
            "- On-demand pricing: $4.89/hr per B200 (was $5.49)\n"
            "- Reserved instances: $2.99/hr with 6-month commitment\n\n"
            "Community highlights:\n"
            "- Tutorial: Fine-tuning Llama 4 70B on Lambda Cloud with DeepSpeed\n"
            "- Benchmark: B200 vs H100 for inference — 2.8x throughput improvement\n"
            "- Customer story: How Perplexity reduced serving costs by 60%\n\n"
            "Platform updates:\n"
            "- New: Persistent storage (up to 10TB per instance)\n"
            "- New: Multi-node training with 1-click NCCL setup\n"
            "- Fix: JupyterLab auto-save now works with large notebooks\n\n"
            "Happy training,\n"
            "The Lambda Team"
        ),
        "body_html": "",
        "category": "CATEGORY_PROMOTIONS",
        "is_read_probability": 0.4,
        "age_range": (1, 14),
    },
    {
        "sender_name": "Google Workspace",
        "sender_email": "googleworkspace-noreply@google.com",
        "subject": "Get up to $1,500 in Google Ads credit",
        "body_plain": (
            "Grow your business with Google Ads\n\n"
            "As a Google Workspace customer, you're eligible for up to\n"
            "$1,500 in Google Ads credit when you start a new campaign.\n\n"
            "Here's how it works:\n"
            "- Spend $500 in the first 60 days -> get $500 credit\n"
            "- Spend $1,000 -> get $1,000 credit\n"
            "- Spend $1,500 -> get $1,500 credit (maximum)\n\n"
            "Popular campaign types for startups:\n"
            "1. Search ads targeting competitor keywords\n"
            "2. YouTube pre-roll for product demos\n"
            "3. Performance Max for lead generation\n\n"
            "Claim your credit: https://ads.google.com/workspace-offer\n"
            "Offer expires March 31, 2026.\n\n"
            "-- Google Workspace Team"
        ),
        "body_html": "",
        "category": "CATEGORY_PROMOTIONS",
        "is_read_probability": 0.2,
        "age_range": (3, 21),
    },
    {
        "sender_name": "Vanta",
        "sender_email": "vantateam@vanta.com",
        "subject": "Still re-proving trust every quarter?",
        "body_plain": (
            "Hi,\n\n"
            "If your team is still scrambling to collect evidence for every\n"
            "security questionnaire, there's a better way.\n\n"
            "Vanta's Trust Center lets you:\n"
            "- Auto-share your SOC 2 report with prospects (gated by NDA)\n"
            "- Pre-fill security questionnaires with AI in under 5 minutes\n"
            "- Track which prospects viewed your compliance docs\n\n"
            "Companies using Vanta close deals 30% faster because buyers\n"
            "get instant access to the security info they need.\n\n"
            "350+ integrations including AWS, GitHub, Okta, and Datadog.\n\n"
            "Start a free trial: https://vanta.com/trust-center\n\n"
            "Best,\n"
            "The Vanta Team"
        ),
        "body_html": "",
        "category": "CATEGORY_PROMOTIONS",
        "is_read_probability": 0.2,
        "age_range": (5, 30),
    },
    {
        "sender_name": "Pilot",
        "sender_email": "hithere@pilot.com",
        "subject": "Give $500, get $500 when you share Pilot",
        "body_plain": (
            "Love Pilot? Share the love.\n\n"
            "For every founder you refer who signs up for Pilot bookkeeping,\n"
            "you both get a $500 credit toward your next month.\n\n"
            "Why founders choose Pilot:\n"
            "- Dedicated bookkeeper who knows startups\n"
            "- Monthly close within 15 business days\n"
            "- Tax-ready financials for investors\n"
            "- R&D tax credit identification included\n\n"
            "Over 2,000 startups trust Pilot with their books,\n"
            "including companies backed by Sequoia, a16z, and YC.\n\n"
            "Share your referral link: https://pilot.com/refer/abc123\n\n"
            "-- Team Pilot"
        ),
        "body_html": "",
        "category": "CATEGORY_PROMOTIONS",
        "is_read_probability": 0.15,
        "age_range": (7, 30),
    },
    {
        "sender_name": "Ben (Strong Compute)",
        "sender_email": "ben=strongcompute.com@hubspotstarter.ap1.hs-send.com",
        "subject": "Cost tracking for all GPU clouds",
        "body_plain": (
            "Hey,\n\n"
            "We just shipped something I think you'll find useful: unified\n"
            "cost tracking across every GPU cloud provider.\n\n"
            "If you're training on Lambda + CoreWeave + your own on-prem\n"
            "cluster, you've probably lost track of real per-token costs.\n\n"
            "Strong Compute Dashboard now shows:\n"
            "- Real-time spend across all providers in one view\n"
            "- Per-experiment cost breakdown (training, eval, inference)\n"
            "- Automatic spot vs on-demand optimization suggestions\n"
            "- Forecasting: predicted monthly spend based on current usage\n\n"
            "We've seen teams save 20-40% just by switching workloads\n"
            "to the cheapest available provider in real time.\n\n"
            "Try it free: https://strongcompute.com/dashboard\n\n"
            "Cheers,\n"
            "Ben @ Strong Compute"
        ),
        "body_html": "",
        "category": "CATEGORY_PROMOTIONS",
        "is_read_probability": 0.3,
        "age_range": (2, 14),
    },
    {
        "sender_name": "Venture Forum",
        "sender_email": "john-ventureforum.net@shared1.ccsend.com",
        "subject": "Reminder: Investor Meeting - March 6th, SF",
        "body_plain": (
            "VENTURE FORUM - MARCH INVESTOR MEETING\n\n"
            "Date: Thursday, March 6, 2026\n"
            "Time: 6:00 PM - 9:00 PM PST\n"
            "Location: The Vault SF, 415 Jackson St, San Francisco\n\n"
            "This month's presenting companies:\n"
            "1. NeuralForge - AI-powered code generation (Seed, raising $3M)\n"
            "2. ClimateLens - Satellite analytics for ESG compliance (Series A, $12M)\n"
            "3. MediBot - Autonomous surgical robotics (Series B, $40M)\n\n"
            "Confirmed investors attending:\n"
            "- Partners from Sequoia, Greylock, Founders Fund\n"
            "- Angels from the ex-Stripe, ex-Google network\n\n"
            "RSVP required. Space limited to 80 attendees.\n"
            "Register: https://ventureforum.net/march-2026\n\n"
            "See you there,\n"
            "John @ Venture Forum"
        ),
        "body_html": "",
        "category": "CATEGORY_PROMOTIONS",
        "is_read_probability": 0.4,
        "age_range": (1, 10),
    },
    {
        "sender_name": "AI Council",
        "sender_email": "community@aicouncil.com",
        "subject": "Last Day to Save $700 - AI Council Summit 2026",
        "body_plain": (
            "EARLY BIRD PRICING ENDS TONIGHT\n\n"
            "AI Council Summit 2026\n"
            "April 14-16, 2026 | Moscone Center, San Francisco\n\n"
            "Early bird: $899 (regular $1,599) -- SAVE $700\n"
            "Use code: EARLYAI26\n\n"
            "Confirmed speakers:\n"
            "- Dario Amodei, CEO of Anthropic\n"
            "- Jensen Huang, CEO of NVIDIA\n"
            "- Mira Murati, CEO of Thinking Machines\n"
            "- Demis Hassabis, CEO of Google DeepMind\n\n"
            "Tracks: Foundation Models, AI Safety, Enterprise AI,\n"
            "Autonomous Agents, Robotics, AI Policy\n\n"
            "3,000+ attendees. 200+ speakers. 50+ workshops.\n\n"
            "Register now: https://aicouncil.com/summit-2026\n\n"
            "-- AI Council Team"
        ),
        "body_html": "",
        "category": "CATEGORY_PROMOTIONS",
        "is_read_probability": 0.4,
        "age_range": (1, 5),
    },
    {
        "sender_name": "Cassie at Devpost",
        "sender_email": "cassie@devpost.com",
        "subject": "HACKATHONS just for you, March 2026",
        "body_plain": (
            "Hey there!\n\n"
            "Here are this month's top hackathons matched to your interests:\n\n"
            "1. Google Gemini API Challenge\n"
            "   Prize pool: $100,000 | Deadline: Mar 15\n"
            "   Build with Gemini 3.1 Pro, Vertex AI, or AI Studio\n\n"
            "2. Anthropic Tool Use Hackathon\n"
            "   Prize pool: $50,000 | Deadline: Mar 22\n"
            "   Create agents that use MCP tools effectively\n\n"
            "3. AWS Agentic AI Challenge\n"
            "   Prize pool: $75,000 | Deadline: Mar 31\n"
            "   Build autonomous agents on Amazon Bedrock\n\n"
            "4. Replit x Cloudflare Buildathon\n"
            "   Prize pool: $25,000 | Deadline: Apr 5\n"
            "   Deploy AI apps with Workers AI + Replit Agent\n\n"
            "See all hackathons: https://devpost.com/hackathons\n\n"
            "Happy hacking!\n"
            "Cassie @ Devpost"
        ),
        "body_html": "",
        "category": "CATEGORY_PROMOTIONS",
        "is_read_probability": 0.35,
        "age_range": (1, 10),
    },
    {
        "sender_name": "Openmart",
        "sender_email": "kathryn@updates.openmart.ai",
        "subject": "Openmart v2: AI-powered lead enrichment is here",
        "body_plain": (
            "Hi there,\n\n"
            "Big product update from Openmart.\n\n"
            "We just launched v2 of our AI-powered lead database with\n"
            "automatic enrichment from 40+ data sources.\n\n"
            "What's new:\n"
            "- Enrichment: technographics, funding data, hiring signals\n"
            "- AI Scoring: predict which SMBs will buy, not just who matches\n"
            "- Integrations: push leads to HubSpot, Salesforce, or webhook\n"
            "- Coverage: 12M+ US small businesses (up from 8M)\n\n"
            "Early customers are seeing 3x reply rates on outbound\n"
            "by targeting businesses with active buying signals.\n\n"
            "Try it: https://openmart.ai/v2\n\n"
            "Best,\n"
            "Kathryn @ Openmart"
        ),
        "body_html": "",
        "category": "CATEGORY_PROMOTIONS",
        "is_read_probability": 0.25,
        "age_range": (3, 14),
    },
    {
        "sender_name": "Sensible",
        "sender_email": "jason@sensible.so",
        "subject": "2026 guide to the IDP landscape",
        "body_plain": (
            "Hi,\n\n"
            "We just published our 2026 Intelligent Document Processing\n"
            "landscape report. Free download, no gating.\n\n"
            "Key findings:\n"
            "- The IDP market hit $4.2B in 2025 (up 38% YoY)\n"
            "- LLM-based extraction now outperforms template-based on 73% of doc types\n"
            "- Top use cases: insurance claims, mortgage processing, invoice extraction\n"
            "- Average implementation time dropped from 6 months to 3 weeks\n\n"
            "We compared 14 IDP vendors across accuracy, speed, and pricing.\n"
            "Spoiler: Sensible's LLM extraction ranked #1 on accuracy for\n"
            "semi-structured documents (invoices, receipts, contracts).\n\n"
            "Download the report: https://sensible.so/idp-landscape-2026\n\n"
            "Best,\n"
            "Jason @ Sensible"
        ),
        "body_html": "",
        "category": "CATEGORY_PROMOTIONS",
        "is_read_probability": 0.2,
        "age_range": (5, 21),
    },
    {
        "sender_name": "Anyscale",
        "sender_email": "productnewsletter@anyscale.com",
        "subject": "How Robotics Teams Run AI at Scale with Ray",
        "body_plain": (
            "ANYSCALE PRODUCT NEWSLETTER - FEBRUARY 2026\n\n"
            "FEATURE: How robotics teams run AI at scale\n\n"
            "Boston Dynamics, Figure, and Covariant all use Ray to train\n"
            "their foundation models for robotics. Here's the common pattern:\n\n"
            "1. Simulation at scale: 10,000+ parallel envs with Ray RLlib\n"
            "2. Multi-modal training: vision + proprioception + language\n"
            "3. Distributed eval: test policies across 500 scenarios simultaneously\n\n"
            "Average cluster: 256 GPUs, 90%+ utilization with Ray autoscaler.\n\n"
            "NEW IN RAY 3.2:\n"
            "- Ray Data: 5x faster for image/video preprocessing\n"
            "- Ray Serve: vLLM integration for 3x inference throughput\n"
            "- Ray Train: native DeepSpeed ZeRO-3 support\n\n"
            "Read more: https://anyscale.com/blog/robotics-at-scale\n\n"
            "-- Anyscale Team"
        ),
        "body_html": "",
        "category": "CATEGORY_PROMOTIONS",
        "is_read_probability": 0.35,
        "age_range": (3, 14),
    },
    {
        "sender_name": "Shortwave",
        "sender_email": "aloha@shortwave.com",
        "subject": "Tasklet Instant Apps + Claude Sonnet & Opus 4.6 support",
        "body_plain": (
            "What's new in Shortwave\n\n"
            "TASKLET INSTANT APPS\n"
            "Turn any email into an interactive workflow. When Shortwave AI\n"
            "detects an actionable email (invoice, event invite, approval\n"
            "request), it generates a Tasklet — a mini app right in your inbox.\n\n"
            "Examples:\n"
            "- Invoice email -> one-click approve/reject + auto-file to accounting\n"
            "- Calendar invite -> see conflicts, suggest alternatives, accept\n"
            "- PR review request -> see diff stats, approve, merge from email\n\n"
            "NEW MODEL SUPPORT\n"
            "- Claude Sonnet 4.5 and Opus 4.6 now available as AI backend\n"
            "- Gemini 3.1 Pro support coming next week\n"
            "- Choose your model per-workspace in Settings > AI\n\n"
            "Try Shortwave: https://shortwave.com\n\n"
            "Mahalo,\n"
            "The Shortwave Team"
        ),
        "body_html": "",
        "category": "CATEGORY_PROMOTIONS",
        "is_read_probability": 0.5,
        "age_range": (1, 7),
    },
    {
        "sender_name": "Industrious",
        "sender_email": "hello@email.industriousoffice.com",
        "subject": "Your private office awaits: 2 months free",
        "body_plain": (
            "Hi,\n\n"
            "Looking for a workspace that works as hard as you do?\n\n"
            "Industrious is offering 2 months free on 12-month agreements\n"
            "at our SF and NYC locations.\n\n"
            "What's included:\n"
            "- Private office for 1-20 people\n"
            "- Premium amenities: espresso bar, phone booths, wellness rooms\n"
            "- IT support and enterprise-grade WiFi (1 Gbps symmetric)\n"
            "- Access to all 200+ Industrious locations nationwide\n"
            "- Community events: founder dinners, investor office hours\n\n"
            "Locations available:\n"
            "- One Embarcadero Center, SF (3 offices left)\n"
            "- 175 Varick St, NYC (5 offices left)\n"
            "- 600 Congress Ave, Austin (available immediately)\n\n"
            "Schedule a tour: https://industriousoffice.com/tour\n\n"
            "Best,\n"
            "The Industrious Team"
        ),
        "body_html": "",
        "category": "CATEGORY_PROMOTIONS",
        "is_read_probability": 0.15,
        "age_range": (3, 21),
    },
    {
        "sender_name": "Vapi",
        "sender_email": "team@mail.vapi.ai",
        "subject": "Building Vapi agents just got 10x faster",
        "body_plain": (
            "Hey builders,\n\n"
            "We just shipped the biggest update to Vapi since launch.\n\n"
            "NEW: VISUAL AGENT BUILDER\n"
            "Design voice agent flows with a drag-and-drop canvas.\n"
            "Connect nodes for greeting, intent detection, tool calls,\n"
            "and handoff — then deploy with one click.\n\n"
            "NEW: FUNCTION CALLING V2\n"
            "- Parallel function execution (3x faster for multi-step tasks)\n"
            "- Built-in retry with exponential backoff\n"
            "- Real-time streaming of function results to the caller\n\n"
            "NEW: ANALYTICS DASHBOARD\n"
            "- Call duration, completion rate, sentiment by intent\n"
            "- Transcript search across all calls\n"
            "- Export to BigQuery for custom analysis\n\n"
            "Start building: https://vapi.ai/dashboard\n\n"
            "Ship it,\n"
            "The Vapi Team"
        ),
        "body_html": "",
        "category": "CATEGORY_PROMOTIONS",
        "is_read_probability": 0.35,
        "age_range": (2, 10),
    },
    {
        "sender_name": "Claw via beehiiv",
        "sender_email": "claw@mail.beehiiv.com",
        "subject": "You are invited to AI+ Summit SF, March 20",
        "body_plain": (
            "YOU'RE INVITED\n\n"
            "AI+ SUMMIT SAN FRANCISCO\n"
            "March 20, 2026 | 9:00 AM - 6:00 PM\n"
            "The Pearl SF, 601 19th Street\n\n"
            "300 founders, investors, and AI engineers.\n"
            "Single-track. No sponsor talks. Real conversations.\n\n"
            "Speakers confirmed:\n"
            "- Harrison Chase (LangChain) - Building Production Agents\n"
            "- Amjad Masad (Replit) - Code Generation at Scale\n"
            "- Sarah Guo (Conviction) - What VCs Actually Look For in AI Startups\n"
            "- Kanjun Qiu (Imbue) - Agents That Reason\n\n"
            "Tickets: $149 (founders) / $249 (general)\n"
            "Apply: https://lu.ma/ai-plus-summit-sf-march\n\n"
            "See you there."
        ),
        "body_html": "",
        "category": "CATEGORY_PROMOTIONS",
        "is_read_probability": 0.5,
        "age_range": (1, 10),
    },
    {
        "sender_name": "Luma",
        "sender_email": "maya@user.luma-mail.com",
        "subject": "You are invited to Autonomous Business Hackathon",
        "body_plain": (
            "Maya Chen invited you to:\n\n"
            "AUTONOMOUS BUSINESS HACKATHON\n"
            "March 8-9, 2026 (48 hours)\n"
            "Shack15, San Francisco\n\n"
            "Build an autonomous business that runs itself.\n"
            "Teams of 2-4. $25K in prizes.\n\n"
            "Sponsors: Anthropic, Vercel, Supabase, Resend\n\n"
            "Prize categories:\n"
            "- Best fully autonomous agent ($10K)\n"
            "- Most revenue generated during hackathon ($8K)\n"
            "- Best use of MCP tools ($5K)\n"
            "- Audience choice ($2K)\n\n"
            "Free API credits from all sponsors.\n"
            "Meals provided. Limited to 100 hackers.\n\n"
            "RSVP: https://lu.ma/autonomous-biz-hack\n"
        ),
        "body_html": "",
        "category": "CATEGORY_PROMOTIONS",
        "is_read_probability": 0.45,
        "age_range": (1, 7),
    },

    # -----------------------------------------------------------------------
    # CATEGORY_UPDATES  (~15)
    # -----------------------------------------------------------------------
    {
        "sender_name": "Turing Post",
        "sender_email": "turingpost@mail.beehiiv.com",
        "subject": "What is Agentic RL and why it matters",
        "body_plain": (
            "TURING POST - WEEKLY DEEP DIVE\n\n"
            "This week: Agentic Reinforcement Learning\n\n"
            "Standard RL optimizes for a reward signal. Agentic RL optimizes\n"
            "for an agent's ability to accomplish open-ended goals in\n"
            "environments it has never seen before.\n\n"
            "The key insight from DeepMind's latest paper (Feb 2026):\n"
            "training on diverse environments with sparse rewards produces\n"
            "agents that generalize better than dense-reward specialists.\n\n"
            "Three approaches gaining traction:\n"
            "1. Environment-agnostic pretraining (HAIRL from Berkeley)\n"
            "2. Reward model distillation (Anthropic's Constitutional RL)\n"
            "3. Hindsight relabeling at scale (OpenAI's approach)\n\n"
            "The practical impact: foundation agents that can handle new\n"
            "tools and APIs without fine-tuning. Companies like Cognition\n"
            "and Factory are already shipping products built on this.\n\n"
            "Read the full analysis with paper links on our site.\n\n"
            "-- Turing Post"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.55,
        "age_range": (1, 7),
    },
    {
        "sender_name": "Turing Post",
        "sender_email": "turingpost@mail.beehiiv.com",
        "subject": "The Scaling Wall: are we really hitting limits?",
        "body_plain": (
            "TURING POST - WEEKLY DEEP DIVE\n\n"
            "This week: The Scaling Debate\n\n"
            "In January, two narratives collided. Ilya Sutskever said\n"
            "'the age of scaling is over.' A week later, Anthropic shipped\n"
            "Opus 4.6, the most capable model ever, trained on more compute\n"
            "than any previous model.\n\n"
            "So which is it?\n\n"
            "The nuance: pre-training scaling may be plateauing for pure\n"
            "next-token prediction. But three new scaling dimensions\n"
            "are opening up:\n\n"
            "1. Test-time compute (thinking longer, not training bigger)\n"
            "2. Agentic RL (learning from environment interaction)\n"
            "3. Synthetic data loops (self-improvement through practice)\n\n"
            "We interviewed 8 ML researchers for this piece. The consensus:\n"
            "scaling isn't dead, it's evolving. The cost curve is just\n"
            "shifting from training to inference.\n\n"
            "Full article with interviews on our site.\n\n"
            "-- Turing Post"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.6,
        "age_range": (3, 14),
    },
    {
        "sender_name": "Founders Creative",
        "sender_email": "founderscreative@mail.beehiiv.com",
        "subject": "The AI Inner Circle: who's really building the future",
        "body_plain": (
            "FOUNDERS CREATIVE - ISSUE #47\n\n"
            "THE AI INNER CIRCLE\n\n"
            "There's a group of ~200 people who show up in every AI story.\n"
            "They hop between Anthropic, OpenAI, Google DeepMind, and startups\n"
            "like musical chairs. They all know each other. They all went to\n"
            "the same 5 schools.\n\n"
            "This week we mapped the network:\n\n"
            "- 67% have PhDs from Stanford, MIT, CMU, Berkeley, or Toronto\n"
            "- Average time at one company: 18 months\n"
            "- Median angel portfolio: 12 AI startups\n"
            "- Most connected person: a researcher who's co-authored papers\n"
            "  with people at all 6 major labs\n\n"
            "What this means for founders:\n"
            "1. Warm intros matter more than cold outreach\n"
            "2. Conference hallway conversations close deals\n"
            "3. The talent pool for senior AI roles is shockingly small\n\n"
            "Full network visualization and analysis on our site.\n\n"
            "-- Founders Creative"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.5,
        "age_range": (2, 10),
    },
    {
        "sender_name": "AI+ Founders Community",
        "sender_email": "aiplusfounderscommunity@substack.com",
        "subject": "AI+ Renaissance Conference: applications now open",
        "body_plain": (
            "AI+ FOUNDERS COMMUNITY\n\n"
            "ANNOUNCING: AI+ RENAISSANCE CONFERENCE\n"
            "May 15-16, 2026 | Brooklyn, NY\n\n"
            "We're hosting our first in-person conference focused on the\n"
            "intersection of AI and creative industries.\n\n"
            "Tracks:\n"
            "1. AI x Music (generative composition, rights management)\n"
            "2. AI x Film (pre-viz, VFX, script analysis)\n"
            "3. AI x Design (brand generation, spatial computing)\n"
            "4. AI x Writing (collaborative fiction, journalism tools)\n\n"
            "Each track features a live demo hour where speakers build\n"
            "something on stage with AI tools.\n\n"
            "Speaker applications open now. Attendee tickets: $79 early bird.\n\n"
            "Apply: https://aiplus.community/renaissance\n\n"
            "Deadline: April 1, 2026\n\n"
            "-- AI+ Community Team"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.4,
        "age_range": (3, 14),
    },
    {
        "sender_name": "Cities Decoded",
        "sender_email": "samlicitiesdecoded@substack.com",
        "subject": "The future of AI and the law: who owns what?",
        "body_plain": (
            "CITIES DECODED - ISSUE #112\n\n"
            "THE FUTURE OF AI AND THE LAW\n\n"
            "Three landmark cases this quarter are reshaping AI copyright:\n\n"
            "1. NYT v. OpenAI: Settled for $200M + content licensing deal.\n"
            "   OpenAI will pay per-article training fees going forward.\n\n"
            "2. Getty v. Stability AI: Jury found 'substantial similarity'\n"
            "   in 14% of tested outputs. Damages TBD.\n\n"
            "3. Doe v. Anthropic: First case testing whether AI-generated\n"
            "   code can be copyrighted. Ruling expected by April.\n\n"
            "The practical impact for startups:\n"
            "- Budget 5-15% of training costs for licensing\n"
            "- Document your training data provenance now\n"
            "- Consider training on permissive-license data only for v1\n\n"
            "The EU AI Act enforcement begins June 2026. US regulation\n"
            "is still a patchwork of state laws.\n\n"
            "Full analysis with case links on our site.\n\n"
            "-- Sam @ Cities Decoded"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.4,
        "age_range": (2, 14),
    },
    {
        "sender_name": "EntreConnect",
        "sender_email": "entreconnect@substack.com",
        "subject": "From Scroll to Sale: how AI founders sell on social",
        "body_plain": (
            "ENTRECONNECT - WEEKLY\n\n"
            "FROM SCROLL TO SALE\n\n"
            "We interviewed 15 AI founders doing >$1M ARR about their\n"
            "social media strategy. The playbook is surprisingly consistent:\n\n"
            "LinkedIn (not Twitter) is the #1 channel for B2B AI startups.\n\n"
            "The formula that works:\n"
            "1. Post a specific result (not a feature announcement)\n"
            "   'Our customer reduced invoice processing from 4 hours to 8 minutes'\n"
            "2. Show the before/after in a carousel or video\n"
            "3. End with a soft CTA (DM me for the case study)\n\n"
            "Numbers from the top performers:\n"
            "- Average 3 posts per week\n"
            "- 40% of inbound pipeline comes from LinkedIn\n"
            "- Best posting time: Tuesday 8-9 AM in your buyer's timezone\n"
            "- Founders personal accounts outperform company pages 5:1\n\n"
            "The full playbook with templates is on our site.\n\n"
            "-- EntreConnect"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.35,
        "age_range": (1, 10),
    },
    {
        "sender_name": "Sunday Swervice",
        "sender_email": "sundayswervice@substack.com",
        "subject": "Black Futures, Fashion Week, and AI-generated everything",
        "body_plain": (
            "SUNDAY SWERVICE - ISSUE #89\n\n"
            "Happy Sunday.\n\n"
            "This week's swerves:\n\n"
            "BLACK FUTURES\n"
            "The Afrofuturism exhibit at MoMA PS1 uses AI-generated\n"
            "sculptures that respond to viewer movement. Artist Rashida\n"
            "Holmes trained a model on 10,000 pieces of African art\n"
            "spanning 3,000 years. The result is haunting and beautiful.\n\n"
            "FASHION WEEK\n"
            "Coperni showed a collection designed entirely by AI.\n"
            "Creative director Arnaud Vaillant used Midjourney + CLO 3D\n"
            "to generate 40 looks, then manufactured 12 for the runway.\n"
            "The internet is divided. The clothes are fire.\n\n"
            "AI-GENERATED EVERYTHING\n"
            "Sony announced AI-composed film scores. A24 is using AI\n"
            "for script coverage. Even Pixar admitted to using AI\n"
            "for background generation. The creative industry is\n"
            "in full adoption mode, not resistance mode.\n\n"
            "Links and hot takes on our site.\n\n"
            "-- Sunday Swervice"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.45,
        "age_range": (1, 7),
    },
    {
        "sender_name": "Stabledash",
        "sender_email": "stabledashnewsletter@mail.beehiiv.com",
        "subject": "Win a $10K KAST Gold Card + our picks for March",
        "body_plain": (
            "STABLEDASH NEWSLETTER\n\n"
            "GIVEAWAY: $10K KAST GOLD CARD\n"
            "We partnered with KAST to give away a $10,000 prepaid card.\n"
            "Enter by sharing this newsletter with 3 friends.\n"
            "Drawing: March 15, 2026.\n\n"
            "OUR PICKS FOR MARCH:\n\n"
            "1. Best AI tool: Cursor Composer (still undefeated)\n"
            "   The inline edit + chat + terminal combo is chef's kiss.\n\n"
            "2. Best new startup: Paradigm (stealth, YC W26)\n"
            "   AI-native accounting. Actually understands GAAP.\n\n"
            "3. Best funding round: Cartesia ($100M Series B)\n"
            "   State-space models are the real competition to transformers.\n\n"
            "4. Best open source: BrowserBase MCP Server\n"
            "   Give any LLM a real browser. Production-ready.\n\n"
            "Full reviews on our site.\n\n"
            "-- Stabledash"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.3,
        "age_range": (1, 10),
    },
    {
        "sender_name": "CoffeeSpace",
        "sender_email": "hazim@coffeespace.com",
        "subject": "From cofounders to founding teams: what we learned",
        "body_plain": (
            "Hey,\n\n"
            "Quick update from CoffeeSpace.\n\n"
            "We've matched 2,400 cofounders in the last 6 months.\n"
            "Here's what we learned from the successful matches:\n\n"
            "WHAT WORKS:\n"
            "- Technical + technical founders (not technical + business)\n"
            "  73% of our successful pairings are two engineers\n"
            "- Founders who meet in person within 7 days of matching\n"
            "  are 4x more likely to start a company together\n"
            "- Shared problem obsession > shared skill set\n\n"
            "WHAT DOESN'T:\n"
            "- Equity splits discussed before product-market fit\n"
            "- More than 3 cofounders (failure rate doubles)\n"
            "- Remote-first cofounder relationships before building trust\n\n"
            "NEW FEATURE: Founding Team Mixer events in SF, NYC, London.\n"
            "3 hours, 12 structured 1:1s, followed by optional dinner.\n\n"
            "Next mixer: March 12, SF. RSVP: https://coffeespace.com/mixer\n\n"
            "-- Hazim @ CoffeeSpace"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.4,
        "age_range": (2, 14),
    },
    {
        "sender_name": "Founders Creative",
        "sender_email": "founderscreative@mail.beehiiv.com",
        "subject": "Why your landing page isn't converting (data from 500 startups)",
        "body_plain": (
            "FOUNDERS CREATIVE - ISSUE #48\n\n"
            "WHY YOUR LANDING PAGE ISN'T CONVERTING\n\n"
            "We analyzed 500 startup landing pages. The median conversion\n"
            "rate is 2.1%. The top 10% convert at 8.4%.\n\n"
            "The 5 biggest differences:\n\n"
            "1. HERO COPY: Top performers use specific outcomes.\n"
            "   Bad: 'AI-powered analytics platform'\n"
            "   Good: 'Find revenue leaks in 5 minutes'\n\n"
            "2. SOCIAL PROOF: Logo bars don't work anymore.\n"
            "   What works: inline testimonials next to features.\n\n"
            "3. CTA: 'Start free trial' converts 34% better than 'Get started'\n\n"
            "4. SPEED: Pages loading in <2s convert 2x those loading in 4s+\n\n"
            "5. DEMO VIDEO: 45-second product video above the fold\n"
            "   increases conversion by 22%\n\n"
            "Full data and examples on our site.\n\n"
            "-- Founders Creative"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.5,
        "age_range": (5, 14),
    },
    {
        "sender_name": "AI+ Founders Community",
        "sender_email": "aiplusfounderscommunity@substack.com",
        "subject": "Weekly Digest: top posts from the AI+ community",
        "body_plain": (
            "AI+ FOUNDERS COMMUNITY - WEEKLY DIGEST\n\n"
            "Top posts this week:\n\n"
            "1. 'I spent $80K on GPU compute and all I got was this 7B model'\n"
            "   by @marcus_ai - Real talk about the economics of training\n"
            "   small models vs. using API providers. TL;DR: unless you need\n"
            "   custom architecture, fine-tuning is almost always cheaper.\n\n"
            "2. 'Our agent hallucinated a customer refund'\n"
            "   by @sarah_builds - Post-mortem on an agent that issued\n"
            "   $12,000 in unauthorized refunds. Root cause: no confirmation\n"
            "   step before executing financial actions.\n\n"
            "3. 'The MCP protocol will eat the API economy'\n"
            "   by @devtools_dan - Why Model Context Protocol changes\n"
            "   everything about how agents interact with services.\n\n"
            "4. AMA with Kanjun Qiu (Imbue CEO) - transcript now available\n\n"
            "Join the conversation: https://aiplus.community\n\n"
            "-- AI+ Community"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.45,
        "age_range": (1, 7),
    },
    {
        "sender_name": "Cities Decoded",
        "sender_email": "samlicitiesdecoded@substack.com",
        "subject": "San Francisco is back (the data proves it)",
        "body_plain": (
            "CITIES DECODED - ISSUE #113\n\n"
            "SAN FRANCISCO IS BACK\n\n"
            "I know, I know. Everyone says this. But this time we have data.\n\n"
            "Office vacancy rate: down to 28% from 35% peak (still bad, but trending)\n"
            "VC deals in SF: $18.2B in 2025, up 42% from 2024\n"
            "Net migration: positive for first time since 2020\n"
            "Restaurant openings: 340 new restaurants in 2025 (vs 180 in 2023)\n\n"
            "What's driving it:\n"
            "1. AI companies are SF-first. Anthropic, OpenAI, Databricks,\n"
            "   Scale AI — all headquartered here, all hiring aggressively.\n"
            "2. Rent is 25% below 2019 peak. Young founders can afford it.\n"
            "3. The doom narrative created a contrarian opportunity.\n\n"
            "The vibe shift is real. Hayes Valley on a Saturday feels like\n"
            "2018 again. SOMA is still rough, but improving.\n\n"
            "Full analysis with charts on our site.\n\n"
            "-- Sam @ Cities Decoded"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.55,
        "age_range": (2, 10),
    },
    {
        "sender_name": "EntreConnect",
        "sender_email": "entreconnect@substack.com",
        "subject": "The $0 to $1M playbook for AI dev tools",
        "body_plain": (
            "ENTRECONNECT - WEEKLY\n\n"
            "THE $0 TO $1M PLAYBOOK FOR AI DEV TOOLS\n\n"
            "We tracked 30 AI dev tool startups from launch to $1M ARR.\n"
            "The fastest took 4 months. The median: 11 months.\n\n"
            "The playbook they all followed:\n\n"
            "MONTH 1-2: BUILD IN PUBLIC\n"
            "- Ship a free open-source version first\n"
            "- Post daily updates on Twitter/X with GIFs\n"
            "- Target a single, specific pain point\n\n"
            "MONTH 3-4: CAPTURE DEMAND\n"
            "- Launch on Product Hunt (still works for dev tools)\n"
            "- Write technical blog posts targeting long-tail SEO\n"
            "- Offer a hosted version with a generous free tier\n\n"
            "MONTH 5-8: CONVERT TO PAID\n"
            "- Team features (SSO, shared workspaces, audit logs)\n"
            "- Usage limits on free tier (not feature limits)\n"
            "- Annual pricing with 2 months free\n\n"
            "MONTH 9-12: EXPAND\n"
            "- Enterprise tier with SLA and support\n"
            "- Integrations with the tools teams already use\n\n"
            "Full breakdown with revenue curves on our site.\n\n"
            "-- EntreConnect"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.45,
        "age_range": (3, 14),
    },
    {
        "sender_name": "Sunday Swervice",
        "sender_email": "sundayswervice@substack.com",
        "subject": "Oscars recap, AI music lawsuits, and the vibe shift",
        "body_plain": (
            "SUNDAY SWERVICE - ISSUE #90\n\n"
            "Happy Sunday.\n\n"
            "OSCARS RECAP\n"
            "The big story wasn't the winners — it was the AI elephant\n"
            "in the room. Best Visual Effects went to a film that used\n"
            "AI-assisted compositing. The acceptance speech didn't mention it.\n"
            "Hollywood's don't-ask-don't-tell era for AI has begun.\n\n"
            "AI MUSIC LAWSUITS\n"
            "Universal Music sued Suno and Udio for $150M each.\n"
            "Meanwhile, Grimes launched a platform where anyone can make\n"
            "AI songs using her voice and split royalties 50/50.\n"
            "Two futures of AI music, playing out simultaneously.\n\n"
            "THE VIBE SHIFT\n"
            "Something changed this month. The AI skeptic crowd went quiet.\n"
            "Not because they changed their minds — because the products\n"
            "got undeniably useful. Cursor, Claude Code, Replit Agent.\n"
            "The gap between 'demo' and 'daily tool' finally closed.\n\n"
            "Links and takes on our site.\n\n"
            "-- Sunday Swervice"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.5,
        "age_range": (1, 7),
    },
    {
        "sender_name": "Stabledash",
        "sender_email": "stabledashnewsletter@mail.beehiiv.com",
        "subject": "The MCP ecosystem is exploding: our top 10 servers",
        "body_plain": (
            "STABLEDASH NEWSLETTER\n\n"
            "THE MCP ECOSYSTEM IS EXPLODING\n\n"
            "Model Context Protocol went from 'interesting spec' to\n"
            "'industry standard' in about 3 months. There are now 400+\n"
            "MCP servers on the registry. Here are our top 10:\n\n"
            "1. Playwright MCP - Browser automation for agents\n"
            "2. GitHub MCP - Full repo management from Claude\n"
            "3. PostgreSQL MCP - Query and manage databases\n"
            "4. Stripe MCP - Payments and billing management\n"
            "5. Slack MCP - Read, search, and post messages\n"
            "6. Google Calendar MCP - Event management\n"
            "7. Linear MCP - Issue tracking and project management\n"
            "8. Figma MCP - Design file reading and generation\n"
            "9. Notion MCP - Document and wiki management\n"
            "10. BrowserBase MCP - Headless browser in the cloud\n\n"
            "The killer pattern: chaining 3-4 MCP servers together\n"
            "to create workflows that would take humans hours.\n\n"
            "Full reviews on our site.\n\n"
            "-- Stabledash"
        ),
        "body_html": "",
        "category": "CATEGORY_UPDATES",
        "is_read_probability": 0.55,
        "age_range": (1, 7),
    },

    # -----------------------------------------------------------------------
    # CATEGORY_SOCIAL  (~5)
    # -----------------------------------------------------------------------
    {
        "sender_name": "Jessica Wu",
        "sender_email": "jessica.wu@techrecruit.io",
        "subject": "Your project on GitHub caught my eye",
        "body_plain": (
            "Hi there,\n\n"
            "I came across your open-source project on GitHub and was\n"
            "really impressed by the architecture. The way you structured\n"
            "the agent orchestration layer is exactly the kind of thinking\n"
            "we see in top-tier ML engineers.\n\n"
            "I'm working with a well-funded AI startup (Series B, $80M)\n"
            "that's building autonomous coding agents. They're looking\n"
            "for a founding engineer to lead their agent infrastructure.\n\n"
            "Comp range: $280K-$350K base + 0.5-1.0% equity.\n"
            "Fully remote, US-based. Team of 15, all ex-FAANG/top labs.\n\n"
            "Would you be open to a 15-minute chat this week? No pressure,\n"
            "just want to share more about what they're building.\n\n"
            "Best,\n"
            "Jessica Wu\n"
            "Senior Technical Recruiter\n"
            "TechRecruit.io"
        ),
        "body_html": "",
        "category": "CATEGORY_SOCIAL",
        "is_read_probability": 0.6,
        "age_range": (1, 10),
    },
    {
        "sender_name": "Daniel Park",
        "sender_email": "daniel@foundernetwork.co",
        "subject": "Intro: fellow YC founder working on adjacent problem",
        "body_plain": (
            "Hey!\n\n"
            "Got your email from a mutual friend (Sarah at Anthropic).\n\n"
            "I'm Daniel, founder of ContextQL (YC W25). We're building\n"
            "a semantic search layer for enterprise knowledge bases.\n\n"
            "I noticed you're working in the agent tooling space and\n"
            "thought there might be some interesting overlap. Several\n"
            "of our customers have asked about agent-driven search\n"
            "and I keep pointing them to your work.\n\n"
            "Would love to grab coffee and compare notes. I'm in SF\n"
            "most weeks. How's Thursday or Friday this week?\n\n"
            "No agenda — just two founders building in the same space\n"
            "trading war stories.\n\n"
            "Best,\n"
            "Daniel Park\n"
            "Founder, ContextQL (YC W25)"
        ),
        "body_html": "",
        "category": "CATEGORY_SOCIAL",
        "is_read_probability": 0.7,
        "age_range": (1, 5),
    },
    {
        "sender_name": "Marcus Thompson",
        "sender_email": "marcus@bloomvc.com",
        "subject": "Impressed by your demo at the AI meetup",
        "body_plain": (
            "Hi,\n\n"
            "I was at the SF AI Builders meetup last Tuesday and saw\n"
            "your demo. Really compelling stuff — especially the part\n"
            "where the agent recovered from a failed API call without\n"
            "any explicit error handling code.\n\n"
            "I'm a partner at Bloom Ventures. We write $500K-$2M checks\n"
            "into pre-seed and seed AI infrastructure companies.\n\n"
            "I don't know if you're raising or even thinking about it,\n"
            "but I'd love to learn more about what you're building.\n"
            "Happy to be helpful regardless — we have a good network\n"
            "of AI founders and potential design partners.\n\n"
            "Free for coffee next week?\n\n"
            "Best,\n"
            "Marcus Thompson\n"
            "Partner, Bloom Ventures"
        ),
        "body_html": "",
        "category": "CATEGORY_SOCIAL",
        "is_read_probability": 0.8,
        "age_range": (1, 7),
    },
    {
        "sender_name": "Rachel Kim",
        "sender_email": "rachel.kim@enterprise-corp.com",
        "subject": "Exploring AI agent solutions for our operations team",
        "body_plain": (
            "Hello,\n\n"
            "I'm the VP of Operations at Enterprise Corp (Fortune 500,\n"
            "manufacturing sector). We're exploring AI agent platforms\n"
            "to automate our procurement and vendor management workflows.\n\n"
            "Currently our team processes ~2,000 purchase orders per month\n"
            "manually. We've evaluated UiPath and Automation Anywhere but\n"
            "find them too rigid for our use case.\n\n"
            "Your product came up in a Gartner briefing last week.\n"
            "I'd like to schedule a 30-minute call to understand:\n\n"
            "1. Can your agents handle unstructured PO documents?\n"
            "2. Do you support SAP integration?\n"
            "3. What's your pricing model for enterprise?\n"
            "4. SOC 2 Type II compliance status?\n\n"
            "We have budget allocated for Q2 and are targeting a\n"
            "decision by end of March.\n\n"
            "Best regards,\n"
            "Rachel Kim\n"
            "VP Operations, Enterprise Corp"
        ),
        "body_html": "",
        "category": "CATEGORY_SOCIAL",
        "is_read_probability": 0.85,
        "age_range": (1, 5),
    },
    {
        "sender_name": "Alex Rivera",
        "sender_email": "alex.rivera@techcrunch.com",
        "subject": "Working on a story about AI agent startups",
        "body_plain": (
            "Hi there,\n\n"
            "I'm a reporter at TechCrunch covering the AI infrastructure\n"
            "and developer tools beat.\n\n"
            "I'm working on a piece about the emerging AI agent ecosystem\n"
            "— specifically the tooling layer (frameworks, testing,\n"
            "deployment) that's forming around autonomous agents.\n\n"
            "Your name came up in multiple conversations with VCs and\n"
            "other founders in the space. Would you be willing to do\n"
            "a brief interview (20 min, on or off the record, your choice)?\n\n"
            "Some questions I'm exploring:\n"
            "- How do you test agents before deploying to production?\n"
            "- What's the biggest unsolved problem in the agent stack?\n"
            "- Is the current 'MCP everything' trend sustainable?\n\n"
            "Piece is slated for publication next Thursday.\n\n"
            "Thanks,\n"
            "Alex Rivera\n"
            "Reporter, TechCrunch"
        ),
        "body_html": "",
        "category": "CATEGORY_SOCIAL",
        "is_read_probability": 0.9,
        "age_range": (1, 5),
    },
]
