"""GitHub push protection's provider patterns, reproduced (the push-protection column of GitHub's
"Supported secret scanning patterns"), plus the key shapes BenchFlow's redactor tests use.

``tools/secret_sweep.py`` reads every pushed commit and the tree with them, and ``tools/keyshape_fix.py`` splits a
literal that matches one so the value is built at runtime. A hit names the rule, never the value. GitHub can add a
pattern at any time: refresh this list at each release (docs/contributing/release-checklist.md).
"""

from __future__ import annotations

import re

A = r"[A-Za-z0-9]"
PATTERNS: dict[str, str] = {
    # Stripe
    "stripe-secret-live": r"\b[sr]k_live_[A-Za-z0-9]{20,247}\b",
    "stripe-secret-test": r"\b[sr]k_test_[A-Za-z0-9]{20,247}\b",
    "stripe-webhook-secret": r"\bwhsec_[A-Za-z0-9]{32,}\b",
    # AWS
    "aws-access-key-id": r"\b(?:AKIA|ASIA|ABIA|ACCA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|APKA)[A-Z0-9]{16}\b",
    "aws-secret-key-assignment": r"(?i)aws.{0,20}?(?:secret|private).{0,20}?['\"=: ]{1,4}[A-Za-z0-9/+]{40}\b",
    # GitHub
    "github-token": r"\bgh[pousr]_[A-Za-z0-9]{36,255}\b",
    "github-fine-grained-pat": r"\bgithub_pat_[A-Za-z0-9_]{50,255}\b",
    # Slack
    "slack-token": r"\bxox[abposre]-[A-Za-z0-9-]{10,}\b",
    "slack-config-token": r"\bxoxe(?:\.xox[bp])?-\d-[A-Za-z0-9]{100,}\b",
    "slack-webhook": r"https://hooks\.slack\.com/(?:services|workflows|triggers)/T[A-Z0-9]+/[A-Z0-9]+/[A-Za-z0-9]{20,}",
    # Google
    "google-api-key": r"\bAIza[0-9A-Za-z_\-]{35}\b",
    "google-oauth-client-secret": r"\bGOCSPX-[0-9A-Za-z_\-]{28}\b",
    "google-oauth-access-token": r"\bya29\.[0-9A-Za-z_\-]{30,}\b",
    "gcp-service-account-key": r"\"private_key\"\s*:\s*\"-----BEGIN",
    # Model vendors
    "openai-project-key": r"\bsk-(?:proj|svcacct|admin)-[A-Za-z0-9_\-]{20,}\b",
    "openai-legacy-key": r"\bsk-[A-Za-z0-9]{20}T3BlbkFJ[A-Za-z0-9]{20}\b",
    "anthropic-key": r"\bsk-ant-(?:api|admin|oat|sid)\d\d-[A-Za-z0-9_\-]{20,}",
    "huggingface-token": r"\bhf_[A-Za-z]{34}\b",
    "groq-key": r"\bgsk_[A-Za-z0-9]{52}\b",
    "xai-key": r"\bxai-[A-Za-z0-9]{80}\b",
    "perplexity-key": r"\bpplx-[A-Za-z0-9]{48}\b",
    "replicate-token": r"\br8_[A-Za-z0-9]{37}\b",
    "langsmith-key": r"\blsv2_(?:pt|sk)_[a-f0-9]{32}_[a-f0-9]{10}\b",
    # Package registries, forges, CI
    "npm-token": r"\bnpm_[A-Za-z0-9]{36}\b",
    "pypi-token": r"\bpypi-AgEIcHlwaS5vcmc[A-Za-z0-9_\-]{50,}",
    "gitlab-token": r"\bgl(?:pat|dt|ptt|rt|cbt|imt|oas|soat|ffct|agent)-[A-Za-z0-9_\-]{20,}\b",
    "atlassian-token": r"\bATATT3[A-Za-z0-9_\-=]{150,}",
    "docker-pat": r"\bdckr_pat_[A-Za-z0-9_\-]{27,}\b",
    "databricks-token": r"\bdapi[a-f0-9]{32}(?:-\d)?\b",
    "pulumi-token": r"\bpul-[a-f0-9]{40}\b",
    "terraform-token": r"\b[A-Za-z0-9]{14}\.atlasv1\.[A-Za-z0-9_\-=]{60,}",
    "doppler-token": r"\bdp\.(?:pt|st|sa|ct|scim|audit)\.[A-Za-z0-9]{40,44}\b",
    "vercel-token": r"\b(?:vcp|vci|vca|vcr|vck)_[A-Za-z0-9]{24,}\b",
    "netlify-token": r"\bnfp_[A-Za-z0-9]{36}\b",
    "supabase-token": r"\bsbp_[a-f0-9]{40}\b",
    "linear-key": r"\blin_api_[A-Za-z0-9]{40}\b",
    "notion-token": r"\b(?:secret_[A-Za-z0-9]{43}|ntn_[A-Za-z0-9]{46})\b",
    "figma-token": r"\bfig[dpu]_[A-Za-z0-9_\-]{40,}\b",
    "postman-key": r"\bPMAK-[a-f0-9]{24}-[a-f0-9]{34}\b",
    "grafana-token": r"\b(?:glc_[A-Za-z0-9+/=]{32,}|glsa_[A-Za-z0-9]{32}_[a-f0-9]{8})",
    "newrelic-key": r"\bNR(?:AK|II|JS|AA)-[A-Za-z0-9]{27}\b",
    "sentry-token": r"\bsntr[yu]s_[A-Za-z0-9+/=_\-]{40,}",
    "dynatrace-token": r"\bdt0c01\.[A-Z0-9]{24}\.[A-Z0-9]{64}\b",
    # Messaging, payments, mail
    "twilio-key": r"\b(?:SK|AC)[a-f0-9]{32}\b",
    "sendgrid-key": r"\bSG\.[A-Za-z0-9_\-]{22}\.[A-Za-z0-9_\-]{43}\b",
    "mailgun-key": r"\bkey-[a-f0-9]{32}\b",
    "mailchimp-key": r"\b[a-f0-9]{32}-us\d{1,2}\b",
    "telegram-bot-token": r"\b\d{8,10}:AA[A-Za-z0-9_\-]{33}\b",
    "discord-bot-token": r"\b[MNO][A-Za-z0-9_\-]{23,25}\.[A-Za-z0-9_\-]{6}\.[A-Za-z0-9_\-]{27,38}\b",
    "square-token": r"\b(?:sq0atp|sq0csp|EAAA)[A-Za-z0-9_\-]{22,60}\b",
    "shopify-token": r"\bshp(?:at|ca|pa|ss)_[a-fA-F0-9]{32}\b",
    "paypal-braintree": r"\baccess_token\$production\$[a-z0-9]{16}\$[a-f0-9]{32}\b",
    # Clouds
    "digitalocean-token": r"\bdo[opr]_v1_[a-f0-9]{64}\b",
    "azure-storage-key": r"AccountKey=[A-Za-z0-9+/]{86}==",
    "azure-ad-client-secret": r"\b[A-Za-z0-9_~.]{3}\dQ~[A-Za-z0-9_~.\-]{31,34}\b",
    "azure-sas": r"\bsig=[A-Za-z0-9%]{43,64}%3D\b",
    "alibaba-key-id": r"\bLTAI[A-Za-z0-9]{12,20}\b",
    "cloudflare-token": r"\bcf(?:ut|at|k)_[A-Za-z0-9]{40,}\b",
    # Private keys
    "private-key-block": r"-----BEGIN[ A-Z0-9_-]{0,100}PRIVATE KEY(?: BLOCK)?-----",
}

RX: dict[str, re.Pattern[str]] = {k: re.compile(v) for k, v in PATTERNS.items()}


def scan_line(line: str) -> list[tuple[str, int, int, str]]:
    """``(rule, column, length, value)`` for each match in ``line`` (column from 1). Callers print the rule and
    the place, never the value."""
    out = []
    for name, rx in RX.items():
        for m in rx.finditer(line):
            out.append((name, m.start() + 1, len(m.group(0)), m.group(0)))
    return out
