"""Provider pricing metadata used for best-effort cost estimates.

The table is intentionally small and explicit. Price values should be reviewed
before they change; source metadata can be refreshed with:

    uv run python scripts/update_pricing_sources.py
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PricingEntry:
    """Per-million-token prices and source metadata for one model prefix."""

    input: float
    output: float
    cache_read: float
    cache_creation: float
    source_url: str
    retrieved_at: str
    source_hash: str

    @property
    def price_source(self) -> str:
        return f"{self.source_url}@{self.retrieved_at}#sha256:{self.source_hash[:12]}"


# Per-million token prices. Cache prices are separate because providers often
# discount cache reads and surcharge cache writes. Unknown models still get
# token telemetry but no cost estimate.
PRICING_USD_PER_MTOK: dict[str, PricingEntry] = {
    "claude-haiku-4-5": PricingEntry(
        input=1.0,
        output=5.0,
        cache_read=0.1,
        cache_creation=1.25,
        source_url="https://www.anthropic.com/pricing",
        retrieved_at="2026-05-18",
        source_hash="e7518460f64c8b3f9f2bb816fdb7c060c500d20d7f5da88b9dbfde5e08e8338e",
    ),
    "gpt-4.1-mini": PricingEntry(
        input=0.4,
        output=1.6,
        cache_read=0.1,
        cache_creation=0.4,
        source_url="https://openai.com/api/pricing/",
        retrieved_at="2026-05-18",
        source_hash="081d1608413e6a8984cf85721f637cf300c1511903aac34d28da8b821598a33d",
    ),
    "gemini-2.5-flash": PricingEntry(
        input=0.3,
        output=2.5,
        cache_read=0.075,
        cache_creation=0.3,
        source_url="https://ai.google.dev/gemini-api/docs/pricing",
        retrieved_at="2026-05-18",
        source_hash="2d296a04bef921506183ce909ebfd0091ae74c6c18f5d10b55d2f3fa8dc8653d",
    ),
}
