"""Post-hoc cost recomputation for a rollout directory.

``recompute_rollout_cost(rollout_dir)`` reads per-turn usage from whichever
capture the run produced and prices it with :mod:`benchflow.pricing`:

1. ``trajectory/llm_trajectory.jsonl`` — LiteLLM proxy capture (API-key mode).
   One :class:`~benchflow.trajectories.types.LLMExchange` per line; usage comes
   from ``response.body.usage`` (provider convention, normalized by
   ``_exchange_token_usage``), the model from ``metadata.provider_model`` /
   ``response.body.model``.
2. ``trajectory/otel_usage.jsonl`` — OAuth / subscription mode capture written
   by :mod:`benchflow.otel_capture` from Claude Code's ``claude_code.api_request``
   telemetry events. One request per line with uncached ``input_tokens``,
   ``output_tokens``, ``cache_read_tokens``, ``cache_creation_tokens``, ``model``.
3. ``result.json`` aggregate — ``agent_result.n_*_tokens`` (whole-run totals) as
   the fallback when neither per-turn file exists. ``usage_source`` decides the
   input convention: ``provider_response`` is cache-inclusive (LiteLLM path),
   ``agent_native_acp`` is uncached (claude-agent-acp forwards the SDK's raw
   ``inputTokens``).

The result is a :class:`CostReport` with per-model token totals and cost, the
recorded ``final_metrics.total_cost_usd`` for comparison, and the price table
version. ``--update-result`` writes the recomputed cost back into
``result.json`` (``final_metrics.total_cost_usd``, ``agent_result.cost_usd``,
``agent_result.price_source``) keeping the previous values under
``agent_result.cost_history``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from benchflow.pricing import (
    LIST_PRICES,
    PriceTable,
    UnknownModelPrice,
    UsageCounts,
    canonical_model,
    compute_cost,
    price_table,
)

logger = logging.getLogger(__name__)

UsageSourceKind = Literal["llm_trajectory", "otel_usage", "result_aggregate"]
OTEL_USAGE_FILENAME = "otel_usage.jsonl"
LLM_TRAJECTORY_FILENAME = "llm_trajectory.jsonl"


@dataclass
class ModelCost:
    model: str
    n_records: int = 0
    usage: UsageCounts = field(default_factory=UsageCounts)
    cost_usd: float | None = None
    breakdown: dict[str, float] = field(default_factory=dict)
    unpriced: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "canonical_model": canonical_model(self.model),
            "n_records": self.n_records,
            "input_tokens_uncached": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "cache_read_tokens": self.usage.cache_read_tokens,
            "cache_creation_tokens": self.usage.cache_creation_tokens,
            "cache_creation_1h_tokens": self.usage.cache_creation_1h_tokens,
            "input_tokens_inclusive": self.usage.total_input_inclusive,
            "cost_usd": self.cost_usd,
            "breakdown_usd": self.breakdown,
            "unpriced": self.unpriced,
        }


@dataclass
class CostReport:
    rollout_dir: str
    source: UsageSourceKind
    source_path: str
    pricing_version: str
    n_records: int
    per_model: list[ModelCost]
    total_cost_usd: float | None
    recorded_cost_usd: float | None
    recorded_price_source: str | None
    recorded_usage_source: str | None
    unpriced_models: list[str]
    computed_at: str

    @property
    def delta_usd(self) -> float | None:
        if self.total_cost_usd is None or self.recorded_cost_usd is None:
            return None
        return self.total_cost_usd - self.recorded_cost_usd

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["per_model"] = [m.to_dict() for m in self.per_model]
        d["delta_usd"] = self.delta_usd
        d["usage_totals"] = sum(
            (m.usage for m in self.per_model), UsageCounts()
        ).__dict__
        return d


# ---------------------------------------------------------------- readers


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        logger.warning("unreadable JSON %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def usage_records_from_llm_trajectory(
    path: Path, *, default_model: str | None = None
) -> list[tuple[str, UsageCounts]]:
    """(model, usage) per successful exchange in ``llm_trajectory.jsonl``."""
    from benchflow.trajectories.types import LLMExchange, _exchange_token_usage

    records: list[tuple[str, UsageCounts]] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            ex = LLMExchange.model_validate_json(raw)
        except Exception as exc:  # malformed line: skip, do not abort
            logger.warning("skipping malformed %s line %d: %s", path.name, lineno, exc)
            continue
        body = ex.response.body if isinstance(ex.response.body, dict) else {}
        usage = body.get("usage")
        if not isinstance(usage, dict) or not any(
            v is not None for v in usage.values()
        ):
            continue  # failed call or no usage: nothing billed
        tu = _exchange_token_usage(ex)
        cc = usage.get("cache_creation")
        one_hour = 0
        if isinstance(cc, dict):
            try:
                one_hour = int(cc.get("ephemeral_1h_input_tokens") or 0)
            except (TypeError, ValueError):
                one_hour = 0
        counts = UsageCounts.from_inclusive_input(
            input_tokens_inclusive=tu.input_tokens,
            output_tokens=tu.output_tokens,
            cache_read_tokens=tu.cache_read_tokens,
            cache_creation_tokens=tu.cache_creation_tokens,
            cache_creation_1h_tokens=one_hour,
        )
        model = (
            ex.metadata.get("provider_model")
            or body.get("model")
            or ex.metadata.get("request_model")
            or ex.metadata.get("model_group")
            or (ex.request.body or {}).get("model")
            or default_model
            or "unknown"
        )
        records.append((str(model), counts))
    return records


def usage_records_from_otel_usage(
    path: Path, *, default_model: str | None = None
) -> list[tuple[str, UsageCounts]]:
    """(model, usage) per ``api_request`` record in ``otel_usage.jsonl``."""
    records: list[tuple[str, UsageCounts]] = []
    seen_request_ids: set[str] = set()
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("skipping malformed %s line %d: %s", path.name, lineno, exc)
            continue
        if not isinstance(rec, dict) or rec.get("event") not in (None, "api_request"):
            continue
        rid = rec.get("request_id")
        if isinstance(rid, str) and rid:
            if rid in seen_request_ids:
                continue  # duplicate export of the same request
            seen_request_ids.add(rid)
        counts = UsageCounts.from_anthropic_usage(rec)
        model = rec.get("model") or default_model or "unknown"
        records.append((str(model), counts))
    return records


def usage_records_from_result_aggregate(
    result: dict[str, Any],
) -> list[tuple[str, UsageCounts]]:
    """Whole-run totals from ``result.json`` as a single (model, usage) record."""
    ar = result.get("agent_result") or {}
    fm = result.get("final_metrics") or {}
    model = str(result.get("model") or ar.get("model") or "unknown")
    usage_source = ar.get("usage_source") or (result.get("usage_tracking") or {}).get(
        "usage_source"
    )
    n_in = int(ar.get("n_input_tokens") or fm.get("total_prompt_tokens") or 0)
    n_out = int(ar.get("n_output_tokens") or fm.get("total_completion_tokens") or 0)
    n_cr = int(ar.get("n_cache_read_tokens") or fm.get("total_cached_tokens") or 0)
    n_cw = int(ar.get("n_cache_creation_tokens") or 0)
    if n_in == 0 and n_out == 0:
        return []
    if usage_source == "agent_native_acp":
        # claude-agent-acp forwards the SDK's raw inputTokens (uncached).
        counts = UsageCounts(n_in, n_out, n_cr, n_cw)
    else:
        counts = UsageCounts.from_inclusive_input(
            input_tokens_inclusive=n_in,
            output_tokens=n_out,
            cache_read_tokens=n_cr,
            cache_creation_tokens=n_cw,
        )
    return [(model, counts)]


# ---------------------------------------------------------------- recompute


def detect_source(rollout_dir: Path) -> tuple[UsageSourceKind, Path]:
    traj = rollout_dir / "trajectory"
    if (traj / LLM_TRAJECTORY_FILENAME).is_file():
        return "llm_trajectory", traj / LLM_TRAJECTORY_FILENAME
    if (traj / OTEL_USAGE_FILENAME).is_file():
        return "otel_usage", traj / OTEL_USAGE_FILENAME
    return "result_aggregate", rollout_dir / "result.json"


def recompute_rollout_cost(
    rollout_dir: str | Path,
    *,
    pricing: str | PriceTable | None = None,
    source: UsageSourceKind | Literal["auto"] = "auto",
) -> CostReport:
    rollout_dir = Path(rollout_dir).expanduser()
    table = (
        pricing
        if isinstance(pricing, PriceTable)
        else price_table(pricing)
        if pricing
        else LIST_PRICES
    )
    result = _read_json(rollout_dir / "result.json") or {}
    default_model = result.get("model")

    if source == "auto":
        kind, path = detect_source(rollout_dir)
    else:
        kind = source
        path = {
            "llm_trajectory": rollout_dir / "trajectory" / LLM_TRAJECTORY_FILENAME,
            "otel_usage": rollout_dir / "trajectory" / OTEL_USAGE_FILENAME,
            "result_aggregate": rollout_dir / "result.json",
        }[kind]

    if kind == "llm_trajectory":
        records = (
            usage_records_from_llm_trajectory(path, default_model=default_model)
            if path.is_file()
            else []
        )
    elif kind == "otel_usage":
        records = (
            usage_records_from_otel_usage(path, default_model=default_model)
            if path.is_file()
            else []
        )
    else:
        records = usage_records_from_result_aggregate(result)

    per_model: dict[str, ModelCost] = {}
    for model, counts in records:
        mc = per_model.setdefault(model, ModelCost(model=model))
        mc.n_records += 1
        mc.usage = mc.usage + counts

    unpriced: list[str] = []
    total: float | None = 0.0
    for mc in per_model.values():
        try:
            bd = compute_cost(mc.model, mc.usage, table)
        except UnknownModelPrice:
            mc.unpriced = True
            unpriced.append(mc.model)
            continue
        mc.cost_usd = bd.total_usd
        mc.breakdown = {
            "input_usd": bd.input_usd,
            "output_usd": bd.output_usd,
            "cache_read_usd": bd.cache_read_usd,
            "cache_write_usd": bd.cache_write_usd,
        }
        total = (total or 0.0) + bd.total_usd
    if unpriced or not per_model:
        total = None if unpriced or not per_model else total

    ar = result.get("agent_result") or {}
    fm = result.get("final_metrics") or {}
    recorded = fm.get("total_cost_usd", ar.get("cost_usd"))
    return CostReport(
        rollout_dir=str(rollout_dir),
        source=kind,
        source_path=str(path),
        pricing_version=table.version,
        n_records=len(records),
        per_model=sorted(per_model.values(), key=lambda m: -(m.cost_usd or 0)),
        total_cost_usd=total,
        recorded_cost_usd=float(recorded)
        if isinstance(recorded, int | float)
        else None,
        recorded_price_source=ar.get("price_source"),
        recorded_usage_source=ar.get("usage_source")
        or (result.get("usage_tracking") or {}).get("usage_source"),
        unpriced_models=unpriced,
        computed_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def write_cost_report(
    report: CostReport, rollout_dir: str | Path | None = None
) -> Path:
    out_dir = Path(rollout_dir or report.rollout_dir)
    path = out_dir / "cost_recompute.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, default=str) + "\n")
    return path


def update_result_json(
    report: CostReport, rollout_dir: str | Path | None = None
) -> Path | None:
    """Write the recomputed cost into ``result.json``; keep the old values."""
    if report.total_cost_usd is None:
        return None
    path = Path(rollout_dir or report.rollout_dir) / "result.json"
    result = _read_json(path)
    if result is None:
        return None
    ar = result.setdefault("agent_result", {}) or {}
    fm = result.setdefault("final_metrics", {}) or {}
    history = list(ar.get("cost_history") or [])
    history.append(
        {
            "cost_usd": ar.get("cost_usd", fm.get("total_cost_usd")),
            "price_source": ar.get("price_source"),
            "replaced_at": report.computed_at,
        }
    )
    ar["cost_history"] = history
    ar["cost_usd"] = report.total_cost_usd
    ar["price_source"] = report.pricing_version
    ar["cost_source"] = report.source
    fm["total_cost_usd"] = report.total_cost_usd
    result["agent_result"], result["final_metrics"] = ar, fm
    path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    return path
