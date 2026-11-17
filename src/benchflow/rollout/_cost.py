"""One pricing step: every trial with tokens gets a cost.

A run around BenchFlow's model proxy (a Claude or ChatGPT login, the native
harness, an agent that routes its own calls) has tokens and, before that change,
no cost unless Claude Code's own session log could be priced.
:func:`apply_list_price` prices every trial that has tokens and no cost, at
list prices (:mod:`benchflow._utils.pricing`), and labels every cost:
``cost_source`` is ``provider`` for a provider's billed figure, else
``list_price_estimate``, with the price table's date (``price_table_date``).
It never fails the rollout.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Iterable
from typing import Any

from benchflow._utils.pricing import (
    SNAPSHOT_TABLE,
    TokenSlice,
    list_rates,
    price_slices,
)
from benchflow.usage_tracking import (
    COST_SOURCE_LIST_PRICE,
    COST_SOURCE_PROVIDER,
    USAGE_END_OF_TURN,
    USAGE_SESSION_LOG,
    USAGE_SOURCE_AGENT_NATIVE_ACP,
    USAGE_SOURCE_AGENT_REPORTED,
    USAGE_SOURCE_PROVIDER_RESPONSE,
    USAGE_STREAM_LOG,
    is_token_usage_available,
)

logger = logging.getLogger(__name__)

#: ``price_source`` of a cost this module set (tokens at list prices).
LIST_PRICE_SOURCE = "list_price"
#: ``price_source`` of the provider's own billed figure (an OpenRouter
#: response's ``usage.cost``, summed by the gateway).
PROVIDER_PRICE_SOURCE = "provider"


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


#: ``usage_incomplete_reason``: why a trial's tokens (and the cost priced
#: from them) are those of some of its calls, not all.
INCOMPLETE_UNREPORTED = (
    "a turn ended without its usage report and no log of the agent's could count it"
)
INCOMPLETE_MIRROR = (
    "counted from the copy of the agent's session log made while its turns "
    "ran, which had not caught up when the sandbox was lost"
)
INCOMPLETE_CODEX_ACP = (
    "codex-acp reports each turn's last model call only, and no Codex session "
    "log was left to count the others"
)
INCOMPLETE_TURN_CUT = (
    "turn cut on a subscription route; a call in flight may be missing"
)
#: Counts of the calls an agent made itself, on a login or a native CLI:
#: its end-of-turn report, or its own logs.
_SUBSCRIPTION_SOURCES = frozenset(
    {
        USAGE_SOURCE_AGENT_NATIVE_ACP,
        USAGE_STREAM_LOG,
        USAGE_SESSION_LOG,
        USAGE_END_OF_TURN,
    }
)
INCOMPLETE_REDACTION = (
    "the agent's session log could not be redacted ({error}), so no copy of it "
    "was kept and the calls only it records were not counted"
)
#: An in-sandbox model proxy's log that may miss calls: read at the end of
#: the agent's turn, the proxy was not running, a request had no record yet,
#: or the log read back was not the one the proxy counted; or the
#: log could not be read at all, and only what the live capture read while
#: the turn ran is counted.
#: An in-place branch child whose calls went through a model proxy: the
#: proxy's log is not read into the child's record (its full import is
#: so the child's count is the agent's own reports, which can miss
#: calls.
INCOMPLETE_BRANCH_CHILD_PROXY = (
    "this in-place branch child's calls went through the model proxy, whose log "
    "is not read into the child's record: its count is the agent's own reports, "
    "which can miss calls"
)
INCOMPLETE_SANDBOX_PROXY = (
    "the in-sandbox model proxy's usage log may miss calls: {detail}"
)


def redaction_failure(rollout: Any) -> str | None:
    """The exception class a copy of the agent's session log failed its
    redaction with, when the trial's count needed that copy: the teardown
    copy's, or the mirror's when no teardown copy was kept."""
    failed = getattr(rollout, "_teardown_redaction_failed", None)
    if failed:
        return failed
    if getattr(rollout, "_session_log_kept", False):
        return None
    return getattr(rollout, "_mirror_redaction_failed", None) or None


def mark_incomplete_usage(rollout: Any) -> None:
    """Mark a count that misses calls: ``usage_incomplete`` with
    ``usage_incomplete_reason``, so a partial count (and its cost) is never
    read as the trial's whole.

    A count step that knows its count is short says why
    (``rollout._usage_incomplete_reason``: a session-log copy that had not
    caught up, codex-acp's last-call reports with no log). Otherwise a turn
    that ended without its usage report (a timeout, a cancel, a lost
    sandbox) and that no log of the agent's counted leaves the trial with
    the tokens of the turns that reported. A turn cut that way on a login
    or a native CLI leaves a count a floor even when a log counted it: a
    call in flight is in no log (``INCOMPLETE_TURN_CUT``; an agent error
    is no cut). A proxy-only trial is complete:
    the proxy counted every call it forwarded, unless it saw a stream cut
    before the provider's usage report, which the proxy's import marks
    (``litellm_logging._mark_interrupted_streams``). Never fails
    the rollout.
    """
    try:
        metrics = getattr(rollout, "_usage_metrics", None) or {}
        error = redaction_failure(rollout)
        if error is not None:
            # A copy skipped for its redaction is never silent, whatever
            # else was counted, tokens or none.
            reason = INCOMPLETE_REDACTION.format(error=error)
            if metrics.get("usage_incomplete_reason") != reason:
                rollout._usage_metrics = {
                    **metrics,
                    "usage_incomplete": True,
                    "usage_incomplete_reason": reason,
                }
                logger.warning("The trial's token count is incomplete: %s", reason)
            return
        proxy = getattr(rollout, "_proxy_usage_incomplete_reason", None)
        if isinstance(proxy, str):
            # Whatever counted the tokens, and when nothing did: the proxy's
            # log was the trial's record of its calls. A reason another step
            # gave (already marked, or a count step's) is kept before it.
            held = (
                metrics.get("usage_incomplete_reason")
                if metrics.get("usage_incomplete")
                else None
            ) or getattr(rollout, "_usage_incomplete_reason", None)
            if isinstance(held, str) and held:
                reason = held if proxy in held else f"{held}; {proxy}"
            else:
                reason = proxy
            if (
                metrics.get("usage_incomplete") is not True
                or metrics.get("usage_incomplete_reason") != reason
            ):
                rollout._usage_metrics = {
                    **metrics,
                    "usage_incomplete": True,
                    "usage_incomplete_reason": reason,
                }
                logger.warning("The trial's token count is incomplete: %s", reason)
            return
        if not is_token_usage_available(metrics) or metrics.get("usage_incomplete"):
            return
        reason = getattr(rollout, "_usage_incomplete_reason", None)
        if reason is None:
            details = metrics.get("usage_details") or {}
            proxied_only = (
                metrics.get("usage_source") == USAGE_SOURCE_PROVIDER_RESPONSE
                and USAGE_SOURCE_AGENT_NATIVE_ACP not in details
            )
            if (
                getattr(rollout, "_usage_report_missing", False)
                and not getattr(rollout, "_usage_salvaged", False)
                and not proxied_only
            ):
                reason = INCOMPLETE_UNREPORTED
            elif getattr(rollout, "_turn_cut", None) and (
                metrics.get("usage_source") in _SUBSCRIPTION_SOURCES
                or (
                    metrics.get("usage_source") == USAGE_SOURCE_PROVIDER_RESPONSE
                    and USAGE_SOURCE_AGENT_NATIVE_ACP in details
                )
            ):
                # A turn cut (a timeout, a cancel, a budget stop, a lost
                # sandbox) on a login or a native CLI: the agent's log keeps
                # only the responses that ended, so a call in flight is in
                # neither it nor the count.
                reason = INCOMPLETE_TURN_CUT
        if reason is None:
            return
        rollout._usage_metrics = {
            **metrics,
            "usage_incomplete": True,
            "usage_incomplete_reason": reason,
        }
        logger.warning("The trial's token count is incomplete: %s", reason)
    except Exception as exc:  # never fail the rollout over a marker
        logger.info("Could not mark the trial's count incomplete: %s", exc)


def _fallback_model(rollout: Any) -> str:
    """The model to price an end-of-turn count with when its role named
    none: the one model the agent's own log says answered, else the run's."""
    reported = list(
        dict.fromkeys(getattr(rollout, "_session_reported_models", None) or [])
    )
    if len(reported) == 1:
        return reported[0]
    return getattr(rollout._config, "model", None) or ""


def _end_of_turn_slices(rollout: Any, *, unproxied_only: bool) -> list[TokenSlice]:
    """The trial's end-of-turn counts, one slice per model (input without
    the cache, as every report gives it)."""
    slices = []
    for (model, unproxied), counts in (
        getattr(rollout, "_native_usage_slices", None) or {}
    ).items():
        if unproxied_only and not unproxied:
            continue
        slices.append(
            TokenSlice(
                model=model or _fallback_model(rollout),
                input=counts.get("input_tokens", 0),
                output=counts.get("output_tokens", 0),
                cache_read=counts.get("cached_read_tokens", 0),
                cache_write=counts.get("cached_write_tokens", 0),
            )
        )
    return slices


def _snapshot_date_for(models: Iterable[str]) -> str | None:
    """The snapshot's date when every model is priced from it, else None."""
    from benchflow.providers.pricing_snapshot import SNAPSHOT_DATE

    names = [m for m in models if m]
    if not names:
        return None
    tables = []
    for name in names:
        rates = list_rates(name)
        tables.append(rates.table if rates else None)
    return (
        SNAPSHOT_DATE if tables and all(t == SNAPSHOT_TABLE for t in tables) else None
    )


def _label_cost(rollout: Any, metrics: dict[str, Any]) -> dict[str, Any]:
    """``cost_source`` and ``price_table_date`` for a cost someone else set."""
    if metrics.get("cost_source") is not None:
        return metrics
    source = metrics.get("price_source")
    labelled = dict(metrics)
    if source == PROVIDER_PRICE_SOURCE:
        labelled.update(cost_source=COST_SOURCE_PROVIDER, price_table_date=None)
        return labelled
    labelled["cost_source"] = COST_SOURCE_LIST_PRICE
    if labelled.get("price_table_date") is None:
        estimate = (metrics.get("usage_details") or {}).get("cost_estimate") or {}
        if source == "litellm":
            models = list(getattr(rollout, "_proxy_reported_models", None) or []) or [
                getattr(rollout._config, "model", None) or ""
            ]
            labelled["price_table_date"] = _snapshot_date_for(models)
        elif estimate.get("method") == "usage-at-list-price":
            labelled["price_table_date"] = _snapshot_date_for(
                (estimate.get("models") or {}).keys()
            )
    return labelled


def apply_list_price(rollout: Any) -> None:
    """Price a trial that has tokens and no cost; label every cost.

    A cost already set (the gateway's, Claude Code's session-log estimate)
    keeps its figure and gets ``cost_source`` and ``price_table_date``.
    Otherwise the trial's tokens -- salvaged from a log, or the end-of-turn
    reports, per model -- are priced at list prices. A trial whose proxied
    calls the gateway priced and whose login calls it could not see (a
    mixed trial) costs the sum of the two. A trial with no tokens, or with a
    model no table prices, keeps a null cost.
    """
    try:
        metrics = dict(getattr(rollout, "_usage_metrics", None) or {})
        if metrics.get("cost_usd") is not None:
            rollout._usage_metrics = _label_cost(rollout, metrics)
            return
        details = dict(metrics.get("usage_details") or {})
        mixed = isinstance(details.get(USAGE_SOURCE_PROVIDER_RESPONSE), dict)
        provider_usd: float | None = None
        if metrics.get("usage_source") == USAGE_SOURCE_AGENT_REPORTED:
            # An agent that routes its own model calls: its own count, priced
            # as reported (still never pooled with metered trials).
            if not _int(metrics.get("total_tokens")):
                return
            slices = [
                TokenSlice(
                    model=_fallback_model(rollout),
                    input=_int(metrics.get("n_input_tokens")),
                    output=_int(metrics.get("n_output_tokens")),
                    cache_read=_int(metrics.get("n_cache_read_tokens")),
                    cache_write=_int(metrics.get("n_cache_creation_tokens")),
                )
            ]
        elif not is_token_usage_available(metrics):
            return
        elif (
            metrics.get("usage_source") == USAGE_SOURCE_PROVIDER_RESPONSE and not mixed
        ):
            # The gateway priced what it could; a model no table it reads
            # prices is no model a bare-name guess should price (a private
            # deployment of an open model is not its vendor's API).
            return
        elif mixed:
            provider_usd = details[USAGE_SOURCE_PROVIDER_RESPONSE].get("cost_usd")
            if provider_usd is None:
                return  # the proxied part has no price: the sum would be short
            slices = _end_of_turn_slices(rollout, unproxied_only=True)
        else:
            slices = getattr(rollout, "_usage_slices", None) or _end_of_turn_slices(
                rollout, unproxied_only=False
            )
        priced = price_slices(list(slices))
        if priced is None:
            logger.info(
                "No list price for the trial's model(s) %s: its cost stays unknown",
                sorted({piece.model for piece in slices}),
            )
            return
        usd = priced.usd + (provider_usd or 0.0)
        details["cost_estimate"] = {
            "source": "benchflow-list-price",
            "method": "usage-at-list-price",
            "models": priced.models,
            "tables": priced.tables,
            **({"provider_response_usd": provider_usd} if mixed else {}),
        }
        table_date = priced.table_date
        proxied_models = getattr(rollout, "_proxy_reported_models", None) or []
        if mixed and table_date != _snapshot_date_for(proxied_models):
            table_date = None  # the two parts were priced from different tables
        rollout._usage_metrics = {
            **metrics,
            "cost_usd": round(usd, 10),
            "price_source": LIST_PRICE_SOURCE,
            "cost_source": COST_SOURCE_LIST_PRICE,
            "price_table_date": table_date,
            "usage_details": details,
            # A login's calls are not counted one by one: how many of a
            # mixed trial's calls were priced at list price is unknown.
            **({"n_calls_cost_estimated": None} if mixed else {}),
        }
        logger.info(
            "Cost at list prices: $%.4f (%s)",
            usd,
            ", ".join(f"{m}: {t}" for m, t in sorted(priced.tables.items())),
        )
    except Exception as exc:  # never fail the rollout over a price
        logger.info("Could not price the trial's tokens: %s", exc)


#: OpenRouter's record of one generation: its bill and its native tokens.
OPENROUTER_GENERATION_URL = "https://openrouter.ai/api/v1/generation"
#: A generation's record can lag its response by a few seconds.
BILL_LOOKUP_ATTEMPTS = 3
BILL_LOOKUP_DELAY_SEC = 2.0
BILL_LOOKUP_TIMEOUT_SEC = 10.0
#: Calls looked up per trial at most; the rest keep their estimate.
BILL_LOOKUP_MAX_CALLS = 200
_TOKEN_KEYS = (
    "n_input_tokens",
    "n_output_tokens",
    "n_cache_read_tokens",
    "n_cache_creation_tokens",
)


def _count(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


async def fetch_openrouter_generation(
    client: Any, generation_id: str, api_key: str
) -> dict[str, Any] | None:
    """OpenRouter's bill for one generation: ``{"usd", "n_input_tokens",
    "n_output_tokens", "n_cache_read_tokens"}`` from ``GET
    /api/v1/generation?id=``, or None when it cannot be had (a 4xx/5xx
    after a few tries, a timeout, a record without a cost). The key travels
    in the request header only, never in a log line."""
    for attempt in range(BILL_LOOKUP_ATTEMPTS):
        if attempt:
            await asyncio.sleep(BILL_LOOKUP_DELAY_SEC)
        try:
            response = await client.get(
                OPENROUTER_GENERATION_URL,
                params={"id": generation_id},
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=BILL_LOOKUP_TIMEOUT_SEC,
            )
        except Exception as exc:  # a timeout, a lost connection: try again
            logger.info(
                "OpenRouter generation %s: %s", generation_id, type(exc).__name__
            )
            continue
        if response.status_code != 200:
            logger.info(
                "OpenRouter generation %s: HTTP %s", generation_id, response.status_code
            )
            continue
        try:
            data = response.json().get("data")
        except Exception:
            data = None
        if not isinstance(data, dict):
            continue
        cost = data.get("total_cost")
        if isinstance(cost, bool) or not isinstance(cost, int | float) or cost < 0:
            continue

        prompt = _count(data, "native_tokens_prompt") or _count(data, "tokens_prompt")
        cached = _count(data, "native_tokens_cached")
        return {
            "usd": float(cost),
            "n_input_tokens": max(prompt - cached, 0),
            "n_output_tokens": _count(data, "native_tokens_completion")
            or _count(data, "tokens_completion"),
            "n_cache_read_tokens": cached,
        }
    return None


def _provider_key(rollout: Any, provider: str) -> str | None:
    """The run's own key for *provider* (``OPENROUTER_API_KEY``)."""
    name = f"{provider.upper()}_API_KEY"
    env = getattr(rollout, "_agent_env", None) or {}
    key = env.get(name) or os.environ.get(name)
    return key if isinstance(key, str) and key.strip() else None


async def resolve_provider_bills(rollout: Any, *, client: Any = None) -> None:
    """Look up the bill of each proxied call whose response carried none.

    An OpenRouter stream that ended without its usage (an aborted request)
    carried no ``usage.cost``; OpenRouter still keeps its generation record,
    which can say $0 and 0 tokens. Each such call's bill and tokens come
    from that record (``cost_source`` provider, not an estimate); a call
    whose lookup fails (no id, an HTTP error, a timeout) keeps its
    list-price estimate and counts in ``n_calls_cost_estimated``. Never
    fails the rollout.
    """
    import httpx

    try:
        metrics = dict(getattr(rollout, "_usage_metrics", None) or {})
        details = dict(metrics.get("usage_details") or {})
        unbilled = details.get("unbilled_calls")
        if metrics.get("usage_source") != USAGE_SOURCE_PROVIDER_RESPONSE or not (
            isinstance(unbilled, list) and unbilled
        ):
            return
        keys = {
            call.get("provider"): _provider_key(rollout, str(call.get("provider")))
            for call in unbilled
            if isinstance(call, dict)
        }
        owned = client is None
        http = client or httpx.AsyncClient()
        try:
            looked = 0
            resolved: list[dict[str, Any]] = []
            for call in unbilled:
                key = keys.get(call.get("provider"))
                generation = call.get("generation_id")
                bill = None
                if key and generation and looked < BILL_LOOKUP_MAX_CALLS:
                    looked += 1
                    bill = await fetch_openrouter_generation(http, generation, key)
                resolved.append({**call, "bill": bill})
        finally:
            if owned:
                await http.aclose()
        billed = float(metrics.get("cost_usd_provider") or 0.0)
        tokens = {field: metrics.get(field) for field in _TOKEN_KEYS}
        estimates: list[float | None] = []
        # The calls priced at list price on a route with no bill to look up
        # (another provider's): their estimates stay in the cost.
        elsewhere = details.get("estimated_elsewhere")
        elsewhere = elsewhere if isinstance(elsewhere, dict) else {}
        n_elsewhere = _count(elsewhere, "calls")
        elsewhere_usd = elsewhere.get("estimate_usd") if n_elsewhere else 0.0
        for call in resolved:
            bill = call["bill"]
            if bill is None:
                estimates.append(call.get("estimate_usd"))
                continue
            billed += bill["usd"]
            # The record's tokens replace what the proxy saw of the call.
            for field in ("n_input_tokens", "n_output_tokens", "n_cache_read_tokens"):
                seen = call.get(field) or 0
                if tokens.get(field) is not None or bill[field]:
                    tokens[field] = max(
                        (tokens.get(field) or 0) - seen + bill[field], 0
                    )
            seen_writes = call.get("n_cache_creation_tokens") or 0
            if tokens.get("n_cache_creation_tokens") is not None and seen_writes:
                tokens["n_cache_creation_tokens"] = max(
                    tokens["n_cache_creation_tokens"] - seen_writes, 0
                )
        n_estimated = len(estimates) + n_elsewhere
        details["unbilled_calls"] = [
            {
                "provider": call.get("provider"),
                "generation_id": call.get("generation_id"),
                "cost_source": "provider" if call["bill"] else "list_price_estimate",
                "usd": call["bill"]["usd"]
                if call["bill"]
                else call.get("estimate_usd"),
            }
            for call in resolved
        ]
        provider = next(
            (str(c.get("provider")) for c in resolved if c.get("provider")), "provider"
        )
        update: dict[str, Any] = {**tokens, "usage_details": details}
        if not n_estimated:
            update.update(
                cost_usd=round(billed, 10),
                cost_source=COST_SOURCE_PROVIDER,
                price_source=provider,
                price_table_date=None,
                cost_usd_provider=round(billed, 10),
                n_calls_cost_estimated=0,
            )
        else:
            known = all(
                isinstance(e, int | float) and not isinstance(e, bool)
                for e in [*estimates, elsewhere_usd]
            )
            estimated_usd = sum(float(e or 0.0) for e in [*estimates, elsewhere_usd])
            update.update(
                cost_usd=round(billed + estimated_usd, 10) if known else None,
                cost_source=COST_SOURCE_LIST_PRICE,
                price_source="litellm",
                cost_usd_provider=round(billed, 10),
                n_calls_cost_estimated=n_estimated,
            )
        rollout._usage_metrics = {**metrics, **update}
        logger.info(
            "Provider bills looked up: %d of %d unbilled call(s) found",
            len(resolved) - (n_estimated - n_elsewhere),
            len(resolved),
        )
    except Exception as exc:  # never fail the rollout over a bill
        logger.info("Could not look up the provider's bills: %s", exc)
