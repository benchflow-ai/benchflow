"""What a trial's LLM judge spent: its tokens and cost.

The llm-judge and agent-judge strategies (:mod:`benchflow.rewards.llm`) and
task.md's judges (:mod:`benchflow.taskmd.judging`) call a model from the
host while a rollout's verifier runs. Each call's usage, as the provider's
response reports it, is recorded (:func:`record`) while the rollout listens
(:func:`recording`), and the rollout writes the trial's total to
result.json as ``judge_usage``: apart from ``agent_result``, in the same
fields, with ``usage_source`` ``provider`` (the provider's own count, read
from each response) and the cost at list prices
(:mod:`benchflow._utils.pricing`). Recording never fails a judge call.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from benchflow._utils.pricing import TokenSlice, anthropic_slice, price_slices
from benchflow.usage_tracking import COST_SOURCE_LIST_PRICE, USAGE_PROVIDER, token_split

logger = logging.getLogger(__name__)

#: ``price_source`` of a judge's cost: its tokens at list prices, as
#: :data:`benchflow.rollout._cost.LIST_PRICE_SOURCE` names it for an agent.
LIST_PRICE_SOURCE = "list_price"
#: agent_result's token fields, in BenchFlow's split.
TOKEN_FIELDS = (
    "n_input_tokens",
    "n_output_tokens",
    "n_cache_read_tokens",
    "n_cache_write_tokens",
    "n_cache_write_5m_tokens",
    "n_cache_write_1h_tokens",
    "total_tokens",
)


@dataclass(frozen=True)
class JudgeCall:
    """One judge response: the model that answered and its tokens, None
    when the response reported no usage. ``writes_reported``: the response
    counted cache writes (Anthropic's do; OpenAI's and Gemini's report none,
    which is unknown, not 0); ``write_split_reported``: it split them into
    5-minute and 1-hour writes."""

    model: str
    tokens: TokenSlice | None
    writes_reported: bool = False
    write_split_reported: bool = False


# Every recording open in this context, outermost first: a verifier
# recovery records its own calls inside the rollout that records all of them.
_calls: ContextVar[tuple[list[JudgeCall], ...]] = ContextVar("judge_calls", default=())


@contextmanager
def recording() -> Iterator[list[JudgeCall]]:
    """Collect the judge calls made in this context (and the tasks it
    starts) into the list it yields; an enclosing recording gets them too."""
    calls: list[JudgeCall] = []
    token = _calls.set((*_calls.get(), calls))
    try:
        yield calls
    finally:
        _calls.reset(token)


def record(
    model: str,
    tokens: TokenSlice | None,
    *,
    writes_reported: bool = False,
    write_split_reported: bool = False,
) -> None:
    """Add one judge response to every recording listening, if any."""
    call = JudgeCall(
        model=model,
        tokens=tokens,
        writes_reported=writes_reported,
        write_split_reported=write_split_reported,
    )
    for calls in _calls.get():
        calls.append(call)


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _field(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _answered_by(response: Any, asked: str) -> str:
    model = _field(response, "model")
    return model if isinstance(model, str) and model else asked


def record_anthropic(asked: str, response: Any) -> None:
    """A Messages API response: ``input_tokens`` leaves the cache out."""
    try:
        model = _answered_by(response, asked)
        usage = _field(response, "usage")
        counts = {
            key: _field(usage, key)
            for key in (
                "input_tokens",
                "output_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
            )
        }
        written = _field(usage, "cache_creation")
        hour = _int(_field(written, "ephemeral_1h_input_tokens"))
        if hour is not None:
            counts["cache_creation"] = {"ephemeral_1h_input_tokens": hour}
        known = _int(counts["input_tokens"]) is not None and (
            _int(counts["output_tokens"]) is not None
        )
        record(
            model,
            anthropic_slice(model, counts) if known else None,
            writes_reported=_int(counts["cache_creation_input_tokens"]) is not None,
            write_split_reported=hour is not None,
        )
    except Exception as exc:  # never fail the judge over its usage
        logger.debug("Could not read the judge's usage: %s", exc)


def record_openai(asked: str, response: Any) -> None:
    """A Chat Completions response: ``prompt_tokens`` counts the cache."""
    try:
        model = _answered_by(response, asked)
        usage = _field(response, "usage")
        prompt = _int(_field(usage, "prompt_tokens"))
        output = _int(_field(usage, "completion_tokens"))
        if prompt is None or output is None:
            record(model, None)
            return
        cached = _int(_field(_field(usage, "prompt_tokens_details"), "cached_tokens"))
        read = min(cached or 0, prompt)
        record(
            model,
            TokenSlice(
                model=model,
                input=prompt - read,
                output=output,
                cache_read=read,
                prompt=prompt,
            ),
        )
    except Exception as exc:  # never fail the judge over its usage
        logger.debug("Could not read the judge's usage: %s", exc)


def record_google(asked: str, response: Any) -> None:
    """A Gemini response: ``prompt_token_count`` counts the cached content,
    and thinking is billed as output beside ``candidates_token_count``."""
    try:
        model = asked
        usage = _field(response, "usage_metadata")
        prompt = _int(_field(usage, "prompt_token_count"))
        answer = _int(_field(usage, "candidates_token_count"))
        if prompt is None or answer is None:
            record(model, None)
            return
        cached = min(_int(_field(usage, "cached_content_token_count")) or 0, prompt)
        thoughts = _int(_field(usage, "thoughts_token_count")) or 0
        record(
            model,
            TokenSlice(
                model=model,
                input=prompt - cached,
                output=answer + thoughts,
                cache_read=cached,
                prompt=prompt,
            ),
        )
    except Exception as exc:  # never fail the judge over its usage
        logger.debug("Could not read the judge's usage: %s", exc)


def usage_block(calls: Sequence[JudgeCall]) -> dict[str, Any] | None:
    """result.json's ``judge_usage`` for a trial's judge calls; None when
    the trial's verifier called no judge.

    The token fields and ``cost_usd`` are those of ``agent_result``, in
    BenchFlow's split: ``n_input_tokens`` leaves the cache out, cache writes
    nothing reported are null, and ``total_tokens`` sums the known parts. A
    call whose response reported no usage leaves every token field and the
    cost null, since a sum without it would read as the whole; so does a
    model no price table has, for the cost.
    """
    if not calls:
        return None
    counted = [call.tokens for call in calls if call.tokens is not None]
    block: dict[str, Any] = {
        "calls": len(calls),
        "calls_without_usage": len(calls) - len(counted),
        "models": dict(sorted(Counter(call.model for call in calls).items())),
        **dict.fromkeys(TOKEN_FIELDS),
        "cost_usd": None,
        "usage_source": None,
        "cost_source": None,
        "price_source": None,
        "price_table_date": None,
    }
    if len(counted) < len(calls):
        return block
    fresh = sum(piece.input for piece in counted)
    output = sum(piece.output for piece in counted)
    read = sum(piece.cache_read for piece in counted)
    write = (
        sum(piece.cache_write for piece in counted)
        if all(call.writes_reported for call in calls)
        else None
    )
    hour = (
        sum(piece.cache_write_1h for piece in counted)
        if write is not None and all(call.write_split_reported for call in calls)
        else None
    )
    block.update(
        n_input_tokens=fresh,
        n_output_tokens=output,
        n_cache_read_tokens=read,
        n_cache_write_tokens=write,
        n_cache_write_5m_tokens=(
            None if write is None or hour is None else write - hour
        ),
        n_cache_write_1h_tokens=hour,
        total_tokens=token_split(
            n_input_tokens=fresh,
            n_output_tokens=output,
            n_cache_read_tokens=read,
            n_cache_write_tokens=write,
        ),
        usage_source=USAGE_PROVIDER,
    )
    priced = price_slices(counted)
    if priced is not None:
        block.update(
            cost_usd=priced.usd,
            cost_source=COST_SOURCE_LIST_PRICE,
            price_source=LIST_PRICE_SOURCE,
            price_table_date=priced.table_date,
        )
    return block


def combine(
    first: dict[str, Any] | None, second: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Two ``judge_usage`` blocks of one trial as one (its first
    verification's and a verifier recovery's). A field either leaves
    unknown is unknown, never a partial sum."""
    if not isinstance(first, dict):
        return second
    if not isinstance(second, dict):
        return first

    def total(field: str) -> Any:
        a, b = first.get(field), second.get(field)
        if (
            isinstance(a, int | float)
            and not isinstance(a, bool)
            and isinstance(b, int | float)
            and not isinstance(b, bool)
        ):
            return a + b
        return None

    models = dict(first.get("models") or {})
    for name, n in (second.get("models") or {}).items():
        models[name] = models.get(name, 0) + n
    cost = total("cost_usd")
    same = {
        key: first.get(key) if first.get(key) == second.get(key) else None
        for key in ("usage_source", "cost_source", "price_source", "price_table_date")
    }
    return {
        "calls": (first.get("calls") or 0) + (second.get("calls") or 0),
        "calls_without_usage": (first.get("calls_without_usage") or 0)
        + (second.get("calls_without_usage") or 0),
        "models": dict(sorted(models.items())),
        **{field: total(field) for field in TOKEN_FIELDS},
        "cost_usd": round(cost, 10) if cost is not None else None,
        **same,
        **(
            {}
            if cost is not None
            else {"cost_source": None, "price_source": None, "price_table_date": None}
        ),
    }
