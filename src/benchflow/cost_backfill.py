"""Re-derive a finished job's tokens and cost from the logs its trials left.

BenchFlow up to 0.10.0rc3 wrote many trials with no cost, or with 0 tokens: a
Codex trial on a ChatGPT login counted one model call of its turn and had no
price, and a native Claude trial that timed out recorded 0 tokens beside its
session log's cost. The logs that hold the counts stayed in the trial folder.
:func:`backfill_job` reads them and counts and prices them with the steps a
0.10 trial takes at teardown (:mod:`benchflow.rollout._salvage`,
:mod:`benchflow.rollout._cost`), at no model spend:

- Claude Code's session logs, main and subagents (``agent/claude-sessions``),
  each response once by (message id, request id); else its stream log
  (``agent/claude-code.jsonl``);
- Codex's session rollouts (``agent/codex-sessions``), each response once by
  its response id, never a sum of Codex's running totals;
- the model proxy's log (``trajectory/llm_trajectory.jsonl``), else the
  trial's model-call records (``trajectory/model-calls.jsonl``): each call's
  provider bill where it carried one, the gateway's list-price figure for
  the rest; a call with neither leaves the cost unknown (never priced from
  its model's name), the billed part kept as a subtotal;
- a log compressed with zstd (``.jsonl.zst``) is read decompressed.

A provider's cost is never replaced, nor is any other cost already there,
unless it is a 0 beside positive usage in the logs; tokens are filled where
the record has none. A cost that stays but lacks its labels gets them
from the evidence (:func:`_labelled`), its counts and cost unchanged; a
proxied record from before the split, whose input counted the cache, is put
on the one split (:func:`_on_the_split`). result.json's
``agent_result`` keeps its earlier values verbatim under
``backfill.original``, with ``backfilled_at`` and ``backfill_version`` beside
it; ``final_metrics`` there and in ``trainer/atif.json`` get the same
figures, and the trial's and the job's results.jsonl rows its tokens.
summary.json's totals are recomputed by summary.json's own builder and
run-card.json's ``after_run.usage`` repeats them. Nothing outside the job
folder is written: the report says when the jobs folder's copy of
summary.json is stale. A second run changes nothing.

Every trial gets a line in the report: ``covered`` (rewritten with a
cost), ``tokens_filled`` (rewritten with its logged tokens, its cost still
unknown or an unverified 0), ``labelled`` (its labels added to kept
figures), ``already_had_cost``, ``uncovered`` with the reason (not
rewritten), or ``not_applicable`` (no model call); ``changes_record`` says
whether the run rewrites it. Each run's report is kept beside the latest
(``backfill-report.<UTC>.json``).
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA = "benchflow.backfill-cost.v1"
REPORT_FILE = "backfill-report.json"

COVERED = "covered"
# Rewritten with its logged tokens; its cost still unknown (or an unverified 0).
TOKENS_FILLED = "tokens_filled"
LABELLED = "labelled"
ALREADY_HAD_COST = "already_had_cost"
UNCOVERED = "uncovered"
NOT_APPLICABLE = "not_applicable"

# Why a trial was left as it was.
PROVIDER_COST = "provider_cost"
COST_PRESENT = "cost_present"
BACKFILLED_EARLIER = "backfilled_earlier"
NO_LOGS = "no_logs"
NO_PRICE = "no_price"
LOGS_TRUNCATED = "logs_truncated"
LOGS_UNREADABLE = "logs_unreadable"
NOT_BENCHFLOW_TRIAL = "not_benchflow_trial"
NO_RESULT = "no_result_json"
SHARED_TURNS = "shares_another_rollouts_turns"
SEVERAL_AGENTS = "several_agents"
MIXED_ROLES = "proxied_and_login_roles"
NO_MODEL_CALL = "no_model_call"
# Why a kept cost is labelled with no cost_source.
LEGACY_UNLABELLED = "legacy_unlabelled"
PARTLY_BILLED = "provider_bills_cover_some_calls"

#: result.json's ``usage_source`` names before that change, and the name
#: each becomes (``usage_tracking.RESULT_USAGE_SOURCES``); None when no source
#: counted the tokens. A name not here stays null, its value kept as
#: ``usage_source_legacy``. ``agent-reported`` keeps its legacy name too, which
#: is what tells an agent's own unmetered report from a login's.
LEGACY_USAGE_SOURCES: dict[str, str | None] = {
    "provider_response": "proxy",  # the model proxy's count of what it forwarded
    "agent_native_acp": "end_of_turn",  # the agent's end-of-turn reports
    "agent-reported": "end_of_turn",  # an agent that routes its own calls
    "agent_session_log": "session_log",  # Claude Code's session log
    "unavailable": None,
    "none": None,
}
#: ``price_source`` names before that change that name a priced table: a cost they
#: priced is a list-price estimate.
LEGACY_PRICED_TABLES: dict[str, str] = {
    "litellm": "the model gateway's LiteLLM table",
    "agent_session_log": "Claude Code's session log, at its own or BenchFlow's prices",
    "list_price": "BenchFlow's pricing step",
    "agent_usage_update": "Claude Code's running cost over ACP",
}
#: The first release that priced with the dated snapshot
#: (:mod:`benchflow.providers.pricing_snapshot`, added after 0.10.0rc1): an
#: older record's list price gets no ``price_table_date``.
SNAPSHOT_SINCE = "0.10.0rc2"

TOKEN_FIELDS = (
    "n_input_tokens",
    "n_output_tokens",
    "n_cache_read_tokens",
    "n_cache_write_tokens",
    "n_cache_write_5m_tokens",
    "n_cache_write_1h_tokens",
    "n_cache_creation_tokens",
    "total_tokens",
)
COST_FIELDS = (
    "cost_usd",
    "cost_source",
    "price_table_date",
    "price_source",
    "cost_usd_provider",
    "n_calls_cost_estimated",
)
_PROXY_LOG = Path("trajectory") / "llm_trajectory.jsonl"
_CALLS = Path("trajectory") / "model-calls.jsonl"
# OpenRouter's own records of the trial's generations (GET /api/v1/generation,
# one ``data`` object a line, by ``id``), saved beside the proxy log by a bill
# lookup (FDB PR #17's or1112b trials): the bill of a call whose answer
# carried none, such as an aborted stream's $0.
_GENERATIONS = Path("trajectory") / "or-generations.jsonl"
_STREAM_LOG = Path("agent") / "claude-code.jsonl"
_CLAUDE_SESSIONS = Path("agent") / "claude-sessions"
_CODEX_SESSIONS = Path("agent") / "codex-sessions"
_ATIF = Path("trainer") / "atif.json"
_ZST = ".zst"
_ABSENT = object()
# How far back from a log's end to look for its last line's start.
_TAIL_SCAN_BYTES = 64 * 1024 * 1024


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _number(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def backfill_version() -> dict[str, Any]:
    """The BenchFlow that backfilled: its version, and the git commit of its
    source when it runs from a BenchFlow checkout (``dirty`` when the
    package's files differ from that commit); None for an installed package,
    even one inside another project's repository."""
    from benchflow import __version__

    here = Path(__file__).resolve()
    commit: str | None = None
    dirty: bool | None = None
    git = shutil.which("git")

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(git), "-C", str(here.parent), *args],
            capture_output=True,
            text=True,
            timeout=30,
        )

    if git is not None:
        try:
            top = run("rev-parse", "--show-toplevel")
            checkout = (
                top.returncode == 0
                and (
                    Path(top.stdout.strip()) / "src" / "benchflow" / here.name
                ).resolve()
                == here
            )
            head = run("rev-parse", "HEAD") if checkout else None
            if head is not None and head.returncode == 0:
                commit = head.stdout.strip() or None
                dirty = run("diff", "--quiet", "HEAD", "--", ".").returncode != 0
        except (OSError, subprocess.SubprocessError):
            commit = dirty = None
    return {"benchflow": __version__, "commit": commit, "dirty": dirty}


# Logs, zstd-compressed or not


def _zstd_bytes(path: Path) -> bytes | None:
    """``path`` decompressed (zstd), or None when it cannot be: Python's own
    ``compression.zstd`` (3.14), the ``zstandard`` package, else the ``zstd``
    program."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    for module, decode in (
        ("compression.zstd", lambda m: m.decompress(data)),
        ("zstandard", lambda m: m.ZstdDecompressor().decompressobj().decompress(data)),
    ):
        try:
            found = importlib.import_module(module)
        except ImportError:
            continue
        try:
            return decode(found)
        except Exception:  # a corrupt or cut frame
            return None
    program = shutil.which("zstd")
    if program is None:
        return None
    try:
        done = subprocess.run(
            [program, "-dc", "--", str(path)], capture_output=True, timeout=600
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def _tail_cut(path: Path) -> bool:
    """Whether a JSON-lines log ends in a line cut off mid-write: its last
    line, with no newline after it, is no JSON."""
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            if size == 0:
                return False
            stream.seek(size - 1)
            if stream.read(1) == b"\n":
                return False
            start = max(0, size - _TAIL_SCAN_BYTES)
            stream.seek(start)
            tail = stream.read()
    except OSError:
        return False
    last = tail.rsplit(b"\n", 1)[-1]
    try:
        json.loads(last)
    except ValueError:
        return True
    return False


class _Logs:
    """A trial's logs as the parsers read them: in place, or, for a log
    compressed with zstd (``x.jsonl.zst``), decompressed into a temporary
    folder that mirrors the trial's layout. A plain log beside its
    compressed copy is read and the copy left out."""

    def __init__(self, trial: Path) -> None:
        self.trial = trial
        self._temp: tempfile.TemporaryDirectory[str] | None = None
        self.unreadable: list[str] = []
        self.cut: list[str] = []

    def __enter__(self) -> _Logs:
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._temp is not None:
            self._temp.cleanup()

    def _mirror(self) -> Path:
        if self._temp is None:
            self._temp = tempfile.TemporaryDirectory(prefix="bf-backfill-")
        return Path(self._temp.name)

    def _decompress(self, path: Path) -> Path | None:
        relative = path.relative_to(self.trial)
        target = self._mirror() / relative.parent / relative.name[: -len(_ZST)]
        if target.exists():
            return target
        data = _zstd_bytes(path)
        if data is None:
            self.unreadable.append(f"{relative}: zstd: no decoder, or a corrupt file")
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return target

    def _note_cut(self, path: Path, shown: Path) -> None:
        if _tail_cut(path):
            self.cut.append(str(shown))

    def file(self, relative: Path) -> Path | None:
        """The log at ``relative`` (or its ``.zst``), readable; None when absent."""
        plain = self.trial / relative
        if plain.is_file() and not plain.is_symlink():
            self._note_cut(plain, relative)
            return plain
        packed = plain.with_name(plain.name + _ZST)
        if packed.is_file() and not packed.is_symlink():
            found = self._decompress(packed)
            if found is not None:
                self._note_cut(found, relative)
            return found
        return None

    def tree(self, relative: Path, suffix: str = ".jsonl") -> Path | None:
        """The folder at ``relative`` with every log in it readable (in
        place when none is compressed); None when absent or empty."""
        folder = self.trial / relative
        if not folder.is_dir() or folder.is_symlink():
            return None
        logs: list[Path] = []
        for root, dirs, files in os.walk(folder, followlinks=False):
            dirs[:] = sorted(
                d for d in dirs if not os.path.islink(os.path.join(root, d))
            )
            for name in sorted(files):
                path = Path(root) / name
                if path.is_symlink() or not path.is_file():
                    continue
                if name.endswith(suffix) or name.endswith(suffix + _ZST):
                    logs.append(path)
        if not logs:
            return None
        packed = [p for p in logs if p.name.endswith(_ZST)]
        plain = [p for p in logs if not p.name.endswith(_ZST)]
        for path in plain:
            self._note_cut(path, path.relative_to(self.trial))
        if not packed:
            return folder
        mirror = self._mirror() / relative
        names = {p.relative_to(folder) for p in plain}
        for path in plain:
            target = mirror / path.relative_to(folder)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        for path in packed:
            inner = path.relative_to(folder)
            if inner.with_name(inner.name[: -len(_ZST)]) in names:
                continue  # its plain copy is read
            found = self._decompress(path)
            if found is not None:
                self._note_cut(found, path.relative_to(self.trial))
        return mirror if mirror.is_dir() else None

    def trial_with(self, relative: Path) -> Path | None:
        """A folder that holds ``relative`` as a trial folder does (for the
        readers that take a trial folder), or None when it is absent."""
        found = self.tree(relative)
        if found is None:
            return None
        return self.trial if found == self.trial / relative else self._mirror()


# Counting


@dataclass
class _Count:
    """What a trial's logs say it used, in result.json's terms."""

    source: str  # usage_source: session_log, stream_log, proxy, end_of_turn
    legacy: str  # usage_source_legacy
    counted: dict[str, Any]  # the token fields
    slices: list[Any]  # TokenSlices to price (a proxied count: its unpriced calls)
    detail: dict[str, Any]  # usage_details.token_count
    billed_usd: float = 0.0  # what the calls' provider bills say
    billed_calls: int = 0
    estimated_usd: float = 0.0  # list-price figures the calls already carry
    estimated_calls: int = 0
    calls: list[dict[str, Any]] | None = None  # per-call records (effort)
    thought_tokens: int | None = None
    models: list[str] = field(default_factory=list)
    cross_check: dict[str, Any] | None = None
    # The input the proxy's earlier parser recorded for these calls
    # (:func:`_pre_split_input`), to tell a record made from them.
    recorded_input: int | None = None
    # Who billed the billed calls (:func:`_biller`).
    billers: frozenset[str] = frozenset()

    def total(self) -> int:
        return _int(self.counted.get("total_tokens")) or 0


def _token_view(fields: dict[str, Any]) -> dict[str, Any]:
    return {key: fields.get(key) for key in TOKEN_FIELDS}


def _claude_count(logs: _Logs) -> _Count | None:
    from benchflow.rollout._salvage import (
        SESSION_TOKENS_SOURCE,
        STREAM_LOG,
        STREAM_TOKENS_SOURCE,
        claude_session_usage,
        claude_stream_usage,
    )
    from benchflow.usage_tracking import (
        USAGE_SESSION_LOG,
        USAGE_SOURCE_AGENT_NATIVE_ACP,
        USAGE_STREAM_LOG,
    )

    sessions = logs.tree(_CLAUDE_SESSIONS)
    stream = logs.file(_STREAM_LOG)
    from_sessions = claude_session_usage(sessions) if sessions else None
    from_stream = claude_stream_usage(stream) if stream else None
    if from_sessions is not None:
        count = _Count(
            source=USAGE_SESSION_LOG,
            legacy=USAGE_SOURCE_AGENT_NATIVE_ACP,
            counted=from_sessions,
            slices=list(from_sessions["slices"]),
            detail={
                "source": SESSION_TOKENS_SOURCE,
                "path": str(_CLAUDE_SESSIONS),
                "responses": from_sessions["responses"],
            },
        )
        if from_stream is not None:
            count.cross_check = {
                "stream_log": {
                    **_token_fields(from_stream),
                    "turns_without_report": from_stream["turns_without_report"],
                }
            }
    elif from_stream is not None:
        count = _Count(
            source=USAGE_STREAM_LOG,
            legacy=USAGE_SOURCE_AGENT_NATIVE_ACP,
            counted=from_stream,
            slices=list(from_stream["slices"]),
            detail={
                "source": STREAM_TOKENS_SOURCE,
                "path": f"agent/{STREAM_LOG}",
                "turns": from_stream["turns"],
                "turns_without_report": from_stream["turns_without_report"],
                "messages": from_stream["messages"],
            },
        )
    else:
        return None
    count.models = sorted({piece.model for piece in count.slices if piece.model})
    return count


def _token_fields(counted: dict[str, Any]) -> dict[str, Any]:
    from benchflow.rollout._salvage import TOKEN_FIELDS as SALVAGE_FIELDS

    return {key: counted.get(key) for key in SALVAGE_FIELDS}


def _codex_count(logs: _Logs) -> _Count | None:
    from benchflow.rollout._salvage import codex_session_slices
    from benchflow.rollout._session_log import CODEX_SESSION_DIR, CODEX_TOKENS_SOURCE
    from benchflow.trajectories.model_calls import (
        CODEX_SESSIONS,
        _codex_calls,
        codex_session_tokens,
    )
    from benchflow.usage_tracking import (
        USAGE_SESSION_LOG,
        USAGE_SOURCE_AGENT_NATIVE_ACP,
    )

    trial = logs.trial_with(_CODEX_SESSIONS)
    if trial is None:
        return None
    totals = codex_session_tokens(trial)
    if totals is None:
        return None
    slices = codex_session_slices(trial)
    counted = {
        "n_input_tokens": totals["n_input_tokens"],
        "n_output_tokens": totals["n_output_tokens"],
        "n_cache_read_tokens": totals["n_cache_read_tokens"],
        "n_cache_creation_tokens": totals["n_cache_creation_tokens"],
        # Codex logs no 5-minute/1-hour split of cache writes.
        "n_cache_write_5m_tokens": None,
        "n_cache_write_1h_tokens": None,
        "total_tokens": totals["total_tokens"],
    }
    return _Count(
        source=USAGE_SESSION_LOG,
        legacy=USAGE_SOURCE_AGENT_NATIVE_ACP,
        counted=counted,
        slices=slices,
        detail={
            "source": CODEX_TOKENS_SOURCE,
            "path": f"agent/{CODEX_SESSION_DIR}",
            "calls": totals["calls"],
        },
        calls=_codex_calls(trial / CODEX_SESSIONS),
        thought_tokens=_int(totals.get("thought_tokens")),
        models=sorted({piece.model for piece in slices if piece.model}),
    )


def _call_model(call: dict[str, Any]) -> str:
    for key in ("route", "model_served", "model_requested"):
        value = call.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _calls_count(
    calls: list[dict[str, Any]],
    usages: list[Any],
    *,
    source: str,
    legacy: str,
    detail: dict[str, Any],
) -> _Count | None:
    """Per-call records (the proxy's, or model-calls.jsonl's) and each call's
    tokens in BenchFlow's split (``usages``, a
    :class:`benchflow.trajectories.types.TokenUsage` per call) as a count:
    each call once, a count no call reported null (never 0), its provider
    bill where it carried one, the gateway's list-price figure where it has
    one; the calls with neither are kept as slices, unpriced."""
    from benchflow._utils.pricing import TokenSlice
    from benchflow.usage_tracking import token_split

    def known(values: list[int | None]) -> int | None:
        found = [value for value in values if value is not None]
        return sum(found) if found else None

    slices: list[TokenSlice] = []
    billed = estimated = 0.0
    billed_calls = estimated_calls = 0
    used: list[Any] = []
    for call, usage in zip(calls, usages, strict=True):
        if usage.input_tokens is None and usage.output_tokens is None:
            continue  # a failed call: nothing was counted
        used.append(usage)
        cost = _number(call.get("cost_usd"))
        if cost is not None and call.get("cost_source") == "provider":
            billed += cost
            billed_calls += 1
        elif cost is not None:
            estimated += cost
            estimated_calls += 1
        else:
            written = usage.cache_creation_tokens or 0
            slices.append(
                TokenSlice(
                    model=_call_model(call),
                    input=usage.input_tokens or 0,
                    output=usage.output_tokens or 0,
                    cache_read=usage.cache_read_tokens or 0,
                    cache_write=written,
                    cache_write_1h=min(usage.cache_write_1h_tokens or 0, written),
                    prompt=(usage.input_tokens or 0)
                    + (usage.cache_read_tokens or 0)
                    + written,
                )
            )
    if not used:
        return None
    counted = {
        "n_input_tokens": known([u.input_tokens for u in used]),
        "n_output_tokens": known([u.output_tokens for u in used]),
        "n_cache_read_tokens": known([u.cache_read_tokens for u in used]),
        "n_cache_creation_tokens": known([u.cache_creation_tokens for u in used]),
        "n_cache_write_5m_tokens": known([u.cache_write_5m_tokens for u in used]),
        "n_cache_write_1h_tokens": known([u.cache_write_1h_tokens for u in used]),
    }
    counted["total_tokens"] = token_split(
        n_input_tokens=counted["n_input_tokens"],
        n_output_tokens=counted["n_output_tokens"],
        n_cache_read_tokens=counted["n_cache_read_tokens"],
        n_cache_write_tokens=counted["n_cache_creation_tokens"],
    )
    models = sorted({_call_model(call) for call in calls if _call_model(call)})
    return _Count(
        source=source,
        legacy=legacy,
        counted=counted,
        slices=slices,
        detail={**detail, "calls": len(used)},
        billed_usd=billed,
        billed_calls=billed_calls,
        estimated_usd=estimated,
        estimated_calls=estimated_calls,
        calls=calls,
        models=models,
    )


def _response_usage(row: dict[str, Any]) -> Any:
    """One proxied call's tokens in BenchFlow's split, by the rule the live
    path applies to the call (``trajectories.types._exchange_token_usage``):
    an OpenAI-shaped answer's prompt counts its cache, the reads and (from
    LiteLLM's Anthropic bridge) the writes; an Anthropic one lists them
    beside its input, with the 5-minute/1-hour split."""
    from benchflow.trajectories.types import (
        LLMExchange,
        LLMRequest,
        LLMResponse,
        _exchange_token_usage,
    )

    response = row.get("response")
    body = response.get("body") if isinstance(response, dict) else None
    exchange = LLMExchange.model_construct(
        request=LLMRequest.model_construct(),
        response=LLMResponse.model_construct(
            body=body if isinstance(body, dict) else {}
        ),
    )
    return _exchange_token_usage(exchange)


def _record_usage(call: dict[str, Any]) -> Any:
    """A model-call record's tokens in BenchFlow's split: its
    ``input_tokens`` count the cache reads and writes, as the record writer
    takes them from the proxy (``model_calls._proxy_scan``)."""
    from benchflow.trajectories.types import TokenUsage

    prompt = _int(call.get("input_tokens"))
    read = _int(call.get("cache_read_tokens"))
    written = _int(call.get("cache_write_tokens"))
    return TokenUsage(
        input_tokens=None
        if prompt is None
        else max(prompt - (read or 0) - (written or 0), 0),
        output_tokens=_int(call.get("output_tokens")),
        cache_read_tokens=read,
        cache_creation_tokens=written,
    )


def _pre_split_input(row: dict[str, Any]) -> int:
    """The input BenchFlow 0.10.0rc1 to rc3 recorded for one proxied call
    (``trajectories.types._exchange_token_usage`` at 9b148715f, the same in
    all three): the answer's input or prompt count, plus its cache reads
    where they are only additive, its cache writes and Gemini's tool-use
    prompt. An answer from Anthropic through LiteLLM's OpenAI route counts
    its writes in its prompt already, so that parser counted them twice
    (prompt 10,000 with 3,000 writes recorded as 13,000)."""

    def first(*values: Any) -> int:
        for value in values:
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return 0

    response = row.get("response")
    body = response.get("body") if isinstance(response, dict) else None
    body = body if isinstance(body, dict) else {}
    usage = body.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    meta = body.get("usageMetadata")
    meta = meta if isinstance(meta, dict) else {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    prompt_details = prompt_details if isinstance(prompt_details, dict) else {}
    input_details = usage.get("input_tokens_details") or {}
    input_details = input_details if isinstance(input_details, dict) else {}
    additive_read = first(
        usage.get("cache_read_input_tokens"),
        usage.get("cacheReadInputTokens"),
        usage.get("cacheReadInputTokenCount"),
    )
    inclusive_read = first(
        prompt_details.get("cached_tokens"),
        input_details.get("cached_tokens"),
        meta.get("cachedContentTokenCount"),
    )
    written = first(
        usage.get("cache_creation_input_tokens"),
        usage.get("cacheWriteInputTokens"),
        usage.get("cacheWriteInputTokenCount"),
    )
    raw = first(
        usage.get("input_tokens"),
        usage.get("prompt_tokens"),
        usage.get("inputTokens"),
        meta.get("promptTokenCount"),
    )
    return (
        raw
        + (0 if inclusive_read else additive_read)
        + written
        + first(meta.get("toolUsePromptTokenCount"))
    )


def _proxy_rows(path: Path) -> list[dict[str, Any]]:
    """The proxy log's call lines, as ``model_calls._proxy_scan`` takes them
    (a request, no lifecycle event)."""
    from benchflow.trajectories.model_calls import _jsonl

    return [
        row
        for row in _jsonl(path)
        if isinstance(row.get("request"), dict) and not row.get("event")
    ]


def _billed_call(
    call: dict[str, Any], row: dict[str, Any], generations: dict[str, float]
) -> dict[str, Any]:
    bill = _row_bill(row, generations)
    if bill is not None:
        return {**call, "cost_usd": bill, "cost_source": "provider"}
    if call.get("cost_source") == "provider":
        return {**call, "cost_source": "list_price_estimate"}
    return call


def _proxy_count(logs: _Logs) -> _Count | None:
    from benchflow.run_checks import read_calls
    from benchflow.trajectories.model_calls import PROXY, _proxy_calls
    from benchflow.usage_tracking import USAGE_PROXY, USAGE_SOURCE_PROVIDER_RESPONSE

    path = logs.file(_PROXY_LOG)
    calls = _proxy_calls(path) if path is not None else []
    recorded_input: int | None = None
    billers: set[str] = set()
    if calls and path is not None:
        rows = _proxy_rows(path)
        usages = [_response_usage(row) for row in rows]
        recorded_input = sum(_pre_split_input(row) for row in rows)
        if len(rows) == len(calls):
            # Each call's bill by the gateway's rule (:func:`_row_bill`); a
            # figure that is no bill is the gateway's list price.
            generations = _generations(logs)
            calls = [
                _billed_call(call, row, generations)
                for call, row in zip(calls, rows, strict=True)
            ]
            billers = {
                _biller(row, generations)
                for call, row in zip(calls, rows, strict=True)
                if call.get("cost_source") == "provider"
            }
        detail = {"source": "proxy-log", "path": str(_PROXY_LOG)}
    else:
        records = logs.file(_CALLS)
        # read_calls takes the trial folder: the one holding trajectory/.
        recorded = read_calls(records.parent.parent) if records is not None else []
        calls = [c for c in recorded if c.get("source") == PROXY]
        for call in calls:
            # Records from before each call named its cost's source (v1, v2):
            # a figure there is the gateway's list price, not a bill.
            if call.get("cost_usd") is not None and call.get("cost_source") is None:
                call["cost_source"] = "list_price_estimate"
        usages = [_record_usage(call) for call in calls]
        # Each record's input is the answer's whole prompt.
        recorded_input = sum(_int(call.get("input_tokens")) or 0 for call in calls)
        detail = {"source": "model-call-records", "path": str(_CALLS)}
    if len(usages) != len(calls):
        return None
    count = _calls_count(
        calls,
        usages,
        source=USAGE_PROXY,
        legacy=USAGE_SOURCE_PROVIDER_RESPONSE,
        detail=detail,
    )
    if count is not None:
        count.recorded_input = recorded_input
        count.billers = frozenset(billers)
    return count


def _end_of_turn_count(agent_result: dict[str, Any], model: str) -> _Count | None:
    """The end-of-turn count the record already holds, to price when no log
    is left (a Claude trial over ACP whose session log was not copied)."""
    from benchflow._utils.pricing import TokenSlice
    from benchflow.usage_tracking import (
        USAGE_END_OF_TURN,
        USAGE_SOURCE_AGENT_NATIVE_ACP,
    )

    fresh = _int(agent_result.get("n_input_tokens"))
    out = _int(agent_result.get("n_output_tokens"))
    if not fresh and not out:
        return None
    read = _int(agent_result.get("n_cache_read_tokens")) or 0
    written = _int(agent_result.get("n_cache_write_tokens"))
    if written is None:
        written = _int(agent_result.get("n_cache_creation_tokens"))
    hour = _int(agent_result.get("n_cache_write_1h_tokens")) or 0
    piece = TokenSlice(
        model=model,
        input=fresh or 0,
        output=out or 0,
        cache_read=read,
        cache_write=written or 0,
        cache_write_1h=min(hour, written or 0),
    )
    counted = {
        "n_input_tokens": fresh,
        "n_output_tokens": out,
        "n_cache_read_tokens": _int(agent_result.get("n_cache_read_tokens")),
        "n_cache_creation_tokens": written,
        "n_cache_write_5m_tokens": _int(agent_result.get("n_cache_write_5m_tokens")),
        "n_cache_write_1h_tokens": _int(agent_result.get("n_cache_write_1h_tokens")),
        "total_tokens": (fresh or 0) + (out or 0) + read + (written or 0),
    }
    return _Count(
        source=USAGE_END_OF_TURN,
        legacy=USAGE_SOURCE_AGENT_NATIVE_ACP,
        counted=counted,
        slices=[piece],
        detail={"source": "end-of-turn-report", "path": None},
        models=[model] if model else [],
    )


# Pricing, with the one pricing step


def _priced_metrics(
    count: _Count, metrics: dict[str, Any], run_model: str
) -> dict[str, Any]:
    """``metrics`` (the trial's usage, the live rollout's shape) after the
    pricing step a 0.10 trial goes through at teardown
    (:func:`benchflow.rollout._cost.apply_list_price`), with ``count``'s
    slices to price."""
    from benchflow.rollout._cost import apply_list_price

    rollout = SimpleNamespace(
        _usage_metrics=dict(metrics),
        _usage_slices=list(count.slices),
        _config=SimpleNamespace(model=run_model, primary_agent=None),
        _session_reported_models=list(count.models),
        _native_usage_slices={},
        _proxy_reported_models=[],
    )
    # Its "Cost at list prices" line is a trial's teardown log; here the
    # report says it, once per trial.
    cost_log = logging.getLogger("benchflow.rollout._cost")
    level = cost_log.level
    cost_log.setLevel(logging.WARNING)
    try:
        apply_list_price(rollout)
    finally:
        cost_log.setLevel(level)
    return rollout._usage_metrics


def _proxy_priced(count: _Count, metrics: dict[str, Any]) -> dict[str, Any]:
    """A proxied trial's cost from its calls: the provider-billed part plus
    the list-price figures the gateway gave the rest (litellm_logging's
    mixed case). ``cost_source`` is ``provider`` only when every call was
    billed. A call with neither is not priced from its model's name (the
    pricing step's rule: a route the gateway could not price may be a
    private deployment, not the vendor's API), so the cost stays unknown and
    the billed part is kept as ``cost_usd_provider``."""
    from benchflow.rollout._cost import _snapshot_date_for
    from benchflow.usage_tracking import COST_SOURCE_LIST_PRICE, COST_SOURCE_PROVIDER

    out = dict(metrics)
    out["cost_usd_provider"] = (
        round(count.billed_usd, 10) if count.billed_calls else None
    )
    if count.slices:
        # Unknown, with the billed part as a subtotal.
        out.update(cost_usd=None, cost_source=None, price_source=None)
        out.update(price_table_date=None, n_calls_cost_estimated=None)
        return out
    out["n_calls_cost_estimated"] = count.estimated_calls
    if not count.estimated_calls:
        out.update(
            cost_usd=round(count.billed_usd, 10),
            cost_source=COST_SOURCE_PROVIDER,
            # Named for who billed (the gateway's route_provider), as a kept
            # provider cost is labelled.
            price_source=next(iter(count.billers))
            if len(count.billers) == 1
            else "provider",
            price_table_date=None,
        )
        return out
    out.update(
        cost_usd=round(count.billed_usd + count.estimated_usd, 10),
        cost_source=COST_SOURCE_LIST_PRICE,
        price_source="litellm",
        # The gateway's figures: dated only when every model is in the
        # snapshot, as rollout._cost labels a litellm cost.
        price_table_date=_snapshot_date_for(count.models),
    )
    return out


# One trial


@dataclass
class _Outcome:
    status: str
    reason: str | None = None
    source: str | None = None
    agent_result: dict[str, Any] | None = None  # the new agent_result, to write
    effort: dict[str, Any] | None = None  # the new effort block, to write
    log_check: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)
    # What the report flags about the record, beside its status.
    flags: list[str] = field(default_factory=list)
    split: dict[str, Any] | None = None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _agents(trial: Path, result: dict[str, Any]) -> set[str]:
    """Every agent that made the trial's calls: the run's and each scene
    role's, by config.json."""
    found = {str(result.get("agent") or "")} - {""}
    config = _read_json(trial / "config.json")
    if isinstance(config, dict):
        if config.get("agent"):
            found.add(str(config["agent"]))
        for scene in config.get("scenes") or []:
            for role in (
                (scene or {}).get("roles") or [] if isinstance(scene, dict) else []
            ):
                if isinstance(role, dict) and role.get("agent"):
                    found.add(str(role["agent"]))
    return found


def _family(agent: str) -> str:
    from benchflow.rollout._session_log import _runs_claude_code, _runs_codex

    if _runs_claude_code(agent):
        return "claude"
    if _runs_codex(agent):
        return "codex"
    return agent


def _shares_turns(trial: Path, result: dict[str, Any]) -> bool:
    """The trial's sandbox held turns another trial is charged for: a fork's
    parent, a branch child, a run continued from a kept checkpoint."""
    return bool(
        result.get("branches")
        or result.get("parent_rollout")
        or (trial / "checkpoint_source.json").exists()
    )


def _no_tokens(agent_result: dict[str, Any]) -> bool:
    return not any(
        _int(agent_result.get(key))
        for key in (
            "total_tokens",
            "n_input_tokens",
            "n_output_tokens",
            "n_cache_read_tokens",
            "n_cache_creation_tokens",
            "n_cache_write_tokens",
        )
    )


def _unknown_not_zero(
    before: dict[str, Any], reason: str, stamp: dict[str, Any]
) -> dict[str, Any] | None:
    """The record of a trial nothing counted, its 0 tokens null with why
    (``no_usage_reason``: ``no_model_call``, or the reason the backfill found
    nothing to count, such as ``no_logs``): a 0 no source counted is unknown,
    not 0. Only a record whose own source counted nothing (``unavailable``,
    ``none``, an agent's unmetered report) or a trial with no model call;
    ``usage_source`` is then null, its old name kept as
    ``usage_source_legacy``. None when the record holds no such 0 (null
    already, so a second run changes nothing)."""
    from benchflow.usage_tracking import is_trusted_usage_source, legacy_usage_source

    if reason != NO_MODEL_CALL and is_trusted_usage_source(
        before.get("usage_source"), before.get("usage_source_legacy")
    ):
        return None
    if not any(
        isinstance(before.get(key), int)
        and not isinstance(before.get(key), bool)
        and before.get(key) == 0
        for key in TOKEN_FIELDS
    ):
        return None
    after = dict(before)
    after.update({key: None for key in TOKEN_FIELDS})
    after["usage_source_legacy"] = legacy_usage_source(
        before.get("usage_source"), before.get("usage_source_legacy")
    )
    after["usage_source"] = None
    after["no_usage_reason"] = reason
    earlier = (
        before.get("backfill") if isinstance(before.get("backfill"), dict) else None
    )
    after["backfilled_at"] = stamp["at"]
    after["backfill_version"] = stamp["version"]
    after["backfill"] = {
        "original": earlier["original"] if earlier is not None else before,
        "source": None,
        "filled": ["tokens_unknown"],
    }
    return after


def _metrics_from(agent_result: dict[str, Any]) -> dict[str, Any]:
    """A result's agent_result in the live rollout's usage shape, which the
    pricing step reads: usage_source as the earlier name it trusts."""
    from benchflow.usage_tracking import legacy_usage_source

    metrics = dict(agent_result)
    metrics["usage_source"] = legacy_usage_source(
        agent_result.get("usage_source"), agent_result.get("usage_source_legacy")
    )
    written = agent_result.get("n_cache_write_tokens")
    if written is None:
        written = agent_result.get("n_cache_creation_tokens")
    metrics["n_cache_creation_tokens"] = written
    return metrics


def _counted_metrics(count: _Count, metrics: dict[str, Any]) -> dict[str, Any]:
    """``metrics`` with ``count``'s tokens, from its source, as teardown's
    salvage writes them (:func:`benchflow.rollout._salvage._replace_usage`):
    the earlier count kept in ``usage_details.token_count``."""
    from benchflow.rollout._salvage import _replace_usage
    from benchflow.usage_tracking import USAGE_SOURCE_AGENT_NATIVE_ACP

    rollout = SimpleNamespace(_usage_metrics=dict(metrics))
    _replace_usage(rollout, count.counted, source=count.source, detail=count.detail)
    counted = rollout._usage_metrics
    counted["usage_source_legacy"] = count.legacy
    if count.legacy != USAGE_SOURCE_AGENT_NATIVE_ACP:
        # The earlier count was the proxy's own, not an end-of-turn report.
        details = dict(counted.get("usage_details") or {})
        token_count = dict(details.get("token_count") or {})
        token_count["recorded"] = token_count.pop("end_of_turn_reported", None)
        details["token_count"] = token_count
        counted["usage_details"] = details
    if count.thought_tokens is not None:
        counted["usage_details"] = {
            **(counted.get("usage_details") or {}),
            "thought_tokens": count.thought_tokens,
        }
    return counted


def _agent_result_from(
    metrics: dict[str, Any], before: dict[str, Any]
) -> dict[str, Any]:
    """result.json's agent_result from the usage metrics, as the rollout's
    result writer lays it out (split), the other fields kept."""
    from benchflow.usage_tracking import result_usage_source, token_split

    written = metrics.get("n_cache_creation_tokens")
    out = dict(before)
    out.update(
        n_input_tokens=metrics.get("n_input_tokens"),
        n_output_tokens=metrics.get("n_output_tokens"),
        n_cache_read_tokens=metrics.get("n_cache_read_tokens"),
        n_cache_write_tokens=written,
        n_cache_write_5m_tokens=metrics.get("n_cache_write_5m_tokens"),
        n_cache_write_1h_tokens=metrics.get("n_cache_write_1h_tokens"),
        n_cache_creation_tokens=written,
        cost_usd=metrics.get("cost_usd"),
        usage_source=result_usage_source(metrics.get("usage_source")),
        cost_source=metrics.get("cost_source"),
        price_table_date=metrics.get("price_table_date"),
        cost_usd_provider=metrics.get("cost_usd_provider"),
        n_calls_cost_estimated=metrics.get("n_calls_cost_estimated"),
        price_source=metrics.get("price_source"),
        usage_source_legacy=metrics.get("usage_source_legacy"),
    )
    out["total_tokens"] = token_split(
        n_input_tokens=out["n_input_tokens"],
        n_output_tokens=out["n_output_tokens"],
        n_cache_read_tokens=out["n_cache_read_tokens"],
        n_cache_write_tokens=out["n_cache_write_tokens"],
    )
    if metrics.get("usage_details") is not None:
        out["usage_details"] = metrics["usage_details"]
    return out


def _effort_from_calls(
    trial: Path, result: dict[str, Any], calls: list[dict[str, Any]] | None
) -> dict[str, Any] | None:
    """result.json's ``effort`` with the efforts the logged calls carried
    (``sent``), where it had none; ``applied`` from them when one effort ran
    every main-thread call and the record had none. ``matches`` stays
    unknown without the effort asked for. None when nothing changes."""
    from benchflow.trajectories.model_calls import effort_outcome, summarize

    effort = result.get("effort")
    effort = dict(effort) if isinstance(effort, dict) else {}
    if effort.get("sent") is not None or not calls:
        return None
    summary = summarize(calls)
    observed = summary.get("agent_effort") or {}
    if not observed:
        return None
    requested = effort.get("requested")
    if requested is None:
        config = _read_json(trial / "config.json")
        if isinstance(config, dict) and isinstance(config.get("reasoning_effort"), str):
            requested = config["reasoning_effort"]
    applied = effort.get("applied")
    from_log = applied is None and len(observed) == 1
    if from_log:
        applied = next(iter(observed))
    rebuilt = effort_outcome(
        summary, requested=requested, applied=applied, route=effort.get("route")
    )
    if from_log:
        rebuilt["applied_source"] = summary.get("source")
    return rebuilt if rebuilt != result.get("effort") else None


@dataclass
class _Bills:
    """What the proxy's raw responses say the provider billed: the calls
    that answered with a usage, those with a bill (:func:`_row_bill`, the
    gateway's rule), and the bills' sum."""

    answered: int
    billed: int
    usd: float
    providers: frozenset[str] = frozenset()

    def is_cost(self, cost: float | None) -> bool:
        """Every call billed, and the bills sum to ``cost`` (the matrix's
        closeness: 1e-6 of the larger, at least $1)."""
        return (
            cost is not None
            and self.billed > 0
            and self.billed == self.answered
            and _same_usd(self.usd, cost)
        )


def _row_bill(row: dict[str, Any], generations: dict[str, float]) -> float | None:
    """The provider's bill for one proxied call, by the gateway's rule
    (``litellm_logging.billed_cost``): the proxy's own record of it
    (``metadata.provider_cost``) where it has the field, else the answer's
    ``usage.cost`` on a route that bills each call (OpenRouter). LiteLLM puts
    its own list price in ``usage.cost`` on other routes, which is no bill.
    Else OpenRouter's own record of the call's generation, by its id
    (:data:`_GENERATIONS`), as the gateway's bill lookup reads it."""
    from benchflow.providers.litellm_logging import billed_cost

    response = row.get("response")
    body = response.get("body") if isinstance(response, dict) else None
    body = body if isinstance(body, dict) else {}
    # The whole line: its route is read from the model group, the request's
    # model and its endpoint as well as its metadata (route_provider).
    bill = billed_cost(body, row)
    if bill is None and isinstance(body.get("id"), str):
        bill = generations.get(body["id"])
    return bill


def _generations(logs: _Logs) -> dict[str, float]:
    """OpenRouter's billed USD per generation id (:data:`_GENERATIONS`)."""
    from benchflow.trajectories.model_calls import _jsonl

    path = logs.file(_GENERATIONS)
    found: dict[str, float] = {}
    for record in _jsonl(path) if path is not None else []:
        cost = _number(record.get("total_cost"))
        if isinstance(record.get("id"), str) and cost is not None and cost >= 0:
            found[record["id"]] = cost
    return found


def _raw_bills(logs: _Logs) -> _Bills | None:
    path = logs.file(_PROXY_LOG)
    if path is None:
        return None
    generations = _generations(logs)
    answered = billed = 0
    usd = 0.0
    providers: set[str] = set()
    for row in _proxy_rows(path):
        response = row.get("response")
        body = response.get("body") if isinstance(response, dict) else None
        if not isinstance(body, dict) or not isinstance(body.get("usage"), dict):
            continue
        answered += 1
        bill = _row_bill(row, generations)
        if bill is not None:
            billed += 1
            usd += bill
            providers.add(_biller(row, generations))
    return _Bills(answered, billed, usd, frozenset(providers)) if answered else None


def _biller(row: dict[str, Any], generations: dict[str, float]) -> str:
    """Who billed a call: who its route goes through, by the gateway's rule
    (``litellm_logging.route_provider``: the model group, ``runtime-`` or
    0.9.x's ``benchflow-``, the request's model, an openrouter.ai endpoint),
    OpenRouter when its generation record billed it; else ``provider``."""
    from benchflow.providers.litellm_logging import route_provider

    response = row.get("response")
    body = response.get("body") if isinstance(response, dict) else None
    if isinstance(body, dict) and body.get("id") in generations:
        return "openrouter"
    return route_provider(row) or "provider"


def _summary_line(agent_result: dict[str, Any]) -> dict[str, Any]:
    return {
        **{key: agent_result.get(key) for key in TOKEN_FIELDS},
        "cost_usd": agent_result.get("cost_usd"),
        "usage_source": agent_result.get("usage_source"),
        "cost_source": agent_result.get("cost_source"),
        "price_table_date": agent_result.get("price_table_date"),
    }


@dataclass
class _Found:
    """What a trial's logs say: its count (None, with ``reason`` when one is
    known), the proxy's raw bills, the logs that could not be read and those
    cut off mid-line."""

    count: _Count | None
    bills: _Bills | None
    unreadable: list[str]
    cut: list[str]
    reason: str | None


def _derive(
    trial: Path, result: dict[str, Any], agent_result: dict[str, Any]
) -> _Found:
    """The trial's count from its logs (see :class:`_Found`)."""
    from benchflow.usage_tracking import (
        USAGE_SOURCE_PROVIDER_RESPONSE,
        result_usage_source,
    )

    families = {_family(agent) for agent in _agents(trial, result)}
    proxied = (
        result_usage_source(agent_result.get("usage_source")) in ("proxy", "provider")
        or agent_result.get("usage_source_legacy") == USAGE_SOURCE_PROVIDER_RESPONSE
    )
    with _Logs(trial) as logs:
        count: _Count | None = None
        reason: str | None = None
        if proxied:
            details = agent_result.get("usage_details") or {}
            if isinstance(details, dict) and isinstance(
                details.get("agent_native_acp"), dict
            ):
                reason = MIXED_ROLES
            else:
                count = _proxy_count(logs)
        elif len(families) > 1:
            reason = SEVERAL_AGENTS
        elif families == {"claude"}:
            count = _claude_count(logs)
        elif families == {"codex"}:
            count = _codex_count(logs)
        if (
            count is None
            and reason is None
            and not proxied
            and result_usage_source(agent_result.get("usage_source")) is None
        ):
            # Nothing counted the trial, and no log of its agent's says: the
            # proxy's log, when its calls went through the proxy.
            count = _proxy_count(logs)
        bills = _raw_bills(logs)
        return _Found(count, bills, list(logs.unreadable), list(logs.cut), reason)


def _backfill_trial(
    trial: Path, result: dict[str, Any], stamp: dict[str, Any]
) -> _Outcome:
    """What the backfill does to one trial (nothing is written here)."""
    from benchflow.agents.registry import is_scripted_agent
    from benchflow.usage_tracking import NO_MODEL_CALL as NO_CALL
    from benchflow.usage_tracking import USAGE_SOURCE_AGENT_NATIVE_ACP

    before = result["agent_result"]
    agent = str(result.get("agent") or "")
    if before.get("no_usage_reason") == NO_CALL or (agent and is_scripted_agent(agent)):
        return _Outcome(
            NOT_APPLICABLE,
            NO_MODEL_CALL,
            agent_result=_unknown_not_zero(before, NO_MODEL_CALL, stamp),
        )
    cost = _number(before.get("cost_usd"))
    earlier = (
        before.get("backfill") if isinstance(before.get("backfill"), dict) else None
    )
    if earlier is not None and cost is not None:
        return _Outcome(ALREADY_HAD_COST, BACKFILLED_EARLIER)

    found = _derive(trial, result, before)
    count, unreadable, cut, why_not = (
        found.count,
        found.unreadable,
        found.cut,
        found.reason,
    )
    notes = [f"log unreadable: {item}" for item in unreadable]
    notes += [f"log ends mid-line (missing tail): {item}" for item in cut]
    if count is not None and count.detail.get("turns_without_report"):
        notes.append(
            f"{count.detail['turns_without_report']} turn(s) ended without their "
            "report: counted from their messages"
        )
    log_check = _log_check(count, before, result) if count is not None else None
    if before.get("cost_source") is not None:
        # A labelled cost is kept, a 0 too: only an unlabelled 0 is replaced.
        reason = PROVIDER_COST if before["cost_source"] == "provider" else COST_PRESENT
        return _kept(
            reason, trial, result, found, stamp, log_check=log_check, notes=notes
        )
    if found.bills is not None and found.bills.is_cost(cost):
        return _kept(
            PROVIDER_COST, trial, result, found, stamp, log_check=log_check, notes=notes
        )

    needs_cost = cost is None or (cost == 0 and count is not None and count.total() > 0)
    needs_tokens = _no_tokens(before) and count is not None and count.total() > 0
    if not needs_cost and not needs_tokens:
        return _kept(
            COST_PRESENT, trial, result, found, stamp, log_check=log_check, notes=notes
        )
    if _shares_turns(trial, result):
        return _Outcome(UNCOVERED, SHARED_TURNS, notes=notes)
    if count is None and why_not is None:
        legacy = before.get("usage_source_legacy") or before.get("usage_source")
        if (
            cost is None
            and legacy in (USAGE_SOURCE_AGENT_NATIVE_ACP, "end_of_turn")
            and _family(agent) != "codex"
            and result.get("error") is None
        ):
            # No log left; the turns all reported (no error): price their count.
            count = _end_of_turn_count(before, str(result.get("model") or ""))
    if count is None or count.total() == 0:
        reason = why_not or (
            LOGS_UNREADABLE if unreadable else LOGS_TRUNCATED if cut else NO_LOGS
        )
        # Nothing to count from: a 0 the record holds is unknown, not 0.
        return _Outcome(
            UNCOVERED,
            reason,
            notes=notes,
            agent_result=_unknown_not_zero(before, reason, stamp),
        )

    metrics = _metrics_from(before)
    if count.source not in ("end_of_turn",):
        metrics = _counted_metrics(count, metrics)
    if needs_cost:
        metrics.update(cost_usd=None, price_source=None, cost_source=None)
        metrics["price_table_date"] = None
    if count.source == "proxy" and needs_cost:
        metrics = _proxy_priced(count, metrics)
    else:
        metrics = _priced_metrics(count, metrics, str(result.get("model") or ""))
    after = _agent_result_from(metrics, before)
    if not needs_cost:
        after, dated = _dated_by_reprice(after, log_check)
        notes += [dated] if dated else []
    flags: list[str] = []
    zero_kept = cost == 0 and after.get("cost_usd") is None
    if zero_kept:
        # A 0 beside logged usage that cannot be priced: kept, never nulled.
        if after.get("cost_usd_provider") is not None:
            notes.append(
                f"the calls' provider bills sum to ${after['cost_usd_provider']:.6f}"
                " over some calls; the recorded 0 is kept"
            )
        after.update(
            cost_usd=before.get("cost_usd"),
            cost_source=None,
            cost_source_reason=ZERO_COST_UNVERIFIED,
            price_source=before.get("price_source"),
            price_table_date=None,
            cost_usd_provider=None,
            n_calls_cost_estimated=None,
        )
        flags.append(ZERO_COST_UNVERIFIED)
    reasons = (["cost_missing"] if cost is None else []) + (
        ["cost_zero_beside_log_usage"] if cost == 0 and not zero_kept else []
    )
    reasons += ["tokens_missing"] if needs_tokens else []
    tokens_changed = _token_view(after) != _token_view(before)
    if not needs_tokens and tokens_changed:
        reasons.append("tokens_recounted")
    after["backfilled_at"] = stamp["at"]
    after["backfill_version"] = stamp["version"]
    after["backfill"] = {
        "original": earlier["original"] if earlier is not None else before,
        "source": count.detail,
        "filled": reasons,
        **({"notes": notes} if notes else {}),
        **({"cross_check": count.cross_check} if count.cross_check else {}),
    }
    priced = after.get("cost_usd") is not None and not zero_kept
    status = COVERED if priced else TOKENS_FILLED if tokens_changed else LABELLED
    reason = None if priced else NO_PRICE
    effort = _effort_from_calls(trial, result, count.calls)
    if effort is not None:
        after["backfill"]["original_effort"] = result.get("effort")
    if earlier is not None and not priced:
        # Backfilled before with no price, and still none: nothing new.
        return _Outcome(UNCOVERED, NO_PRICE, source=count.source, notes=notes)
    return _Outcome(
        status,
        reason,
        source=count.source,
        agent_result=after,
        effort=effort,
        log_check=log_check,
        notes=notes,
        flags=flags + _flags(after, result, count),
    )


def _kept(
    reason: str,
    trial: Path,
    result: dict[str, Any],
    found: _Found | None,
    stamp: dict[str, Any],
    *,
    log_check: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> _Outcome:
    """A trial whose cost stays: ``labelled`` when it lacked its labels
    and got them, else ``already_had_cost``."""
    notes = list(notes or [])
    count = found.count if found is not None else None
    labelled = _labelled(trial, result, found, provider=reason == PROVIDER_COST)
    if labelled is None:
        return _Outcome(
            ALREADY_HAD_COST,
            reason,
            log_check=log_check,
            notes=notes,
            flags=_flags(result["agent_result"], result, count),
        )
    after, label_notes = labelled
    notes += label_notes
    after, dated = _dated_by_reprice(after, log_check)
    notes += [dated] if dated else []
    before = result["agent_result"]
    after, split = _on_the_split(after, before, count)
    if split.get("split_reason"):
        notes.append(f"input left as recorded: {split['split_reason']}")
        # Off its split, its counts are not what its name says.
        after["usage_source"] = before.get("usage_source")
        after["usage_source_legacy"] = before.get("usage_source_legacy")
    changed = sorted(key for key in after if before.get(key, _ABSENT) != after[key])
    after["backfilled_at"] = stamp["at"]
    after["backfill_version"] = stamp["version"]
    after["backfill"] = {
        "original": before,
        "labelled": changed,
        **split,
        **({"notes": notes} if notes else {}),
    }
    return _Outcome(
        LABELLED,
        reason,
        agent_result=after,
        log_check=log_check,
        notes=notes,
        flags=_flags(after, result, count),
        split=split or None,
    )


#: Why a cache-inclusive input was left as recorded.
SPLIT_INPUT_BELOW_CACHE = "input_below_cache"
SPLIT_CACHE_UNKNOWN = "cache_counts_unknown"
SPLIT_LOG_DISAGREES = "proxy_log_disagrees"
SPLIT_WRITES_WITHOUT_LOG = "cache_writes_without_proxy_log"
#: A record with input and no output from a trial whose agent's connection
#: closed: the 0 is kept, never guessed.
ZERO_OUTPUT_ON_TRANSPORT_LOSS = "zero_output_on_transport_loss"
ZERO_OUTPUT = "zero_output"
#: An unlabelled 0 cost beside logged usage the backfill cannot price: kept.
ZERO_COST_UNVERIFIED = "zero_cost_unverified"
#: A labelled 0 cost beside logged usage: kept (a labelled cost is never
#: replaced), and flagged.
ZERO_COST_BESIDE_LOG_USAGE = "zero_cost_beside_log_usage"


def _on_the_split(
    after: dict[str, Any], before: dict[str, Any], count: _Count | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """A earlier proxied record on its one split, its counts and cost
    the same, only split; ``total_tokens`` the sum of the parts. Its input
    counted the cache (:func:`_input_includes_cache`).

    With the proxy's log, when the record is what the old parser made of
    the logged calls (its input :attr:`_Count.recorded_input`, its output
    the log's), the split is the log's by the live path's rule
    (:func:`_response_usage`): an answer from Anthropic through LiteLLM's
    OpenAI route had its cache writes counted twice, and its uncached input
    is its prompt less the reads and the writes. With no log, the split is
    the input less the record's cache reads, only where its cache writes are
    known to be none: a bridged answer's writes cannot be placed without its
    log. Otherwise left as recorded, with the reason. Returns the record and
    ``backfill``'s split fields (none for a record already on the split)."""
    from benchflow.usage_tracking import token_split

    if not _input_includes_cache(before):
        return after, {}
    fields: dict[str, Any] = {"input_includes_cache": True}
    recorded = _int(before.get("n_input_tokens"))

    def left(reason: str) -> tuple[dict[str, Any], dict[str, Any]]:
        return after, {**fields, "split_converted": False, "split_reason": reason}

    if count is not None and count.source == "proxy":
        logged = count.counted
        if (
            recorded is None
            or logged.get("n_input_tokens") is None
            or recorded != count.recorded_input
            or logged.get("n_output_tokens") != _int(before.get("n_output_tokens"))
        ):
            return left(SPLIT_LOG_DISAGREES)
        converted = {
            **after,
            "n_input_tokens": logged["n_input_tokens"],
            "n_cache_read_tokens": logged["n_cache_read_tokens"],
            "n_cache_write_tokens": logged["n_cache_creation_tokens"],
            "n_cache_creation_tokens": logged["n_cache_creation_tokens"],
            "n_cache_write_5m_tokens": logged["n_cache_write_5m_tokens"],
            "n_cache_write_1h_tokens": logged["n_cache_write_1h_tokens"],
            "total_tokens": logged["total_tokens"],
        }
        return converted, {
            **fields,
            "split_converted": True,
            "split_from": count.detail.get("source"),
        }
    read = _int(before.get("n_cache_read_tokens"))
    written = _int(after.get("n_cache_write_tokens"))
    if written is not None and written > 0:
        return left(SPLIT_WRITES_WITHOUT_LOG)
    if recorded is None or read is None or written is None:
        return left(SPLIT_CACHE_UNKNOWN)
    uncached = recorded - read
    if uncached < 0:
        return left(SPLIT_INPUT_BELOW_CACHE)
    converted = {**after, "n_input_tokens": uncached}
    converted["total_tokens"] = token_split(
        n_input_tokens=uncached,
        n_output_tokens=_int(converted.get("n_output_tokens")),
        n_cache_read_tokens=read,
        n_cache_write_tokens=written,
    )
    return converted, {**fields, "split_converted": True, "split_from": "record"}


def _flags(
    after: dict[str, Any], result: dict[str, Any], count: _Count | None
) -> list[str]:
    """What the report flags about a record's kept numbers: 0 output beside
    input (on a transport loss when the trial ended with its agent's
    connection closed, ``pipe_closed``: rc3's opencode trial on MiMo, whose
    PTY closed mid-turn while its two calls had answered empty), and a 0
    cost beside usage its logs show."""
    from benchflow._utils.scoring import PIPE_CLOSED

    flags: list[str] = []
    if _int(after.get("n_output_tokens")) == 0 and _int(after.get("n_input_tokens")):
        lost = result.get("error_category") == PIPE_CLOSED
        flags.append(ZERO_OUTPUT_ON_TRANSPORT_LOSS if lost else ZERO_OUTPUT)
    if (
        _number(after.get("cost_usd")) == 0
        and after.get("cost_source") is not None
        and count is not None
        and count.total() > 0
    ):
        flags.append(ZERO_COST_BESIDE_LOG_USAGE)
    return flags


def _reports_cache_writes(result: dict[str, Any]) -> bool:
    """Whether the record's usage came from a source that reports cache
    writes: Claude Code's own reports (Anthropic's usage), not a proxy's."""
    from benchflow.usage_tracking import USAGE_SOURCE_PROVIDER_RESPONSE

    record = result["agent_result"]
    legacy = record.get("usage_source_legacy") or record.get("usage_source")
    return (
        legacy != USAGE_SOURCE_PROVIDER_RESPONSE
        and _family(str(result.get("agent") or "")) == "claude"
    )


def _same_usd(a: float, b: float) -> bool:
    """Two USD figures the same, within 1e-6 of the larger (at least $1),
    the cost matrix's closeness."""
    return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))


def _input_includes_cache(agent_result: dict[str, Any]) -> bool:
    """A proxied record written before that change's split: its n_input_tokens
    counts the cache reads and writes too (the proxy's prompt count)."""
    from benchflow.usage_tracking import USAGE_SOURCE_PROVIDER_RESPONSE

    legacy = agent_result.get("usage_source_legacy") or agent_result.get("usage_source")
    return (
        legacy == USAGE_SOURCE_PROVIDER_RESPONSE
        and "n_cache_write_tokens" not in agent_result
    )


def _has_snapshot(trial: Path) -> bool:
    """Whether the BenchFlow that ran the trial (config.json) priced with the
    dated snapshot (:data:`SNAPSHOT_SINCE` on)."""
    from packaging.version import InvalidVersion, Version

    config = _read_json(trial / "config.json")
    version = config.get("benchflow_version") if isinstance(config, dict) else None
    try:
        return Version(str(version)) >= Version(SNAPSHOT_SINCE)
    except InvalidVersion:
        return False


def _labelled(
    trial: Path, result: dict[str, Any], found: _Found | None, *, provider: bool
) -> tuple[dict[str, Any], list[str]] | None:
    """A kept cost's agent_result with its labels, every number as it
    was, and notes for the report; None when it has them already.

    ``usage_source`` takes its name (:data:`LEGACY_USAGE_SOURCES`;
    an unknown one is null, its value kept in ``usage_source_legacy``).
    ``cost_source`` is ``provider`` when every call's raw response carried
    the provider's cost and those sum to the cost (:class:`_Bills`); else
    ``list_price_estimate`` when ``price_source`` names a priced table
    (:data:`LEGACY_PRICED_TABLES`), dated as the pricing step dates it when
    the run had the snapshot; else null with ``cost_source_reason``.
    """
    from benchflow.rollout._cost import _label_cost
    from benchflow.usage_tracking import COST_SOURCE_PROVIDER, RESULT_USAGE_SOURCES

    before = result["agent_result"]
    after = dict(before)
    notes: list[str] = []
    usage = before.get("usage_source")
    if usage is not None and usage not in RESULT_USAGE_SOURCES:
        after["usage_source"] = LEGACY_USAGE_SOURCES.get(str(usage))
        if before.get("usage_source_legacy") is None:
            after["usage_source_legacy"] = usage
    if "n_cache_write_tokens" not in before and "n_cache_creation_tokens" in before:
        # its name for the same count, its parts unknown. A 0 is a count
        # only where the source reports cache writes (Claude Code's own
        # reports, from Anthropic's usage); elsewhere it is the old parsers'
        # default for a count nobody reported, so unknown.
        written = before["n_cache_creation_tokens"]
        if written == 0 and not _reports_cache_writes(result):
            written = None
        after["n_cache_write_tokens"] = after["n_cache_creation_tokens"] = written
        after.setdefault("n_cache_write_5m_tokens", None)
        after.setdefault("n_cache_write_1h_tokens", None)
    cost = _number(before.get("cost_usd"))
    bills = found.bills if found is not None else None
    if before.get("cost_source") is None and cost is not None:
        price_source = before.get("price_source")
        count = found.count if found is not None else None
        # The rc-era mixed case: the billed calls' bills plus the gateway's
        # list-price figures for the rest (litellm_logging's rule).
        mixed = (
            count is not None
            and count.source == "proxy"
            and count.billed_calls > 0
            and count.estimated_calls > 0
            and not count.slices
            and _same_usd(count.billed_usd + count.estimated_usd, cost)
        )
        if provider and bills is not None:
            after.update(
                cost_source=COST_SOURCE_PROVIDER,
                # Named for who billed, as the gateway names a billed cost.
                price_source=next(iter(bills.providers))
                if len(bills.providers) == 1
                else "provider",
                price_table_date=None,
                cost_usd_provider=cost,
                n_calls_cost_estimated=0,
            )
        elif (
            bills is not None
            and 0 < bills.billed < bills.answered
            and _same_usd(bills.usd, cost)
        ):
            # The cost is the billed calls' sum; the others have none in it.
            after.update(
                cost_source=None,
                cost_source_reason=PARTLY_BILLED,
                cost_usd_provider=round(bills.usd, 10),
            )
        elif mixed or (
            isinstance(price_source, str) and price_source in LEGACY_PRICED_TABLES
        ):
            # The proxy's routes name the models (a response names the
            # gateway's model group, runtime-xiaomi-mimo-v2.6-flash).
            routes = (
                count.models if count is not None and count.source == "proxy" else []
            )
            rollout = SimpleNamespace(
                _proxy_reported_models=list(routes),
                _config=SimpleNamespace(model=result.get("model") or ""),
            )
            after = _label_cost(rollout, after)
            if not _has_snapshot(trial):
                after["price_table_date"] = None
            if mixed and count is not None:
                after.update(
                    cost_usd_provider=round(count.billed_usd, 10),
                    n_calls_cost_estimated=count.estimated_calls,
                )
        else:
            after.update(cost_source=None, cost_source_reason=LEGACY_UNLABELLED)
        for key in ("price_table_date", "cost_usd_provider", "n_calls_cost_estimated"):
            after.setdefault(key, None)
        if bills is not None and bills.billed and not provider:
            notes.append(
                f"{bills.billed} of {bills.answered} raw responses carry a provider "
                f"cost summing to ${bills.usd:.6f}; the recorded cost "
                f"${cost:.6f} is kept"
            )
    if after == before:
        return None
    return after, notes


def _log_check(
    count: _Count, before: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    """What the logs say beside what the record says, for the report."""
    metrics = _counted_metrics(count, _metrics_from(before))
    metrics.update(cost_usd=None, price_source=None, cost_source=None)
    if count.source == "proxy":
        priced = _proxy_priced(count, metrics)
    else:
        priced = _priced_metrics(count, metrics, str(result.get("model") or ""))
    return {
        "source": count.source,
        **_token_fields(count.counted),
        "cost_usd_at_list_price": priced.get("cost_usd"),
        "price_table_date": priced.get("price_table_date"),
        "cost_usd_provider": priced.get("cost_usd_provider"),
    }


def _dated_by_reprice(
    after: dict[str, Any], log_check: dict[str, Any] | None
) -> tuple[dict[str, Any], str | None]:
    """A kept list-price figure with no table date (Claude Code's own
    totals, priced by its own table) dated by the evidence: when BenchFlow's
    dated snapshot prices the logged responses to the same figure (within
    1e-6 of it), that is the table it agrees with. The note says so."""
    from benchflow.usage_tracking import COST_SOURCE_LIST_PRICE

    cost = _number(after.get("cost_usd"))
    repriced = _number((log_check or {}).get("cost_usd_at_list_price"))
    date = (log_check or {}).get("price_table_date")
    if (
        after.get("cost_source") != COST_SOURCE_LIST_PRICE
        or after.get("price_table_date") is not None
        or cost is None
        or repriced is None
        or date is None
        or abs(cost - repriced) > max(1e-9, 1e-6 * max(abs(cost), abs(repriced)))
    ):
        return after, None
    return {**after, "price_table_date": date}, (
        f"the recorded cost equals the {date} snapshot's price of the logged "
        f"responses (${repriced:.6f}): dated {date}"
    )


def _unrecorded_spend(folder: Path) -> dict[str, Any] | None:
    """What a trial with no result.json (stopped before 0.10 wrote one)
    spent, by its logs, for the report: there is no record to write it to."""
    config = _read_json(folder / "config.json")
    config = config if isinstance(config, dict) else {}
    stub = {"agent": config.get("agent"), "model": config.get("model")}
    try:
        count = _derive(folder, stub, {}).count
        return _log_check(count, {}, stub) if count is not None else None
    except Exception as exc:  # the report says what it can
        logger.warning("Could not read the logs of %s: %s", folder, exc)
        return None


# The job


def _trials(job_dir: Path) -> tuple[list[Path], list[Path], list[Path]]:
    """The job's trial result files (every attempt, stopped trials' spend
    records included), the trial folders with no result.json, and folders of
    another harness's trials (an ``roll.json``)."""
    from benchflow._utils.result_paths import (
        is_interrupted_result,
        is_job_summary,
        is_set_aside,
        iter_task_result_paths,
    )
    from benchflow.jobs import _interrupted

    results = list(iter_task_result_paths(job_dir))
    taken = {path.parent for path in results}
    for path in sorted(job_dir.rglob("result.json")):
        if path in results or is_set_aside(path, job_dir) or is_job_summary(path):
            continue
        if any(parent in taken for parent in path.parent.parents):
            continue
        if is_interrupted_result(_read_json(path)):
            results.append(path)
    stopped = {path.parent for path in results}
    missing = [folder for folder in _interrupted(job_dir) if folder not in stopped]
    known = stopped | set(missing)
    others = sorted(
        {
            path.parent
            for path in job_dir.rglob("roll.json")
            if not is_set_aside(path, job_dir)
            and not any(path.parent.is_relative_to(folder) for folder in known)
        }
    )
    return sorted(results), missing, others


def _nested_jobs(job_dir: Path) -> list[str]:
    """Job folders below ``job_dir`` (a job record of their own): a folder
    of jobs is no job, and one summary.json cannot total them."""
    from benchflow._utils.result_paths import is_set_aside

    return sorted(
        {
            str(path.parent.relative_to(job_dir))
            for name in ("evaluation.json", "summary.json")
            for path in job_dir.rglob(name)
            if path.parent != job_dir and not is_set_aside(path, job_dir)
        }
    )


def _dump(data: Any, *, newline: bool) -> str:
    from benchflow._utils.json_safe import dumps_finite

    return dumps_finite(data, indent=2) + ("\n" if newline else "")


def _rewrite(path: Path, data: Any) -> bool:
    """Replace a JSON file with ``data`` in the layout BenchFlow writes it
    (indent 2, its trailing newline kept). False when nothing changed."""
    old = path.read_text()
    return _replace_text(path, old, _dump(data, newline=old.endswith("\n")))


def _replace_text(path: Path, old: str, new: str) -> bool:
    """Replace ``path``'s text whole, keeping its mode and its time (attempts
    of a task are ranked by it). False when nothing changed."""
    from benchflow._utils.disk_space import write_text_atomic

    if new == old:
        return False
    stat = path.stat()
    write_text_atomic(path, new)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    return True


def _final_metrics(
    old: Any, agent_result: dict[str, Any], *, keep_totals: bool
) -> dict[str, Any]:
    """final_metrics from the new agent_result (ATIF's terms: the prompt
    counts the cache); with ``keep_totals`` (a record left off 's
    split), the old totals with the new provenance in ``extra``."""
    from benchflow._utils.result_metadata import (
        cost_provenance,
        final_metrics_from_agent_result,
    )

    old = dict(old) if isinstance(old, dict) else {}
    if keep_totals:
        return {**old, "extra": cost_provenance(agent_result)}
    return {**old, **final_metrics_from_agent_result(agent_result)}


def _keeps_totals(agent_result: dict[str, Any]) -> bool:
    """A backfilled record left off its split: its final_metrics keep
    their totals (:func:`_final_metrics`)."""
    earlier = agent_result.get("backfill")
    return isinstance(earlier, dict) and earlier.get("split_converted") is False


def _atif_updated(trial: Path, agent_result: dict[str, Any]) -> dict[str, Any] | None:
    """trainer/atif.json with final_metrics from the backfilled record
    (:func:`_final_metrics`), its step count and the rest kept, stamped with
    the record's own ``backfilled_at``. Its earlier figures go under
    ``extra.backfill.original`` the first time and stay: a run stopped after
    this write and before result.json's keeps the true originals when run
    again. None when there is no ATIF document."""
    document = _read_json(trial / _ATIF)
    if not isinstance(document, dict):
        return None
    old = document.get("final_metrics")
    old = dict(old) if isinstance(old, dict) else {}
    held = (
        (old.get("extra") or {}).get("backfill")
        if isinstance(old.get("extra"), dict)
        else None
    )
    if isinstance(held, dict) and "original" in held:
        original = held["original"]
    else:
        original = {k: v for k, v in old.items() if k != "total_steps"}
    metrics = _final_metrics(
        old if not isinstance(held, dict) else {**old, **original},
        agent_result,
        keep_totals=_keeps_totals(agent_result),
    )
    extra = dict(metrics["extra"])
    extra["backfill"] = {
        "backfilled_at": agent_result.get("backfilled_at"),
        "backfill_version": agent_result.get("backfill_version"),
        "original": original,
    }
    document["final_metrics"] = {**metrics, "extra": extra}
    return document


def _sync_atif(job_dir: Path, result_paths: list[Path]) -> list[str]:
    """Bring each backfilled trial's trainer/atif.json in line with its
    result.json (the commit point): the files written."""
    written: list[str] = []
    for path in result_paths:
        result = _read_json(path)
        record = result.get("agent_result") if isinstance(result, dict) else None
        if not isinstance(record, dict) or not isinstance(record.get("backfill"), dict):
            continue
        document = _atif_updated(path.parent, record)
        if document is not None and _rewrite(path.parent / _ATIF, document):
            written.append(str((path.parent / _ATIF).relative_to(job_dir)))
    return written


_ROW_COST_FIELDS = ("cost_usd", "cost_source", "usage_source", "price_table_date")


def _numbers_changed(agent_result: dict[str, Any]) -> bool:
    """Whether a backfill changed what a results.jsonl row shows of the
    record (its ``token_usage``) or its cost; labels added to kept numbers
    change neither."""
    from benchflow.trajectories.results import _token_usage_from_agent_result

    earlier = agent_result.get("backfill")
    original = earlier.get("original") if isinstance(earlier, dict) else None
    if not isinstance(original, dict):
        return False
    return _token_usage_from_agent_result(original) != _token_usage_from_agent_result(
        agent_result
    ) or original.get("cost_usd") != agent_result.get("cost_usd")


def _row_updated(row: dict[str, Any], agent_result: dict[str, Any]) -> dict[str, Any]:
    """A results.jsonl row with the backfilled record's ``token_usage`` (as
    the row writer derives it) and, in ``info.backfill``, its cost and where
    they came from, the row's earlier ``token_usage`` kept as ``original``."""
    from benchflow.trajectories.results import _token_usage_from_agent_result

    info = dict(row.get("info") or {})
    earlier = info.get("backfill")
    original = (earlier.get("original") if isinstance(earlier, dict) else None) or {
        "token_usage": row.get("token_usage")
    }
    info["backfill"] = {
        **{key: agent_result.get(key) for key in _ROW_COST_FIELDS},
        "backfilled_at": agent_result.get("backfilled_at"),
        "backfill_version": agent_result.get("backfill_version"),
        "original": original,
    }
    return {
        **row,
        "info": info,
        "token_usage": _token_usage_from_agent_result(agent_result),
    }


def _row_owner(info: dict[str, Any], trials: dict[str, dict[str, Any]]) -> str | None:
    """The trial a results.jsonl row is of, among ``trials`` (by path
    relative to the job): the one whose path its ``rollout_dir`` ends with
    (the longest such, where the job ran elsewhere), and whose
    ``rollout_name`` the row names; without ``rollout_dir``, the one whose
    ``rollout_name`` and ``task_name`` it names."""
    from pathlib import PurePosixPath

    rollout_dir = info.get("rollout_dir")
    name = info.get("rollout_name")
    if isinstance(rollout_dir, str) and rollout_dir:
        parts = PurePosixPath(rollout_dir).parts
        best: str | None = None
        for key, result in trials.items():
            own = PurePosixPath(key).parts
            if (
                parts[-len(own) :] == own
                and name in (None, result.get("rollout_name"))
                and (best is None or len(own) > len(PurePosixPath(best).parts))
            ):
                best = key
        return best
    matches = [
        key
        for key, result in trials.items()
        if name is not None
        and name == result.get("rollout_name")
        and info.get("task_name") == result.get("task_name")
    ]
    return matches[0] if len(matches) == 1 else None


def _sync_rows(path: Path, trials: dict[str, dict[str, Any]]) -> bool:
    """Rewrite the rows of a results.jsonl that are of a trial in ``trials``
    (result.json payloads by path relative to the job, :func:`_row_owner`);
    every other line stays as it is. False when nothing changed."""
    from benchflow.trajectories.results import _record_to_redacted_json_line

    try:
        old = path.read_text()
    except OSError:
        return False
    lines = old.splitlines(keepends=True)
    out: list[str] = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            out.append(line)
            continue
        info = row.get("info") if isinstance(row, dict) else None
        owner = _row_owner(info if isinstance(info, dict) else {}, trials)
        if owner is None:
            out.append(line)
            continue
        updated = _row_updated(row, trials[owner]["agent_result"])
        if updated == row:
            out.append(line)
            continue
        end = "\n" if line.endswith("\n") else ""
        out.append(_record_to_redacted_json_line(updated) + end)
    return _replace_text(path, old, "".join(out))


def _sync_results_jsonl(job_dir: Path, result_paths: list[Path]) -> list[str]:
    """Bring the trials' and the job's results.jsonl rows in line with the
    records whose tokens or cost a backfill changed; the files written."""
    from benchflow.trajectories.results import (
        JOB_RESULTS_FILENAME,
        ROLLOUT_RESULTS_FILENAME,
    )

    changed: dict[str, dict[str, Any]] = {}
    written: list[str] = []
    for path in result_paths:
        result = _read_json(path)
        record = result.get("agent_result") if isinstance(result, dict) else None
        if not isinstance(record, dict) or not _numbers_changed(record):
            continue
        key = path.parent.relative_to(job_dir).as_posix()
        changed[key] = result
        rows = path.parent / ROLLOUT_RESULTS_FILENAME
        if rows.is_file() and _sync_rows(rows, {key: result}):
            written.append(str(rows.relative_to(job_dir)))
    job_rows = job_dir / JOB_RESULTS_FILENAME
    if changed and job_rows.is_file() and _sync_rows(job_rows, changed):
        written.append(JOB_RESULTS_FILENAME)
    return written


def _sync_run_card(job_dir: Path, stamp: dict[str, Any]) -> tuple[str, bool]:
    """run-card.json's ``after_run.usage`` set to summary.json's totals, as
    the job writes it (its earlier value kept under ``after_run.backfill``).
    Returns the card's state and whether it was written."""
    from benchflow.evaluation import RUN_CARD_USAGE_FIELDS
    from benchflow.run_card import RUN_CARD

    path = job_dir / RUN_CARD
    card = _read_json(path)
    summary = _read_json(job_dir / "summary.json")
    if not isinstance(card, dict):
        return "absent", False
    after_run = card.get("after_run")
    if not isinstance(after_run, dict):
        return "no_after_run", False
    if not isinstance(summary, dict):
        return "no_summary", False
    usage = {key: summary.get(key) for key in RUN_CARD_USAGE_FIELDS}
    if after_run.get("usage") == usage:
        return "current", False
    earlier = after_run.get("backfill")
    original = (earlier.get("original") if isinstance(earlier, dict) else None) or {
        "usage": after_run.get("usage")
    }
    updated = {
        **card,
        "after_run": {
            **after_run,
            "usage": usage,
            "backfill": {
                "backfilled_at": stamp["at"],
                "backfill_version": stamp["version"],
                "original": original,
            },
        },
    }
    return "updated", _rewrite(path, updated)


def _jobs_root_copy(job_dir: Path) -> dict[str, Any] | None:
    """The copy of summary.json a job writes in the folder above it, for
    tooling that reads it there: whether it still matches the job's own (it
    is outside the job folder, so a backfill never writes it)."""
    copy = job_dir.parent / "summary.json"
    own = job_dir / "summary.json"
    if not copy.is_file() or not own.is_file():
        return None
    held, mine = _read_json(copy), _read_json(own)
    if not isinstance(held, dict) or not isinstance(mine, dict):
        return None
    if held.get("job_name") != mine.get("job_name"):
        return None  # another job's summary
    stale = held != mine
    return {
        "path": str(copy),
        "state": "stale" if stale else "current",
        **(
            {"refresh": f"cp {own} {copy}  (the job writes the same text to both)"}
            if stale
            else {}
        ),
    }


#: summary.json's counts, beside total_cost_usd, of the trials whose figures
#: in the totals a backfill could not make whole: a cost that is a provider
#: subtotal (``provider_bills_cover_some_calls``), an unverified $0
#: (``zero_cost_unverified``), and an input still counting its cache (left
#: off its split).
_CAVEATS = (
    "trials_cost_provider_subtotal",
    "trials_cost_zero_unverified",
    "trials_input_includes_cache",
)


def _caveats(results: dict[str, dict[str, Any]]) -> dict[str, int]:
    def record(result: dict[str, Any]) -> dict[str, Any]:
        found = result.get("agent_result")
        return found if isinstance(found, dict) else {}

    def left_off(result: dict[str, Any]) -> bool:
        earlier = record(result).get("backfill")
        return isinstance(earlier, dict) and earlier.get("split_converted") is False

    values = list(results.values())
    return {
        "trials_cost_provider_subtotal": sum(
            record(r).get("cost_source_reason") == PARTLY_BILLED for r in values
        ),
        "trials_cost_zero_unverified": sum(
            record(r).get("cost_source_reason") == ZERO_COST_UNVERIFIED for r in values
        ),
        "trials_input_includes_cache": sum(left_off(r) for r in values),
    }


def _summary_updated(job_dir: Path, stamp: dict[str, Any]) -> dict[str, Any] | None:
    """summary.json with its token and cost totals recomputed by its own
    builder over the job's results as they now are; its earlier values kept
    under ``backfill.original``. None when there is no summary.json or the
    totals are unchanged."""
    from benchflow._utils.evaluation_results import usage_summary
    from benchflow._utils.result_paths import (
        interrupted_trial_results,
        load_task_results,
    )

    summary = _read_json(job_dir / "summary.json")
    if not isinstance(summary, dict):
        return None
    results = {
        **load_task_results(job_dir, include_terminal_errors=True),
        **interrupted_trial_results(job_dir),
    }
    from benchflow.evaluation import _spend_all_attempts

    usage = usage_summary(results)
    caveats = _caveats(results)
    fresh = {**usage, **caveats}
    if "spend_all_attempts" in summary:
        # Every attempt's spend, by the job's own builder: a backfill that
        # prices a trial changes it as it changes the kept totals. A summary
        # written before the block existed gets none (a clean job stays as it
        # is).
        fresh.update(_spend_all_attempts(job_dir))
    changed = {k: v for k, v in fresh.items() if summary.get(k, object()) != v}
    # The caveat counts are no reason to write on their own while they are
    # 0 and absent: a job that needs nothing is left as it is.
    if not any(k in usage or k in summary or v for k, v in changed.items()):
        return None
    earlier = (
        summary.get("backfill") if isinstance(summary.get("backfill"), dict) else {}
    )
    original = dict(earlier.get("original") or {})
    added = list(earlier.get("added") or [])
    for key in changed:
        if key in summary and key not in original and key not in added:
            original[key] = summary[key]
        elif key not in summary and key not in added and key not in original:
            added.append(key)
    updated: dict[str, Any] = {}
    for key, value in {**summary, **fresh}.items():
        if key in _CAVEATS:
            continue
        updated[key] = value
        if key == "total_cost_usd":
            # Beside the total they qualify.
            updated.update({k: fresh[k] for k in _CAVEATS})
    updated.update({k: fresh[k] for k in _CAVEATS if k not in updated})
    updated["backfilled_at"] = stamp["at"]
    updated["backfill_version"] = stamp["version"]
    updated["backfill"] = {"original": original, "added": sorted(added)}
    return updated


def _totals(lines: Iterable[dict[str, Any]]) -> dict[str, Any]:
    lines = list(lines)
    costs = [c for line in lines if (c := _number(line.get("cost_usd"))) is not None]

    def total(key: str) -> int | None:
        values = [v for line in lines if (v := _int(line.get(key))) is not None]
        return sum(values) if values else None

    return {
        "trials_with_cost": len(costs),
        "cost_usd": round(sum(costs), 10) if costs else None,
        **{
            key: total(key)
            for key in (
                "n_input_tokens",
                "n_output_tokens",
                "n_cache_read_tokens",
                "n_cache_write_tokens",
                "total_tokens",
            )
        },
    }


def backfill_job(
    job_dir: Path,
    *,
    dry_run: bool = False,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Backfill one job folder's tokens and cost from its trials' logs (see
    the module); write its report (``backfill-report.json`` in the job
    folder unless ``report_path`` names another) and return it. With
    ``dry_run`` only the report is written."""
    job_dir = Path(job_dir).resolve()
    if not job_dir.is_dir():
        raise FileNotFoundError(f"No job folder at {job_dir}")
    nested = _nested_jobs(job_dir)
    if nested:
        raise ValueError(
            f"{job_dir} holds other jobs ({', '.join(nested[:3])}"
            f"{', ...' if len(nested) > 3 else ''}): pass each job folder"
        )
    now = datetime.now(UTC)
    stamp = {
        "at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "id": now.strftime("%Y%m%dT%H%M%S%fZ"),
        "version": backfill_version(),
    }
    results, missing, others = _trials(job_dir)
    if not results and not missing and not others:
        raise ValueError(f"No trials in {job_dir}: is it a job folder?")
    lines: list[dict[str, Any]] = []
    before_lines: list[dict[str, Any]] = []
    after_lines: list[dict[str, Any]] = []
    written: list[str] = []
    for path in results:
        trial = path.parent
        relative = str(trial.relative_to(job_dir))
        result = _read_json(path)
        if not isinstance(result, dict) or not isinstance(
            result.get("agent_result"), dict
        ):
            line = {
                "trial": relative,
                "status": UNCOVERED,
                "reason": NOT_BENCHFLOW_TRIAL,
            }
            lines.append(line)
            continue
        try:
            outcome = _backfill_trial(trial, result, stamp)
        except Exception as exc:  # one trial's surprise never stops the job
            logger.warning("Could not backfill %s: %s", relative, exc)
            outcome = _Outcome(UNCOVERED, LOGS_UNREADABLE, notes=[f"error: {exc}"])
        before = result["agent_result"]
        after = outcome.agent_result or before
        before_lines.append(before)
        after_lines.append(after)
        line = {
            "trial": relative,
            "task": result.get("task_name"),
            "agent": result.get("agent"),
            "model": result.get("model"),
            "error_category": result.get("error_category"),
            "interrupted": result.get("interrupted") is True,
            "status": outcome.status,
            "reason": outcome.reason,
            # Whether this run rewrites the record (in a dry run: would).
            "changes_record": outcome.agent_result is not None,
            "source": outcome.source,
            "before": _summary_line(before),
            "after": _summary_line(after),
            **({"split": outcome.split} if outcome.split else {}),
            **({"flags": outcome.flags} if outcome.flags else {}),
            **({"log_check": outcome.log_check} if outcome.log_check else {}),
            **({"effort_after": outcome.effort} if outcome.effort else {}),
            **({"notes": outcome.notes} if outcome.notes else {}),
        }
        files: list[str] = []
        if outcome.agent_result is not None and not dry_run:
            from benchflow._utils.result_metadata import incomplete_usage_fields

            updated = dict(result)
            # A count that misses calls keeps saying so once priced here:
            # cost_incomplete beside the cost, as a trial's own teardown writes.
            agent_result = {
                **outcome.agent_result,
                **incomplete_usage_fields(outcome.agent_result),
            }
            updated["agent_result"] = agent_result
            updated["final_metrics"] = _final_metrics(
                result.get("final_metrics"),
                agent_result,
                keep_totals=_keeps_totals(agent_result),
            )
            if outcome.effort is not None:
                updated["effort"] = outcome.effort
            # Every derived file first, result.json (the commit point) last:
            # a run stopped between them redoes the trial when run again.
            atif = _atif_updated(trial, outcome.agent_result)
            if atif is not None and _rewrite(trial / _ATIF, atif):
                files.append(str((trial / _ATIF).relative_to(job_dir)))
            if _rewrite(path, updated):
                files.append(str(path.relative_to(job_dir)))
        line["files_written"] = files
        written += files
        lines.append(line)
    for folder in missing:
        line: dict[str, Any] = {
            "trial": str(folder.relative_to(job_dir)),
            "status": UNCOVERED,
            "reason": NO_RESULT,
        }
        spent = _unrecorded_spend(folder)
        if spent is not None:
            line["log_check"] = spent
        lines.append(line)
    for folder in others:
        lines.append(
            {
                "trial": str(folder.relative_to(job_dir)),
                "status": UNCOVERED,
                "reason": NOT_BENCHFLOW_TRIAL,
            }
        )
    summary_before = _read_json(job_dir / "summary.json")
    summary_state = "absent" if not isinstance(summary_before, dict) else "unchanged"
    summary_after = summary_before
    summary_changed: list[str] = []
    if isinstance(summary_before, dict):
        updated_summary = None if dry_run else _summary_updated(job_dir, stamp)
        if updated_summary is not None and _rewrite(
            job_dir / "summary.json", updated_summary
        ):
            summary_state = "updated"
            summary_after = updated_summary
            summary_changed = sorted(
                key
                for key, value in updated_summary.items()
                if key not in ("backfill", "backfilled_at", "backfill_version")
                and summary_before.get(key, _ABSENT) != value
            )
            written.append("summary.json")
        elif dry_run:
            summary_state = "not_written_dry_run"
    rows_written: list[str] = []
    card_state = "not_written_dry_run"
    if not dry_run:
        written += _sync_atif(job_dir, results)
        rows_written = _sync_results_jsonl(job_dir, results)
        written += rows_written
        card_state, card_written = _sync_run_card(job_dir, stamp)
        if card_written:
            written.append("run-card.json")
    jobs_root_summary = _jobs_root_copy(job_dir)
    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    flag_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for line in lines:
        for flag in line.get("flags") or []:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
        split = line.get("split") or {}
        if split:
            key = (
                "converted"
                if split.get("split_converted")
                else (f"left:{split.get('split_reason')}")
            )
            split_counts[key] = split_counts.get(key, 0) + 1
        status_counts[line["status"]] = status_counts.get(line["status"], 0) + 1
        if line.get("reason"):
            key = f"{line['status']}:{line['reason']}"
            reason_counts[key] = reason_counts.get(key, 0) + 1
    from benchflow.providers.pricing_snapshot import SNAPSHOT_DATE

    summary_keys = (
        "total_cost_usd",
        *_CAVEATS,
        "total_input_tokens",
        "total_output_tokens",
        "total_cache_read_tokens",
        "total_cache_write_tokens",
        "total_tokens",
        "trials_without_cost",
        "trials_without_usage",
        "trials_usage_incomplete",
        "trials_cost_incomplete",
        "usage_sources",
        "cost_sources",
        "spend_all_attempts",
    )
    report = {
        "schema": SCHEMA,
        "job_dir": str(job_dir),
        "generated_at": stamp["at"],
        "report_file": None,
        "backfill_version": stamp["version"],
        "dry_run": dry_run,
        "price_snapshot_date": SNAPSHOT_DATE,
        "job": {
            "trials": len(lines),
            "status_counts": dict(sorted(status_counts.items())),
            "reason_counts": dict(sorted(reason_counts.items())),
            # Pre-proxied inputs that counted the cache, put on the
            # split or left with their reason.
            "split_counts": dict(sorted(split_counts.items())),
            "flag_counts": dict(sorted(flag_counts.items())),
            "before": _totals(before_lines),
            "after": _totals(after_lines),
            "summary_json": summary_state,
            "summary_keys_changed": summary_changed,
            "results_jsonl_written": rows_written,
            "run_card": card_state,
            "jobs_root_summary": jobs_root_summary,
            "summary_before": {
                key: (summary_before or {}).get(key) for key in summary_keys
            }
            if isinstance(summary_before, dict)
            else None,
            "summary_after": {
                key: (summary_after or {}).get(key) for key in summary_keys
            }
            if isinstance(summary_after, dict)
            else None,
            "files_written": written,
        },
        "trials": lines,
    }
    # Each run's report is kept (backfill-report.<UTC>.json); the stable name
    # holds the latest, naming its run's file.
    from benchflow._utils.disk_space import write_text_atomic

    target = Path(report_path) if report_path is not None else job_dir / REPORT_FILE
    own = target.with_name(f"{target.stem}.{stamp['id']}{target.suffix}")
    report["report_file"] = own.name
    text = _dump(report, newline=True)
    write_text_atomic(own, text)
    write_text_atomic(target, text)
    return report
