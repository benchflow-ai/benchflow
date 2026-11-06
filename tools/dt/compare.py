"""Compare two arms' snapshots cell by cell, sort every difference into a
class, and check each against the divergence ledger.

A snapshot is the JSONL ``run.py`` writes: one line per cell (task, agent,
round) with the pinned reader's record (``normalise.py``), the raw view of
``result.json``/``timing.json``, and the arm's own reader output
(``bench eval inspect --json``). Cells pair on the task's corpus name and
its source tree hash, never on ``task_digest`` (the engines compute it
differently for split layouts).

Two modes:

- ``before-after`` (two builds of one engine): ``regression``, ``fix``,
  ``newly_refused``, ``newly_accepted``, ``outcome_flip``, ``reward_change``,
  ``verifier_output_change``, ``field_change``, ``key_added``,
  ``key_removed``, ``raw_value_change``, ``summary_change``,
  ``reader_drift``, ``nop_above_zero``, ``probe_above_nop``,
  ``timing_shift``.
- ``cross-engine`` (Python against Rust): ``python_refuses_rust_runs``,
  ``rust_refuses_python_runs``, ``score_disagree``, ``spec_mismatch``,
  ``field_change``, ``verifier_output_change``, ``summary_change``,
  ``reader_drift``.

A class that says the whole cell changed kind (``nop_above_zero``, a refusal
that starts or stops, either engine refusing alone) stands alone on its cell:
the field, verifier, reward and summary-count changes that follow from it
are not listed again. Scoring exports remain independent findings.

Every class found must be matched by a row of ``divergences/accepted.toml``,
``triaged.toml`` or ``flaky.toml``, or by an ``expected-changes.toml`` entry,
or the run fails. An accepted or triaged row that matches nothing in a full
run of its tier fails too (a stale ignore hides the next real change), as
does a triaged row that still matches after its
owner landed (``results/status.md``), a flaky row with fewer than 3 repeats
of evidence, and an expected-changes entry without exact cells, a target id
and a score-changing item. A cell the reader could not read (a changed refusal
format included) is never matched by anything.

Imports nothing from ``benchflow`` (a test checks it); timing statistics
come from ``tools/perf/summarise``.

    python3 -I tools/dt/compare.py before.jsonl after.jsonl --mode before-after \\
        --out report.json --markdown report.md
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
import tomllib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    # Run as a script (python3 -I tools/dt/compare.py): make `tools`
    # importable from the repository root, never from the current folder.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.dt import normalise
from tools.perf.summarise import bootstrap_interval, median

HERE = Path(__file__).parent
DIVERGENCES = HERE / "divergences"
EXPECTED_CHANGES = HERE / "expected-changes.toml"
SPEC = HERE / "spec" / "expected.toml"
TIMING_NULL = HERE / "timing-null.toml"

BEFORE_AFTER_CLASSES = (
    "regression",
    "fix",
    "newly_refused",
    "newly_accepted",
    "outcome_flip",
    "reward_change",
    "verifier_output_change",
    "field_change",
    "key_added",
    "key_removed",
    "raw_value_change",
    "summary_change",
    "reader_drift",
    "nop_above_zero",
    "probe_above_nop",
    "timing_shift",
)
CROSS_ENGINE_CLASSES = (
    "python_refuses_rust_runs",
    "rust_refuses_python_runs",
    "score_disagree",
    "spec_mismatch",
    "field_change",
    "verifier_output_change",
    "summary_change",
    "reader_drift",
)
# Classes that stand alone on a cell (everything else there follows).
SUBSUMING = (
    "nop_above_zero",
    "newly_refused",
    "newly_accepted",
    "python_refuses_rust_runs",
    "rust_refuses_python_runs",
)
# Never matched by a ledger row: the reader could not read the cell.
UNREADABLE = "unreadable"
TIMING_CLASSES = ("timing_shift",)
PHASES = ("setup", "agent", "verify")
# A phase shifts when every paired round moves the same way (the bootstrap
# interval of the paired median excludes 0) by at least this much. Python
# writes phases to 0.1 s.
TIMING_MIN_SHIFT_SEC = 1.0
TIMING_MIN_ROUNDS = 3

# Record fields compared field by field (field_change) in each mode. The
# cross-engine set leaves out what one engine does not record at all
# (Rust writes no trajectory_summary and its token counts for scripted
# agents are always 0): those are DROP-like gaps, not divergences.
BEFORE_AFTER_FIELDS = (
    "family",
    "code",
    "error_kind",
    "verifier_error_kind",
    "refusal.paths",
    "counts.*",
    "prompt.n",
    "prompt.raw_sha256",
    "prompt.stripped_sha256",
    "task_digest",
    "skill_mode",
    "attempts",
)
CROSS_ENGINE_FIELDS = (
    "family",
    "refusal.paths",
    "counts.n_prompts",
    "counts.n_tool_calls",
    "prompt.n",
    "prompt.raw_sha256",
    "prompt.stripped_sha256",
    "task_digest",
)
REFUSAL_FIELDS = ("family", "refusal.paths")
# The canonical outcome vocabulary (normalise.py).
REFUSED = {"refused_at_load"}

# The score-changing items; an expected-changes entry names one
# (GOALS DT3: "never a whole item"). "DT1" is the one records change GOALS
# DT1 lists for REL (refusals counted as unsupported_task_feature).
D44_ITEMS = (
    "P4",
    "K2",
    "K1",
    "S1",
    "S2",
    "S3",
    "R3 B10",
    "R3 B11",
    "U1",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "J1",
    "P78",
    "U4",
    "AG",
    "PT",
    "DT1",
)
SPEC_SIDES = ("python", "rust", "both", "neither", "n/a")


class LedgerError(ValueError):
    """A ledger row or expected-changes entry that cannot be used."""


@dataclass
class Finding:
    cell: str
    cls: str
    rounds: list[int] = field(default_factory=list)
    detail: list[str] = field(default_factory=list)
    arm: str | None = None
    matched_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell": self.cell,
            "class": self.cls,
            "rounds": sorted(set(self.rounds)),
            "detail": sorted(set(self.detail)),
            "arm": self.arm,
            "matched_by": self.matched_by,
        }


# --- snapshots -------------------------------------------------------------


def load_snapshot(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def cell_key(cell: Mapping[str, Any]) -> str:
    return f"{cell['name']}/{cell['agent']}"


def _cell_parts(rounds: Mapping[int, Mapping[str, Any]]) -> tuple[str, str]:
    """A cell's task and agent, from its rows' own fields: never split back
    out of the key, as an agent can hold a slash
    ("script:agents/write-hello.sh")."""
    cell = next(iter(rounds.values()))["cell"]
    return str(cell["name"]), str(cell["agent"])


def _index(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[int, Mapping[str, Any]]]:
    out: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        cell = row["cell"]
        out[cell_key(cell)][int(cell.get("round", 1))] = row
    return out


def _get(record: Mapping[str, Any] | None, dotted: str) -> Any:
    value: Any = record
    for part in dotted.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def _field_values(
    record: Mapping[str, Any] | None, fields: Sequence[str]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for spec in fields:
        if spec.endswith(".*"):
            sub = _get(record, spec[:-2])
            if isinstance(sub, Mapping):
                for key, value in sub.items():
                    out[f"{spec[:-2]}.{key}"] = value
            else:
                out[spec[:-2]] = sub
        else:
            out[spec] = _get(record, spec)
    return out


def _refused(record: Mapping[str, Any] | None) -> bool:
    if record is None:
        return False
    return record.get("outcome") in REFUSED or record.get("refusal") is not None


def _reward(record: Mapping[str, Any] | None) -> float | None:
    return None if record is None else record.get("reward")


# --- classes ---------------------------------------------------------------


def outcome_classes(
    a: Mapping[str, Any], b: Mapping[str, Any]
) -> list[tuple[str, str]]:
    """Before/after classes from outcome and reward."""
    out: list[tuple[str, str]] = []
    oa, ob = a.get("outcome"), b.get("outcome")
    ra, rb = _reward(a), _reward(b)
    agent = b.get("agent") or a.get("agent") or ""
    if agent == "nop" and (rb or 0) > 0 and not (ra or 0) > 0:
        # The empty control now scores: everything else on the cell follows.
        return [("nop_above_zero", f"reward {ra} -> {rb}")]
    if not _refused(a) and _refused(b):
        out.append(("newly_refused", f"{oa} -> {ob}"))
    elif _refused(a) and not _refused(b):
        out.append(("newly_accepted", f"{oa} -> {ob}"))
    elif oa != ob:
        if oa == "passed":
            out.append(("regression", f"{oa} -> {ob}"))
        elif ob == "passed":
            out.append(("fix", f"{oa} -> {ob}"))
        else:
            out.append(("outcome_flip", f"{oa} -> {ob}"))
    if ra != rb:
        out.append(("reward_change", f"{ra} -> {rb}"))
    return out


def field_classes(
    a: Mapping[str, Any], b: Mapping[str, Any], fields: Sequence[str]
) -> list[tuple[str, str]]:
    va, vb = _field_values(a, fields), _field_values(b, fields)
    changed = sorted(k for k in set(va) | set(vb) if va.get(k) != vb.get(k))
    return [("field_change", k) for k in changed]


def verifier_classes(
    a: Mapping[str, Any], b: Mapping[str, Any]
) -> list[tuple[str, str]]:
    va, vb = a.get("verifier") or {}, b.get("verifier") or {}
    if _refused(a) or _refused(b):
        return []
    changed = sorted(k for k in set(va) | set(vb) if va.get(k) != vb.get(k))
    return [("verifier_output_change", k) for k in changed]


def _same_json(a: Any, b: Any) -> bool:
    """JSON numbers may use either numeric spelling; booleans are distinct."""
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(_same_json(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            _same_json(x, y) for x, y in zip(a, b, strict=True)
        )
    if type(a) is not type(b):
        return type(a) in (int, float) and type(b) in (int, float) and a == b
    return a == b


def _scoring_display(value: Any) -> str:
    # Preserve historical null details, but quote strings and preserve the
    # JSON types inside collections. Exact expectations must not accept
    # numeric-looking strings or bools as numbers (CONF2-DT1-2).
    return "None" if value is None else json.dumps(value, sort_keys=True)


def summary_classes(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    cross: bool,
    allow_keys: Sequence[str] = (),
) -> list[tuple[str, str]]:
    sa, sb = a.get("summary"), b.get("summary")
    load_transition = False
    if sa is None or sb is None:
        # A planner refusal writes no job/trial summary. Its appearance or
        # disappearance follows the accepted/refused transition, unlike a
        # completed job losing its scoring artifact (CONF-DT1-2).
        load_transition = (sa is None and _refused_at_load(a)) or (
            sb is None and _refused_at_load(b)
        )
        if not load_transition and (sa is None) != (sb is None):
            return [("key_removed" if sb is None else "key_added", "summary.json")]
        if not load_transition:
            return []
        # No artifact is expected from the refused arm, but every scoring
        # field in the completed arm still needs an exact transition entry
        # (CONF2-DT1-1). Accepting a refusal never accepts arbitrary scores.
        sa, sb = sa or {}, sb or {}
    out = []
    # The scoring fields: all of them between two builds of
    # one engine; across engines only those both define one way.
    ga, gb = sa.get("scoring") or {}, sb.get("scoring") or {}
    if cross:
        ga = {k: v for k, v in ga.items() if k in normalise.SUMMARY_SCORING_CROSS}
        gb = {k: v for k, v in gb.items() if k in normalise.SUMMARY_SCORING_CROSS}
    scoring_keys = set(ga) | set(gb)
    if load_transition and not (_refused_at_load(a) and _refused_at_load(b)):
        # A completed arm must export both shared ratios (CONF3-DT1-1).
        # The refused arm has neither, so an observed-key union alone loses
        # an omitted export. Keep the fixed required keys even when both
        # maps lack them; their missing -> missing detail is independent of
        # the accepted refusal and cannot match a valid score transition.
        scoring_keys.update(normalise.SUMMARY_SCORING_CROSS)
    for key in sorted(scoring_keys):
        if key not in ga or key not in gb or not _same_json(ga[key], gb[key]):
            va = _scoring_display(ga[key]) if key in ga else "<missing>"
            vb = _scoring_display(gb[key]) if key in gb else "<missing>"
            out.append(("summary_change", f"{key} {va} -> {vb}"))
    if load_transition:
        return out
    if not cross and sa.get("keys") is not None and sb.get("keys") is not None:
        # The raw fallback: a summary field that comes or goes.
        ka, kb = set(sa["keys"]), set(sb["keys"])
        for key in sorted(kb - ka):
            if not _allowed(f"summary.json:{key}", allow_keys):
                out.append(("key_added", f"summary.json:{key}"))
        for key in sorted(ka - kb):
            if not _allowed(f"summary.json:{key}", allow_keys):
                out.append(("key_removed", f"summary.json:{key}"))
    ca, cb = sa["counts"], sb["counts"]
    if cross:
        # The engines bucket a trial with no verdict differently (Python:
        # verifier_errored; Rust counts it as verifier_errored and as
        # unscored); compare what both mean.
        ca, cb = _cross_counts(ca), _cross_counts(cb)
    for key in sorted(set(ca) | set(cb)):
        if ca.get(key) != cb.get(key):
            out.append(
                (
                    "summary_change",
                    f"{key} {ca.get(key)} -> {cb.get(key)}",
                )
            )
    if cross:
        if sa.get("families") != sb.get("families"):
            out.append(
                (
                    "summary_change",
                    f"families {sa.get('families')} -> {sb.get('families')}",
                )
            )
    elif sa.get("raw") != sb.get("raw"):
        out.append(("summary_change", f"categories {sa.get('raw')} -> {sb.get('raw')}"))
    return out


def _cross_counts(counts: Mapping[str, Any]) -> dict[str, Any]:
    def n(key: str) -> int:
        return int(counts.get(key) or 0)

    return {
        "total": n("total"),
        "passed": n("passed"),
        "failed": n("failed"),
        "errored": n("errored"),
        "no_verdict": n("total") - n("passed") - n("failed") - n("errored"),
    }


def raw_classes(
    a: Mapping[str, Any] | None,
    b: Mapping[str, Any] | None,
    allow_keys: Sequence[str] = (),
) -> list[tuple[str, str]]:
    """The raw key-set and value diff beside the normalised one."""
    if not a or not b:
        return []
    out: list[tuple[str, str]] = []
    for name in ("result.json", "timing.json"):
        fa, fb = a.get(name), b.get(name)
        if fa is None or fb is None:
            if (fa is None) != (fb is None):
                out.append(("key_removed" if fb is None else "key_added", name))
            continue
        ka, kb = set(fa["keys"]), set(fb["keys"])
        for key in sorted(kb - ka):
            if not _allowed(f"{name}:{key}", allow_keys):
                out.append(("key_added", f"{name}:{key}"))
        for key in sorted(ka - kb):
            if not _allowed(f"{name}:{key}", allow_keys):
                out.append(("key_removed", f"{name}:{key}"))
        if name == "result.json":
            va, vb = fa.get("values") or {}, fb.get("values") or {}
            for key in sorted(set(va) & set(vb)):
                if va[key] != vb[key] and not normalise.covered(key):
                    out.append(("raw_value_change", f"{name}:{key}"))
    return out


def _allowed(key: str, allow_keys: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(key, pattern) for pattern in allow_keys)


def reader_drift(row: Mapping[str, Any]) -> list[str]:
    """Where the arm's own reader (bench eval inspect --json) disagrees with
    the pinned reader on this cell."""
    reader, record = row.get("reader"), row.get("record")
    # None is an arm with no reader of its own (Rust); an empty mapping is a
    # Python reader that listed nothing, which is drift.
    if reader is None or not record or _refused_at_load(record):
        return []
    seen = reader.get(record.get("task")) if isinstance(reader, Mapping) else None
    if seen is None:
        return ["the reader lists no trial for this task"]
    out = []
    others = sorted(str(k) for k in reader if k != record.get("task"))
    if others:
        out.append(f"the reader lists other trials: {', '.join(others)}")
    if seen.get("outcome") != record.get("outcome"):
        out.append(
            f"outcome: reader {seen.get('outcome')}, pinned {record.get('outcome')}"
        )
    if seen.get("reward") != record.get("reward"):
        out.append(
            f"reward: reader {seen.get('reward')}, pinned {record.get('reward')}"
        )
    # Who failed: the reader's category against result.json's, which the
    # pinned record keeps. A record from before that field
    # existed has no key and is not checked.
    for key in ("error_category", "verifier_error_category"):
        if key in record and seen.get(key) != record.get(key):
            out.append(f"{key}: reader {seen.get(key)}, pinned {record.get(key)}")
    return out


def _refused_at_load(record: Mapping[str, Any]) -> bool:
    return record.get("outcome") == "refused_at_load"


def cross_classes(
    py: Mapping[str, Any], rs: Mapping[str, Any]
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if _refused(py) and not _refused(rs):
        out.append(
            (
                "python_refuses_rust_runs",
                f"python {py.get('outcome')}, rust {rs.get('outcome')}",
            )
        )
    elif _refused(rs) and not _refused(py):
        out.append(
            (
                "rust_refuses_python_runs",
                f"rust {rs.get('outcome')}, python {py.get('outcome')}",
            )
        )
    elif (not _refused(py) and not _refused(rs)) and (
        py.get("outcome") != rs.get("outcome") or py.get("reward") != rs.get("reward")
    ):
        out.append(
            (
                "score_disagree",
                f"python {py.get('outcome')} {py.get('reward')}, rust {rs.get('outcome')} {rs.get('reward')}",
            )
        )
    return out


def spec_classes(
    name: str,
    agent: str,
    records: Mapping[str, Mapping[str, Any] | None],
    spec_rows: Sequence[Mapping[str, Any]],
    categories: Mapping[str, Any],
) -> list[tuple[str, str]]:
    """spec_mismatch for every side whose record misses the documented
    behaviour (spec/expected.toml), with the side named."""
    out = []
    for row in spec_rows:
        if row["task"] != name or row.get("agent", "oracle") != agent:
            continue
        for side, record in records.items():
            if record is None or side not in row.get("engines", ("python", "rust")):
                continue
            problem = spec_problem(row, record, categories)
            if problem:
                out.append(("spec_mismatch", f"{side}: {problem} ({row['id']})"))
    return out


def spec_problem(
    row: Mapping[str, Any], record: Mapping[str, Any], categories: Mapping[str, Any]
) -> str | None:
    expected = row["outcome"]
    outcome = record.get("outcome")
    # A refusal at load and an errored refusal are one documented outcome:
    # the task is refused before any sandbox starts.
    if expected == "refused" and _refused(record):
        outcome = "refused"
    if outcome != expected:
        return f"outcome {outcome}, spec {expected}"
    if "reward" in row and record.get("reward") != row["reward"]:
        return f"reward {record.get('reward')}, spec {row['reward']}"
    code = row.get("code") or ""
    if code:
        # Rust records its R-code: it must be the documented one, as two codes
        # of one family can name different causes and owners (R101 the task's
        # image, R102 a build timeout). Python has no R-code
        # outside refusals, so its failure family stands in.
        if record.get("engine") == "rust":
            if record.get("code") != code:
                return f"code {record.get('code')}, spec {code}"
            return None
        family = categories["rust"]["code"].get(code)
        if record.get("family") != family:
            return f"family {record.get('family')}, spec {family} ({code})"
    return None


def load_timing_null(path: Path = TIMING_NULL) -> dict[str, float]:
    return {k: float(v) for k, v in _load_toml(path).get("seconds", {}).items()}


def timing_classes(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    min_shift: float = TIMING_MIN_SHIFT_SEC,
    null: Mapping[str, float] | None = None,
) -> list[tuple[str, str]]:
    """timing_shift per phase, from rounds paired a/b (tools/perf/summarise).

    A phase shifts when the median of its paired differences is larger than
    both ``min_shift`` and the A/A null's largest difference for that phase
    (timing-null.toml), and the bootstrap interval of that median excludes 0.
    """
    out = []
    null = null or {}
    for phase in PHASES:
        threshold = max(min_shift, null.get(phase, 0.0))
        diffs = []
        for a, b in pairs:
            ta, tb = _get(a, f"timing.{phase}"), _get(b, f"timing.{phase}")
            if isinstance(ta, int | float) and isinstance(tb, int | float):
                diffs.append(float(tb) - float(ta))
        if len(diffs) < TIMING_MIN_ROUNDS:
            continue
        mid = median(diffs)
        low, high = bootstrap_interval(diffs)
        if abs(mid) > threshold and (low > 0 or high < 0):
            out.append(
                ("timing_shift", f"{phase} {mid:+.2f} s [{low:+.2f}, {high:+.2f}]")
            )
    return out


# --- the comparison --------------------------------------------------------


def compare(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    spec_rows: Sequence[Mapping[str, Any]] = (),
    allow_keys: Sequence[str] = (),
    categories: Mapping[str, Any] | None = None,
    timing_null: Mapping[str, float] | None = None,
) -> list[Finding]:
    """Every class on every cell; side ``a`` is before (or Python), ``b`` after
    (or Rust)."""
    if mode not in ("before-after", "cross-engine"):
        raise ValueError(f"mode is before-after or cross-engine, not {mode!r}")
    categories = categories if categories is not None else normalise.load_categories()
    cross = mode == "cross-engine"
    ia, ib = _index(before), _index(after)
    findings: dict[tuple[str, str, str | None], Finding] = {}

    def add(cell: str, cls: str, rnd: int, detail: str, arm: str | None = None) -> None:
        key = (cell, cls, arm)
        finding = findings.setdefault(key, Finding(cell, cls, arm=arm))
        finding.rounds.append(rnd)
        finding.detail.append(detail)

    for cell in sorted(set(ia) | set(ib)):
        ra, rb = ia.get(cell, {}), ib.get(cell, {})
        if not ra or not rb:
            add(cell, UNREADABLE, 0, f"only in {'after' if not ra else 'before'}")
            continue
        trees = {r["cell"].get("tree") for r in [*ra.values(), *rb.values()]}
        if len(trees) != 1:
            add(cell, UNREADABLE, 0, f"tree hashes differ: {sorted(map(str, trees))}")
            continue
        pairs = []
        for rnd in sorted(set(ra) | set(rb)):
            a, b = ra.get(rnd), rb.get(rnd)
            if a is None or b is None:
                add(cell, UNREADABLE, rnd, "round missing in one arm")
                continue
            for side, row in (("a", a), ("b", b)):
                if row.get("error") or row.get("record") is None:
                    add(
                        cell,
                        UNREADABLE,
                        rnd,
                        f"{side}: {row.get('error') or 'no record'}",
                        side,
                    )
                for problem in reader_drift(row):
                    add(cell, "reader_drift", rnd, problem, side)
            rec_a, rec_b = a.get("record"), b.get("record")
            if rec_a is None or rec_b is None:
                continue
            if cross:
                classes = cross_classes(rec_a, rec_b)
                # Both engines refuse: only what they refuse is compared (a
                # task refused at load has no counts, prompt or digest).
                both_refused = _refused(rec_a) and _refused(rec_b)
                fields = REFUSAL_FIELDS if both_refused else CROSS_ENGINE_FIELDS
                classes += field_classes(rec_a, rec_b, fields)
                classes += verifier_classes(rec_a, rec_b)
                classes += summary_classes(rec_a, rec_b, cross=True)
                name, agent = _cell_parts(ra)
                classes += spec_classes(
                    name, agent, {"python": rec_a, "rust": rec_b}, spec_rows, categories
                )
            else:
                classes = outcome_classes(rec_a, rec_b)
                classes += field_classes(rec_a, rec_b, BEFORE_AFTER_FIELDS)
                classes += verifier_classes(rec_a, rec_b)
                classes += raw_classes(a.get("raw"), b.get("raw"), allow_keys)
                classes += summary_classes(
                    rec_a, rec_b, cross=False, allow_keys=allow_keys
                )
            classes = _drop_consequences(classes)
            for cls, detail in classes:
                add(cell, cls, rnd, detail)
            pairs.append((rec_a, rec_b))
        if not cross:
            for cls, detail in timing_classes(pairs, null=timing_null):
                add(cell, cls, 0, detail)
    if not cross:
        _probe_classes(ib, ia, add)
    return sorted(findings.values(), key=lambda f: (f.cell, f.cls, f.arm or ""))


OUTCOME_CLASSES = (
    "regression",
    "fix",
    "outcome_flip",
    "reward_change",
    "score_disagree",
)
# What a changed outcome or reward writes everywhere: the job's counts, the
# raw category fields, the verifier's files, the failure family, code and
# number of attempts, and timing.json's keys (a phase that did not run
# writes no key).
CONSEQUENCES = {
    "summary_change": None,
    "raw_value_change": (
        "result.json:error_category",
        "result.json:verifier_error_category",
    ),
    "verifier_output_change": None,
    "field_change": (
        "family",
        "code",
        "attempts",
        "error_kind",
        "verifier_error_kind",
    ),
    "key_added": ("timing.json:*", "result.json:rewards", "result.json:rewards.*"),
    "key_removed": ("timing.json:*", "result.json:rewards", "result.json:rewards.*"),
}


def _drop_consequences(classes: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Keep each change once: a class that stands alone takes its cell, and
    a changed outcome or reward is not listed again as what it changed
    (``CONSEQUENCES``). Scoring exports and summary presence remain
    independent, including when result.json stores no changed outcome."""

    # An outcome transition establishes no particular scoring export. Keep
    # every scoring-field change, even on an accepted/refused/nop cell, and
    # ledger the exact intended transitions (CONF-DT1-1). Counts and category
    # buckets remain consequences under the existing one-trial contract.
    def independent(cls: str, detail: str) -> bool:
        return (
            cls == "summary_change"
            and any(
                fnmatch.fnmatchcase(detail.split(" ", 1)[0].split(".", 1)[0], p)
                for p in normalise.SUMMARY_SCORING
            )
        ) or (
            cls in ("key_added", "key_removed")
            and (
                detail == "summary.json"
                or (
                    detail.startswith("summary.json:")
                    and any(
                        fnmatch.fnmatchcase(detail.split(":", 1)[1], p)
                        for p in normalise.SUMMARY_SCORING
                    )
                )
            )
        )

    if any(cls in SUBSUMING for cls, _ in classes):
        return [
            c
            for c in classes
            if c[0] in SUBSUMING or c[0] == "spec_mismatch" or independent(*c)
        ]
    if not any(cls in OUTCOME_CLASSES for cls, _ in classes):
        return classes
    kept = []
    for cls, detail in classes:
        if independent(cls, detail):
            kept.append((cls, detail))
            continue
        if cls in CONSEQUENCES:
            paths = CONSEQUENCES[cls]
            if paths is None or any(fnmatch.fnmatchcase(detail, p) for p in paths):
                continue
        kept.append((cls, detail))
    return kept


def _probe_classes(after, before, add) -> None:
    """probe_above_nop: a probe agent scores above the nop on the same task
    and round in the after arm, where it did not before."""
    for cell, rounds in after.items():
        name, agent = _cell_parts(rounds)
        if not agent.startswith("probe:"):
            continue
        for rnd, row in rounds.items():
            nop_b = after.get(f"{name}/nop", {}).get(rnd)
            probe_a = before.get(cell, {}).get(rnd)
            nop_a = before.get(f"{name}/nop", {}).get(rnd)
            rb = _reward(row.get("record"))
            nb = _reward(nop_b.get("record")) if nop_b else None
            if rb is None or nb is None or rb <= nb:
                continue
            ra = _reward(probe_a.get("record")) if probe_a else None
            na = _reward(nop_a.get("record")) if nop_a else None
            if ra is not None and na is not None and ra > na:
                continue
            add(cell, "probe_above_nop", rnd, f"probe {rb} > nop {nb}")


# --- the ledger ------------------------------------------------------------


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def load_rows(path: Path, key: str = "row") -> list[dict[str, Any]]:
    return list(_load_toml(path).get(key, []))


def validate_ledger_row(row: Mapping[str, Any], kind: str) -> list[str]:
    """Problems with one accepted/triaged/flaky row (empty when usable)."""
    rid = row.get("id", "<no id>")
    problems = []
    for key in ("id", "mode", "cells", "class", "cite"):
        if not row.get(key):
            problems.append(f"{kind} row {rid}: no {key}")
    if row.get("kind") == "drop":
        if not row.get("field"):
            problems.append(f"{kind} row {rid}: a drop row names the dropped field")
        elif row["field"] not in normalise.DROP:
            problems.append(
                f"{kind} row {rid}: drop row names {row['field']!r}, which normalise.DROP does not drop (stale)"
            )
    if kind in ("accepted", "triaged"):
        if row.get("spec_side") not in SPEC_SIDES:
            problems.append(
                f"{kind} row {rid}: spec_side is one of {', '.join(SPEC_SIDES)}"
            )
        if not row.get("owner"):
            problems.append(f"{kind} row {rid}: no owner")
    if kind == "flaky":
        repeats = row.get("repeats")
        if not isinstance(repeats, int) or repeats < 3:
            problems.append(
                f"flaky row {rid}: {repeats!r} repeats of evidence; a flaky row needs at least 3"
            )
        if not row.get("flip") or not row.get("reason"):
            problems.append(f"flaky row {rid}: names the flip and a reason")
    if row.get("mode") not in (None, "before-after", "cross-engine", "any"):
        problems.append(f"{kind} row {rid}: mode is before-after, cross-engine or any")
    return problems


_GLOB = re.compile(r"[*?\[\]]")


def validate_expected_change(
    entry: Mapping[str, Any], *, items: Sequence[str] = D44_ITEMS
) -> list[str]:
    """An expected-changes entry names exact cells, a target id and a
    score-changing item (never a whole item: DT3)."""
    eid = entry.get("id", "<no id>")
    problems = []
    cells = entry.get("cells")
    if not cells or not isinstance(cells, list):
        problems.append(f"expected change {eid}: names no cells")
    elif any(_GLOB.search(str(c)) or "/" not in str(c) for c in cells):
        problems.append(
            f"expected change {eid}: cells must be exact <task>/<agent>, no globs"
        )
    if not entry.get("target"):
        problems.append(f"expected change {eid}: names no target id")
    item = str(entry.get("item") or "")
    if not item:
        problems.append(f"expected change {eid}: names no score-changing item")
    elif not any(
        item == i or item.startswith(f"{i}:") or item.startswith(f"{i} ") for i in items
    ):
        problems.append(
            f"expected change {eid}: {item!r} is not a score-changing item ({', '.join(items)})"
        )
    if not entry.get("class"):
        problems.append(f"expected change {eid}: names no class")
    return problems


def landed_targets(status_text: str) -> set[str]:
    """Target ids the integrator recorded as landed in results/status.md
    ("integrator: merged T0 ...", "integrator: M0 merged ...", "LANDED <id>")."""
    landed = set()
    for line in status_text.splitlines():
        if "integrator:" in line:
            tail = line.split("integrator:", 1)[1]
            for match in re.finditer(r"\bmerged:? (?:([A-Z][A-Za-z0-9]*)\b)", tail):
                landed.add(match.group(1))
            for match in re.finditer(r"\b([A-Z][A-Za-z0-9]*) merged\b", tail):
                landed.add(match.group(1))
        for match in re.finditer(r"\bLANDED ([A-Z][A-Za-z0-9]*)\b", line):
            landed.add(match.group(1))
    return landed


def _row_covers(row: Mapping[str, Any], finding: Finding, mode: str) -> set[str]:
    """The details of ``finding`` that ``row`` explains (empty: none).

    A row with ``fields`` explains only the details it names (a field_change
    on prompt.raw_sha256 and task_digest needs a row for each); a row
    without explains every detail of its class on its cells.
    """
    if row.get("kind") == "drop":
        return set()
    if row.get("mode", "any") not in ("any", mode):
        return set()
    classes = row["class"] if isinstance(row["class"], list) else [row["class"]]
    if finding.cls not in classes:
        return set()
    if not any(fnmatch.fnmatchcase(finding.cell, pattern) for pattern in row["cells"]):
        return set()
    details = set(finding.detail) or {""}
    fields = row.get("fields")
    if not fields:
        return details
    if row.get("exact_fields"):
        return {d for d in details if d in fields}
    return {
        d
        for d in details
        if any(fnmatch.fnmatchcase(d, f) or d.startswith(f) for f in fields)
    }


def _row_matches(row: Mapping[str, Any], finding: Finding, mode: str) -> bool:
    """Whether ``row`` alone explains every detail of ``finding``."""
    return _row_covers(row, finding, mode) == (set(finding.detail) or {""})


@dataclass
class Verdict:
    findings: list[Finding]
    failures: list[str]
    matched_rows: dict[str, int]

    @property
    def ok(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "failures": self.failures,
            "findings": [f.to_dict() for f in self.findings],
            "matched_rows": self.matched_rows,
        }


def judge(
    findings: Sequence[Finding],
    *,
    mode: str,
    accepted: Sequence[Mapping[str, Any]] = (),
    triaged: Sequence[Mapping[str, Any]] = (),
    flaky: Sequence[Mapping[str, Any]] = (),
    expected: Sequence[Mapping[str, Any]] = (),
    expected_items: Sequence[str] = D44_ITEMS,
    status_text: str = "",
    tier: str = "smoke",
    full: bool = True,
    timing_fails: bool = False,
    rounds: int | None = None,
) -> Verdict:
    """Match findings to the ledger; every rule that fails the run is here.

    Every cell an expected-changes entry names must show its class; with
    ``rounds`` (a mutant's proof), in every one of the run's rounds.
    """
    failures: list[str] = []
    rows: list[tuple[str, Mapping[str, Any]]] = []
    for kind, group in (("accepted", accepted), ("triaged", triaged), ("flaky", flaky)):
        for row in group:
            problems = validate_ledger_row(row, kind)
            failures.extend(problems)
            if not problems:
                rows.append((kind, row))
    entries = []
    for entry in expected:
        problems = validate_expected_change(entry, items=expected_items)
        failures.extend(problems)
        # An entry for one mode (a merge's before/after diff) neither matches
        # nor is required in the other.
        if not problems and entry.get("mode", "any") in ("any", mode):
            entries.append(entry)
    counts: dict[str, int] = {f"{kind}:{row['id']}": 0 for kind, row in rows}
    counts.update({f"expected:{e.get('id', i)}": 0 for i, e in enumerate(entries)})
    seen_cells: dict[str, dict[str, set[int]]] = {}
    for finding in findings:
        finding.matched_by = []  # judge again from scratch
    for finding in findings:
        if finding.cls == UNREADABLE:
            failures.append(f"{finding.cell}: unreadable: {'; '.join(finding.detail)}")
            continue
        covered: set[str] = set()
        for kind, row in rows:
            got = _row_covers(row, finding, mode)
            if got:
                name = f"{kind}:{row['id']}"
                finding.matched_by.append(name)
                counts[name] += 1
                covered |= got
        for i, entry in enumerate(entries):
            probe = {**entry, "cells": entry["cells"], "mode": entry.get("mode", "any")}
            got = _row_covers(probe, finding, mode)
            if got:
                name = f"expected:{entry.get('id', i)}"
                finding.matched_by.append(name)
                counts[name] += 1
                covered |= got
                seen_cells.setdefault(name, {}).setdefault(finding.cell, set()).update(
                    finding.rounds
                )
        missing = (set(finding.detail) or {""}) - covered
        if missing:
            if finding.cls in TIMING_CLASSES and not timing_fails:
                continue
            failures.append(
                f"{finding.cell}: {finding.cls} ({'; '.join(sorted(missing))}) "
                "matches no ledger row or expected change"
            )
            continue
    landed = landed_targets(status_text)
    for kind, row in rows:
        name = f"{kind}:{row['id']}"
        in_scope = (
            full
            and tier in row.get("tiers", ["smoke"])
            and row.get("mode", "any") in ("any", mode)
        )
        # A flaky row is exempt: a flip that happens only sometimes may not
        # happen in this run (its 3 repeats of evidence are its check).
        stale = counts[name] == 0 and in_scope and kind != "flaky"
        # A drop row explains no finding: it is stale when DROP no longer
        # drops its field (validate_ledger_row).
        if stale and row.get("kind") != "drop":
            failures.append(
                f"{name} matches nothing in this run: a stale row hides the next change"
            )
        if kind == "triaged" and counts[name] and str(row.get("owner")) in landed:
            failures.append(
                f"{name} still matches {counts[name]} finding(s) after its owner {row['owner']} landed"
            )
    for i, entry in enumerate(entries):
        name = f"expected:{entry.get('id', i)}"
        if counts[name] == 0:
            failures.append(
                f"{name} ({entry.get('target')}, {entry.get('item')}) matches nothing"
            )
            continue
        timing = entry.get("class") in TIMING_CLASSES
        for cell in entry["cells"]:
            got = seen_cells.get(name, {}).get(cell)
            if got is None:
                failures.append(f"{name}: {cell} does not show {entry.get('class')}")
            elif rounds and not timing and not set(range(1, rounds + 1)) <= got:
                failures.append(
                    f"{name}: {cell} shows {entry.get('class')} in rounds "
                    f"{sorted(got)} of {rounds}"
                )
    return Verdict(list(findings), failures, counts)


def load_spec(path: Path = SPEC) -> list[dict[str, Any]]:
    return load_rows(path)


def render_markdown(verdict: Verdict, *, title: str) -> str:
    lines = [f"# {title}", "", f"Verdict: {'clean' if verdict.ok else 'FAILED'}", ""]
    if verdict.failures:
        lines += ["## Failures", ""] + [f"- {f}" for f in verdict.failures] + [""]
    lines += [
        "## Findings",
        "",
        "| Cell | Class | Rounds | Detail | Matched by |",
        "|---|---|---|---|---|",
    ]
    for f in verdict.findings:
        d = f.to_dict()
        lines.append(
            f"| {d['cell']}{' (' + d['arm'] + ')' if d['arm'] else ''} | {d['class']} | "
            f"{','.join(map(str, d['rounds']))} | {'; '.join(d['detail'])[:300]} | "
            f"{', '.join(d['matched_by']) or '-'} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "before", type=Path, help="snapshot JSONL of arm a (before, or Python)"
    )
    parser.add_argument(
        "after", type=Path, help="snapshot JSONL of arm b (after, or Rust)"
    )
    parser.add_argument(
        "--mode", required=True, choices=("before-after", "cross-engine")
    )
    parser.add_argument("--divergences", type=Path, default=DIVERGENCES)
    parser.add_argument(
        "--expected",
        type=Path,
        default=EXPECTED_CHANGES,
        help="expected-changes.toml, or a mutant's predictions",
    )
    parser.add_argument(
        "--expected-item-prefix",
        action="append",
        default=None,
        help="accept these item names instead of the score-changing items (mutant predictions)",
    )
    parser.add_argument(
        "--no-ledger",
        action="store_true",
        help="match against --expected only (mutants, A/A)",
    )
    parser.add_argument(
        "--status", type=Path, help="results/status.md, for landed owners"
    )
    parser.add_argument("--tier", default="smoke")
    parser.add_argument(
        "--partial",
        action="store_true",
        help="not a full run of the tier: skip the stale-row check",
    )
    parser.add_argument(
        "--timing-fails",
        action="store_true",
        help="an unmatched timing_shift fails the run",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        help="every expected cell must show its class in each of this many rounds",
    )
    parser.add_argument("--allow-key", action="append", default=[])
    parser.add_argument("--timing-null", type=Path, default=TIMING_NULL)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    findings = compare(
        load_snapshot(args.before),
        load_snapshot(args.after),
        mode=args.mode,
        spec_rows=load_spec(),
        allow_keys=args.allow_key,
        timing_null=load_timing_null(args.timing_null),
    )
    ledger = {"accepted": [], "triaged": [], "flaky": []}
    for kind in ledger:
        rows = load_rows(args.divergences / f"{kind}.toml")
        if args.no_ledger and kind != "flaky":
            rows = []
        ledger[kind] = rows
    verdict = judge(
        findings,
        mode=args.mode,
        accepted=ledger["accepted"],
        triaged=ledger["triaged"],
        flaky=ledger["flaky"],
        expected=load_rows(args.expected, "change"),
        expected_items=tuple(args.expected_item_prefix)
        if args.expected_item_prefix
        else D44_ITEMS,
        status_text=args.status.read_text(encoding="utf-8") if args.status else "",
        tier=args.tier,
        full=not args.partial,
        timing_fails=args.timing_fails,
        rounds=args.rounds,
    )
    if args.out:
        args.out.write_text(
            json.dumps(verdict.to_dict(), indent=2, sort_keys=True) + "\n"
        )
    if args.markdown:
        args.markdown.write_text(
            render_markdown(verdict, title=f"{args.before.name} vs {args.after.name}")
        )
    for failure in verdict.failures:
        print(f"FAIL {failure}", file=sys.stderr)
    print(f"{len(verdict.findings)} findings, {len(verdict.failures)} failures")
    return 0 if verdict.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
