"""Read an oracle-parity run against Harbor 0.26.0, with the classes the 0.10.0rc4 gate used.

    python -m tools.parity.tally --rows rows.tsv [--reruns reruns.tsv] [--mode pr|nightly] [--summary FILE]

Each task's oracle reward is compared with Harbor's (``harbor-ref-0.26.0.tsv``). A mismatch was re-run alone
(``tools/parity/run.py``) before it is read. The classes, in the order they are decided:

- ``COUNT`` (blocks): the detector changed a reward, an honest run scored ``provenance_incomplete``, a trial was
  refused for ``integrity_unavailable`` on Docker with the pinned bfx, or a walker other than bfx read the links;
- ``MUST_SHOW`` (blocks): a must-show rule of ``ledger/mustshow.toml`` broken (algotune never 0, sldbench equal to
  Harbor including its -1, the refused task refused before its turn);
- ``REFERENCE_NO_SCORE``: Harbor has no score; context only;
- ``ACCEPTED_UNSCORED``: unscored where Harbor scores, for a reason ``ledger/accepted.toml`` names;
- ``MATCH``; ``FLAKE``: a re-run alone matched (``ledger/flaky.toml`` names tasks that get more re-runs);
- ``ACCEPTED``: a difference ``ledger/accepted.toml`` names with both values (a hardware fact of the task);
- ``TIMEOUT_UNDER_LOAD``: the verifier ran to at least 95% of its cap; a pull request names it, a nightly blocks;
- ``REGRESSION`` (blocks): anything else.

A task ``ledger/excluded.toml`` names (broken upstream, with its evidence) is ``EXCLUDED``, never a
regression; the sampler never draws it.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import re
import sys
import tomllib
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "harbor-ref-0.26.0.tsv"
LEDGER = HERE / "ledger"
ROW_FIELDS = (
    "key",
    "end",
    "reward",
    "why",
    "error_category",
    "reward_changed",
    "provenance_outcome",
    "walker",
    "verifier_secs",
    "verifier_cap",
    "secs",
)
CAP_SHARE = 0.95

#: The row end ``run.py`` writes for a task bigger than the runner (not run).
RUNNER_CAPACITY = "RUNNER_CAPACITY"


@dataclasses.dataclass
class Row:
    key: str
    end: str = "SCORED"
    reward: str = ""
    why: str = ""
    error_category: str = ""
    reward_changed: str = ""
    provenance_outcome: str = ""
    walker: str = ""
    verifier_secs: float | None = None
    verifier_cap: float | None = None
    secs: float | None = None

    @property
    def value(self) -> float | None:
        try:
            return (
                float(self.reward)
                if self.end == "SCORED" and self.reward != ""
                else None
            )
        except ValueError:
            return None


@dataclasses.dataclass
class Ledger:
    accepted: dict[str, tuple[float, float, str]]
    unscored: list[tuple[re.Pattern[str], str]]
    flaky: dict[str, tuple[int, str]]
    mustshow: dict[str, tuple[str, str]]
    excluded: dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Verdict:
    key: str
    cls: str
    blocking: bool
    note: str = ""


def load_reference(path: Path = REFERENCE) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        key, _, score = line.partition("\t")
        out[key] = float(score) if score.strip() not in ("", "-", "None") else None
    return out


def load_ledger(path: Path = LEDGER) -> Ledger:
    acc = tomllib.loads((path / "accepted.toml").read_text())
    flaky = tomllib.loads((path / "flaky.toml").read_text())
    must = tomllib.loads((path / "mustshow.toml").read_text())
    excluded = (
        tomllib.loads((path / "excluded.toml").read_text())
        if (path / "excluded.toml").exists()
        else {}
    )
    return Ledger(
        accepted={
            a["key"]: (float(a["benchflow"]), float(a["harbor"]), a["reason"])
            for a in acc.get("accepted", [])
        },
        unscored=[
            (re.compile(u["pattern"]), u["reason"]) for u in acc.get("unscored", [])
        ],
        flaky={
            f["key"]: (int(f["reruns"]), f["reason"]) for f in flaky.get("flaky", [])
        },
        mustshow={m["key"]: (m["rule"], m["reason"]) for m in must.get("must", [])},
        excluded={e["key"]: e["reason"] for e in excluded.get("excluded", [])},
    )


def reruns_allowed(key: str, ledger: Ledger) -> int:
    return ledger.flaky[key][0] if key in ledger.flaky else 1


def _eq(a: float | None, b: float | None) -> bool:
    return a is not None and b is not None and abs(a - b) < 1e-9


def _counts(row: Row) -> str | None:
    if row.reward_changed.strip().lower() == "true":
        return "the detector changed the reward"
    if row.provenance_outcome == "provenance_incomplete":
        return "an honest oracle run scored provenance_incomplete"
    if row.error_category == "integrity_unavailable":
        return "refused for integrity_unavailable with the pinned bfx"
    if row.walker and row.walker != "bfx":
        return f"the links were read by {row.walker}, not bfx"
    return None


def _must_show(
    row: Row, harbor: float | None, rule: str, reruns: list[Row]
) -> str | None:
    values = [row.value] + [r.value for r in reruns]
    if rule == "never_zero":
        if row.end == "SCORED" and _eq(row.value, 0.0):
            return "scored 0; it must never be 0"
        return None
    if rule == "equals_harbor":
        if not any(_eq(v, harbor) for v in values):
            return f"must equal Harbor's {harbor}, got {row.reward or row.end}"
        return None
    if rule == "refused_before_turn":
        if row.end == "SCORED" or "task refused before the turn" not in row.why:
            return "must be refused before its turn, unscored"
        return None
    raise ValueError(f"unknown must-show rule {rule}")


def classify(
    row: Row,
    ref: dict[str, float | None],
    ledger: Ledger,
    reruns: list[Row],
    mode: str = "pr",
) -> Verdict:
    key = row.key
    if key in ledger.excluded:
        return Verdict(key, "EXCLUDED", False, ledger.excluded[key])
    if row.end == RUNNER_CAPACITY:
        # Not run here; the nightly's runners are sized for every task, so
        # there it blocks.
        return Verdict(key, RUNNER_CAPACITY, mode == "nightly", row.why)
    count = _counts(row)
    if count:
        return Verdict(key, "COUNT", True, count)
    harbor = ref.get(key)
    if key in ledger.mustshow:
        broken = _must_show(row, harbor, ledger.mustshow[key][0], reruns)
        if broken:
            return Verdict(key, "MUST_SHOW", True, broken)
    if harbor is None:
        return Verdict(key, "REFERENCE_NO_SCORE", False, row.end)
    if row.value is None:
        for pattern, reason in ledger.unscored:
            if pattern.search(row.why):
                return Verdict(key, "ACCEPTED_UNSCORED", False, reason)
        return Verdict(
            key,
            "REGRESSION",
            True,
            f"unscored where Harbor scores {harbor}: {row.why[:160]}",
        )
    if _eq(row.value, harbor):
        return Verdict(key, "MATCH", False)
    if any(_eq(r.value, harbor) for r in reruns):
        return Verdict(key, "FLAKE", False, f"{row.reward} in the run, {harbor} alone")
    if key in ledger.accepted:
        bench, href, reason = ledger.accepted[key]
        if _eq(row.value, bench) and _eq(harbor, href):
            return Verdict(key, "ACCEPTED", False, reason)
    if (
        row.verifier_secs
        and row.verifier_cap
        and row.verifier_secs >= CAP_SHARE * row.verifier_cap
    ):
        return Verdict(
            key,
            "TIMEOUT_UNDER_LOAD",
            mode == "nightly",
            f"verifier {row.verifier_secs:.0f}s of its {row.verifier_cap:.0f}s cap",
        )
    return Verdict(
        key, "REGRESSION", True, f"{row.reward} where Harbor scores {harbor}"
    )


def _number(text: dict[str, str], field: str) -> float | None:
    return float(text[field]) if text[field] else None


def load_rows(path: Path | None) -> list[Row]:
    if path is None or not path.exists():
        return []
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            text = {f: (r.get(f) or "") for f in ROW_FIELDS}
            out.append(
                Row(
                    key=text["key"],
                    end=text["end"],
                    reward=text["reward"],
                    why=text["why"],
                    error_category=text["error_category"],
                    reward_changed=text["reward_changed"],
                    provenance_outcome=text["provenance_outcome"],
                    walker=text["walker"],
                    verifier_secs=_number(text, "verifier_secs"),
                    verifier_cap=_number(text, "verifier_cap"),
                    secs=_number(text, "secs"),
                )
            )
    return out


def tally(
    rows: list[Row], reruns: list[Row], ref: dict, ledger: Ledger, mode: str = "pr"
) -> list[Verdict]:
    by_key: dict[str, list[Row]] = {}
    for r in reruns:
        by_key.setdefault(r.key, []).append(r)
    return [classify(r, ref, ledger, by_key.get(r.key, []), mode) for r in rows]


def exit_code(verdicts: list[Verdict]) -> int:
    return 1 if any(v.blocking for v in verdicts) else 0


def render(verdicts: list[Verdict]) -> str:
    counts = Counter(v.cls for v in verdicts)
    lines = [
        "## Oracle parity against Harbor 0.26.0",
        "",
        ", ".join(f"{n} {c}" for c, n in sorted(counts.items())),
        "",
        "| task | class | blocks | note |",
        "|---|---|---|---|",
    ]
    for v in sorted(verdicts, key=lambda v: (not v.blocking, v.cls, v.key)):
        lines.append(
            f"| `{v.key}` | {v.cls} | {'yes' if v.blocking else ''} | {v.note.replace('|', '/')} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--reruns", type=Path)
    ap.add_argument("--reference", type=Path, default=REFERENCE)
    ap.add_argument("--ledger", type=Path, default=LEDGER)
    ap.add_argument("--mode", choices=("pr", "nightly"), default="pr")
    ap.add_argument(
        "--summary", type=Path, help="append the table here (GITHUB_STEP_SUMMARY)"
    )
    args = ap.parse_args(argv)
    rows = load_rows(args.rows)
    if not rows:
        print("tally: no rows", file=sys.stderr)
        return 2
    verdicts = tally(
        rows,
        load_rows(args.reruns),
        load_reference(args.reference),
        load_ledger(args.ledger),
        args.mode,
    )
    text = render(verdicts)
    print(text)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as fh:
            fh.write(text)
    return exit_code(verdicts)


if __name__ == "__main__":
    sys.exit(main())
