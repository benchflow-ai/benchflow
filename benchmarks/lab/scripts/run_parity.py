"""Single-shot parity runner for the LAB adapter.

For each task in the parity subset:

  1. Concatenate the task's source documents (extracted with the same
     readers LAB and the BenchFlow verifier use) and the task instructions
     into a single Gemini prompt.
  2. Generate one deliverable per declared output filename (the same
     Gemini call, fanned out by deliverable name).
  3. Save the generated text as both `.md` (for criteria scoring) and the
     declared `.docx`/`.xlsx` filename (so deliverable matching works).
  4. Score the produced output two ways:
       - **LAB native** path: load the rubric directly from the LAB
         ``task.json`` and call our rubric judge against the same agent
         output.  This is the "original benchmark" arm — it bypasses
         only the harness, not the scoring rubric.
       - **BenchFlow** path: call the translated task's
         ``tests/rubric_judge.py`` (the verifier the BenchFlow runtime
         would invoke) against the same output.
  5. Compare the per-criterion verdicts and the all-pass reward across
     both arms.

Why a one-shot generator?  The harbor parity recipe asks for "same agents,
same models, same settings, both sides".  Running the full LAB podman /
BenchFlow Docker harness on N×3 tasks needs a Docker-permitted host and
hours of wall clock; for the dev sanity-check arm of the recipe a one-shot
Gemini call is enough to exercise translation fidelity (instructions,
documents, rubric, deliverables, judge) end-to-end.  The full agentic
parity (steps 2 and 3 in the harbor recipe) re-uses this script's I/O
contract — see the README for how to swap in `bench run` and
`harness.run`.

Output:
    parity-results/<runs>/{lab,bench}/<task-id>/
        agent_output/<deliverable>          generated text
        scores.json                         per-criterion verdicts + reward
    parity-results/summary.json             aggregated mean ± SEM across runs
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

LAB_DEFAULT_REPO = Path(os.environ.get("LAB_DIR", "/home/user/workspace/harvey-labs"))
BENCHFLOW_REPO = Path(__file__).resolve().parents[3]   # benchflow repo root
ADAPTER_DIR = Path(__file__).resolve().parents[1]      # benchmarks/lab/

sys.path.insert(0, str(ADAPTER_DIR))
from adapter.translate import discover_tasks, sanitize_task_id, write_task  # noqa: E402

LOG = logging.getLogger("lab-parity")

GEMINI_MODEL = os.environ.get("LAB_GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
GEMINI_JUDGE = os.environ.get("LAB_JUDGE_MODEL", GEMINI_MODEL)


# ── Document extraction (host-side, mirrors verifier) ─────────────────


def _read_doc(path: Path) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix == ".docx":
            r = subprocess.run(
                ["pandoc", str(path), "-t", "markdown", "--wrap=none",
                 "--track-changes=accept"],
                capture_output=True, text=True, timeout=60,
            )
            return r.stdout if r.returncode == 0 else f"(pandoc failed: {r.stderr})"
        if suffix == ".xlsx":
            import pandas as pd
            sheets = pd.read_excel(path, sheet_name=None)
            return "\n".join(
                f"=== Sheet: {n} ===\n{df.to_string(index=False)}"
                for n, df in sheets.items()
            )
        if suffix == ".pptx":
            from markitdown import MarkItDown
            return MarkItDown().convert(str(path)).text_content
        if suffix == ".pdf":
            import pdfplumber
            parts = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    if t := page.extract_text():
                        parts.append(t)
            return "\n".join(parts)
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"(error reading {path.name}: {e})"


def load_documents(docs_dir: Path, *, max_chars: int = 200_000) -> str:
    """Render the task's documents folder as one big text block."""
    if not docs_dir.is_dir():
        return "(no documents/ dir)"
    parts = []
    for f in sorted(docs_dir.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(docs_dir)
        body = _read_doc(f)
        parts.append(f"\n\n===== {rel} =====\n{body}")
    text = "".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n(... truncated to {max_chars} chars)"
    return text


# ── One-shot agent ────────────────────────────────────────────────────

_AGENT_PROMPT = """\
You are completing a legal work assignment.  The source documents are
attached after the instructions.  Produce a complete deliverable that
satisfies the instructions.  Reply with **only the deliverable text**,
formatted as Markdown.  Do not wrap it in code fences.  Do not include
any commentary, headers about your process, or "Here is the deliverable"
preamble.  This text will be saved verbatim and graded by a rubric.

If the instructions ask for multiple deliverables, separate each one with
a line containing exactly:

    ===== DELIVERABLE: <filename> =====

(matching the deliverable filename declared in the instructions).

## Instructions

{instructions}

## Source Documents

{documents}
"""


def _gemini_client():
    from google import genai
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def run_one_shot_agent(client, instructions: str, documents: str) -> str:
    prompt = _AGENT_PROMPT.format(instructions=instructions, documents=documents)
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={"temperature": 0.0},
    )
    return resp.text or ""


def split_deliverables(text: str, declared: list[str]) -> dict[str, str]:
    """Split a one-shot reply into the declared deliverable files.

    Looks for ``===== DELIVERABLE: <name> =====`` markers; falls back to
    the whole text under the first declared filename when the model
    didn't comply with the marker convention.
    """
    if not declared:
        return {"response.md": text.strip() + "\n"}

    if len(declared) == 1 or "===== DELIVERABLE:" not in text:
        return {declared[0]: text.strip() + "\n"}

    out: dict[str, str] = {}
    current_name = declared[0]
    current_buf: list[str] = []
    for line in text.splitlines():
        line_strip = line.strip()
        if line_strip.startswith("===== DELIVERABLE:") and line_strip.endswith("====="):
            if current_buf:
                out[current_name] = "\n".join(current_buf).strip() + "\n"
                current_buf = []
            name = line_strip.removeprefix("===== DELIVERABLE:").removesuffix("=====").strip()
            current_name = name or current_name
        else:
            current_buf.append(line)
    if current_buf:
        out[current_name] = "\n".join(current_buf).strip() + "\n"
    # Ensure every declared deliverable has *something* (empty if missing)
    for d in declared:
        out.setdefault(d, "")
    return out


def materialise_outputs(out_dir: Path, parts: dict[str, str]) -> None:
    """Save each deliverable as both its declared name and as .md fallback."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, body in parts.items():
        target = out_dir / name
        suffix = target.suffix.lower()
        # Always keep a markdown copy for binary deliverables — both
        # judges fall back to fuzzy-matching by extension/keywords.
        md_path = target.with_suffix(".md")
        md_path.write_text(body)
        if suffix in (".docx",):
            try:
                subprocess.run(
                    ["pandoc", str(md_path), "-o", str(target)],
                    check=True, capture_output=True, timeout=60,
                )
            except Exception as e:
                LOG.warning("pandoc failed for %s: %s — keeping .md only", name, e)
        elif suffix in (".xlsx", ".pptx", ".pdf"):
            # Don't try to fake binary formats; the judge falls back to
            # fuzzy-match on the .md sibling.
            pass
        else:
            target.write_text(body)


# ── Judge (single function used by both arms) ─────────────────────────

_JUDGE_SYSTEM = (
    "You are an evaluator grading legal-work agent output against a "
    "single pass/fail rubric criterion. Reply strictly with a JSON "
    'object: {"verdict": "pass"|"fail", "reasoning": "..."}.'
)

_JUDGE_USER = """\
TASK: {task_desc}

CRITERION TITLE: {criterion_title}

PASS/FAIL CRITERIA:
{match_criteria}

AGENT OUTPUT:
{agent_output}

Decide pass or fail for this single criterion only. JSON only.
"""


def gemini_judge(client, task_desc: str, criterion: dict, agent_output: str) -> dict:
    prompt = _JUDGE_USER.format(
        task_desc=task_desc,
        criterion_title=criterion["title"],
        match_criteria=criterion["match_criteria"],
        agent_output=agent_output[:200_000],
    )
    try:
        resp = client.models.generate_content(
            model=GEMINI_JUDGE,
            contents=prompt,
            config={
                "temperature": 0.0,
                "system_instruction": _JUDGE_SYSTEM,
                "response_mime_type": "application/json",
            },
        )
        text = (resp.text or "").strip()
        data = json.loads(text)
        verdict = str(data.get("verdict", "fail")).lower()
        if verdict not in ("pass", "fail"):
            verdict = "fail"
        return {"verdict": verdict, "reasoning": data.get("reasoning", "")}
    except Exception as e:
        return {"verdict": "fail", "reasoning": f"judge error: {e}"}


# ── Per-arm scoring ───────────────────────────────────────────────────


def collect_agent_output_text(out_dir: Path, declared: list[str]) -> dict[str, str]:
    """Read each declared deliverable as text, falling back to the .md sibling."""
    rendered: dict[str, str] = {}
    for name in declared:
        p = out_dir / name
        md = p.with_suffix(".md")
        if p.exists() and p.stat().st_size > 0 and p.suffix.lower() not in (".docx", ".xlsx", ".pptx", ".pdf"):
            rendered[name] = p.read_text()
        elif md.exists():
            rendered[name] = md.read_text()
        elif p.exists():
            rendered[name] = _read_doc(p)
        else:
            rendered[name] = ""
    return rendered


def score_lab_native(client, task_cfg: dict, output_dir: Path,
                     parallel: int = 8) -> dict:
    """LAB-native scoring path.

    Mirrors LAB's ``evaluation/scoring.py`` semantics: per-criterion
    pass/fail, all-pass for reward = 1.0.  Same judge model as the
    BenchFlow side (controlled by ``LAB_JUDGE_MODEL``) so the only
    variable across arms is the framework wiring.
    """
    from concurrent.futures import ThreadPoolExecutor

    criteria = task_cfg["criteria"]
    declared = sorted({d for c in criteria for d in c.get("deliverables", [])})
    rendered = collect_agent_output_text(output_dir, declared)
    full_output = "\n\n".join(f"## {n}\n{t}" for n, t in rendered.items())
    title = task_cfg.get("title", "")

    def _score(c: dict) -> dict:
        if cd := c.get("deliverables"):
            agent_text = "\n\n".join(
                f"## Deliverable: {n}\n{rendered.get(n, '')}" for n in cd
            )
        else:
            agent_text = full_output
        verdict = gemini_judge(client, title, c, agent_text)
        return {
            "id": c["id"],
            "title": c["title"],
            "verdict": verdict["verdict"],
            "reasoning": verdict["reasoning"],
        }

    with ThreadPoolExecutor(max_workers=max(parallel, 1)) as pool:
        results = list(pool.map(_score, criteria))

    n = len(results)
    n_pass = sum(1 for r in results if r["verdict"] == "pass")
    return {
        "n_criteria": n,
        "n_passed": n_pass,
        "all_pass": n > 0 and n_pass == n,
        "reward": 1.0 if n > 0 and n_pass == n else 0.0,
        "criteria": results,
    }


def score_benchflow_translated(translated_task_dir: Path, output_dir: Path,
                               judge_model: str) -> dict:
    """BenchFlow scoring path: invokes the verifier exactly as the runtime would."""
    report = output_dir.parent / "bench_report.json"
    reward = output_dir.parent / "bench_reward.txt"
    cmd = [
        sys.executable,
        str(translated_task_dir / "tests" / "rubric_judge.py"),
        "--output-dir", str(output_dir),
        "--criteria", str(translated_task_dir / "tests" / "criteria.json"),
        "--task-desc-file", str(translated_task_dir / "tests" / "task_desc.txt"),
        "--report", str(report),
        "--reward", str(reward),
        "--judge-model", judge_model,
    ]
    env = os.environ.copy()
    subprocess.run(cmd, check=True, env=env)
    data = json.loads(report.read_text())
    data["reward"] = float(reward.read_text().strip())
    return data


# ── Per-task orchestration ────────────────────────────────────────────


@dataclass
class TaskResult:
    task_id: str            # sanitised
    relative_id: str
    lab_score: float = math.nan
    bench_score: float = math.nan
    lab_passed: int = 0
    bench_passed: int = 0
    n_criteria: int = 0
    agreement: bool = False  # per-criterion verdicts identical
    error: str | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class RunResult:
    run_index: int
    started_at: float
    tasks: list[TaskResult] = field(default_factory=list)

    def lab_scores(self) -> list[float]:
        return [t.lab_score for t in self.tasks if not math.isnan(t.lab_score)]

    def bench_scores(self) -> list[float]:
        return [t.bench_score for t in self.tasks if not math.isnan(t.bench_score)]


def run_one_task(client, lab_root: Path, translated_root: Path, relative_id: str,
                 run_dir: Path) -> TaskResult:
    """Execute the one-shot agent + both scoring arms for a single task."""
    parts = relative_id.split("/")
    sanitised = sanitize_task_id(parts)
    lab_task_dir = lab_root / "tasks" / Path(*parts)
    cfg = json.loads((lab_task_dir / "task.json").read_text())
    instructions = cfg.get("instructions", "")
    if not instructions:
        ip = lab_task_dir / "instructions.md"
        instructions = ip.read_text() if ip.exists() else ""

    declared = sorted({d for c in cfg.get("criteria", []) for d in c.get("deliverables", [])})
    if not declared:
        declared = list(cfg.get("deliverables", {}).keys())

    LOG.info("[%s] reading documents", relative_id)
    documents = load_documents(lab_task_dir / "documents")

    LOG.info("[%s] generating one-shot agent output", relative_id)
    try:
        text = run_one_shot_agent(client, instructions, documents)
    except Exception as e:
        return TaskResult(task_id=sanitised, relative_id=relative_id, error=f"agent: {e}")

    parts_text = split_deliverables(text, declared)

    out_dir = run_dir / sanitised / "agent_output"
    materialise_outputs(out_dir, parts_text)

    LOG.info("[%s] scoring (LAB-native arm)", relative_id)
    try:
        lab_scores = score_lab_native(client, cfg, out_dir)
    except Exception as e:
        return TaskResult(task_id=sanitised, relative_id=relative_id,
                          error=f"lab-score: {e}")
    (run_dir / sanitised / "lab_scores.json").write_text(
        json.dumps(lab_scores, indent=2)
    )

    LOG.info("[%s] scoring (BenchFlow arm)", relative_id)
    bench_task_dir = translated_root / sanitised
    try:
        bench_scores = score_benchflow_translated(bench_task_dir, out_dir, GEMINI_JUDGE)
    except Exception as e:
        return TaskResult(task_id=sanitised, relative_id=relative_id,
                          error=f"bench-score: {e}")
    (run_dir / sanitised / "bench_scores.json").write_text(
        json.dumps(bench_scores, indent=2)
    )

    # Per-criterion agreement
    lab_by_id = {c["id"]: c["verdict"] for c in lab_scores["criteria"]}
    bench_by_id = {c["id"]: c["verdict"] for c in bench_scores["criteria"]}
    agreement = lab_by_id == bench_by_id

    return TaskResult(
        task_id=sanitised,
        relative_id=relative_id,
        lab_score=lab_scores["reward"],
        bench_score=bench_scores["reward"],
        lab_passed=lab_scores["n_passed"],
        bench_passed=bench_scores["n_passed"],
        n_criteria=lab_scores["n_criteria"],
        agreement=agreement,
    )


# ── Aggregation ───────────────────────────────────────────────────────


def mean_sem(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in xs) / (n * (n - 1))
    return m, math.sqrt(var)


def summarise(runs: list[RunResult]) -> dict:
    """Aggregate mean ± sample SEM across runs (harbor parity reporting)."""
    by_task: dict[str, list[tuple[float, float]]] = {}
    for r in runs:
        for t in r.tasks:
            by_task.setdefault(t.relative_id, []).append((t.lab_score, t.bench_score))

    # Per-run dataset-level scores
    per_run_lab = [sum(r.lab_scores()) / max(len(r.lab_scores()), 1) for r in runs]
    per_run_bench = [sum(r.bench_scores()) / max(len(r.bench_scores()), 1) for r in runs]

    lab_mean, lab_sem = mean_sem(per_run_lab)
    bench_mean, bench_sem = mean_sem(per_run_bench)

    overlap = (
        max(per_run_lab) >= min(per_run_bench) if per_run_lab and per_run_bench else False
    ) and (
        max(per_run_bench) >= min(per_run_lab) if per_run_lab and per_run_bench else False
    )

    return {
        "n_runs": len(runs),
        "n_tasks": len(by_task),
        "per_run_lab": per_run_lab,
        "per_run_bench": per_run_bench,
        "lab_mean_pm_sem": f"{lab_mean:.3f} ± {lab_sem:.3f}",
        "bench_mean_pm_sem": f"{bench_mean:.3f} ± {bench_sem:.3f}",
        "ranges_overlap": overlap,
        "per_task": {
            rid: {
                "lab_runs": [s[0] for s in scores],
                "bench_runs": [s[1] for s in scores],
            }
            for rid, scores in by_task.items()
        },
    }


# ── CLI ───────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lab-dir", default=str(LAB_DEFAULT_REPO))
    ap.add_argument("--translated-dir", default="/tmp/lab-tasks")
    ap.add_argument("--task-list", default=str(ADAPTER_DIR / "scripts" / "parity_subset.txt"))
    ap.add_argument("--results-dir", default="parity-results")
    ap.add_argument("--runs", type=int, default=1,
                    help="Number of independent runs per side (harbor recipe: 3)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not os.environ.get("GEMINI_API_KEY"):
        print("error: set GEMINI_API_KEY", file=sys.stderr)
        return 2

    lab_root = Path(args.lab_dir).resolve()
    translated_root = Path(args.translated_dir).resolve()
    results_root = Path(args.results_dir).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    # Materialise translated tasks
    translated_root.mkdir(parents=True, exist_ok=True)
    rids: list[str] = [
        line.strip()
        for line in Path(args.task_list).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if args.limit:
        rids = rids[: args.limit]

    LOG.info("Translating %d task(s) to %s", len(rids), translated_root)
    tasks = discover_tasks(lab_root)
    by_rid = {t.relative_id: t for t in tasks}
    for rid in rids:
        if rid not in by_rid:
            raise SystemExit(f"task not in LAB: {rid}")
        write_task(by_rid[rid], translated_root, force=True)

    client = _gemini_client()

    runs: list[RunResult] = []
    for run_i in range(1, args.runs + 1):
        run_dir = results_root / f"run-{run_i:02d}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        run = RunResult(run_index=run_i, started_at=time.time())
        for rid in rids:
            LOG.info("=== run %d / task %s ===", run_i, rid)
            res = run_one_task(client, lab_root, translated_root, rid, run_dir)
            run.tasks.append(res)
            (run_dir / "tasks.jsonl").open("a").write(
                json.dumps(res.to_dict()) + "\n"
            )
        runs.append(run)

    summary = summarise(runs)
    summary_path = results_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
