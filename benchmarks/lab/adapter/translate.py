"""Translate a Harvey-LAB task into a BenchFlow task directory.

Source (LAB native):
    tasks/<area>/<slug>[/<scenario>]/
        task.json            -- title / instructions / criteria / deliverables
        documents/           -- read-only source materials (.docx, .pdf, .xlsx, .pptx)

Target (BenchFlow):
    <out>/<sanitized-task-id>/
        task.toml
        instruction.md
        environment/
            Dockerfile
            documents/       -- copied from source
        tests/
            test.sh          -- runs rubric_judge.py and writes /logs/verifier/reward.txt
            rubric_judge.py  -- LLM judge against task.json criteria
            criteria.json    -- detached copy of just the rubric (kept out of /app)
        solution/
            solve.sh         -- empty stub (LAB tasks have no oracle solutions)

The translation is intentionally faithful: the agent sees the same
instructions and the same documents that LAB shows it, and the verifier
applies the same all-pass rubric semantics that LAB's `evaluation/scoring.py`
applies (every criterion must `pass` for the task to score 1.0).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

# ── Task ID sanitisation ──────────────────────────────────────────────


def sanitize_task_id(parts: list[str]) -> str:
    """Lower-case, hyphen-joined, alnum-or-hyphen identifier.

    ``["corporate-ma", "review-data-room", "scenario-01"]`` →
    ``"corporate-ma__review-data-room__scenario-01"``.

    The double-underscore separator preserves the LAB practice-area /
    slug / scenario hierarchy while keeping the result a single
    filesystem-safe directory name.
    """
    cleaned = []
    for p in parts:
        s = p.lower().strip().replace(" ", "-").replace("/", "-")
        s = "".join(c for c in s if c.isalnum() or c in "-_")
        if s:
            cleaned.append(s)
    if not cleaned:
        raise ValueError(f"Empty task id from parts: {parts!r}")
    return "__".join(cleaned)


# ── LAB task discovery ───────────────────────────────────────────────


@dataclass(frozen=True)
class LabTask:
    """A discovered LAB task on disk."""

    task_id: str           # sanitised, BenchFlow-side identifier
    lab_path: Path         # source directory under tasks/
    relative_id: str       # original LAB id, e.g. "corporate-ma/review-foo"
    config: dict


def discover_tasks(lab_root: Path) -> list[LabTask]:
    """Find every ``task.json`` under ``<lab_root>/tasks/``."""
    tasks_dir = lab_root / "tasks"
    if not tasks_dir.is_dir():
        raise FileNotFoundError(f"LAB tasks dir not found: {tasks_dir}")

    found: list[LabTask] = []
    for cfg in sorted(tasks_dir.rglob("task.json")):
        rel = cfg.parent.relative_to(tasks_dir)
        parts = list(rel.parts)
        config = json.loads(cfg.read_text())
        found.append(
            LabTask(
                task_id=sanitize_task_id(parts),
                lab_path=cfg.parent,
                relative_id="/".join(parts),
                config=config,
            )
        )
    return found


# ── Instruction.md and task.toml ─────────────────────────────────────


_AGENT_PREAMBLE = """\
You are an AI agent executing a legal work task.

## Workspace layout

You are running inside a sandbox. Your working directory is `/app/`:

- `/app/documents/` — source documents (read-only). Includes binary files
  (.docx, .xlsx, .pptx, .pdf) and plain-text files. Use `pandoc`, `python -m
  pdfplumber`, `python -m markitdown`, or `python -c "import pandas; ..."`
  to extract content.
- `/app/` — write deliverables here as ordinary files.

## Producing deliverables

- Plain markdown / .txt: write the file directly (`cat > /app/foo.md`).
- `.docx`: use `pandoc input.md -o /app/foo.docx`.
- `.xlsx`: use `python -c "import pandas as pd; ...; df.to_excel('/app/foo.xlsx')"`.
- `.pptx`: use `python -c "from pptx import Presentation; ..."`.

When you finish, stop responding — do not write a summary or wait for
confirmation.
"""


def _instruction_for(task: LabTask) -> str:
    """Build the agent prompt: preamble + LAB instructions."""
    cfg = task.config
    title = cfg.get("title", task.relative_id)
    body = cfg.get("instructions") or ""
    if not body:
        # LAB allows external instructions.md
        ext = task.lab_path / "instructions.md"
        if ext.exists():
            body = ext.read_text(encoding="utf-8")
    return f"{_AGENT_PREAMBLE}\n## Task: {title}\n\n{body.strip()}\n"


def _task_toml(task: LabTask) -> str:
    """Render task.toml. LAB tasks are free-form documents, so we keep
    timeouts generous and leave the network on (the verifier needs it
    to call the Gemini judge)."""
    cfg = task.config
    title = cfg.get("title", task.relative_id).replace('"', "'")
    tags = cfg.get("tags") or []
    tags_toml = ", ".join(f'"{t}"' for t in tags)
    work_type = cfg.get("work_type", "analyze")
    return f"""version = "1.0"

[metadata]
author_name = "harveyai (LAB) — translated by benchflow lab adapter"
title = "{title}"
category = "legal"
work_type = "{work_type}"
tags = [{tags_toml}]
source_id = "{task.relative_id}"

[agent]
timeout_sec = 1800

[verifier]
timeout_sec = 600

[environment]
cpus = 2
memory_mb = 4096
storage_mb = 10240
allow_internet = true
"""


# ── Dockerfile ────────────────────────────────────────────────────────

_DOCKERFILE = """\
# LAB task environment.
#
# The image ships the file-format tools that the agent uses to read the
# source documents (pandoc, pdfplumber, pandas+openpyxl, markitdown,
# python-pptx) and the genai SDK that the verifier uses to call the
# Gemini judge.

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \\
        pandoc \\
        curl \\
        ca-certificates \\
        git \\
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \\
        google-genai==2.0.1 \\
        pdfplumber==0.11.4 \\
        pandas==2.2.3 \\
        openpyxl==3.1.5 \\
        python-pptx==1.0.2 \\
        markitdown==0.0.1a3

WORKDIR /app

# Source documents are read-only, mounted under /app/documents.
COPY documents /app/documents
RUN find /app/documents -type f -exec chmod a-w {} +

# An empty marker so the agent's `ls` shows the layout immediately.
RUN touch /app/.workspace
"""


# ── Verifier (rubric judge) ──────────────────────────────────────────

_TEST_SH = """\
#!/bin/bash
# LAB rubric verifier.
# Runs the LLM judge over each criterion in /tests/criteria.json against
# the agent's output in /app/, then writes the all-pass float reward to
# /logs/verifier/reward.txt.

set -uo pipefail

mkdir -p /logs/verifier

python3 /tests/rubric_judge.py \\
    --output-dir /app \\
    --criteria   /tests/criteria.json \\
    --task-desc-file /tests/task_desc.txt \\
    --report     /logs/verifier/criteria.json \\
    --reward     /logs/verifier/reward.txt
"""


# rubric_judge.py is a self-contained script — it has to run inside the
# verifier container without depending on the rest of the adapter.
_RUBRIC_JUDGE = '''\
"""LAB rubric judge. Scores each criterion pass/fail with Gemini.

The all-pass rule (every criterion must pass for the task to score 1.0)
mirrors LAB's own ``evaluation/scoring.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


# ── File extraction ──────────────────────────────────────────────────

def _read_file_as_text(path: Path) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix == ".docx":
            r = subprocess.run(
                ["pandoc", str(path), "-t", "markdown", "--wrap=none",
                 "--track-changes=accept"],
                capture_output=True, text=True, timeout=60,
            )
            if r.returncode != 0:
                return f"(pandoc failed: {r.stderr})"
            return r.stdout
        if suffix == ".xlsx":
            import pandas as pd
            sheets = pd.read_excel(path, sheet_name=None)
            return "\\n".join(
                f"=== Sheet: {name} ===\\n{df.to_string(index=False)}"
                for name, df in sheets.items()
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
            return "\\n".join(parts)
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001 — judge should never crash
        return f"(error reading {path.name}: {e})"


# ── Deliverable matching ─────────────────────────────────────────────

_SKIP_DIRS = {"node_modules", ".npm", "__pycache__", ".git", "venv", ".venv"}
_SKIP_EXTS = {".lock", ".map"}
_SKIP_FILES = {"package-lock.json", ".workspace"}


def _list_outputs(output_dir: Path) -> list[Path]:
    out = []
    if not output_dir.exists():
        return out
    for f in sorted(output_dir.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(output_dir)
        if any(p in _SKIP_DIRS for p in rel.parts):
            continue
        if rel.parts and rel.parts[0] == "documents":  # source docs
            continue
        if f.suffix in _SKIP_EXTS or f.name in _SKIP_FILES:
            continue
        out.append(f)
    return out


def _fuzzy_match(expected: str, candidates: list[Path]) -> Path | None:
    expected_words = set(
        Path(expected).stem.lower().replace("-", " ").replace("_", " ").split()
    )
    best, best_score = None, 0
    for c in candidates:
        cand_words = set(
            c.stem.lower().replace("-", " ").replace("_", " ").split()
        )
        score = len(expected_words & cand_words)
        if score > best_score:
            best_score, best = score, c
    return best if best_score > 0 else None


def _resolve_deliverables(criteria: list[dict], output_dir: Path) -> dict:
    """Map each criterion deliverable name → actual Path (or None).

    Resolution order, preserving LAB's ``_match_deliverables`` semantics:

      1. Exact filename match.
      2. Same-extension fuzzy match (sole candidate, then keyword overlap).
      3. ``<stem>.md`` sibling — agents that produced markdown instead of a
         binary deliverable should still be gradeable. LAB's text-mode
         readers tolerate this; we mirror it here.
    """
    actual = _list_outputs(output_dir)
    by_name = {p.name: p for p in actual}
    resolved: dict[str, Path | None] = {}
    used: set[Path] = set()

    wanted = sorted({d for c in criteria for d in c.get("deliverables", [])})
    for name in wanted:
        if name in by_name and by_name[name] not in used:
            resolved[name] = by_name[name]
            used.add(by_name[name])
            continue
        ext = Path(name).suffix.lower()
        candidates = [
            p for p in actual if p not in used and p.suffix.lower() == ext
        ]
        if len(candidates) == 1:
            resolved[name] = candidates[0]
            used.add(candidates[0])
            continue
        match = _fuzzy_match(name, candidates)
        if match is not None:
            resolved[name] = match
            used.add(match)
            continue
        # Markdown sibling fallback: <stem>.md in the same dir.
        md_candidate = output_dir / (Path(name).stem + ".md")
        if md_candidate.exists() and md_candidate not in used:
            resolved[name] = md_candidate
            used.add(md_candidate)
            continue
        resolved[name] = None
    return resolved


# ── Judge ────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """\\
You are an evaluator grading legal-work agent output against a single \\
pass/fail rubric criterion. Read the criterion carefully and answer \\
strictly with a JSON object: {"verdict": "pass"|"fail", "reasoning": "..."}.\\
"""

_JUDGE_USER = """\\
TASK: {task_desc}

CRITERION TITLE: {criterion_title}

PASS/FAIL CRITERIA:
{match_criteria}

AGENT OUTPUT:
{agent_output}

Decide pass or fail for this single criterion only. Respond with JSON only.\\
"""


def _judge(client, model: str, task_desc: str, criterion: dict, output_text: str) -> dict:
    prompt = _JUDGE_USER.format(
        task_desc=task_desc,
        criterion_title=criterion["title"],
        match_criteria=criterion["match_criteria"],
        agent_output=output_text[:200_000],  # cap context
    )
    try:
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "temperature": 0.0,
                "system_instruction": _JUDGE_SYSTEM,
                "response_mime_type": "application/json",
            },
        )
        text = (resp.text or "").strip()
        # Strip markdown fences if any
        if text.startswith("```"):
            text = text.strip("`")
            text = text.split("\\n", 1)[1] if "\\n" in text else text
            if text.endswith("```"):
                text = text[: -3]
        data = json.loads(text)
        verdict = str(data.get("verdict", "fail")).lower()
        if verdict not in ("pass", "fail"):
            verdict = "fail"
        return {"verdict": verdict, "reasoning": data.get("reasoning", "")}
    except Exception as e:  # noqa: BLE001
        return {"verdict": "fail", "reasoning": f"judge error: {e}"}


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--criteria", type=Path, required=True)
    ap.add_argument("--task-desc-file", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--reward", type=Path, required=True)
    ap.add_argument("--judge-model",
                    default=os.environ.get("LAB_JUDGE_MODEL",
                                           "gemini-3.1-flash-lite-preview"))
    args = ap.parse_args()

    criteria = json.loads(args.criteria.read_text())
    task_desc = args.task_desc_file.read_text().strip()

    # Resolve which output files map to which deliverable names
    resolved = _resolve_deliverables(criteria, args.output_dir)

    # Lazy import — keeps the script importable for unit testing without
    # the genai SDK installed on the host.
    from google import genai
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    from concurrent.futures import ThreadPoolExecutor

    fallback_sections = None  # cache full-output once

    def _grade(c: dict) -> dict:
        nonlocal fallback_sections
        sections = []
        for name in c.get("deliverables", []):
            path = resolved.get(name)
            if path is None or not path.exists():
                sections.append(f"## Deliverable: {name}\\n(File not found)")
                continue
            sections.append(
                f"## Deliverable: {name}\\n{_read_file_as_text(path)}"
            )
        if not sections:
            if fallback_sections is None:
                fallback_sections = []
                for p in _list_outputs(args.output_dir):
                    fallback_sections.append(
                        f"## File: {p.name}\\n{_read_file_as_text(p)}"
                    )
            sections = list(fallback_sections)
        agent_output = "\\n\\n".join(sections) if sections else "(no output)"
        verdict = _judge(client, args.judge_model, task_desc, c, agent_output)
        return {
            "id": c["id"],
            "title": c["title"],
            "verdict": verdict["verdict"],
            "reasoning": verdict["reasoning"],
        }

    parallel = int(os.environ.get("LAB_JUDGE_PARALLEL", "8"))
    with ThreadPoolExecutor(max_workers=max(parallel, 1)) as pool:
        results = list(pool.map(_grade, criteria))

    n = len(results)
    n_pass = sum(1 for r in results if r["verdict"] == "pass")
    all_pass = n > 0 and n_pass == n
    reward = 1.0 if all_pass else 0.0

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "n_criteria": n,
        "n_passed": n_pass,
        "all_pass": all_pass,
        "criteria": results,
    }, indent=2))
    args.reward.parent.mkdir(parents=True, exist_ok=True)
    args.reward.write_text(f"{reward}\\n")

    print(f"LAB rubric: {n_pass}/{n} passed (reward={reward})")
    sys.exit(0)


if __name__ == "__main__":
    main()
'''


# ── Public API ────────────────────────────────────────────────────────


def write_task(task: LabTask, out_dir: Path, *, force: bool = False) -> Path:
    """Materialise one LAB task as a BenchFlow task directory.

    Returns the path of the generated directory.
    """
    target = out_dir / task.task_id
    if target.exists():
        if not force:
            return target
        shutil.rmtree(target)
    target.mkdir(parents=True)

    # Top-level metadata
    (target / "task.toml").write_text(_task_toml(task))
    (target / "instruction.md").write_text(_instruction_for(task))

    # Environment + documents
    env = target / "environment"
    env.mkdir()
    (env / "Dockerfile").write_text(_DOCKERFILE)

    docs_src = task.lab_path / "documents"
    docs_dst = env / "documents"
    docs_dst.mkdir()
    if docs_src.is_dir():
        for entry in docs_src.iterdir():
            if entry.is_file():
                shutil.copy2(entry, docs_dst / entry.name)
            elif entry.is_dir():
                shutil.copytree(entry, docs_dst / entry.name)
    else:
        # Empty marker so COPY documents doesn't fail in Docker
        (docs_dst / ".empty").write_text("")

    # Verifier
    tests = target / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text(_TEST_SH)
    (tests / "test.sh").chmod(0o755)
    (tests / "rubric_judge.py").write_text(_RUBRIC_JUDGE)
    (tests / "criteria.json").write_text(
        json.dumps(task.config["criteria"], indent=2)
    )
    (tests / "task_desc.txt").write_text(task.config.get("title", task.relative_id))

    # Solution stub (no oracle for free-form drafting tasks)
    sol = target / "solution"
    sol.mkdir()
    (sol / "solve.sh").write_text(
        "#!/bin/bash\n# LAB tasks have no canonical oracle; left intentionally empty.\n"
        "exit 0\n"
    )
    (sol / "solve.sh").chmod(0o755)

    return target
