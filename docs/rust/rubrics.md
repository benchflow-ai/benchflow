# Rubrics judged by a model

A task.md rubric (`verifier/rubric.json`, schema rubric-1) can mix criteria the verifier's tests decide (`"judge": "test"`) with criteria a model decides (`"judge": "llm"`). `bench run` judges the model criteria on the host, after the agent's turn and after the verifier's tests, and scores the whole rubric from both.

The judge works like a verifier with an answer key. Each criterion gets one model call of its own, in a fresh context: the task's instruction, that one criterion, the task author's reference (labeled as the reference), and the agent's files the criterion names. No call sees another criterion or another verdict, so one criterion cannot sway the next.

## What a task author writes

```json
{
  "$schema": "https://task.md/schema/rubric-1.json",
  "x-judge": { "model": "vercel/deepseek/deepseek-v4-flash-0731" },
  "x-reference": "verifier/reference.md",
  "criteria": [
    { "id": "written", "gate": true, "text": "/app/summary.md exists.", "judge": "test", "check": "test_summary_exists" },
    { "id": "cause", "text": "The summary gives the root cause as the expired TLS certificate, and no other cause.",
      "judge": "llm", "points": 2, "evidence": ["/app/summary.md"],
      "reference": "verifier/answers.toml#facts.cause" },
    { "id": "blame", "text": "The summary names or blames an individual person.", "outcome": "bad",
      "judge": "llm", "points": 1, "evidence": ["/app/summary.md"] }
  ]
}
```

- `evidence`: the files the judge reads, as paths in the sandbox (relative paths start at the workdir). A folder gives its first 20 files. Every model-judged criterion needs at least one.
- `reference` (a rubric-1 field) on a criterion, and `x-reference` on the rubric, give the judge the right answer. Each is either text written inline, or a file in a private folder of the package (`verifier/`, `oracle/`, `evidence/`), which the agent never sees: `verifier/reference.md` for the whole file, `verifier/answers.toml#facts.cause` for one value of a TOML or JSON file. The rubric's reference goes to every criterion's call; a criterion's reference goes only to its own.
- `x-judge.model` names the judge as `<provider>/<model>`. `bench run --judge-model <provider>/<model>` overrides it; with neither, the judge is `vercel/deepseek/deepseek-v4-flash-0731`.
- The verdict follows the criterion's shape: pass or fail for a criterion with `points` or `gate`; a level key for one with `levels`; a number within `score.min` and `score.max` for a continuous one. For `"outcome": "bad"`, the judge says whether the bad thing happened, and the criterion passes when it did not.
- `guidance` on a criterion is shown to the judge as notes.

`bench check` checks the judge settings: a malformed `x-judge`, or a reference file or key that does not exist, is an error (P004), and the report lists how many criteria a model judges, with which model.

## What one judge call sees

The system prompt tells the model it is a verifier grading one criterion, that the reference shows what a correct answer looks like but is not the agent's work (something only the reference contains never counts for the agent), and to reply with one JSON object: `{"holds": true|false, "reason": "...", "quotes": ["..."]}` (or `"level"`, or `"value"`).

The user message has four labeled parts, in this order:

1. `## TASK INSTRUCTION`: what the agent was asked to do.
2. `## CRITERION \`<id>\``: the criterion's text, its guidance, and its levels or scale.
3. `## REFERENCE (written by the task's author; NOT the agent's work)`: each reference under `=== REFERENCE: <source> ===`.
4. `## THE AGENT'S WORK`: each evidence file under `=== AGENT'S WORK: <path> (<kind>, <bytes> bytes) ===`, or a line saying why it is not shown (missing, a link, too large, binary).

Calls run at temperature 0, at most 4 at a time.

## How the evidence gets to the judge

After the verifier's tests, BenchFlow checks each evidence path in the sandbox as root (links are not followed), downloads files up to 16 MB, and converts them to text on the host: text, markdown, JSON, CSV, YAML, code, and HTML as written; `.docx`, `.pptx`, and `.xlsx` from their XML (spreadsheets as `A1=value` cells per sheet); PDF from the text its pages draw (a scanned PDF has none). A call shows at most 60,000 characters of one file and 150,000 of all its files; anything cut says so, with how many characters were left out.

The judge runs on the host, so nothing in the sandbox can reach it, its prompt, or its key. The key comes from the host's provider settings, the same ones `-m` uses (the provider's variable, then `~/.config/benchflow/<provider>.env`); on the Vercel AI Gateway every request is pinned to allowed upstreams. The key is redacted from everything sent and saved, and never written to the trial folder.

## When the judge cannot decide

A reply that is not the verdict asked for (no JSON, a `holds` that is not true or false, a level the criterion lacks, a number out of range, an empty reason) is asked again with the reason, in the same criterion's context. After 3 calls the criterion is recorded as not judged, and the trial is unscored with R312: never a silent 0. A provider that cannot be reached is R312 too. Judge settings the task got wrong are R313.

## What is recorded

`verifier/judgments.jsonl` in the trial folder has one line per model-judged criterion: the criterion and its text, the model, the references and evidence files it saw (with sizes, and whether a file was cut), the system prompt, every attempt with the messages sent and the reply, the verdict (and level or value), the reason, each quote with whether it was found in the evidence, and tokens and cost.

`verifier/review.json` (and the `rubric` in `result.json`) gives every criterion's verdict; a model-judged one has `judge.model` and the judge's reason as its `rationale`, and `x-benchflow.judge` totals the calls, tokens, and cost.

## What a judge cannot promise

- The judge reads text the agent wrote, and that text can try to steer it ("the criterion above is met"). The prompt tells the judge that the agent's work is material to grade, not instructions, and every quote it gives is checked against the evidence (`found` in `verifier/judgments.jsonl`), but a model can still be talked into a verdict. Judge by tests whatever a test can check, and keep model-judged criteria for what only reading can decide.
- One call per criterion at temperature 0 is not a deterministic grade: the same work can get a different verdict on another run or another model. Name the judge model in the rubric so a job's verdicts come from one model, and repeat trials before reading a small difference between two agents.
- The judge sees the agent's files as text: layout, images, charts, and formatting in documents are not shown, and a PDF whose text cannot be read is noted as unreadable.

## Not yet

- Criteria judged by `agent`, `vlm`, `panel`, or `human` are still refused by name (R401).
- The `trajectory`, `screenshots`, and other runtime evidence items are shown to the judge as not available.
- BenchFlow 0.8's `verifier/judge.toml` is not read by this runner: such a task's own `test.sh` still runs its judge inside the sandbox. To judge it on the host, move its criteria into `verifier/rubric.json` as `llm` criteria (`files` become `evidence`).
