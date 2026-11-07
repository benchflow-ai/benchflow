#!/usr/bin/env python3
"""A model judge for the packet judge.html shows a person: the brief, the reason keys, and pictures of
both outputs of each pair as A and B. Its answers are written as votes `bench preferences import` reads.

The pictures come from headless Chromium (Playwright): the first screen in a 1440 x 900 window, the whole
desktop page (cut at 2,700 px), and the whole page at phone width, 390 x 844 (cut at 2,532 px). Each pair
gets a folder in --shots with A-*.png, B-*.png and prompt.md; pictures already there are kept, so the
store-outputs folder is read only to make missing ones. Pairs already in --votes are skipped.

usage: python3 judge_model.py <store-outputs dir> <assignment.json> --brief task.md --reasons reasons.json \
         --shots shots/ --votes votes.jsonl --model NAME (--base-url-env VAR --key-env VAR | --command 'CLI')

--command runs CLI in the pair's folder with the prompt on stdin (it is also in prompt.md) and reads the
answer from its stdout, so an agent such as Claude Code (`claude -p`) can judge too.
"""

import argparse
import base64
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from serve import brief, entry, read_votes, write_votes

# name, window width, window height, where the picture is cut (the window height: the first screen only)
SHOTS = (("desktop", 1440, 900, 900), ("desktop-full", 1440, 900, 2700), ("phone", 390, 844, 2532))
SHOWN = ("the first screen in a 1440 x 900 window", "the whole page at 1440 wide, cut at 2,700 px",
         "the whole page on a 390 x 844 phone, cut at 2,532 px")
CHOICES = {"a": "left", "b": "right", "tie": "tie"}


def render(outputs: Path, items: set, folder: Path) -> None:
    from playwright.sync_api import sync_playwright

    folder.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chromium")
        for item in sorted(items):
            url = (outputs / item / entry(outputs / item)).resolve().as_uri()
            for name, width, height, cut in SHOTS:
                page = browser.new_page(viewport={"width": width, "height": height}, offline=True)
                page.goto(url, wait_until="load")
                page.wait_for_timeout(500)
                tall = min(cut, page.evaluate("document.documentElement.scrollHeight"))
                clip = {"x": 0, "y": 0, "width": width, "height": max(tall, height)}
                page.screenshot(path=folder / f"{item}-{name}.png", full_page=cut > height, clip=clip)
                page.close()
        browser.close()


def prompt(brief_text: str, reasons: dict, files: list) -> str:
    keys = "\n".join(f"{r['key']}. {r['text']}" for r in reasons["reasons"])
    shown = "; ".join(f"{name}: {how}" for (name, *_), how in zip(SHOTS, SHOWN))
    look = ("The pictures are these PNG files in this folder; open every one before you answer:\n"
            + "\n".join(f"- {f}" for f in files) + "\n\n") if files else ""
    return f"""You are judging two web pages, A and B, made for the same brief. You are not told who made them.

The brief:
{brief_text}

{look}Each page is shown three ways ({shown}).

Which page answers the brief better for the people it is for? Look at all the pictures of both pages before you
decide; say tie only if neither is better. Then mark the reason keys that decided it ({reasons.get('hint', '')}):
{keys}

Answer with one JSON object and nothing else:
{{"choice": "A" or "B" or "tie", "reasons": ["<key>", ...], "note": "<one sentence>"}}
"""


def ask_endpoint(a, text: str, pictures: list) -> str:
    content = [{"type": "text", "text": text}]
    for caption, path in pictures:
        content.append({"type": "text", "text": caption})
        data = base64.b64encode(path.read_bytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}})
    body = {"model": a.model, "messages": [{"role": "user", "content": content}], "max_tokens": a.max_tokens}
    request = urllib.request.Request(
        os.environ[a.base_url_env].rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {os.environ[a.key_env]}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            answer = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{exc}: {exc.read().decode(errors='replace')[:300]}") from None
    usage = answer.get("usage") or {}
    print(f"  tokens: {usage.get('prompt_tokens')} in, {usage.get('completion_tokens')} out", flush=True)
    return answer["choices"][0]["message"].get("content") or ""


def ask_command(a, text: str, folder: Path) -> str:
    run = subprocess.run(shlex.split(a.command), cwd=folder, input=text, capture_output=True, text=True,
                         timeout=1200)
    if run.returncode:
        raise RuntimeError(f"{a.command} exited {run.returncode}: {run.stderr.strip()[-300:]}")
    return run.stdout


def parse(text: str, keys: set) -> dict:
    """The last JSON object in the text that has a choice."""
    decoder = json.JSONDecoder()
    for start in reversed([m.start() for m in re.finditer(r"\{", text)]):
        try:
            found, _ = decoder.raw_decode(text, start)
        except ValueError:
            continue
        if isinstance(found, dict) and str(found.get("choice", "")).strip().lower() in CHOICES:
            reasons = [str(k) for k in found.get("reasons") or [] if str(k) in keys]
            return {"choice": str(found["choice"]).strip().lower(), "reasons": sorted(set(reasons)),
                    "note": " ".join(str(found.get("note") or "").split())}
    raise ValueError(f"no answer with a choice in: {text.strip()[-300:]!r}")


def main():
    ap = argparse.ArgumentParser(description="Let a model judge the blind pairs, from pictures of each output.")
    ap.add_argument("outputs", type=Path, help="the store-outputs folder (read only to make missing pictures)")
    ap.add_argument("assignment", type=Path)
    ap.add_argument("--brief", type=Path, required=True)
    ap.add_argument("--reasons", type=Path, required=True)
    ap.add_argument("--shots", type=Path, required=True, help="a folder for the pictures, one folder per pair")
    ap.add_argument("--votes", type=Path, required=True, help="this judge's votes file (JSON lines)")
    ap.add_argument("--model", required=True, help="the model's name; the rater is model:NAME")
    ap.add_argument("--base-url-env", help="the variable holding an OpenAI-compatible base URL (ending in /v1)")
    ap.add_argument("--key-env", help="the variable holding its API key")
    ap.add_argument("--max-tokens", type=int, default=8000, help="room to reason before answering")
    ap.add_argument("--command", help="an agent CLI to run in each pair's folder instead of the endpoint")
    a = ap.parse_args()
    if not a.command and not (a.base_url_env and a.key_env and os.environ.get(a.base_url_env)
                              and os.environ.get(a.key_env)):
        ap.error("give --command, or --base-url-env and --key-env naming variables that are set")
    rater = f"model:{a.model}"
    pairs = json.loads(a.assignment.read_text())["pairs"]
    reasons = json.loads(a.reasons.read_text())
    keys = {r["key"] for r in reasons["reasons"]}
    votes = read_votes(a.votes)
    if {v.get("rater") for v in votes} - {rater}:
        raise SystemExit(f"{a.votes} holds votes by other raters: give each judge a file")
    names = [f"{side}-{name}.png" for side in "AB" for name, *_ in SHOTS]
    missing = [p for p in pairs if not all((a.shots / p["pair_id"] / n).is_file() for n in names)]
    if missing:
        render(a.outputs, {p[k] for p in missing for k in ("left", "right")}, a.shots / "items")
    text_brief = brief(a.brief)
    for number, pair in enumerate(pairs, 1):
        if any(v["pair_id"] == pair["pair_id"] for v in votes):
            continue
        folder = a.shots / pair["pair_id"]
        folder.mkdir(parents=True, exist_ok=True)
        pictures = []
        for side, item in (("A", pair["left"]), ("B", pair["right"])):
            for (name, *_), how in zip(SHOTS, SHOWN):
                path = folder / f"{side}-{name}.png"
                if not path.is_file():
                    shutil.copyfile(a.shots / "items" / f"{item}-{name}.png", path)
                pictures.append((f"Page {side}, {how}:", path))
        text = prompt(text_brief, reasons, names if a.command else [])
        (folder / "prompt.md").write_text(text)
        print(f"pair {number} of {len(pairs)} ({pair['pair_id']})", flush=True)
        try:
            raw = ask_command(a, text, folder) if a.command else ask_endpoint(a, text, pictures)
            # Kept outside the pair's folder, where the next judge works.
            (a.shots / "answers").mkdir(exist_ok=True)
            (a.shots / "answers" / f"{pair['pair_id']}-{re.sub(r'[^A-Za-z0-9.-]', '_', a.model)}.txt").write_text(raw)
            answer = parse(raw, keys)
        except Exception as exc:  # one failed pair keeps the others' votes; run again to retry it
            print(f"  no vote: {exc}", flush=True)
            continue
        reason = "; ".join(filter(None, [answer["reasons"] and "keys: " + ",".join(answer["reasons"]),
                                         answer["note"] and "note: " + answer["note"]]))
        vote = {"pair_id": pair["pair_id"], "choice": CHOICES[answer["choice"]], "rater": rater,
                "rated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "source": "judge_model.py"}
        if reason:
            vote["reason"] = reason
        votes.append(vote)
        write_votes(a.votes, votes)
        print(f"  {answer['choice']}: {reason}", flush=True)
    print(f"{len(votes)} of {len(pairs)} pairs have a vote from {rater} in {a.votes}")


if __name__ == "__main__":
    main()
