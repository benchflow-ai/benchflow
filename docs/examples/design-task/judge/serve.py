#!/usr/bin/env python3
"""Serve the blind pairs `bench preferences pairs` made to a person judging in judge.html.

The judging page and its votes are on 127.0.0.1:PORT. The outputs are served read-only on PORT+1
(side A) and PORT+2 (side B): each side is another origin, so a page an agent wrote can't read the
judging page or the other side, or cast a vote. Every vote is written to --votes at once, in the
format `bench preferences import` reads.

usage: python3 serve.py <store-outputs dir> <assignment.json> --brief task.md --reasons reasons.json \
           --rater NAME --votes votes.jsonl [--port 8700]
"""

import argparse
import contextlib
import json
import mimetypes
import os
import re
import secrets
import stat
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

HERE = Path(__file__).resolve().parent
HOST = "127.0.0.1"
CHOICES = ("left", "right", "tie", "abstain")
FIELDS = {"pair_id", "choice", "rater", "rated_at", "reason", "source", "aspect", "vote_id"}


def entry(item: Path) -> str:
    """The page to open in an output: its shallowest index.html, else its shallowest .html file."""
    for pattern in ("index.html", "*.html"):
        pages = sorted(item.rglob(pattern), key=lambda p: (len(p.parts), str(p)))
        if pages:
            return pages[0].relative_to(item).as_posix()
    raise SystemExit(f"{item} holds no .html page")


def brief(path: Path) -> str:
    """The task's instruction: task.md without its front matter, only its `## prompt` section if it has one."""
    text = re.sub(r"\A---\n.*?\n---\n", "", path.read_text(), flags=re.S)
    section = re.search(r"^## prompt[ \t]*\n(.*?)(?=^#{1,2} |\Z)", text, flags=re.M | re.S)
    return (section.group(1) if section else text).strip()


def read_votes(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def write_votes(path: Path, votes: list) -> None:
    """Replace the file in one step: a crash leaves the old list or the new one, never half of one."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w") as f:
        f.writelines(json.dumps(vote, ensure_ascii=False) + "\n" for vote in votes)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def check(votes, pair_ids: set, rater: str) -> None:
    if not isinstance(votes, list):
        raise ValueError("expected a list of votes")
    for vote in votes:
        if not isinstance(vote, dict) or set(vote) - FIELDS:
            raise ValueError(f"not a vote: {vote!r}")
        if (vote.get("pair_id") not in pair_ids or vote.get("choice") not in CHOICES
                or vote.get("rater") != rater or vote.get("source") != "judge.html"
                or not isinstance(vote.get("rated_at"), str) or not isinstance(vote.get("reason", ""), str)):
            raise ValueError(f"not a vote on these pairs by {rater}: {vote!r}")


def outputs_handler(root: Path, judge_origin: str):
    class Outputs(BaseHTTPRequestHandler):
        """GET and HEAD of regular files under root; no listing, no '..', and no link is followed."""

        def do_GET(self):
            self.send_file(body=True)

        def do_HEAD(self):
            self.send_file(body=False)

        def send_file(self, body: bool):
            path, info = root, None
            for part in unquote(urlsplit(self.path).path).split("/")[1:]:
                if part in ("", ".", "..") or "\\" in part or "\0" in part:
                    return self.send_error(404)
                path = path / part
                try:
                    info = path.lstat()
                except OSError:
                    return self.send_error(404)
                if stat.S_ISLNK(info.st_mode):
                    return self.send_error(404)
            if info is None or not stat.S_ISREG(info.st_mode):
                return self.send_error(404)
            data = path.read_bytes() if body else b""
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data) if body else info.st_size))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", f"frame-ancestors {judge_origin}")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    return Outputs


def judge_handler(config: dict, votes_path: Path, origin: str, sides: list):
    page = (HERE / "judge.html").read_bytes()
    pair_ids = {pair["pair_id"] for pair in config["pairs"]}
    lock = threading.Lock()
    csp = (f"default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; "
           f"frame-src {' '.join(sides)}; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")

    class Judge(BaseHTTPRequestHandler):
        def reply(self, code: int, data: bytes, kind: str = "text/plain; charset=utf-8"):
            self.send_response(code)
            for name, value in (("Content-Type", kind), ("Content-Length", str(len(data))),
                                ("Cache-Control", "no-store"), ("Content-Security-Policy", csp)):
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                return self.reply(200, page, "text/html; charset=utf-8")
            if path == "/config.json":
                with lock:
                    data = json.dumps({**config, "votes": read_votes(votes_path)}).encode()
                return self.reply(200, data, "application/json")
            self.reply(404, b"not found")

        def do_POST(self):
            if urlsplit(self.path).path != "/votes":
                return self.reply(404, b"not found")
            # Only the judging page itself: its origin, its token, and JSON (a request from another
            # origin with a custom header and this type needs a preflight, which is never answered).
            if (self.headers.get("Origin") != origin
                    or not secrets.compare_digest(self.headers.get("X-Judge-Token", ""), config["token"])
                    or self.headers.get_content_type() != "application/json"):
                return self.reply(403, b"refused")
            try:
                votes = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                check(votes, pair_ids, config["rater"])
            except ValueError as exc:
                return self.reply(400, str(exc).encode())
            with lock:
                write_votes(votes_path, votes)
            print(f"{len(votes)} vote{'' if len(votes) == 1 else 's'} in {votes_path}", flush=True)
            self.reply(200, json.dumps({"saved": len(votes)}).encode(), "application/json")

        def log_message(self, *args):
            pass

    return Judge


def main():
    ap = argparse.ArgumentParser(description="Serve blind pairs to a person judging in judge.html.")
    ap.add_argument("outputs", type=Path, help="the store-outputs folder `bench preferences pairs` made")
    ap.add_argument("assignment", type=Path, help="the assignment.json it wrote")
    ap.add_argument("--brief", type=Path, required=True, help="the task's task.md (or its instruction)")
    ap.add_argument("--reasons", type=Path, required=True, help="reasons.json: the reason keys 1 to 8")
    ap.add_argument("--rater", required=True, help="the judge's name, recorded on every vote")
    ap.add_argument("--votes", type=Path, required=True, help="this judge's votes file (JSON lines)")
    ap.add_argument("--port", type=int, default=8700, help="the judging page's port; the outputs use the next two")
    a = ap.parse_args()
    root = a.outputs.resolve()
    origin, side_a, side_b = (f"http://{HOST}:{a.port + n}" for n in range(3))
    pairs = []
    for pair in json.loads(a.assignment.read_text())["pairs"]:
        ids = pair["left"], pair["right"]
        if not all(re.fullmatch(r"[A-Za-z0-9_-]+", i) and (root / i).is_dir() for i in ids):
            raise SystemExit(f"pair {pair['pair_id']}: an output is missing from {root}")
        pairs.append({"pair_id": pair["pair_id"],
                      "a": f"{side_a}/{ids[0]}/{entry(root / ids[0])}",
                      "b": f"{side_b}/{ids[1]}/{entry(root / ids[1])}"})
    others = {vote.get("rater") for vote in read_votes(a.votes)} - {a.rater}
    if others:
        raise SystemExit(f"{a.votes} holds votes by {', '.join(sorted(map(str, others)))}: give each judge a file")
    config = {"rater": a.rater, "brief": brief(a.brief), **json.loads(a.reasons.read_text()),
              "pairs": pairs, "token": secrets.token_urlsafe(18)}
    judge = ThreadingHTTPServer((HOST, a.port), judge_handler(config, a.votes.resolve(), origin, [side_a, side_b]))
    for port in (a.port + 1, a.port + 2):
        server = ThreadingHTTPServer((HOST, port), outputs_handler(root, origin))
        threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"{len(pairs)} pairs for {a.rater}: open {origin}/ (votes go to {a.votes}; Ctrl-C stops)", flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        judge.serve_forever()


if __name__ == "__main__":
    main()
