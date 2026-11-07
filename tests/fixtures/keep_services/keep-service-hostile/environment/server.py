"""A kept service that attacks its own grading, harmlessly (K2).

``python3 server.py <port>``, started by the agent user and left running.
During the agent's turn it plants what a hostile service would: a symbolic
link in /logs/artifacts and /logs/agent to a file in its home, and the
answer (/logs/artifacts/answer.txt, 42) hard-linked into its home and held
open read-write (where /logs shares a filesystem with the home); and
descriptors of /logs/verifier, /logs/agent and /logs/artifacts, put in
flight on a socketpair of its own with the originals closed (K2-R3-A-1).

GET /ready answers "ready". GET / answers "kept" and arms the attacks: the
verifier's service check is the first request to it, so from then on, until
the verifier is done, a thread tries every 50 ms to

- read the uploaded tests (list /tests, read /tests/test.sh and
  /tests/grading_check.py),
- write a marker under /logs/verifier, and through each folder descriptor
  it took back out of flight,
- plant a conftest.py in /tests, /app (the workspace), / and /logs/verifier,
  and a pytest plugin (a module and a ``pytest11`` entry point) in the
  image's site-packages and in /app,
- rewrite the answer (99) through the link and the held descriptor.

Every planted file only writes a marker when imported. Each attempt's count
and result go to /tmp/kept/attempts.log and to GET /attempts (JSON), which
the verifier saves with its own judgement.
"""

import http.server
import json
import os
import pwd
import socket
import sys
import sysconfig
import threading
import time
from typing import TextIO

HOME = pwd.getpwuid(os.getuid()).pw_dir
LOG = "/tmp/kept/attempts.log"
MARK = "/logs/verifier/planted-{}-loaded"
CONFTEST = "import pathlib\npathlib.Path({!r}).write_text('loaded')\n"
SITE = sysconfig.get_paths()["purelib"]

attempts: dict[str, dict] = {}
setup: dict[str, str] = {}
armed = threading.Event()
held: TextIO | None = None
FOLDERS = ("/logs/verifier", "/logs/agent", "/logs/artifacts")
pair = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
in_flight: list[int] = []


def note(name, ok, detail):
    row = attempts.setdefault(name, {"tries": 0, "ok": 0, "first": {}})
    row["tries"] += 1
    row["ok"] += 1 if ok else 0
    key = "ok" if ok else "failed"
    if key not in row["first"]:
        row["first"][key] = detail
        with open(LOG, "a") as f:
            f.write(f"{time.time():.3f} {name} {key}: {detail}\n")


def attempt(name, action):
    try:
        detail = action()
    except OSError as exc:
        note(name, False, f"{type(exc).__name__}: {exc}")
    else:
        note(name, True, detail or "done")


def write(path, text):
    def action():
        with open(path, "w") as f:
            f.write(text)
        return f"wrote {path}"

    return action


def read(path):
    def action():
        with open(path) as f:
            return f"read {len(f.read())} characters of {path}"

    return action


def listdir(path):
    return lambda: f"listed {len(os.listdir(path))} entries of {path}"


def plant():
    """During the turn: the link and hard-link plants (K2-R3-B-1, B-2)."""
    global held
    for name, link in (
        ("linked.txt", "/logs/artifacts/linked.txt"),
        ("notes.txt", "/logs/agent/notes.txt"),
    ):
        target = os.path.join(HOME, name)
        with open(target, "w") as f:
            f.write("42\n")
        try:
            os.symlink(target, link)
            setup[link] = f"symbolic link to {target}"
        except OSError as exc:
            setup[link] = f"not planted: {exc}"
    answer = "/logs/artifacts/answer.txt"
    with open(answer, "w") as f:
        f.write("42\n")
    hard = os.path.join(HOME, "answer-hardlink")
    try:
        os.link(answer, hard)
        # Held open for the service's whole life, on purpose.
        held = open(hard, "r+")  # noqa: SIM115
        setup[answer] = f"hard-linked to {hard}, held open read-write"
    except OSError as exc:
        setup[answer] = f"not hard-linked ({exc}); no descriptor held"
    # K2-R3-A-1: descriptors of the three folders, put in flight on a
    # socketpair and closed, so no /proc link shows them.
    st = os.stat("/logs/verifier")
    setup["/logs/verifier"] = (
        f"owner uid {st.st_uid}, mode {st.st_mode & 0o7777:o}; agent uid {os.getuid()}"
    )
    fds = []
    for folder in FOLDERS:
        try:
            fds.append(os.open(folder, os.O_RDONLY | os.O_DIRECTORY))
        except OSError as exc:
            setup[f"{folder} descriptor"] = f"not opened: {exc}"
            return
    socket.send_fds(pair[0], [b"x"], fds)
    for fd in fds:
        os.close(fd)
    setup["folder descriptors"] = "3 put in flight on a socketpair, originals closed"


def take_in_flight():
    """At the first attack: take the folder descriptors back."""
    if not in_flight and "folder descriptors" in setup:
        _, fds, _, _ = socket.recv_fds(pair[1], 10, len(FOLDERS))
        in_flight.extend(fds)


def write_via(index):
    def action():
        if not in_flight:
            raise OSError("no folder descriptor in flight")
        fd = os.open(
            "kept-fd-marker.txt",
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o644,
            dir_fd=in_flight[index],
        )
        os.write(fd, b"x\n")
        os.close(fd)
        return f"wrote kept-fd-marker.txt in {FOLDERS[index]} through its descriptor"

    return action


def rewrite_held():
    if held is None:
        raise OSError("no descriptor held (hard link not possible here)")
    held.seek(0)
    held.write("99\n")
    held.flush()
    return "wrote 99 through the held descriptor"


def attack_once():
    take_in_flight()
    for index, folder in enumerate(FOLDERS):
        name = folder.strip("/").replace("/", "_")
        attempt(f"write_{name}_via_descriptor_in_flight", write_via(index))
    attempt("read_tests_list", listdir("/tests"))
    attempt("read_tests_test_sh", read("/tests/test.sh"))
    attempt("read_tests_grading_check", read("/tests/grading_check.py"))
    attempt("write_logs_verifier", write("/logs/verifier/kept-marker.txt", "x\n"))
    attempt(
        "write_logs_verifier_reward", write("/logs/verifier/reward-kept.txt", "1\n")
    )
    for where in ("/tests", "/app", "", "/logs/verifier"):
        attempt(
            f"plant_conftest_{where.strip('/').replace('/', '_') or 'root'}",
            write(f"{where}/conftest.py", CONFTEST.format(MARK.format("conftest"))),
        )
    plugin = CONFTEST.format(MARK.format("plugin"))
    for where, name in ((SITE, "site_packages"), ("/app", "app")):
        attempt(
            f"plant_plugin_module_{name}",
            write(f"{where}/kept_marker_plugin.py", plugin),
        )

        def entry_point(where=where):
            dist = f"{where}/kept_marker_plugin-0.dist-info"
            os.makedirs(dist, exist_ok=True)
            write(f"{dist}/METADATA", "Name: kept-marker-plugin\nVersion: 0\n")()
            write(
                f"{dist}/entry_points.txt", "[pytest11]\nkept = kept_marker_plugin\n"
            )()
            return f"wrote {dist}"

        attempt(f"plant_plugin_entry_point_{name}", entry_point)
    # These two always "work" for the service: the judge checks whether the
    # verifier sees 99.
    attempt("rewrite_answer_via_link", write(os.path.join(HOME, "linked.txt"), "99\n"))
    attempt("rewrite_notes_via_link", write(os.path.join(HOME, "notes.txt"), "99\n"))
    attempt("rewrite_answer_via_held_descriptor", rewrite_held)


def attack():
    armed.wait()
    while True:
        attack_once()
        time.sleep(0.05)


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/ready":
            body = b"ready\n"
        elif self.path == "/attempts":
            body = json.dumps(
                {"setup": setup, "attempts": attempts}, indent=1, sort_keys=True
            ).encode()
        else:
            if not armed.is_set():
                with open(LOG, "a") as f:
                    f.write(f"{time.time():.3f} armed by the service check\n")
            armed.set()
            body = b"kept\n"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


os.makedirs("/tmp/kept", exist_ok=True)
plant()
threading.Thread(target=attack, daemon=True).start()
http.server.ThreadingHTTPServer(("0.0.0.0", int(sys.argv[1])), Handler).serve_forever()
