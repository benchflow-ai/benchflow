"""Tiny SMTP/IMAP service for Toolathlon email tasks.

The service intentionally implements only the protocol subset exercised by
Toolathlon's ``emails-mcp`` package and verifier helpers:

* SMTP/Submission accepts plaintext mail, optional AUTH, and DATA.
* Delivered messages are copied to each recipient INBOX and the envelope
  sender's Sent folder, matching the verifier's sender-outbox checks.
* IMAP supports LOGIN, LIST, SELECT, APPEND, SEARCH ALL/FROM, FETCH RFC822,
  STORE +FLAGS \\Deleted, EXPUNGE, CLOSE, NOOP, and LOGOUT.

It is not a general mail server. It exists to avoid pulling the much larger
poste.io appliance inside 10 GB Daytona DinD sandboxes.
"""

from __future__ import annotations

import argparse
import email
import http.server
import re
import socketserver
import threading
import time
from pathlib import Path

_ADDR_RE = re.compile(r"<([^>]+)>")
_SAFE_RE = re.compile(r"[^A-Za-z0-9_.@+-]+")


def _clean_mailbox(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    value = (value or "unknown@mcp.com").strip().strip('"').strip("'")
    match = _ADDR_RE.search(value)
    if match:
        value = match.group(1)
    value = value.strip()
    if not value:
        value = "unknown@mcp.com"
    return value.lower()


def _safe_component(value: str) -> str:
    return _SAFE_RE.sub("_", value)[:180] or "mailbox"


def _dequote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1]
    return value


class MailStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = threading.Lock()
        self._counter = 0

    def _folder_path(self, mailbox: str, folder: str) -> Path:
        mailbox_dir = self.root / _safe_component(mailbox)
        folder_dir = mailbox_dir / ("Sent" if folder.lower() == "sent" else "INBOX")
        folder_dir.mkdir(parents=True, exist_ok=True)
        return folder_dir

    def ensure_mailbox(self, mailbox: str) -> None:
        self._folder_path(mailbox, "INBOX")
        self._folder_path(mailbox, "Sent")

    def append(self, mailbox: str, folder: str, raw: bytes) -> None:
        mailbox = _clean_mailbox(mailbox)
        with self._lock:
            self._counter += 1
            name = f"{time.time_ns()}_{self._counter:06d}.eml"
            self._folder_path(mailbox, folder).joinpath(name).write_bytes(raw)

    def deliver(self, sender: str, recipients: list[str], raw: bytes) -> None:
        sender = _clean_mailbox(sender)
        self.append(sender, "Sent", raw)
        for recipient in recipients:
            self.append(_clean_mailbox(recipient), "INBOX", raw)

    def messages(self, mailbox: str, folder: str) -> list[Path]:
        self.ensure_mailbox(mailbox)
        folder_path = self._folder_path(mailbox, folder)
        return [
            path
            for path in sorted(folder_path.glob("*.eml"))
            if not path.with_suffix(path.suffix + ".deleted").exists()
        ]

    def mark_deleted(self, mailbox: str, folder: str, seqs: list[int]) -> None:
        messages = self.messages(mailbox, folder)
        with self._lock:
            for seq in seqs:
                if 1 <= seq <= len(messages):
                    messages[seq - 1].with_suffix(
                        messages[seq - 1].suffix + ".deleted"
                    ).touch()

    def expunge(self, mailbox: str, folder: str) -> list[int]:
        self.ensure_mailbox(mailbox)
        folder_path = self._folder_path(mailbox, folder)
        deleted = sorted(folder_path.glob("*.eml.deleted"))
        expunged: list[int] = []
        with self._lock:
            for marker in deleted:
                message = marker.with_suffix("")
                all_messages = sorted(folder_path.glob("*.eml"))
                try:
                    seq = all_messages.index(message) + 1
                except ValueError:
                    seq = 1
                marker.unlink(missing_ok=True)
                message.unlink(missing_ok=True)
                expunged.append(seq)
        return expunged


class SMTPHandler(socketserver.StreamRequestHandler):
    store: MailStore

    def _line(self, text: str) -> None:
        self.wfile.write(text.encode("utf-8") + b"\r\n")

    def handle(self) -> None:
        sender = ""
        recipients: list[str] = []
        self._line("220 toolathlon fake mail ready")
        while True:
            raw_line = self.rfile.readline(65536)
            if not raw_line:
                return
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            command, _, rest = line.partition(" ")
            upper = command.upper()

            if upper in {"EHLO", "HELO"}:
                self._line("250-poste")
                self._line("250-AUTH PLAIN LOGIN")
                self._line("250 OK")
            elif upper == "AUTH":
                if rest.upper().startswith("LOGIN") and len(rest.split()) == 1:
                    self._line("334 VXNlcm5hbWU6")
                    self.rfile.readline(65536)
                    self._line("334 UGFzc3dvcmQ6")
                    self.rfile.readline(65536)
                self._line("235 Authentication successful")
            elif upper == "MAIL":
                sender = _clean_mailbox(rest)
                self._line("250 OK")
            elif upper == "RCPT":
                recipients.append(_clean_mailbox(rest))
                self._line("250 OK")
            elif upper == "DATA":
                self._line("354 End data with <CR><LF>.<CR><LF>")
                chunks: list[bytes] = []
                while True:
                    data_line = self.rfile.readline(65536)
                    if not data_line:
                        return
                    if data_line in {b".\r\n", b".\n", b"."}:
                        break
                    if data_line.startswith(b".."):
                        data_line = data_line[1:]
                    chunks.append(data_line)
                raw_message = b"".join(chunks)
                self.store.deliver(sender, recipients, raw_message)
                self._line("250 Queued")
            elif upper == "RSET":
                sender = ""
                recipients = []
                self._line("250 OK")
            elif upper == "NOOP":
                self._line("250 OK")
            elif upper == "QUIT":
                self._line("221 Bye")
                return
            else:
                self._line("250 OK")


class IMAPHandler(socketserver.StreamRequestHandler):
    store: MailStore

    def setup(self) -> None:
        super().setup()
        self.mailbox = "anonymous@mcp.com"
        self.selected_folder: str | None = None

    def _line(self, text: str) -> None:
        self.wfile.write(text.encode("utf-8") + b"\r\n")

    def _raw(self, data: bytes) -> None:
        self.wfile.write(data)

    def _tagged_ok(self, tag: str, text: str = "OK") -> None:
        self._line(f"{tag} OK {text}")

    def _parse_seqs(self, value: str) -> list[int]:
        out: list[int] = []
        for part in value.replace(",", " ").split():
            if ":" in part:
                start_s, end_s = part.split(":", 1)
                try:
                    start = int(start_s)
                    end = len(self._selected_messages()) if end_s == "*" else int(end_s)
                except ValueError:
                    continue
                out.extend(range(start, end + 1))
            else:
                try:
                    out.append(int(part))
                except ValueError:
                    continue
        return out

    def _selected_messages(self) -> list[Path]:
        folder = self.selected_folder or "INBOX"
        return self.store.messages(self.mailbox, folder)

    def _search(self, criteria: str) -> list[int]:
        messages = self._selected_messages()
        criteria_upper = criteria.upper().strip()
        if not criteria_upper or criteria_upper == "ALL":
            return list(range(1, len(messages) + 1))
        from_match = re.search(r'FROM\s+"?([^")]+)"?', criteria, flags=re.IGNORECASE)
        if from_match:
            needle = from_match.group(1).lower()
            hits: list[int] = []
            for seq, path in enumerate(messages, start=1):
                msg = email.message_from_bytes(path.read_bytes())
                if needle in (msg.get("From") or "").lower():
                    hits.append(seq)
            return hits
        return list(range(1, len(messages) + 1))

    def _append(self, tag: str, rest: str) -> None:
        mailbox, _, append_args = rest.partition(" ")
        folder = _dequote(mailbox or "INBOX")
        literal_match = re.search(r"\{(\d+)\}\s*$", append_args)
        if not literal_match:
            self._line(f"{tag} BAD APPEND expects literal data")
            return

        size = int(literal_match.group(1))
        self._line("+ Ready for literal data")
        raw = self.rfile.read(size)
        # Consume the trailing command newline after the literal.
        self.rfile.readline(65536)
        self.store.append(self.mailbox, folder, raw)
        self._tagged_ok(tag, "APPEND completed")

    def handle(self) -> None:
        self._line("* OK Toolathlon fake IMAP ready")
        while True:
            raw_line = self.rfile.readline(65536)
            if not raw_line:
                return
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                continue
            tag, _, remainder = line.partition(" ")
            command, _, rest = remainder.partition(" ")
            upper = command.upper()

            if upper == "CAPABILITY":
                self._line("* CAPABILITY IMAP4rev1 AUTH=PLAIN")
                self._tagged_ok(tag, "CAPABILITY completed")
            elif upper == "LOGIN":
                username = rest.split(" ", 1)[0] if rest else self.mailbox
                self.mailbox = _clean_mailbox(_dequote(username))
                self.store.ensure_mailbox(self.mailbox)
                self._tagged_ok(tag, "LOGIN completed.")
            elif upper == "LIST":
                self.store.ensure_mailbox(self.mailbox)
                self._line('* LIST () "." "INBOX"')
                self._line('* LIST () "." "Sent"')
                self._tagged_ok(tag, "LIST completed")
            elif upper in {"SELECT", "EXAMINE"}:
                self.selected_folder = _dequote(rest or "INBOX")
                messages = self._selected_messages()
                self._line("* FLAGS (\\Seen \\Deleted)")
                self._line("* OK [PERMANENTFLAGS (\\Seen \\Deleted)] flags permitted")
                self._line(f"* {len(messages)} EXISTS")
                self._line("* 0 RECENT")
                self._tagged_ok(tag, "[READ-WRITE] SELECT completed")
            elif upper == "SEARCH":
                hits = " ".join(str(seq) for seq in self._search(rest))
                self._line(f"* SEARCH {hits}".rstrip())
                self._tagged_ok(tag, "SEARCH completed")
            elif upper == "APPEND":
                self._append(tag, rest)
            elif upper == "FETCH":
                seq_part, _, _fetch_spec = rest.partition(" ")
                messages = self._selected_messages()
                for seq in self._parse_seqs(seq_part):
                    if 1 <= seq <= len(messages):
                        raw = messages[seq - 1].read_bytes()
                        self._raw(f"* {seq} FETCH (RFC822 {{{len(raw)}}}\r\n".encode())
                        self._raw(raw)
                        self._raw(b")\r\n")
                self._tagged_ok(tag, "FETCH completed")
            elif upper == "STORE":
                seq_part, _, _flags = rest.partition(" ")
                if self.selected_folder:
                    self.store.mark_deleted(
                        self.mailbox, self.selected_folder, self._parse_seqs(seq_part)
                    )
                self._tagged_ok(tag, "STORE completed")
            elif upper == "EXPUNGE":
                if self.selected_folder:
                    for seq in self.store.expunge(self.mailbox, self.selected_folder):
                        self._line(f"* {seq} EXPUNGE")
                self._tagged_ok(tag, "EXPUNGE completed")
            elif upper == "CLOSE":
                if self.selected_folder:
                    self.store.expunge(self.mailbox, self.selected_folder)
                self.selected_folder = None
                self._tagged_ok(tag, "CLOSE completed")
            elif upper == "NOOP":
                self._tagged_ok(tag, "NOOP completed")
            elif upper == "LOGOUT":
                self._line("* BYE LOGOUT requested")
                self._tagged_ok(tag, "LOGOUT completed")
                return
            else:
                self._line(f"{tag} BAD unsupported command")


class HealthHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok\n")

    def log_message(self, format: str, *args: object) -> None:
        return


class ThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _serve(server: ThreadingTCPServer) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/tmp/toolathlon-mail")
    parser.add_argument("--smtp-port", type=int, default=25)
    parser.add_argument("--submission-port", type=int, default=587)
    parser.add_argument("--imap-port", type=int, default=143)
    parser.add_argument("--http-port", type=int, default=80)
    parser.add_argument("--ready-file", default="/tmp/poste-ready")
    args = parser.parse_args()

    store = MailStore(Path(args.root))
    SMTPHandler.store = store
    IMAPHandler.store = store

    servers: list[ThreadingTCPServer] = []
    for port, handler in (
        (args.smtp_port, SMTPHandler),
        (args.submission_port, SMTPHandler),
        (args.imap_port, IMAPHandler),
        (args.http_port, HealthHandler),
    ):
        server = ThreadingTCPServer(("0.0.0.0", port), handler)
        servers.append(server)
        _serve(server)

    Path(args.ready_file).write_text("ready\n")
    print(
        "toolathlon fake mail ready "
        f"(smtp={args.smtp_port}, submission={args.submission_port}, "
        f"imap={args.imap_port}, http={args.http_port})",
        flush=True,
    )
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
