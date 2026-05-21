"""MIME / RFC 2822 utilities for Gmail API fidelity."""

from __future__ import annotations

import base64
import uuid
from datetime import datetime, timezone
from email import message_from_bytes, policy
from email.headerregistry import Address
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import format_datetime, parsedate_to_datetime

from .schemas import Header as HeaderSchema, MessagePart, MessagePartBody


def base64url_encode(data: bytes) -> str:
    """Encode bytes to base64url string (no padding)."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def base64url_decode(s: str) -> bytes:
    """Decode base64url string to bytes."""
    # Add padding
    padded = s + "=" * (4 - len(s) % 4)
    return base64.urlsafe_b64decode(padded)


def generate_message_id(domain: str = "claw-gmail.local") -> str:
    """Generate an RFC 2822 Message-ID."""
    return f"<{uuid.uuid4()}@{domain}>"


def build_rfc2822(
    sender: str,
    to: str,
    subject: str,
    body_plain: str = "",
    body_html: str = "",
    cc: str = "",
    bcc: str = "",
    in_reply_to: str = "",
    references: str = "",
    message_id: str = "",
    date: datetime | None = None,
    attachments: list | None = None,
) -> bytes:
    """Build a proper RFC 2822 MIME message.

    Returns raw bytes with CRLF line endings.
    """
    if not message_id:
        message_id = generate_message_id()
    if date is None:
        date = datetime.now(timezone.utc)

    has_html = bool(body_html)
    has_attachments = bool(attachments)

    # Build text part(s)
    if has_html:
        text_part = MIMEMultipart("alternative")
        text_part.attach(MIMEText(body_plain, "plain", "utf-8"))
        text_part.attach(MIMEText(body_html, "html", "utf-8"))
    else:
        text_part = MIMEText(body_plain, "plain", "utf-8")

    # Wrap with attachments if needed
    if has_attachments:
        msg = MIMEMultipart("mixed")
        msg.attach(text_part)
        for att in attachments:
            att_part = MIMEBase(
                att.mime_type.split("/")[0] if "/" in att.mime_type else "application",
                att.mime_type.split("/")[1] if "/" in att.mime_type else "octet-stream",
            )
            att_part.set_payload(base64.b64decode(att.data) if att.data else b"")
            att_part.add_header("Content-Disposition", "attachment", filename=att.filename)
            att_part.add_header("Content-Transfer-Encoding", "base64")
            msg.attach(att_part)
    else:
        msg = text_part

    # Set top-level headers
    msg["From"] = sender
    msg["To"] = to
    msg["Subject"] = subject
    msg["Date"] = format_datetime(date)
    msg["Message-ID"] = message_id
    msg["MIME-Version"] = "1.0"
    if cc:
        msg["Cc"] = cc
    if bcc:
        msg["Bcc"] = bcc
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
    if references:
        msg["References"] = references

    return msg.as_bytes(policy=policy.SMTP)


def parse_rfc2822(raw_bytes: bytes) -> dict:
    """Parse RFC 2822 message bytes into a dict of fields.

    Returns dict with: sender, to, cc, bcc, subject, body_plain, body_html,
    message_id, in_reply_to, references, date, attachments.
    """
    msg = message_from_bytes(raw_bytes, policy=policy.default)

    body_plain = ""
    body_html = ""
    attachments = []

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition", ""))

            if "attachment" in disposition:
                data = part.get_payload(decode=True)
                attachments.append({
                    "filename": part.get_filename() or "attachment",
                    "mime_type": content_type,
                    "size": len(data) if data else 0,
                    "data": base64.b64encode(data).decode() if data else "",
                })
            elif content_type == "text/plain" and not body_plain:
                payload = part.get_payload(decode=True)
                if payload:
                    body_plain = payload.decode("utf-8", errors="replace")
            elif content_type == "text/html" and not body_html:
                payload = part.get_payload(decode=True)
                if payload:
                    body_html = payload.decode("utf-8", errors="replace")
    else:
        content_type = msg.get_content_type()
        payload = msg.get_payload(decode=True)
        if payload:
            text = payload.decode("utf-8", errors="replace")
            if content_type == "text/html":
                body_html = text
            else:
                body_plain = text

    # Parse date
    date = None
    date_str = msg.get("Date", "")
    if date_str:
        try:
            date = parsedate_to_datetime(str(date_str))
        except (ValueError, TypeError):
            pass

    return {
        "sender": str(msg.get("From", "")),
        "to": str(msg.get("To", "")),
        "cc": str(msg.get("Cc", "")),
        "bcc": str(msg.get("Bcc", "")),
        "subject": str(msg.get("Subject", "")),
        "body_plain": body_plain,
        "body_html": body_html,
        "message_id": str(msg.get("Message-ID", "")),
        "in_reply_to": str(msg.get("In-Reply-To", "")),
        "references": str(msg.get("References", "")),
        "date": date,
        "attachments": attachments,
    }


def build_payload_tree(
    msg_model,
    attachments: list | None = None,
    include_body: bool = True,
) -> MessagePart:
    """Build the nested Gmail-style payload tree from a Message model.

    Args:
        msg_model: Message ORM object
        attachments: list of Attachment ORM objects
        include_body: if True, include body.data (format=full); if False, omit (format=metadata)

    Returns:
        MessagePart schema with proper nested structure.
    """
    if attachments is None:
        attachments = list(msg_model.attachments) if hasattr(msg_model, "attachments") else []

    date_str = ""
    if msg_model.internal_date:
        try:
            dt = msg_model.internal_date
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            date_str = format_datetime(dt)
        except Exception:
            date_str = msg_model.internal_date.isoformat()

    # Build top-level headers
    headers = [
        HeaderSchema(name="From", value=msg_model.sender or ""),
        HeaderSchema(name="To", value=msg_model.to or ""),
        HeaderSchema(name="Subject", value=msg_model.subject or ""),
        HeaderSchema(name="Date", value=date_str),
        HeaderSchema(name="MIME-Version", value="1.0"),
    ]
    if msg_model.cc:
        headers.append(HeaderSchema(name="Cc", value=msg_model.cc))
    if msg_model.bcc:
        headers.append(HeaderSchema(name="Bcc", value=msg_model.bcc))
    if msg_model.reply_to:
        headers.append(HeaderSchema(name="Reply-To", value=msg_model.reply_to))

    message_id_header = getattr(msg_model, "message_id_header", "") or ""
    if message_id_header:
        headers.append(HeaderSchema(name="Message-ID", value=message_id_header))
    in_reply_to = getattr(msg_model, "in_reply_to", "") or ""
    if in_reply_to:
        headers.append(HeaderSchema(name="In-Reply-To", value=in_reply_to))
    references = getattr(msg_model, "references", "") or ""
    if references:
        headers.append(HeaderSchema(name="References", value=references))

    has_html = bool(msg_model.body_html)
    has_attachments = bool(attachments)

    def _encode_body(text: str) -> MessagePartBody:
        if not include_body:
            return MessagePartBody(size=len(text.encode("utf-8")) if text else 0)
        encoded = base64url_encode(text.encode("utf-8")) if text else None
        return MessagePartBody(
            size=len(text.encode("utf-8")) if text else 0,
            data=encoded,
        )

    # Real Gmail partId scheme: root="" , children="0","1","2",...
    # Nested multipart: root="", child="0" (multipart/alt), grandchildren="0.0","0.1"

    # Case 1: plain only, no attachments
    if not has_html and not has_attachments:
        return MessagePart(
            partId="",
            mimeType="text/plain",
            headers=headers,
            body=_encode_body(msg_model.body_plain or ""),
        )

    # Case 2: plain + html, no attachments → multipart/alternative
    if has_html and not has_attachments:
        plain_part = MessagePart(
            partId="0",
            mimeType="text/plain",
            headers=[
                HeaderSchema(name="Content-Type", value='text/plain; charset="UTF-8"'),
                HeaderSchema(name="Content-Transfer-Encoding", value="quoted-printable"),
            ],
            body=_encode_body(msg_model.body_plain or ""),
        )
        html_part = MessagePart(
            partId="1",
            mimeType="text/html",
            headers=[
                HeaderSchema(name="Content-Type", value='text/html; charset="UTF-8"'),
                HeaderSchema(name="Content-Transfer-Encoding", value="quoted-printable"),
            ],
            body=_encode_body(msg_model.body_html or ""),
        )
        return MessagePart(
            partId="",
            mimeType="multipart/alternative",
            headers=headers,
            body=MessagePartBody(size=0),
            parts=[plain_part, html_part],
        )

    # Case 3: with attachments → multipart/mixed
    # Build text part(s) first
    if has_html:
        plain_part = MessagePart(
            partId="0.0",
            mimeType="text/plain",
            headers=[
                HeaderSchema(name="Content-Type", value='text/plain; charset="UTF-8"'),
                HeaderSchema(name="Content-Transfer-Encoding", value="quoted-printable"),
            ],
            body=_encode_body(msg_model.body_plain or ""),
        )
        html_part = MessagePart(
            partId="0.1",
            mimeType="text/html",
            headers=[
                HeaderSchema(name="Content-Type", value='text/html; charset="UTF-8"'),
                HeaderSchema(name="Content-Transfer-Encoding", value="quoted-printable"),
            ],
            body=_encode_body(msg_model.body_html or ""),
        )
        text_container = MessagePart(
            partId="0",
            mimeType="multipart/alternative",
            body=MessagePartBody(size=0),
            parts=[plain_part, html_part],
        )
    else:
        text_container = MessagePart(
            partId="0",
            mimeType="text/plain",
            headers=[
                HeaderSchema(name="Content-Type", value='text/plain; charset="UTF-8"'),
                HeaderSchema(name="Content-Transfer-Encoding", value="quoted-printable"),
            ],
            body=_encode_body(msg_model.body_plain or ""),
        )

    # Build attachment parts
    att_parts = []
    for idx, att in enumerate(attachments):
        att_body = MessagePartBody(
            attachmentId=att.id,
            size=att.size or 0,
        )
        att_parts.append(MessagePart(
            partId=str(idx + 1),
            mimeType=att.mime_type or "application/octet-stream",
            filename=att.filename or "",
            body=att_body,
        ))

    return MessagePart(
        partId="",
        mimeType="multipart/mixed",
        headers=headers,
        body=MessagePartBody(size=0),
        parts=[text_container] + att_parts,
    )


def build_raw_field(msg_model, attachments: list | None = None) -> str:
    """Build base64url-encoded RFC 2822 for format=raw."""
    if attachments is None:
        attachments = list(msg_model.attachments) if hasattr(msg_model, "attachments") else []

    date = msg_model.internal_date
    if date and date.tzinfo is None:
        date = date.replace(tzinfo=timezone.utc)

    raw_bytes = build_rfc2822(
        sender=msg_model.sender or "",
        to=msg_model.to or "",
        subject=msg_model.subject or "",
        body_plain=msg_model.body_plain or "",
        body_html=msg_model.body_html or "",
        cc=msg_model.cc or "",
        bcc=msg_model.bcc or "",
        in_reply_to=getattr(msg_model, "in_reply_to", "") or "",
        references=getattr(msg_model, "references", "") or "",
        message_id=getattr(msg_model, "message_id_header", "") or "",
        date=date,
        attachments=attachments,
    )
    return base64url_encode(raw_bytes)
