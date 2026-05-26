"""Messages API routes — mirrors Gmail REST API."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from claw_gmail.models import Message, Thread, MessageLabel, Label, User
from .deps import get_db, resolve_user_id
from .history_tracker import (
    record_message_added,
    record_message_deleted,
    record_labels_added,
    record_labels_removed,
)
from .mime import (
    base64url_decode,
    generate_message_id,
    parse_rfc2822,
    build_payload_tree,
    build_raw_field,
)
from .schemas import (
    MessageSchema,
    MessageMinimalSchema,
    MessageListResponse,
    MessageListItem,
    MessageSendRequest,
    MessageModifyRequest,
    MessageBatchModifyRequest,
    MessageBatchDeleteRequest,
    MessageInsertRequest,
    MessagePart,
    MessagePartBody,
    Header as HeaderSchema,
)

router = APIRouter()


def _build_label_ids(msg: Message) -> list[str]:
    """Compute label IDs from message state + explicit labels."""
    ids = [ml.label_id for ml in msg.labels]
    if not msg.is_read and "UNREAD" not in ids:
        ids.append("UNREAD")
    if msg.is_starred and "STARRED" not in ids:
        ids.append("STARRED")
    if msg.is_trash and "TRASH" not in ids:
        ids.append("TRASH")
    if msg.is_spam and "SPAM" not in ids:
        ids.append("SPAM")
    if msg.is_draft and "DRAFT" not in ids:
        ids.append("DRAFT")
    if msg.is_sent and "SENT" not in ids:
        ids.append("SENT")
    return ids


def _message_to_schema(msg: Message, format: str = "full") -> dict:
    """Convert a Message ORM object to API-response dict.

    Returns a dict (not a Pydantic model) so that absent fields like
    payload/raw are omitted entirely rather than serialized as null.
    """
    label_ids = _build_label_ids(msg)
    internal_date = str(int(msg.internal_date.timestamp() * 1000)) if msg.internal_date else None
    history_id = str(msg.user.history_id) if msg.user else None
    size_estimate = len((msg.body_plain or "").encode()) + len((msg.body_html or "").encode())

    data: dict = {
        "id": msg.id,
        "threadId": msg.thread_id,
        "labelIds": label_ids,
        "snippet": msg.snippet,
        "historyId": history_id,
        "internalDate": internal_date,
        "sizeEstimate": size_estimate,
    }

    if format == "raw":
        data["raw"] = build_raw_field(msg)
    elif format == "full":
        payload = build_payload_tree(msg, include_body=True)
        data["payload"] = payload.model_dump(exclude_none=True)
    elif format == "metadata":
        # Real Gmail metadata payload has only {mimeType, headers} (Bug 3)
        full_payload = build_payload_tree(msg, include_body=False)
        data["payload"] = {
            "mimeType": full_payload.mimeType,
            "headers": [h.model_dump() for h in full_payload.headers],
        }
    # format == "minimal": no payload, no raw

    return {k: v for k, v in data.items() if v is not None}


def _parse_search_query(q: str) -> dict:
    """Parse Gmail search query syntax into filter dict."""
    filters = {}
    tokens = []
    i = 0
    while i < len(q):
        # Check for operator:value patterns
        for op in ("from:", "to:", "subject:", "label:", "is:", "in:", "category:", "after:", "before:", "has:", "older_than:", "newer_than:"):
            if q[i:].lower().startswith(op):
                i += len(op)
                # Read value (quoted or unquoted)
                if i < len(q) and q[i] == '"':
                    end = q.index('"', i + 1)
                    value = q[i + 1:end]
                    i = end + 1
                else:
                    end = q.find(" ", i)
                    if end == -1:
                        end = len(q)
                    value = q[i:end]
                    i = end
                key = op.rstrip(":")
                filters.setdefault(key, []).append(value)
                break
        else:
            if q[i] == " ":
                i += 1
            else:
                end = q.find(" ", i)
                if end == -1:
                    end = len(q)
                tokens.append(q[i:end])
                i = end
    if tokens:
        filters["text"] = tokens
    return filters


def _apply_search_filters(query, filters: dict, user_id: str, db: Session):
    """Apply parsed search filters to a SQLAlchemy query."""
    for key, values in filters.items():
        for val in values:
            val_lower = val.lower()
            if key == "from":
                query = query.filter(Message.sender.ilike(f"%{val}%"))
            elif key == "to":
                query = query.filter(Message.to.ilike(f"%{val}%"))
            elif key == "subject":
                query = query.filter(Message.subject.ilike(f"%{val}%"))
            elif key == "is":
                if val_lower == "unread":
                    query = query.filter(Message.is_read == False)
                elif val_lower == "read":
                    query = query.filter(Message.is_read == True)
                elif val_lower == "starred":
                    query = query.filter(Message.is_starred == True)
                elif val_lower == "important":
                    query = query.filter(Message.labels.any(MessageLabel.label_id == "IMPORTANT"))
            elif key == "label":
                query = query.filter(Message.labels.any(MessageLabel.label_id == val))
            elif key == "in":
                label_map = {"inbox": "INBOX", "sent": "SENT", "trash": "TRASH", "spam": "SPAM", "drafts": "DRAFT"}
                label_id = label_map.get(val_lower, val.upper())
                query = query.filter(Message.labels.any(MessageLabel.label_id == label_id))
            elif key == "category":
                label_id = f"CATEGORY_{val.upper()}"
                query = query.filter(Message.labels.any(MessageLabel.label_id == label_id))
            elif key == "after":
                try:
                    dt = datetime.strptime(val, "%Y/%m/%d")
                    query = query.filter(Message.internal_date >= dt)
                except ValueError:
                    pass
            elif key == "before":
                try:
                    dt = datetime.strptime(val, "%Y/%m/%d")
                    query = query.filter(Message.internal_date < dt)
                except ValueError:
                    pass
            elif key == "older_than":
                # Parse values like "30d", "2m", "1y"
                import re
                m = re.match(r"(\d+)([dmy])", val_lower)
                if m:
                    num, unit = int(m.group(1)), m.group(2)
                    if unit == "d":
                        delta = timedelta(days=num)
                    elif unit == "m":
                        delta = timedelta(days=num * 30)
                    else:
                        delta = timedelta(days=num * 365)
                    cutoff = datetime.utcnow() - delta
                    query = query.filter(Message.internal_date < cutoff)
            elif key == "newer_than":
                import re
                m = re.match(r"(\d+)([dmy])", val_lower)
                if m:
                    num, unit = int(m.group(1)), m.group(2)
                    if unit == "d":
                        delta = timedelta(days=num)
                    elif unit == "m":
                        delta = timedelta(days=num * 30)
                    else:
                        delta = timedelta(days=num * 365)
                    cutoff = datetime.utcnow() - delta
                    query = query.filter(Message.internal_date >= cutoff)
            elif key == "has" and val_lower == "attachment":
                from claw_gmail.models import Attachment
                query = query.filter(Message.attachments.any())
            elif key == "text":
                query = query.filter(
                    Message.subject.ilike(f"%{val}%")
                    | Message.body_plain.ilike(f"%{val}%")
                    | Message.sender.ilike(f"%{val}%")
                )
    return query


@router.get("/users/{userId}/messages")
def list_messages(
    userId: str,
    q: str | None = Query(None),
    maxResults: int = Query(100, ge=1, le=500),
    pageToken: str | None = Query(None),
    labelIds: list[str] | None = Query(None),
    includeSpamTrash: bool = Query(False),
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    query = db.query(Message).filter(Message.user_id == _user_id)

    if not includeSpamTrash:
        query = query.filter(Message.is_trash == False, Message.is_spam == False)

    if labelIds:
        # Normalize: handle both repeated params and comma-separated
        normalized = []
        for item in labelIds:
            for lid in item.split(","):
                lid = lid.strip()
                if lid:
                    normalized.append(lid)
        for lid in normalized:
            if lid == "UNREAD":
                query = query.filter(Message.is_read == False)
            elif lid == "STARRED":
                query = query.filter(Message.is_starred == True)
            elif lid == "TRASH":
                query = query.filter(Message.is_trash == True)
            elif lid == "SPAM":
                query = query.filter(Message.is_spam == True)
            elif lid == "DRAFT":
                query = query.filter(Message.is_draft == True)
            elif lid == "SENT":
                query = query.filter(Message.is_sent == True)
            else:
                query = query.join(MessageLabel).filter(MessageLabel.label_id == lid)

    if q:
        filters = _parse_search_query(q)
        query = _apply_search_filters(query, filters, _user_id, db)

    # Pagination
    offset = 0
    if pageToken:
        try:
            offset = int(pageToken)
        except ValueError:
            pass

    total = query.count()
    messages = query.order_by(Message.internal_date.desc()).offset(offset).limit(maxResults).all()

    next_token = None
    if offset + maxResults < total:
        next_token = str(offset + maxResults)

    # Real Gmail omits nextPageToken when absent, omits messages key when empty (Bug 2/4)
    data: dict = {"resultSizeEstimate": total}
    if messages:
        data["messages"] = [{"id": m.id, "threadId": m.thread_id} for m in messages]
    if next_token:
        data["nextPageToken"] = next_token
    return data


@router.get("/users/{userId}/messages/{messageId}")
def get_message(
    userId: str,
    messageId: str,
    format: str = Query("full"),
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    msg = db.query(Message).filter(Message.id == messageId, Message.user_id == _user_id).first()
    if not msg:
        raise HTTPException(404, f"Message {messageId!r} not found")
    return _message_to_schema(msg, format=format)


@router.post("/users/{userId}/messages/send", response_model=MessageMinimalSchema)
def send_message(
    userId: str,
    body: MessageSendRequest,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    user = db.query(User).filter(User.id == _user_id).first()
    msg_id = uuid.uuid4().hex[:16]
    now = datetime.utcnow()

    # Parse raw RFC 2822 (matches real Gmail API which requires raw field)
    raw_bytes = base64url_decode(body.raw)
    parsed = parse_rfc2822(raw_bytes)
    to = parsed["to"]
    cc = parsed["cc"]
    bcc = parsed["bcc"]
    subject = parsed["subject"]
    body_plain = parsed["body_plain"]
    body_html = parsed["body_html"]
    message_id_header = parsed["message_id"]
    in_reply_to = parsed["in_reply_to"]
    references = parsed["references"]
    if parsed["date"]:
        now = parsed["date"].replace(tzinfo=None) if parsed["date"].tzinfo else parsed["date"]

    thread_id = body.threadId
    snippet = body_plain[:200]

    if not thread_id:
        thread_id = uuid.uuid4().hex[:16]
        thread = Thread(id=thread_id, user_id=_user_id, snippet=snippet)
        db.add(thread)
    else:
        thread = db.query(Thread).filter(Thread.id == thread_id).first()
        if thread:
            thread.snippet = snippet

    msg = Message(
        id=msg_id,
        thread_id=thread_id,
        user_id=_user_id,
        sender=user.email_address,
        to=to,
        cc=cc,
        bcc=bcc,
        subject=subject,
        snippet=snippet,
        body_plain=body_plain,
        body_html=body_html,
        internal_date=now,
        is_sent=True,
        message_id_header=message_id_header,
        in_reply_to=in_reply_to,
        references=references,
    )
    db.add(msg)

    # Add SENT and INBOX labels
    for lid in ("SENT", "INBOX"):
        label = db.query(Label).filter(Label.id == lid, Label.user_id == _user_id).first()
        if label:
            db.add(MessageLabel(message_id=msg_id, label_id=lid))

    # Deliver to recipient if they exist in the system
    _deliver_to_recipients(db, msg, user)

    record_message_added(db, _user_id, msg_id)
    record_labels_added(db, _user_id, msg_id, ["SENT", "INBOX"])
    db.commit()
    db.refresh(msg)
    return MessageMinimalSchema(id=msg.id, threadId=msg.thread_id, labelIds=_build_label_ids(msg))


def _deliver_to_recipients(db: Session, msg: Message, sender: User):
    """Deliver a copy of the message to local recipients."""
    recipients = [r.strip() for r in msg.to.split(",") if r.strip()]
    for recipient_email in recipients:
        recipient = db.query(User).filter(User.email_address == recipient_email).first()
        if recipient and recipient.id != sender.id:
            recv_msg_id = uuid.uuid4().hex[:16]
            # Find or create thread for recipient
            recv_thread_id = uuid.uuid4().hex[:16]
            recv_thread = Thread(id=recv_thread_id, user_id=recipient.id, snippet=msg.snippet)
            db.add(recv_thread)

            recv_msg = Message(
                id=recv_msg_id,
                thread_id=recv_thread_id,
                user_id=recipient.id,
                sender=sender.email_address,
                to=msg.to,
                cc=msg.cc,
                subject=msg.subject,
                snippet=msg.snippet,
                body_plain=msg.body_plain,
                body_html=msg.body_html,
                internal_date=msg.internal_date,
                is_read=False,
            )
            db.add(recv_msg)
            for lid in ("INBOX",):
                label = db.query(Label).filter(Label.id == lid, Label.user_id == recipient.id).first()
                if label:
                    db.add(MessageLabel(message_id=recv_msg_id, label_id=lid))
            record_message_added(db, recipient.id, recv_msg_id)
            record_labels_added(db, recipient.id, recv_msg_id, ["INBOX"])


@router.post("/users/{userId}/messages/{messageId}/modify", response_model=MessageMinimalSchema)
def modify_message(
    userId: str,
    messageId: str,
    body: MessageModifyRequest,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    msg = db.query(Message).filter(Message.id == messageId, Message.user_id == _user_id).first()
    if not msg:
        raise HTTPException(404, f"Message {messageId!r} not found")

    for lid in body.addLabelIds:
        if lid == "UNREAD":
            msg.is_read = False
        elif lid == "STARRED":
            msg.is_starred = True
        elif lid == "TRASH":
            msg.is_trash = True
        elif lid == "SPAM":
            msg.is_spam = True
        elif lid == "IMPORTANT":
            existing = db.query(MessageLabel).filter(
                MessageLabel.message_id == messageId, MessageLabel.label_id == lid
            ).first()
            if not existing:
                db.add(MessageLabel(message_id=messageId, label_id=lid))
        else:
            existing = db.query(MessageLabel).filter(
                MessageLabel.message_id == messageId, MessageLabel.label_id == lid
            ).first()
            if not existing:
                db.add(MessageLabel(message_id=messageId, label_id=lid))

    for lid in body.removeLabelIds:
        if lid == "UNREAD":
            msg.is_read = True
        elif lid == "STARRED":
            msg.is_starred = False
        elif lid == "TRASH":
            msg.is_trash = False
        elif lid == "SPAM":
            msg.is_spam = False
        elif lid == "INBOX":
            db.query(MessageLabel).filter(
                MessageLabel.message_id == messageId, MessageLabel.label_id == "INBOX"
            ).delete()
        else:
            db.query(MessageLabel).filter(
                MessageLabel.message_id == messageId, MessageLabel.label_id == lid
            ).delete()

    if body.addLabelIds:
        record_labels_added(db, _user_id, messageId, body.addLabelIds)
    if body.removeLabelIds:
        record_labels_removed(db, _user_id, messageId, body.removeLabelIds)
    db.commit()
    db.refresh(msg)
    return MessageMinimalSchema(id=msg.id, threadId=msg.thread_id, labelIds=_build_label_ids(msg))


@router.post("/users/{userId}/messages/{messageId}/trash", response_model=MessageMinimalSchema)
def trash_message(
    userId: str,
    messageId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    msg = db.query(Message).filter(Message.id == messageId, Message.user_id == _user_id).first()
    if not msg:
        raise HTTPException(404, f"Message {messageId!r} not found")
    msg.is_trash = True
    # Remove from INBOX
    db.query(MessageLabel).filter(
        MessageLabel.message_id == messageId, MessageLabel.label_id == "INBOX"
    ).delete()
    record_labels_added(db, _user_id, messageId, ["TRASH"])
    record_labels_removed(db, _user_id, messageId, ["INBOX"])
    db.commit()
    db.refresh(msg)
    return MessageMinimalSchema(id=msg.id, threadId=msg.thread_id, labelIds=_build_label_ids(msg))


@router.post("/users/{userId}/messages/{messageId}/untrash", response_model=MessageMinimalSchema)
def untrash_message(
    userId: str,
    messageId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    msg = db.query(Message).filter(Message.id == messageId, Message.user_id == _user_id).first()
    if not msg:
        raise HTTPException(404, f"Message {messageId!r} not found")
    msg.is_trash = False
    # Real Gmail does NOT restore INBOX on untrash — only removes TRASH
    record_labels_removed(db, _user_id, messageId, ["TRASH"])
    db.commit()
    db.refresh(msg)
    return MessageMinimalSchema(id=msg.id, threadId=msg.thread_id, labelIds=_build_label_ids(msg))


@router.delete("/users/{userId}/messages/{messageId}")
def delete_message(
    userId: str,
    messageId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    msg = db.query(Message).filter(Message.id == messageId, Message.user_id == _user_id).first()
    if not msg:
        raise HTTPException(404, f"Message {messageId!r} not found")
    record_message_deleted(db, _user_id, messageId)
    # Remove label associations first
    db.query(MessageLabel).filter(MessageLabel.message_id == messageId).delete()
    # Remove any draft pointing to this message
    from claw_gmail.models import Draft
    db.query(Draft).filter(Draft.message_id == messageId).delete()
    db.delete(msg)
    db.commit()
    return {}


@router.post("/users/{userId}/messages/batchModify")
def batch_modify_messages(
    userId: str,
    body: MessageBatchModifyRequest,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    for msg_id in body.ids:
        msg = db.query(Message).filter(Message.id == msg_id, Message.user_id == _user_id).first()
        if msg:
            for lid in body.addLabelIds:
                if lid == "UNREAD":
                    msg.is_read = False
                elif lid == "STARRED":
                    msg.is_starred = True
                elif lid == "TRASH":
                    msg.is_trash = True
                else:
                    existing = db.query(MessageLabel).filter(
                        MessageLabel.message_id == msg_id, MessageLabel.label_id == lid
                    ).first()
                    if not existing:
                        db.add(MessageLabel(message_id=msg_id, label_id=lid))
            for lid in body.removeLabelIds:
                if lid == "UNREAD":
                    msg.is_read = True
                elif lid == "STARRED":
                    msg.is_starred = False
                elif lid == "TRASH":
                    msg.is_trash = False
                else:
                    db.query(MessageLabel).filter(
                        MessageLabel.message_id == msg_id, MessageLabel.label_id == lid
                    ).delete()
            if body.addLabelIds:
                record_labels_added(db, _user_id, msg_id, body.addLabelIds)
            if body.removeLabelIds:
                record_labels_removed(db, _user_id, msg_id, body.removeLabelIds)
    db.commit()
    return {}


@router.post("/users/{userId}/messages/batchDelete")
def batch_delete_messages(
    userId: str,
    body: MessageBatchDeleteRequest,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    from claw_gmail.models import Draft
    for msg_id in body.ids:
        msg = db.query(Message).filter(Message.id == msg_id, Message.user_id == _user_id).first()
        if msg:
            record_message_deleted(db, _user_id, msg_id)
            db.query(MessageLabel).filter(MessageLabel.message_id == msg_id).delete()
            db.query(Draft).filter(Draft.message_id == msg_id).delete()
            db.delete(msg)
    db.commit()
    return {}


def _insert_raw_message(
    db: Session,
    user_id: str,
    raw_b64: str,
    label_ids: list[str] | None = None,
    thread_id: str | None = None,
) -> Message:
    """Insert a raw RFC 2822 message directly into the mailbox (shared by insert and import)."""
    raw_bytes = base64url_decode(raw_b64)
    parsed = parse_rfc2822(raw_bytes)

    user = db.query(User).filter(User.id == user_id).first()
    msg_id = uuid.uuid4().hex[:16]
    now = datetime.utcnow()

    if parsed["date"]:
        internal_date = parsed["date"].replace(tzinfo=None) if parsed["date"].tzinfo else parsed["date"]
    else:
        internal_date = now

    if not thread_id:
        thread_id = uuid.uuid4().hex[:16]
        snippet = parsed["body_plain"][:200]
        db.add(Thread(id=thread_id, user_id=user_id, snippet=snippet))
    else:
        thread = db.query(Thread).filter(Thread.id == thread_id).first()
        if thread:
            thread.snippet = parsed["body_plain"][:200]

    msg = Message(
        id=msg_id,
        thread_id=thread_id,
        user_id=user_id,
        sender=parsed["sender"],
        to=parsed["to"],
        cc=parsed["cc"],
        bcc=parsed["bcc"],
        subject=parsed["subject"],
        snippet=parsed["body_plain"][:200],
        body_plain=parsed["body_plain"],
        body_html=parsed["body_html"],
        internal_date=internal_date,
        is_read=False,
        message_id_header=parsed["message_id"],
        in_reply_to=parsed["in_reply_to"],
        references=parsed["references"],
    )
    db.add(msg)

    # Apply labels
    effective_labels = list(label_ids) if label_ids else []
    for lid in effective_labels:
        if lid in ("UNREAD",):
            msg.is_read = False
        elif lid in ("STARRED",):
            msg.is_starred = True
        elif lid in ("TRASH",):
            msg.is_trash = True
        elif lid in ("SPAM",):
            msg.is_spam = True
        else:
            label = db.query(Label).filter(Label.id == lid, Label.user_id == user_id).first()
            if label:
                db.add(MessageLabel(message_id=msg_id, label_id=lid))

    record_message_added(db, user_id, msg_id)
    if effective_labels:
        record_labels_added(db, user_id, msg_id, effective_labels)
    db.commit()
    db.refresh(msg)
    return msg


@router.post("/users/{userId}/messages", response_model=MessageSchema, response_model_exclude_none=True)
def insert_message(
    userId: str,
    body: MessageInsertRequest,
    internalDateSource: str = Query("receivedTime"),
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    """Insert a message directly into the mailbox (messages.insert)."""
    msg = _insert_raw_message(db, _user_id, body.raw, body.labelIds, body.threadId)
    return _message_to_schema(msg)


@router.post("/users/{userId}/messages/import", response_model=MessageSchema, response_model_exclude_none=True)
def import_message(
    userId: str,
    body: MessageInsertRequest,
    neverMarkSpam: bool = Query(False),
    processForCalendar: bool = Query(False),
    internalDateSource: str = Query("dateHeader"),
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    """Import a message (messages.import). Same as insert for mock purposes."""
    msg = _insert_raw_message(db, _user_id, body.raw, body.labelIds, body.threadId)
    return _message_to_schema(msg)
