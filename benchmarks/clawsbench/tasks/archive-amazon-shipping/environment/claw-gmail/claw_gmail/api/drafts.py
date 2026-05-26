"""Drafts API routes."""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from claw_gmail.models import Draft, Message, Thread, MessageLabel, Label, User
from .deps import get_db, resolve_user_id
from .history_tracker import (
    record_message_added,
    record_message_deleted,
    record_labels_added,
    record_labels_removed,
)
from .messages import _message_to_schema, _deliver_to_recipients, _build_label_ids
from .mime import base64url_decode, generate_message_id, parse_rfc2822
from .schemas import (
    DraftSchema,
    DraftListResponse,
    DraftListItem,
    DraftCreateRequest,
    MessageListItem,
    MessageSchema,
    MessageMinimalSchema,
)

router = APIRouter()


@router.get("/users/{userId}/drafts")
def list_drafts(
    userId: str,
    q: str | None = Query(None),
    maxResults: int = Query(100, ge=1, le=500),
    pageToken: str | None = Query(None),
    includeSpamTrash: bool = Query(False),
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    from .messages import _parse_search_query, _apply_search_filters

    query = db.query(Draft).join(Message, Draft.message_id == Message.id).filter(Draft.user_id == _user_id)

    if not includeSpamTrash:
        query = query.filter(Message.is_trash == False, Message.is_spam == False)

    if q:
        filters = _parse_search_query(q)
        query = _apply_search_filters(query, filters, _user_id, db)

    offset = 0
    if pageToken:
        try:
            offset = int(pageToken)
        except ValueError:
            pass

    total = query.count()
    drafts = query.offset(offset).limit(maxResults).all()

    next_token = None
    if offset + maxResults < total:
        next_token = str(offset + maxResults)

    items = []
    for d in drafts:
        items.append(DraftListItem(
            id=d.id,
            message=MessageListItem(id=d.message_id, threadId=d.message.thread_id),
        ))

    # Real Gmail: empty list returns {"resultSizeEstimate": 0} with no drafts key (Bug 4).
    # nextPageToken omitted when absent (Bug 2).
    data: dict = {"resultSizeEstimate": total}
    if items:
        data["drafts"] = [item.model_dump() for item in items]
    if next_token:
        data["nextPageToken"] = next_token
    return data


@router.get("/users/{userId}/drafts/{draftId}", response_model=DraftSchema, response_model_exclude_none=True)
def get_draft(
    userId: str,
    draftId: str,
    format: str = Query("full"),
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    draft = db.query(Draft).filter(Draft.id == draftId, Draft.user_id == _user_id).first()
    if not draft:
        raise HTTPException(404, f"Draft {draftId!r} not found")
    return DraftSchema(
        id=draft.id,
        message=_message_to_schema(draft.message, format=format),
    )


@router.post("/users/{userId}/drafts", response_model=DraftSchema, response_model_exclude_none=True, status_code=201)
def create_draft(
    userId: str,
    body: DraftCreateRequest,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    user = db.query(User).filter(User.id == _user_id).first()
    msg_id = uuid.uuid4().hex[:16]
    draft_id = uuid.uuid4().hex[:16]
    thread_id = body.message.threadId or uuid.uuid4().hex[:16]

    # Parse raw RFC 2822 (matches real Gmail API which requires raw field)
    raw_bytes = base64url_decode(body.message.raw)
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

    snippet = body_plain[:200]

    # Create thread if needed
    existing_thread = db.query(Thread).filter(Thread.id == thread_id).first()
    if not existing_thread:
        db.add(Thread(id=thread_id, user_id=_user_id, snippet=snippet))

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
        internal_date=datetime.utcnow(),
        is_draft=True,
        message_id_header=message_id_header,
        in_reply_to=in_reply_to,
        references=references,
    )
    db.add(msg)

    # Add DRAFT label
    label = db.query(Label).filter(Label.id == "DRAFT", Label.user_id == _user_id).first()
    if label:
        db.add(MessageLabel(message_id=msg_id, label_id="DRAFT"))

    draft = Draft(id=draft_id, user_id=_user_id, message_id=msg_id)
    db.add(draft)
    record_message_added(db, _user_id, msg_id)
    record_labels_added(db, _user_id, msg_id, ["DRAFT"])
    db.commit()
    db.refresh(draft)

    return DraftSchema(
        id=draft.id,
        message=MessageMinimalSchema(
            id=draft.message.id,
            threadId=draft.message.thread_id,
            labelIds=_build_label_ids(draft.message),
        ),
    )


@router.put("/users/{userId}/drafts/{draftId}", response_model=DraftSchema, response_model_exclude_none=True)
def update_draft(
    userId: str,
    draftId: str,
    body: DraftCreateRequest,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    draft = db.query(Draft).filter(Draft.id == draftId, Draft.user_id == _user_id).first()
    if not draft:
        raise HTTPException(404, f"Draft {draftId!r} not found")

    msg = draft.message
    raw_bytes = base64url_decode(body.message.raw)
    parsed = parse_rfc2822(raw_bytes)
    msg.to = parsed["to"]
    msg.cc = parsed["cc"]
    msg.bcc = parsed["bcc"]
    msg.subject = parsed["subject"]
    msg.body_plain = parsed["body_plain"]
    msg.body_html = parsed["body_html"]
    msg.message_id_header = parsed["message_id"]
    msg.in_reply_to = parsed["in_reply_to"]
    msg.references = parsed["references"]
    msg.snippet = parsed["body_plain"][:200]

    db.commit()
    db.refresh(draft)
    return DraftSchema(id=draft.id, message=_message_to_schema(draft.message))


@router.post("/users/{userId}/drafts/send", response_model=MessageSchema)
def send_draft(
    userId: str,
    body: dict,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    draft_id = body.get("id")
    if not draft_id:
        raise HTTPException(400, "Draft id required")

    draft = db.query(Draft).filter(Draft.id == draft_id, Draft.user_id == _user_id).first()
    if not draft:
        raise HTTPException(404, f"Draft {draft_id!r} not found")

    msg = draft.message
    msg.is_draft = False
    msg.is_sent = True
    msg.internal_date = datetime.utcnow()

    # Remove DRAFT label, add SENT
    db.query(MessageLabel).filter(
        MessageLabel.message_id == msg.id, MessageLabel.label_id == "DRAFT"
    ).delete()
    existing = db.query(MessageLabel).filter(
        MessageLabel.message_id == msg.id, MessageLabel.label_id == "SENT"
    ).first()
    if not existing:
        db.add(MessageLabel(message_id=msg.id, label_id="SENT"))

    user = db.query(User).filter(User.id == _user_id).first()
    _deliver_to_recipients(db, msg, user)

    record_labels_removed(db, _user_id, msg.id, ["DRAFT"])
    record_labels_added(db, _user_id, msg.id, ["SENT"])

    db.delete(draft)
    db.commit()
    db.refresh(msg)
    return _message_to_schema(msg)


@router.delete("/users/{userId}/drafts/{draftId}")
def delete_draft(
    userId: str,
    draftId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    draft = db.query(Draft).filter(Draft.id == draftId, Draft.user_id == _user_id).first()
    if not draft:
        raise HTTPException(404, f"Draft {draftId!r} not found")
    # Delete the associated message too
    msg = draft.message
    record_message_deleted(db, _user_id, msg.id)
    db.delete(draft)
    db.delete(msg)
    db.commit()
    return {}
