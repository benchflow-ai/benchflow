"""Threads API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from claw_gmail.models import Thread, Message, MessageLabel
from .deps import get_db, resolve_user_id
from .history_tracker import (
    record_message_deleted,
    record_labels_added,
    record_labels_removed,
)
from .messages import _message_to_schema, _parse_search_query, _apply_search_filters
from .schemas import (
    ThreadSchema,
    ThreadListResponse,
    ThreadListItem,
    ThreadModifyRequest,
)

router = APIRouter()


@router.get("/users/{userId}/threads")
def list_threads(
    userId: str,
    q: str | None = Query(None),
    maxResults: int = Query(100, ge=1, le=500),
    pageToken: str | None = Query(None),
    labelIds: list[str] | None = Query(None),
    includeSpamTrash: bool = Query(False),
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    query = db.query(Thread).filter(Thread.user_id == _user_id)

    if labelIds:
        # Normalize: handle both repeated params and comma-separated
        normalized = []
        for item in labelIds:
            for lid in item.split(","):
                lid = lid.strip()
                if lid:
                    normalized.append(lid)
        for lid in normalized:
            query = query.filter(
                Thread.messages.any(
                    Message.labels.any(MessageLabel.label_id == lid)
                    if lid not in ("UNREAD", "STARRED", "TRASH", "SPAM", "DRAFT", "SENT")
                    else (
                        Message.is_read == False if lid == "UNREAD"
                        else Message.is_starred == True if lid == "STARRED"
                        else Message.is_trash == True if lid == "TRASH"
                        else Message.is_spam == True if lid == "SPAM"
                        else Message.is_draft == True if lid == "DRAFT"
                        else Message.is_sent == True
                    )
                )
            )

    if not includeSpamTrash:
        query = query.filter(
            Thread.messages.any(
                (Message.is_trash == False) & (Message.is_spam == False)
            )
        )

    if q:
        # Filter threads that contain matching messages
        filters = _parse_search_query(q)
        msg_query = db.query(Message.thread_id).filter(Message.user_id == _user_id)
        msg_query = _apply_search_filters(msg_query, filters, _user_id, db)
        matching_thread_ids = [row[0] for row in msg_query.distinct().all()]
        if matching_thread_ids:
            query = query.filter(Thread.id.in_(matching_thread_ids))
        else:
            query = query.filter(False)  # No matches

    offset = 0
    if pageToken:
        try:
            offset = int(pageToken)
        except ValueError:
            pass

    total = query.count()
    threads = query.offset(offset).limit(maxResults).all()

    next_token = None
    if offset + maxResults < total:
        next_token = str(offset + maxResults)

    # Real Gmail omits nextPageToken when absent, omits threads key when empty (Bug 2/4)
    data: dict = {"resultSizeEstimate": total}
    if threads:
        data["threads"] = [
            {"id": t.id, "snippet": t.snippet, "historyId": str(t.history_id)}
            for t in threads
        ]
    if next_token:
        data["nextPageToken"] = next_token
    return data


@router.get("/users/{userId}/threads/{threadId}")
def get_thread(
    userId: str,
    threadId: str,
    format: str = Query("full"),
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    thread = db.query(Thread).filter(Thread.id == threadId, Thread.user_id == _user_id).first()
    if not thread:
        raise HTTPException(404, f"Thread {threadId!r} not found")

    msgs = (
        db.query(Message)
        .filter(Message.thread_id == threadId, Message.user_id == _user_id)
        .order_by(Message.internal_date.asc())
        .all()
    )

    # Real Gmail threads.get returns {id, historyId, messages} — no snippet (Bug 2)
    return {
        "id": thread.id,
        "historyId": str(thread.history_id),
        "messages": [_message_to_schema(m, format=format) for m in msgs],
    }


@router.post("/users/{userId}/threads/{threadId}/modify", response_model=ThreadSchema, response_model_exclude_none=True)
def modify_thread(
    userId: str,
    threadId: str,
    body: ThreadModifyRequest,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    thread = db.query(Thread).filter(Thread.id == threadId, Thread.user_id == _user_id).first()
    if not thread:
        raise HTTPException(404, f"Thread {threadId!r} not found")

    msgs = db.query(Message).filter(Message.thread_id == threadId, Message.user_id == _user_id).all()
    for msg in msgs:
        for lid in body.addLabelIds:
            if lid == "UNREAD":
                msg.is_read = False
            elif lid == "STARRED":
                msg.is_starred = True
            elif lid == "TRASH":
                msg.is_trash = True
            else:
                existing = db.query(MessageLabel).filter(
                    MessageLabel.message_id == msg.id, MessageLabel.label_id == lid
                ).first()
                if not existing:
                    db.add(MessageLabel(message_id=msg.id, label_id=lid))
        for lid in body.removeLabelIds:
            if lid == "UNREAD":
                msg.is_read = True
            elif lid == "STARRED":
                msg.is_starred = False
            elif lid == "TRASH":
                msg.is_trash = False
            else:
                db.query(MessageLabel).filter(
                    MessageLabel.message_id == msg.id, MessageLabel.label_id == lid
                ).delete()
        if body.addLabelIds:
            record_labels_added(db, _user_id, msg.id, body.addLabelIds)
        if body.removeLabelIds:
            record_labels_removed(db, _user_id, msg.id, body.removeLabelIds)

    db.commit()
    return get_thread(userId, threadId, db=db, _user_id=_user_id)


@router.post("/users/{userId}/threads/{threadId}/trash", response_model=ThreadSchema, response_model_exclude_none=True)
def trash_thread(
    userId: str,
    threadId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    thread = db.query(Thread).filter(Thread.id == threadId, Thread.user_id == _user_id).first()
    if not thread:
        raise HTTPException(404, f"Thread {threadId!r} not found")

    msgs = db.query(Message).filter(Message.thread_id == threadId, Message.user_id == _user_id).all()
    for msg in msgs:
        msg.is_trash = True
        db.query(MessageLabel).filter(
            MessageLabel.message_id == msg.id, MessageLabel.label_id == "INBOX"
        ).delete()
        record_labels_added(db, _user_id, msg.id, ["TRASH"])
        record_labels_removed(db, _user_id, msg.id, ["INBOX"])
    db.commit()
    return get_thread(userId, threadId, db=db, _user_id=_user_id)


@router.post("/users/{userId}/threads/{threadId}/untrash", response_model=ThreadSchema, response_model_exclude_none=True)
def untrash_thread(
    userId: str,
    threadId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    thread = db.query(Thread).filter(Thread.id == threadId, Thread.user_id == _user_id).first()
    if not thread:
        raise HTTPException(404, f"Thread {threadId!r} not found")

    msgs = db.query(Message).filter(Message.thread_id == threadId, Message.user_id == _user_id).all()
    for msg in msgs:
        msg.is_trash = False
        # Real Gmail does NOT restore INBOX on untrash — only removes TRASH
        record_labels_removed(db, _user_id, msg.id, ["TRASH"])
    db.commit()
    return get_thread(userId, threadId, db=db, _user_id=_user_id)


@router.delete("/users/{userId}/threads/{threadId}")
def delete_thread(
    userId: str,
    threadId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    thread = db.query(Thread).filter(Thread.id == threadId, Thread.user_id == _user_id).first()
    if not thread:
        raise HTTPException(404, f"Thread {threadId!r} not found")
    msgs = db.query(Message).filter(Message.thread_id == threadId, Message.user_id == _user_id).all()
    for msg in msgs:
        record_message_deleted(db, _user_id, msg.id)
    db.delete(thread)
    db.commit()
    return {}
