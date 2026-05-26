"""History API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from claw_gmail.models import HistoryRecord, User, Message
from .deps import get_db, resolve_user_id
from .messages import _build_label_ids
from .schemas import (
    HistoryListResponse,
    HistoryEntry,
    HistoryMessageAdded,
    HistoryMessageDeleted,
    HistoryLabelAdded,
    HistoryLabelRemoved,
    HistoryMessageItem,
    MessageListItem,
)

router = APIRouter()


@router.get("/users/{userId}/history")
def list_history(
    userId: str,
    startHistoryId: str = Query(...),
    maxResults: int = Query(100, ge=1, le=500),
    pageToken: str | None = Query(None),
    labelId: str | None = Query(None),
    historyTypes: str | None = Query(None),
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    try:
        start_id = int(startHistoryId)
    except ValueError:
        raise HTTPException(400, "Invalid startHistoryId")

    query = db.query(HistoryRecord).filter(
        HistoryRecord.user_id == _user_id,
        HistoryRecord.history_id > start_id,
    )

    if historyTypes:
        types = [t.strip() for t in historyTypes.split(",")]
        type_map = {
            "messageAdded": "messageAdded",
            "messageDeleted": "messageDeleted",
            "labelAdded": "labelAdded",
            "labelRemoved": "labelRemoved",
        }
        allowed = [type_map[t] for t in types if t in type_map]
        if allowed:
            query = query.filter(HistoryRecord.event_type.in_(allowed))

    records = query.order_by(HistoryRecord.history_id.asc()).limit(maxResults).all()

    entries = []
    for rec in records:
        # Look up actual threadId and labelIds from the message
        thread_id = ""
        label_ids_list: list[str] = []
        if rec.message_id:
            msg_obj = db.query(Message).filter(Message.id == rec.message_id).first()
            if msg_obj:
                thread_id = msg_obj.thread_id
                label_ids_list = _build_label_ids(msg_obj)

        msg_ref = {"id": rec.message_id or "", "threadId": thread_id}
        msg_item = {"id": rec.message_id or "", "threadId": thread_id, "labelIds": label_ids_list}

        # Build sparse entry — only include the relevant event key (Bug 2)
        entry: dict = {
            "id": str(rec.history_id),
            "messages": [msg_ref],
        }

        if rec.event_type == "messageAdded":
            entry["messagesAdded"] = [{"message": msg_item}]
        elif rec.event_type == "messageDeleted":
            entry["messagesDeleted"] = [{"message": msg_item}]
        elif rec.event_type == "labelAdded":
            label_ids = [l.strip() for l in rec.label_ids_added.split(",") if l.strip()]
            entry["labelsAdded"] = [{"message": msg_item, "labelIds": label_ids}]
        elif rec.event_type == "labelRemoved":
            label_ids = [l.strip() for l in rec.label_ids_removed.split(",") if l.strip()]
            entry["labelsRemoved"] = [{"message": msg_item, "labelIds": label_ids}]

        entries.append(entry)

    user = db.query(User).filter(User.id == _user_id).first()
    data: dict = {"historyId": str(user.history_id) if user else "1"}
    if entries:
        data["history"] = entries
    return data
