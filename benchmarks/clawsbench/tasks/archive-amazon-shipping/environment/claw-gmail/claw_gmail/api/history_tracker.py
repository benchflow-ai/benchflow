"""Auto-track history records on every mutation, matching real Gmail's history.list behavior."""

from __future__ import annotations

from sqlalchemy.orm import Session

from claw_gmail.models import HistoryRecord, User


def _bump_history(db: Session, user_id: str) -> int:
    """Increment user's history_id and return the new value."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return 1
    user.history_id += 1
    return user.history_id


def record_message_added(db: Session, user_id: str, message_id: str) -> None:
    history_id = _bump_history(db, user_id)
    db.add(HistoryRecord(
        user_id=user_id,
        history_id=history_id,
        event_type="messageAdded",
        message_id=message_id,
    ))


def record_message_deleted(db: Session, user_id: str, message_id: str) -> None:
    history_id = _bump_history(db, user_id)
    db.add(HistoryRecord(
        user_id=user_id,
        history_id=history_id,
        event_type="messageDeleted",
        message_id=message_id,
    ))


def record_labels_added(
    db: Session, user_id: str, message_id: str, label_ids: list[str]
) -> None:
    if not label_ids:
        return
    history_id = _bump_history(db, user_id)
    db.add(HistoryRecord(
        user_id=user_id,
        history_id=history_id,
        event_type="labelAdded",
        message_id=message_id,
        label_ids_added=",".join(label_ids),
    ))


def record_labels_removed(
    db: Session, user_id: str, message_id: str, label_ids: list[str]
) -> None:
    if not label_ids:
        return
    history_id = _bump_history(db, user_id)
    db.add(HistoryRecord(
        user_id=user_id,
        history_id=history_id,
        event_type="labelRemoved",
        message_id=message_id,
        label_ids_removed=",".join(label_ids),
    ))
