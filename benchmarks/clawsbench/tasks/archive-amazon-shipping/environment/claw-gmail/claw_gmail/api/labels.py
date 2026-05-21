"""Labels API routes."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from sqlalchemy import func

from claw_gmail.models import Label, LabelType, MessageLabel, Message
from .deps import get_db, resolve_user_id
from .schemas import LabelSchema, LabelListResponse, LabelCreateRequest, LabelUpdateRequest, LabelColor

router = APIRouter()


def _label_to_schema(label: Label, db: Session, include_counts: bool = False) -> LabelSchema:
    # Real Gmail: system labels in list response omit counts; counts only on labels.get
    if include_counts:
        msg_total = db.query(MessageLabel).filter(MessageLabel.label_id == label.id).count()
        msg_unread = (
            db.query(MessageLabel)
            .join(Message)
            .filter(MessageLabel.label_id == label.id, Message.is_read == False)
            .count()
        )
        # Compute thread counts dynamically from messages with this label
        threads_total = (
            db.query(func.count(func.distinct(Message.thread_id)))
            .join(MessageLabel)
            .filter(MessageLabel.label_id == label.id)
            .scalar()
        ) or 0
        threads_unread = (
            db.query(func.count(func.distinct(Message.thread_id)))
            .join(MessageLabel)
            .filter(MessageLabel.label_id == label.id, Message.is_read == False)
            .scalar()
        ) or 0
    else:
        msg_total = None
        msg_unread = None
        threads_total = None
        threads_unread = None

    return LabelSchema(
        id=label.id,
        name=label.name,
        type=label.type,
        messageListVisibility=label.message_list_visibility,
        labelListVisibility=label.label_list_visibility,
        messagesTotal=msg_total,
        messagesUnread=msg_unread,
        threadsTotal=threads_total,
        threadsUnread=threads_unread,
        color=LabelColor(backgroundColor=label.color_bg, textColor=label.color_text)
        if label.color_bg or label.color_text
        else None,
    )


def _label_mutation_response(label: Label) -> dict:
    """Build sparse response for label create/update (Bug 1).

    Real Gmail returns only {id, name, messageListVisibility, labelListVisibility}.
    """
    return {
        "id": label.id,
        "name": label.name,
        "messageListVisibility": label.message_list_visibility,
        "labelListVisibility": label.label_list_visibility,
    }


@router.get("/users/{userId}/labels", response_model=LabelListResponse, response_model_exclude_none=True)
def list_labels(
    userId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    labels = db.query(Label).filter(Label.user_id == _user_id).all()
    return LabelListResponse(labels=[_label_to_schema(l, db) for l in labels])


@router.get("/users/{userId}/labels/{labelId}", response_model=LabelSchema, response_model_exclude_none=True)
def get_label(
    userId: str,
    labelId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    label = db.query(Label).filter(Label.id == labelId, Label.user_id == _user_id).first()
    if not label:
        raise HTTPException(404, f"Label {labelId!r} not found")
    return _label_to_schema(label, db, include_counts=True)


@router.post("/users/{userId}/labels", status_code=201)
def create_label(
    userId: str,
    body: LabelCreateRequest,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    label_id = f"Label_{uuid.uuid4().hex[:8]}"
    label = Label(
        id=label_id,
        user_id=_user_id,
        name=body.name,
        type=LabelType.user,
        message_list_visibility=body.messageListVisibility,
        label_list_visibility=body.labelListVisibility,
        color_bg=body.color.backgroundColor if body.color else None,
        color_text=body.color.textColor if body.color else None,
    )
    db.add(label)
    db.commit()
    db.refresh(label)
    # Real Gmail create returns only {id, name, messageListVisibility, labelListVisibility} (Bug 1)
    return _label_mutation_response(label)


@router.put("/users/{userId}/labels/{labelId}")
def update_label(
    userId: str,
    labelId: str,
    body: LabelUpdateRequest,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    label = db.query(Label).filter(Label.id == labelId, Label.user_id == _user_id).first()
    if not label:
        raise HTTPException(404, f"Label {labelId!r} not found")
    if label.type == LabelType.system.value:
        raise HTTPException(400, "Cannot modify system labels")

    if body.name is not None:
        label.name = body.name
    if body.messageListVisibility is not None:
        label.message_list_visibility = body.messageListVisibility
    if body.labelListVisibility is not None:
        label.label_list_visibility = body.labelListVisibility
    if body.color:
        if body.color.backgroundColor is not None:
            label.color_bg = body.color.backgroundColor
        if body.color.textColor is not None:
            label.color_text = body.color.textColor
    db.commit()
    db.refresh(label)
    # Real Gmail update returns only {id, name, messageListVisibility, labelListVisibility} (Bug 1)
    return _label_mutation_response(label)


@router.patch("/users/{userId}/labels/{labelId}")
def patch_label(
    userId: str,
    labelId: str,
    body: LabelUpdateRequest,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    label = db.query(Label).filter(Label.id == labelId, Label.user_id == _user_id).first()
    if not label:
        raise HTTPException(404, f"Label {labelId!r} not found")
    if label.type == LabelType.system.value:
        raise HTTPException(400, "Cannot modify system labels")

    if body.name is not None:
        label.name = body.name
    if body.messageListVisibility is not None:
        label.message_list_visibility = body.messageListVisibility
    if body.labelListVisibility is not None:
        label.label_list_visibility = body.labelListVisibility
    if body.color:
        if body.color.backgroundColor is not None:
            label.color_bg = body.color.backgroundColor
        if body.color.textColor is not None:
            label.color_text = body.color.textColor
    db.commit()
    db.refresh(label)
    # Real Gmail patch returns full label with counts (like labels.get)
    return _label_to_schema(label, db, include_counts=True)


@router.delete("/users/{userId}/labels/{labelId}")
def delete_label(
    userId: str,
    labelId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    label = db.query(Label).filter(Label.id == labelId, Label.user_id == _user_id).first()
    if not label:
        raise HTTPException(404, f"Label {labelId!r} not found")
    if label.type == LabelType.system.value:
        raise HTTPException(400, "Cannot delete system labels")
    db.delete(label)
    db.commit()
    return {}
