"""State snapshots, reset, and diff functionality."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from datetime import datetime

from sqlalchemy.orm import Session

from claw_gmail.models import (
    get_session_factory,
    User,
    Message,
    Thread,
    Label,
    MessageLabel,
    Draft,
    Attachment,
    Contact,
    Filter,
    SendAs,
    ForwardingAddress,
    Delegate,
    VacationSettings,
    AutoForwarding,
)

SNAPSHOTS_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent.parent / ".data" / "snapshots"


def _serialize_messages(db: Session, user_id: str) -> list[dict]:
    """Serialize all messages for a user."""
    messages = db.query(Message).filter(Message.user_id == user_id).all()
    result = []
    for m in messages:
        label_ids = [ml.label_id for ml in m.labels]
        result.append({
            "id": m.id,
            "threadId": m.thread_id,
            "sender": m.sender,
            "to": m.to,
            "cc": m.cc,
            "subject": m.subject,
            "snippet": m.snippet,
            "body": m.body_plain,
            "bodyHtml": m.body_html or "",
            "internalDate": m.internal_date.isoformat() if m.internal_date else None,
            "isRead": m.is_read,
            "isStarred": m.is_starred,
            "isTrash": m.is_trash,
            "isSpam": m.is_spam,
            "isDraft": m.is_draft,
            "isSent": m.is_sent,
            "labelIds": label_ids,
            "messageIdHeader": m.message_id_header or "",
            "inReplyTo": m.in_reply_to or "",
            "references": m.references or "",
        })
    return result


def _serialize_user(db: Session, user: User) -> dict:
    """Serialize full user state."""
    messages = _serialize_messages(db, user.id)
    threads = db.query(Thread).filter(Thread.user_id == user.id).all()
    labels = db.query(Label).filter(Label.user_id == user.id).all()
    drafts = db.query(Draft).filter(Draft.user_id == user.id).all()
    contacts = db.query(Contact).filter(Contact.user_id == user.id).all()
    send_as = db.query(SendAs).filter(SendAs.user_id == user.id).all()
    fwd_addrs = db.query(ForwardingAddress).filter(ForwardingAddress.user_id == user.id).all()
    delegates = db.query(Delegate).filter(Delegate.user_id == user.id).all()
    filters = db.query(Filter).filter(Filter.user_id == user.id).all()
    vacation = db.query(VacationSettings).filter(VacationSettings.user_id == user.id).first()
    auto_fwd = db.query(AutoForwarding).filter(AutoForwarding.user_id == user.id).first()

    result = {
        "user": {
            "id": user.id,
            "email": user.email_address,
            "displayName": user.display_name,
            "historyId": user.history_id,
        },
        "messages": messages,
        "threads": [
            {"id": t.id, "snippet": t.snippet, "historyId": t.history_id}
            for t in threads
        ],
        "labels": [
            {"id": l.id, "name": l.name, "type": l.type}
            for l in labels
        ],
        "drafts": [
            {"id": d.id, "messageId": d.message_id}
            for d in drafts
        ],
        "filters": [
            {
                "id": f.id,
                "criteria": {
                    "from": f.criteria_from,
                    "to": f.criteria_to,
                    "subject": f.criteria_subject,
                    "query": f.criteria_query,
                    "hasAttachment": f.criteria_has_attachment,
                    "negatedQuery": f.criteria_negated_query,
                    "excludeChats": f.criteria_exclude_chats,
                    "size": f.criteria_size,
                    "sizeComparison": f.criteria_size_comparison,
                },
                "action": {
                    "addLabelIds": [x for x in f.action_add_label_ids.split(",") if x] if f.action_add_label_ids else [],
                    "removeLabelIds": [x for x in f.action_remove_label_ids.split(",") if x] if f.action_remove_label_ids else [],
                    "forward": f.action_forward,
                },
            }
            for f in filters
        ],
        "contacts": [
            {"id": c.id, "name": c.name, "email": c.email}
            for c in contacts
        ],
        "sendAs": [
            {
                "sendAsEmail": sa.send_as_email,
                "displayName": sa.display_name,
                "replyToAddress": sa.reply_to_address,
                "signature": sa.signature,
                "isPrimary": sa.is_primary,
                "isDefault": sa.is_default,
                "treatAsAlias": sa.treat_as_alias,
                "verificationStatus": sa.verification_status,
            }
            for sa in send_as
        ],
        "forwardingAddresses": [
            {"forwardingEmail": fa.forwarding_email, "verificationStatus": fa.verification_status}
            for fa in fwd_addrs
        ],
        "delegates": [
            {"delegateEmail": d.delegate_email, "verificationStatus": d.verification_status}
            for d in delegates
        ],
    }

    if vacation:
        result["vacationSettings"] = {
            "enableAutoReply": vacation.enable_auto_reply,
            "responseSubject": vacation.response_subject,
            "responseBodyHtml": vacation.response_body_html,
            "responseBodyPlainText": vacation.response_body_plain_text,
            "restrictToContacts": vacation.restrict_to_contacts,
            "restrictToDomain": vacation.restrict_to_domain,
            "startTime": vacation.start_time,
            "endTime": vacation.end_time,
        }

    if auto_fwd:
        result["autoForwarding"] = {
            "enabled": auto_fwd.enabled,
            "emailAddress": auto_fwd.email_address,
            "disposition": auto_fwd.disposition,
        }

    return result


def get_state_dump() -> dict:
    """Get full state dump for all users."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        users = db.query(User).all()
        return {
            "users": {u.id: _serialize_user(db, u) for u in users},
            "timestamp": datetime.utcnow().isoformat(),
        }
    finally:
        db.close()


def take_snapshot(name: str) -> Path:
    """Save current state to a JSON snapshot."""
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    state = get_state_dump()
    path = SNAPSHOTS_DIR / f"{name}.json"
    path.write_text(json.dumps(state, indent=2))
    return path


def restore_snapshot(name: str) -> bool:
    """Restore DB from a snapshot. Returns True if successful."""
    path = SNAPSHOTS_DIR / f"{name}.json"
    if not path.exists():
        return False

    state = json.loads(path.read_text())
    _restore_from_state(state)
    return True


def _restore_from_state(state: dict):
    """Rebuild DB from a state dict."""
    from claw_gmail.models import Base, get_engine
    engine = get_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        for user_id, user_data in state.get("users", {}).items():
            u = user_data["user"]
            db.add(User(
                id=u["id"],
                email_address=u["email"],
                display_name=u["displayName"],
                history_id=u.get("historyId", 1),
            ))

            for l in user_data.get("labels", []):
                db.add(Label(
                    id=l["id"],
                    user_id=user_id,
                    name=l["name"],
                    type=l.get("type", "user"),
                ))

            for t in user_data.get("threads", []):
                db.add(Thread(
                    id=t["id"],
                    user_id=user_id,
                    snippet=t.get("snippet", ""),
                    history_id=t.get("historyId", 1),
                ))

            for m in user_data.get("messages", []):
                internal_date = None
                if m.get("internalDate"):
                    try:
                        internal_date = datetime.fromisoformat(m["internalDate"])
                    except (ValueError, TypeError):
                        internal_date = datetime.utcnow()

                db.add(Message(
                    id=m["id"],
                    thread_id=m["threadId"],
                    user_id=user_id,
                    sender=m.get("sender", ""),
                    to=m.get("to", ""),
                    cc=m.get("cc", ""),
                    subject=m.get("subject", ""),
                    snippet=m.get("snippet", ""),
                    body_plain=m.get("body", ""),
                    body_html=m.get("bodyHtml", ""),
                    internal_date=internal_date or datetime.utcnow(),
                    is_read=m.get("isRead", False),
                    is_starred=m.get("isStarred", False),
                    is_trash=m.get("isTrash", False),
                    is_spam=m.get("isSpam", False),
                    is_draft=m.get("isDraft", False),
                    is_sent=m.get("isSent", False),
                    message_id_header=m.get("messageIdHeader", ""),
                    in_reply_to=m.get("inReplyTo", ""),
                    references=m.get("references", ""),
                ))

                for lid in m.get("labelIds", []):
                    # UNREAD and STARRED are purely derived from boolean
                    # fields and never stored as MessageLabel rows.
                    # All other labels (INBOX, SENT, DRAFT, SPAM, TRASH,
                    # CATEGORY_*, user labels) are stored as MessageLabel
                    # and must be restored to keep snapshot↔state consistent.
                    if lid not in ("UNREAD", "STARRED"):
                        db.add(MessageLabel(message_id=m["id"], label_id=lid))

            for d in user_data.get("drafts", []):
                db.add(Draft(id=d["id"], user_id=user_id, message_id=d["messageId"]))

            for c in user_data.get("contacts", []):
                db.add(Contact(id=c["id"], user_id=user_id, name=c["name"], email=c["email"]))

            # Restore settings
            for sa in user_data.get("sendAs", []):
                db.add(SendAs(
                    user_id=user_id,
                    send_as_email=sa["sendAsEmail"],
                    display_name=sa.get("displayName", ""),
                    reply_to_address=sa.get("replyToAddress", ""),
                    signature=sa.get("signature", ""),
                    is_primary=sa.get("isPrimary", False),
                    is_default=sa.get("isDefault", False),
                    treat_as_alias=sa.get("treatAsAlias", False),
                    verification_status=sa.get("verificationStatus", "accepted"),
                ))
            for fa in user_data.get("forwardingAddresses", []):
                db.add(ForwardingAddress(
                    user_id=user_id,
                    forwarding_email=fa["forwardingEmail"],
                    verification_status=fa.get("verificationStatus", "accepted"),
                ))
            for d in user_data.get("delegates", []):
                db.add(Delegate(
                    user_id=user_id,
                    delegate_email=d["delegateEmail"],
                    verification_status=d.get("verificationStatus", "accepted"),
                ))
            vs_data = user_data.get("vacationSettings")
            if vs_data:
                db.add(VacationSettings(
                    user_id=user_id,
                    enable_auto_reply=vs_data.get("enableAutoReply", False),
                    response_subject=vs_data.get("responseSubject", ""),
                    response_body_html=vs_data.get("responseBodyHtml", ""),
                    response_body_plain_text=vs_data.get("responseBodyPlainText", ""),
                    restrict_to_contacts=vs_data.get("restrictToContacts", False),
                    restrict_to_domain=vs_data.get("restrictToDomain", False),
                    start_time=vs_data.get("startTime"),
                    end_time=vs_data.get("endTime"),
                ))
            af_data = user_data.get("autoForwarding")
            if af_data:
                db.add(AutoForwarding(
                    user_id=user_id,
                    enabled=af_data.get("enabled", False),
                    email_address=af_data.get("emailAddress", ""),
                    disposition=af_data.get("disposition", "leaveInInbox"),
                ))

        db.commit()
    finally:
        db.close()


def get_diff(initial_state: dict | None = None) -> dict:
    """Compute diff between initial state (snapshot) and current state."""
    current = get_state_dump()

    if initial_state is None:
        # Try loading the 'initial' snapshot
        path = SNAPSHOTS_DIR / "initial.json"
        if path.exists():
            initial_state = json.loads(path.read_text())
        else:
            return {"error": "No initial snapshot found"}

    diff = {"added": {}, "updated": {}, "deleted": {}}

    for user_id, current_user in current.get("users", {}).items():
        initial_user = initial_state.get("users", {}).get(user_id)
        if not initial_user:
            diff["added"][user_id] = current_user
            continue

        # Compare messages
        curr_msgs = {m["id"]: m for m in current_user.get("messages", [])}
        init_msgs = {m["id"]: m for m in initial_user.get("messages", [])}

        user_diff = {"messages": {"added": [], "updated": [], "deleted": []}}

        for mid, msg in curr_msgs.items():
            if mid not in init_msgs:
                user_diff["messages"]["added"].append(msg)
            elif msg != init_msgs[mid]:
                changes = {}
                for k, v in msg.items():
                    if init_msgs[mid].get(k) != v:
                        changes[k] = v
                # Compute labelsAdded / labelsRemoved when labelIds changed
                if "labelIds" in changes:
                    old_labels = set(init_msgs[mid].get("labelIds", []))
                    new_labels = set(msg.get("labelIds", []))
                    added_labels = sorted(new_labels - old_labels)
                    removed_labels = sorted(old_labels - new_labels)
                    if added_labels:
                        changes["labelsAdded"] = added_labels
                    if removed_labels:
                        changes["labelsRemoved"] = removed_labels
                user_diff["messages"]["updated"].append({"id": mid, **changes})

        for mid in init_msgs:
            if mid not in curr_msgs:
                user_diff["messages"]["deleted"].append(init_msgs[mid])

        # Compare labels
        curr_labels = {l["id"]: l for l in current_user.get("labels", [])}
        init_labels = {l["id"]: l for l in initial_user.get("labels", [])}

        labels_diff = {"added": [], "removed": []}
        for lid, label in curr_labels.items():
            if lid not in init_labels:
                labels_diff["added"].append(label)
        for lid in init_labels:
            if lid not in curr_labels:
                labels_diff["removed"].append(init_labels[lid])

        if any(labels_diff.values()):
            user_diff["labels"] = labels_diff

        if any(user_diff["messages"].values()) or "labels" in user_diff:
            diff["updated"][user_id] = user_diff

    return diff
