"""Settings API routes — filters, sendAs, forwarding, delegates, vacation, autoForwarding, watch/stop."""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from claw_gmail.models import (
    Filter,
    User,
    SendAs,
    ForwardingAddress,
    Delegate,
    VacationSettings,
    AutoForwarding,
)
from .deps import get_db, resolve_user_id
from .schemas import (
    FilterSchema,
    FilterListResponse,
    FilterCriteria,
    FilterAction,
    FilterCreateRequest,
    SendAsSchema,
    SendAsListResponse,
    SendAsCreateRequest,
    SendAsUpdateRequest,
    ForwardingAddressSchema,
    ForwardingAddressListResponse,
    ForwardingAddressCreateRequest,
    DelegateSchema,
    DelegateListResponse,
    DelegateCreateRequest,
    VacationSettingsSchema,
    VacationUpdateRequest,
    AutoForwardingSchema,
    AutoForwardingUpdateRequest,
    ImapSettingsSchema,
    ImapSettingsUpdateRequest,
    PopSettingsSchema,
    PopSettingsUpdateRequest,
    LanguageSettingsSchema,
    LanguageSettingsUpdateRequest,
    WatchRequest,
    WatchResponse,
)

router = APIRouter()


# ============================================================
# Filters
# ============================================================

def _filter_to_schema(f: Filter) -> FilterSchema:
    criteria = FilterCriteria.model_construct(
        **{
            "from": f.criteria_from or None,
            "to": f.criteria_to or None,
            "subject": f.criteria_subject or None,
            "query": f.criteria_query or None,
            "hasAttachment": f.criteria_has_attachment,
            "negatedQuery": f.criteria_negated_query or None,
            "excludeChats": f.criteria_exclude_chats,
            "size": f.criteria_size,
            "sizeComparison": f.criteria_size_comparison or None,
        }
    )
    add_ids = [x.strip() for x in f.action_add_label_ids.split(",") if x.strip()] if f.action_add_label_ids else []
    remove_ids = [x.strip() for x in f.action_remove_label_ids.split(",") if x.strip()] if f.action_remove_label_ids else []
    action = FilterAction(
        addLabelIds=add_ids,
        removeLabelIds=remove_ids,
        forward=f.action_forward or None,
    )
    return FilterSchema(id=f.id, criteria=criteria, action=action)


@router.get("/users/{userId}/settings/filters")
def list_filters(
    userId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    filters = db.query(Filter).filter(Filter.user_id == _user_id).all()
    if not filters:
        return JSONResponse(content={})
    return FilterListResponse(filter=[_filter_to_schema(f) for f in filters])


@router.get("/users/{userId}/settings/filters/{filterId}", response_model=FilterSchema)
def get_filter(
    userId: str,
    filterId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    f = db.query(Filter).filter(Filter.id == filterId, Filter.user_id == _user_id).first()
    if not f:
        raise HTTPException(404, f"Filter {filterId!r} not found")
    return _filter_to_schema(f)


@router.post("/users/{userId}/settings/filters", response_model=FilterSchema, status_code=201)
def create_filter(
    userId: str,
    body: FilterCreateRequest,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    filter_id = uuid.uuid4().hex[:16]
    c = body.criteria or FilterCriteria.model_construct()
    a = body.action or FilterAction()

    f = Filter(
        id=filter_id,
        user_id=_user_id,
        criteria_from=getattr(c, "from_", None) or "",
        criteria_to=c.to or "",
        criteria_subject=c.subject or "",
        criteria_query=c.query or "",
        criteria_has_attachment=c.hasAttachment if c.hasAttachment else False,
        criteria_negated_query=c.negatedQuery or "",
        criteria_exclude_chats=c.excludeChats if c.excludeChats else False,
        criteria_size=c.size,
        criteria_size_comparison=c.sizeComparison or "",
        action_add_label_ids=",".join(a.addLabelIds),
        action_remove_label_ids=",".join(a.removeLabelIds),
        action_forward=a.forward or "",
    )
    db.add(f)
    db.commit()
    db.refresh(f)
    return _filter_to_schema(f)


@router.delete("/users/{userId}/settings/filters/{filterId}")
def delete_filter(
    userId: str,
    filterId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    f = db.query(Filter).filter(Filter.id == filterId, Filter.user_id == _user_id).first()
    if not f:
        raise HTTPException(404, f"Filter {filterId!r} not found")
    db.delete(f)
    db.commit()
    return {}


# ============================================================
# SendAs
# ============================================================

def _send_as_to_schema(sa: SendAs) -> dict:
    """Build sendAs response dict.

    Real Gmail primary sendAs omits treatAsAlias and verificationStatus (Bug 5).
    """
    data = {
        "sendAsEmail": sa.send_as_email,
        "displayName": sa.display_name,
        "replyToAddress": sa.reply_to_address,
        "signature": sa.signature,
        "isPrimary": sa.is_primary,
        "isDefault": sa.is_default,
    }
    if not sa.is_primary:
        data["treatAsAlias"] = sa.treat_as_alias
        data["verificationStatus"] = sa.verification_status
    return data


@router.get("/users/{userId}/settings/sendAs")
def list_send_as(
    userId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    entries = db.query(SendAs).filter(SendAs.user_id == _user_id).all()
    return {"sendAs": [_send_as_to_schema(e) for e in entries]}


@router.get("/users/{userId}/settings/sendAs/{sendAsEmail}", response_model=SendAsSchema)
def get_send_as(
    userId: str,
    sendAsEmail: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    sa = db.query(SendAs).filter(SendAs.user_id == _user_id, SendAs.send_as_email == sendAsEmail).first()
    if not sa:
        raise HTTPException(404, f"SendAs {sendAsEmail!r} not found")
    return _send_as_to_schema(sa)


@router.post("/users/{userId}/settings/sendAs", response_model=SendAsSchema, status_code=201)
def create_send_as(
    userId: str,
    body: SendAsCreateRequest,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    existing = db.query(SendAs).filter(
        SendAs.user_id == _user_id, SendAs.send_as_email == body.sendAsEmail
    ).first()
    if existing:
        raise HTTPException(409, f"SendAs {body.sendAsEmail!r} already exists")

    sa = SendAs(
        user_id=_user_id,
        send_as_email=body.sendAsEmail,
        display_name=body.displayName,
        reply_to_address=body.replyToAddress,
        signature=body.signature,
        treat_as_alias=body.treatAsAlias,
        is_default=body.isDefault,
        verification_status="accepted",
    )
    db.add(sa)
    db.commit()
    db.refresh(sa)
    return _send_as_to_schema(sa)


@router.put("/users/{userId}/settings/sendAs/{sendAsEmail}", response_model=SendAsSchema)
def update_send_as(
    userId: str,
    sendAsEmail: str,
    body: SendAsUpdateRequest,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    sa = db.query(SendAs).filter(SendAs.user_id == _user_id, SendAs.send_as_email == sendAsEmail).first()
    if not sa:
        raise HTTPException(404, f"SendAs {sendAsEmail!r} not found")
    if sa.is_primary:
        raise HTTPException(400, "Cannot modify primary sendAs")

    if body.displayName is not None:
        sa.display_name = body.displayName
    if body.replyToAddress is not None:
        sa.reply_to_address = body.replyToAddress
    if body.signature is not None:
        sa.signature = body.signature
    if body.treatAsAlias is not None:
        sa.treat_as_alias = body.treatAsAlias
    if body.isDefault is not None:
        sa.is_default = body.isDefault

    db.commit()
    db.refresh(sa)
    return _send_as_to_schema(sa)


@router.delete("/users/{userId}/settings/sendAs/{sendAsEmail}")
def delete_send_as(
    userId: str,
    sendAsEmail: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    sa = db.query(SendAs).filter(SendAs.user_id == _user_id, SendAs.send_as_email == sendAsEmail).first()
    if not sa:
        raise HTTPException(404, f"SendAs {sendAsEmail!r} not found")
    if sa.is_primary:
        raise HTTPException(400, "Cannot delete primary sendAs")
    db.delete(sa)
    db.commit()
    return {}


@router.post("/users/{userId}/settings/sendAs/{sendAsEmail}/verify")
def verify_send_as(
    userId: str,
    sendAsEmail: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    sa = db.query(SendAs).filter(SendAs.user_id == _user_id, SendAs.send_as_email == sendAsEmail).first()
    if not sa:
        raise HTTPException(404, f"SendAs {sendAsEmail!r} not found")
    sa.verification_status = "accepted"
    db.commit()
    return {}


# ============================================================
# ForwardingAddresses
# ============================================================

@router.get("/users/{userId}/settings/forwardingAddresses")
def list_forwarding_addresses(
    userId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    addrs = db.query(ForwardingAddress).filter(ForwardingAddress.user_id == _user_id).all()
    if not addrs:
        return JSONResponse(content={})
    return ForwardingAddressListResponse(
        forwardingAddresses=[
            ForwardingAddressSchema(forwardingEmail=a.forwarding_email, verificationStatus=a.verification_status)
            for a in addrs
        ]
    )


@router.get("/users/{userId}/settings/forwardingAddresses/{forwardingEmail}", response_model=ForwardingAddressSchema)
def get_forwarding_address(
    userId: str,
    forwardingEmail: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    addr = db.query(ForwardingAddress).filter(
        ForwardingAddress.user_id == _user_id, ForwardingAddress.forwarding_email == forwardingEmail
    ).first()
    if not addr:
        raise HTTPException(404, f"ForwardingAddress {forwardingEmail!r} not found")
    return ForwardingAddressSchema(forwardingEmail=addr.forwarding_email, verificationStatus=addr.verification_status)


@router.post("/users/{userId}/settings/forwardingAddresses", response_model=ForwardingAddressSchema, status_code=201)
def create_forwarding_address(
    userId: str,
    body: ForwardingAddressCreateRequest,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    existing = db.query(ForwardingAddress).filter(
        ForwardingAddress.user_id == _user_id, ForwardingAddress.forwarding_email == body.forwardingEmail
    ).first()
    if existing:
        raise HTTPException(409, f"ForwardingAddress {body.forwardingEmail!r} already exists")

    addr = ForwardingAddress(
        user_id=_user_id,
        forwarding_email=body.forwardingEmail,
        verification_status="accepted",
    )
    db.add(addr)
    db.commit()
    db.refresh(addr)
    return ForwardingAddressSchema(forwardingEmail=addr.forwarding_email, verificationStatus=addr.verification_status)


@router.delete("/users/{userId}/settings/forwardingAddresses/{forwardingEmail}")
def delete_forwarding_address(
    userId: str,
    forwardingEmail: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    addr = db.query(ForwardingAddress).filter(
        ForwardingAddress.user_id == _user_id, ForwardingAddress.forwarding_email == forwardingEmail
    ).first()
    if not addr:
        raise HTTPException(404, f"ForwardingAddress {forwardingEmail!r} not found")
    db.delete(addr)
    db.commit()
    return {}


# ============================================================
# Delegates
# ============================================================

@router.get("/users/{userId}/settings/delegates")
def list_delegates(
    userId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    delegates = db.query(Delegate).filter(Delegate.user_id == _user_id).all()
    if not delegates:
        return JSONResponse(content={})
    return DelegateListResponse(
        delegates=[
            DelegateSchema(delegateEmail=d.delegate_email, verificationStatus=d.verification_status)
            for d in delegates
        ]
    )


@router.get("/users/{userId}/settings/delegates/{delegateEmail}", response_model=DelegateSchema)
def get_delegate(
    userId: str,
    delegateEmail: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    d = db.query(Delegate).filter(
        Delegate.user_id == _user_id, Delegate.delegate_email == delegateEmail
    ).first()
    if not d:
        raise HTTPException(404, f"Delegate {delegateEmail!r} not found")
    return DelegateSchema(delegateEmail=d.delegate_email, verificationStatus=d.verification_status)


@router.post("/users/{userId}/settings/delegates", response_model=DelegateSchema, status_code=201)
def create_delegate(
    userId: str,
    body: DelegateCreateRequest,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    existing = db.query(Delegate).filter(
        Delegate.user_id == _user_id, Delegate.delegate_email == body.delegateEmail
    ).first()
    if existing:
        raise HTTPException(409, f"Delegate {body.delegateEmail!r} already exists")

    d = Delegate(
        user_id=_user_id,
        delegate_email=body.delegateEmail,
        verification_status="accepted",
    )
    db.add(d)
    db.commit()
    db.refresh(d)
    return DelegateSchema(delegateEmail=d.delegate_email, verificationStatus=d.verification_status)


@router.delete("/users/{userId}/settings/delegates/{delegateEmail}")
def delete_delegate(
    userId: str,
    delegateEmail: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    d = db.query(Delegate).filter(
        Delegate.user_id == _user_id, Delegate.delegate_email == delegateEmail
    ).first()
    if not d:
        raise HTTPException(404, f"Delegate {delegateEmail!r} not found")
    db.delete(d)
    db.commit()
    return {}


# ============================================================
# Vacation Settings
# ============================================================

def _get_or_create_vacation(db: Session, user_id: str) -> VacationSettings:
    vs = db.query(VacationSettings).filter(VacationSettings.user_id == user_id).first()
    if not vs:
        vs = VacationSettings(user_id=user_id)
        db.add(vs)
        db.flush()
    return vs


def _vacation_to_schema(vs: VacationSettings) -> dict:
    """Build vacation response, omitting fields at default values (Bug 5).

    Real Gmail returns only {enableAutoReply, responseSubject, restrictToContacts}
    when vacation is disabled / at defaults.  Fields like responseBodyHtml,
    responseBodyPlainText, and restrictToDomain are omitted unless explicitly set.
    """
    data: dict = {
        "enableAutoReply": vs.enable_auto_reply,
        "responseSubject": vs.response_subject,
        "restrictToContacts": vs.restrict_to_contacts,
    }
    # Only include these when they differ from the unset default
    if vs.response_body_html:
        data["responseBodyHtml"] = vs.response_body_html
    if vs.response_body_plain_text:
        data["responseBodyPlainText"] = vs.response_body_plain_text
    if vs.restrict_to_domain:
        data["restrictToDomain"] = vs.restrict_to_domain
    if vs.start_time is not None:
        data["startTime"] = str(vs.start_time)
    if vs.end_time is not None:
        data["endTime"] = str(vs.end_time)
    return data


@router.get("/users/{userId}/settings/vacation")
def get_vacation(
    userId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    vs = _get_or_create_vacation(db, _user_id)
    db.commit()
    return _vacation_to_schema(vs)


@router.put("/users/{userId}/settings/vacation", response_model=VacationSettingsSchema)
def update_vacation(
    userId: str,
    body: VacationUpdateRequest,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    vs = _get_or_create_vacation(db, _user_id)
    if body.enableAutoReply is not None:
        vs.enable_auto_reply = body.enableAutoReply
    if body.responseSubject is not None:
        vs.response_subject = body.responseSubject
    if body.responseBodyHtml is not None:
        vs.response_body_html = body.responseBodyHtml
    if body.responseBodyPlainText is not None:
        vs.response_body_plain_text = body.responseBodyPlainText
    if body.restrictToContacts is not None:
        vs.restrict_to_contacts = body.restrictToContacts
    if body.restrictToDomain is not None:
        vs.restrict_to_domain = body.restrictToDomain
    if body.startTime is not None:
        vs.start_time = int(body.startTime)
    if body.endTime is not None:
        vs.end_time = int(body.endTime)
    db.commit()
    db.refresh(vs)
    return _vacation_to_schema(vs)


# ============================================================
# Auto-Forwarding
# ============================================================

def _get_or_create_auto_forwarding(db: Session, user_id: str) -> AutoForwarding:
    af = db.query(AutoForwarding).filter(AutoForwarding.user_id == user_id).first()
    if not af:
        af = AutoForwarding(user_id=user_id)
        db.add(af)
        db.flush()
    return af


@router.get("/users/{userId}/settings/autoForwarding")
def get_auto_forwarding(
    userId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    af = _get_or_create_auto_forwarding(db, _user_id)
    db.commit()
    # Real Gmail returns only {"enabled": false} at defaults (Bug 5).
    # emailAddress and disposition only appear when forwarding is configured.
    data: dict = {"enabled": af.enabled}
    if af.enabled:
        data["emailAddress"] = af.email_address
        data["disposition"] = af.disposition
    return data


@router.put("/users/{userId}/settings/autoForwarding", response_model=AutoForwardingSchema)
def update_auto_forwarding(
    userId: str,
    body: AutoForwardingUpdateRequest,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    af = _get_or_create_auto_forwarding(db, _user_id)
    if body.enabled is not None:
        af.enabled = body.enabled
    if body.emailAddress is not None:
        af.email_address = body.emailAddress
    if body.disposition is not None:
        af.disposition = body.disposition
    db.commit()
    db.refresh(af)
    return AutoForwardingSchema(
        enabled=af.enabled,
        emailAddress=af.email_address,
        disposition=af.disposition,
    )


# ============================================================
# IMAP Settings (stub — not needed for agent use cases)
# ============================================================

# In-memory per-user stores; no DB table needed for these simple settings.
_imap_settings: dict[str, dict] = {}
_pop_settings: dict[str, dict] = {}
_language_settings: dict[str, dict] = {}


def reset_in_memory_settings():
    """Clear in-memory settings stores (called on engine reset for test isolation)."""
    _imap_settings.clear()
    _pop_settings.clear()
    _language_settings.clear()


@router.get("/users/{userId}/settings/imap", response_model=ImapSettingsSchema)
def get_imap(
    userId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    s = _imap_settings.get(_user_id, {})
    return ImapSettingsSchema(**s)


@router.put("/users/{userId}/settings/imap", response_model=ImapSettingsSchema)
def update_imap(
    userId: str,
    body: ImapSettingsUpdateRequest,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    cur = _imap_settings.get(_user_id, {})
    for field in ("enabled", "autoExpunge", "expungeBehavior", "maxFolderSize"):
        val = getattr(body, field, None)
        if val is not None:
            cur[field] = val
    _imap_settings[_user_id] = cur
    return ImapSettingsSchema(**cur)


# ============================================================
# POP Settings (stub — not needed for agent use cases)
# ============================================================

@router.get("/users/{userId}/settings/pop", response_model=PopSettingsSchema)
def get_pop(
    userId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    s = _pop_settings.get(_user_id, {})
    return PopSettingsSchema(**s)


@router.put("/users/{userId}/settings/pop", response_model=PopSettingsSchema)
def update_pop(
    userId: str,
    body: PopSettingsUpdateRequest,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    cur = _pop_settings.get(_user_id, {})
    for field in ("accessWindow", "disposition"):
        val = getattr(body, field, None)
        if val is not None:
            cur[field] = val
    _pop_settings[_user_id] = cur
    return PopSettingsSchema(**cur)


# ============================================================
# Language Settings (stub — not needed for agent use cases)
# ============================================================

@router.get("/users/{userId}/settings/language", response_model=LanguageSettingsSchema)
def get_language(
    userId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    s = _language_settings.get(_user_id, {})
    return LanguageSettingsSchema(**s)


@router.put("/users/{userId}/settings/language", response_model=LanguageSettingsSchema)
def update_language(
    userId: str,
    body: LanguageSettingsUpdateRequest,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    cur = _language_settings.get(_user_id, {})
    if body.displayLanguage is not None:
        cur["displayLanguage"] = body.displayLanguage
    _language_settings[_user_id] = cur
    return LanguageSettingsSchema(**cur)


# ============================================================
# Watch / Stop (no-ops)
# ============================================================

@router.post("/users/{userId}/watch", response_model=WatchResponse)
def watch(
    userId: str,
    body: WatchRequest | None = None,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    user = db.query(User).filter(User.id == _user_id).first()
    history_id = str(user.history_id) if user else "1"
    # Expiration: 7 days from now in epoch ms
    expiration = str(int((time.time() + 7 * 86400) * 1000))
    return WatchResponse(historyId=history_id, expiration=expiration)


@router.post("/users/{userId}/stop")
def stop(
    userId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    return {}
