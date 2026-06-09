"""Pydantic schemas mirroring Gmail API response format."""

from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field


# Profile
class Profile(BaseModel):
    emailAddress: str
    messagesTotal: int = 0
    threadsTotal: int = 0
    historyId: str = "1"


# Labels
class LabelColor(BaseModel):
    backgroundColor: str | None = None
    textColor: str | None = None


class LabelSchema(BaseModel):
    id: str
    name: str
    type: str = "user"
    messageListVisibility: str | None = None
    labelListVisibility: str | None = None
    messagesTotal: int | None = None
    messagesUnread: int | None = None
    threadsTotal: int | None = None
    threadsUnread: int | None = None
    color: LabelColor | None = None


class LabelListResponse(BaseModel):
    labels: list[LabelSchema]


class LabelCreateRequest(BaseModel):
    name: str
    messageListVisibility: str = "show"
    labelListVisibility: str = "labelShow"
    color: LabelColor | None = None


class LabelUpdateRequest(BaseModel):
    name: str | None = None
    messageListVisibility: str | None = None
    labelListVisibility: str | None = None
    color: LabelColor | None = None


# Messages
class Header(BaseModel):
    name: str
    value: str


class MessagePartBody(BaseModel):
    attachmentId: str | None = None
    size: int = 0
    data: str | None = None  # base64url


class MessagePart(BaseModel):
    partId: str = ""
    mimeType: str = "text/plain"
    filename: str = ""
    headers: list[Header] = Field(default_factory=list)
    body: MessagePartBody = Field(default_factory=MessagePartBody)
    parts: list[MessagePart] | None = None


class MessageSchema(BaseModel):
    """Full message response. Use model_config exclude_none to match Gmail's behavior
    of omitting absent fields (e.g. no 'payload' key in raw format, not payload=null)."""
    model_config = {"exclude_none": True}

    id: str
    threadId: str
    labelIds: list[str] = Field(default_factory=list)
    snippet: str | None = None
    historyId: str | None = None
    internalDate: str | None = None  # epoch ms string
    payload: MessagePart | None = None
    sizeEstimate: int | None = None
    raw: str | None = None


class MessageMinimalSchema(BaseModel):
    """Minimal response for send/modify/trash/untrash — matches real Gmail."""
    id: str
    threadId: str
    labelIds: list[str] = Field(default_factory=list)


class MessageListItem(BaseModel):
    id: str
    threadId: str


class MessageListResponse(BaseModel):
    messages: list[MessageListItem] = Field(default_factory=list)
    nextPageToken: str | None = None
    resultSizeEstimate: int = 0


class MessageSendRequest(BaseModel):
    raw: str
    threadId: str | None = None


class MessageModifyRequest(BaseModel):
    addLabelIds: list[str] = Field(default_factory=list)
    removeLabelIds: list[str] = Field(default_factory=list)


class MessageBatchModifyRequest(BaseModel):
    ids: list[str]
    addLabelIds: list[str] = Field(default_factory=list)
    removeLabelIds: list[str] = Field(default_factory=list)


class MessageBatchDeleteRequest(BaseModel):
    ids: list[str]


# Threads
class ThreadSchema(BaseModel):
    id: str
    historyId: str | None = None
    snippet: str = ""
    messages: list[MessageSchema] = Field(default_factory=list)


class ThreadListItem(BaseModel):
    id: str
    snippet: str = ""
    historyId: str | None = None


class ThreadListResponse(BaseModel):
    threads: list[ThreadListItem] = Field(default_factory=list)
    nextPageToken: str | None = None
    resultSizeEstimate: int = 0


class ThreadModifyRequest(BaseModel):
    addLabelIds: list[str] = Field(default_factory=list)
    removeLabelIds: list[str] = Field(default_factory=list)


# Drafts
class DraftMessage(BaseModel):
    raw: str
    threadId: str | None = None


class DraftCreateRequest(BaseModel):
    message: DraftMessage


class DraftSchema(BaseModel):
    id: str
    message: MessageSchema | MessageMinimalSchema


class DraftListItem(BaseModel):
    id: str
    message: MessageListItem


class DraftListResponse(BaseModel):
    drafts: list[DraftListItem] = Field(default_factory=list)
    nextPageToken: str | None = None
    resultSizeEstimate: int = 0


# History
class HistoryMessageItem(BaseModel):
    id: str
    threadId: str
    labelIds: list[str] = Field(default_factory=list)


class HistoryMessageAdded(BaseModel):
    message: HistoryMessageItem


class HistoryMessageDeleted(BaseModel):
    message: HistoryMessageItem


class HistoryLabelAdded(BaseModel):
    message: HistoryMessageItem
    labelIds: list[str]


class HistoryLabelRemoved(BaseModel):
    message: HistoryMessageItem
    labelIds: list[str]


class HistoryEntry(BaseModel):
    id: str
    messages: list[MessageListItem] = Field(default_factory=list)
    messagesAdded: list[HistoryMessageAdded] | None = None
    messagesDeleted: list[HistoryMessageDeleted] | None = None
    labelsAdded: list[HistoryLabelAdded] | None = None
    labelsRemoved: list[HistoryLabelRemoved] | None = None


class HistoryListResponse(BaseModel):
    history: list[HistoryEntry] = Field(default_factory=list)
    nextPageToken: str | None = None
    historyId: str = "1"


# Attachments
class AttachmentSchema(BaseModel):
    attachmentId: str
    size: int = 0
    data: str | None = None


# Filters
class FilterCriteria(BaseModel):
    from_: str | None = Field(None, alias="from")
    to: str | None = None
    subject: str | None = None
    query: str | None = None
    hasAttachment: bool = False
    negatedQuery: str | None = None
    excludeChats: bool = False
    size: int | None = None
    sizeComparison: str | None = None


class FilterAction(BaseModel):
    addLabelIds: list[str] = Field(default_factory=list)
    removeLabelIds: list[str] = Field(default_factory=list)
    forward: str | None = None


class FilterSchema(BaseModel):
    id: str
    criteria: FilterCriteria
    action: FilterAction


class FilterListResponse(BaseModel):
    filter: list[FilterSchema] = Field(default_factory=list)


# Settings: SendAs
class SendAsSchema(BaseModel):
    sendAsEmail: str
    displayName: str = ""
    replyToAddress: str = ""
    signature: str = ""
    isPrimary: bool = False
    isDefault: bool = False
    treatAsAlias: bool = False
    verificationStatus: str = "accepted"


class SendAsListResponse(BaseModel):
    sendAs: list[SendAsSchema] = Field(default_factory=list)


class SendAsCreateRequest(BaseModel):
    sendAsEmail: str
    displayName: str = ""
    replyToAddress: str = ""
    signature: str = ""
    treatAsAlias: bool = False
    isDefault: bool = False


class SendAsUpdateRequest(BaseModel):
    displayName: str | None = None
    replyToAddress: str | None = None
    signature: str | None = None
    treatAsAlias: bool | None = None
    isDefault: bool | None = None


# Settings: ForwardingAddresses
class ForwardingAddressSchema(BaseModel):
    forwardingEmail: str
    verificationStatus: str = "accepted"


class ForwardingAddressListResponse(BaseModel):
    forwardingAddresses: list[ForwardingAddressSchema] = Field(default_factory=list)


class ForwardingAddressCreateRequest(BaseModel):
    forwardingEmail: str


# Settings: Delegates
class DelegateSchema(BaseModel):
    delegateEmail: str
    verificationStatus: str = "accepted"


class DelegateListResponse(BaseModel):
    delegates: list[DelegateSchema] = Field(default_factory=list)


class DelegateCreateRequest(BaseModel):
    delegateEmail: str


# Settings: Vacation
class VacationSettingsSchema(BaseModel):
    enableAutoReply: bool = False
    responseSubject: str = ""
    responseBodyHtml: str = ""
    responseBodyPlainText: str = ""
    restrictToContacts: bool = False
    restrictToDomain: bool = False
    startTime: str | None = None  # epoch ms string
    endTime: str | None = None


class VacationUpdateRequest(BaseModel):
    enableAutoReply: bool | None = None
    responseSubject: str | None = None
    responseBodyHtml: str | None = None
    responseBodyPlainText: str | None = None
    restrictToContacts: bool | None = None
    restrictToDomain: bool | None = None
    startTime: str | None = None
    endTime: str | None = None


# Settings: AutoForwarding
class AutoForwardingSchema(BaseModel):
    enabled: bool = False
    emailAddress: str = ""
    disposition: str = "leaveInInbox"


class AutoForwardingUpdateRequest(BaseModel):
    enabled: bool | None = None
    emailAddress: str | None = None
    disposition: str | None = None


# Settings: IMAP
class ImapSettingsSchema(BaseModel):
    enabled: bool = False
    autoExpunge: bool = True
    expungeBehavior: str = "archive"
    maxFolderSize: int = 0


class ImapSettingsUpdateRequest(BaseModel):
    enabled: bool | None = None
    autoExpunge: bool | None = None
    expungeBehavior: str | None = None
    maxFolderSize: int | None = None


# Settings: POP
class PopSettingsSchema(BaseModel):
    accessWindow: str = "disabled"
    disposition: str = "leaveInInbox"


class PopSettingsUpdateRequest(BaseModel):
    accessWindow: str | None = None
    disposition: str | None = None


# Settings: Language
class LanguageSettingsSchema(BaseModel):
    displayLanguage: str = "en"


class LanguageSettingsUpdateRequest(BaseModel):
    displayLanguage: str | None = None


# Settings: Filters (create request)
class FilterCreateRequest(BaseModel):
    criteria: FilterCriteria | None = None
    action: FilterAction | None = None


# Watch/Stop
class WatchRequest(BaseModel):
    topicName: str = ""
    labelIds: list[str] = Field(default_factory=list)
    labelFilterAction: str | None = None


class WatchResponse(BaseModel):
    historyId: str = "1"
    expiration: str = ""


# Message Insert Request
class MessageInsertRequest(BaseModel):
    raw: str
    labelIds: list[str] = Field(default_factory=list)
    threadId: str | None = None


# Admin
class AdminResetResponse(BaseModel):
    status: str = "ok"
    message: str = ""


class ActionLogEntry(BaseModel):
    timestamp: str
    method: str
    path: str
    user_id: str
    request_body: dict | None = None
    response_status: int = 200
