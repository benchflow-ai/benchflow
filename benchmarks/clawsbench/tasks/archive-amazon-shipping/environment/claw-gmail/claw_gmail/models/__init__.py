"""Database models."""

from .base import Base, get_engine, get_session_factory, init_db, reset_engine
from .user import User
from .message import Thread, Message, MessageLabel
from .label import Label, LabelType, SYSTEM_LABELS, HIDDEN_SYSTEM_LABELS
from .draft import Draft
from .attachment import Attachment
from .history import HistoryRecord
from .filter import Filter
from .contact import Contact
from .settings import SendAs, ForwardingAddress, Delegate, VacationSettings, AutoForwarding

__all__ = [
    "Base",
    "get_engine",
    "get_session_factory",
    "init_db",
    "reset_engine",
    "User",
    "Thread",
    "Message",
    "MessageLabel",
    "Label",
    "LabelType",
    "SYSTEM_LABELS",
    "HIDDEN_SYSTEM_LABELS",
    "Draft",
    "Attachment",
    "HistoryRecord",
    "Filter",
    "Contact",
    "SendAs",
    "ForwardingAddress",
    "Delegate",
    "VacationSettings",
    "AutoForwarding",
]
