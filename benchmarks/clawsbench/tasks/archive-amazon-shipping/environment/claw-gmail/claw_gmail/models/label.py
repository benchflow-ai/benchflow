"""Label model."""

from sqlalchemy import String, Integer, ForeignKey, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

import enum


class LabelType(str, enum.Enum):
    system = "system"
    user = "user"


# (id, name) — system labels seeded for every user
SYSTEM_LABELS = [
    ("INBOX", "INBOX"),
    ("SENT", "SENT"),
    ("DRAFT", "DRAFT"),
    ("TRASH", "TRASH"),
    ("SPAM", "SPAM"),
    ("STARRED", "STARRED"),
    ("UNREAD", "UNREAD"),
    ("IMPORTANT", "IMPORTANT"),
    ("CHAT", "CHAT"),
    ("CATEGORY_PERSONAL", "CATEGORY_PERSONAL"),
    ("CATEGORY_SOCIAL", "CATEGORY_SOCIAL"),
    ("CATEGORY_PROMOTIONS", "CATEGORY_PROMOTIONS"),
    ("CATEGORY_UPDATES", "CATEGORY_UPDATES"),
    ("CATEGORY_FORUMS", "CATEGORY_FORUMS"),
]

# System labels that have messageListVisibility=hide, labelListVisibility=labelHide
# (others like SENT, INBOX, DRAFT, STARRED, UNREAD have no visibility fields)
HIDDEN_SYSTEM_LABELS = {
    "CHAT", "IMPORTANT", "TRASH", "SPAM",
    "CATEGORY_FORUMS", "CATEGORY_UPDATES",
    "CATEGORY_PERSONAL", "CATEGORY_PROMOTIONS", "CATEGORY_SOCIAL",
}


class Label(Base):
    __tablename__ = "labels"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    type: Mapped[str] = mapped_column(SAEnum(LabelType), default=LabelType.user)
    color_bg: Mapped[str | None] = mapped_column(String, nullable=True)
    color_text: Mapped[str | None] = mapped_column(String, nullable=True)
    message_list_visibility: Mapped[str | None] = mapped_column(String, nullable=True)
    label_list_visibility: Mapped[str | None] = mapped_column(String, nullable=True)
    messages_total: Mapped[int] = mapped_column(Integer, default=0)
    messages_unread: Mapped[int] = mapped_column(Integer, default=0)
    threads_total: Mapped[int] = mapped_column(Integer, default=0)
    threads_unread: Mapped[int] = mapped_column(Integer, default=0)

    user: Mapped["User"] = relationship(back_populates="labels")  # noqa: F821
    message_labels: Mapped[list["MessageLabel"]] = relationship(back_populates="label", cascade="all, delete-orphan")  # noqa: F821
