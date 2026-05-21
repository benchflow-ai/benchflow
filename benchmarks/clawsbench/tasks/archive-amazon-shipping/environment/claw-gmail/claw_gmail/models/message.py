"""Message and Thread models."""

from datetime import datetime

from sqlalchemy import String, Integer, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class Thread(Base):
    __tablename__ = "threads"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    snippet: Mapped[str] = mapped_column(String(200), default="")
    history_id: Mapped[int] = mapped_column(Integer, default=1)

    user: Mapped["User"] = relationship(back_populates="threads")  # noqa: F821
    messages: Mapped[list["Message"]] = relationship(back_populates="thread", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    thread_id: Mapped[str] = mapped_column(ForeignKey("threads.id"), nullable=False)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)

    sender: Mapped[str] = mapped_column(String, nullable=False)
    to: Mapped[str] = mapped_column(String, default="")  # comma-separated
    cc: Mapped[str] = mapped_column(String, default="")
    bcc: Mapped[str] = mapped_column(String, default="")
    reply_to: Mapped[str] = mapped_column(String, default="")
    subject: Mapped[str] = mapped_column(String, default="")
    snippet: Mapped[str] = mapped_column(String(200), default="")
    body_plain: Mapped[str] = mapped_column(Text, default="")
    body_html: Mapped[str] = mapped_column(Text, default="")
    internal_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    message_id_header: Mapped[str] = mapped_column(String, default="")
    in_reply_to: Mapped[str] = mapped_column(String, default="")
    references: Mapped[str] = mapped_column(Text, default="")

    is_read: Mapped[bool] = mapped_column(Boolean, default=False)
    is_starred: Mapped[bool] = mapped_column(Boolean, default=False)
    is_trash: Mapped[bool] = mapped_column(Boolean, default=False)
    is_spam: Mapped[bool] = mapped_column(Boolean, default=False)
    is_draft: Mapped[bool] = mapped_column(Boolean, default=False)
    is_sent: Mapped[bool] = mapped_column(Boolean, default=False)

    user: Mapped["User"] = relationship(back_populates="messages")  # noqa: F821
    thread: Mapped["Thread"] = relationship(back_populates="messages")
    labels: Mapped[list["MessageLabel"]] = relationship(back_populates="message", cascade="all, delete-orphan")
    attachments: Mapped[list["Attachment"]] = relationship(back_populates="message", cascade="all, delete-orphan")  # noqa: F821


class MessageLabel(Base):
    __tablename__ = "message_labels"

    message_id: Mapped[str] = mapped_column(ForeignKey("messages.id"), primary_key=True)
    label_id: Mapped[str] = mapped_column(ForeignKey("labels.id"), primary_key=True)

    message: Mapped["Message"] = relationship(back_populates="labels")
    label: Mapped["Label"] = relationship(back_populates="message_labels")  # noqa: F821
