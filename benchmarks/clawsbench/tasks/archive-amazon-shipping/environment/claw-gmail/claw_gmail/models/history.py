"""History record model for change tracking."""

from datetime import datetime

from sqlalchemy import String, Integer, DateTime, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class HistoryRecord(Base):
    __tablename__ = "history_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    history_id: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String, nullable=False)  # messageAdded, messageDeleted, labelAdded, labelRemoved
    message_id: Mapped[str | None] = mapped_column(String, nullable=True)
    label_ids_added: Mapped[str] = mapped_column(Text, default="")  # comma-separated
    label_ids_removed: Mapped[str] = mapped_column(Text, default="")  # comma-separated
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="history_records")  # noqa: F821
