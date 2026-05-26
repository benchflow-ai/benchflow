"""Filter model for auto-labeling rules."""

from sqlalchemy import String, Boolean, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class Filter(Base):
    __tablename__ = "filters"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)

    # Criteria
    criteria_from: Mapped[str] = mapped_column(String, default="")
    criteria_to: Mapped[str] = mapped_column(String, default="")
    criteria_subject: Mapped[str] = mapped_column(String, default="")
    criteria_query: Mapped[str] = mapped_column(String, default="")
    criteria_has_attachment: Mapped[bool] = mapped_column(Boolean, default=False)
    criteria_negated_query: Mapped[str] = mapped_column(String, default="")
    criteria_exclude_chats: Mapped[bool] = mapped_column(Boolean, default=False)
    criteria_size: Mapped[int | None] = mapped_column(default=None)
    criteria_size_comparison: Mapped[str] = mapped_column(String, default="")  # larger, smaller

    # Actions
    action_add_label_ids: Mapped[str] = mapped_column(String, default="")  # comma-separated
    action_remove_label_ids: Mapped[str] = mapped_column(String, default="")
    action_forward: Mapped[str] = mapped_column(String, default="")
    action_mark_read: Mapped[bool] = mapped_column(Boolean, default=False)
    action_star: Mapped[bool] = mapped_column(Boolean, default=False)
    action_trash: Mapped[bool] = mapped_column(Boolean, default=False)
    action_never_spam: Mapped[bool] = mapped_column(Boolean, default=False)
    action_important: Mapped[bool] = mapped_column(Boolean, default=False)

    user: Mapped["User"] = relationship(back_populates="filters")  # noqa: F821
