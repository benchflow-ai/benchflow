"""User model."""

from sqlalchemy import String, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    email_address: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    history_id: Mapped[int] = mapped_column(Integer, default=1)

    messages: Mapped[list["Message"]] = relationship(back_populates="user", cascade="all, delete-orphan")  # noqa: F821
    threads: Mapped[list["Thread"]] = relationship(back_populates="user", cascade="all, delete-orphan")  # noqa: F821
    labels: Mapped[list["Label"]] = relationship(back_populates="user", cascade="all, delete-orphan")  # noqa: F821
    drafts: Mapped[list["Draft"]] = relationship(back_populates="user", cascade="all, delete-orphan")  # noqa: F821
    filters: Mapped[list["Filter"]] = relationship(back_populates="user", cascade="all, delete-orphan")  # noqa: F821
    contacts: Mapped[list["Contact"]] = relationship(back_populates="user", cascade="all, delete-orphan")  # noqa: F821
    history_records: Mapped[list["HistoryRecord"]] = relationship(back_populates="user", cascade="all, delete-orphan")  # noqa: F821
    send_as_entries: Mapped[list["SendAs"]] = relationship(back_populates="user", cascade="all, delete-orphan")  # noqa: F821
    forwarding_addresses: Mapped[list["ForwardingAddress"]] = relationship(back_populates="user", cascade="all, delete-orphan")  # noqa: F821
    delegates: Mapped[list["Delegate"]] = relationship(back_populates="user", cascade="all, delete-orphan")  # noqa: F821
    vacation_settings: Mapped["VacationSettings | None"] = relationship(back_populates="user", uselist=False, cascade="all, delete-orphan")  # noqa: F821
    auto_forwarding: Mapped["AutoForwarding | None"] = relationship(back_populates="user", uselist=False, cascade="all, delete-orphan")  # noqa: F821
