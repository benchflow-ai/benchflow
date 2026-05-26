"""Settings models: SendAs, ForwardingAddress, Delegate, VacationSettings, AutoForwarding."""

from sqlalchemy import String, Boolean, Integer, Text, ForeignKey, BigInteger
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class SendAs(Base):
    __tablename__ = "send_as"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    send_as_email: Mapped[str] = mapped_column(String, nullable=False)
    display_name: Mapped[str] = mapped_column(String, default="")
    reply_to_address: Mapped[str] = mapped_column(String, default="")
    signature: Mapped[str] = mapped_column(Text, default="")
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    treat_as_alias: Mapped[bool] = mapped_column(Boolean, default=False)
    verification_status: Mapped[str] = mapped_column(String, default="accepted")

    user: Mapped["User"] = relationship(back_populates="send_as_entries")  # noqa: F821


class ForwardingAddress(Base):
    __tablename__ = "forwarding_addresses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    forwarding_email: Mapped[str] = mapped_column(String, nullable=False)
    verification_status: Mapped[str] = mapped_column(String, default="accepted")

    user: Mapped["User"] = relationship(back_populates="forwarding_addresses")  # noqa: F821


class Delegate(Base):
    __tablename__ = "delegates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    delegate_email: Mapped[str] = mapped_column(String, nullable=False)
    verification_status: Mapped[str] = mapped_column(String, default="accepted")

    user: Mapped["User"] = relationship(back_populates="delegates")  # noqa: F821


class VacationSettings(Base):
    __tablename__ = "vacation_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, unique=True)
    enable_auto_reply: Mapped[bool] = mapped_column(Boolean, default=False)
    response_subject: Mapped[str] = mapped_column(String, default="")
    response_body_html: Mapped[str] = mapped_column(Text, default="")
    response_body_plain_text: Mapped[str] = mapped_column(Text, default="")
    restrict_to_contacts: Mapped[bool] = mapped_column(Boolean, default=False)
    restrict_to_domain: Mapped[bool] = mapped_column(Boolean, default=False)
    start_time: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    end_time: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    user: Mapped["User"] = relationship(back_populates="vacation_settings")  # noqa: F821


class AutoForwarding(Base):
    __tablename__ = "auto_forwarding"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, unique=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    email_address: Mapped[str] = mapped_column(String, default="")
    disposition: Mapped[str] = mapped_column(String, default="leaveInInbox")

    user: Mapped["User"] = relationship(back_populates="auto_forwarding")  # noqa: F821
