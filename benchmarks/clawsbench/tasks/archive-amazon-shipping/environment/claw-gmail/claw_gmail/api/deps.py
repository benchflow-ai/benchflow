"""Shared dependencies for API routes."""

from __future__ import annotations

from fastapi import Header, HTTPException, Depends
from sqlalchemy.orm import Session

from claw_gmail.models import get_session_factory, User


def get_db() -> Session:
    """Yield a DB session."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def resolve_user_id(
    userId: str,
    x_claw_gmail_user: str | None = Header(None),
    db: Session = Depends(get_db),
) -> str:
    """Resolve 'me' to the actual user ID.

    Priority: userId path param -> X-Mock-Gmail-User header -> first user in DB.
    """
    if userId != "me":
        user = db.query(User).filter(User.id == userId).first()
        if not user:
            raise HTTPException(404, f"User {userId!r} not found")
        return userId

    if x_claw_gmail_user:
        user = db.query(User).filter(
            (User.id == x_claw_gmail_user) | (User.email_address == x_claw_gmail_user)
        ).first()
        if user:
            return user.id

    # Fallback: first user
    user = db.query(User).first()
    if not user:
        raise HTTPException(404, "No users in database. Run `claw-gmail seed` first.")
    return user.id
