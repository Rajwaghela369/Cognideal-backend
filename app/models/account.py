import uuid
from typing import Optional

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.mixins import TimestampMixin, UUIDPrimaryKeyMixin


class Account(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "accounts"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    industry: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    website: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    employee_band: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    hq_region: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)


class Contact(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A person at a customer account.

    Contacts hang off the *account*, not off a deal: the same stakeholder can
    appear in several deals with the same company, and everything specific to
    one deal (buying role, influence, sentiment) lives on ``deal_contacts``.
    """

    __tablename__ = "contacts"
    __table_args__ = (UniqueConstraint("account_id", "email", name="account_email"),)

    account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("accounts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    first_name: Mapped[str] = mapped_column(String(120), nullable=False)
    last_name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[Optional[str]] = mapped_column(String(320), nullable=True)
    title: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    phone: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
