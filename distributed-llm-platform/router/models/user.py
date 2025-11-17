"""User model for database."""
from datetime import datetime
from decimal import Decimal
from typing import Optional
from sqlalchemy import Column, String, Integer, Numeric, Boolean, DateTime, Index
from sqlalchemy.sql import func
from router.core.database import Base


class User(Base):
    """User model - represents platform users."""

    __tablename__ = "users"

    user_id = Column(String(64), primary_key=True)
    email = Column(String(255), nullable=False, unique=True)
    api_key = Column(String(128), nullable=False, unique=True)
    status = Column(String(32), nullable=False, default="active")  # active, suspended, inactive
    tier = Column(String(32), nullable=False, default="free")  # free, starter, pro, enterprise

    # Billing
    billing_email = Column(String(255), nullable=True)
    payment_method_id = Column(String(128), nullable=True)
    credit_balance = Column(Numeric(12, 2), default=Decimal("0.00"))  # Prepaid credits

    # Limits
    rate_limit_per_minute = Column(Integer, default=60)
    monthly_budget_usd = Column(Numeric(10, 2), nullable=True)

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        Index("idx_user_api_key", "api_key"),
        Index("idx_user_status", "status"),
        Index("idx_user_tier", "tier"),
    )

    def __repr__(self) -> str:
        return f"<User(id={self.user_id}, email={self.email}, tier={self.tier})>"
