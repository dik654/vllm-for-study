"""Request log model for database."""
from datetime import datetime
from decimal import Decimal
from sqlalchemy import Column, String, Integer, Numeric, Boolean, DateTime, Text, ForeignKey, Index
from sqlalchemy.sql import func
from router.core.database import Base


class Request(Base):
    """Request model - logs all LLM API requests."""

    __tablename__ = "requests"

    request_id = Column(String(64), primary_key=True)
    user_id = Column(String(64), ForeignKey("users.user_id"), nullable=False)
    provider_id = Column(String(64), ForeignKey("providers.provider_id"), nullable=False)

    # Request Info
    model = Column(String(128), nullable=False)
    endpoint = Column(String(128), nullable=False)
    is_batch = Column(Boolean, default=False)

    # Token Usage
    input_tokens = Column(Integer, nullable=False)
    cached_input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, nullable=False)
    latency_ms = Column(Integer, nullable=False)

    # Cost Breakdown
    provider_cost_usd = Column(Numeric(12, 8), nullable=False)
    commission_usd = Column(Numeric(12, 8), nullable=False)
    user_charge_usd = Column(Numeric(12, 8), nullable=False)

    # Status
    status = Column(String(32), nullable=False)  # success, error, timeout
    error_message = Column(Text, nullable=True)

    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("idx_request_user_created", "user_id", "created_at"),
        Index("idx_request_provider_created", "provider_id", "created_at"),
        Index("idx_request_created_at", "created_at"),
        Index("idx_request_is_batch", "is_batch"),
    )

    def __repr__(self) -> str:
        return f"<Request(id={self.request_id}, user={self.user_id}, status={self.status})>"
