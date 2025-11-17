"""Provider model for database."""
from datetime import datetime
from decimal import Decimal
from typing import Optional
from sqlalchemy import Column, String, Integer, Numeric, Boolean, DateTime, JSON, Index
from sqlalchemy.sql import func
from router.core.database import Base


class Provider(Base):
    """Provider model - represents LLM/Resource providers."""

    __tablename__ = "providers"

    provider_id = Column(String(64), primary_key=True)
    provider_name = Column(String(255), nullable=False)
    endpoint_url = Column(String(512), nullable=False)
    provider_token = Column(String(128), nullable=False, unique=True)
    status = Column(String(32), nullable=False, default="active")  # active, suspended, inactive

    # LLM Pricing
    token_input_rate = Column(Numeric(12, 10), nullable=False)
    token_output_rate = Column(Numeric(12, 10), nullable=False)
    cached_input_discount = Column(Numeric(3, 2), default=Decimal("0.9"))
    request_base_rate = Column(Numeric(10, 6), nullable=False)
    batch_discount = Column(Numeric(3, 2), default=Decimal("0.5"))

    # Context Memory/Storage Pricing
    context_memory_hourly_rate = Column(Numeric(10, 6), nullable=False)
    context_storage_hourly_rate = Column(Numeric(10, 6), nullable=False)

    # GP Memory/Storage Pricing
    gp_memory_hourly_rate = Column(Numeric(10, 6), nullable=False)
    gp_storage_hot_hourly_rate = Column(Numeric(10, 6), nullable=False)
    gp_storage_cold_hourly_rate = Column(Numeric(10, 6), nullable=False)

    # Capacity
    max_concurrent_requests = Column(Integer, nullable=False)
    context_memory_capacity_gb = Column(Integer, nullable=False)
    context_storage_capacity_tb = Column(Integer, nullable=False)
    gp_memory_capacity_gb = Column(Integer, nullable=False)
    gp_storage_capacity_tb = Column(Integer, nullable=False)

    # Metadata
    supported_models = Column(JSON, nullable=False)  # List of model names
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Health tracking (not persisted - managed by health checker)
    # These are tracked in Redis in real implementation
    # current_requests: int
    # avg_latency_ms: float
    # last_health_check: datetime
    # is_healthy: bool

    __table_args__ = (
        Index("idx_provider_status", "status"),
        Index("idx_provider_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Provider(id={self.provider_id}, name={self.provider_name}, status={self.status})>"
