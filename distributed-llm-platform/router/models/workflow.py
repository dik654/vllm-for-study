"""Workflow models for database."""
from datetime import datetime
from decimal import Decimal
from sqlalchemy import Column, String, Integer, Numeric, Boolean, DateTime, Text, JSON, ForeignKey, Index
from sqlalchemy.sql import func
from router.core.database import Base


class Workflow(Base):
    """Workflow model - represents user-created workflows."""

    __tablename__ = "workflows"

    workflow_id = Column(String(64), primary_key=True)
    user_id = Column(String(64), ForeignKey("users.user_id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    version = Column(String(16), nullable=False, default="1.0.0")

    # Workflow Definition (JSON)
    nodes = Column(JSON, nullable=False)  # List of nodes
    connections = Column(JSON, nullable=False)  # List of connections
    metadata = Column(JSON, nullable=False)  # Tags, category, etc.
    settings = Column(JSON, nullable=False)  # Timeout, error_handling, etc.

    # Public/Private
    is_public = Column(Boolean, default=False)
    fork_count = Column(Integer, default=0)
    execution_count = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        Index("idx_workflow_user_created", "user_id", "created_at"),
        Index("idx_workflow_public_created", "is_public", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Workflow(id={self.workflow_id}, name={self.name}, user={self.user_id})>"


class WorkflowExecution(Base):
    """WorkflowExecution model - tracks workflow execution."""

    __tablename__ = "workflow_executions"

    execution_id = Column(String(64), primary_key=True)
    workflow_id = Column(String(64), ForeignKey("workflows.workflow_id"), nullable=False)
    user_id = Column(String(64), ForeignKey("users.user_id"), nullable=False)

    # Execution State
    state = Column(String(32), nullable=False)  # running, completed, failed, cancelled
    input_data = Column(JSON, nullable=False)
    output_data = Column(JSON, nullable=True)

    # Progress Tracking
    progress = Column(JSON, nullable=True)  # {completed_nodes: 3, total_nodes: 5}
    node_outputs = Column(JSON, nullable=True)  # {node_id: output}

    # Timestamps
    started_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Metrics
    total_duration_ms = Column(Integer, nullable=True)
    total_cost_usd = Column(Numeric(12, 8), nullable=True)

    # Error Info
    error_message = Column(Text, nullable=True)

    __table_args__ = (
        Index("idx_execution_user_started", "user_id", "started_at"),
        Index("idx_execution_workflow_started", "workflow_id", "started_at"),
        Index("idx_execution_state", "state"),
    )

    def __repr__(self) -> str:
        return f"<WorkflowExecution(id={self.execution_id}, workflow={self.workflow_id}, state={self.state})>"
