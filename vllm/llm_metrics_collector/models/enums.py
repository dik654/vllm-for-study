"""Enumerations for LLM metrics collection."""

from enum import Enum


class FinishReason(str, Enum):
    """Reason why a request finished.

    Maps to vLLM's finish reasons for compatibility.
    """
    STOP = "stop"  # Normal completion (EOS token or stop sequence)
    LENGTH = "length"  # Reached max_tokens limit
    ABORT = "abort"  # Client cancelled or error occurred
    PREEMPTED = "preempted"  # Request was preempted (may retry)

    @classmethod
    def from_vllm_finish_reason(cls, reason: str) -> "FinishReason":
        """Convert vLLM finish reason to our enum.

        Args:
            reason: vLLM finish reason string

        Returns:
            Corresponding FinishReason enum value
        """
        reason_lower = reason.lower()
        if reason_lower == "stop":
            return cls.STOP
        elif reason_lower == "length":
            return cls.LENGTH
        elif reason_lower in ("abort", "cancelled", "error"):
            return cls.ABORT
        elif reason_lower == "preempted":
            return cls.PREEMPTED
        else:
            # Default to ABORT for unknown reasons
            return cls.ABORT


class PricingTier(str, Enum):
    """Pricing tier based on priority/SLA requirements.

    Different tiers can have different pricing multipliers:
    - REALTIME: Premium pricing for guaranteed low latency (TTFT < 1s)
    - STANDARD: Standard pricing for normal requests
    - BATCH: Discounted pricing for batch/async requests (no latency guarantee)
    """
    REALTIME = "realtime"
    STANDARD = "standard"
    BATCH = "batch"


class StorageFormat(str, Enum):
    """Supported storage/export formats.

    Different formats have different trade-offs:
    - JSON: Human-readable, good for debugging
    - JSONL: JSON Lines, efficient for streaming and appending
    - CSV: Universal format, easy to import into spreadsheets
    - PARQUET: Columnar format, best for analytics and compression
    """
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"


class AggregationPeriod(str, Enum):
    """Time period for metrics aggregation.

    Used by analyzers to group metrics by time windows.
    """
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"
