"""
LLM Metrics Collector for vLLM

A comprehensive metrics collection system for tracking LLM usage,
resource consumption, and cost calculation.

Main components:
- models: Data models (RequestMetrics, AggregatedMetrics)
- collectors: Metrics collectors (VLLMMetricsCollector)
- storage: Storage backends (File, Database, Memory)
- pricing: Pricing calculators (TokenBased, Hybrid, etc.)
- exporters: Data exporters (JSON, CSV, Parquet)
- analyzers: Metrics aggregators and reporters
"""

__version__ = "0.1.0"

# Import main classes for easy access
from vllm.llm_metrics_collector.models.enums import (
    FinishReason,
    PricingTier,
    StorageFormat,
    AggregationPeriod,
)
from vllm.llm_metrics_collector.models.request_metrics import RequestMetrics
from vllm.llm_metrics_collector.models.aggregated_metrics import (
    AggregatedMetrics,
)
from vllm.llm_metrics_collector.collectors.base import MetricsCollectorBase
from vllm.llm_metrics_collector.collectors.vllm_collector import (
    VLLMMetricsCollector,
)
from vllm.llm_metrics_collector.storage.base import StorageBackend
from vllm.llm_metrics_collector.storage.memory_storage import (
    MemoryStorageBackend,
)
from vllm.llm_metrics_collector.storage.file_storage import FileStorageBackend
from vllm.llm_metrics_collector.storage.pending_queue import PendingQueueStorage
from vllm.llm_metrics_collector.pricing.base import PricingCalculator
from vllm.llm_metrics_collector.pricing.token_based import TokenBasedPricing

__all__ = [
    "__version__",
    # Enums
    "FinishReason",
    "PricingTier",
    "StorageFormat",
    "AggregationPeriod",
    # Models
    "RequestMetrics",
    "AggregatedMetrics",
    # Collectors
    "MetricsCollectorBase",
    "VLLMMetricsCollector",
    # Storage
    "StorageBackend",
    "MemoryStorageBackend",
    "FileStorageBackend",
    "PendingQueueStorage",
    # Pricing
    "PricingCalculator",
    "TokenBasedPricing",
]
