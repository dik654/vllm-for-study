"""Base class for metrics collectors."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from vllm.llm_metrics_collector.models.request_metrics import RequestMetrics


class MetricsCollectorBase(ABC):
    """Abstract base class for metrics collectors.

    Collectors are responsible for:
    1. Collecting metrics from various sources (vLLM, resources, etc.)
    2. Enriching metrics with additional information
    3. Validating collected metrics

    Subclasses should implement the collect() method to gather metrics
    from their specific source.
    """

    @abstractmethod
    def collect(self, **kwargs) -> RequestMetrics:
        """Collect metrics from the source.

        Args:
            **kwargs: Source-specific arguments for collection

        Returns:
            RequestMetrics object with collected data

        Raises:
            ValueError: If required data is missing or invalid
        """
        pass

    def enrich(self, metrics: RequestMetrics, **kwargs) -> RequestMetrics:
        """Enrich metrics with additional information.

        This method can be overridden to add extra data to metrics,
        such as:
        - User information lookup
        - Cost calculation
        - Resource usage estimation
        - Custom metadata

        Args:
            metrics: Metrics to enrich
            **kwargs: Additional context for enrichment

        Returns:
            Enriched RequestMetrics object
        """
        return metrics

    def validate(self, metrics: RequestMetrics) -> bool:
        """Validate collected metrics.

        Args:
            metrics: Metrics to validate

        Returns:
            True if valid, False otherwise
        """
        return metrics.validate()

    def collect_and_enrich(self, **kwargs) -> RequestMetrics:
        """Convenience method to collect and enrich in one call.

        Args:
            **kwargs: Arguments passed to collect() and enrich()

        Returns:
            Enriched RequestMetrics object

        Raises:
            ValueError: If metrics are invalid after collection
        """
        metrics = self.collect(**kwargs)
        metrics = self.enrich(metrics, **kwargs)

        if not self.validate(metrics):
            raise ValueError(f"Invalid metrics collected: {metrics}")

        return metrics
