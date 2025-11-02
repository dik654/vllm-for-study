"""In-memory storage backend for testing and development."""

from typing import List, Dict, Any, Optional

from vllm.llm_metrics_collector.storage.base import StorageBackend
from vllm.llm_metrics_collector.models.request_metrics import RequestMetrics


class MemoryStorageBackend(StorageBackend):
    """In-memory storage backend.

    Stores all metrics in a Python list. Suitable for:
    - Testing
    - Development
    - Small-scale deployments
    - Temporary caching

    Not suitable for:
    - Production with high volume
    - Persistence across restarts
    - Multi-process scenarios

    Usage:
        storage = MemoryStorageBackend()
        storage.save(metrics)
        results = storage.load(filters={"user_id": "user-123"})
    """

    def __init__(self):
        """Initialize in-memory storage."""
        self._storage: List[RequestMetrics] = []

    def save(self, metrics: RequestMetrics) -> None:
        """Save a single metrics record.

        Args:
            metrics: RequestMetrics to save
        """
        self._storage.append(metrics)

    def save_batch(self, metrics_list: List[RequestMetrics]) -> None:
        """Save multiple metrics records.

        Args:
            metrics_list: List of RequestMetrics to save
        """
        self._storage.extend(metrics_list)

    def load(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[RequestMetrics]:
        """Load metrics records with filtering.

        Supported filters:
        - request_id: Exact match
        - user_id: Exact match
        - organization_id: Exact match
        - model_name: Exact match
        - start_time: arrival_time >= start_time
        - end_time: arrival_time <= end_time
        - finish_reason: Exact match (as string)
        - min_cost: estimated_cost >= min_cost
        - max_cost: estimated_cost <= max_cost

        Args:
            filters: Filter conditions
            limit: Maximum records to return
            offset: Records to skip

        Returns:
            List of matching RequestMetrics
        """
        # Start with all records
        results = self._storage

        # Apply filters
        if filters:
            results = [m for m in results if self._matches_filters(m, filters)]

        # Apply offset and limit
        if offset > 0:
            results = results[offset:]
        if limit is not None:
            results = results[:limit]

        return results

    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count matching records.

        Args:
            filters: Filter conditions

        Returns:
            Number of matching records
        """
        if not filters:
            return len(self._storage)

        return sum(1 for m in self._storage
                   if self._matches_filters(m, filters))

    def close(self) -> None:
        """Close storage (no-op for memory backend)."""
        pass

    def clear(self) -> None:
        """Clear all stored metrics.

        Useful for testing.
        """
        self._storage.clear()

    def get_all(self) -> List[RequestMetrics]:
        """Get all stored metrics.

        Returns:
            List of all RequestMetrics
        """
        return self._storage.copy()

    def _matches_filters(self, metrics: RequestMetrics,
                         filters: Dict[str, Any]) -> bool:
        """Check if metrics match all filter conditions.

        Args:
            metrics: Metrics to check
            filters: Filter conditions

        Returns:
            True if all filters match
        """
        for key, value in filters.items():
            if key == "request_id":
                if metrics.request_id != value:
                    return False
            elif key == "user_id":
                if metrics.user_id != value:
                    return False
            elif key == "organization_id":
                if metrics.organization_id != value:
                    return False
            elif key == "model_name":
                if metrics.model_name != value:
                    return False
            elif key == "start_time":
                if metrics.arrival_time < value:
                    return False
            elif key == "end_time":
                if metrics.arrival_time > value:
                    return False
            elif key == "finish_reason":
                if metrics.finish_reason.value != value:
                    return False
            elif key == "min_cost":
                if metrics.estimated_cost < value:
                    return False
            elif key == "max_cost":
                if metrics.estimated_cost > value:
                    return False

        return True
