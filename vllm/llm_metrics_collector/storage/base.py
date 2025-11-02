"""Base class for storage backends."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from vllm.llm_metrics_collector.models.request_metrics import RequestMetrics


class StorageBackend(ABC):
    """Abstract base class for metrics storage backends.

    Storage backends are responsible for persisting RequestMetrics data
    and providing query capabilities. Different backends can be used
    for different use cases:
    - MemoryStorageBackend: Testing and development
    - FileStorageBackend: Simple file-based storage (JSONL)
    - DatabaseStorageBackend: SQL database for complex queries
    - CloudStorageBackend: S3/GCS for scalable storage

    All storage backends must implement the core CRUD operations.
    """

    @abstractmethod
    def save(self, metrics: RequestMetrics) -> None:
        """Save a single metrics record.

        Args:
            metrics: RequestMetrics to save

        Raises:
            IOError: If save operation fails
        """
        pass

    @abstractmethod
    def save_batch(self, metrics_list: List[RequestMetrics]) -> None:
        """Save multiple metrics records in a batch.

        This is more efficient than calling save() multiple times.

        Args:
            metrics_list: List of RequestMetrics to save

        Raises:
            IOError: If save operation fails
        """
        pass

    @abstractmethod
    def load(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[RequestMetrics]:
        """Load metrics records with optional filtering.

        Args:
            filters: Dictionary of filter conditions, e.g.:
                {
                    "user_id": "user-123",
                    "model_name": "gpt-3.5-turbo",
                    "start_time": 1234567890.0,
                    "end_time": 1234567900.0,
                }
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            List of RequestMetrics matching the filters

        Raises:
            IOError: If load operation fails
        """
        pass

    @abstractmethod
    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count metrics records matching filters.

        Args:
            filters: Dictionary of filter conditions (same as load())

        Returns:
            Number of matching records

        Raises:
            IOError: If count operation fails
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the storage backend and release resources.

        This should be called when done using the backend to ensure
        proper cleanup of file handles, database connections, etc.
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
