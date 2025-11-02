"""File-based storage backend using JSONL format."""

import json
import gzip
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from vllm.llm_metrics_collector.storage.base import StorageBackend
from vllm.llm_metrics_collector.models.request_metrics import RequestMetrics


class FileStorageBackend(StorageBackend):
    """File-based storage backend using JSONL format.

    Features:
    - JSONL (JSON Lines) format for easy appending
    - Date-based partitioning (hour/day/week/month)
    - Optional gzip compression
    - Efficient for append-only workloads
    - Easy to process with standard tools

    File structure:
        base_dir/
            2025-11-02.jsonl      # Daily partitioning
            2025-11-03.jsonl
            ...

    Or with hourly partitioning:
        base_dir/
            2025-11-02/
                00.jsonl
                01.jsonl
                ...

    Usage:
        storage = FileStorageBackend(
            base_dir="/var/log/vllm/metrics",
            partition_by="day",
            compress=False
        )
        storage.save(metrics)
    """

    def __init__(
        self,
        base_dir: str,
        partition_by: str = "day",  # hour, day, week, month
        compress: bool = False,
    ):
        """Initialize file storage backend.

        Args:
            base_dir: Base directory for storing files
            partition_by: Partition granularity (hour, day, week, month)
            compress: Whether to use gzip compression
        """
        self.base_dir = Path(base_dir)
        self.partition_by = partition_by
        self.compress = compress

        # Create base directory if it doesn't exist
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Validate partition_by
        if partition_by not in ("hour", "day", "week", "month"):
            raise ValueError(
                f"Invalid partition_by: {partition_by}. "
                f"Must be one of: hour, day, week, month")

    def save(self, metrics: RequestMetrics) -> None:
        """Save a single metrics record.

        Args:
            metrics: RequestMetrics to save
        """
        self.save_batch([metrics])

    def save_batch(self, metrics_list: List[RequestMetrics]) -> None:
        """Save multiple metrics records.

        Groups metrics by partition and writes to appropriate files.

        Args:
            metrics_list: List of RequestMetrics to save
        """
        # Group metrics by partition
        partition_groups: Dict[Path, List[RequestMetrics]] = {}

        for metrics in metrics_list:
            file_path = self._get_file_path(metrics.arrival_time)
            if file_path not in partition_groups:
                partition_groups[file_path] = []
            partition_groups[file_path].append(metrics)

        # Write each partition
        for file_path, metrics_group in partition_groups.items():
            self._append_to_file(file_path, metrics_group)

    def load(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[RequestMetrics]:
        """Load metrics records with filtering.

        Optimized to only read files within the time range if
        start_time/end_time filters are provided.

        Supported filters (same as MemoryStorageBackend):
        - request_id, user_id, organization_id, model_name
        - start_time, end_time
        - finish_reason
        - min_cost, max_cost

        Args:
            filters: Filter conditions
            limit: Maximum records to return
            offset: Records to skip

        Returns:
            List of matching RequestMetrics
        """
        # Determine which files to read based on time filters
        if filters and "start_time" in filters and "end_time" in filters:
            files_to_read = self._get_files_in_range(filters["start_time"],
                                                      filters["end_time"])
        else:
            files_to_read = self._get_all_files()

        # Read and filter metrics
        results: List[RequestMetrics] = []
        records_seen = 0

        for file_path in sorted(files_to_read):
            metrics_batch = self._read_from_file(file_path)

            for metrics in metrics_batch:
                # Apply filters
                if filters and not self._matches_filters(metrics, filters):
                    continue

                # Apply offset
                if records_seen < offset:
                    records_seen += 1
                    continue

                # Add to results
                results.append(metrics)
                records_seen += 1

                # Check limit
                if limit is not None and len(results) >= limit:
                    return results

        return results

    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count matching records.

        Args:
            filters: Filter conditions

        Returns:
            Number of matching records
        """
        # For simplicity, load all and count
        # In production, this could be optimized with indexing
        matching = self.load(filters=filters)
        return len(matching)

    def close(self) -> None:
        """Close storage (no-op for file backend)."""
        pass

    def _get_file_path(self, timestamp: float) -> Path:
        """Get file path for a given timestamp.

        Args:
            timestamp: Unix timestamp

        Returns:
            Path to the partition file
        """
        dt = datetime.fromtimestamp(timestamp)

        if self.partition_by == "hour":
            # e.g., 2025-11-02/14.jsonl
            date_dir = self.base_dir / dt.strftime("%Y-%m-%d")
            date_dir.mkdir(parents=True, exist_ok=True)
            filename = dt.strftime("%H.jsonl")
        elif self.partition_by == "day":
            # e.g., 2025-11-02.jsonl
            filename = dt.strftime("%Y-%m-%d.jsonl")
        elif self.partition_by == "week":
            # e.g., 2025-W44.jsonl (ISO week)
            filename = dt.strftime("%Y-W%W.jsonl")
        elif self.partition_by == "month":
            # e.g., 2025-11.jsonl
            filename = dt.strftime("%Y-%m.jsonl")
        else:
            raise ValueError(f"Invalid partition_by: {self.partition_by}")

        if self.compress:
            filename += ".gz"

        if self.partition_by == "hour":
            return date_dir / filename
        else:
            return self.base_dir / filename

    def _append_to_file(self, file_path: Path,
                        metrics_list: List[RequestMetrics]) -> None:
        """Append metrics to a file.

        Args:
            file_path: Path to the file
            metrics_list: Metrics to append
        """
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Open file for appending
        if self.compress:
            with gzip.open(file_path, "at", encoding="utf-8") as f:
                for metrics in metrics_list:
                    json_line = metrics.to_json()
                    f.write(json_line + "\n")
        else:
            with open(file_path, "a", encoding="utf-8") as f:
                for metrics in metrics_list:
                    json_line = metrics.to_json()
                    f.write(json_line + "\n")

    def _read_from_file(self, file_path: Path) -> List[RequestMetrics]:
        """Read all metrics from a file.

        Args:
            file_path: Path to the file

        Returns:
            List of RequestMetrics
        """
        if not file_path.exists():
            return []

        metrics_list: List[RequestMetrics] = []

        try:
            if self.compress or file_path.suffix == ".gz":
                with gzip.open(file_path, "rt", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            metrics = RequestMetrics.from_json(line)
                            metrics_list.append(metrics)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            metrics = RequestMetrics.from_json(line)
                            metrics_list.append(metrics)
        except Exception as e:
            # Log error but don't fail completely
            print(f"Error reading {file_path}: {e}")

        return metrics_list

    def _get_all_files(self) -> List[Path]:
        """Get all JSONL files in the base directory.

        Returns:
            List of file paths
        """
        if self.compress:
            pattern = "*.jsonl.gz"
        else:
            pattern = "*.jsonl"

        files = list(self.base_dir.glob(pattern))

        # For hourly partitioning, also check subdirectories
        if self.partition_by == "hour":
            for date_dir in self.base_dir.glob("*"):
                if date_dir.is_dir():
                    files.extend(date_dir.glob(pattern))

        return files

    def _get_files_in_range(self, start_time: float,
                            end_time: float) -> List[Path]:
        """Get files that might contain data in the time range.

        Args:
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            List of file paths to read
        """
        # For simplicity, just return all files
        # In production, this could be optimized based on partition format
        return self._get_all_files()

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
