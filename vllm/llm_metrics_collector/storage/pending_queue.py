"""Pending queue storage for BFT Agent integration.

This storage backend creates individual JSONL files for each metric submission
in a pending directory. The BFT Agent monitors this directory, reads the files,
submits them to the BFT Coordinator, and moves them to a sent directory.

Directory structure:
    base_dir/
        pending/
            req-123.jsonl
            req-456.jsonl
        sent/
            req-123.jsonl
            req-456.jsonl
        failed/
            req-789.jsonl

This design ensures:
- No data loss (files persist until successfully sent)
- Simple file-based communication between Python and Rust
- Easy debugging (inspect individual files)
- Automatic retry (failed files can be moved back to pending)
"""

import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from vllm.llm_metrics_collector.models.request_metrics import RequestMetrics


class PendingQueueStorage:
    """File-based pending queue for metrics awaiting BFT verification.

    Usage:
        # In vLLM Metrics Collector
        queue = PendingQueueStorage("/var/log/vllm/metrics")
        queue.enqueue(metrics)

        # In BFT Agent (Rust reads from pending/ directory)
        # After successful submission:
        queue.mark_sent("req-123")
    """

    def __init__(self, base_dir: str):
        """Initialize pending queue storage.

        Args:
            base_dir: Base directory for queue files
        """
        self.base_dir = Path(base_dir)
        self.pending_dir = self.base_dir / "pending"
        self.sent_dir = self.base_dir / "sent"
        self.failed_dir = self.base_dir / "failed"

        # Create directories
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.sent_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)

    def enqueue(self, metrics: RequestMetrics) -> Path:
        """Add metrics to pending queue.

        Args:
            metrics: RequestMetrics to enqueue

        Returns:
            Path to the created file
        """
        # Create filename from request_id
        filename = f"{metrics.request_id}.jsonl"
        file_path = self.pending_dir / filename

        # Write metrics to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(metrics.to_json())

        return file_path

    def enqueue_batch(self, metrics_list: List[RequestMetrics]) -> List[Path]:
        """Add multiple metrics to pending queue.

        Args:
            metrics_list: List of RequestMetrics to enqueue

        Returns:
            List of paths to created files
        """
        return [self.enqueue(metrics) for metrics in metrics_list]

    def get_pending_files(self) -> List[Path]:
        """Get list of all pending files.

        Returns:
            List of pending file paths, sorted by modification time (oldest first)
        """
        files = list(self.pending_dir.glob("*.jsonl"))
        # Sort by modification time (oldest first for FIFO processing)
        return sorted(files, key=lambda p: p.stat().st_mtime)

    def read_pending(self, request_id: str) -> Optional[RequestMetrics]:
        """Read a specific pending metrics file.

        Args:
            request_id: Request ID to read

        Returns:
            RequestMetrics if found, None otherwise
        """
        file_path = self.pending_dir / f"{request_id}.jsonl"

        if not file_path.exists():
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                json_line = f.read().strip()
                return RequestMetrics.from_json(json_line)
        except Exception as e:
            print(f"Error reading pending file {file_path}: {e}")
            return None

    def mark_sent(self, request_id: str) -> bool:
        """Move metrics from pending to sent directory.

        Args:
            request_id: Request ID to mark as sent

        Returns:
            True if successful, False otherwise
        """
        src_path = self.pending_dir / f"{request_id}.jsonl"

        if not src_path.exists():
            return False

        # Add timestamp to sent filename to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst_path = self.sent_dir / f"{request_id}_{timestamp}.jsonl"

        try:
            src_path.rename(dst_path)
            return True
        except Exception as e:
            print(f"Error moving file to sent: {e}")
            return False

    def mark_failed(self, request_id: str, error_message: str = "") -> bool:
        """Move metrics from pending to failed directory.

        Args:
            request_id: Request ID to mark as failed
            error_message: Optional error message to append

        Returns:
            True if successful, False otherwise
        """
        src_path = self.pending_dir / f"{request_id}.jsonl"

        if not src_path.exists():
            return False

        # Add timestamp to failed filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst_path = self.failed_dir / f"{request_id}_{timestamp}.jsonl"

        try:
            # If error message provided, append it as a second JSON line
            if error_message:
                with open(src_path, "a", encoding="utf-8") as f:
                    error_info = {
                        "error": error_message,
                        "failed_at": timestamp
                    }
                    f.write("\n" + json.dumps(error_info))

            src_path.rename(dst_path)
            return True
        except Exception as e:
            print(f"Error moving file to failed: {e}")
            return False

    def delete_pending(self, request_id: str) -> bool:
        """Delete a pending metrics file.

        Args:
            request_id: Request ID to delete

        Returns:
            True if successful, False otherwise
        """
        file_path = self.pending_dir / f"{request_id}.jsonl"

        if not file_path.exists():
            return False

        try:
            file_path.unlink()
            return True
        except Exception as e:
            print(f"Error deleting pending file: {e}")
            return False

    def count_pending(self) -> int:
        """Count number of pending files.

        Returns:
            Number of pending files
        """
        return len(list(self.pending_dir.glob("*.jsonl")))

    def count_sent(self) -> int:
        """Count number of sent files.

        Returns:
            Number of sent files
        """
        return len(list(self.sent_dir.glob("*.jsonl")))

    def count_failed(self) -> int:
        """Count number of failed files.

        Returns:
            Number of failed files
        """
        return len(list(self.failed_dir.glob("*.jsonl")))

    def clear_sent(self, older_than_days: int = 7) -> int:
        """Clear old sent files.

        Args:
            older_than_days: Delete files older than this many days

        Returns:
            Number of files deleted
        """
        import time
        cutoff_time = time.time() - (older_than_days * 24 * 3600)
        deleted = 0

        for file_path in self.sent_dir.glob("*.jsonl"):
            if file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    deleted += 1
                except Exception as e:
                    print(f"Error deleting sent file {file_path}: {e}")

        return deleted

    def clear_failed(self, older_than_days: int = 30) -> int:
        """Clear old failed files.

        Args:
            older_than_days: Delete files older than this many days

        Returns:
            Number of files deleted
        """
        import time
        cutoff_time = time.time() - (older_than_days * 24 * 3600)
        deleted = 0

        for file_path in self.failed_dir.glob("*.jsonl"):
            if file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    deleted += 1
                except Exception as e:
                    print(f"Error deleting failed file {file_path}: {e}")

        return deleted

    def get_stats(self) -> dict:
        """Get queue statistics.

        Returns:
            Dictionary with queue stats
        """
        return {
            "pending": self.count_pending(),
            "sent": self.count_sent(),
            "failed": self.count_failed(),
            "pending_dir": str(self.pending_dir),
            "sent_dir": str(self.sent_dir),
            "failed_dir": str(self.failed_dir),
        }
