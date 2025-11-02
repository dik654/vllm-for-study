"""Aggregated metrics data model."""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List


@dataclass
class AggregatedMetrics:
    """Aggregated metrics over a time window or group.

    Used for generating reports, dashboards, and billing summaries.
    Can be aggregated by:
    - User (for user-specific billing)
    - Organization (for org-level billing)
    - Model (for model performance analysis)
    - Time period (hour, day, week, month)

    Attributes:
        # Aggregation criteria
        aggregation_key: Key used for aggregation (e.g., user_id, model_name)
        time_window_start: Start of the time window
        time_window_end: End of the time window

        # Request statistics
        total_requests: Total number of requests
        successful_requests: Requests that completed successfully
        failed_requests: Requests that failed/aborted
        aborted_requests: Requests explicitly aborted
        preempted_requests: Requests that were preempted

        # Token statistics
        total_prompt_tokens: Sum of all prompt tokens
        total_completion_tokens: Sum of all completion tokens
        total_tokens: Sum of all tokens
        total_cached_tokens: Sum of all cached tokens
        total_uncached_tokens: Sum of all uncached tokens

        # Cost statistics
        total_cost: Sum of all costs
        total_input_cost: Sum of input token costs
        total_output_cost: Sum of output token costs
        total_cache_discount: Sum of cache discounts

        # Performance statistics
        avg_e2e_latency: Average end-to-end latency
        p50_e2e_latency: Median latency (50th percentile)
        p95_e2e_latency: 95th percentile latency
        p99_e2e_latency: 99th percentile latency
        avg_time_to_first_token: Average TTFT
        avg_time_per_output_token: Average TPOT
        avg_tokens_per_second: Average throughput

        # Resource usage statistics
        total_gpu_compute_seconds: Sum of GPU compute time
        avg_batch_size: Average batch size
        total_preemptions: Total number of preemptions
        avg_kv_cache_blocks: Average KV cache blocks used
        avg_prefix_cache_hit_rate: Average cache hit rate
    """

    # Aggregation criteria
    aggregation_key: str
    time_window_start: datetime
    time_window_end: datetime

    # Request statistics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    aborted_requests: int = 0
    preempted_requests: int = 0

    # Token statistics
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cached_tokens: int = 0
    total_uncached_tokens: int = 0

    # Cost statistics
    total_cost: float = 0.0
    total_input_cost: float = 0.0
    total_output_cost: float = 0.0
    total_cache_discount: float = 0.0

    # Performance statistics
    avg_e2e_latency: float = 0.0
    p50_e2e_latency: float = 0.0
    p95_e2e_latency: float = 0.0
    p99_e2e_latency: float = 0.0
    avg_time_to_first_token: float = 0.0
    avg_time_per_output_token: float = 0.0
    avg_tokens_per_second: float = 0.0

    # Resource usage statistics
    total_gpu_compute_seconds: float = 0.0
    avg_batch_size: float = 0.0
    total_preemptions: int = 0
    avg_kv_cache_blocks: float = 0.0
    avg_prefix_cache_hit_rate: float = 0.0

    def get_success_rate(self) -> float:
        """Calculate success rate.

        Returns:
            Success rate as fraction (0.0-1.0)
        """
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    def get_failure_rate(self) -> float:
        """Calculate failure rate.

        Returns:
            Failure rate as fraction (0.0-1.0)
        """
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    def get_cache_hit_rate(self) -> float:
        """Calculate overall cache hit rate.

        Returns:
            Cache hit rate as fraction (0.0-1.0)
        """
        if self.total_prompt_tokens == 0:
            return 0.0
        return self.total_cached_tokens / self.total_prompt_tokens

    def get_avg_cost_per_request(self) -> float:
        """Calculate average cost per request.

        Returns:
            Average cost in dollars
        """
        if self.total_requests == 0:
            return 0.0
        return self.total_cost / self.total_requests

    def get_avg_cost_per_1k_tokens(self) -> float:
        """Calculate average cost per 1K tokens.

        Returns:
            Average cost per 1K tokens in dollars
        """
        if self.total_tokens == 0:
            return 0.0
        return (self.total_cost / self.total_tokens) * 1000

    def get_window_duration_hours(self) -> float:
        """Get time window duration in hours.

        Returns:
            Duration in hours
        """
        delta = self.time_window_end - self.time_window_start
        return delta.total_seconds() / 3600

    def get_requests_per_hour(self) -> float:
        """Calculate requests per hour.

        Returns:
            Average requests per hour
        """
        hours = self.get_window_duration_hours()
        if hours == 0:
            return 0.0
        return self.total_requests / hours

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        data = asdict(self)
        # Convert datetime to ISO format strings
        data["time_window_start"] = self.time_window_start.isoformat()
        data["time_window_end"] = self.time_window_end.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AggregatedMetrics":
        """Create from dictionary.

        Args:
            data: Dictionary with aggregated metrics data

        Returns:
            AggregatedMetrics instance
        """
        # Convert ISO strings back to datetime
        if isinstance(data.get("time_window_start"), str):
            data["time_window_start"] = datetime.fromisoformat(
                data["time_window_start"])
        if isinstance(data.get("time_window_end"), str):
            data["time_window_end"] = datetime.fromisoformat(
                data["time_window_end"])

        return cls(**data)

    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert to JSON string.

        Args:
            indent: JSON indentation (None for compact)

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "AggregatedMetrics":
        """Create from JSON string.

        Args:
            json_str: JSON string

        Returns:
            AggregatedMetrics instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def to_report_string(self) -> str:
        """Generate human-readable report.

        Returns:
            Formatted text report
        """
        lines = [
            "=" * 60,
            "AGGREGATED METRICS REPORT",
            "=" * 60,
            f"Aggregation Key: {self.aggregation_key}",
            f"Time Window: {self.time_window_start.strftime('%Y-%m-%d %H:%M')} "
            f"to {self.time_window_end.strftime('%Y-%m-%d %H:%M')}",
            f"Duration: {self.get_window_duration_hours():.2f} hours",
            "",
            "REQUEST STATISTICS:",
            f"  Total Requests: {self.total_requests:,}",
            f"  Successful: {self.successful_requests:,} "
            f"({self.get_success_rate()*100:.1f}%)",
            f"  Failed: {self.failed_requests:,} "
            f"({self.get_failure_rate()*100:.1f}%)",
            f"  Aborted: {self.aborted_requests:,}",
            f"  Preempted: {self.preempted_requests:,}",
            f"  Requests/Hour: {self.get_requests_per_hour():.1f}",
            "",
            "TOKEN STATISTICS:",
            f"  Prompt Tokens: {self.total_prompt_tokens:,}",
            f"  Completion Tokens: {self.total_completion_tokens:,}",
            f"  Total Tokens: {self.total_tokens:,}",
            f"  Cached Tokens: {self.total_cached_tokens:,} "
            f"({self.get_cache_hit_rate()*100:.1f}% hit rate)",
            "",
            "COST STATISTICS:",
            f"  Total Cost: ${self.total_cost:.2f}",
            f"  Input Cost: ${self.total_input_cost:.2f}",
            f"  Output Cost: ${self.total_output_cost:.2f}",
            f"  Cache Discount: -${self.total_cache_discount:.2f}",
            f"  Avg Cost/Request: ${self.get_avg_cost_per_request():.4f}",
            f"  Avg Cost/1K Tokens: ${self.get_avg_cost_per_1k_tokens():.4f}",
            "",
            "PERFORMANCE STATISTICS:",
            f"  Avg E2E Latency: {self.avg_e2e_latency:.3f}s",
            f"  P50 Latency: {self.p50_e2e_latency:.3f}s",
            f"  P95 Latency: {self.p95_e2e_latency:.3f}s",
            f"  P99 Latency: {self.p99_e2e_latency:.3f}s",
            f"  Avg TTFT: {self.avg_time_to_first_token:.3f}s",
            f"  Avg TPOT: {self.avg_time_per_output_token:.4f}s",
            f"  Avg Throughput: {self.avg_tokens_per_second:.1f} tokens/s",
            "",
            "RESOURCE USAGE:",
            f"  Total GPU Compute: {self.total_gpu_compute_seconds:.1f}s "
            f"({self.total_gpu_compute_seconds/3600:.2f}h)",
            f"  Avg Batch Size: {self.avg_batch_size:.2f}",
            f"  Total Preemptions: {self.total_preemptions:,}",
            f"  Avg KV Cache Blocks: {self.avg_kv_cache_blocks:.1f}",
            f"  Avg Cache Hit Rate: {self.avg_prefix_cache_hit_rate*100:.1f}%",
            "=" * 60,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"AggregatedMetrics(key={self.aggregation_key!r}, "
            f"requests={self.total_requests:,}, "
            f"tokens={self.total_tokens:,}, "
            f"cost=${self.total_cost:.2f}, "
            f"window={self.time_window_start.date()} to "
            f"{self.time_window_end.date()})")
