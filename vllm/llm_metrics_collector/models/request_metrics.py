"""Request-level metrics data model."""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any

from vllm.llm_metrics_collector.models.enums import FinishReason, PricingTier


@dataclass
class RequestMetrics:
    """Comprehensive metrics for a single LLM request.

    This class captures all relevant metrics for usage tracking, pricing,
    and performance analysis. It's designed to be:
    - Compatible with vLLM's FinishedRequestStats
    - Compatible with OpenAI's usage format
    - Suitable for cost calculation
    - Suitable for performance analysis

    Attributes:
        # Identification
        request_id: Unique identifier for this request
        user_id: User identifier (optional, can be hashed)
        organization_id: Organization identifier (optional)
        api_key_hash: Hashed API key for tracking (optional)

        # Model information
        model_name: Name of the model used
        model_version: Version of the model (optional)
        engine_id: vLLM engine instance ID

        # Token usage (primary billing metrics)
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens generated
        total_tokens: Total tokens (prompt + completion)
        cached_tokens: Number of tokens served from cache
        uncached_tokens: Tokens that required computation

        # Timing metrics (seconds)
        arrival_time: Wall-clock timestamp when request arrived
        queued_time: Time spent waiting in queue
        prefill_time: Time for prefill/prompt processing phase
        decode_time: Time for token generation phase
        inference_time: Total inference time (prefill + decode)
        e2e_latency: End-to-end latency (queue + inference)
        time_to_first_token: Time until first token (TTFT)
        time_per_output_token: Average time per output token (TPOT)

        # Resource usage
        kv_cache_blocks_used: Number of KV cache blocks used
        kv_cache_memory_bytes: KV cache memory in bytes
        peak_gpu_memory_bytes: Peak GPU memory usage (optional)
        gpu_compute_time: GPU computation time in seconds
        num_preemptions: Number of times request was preempted

        # Request parameters
        max_tokens: Requested maximum tokens
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter (optional)
        n: Number of completion choices
        stream: Whether request was streaming

        # Completion information
        finish_reason: Why the request finished
        error_code: Error code if failed (optional)
        error_message: Error message if failed (optional)

        # Cost information (calculated by pricing calculator)
        estimated_cost: Total estimated cost
        input_cost: Cost for input tokens
        output_cost: Cost for output tokens
        cache_discount: Discount for cached tokens

        # Efficiency metrics
        prefix_cache_hit_rate: Cache hit rate (0.0-1.0)
        batch_size: Batch size when processed
        tokens_per_second: Token generation throughput
        pricing_tier: Pricing tier applied
    """

    # Identification
    request_id: str
    user_id: Optional[str] = None
    organization_id: Optional[str] = None
    api_key_hash: Optional[str] = None

    # Model information
    model_name: str = ""
    model_version: str = ""
    engine_id: str = ""

    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    uncached_tokens: int = 0

    # Timing metrics (seconds)
    arrival_time: float = 0.0
    queued_time: float = 0.0
    prefill_time: float = 0.0
    decode_time: float = 0.0
    inference_time: float = 0.0
    e2e_latency: float = 0.0
    time_to_first_token: float = 0.0
    time_per_output_token: float = 0.0

    # Resource usage
    kv_cache_blocks_used: int = 0
    kv_cache_memory_bytes: int = 0
    peak_gpu_memory_bytes: Optional[int] = None
    gpu_compute_time: float = 0.0
    num_preemptions: int = 0

    # Request parameters
    max_tokens: int = 0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    n: int = 1
    stream: bool = False

    # Completion information
    finish_reason: FinishReason = FinishReason.STOP
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    # Cost information
    estimated_cost: float = 0.0
    input_cost: float = 0.0
    output_cost: float = 0.0
    cache_discount: float = 0.0

    # Efficiency metrics
    prefix_cache_hit_rate: float = 0.0
    batch_size: int = 1
    tokens_per_second: float = 0.0
    pricing_tier: PricingTier = PricingTier.STANDARD

    def __post_init__(self):
        """Calculate derived fields after initialization."""
        # Calculate total tokens if not set
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens

        # Calculate uncached tokens if not set
        if self.uncached_tokens == 0:
            self.uncached_tokens = max(0, self.prompt_tokens - self.cached_tokens)

        # Calculate tokens per second if not set and decode time is available
        if (self.tokens_per_second == 0.0 and self.decode_time > 0
                and self.completion_tokens > 0):
            self.tokens_per_second = self.completion_tokens / self.decode_time

    def validate(self) -> bool:
        """Validate metrics data.

        Returns:
            True if all required fields are valid, False otherwise
        """
        # Required fields
        if not self.request_id:
            return False
        if not self.model_name:
            return False

        # Token counts should be non-negative
        if self.prompt_tokens < 0 or self.completion_tokens < 0:
            return False
        if self.total_tokens < 0 or self.cached_tokens < 0:
            return False

        # Cached tokens shouldn't exceed prompt tokens
        if self.cached_tokens > self.prompt_tokens:
            return False

        # Timing should be non-negative
        if any(t < 0 for t in [
                self.queued_time, self.prefill_time, self.decode_time,
                self.inference_time, self.e2e_latency
        ]):
            return False

        # Costs should be non-negative
        if any(c < 0 for c in
               [self.estimated_cost, self.input_cost, self.output_cost]):
            return False

        return True

    def is_successful(self) -> bool:
        """Check if request completed successfully.

        Returns:
            True if finished with STOP or LENGTH, False if ABORT
        """
        return self.finish_reason in (FinishReason.STOP, FinishReason.LENGTH)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation with enum values as strings
        """
        data = asdict(self)
        # Convert enums to strings
        data["finish_reason"] = self.finish_reason.value
        data["pricing_tier"] = self.pricing_tier.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RequestMetrics":
        """Create from dictionary.

        Args:
            data: Dictionary with metrics data

        Returns:
            RequestMetrics instance
        """
        # Convert string values back to enums
        if "finish_reason" in data and isinstance(data["finish_reason"], str):
            data["finish_reason"] = FinishReason(data["finish_reason"])
        if "pricing_tier" in data and isinstance(data["pricing_tier"], str):
            data["pricing_tier"] = PricingTier(data["pricing_tier"])

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
    def from_json(cls, json_str: str) -> "RequestMetrics":
        """Create from JSON string.

        Args:
            json_str: JSON string

        Returns:
            RequestMetrics instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def to_openai_usage(self) -> Dict[str, Any]:
        """Convert to OpenAI-compatible usage format.

        Returns:
            Dictionary matching OpenAI's UsageInfo format
        """
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "prompt_tokens_details": {
                "cached_tokens": self.cached_tokens,
            },
        }

    def get_billable_tokens(self) -> int:
        """Get number of tokens that should be billed.

        Typically this is uncached input tokens + all output tokens,
        but can be customized based on pricing model.

        Returns:
            Number of billable tokens
        """
        return self.uncached_tokens + self.completion_tokens

    def get_arrival_datetime(self) -> datetime:
        """Get arrival time as datetime object.

        Returns:
            Datetime object in UTC
        """
        return datetime.fromtimestamp(self.arrival_time)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"RequestMetrics(request_id={self.request_id!r}, "
            f"model={self.model_name!r}, "
            f"tokens={self.prompt_tokens}+{self.completion_tokens}="
            f"{self.total_tokens}, "
            f"latency={self.e2e_latency:.3f}s, "
            f"finish={self.finish_reason.value}, "
            f"cost=${self.estimated_cost:.4f})")
