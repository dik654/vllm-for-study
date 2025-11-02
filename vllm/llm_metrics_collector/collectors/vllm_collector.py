"""vLLM metrics collector."""

import time
from typing import Optional, Dict, Any

from vllm.llm_metrics_collector.collectors.base import MetricsCollectorBase
from vllm.llm_metrics_collector.models.request_metrics import RequestMetrics
from vllm.llm_metrics_collector.models.enums import FinishReason, PricingTier

try:
    from vllm.v1.metrics.stats import FinishedRequestStats
    from vllm.v1.engine import FinishReason as VLLMFinishReason
except ImportError:
    # Fallback for testing or when vLLM is not available
    FinishedRequestStats = None
    VLLMFinishReason = None


class VLLMMetricsCollector(MetricsCollectorBase):
    """Collector for vLLM FinishedRequestStats.

    Converts vLLM's FinishedRequestStats into our RequestMetrics format.
    This collector handles the core metrics available from vLLM, while
    additional enrichment (user info, costs, etc.) can be done via enrich().

    Usage:
        collector = VLLMMetricsCollector(
            model_name="gpt-3.5-turbo",
            engine_id="engine-1"
        )
        metrics = collector.collect(
            finished_stats=stats,
            request_id="req-123",
            arrival_time=1234567890.0
        )
    """

    def __init__(
        self,
        model_name: str = "",
        model_version: str = "",
        engine_id: str = "",
    ):
        """Initialize collector.

        Args:
            model_name: Name of the model being used
            model_version: Version of the model (optional)
            engine_id: vLLM engine instance identifier
        """
        self.model_name = model_name
        self.model_version = model_version
        self.engine_id = engine_id

    def collect(self, **kwargs) -> RequestMetrics:
        """Collect metrics from vLLM FinishedRequestStats.

        Args:
            **kwargs: Must include:
                - finished_stats: FinishedRequestStats object
                - request_id: Unique request identifier
                - arrival_time: Wall-clock timestamp (optional)
                Additional optional fields:
                - user_id: User identifier
                - organization_id: Organization identifier
                - api_key_hash: Hashed API key
                - cached_tokens: Number of cached tokens
                - temperature: Sampling temperature
                - top_p: Nucleus sampling parameter
                - top_k: Top-k sampling parameter
                - n: Number of choices
                - stream: Whether streaming

        Returns:
            RequestMetrics object

        Raises:
            ValueError: If required arguments are missing
        """
        finished_stats = kwargs.get("finished_stats")
        if finished_stats is None:
            raise ValueError("finished_stats is required")

        request_id = kwargs.get("request_id")
        if not request_id:
            raise ValueError("request_id is required")

        # Extract basic identification
        arrival_time = kwargs.get("arrival_time", time.time())
        user_id = kwargs.get("user_id")
        organization_id = kwargs.get("organization_id")
        api_key_hash = kwargs.get("api_key_hash")

        # Extract token counts from FinishedRequestStats
        prompt_tokens = getattr(finished_stats, "num_prompt_tokens", 0)
        completion_tokens = getattr(finished_stats, "num_generation_tokens",
                                     0)
        cached_tokens = kwargs.get("cached_tokens", 0)

        # Extract timing metrics from FinishedRequestStats
        e2e_latency = getattr(finished_stats, "e2e_latency", 0.0)
        queued_time = getattr(finished_stats, "queued_time", 0.0)
        prefill_time = getattr(finished_stats, "prefill_time", 0.0)
        decode_time = getattr(finished_stats, "decode_time", 0.0)
        inference_time = getattr(finished_stats, "inference_time", 0.0)
        time_per_output_token = getattr(finished_stats,
                                         "mean_time_per_output_token", 0.0)

        # Calculate TTFT (prefill_time is a good approximation)
        # Note: vLLM V1 may have first_token_latency in RequestStateStats
        time_to_first_token = prefill_time

        # Extract request parameters
        max_tokens = getattr(finished_stats, "max_tokens_param", 0) or 0
        temperature = kwargs.get("temperature", 1.0)
        top_p = kwargs.get("top_p", 1.0)
        top_k = kwargs.get("top_k")
        n = kwargs.get("n", 1)
        stream = kwargs.get("stream", False)

        # Extract finish reason
        vllm_finish_reason = finished_stats.finish_reason
        finish_reason = self._convert_finish_reason(vllm_finish_reason)

        # Create RequestMetrics
        metrics = RequestMetrics(
            # Identification
            request_id=request_id,
            user_id=user_id,
            organization_id=organization_id,
            api_key_hash=api_key_hash,
            # Model information
            model_name=self.model_name,
            model_version=self.model_version,
            engine_id=self.engine_id,
            # Token usage
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cached_tokens=cached_tokens,
            uncached_tokens=max(0, prompt_tokens - cached_tokens),
            # Timing metrics
            arrival_time=arrival_time,
            queued_time=queued_time,
            prefill_time=prefill_time,
            decode_time=decode_time,
            inference_time=inference_time,
            e2e_latency=e2e_latency,
            time_to_first_token=time_to_first_token,
            time_per_output_token=time_per_output_token,
            # Request parameters
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            n=n,
            stream=stream,
            # Completion information
            finish_reason=finish_reason,
        )

        return metrics

    def enrich(self, metrics: RequestMetrics, **kwargs) -> RequestMetrics:
        """Enrich metrics with additional information.

        Can add:
        - Resource usage metrics (if available)
        - Cost calculation (if pricing calculator provided)
        - Custom metadata

        Args:
            metrics: Metrics to enrich
            **kwargs: Additional context (pricing_calculator, etc.)

        Returns:
            Enriched RequestMetrics
        """
        # Add resource metrics if provided
        kv_cache_blocks = kwargs.get("kv_cache_blocks_used")
        if kv_cache_blocks is not None:
            metrics.kv_cache_blocks_used = kv_cache_blocks

        kv_cache_memory = kwargs.get("kv_cache_memory_bytes")
        if kv_cache_memory is not None:
            metrics.kv_cache_memory_bytes = kv_cache_memory

        peak_gpu_memory = kwargs.get("peak_gpu_memory_bytes")
        if peak_gpu_memory is not None:
            metrics.peak_gpu_memory_bytes = peak_gpu_memory

        # Calculate costs if pricing calculator provided
        pricing_calculator = kwargs.get("pricing_calculator")
        if pricing_calculator:
            try:
                metrics.estimated_cost = pricing_calculator.calculate_cost(
                    metrics)
                metrics.input_cost = (
                    pricing_calculator.calculate_input_cost(metrics))
                metrics.output_cost = (
                    pricing_calculator.calculate_output_cost(metrics))
            except Exception:
                # Don't fail enrichment if pricing calculation fails
                pass

        # Set pricing tier based on TTFT if not already set
        if metrics.pricing_tier == PricingTier.STANDARD:
            if metrics.time_to_first_token < 1.0:
                metrics.pricing_tier = PricingTier.REALTIME
            elif metrics.time_to_first_token > 5.0:
                metrics.pricing_tier = PricingTier.BATCH

        return metrics

    def _convert_finish_reason(self, vllm_reason: Any) -> FinishReason:
        """Convert vLLM finish reason to our enum.

        Args:
            vllm_reason: vLLM FinishReason enum or string

        Returns:
            Our FinishReason enum
        """
        if vllm_reason is None:
            return FinishReason.ABORT

        # Convert to string if it's an enum
        if hasattr(vllm_reason, "name"):
            reason_str = vllm_reason.name.lower()
        elif hasattr(vllm_reason, "value"):
            reason_str = str(vllm_reason.value).lower()
        else:
            reason_str = str(vllm_reason).lower()

        # Map vLLM reasons to our enum
        return FinishReason.from_vllm_finish_reason(reason_str)
