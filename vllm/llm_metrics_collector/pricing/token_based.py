"""Token-based pricing calculator."""

from vllm.llm_metrics_collector.pricing.base import PricingCalculator
from vllm.llm_metrics_collector.models.request_metrics import RequestMetrics


class TokenBasedPricing(PricingCalculator):
    """Token-based pricing calculator.

    This is the most common pricing model, used by OpenAI, Anthropic, etc.
    Charges based on:
    - Input tokens (prompt)
    - Output tokens (completion)
    - Optional cache discount for cached tokens

    Pricing formula:
        input_cost = (uncached_tokens / 1000) * input_price_per_1k
        output_cost = (completion_tokens / 1000) * output_price_per_1k
        cache_discount = (cached_tokens / 1000) * cache_discount_per_1k
        total_cost = input_cost + output_cost - cache_discount

    Optional tier multipliers can be applied for REALTIME/BATCH tiers.

    Example pricing (GPT-4 style):
        pricing = TokenBasedPricing(
            input_price_per_1k=0.03,    # $30/1M tokens
            output_price_per_1k=0.06,   # $60/1M tokens
            cache_discount_per_1k=0.015 # 50% discount on cached
        )

    Example pricing (GPT-3.5 style):
        pricing = TokenBasedPricing(
            input_price_per_1k=0.0005,   # $0.50/1M tokens
            output_price_per_1k=0.0015,  # $1.50/1M tokens
            cache_discount_per_1k=0.00025
        )
    """

    def __init__(
        self,
        input_price_per_1k: float,
        output_price_per_1k: float,
        cache_discount_per_1k: float = 0.0,
        apply_tier_multiplier: bool = False,
    ):
        """Initialize token-based pricing.

        Args:
            input_price_per_1k: Price per 1,000 input tokens (in dollars)
            output_price_per_1k: Price per 1,000 output tokens (in dollars)
            cache_discount_per_1k: Discount per 1,000 cached tokens (in dollars)
            apply_tier_multiplier: Whether to apply tier multipliers
        """
        self.input_price_per_1k = input_price_per_1k
        self.output_price_per_1k = output_price_per_1k
        self.cache_discount_per_1k = cache_discount_per_1k
        self.should_apply_tier_multiplier = apply_tier_multiplier

    def calculate_cost(self, metrics: RequestMetrics) -> float:
        """Calculate total cost.

        Args:
            metrics: Request metrics

        Returns:
            Total cost in dollars
        """
        input_cost = self.calculate_input_cost(metrics)
        output_cost = self.calculate_output_cost(metrics)
        cache_discount = self.calculate_cache_discount(metrics)

        base_cost = input_cost + output_cost - cache_discount

        if self.should_apply_tier_multiplier:
            return self.apply_tier_multiplier(base_cost, metrics)
        else:
            return base_cost

    def calculate_input_cost(self, metrics: RequestMetrics) -> float:
        """Calculate input token cost.

        Only uncached tokens are charged.

        Args:
            metrics: Request metrics

        Returns:
            Input cost in dollars
        """
        return (metrics.uncached_tokens / 1000.0) * self.input_price_per_1k

    def calculate_output_cost(self, metrics: RequestMetrics) -> float:
        """Calculate output token cost.

        Args:
            metrics: Request metrics

        Returns:
            Output cost in dollars
        """
        return (metrics.completion_tokens / 1000.0) * self.output_price_per_1k

    def calculate_cache_discount(self, metrics: RequestMetrics) -> float:
        """Calculate cache discount.

        Args:
            metrics: Request metrics

        Returns:
            Cache discount in dollars
        """
        return (metrics.cached_tokens / 1000.0) * self.cache_discount_per_1k

    def get_price_summary(self) -> str:
        """Get human-readable price summary.

        Returns:
            Price summary string
        """
        return (f"Token-based Pricing:\n"
                f"  Input: ${self.input_price_per_1k:.4f} per 1K tokens\n"
                f"  Output: ${self.output_price_per_1k:.4f} per 1K tokens\n"
                f"  Cache Discount: ${self.cache_discount_per_1k:.4f} per 1K tokens\n"
                f"  Tier Multiplier: {'Enabled' if self.should_apply_tier_multiplier else 'Disabled'}"
                )

    def estimate_cost_for_tokens(self, prompt_tokens: int,
                                  completion_tokens: int,
                                  cached_tokens: int = 0) -> float:
        """Estimate cost for given token counts.

        Useful for pre-request cost estimation.

        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            cached_tokens: Number of cached tokens

        Returns:
            Estimated cost in dollars
        """
        uncached_tokens = max(0, prompt_tokens - cached_tokens)
        input_cost = (uncached_tokens / 1000.0) * self.input_price_per_1k
        output_cost = (completion_tokens / 1000.0) * self.output_price_per_1k
        cache_discount = (cached_tokens /
                          1000.0) * self.cache_discount_per_1k

        return input_cost + output_cost - cache_discount

    def __repr__(self) -> str:
        """String representation."""
        return (f"TokenBasedPricing(input=${self.input_price_per_1k:.4f}/1k, "
                f"output=${self.output_price_per_1k:.4f}/1k, "
                f"cache_discount=${self.cache_discount_per_1k:.4f}/1k)")
