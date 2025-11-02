"""Base class for pricing calculators."""

from abc import ABC, abstractmethod

from vllm.llm_metrics_collector.models.request_metrics import RequestMetrics


class PricingCalculator(ABC):
    """Abstract base class for pricing calculators.

    Pricing calculators are responsible for:
    1. Calculating total cost for a request
    2. Breaking down costs by component (input, output, cache, etc.)
    3. Applying discounts and surcharges
    4. Supporting different pricing models

    Subclasses should implement the calculate_cost() method and
    optionally override calculate_input_cost() and calculate_output_cost()
    for more detailed breakdowns.
    """

    @abstractmethod
    def calculate_cost(self, metrics: RequestMetrics) -> float:
        """Calculate total cost for a request.

        Args:
            metrics: Request metrics

        Returns:
            Total cost in dollars
        """
        pass

    def calculate_input_cost(self, metrics: RequestMetrics) -> float:
        """Calculate input token cost.

        Default implementation returns 0. Override for detailed breakdown.

        Args:
            metrics: Request metrics

        Returns:
            Input cost in dollars
        """
        return 0.0

    def calculate_output_cost(self, metrics: RequestMetrics) -> float:
        """Calculate output token cost.

        Default implementation returns 0. Override for detailed breakdown.

        Args:
            metrics: Request metrics

        Returns:
            Output cost in dollars
        """
        return 0.0

    def calculate_cache_discount(self, metrics: RequestMetrics) -> float:
        """Calculate cache discount.

        Default implementation returns 0. Override if supporting cache discounts.

        Args:
            metrics: Request metrics

        Returns:
            Cache discount in dollars (positive value)
        """
        return 0.0

    def apply_tier_multiplier(self, base_cost: float,
                              metrics: RequestMetrics) -> float:
        """Apply pricing tier multiplier.

        Default multipliers:
        - REALTIME: 2.0x (premium for guaranteed low latency)
        - STANDARD: 1.0x (normal pricing)
        - BATCH: 0.5x (50% discount for batch/async)

        Args:
            base_cost: Base cost before tier adjustment
            metrics: Request metrics

        Returns:
            Adjusted cost
        """
        from vllm.llm_metrics_collector.models.enums import PricingTier

        multipliers = {
            PricingTier.REALTIME: 2.0,
            PricingTier.STANDARD: 1.0,
            PricingTier.BATCH: 0.5,
        }

        multiplier = multipliers.get(metrics.pricing_tier, 1.0)
        return base_cost * multiplier
