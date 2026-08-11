"""External-oracle and private-campaign adapters; never imported by the core."""

from .parity_metrics import boolean_parity, first_divergence, float_parity

__all__ = ["boolean_parity", "first_divergence", "float_parity"]
