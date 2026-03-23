"""Venue registry: register and retrieve venue adapters by type."""

from __future__ import annotations

import structlog

from ep2_crypto.execution.venue import VenueAdapter, VenueType

logger = structlog.get_logger(__name__)


class VenueRegistry:
    """Registry of available venue adapters.

    Usage:
        registry = VenueRegistry()
        registry.register(polymarket_adapter)
        adapter = registry.get(VenueType.POLYMARKET_BINARY)
    """

    def __init__(self) -> None:
        self._adapters: dict[VenueType, VenueAdapter] = {}

    def register(self, adapter: VenueAdapter) -> None:
        """Register a venue adapter."""
        venue = adapter.venue_type
        if venue in self._adapters:
            logger.warning("venue_adapter_replaced", venue=venue.value)
        self._adapters[venue] = adapter
        logger.info("venue_adapter_registered", venue=venue.value)

    def get(self, venue_type: VenueType) -> VenueAdapter:
        """Get a registered venue adapter.

        Raises:
            KeyError: If no adapter registered for the venue type.
        """
        if venue_type not in self._adapters:
            raise KeyError(
                f"No adapter registered for venue {venue_type.value}. "
                f"Available: {[v.value for v in self._adapters]}"
            )
        return self._adapters[venue_type]

    def list_venues(self) -> list[VenueType]:
        """List all registered venue types."""
        return list(self._adapters.keys())

    @property
    def count(self) -> int:
        return len(self._adapters)
