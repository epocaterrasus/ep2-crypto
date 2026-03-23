"""Multi-tier alerting system with rate limiting.

Tiers:
  INFO:      Daily summary (max 1/hour)
  WARNING:   Alpha decay, drift, slippage anomaly (max 5/hour)
  CRITICAL:  Kill switch, model validation fail, API down (unlimited)
  EMERGENCY: Exchange error, position stuck, catastrophic loss (unlimited)

Backends:
  - Telegram bot (primary)
  - Slack webhook (secondary, optional)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Protocol

import structlog

logger = structlog.get_logger(__name__)


class AlertTier(IntEnum):
    INFO = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3


@dataclass(frozen=True)
class Alert:
    """A single alert to be sent."""

    tier: AlertTier
    title: str
    message: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def formatted(self) -> str:
        prefix = {
            AlertTier.INFO: "[i]",
            AlertTier.WARNING: "⚠️",
            AlertTier.CRITICAL: "🔴",
            AlertTier.EMERGENCY: "🚨",
        }[self.tier]
        return f"{prefix} [{self.tier.name}] {self.title}\n{self.message}"


class AlertSender(Protocol):
    """Protocol for alert delivery backends."""

    def send(self, alert: Alert) -> bool:
        """Send an alert. Returns True if delivered successfully."""
        ...


class TelegramSender:
    """Sends alerts via Telegram Bot API.

    Requires EP2_TELEGRAM_BOT_TOKEN and EP2_TELEGRAM_CHAT_ID env vars.
    Actual HTTP calls are deferred to avoid import-time deps.
    """

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._enabled = bool(bot_token and chat_id)

    def send(self, alert: Alert) -> bool:
        if not self._enabled:
            logger.debug("telegram_disabled", title=alert.title)
            return False

        try:
            import json
            import urllib.request

            url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
            payload = json.dumps({
                "chat_id": self._chat_id,
                "text": alert.formatted,
                "parse_mode": "HTML",
            }).encode()
            req = urllib.request.Request(  # noqa: S310
                url, data=payload, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                return resp.status == 200
        except Exception:
            logger.exception("telegram_send_failed", title=alert.title)
            return False

    @property
    def enabled(self) -> bool:
        return self._enabled


class SlackSender:
    """Sends alerts via Slack incoming webhook.

    Requires EP2_SLACK_WEBHOOK_URL env var.
    """

    def __init__(self, webhook_url: str) -> None:
        self._webhook_url = webhook_url
        self._enabled = bool(webhook_url)

    def send(self, alert: Alert) -> bool:
        if not self._enabled:
            logger.debug("slack_disabled", title=alert.title)
            return False

        try:
            import json
            import urllib.request

            payload = json.dumps({"text": alert.formatted}).encode()
            req = urllib.request.Request(  # noqa: S310
                self._webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                return resp.status == 200
        except Exception:
            logger.exception("slack_send_failed", title=alert.title)
            return False

    @property
    def enabled(self) -> bool:
        return self._enabled


class RateLimiter:
    """Token-bucket rate limiter per alert tier."""

    def __init__(self, limits: dict[AlertTier, int] | None = None) -> None:
        # Default: max alerts per hour
        self._limits = limits or {
            AlertTier.INFO: 1,
            AlertTier.WARNING: 5,
            AlertTier.CRITICAL: 0,  # 0 = unlimited
            AlertTier.EMERGENCY: 0,
        }
        self._history: dict[AlertTier, list[float]] = {t: [] for t in AlertTier}

    def allow(self, tier: AlertTier) -> bool:
        """Check if an alert of this tier is allowed right now."""
        limit = self._limits.get(tier, 0)
        if limit == 0:
            return True  # Unlimited

        now = time.time()
        hour_ago = now - 3600
        # Prune old entries
        self._history[tier] = [t for t in self._history[tier] if t > hour_ago]

        return len(self._history[tier]) < limit

    def record(self, tier: AlertTier) -> None:
        """Record that an alert was sent."""
        self._history[tier].append(time.time())

    def get_remaining(self, tier: AlertTier) -> int | None:
        """Get remaining allowed alerts this hour. None = unlimited."""
        limit = self._limits.get(tier, 0)
        if limit == 0:
            return None
        now = time.time()
        hour_ago = now - 3600
        recent = len([t for t in self._history[tier] if t > hour_ago])
        return max(0, limit - recent)


class AlertManager:
    """Orchestrates alert delivery with rate limiting and multiple backends.

    Primary: Telegram
    Secondary: Slack (optional, for redundancy)
    """

    def __init__(
        self,
        senders: list[AlertSender] | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._senders = senders or []
        self._rate_limiter = rate_limiter or RateLimiter()
        self._sent_count = 0
        self._suppressed_count = 0
        self._history: list[Alert] = []

    def send(self, alert: Alert) -> bool:
        """Send an alert through all backends, respecting rate limits.

        Returns True if at least one backend delivered successfully.
        """
        if not self._rate_limiter.allow(alert.tier):
            self._suppressed_count += 1
            logger.info(
                "alert_rate_limited",
                tier=alert.tier.name,
                title=alert.title,
            )
            return False

        self._rate_limiter.record(alert.tier)
        self._history.append(alert)

        delivered = False
        for sender in self._senders:
            try:
                if sender.send(alert):
                    delivered = True
            except Exception:
                logger.exception(
                    "alert_sender_failed",
                    sender=type(sender).__name__,
                )

        if delivered:
            self._sent_count += 1
        else:
            # Log locally even if no backend delivered
            logger.warning(
                "alert_no_delivery",
                tier=alert.tier.name,
                title=alert.title,
                message=alert.message,
            )

        return delivered

    def send_info(self, title: str, message: str, **metadata: Any) -> bool:
        return self.send(Alert(
            tier=AlertTier.INFO, title=title, message=message, metadata=metadata
        ))

    def send_warning(self, title: str, message: str, **metadata: Any) -> bool:
        return self.send(Alert(
            tier=AlertTier.WARNING, title=title, message=message, metadata=metadata
        ))

    def send_critical(self, title: str, message: str, **metadata: Any) -> bool:
        return self.send(Alert(
            tier=AlertTier.CRITICAL, title=title, message=message, metadata=metadata
        ))

    def send_emergency(self, title: str, message: str, **metadata: Any) -> bool:
        return self.send(Alert(
            tier=AlertTier.EMERGENCY, title=title, message=message, metadata=metadata
        ))

    @property
    def sent_count(self) -> int:
        return self._sent_count

    @property
    def suppressed_count(self) -> int:
        return self._suppressed_count

    @property
    def history(self) -> list[Alert]:
        return list(self._history)
