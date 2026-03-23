"""Tests for alert system."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ep2_crypto.monitoring.alerts import (
    Alert,
    AlertManager,
    AlertTier,
    RateLimiter,
    SlackSender,
    TelegramSender,
)


class TestAlert:
    def test_formatted_info(self) -> None:
        alert = Alert(tier=AlertTier.INFO, title="Daily Summary", message="PnL +$50")
        assert "[INFO]" in alert.formatted
        assert "Daily Summary" in alert.formatted

    def test_formatted_emergency(self) -> None:
        alert = Alert(tier=AlertTier.EMERGENCY, title="Exchange Down", message="Binance WS lost")
        assert "[EMERGENCY]" in alert.formatted

    def test_all_tiers_format(self) -> None:
        for tier in AlertTier:
            alert = Alert(tier=tier, title="Test", message="msg")
            assert tier.name in alert.formatted


class TestRateLimiter:
    def test_info_limited_to_1_per_hour(self) -> None:
        limiter = RateLimiter()
        assert limiter.allow(AlertTier.INFO)
        limiter.record(AlertTier.INFO)
        assert not limiter.allow(AlertTier.INFO)

    def test_warning_limited_to_5_per_hour(self) -> None:
        limiter = RateLimiter()
        for _ in range(5):
            assert limiter.allow(AlertTier.WARNING)
            limiter.record(AlertTier.WARNING)
        assert not limiter.allow(AlertTier.WARNING)

    def test_critical_unlimited(self) -> None:
        limiter = RateLimiter()
        for _ in range(100):
            assert limiter.allow(AlertTier.CRITICAL)
            limiter.record(AlertTier.CRITICAL)

    def test_emergency_unlimited(self) -> None:
        limiter = RateLimiter()
        for _ in range(100):
            assert limiter.allow(AlertTier.EMERGENCY)
            limiter.record(AlertTier.EMERGENCY)

    def test_remaining_info(self) -> None:
        limiter = RateLimiter()
        assert limiter.get_remaining(AlertTier.INFO) == 1
        limiter.record(AlertTier.INFO)
        assert limiter.get_remaining(AlertTier.INFO) == 0

    def test_remaining_critical_unlimited(self) -> None:
        limiter = RateLimiter()
        assert limiter.get_remaining(AlertTier.CRITICAL) is None

    def test_custom_limits(self) -> None:
        limiter = RateLimiter(limits={
            AlertTier.INFO: 3,
            AlertTier.WARNING: 10,
            AlertTier.CRITICAL: 0,
            AlertTier.EMERGENCY: 0,
        })
        for _ in range(3):
            assert limiter.allow(AlertTier.INFO)
            limiter.record(AlertTier.INFO)
        assert not limiter.allow(AlertTier.INFO)


class TestTelegramSender:
    def test_disabled_when_no_token(self) -> None:
        sender = TelegramSender(bot_token="", chat_id="")
        assert not sender.enabled
        alert = Alert(tier=AlertTier.INFO, title="Test", message="msg")
        assert not sender.send(alert)

    def test_enabled_with_credentials(self) -> None:
        sender = TelegramSender(bot_token="tok123", chat_id="chat456")
        assert sender.enabled


class TestSlackSender:
    def test_disabled_when_no_url(self) -> None:
        sender = SlackSender(webhook_url="")
        assert not sender.enabled
        alert = Alert(tier=AlertTier.INFO, title="Test", message="msg")
        assert not sender.send(alert)

    def test_enabled_with_url(self) -> None:
        sender = SlackSender(webhook_url="https://hooks.slack.com/services/xxx")
        assert sender.enabled


class TestAlertManager:
    @pytest.fixture()
    def mock_sender(self) -> MagicMock:
        sender = MagicMock()
        sender.send.return_value = True
        return sender

    @pytest.fixture()
    def manager(self, mock_sender: MagicMock) -> AlertManager:
        return AlertManager(senders=[mock_sender])

    def test_send_info(self, manager: AlertManager, mock_sender: MagicMock) -> None:
        result = manager.send_info("Daily Report", "PnL: +$50")
        assert result
        assert manager.sent_count == 1
        mock_sender.send.assert_called_once()

    def test_send_warning(self, manager: AlertManager, mock_sender: MagicMock) -> None:
        result = manager.send_warning("Drift Detected", "OBI PSI=0.35")
        assert result
        assert manager.sent_count == 1

    def test_send_critical(self, manager: AlertManager, mock_sender: MagicMock) -> None:
        result = manager.send_critical("Kill Switch", "Daily loss limit hit")
        assert result

    def test_send_emergency(self, manager: AlertManager, mock_sender: MagicMock) -> None:
        result = manager.send_emergency("Exchange Error", "WS connection lost")
        assert result

    def test_rate_limited_alert_suppressed(self, mock_sender: MagicMock) -> None:
        manager = AlertManager(senders=[mock_sender])
        # First INFO goes through
        assert manager.send_info("First", "msg")
        # Second INFO rate limited
        result = manager.send_info("Second", "msg")
        assert not result
        assert manager.suppressed_count == 1
        assert mock_sender.send.call_count == 1

    def test_history_recorded(self, manager: AlertManager) -> None:
        manager.send_info("Test1", "msg1")
        manager.send_warning("Test2", "msg2")
        history = manager.history
        assert len(history) == 2
        assert history[0].title == "Test1"
        assert history[1].title == "Test2"

    def test_no_senders_logs_warning(self) -> None:
        manager = AlertManager(senders=[])
        result = manager.send_critical("Test", "msg")
        assert not result

    def test_sender_exception_handled(self) -> None:
        bad_sender = MagicMock()
        bad_sender.send.side_effect = RuntimeError("connection failed")
        manager = AlertManager(senders=[bad_sender])
        # Should not raise
        result = manager.send_critical("Test", "msg")
        assert not result

    def test_multiple_senders(self) -> None:
        sender1 = MagicMock()
        sender1.send.return_value = True
        sender2 = MagicMock()
        sender2.send.return_value = True
        manager = AlertManager(senders=[sender1, sender2])
        result = manager.send_critical("Test", "msg")
        assert result
        sender1.send.assert_called_once()
        sender2.send.assert_called_once()

    def test_partial_delivery_counts(self) -> None:
        good_sender = MagicMock()
        good_sender.send.return_value = True
        bad_sender = MagicMock()
        bad_sender.send.return_value = False
        manager = AlertManager(senders=[good_sender, bad_sender])
        result = manager.send_critical("Test", "msg")
        assert result  # At least one succeeded
        assert manager.sent_count == 1

    def test_metadata_passed_through(self, manager: AlertManager) -> None:
        manager.send_info("Test", "msg", regime="trending", sharpe=1.5)
        alert = manager.history[0]
        assert alert.metadata["regime"] == "trending"
        assert alert.metadata["sharpe"] == 1.5
