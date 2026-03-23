"""Tests for structured logging configuration."""

import json
import logging

import structlog

from ep2_crypto.logging import configure_logging


class TestConfigureLogging:
    def test_json_output_to_stderr(self, capsys):
        configure_logging(level="INFO", json_output=True)
        logger = structlog.get_logger("test")
        logger.info("test_event", key="value")

        captured = capsys.readouterr()
        assert captured.out == ""  # nothing on stdout
        line = captured.err.strip()
        data = json.loads(line)
        assert data["event"] == "test_event"
        assert data["key"] == "value"
        assert data["level"] == "info"
        assert "timestamp" in data

    def test_callsite_info_included(self, capsys):
        configure_logging(level="DEBUG", json_output=True)
        logger = structlog.get_logger("test")
        logger.debug("callsite_test")

        captured = capsys.readouterr()
        data = json.loads(captured.err.strip())
        assert "filename" in data
        assert "func_name" in data
        assert "lineno" in data

    def test_log_level_filtering(self, capsys):
        configure_logging(level="WARNING", json_output=True)
        logger = structlog.get_logger("test")
        logger.info("should_not_appear")
        logger.warning("should_appear")

        captured = capsys.readouterr()
        lines = [l for l in captured.err.strip().split("\n") if l]
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["event"] == "should_appear"

    def test_console_renderer_mode(self, capsys):
        configure_logging(level="INFO", json_output=False)
        logger = structlog.get_logger("test")
        logger.info("human_readable")

        captured = capsys.readouterr()
        assert "human_readable" in captured.err
        # Should NOT be valid JSON
        try:
            json.loads(captured.err.strip())
            is_json = True
        except json.JSONDecodeError:
            is_json = False
        assert not is_json

    def test_third_party_loggers_silenced(self):
        configure_logging(level="DEBUG")
        for name in ("urllib3", "websockets", "ccxt", "asyncio"):
            assert logging.getLogger(name).level == logging.WARNING
