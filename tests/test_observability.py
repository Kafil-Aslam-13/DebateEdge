"""Tests for observability — no API keys needed."""

import os
import pytest
from unittest.mock import patch, MagicMock


# ── LangSmith setup tests ──────────────────────────────────────────────────────

def test_langsmith_setup_disabled_by_config():
    """LangSmith skips setup when disabled in config."""
    with patch("src.observability.langsmith_setup.get_settings") as mock:
        mock.return_value.langsmith_enabled = False
        from src.observability.langsmith_setup import setup_langsmith
        result = setup_langsmith()
        assert result is False


def test_langsmith_setup_no_key():
    """LangSmith skips gracefully without API key."""
    with patch("src.observability.langsmith_setup.get_settings") as mock:
        mock.return_value.langsmith_enabled = True
        mock.return_value.langsmith_project = "test"
        with patch.dict(os.environ, {"LANGSMITH_API_KEY": ""}, clear=False):
            from src.observability.langsmith_setup import setup_langsmith
            result = setup_langsmith()
            assert result is False


# ── Logfire setup tests ────────────────────────────────────────────────────────

def test_logfire_setup_disabled_by_config():
    """Logfire skips setup when disabled in config."""
    with patch("src.observability.logfire_setup.get_settings") as mock:
        mock.return_value.logfire_enabled = False
        from src.observability.logfire_setup import setup_logfire
        result = setup_logfire()
        assert result is False


def test_logfire_setup_no_token():
    """Logfire skips gracefully without token."""
    with patch("src.observability.logfire_setup.get_settings") as mock:
        mock.return_value.logfire_enabled = True
        mock.return_value.logfire_service = "test"
        mock.return_value.environment     = "test"
        with patch.dict(os.environ, {"LOGFIRE_TOKEN": ""}, clear=False):
            from src.observability.logfire_setup import setup_logfire
            result = setup_logfire()
            assert result is False


def test_logfire_span_noop_without_logfire():
    """get_logfire_span returns no-op if logfire not configured."""
    from src.observability.logfire_setup import get_logfire_span
    # Should not raise even if logfire not configured
    with get_logfire_span("test_span", key="value"):
        pass   # no-op


# ── Metrics tests ──────────────────────────────────────────────────────────────

def test_metrics_noop_without_logfire():
    """All metric functions are no-ops if logfire not available."""
    from src.observability.metrics import (
        record_argument_score,
        record_fallacy_detected,
        record_llm_cost,
        record_session_turn,
        record_rag_retrieval,
        record_guard_result,
    )
    # None of these should raise
    record_argument_score(7, "strong", 1, "social media")
    record_fallacy_detected("ad_hominem", "high", 2)
    record_llm_cost(0.001, 150, "debate", "groq/llama3-70b", False)
    record_session_turn(3, "social media")
    record_rag_retrieval(2, 0)
    record_guard_result("length_check", "pass", "input")