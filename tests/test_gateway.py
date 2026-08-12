"""Tests for Sprint 6 LLM Gateway — no real API calls needed."""

import pytest
from unittest.mock import MagicMock, patch

from src.gateway.cost_tracker import CostTracker, CallRecord
from src.gateway.llm_gateway import LLMGateway


# ── CostTracker tests — no mocking needed ────────────────────────────────────

def test_cost_tracker_records_call():
    tracker = CostTracker()
    record = tracker.record(
        task_type="classification",
        model="groq/llama3-8b-8192",
        prompt_tokens=100,
        completion_tokens=50,
        cached=False,
    )
    assert record.total_tokens == 150
    assert record.cost_usd >= 0
    assert record.cached is False


def test_cost_tracker_records_cache_hit():
    tracker = CostTracker()
    tracker.record(
        task_type="debate",
        model="groq/llama3-70b-8192",
        prompt_tokens=0,
        completion_tokens=0,
        cached=True,
    )
    summary = tracker.get_summary()
    assert summary["cache_hits"] == 1
    assert summary["total_tokens"] == 0


def test_cost_tracker_summary():
    tracker = CostTracker()
    tracker.record("classification", "groq/llama3-8b-8192", 100, 50)
    tracker.record("debate",         "groq/llama3-70b-8192", 200, 100)
    summary = tracker.get_summary()
    assert summary["total_calls"]  == 2
    assert summary["total_tokens"] == 450
    assert "classification" in summary["tokens_by_task"]
    assert "debate"         in summary["tokens_by_task"]


def test_cost_tracker_reset():
    tracker = CostTracker()
    tracker.record("debate", "groq/llama3-70b-8192", 100, 50)
    tracker.reset()
    summary = tracker.get_summary()
    assert summary["total_calls"]  == 0
    assert summary["total_tokens"] == 0


def test_cost_tracker_unknown_model_uses_default():
    tracker = CostTracker()
    record = tracker.record(
        task_type="debate",
        model="unknown/model-xyz",
        prompt_tokens=100,
        completion_tokens=50,
    )
    # Should not crash — uses default rates
    assert record.cost_usd >= 0


# ── Gateway cache tests — mock the router ────────────────────────────────────

def _make_mock_response(content: str, model: str = "groq/llama3-8b-8192"):
    """Build a mock litellm response object."""
    response = MagicMock()
    response.choices[0].message.content = content
    response.model  = model
    response.usage.prompt_tokens     = 100
    response.usage.completion_tokens = 50
    return response


def test_gateway_caches_identical_prompts():
    """Second call with same messages returns cached result."""
    with patch("src.gateway.llm_gateway.Router") as MockRouter:
        instance = MockRouter.return_value
        instance.completion.return_value = _make_mock_response(
            '{"quality": "weak", "reasoning": "test"}'
        )

        gateway = LLMGateway()
        messages = [{"role": "user", "content": "test argument"}]

        result1 = gateway.complete(messages, task_type="classification")
        result2 = gateway.complete(messages, task_type="classification")

        # Router called only once — second call served from cache
        assert instance.completion.call_count == 1
        assert result1 == result2


def test_gateway_different_task_types_different_models():
    """Classification and debate should use different model aliases."""
    with patch("src.gateway.llm_gateway.Router") as MockRouter:
        instance = MockRouter.return_value
        instance.completion.return_value = _make_mock_response("response")

        gateway = LLMGateway()

        gateway.complete(
            [{"role": "user", "content": "classify this"}],
            task_type="classification",
        )
        gateway.complete(
            [{"role": "user", "content": "debate this"}],
            task_type="debate",
        )

        calls = instance.completion.call_args_list
        # First call should use "fast", second "powerful"
        assert calls[0][1]["model"] == "fast"
        assert calls[1][1]["model"] == "powerful"


def test_gateway_reset_clears_cache():
    with patch("src.gateway.llm_gateway.Router") as MockRouter:
        instance = MockRouter.return_value
        instance.completion.return_value = _make_mock_response("response")

        gateway = LLMGateway()
        messages = [{"role": "user", "content": "test"}]

        gateway.complete(messages, task_type="classification")
        gateway.reset()
        gateway.complete(messages, task_type="classification")

        # After reset cache is empty → router called twice
        assert instance.completion.call_count == 2