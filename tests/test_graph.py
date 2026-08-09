"""Tests for Sprint 2 LangGraph workflow.

Tests the graph structure and routing logic independently
from the LLM — so tests run fast without API calls.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.core.constants import (
    QUALITY_STRONG, QUALITY_WEAK, QUALITY_FALLACY,
    NODE_STRONG, NODE_WEAK, NODE_FALLACY,
)
from src.graphs.debate_graph import route_by_quality
from src.graphs.state import DebateState


def make_state(**kwargs) -> DebateState:
    """Build a minimal valid DebateState for testing."""
    defaults = {
        "topic": "AI will replace all jobs",
        "user_side": "for",
        "user_argument": "test argument",
        "turn_number": 1,
        "debate_history": [],
        "argument_quality": QUALITY_WEAK,
        "quality_reasoning": "",
        "argument_score": 5,
        "score_breakdown": {},
        "handler_note": "",
        "ai_response": "",
        "error": "",
        "has_error": False,
    }
    defaults.update(kwargs)
    return defaults


# ── Router tests — no LLM needed ─────────────────────────────────────────────

def test_route_strong_argument():
    state = make_state(argument_quality=QUALITY_STRONG)
    assert route_by_quality(state) == NODE_STRONG


def test_route_weak_argument():
    state = make_state(argument_quality=QUALITY_WEAK)
    assert route_by_quality(state) == NODE_WEAK


def test_route_fallacy_argument():
    state = make_state(argument_quality=QUALITY_FALLACY)
    assert route_by_quality(state) == NODE_FALLACY


def test_route_unknown_defaults_to_weak():
    state = make_state(argument_quality="unknown_quality")
    assert route_by_quality(state) == NODE_WEAK


# ── State structure tests ─────────────────────────────────────────────────────

def test_state_has_required_fields():
    state = make_state()
    required = [
        "topic", "user_side", "user_argument", "turn_number",
        "debate_history", "argument_quality", "argument_score",
        "ai_response", "has_error",
    ]
    for field in required:
        assert field in state, f"Missing field: {field}"


def test_state_default_values():
    state = make_state()
    assert state["has_error"] is False
    assert state["debate_history"] == []
    assert state["turn_number"] == 1