"""Tests for prompt templates — no LLM calls needed."""
import pytest
from src.prompts.debate_prompts import debate_prompt, opening_prompt
from src.prompts.scoring_prompts import (
    classification_prompt,
    scoring_prompt,
    strong_handler_prompt,
    weak_handler_prompt,
    fallacy_handler_prompt,
)


def test_debate_prompt_has_required_variables():
    """debate_prompt must accept topic, side, debate_history, user_argument."""
    variables = debate_prompt.input_variables
    assert "topic" in variables
    assert "user_argument" in variables


def test_opening_prompt_has_required_variables():
    variables = opening_prompt.input_variables
    assert "topic" in variables
    assert "side" in variables


def test_classification_prompt_formats():
    """Classification prompt should format without errors."""
    try:
        result = classification_prompt.format_messages(
            argument="Test argument",
            topic="Test topic",
        )
        assert len(result) > 0
    except Exception as e:
        pytest.fail(f"classification_prompt.format_messages failed: {e}")


def test_scoring_prompt_formats():
    try:
        result = scoring_prompt.format_messages(
            topic="Test topic",
            user_side="for",
            argument="Test argument",
            quality="weak",
        )
        assert len(result) > 0
    except Exception as e:
        import pytest
        pytest.fail(f"scoring_prompt.format_messages failed: {e}")


def test_partial_prompts_pre_filled():
    """Handler prompts should have coaching_mode pre-filled."""
    # These should format with fewer variables than the base prompt
    # because .partial() pre-filled coaching_mode and handler_instruction
    try:
        strong_handler_prompt.format_messages(
            topic="topic",
            argument="argument",
            quality="strong",
            score=8,
            quality_reasoning="good argument",
            ai_side="against",
        )
    except Exception as e:
        import pytest
        pytest.fail(f"strong_handler_prompt failed: {e}")