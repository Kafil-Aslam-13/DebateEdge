"""Tests for Sprint 3 parsers — no LLM calls needed."""

import pytest
from src.parsers.fallacy_parsers import (
    FallacyDetectionResult,
    FallacyType,
    FallacySeverity,
    parse_fallacy_safe,
    get_format_instructions,
)


def test_no_fallacy_normalises_fields():
    """If contains_fallacy=False all related fields forced to none."""
    result = FallacyDetectionResult(
        contains_fallacy=False,
        fallacy_name="ad_hominem",
        fallacy_type=FallacyType.INFORMAL,
        explanation="test",
        severity=FallacySeverity.HIGH,
        correction="fix it",
    )
    assert result.fallacy_name == "none"
    assert result.fallacy_type == FallacyType.NONE
    assert result.severity == FallacySeverity.NONE
    assert result.correction == "none"


def test_fallacy_name_normalised():
    """Fallacy name lowercased and underscored."""
    result = FallacyDetectionResult(
        contains_fallacy=True,
        fallacy_name="Ad Hominem",
        fallacy_type=FallacyType.INFORMAL,
        explanation="test",
        severity=FallacySeverity.HIGH,
        correction="fix it",
    )
    assert result.fallacy_name == "ad_hominem"


def test_parse_fallacy_safe_valid_json():
    """parse_fallacy_safe handles valid JSON."""
    text = '{"contains_fallacy": false, "fallacy_name": "none", "fallacy_type": "none", "explanation": "no fallacy", "severity": "none", "correction": "none"}'
    result = parse_fallacy_safe(text)
    assert result.contains_fallacy is False


def test_parse_fallacy_safe_bad_input():
    """parse_fallacy_safe returns safe default on bad input."""
    result = parse_fallacy_safe("this is not json at all !!!")
    assert result.contains_fallacy is False
    assert result.fallacy_name == "none"


def test_format_instructions_is_string():
    instructions = get_format_instructions()
    assert isinstance(instructions, str)
    assert len(instructions) > 0