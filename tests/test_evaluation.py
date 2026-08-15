"""Tests for Sprint 9 evaluation — rule-based tests need no API."""

import pytest
from src.evaluation.evaluator import DebateEvaluator
from src.core.constants import (
    IMPROVING, DECLINING, STABLE, INSUFFICIENT,
    EVAL_EXCELLENT, EVAL_GOOD, EVAL_AVERAGE, EVAL_POOR,
)


# ── Grade computation tests (rule-based, no API) ──────────────────────────────

def test_grade_excellent():
    ev = DebateEvaluator.__new__(DebateEvaluator)
    assert ev._compute_grade(9)  == EVAL_EXCELLENT
    assert ev._compute_grade(8)  == EVAL_EXCELLENT


def test_grade_good():
    ev = DebateEvaluator.__new__(DebateEvaluator)
    assert ev._compute_grade(7) == EVAL_GOOD
    assert ev._compute_grade(6) == EVAL_GOOD


def test_grade_average():
    ev = DebateEvaluator.__new__(DebateEvaluator)
    assert ev._compute_grade(5) == EVAL_AVERAGE
    assert ev._compute_grade(4) == EVAL_AVERAGE


def test_grade_poor():
    ev = DebateEvaluator.__new__(DebateEvaluator)
    assert ev._compute_grade(3) == EVAL_POOR
    assert ev._compute_grade(0) == EVAL_POOR


# ── Trend computation tests (rule-based, no API) ──────────────────────────────

def test_trend_improving():
    ev = DebateEvaluator.__new__(DebateEvaluator)
    direction, f, s = ev._compute_trend([3, 4, 7, 8, 9])
    assert direction == IMPROVING
    assert s > f


def test_trend_declining():
    ev = DebateEvaluator.__new__(DebateEvaluator)
    direction, f, s = ev._compute_trend([9, 8, 7, 4, 3])
    assert direction == DECLINING
    assert s < f


def test_trend_stable():
    ev = DebateEvaluator.__new__(DebateEvaluator)
    direction, f, s = ev._compute_trend([5, 6, 5, 6, 5])
    assert direction == STABLE


def test_trend_insufficient_data():
    ev = DebateEvaluator.__new__(DebateEvaluator)
    direction, f, s = ev._compute_trend([5])
    assert direction == INSUFFICIENT


def test_trend_two_scores():
    ev = DebateEvaluator.__new__(DebateEvaluator)
    direction, f, s = ev._compute_trend([3, 8])
    assert direction == IMPROVING


# ── Session evaluation requires turns ────────────────────────────────────────

def test_session_eval_none_with_no_turns():
    ev = DebateEvaluator.__new__(DebateEvaluator)
    ev._turn_evals = []
    ev._gateway    = None
    result = ev.evaluate_session()
    assert result is None


def test_session_eval_none_with_one_turn():
    from src.evaluation.evaluator import TurnEvaluation
    ev = DebateEvaluator.__new__(DebateEvaluator)
    ev._gateway    = None
    ev._turn_evals = [
        TurnEvaluation(
            turn_number=1,
            argument_score=5,
            argument_quality="weak",
            ai_response_score=7,
            ai_relevance=7,
            ai_evidence=6,
            ai_persuasion=7,
            ai_coaching=8,
            ai_feedback="Good response",
            grade="good",
        )
    ]
    result = ev.evaluate_session()
    assert result is None


# ── Reset ─────────────────────────────────────────────────────────────────────

def test_evaluator_reset():
    from src.evaluation.evaluator import TurnEvaluation
    ev = DebateEvaluator.__new__(DebateEvaluator)
    ev._turn_evals = [
        TurnEvaluation(1, 5, "weak", 7, 7, 6, 7, 8, "Good", "good")
    ]
    ev.reset()
    assert len(ev._turn_evals) == 0