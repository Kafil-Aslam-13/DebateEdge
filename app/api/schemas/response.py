"""Api response schema"""
from pydantic import BaseModel
from typing import Optional

class StartDebateResponse(BaseModel):
    opening_statement: str
    ai_side:           str
    topic:             str


class ScoreBreakdown(BaseModel):
    logic:    int
    evidence: int
    clarity:  int

class FallacyInfo(BaseModel):
    detected:     bool
    name:         str
    severity:     str
    explanation:  str
    correction:   str

class GuardInfo(BaseModel):
    passed: bool
    reason: str

class EvalInfo(BaseModel):
    score:    int
    grade:    str
    feedback: str

class CostInfo(BaseModel):
    total_tokens:   int
    total_cost_usd: float
    cache_hits:     int

class ArgueResponse(BaseModel):
    ai_response:      str
    argument_quality: str
    argument_score:   float
    score_breakdown:  ScoreBreakdown
    quality_reasoning: str
    fallacy:          FallacyInfo
    input_guard:      GuardInfo
    evaluation:       EvalInfo
    debate_summary:   str
    similar_past_args: list
    cost:             CostInfo
    rag_context:      str

class SessionEvalResponse(BaseModel):
    total_turns:           int
    user_improvement:      str
    avg_score_first_half:  float
    avg_score_second_half: float
    score_trend:           list[float]
    fallacy_count:         int
    strong_count:          int
    weak_count:            int
    best_turn:             int
    worst_turn:            int
    overall_grade:         str
    improvement_advice:    str
    total_tokens:          int
    total_cost_usd:        float
    cache_hits:            int

class HealthResponse(BaseModel):
    status:      str
    version:     str
    environment: str