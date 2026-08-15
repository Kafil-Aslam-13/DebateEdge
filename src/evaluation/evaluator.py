"""Debate Evaluation
2 TYPES:
1-> TURN LEVEL (runs every turn)
2-> Session level - (runs on debate end)

"""

from dataclasses import dataclass, field
from statistics import mean
from typing import Optional

from src.core.constants import(
    DECLINING,
    EVAL_AVERAGE,
    EVAL_GOOD,
    EVAL_EXCELLENT,
    EVAL_POOR,
    IMPROVING,
    INSUFFICIENT,
    STABLE,
    TASK_CLASSIFICATION
)
from src.core.logger import get_logger
from src.gateway.llm_gateway import get_gateway
from src.observability.logfire_setup import get_logfire_span
from src.observability.metrics import record_argument_score

from pydantic import BaseModel, Field

logger = get_logger(__name__)

class AIResponseJudgeOutput(BaseModel):
    relevance: int = Field(ge=0, le=10)
    evidence: int = Field(ge=0, le=10)
    persuasion: int = Field(ge=0, le=10)
    coaching: int = Field(ge=0, le=10)
    feedback: str

@dataclass
class TurnEvaluation:
    """Evaluation result for a single debate turn"""
    turn_number:     int
    argument_score:  int          
    argument_quality: str         
    ai_response_score: int       
    ai_relevance:    int           # 0-10
    ai_evidence:     int           # 0-10
    ai_persuasion:   int           # 0-10
    ai_coaching:     int           # 0-10
    ai_feedback:     str           # one sentence from judge
    grade:           str  


@dataclass
class SessionEvaluation:
    """Comprehensive evaluation of full debate session."""
    total_turns:         int
    user_improvement:    str        # improving/declining/stable/insufficient
    avg_score_first_half: float
    avg_score_second_half: float
    score_trend:         list[int]  # all scores in order
    fallacy_count:       int
    fallacy_types:       list[str]
    strong_count:        int
    weak_count:          int
    fallacy_argument_count: int
    best_turn:           int        # turn with highest score
    worst_turn:          int        # turn with lowest score
    total_tokens:        int
    total_cost_usd:      float
    cache_hits:          int
    overall_grade:       str
    improvement_advice:  str        # LLM-generated coaching summary

class DebateEvaluator:
    """evaluate debate quality at turn and session level"""

    def __init__(self):
        self._gateway=get_gateway()
        self._turn_evals:list[TurnEvaluation]=[]
        logger.info("Debate evaluator")

    def evaluate_turn(
            self,
            turn_number: int,
            user_argument: str,
            ai_response: str,
            argument_score: int,
            argument_quality:str,
            topic: str

        ) -> TurnEvaluation:
        """Evaluate 1 debate turn
        LLM as a judge scores the Ai counterargument on 4 dimensions.
        """
        with get_logfire_span(
            "evaluator.turn",
            turn=turn_number,
        ):
            ai_scores=self._judge_ai_response(
                user_argument=user_argument,
                ai_response=ai_response,
                topic = topic,
            )

            overall_ai_score = int(mean([
                ai_scores["relevance"],
                ai_scores["evidence"],
                ai_scores["persuasion"],
                ai_scores["coaching"],
            ]))

            grade = self._compute_grade(overall_ai_score)
            eval_result = TurnEvaluation(
                turn_number=turn_number,
                argument_quality=argument_quality,
                argument_score=argument_score,
                ai_response_score=overall_ai_score,
                ai_relevance=ai_scores["relevance"],
                ai_evidence=ai_scores["evidence"],
                ai_persuasion=ai_scores["persuasion"],
                ai_coaching=ai_scores["coaching"],
                ai_feedback=ai_scores["feedback"],
                grade=grade,
            )

            self._turn_evals.append(eval_result)

            logger.info(
                f"TurnEval | turn={turn_number} | "
                f"user_score={argument_score} | "
                f"ai_score={overall_ai_score} | "
                f"grade={grade}"
            )

            return eval_result



        
    def evaluate_session(self) -> Optional[SessionEvaluation]:
        """Evaluate the full debate session.

        Requires at least 2 turns of data.
        Combines rule-based trend analysis with
        LLM-generated improvement advice.

        Returns:
            SessionEvaluation or None if insufficient data
        """
        if len(self._turn_evals) < 2:
            logger.info("Session eval: insufficient turns (need >= 2)")
            return None

        with get_logfire_span(
            "evaluator.session",
            total_turns=len(self._turn_evals),
        ):
            scores  = [e.argument_score   for e in self._turn_evals]
            turns   = [e.turn_number      for e in self._turn_evals]
            quality = [e.argument_quality for e in self._turn_evals]

            
            improvement, avg_first, avg_second = self._compute_trend(scores)

            strong_count  = quality.count("strong")
            weak_count    = quality.count("weak")
            fallacy_count = quality.count("fallacy")

            fallacy_types = list({
                e.argument_quality
                for e in self._turn_evals
                if e.argument_quality == "fallacy"
            })

            best_turn  = turns[scores.index(max(scores))]
            worst_turn = turns[scores.index(min(scores))]

            avg_user_score = mean(scores)
            overall_grade  = self._compute_grade(int(avg_user_score))

            cost_summary = self._gateway.get_cost_summary()

            advice = self._generate_advice(
                scores=scores,
                improvement=improvement,
                strong_count=strong_count,
                weak_count=weak_count,
                fallacy_count=fallacy_count,
            )

            session_eval = SessionEvaluation(
                total_turns=len(self._turn_evals),
                user_improvement=improvement,
                avg_score_first_half=round(avg_first, 2),
                avg_score_second_half=round(avg_second, 2),
                score_trend=scores,
                fallacy_count=fallacy_count,
                fallacy_types=fallacy_types,
                strong_count=strong_count,
                weak_count=weak_count,
                fallacy_argument_count=fallacy_count,
                best_turn=best_turn,
                worst_turn=worst_turn,
                total_tokens=cost_summary.get("total_tokens", 0),
                total_cost_usd=cost_summary.get("total_cost_usd", 0.0),
                cache_hits=cost_summary.get("cache_hits", 0),
                overall_grade=overall_grade,
                improvement_advice=advice,
            )

            logger.info(
                f"SessionEval | turns={len(self._turn_evals)} | "
                f"improvement={improvement} | "
                f"avg_score={avg_user_score:.1f} | "
                f"grade={overall_grade}"
            )

            return session_eval




    def _judge_ai_response(
        self,
        user_argument: str,
        ai_response:   str,
        topic:         str,
    ) -> dict:
        """LLM-as-judge: score AI counterargument on 4 dimensions.
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert debate coach evaluating "
                        "AI counterarguments. Be rigorous and fair. "
                        "Respond with valid JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Debate topic: {topic}\n"
                        f"User's argument: {user_argument}\n"
                        f"AI counterargument: {ai_response}\n\n"
                        f"Score the AI counterargument 0-10 on:\n"
                        f"- relevance:   directly addresses user's argument\n"
                        f"- evidence:    uses facts, data, examples\n"
                        f"- persuasion:  logically convincing\n"
                        f"- coaching:    helps user understand weakness\n\n"
                        f"Respond JSON only:\n"
                        f'{{"relevance":0-10,"evidence":0-10,'
                        f'"persuasion":0-10,"coaching":0-10,'
                        f'"feedback":"one sentence"}}'
                    ),
                },
            ]

            
            result = self._gateway.complete(
                messages=messages, task_type=TASK_CLASSIFICATION,response_model=AIResponseJudgeOutput
            )
            assert isinstance(result , AIResponseJudgeOutput)

            return {
                "relevance": result.relevance,
                "evidence": result.evidence,
                "persuasion": result.persuasion,
                "coaching": result.coaching,
                "feedback": result.feedback,
            }

        except Exception as e:
            logger.warning(f"AI response judge failed: {e}")
            return {
                "relevance":  5,
                "evidence":   5,
                "persuasion": 5,
                "coaching":   5,
                "feedback":   "Evaluation unavailable.",
            }

    def _compute_trend(
        self,
        scores: list[int],
    ) -> tuple[str, float, float]:
        """Compute improvement trend from score list.

        Rule-based: compare first half avg to second half avg.
        No LLM needed — this is pure maths.

        Returns:
            (improvement_direction, avg_first_half, avg_second_half)
        """
        if len(scores) < 2:
            return INSUFFICIENT, 0.0, 0.0

        mid       = len(scores) // 2
        first_half  = scores[:mid]  if mid > 0 else scores[:1]
        second_half = scores[mid:]  if mid > 0 else scores[1:]

        avg_first  = mean(first_half)
        avg_second = mean(second_half)
        diff       = avg_second - avg_first

        if diff >= 1.5:
            direction = IMPROVING
        elif diff <= -1.5:
            direction = DECLINING
        else:
            direction = STABLE

        return direction, avg_first, avg_second

    def _compute_grade(self, score: int) -> str:
        """Convert numeric score to grade label.

        Rule-based thresholds — no LLM needed.
        """
        if score >= 8:
            return EVAL_EXCELLENT
        if score >= 6:
            return EVAL_GOOD
        if score >= 4:
            return EVAL_AVERAGE
        return EVAL_POOR

    def _generate_advice(
        self,
        scores:        list[int],
        improvement:   str,
        strong_count:  int,
        weak_count:    int,
        fallacy_count: int,
    ) -> str:
        """Generate personalised coaching advice using LLM.

        Rule-based analysis identifies patterns.
        LLM converts patterns into natural language advice.
        Hybrid approach — rules do the analysis,
        LLM does the communication.
        """
        try:
            pattern_summary = (
                f"Score trend: {scores}\n"
                f"Improvement direction: {improvement}\n"
                f"Strong arguments: {strong_count}\n"
                f"Weak arguments: {weak_count}\n"
                f"Fallacies committed: {fallacy_count}\n"
                f"Score range: {min(scores)}-{max(scores)}"
            )

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert debate coach. "
                        "Give concise, specific, actionable advice. "
                        "Be encouraging but honest."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Based on this debate session analysis:\n"
                        f"{pattern_summary}\n\n"
                        f"Write 2-3 sentences of specific coaching advice "
                        f"for how this debater can improve. "
                        f"Reference their actual performance patterns."
                    ),
                },
            ]

            return self._gateway.complete(
                messages, task_type="explanation"
            )

        except Exception as e:
            logger.warning(f"Advice generation failed: {e}")
            return (
                "Keep practising! Focus on supporting your arguments "
                "with specific evidence and avoid logical fallacies."
            )

    def reset(self) -> None:
        """Clear all turn evaluations for new session."""
        self._turn_evals.clear()
        logger.info("DebateEvaluator reset.")

    def get_turn_history(self) -> list[TurnEvaluation]:
        """Return all turn evaluations so far."""
        return list(self._turn_evals)